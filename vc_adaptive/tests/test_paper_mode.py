"""Regression checks for paper job coverage and the actual plotted data."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as runner
import plot
from experiment_config import result_filename


class PaperModeTests(unittest.TestCase):
    def test_normal_defaults_and_paper_overrides(self):
        normal = runner.parse_args([])
        self.assertEqual(normal.repeats, 1)
        self.assertEqual(normal.alpha, [0.1])
        self.assertEqual(normal.line_search_mode, ["parallel"])
        self.assertEqual(normal.time_limit, 100)
        paper = runner.parse_args(["--paper"])
        self.assertEqual(paper.repeats, 10)
        self.assertEqual(paper.time_limit, 100)
        self.assertEqual(paper.line_search_mode, ["parallel"])
        custom = runner.parse_args(["--vc-paper-mode", "--repeats", "2", "--time-limit", "1"])
        self.assertEqual((custom.repeats, custom.time_limit), (2, 1))
        for option in (["--R", "0.5"], ["--alpha", "0.1"],
                       ["--algorithms", "ours"], ["--line-search-mode", "parallel"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["--paper", *option])

    def test_all_problem_job_coverage_without_duplicates(self):
        for problem in ("quad_arith", "rosenbrock", "dixon_price", "powell", "qing"):
            with self.subTest(problem=problem):
                args = runner.parse_args(["--paper", "--problem", problem])
                jobs = runner.make_paper_jobs(args)
                baselines = {"gd", "hb", "agd"}
                if problem == "quad_arith":
                    baselines.add("nag")
                if problem == "powell":
                    baselines.add("nagc")
                self.assertEqual(len(jobs), len(baselines) + 10)
                self.assertEqual(len({j["job_id"] for j in jobs}), len(jobs))
                self.assertEqual({j["algorithm"] for j in jobs}, baselines | {"ours", "ours_defer_cp"})
                vc = [j for j in jobs if j["algorithm"] == "ours"]
                self.assertEqual({(j["R"], j["alpha"]) for j in vc},
                                 {(r, a) for r in (0.25, 0.5, 0.75) for a in (0.05, 0.1, 0.5)})
                pred = [j for j in jobs if j["algorithm"] == "ours_defer_cp"]
                self.assertEqual([(j["R"], j["alpha"]) for j in pred], [(0.5, 0.1)])
                self.assertTrue(all(j["line_search_mode"] == "parallel" for j in vc + pred))

    def test_section_selection_and_threshold(self):
        args = runner.parse_args(["--paper-compare", "--problem", "qing", "--checkpoint-threshold", "7"])
        jobs = runner.make_paper_jobs(args)
        self.assertEqual(len(jobs), 5)
        pred = next(j for j in jobs if j["algorithm"] == "ours_defer_cp")
        self.assertEqual(pred["checkpoint_threshold"], 7)
        self.assertTrue(str(runner.result_path_for_job(Path("."), "qing", pred)).endswith("cpT7.npz"))
        sweep = runner.make_paper_jobs(runner.parse_args(["--vc-paper-sweep"]))
        self.assertEqual(len(sweep), 9)
        self.assertEqual({j["algorithm"] for j in sweep}, {"ours"})
        both = runner.make_paper_jobs(runner.parse_args(["--paper-compare", "--paper-sweep"]))
        self.assertEqual(len(both), 14)

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "absent"
            with patch.object(sys, "argv", ["main.py", "--paper", "--dry-run", "--result-dir", str(out)]):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    runner.main()
            self.assertFalse(out.exists())
            self.assertIn("Total trials: 14 jobs x 10 repeats = 140", stdout.getvalue())

    def test_figure_data_labels_and_custom_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(algorithm, value, **kwargs):
                np.savez(root / result_filename("powell", algorithm, **kwargs),
                         val=np.array([value, value / 2]), grad_call=[1, 2],
                         grad_val_call=[3, 6], oracle_call=[2, 4], time=[0.1, 0.2])
            for algorithm in ("gd", "hb", "agd", "nagc"):
                save(algorithm, 100)
            for r in (0.25, 0.5, 0.75):
                for a in (0.05, 0.1, 0.5):
                    save("ours", 8, R=r, alpha=a, line_search_mode="parallel")
                    save("ours", 900, R=r, alpha=a, line_search_mode="short_circuit")
            save("ours_defer_cp", 4, R=0.5, alpha=0.1, line_search_mode="parallel", checkpoint_threshold=7)
            save("ours_defer_cp", 800, R=0.5, alpha=0.1, line_search_mode="parallel", checkpoint_threshold=10)
            save("ours_defer_cp", 900, R=0.5, alpha=0.1, line_search_mode="short_circuit", checkpoint_threshold=7)
            with contextlib.redirect_stdout(io.StringIO()):
                figures = plot.plot_vc_paper_set(["powell"], result_dir=root, show=False, checkpoint_threshold=7)
            self.assertEqual(len(list((root / "fig").glob("*.pdf"))), 4)
            for _, ax in figures[:3]:
                lines = {line.get_label(): line for line in ax.lines}
                self.assertEqual(set(lines), {"GD", "HB", "AGD", "NAG-C", "VC", "Predictive VC"})
                np.testing.assert_array_equal(lines["VC"].get_ydata(), [8, 4])
                np.testing.assert_array_equal(lines["Predictive VC"].get_ydata(), [4, 2])
            self.assertEqual(len(figures[3][1].lines), 9)
            self.assertEqual(figures[1][1].get_xlabel(), "#oracle calls")
            self.assertEqual(figures[3][1].get_xlabel(), "#oracle calls")
            for line in figures[1][1].lines:
                np.testing.assert_array_equal(line.get_xdata(), [2, 4])
            for line in figures[3][1].lines:
                np.testing.assert_array_equal(line.get_ydata(), [8, 4])
                np.testing.assert_array_equal(line.get_xdata(), [2, 4])
            plt.close("all")

    def test_paper_plots_do_not_substitute_short_circuit_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for r in (0.25, 0.5, 0.75):
                for a in (0.05, 0.1, 0.5):
                    for algorithm in ("ours", "ours_defer_cp"):
                        np.savez(root / result_filename("powell", algorithm, R=r, alpha=a,
                                                       line_search_mode="short_circuit", checkpoint_threshold=10),
                                 val=[900], grad_call=[1], grad_val_call=[2], time=[0.1])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with contextlib.redirect_stdout(io.StringIO()):
                    figures = plot.plot_vc_paper_set(["powell"], result_dir=root, show=False)
            self.assertTrue(all(len(ax.lines) == 0 for _, ax in figures))
            self.assertTrue(any("skip missing VC result" in str(w.message) for w in caught))
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
