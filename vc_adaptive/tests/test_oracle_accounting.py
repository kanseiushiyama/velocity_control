"""Check actual evaluation paths, rollback charging, and saved plot coordinates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as runner
import plot
from obj_func import Oracle
from optimizer import (
    AdaRestartAGD, AdaRestartHB, AdaVeloControlDeferredRCheckpoint,
    Gradient_descent, NAGc, NAGsc,
)
from run_trial import extract_history


class Quadratic:
    name = "OracleAccountingTest"
    mu = .125

    def __init__(self):
        self.param = jnp.asarray([1., -2.], dtype=jnp.float64)

    def loss(self, x):
        return .125 * jnp.vdot(x, x)


class OracleAccountingTests(unittest.TestCase):
    def test_separate_joint_ignored_and_partial_evaluations(self):
        model = Quadratic()
        oracle = Oracle(model)
        oracle.loss(model.param)
        oracle.grad(model.param)
        self.assertEqual(oracle.oracle_calls, 2)
        oracle.value_and_grad(model.param)
        self.assertEqual(oracle.count, dict(val=2, grad=2, joint=1))
        self.assertEqual(oracle.oracle_calls, 3)
        oracle.value_and_grad(model.param, count=False)
        self.assertEqual(oracle.oracle_calls, 3)
        oracle.value_and_grad(model.param, count_val=False, count_grad=True)
        self.assertEqual(oracle.oracle_calls, 4)
        oracle.reset_count()
        self.assertEqual(oracle.count, dict(val=0, grad=0, joint=0))

    def test_baseline_paths_and_direct_npz_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cls in (Gradient_descent, AdaRestartHB, AdaRestartAGD, NAGsc, NAGc):
                with self.subTest(algorithm=cls.__name__):
                    opt = cls(Quadratic())
                    opt.stop_mode, opt.n_iter = "iter", 8
                    opt.result_dir = Path(tmp)
                    opt.run()
                    counts = opt.oracle.count
                    if cls is Gradient_descent:
                        self.assertEqual(counts["joint"], 0)
                        # The first reported gradient must already be charged.
                        self.assertEqual(opt.grad_call_hist[0], 1)
                        self.assertEqual(opt.oracle_call_hist[0], 2)
                    else:
                        # Every algorithmic gradient in these kernels is joint.
                        self.assertEqual(counts["joint"], counts["grad"])
                        self.assertEqual(opt.oracle.oracle_calls, counts["val"])
                    hist = extract_history(opt)
                    np.testing.assert_array_equal(hist["oracle_call"],
                                                  hist["grad_val_call"] - hist["joint_call"])
                    self.assertEqual(len(hist["val"]), len(hist["oracle_call"]))
            for path in Path(tmp).glob("*.npz"):
                with np.load(path) as data:
                    self.assertIn("oracle_call", data.files)
                    self.assertIn("joint_call", data.files)

    def test_discarded_prediction_remains_charged(self):
        opt = AdaVeloControlDeferredRCheckpoint(
            Quadratic(), R=.5, alpha=.1, checkpoint_threshold=2,
            line_search_mode="parallel")
        opt.L_est = 4.
        opt.stop_mode, opt.n_iter, opt.save_hist = "iter", 60, False
        opt.warmup()
        self.assertEqual(opt.oracle.oracle_calls, 0)
        calls = dict(full=0, checkpoint=0, predicted=0)
        full, checkpoint, batch = opt._ada_full_step, opt._ada_checkpoint_step, opt._ada_r1_batch_step
        rejected = False

        def full_counted(*args):
            calls["full"] += 1
            return full(*args)

        def checkpoint_counted(*args):
            calls["checkpoint"] += 1
            return checkpoint(*args)

        def reject_first_batch(*args):
            nonlocal rejected
            out = batch(*args)
            calls["predicted"] += int(out.attempted)
            if not rejected:
                rejected = True
                # Exercise host rollback after real evaluations have run.
                return out._replace(success=jnp.asarray(False))
            return out

        opt._ada_full_step = full_counted
        opt._ada_checkpoint_step = checkpoint_counted
        opt._ada_r1_batch_step = reject_first_batch
        opt.optimize()
        self.assertTrue(rejected)
        self.assertGreater(opt.rollback_count, 0)
        self.assertEqual(opt.oracle.count["joint"], 1 + sum(calls.values()))
        self.assertGreater(calls["predicted"], 0)
        hist = extract_history(opt)
        self.assertGreater(len(hist["oracle_call"]), 0)
        self.assertEqual(len(hist["oracle_call"]), len(hist["val"]))
        self.assertEqual(len(hist["joint_call"]), len(hist["val"]))
        self.assertTrue(np.all(np.diff(hist["oracle_call"]) > 0))
        np.testing.assert_array_equal(hist["oracle_call"],
                                      hist["val_call"] + hist["grad_call"] - hist["joint_call"])

    def test_aggregate_uses_recorded_oracle_coordinates(self):
        repeats = []
        # Different joint fractions: summing marginal medians is not valid.
        for values, gradients, joint in [(9, 1, 1), (1, 9, 1), (8, 8, 8)]:
            repeats.append(dict(
                time=np.array([.1]), grad_call=np.array([gradients]),
                val_call=np.array([values]), grad_val_call=np.array([values + gradients]),
                joint_call=np.array([joint]), oracle_call=np.array([values + gradients - joint]),
                grad=np.array([1.]), val=np.array([1.]),
                summary=dict(elapsed_time=.1, grad_calls=gradients, val_calls=values,
                             grad_val_calls=values + gradients, joint_calls=joint,
                             oracle_calls=values + gradients - joint)))
        result = runner.aggregate_histories(repeats)
        np.testing.assert_array_equal(result["oracle_call"], [9])
        np.testing.assert_array_equal(result["oracle_calls_repeats"], [9, 9, 8])
        x, _ = plot._xy(result, "val", "oracle_call")
        np.testing.assert_array_equal(x, [9])

    def test_old_histories_are_stale_and_never_relabelled(self):
        args = runner.parse_args(["--algorithms", "gd"])
        config = runner.trial_config(args, runner.make_jobs(args, ["gd"])[0], 0)
        self.assertIn("optimizer.py", config["source_hashes"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text(json.dumps(dict(
                status="complete", config_hash=runner.config_hash(config))))
            old = {name: np.array([1.]) for name in
                   ("time", "grad_call", "val_call", "grad_val_call", "grad", "val")}
            np.savez(root / "history.npz", **old)
            self.assertEqual(runner.trial_status(root, config), "stale")
            with self.assertRaisesRegex(ValueError, "Rerun main.py"):
                plot._xy(old, "val", "oracle_call")

    def test_normal_plot_default_and_limits(self):
        with patch.object(sys, "argv", ["plot.py"]):
            self.assertEqual(plot.parse_args().xaxis, ["oracle_call"])
        with patch.object(sys, "argv", ["plot.py", "--vc-paper-compare-oracle-max", "100",
                                       "--vc-paper-sweep-oracle-max", "200"]):
            args = plot.parse_args()
            self.assertEqual((args.vc_paper_compare_oracle_max,
                              args.vc_paper_sweep_oracle_max), (100., 200.))


if __name__ == "__main__":
    unittest.main()
