"""Fast regression tests for the restored Figure 3 orchestration interface."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MAML = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location('vc_fig3_runner_test', MAML / 'run_fig3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def value(command, flag):
    return command[command.index(flag) + 1]


class Figure3HelperTests(unittest.TestCase):
    def test_import_has_no_trial_execution(self):
        module = load_runner()
        self.assertTrue(callable(module.main))
        self.assertTrue(callable(module.validate_and_export))

    def test_full_plan_keeps_paper_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'new'
            jobs = load_runner().build_jobs(out)
            self.assertEqual(len(jobs), 2)
            self.assertEqual(sum(j['repeats'] * j['candidates'] for j in jobs), 300)
            self.assertEqual(sum(j['repeats'] * j['candidates'] * j['budget'] for j in jobs), 18000)
            for job, k, budget in zip(jobs, (5, 20), (40, 80)):
                cmd = job['command']
                expected = {'--inner-steps': str(k), '--time-budget-seconds': str(budget),
                            '--repeats': '10', '--seed': '0', '--seed-stride': '1',
                            '--hidden-width': '40', '--eval-tasks': '256',
                            '--support-size': '10', '--query-size': '10',
                            '--meta-batch-size': '16', '--log-every': '5',
                            '--inner-vc-alpha': '0.1', '--inner-vc-r': '0.5',
                            '--outer-adam-lr': '0.001'}
                for flag, expected_value in expected.items():
                    self.assertEqual(value(cmd, flag), expected_value)
                for flag in ('--gd-L-grid', '--nag-L-grid', '--vc-L-grid', '--vc-M-grid'):
                    pos = cmd.index(flag)
                    self.assertEqual(cmd[pos+1:pos+4], ['30', '100', '300'])
                self.assertEqual(value(cmd, '--output-dir'), str(out / 'result' / f'inner_{k}'))
            self.assertFalse(out.exists())

    def test_smoke_is_separately_labeled_small_plan(self):
        jobs = load_runner().build_jobs(Path('/unused'), smoke=True)
        self.assertEqual(sum(j['repeats'] * j['candidates'] for j in jobs), 6)
        for job in jobs:
            self.assertEqual(job['budget'], 0.05)
            self.assertEqual(value(job['command'], '--hidden-width'), '4')
            self.assertEqual(value(job['command'], '--meta-batch-size'), '2')

    def test_dry_run_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'new'
            result = subprocess.run([sys.executable, str(MAML / 'run_fig3.py'), '--dry-run', '--output-dir', str(output)],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('18000 s', result.stdout)
            self.assertFalse(output.exists())

    def test_runner_refuses_nonempty_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            marker = out / 'keep.json'
            marker.write_text('{"keep": true}')
            result = subprocess.run([sys.executable, str(MAML / 'run_fig3.py'), '--smoke', '--output-dir', temp],
                                    capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing to overwrite', result.stderr)
            self.assertEqual(json.loads(marker.read_text()), {'keep': True})
            self.assertEqual(list(out.iterdir()), [marker])

    def test_plot_refuses_existing_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            marker = out / 'fig3_maml.pdf'
            marker.write_bytes(b'preserve')
            result = subprocess.run([sys.executable, str(MAML / 'plot_fig3.py'), '--output-dir', temp],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing to overwrite', result.stderr)
            self.assertEqual(marker.read_bytes(), b'preserve')
            self.assertFalse((out / 'fig3_maml.png').exists())

    def test_plot_refuses_bundled_figure_directory(self):
        result = subprocess.run([sys.executable, str(MAML / 'plot_fig3.py'), '--output-dir', str(MAML / 'figures')],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output must not be inside bundled data or code', result.stderr)


if __name__ == '__main__':
    unittest.main()
