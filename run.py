"""Run supplied experiments without changing numerical code or saved results.

Examples (from the repository root):
  python run.py smoke --suite all
  python run.py reproduce --suite maml --dry-run
  python run.py reproduce --suite adaptive --output-dir runs/adaptive-new
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROBLEMS = ('quad_arith', 'rosenbrock', 'dixon_price', 'powell', 'qing')


def build_jobs(mode: str, suite: str, output: Path, problems: list[str]) -> list[dict]:
    smoke = mode == 'smoke'
    suites = ('differentiable', 'maml', 'adaptive') if suite == 'all' else (suite,)
    jobs: list[dict] = []

    def add(name: str, script: str, args: list[str], x64: bool) -> None:
        path = ROOT / script
        jobs.append({'name': name, 'command': [sys.executable, str(path), *args],
                     'cwd': str(path.parent),
                     'environment': {'MPLBACKEND': 'Agg', 'JAX_PLATFORMS': 'cpu',
                                     'JAX_ENABLE_X64': 'true' if x64 else 'false'}})

    for current in suites:
        if current == 'differentiable':
            args = ['--output-dir', str(output / 'differentiable'), '--steps',
                    '20' if smoke else '5000', '--dim', '8' if smoke else '500',
                    '--seed', '7', '--alpha', '0.1', '--r', '0.5',
                    '--x0-radius', '0.5', '--gd-step-factor', '1.0',
                    '--nag-step-factor', '1.0', '--fd-epsilon', '1e-6']
            add('differentiable', 'VC_differentiable/diffable.py', args, True)
        elif current == 'maml':
            for steps in (5, 20):
                cfg = json.loads((ROOT / f'VC_MAML/result/inner_{steps}/base_config.json').read_text())
                grid = ['100'] if smoke else ['30', '100', '300']
                args = ['--output-dir', str(output / 'maml' / f'inner_{steps}'),
                        '--inner-steps', str(steps), '--time-budget-seconds',
                        '0.05' if smoke else str(cfg['time_budget_seconds']),
                        '--repeats', '1' if smoke else '10', '--seed', str(cfg['seed']),
                        '--seed-stride', '1', '--hidden-width', str(cfg['hidden_width']),
                        '--eval-tasks', '8' if smoke else str(cfg['eval_tasks']),
                        '--support-size', str(cfg['support_size']),
                        '--query-size', str(cfg['query_size']),
                        '--meta-batch-size', '2' if smoke else str(cfg['meta_batch_size']),
                        '--log-every', str(cfg['log_every']),
                        '--inner-vc-alpha', str(cfg['inner_vc_alpha']),
                        '--inner-vc-r', str(cfg['inner_vc_r']),
                        '--outer-adam-lr', str(cfg['outer_adam_lr']),
                        '--gd-L-grid', *grid, '--nag-L-grid', *grid,
                        '--vc-L-grid', *grid, '--vc-M-grid', *grid]
                add(f'maml-{steps}', 'VC_MAML/maml_code/benchmark_maml.py', args, False)
        elif current == 'adaptive':
            if smoke:
                add('adaptive-smoke', 'vc_adaptive/main.py',
                    ['--paper-compare', '--problem', 'powell', '--dim', '8', '--repeats', '1',
                     '--stop-mode', 'iter', '--n-iter', '20', '--result-dir',
                     str(output / 'adaptive' / 'powell'), '--plot', '--no-show'], True)
            else:
                for problem in problems:
                    for section in ('sweep', 'compare'):
                        cfg = json.loads((ROOT / f'vc_adaptive/result/{problem}/{section}/base_config.json').read_text())
                        args = [f'--paper-{section}', '--problem', problem, '--repeats', str(cfg['repeats']),
                                '--stop-mode', cfg['stop_mode'], '--time-limit', str(cfg['time_limit']),
                                '--n-iter', str(cfg['n_iter']), '--seed', str(cfg['seed']),
                                '--seed-stride', str(cfg['seed_stride']), '--L', str(cfg['L']),
                                '--sigma-x0', str(cfg['sigma_x0']), '--hmax', str(cfg['hmax']),
                                '--checkpoint-threshold', str(cfg['checkpoint_threshold']),
                                '--result-dir', str(output / 'adaptive' / problem / section),
                                '--plot', '--no-show']
                        if cfg['dim'] is not None:
                            args += ['--dim', str(cfg['dim'])]
                        add(f'adaptive-{problem}-{section}', 'vc_adaptive/main.py', args, True)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'reproduce'))
    parser.add_argument('--suite', choices=('all', 'maml', 'adaptive', 'differentiable'), default='all')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--problems', nargs='+', choices=PROBLEMS, default=list(PROBLEMS))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = (args.output_dir or ROOT / 'runs' / f'{args.mode}-{args.suite}').expanduser().resolve()
    jobs = build_jobs(args.mode, args.suite, output, args.problems)
    for job in jobs:
        print(f"[{job['name']}] {shlex.join(job['command'])}", flush=True)
    if args.dry_run:
        return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f'Refusing to overwrite a nonempty output directory: {output}. Choose a new --output-dir.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'commands.json').write_text(json.dumps(jobs, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/record_environment.py'),
                    '--output', str(output / 'actual_environment.json')], check=True)
    for job in jobs:
        env = os.environ.copy()
        env.update(job['environment'])
        subprocess.run(job['command'], cwd=job['cwd'], env=env, check=True)
    print(f'Finished. New results: {output}', flush=True)


if __name__ == '__main__':
    main()
