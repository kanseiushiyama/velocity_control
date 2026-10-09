"""Run, validate and plot the supplied MAML Figure 3 experiment.

The default is the paper setting: 300 trials, with 40/80 seconds of optimizer
work for K=5/20 (five hours in total, excluding compilation and evaluation).
Use --dry-run to inspect the commands, or --smoke for a small integration test.
All generated files go into a NEW workspace; bundled results are not reused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'maml_code'
REPOSITORY = ROOT.parent


def build_jobs(workspace: Path, *, smoke: bool = False) -> list[dict]:
    """Keep the original full-run arguments; smoke is a separate test preset."""
    jobs = []
    for steps in (5, 20):
        budget = 0.05 if smoke else (40 if steps == 5 else 80)
        repeats = 1 if smoke else 10
        grid = ['100'] if smoke else ['30', '100', '300']
        out = workspace / 'result' / f'inner_{steps}'
        argv = ['--repeats', str(repeats), '--seed', '0', '--seed-stride', '1',
                '--inner-steps', str(steps), '--time-budget-seconds', str(budget),
                '--hidden-width', '4' if smoke else '40',
                '--eval-tasks', '4' if smoke else '256',
                '--support-size', '10', '--query-size', '10',
                '--meta-batch-size', '2' if smoke else '16', '--log-every', '5',
                '--inner-vc-alpha', '0.1', '--inner-vc-r', '0.5', '--outer-adam-lr', '0.001',
                '--gd-L-grid', *grid, '--nag-L-grid', *grid,
                '--vc-L-grid', *grid, '--vc-M-grid', *grid,
                '--output-dir', str(out)]
        jobs.append(dict(inner_steps=steps, budget=budget, repeats=repeats,
                         candidates=3 if smoke else 15,
                         command=[sys.executable, '-u', str(CODE / 'benchmark_maml.py'), *argv]))
    return jobs


def write_status(workspace: Path, started: float, **kwargs) -> None:
    path = workspace / 'execution_status.tmp'
    path.write_text(json.dumps(dict(started=started, updated=time.time(), **kwargs), indent=2) + '\n')
    path.replace(workspace / 'execution_status.json')


def validate_and_export(steps: int, workspace: Path, *, smoke: bool = False) -> dict:
    """Apply the supplied configuration/history checks and curve aggregation."""
    import numpy as np
    import benchmark_maml as bench

    repeats = 1 if smoke else 10
    candidate_count = 3 if smoke else 15
    out = workspace / 'result' / f'inner_{steps}'
    candidates = json.loads((out / 'candidate_grid.json').read_text())
    assert len(candidates) == candidate_count
    configs = json.loads((out / 'base_config.json').read_text())
    budget = 0.05 if smoke else (40 if steps == 5 else 80)
    assert configs['time_budget_seconds'] == budget
    summaries, histories = bench.load_candidate_results(out, candidates, repeats)
    curves = {}
    summary_rows = []
    for c in candidates:
        cid = c['candidate_id']
        assert len(summaries[cid]) == len(histories[cid]) == repeats, cid
        for repeat, (s, h) in enumerate(zip(summaries[cid], histories[cid])):
            trial = out / 'trials' / f'repeat_{repeat:03d}' / cid
            cfg = json.loads((trial / 'config.json').read_text())
            expected = dict(configs, seed=repeat)
            if c['method'] in ['gd','nag']:
                expected[f"inner_{c['method']}_L"] = c['L']
            else:
                expected.update(inner_vc_L_proxy=c['L'], inner_vc_M_proxy=c['M'])
            assert cfg == expected, trial
            assert s['seed'] == repeat and s['candidate_id'] == cid
            assert s['elapsed_seconds'] >= budget and s['time_budget_seconds'] == budget
            assert s['inner_output'] == ('weighted_average_wt' if c['method']=='vc' else 'last_iterate')
            a = np.asarray([[float(r[k]) for k in ('elapsed_seconds','eval_query_loss','outer_iteration','oracle_calls')] for r in h])
            assert np.isfinite(a).all() and (a[:,1] > 0).all(), trial
            assert np.all(np.diff(a[:,0]) >= 0) and a[0,0] == 0, trial
            assert np.all(a[:,3] == a[:,2]*configs['meta_batch_size']*steps), trial
            summary_rows.append(s)
        grid, median, q25, q75 = bench.aggregate_curve(histories[cid], 'elapsed_seconds', max_x=budget)
        for suffix, data in [('time',grid),('median',median),('q25',q25),('q75',q75)]:
            curves[cid+'__'+suffix] = data
        curves[cid+'__interpolated_runs'] = np.asarray([
            np.interp(grid, [float(r['elapsed_seconds']) for r in h], [float(r['eval_query_loss']) for r in h])
            for h in histories[cid]])
    np.savez_compressed(out / 'fig3_curves.npz', **curves)
    fields = list(dict.fromkeys(k for s in summary_rows for k in s))
    with (out / 'all_trial_summaries.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary_rows)
    rows = bench.read_csv(out / 'hyperparameter_grid_results.csv')
    assert len(rows) == candidate_count and all(int(r['n']) == repeats for r in rows)
    best = json.loads((out / 'best_hyperparameters.json').read_text())
    for method in ['gd','nag','vc']:
        expected_best = min((c for c in candidates if c['method']==method),
            key=lambda c: np.median([s['final_eval_query_loss'] for s in summaries[c['candidate_id']]]))
        assert best[method]['candidate_id'] == expected_best['candidate_id']
    return dict(inner_steps=steps, trials=candidate_count*repeats, candidates=candidate_count, repeats=repeats, time_budget_seconds=budget, best=best)


def write_results_readme(workspace: Path, jobs: list[dict], smoke: bool) -> None:
    count = sum(job['candidates'] * job['repeats'] for job in jobs)
    lines = [
        '# MAML Figure 3 run', '',
        '**Small smoke test; not a paper experiment.**' if smoke else 'Fresh execution of the supplied paper settings.', '',
        f'{count} trials completed. VC uses the dyadic weighted average; GD/NAG use their last inner iterate.',
        'The three benchmark source files were not changed by this orchestration script.', '',
        '## Settings', '',
    ]
    for job in jobs:
        lines.append(f"- K={job['inner_steps']}: {job['candidates']} candidates x {job['repeats']} repeats x {job['budget']} seconds of optimizer time.")
    lines += [
        '', 'The precise arguments are recorded in commands.json.',
        'actual_environment.json and requirements-lock.txt describe THIS run, not a recovered historical environment.',
        'The backend is CPU and MAML uses float32. JIT warmup and held-out evaluation are excluded from optimizer time.',
        '', '## Outputs', '',
        'figures/fig3_maml.pdf and figures/fig3_maml.png contain the supplied two-by-two layout.',
        'Left panels: median and 25th--75th percentiles for the selected candidate.',
        'Right panels: medians for all tested candidates.',
        'Selection minimizes median final held-out query loss on the same runs; no independent post-selection test set is used.',
        'Curves use the benchmark aggregate_curve function and its 121-point linear interpolation grid.',
        'result/inner_5 and result/inner_20 include curve arrays, all-trial summaries and full trial histories.',
        'validation.json records the checks of configurations, output conventions, histories and selection.',
        '', '## Optional archive', '',
        'With --archive, maml_weighted_fig3_github_bundle.zip contains code, aggregates, figures and run metadata.',
        'The optional archive excludes trials/ and caches; full histories remain in this workspace.',
        'Nothing is uploaded to GitHub by this script.', '',
    ]
    (workspace / 'RESULTS_README.md').write_text('\n'.join(lines), encoding='utf-8')


def create_archive(workspace: Path) -> Path:
    """Create the optional standalone MAML bundle without the heavy trials."""
    members = []
    for path in CODE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.txt'):
            members.append((path, Path('maml_code') / path.name))
    for name in ('run_fig3.py', 'plot_fig3.py', 'check_weighted_output.py'):
        members.append((ROOT / name, Path(name)))
    # Include the shared recorder needed by this updated runner in standalone form.
    recorder = REPOSITORY / 'scripts' / 'record_environment.py'
    if not recorder.is_file():
        recorder = ROOT / 'record_environment.py'
    members.append((recorder, Path('record_environment.py')))
    license_path = REPOSITORY / 'LICENSE'
    if not license_path.is_file():
        license_path = ROOT / 'LICENSE'
    if license_path.is_file():
        members.append((license_path, Path('LICENSE')))
    for path in sorted(workspace.rglob('*')):
        rel = path.relative_to(workspace)
        if not path.is_file() or any(part in ('trials', '__pycache__', 'matplotlib_cache') for part in rel.parts):
            continue
        if path.suffix == '.zip':
            continue
        members.append((path, rel))
    archive = workspace / 'maml_weighted_fig3_github_bundle.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('maml_fig3/README.md', (
            '# MAML experiment bundle\n\n'
            'Run commands from this extracted directory. Install dependencies from '
            '`maml_code/requirements.txt`. The library pins are the supplied reference '
            'environment, whereas `actual_environment.json` records this run.\n\n'
            'Replot aggregates (no training):\n'
            '```bash\npython plot_fig3.py --output-dir runs/replot\n```\n\n'
            'Check weighted output and its meta-gradient:\n'
            '```bash\npython check_weighted_output.py --output runs/check.json\n```\n\n'
            'Preview or rerun the experiment into a new workspace:\n'
            '```bash\npython run_fig3.py --dry-run --output-dir runs/new\n'
            'python run_fig3.py --smoke --output-dir runs/smoke\n'
            'python run_fig3.py --output-dir runs/new\n```\n\n'
            'The final command uses five hours of cumulative optimizer time plus overhead. '
            'See RESULTS_README.md for the settings and results in this bundle. '
            'Individual trial histories are excluded from the ZIP but retained in '
            'the workspace that created it.\n'
        ))
        for path, rel in members:
            z.write(path, 'maml_fig3/' + rel.as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive CRC check failed')
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, help='New workspace (default: runs/maml-fig3 or runs/maml-fig3-smoke).')
    parser.add_argument('--dry-run', action='store_true', help='Show the full command plan without creating files or executing trials.')
    parser.add_argument('--smoke', action='store_true', help='Run a tiny six-trial integration test, not a paper experiment.')
    parser.add_argument('--archive', action='store_true', help='Also build a standalone ZIP of this run, excluding raw trials.')
    args = parser.parse_args()
    workspace = (args.output_dir or REPOSITORY / 'runs' / ('maml-fig3-smoke' if args.smoke else 'maml-fig3')).expanduser().resolve()
    jobs = build_jobs(workspace, smoke=args.smoke)
    for job in jobs:
        print(f"[K={job['inner_steps']}] {shlex.join(job['command'])}", flush=True)
    optimizer_seconds = sum(job['candidates'] * job['repeats'] * job['budget'] for job in jobs)
    print(f'Nominal cumulative optimizer time: {optimizer_seconds:g} s (compilation/evaluation excluded).', flush=True)
    print(f'Output workspace: {workspace}', flush=True)
    if args.dry_run:
        return
    if not __debug__:
        parser.error('Validation uses assertions; run Python without -O or -OO.')
    if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
        parser.error(f'Refusing to overwrite a nonempty output directory: {workspace}. Choose a new --output-dir.')
    if not (CODE / 'benchmark_maml.py').is_file():
        parser.error(f'Missing benchmark code under {CODE}')
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(workspace / 'matplotlib_cache'),
                      PYTHONUNBUFFERED='1', JAX_ENABLE_X64='False', JAX_PLATFORMS='cpu')
    sys.path.insert(0, str(CODE))
    started = time.time()
    validations = []
    try:
        write_status(workspace, started, state='starting', mode='smoke' if args.smoke else 'paper')
        (workspace / 'commands.json').write_text(json.dumps([job['command'] for job in jobs], indent=2) + '\n')
        (workspace / 'source_hashes.json').write_text(json.dumps(
            {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(CODE.glob('*.py'))}, indent=2) + '\n')
        recorder = REPOSITORY / 'scripts' / 'record_environment.py'
        if not recorder.is_file():
            recorder = ROOT / 'record_environment.py'  # Optional standalone archive.
        subprocess.run([sys.executable, str(recorder), '--output', str(workspace / 'actual_environment.json')], check=True)
        with (workspace / 'requirements-lock.txt').open('w') as handle:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)
        for job in jobs:
            steps = job['inner_steps']
            log = workspace / f'inner_{steps}.log'
            write_status(workspace, started, state='running', inner_steps=steps, log=log.name)
            print(f"START K={steps}: {job['candidates']} candidates x {job['repeats']} repeats x {job['budget']} s", flush=True)
            with log.open('w') as handle:
                subprocess.run(job['command'], cwd=CODE, stdout=handle, stderr=subprocess.STDOUT, check=True)
            validations.append(validate_and_export(steps, workspace, smoke=args.smoke))
            (workspace / 'validation.json').write_text(json.dumps(validations, indent=2) + '\n')
            print(f'DONE K={steps}: trial histories validated and aggregate curves exported.', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'plot_fig3.py'), '--result-dir', str(workspace / 'result'),
                        '--output-dir', str(workspace / 'figures')], check=True)
        write_results_readme(workspace, jobs, args.smoke)
        write_status(workspace, started, state='complete', trials=sum(v['trials'] for v in validations),
                     elapsed_s=time.time() - started, mode='smoke' if args.smoke else 'paper')
        if args.archive:
            print(f'Created {create_archive(workspace)}', flush=True)
        print(f'Complete. Results, checks and Figure 3: {workspace}', flush=True)
    except BaseException as exc:
        write_status(workspace, started, state='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
