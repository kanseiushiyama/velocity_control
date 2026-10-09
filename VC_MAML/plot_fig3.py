"""Recreate the four-panel Fig. 3 from exported aggregate curves."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent


def plot(result_dir: Path | None = None, output_dir: Path | None = None) -> None:
    result_dir = (result_dir or ROOT / 'result').expanduser().resolve()
    target = (output_dir or ROOT.parent / 'runs' / 'maml-fig3-replot').expanduser().resolve()
    for protected in (ROOT / 'result', ROOT / 'figures', ROOT / 'maml_code'):
        if target.is_relative_to(protected.resolve()):
            raise ValueError(f'Output must not be inside bundled data or code: {target}')
    for name in ('fig3_maml.pdf', 'fig3_maml.png'):
        if (target / name).exists():
            raise FileExistsError(f'Refusing to overwrite {target / name}; choose another --output-dir.')
    for steps in (5, 20):
        for name in ('base_config.json', 'candidate_grid.json', 'best_hyperparameters.json', 'fig3_curves.npz'):
            path = result_dir / f'inner_{steps}' / name
            if not path.is_file():
                raise FileNotFoundError(f'Missing figure input: {path}')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    cmap = plt.get_cmap('tab10')
    colors = {30: cmap(0), 100: cmap(1), 300: cmap(2)}
    method_colors = dict(gd=cmap(0), nag=cmap(1), vc=cmap(2))
    widths = dict(gd=2.8, nag=1.4, vc=1.4)
    styles = {30: '--', 100: '-.', 300: ':'}
    for row, steps in enumerate([5, 20]):
        root = result_dir / f'inner_{steps}'
        candidates = json.loads((root / 'candidate_grid.json').read_text())
        best = json.loads((root / 'best_hyperparameters.json').read_text())
        with np.load(root / 'fig3_curves.npz', allow_pickle=False) as curves:
            for c in candidates:
                cid, method = c['candidate_id'], c['method']
                is_best = best[method]['candidate_id'] == cid
                label = f"{method.upper()} L={c['L']:g}"
                if method == 'vc':
                    label += f", M={c['M']:g}"
                if is_best:
                    label += ' (best)'
                x = curves[cid + '__time']
                y = curves[cid + '__median']
                linestyle = styles[c['M']] if method == 'vc' else '-'
                axes[row, 1].semilogy(x, y, color=colors[c['L']],
                    linestyle=linestyle, linewidth=widths[method],
                    alpha=1 if is_best else .5, label=label)
                if is_best:
                    axes[row, 0].semilogy(x, y, color=method_colors[method],
                        linestyle=linestyle, linewidth=widths[method], label=label)
                    axes[row, 0].fill_between(x, curves[cid+'__q25'], curves[cid+'__q75'],
                        color=method_colors[method], alpha=.15, linewidth=0)
        for col in range(2):
            ax = axes[row, col]
            budget = float(json.loads((root / 'base_config.json').read_text())['time_budget_seconds'])
            ax.set_xlim(-0.05*budget, 1.05*budget)
            ax.set_xlabel('execution time [s]')
            ax.set_ylabel('meta loss')
            ax.grid(True, which='both', alpha=.3)
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(
                base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.legend(loc='upper right', fontsize=8 if col == 0 else 6,
                      ncol=1 if col == 0 else 2, handlelength=3)
    fig.subplots_adjust(left=.085, right=.985, bottom=.075, top=.925,
                        hspace=.42, wspace=.24)
    fig.text(.535, .957, '5 iterations', ha='center', fontsize=17)
    fig.text(.535, .455, '20 iterations', ha='center', fontsize=17)
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / 'fig3_maml.pdf', bbox_inches='tight')
    fig.savefig(target / 'fig3_maml.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=ROOT / 'result',
                        help='Directory containing inner_5/ and inner_20/ curve arrays.')
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent / 'runs' / 'maml-fig3-replot',
                        help='New figure destination (the original figures are never overwritten).')
    args = parser.parse_args()
    try:
        plot(args.result_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f'Wrote fig3_maml.pdf and fig3_maml.png to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
