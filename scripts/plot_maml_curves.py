"""Plot four separate PNG panels from the bundled MAML interpolation arrays.

No training, raw trial histories, JAX import, or modification of saved data.
The supplied combined figure is preserved separately under VC_MAML/figures/.
"""
from __future__ import annotations
import argparse
import json
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=ROOT / 'VC_MAML/result')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/maml-replot')
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f'Refusing to overwrite {output}; choose an empty output directory.')
    sources = {}
    for steps in (5, 20):
        folder = args.result_dir / f'inner_{steps}'
        for name in ('candidate_grid.json', 'best_hyperparameters.json', 'fig3_curves.npz'):
            if not (folder / name).is_file():
                parser.error(f'Missing input: {folder / name}')
        candidates = json.loads((folder / 'candidate_grid.json').read_text())
        best = json.loads((folder / 'best_hyperparameters.json').read_text())
        with np.load(folder / 'fig3_curves.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for candidate in candidates:
            cid = candidate['candidate_id']
            for field in ('time', 'median', 'q25', 'q75'):
                key = f'{cid}__{field}'
                if key not in arrays or not np.isfinite(arrays[key]).all():
                    parser.error(f'Missing or nonfinite array: {key}')
        sources[steps] = (candidates, best, arrays)
    output.mkdir(parents=True, exist_ok=True)
    for steps, (candidates, best, arrays) in sources.items():
        for panel in ('best', 'all'):
            chosen = [c for c in candidates if panel == 'all' or c['candidate_id'] == best[c['method']]['candidate_id']]
            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            for candidate in chosen:
                cid, method = candidate['candidate_id'], candidate['method']
                label = f"{method.upper()} L={candidate['L']:g}"
                if method == 'vc':
                    label += f", M={candidate['M']:g}"
                linestyle = {'gd': '-', 'nag': '--', 'vc': '-.'}[method]
                x = arrays[f'{cid}__time']
                ax.plot(x, arrays[f'{cid}__median'], label=label, linestyle=linestyle)
                if panel == 'best':
                    ax.fill_between(x, arrays[f'{cid}__q25'], arrays[f'{cid}__q75'], alpha=0.15)
            ax.set_yscale('log')
            ax.set_xlabel('Cumulative optimizer time (s)')
            ax.set_ylabel('Evaluation meta-loss')
            ax.set_title(f'{steps} inner steps: {panel} configurations')
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7 if panel == 'all' else 9, ncol=2 if panel == 'all' else 1)
            fig.tight_layout()
            path = output / f'maml_{steps}_{panel}.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            print(path)


if __name__ == '__main__':
    main()
