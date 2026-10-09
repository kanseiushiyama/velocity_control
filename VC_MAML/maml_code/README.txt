MAML weighted-output benchmark

Environment from the supplied MAML rerun
Python 3.14.7; JAX/jaxlib 0.11.1; NumPy 2.5.3; Matplotlib 3.11.1.
Apple M3 (8 CPU cores), 24 GiB unified memory, macOS 26.6.2, arm64.
JAX CPU backend, float32.
The three numerical Python files and requirements.txt are unchanged.

From the repository root:
  python3.14 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python scripts/check_weighted_output.py
  python VC_MAML/check_weighted_output.py
  python run.py smoke --suite maml
  python run.py reproduce --suite maml --dry-run
  python run.py reproduce --suite maml
  python VC_MAML/plot_fig3.py
  python VC_MAML/run_fig3.py --dry-run
  python VC_MAML/run_fig3.py --smoke
  python VC_MAML/run_fig3.py

See ../README.md for model, output, timing, and saved-data details.
See ../../README.md and ../../environment.json for the common environment.
The full benchmark uses K=5 / 40 seconds and K=20 / 80 seconds,
15 candidates, 10 repeats, and the VC weighted-average output.
The previously missing three helpers are now included beside maml_code/.
The restored Figure 3 runner validates new trials, exports aggregates and plots
the combined PDF/PNG. New outputs default to runs/; bundled files are preserved.
Raw historical trial directories and original machine/lock files are still absent.
