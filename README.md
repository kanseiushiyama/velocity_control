# Restart-free acceleration for nonconvex optimization

Experimental code accompanying **A Restart-Free Accelerated Algorithm for Non-Convex Minimization: Continuous and Discrete Analysis**, by **Kansei Ushiyama and Shun Sato**.

The repository contains the differentiable velocity-control method (VC), the MAML experiment, and the adaptive VC benchmarks, including Predictive VC and comparison methods. The numerical source files from the supplied experiment bundle are preserved.

## Repository layout

```text
VC_MAML/
  check_weighted_output.py  Supplied independent averaging/meta-gradient check
  run_fig3.py               Full run, validation and combined Figure 3
  plot_fig3.py              Combined Figure 3 from saved curves, no training
  maml_code/           Inner GD/NAG/VC and outer Adam benchmark
  result/              Recorded configurations, summaries and reusable curves
  figures/             Supplied combined MAML figure (PDF and PNG)
VC_differentiable/
  diffable.py          Jacobian and solution-sensitivity experiment
  results/             Supplied PDF figures
vc_adaptive/
  *.py                 Adaptive methods, objectives, runner and plotter
  tests/               Existing unit tests
  result/              Recorded configurations for each problem
  figures/             Supplied publication figures
scripts/               Environment, source-integrity and MAML checks
run.py                 Safe entry point for small tests and full experiments
requirements.txt       Shared MAML package pins plus the adaptive dependency
environment.json      Reference environment and its provenance
```

See [MAML details](VC_MAML/README.md), [the differentiable experiment](VC_differentiable/README.md), [the adaptive runner](vc_adaptive/README.md), and [data/provenance notes](docs/REPRODUCIBILITY.md).

## Reference experiment environment

The environment description throughout this repository follows the supplied MAML README. The 300 MAML trial-summary rows also record Python 3.14.7 and the CPU backend.

| Component | Recorded MAML environment |
| --- | --- |
| Python | 3.14.7 |
| JAX / jaxlib | 0.11.1 / 0.11.1 |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.1 |
| Processor | Apple M3, 8 CPU cores |
| Unified memory | 24 GiB |
| Operating system | macOS 26.6.2 |
| Architecture | arm64 |
| JAX backend | CPU |

The numerical precision remains **float32 for MAML** and **float64 for the differentiable and adaptive experiments**, following their implementations. The common environment description does not change those computational settings.

`scikit-learn` is additionally required by the adaptive objective-module imports. Its original version was not recorded in the supplied MAML environment, so it is intentionally not assigned a version pin. `requirements.txt` is therefore not a complete environment lock. See [environment.json](environment.json) for provenance and scope.

## Installation

From the repository root, with Python 3.14.7 installed:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/record_environment.py --check-reference
```

For MAML alone, the four pinned packages in `VC_MAML/maml_code/requirements.txt` are sufficient. The original package pins are retained as supplied; availability of those exact builds and reproduction on the recorded Mac are separate from the packaging checks.

Keep each experiment in a separate Python process. Do not import the MAML and float64 benchmark modules into the same notebook kernel when making precision-sensitive comparisons.

## Quick checks

```bash
python scripts/verify_sources.py
python scripts/check_weighted_output.py
python VC_MAML/check_weighted_output.py
python -m unittest discover -s VC_MAML/tests -v
python -m unittest discover -s vc_adaptive/tests -v
python run.py smoke --suite all
```

The smoke test uses small dimensions and short budgets, not the paper settings. JIT compilation may take longer than its timed optimization. New results go to `runs/smoke-all/`; a nonempty output directory is rejected to prevent accidental overwrite. Use `--output-dir runs/another-name` for another run.

## Run the experiments

Inspect a complete command plan without executing it:

```bash
python run.py reproduce --suite all --dry-run
```

Run one suite:

```bash
python run.py reproduce --suite maml
python run.py reproduce --suite differentiable
python run.py reproduce --suite adaptive
```

For one adaptive problem:

```bash
python run.py reproduce --suite adaptive --problems qing --output-dir runs/qing-new
```

The wrapper launches the original programs, records the exact commands and the actual local environment, and writes only to a new output directory. It selects the CPU backend and each suite's original precision. It does not rewrite the bundled configurations, results, or algorithm updates.

| Suite | Settings used by the wrapper |
| --- | --- |
| MAML | 5 inner steps / 40 seconds and 20 inner steps / 80 seconds; 15 candidates and 10 seeds per setting |
| Differentiable experiment | Current script defaults: dimension 500, 5,000 steps, seed 7, alpha 0.1, r 0.5 |
| Adaptive comparison | Five repeats per method; 50 seconds, except Qing at 15 seconds |
| Adaptive VC sweep | One repeat per parameter pair; the same problem-specific time budgets |

For the restored Figure 3 pipeline (run, validate all trials, export curves and draw the combined PDF/PNG), use:

```bash
python VC_MAML/run_fig3.py --dry-run
python VC_MAML/run_fig3.py --smoke
python VC_MAML/run_fig3.py
```

The first command is a preview only. The second is a six-trial smoke test; the third uses the complete paper settings. Results go to `runs/maml-fig3-smoke/` and `runs/maml-fig3/`, respectively. Both reject a nonempty destination. Use `--output-dir runs/a-new-name` for another execution. `--archive` optionally writes a separate MAML results bundle (raw `trials/` are excluded). The existing root `run.py` remains available for suite-by-suite execution, but does not perform the supplied Figure 3 validation/export/combined-plot steps.

The MAML suite alone requires **five hours of cumulative optimizer time**, plus compilation, evaluation, and process overhead. Its VC output is the **weighted average**, not the last iterate. GD and NAG return their last inner iterates.

Adaptive problem sizes are 1,000,000, except Qing at 100,000. The wrapper reads the saved adaptive configurations rather than silently using the runner's default ten repeats and 100-second limit. The differentiable experiment does not include an original command/configuration record; the wrapper's full-run settings are the current script defaults, not a claim that the saved figures were independently reproduced.

## View or replot the supplied results

The original figure files are included unchanged. Small MAML tables and reusable curves are included directly; the larger saved results are distributed in **`VC_paper_saved_results.zip`** alongside the code archive.

Extract the saved-results archive beside this repository folder. It uses the same `VC_paper_github_ready/` root and restores the existing `result/` and `results/` paths. Overlapping files are identical. See [the data notes](docs/REPRODUCIBILITY.md) before rerunning a benchmark.

The original three Figure 3 helper scripts have now been supplied and integrated. To regenerate the author's combined, two-by-two Figure 3 directly from the bundled arrays, without training:

```bash
python VC_MAML/plot_fig3.py
```

It writes `runs/maml-fig3-replot/fig3_maml.pdf` and `fig3_maml.png`; existing files are not overwritten. The new runner uses the same plotting code with its own result/output paths. The supplied line styles, colors, labels and panel layout are retained. A byte-identical historical PDF is not promised across library versions.

The previously added helper is also retained for four separate PNG panels:

```bash
python scripts/plot_maml_curves.py
```

The original combined figure remains unchanged at `VC_MAML/figures/maml.pdf`.

After restoring the larger results, use the original adaptive plotter, for example:

```bash
python vc_adaptive/plot.py --paper-compare --problem qing --result-dir vc_adaptive/result/qing/compare --fig-dir runs/qing-replot/compare --no-show
python vc_adaptive/plot.py --paper-sweep --problem qing --result-dir vc_adaptive/result/qing/sweep --fig-dir runs/qing-replot/sweep --no-show
```

Individual MAML trial histories were not present in the supplied archive. The interpolation arrays support time-curve plotting, but they cannot recover omitted raw histories or serve as a completed-trial cache. A full rerun creates fresh trial histories in `runs/`.

## Timing and reported metrics

MAML measures cumulative optimizer time: task sampling, meta-gradient computation, Adam updates, and synchronization. JIT warmup and periodic evaluation are excluded. Evaluation uses a separate fixed set of 256 tasks within each run. Configurations are selected by median final evaluation query loss over the same ten runs, not by an independent post-selection test set.

MAML's `oracle_calls` field counts `outer_iterations * meta_batch_size * inner_steps`; it does not count higher-order reverse-mode work or held-out evaluation. It is not the same accounting convention as the adaptive benchmark, which counts each value-only, gradient-only, or joint value-and-gradient request as one oracle call. See the suite-specific documentation before comparing axes.

Fixed-time results depend on hardware, load, and numerical-library versions. Matching seeds does not by itself imply identical elapsed-time curves.

## Citation and license

Please cite the accompanying paper when using this code in academic work. [CITATION.cff](CITATION.cff) contains the paper title and author information. Add the public repository URL and any final publication DOI when they are available.

The original source code is distributed under the [MIT License](LICENSE). Third-party code and installed dependencies remain subject to their own licenses. Upstream attribution comments in the supplied source are retained. The citation request is separate from the MIT license conditions.
