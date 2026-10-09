# MAML with weighted VC output

This experiment compares inner GD, NAG, and velocity control while training a shared neural-network initialization using outer Adam.

## Environment

The supplied rerun README reports **Python 3.14.7, JAX/jaxlib 0.11.1, NumPy 2.5.3, and Matplotlib 3.11.1**, on an **Apple M3 with 8 CPU cores and 24 GiB unified memory, macOS 26.6.2, arm64**, using the **CPU backend and float32**. This is the common reference environment documented in the [root README](../README.md#reference-experiment-environment) and [environment.json](../environment.json). The three numerical Python files and their four-package dependency pins are unchanged.

## Model and output convention

- Targets are noiseless sine functions `a*sin(x+phi)`, with independent `a ~ Uniform[0.1,5]` and `phi ~ Uniform[0,pi]`.
- The trainable regressor has one tanh hidden layer of width 40 and 121 parameters. The task's amplitude and phase are not the regressor parameters.
- Each task has 10 support and 10 query points, sampled from `[-5,5]`. The outer mini-batch contains 16 tasks. Evaluation uses 256 separate, fixed tasks within a run.
- The loss is half the mean squared error. Adam uses learning rate 0.001, beta1 0.9, beta2 0.999, and epsilon 1e-8.
- VC returns the paper's dyadic weighted output: average `x_tau` for `tau = 2**(floor(log2(K))-1), ..., K-1` with weights proportional to `exp(alpha*tau**(6/7))`. Thus K=5 uses `x_2,...,x_4`, and K=20 uses `x_8,...,x_19`. This output is used in both the meta-gradient and held-out evaluation. GD and NAG return `x_K`.

`maml_code/maml_benchmark_core.py` implements these choices. The control polynomial actually evaluated in the supplied code is `3*z**2 - 2*z**3`; an older docstring still calls it quintic/C2. The source has not been changed to alter either the algorithm or that historical comment.

## Run from the repository root

```bash
python scripts/check_weighted_output.py
python VC_MAML/check_weighted_output.py
python run.py smoke --suite maml
python run.py reproduce --suite maml --dry-run
python run.py reproduce --suite maml
```

The newly supplied orchestration pipeline also performs the trial validation, curve export and combined Figure 3 plotting:

```bash
python VC_MAML/run_fig3.py --dry-run
python VC_MAML/run_fig3.py --smoke
python VC_MAML/run_fig3.py
```

The dry run starts no experiments. The smoke preset is explicitly separate from the paper setting: width 4, 4 evaluation tasks, meta-batch size 2, one seed, L=M=100, and 0.05 seconds of optimizer time for each of GD/NAG/VC at K=5 and K=20 (six trials). The default full run uses the original supplied arguments unchanged. All new files are written below a fresh `runs/maml-fig3/` workspace (or `runs/maml-fig3-smoke/`). Specify `--output-dir` to choose another new workspace. Existing nonempty destinations are rejected, so archived results are never used as a trial cache.

The full run uses K=5 with 40 seconds and K=20 with 80 seconds for every candidate. Each setting has 10 repeats (seeds 0 through 9). GD and NAG each use L in `{30,100,300}`; VC uses all nine pairs of L and M in that set, with alpha 0.1 and r 0.5. The exact saved settings are in `result/inner_5/base_config.json` and `result/inner_20/base_config.json`.

For direct use, the original CLI is available:

```bash
python VC_MAML/maml_code/benchmark_maml.py --help
```

Use a **new absolute output path** when calling that CLI directly because its trial subprocess changes working directory. The root wrapper handles absolute paths and rejects nonempty output folders. The original cache check checks the output convention, but not every experiment setting; do not reuse an old result directory for a different configuration.

## Saved data and plotting

`result/inner_5/` and `result/inner_20/` include the configuration, 150 trial summaries each, candidate grid, best-configuration information, and `fig3_curves.npz`. The latter stores 121-point time grids, medians, quartiles, and ten interpolated run curves for each candidate.

The original combined figure is at `figures/maml.pdf` and `figures/fig3_maml.png`. The restored plotter generates a combined two-by-two PDF and PNG from the existing arrays, without retraining:

```bash
python VC_MAML/plot_fig3.py
```

Outputs: `runs/maml-fig3-replot/fig3_maml.pdf` and `fig3_maml.png`. The original figures are not modified. Input and output directories can be set explicitly, for example:

```bash
python VC_MAML/plot_fig3.py --result-dir runs/maml-fig3/result --output-dir runs/another-fig3
```

The plotter keeps the supplied panel layout, colors, line styles and labels; axis time limits follow the input configuration (40/80 seconds for the supplied paper data). For the four separate PNG panels added in the previous package, use:

```bash
python scripts/plot_maml_curves.py
```

Best configuration means the minimum median final query loss over the same ten runs; selection and reporting do not use different task sets. The final loss is measured at the end of outer training and does not change the weighted inner-output convention.

The author has now supplied `run_fig3.py`, `plot_fig3.py`, and `check_weighted_output.py`; the integrated copies are beside `maml_code/`. Their I/O and command-line handling have been adapted for safe repository use. `check_weighted_output.py` retains the supplied independent reference trajectory, K=2,4,5,8,20 cases, output/meta-gradient tolerances and rejection of K=1; its JSON report goes to `runs/weighted-output-check/` by default. The older `scripts/check_weighted_output.py` remains a separate full-Jacobian check.

Raw historical `trials/`, the original machine dump/full lock file, `commands.json`, and `validation.json` were still not included in the uploads. New runs record their own `actual_environment.json`, `requirements-lock.txt`, `commands.json`, source hashes and validation results inside the new workspace; these are NOT historical metadata. The provided summaries and curve arrays remain unchanged. For integration details, see [the helper update notes](../metadata/maml_helpers_update.md).

## Timing

Optimizer time includes task sampling, meta-gradient computation, Adam updates, and synchronization. It excludes JIT warmup and held-out evaluation. The recorded oracle coordinate is `outer_iterations * meta_batch_size * inner_steps`; it excludes higher-order reverse-mode work and evaluation. Fixed-time results vary with machine load and hardware even with identical seeds.
