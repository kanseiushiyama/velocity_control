# VC numerical experiments

JAX implementations of velocity control (VC) and comparison methods, with resumable trials, repeated-run aggregation, and publication plots. Use `main.py` to run and aggregate experiments and `plot.py` to plot saved aggregate results.

## Setup

Use the [shared reference environment](../README.md#reference-experiment-environment), following the supplied MAML README: **Python 3.14.7, JAX/jaxlib 0.11.1, NumPy 2.5.3, Matplotlib 3.11.1; Apple M3 (8 CPU cores), 24 GiB unified memory, macOS 26.6.2, arm64, CPU backend**.

From the repository root, install `requirements.txt`. It reuses the four MAML version pins and additionally installs `scikit-learn`, which the objective module imports. The supplied MAML information does not record a scikit-learn version, so that dependency is not assigned a historical version number.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The adaptive code explicitly enables **float64**, unlike the float32 MAML experiment. Keep this setting for these benchmarks. The six Python modules remain in their original directory. Unless otherwise stated, commands later in this README are run from `vc_adaptive/`.

### Match the bundled result configurations

The bundled comparison uses **five** repeats; the VC parameter sweep uses **one** repeat. The time budget is **50 seconds**, except **Qing at 15 seconds**. The original command-line defaults of ten repeats and 100 seconds are retained in the source, but are not the bundled experiment settings.

Use the root wrapper to read the saved configurations and write to new directories:

```bash
# Run these commands from the repository root.
python run.py reproduce --suite adaptive --dry-run
python run.py reproduce --suite adaptive
```

The ten-repeat workflow below is retained as an optional usage example and does not describe the bundled five-repeat comparison. Do not run aggregation commands over the bundled result directories; their raw trial directories are absent.

## Numerical evaluation and updating existing experiments

Qing now uses the verified compensated evaluation of `x_i*x_i - i` directly in
`obj_func.py`. It uses float64 split-product compensation and TwoSum, with a
custom JVP for the mathematical derivative. The Qing polynomial and VC
acceptance conditions are unchanged. Every method uses the same compensated
Qing oracle. Each evaluation performs more arithmetic but does not add an
extra oracle call. No arbitrary-precision calculation is used during optimization.

`optimizer.py` now includes the previously delivered joint-evaluation version:
ordinary/replay VC steps, predictive steps, and checkpoints evaluate the r=1
candidate with `value_and_grad`. No separate `optimizer_joint.py` replacement
is needed. The ordinary step still performs its original line-search value
checks; that work remains charged in addition to the later joint evaluation.
The general M formula is unchanged. This reduces a reproduced Qing rounding
problem but does not guarantee elimination of every finite-precision plateau.

The experiment version has changed, and the configuration hash now includes
hashes of the numerical source files. Existing trials from the earlier code
are marked stale and rerun by `main.py`. Old histories without `oracle_call`
cannot be converted reliably from `grad_val_call` alone; `plot.py` gives a clear
error instead of relabeling those counts. Rerun all methods for a fair Qing
comparison. Use a new result directory to keep the earlier results separately.

```bash
python main.py --problem qing --algorithms ours_defer_cp --R 0.5 \
  --repeats 1 --time-limit 10 --result-dir result/qing_compensated --plot --no-show
```

## Optional workflow: one sweep run and ten comparison runs

Run the sweep first, run the comparison second, and then generate all four figures:

```bash
# Figure 4: one run for each of the nine VC parameter combinations.
python main.py --paper-sweep --problem powell --repeats 1 \
  --result-dir result/paper/powell

# Figures 1--3: ten runs for each comparison method.
python main.py --paper-compare --problem powell --repeats 10 \
  --result-dir result/paper/powell

# Generate the four PDFs from the final aggregate results.
python plot.py --paper --problem powell \
  --result-dir result/paper/powell --no-show
```

This produces the following repeat counts:

| Experiment | Completed trials per parameter setting |
| --- | ---: |
| Comparison baselines | 10 |
| VC at `R=0.5, alpha=0.1`, shared by the comparison and sweep | 10 |
| Predictive VC at `R=0.5, alpha=0.1` | 10 |
| The other eight VC sweep combinations | 1 each |

The shared VC job reuses its completed first trial from the sweep and runs only the remaining nine trials in the comparison phase. This requires matching trial settings and no `--overwrite`. Keep the same problem, dimension, stopping settings, seed settings, and VC parameters in both commands.

**Run sweep before comparison when sharing a result directory.** Each invocation rebuilds aggregate NPZ files for its selected jobs using its requested repeat count. Running the one-repeat sweep after the comparison would replace the shared VC aggregate with a one-repeat result. The individual trial files remain available; rerun the comparison command with `--status-only` to restore the ten-repeat aggregate without recomputing the trials.

**`--paper` does not automatically use different repeat counts for comparison and sweep.** All paper flags default to ten repeats, including `--paper-sweep`. Supplying `--paper-compare --paper-sweep` in one command selects both sets with a single common `--repeats` value. Use the two separate commands above for the mixed repeat counts.

| Problems | Unique trials with the recommended workflow | Trials with `--paper` defaults |
| --- | ---: | ---: |
| `quad_arith`, `powell` | 68 per problem | 140 per problem |
| `rosenbrock`, `dixon_price`, `qing` | 58 per problem | 130 per problem |

These totals assume all requested trials complete and the shared first trial is reused.

### Run all five problems

```bash
for problem in quad_arith rosenbrock dixon_price powell qing; do
  python main.py --paper-sweep --problem "$problem" --repeats 1 \
    --result-dir "result/paper/$problem"
  python main.py --paper-compare --problem "$problem" --repeats 10 \
    --result-dir "result/paper/$problem"
  python plot.py --paper --problem "$problem" \
    --result-dir "result/paper/$problem" --no-show
done
```

### Inspect settings or run a small example

```bash
# Print planned jobs and settings without running or writing files.
python main.py --paper-sweep --problem powell --repeats 1 --dry-run
python main.py --paper-compare --problem powell --repeats 10 --dry-run

# Small example: one repeat for all jobs, followed by all four plots.
python main.py --paper --problem powell --dim 8 --repeats 1 \
  --stop-mode iter --n-iter 40 --result-dir result/smoke --plot --no-show
```

`--dry-run` describes the requested jobs; it does not inspect completion status. Its total includes already completed jobs. Use `--status-only` to inspect saved trial status and reaggregate results.

## Paper figures and settings

All four paper figures plot the objective value. The comparison uses `R=0.5, alpha=0.1`. The sweep uses all nine combinations of `R` in `{0.25, 0.5, 0.75}` and `alpha` in `{0.05, 0.1, 0.5}`.

| Figure | Horizontal axis | VC variants |
| --- | --- | --- |
| 1 | Gradient evaluations | VC and Predictive VC at the comparison parameters |
| 2 | Oracle calls (joint value/gradient = 1) | Same as Figure 1 |
| 3 | Measured optimization time | Same as Figure 1 |
| 4 | Oracle calls (joint value/gradient = 1) | VC at the nine sweep combinations |

| Paper legend | Algorithm ID | Implementation |
| --- | --- | --- |
| VC | `ours` | `AdaVeloControl`, with `parallel` |
| Predictive VC | `ours_defer_cp` | `AdaVeloControlDeferredRCheckpoint`, with `parallel` |

The comparison also includes GD, restart HB, and restart AGD. It adds NAG-SC for `quad_arith` and adaptive NAG-C for `powell`. All VC variants used in the paper figures use the parallel line-search mode. The plotting code reads the parallel results explicitly; it does not substitute short-circuit results for missing paper data.

In the command line and configuration, `R` is the fixed intermediate control value denoted by `r` in the paper. The optimizer's recorded `r` history contains the selected values denoted by `r_hat_t` in the paper, taking values in `{1, R, 0}`.

Predictive VC combines deferred evaluation of the intermediate candidate with checkpointing. Its default prediction threshold is ten consecutive selections of `r_hat_t=1`. During prediction, the selected candidate's upper-model condition and M-test are checked at every iteration. Checkpoints additionally reconstruct and certify the zero-control fallback. A failed prediction triggers rollback and non-predictive replay. The checkpoint schedule uses streak lengths `T + j^2` after the threshold `T`.

### Joint evaluation in ordinary and predictive steps

All VC variants use `jax.value_and_grad` for the r=1 candidate in ordinary and
replay steps. The joint value is used for M and cached with its gradient if
r=1 is selected. Line-search value evaluations are still performed and counted.
Prediction and checkpoint kernels reuse the joint pair for their upper-model
certificate, M-test, and accepted state.

For `ours_defer_cp`, the following counts apply when the certificates and
M-test pass. The ordinary row assumes no backtracking and selection of r=1.
Initialization and output diagnostics are not included in these per-step rows.

| Step type | Function evaluations | Gradient evaluations | Joint evaluations | Oracle calls |
| --- | ---: | ---: | ---: | ---: |
| Ordinary or replay step with defer-R | 3 | 1 | 1 | 3 |
| Predictive step between checkpoints | 1 | 1 | 1 | 1 |
| Predictive checkpoint with defer-R | 2 | 1 | 1 | 2 |

Without defer-R, an ordinary successful r=1 step costs four oracle calls, and
an `ours_cp` predictive checkpoint costs three. A failed predictive attempt
still costs its joint evaluation, plus any additional candidate certificates
that were actually evaluated before failure. Rolled-back work remains charged.

### Paper defaults

| Setting | Default |
| --- | --- |
| Stop mode | `time` |
| Measured time limit per trial | 100 seconds |
| Repeats in any paper mode | 10; explicitly use `--repeats 1` for the sweep |
| Checkpoint threshold | 10 |
| VC squared-step cap, `hmax` | 1 |
| Quadratic curvature parameter, `L` | 10,000 |
| Initial perturbation scale, `sigma-x0` | 1 |
| Base seed / seed stride | 0 / 0 |

A zero seed stride repeats the same deterministic problem instance and initial point, allowing timing variability to be measured. With `--seed-stride 1`, repeat `i` uses seed `seed + i`, where `i` starts at zero. The quadratic problem has a deterministic initial point and does not use the model seed. Method order is shuffled within each repeat unless `--fixed-method-order` is specified.

The parameter grid and paper legends are shared between the runner and plotter through `experiment_config.py`.

The default time limit is 100 seconds in both normal and paper modes. Use `--time-limit 300` to request the earlier time budget; trials from an earlier numerical implementation still need to be rerun. Parallel and short-circuit VC trials have separate job IDs and filenames.

## Problems and algorithms

| Problem ID | Objective | Default dimension | Additional comparison baseline |
| --- | --- | ---: | --- |
| `quad_arith` | Quadratic with an arithmetic sequence of eigenvalues | 1,000,000 | NAG-SC |
| `rosenbrock` | Rosenbrock | 1,000,000 | None |
| `dixon_price` | Dixon-Price | 1,000,000 | None |
| `powell` | Powell | 1,000,000 | NAG-C |
| `qing` | Qing | 100,000 | None |

Powell requires a dimension divisible by four. `--dim` overrides the default for the selected problem. Non-quadratic initial points are Gaussian perturbations of the reference optimum, scaled by `--sigma-x0`.

| Algorithm ID | Method |
| --- | --- |
| `auto` | GD, HB, AGD, and VC; additionally NAG-SC for the quadratic and NAG-C for Powell |
| `gd` | Gradient descent |
| `hb` | Adaptive restart HB |
| `agd` | Adaptive restart AGD |
| `nag` | NAG-SC; supported by the runner only for `quad_arith` |
| `nagc` | Adaptive NAG-C; supported by the runner only for `powell` |
| `ours` | VC |
| `ours_cp` | VC with checkpointing |
| `ours_defer_r` | VC with deferred intermediate-candidate evaluation |
| `ours_defer_cp` | VC with both deferred evaluation and checkpointing |

Use `auto` on its own, or provide an explicit list of algorithm IDs. In normal mode, the runner executes the Cartesian product of the requested `R`, `alpha`, and line-search modes for each VC variant. Baselines do not have these parameter combinations.

## Resuming and reaggregating

Repeating a command skips trials whose histories and completion metadata match the requested configuration. Missing or incomplete trials run from the beginning. Completed trials with a different configuration are marked `stale` and rerun. This is trial-level resumption; an interrupted individual trial does not resume from an optimizer checkpoint.

For the recommended paper workflow, repeat the sweep and comparison commands in the same order. To inspect status and rebuild aggregates without running trials:

```bash
python main.py --paper-sweep --problem powell --repeats 1 \
  --result-dir result/paper/powell --status-only
python main.py --paper-compare --problem powell --repeats 10 \
  --result-dir result/paper/powell --status-only
```

Use the original dimension, stopping settings, seed settings, checkpoint threshold, and other trial settings. Increasing `--repeats` extends the requested repeat indices and reuses matching completed indices. Decreasing it narrows the aggregation range without deleting individual trial files.

`--status-only` writes configuration and aggregate files, so the sweep-before-comparison order still matters. After the comparison phase, the root CSV summaries and `base_config.json` describe that phase's requested jobs. Sweep-only trial files and their aggregate NPZ files remain available, but the final root CSV files are not a combined report of both phases.

Do not use `main.py --paper --status-only` after the mixed-repeat workflow unless you intend to request ten repeats for every sweep setting. Use the phase-specific commands above to retain the intended requested counts.

Use separate result directories for different problems, dimensions, and experimental conditions. Aggregate filenames do not encode every setting. Missing data cause the plotter to warn and omit the corresponding curve; plotting does not verify trial configuration or the requested repeat count.

## Normal-mode examples

Without paper flags, arbitrary supported algorithms and parameter combinations can be selected:

```bash
python main.py --problem powell \
  --algorithms gd hb agd ours ours_defer_cp \
  --R 0.5 --alpha 0.1 --line-search-mode parallel short_circuit \
  --repeats 10 --result-dir result/custom --plot --no-show

python plot.py --problem powell --mode val \
  --xaxis oracle_call time \
  --algorithms gd hb agd ours ours_defer_cp \
  --R 0.5 --alpha 0.1 --line-search-mode parallel short_circuit \
  --result-dir result/custom --no-show
```

`main.py --plot` in normal mode and standalone `plot.py` both default to gradient norms and objective values against **oracle calls** (two PDFs). Use `plot.py --xaxis oracle_call time` to include time plots. The legacy `grad_call` and `grad_val_call` axes remain available explicitly.

## Complete `main.py` option reference

Arguments shown with `...` accept one or more space-separated values. Boolean flags are disabled unless supplied. The defaults below are the effective defaults after preset resolution; some parser help entries initially display `None`.

### Experiment selection and VC parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--problem NAME` | `quad_arith` | Run one problem from the problem table above. |
| `--algorithms ID ...` | `auto` | Select algorithms from the algorithm table. Cannot be supplied in paper mode. |
| `--R FLOAT ...` | `0.25 0.5 0.75` in normal mode | Fixed intermediate control values for VC variants. Use values in `(0, 1)`. Cannot be supplied in paper mode. |
| `--alpha FLOAT ...` | `0.1` in normal mode | Positive VC schedule parameters. Cannot be supplied in paper mode. |
| `--hmax FLOAT` | `1.0` | Positive cap on the VC squared-step quantity `h_t^2`. Applies to all VC variants. |
| `--checkpoint-threshold INT` | `10` | Positive number of consecutive `r_hat_t=1` selections needed to enter prediction. Applies to `ours_cp` and `ours_defer_cp`. |
| `--line-search-mode MODE ...` | `parallel` in normal mode | Select `parallel`, `short_circuit`, or both for VC variants. Paper mode fixes this to `parallel` and rejects an explicit value. |

### Paper presets

| Option | Default | Meaning |
| --- | --- | --- |
| `--paper`, `--vc-paper-mode` | Disabled | Select the union of comparison and sweep jobs, unless a phase selector restricts it. Shared jobs are deduplicated. All selected jobs use the same repeat count. |
| `--paper-compare`, `--vc-paper-compare` | Disabled | Select the jobs for Figures 1--3 and implicitly enable paper mode. |
| `--paper-sweep`, `--vc-paper-sweep` | Disabled | Select the nine VC jobs for Figure 4 and implicitly enable paper mode. Its repeat default is 10, not 1. |

Supplying both phase selectors selects both phases. `--paper` together with only one phase selector selects that phase only. Presets fix the algorithms, `R`, `alpha`, and line-search mode. Dimension, budgets, repeats, seeds, `hmax`, and checkpoint threshold remain configurable. Figure generation requires `--plot` or a subsequent call to `plot.py`.

### Budgets and problem parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--stop-mode {time,iter}` | `time` | Stop using measured optimization time or the gradient-evaluation budget. |
| `--time-limit FLOAT` | `100.0` | Time budget in seconds for each individual trial; used in `time` mode. It is not a limit on the entire command. |
| `--n-iter INT` | `17500` | Gradient-evaluation budget in `iter` mode, despite the option name. It is not an exact number of optimizer updates. |
| `--repeats INT` | `1` in normal mode; `10` in every paper mode | Positive number of repeat indices to run and aggregate for every selected job. |
| `--dim INT` | Problem-specific | Override the dimension. Powell requires a multiple of four. |
| `--L FLOAT` | `10000` | Largest eigenvalue of the quadratic objective, whose smallest eigenvalue is 1. Does not set the adaptive line search's initial estimate; unused by the other problem factories. |
| `--sigma-x0 FLOAT` | `1.0` | Scale of the Gaussian perturbation of the initial point for non-quadratic problems; unused by the quadratic factory. |
| `--seed INT` | `0` | Base seed for the problem instance and method-order randomization. |
| `--seed-stride INT` | `0` | Repeat `i` uses model seed `seed + i * seed_stride`. Zero keeps the initial point unchanged across repeats. |

Stopping is checked at optimizer control points. A step or predictive batch can overshoot the requested budget. Warmup, process startup, diagnostics, and file output also affect end-to-end runtime.

### Output, resumption, and inspection

| Option | Default | Meaning |
| --- | --- | --- |
| `--result-dir PATH` | `./result` | Directory for trial histories, configuration, aggregate NPZ files, CSV summaries, and default plots. |
| `--overwrite` | Disabled | Rerun selected trials even if they are already complete and match the configuration. |
| `--keep-going` | Disabled | Continue with other trials after a trial subprocess fails. Inspect the failed trial's `process.log`. |
| `--fixed-method-order` | Disabled | Preserve generated job order instead of shuffling jobs within each repeat. |
| `--status-only` | Disabled | Inspect trial status and reaggregate matching completed trials without running new trials. Still writes directories, configuration, and aggregate files. |
| `--dry-run` | Disabled | Print resolved settings, selected jobs, and requested trial count; do not run trials, inspect existing trial status, or write files. |
| `--plot` | Disabled | Plot after aggregation. Paper mode plots only the selected paper figures; normal mode produces the two oracle-call plots described above. Also applies with `--status-only`. |
| `--no-show` | Disabled | With `--plot`, save PDFs without opening the plot window. Has no effect when plotting is not requested. |
| `-h`, `--help` | — | Print command-line help and exit. |

## Complete `plot.py` option reference

`plot.py` reads existing aggregate NPZ files. It does not run experiments or aggregate individual trials, and it has no repeat-count option.

### Data selection and normal plots

| Option | Default | Meaning |
| --- | --- | --- |
| `--problem NAME ...` | `quad_arith` | Plot one or more supported problems. All inputs must be in the selected result directory. |
| `--mode {grad,val} ...` | `grad val` | Plot gradient norm and/or objective value in normal mode. Paper mode always uses objective values. |
| `--xaxis {oracle_call,time,grad_call,grad_val_call,iter} ...` | `oracle_call` | Horizontal axes for normal plots. `oracle_call` counts a joint value/gradient evaluation once. `grad_val_call` is the legacy sum and counts it twice. `iter` is the zero-based saved-history index, not necessarily the optimizer iteration number. Paper axes are fixed. |
| `--algorithms ID ...` | `auto` | Algorithms for normal plots. Paper mode uses its fixed comparison and sweep variants. |
| `--R FLOAT ...` | `0.25 0.5 0.75` | VC parameter values to load for normal plots. Paper mode uses the fixed paper grid. |
| `--alpha FLOAT ...` | `0.1` | VC schedule parameters to load for normal plots. Paper mode uses the fixed paper grid. |
| `--line-search-mode MODE ...` | `parallel` | Load `parallel`, `short_circuit`, or both for normal plots. Paper mode always loads `parallel`. |
| `--checkpoint-threshold INT` | `10` | Checkpoint threshold encoded in the result filename. Must match the experiment, including for Predictive VC in paper comparisons. |
| `--no-control-markers` | Disabled | Suppress gray control markers in normal plots. Normal plots otherwise show them when exactly one line-search mode is selected. Does not affect paper plots. |

In `plot.py`, paper mode ignores `--mode`, `--xaxis`, `--algorithms`, `--R`, `--alpha`, `--line-search-mode`, and `--no-control-markers`, rather than rejecting them. `--problem` and `--checkpoint-threshold` still apply. Use the dedicated paper options below to adjust the figure selection and axes. For normal plots against `grad_call`, a requested parallel VC result may fall back to a short-circuit result if the parallel file is absent. This fallback does not apply to paper plots.

### Axis limits for normal plots

| Option | Default | Meaning |
| --- | --- | --- |
| `--time-max FLOAT` | Automatic | Upper limit of the time axis in seconds. |
| `--grad-call-max FLOAT` | Automatic | Upper limit of the gradient-evaluation axis. |
| `--grad-val-call-max FLOAT` | Automatic | Upper limit of the legacy combined gradient/function-evaluation axis. |
| `--oracle-call-max FLOAT` | Automatic | Upper limit of the oracle-call axis (joint evaluation = 1). |
| `--iter-max FLOAT` | Automatic | Upper limit of the saved-history index axis. |

These five options apply only to normal plots. Automatic limits are based on the plotted data, with a small margin; axes start at zero.

### Paper plots and their axis limits

| Option | Default | Meaning |
| --- | --- | --- |
| `--paper`, `--vc-paper-mode` | Disabled | Generate all four paper figures, unless phase selectors restrict the selection. |
| `--paper-compare`, `--vc-paper-compare` | Disabled | Generate Figures 1--3; implicitly enable paper mode. |
| `--paper-sweep`, `--vc-paper-sweep` | Disabled | Generate Figure 4; implicitly enable paper mode. |
| `--vc-paper-compare-grad-max FLOAT` | Automatic | Upper gradient-evaluation limit for Figure 1. |
| `--vc-paper-compare-oracle-max FLOAT` | Automatic | Upper oracle-call limit for Figure 2. `--vc-paper-compare-gradval-max` remains an alias, now in oracle-call units. |
| `--vc-paper-compare-time-max FLOAT` | Automatic | Upper time limit in seconds for Figure 3. |
| `--vc-paper-sweep-oracle-max FLOAT` | Automatic | Upper oracle-call limit for Figure 4. `--vc-paper-sweep-grad-max` remains an alias, now in oracle-call units. |

Both phase selectors together generate all four figures. Paper comparison plots show gray control markers when available; the sweep has no control markers. All legends are placed at the upper right.

```bash
python plot.py --paper --problem powell \
  --result-dir result/paper/powell --no-show \
  --vc-paper-compare-time-max 100 \
  --vc-paper-compare-grad-max 20000 \
  --vc-paper-compare-oracle-max 100000 \
  --vc-paper-sweep-oracle-max 20000
```

### Files and display

| Option | Default | Meaning |
| --- | --- | --- |
| `--result-dir PATH` | `./result` | Directory containing aggregate NPZ files to read. |
| `--fig-dir PATH` | `RESULT_DIR/fig` | Output directory for PDFs. Existing files with the same names are overwritten. |
| `--no-show` | Disabled | Save PDFs without opening a plot window. |
| `-h`, `--help` | — | Print command-line help and exit. |

## Internal runner: `run_trial.py`

`main.py` launches this script in a fresh process for each trial. Normal use does not require invoking it directly.

| Option | Default | Meaning |
| --- | --- | --- |
| `--config-json PATH` | Required | Read a trial configuration JSON generated by `main.py`. |
| `--output-dir PATH` | Required | Write this trial's `history.npz` and `summary.json` to the specified directory. |
| `-h`, `--help` | — | Print command-line help and exit. |

Algorithm selection and numerical settings are supplied through the JSON file, not separate `run_trial.py` options. The parent `main.py` process captures trial output in `process.log`.

## Saved results and measurement conventions

| Path under `RESULT_DIR` | Contents |
| --- | --- |
| `base_config.json` | Resolved settings and job list for the latest invocation |
| `trials/PROBLEM/repeat_NNN/JOB_ID/config.json` | Exact configuration of one trial |
| `trials/PROBLEM/repeat_NNN/JOB_ID/process.log` | Trial subprocess output and errors |
| `trials/PROBLEM/repeat_NNN/JOB_ID/history.npz` | Individual trial history |
| `trials/PROBLEM/repeat_NNN/JOB_ID/summary.json` | Completion metadata, final statistics, and configuration hash |
| `*.npz` | Aggregate history for each selected method/parameter combination |
| `benchmark_summary.csv` | Per-trial statistics for matching completed trials in the latest aggregation |
| `benchmark_stats.csv` | Requested/completed counts, timing statistics, and evaluation-count medians for the selected jobs |
| `fig/PROBLEM_vc_compare_val_grad_call.pdf` | Figure 1 |
| `fig/PROBLEM_vc_compare_val_oracle_call.pdf` | Figure 2 |
| `fig/PROBLEM_vc_compare_val_time.pdf` | Figure 3 |
| `fig/PROBLEM_vc_sweep_val_oracle_call.pdf` | Figure 4 |

An aggregate built from one repeat contains that repeat's curve. For multiple repeats, histories are truncated to their common length, and coordinate-wise medians are taken at each saved-history position. Time/evaluation coordinates and objective/gradient coordinates are aggregated separately. Curves are not interpolated onto common time or evaluation grids. Individual repeat histories are also retained in the aggregate NPZ, padded with NaNs where needed. The current plots do not display interquartile bands.

VC histories evaluate the weighted output; predictive VC records diagnostics at ordinary steps and predictive batch endpoints. Consequently, the saved-history index can differ from the optimizer iteration count. Gray `+` markers indicate recorded `r_hat_t < 1`. If control histories differ between repeats, the markers use the first completed repeat's history.

JAX warmup is performed before optimization timing and resets oracle counters. Each value-only evaluation, gradient-only evaluation, or joint function-and-gradient evaluation costs **one oracle call**:

```text
oracle_calls = val_calls + grad_calls - joint_calls
```

The component counters retain their previous definitions: joint evaluation adds one to both `val_calls` and `grad_calls`. `grad_val_calls` remains their legacy sum. A separately computed value and gradient therefore cost two oracle calls; they are not merged after the fact. Parallel evaluation at three candidate points costs three calls, and speculative batching counts every attempted point, not one call per batch.

Individual and aggregate NPZ files contain `joint_call` and `oracle_call` histories. Trial summaries and CSV reports include `joint_calls` and `oracle_calls`; aggregates also contain per-repeat histories and final counters. Rolled-back histories are truncated, but the evaluation counters are never rolled back: failed predictions, rejected line-search trials, and replay all remain charged.

The new axis retains the existing measurement scope: compilation/warmup and output-point diagnostic evaluations made with `count=False` are excluded. This includes VC's weighted-output diagnostics, AGD's reported average, and NAG's reporting evaluations. HB's averaged-point evaluations are part of its update and are counted. Thus this is an optimization-work count under the existing experiment convention, not an all-inclusive count of every evaluation used for reporting or output selection. Recorded optimization time is distinct from end-to-end runtime.

## Source files and checks

| File | Role |
| --- | --- |
| `experiment_config.py` | Problem specifications, shared paper settings, and result filenames |
| `main.py` | Job construction, subprocess execution, resumption, and aggregation |
| `run_trial.py` | Execution and saving of one trial |
| `optimizer.py` | Optimization methods, evaluation counting, and checkpoint logic |
| `obj_func.py` | Objective functions, initial points, and oracle wrapper |
| `plot.py` | Normal and paper plots |
| `tests/test_paper_mode.py` | Regression checks for paper job coverage and plot data selection |
| `tests/test_predictive_evaluation.py` | Checks ordinary/predictive/checkpoint state reuse, component counts, and rejected attempts |
| `tests/test_oracle_accounting.py` | Checks joint accounting, rollback costs, aggregation, and legacy-data handling |

```bash
python -m unittest discover -s tests -v
```
