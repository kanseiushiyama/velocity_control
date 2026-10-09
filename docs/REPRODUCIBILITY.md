# Reproducibility and package provenance

## Source and result preservation

This release layout was prepared from `VC_paper(1).zip`. The ten numerical, experiment-runner, and original plotting Python files are byte-identical to that upload. The three existing test modules are retained; only two axis-label expectations in `test_paper_mode.py` are adjusted from `Oracle calls` to `#oracle calls`, matching the unchanged plotter. No numerical assertions are weakened.

All supplied result data and figures are retained without modifying file bytes. The source archive includes a small subset (MAML summaries/curves, recorded configurations, and publication figures); `VC_paper_saved_results.zip` preserves every supplied file under a `result/` or `results/` directory. Original macOS metadata and Python bytecode/cache files are excluded.

`metadata/source_manifest.json` records source checksums and the test-only exception. `metadata/file_manifest.json` identifies unchanged bundled inputs and the location of each saved-result file. New helpers, documentation, reference-environment metadata, and the MIT license are packaging additions.

## Environment descriptions

The repository uses the MAML README's stated environment as its common reference: Python 3.14.7, JAX/jaxlib 0.11.1, NumPy 2.5.3, Matplotlib 3.11.1; Apple M3, 8 CPU cores, 24 GiB unified memory, macOS 26.6.2, arm64, CPU backend.

The 300 supplied MAML summary rows independently contain Python 3.14.7 and CPU. The package/hardware fields are transcribed from the supplied README; the old environment dump and full lock file it mentioned were not included. The new `environment.json` is clearly labeled as reference documentation, not as a recovered historical machine log. The historical environment of the other suites is not independently established by that MAML evidence.

The scikit-learn version was not provided. Do not describe the dependency file as a complete lock. `scripts/record_environment.py` records the actual machine/packages on a new run in a separate file and never edits `environment.json`.

The original numerical precision is preserved: MAML float32, other suites float64. `run.py` launches each suite in a separate process to avoid mixing these settings.

## What can be reproduced without training

The MAML interpolation archives contain time grids and interpolated run curves, not the original per-iteration histories. The newly supplied `VC_MAML/plot_fig3.py` regenerates the combined two-by-two Figure 3 from them. `scripts/plot_maml_curves.py` remains available for four separate PNG panels. The originally supplied combined figure and all stored arrays are preserved.

The author subsequently supplied the three previously missing helper scripts. They are now integrated under `VC_MAML/`, with main guards, safe output paths and explicit command-line arguments. Numerical benchmark sources and the independent reference calculation have not changed. The runner's complete original 300-trial argument list is retained; `--smoke` is a separately labeled six-trial test. Input hashes and exact patches are in `metadata/maml_helpers_manifest.json` and `metadata/maml_helpers.patch`.

The new `VC_MAML/run_fig3.py` runs the original benchmark, validates complete fresh histories, exports curves and generates Figure 3. It does not attempt to reconstruct missing historical histories from aggregate arrays. It records the executing machine under `actual_environment.json` in its new workspace, without changing the MAML reference environment description.

The larger archive contains adaptive aggregate NPZ files and benchmark summary tables, which the original adaptive plotter can read. Its missing per-trial directories cannot be reconstructed from those aggregates. Avoid `--status-only` on aggregate-only bundled directories: the runner expects trial directories and may regenerate aggregate outputs.

The differentiable bundle contains a metrics CSV and four figure PDFs, but no complete invocation/configuration record. `run.py reproduce --suite differentiable` uses current script defaults; it does not claim independent recovery of the missing historical command.

## Reruns and caching

The full MAML plan has 300 trial runs: 15 candidates x 10 seeds x 2 inner-step budgets. Budgets are 40 seconds (K=5) and 80 seconds (K=20) per candidate and seed. The VC output is the dyadic weighted average, as verified by the supplied source and summary labels.

The original MAML runner can reuse a trial when its output convention matches, even if other settings differ. The root wrapper refuses nonempty destinations to avoid that ambiguity. To resume intentionally, use the original CLI with exactly the same configuration and an absolute output path.

The adaptive runner retains its original source-hash compatibility checks. The supplied aggregates are labeled `benchmark-v9-oracle-qing-compensated`; they should not be mixed with earlier oracle accounting or uncompensated Qing runs. The public wrapper preserves these algorithms and reads the saved five-repeat/one-repeat configuration rather than silently changing the defaults in source.

## Attribution

The original source's comparison-method attribution comments are preserved. External dependencies are installed separately and retain their own licenses. The code's MIT license does not replace upstream notices or license terms for copied third-party material.
