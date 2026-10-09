# Supplied MAML helper integration

## Scope

The author subsequently supplied the three scripts previously missing from the archive:
`check_weighted_output.py`, `plot_fig3.py`, and `run_fig3.py`.
They are integrated directly under `VC_MAML/`, next to `maml_code/`, not in the
repository-root `scripts/` directory. The older helpers remain available.

The input for this update is `VC_paper_GitHub_ready.zip` plus the three uploads.
The ten numerical/experiment/original plotting modules, all three existing adaptive
test modules, all saved data, and all supplied figures are unchanged relative to that ZIP.
The earlier test-label correction is retained; no new adaptive-test edit was made.
The MIT license and the MAML-based reference environment files are unchanged.
No files have been published to GitHub.

## Integration changes

- `check_weighted_output.py`: retains the original reference trajectory, averaging
  window calculation, all five K cases (2, 4, 5, 8, 20), comparison tolerances,
  meta-gradient check, and the K=1 rejection. A main guard prevents checks from running
  on import. JSON output defaults to `runs/weighted-output-check/weighted_output_checks.json`;
  `--output` can select another new file. CPU and float32 match the MAML convention.
- `plot_fig3.py`: preserves the uploaded two-by-two panel layout, line styles, colors,
  labels and figure sizes. `--result-dir` selects the aggregate inputs and `--output-dir`
  selects new PDF/PNG files. Default output is `runs/maml-fig3-replot/`; the bundled
  `VC_MAML/figures/` is protected. Time-axis limits follow the input configuration,
  giving the same 40/80-second limits for the supplied paper results.
- `run_fig3.py`: the full-run benchmark arguments are unchanged (300 trials;
  5 steps/40 seconds and 20 steps/80 seconds; the original grids and seeds).
  It still validates per-trial configurations, output conventions, finite histories,
  timing/oracle coordinates, repeat counts and best-candidate selection; aggregates
  still use the original benchmark function. It then generates the combined figure.
  A main guard, `--dry-run`, a separate six-trial `--smoke` preset, and `--output-dir`
  prevent accidental full runs and mixing with old results. Nonempty workspaces are
  rejected. Runtime machine records and pip freeze output go to the new workspace,
  never to the repository's reference environment file.
- The original hardcoded generated prose claiming execution on "this Mac" is replaced
  by an English results summary referring to the actual runtime metadata.
  Optional `--archive` packaging is retained with its own explicit flag and excludes
  raw trials/caches. Its standalone bundle includes the license and needed recorder.
- Updated the English READMEs, provenance notes and checksum verifier. Added seven
  lightweight helper-interface regression tests. Exact differences from the supplied
  scripts are recorded in `maml_helpers.patch`; upload/integrated hashes are recorded
  in `maml_helpers_manifest.json`.

## Commands from the repository root

```bash
# Integrity check and tests (no full paper rerun)
python scripts/verify_sources.py
python VC_MAML/check_weighted_output.py
python -m unittest discover -s VC_MAML/tests -v
python -m unittest discover -s vc_adaptive/tests -v

# Replot bundled curves; no optimization or large result archive required
python VC_MAML/plot_fig3.py

# Preview the complete original 300-trial command plan without execution
python VC_MAML/run_fig3.py --dry-run

# Tiny end-to-end test: execute, validate, export and draw
python VC_MAML/run_fig3.py --smoke

# Full experiment: about five hours of optimizer work, plus other overhead
python VC_MAML/run_fig3.py
```

Outputs default to new folders under `runs/`. Repeating a command with the same
existing output fails safely; choose another `--output` or `--output-dir`.
The previous `python run.py reproduce --suite maml` remains a valid benchmark-only
interface, whereas `VC_MAML/run_fig3.py` also performs the supplied validation/export
and combined-figure generation.

## Verification performed for this update

- All 16 existing adaptive tests passed, and all 7 new helper-interface tests passed.
- The supplied averaging/meta-gradient checks passed for K=2,4,5,8,20; K=1 is rejected.
  Its independent `reference` function is AST-identical to the upload.
- The complete full-run argument vectors for K=5 and K=20 were compared against
  the uploaded runner: identical except for the new output workspace.
- A small end-to-end run completed six trials (GD/NAG/VC at K=5,20), validated all
  histories, exported curves, generated the combined PDF/PNG and assembled the optional ZIP.
- The combined Figure 3 was regenerated from the bundled paper curves without training.
  Its PDF was rendered and visually inspected. In the same validation environment,
  its generated PNG was byte-identical to output from the unmodified uploaded plotter.
  This does not assert byte identity with the historical figure across library versions.
- The final standalone ZIP passed CRC validation, contained the license, and supported
  its documented dry-run and aggregate-replot commands.
- Safe refusal of existing outputs and import/dry-run behavior were tested.
- Input results and reference-environment files were checked for byte preservation.

These checks used Linux x86_64, Python 3.13.5, JAX/jaxlib 0.9.0.1, NumPy 2.3.5,
Matplotlib 3.10.8 and scikit-learn 1.8.0. They are not a rerun on the reported Mac.
The full 300-trial experiment and exact MAML reference environment installation
were NOT executed for this integration. The new uploads supply scripts, not the
still-missing historical raw trial directories or original machine/lock files.

## Distribution

Only the GitHub code ZIP needs replacement. Its top-level folder remains
`VC_paper_github_ready/`, so the previous `VC_paper_saved_results.zip` is compatible
and need not be replaced. Saved result bytes, datasets, and supplied plots were not
modified. New validation/replot files generated during this integration are not
substituted for the paper results in the distribution.
