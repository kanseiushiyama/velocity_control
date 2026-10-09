# Differentiable optimization experiment

`diffable.py` compares GD, NAG, and differentiable velocity control on the separable smooth objective used for the solution-sensitivity/Jacobian experiment. The supplied numerical file and figures are unchanged.

## Environment and precision

Use the [shared reference environment](../README.md#reference-experiment-environment), taken from the supplied MAML README: **Python 3.14.7, JAX/jaxlib 0.11.1, NumPy 2.5.3, Matplotlib 3.11.1; Apple M3 (8 CPU cores), 24 GiB unified memory, macOS 26.6.2, arm64, CPU backend**.

Unlike MAML, this script explicitly enables **float64**. That setting is preserved.

## Run from the repository root

```bash
python run.py smoke --suite differentiable
python run.py reproduce --suite differentiable --dry-run
python run.py reproduce --suite differentiable
```

The full-run wrapper uses the current defaults: dimension 500, 5,000 steps, seed 7, initial radius 0.5, alpha 0.1, r 0.5, and GD/NAG step factors 1.0. The finite-difference check uses epsilon 1e-6.

The uploaded experiment does not contain an original command record for the saved figures, so these defaults should not be interpreted as an independently verified reconstruction of the saved run. View all options with:

```bash
python VC_differentiable/diffable.py --help
```

Always supply a new output directory for a fresh run. The root wrapper handles this check and keeps the supplied `results/` figures untouched. `metrics.csv` is preserved in the separate saved-results archive.
