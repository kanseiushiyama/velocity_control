"""Fixed-time MAML benchmark with direct hyperparameter grids.

Inner optimizers:
  GD:  step size = 1 / L, with L selected from --gd-L-grid
  NAG: step size = 1 / L, with L selected from --nag-L-grid
  VC:  (L_proxy, M_proxy) selected from --vc-L-grid x --vc-M-grid

Every candidate is run under the same main benchmark conditions. There is no
separate tuning phase. For each method, the candidate with the smallest median
final meta-query loss across repeats is reported as the best configuration.
VC uses the weighted inner output x_bar_K from Eq. (wt); GD and NAG use x_K.
"Final" refers to the end of outer meta-training, not the inner output rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from maml_benchmark_core import ExperimentConfig, INNER_OUTPUT

LABELS = {"gd": "GD", "nag": "NAG", "vc": "VC"}

METHOD_LINEWIDTH = {"gd": 2.8, "nag": 1.4, "vc": 1.4}


def build_plot_styles(candidates: list[dict[str, object]]):
    L_values = sorted({float(c["L"]) for c in candidates})
    cmap = plt.get_cmap("tab10")
    colors = {L: cmap(i % cmap.N) for i, L in enumerate(L_values)}
    method_colors = {
        "gd": cmap(0),
        "nag": cmap(1),
        "vc": cmap(2),
    }

    M_values = sorted({
        float(c["M"])
        for c in candidates
        if c["method"] == "vc" and c["M"] is not None
    })
    # Compact patterns so that the difference is visible even in the legend.
    vc_linestyles = [
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
        (0, (4, 1, 1, 1)),
    ]
    vc_styles = {
        M: vc_linestyles[i % len(vc_linestyles)]
        for i, M in enumerate(M_values)
    }
    return colors, method_colors, vc_styles


def candidate_label(candidate: dict[str, object], best: bool = False) -> str:
    method = str(candidate["method"])
    L = float(candidate["L"])
    if method == "vc":
        label = f"VC L={L:g}, M={float(candidate['M']):g}"
    else:
        label = f"{LABELS[method]} L={L:g}"
    return label + (" (best)" if best else "")


def style_line(
    line,
    candidate,
    colors,
    method_colors,
    vc_styles,
    alpha=1.0,
    best_plot: bool = False,
) -> None:
    method = str(candidate["method"])
    if best_plot:
        line.set_color(method_colors[method])
    else:
        line.set_color(colors[float(candidate["L"])])
    line.set_linewidth(METHOD_LINEWIDTH[method])
    line.set_alpha(alpha)
    if method == "vc":
        line.set_linestyle(vc_styles[float(candidate["M"])])
    else:
        line.set_linestyle("-")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-width", type=int, default=40)
    parser.add_argument("--eval-tasks", type=int, default=256)
    parser.add_argument("--support-size", type=int, default=10)
    parser.add_argument("--query-size", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--time-budget-seconds", type=float, default=30.0)
    parser.add_argument("--log-every", type=int, default=5)

    parser.add_argument(
        "--gd-L-grid", nargs="+", type=float, default=[100.0],
        help="L candidates for GD; the inner step size is 1/L.",
    )
    parser.add_argument(
        "--nag-L-grid", nargs="+", type=float, default=[100.0],
        help="L candidates for NAG; the inner step size is 1/L.",
    )
    parser.add_argument(
        "--vc-L-grid", nargs="+", type=float, default=[100.0],
        help="L_proxy candidates for VC.",
    )
    parser.add_argument(
        "--vc-M-grid", nargs="+", type=float, default=[100.0],
        help="M_proxy candidates for VC.",
    )

    parser.add_argument("--inner-vc-alpha", type=float, default=0.1)
    parser.add_argument("--inner-vc-r", type=float, default=0.5)
    parser.add_argument("--outer-adam-lr", type=float, default=1.0e-3)
    parser.add_argument("--meta-batch-size", type=int, default=16)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        seed=args.seed,
        hidden_width=args.hidden_width,
        eval_tasks=args.eval_tasks,
        support_size=args.support_size,
        query_size=args.query_size,
        inner_steps=args.inner_steps,
        time_budget_seconds=args.time_budget_seconds,
        log_every=args.log_every,
        inner_gd_L=float(args.gd_L_grid[0]),
        inner_nag_L=float(args.nag_L_grid[0]),
        inner_vc_alpha=args.inner_vc_alpha,
        inner_vc_r=args.inner_vc_r,
        inner_vc_L_proxy=float(args.vc_L_grid[0]),
        inner_vc_M_proxy=float(args.vc_M_grid[0]),
        outer_adam_lr=args.outer_adam_lr,
        meta_batch_size=args.meta_batch_size,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def make_candidates(args: argparse.Namespace, base: ExperimentConfig) -> list[dict[str, object]]:
    grids = [args.gd_L_grid, args.nag_L_grid, args.vc_L_grid, args.vc_M_grid]
    if any(any(v <= 0.0 for v in grid) for grid in grids):
        raise ValueError("All L and M grid values must be positive")

    candidates: list[dict[str, object]] = []
    for L in args.gd_L_grid:
        candidates.append({
            "candidate_id": f"gd_L_{slug(L)}",
            "method": "gd",
            "config": replace(base, inner_gd_L=float(L)),
            "L": float(L),
            "M": None,
        })
    for L in args.nag_L_grid:
        candidates.append({
            "candidate_id": f"nag_L_{slug(L)}",
            "method": "nag",
            "config": replace(base, inner_nag_L=float(L)),
            "L": float(L),
            "M": None,
        })
    for L in args.vc_L_grid:
        for M in args.vc_M_grid:
            candidates.append({
                "candidate_id": f"vc_L_{slug(L)}_M_{slug(M)}",
                "method": "vc",
                "config": replace(
                    base,
                    inner_vc_L_proxy=float(L),
                    inner_vc_M_proxy=float(M),
                ),
                "L": float(L),
                "M": float(M),
            })
    return candidates


def aggregate_curve(
    histories: list[list[dict[str, str]]], xkey: str, max_x: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    for rows in histories:
        x = np.asarray([float(r[xkey]) for r in rows])
        y = np.asarray([float(r["eval_query_loss"]) for r in rows])
        curves.append((x, y))
    end = min(x[-1] for x, _ in curves)
    if max_x is not None:
        end = min(end, max_x)
    points = max(2, int(end) + 1) if xkey == "outer_iteration" else 121
    grid = np.linspace(0.0, end, points)
    values = np.vstack([np.interp(grid, x, y) for x, y in curves])
    return grid, np.median(values, axis=0), np.quantile(values, 0.25, axis=0), np.quantile(values, 0.75, axis=0)


def save_curve_plot(
    output_dir: Path,
    candidates: list[dict[str, object]],
    best: dict[str, dict[str, object]],
    histories: dict[str, list[list[dict[str, str]]]],
    xkey: str,
    xlabel: str,
    filename: str,
    max_x: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    colors, method_colors, vc_styles = build_plot_styles(candidates)

    candidate_by_id = {str(c["candidate_id"]): c for c in candidates}
    for method in ("gd", "nag", "vc"):
        if method not in best:
            continue
        cid = str(best[method]["candidate_id"])
        candidate = candidate_by_id[cid]
        candidate_histories = histories.get(cid, [])
        if not candidate_histories:
            continue

        grid, median, q25, q75 = aggregate_curve(
            candidate_histories, xkey, max_x=max_x
        )
        line = ax.semilogy(
            grid, median, label=candidate_label(candidate, best=True)
        )[0]
        style_line(
            line, candidate, colors, method_colors, vc_styles,
            alpha=1.0, best_plot=True,
        )

        ax.fill_between(
            grid, q25, q75,
            alpha=0.15,
            color=line.get_color(),
            linewidth=0,
        )

    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
    )
    ax.yaxis.set_major_formatter(
        mticker.LogFormatterSciNotation(
            base=10.0, labelOnlyBase=False,
            minor_thresholds=(np.inf, np.inf),
        )
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("meta loss", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11, handlelength=3.0)
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_all_candidates_plot(
    output_dir: Path,
    candidates: list[dict[str, object]],
    best: dict[str, dict[str, object]],
    histories: dict[str, list[list[dict[str, str]]]],
    xkey: str,
    xlabel: str,
    filename: str,
    max_x: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    colors, method_colors, vc_styles = build_plot_styles(candidates)
    best_ids = {method: str(row["candidate_id"]) for method, row in best.items()}

    for candidate in candidates:
        cid = str(candidate["candidate_id"])
        method = str(candidate["method"])
        candidate_histories = histories.get(cid, [])
        if not candidate_histories:
            continue

        grid, median, _, _ = aggregate_curve(
            candidate_histories, xkey, max_x=max_x
        )
        is_best = best_ids.get(method) == cid
        line = ax.semilogy(
            grid, median, label=candidate_label(candidate, best=is_best)
        )[0]
        style_line(
            line, candidate, colors, method_colors, vc_styles,
            alpha=1.0 if is_best else 0.5,
            best_plot=False,
        )

    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
    )
    ax.yaxis.set_major_formatter(
        mticker.LogFormatterSciNotation(
            base=10.0, labelOnlyBase=False,
            minor_thresholds=(np.inf, np.inf),
        )
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("meta loss", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, handlelength=3.5)
    ax.tick_params(axis="both", which="major", labelsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_candidate_results(
    output_dir: Path,
    candidates: list[dict[str, object]],
    repeats: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[list[dict[str, str]]]]]:
    summaries: dict[str, list[dict[str, object]]] = {}
    histories: dict[str, list[list[dict[str, str]]]] = {}
    for candidate in candidates:
        cid = str(candidate["candidate_id"])
        summaries[cid] = []
        histories[cid] = []
        for repeat in range(repeats):
            trial_dir = output_dir / "trials" / f"repeat_{repeat:03d}" / cid
            summary_path = trial_dir / "summary.json"
            history_path = trial_dir / "history.csv"
            if summary_path.exists() and history_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                if summary.get("inner_output") != INNER_OUTPUT[str(candidate["method"])]:
                    continue
                summaries[cid].append(summary)
                histories[cid].append(read_csv(history_path))
    return summaries, histories


def select_best(
    output_dir: Path,
    candidates: list[dict[str, object]],
    summaries: dict[str, list[dict[str, object]]],
) -> dict[str, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        cid = str(candidate["candidate_id"])
        result_rows = summaries[cid]
        if not result_rows:
            continue
        losses = np.asarray([float(r["final_eval_query_loss"]) for r in result_rows])
        iterations = np.asarray([int(r["outer_iterations_completed"]) for r in result_rows])
        oracle_calls = np.asarray([int(r["oracle_calls_completed"]) for r in result_rows])
        rows.append({
            "method_id": candidate["method"],
            "inner_output": INNER_OUTPUT[str(candidate["method"])],
            "candidate_id": cid,
            "L": candidate["L"],
            "M": "" if candidate["M"] is None else candidate["M"],
            "n": len(result_rows),
            "final_loss_median": float(np.median(losses)),
            "final_loss_q25": float(np.quantile(losses, 0.25)),
            "final_loss_q75": float(np.quantile(losses, 0.75)),
            "outer_iterations_median": float(np.median(iterations)),
            "oracle_calls_median": float(np.median(oracle_calls)),
        })

    rows.sort(key=lambda r: (str(r["method_id"]), float(r["final_loss_median"])))
    write_csv(output_dir / "hyperparameter_grid_results.csv", rows)

    best: dict[str, dict[str, object]] = {}
    for method in ("gd", "nag", "vc"):
        method_rows = [r for r in rows if r["method_id"] == method]
        if method_rows:
            best[method] = min(method_rows, key=lambda r: float(r["final_loss_median"]))

    with (output_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best


def aggregate_best_results(
    output_dir: Path,
    candidates: list[dict[str, object]],
    best: dict[str, dict[str, object]],
    summaries: dict[str, list[dict[str, object]]],
    histories: dict[str, list[list[dict[str, str]]]],
    time_budget: float,
) -> None:
    save_curve_plot(
        output_dir, candidates, best, histories,
        "elapsed_seconds", "execution time [s]",
        "maml_time_median.pdf", max_x=time_budget,
    )
    save_curve_plot(
        output_dir, candidates, best, histories,
        "outer_iteration", "#outer iteration",
        "maml_iter_median.pdf",
    )
    save_curve_plot(
        output_dir, candidates, best, histories,
        "oracle_calls", "#oracle call",
        "maml_oracle_median.pdf",
    )

    save_all_candidates_plot(
        output_dir, candidates, best, histories,
        "elapsed_seconds", "execution time [s]",
        "maml_time_all_candidates.pdf", max_x=time_budget,
    )
    save_all_candidates_plot(
        output_dir, candidates, best, histories,
        "outer_iteration", "#outer iteration",
        "maml_iter_all_candidates.pdf",
    )
    save_all_candidates_plot(
        output_dir, candidates, best, histories,
        "oracle_calls", "#oracle call",
        "maml_oracle_all_candidates.pdf",
    )

    methods = [m for m in ("gd", "nag", "vc") if m in best]
    loss_data = [
        [float(s["final_eval_query_loss"]) for s in summaries[str(best[m]["candidate_id"])]]
        for m in methods
    ]
    plt.figure(figsize=(6.4, 4.3))
    plt.boxplot(loss_data)
    plt.xticks(np.arange(1, len(methods) + 1), [LABELS[m] for m in methods], fontsize=12)
    plt.yscale("log")
    plt.ylabel("final meta query loss", fontsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "maml_final_loss_boxplot.pdf", dpi=180, bbox_inches="tight")
    plt.close()

    print("\nBest configurations (selected on the same benchmark runs)")
    for method in methods:
        row = best[method]
        if method == "vc":
            params = f"L={float(row['L']):g}, M={float(row['M']):g}"
        else:
            params = f"L={float(row['L']):g} (step size={1.0/float(row['L']):g})"
        print(
            f"  {LABELS[method]:3s}: {params}; "
            f"median final loss={float(row['final_loss_median']):.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("maml_grid_benchmark"))
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    add_config_args(parser)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.time_budget_seconds <= 0.0:
        raise ValueError("--time-budget-seconds must be positive")
    if args.inner_steps < 2:
        raise ValueError("--inner-steps must be >= 2 for the VC weighted-average output")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base = config_from_args(args)
    candidates = make_candidates(args, base)

    with (output_dir / "base_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(base), f, indent=2)
    with (output_dir / "candidate_grid.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "candidate_id": c["candidate_id"],
                    "method": c["method"],
                    "L": c["L"],
                    "M": c["M"],
                }
                for c in candidates
            ],
            f,
            indent=2,
        )

    runner = Path(__file__).with_name("run_maml_trial.py")
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    rng = np.random.default_rng(base.seed + 91_337)

    for repeat in range(args.repeats):
        seed = base.seed + repeat * args.seed_stride
        jobs = list(candidates)
        rng.shuffle(jobs)
        print(f"\nRepeat {repeat + 1}/{args.repeats}, seed={seed}, candidates={len(jobs)}")

        for candidate in jobs:
            cid = str(candidate["candidate_id"])
            method = str(candidate["method"])
            trial_dir = output_dir / "trials" / f"repeat_{repeat:03d}" / cid
            summary_path = trial_dir / "summary.json"
            if summary_path.exists() and not args.overwrite:
                with summary_path.open("r", encoding="utf-8") as f:
                    saved_summary = json.load(f)
                if saved_summary.get("inner_output") == INNER_OUTPUT[method]:
                    print(f"  skip {cid}: completed")
                    continue
                print(f"  rerun {cid}: inner output rule changed")

            trial_dir.mkdir(parents=True, exist_ok=True)
            config = replace(candidate["config"], seed=seed)
            config_path = trial_dir / "config_input.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(config), f, indent=2)

            log_path = trial_dir / "process.log"
            cmd = [
                sys.executable, str(runner),
                "--method", method,
                "--candidate-id", cid,
                "--repeat-index", str(repeat),
                "--output-dir", str(trial_dir),
                "--config-json", str(config_path),
            ]
            print(f"  run  {cid}")
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    cmd, cwd=runner.parent, env=env,
                    stdout=log, stderr=subprocess.STDOUT, text=True,
                )
            if result.returncode != 0:
                message = f"Trial failed: repeat={repeat}, candidate={cid}; see {log_path}"
                if args.keep_going:
                    print("  WARNING:", message)
                else:
                    raise RuntimeError(message)

    summaries, histories = load_candidate_results(output_dir, candidates, args.repeats)
    best = select_best(output_dir, candidates, summaries)
    aggregate_best_results(
        output_dir, candidates, best, summaries, histories, base.time_budget_seconds
    )


if __name__ == "__main__":
    main()
