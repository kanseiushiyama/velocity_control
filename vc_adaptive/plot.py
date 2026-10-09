import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from experiment_config import (
    VC_COMPARE_ALPHA, VC_COMPARE_R, VC_PAPER_COMPARE_ALGORITHMS,
    VC_PAPER_LABELS, VC_PAPER_LINE_SEARCH_MODE, VC_SWEEP_ALPHAS, VC_SWEEP_RS,
    default_algorithms, problem_choices, result_filename,
)


LABELS = {
    "gd": "GD",
    "hb": "HB",
    "agd": "AGD",
    "nag": "NAG-SC",
    "nagc": "NAG-C",
}

LINESTYLES = {
    "gd": "-.",
    "hb": "-.",
    "agd": "-.",
    "nag": "--",
    "nagc": "--",
}

VC_ALPHA_COLORS = {0.05: "#1f77b4", 0.1: "#ff7f0e", 0.5: "#2ca02c"}
VC_R_LINESTYLES = {0.25: ":", 0.5: "-", 0.75: "--"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot numerical experiments for the velocity-control paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--problem", nargs="+", choices=problem_choices(), default=["quad_arith"])
    parser.add_argument("--mode", nargs="+", choices=("grad", "val"), default=["grad", "val"])
    parser.add_argument(
        "--xaxis",
        nargs="+",
        choices=("oracle_call", "time", "grad_call", "grad_val_call", "iter"),
        default=["oracle_call"],
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("auto", "gd", "hb", "agd", "ours", "ours_cp", "ours_defer_r", "ours_defer_cp", "nag", "nagc"),
        default=["auto"],
        help="'auto' includes NAG-SC for quad_arith and NAG-C for Powell",
    )
    parser.add_argument("--R", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.1])
    parser.add_argument("--checkpoint-threshold", type=int, default=10)
    parser.add_argument(
        "--line-search-mode",
        nargs="+",
        choices=("parallel", "short_circuit"),
        default=["parallel"],
    )
    parser.add_argument("--result-dir", type=Path, default=Path("./result"))
    parser.add_argument("--fig-dir", type=Path, default=None, help="default: RESULT_DIR/fig")
    parser.add_argument("--no-control-markers", action="store_true")

    parser.add_argument("--time-max", type=float, default=None)
    parser.add_argument("--grad-call-max", type=float, default=None)
    parser.add_argument("--grad-val-call-max", type=float, default=None)
    parser.add_argument("--oracle-call-max", type=float, default=None)
    parser.add_argument("--iter-max", type=float, default=None)

    parser.add_argument("--vc-paper-mode", "--paper", action="store_true")
    parser.add_argument("--vc-paper-compare", "--paper-compare", action="store_true")
    parser.add_argument("--vc-paper-sweep", "--paper-sweep", action="store_true")
    parser.add_argument("--vc-paper-compare-grad-max", type=float, default=None)
    parser.add_argument("--vc-paper-compare-oracle-max", "--vc-paper-compare-gradval-max",
                        type=float, default=None, help="oracle-call limit for Figure 2; old spelling retained as an alias")
    parser.add_argument("--vc-paper-compare-time-max", type=float, default=None)
    parser.add_argument("--vc-paper-sweep-oracle-max", "--vc-paper-sweep-grad-max",
                        type=float, default=None, help="oracle-call limit for Figure 4; old spelling retained as an alias")

    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def resolve_algorithms(problem, requested):
    if requested == "auto" or requested == ["auto"]:
        return default_algorithms(problem)
    if isinstance(requested, str):
        return [requested]
    if "auto" in requested:
        raise ValueError("Use --algorithms auto by itself, or list algorithms explicitly.")
    return requested


def _load(path):
    if not path.exists():
        return None
    return np.load(path)


def _load_with_warning(path):
    data = _load(path)
    if data is None:
        warnings.warn(f"skip missing result: {path}")
    return data


def _load_ours(problem, R, alpha, result_dir, line_search_mode="parallel", xaxis=None, warn=True, algorithm="ours", checkpoint_threshold=10,
               allow_mode_fallback=True):
    modes = [line_search_mode]
    if allow_mode_fallback and xaxis == "grad_call" and line_search_mode == "parallel":
        modes.append("short_circuit")

    tried = []
    for mode in modes:
        path = Path(result_dir) / result_filename(
            problem,
            algorithm,
            R=R,
            alpha=alpha,
            line_search_mode=mode,
            checkpoint_threshold=(checkpoint_threshold if algorithm in {"ours_cp", "ours_defer_cp"} else None),
        )
        tried.append(path)
        data = _load(path)
        if data is not None:
            return data, mode, path
    if warn:
        warnings.warn(f"skip missing VC result: {', '.join(str(path) for path in tried)}")
    return None, line_search_mode, None


def _xy(data, mode, xaxis):
    y = np.asarray(data[mode])
    if xaxis == "iter":
        x = np.arange(len(y), dtype=float)
    else:
        if xaxis == "oracle_call" and xaxis not in data:
            raise ValueError(
                "Result has no oracle_call history. Rerun main.py with the updated "
                "code; old grad+val counts cannot recover the number of joint evaluations."
            )
        x = np.asarray(data[xaxis])
    n = min(len(x), len(y))
    return x[:n], y[:n]


def _plot_line(ax, x, y, label, linestyle="-", color=None, linewidth=None, alpha=1.0, zorder=None):
    ax.plot(x, y, label=label, linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _plot_baseline(ax, problem, algorithm, mode, xaxis, result_dir):
    path = Path(result_dir) / result_filename(problem, algorithm)
    data = _load_with_warning(path)
    if data is None:
        return False
    x, y = _xy(data, mode, xaxis)
    _plot_line(ax, x, y, LABELS[algorithm], linestyle=LINESTYLES[algorithm])
    return True


def _plot_r_markers(ax, data, xaxis, mode):
    if "r" not in data:
        return
    x, y = _xy(data, mode, xaxis)
    r = np.asarray(data["r"])[: len(y)]
    n = min(len(x), len(y), len(r))
    x, y, r = x[:n], y[:n], r[:n]
    mask = r < 1.0 - 1e-12
    ax.plot(x[mask], y[mask], marker="+", linestyle="", color="gray", alpha=0.25)


def _plot_vc(ax, problem, R, alpha, mode, xaxis, result_dir, line_search_mode="parallel", label=None,
             linestyle="-", color=None, linewidth=None, alpha_line=1.0, show_markers=False, suppress_ls_in_label=False,
             zorder=None, algorithm="ours", checkpoint_threshold=10, allow_mode_fallback=True):
    data, used_mode, _ = _load_ours(
        problem, R, alpha, result_dir, line_search_mode=line_search_mode, xaxis=xaxis,
        algorithm=algorithm, checkpoint_threshold=checkpoint_threshold,
        allow_mode_fallback=allow_mode_fallback,
    )
    if data is None:
        return False
    x, y = _xy(data, mode, xaxis)
    if label is None:
        label = f"VC r={R}"
        if alpha is not None:
            label += f", alpha={alpha}"
        if not suppress_ls_in_label and used_mode != "parallel":
            label += ", SC"
    _plot_line(ax, x, y, label, linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha_line, zorder=zorder)
    if show_markers:
        _plot_r_markers(ax, data, xaxis, mode)
    return True


def _adaptive_xmax(ax):
    xmax = 0.0
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            xmax = max(xmax, float(np.max(x)))
    return 1.0 if xmax <= 0 else 1.02 * xmax


def _set_axes(ax, mode, xaxis, time_max=None, grad_call_max=None, grad_val_call_max=None, iter_max=None,
              oracle_call_max=None):
    ax.set_yscale("log")
    if xaxis == "oracle_call":
        ax.set_xlabel("#oracle calls", fontsize=12)
        ax.set_xlim(0, _adaptive_xmax(ax) if oracle_call_max is None else oracle_call_max)
    elif xaxis == "grad_call":
        ax.set_xlabel("#grad calls", fontsize=12)
        ax.set_xlim(0, _adaptive_xmax(ax) if grad_call_max is None else grad_call_max)
    elif xaxis == "grad_val_call":
        ax.set_xlabel("#grad + val calls", fontsize=12)
        ax.set_xlim(0, _adaptive_xmax(ax) if grad_val_call_max is None else grad_val_call_max)
    elif xaxis == "time":
        ax.set_xlabel("time (s)", fontsize=12)
        ax.set_xlim(0, _adaptive_xmax(ax) if time_max is None else time_max)
    else:
        ax.set_xlabel("iteration", fontsize=12)
        ax.set_xlim(0, _adaptive_xmax(ax) if iter_max is None else iter_max)

    ax.set_ylabel("gradient norm" if mode == "grad" else "objective function", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12, loc="upper right")
    ax.tick_params(labelsize=12)


def _mode_tag(line_search_modes):
    modes = list(dict.fromkeys(line_search_modes))
    if modes == ["parallel"]:
        return ""
    names = ["parallel" if m == "parallel" else "short" for m in modes]
    return "_ls-" + "-".join(names)


def plot_one(problem, mode, xaxis, Rs, alphas, algorithms="auto", result_dir=Path("./result"), fig_dir=None,
             control_markers=True, time_max=None, grad_call_max=None, grad_val_call_max=None, iter_max=None,
             line_search_modes=("parallel",), checkpoint_threshold=10, oracle_call_max=None):
    result_dir = Path(result_dir)
    fig_dir = result_dir / "fig" if fig_dir is None else Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    algorithms = resolve_algorithms(problem, algorithms)
    line_search_modes = list(dict.fromkeys(line_search_modes))

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for algorithm in algorithms:
        if algorithm in {"ours", "ours_cp", "ours_defer_r", "ours_defer_cp"}:
            for ls_mode in line_search_modes:
                for alpha in alphas:
                    for R in Rs:
                        suppress = (xaxis == "grad_call")
                        if algorithm == "ours":
                            label = f"VC r={R}"
                        elif algorithm == "ours_defer_r":
                            label = f"VC-deferR r={R}"
                        elif algorithm == "ours_defer_cp":
                            label = f"VC-deferR-CP T={checkpoint_threshold}, r={R}"
                        else:
                            label = f"VC-CP T={checkpoint_threshold}, r={R}"
                        if len(alphas) > 1:
                            label += f", alpha={alpha}"
                        if not suppress and len(line_search_modes) > 1:
                            label += ", Par." if ls_mode == "parallel" else ", SC"
                        _plot_vc(
                            ax, problem, R, alpha, mode, xaxis, result_dir,
                            line_search_mode=ls_mode, label=label,
                            show_markers=control_markers and len(line_search_modes) == 1,
                            suppress_ls_in_label=suppress, algorithm=algorithm,
                            checkpoint_threshold=checkpoint_threshold,
                        )
        else:
            _plot_baseline(ax, problem, algorithm, mode, xaxis, result_dir)

    _set_axes(ax, mode, xaxis, time_max, grad_call_max, grad_val_call_max, iter_max, oracle_call_max)
    fig.tight_layout()
    out = fig_dir / f"{problem}_{mode}_{xaxis}{_mode_tag(line_search_modes)}.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved: {out}")
    return fig, ax


def _vc_compare_figure(problem, xaxis, result_dir, fig_dir, time_max=None, grad_call_max=None, grad_val_call_max=None,
                       checkpoint_threshold=10, oracle_call_max=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for algorithm in [alg for alg in default_algorithms(problem) if alg != "ours"]:
        _plot_baseline(ax, problem, algorithm, "val", xaxis, result_dir)

    style_map = {
        "ours": {"color": "#d62728", "linestyle": "-"},
        "ours_defer_cp": {"color": "#9467bd", "linestyle": "--"},
    }
    for algorithm in VC_PAPER_COMPARE_ALGORITHMS:
        spec = style_map[algorithm]
        _plot_vc(
            ax,
            problem,
            R=VC_COMPARE_R,
            alpha=VC_COMPARE_ALPHA,
            mode="val",
            xaxis=xaxis,
            result_dir=result_dir,
            line_search_mode=VC_PAPER_LINE_SEARCH_MODE,
            allow_mode_fallback=False,
            algorithm=algorithm,
            checkpoint_threshold=checkpoint_threshold,
            label=VC_PAPER_LABELS[algorithm],
            linestyle=spec["linestyle"],
            color=spec["color"],
            linewidth=2.2,
            show_markers=True,
            suppress_ls_in_label=(xaxis == "grad_call"),
            zorder=5,
        )

    _set_axes(ax, "val", xaxis, time_max, grad_call_max, grad_val_call_max, None, oracle_call_max)
    fig.tight_layout()
    out = Path(fig_dir) / f"{problem}_vc_compare_val_{xaxis}.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved: {out}")
    return fig, ax


def _vc_sweep_figure(problem, result_dir, fig_dir, grad_call_max=None, oracle_call_max=None):
    # Compatibility for older Python callers; the sweep now measures oracle calls.
    if oracle_call_max is None:
        oracle_call_max = grad_call_max
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    combos = [(alpha, R) for alpha in VC_SWEEP_ALPHAS for R in VC_SWEEP_RS]
    for alpha, R in combos:
        highlight = (alpha == 0.1 and R == 0.5)
        _plot_vc(
            ax,
            problem,
            R=R,
            alpha=alpha,
            mode="val",
            xaxis="oracle_call",
            result_dir=result_dir,
            line_search_mode=VC_PAPER_LINE_SEARCH_MODE,
            allow_mode_fallback=False,
            label=rf"$\alpha={alpha:g}$, $r={R:g}$",
            linestyle=VC_R_LINESTYLES[R],
            color=VC_ALPHA_COLORS[alpha],
            linewidth=2.7 if highlight else 1.7,
            alpha_line=1.0 if highlight else 0.55,
            zorder=10 if highlight else 3,
        )

    _set_axes(ax, "val", "oracle_call", oracle_call_max=oracle_call_max)
    if problem == "rosenbrock":
        ax.legend(fontsize=11, ncol=1, loc="lower left")
    else:
        ax.legend(fontsize=11, ncol=2, loc="upper right")
    fig.tight_layout()
    out = Path(fig_dir) / f"{problem}_vc_sweep_val_oracle_call.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved: {out}")
    return fig, ax


def plot_vc_paper_set(problems, result_dir=Path("./result"), fig_dir=None,
                      compare=True, sweep=True,
                      compare_grad_max=None, compare_gradval_max=None, compare_time_max=None, sweep_grad_max=None,
                      show=True, checkpoint_threshold=10, compare_oracle_max=None, sweep_oracle_max=None):
    if compare_oracle_max is None:
        compare_oracle_max = compare_gradval_max
    if sweep_oracle_max is None:
        sweep_oracle_max = sweep_grad_max
    result_dir = Path(result_dir)
    fig_dir = result_dir / "fig" if fig_dir is None else Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for problem in problems:
        if compare:
            figures.append(_vc_compare_figure(problem, "grad_call", result_dir, fig_dir, grad_call_max=compare_grad_max,
                                              checkpoint_threshold=checkpoint_threshold))
            figures.append(_vc_compare_figure(problem, "oracle_call", result_dir, fig_dir, oracle_call_max=compare_oracle_max,
                                              checkpoint_threshold=checkpoint_threshold))
            figures.append(_vc_compare_figure(problem, "time", result_dir, fig_dir, time_max=compare_time_max,
                                              checkpoint_threshold=checkpoint_threshold))
        if sweep:
            figures.append(_vc_sweep_figure(problem, result_dir, fig_dir, oracle_call_max=sweep_oracle_max))
    if show:
        plt.show()
    return figures


def plot_grid(problems, modes, xaxes, Rs, alphas, algorithms="auto", result_dir=Path("./result"), fig_dir=None,
              control_markers=True, time_max=None, grad_call_max=None, grad_val_call_max=None, iter_max=None,
              line_search_modes=("parallel",), show=True, checkpoint_threshold=10, oracle_call_max=None):
    figures = []
    for problem in problems:
        for mode in modes:
            for xaxis in xaxes:
                figures.append(
                    plot_one(
                        problem=problem,
                        mode=mode,
                        xaxis=xaxis,
                        Rs=Rs,
                        alphas=alphas,
                        algorithms=algorithms,
                        result_dir=result_dir,
                        fig_dir=fig_dir,
                        control_markers=control_markers,
                        time_max=time_max,
                        grad_call_max=grad_call_max,
                        grad_val_call_max=grad_val_call_max,
                        oracle_call_max=oracle_call_max,
                        iter_max=iter_max,
                        line_search_modes=line_search_modes,
                        checkpoint_threshold=checkpoint_threshold,
                    )
                )
    if show:
        plt.show()
    return figures


def main():
    args = parse_args()
    if args.vc_paper_mode or args.vc_paper_compare or args.vc_paper_sweep:
        compare = True
        sweep = True
        if args.vc_paper_compare or args.vc_paper_sweep:
            compare = args.vc_paper_compare
            sweep = args.vc_paper_sweep
        plot_vc_paper_set(
            problems=args.problem,
            result_dir=args.result_dir,
            fig_dir=args.fig_dir,
            compare=compare,
            sweep=sweep,
            compare_grad_max=args.vc_paper_compare_grad_max,
            compare_oracle_max=args.vc_paper_compare_oracle_max,
            compare_time_max=args.vc_paper_compare_time_max,
            sweep_oracle_max=args.vc_paper_sweep_oracle_max,
            show=not args.no_show,
            checkpoint_threshold=args.checkpoint_threshold,
        )
        return

    plot_grid(
        problems=args.problem,
        modes=args.mode,
        xaxes=args.xaxis,
        Rs=args.R,
        alphas=args.alpha,
        algorithms=args.algorithms,
        result_dir=args.result_dir,
        fig_dir=args.fig_dir,
        control_markers=not args.no_control_markers,
        time_max=args.time_max,
        grad_call_max=args.grad_call_max,
        grad_val_call_max=args.grad_val_call_max,
        oracle_call_max=args.oracle_call_max,
        iter_max=args.iter_max,
        line_search_modes=args.line_search_mode,
        show=not args.no_show,
        checkpoint_threshold=args.checkpoint_threshold,
    )


if __name__ == "__main__":
    main()
