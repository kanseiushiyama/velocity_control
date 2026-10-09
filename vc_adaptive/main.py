"""Run resumable repeated numerical experiments for the velocity-control paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings
from functools import lru_cache

import numpy as np

from experiment_config import (
    VC_COMPARE_ALPHA, VC_COMPARE_R, VC_PAPER_COMPARE_ALGORITHMS,
    VC_PAPER_LINE_SEARCH_MODE, VC_PAPER_REPEATS, VC_SWEEP_ALPHAS, VC_SWEEP_RS,
    default_algorithms, problem_choices, result_filename,
)


LABELS = {
    "gd": "GD",
    "hb": "Restart HB",
    "agd": "Restart AGD",
    "nag": "NAG-SC",
    "nagc": "NAG-C",
}

EXPERIMENT_VERSION = "benchmark-v9-oracle-qing-compensated"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--problem", choices=problem_choices(), default="quad_arith")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("auto", "gd", "hb", "agd", "ours", "ours_cp", "ours_defer_r", "ours_defer_cp", "nag", "nagc"),
        default=None,
        help="'auto' adds NAG-SC for quad_arith and adaptive NAG-C for Powell",
    )
    parser.add_argument("--R", type=float, nargs="+", default=None, help="normal mode default: 0.25 0.5 0.75")
    parser.add_argument("--alpha", type=float, nargs="+", default=None, help="normal mode default: 0.1")
    parser.add_argument("--hmax", type=float, default=1.0)
    parser.add_argument("--checkpoint-threshold", type=int, default=10)
    parser.add_argument(
        "--line-search-mode",
        nargs="+",
        choices=("parallel", "short_circuit"),
        default=None,
        help="normal mode default: parallel; paper mode: parallel",
    )

    parser.add_argument("--stop-mode", choices=("time", "iter"), default="time")
    parser.add_argument("--time-limit", type=float, default=100.0)
    parser.add_argument("--n-iter", type=int, default=17_500, help="gradient-call budget for --stop-mode iter")
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="number of repeat directories (normal: 1; paper: 10)",
    )

    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--L", type=float, default=10_000)
    parser.add_argument("--sigma-x0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=0,
        help="seed increment between repeats; 0 repeats the same deterministic instance",
    )

    parser.add_argument("--result-dir", type=Path, default=Path("./result"))
    parser.add_argument("--overwrite", action="store_true", help="rerun completed trials")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed trial")
    parser.add_argument(
        "--fixed-method-order",
        action="store_true",
        help="do not shuffle method order within each repeat",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="show complete/pending status and aggregate existing trials without running",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--paper", "--vc-paper-mode", dest="paper", action="store_true",
        help="run exactly the experiments needed for paper figures 1--4",
    )
    parser.add_argument(
        "--paper-compare", "--vc-paper-compare", dest="paper_compare", action="store_true",
        help="run paper figures 1--3 only (implies --paper)",
    )
    parser.add_argument(
        "--paper-sweep", "--vc-paper-sweep", dest="paper_sweep", action="store_true",
        help="run paper figure 4 only (implies --paper)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print resolved settings and jobs without running or writing files",
    )
    args = parser.parse_args(argv)
    args.paper = args.paper or args.paper_compare or args.paper_sweep
    if args.paper:
        for field, flag in (
            ("algorithms", "--algorithms"), ("R", "--R"),
            ("alpha", "--alpha"), ("line_search_mode", "--line-search-mode"),
        ):
            if getattr(args, field) is not None:
                parser.error(f"{flag} is fixed by --paper; omit it or use normal mode.")
        if not args.paper_compare and not args.paper_sweep:
            args.paper_compare = args.paper_sweep = True
    args.algorithms = ["auto"] if args.algorithms is None else args.algorithms
    args.R = list(VC_SWEEP_RS) if args.R is None else args.R
    args.alpha = (list(VC_SWEEP_ALPHAS) if args.paper else [0.1]) if args.alpha is None else args.alpha
    if args.line_search_mode is None:
        args.line_search_mode = [VC_PAPER_LINE_SEARCH_MODE if args.paper else "parallel"]
    if args.repeats is None:
        args.repeats = VC_PAPER_REPEATS if args.paper else 1
    # Trials run with cwd=runner.parent, so paths must not depend on that cwd.
    args.result_dir = args.result_dir.expanduser().resolve()
    return args


def resolve_algorithms(problem: str, requested: list[str]) -> list[str]:
    if "auto" in requested:
        if len(requested) != 1:
            raise ValueError("Use --algorithms auto by itself, or list algorithms explicitly.")
        return default_algorithms(problem)
    if "nag" in requested and problem != "quad_arith":
        raise ValueError("NAG-SC is enabled only for --problem quad_arith.")
    if "nagc" in requested and problem != "powell":
        raise ValueError("NAG-C is enabled only for --problem powell.")
    return requested


def fmt_float(x: float) -> str:
    return f"{x:g}".replace("-", "m").replace(".", "p")


def make_jobs(args, algorithms: list[str]) -> list[dict]:
    jobs = []
    for algorithm in algorithms:
        if algorithm not in {"ours", "ours_cp", "ours_defer_r", "ours_defer_cp"}:
            jobs.append({
                "job_id": algorithm,
                "label": LABELS[algorithm],
                "algorithm": algorithm,
                "R": None,
                "alpha": None,
            })
            continue
        for line_search_mode in args.line_search_mode:
            for R in args.R:
                for alpha in args.alpha:
                    mode_suffix = "parallel" if line_search_mode == "parallel" else "short"
                    if algorithm == "ours":
                        job_id = f"ours_R{fmt_float(R)}_a{fmt_float(alpha)}_{mode_suffix}"
                        label = f"AdaVeloControl (R={R:g}, alpha={alpha:g}, {line_search_mode})"
                    elif algorithm == "ours_defer_r":
                        job_id = f"ours_deferR_R{fmt_float(R)}_a{fmt_float(alpha)}_{mode_suffix}"
                        label = f"AdaVeloControl-deferR (R={R:g}, alpha={alpha:g}, {line_search_mode})"
                    elif algorithm == "ours_defer_cp":
                        job_id = (
                            f"ours_deferR_cp_T{args.checkpoint_threshold}_R{fmt_float(R)}_"
                            f"a{fmt_float(alpha)}_{mode_suffix}"
                        )
                        label = (
                            f"AdaVeloControl-deferR-CP (T={args.checkpoint_threshold}, "
                            f"R={R:g}, alpha={alpha:g}, {line_search_mode})"
                        )
                    else:
                        job_id = (
                            f"ours_cp_T{args.checkpoint_threshold}_R{fmt_float(R)}_"
                            f"a{fmt_float(alpha)}_{mode_suffix}"
                        )
                        label = (
                            f"AdaVeloControl-CP (T={args.checkpoint_threshold}, R={R:g}, "
                            f"alpha={alpha:g}, {line_search_mode})"
                        )
                    jobs.append({
                        "job_id": job_id,
                        "label": label,
                        "algorithm": algorithm,
                        "R": float(R),
                        "alpha": float(alpha),
                        "line_search_mode": line_search_mode,
                        "checkpoint_threshold": (
                            int(args.checkpoint_threshold) if algorithm in {"ours_cp", "ours_defer_cp"} else None
                        ),
                    })
    return jobs


def make_paper_jobs(args) -> list[dict]:
    """Build the union of comparison and sweep trials, keeping shared VC once."""
    preset = argparse.Namespace(**vars(args))
    preset.line_search_mode = [VC_PAPER_LINE_SEARCH_MODE]
    jobs = []
    if args.paper_compare:
        preset.R = [VC_COMPARE_R]
        preset.alpha = [VC_COMPARE_ALPHA]
        baselines = [alg for alg in default_algorithms(args.problem) if alg != "ours"]
        jobs.extend(make_jobs(preset, baselines + list(VC_PAPER_COMPARE_ALGORITHMS)))
    if args.paper_sweep:
        preset.R = list(VC_SWEEP_RS)
        preset.alpha = list(VC_SWEEP_ALPHAS)
        jobs.extend(make_jobs(preset, ["ours"]))
    return list({job["job_id"]: job for job in jobs}.values())


def config_hash(config: dict) -> str:
    text = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def source_hashes() -> dict[str, str]:
    """Invalidate cached trials when the numerical implementation changes."""
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("optimizer.py", "obj_func.py", "run_trial.py", "experiment_config.py")}


def trial_config(args, job: dict, repeat: int) -> dict:
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "source_hashes": source_hashes(),
        "problem": args.problem,
        "job_id": job["job_id"],
        "label": job["label"],
        "algorithm": job["algorithm"],
        "stop_mode": args.stop_mode,
        "time_limit": float(args.time_limit),
        "n_iter": int(args.n_iter),
        "dim": args.dim,
        "L": float(args.L),
        "sigma_x0": float(args.sigma_x0),
        "seed": int(args.seed + repeat * args.seed_stride),
        "repeat_index": int(repeat),
    }
    if job["algorithm"] in {"ours", "ours_cp", "ours_defer_r", "ours_defer_cp"}:
        config.update({
            "R": job["R"],
            "alpha": job["alpha"],
            "hmax": float(args.hmax),
            "line_search_mode": job["line_search_mode"],
        })
        if job["algorithm"] in {"ours_cp", "ours_defer_cp"}:
            config["checkpoint_threshold"] = int(job["checkpoint_threshold"])
    return config


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def trial_status(trial_dir: Path, config: dict) -> str:
    summary_path = trial_dir / "summary.json"
    history_path = trial_dir / "history.npz"
    if not summary_path.exists() or not history_path.exists():
        return "pending"
    try:
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return "pending"
    if summary.get("status") != "complete":
        return "pending"
    if summary.get("config_hash") != config_hash(config):
        return "stale"
    try:
        with np.load(history_path, allow_pickle=False) as npz:
            required = {"time", "grad_call", "val_call", "grad_val_call", "joint_call", "oracle_call", "grad", "val"}
            if not required.issubset(npz.files):
                return "stale"
    except (OSError, ValueError):
        return "pending"
    return "complete"


def atomic_json_dump(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def result_path_for_job(result_dir: Path, problem: str, job: dict) -> Path:
    if job["algorithm"] in {"ours", "ours_cp", "ours_defer_r", "ours_defer_cp"}:
        name = result_filename(
            problem,
            job["algorithm"],
            R=job["R"],
            alpha=job["alpha"],
            line_search_mode=job["line_search_mode"],
            checkpoint_threshold=job.get("checkpoint_threshold"),
        )
    else:
        name = result_filename(problem, job["algorithm"])
    return result_dir / name


def pad_histories(repeats: list[dict], key: str):
    lengths = np.asarray([len(repeat[key]) for repeat in repeats], dtype=int)
    max_len = int(lengths.max(initial=0))
    out = np.full((len(repeats), max_len), np.nan, dtype=float)
    for i, repeat in enumerate(repeats):
        arr = np.asarray(repeat[key], dtype=float)
        out[i, : len(arr)] = arr
    return out, lengths


def aggregate_histories(repeats: list[dict]) -> dict:
    if not repeats:
        raise ValueError("No completed repeats to aggregate.")

    hist_keys = ("time", "grad_call", "val_call", "grad_val_call", "joint_call", "oracle_call", "grad", "val")
    n_common = min(len(repeat["time"]) for repeat in repeats)
    data = {}
    for key in hist_keys:
        stack = (
            np.stack([np.asarray(repeat[key], dtype=float)[:n_common] for repeat in repeats])
            if n_common
            else np.empty((len(repeats), 0))
        )
        data[key] = np.median(stack, axis=0) if n_common else np.asarray([])
        repeat_hist, _ = pad_histories(repeats, key)
        data[f"{key}_repeats"] = repeat_hist

    _, hist_len_repeats = pad_histories(repeats, "time")
    data["hist_len_repeats"] = hist_len_repeats
    data["n_repeats_complete"] = np.asarray(len(repeats), dtype=int)

    summaries = [repeat["summary"] for repeat in repeats]
    elapsed = np.asarray([s["elapsed_time"] for s in summaries], dtype=float)
    data["elapsed_time_repeats"] = elapsed
    data["elapsed_time_median"] = np.asarray(np.median(elapsed))
    data["grad_calls_repeats"] = np.asarray([s["grad_calls"] for s in summaries], dtype=int)
    data["val_calls_repeats"] = np.asarray([s["val_calls"] for s in summaries], dtype=int)
    data["grad_val_calls_repeats"] = np.asarray([s["grad_val_calls"] for s in summaries], dtype=int)
    data["joint_calls_repeats"] = np.asarray([s["joint_calls"] for s in summaries], dtype=int)
    data["oracle_calls_repeats"] = np.asarray([s["oracle_calls"] for s in summaries], dtype=int)

    for key in ("r", "acc"):
        if all(key in repeat for repeat in repeats):
            n = min(len(repeat[key]) for repeat in repeats)
            arrays = [np.asarray(repeat[key])[:n] for repeat in repeats]
            ref = arrays[0]
            same = all(
                np.allclose(ref, arr, rtol=1e-12, atol=1e-14, equal_nan=True)
                for arr in arrays[1:]
            )
            if same:
                data[key] = ref
            else:
                warnings.warn(f"{key} histories differ across repeats; saving repeat 0 for markers.")
                data[key] = ref

    if all("line_search_stats" in repeat for repeat in repeats):
        stats = np.stack([np.asarray(repeat["line_search_stats"]) for repeat in repeats])
        data["line_search_stats"] = np.asarray(repeats[0]["line_search_stats"])
        data["line_search_stats_repeats"] = stats
        if "line_search_mode" in repeats[0]:
            data["line_search_mode"] = repeats[0]["line_search_mode"]

    return data


def read_completed_repeat(trial_dir: Path) -> dict:
    with np.load(trial_dir / "history.npz", allow_pickle=False) as npz:
        repeat = {key: np.asarray(npz[key]) for key in npz.files}
    repeat["summary"] = load_json(trial_dir / "summary.json")
    return repeat


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def problem_trials_dir(result_dir: Path, problem: str) -> Path:
    return Path(result_dir) / "trials" / problem


def aggregate_results(args, jobs: list[dict]) -> list[tuple[str, float]]:
    summary_rows = []
    stat_rows = []
    timing_summary = []

    for job in jobs:
        completed = []
        for repeat in range(args.repeats):
            config = trial_config(args, job, repeat)
            trial_dir = problem_trials_dir(args.result_dir, args.problem) / f"repeat_{repeat:03d}" / job["job_id"]
            if trial_status(trial_dir, config) != "complete":
                continue
            repeat_data = read_completed_repeat(trial_dir)
            completed.append(repeat_data)
            summary_rows.append(repeat_data["summary"])

        if not completed:
            warnings.warn(f"No completed repeats for {job['job_id']}; aggregate result not written.")
            continue

        data = aggregate_histories(completed)
        path = result_path_for_job(args.result_dir, args.problem, job)
        np.savez(path, **data)

        elapsed = np.asarray(data["elapsed_time_repeats"], dtype=float)
        med = float(np.median(elapsed))
        timing_summary.append((job["label"], med))
        stat_rows.append({
            "job_id": job["job_id"],
            "label": job["label"],
            "algorithm": job["algorithm"],
            "n_complete": len(completed),
            "n_requested": args.repeats,
            "elapsed_time_median": med,
            "elapsed_time_q25": float(np.quantile(elapsed, 0.25)),
            "elapsed_time_q75": float(np.quantile(elapsed, 0.75)),
            "elapsed_time_mean": float(np.mean(elapsed)),
            "elapsed_time_std": float(np.std(elapsed)),
            "grad_calls_median": float(np.median(data["grad_calls_repeats"])),
            "val_calls_median": float(np.median(data["val_calls_repeats"])),
            "grad_val_calls_median": float(np.median(data["grad_val_calls_repeats"])),
            "joint_calls_median": float(np.median(data["joint_calls_repeats"])),
            "oracle_calls_median": float(np.median(data["oracle_calls_repeats"])),
        })
        print(f"aggregate {job['job_id']}: {len(completed)}/{args.repeats} complete, median={med:.6g} s")

    write_csv(args.result_dir / "benchmark_summary.csv", summary_rows)
    write_csv(args.result_dir / "benchmark_stats.csv", stat_rows)
    return timing_summary


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.checkpoint_threshold <= 0:
        raise ValueError("--checkpoint-threshold must be positive.")
    if args.paper:
        jobs = make_paper_jobs(args)
        algorithms = list(dict.fromkeys(job["algorithm"] for job in jobs))
    else:
        algorithms = resolve_algorithms(args.problem, args.algorithms)
        jobs = make_jobs(args, algorithms)

    base_config = {
        "experiment_version": EXPERIMENT_VERSION,
        "problem": args.problem,
        "algorithms": algorithms,
        "jobs": jobs,
        "repeats": args.repeats,
        "stop_mode": args.stop_mode,
        "time_limit": args.time_limit,
        "n_iter": args.n_iter,
        "dim": args.dim,
        "L": args.L,
        "sigma_x0": args.sigma_x0,
        "seed": args.seed,
        "seed_stride": args.seed_stride,
        "hmax": args.hmax,
        "checkpoint_threshold": args.checkpoint_threshold,
        "line_search_mode": list(args.line_search_mode),
        "paper": args.paper,
        "paper_compare": args.paper_compare,
        "paper_sweep": args.paper_sweep,
    }
    if args.dry_run:
        print(json.dumps(base_config, indent=2, sort_keys=True))
        print(f"Total trials: {len(jobs)} jobs x {args.repeats} repeats = {len(jobs) * args.repeats}")
        return
    args.result_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(args.result_dir / "base_config.json", base_config)

    runner = Path(__file__).with_name("run_trial.py")
    rng = np.random.default_rng(args.seed + 91_337)
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    print(f"problem: {args.problem}")
    print(f"repeats: {args.repeats}")
    print(f"jobs: {', '.join(job['job_id'] for job in jobs)}")

    for repeat in range(args.repeats):
        repeat_dir = problem_trials_dir(args.result_dir, args.problem) / f"repeat_{repeat:03d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        order = list(jobs)
        if not args.fixed_method_order:
            rng.shuffle(order)
        seed = args.seed + repeat * args.seed_stride
        print(f"\nRepeat {repeat + 1}/{args.repeats}, seed={seed}")

        for job in order:
            config = trial_config(args, job, repeat)
            trial_dir = repeat_dir / job["job_id"]
            status = trial_status(trial_dir, config)

            if status == "complete" and not args.overwrite:
                print(f"  complete {job['job_id']}")
                continue
            if args.status_only:
                print(f"  {status:8s} {job['job_id']}")
                continue
            if status == "stale":
                print(f"  stale    {job['job_id']} -> rerun")
            elif status == "complete" and args.overwrite:
                print(f"  overwrite {job['job_id']}")
            else:
                print(f"  run      {job['job_id']}")

            trial_dir.mkdir(parents=True, exist_ok=True)
            config_path = trial_dir / "config.json"
            atomic_json_dump(config_path, config)
            log_path = trial_dir / "process.log"
            cmd = [
                sys.executable,
                str(runner),
                "--config-json",
                str(config_path),
                "--output-dir",
                str(trial_dir),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    cmd,
                    cwd=str(runner.parent),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode != 0:
                message = f"Trial failed: repeat={repeat}, job={job['job_id']}; see {log_path}"
                if args.keep_going:
                    print("  WARNING:", message)
                else:
                    raise RuntimeError(message)
            else:
                print(f"  complete {job['job_id']}")

    timing_summary = aggregate_results(args, jobs)

    if timing_summary:
        print("\n=== median wall-clock summary ===")
        for label, median_time in timing_summary:
            print(f"{label}: {median_time:.6g} s")

    if args.plot:
        if args.paper:
            from plot import plot_vc_paper_set

            plot_vc_paper_set(
                problems=[args.problem],
                result_dir=args.result_dir,
                compare=args.paper_compare,
                sweep=args.paper_sweep,
                show=not args.no_show,
                checkpoint_threshold=args.checkpoint_threshold,
            )
            return
        from plot import plot_grid

        plot_grid(
            problems=[args.problem],
            modes=["grad", "val"],
            xaxes=["oracle_call"],
            Rs=args.R,
            alphas=args.alpha,
            algorithms=algorithms,
            result_dir=args.result_dir,
            line_search_modes=args.line_search_mode,
            show=not args.no_show,
            checkpoint_threshold=args.checkpoint_threshold,
        )


if __name__ == "__main__":
    main()
