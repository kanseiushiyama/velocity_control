"""Run one fixed-time MAML trial in a fresh Python process."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import jax

from maml_benchmark_core import (
    ExperimentConfig,
    INNER_OUTPUT,
    initialize_parameters,
    sample_task_batch,
    train_fixed_time,
)

METHODS = {"gd": "GD", "nag": "NAG", "vc": "VC"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.config_json.open("r", encoding="utf-8") as f:
        config = ExperimentConfig(**json.load(f))

    root_key = jax.random.PRNGKey(config.seed)
    init_key, eval_key, train_key = jax.random.split(root_key, 3)
    theta0 = initialize_parameters(init_key, config.hidden_width)
    eval_tasks = sample_task_batch(eval_key, config, config.eval_tasks)

    _, history = train_fixed_time(
        theta0, train_key, config, args.method, METHODS[args.method], eval_tasks
    )
    write_csv(args.output_dir / "history.csv", history)

    final = history[-1]
    summary = {
        "candidate_id": args.candidate_id,
        "method_id": args.method,
        "method": METHODS[args.method],
        "repeat_index": args.repeat_index,
        "seed": config.seed,
        "inner_steps": config.inner_steps,
        "inner_output": INNER_OUTPUT[args.method],
        "inner_gd_L": config.inner_gd_L,
        "inner_nag_L": config.inner_nag_L,
        "inner_vc_L_proxy": config.inner_vc_L_proxy,
        "inner_vc_M_proxy": config.inner_vc_M_proxy,
        "final_eval_query_loss": float(final["eval_query_loss"]),
        "elapsed_seconds": float(final["elapsed_seconds"]),
        "time_budget_seconds": config.time_budget_seconds,
        "time_budget_overshoot_seconds": max(
            0.0, float(final["elapsed_seconds"]) - config.time_budget_seconds
        ),
        "outer_iterations_completed": int(final["outer_iteration"]),
        "oracle_calls_completed": int(final["oracle_calls"]),
        "jax_backend": jax.default_backend(),
        "jax_devices": "; ".join(str(d) for d in jax.devices()),
        "python": platform.python_version(),
        "pid": os.getpid(),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


if __name__ == "__main__":
    main()
