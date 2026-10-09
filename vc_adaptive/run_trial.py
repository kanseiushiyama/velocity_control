"""Run one velocity-control benchmark trial in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import obj_func as objective_module
import optimizer as optimizer_module

from experiment_config import make_model
from optimizer import (
    AdaRestartAGD, AdaRestartHB, AdaVeloControl, AdaVeloControlCheckpoint,
    AdaVeloControlDeferredR, AdaVeloControlDeferredRCheckpoint, Gradient_descent,
    NAGc, NAGsc,
)

jax.config.update("jax_enable_x64", True)


ALGORITHM_CLASSES = {
    "gd": Gradient_descent,
    "hb": AdaRestartHB,
    "agd": AdaRestartAGD,
    "nag": NAGsc,
    "nagc": NAGc,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def config_hash(config: dict) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_optimizer(model, config):
    algorithm = config["algorithm"]
    if algorithm in {"ours", "ours_cp", "ours_defer_r", "ours_defer_cp"}:
        cls = {
            "ours": AdaVeloControl,
            "ours_cp": AdaVeloControlCheckpoint,
            "ours_defer_r": AdaVeloControlDeferredR,
            "ours_defer_cp": AdaVeloControlDeferredRCheckpoint,
        }[algorithm]
        kwargs = dict(
            R=float(config["R"]),
            alpha=float(config["alpha"]),
            hmax=float(config["hmax"]),
            line_search_mode=config["line_search_mode"],
        )
        if algorithm in {"ours_cp", "ours_defer_cp"}:
            kwargs["checkpoint_threshold"] = int(config["checkpoint_threshold"])
        return cls(model, **kwargs)
    return ALGORITHM_CLASSES[algorithm](model)


def configure_optimizer(optim, config, output_dir: Path) -> None:
    optim.stop_mode = config["stop_mode"]
    optim.time_limit = float(config["time_limit"])
    optim.n_iter = int(config["n_iter"])
    optim.result_dir = output_dir
    optim.save_hist = False


def extract_history(optim) -> dict[str, np.ndarray]:
    data = {
        "grad_call": np.asarray(optim.grad_call_hist),
        "val_call": np.asarray(optim.val_call_hist),
        "grad_val_call": np.asarray(optim.grad_val_call_hist),
        "joint_call": np.asarray(optim.joint_call_hist),
        "oracle_call": np.asarray(optim.oracle_call_hist),
        "grad": np.asarray(optim.grad_hist),
        "val": np.asarray(optim.val_hist),
        "time": np.asarray(optim.time_hist),
    }
    for key in ("r", "acc"):
        attr = f"{key}_hist"
        if hasattr(optim, attr):
            arr = np.asarray(getattr(optim, attr))
            if arr.size:
                data[key] = arr
    if hasattr(optim, "line_search_stats"):
        data["line_search_stats"] = np.asarray(optim.line_search_stats)
        data["line_search_mode"] = np.asarray(optim.line_search_mode)
    return data


def atomic_json_dump(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.config_json.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model = make_model(
        config["problem"],
        dim=config["dim"],
        L=float(config["L"]),
        sigma_x0=float(config["sigma_x0"]),
        seed=int(config["seed"]),
    )
    optim = make_optimizer(model, config)
    configure_optimizer(optim, config, args.output_dir)

    print(f"job: {config['job_id']}")
    print(f"repeat: {config['repeat_index']}")
    print(f"seed: {config['seed']}")
    print(f"dimension: {model.param.size}")

    optim.run()
    history = extract_history(optim)
    np.savez(args.output_dir / "history.npz", **history)

    final_grad = float(history["grad"][-1]) if len(history["grad"]) else float("nan")
    final_val = float(history["val"][-1]) if len(history["val"]) else float("nan")
    summary = {
        "status": "complete",
        "config_hash": config_hash(config),
        "job_id": config["job_id"],
        "label": config["label"],
        "algorithm": config["algorithm"],
        "repeat_index": int(config["repeat_index"]),
        "seed": int(config["seed"]),
        "elapsed_time": float(optim.elapsed_time),
        "grad_calls": int(optim.oracle.count["grad"]),
        "val_calls": int(optim.oracle.count["val"]),
        "grad_val_calls": int(optim.oracle.count["grad"] + optim.oracle.count["val"]),
        "joint_calls": int(optim.oracle.count["joint"]),
        "oracle_calls": int(optim.oracle.oracle_calls),
        "oracle_counting": "value-only=1, gradient-only=1, joint=1; excludes diagnostics and warmup",
        "optimizer_sha256": hashlib.sha256(Path(optimizer_module.__file__).read_bytes()).hexdigest(),
        "objective_sha256": hashlib.sha256(Path(objective_module.__file__).read_bytes()).hexdigest(),
        "history_length": int(len(history["time"])),
        "final_gradient_norm": final_grad,
        "final_objective": final_val,
    }
    if config["algorithm"] in {"ours", "ours_cp", "ours_defer_r", "ours_defer_cp"}:
        summary.update(
            {
                "R": float(config["R"]),
                "alpha": float(config["alpha"]),
                "hmax": float(config["hmax"]),
                "line_search_mode": config["line_search_mode"],
            }
        )
        if config["algorithm"] in {"ours_cp", "ours_defer_cp"}:
            summary.update(optim.checkpoint_summary())
        elif config["algorithm"] == "ours_defer_r":
            summary.update({
                "deferred_R_checks": int(optim.deferred_R_checks),
                "deferred_R_failures": int(optim.line_search_stats[1]),
            })

    # summary.json is written last. Its presence marks a completed trial.
    atomic_json_dump(args.output_dir / "summary.json", summary)

    print(f"wall-clock: {summary['elapsed_time']:.6g} s")
    print(f"grad calls: {summary['grad_calls']}")
    print(f"val calls: {summary['val_calls']}")
    print(f"grad + val calls: {summary['grad_val_calls']}")
    print(f"joint calls: {summary['joint_calls']}")
    print(f"oracle calls: {summary['oracle_calls']}")
    print("status: complete")


if __name__ == "__main__":
    main()
