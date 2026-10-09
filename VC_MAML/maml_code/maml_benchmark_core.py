"""MAML benchmark core: inner GD/NAG/VC, outer stochastic Adam.

This version is intentionally focused on fixed optimizer-time comparisons.
Each outer step samples a fresh meta-batch. JIT compilation and held-out
evaluation are excluded from the measured optimizer time.

VC returns the paper's weighted output x_bar_K (Eq. wt): for
2**i <= K < 2**(i+1), average x_t over 2**(i-1) <= t < K with
weights proportional to exp(alpha * t**(6/7)). GD and NAG return x_K.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "False")

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

# Persist the output rule so cached last-iterate VC trials cannot be reused.
INNER_OUTPUT = {
    "gd": "last_iterate",
    "nag": "last_iterate",
    "vc": "weighted_average_wt",
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    hidden_width: int = 40
    eval_tasks: int = 256
    support_size: int = 10
    query_size: int = 10
    inner_steps: int = 10
    time_budget_seconds: float = 30.0
    log_every: int = 5

    amplitude_min: float = 0.1
    amplitude_max: float = 5.0
    phase_min: float = 0.0
    phase_max: float = float(np.pi)
    x_min: float = -5.0
    x_max: float = 5.0

    inner_gd_L: float = 100.0
    inner_nag_L: float = 100.0

    inner_vc_alpha: float = 0.1
    inner_vc_r: float = 0.5
    inner_vc_L_proxy: float = 100.0
    inner_vc_M_proxy: float = 100.0

    outer_adam_lr: float = 1.0e-3
    outer_adam_beta1: float = 0.9
    outer_adam_beta2: float = 0.999
    outer_adam_epsilon: float = 1.0e-8
    meta_batch_size: int = 16


class TaskBatch(NamedTuple):
    support_x: Array
    support_y: Array
    query_x: Array
    query_y: Array


class AdamState(NamedTuple):
    step: Array
    first_moment: Array
    second_moment: Array


def initialize_parameters(key: Array, hidden_width: int) -> Array:
    key_w1, key_w2 = jax.random.split(key)
    w1 = jax.random.normal(key_w1, (hidden_width,)) * jnp.sqrt(
        2.0 / (1 + hidden_width)
    )
    b1 = jnp.zeros(hidden_width)
    w2 = jax.random.normal(key_w2, (hidden_width,)) * jnp.sqrt(
        2.0 / (hidden_width + 1)
    )
    return jnp.concatenate([w1, b1, w2, jnp.zeros(1)])


def predict(theta: Array, x: Array, hidden_width: int) -> Array:
    w1 = theta[:hidden_width]
    b1 = theta[hidden_width : 2 * hidden_width]
    w2 = theta[2 * hidden_width : 3 * hidden_width]
    b2 = theta[-1]
    hidden = jnp.tanh(x[..., None] * w1 + b1)
    return hidden @ w2 + b2


def mse(theta: Array, x: Array, y: Array, hidden_width: int) -> Array:
    residual = predict(theta, x, hidden_width) - y
    return 0.5 * jnp.mean(residual**2)


def sample_task_batch(
    key: Array, config: ExperimentConfig, batch_size: int
) -> TaskBatch:
    ka, kp, ks, kq = jax.random.split(key, 4)
    amplitude = jax.random.uniform(
        ka,
        (batch_size,),
        minval=config.amplitude_min,
        maxval=config.amplitude_max,
    )
    phase = jax.random.uniform(
        kp,
        (batch_size,),
        minval=config.phase_min,
        maxval=config.phase_max,
    )
    support_x = jax.random.uniform(
        ks,
        (batch_size, config.support_size),
        minval=config.x_min,
        maxval=config.x_max,
    )
    query_x = jax.random.uniform(
        kq,
        (batch_size, config.query_size),
        minval=config.x_min,
        maxval=config.x_max,
    )
    support_y = amplitude[:, None] * jnp.sin(support_x + phase[:, None])
    query_y = amplitude[:, None] * jnp.sin(query_x + phase[:, None])
    return TaskBatch(support_x, support_y, query_x, query_y)


def damping_coefficient(t: Array, alpha: float) -> Array:
    return jnp.expm1(
        alpha * (t ** (6.0 / 7.0) - (t - 1.0) ** (6.0 / 7.0))
    )


def velocity_control_sigma(
    velocity_norm_sq: Array, threshold_sq: Array, r: float
) -> Array:
    """C2 velocity control using a quintic smoothstep."""
    threshold_sq = jnp.maximum(
        threshold_sq, jnp.finfo(velocity_norm_sq.dtype).tiny
    )
    z = jnp.clip((velocity_norm_sq / threshold_sq - 1.0) / 3.0, 0.0, 1.0)
    smoothstep = 3.0 * z**2 - 2.0 * z**3
    return 1.0 - (1.0 - r) * smoothstep


def make_adapter(
    config: ExperimentConfig, method: str
) -> Callable[[Array, Array, Array], Array]:
    """Build the inner output map differentiated by the meta-objective.

    VC uses x_bar_K from Eq. (wt), including x_{K-1} but excluding x_K.
    Its dyadic averaging window is defined for K >= 2.
    """
    support_value_grad = jax.value_and_grad(
        lambda theta, x, y: mse(theta, x, y, config.hidden_width)
    )

    if method == "gd":
        if config.inner_gd_L <= 0.0:
            raise ValueError("inner_gd_L must be positive")

        def adapt(theta0: Array, x: Array, y: Array) -> Array:
            def step(theta, _):
                _, grad = support_value_grad(theta, x, y)
                return theta - grad / config.inner_gd_L, None

            return jax.lax.scan(
                step, theta0, xs=None, length=config.inner_steps
            )[0]

        return adapt

    if method == "nag":
        if config.inner_nag_L <= 0.0:
            raise ValueError("inner_nag_L must be positive")

        def adapt(theta0: Array, x: Array, y: Array) -> Array:
            def step(carry, k):
                theta_prev, theta = carry
                kf = k.astype(theta.dtype)
                beta = jnp.where(k == 0, 0.0, (kf - 1.0) / (kf + 2.0))
                extrapolated = theta + beta * (theta - theta_prev)
                _, grad = support_value_grad(extrapolated, x, y)
                theta_next = extrapolated - grad / config.inner_nag_L
                return (theta, theta_next), None

            ks = jnp.arange(config.inner_steps, dtype=jnp.int32)
            return jax.lax.scan(step, (theta0, theta0), ks)[0][1]

        return adapt

    if method == "vc":
        if config.inner_steps < 2:
            raise ValueError("VC weighted-average output requires inner_steps >= 2")
        if not (0.0 <= config.inner_vc_r < 1.0):
            raise ValueError("inner_vc_r must satisfy 0 <= r < 1")
        if config.inner_vc_alpha <= 0.0:
            raise ValueError("inner_vc_alpha must be positive")
        if config.inner_vc_L_proxy <= 0.0 or config.inner_vc_M_proxy <= 0.0:
            raise ValueError("VC proxies must be positive")

        r_prime = max(config.inner_vc_r, 0.5)
        h_sq = 4.0 * (1.0 - r_prime) / config.inner_vc_L_proxy
        window_start = 1 << (config.inner_steps.bit_length() - 2)

        def adapt(theta0: Array, x: Array, y: Array) -> Array:
            times = jnp.arange(1, config.inner_steps + 1, dtype=theta0.dtype)
            taus = times - 1.0
            # Subtract the largest log weight before exponentiating. The
            # common scale cancels upon normalization and avoids overflow.
            log_weights = config.inner_vc_alpha * (
                taus ** (6.0 / 7.0) - taus[-1] ** (6.0 / 7.0)
            )
            weights = jnp.where(taus >= window_start, jnp.exp(log_weights), 0.0)
            weights = weights / jnp.sum(weights)

            def step(carry, inputs):
                theta, velocity, r_hat_prev, a_prev, theta_bar = carry
                t, weight = inputs
                # Before update t, theta is x_{t-1}. Accumulate the output
                # without storing a separate array of all inner iterates.
                theta_bar = theta_bar + weight * theta
                _, grad = support_value_grad(theta, x, y)
                a_t = damping_coefficient(t, config.inner_vc_alpha)
                v_one = (
                    r_hat_prev * (2.0 + a_t) / (2.0 + a_prev) * velocity
                    - h_sq * grad
                ) / (1.0 + a_t)
                threshold = (
                    6.0 * r_prime * config.inner_vc_alpha
                    / (
                        7.0
                        * h_sq
                        * config.inner_vc_M_proxy
                        * t ** (1.0 / 7.0)
                    )
                )
                r_hat = velocity_control_sigma(
                    jnp.vdot(v_one, v_one).real,
                    threshold**2,
                    config.inner_vc_r,
                )
                velocity = (1.0 + a_t) / (2.0 - r_hat + a_t) * v_one
                theta = theta + velocity
                return (theta, velocity, r_hat, a_t, theta_bar), None

            initial = (
                theta0,
                jnp.zeros_like(theta0),
                jnp.asarray(1.0, theta0.dtype),
                jnp.asarray(0.0, theta0.dtype),
                jnp.zeros_like(theta0),
            )
            return jax.lax.scan(step, initial, (times, weights))[0][4]

        return adapt

    raise ValueError("method must be 'gd', 'nag', or 'vc'")


def make_meta_loss(
    config: ExperimentConfig, method: str
) -> Callable[[Array, TaskBatch], Array]:
    adapter = make_adapter(config, method)

    def one_task(theta0, sx, sy, qx, qy):
        adapted = adapter(theta0, sx, sy)
        return mse(adapted, qx, qy, config.hidden_width)

    batched = jax.vmap(one_task, in_axes=(None, 0, 0, 0, 0))

    def meta_loss(theta0: Array, tasks: TaskBatch) -> Array:
        return jnp.mean(
            batched(
                theta0,
                tasks.support_x,
                tasks.support_y,
                tasks.query_x,
                tasks.query_y,
            )
        )

    return meta_loss


def make_fixed_eval_loss(
    config: ExperimentConfig, method: str, eval_tasks: TaskBatch
) -> Callable[[Array], Array]:
    meta_loss = make_meta_loss(config, method)
    return jax.jit(lambda theta: meta_loss(theta, eval_tasks))


def initialize_adam(theta: Array) -> AdamState:
    zeros = jnp.zeros_like(theta)
    return AdamState(jnp.asarray(0, jnp.int32), zeros, zeros)


def make_adam_update(config: ExperimentConfig):
    b1 = float(config.outer_adam_beta1)
    b2 = float(config.outer_adam_beta2)
    lr = float(config.outer_adam_lr)
    eps = float(config.outer_adam_epsilon)

    @jax.jit
    def update(theta: Array, grad: Array, state: AdamState):
        step = state.step + 1
        first = b1 * state.first_moment + (1.0 - b1) * grad
        second = b2 * state.second_moment + (1.0 - b2) * grad**2
        first_hat = first / (1.0 - b1**step)
        second_hat = second / (1.0 - b2**step)
        theta = theta - lr * first_hat / (jnp.sqrt(second_hat) + eps)
        return theta, AdamState(step, first, second)

    return update


def evaluate(theta: Array, eval_loss: Callable[[Array], Array]) -> float:
    return float(jax.device_get(eval_loss(theta)))


def train_fixed_time(
    theta0: Array,
    task_key: Array,
    config: ExperimentConfig,
    method: str,
    label: str,
    eval_tasks: TaskBatch,
) -> tuple[Array, list[dict[str, object]]]:
    if config.time_budget_seconds <= 0.0:
        raise ValueError("time_budget_seconds must be positive")
    if config.meta_batch_size <= 0:
        raise ValueError("meta_batch_size must be positive")

    stochastic_loss = make_meta_loss(config, method)
    value_grad = jax.jit(jax.value_and_grad(stochastic_loss, argnums=0))
    eval_loss = make_fixed_eval_loss(config, method, eval_tasks)
    adam_update = make_adam_update(config)

    # Warm up all JIT-compiled computations used in the timed loop/evaluation.
    warm_key = jax.random.fold_in(task_key, 2_000_000_001)
    warm_batch = sample_task_batch(warm_key, config, config.meta_batch_size)
    warm_value, warm_grad = value_grad(theta0, warm_batch)
    warm_theta, warm_state = adam_update(theta0, warm_grad, initialize_adam(theta0))
    warm_eval = eval_loss(theta0)
    jax.block_until_ready((warm_value, warm_grad, warm_theta, warm_state, warm_eval))

    theta = theta0
    state = initialize_adam(theta0)
    elapsed = 0.0
    iteration = 0
    key = task_key
    history: list[dict[str, object]] = [
        {
            "method": label,
            "outer_iteration": 0,
            "oracle_calls": 0,
            "elapsed_seconds": 0.0,
            "eval_query_loss": evaluate(theta, eval_loss),
        }
    ]

    while elapsed < config.time_budget_seconds:
        iteration += 1
        tic = time.perf_counter()

        # Fresh meta-batch sampling is part of the optimizer runtime.
        key, batch_key = jax.random.split(key)
        tasks = sample_task_batch(batch_key, config, config.meta_batch_size)
        _, grad = value_grad(theta, tasks)
        theta, state = adam_update(theta, grad, state)

        # theta depends on task sampling, meta-gradient, and Adam update.
        jax.block_until_ready(theta)
        elapsed += time.perf_counter() - tic

        if (
            iteration == 1
            or iteration % config.log_every == 0
            or elapsed >= config.time_budget_seconds
        ):
            loss = evaluate(theta, eval_loss)
            history.append(
                {
                    "method": label,
                    "outer_iteration": iteration,
                    # One inner gradient-oracle call is counted for each
                    # task and each inner step.  GD, NAG, and VC all use one
                    # support-loss gradient evaluation per inner step.
                    "oracle_calls": iteration * config.meta_batch_size * config.inner_steps,
                    "elapsed_seconds": elapsed,
                    "eval_query_loss": loss,
                }
            )
            print(
                f"{label:20s} iter={iteration:5d} "
                f"time={elapsed:8.3f}s eval={loss:.6e}"
            )

    return theta, history
