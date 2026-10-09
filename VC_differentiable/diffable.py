"""Compare GD, NAG, and Algorithm 1 of Ushiyama--Sato in JAX.

Artificial problem:

    f_lambda(x) = 1/2 sum_i a_i x_i^2
                  + sum_i rho_i (1 - cos(x_i)) + lambda b^T x.

At lambda=0, x_star=0 is a strict local minimizer because
A + diag(rho) is positive definite.  The local solution branch therefore satisfies

    dx_star/dlambda |_{lambda=0}
        = -[A + diag(rho)]^{-1} b.

For the velocity-control method, the formal output is the weighted average x_bar_t
from Eq. (13), computed recursively by Algorithm 3 of
"A Restart-Free Accelerated Algorithm for Non-Convex Minimization:
Continuous and Discrete Analysis" (arXiv:2606.30050).

JAX differentiates the GD and NAG iterates, as well as both the raw iterates x_t
and the Algorithm-3 outputs x_bar_t of the velocity-control method, with respect to lambda.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Enable double precision before importing jax.numpy.
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
Array = jax.Array


@dataclass(frozen=True)
class Problem:
    a: Array
    rho: Array
    b: Array
    x0: Array
    x_star: Array
    L: float
    M: float


def make_problem(
    dim: int = 20,
    seed: int = 7,
    x0_radius: float = 0.5,
) -> Problem:
    """Construct a smooth coercive globally nonconvex test problem.

    The final coordinate is exactly

        0.5*x_d^2 + 1 - cos(x_d),

    whose Hessian is 0.5 + cos(x_d).  Hence the objective is globally
    nonconvex (e.g. at x_d=pi) but is locally strongly convex at x=0.
    """
    if dim < 2:
        raise ValueError("dim must be at least 2")
    if x0_radius <= 0.0:
        raise ValueError("x0_radius must be positive")

    rng = np.random.default_rng(seed)

    # Positive diagonal A makes f_lambda coercive for every fixed lambda.
    a_np = np.concatenate([np.geomspace(1.0e-4, 1.0, dim - 1), [0.5]])

    # Keep rho nonnegative, and force the final coordinate to have rho_d=1.
    rho_np = np.concatenate([np.zeros(dim//2+dim%2), rng.uniform(0.0, 1.0, size=dim//2)])
    rho_np[-1] = 1.0

    b_np = rng.normal(size=dim)
    b_np /= np.linalg.norm(b_np)

    # Start near x_star=0 so that both methods target the same local branch.
    x0_np = rng.normal(size=dim)
    x0_np /= np.linalg.norm(x0_np)
    x0_np *= x0_radius

    x_star_np = np.zeros(dim)

    local_hessian_diag = a_np + rho_np
    if np.min(local_hessian_diag) <= 0.0:
        raise RuntimeError("x_star=0 is not a strict local minimizer")

    # The last coordinate guarantees global nonconvexity:
    # Hessian entry at x_d=pi is 0.5 - 1 = -0.5.
    if a_np[-1] - rho_np[-1] >= 0.0:
        raise RuntimeError("The constructed objective is not globally nonconvex")

    # Global bounds with the Euclidean norm:
    #   ||nabla^2 f(x)|| <= max_i (a_i + rho_i),
    #   ||nabla^2 f(x)-nabla^2 f(y)|| <= max_i rho_i ||x-y||.
    L = float(np.max(a_np + rho_np))
    M = float(np.max(rho_np))

    return Problem(
        a=jnp.asarray(a_np),
        rho=jnp.asarray(rho_np),
        b=jnp.asarray(b_np),
        x0=jnp.asarray(x0_np),
        x_star=jnp.asarray(x_star_np),
        L=L,
        M=M,
    )


def objective(problem: Problem, x: Array, lam: Array) -> Array:
    return (
        0.5 * jnp.dot(problem.a * x, x)
        + jnp.sum(problem.rho * (1.0 - jnp.cos(x)))
        + lam * jnp.dot(problem.b, x)
    )


def gradient(problem: Problem, x: Array, lam: Array) -> Array:
    return problem.a * x + problem.rho * jnp.sin(x) + lam * problem.b


def hessian_diag(problem: Problem, x: Array) -> Array:
    return problem.a + problem.rho * jnp.cos(x)


def gd_trajectory(
    problem: Problem,
    lam: Array,
    steps: int,
    step_factor: float = 1.0,
) -> Array:
    """Return [x_0,...,x_steps] for GD with eta=step_factor/L."""
    eta = step_factor / problem.L

    def step(x: Array, _: None) -> tuple[Array, Array]:
        x_next = x - eta * gradient(problem, x, lam)
        return x_next, x_next

    _, xs = jax.lax.scan(step, problem.x0, xs=None, length=steps)
    return jnp.concatenate([problem.x0[None, :], xs], axis=0)



def nag_trajectory(
    problem: Problem,
    lam: Array,
    steps: int,
    step_factor: float = 1.0,
) -> Array:
    """Return [x_0,...,x_steps] for classical convex NAG.

    The update is

        y_k = x_k + beta_k (x_k - x_{k-1}),
        x_{k+1} = y_k - eta * grad f_lambda(y_k),

    with eta=step_factor/L, beta_0=0, and
    beta_k=(k-1)/(k+2) for k>=1.  Thus one gradient evaluation is
    used per iteration, as for GD and Algorithm 1.
    """
    eta = step_factor / problem.L

    def step(
        carry: tuple[Array, Array],
        k: Array,
    ) -> tuple[tuple[Array, Array], Array]:
        x_prev, x = carry
        k_float = k.astype(x.dtype)
        beta = jnp.where(k == 0, 0.0, (k_float - 1.0) / (k_float + 2.0))
        y = x + beta * (x - x_prev)
        x_next = y - eta * gradient(problem, y, lam)
        return (x, x_next), x_next

    ks = jnp.arange(steps, dtype=jnp.int32)
    _, xs = jax.lax.scan(step, (problem.x0, problem.x0), ks)
    return jnp.concatenate([problem.x0[None, :], xs], axis=0)

def damping_coefficient(t: Array, alpha: float, dtype: jnp.dtype) -> Array:
    """a_t = exp(alpha*(t^(6/7)-(t-1)^(6/7)))-1, Eq. (15)."""
    t_float = t.astype(dtype)
    exponent = alpha * (
        t_float ** (6.0 / 7.0) - (t_float - 1.0) ** (6.0 / 7.0)
    )
    return jnp.expm1(exponent)


def smooth_velocity_control(
    u_sq: Array,
    m_sq: Array,
    r: float,
) -> Array:
    """The differentiable sigma in Eq. (17)."""
    q = u_sq / m_sq
    z = jnp.clip((q - 1.0) / 3.0, 0.0, 1.0)
    smoothstep = 3.0 * z**2 - 2.0 * z**3
    return 1.0 - (1.0 - r) * smoothstep


def algorithm3_step(
    *,
    alpha: float,
    x_prev: Array,
    t: Array,
    X_prev: Array,
    A_prev: Array,
    t0: Array,
    X_cut: Array,
    A_cut: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """One recursive output update from Algorithm 3 / Eq. (A1).

    The state starts from X_1=A_1=0, t0=1, X_cut=A_cut=0.
    This routine is called for t>=2 and returns x_bar_t=X_t/A_t.
    """
    dtype = x_prev.dtype
    t_float = t.astype(dtype)
    tm1_float = t_float - 1.0

    # Eq. (A1), written with exponent differences to avoid overflow.
    factor = (
        jnp.exp(
            alpha
            * (tm1_float ** (6.0 / 7.0) - t_float ** (6.0 / 7.0))
        )
        / t_float ** (1.0 / 7.0)
    )
    X_t = factor * (
        tm1_float ** (1.0 / 7.0) * X_prev + alpha * x_prev
    )
    A_t = factor * (
        tm1_float ** (1.0 / 7.0) * A_prev + alpha
    )

    def cut_branch(
        state: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array]:
        X_tmp, A_tmp, t0_tmp, X_cut_tmp, A_cut_tmp = state
        t0_float = t0_tmp.astype(dtype)
        cut_factor = (
            (t0_float / t_float) ** (1.0 / 7.0)
            * jnp.exp(
                alpha
                * (t0_float ** (6.0 / 7.0) - t_float ** (6.0 / 7.0))
            )
        )
        X_new = X_tmp - cut_factor * X_cut_tmp
        A_new = A_tmp - cut_factor * A_cut_tmp
        return X_new, A_new, t, X_new, A_new

    def no_cut_branch(
        state: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array]:
        return state

    X_t, A_t, t0_new, X_cut_new, A_cut_new = jax.lax.cond(
        t == 2 * t0,
        cut_branch,
        no_cut_branch,
        operand=(X_t, A_t, t0, X_cut, A_cut),
    )
    x_bar_t = X_t / A_t
    return X_t, A_t, t0_new, X_cut_new, A_cut_new, x_bar_t


def velocity_control_trajectory(
    problem: Problem,
    lam: Array,
    steps: int,
    alpha: float = 0.1,
    r: float = 0.5,
) -> tuple[Array, Array, Array, Array, Array]:
    """Run Algorithm 1 and compute its output recursively by Algorithm 3.

    Returns
    -------
    xs:
        [x_0,...,x_steps].
    x_bars:
        [x_bar_1,...,x_bar_steps].  The paper defines x_bar_t for t>=2;
        here x_bar_1 is set to x_0 only as a harmless plotting convention.
    r_hats:
        [r_hat_1,...,r_hat_steps].
    thresholds:
        [m_1,...,m_steps].
    trial_velocity_norms:
        [||v^1_1||,...,||v^1_steps||].
    """
    if steps < 2:
        raise ValueError("steps must be at least 2 because x_bar_t is defined for t>=2")
    if not (0.0 <= r < 1.0):
        raise ValueError("r must satisfy 0 <= r < 1")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if problem.M <= 0.0:
        raise ValueError("This implementation expects M>0")

    r_prime = max(r, 0.5)
    h_sq = 4.0 * (1.0 - r_prime) / problem.L

    # Carry also contains the Algorithm-3 state.
    # X_1=A_1=0, t0=1, X_cut=A_cut=0.
    initial = (
        problem.x0,
        jnp.zeros_like(problem.x0),
        jnp.asarray(1.0, dtype=problem.x0.dtype),
        jnp.asarray(0.0, dtype=problem.x0.dtype),
        jnp.zeros_like(problem.x0),
        jnp.asarray(0.0, dtype=problem.x0.dtype),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.zeros_like(problem.x0),
        jnp.asarray(0.0, dtype=problem.x0.dtype),
    )

    def step(
        carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array],
        t: Array,
    ) -> tuple[
        tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array],
        tuple[Array, Array, Array, Array, Array],
    ]:
        (
            x_prev,
            v_prev,
            r_hat_prev,
            a_prev,
            X_prev,
            A_prev,
            t0,
            X_cut,
            A_cut,
        ) = carry

        # Algorithm 3 uses x_{t-1}.  It is defined for t>=2.
        def compute_output(_: None):
            return algorithm3_step(
                alpha=alpha,
                x_prev=x_prev,
                t=t,
                X_prev=X_prev,
                A_prev=A_prev,
                t0=t0,
                X_cut=X_cut,
                A_cut=A_cut,
            )

        def initial_output(_: None):
            return X_prev, A_prev, t0, X_cut, A_cut, x_prev

        X_t, A_t, t0_new, X_cut_new, A_cut_new, x_bar_t = jax.lax.cond(
            t >= 2,
            compute_output,
            initial_output,
            operand=None,
        )

        # Algorithm 1, Lines 5--9.
        a_t = damping_coefficient(t, alpha, problem.x0.dtype)
        trial_velocity = (
            r_hat_prev * (2.0 + a_t) / (2.0 + a_prev) * v_prev
            - h_sq * gradient(problem, x_prev, lam)
        ) / (1.0 + a_t)

        t_float = t.astype(problem.x0.dtype)
        m_t = (
            6.0
            * r_prime
            * alpha
            / (7.0 * h_sq * problem.M * t_float ** (1.0 / 7.0))
        )
        trial_norm_sq = jnp.dot(trial_velocity, trial_velocity)
        r_hat_t = smooth_velocity_control(trial_norm_sq, m_t**2, r)

        v_t = (1.0 + a_t) / (2.0 - r_hat_t + a_t) * trial_velocity
        x_t = x_prev + v_t

        next_carry = (
            x_t,
            v_t,
            r_hat_t,
            a_t,
            X_t,
            A_t,
            t0_new,
            X_cut_new,
            A_cut_new,
        )
        output = (
            x_t,
            x_bar_t,
            r_hat_t,
            m_t,
            jnp.sqrt(trial_norm_sq),
        )
        return next_carry, output

    times = jnp.arange(1, steps + 1, dtype=jnp.int32)
    _, outputs = jax.lax.scan(step, initial, times)
    xs, x_bars, r_hats, thresholds, trial_velocity_norms = outputs
    xs = jnp.concatenate([problem.x0[None, :], xs], axis=0)
    return xs, x_bars, r_hats, thresholds, trial_velocity_norms


def exact_solution_jacobian(problem: Problem) -> Array:
    """Exact d x_star(lambda)/d lambda at lambda=0."""
    return -problem.b / hessian_diag(problem, problem.x_star)


def direct_weighted_output(
    trajectory: np.ndarray,
    iteration: int,
    alpha: float,
) -> np.ndarray:
    """Direct Eq. (13), used only to verify the Algorithm-3 recursion."""
    if iteration < 2:
        return trajectory[0].copy()
    i = int(np.floor(np.log2(iteration)))
    t0 = 2 ** (i - 1)
    indices = np.arange(t0, iteration)
    log_weights = alpha * indices ** (6.0 / 7.0)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    return np.tensordot(weights, trajectory[t0:iteration], axes=(0, 0))


def relative_error(x: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float(np.linalg.norm(x - reference))
    return float(np.linalg.norm(x - reference) / denominator)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def save_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    rows = zip(*(columns[name] for name in names), strict=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerows(rows)


def save_semilogy(
    path: Path,
    iterations: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    ylabel: str,
) -> None:
    plt.figure(figsize=(6.4, 4.3))
    for label, values in series:
        plt.semilogy(iterations, np.maximum(values, 1.0e-18), label=label)
    plt.xlabel("#grad calls",fontsize=12)
    plt.ylabel(ylabel,fontsize=12)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=12)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = make_problem(
        dim=args.dim,
        seed=args.seed,
        x0_radius=args.x0_radius,
    )
    lam0 = jnp.asarray(0.0, dtype=problem.x0.dtype)

    gd_map: Callable[[Array], Array] = lambda lam: gd_trajectory(
        problem, lam, args.steps, args.gd_step_factor
    )
    nag_map: Callable[[Array], Array] = lambda lam: nag_trajectory(
        problem, lam, args.steps, args.nag_step_factor
    )
    vc_map = lambda lam: velocity_control_trajectory(
        problem, lam, args.steps, args.alpha, args.r
    )

    # Since lambda is scalar, a JVP with tangent 1 gives d/dlambda of all outputs.
    tangent_one = jnp.asarray(1.0, dtype=problem.x0.dtype)
    gd_xs, gd_jacobians = jax.jvp(
        gd_map,
        (lam0,),
        (tangent_one,),
    )
    nag_xs, nag_jacobians = jax.jvp(
        nag_map,
        (lam0,),
        (tangent_one,),
    )
    vc_primal, vc_tangent = jax.jvp(
        vc_map,
        (lam0,),
        (tangent_one,),
    )
    vc_xs, vc_xbars, r_hats, thresholds, trial_velocity_norms = vc_primal
    vc_jacobians = vc_tangent[0]
    vc_xbar_jacobians = vc_tangent[1]

    grad_batch = jax.vmap(lambda x: gradient(problem, x, lam0))
    objective_batch = jax.vmap(lambda x: objective(problem, x, lam0))

    f_star = float(objective(problem, problem.x_star, lam0))
    j_star = exact_solution_jacobian(problem)
    j_star_norm = jnp.linalg.norm(j_star)

    gd_distance = jnp.linalg.norm(gd_xs - problem.x_star, axis=1)
    nag_distance = jnp.linalg.norm(nag_xs - problem.x_star, axis=1)
    vc_iterate_distance = jnp.linalg.norm(vc_xs - problem.x_star, axis=1)
    vc_output_distance = jnp.linalg.norm(vc_xbars - problem.x_star, axis=1)

    gd_grad_norm = jnp.linalg.norm(grad_batch(gd_xs), axis=1)
    nag_grad_norm = jnp.linalg.norm(grad_batch(nag_xs), axis=1)
    vc_iterate_grad_norm = jnp.linalg.norm(grad_batch(vc_xs), axis=1)
    vc_output_grad_norm = jnp.linalg.norm(grad_batch(vc_xbars), axis=1)

    gd_function_gap = objective_batch(gd_xs) - f_star
    nag_function_gap = objective_batch(nag_xs) - f_star
    vc_iterate_function_gap = objective_batch(vc_xs) - f_star
    vc_output_function_gap = objective_batch(vc_xbars) - f_star

    gd_jacobian_error = (
        jnp.linalg.norm(gd_jacobians - j_star, axis=1) / j_star_norm
    )
    nag_jacobian_error = (
        jnp.linalg.norm(nag_jacobians - j_star, axis=1) / j_star_norm
    )
    vc_iterate_jacobian_error = (
        jnp.linalg.norm(vc_jacobians - j_star, axis=1) / j_star_norm
    )
    vc_output_jacobian_error = (
        jnp.linalg.norm(vc_xbar_jacobians - j_star, axis=1) / j_star_norm
    )

    # Align Algorithm-3 outputs with iteration indices 1,...,T.
    nan_row = np.full((1,), np.nan)
    arrays = {
        "iteration": np.arange(args.steps + 1),
        "gd_distance": np.asarray(gd_distance),
        "nag_distance": np.asarray(nag_distance),
        "vc_iterate_distance": np.asarray(vc_iterate_distance),
        "vc_output_distance": np.concatenate([nan_row, np.asarray(vc_output_distance)]),
        "gd_gradient_norm": np.asarray(gd_grad_norm),
        "nag_gradient_norm": np.asarray(nag_grad_norm),
        "vc_iterate_gradient_norm": np.asarray(vc_iterate_grad_norm),
        "vc_output_gradient_norm": np.concatenate([nan_row, np.asarray(vc_output_grad_norm)]),
        "gd_function_gap": np.asarray(gd_function_gap),
        "nag_function_gap": np.asarray(nag_function_gap),
        "vc_iterate_function_gap": np.asarray(vc_iterate_function_gap),
        "vc_output_function_gap": np.concatenate([nan_row, np.asarray(vc_output_function_gap)]),
        "gd_jacobian_relative_error": np.asarray(gd_jacobian_error),
        "nag_jacobian_relative_error": np.asarray(nag_jacobian_error),
        "vc_iterate_jacobian_relative_error": np.asarray(vc_iterate_jacobian_error),
        "vc_output_jacobian_relative_error": np.concatenate([nan_row, np.asarray(vc_output_jacobian_error)]),
        "r_hat": np.concatenate([nan_row, np.asarray(r_hats)]),
        "velocity_threshold": np.concatenate([nan_row, np.asarray(thresholds)]),
        "trial_velocity_norm": np.concatenate([nan_row, np.asarray(trial_velocity_norms)]),
    }
    save_csv(output_dir / "metrics.csv", arrays)

    # Main comparison uses the formal output x_bar_t of Algorithm 1 (t>=2).
    iterations = np.arange(2, args.steps + 1)
    save_semilogy(
        output_dir / "diff_dist.pdf",
        iterations,
        [
            ("GD", arrays["gd_distance"][2:]),
            ("NAG", arrays["nag_distance"][2:]),
            ("VC", arrays["vc_output_distance"][2:]),
        ],
        r"Distance to the local minimum",
    )
    save_semilogy(
        output_dir / "diff_grad.pdf",
        iterations,
        [
            ("GD", arrays["gd_gradient_norm"][2:]),
            ("NAG", arrays["nag_gradient_norm"][2:]),
            ("VC", arrays["vc_output_gradient_norm"][2:]),
        ],
        r"Gradient norm",
    )
    save_semilogy(
        output_dir / "diff_jacobian.pdf",
        iterations,
        [
            ("GD", arrays["gd_jacobian_relative_error"][2:]),
            ("NAG", arrays["nag_jacobian_relative_error"][2:]),
            ("VC", arrays["vc_output_jacobian_relative_error"][2:]),
        ],
        r"Relative error in Jacobian",
    )

    plt.figure(figsize=(6.4, 4.3))
    plt.plot(np.arange(1, args.steps + 1), arrays["r_hat"][1:])
    plt.xlabel("#grad calls")
    plt.ylabel(r"$\hat r_t$")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_control_rhat.pdf", dpi=180)
    plt.close()

    gd_xs_np = np.asarray(gd_xs)
    nag_xs_np = np.asarray(nag_xs)
    vc_xs_np = np.asarray(vc_xs)
    vc_xbars_np = np.asarray(vc_xbars)
    gd_jac_np = np.asarray(gd_jacobians)
    nag_jac_np = np.asarray(nag_jacobians)
    vc_jac_np = np.asarray(vc_jacobians)
    vc_xbar_jac_np = np.asarray(vc_xbar_jacobians)
    x_star_np = np.asarray(problem.x_star)
    j_star_np = np.asarray(j_star)

    # Verify that Algorithm 3 equals the direct definition (13).
    alg3_errors = []
    for t in range(2, args.steps + 1):
        direct = direct_weighted_output(vc_xs_np, t, args.alpha)
        alg3_errors.append(np.linalg.norm(vc_xbars_np[t - 1] - direct))
    max_alg3_error = max(alg3_errors) if alg3_errors else 0.0

    default_checkpoints = [2, 5, 10, 20, 50, 100, 200, 500, 1000, args.steps]
    checkpoints = sorted({t for t in default_checkpoints if 2 <= t <= args.steps})

    print("\nProblem information")
    print(f"  dimension                   : {args.dim}")
    print("  lambda0                     : 0")
    print(f"  global gradient Lipschitz L : {problem.L:.6g}")
    print(f"  Hessian Lipschitz M         : {problem.M:.6g}")
    print(
        "  local Hessian condition no. : "
        f"{float(jnp.max(hessian_diag(problem, problem.x_star)) / jnp.min(hessian_diag(problem, problem.x_star))):.6g}"
    )
    print(f"  ||J_star||                  : {float(j_star_norm):.6g}")
    print(f"  max Alg.3/direct discrepancy: {max_alg3_error:.3e}")

    print("\nComparison with the formal Algorithm-1 output")
    header = (
        " iter | GD dist   NAG dist  VC xbar dist | GD Jacobian err  NAG Jacobian err  VC xbar Jacobian err "
        "| VC last-iterate Jacobian err"
    )
    print(header)
    print("-" * len(header))
    for t in checkpoints:
        # vc_xbars[t-1] is x_bar_t because x_bars starts at t=1.
        print(
            f"{t:5d} |"
            f" {np.linalg.norm(gd_xs_np[t] - x_star_np):9.3e}"
            f" {np.linalg.norm(nag_xs_np[t] - x_star_np):9.3e}"
            f" {np.linalg.norm(vc_xbars_np[t - 1] - x_star_np):12.3e} |"
            f" {relative_error(gd_jac_np[t], j_star_np):15.3e}"
            f" {relative_error(nag_jac_np[t], j_star_np):16.3e}"
            f" {relative_error(vc_xbar_jac_np[t - 1], j_star_np):20.3e} |"
            f" {relative_error(vc_jac_np[t], j_star_np):26.3e}"
        )

    print("\nFinal Jacobian diagnostics")
    print(
        f"  GD cosine(J_T,J_star)       : "
        f"{cosine_similarity(gd_jac_np[-1], j_star_np):.9f}"
    )
    print(
        f"  NAG cosine(J_T,J_star)      : "
        f"{cosine_similarity(nag_jac_np[-1], j_star_np):.9f}"
    )
    print(
        f"  VC xbar cosine(J,J_star)    : "
        f"{cosine_similarity(vc_xbar_jac_np[-1], j_star_np):.9f}"
    )
    print(
        f"  VC x_T cosine(J_T,J_star)   : "
        f"{cosine_similarity(vc_jac_np[-1], j_star_np):.9f}"
    )

    if not args.skip_fd_check:
        eps = args.fd_epsilon

        gd_plus = np.asarray(gd_map(lam0 + eps)[-1])
        gd_minus = np.asarray(gd_map(lam0 - eps)[-1])
        gd_fd = (gd_plus - gd_minus) / (2.0 * eps)

        nag_plus = np.asarray(nag_map(lam0 + eps)[-1])
        nag_minus = np.asarray(nag_map(lam0 - eps)[-1])
        nag_fd = (nag_plus - nag_minus) / (2.0 * eps)

        vc_plus = vc_map(lam0 + eps)
        vc_minus = vc_map(lam0 - eps)
        vc_xbar_plus = np.asarray(vc_plus[1][-1])
        vc_xbar_minus = np.asarray(vc_minus[1][-1])
        vc_xbar_fd = (vc_xbar_plus - vc_xbar_minus) / (2.0 * eps)

        print("\nAutodiff versus centered finite differences")
        print(
            f"  GD final iterate discrepancy: "
            f"{relative_error(gd_jac_np[-1], gd_fd):.3e}"
        )
        print(
            f"  NAG final iterate discrepancy: "
            f"{relative_error(nag_jac_np[-1], nag_fd):.3e}"
        )
        print(
            f"  VC Algorithm-3 output       : "
            f"{relative_error(vc_xbar_jac_np[-1], vc_xbar_fd):.3e}"
        )

    print(f"\nSaved results to: {output_dir.resolve()}")
    print("  metrics.csv")
    print("  forward_distance.png")
    print("  gradient_norm.png")
    print("  jacobian_relative_error.png")
    print("  velocity_control_rhat.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--x0-radius", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--gd-step-factor", type=float, default=1.0)
    parser.add_argument("--nag-step-factor", type=float, default=1.0)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--skip-fd-check", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
