"""Check VC's averaged output and its derivative against an explicit trajectory.

This is a small mathematical consistency check, not a rerun of paper experiments.
Run in its own process so the MAML float32 setting cannot leak to other suites.
"""
from __future__ import annotations
import os
os.environ['JAX_ENABLE_X64'] = 'false'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'VC_MAML/maml_code'))
import jax
import jax.numpy as jnp
import numpy as np
from maml_benchmark_core import ExperimentConfig, initialize_parameters, make_adapter, mse


def explicit_reference(theta0, x, y, config):
    theta = theta0
    velocity = jnp.zeros_like(theta)
    rhat = jnp.asarray(1.0, theta.dtype)
    previous_damping = jnp.asarray(0.0, theta.dtype)
    rprime = max(config.inner_vc_r, 0.5)
    h2 = 4.0 * (1.0 - rprime) / config.inner_vc_L_proxy
    trajectory = [theta]
    gradient = jax.grad(lambda w: mse(w, x, y, config.hidden_width))
    for t in range(1, config.inner_steps + 1):
        tf = jnp.asarray(t, theta.dtype)
        damping = jnp.expm1(config.inner_vc_alpha * (tf**(6.0 / 7.0) - (tf - 1)**(6.0 / 7.0)))
        trial = (rhat * (2 + damping) / (2 + previous_damping) * velocity - h2 * gradient(theta)) / (1 + damping)
        threshold = 6 * rprime * config.inner_vc_alpha / (7 * h2 * config.inner_vc_M_proxy * tf**(1.0 / 7.0))
        denominator = jnp.maximum(threshold**2, jnp.finfo(theta.dtype).tiny)
        z = jnp.clip((jnp.vdot(trial, trial).real / denominator - 1) / 3, 0, 1)
        rhat = 1 - (1 - config.inner_vc_r) * (3 * z**2 - 2 * z**3)
        velocity = (1 + damping) / (2 - rhat + damping) * trial
        theta = theta + velocity
        previous_damping = damping
        trajectory.append(theta)
    start = 2 ** (config.inner_steps.bit_length() - 2)
    taus = jnp.arange(start, config.inner_steps, dtype=theta.dtype)
    log_weights = config.inner_vc_alpha * taus**(6.0 / 7.0)
    weights = jax.nn.softmax(log_weights)
    return jnp.sum(weights[:, None] * jnp.stack(trajectory[start:config.inner_steps]), axis=0)


def main():
    for steps in (2, 5, 20):
        config = ExperimentConfig(inner_steps=steps, hidden_width=4)
        theta = initialize_parameters(jax.random.PRNGKey(7), config.hidden_width).astype(jnp.float32)
        x = jnp.linspace(-2, 2, 10, dtype=jnp.float32)
        y = 2 * jnp.sin(x + 0.4)
        candidate = make_adapter(config, 'vc')
        production = jax.jit(lambda w: candidate(w, x, y))
        reference = jax.jit(lambda w: explicit_reference(w, x, y, config))
        a, b = np.asarray(production(theta)), np.asarray(reference(theta))
        jac_a = np.asarray(jax.jacfwd(production)(theta))
        jac_b = np.asarray(jax.jacfwd(reference)(theta))
        np.testing.assert_allclose(a, b, rtol=5e-5, atol=2e-6)
        np.testing.assert_allclose(jac_a, jac_b, rtol=5e-4, atol=2e-5)
        print(f'K={steps}: max output error={np.max(np.abs(a-b)):.3e}, max Jacobian error={np.max(np.abs(jac_a-jac_b)):.3e}')
    print('All weighted-output checks passed (float32).')


if __name__ == '__main__':
    main()
