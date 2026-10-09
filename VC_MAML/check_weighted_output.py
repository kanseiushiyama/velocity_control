"""Check the VC averaging window and its differentiated output independently."""
from pathlib import Path
import argparse
import os
import sys
import json
os.environ['JAX_ENABLE_X64'] = 'False'
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parent / 'maml_code'))
import jax
import jax.numpy as jnp
import numpy as np
import maml_benchmark_core as core


def reference(theta0, x, y, cfg):
    # Retain the full trajectory and average afterwards, independently of
    # the supplied scan's accumulation and window-index calculation.
    theta = theta0
    velocity = jnp.zeros_like(theta)
    rprev, aprev = 1., 0.
    states = [theta]
    rp = max(cfg.inner_vc_r, .5)
    h2 = 4*(1-rp)/cfg.inner_vc_L_proxy
    gradient = jax.grad(lambda z: core.mse(z,x,y,cfg.hidden_width))
    for t in range(1,cfg.inner_steps+1):
        a = core.damping_coefficient(jnp.asarray(t,theta.dtype),cfg.inner_vc_alpha)
        v1 = (rprev*(2+a)/(2+aprev)*velocity-h2*gradient(theta))/(1+a)
        threshold = 6*rp*cfg.inner_vc_alpha/(7*h2*cfg.inner_vc_M_proxy*t**(1/7))
        rh = core.velocity_control_sigma(jnp.vdot(v1,v1).real,threshold**2,cfg.inner_vc_r)
        velocity = (1+a)/(2-rh+a)*v1
        theta = theta+velocity
        states.append(theta)
        rprev,aprev = rh,a
    power = 1
    while 2*power <= cfg.inner_steps:
        power *= 2
    start = power//2
    times = np.arange(start,cfg.inner_steps)
    weights = np.exp(cfg.inner_vc_alpha*times**(6/7))
    weights /= weights.sum()
    return jnp.einsum('i,ij->j',jnp.asarray(weights,dtype=theta.dtype),jnp.stack(states[start:cfg.inner_steps])), states[-1], times


def run_checks():
    records=[]
    for k in [2,4,5,8,20]:
        cfg=core.ExperimentConfig(hidden_width=4,inner_steps=k)
        theta=core.initialize_parameters(jax.random.PRNGKey(3),4)
        task=core.sample_task_batch(jax.random.PRNGKey(7),cfg,1)
        sx,sy,qx,qy=(a[0] for a in task)
        adapter=core.make_adapter(cfg,'vc')
        actual=adapter(theta,sx,sy)
        expected,last,times=reference(theta,sx,sy,cfg)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        grad_actual=jax.grad(lambda z:core.mse(adapter(z,sx,sy),qx,qy,4))(theta)
        grad_expected=jax.grad(lambda z:core.mse(reference(z,sx,sy,cfg)[0],qx,qy,4))(theta)
        np.testing.assert_allclose(grad_actual,grad_expected,rtol=2e-4,atol=2e-5)
        difference=float(jnp.linalg.norm(actual-last))
        assert difference>1e-6
        records.append(dict(inner_steps=k,averaged_indices=times.tolist(),distance_from_last_iterate=difference,status='pass'))
    try:
        core.make_adapter(core.ExperimentConfig(inner_steps=1),'vc')
    except ValueError:
        pass
    else:
        raise AssertionError('K=1 must be rejected')
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'runs' / 'weighted-output-check' / 'weighted_output_checks.json',
                        help='New JSON output file; an existing file is never overwritten.')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f'Refusing to overwrite {output}; choose a new --output.')
    records = run_checks()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects a file created while the checks run.
    with output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))
    print(f'Checks passed. Report: {output}')


if __name__ == '__main__':
    main()
