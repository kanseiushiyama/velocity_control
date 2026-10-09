"""Protect ordinary and predictive joint evaluation and component counts."""

from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer import AdaVeloControlCheckpoint, AdaVeloControlDeferredRCheckpoint


class Quadratic:
    def __init__(self, curvature=0.25):
        self.name = "PredictiveEvaluationTest"
        self.param = jnp.asarray([1., -2.], dtype=jnp.float64)
        self.curvature = curvature

    def loss(self, x):
        return 0.5 * self.curvature * jnp.vdot(x, x)


def initial_state(optim, L=4.):
    x = optim.model.param
    zero = jnp.asarray(0., dtype=x.dtype)
    value, gradient = jax.value_and_grad(optim.model.loss)(x)
    return dict(t=jnp.asarray(1, dtype=jnp.int32), r_b=jnp.asarray(1., dtype=x.dtype),
                a_b=zero, v_b=jnp.zeros_like(x), x=x, x0=x, val=value, val0=value,
                grad=gradient, L_est=jnp.asarray(L), L_b=jnp.asarray(L),
                t0=jnp.asarray(1, dtype=jnp.int32), xbar_num=jnp.zeros_like(x),
                S=zero, x_cut=jnp.zeros_like(x), S_cut=zero)


def batch_args(state, n_steps):
    args = dict(state)
    t = args.pop("t")
    args.pop("r_b")
    return dict(n_steps=jnp.asarray(n_steps, dtype=jnp.int32), t_start=t - 1, **args)


class PredictiveEvaluationTests(unittest.TestCase):
    def assert_cached_pair(self, model, result):
        # Analytic values do not depend on the optimizer's evaluation path.
        x = np.asarray(result.x)
        np.testing.assert_allclose(result.val, .5 * model.curvature * np.dot(x, x), rtol=1e-13)
        np.testing.assert_allclose(result.grad, model.curvature * x, rtol=1e-13)

    def test_predictive_batch_counts_and_cached_pair(self):
        for cls in (AdaVeloControlCheckpoint, AdaVeloControlDeferredRCheckpoint):
            with self.subTest(variant=cls.__name__):
                model = Quadratic()
                opt = cls(model, R=.5, alpha=.1, line_search_mode="parallel")
                out = opt._ada_r1_batch_step(**batch_args(initial_state(opt), 5))
                self.assertTrue(bool(out.success))
                self.assertEqual((int(out.attempted), int(out.completed)), (5, 5))
                self.assertEqual((int(out.n_val), int(out.n_grad)), (5, 5))
                self.assert_cached_pair(model, out)

    def test_checkpoint_counts_and_fallback(self):
        for cls, expected_values in ((AdaVeloControlCheckpoint, 3),
                                     (AdaVeloControlDeferredRCheckpoint, 2)):
            with self.subTest(variant=cls.__name__):
                model = Quadratic()
                opt = cls(model, R=.5, alpha=.1, line_search_mode="parallel")
                out = opt._ada_checkpoint_step(**initial_state(opt))
                self.assertTrue(bool(out.success))
                self.assertEqual((int(out.n_val), int(out.n_grad)), (expected_values, 1))
                self.assert_cached_pair(model, out)
                x0 = np.asarray(out.x0)
                np.testing.assert_allclose(out.val0, .5 * model.curvature * np.dot(x0, x0), rtol=1e-13)

    def test_ordinary_step_charges_joint_value_reevaluation(self):
        for cls, expected_values in ((AdaVeloControlCheckpoint, 4),
                                     (AdaVeloControlDeferredRCheckpoint, 3)):
            with self.subTest(variant=cls.__name__):
                model = Quadratic()
                opt = cls(model, R=.5, alpha=.1, line_search_mode="parallel")
                out = opt._ada_full_step(**initial_state(opt))
                self.assertEqual(float(out.r), 1.)
                self.assertFalse(bool(out.rewrite_prev))
                self.assertEqual((int(out.n_val), int(out.n_grad)), (expected_values, 1))
                self.assert_cached_pair(model, out)

    def test_failed_prediction_is_charged_and_preserves_state(self):
        # Deliberately underestimate curvature so the first E^- test fails.
        model = Quadratic(curvature=8.)
        opt = AdaVeloControlDeferredRCheckpoint(model, R=.5, alpha=.1, line_search_mode="parallel")
        st = initial_state(opt, L=1.)
        batch = opt._ada_r1_batch_step(**batch_args(st, 5))
        checkpoint = opt._ada_checkpoint_step(**st)
        self.assertEqual((int(batch.attempted), int(batch.completed)), (1, 0))
        for out in (batch, checkpoint):
            self.assertFalse(bool(out.success))
            self.assertEqual((int(out.n_val), int(out.n_grad)), (1, 1))
            np.testing.assert_array_equal(out.x, st["x"])
            np.testing.assert_array_equal(out.grad, st["grad"])
            self.assertEqual(float(out.val), float(st["val"]))


if __name__ == "__main__":
    unittest.main()
