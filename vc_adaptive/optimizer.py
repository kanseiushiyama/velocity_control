import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple
from obj_func import *
jax.config.update("jax_enable_x64", True)

class GDBTState(NamedTuple):
    L: Any
    h: Any
    v: Any
    x_cand: Any
    val_cand: Any
    res: Any
    n_trial: Any


class AdaLSState(NamedTuple):
    L: Any
    val_cand: Any
    res: Any
    fail: Any
    n_trial: Any
    n_val: Any
    stats: Any


class AdaStep(NamedTuple):
    r: Any
    a: Any
    v: Any
    x: Any
    x0: Any
    val: Any
    val0: Any
    grad: Any
    L: Any
    L_prev: Any
    t0: Any
    xbar_num: Any
    S: Any
    x_cut: Any
    S_cut: Any
    xbar: Any
    n_val: Any
    n_grad: Any
    rewrite_prev: Any
    M: Any
    ls_stats: Any


class AdaPredictStep(NamedTuple):
    success: Any
    a: Any
    v: Any
    x: Any
    x0: Any
    val: Any
    val0: Any
    grad: Any
    L: Any
    L_prev: Any
    t0: Any
    xbar_num: Any
    S: Any
    x_cut: Any
    S_cut: Any
    xbar: Any
    n_val: Any
    n_grad: Any
    M: Any
    ls_stats: Any


class AdaSpecBatch(NamedTuple):
    success: Any
    attempted: Any
    completed: Any
    a: Any
    v: Any
    x: Any
    x0: Any
    val: Any
    val0: Any
    grad: Any
    L: Any
    L_prev: Any
    t0: Any
    xbar_num: Any
    S: Any
    x_cut: Any
    S_cut: Any
    xbar: Any
    n_val: Any
    n_grad: Any
    M: Any
    ls_stats: Any


class HBStep(NamedTuple):
    k: Any
    h: Any
    x: Any
    xbar: Any
    v: Any
    val: Any
    grad: Any
    val_bar: Any
    grad_bar: Any
    L: Any
    S: Any
    val_best: Any
    x_best: Any
    grad_best: Any
    val_out: Any
    grad_norm_out: Any
    n_val: Any
    n_grad: Any


class AGDStep(NamedTuple):
    k: Any
    M: Any
    S: Any
    Z: Any
    x: Any
    y: Any
    ybar: Any
    val0: Any
    val_x: Any
    grad_x: Any
    grad_y: Any
    L: Any
    n_val: Any
    n_grad: Any


class NAGLSState(NamedTuple):
    L: Any
    A: Any
    d: Any
    y: Any
    grad: Any
    x_cand: Any
    val_y: Any
    val_cand: Any
    ok: Any
    n_trial: Any


class NAGStep(NamedTuple):
    L: Any
    A: Any
    d: Any
    y: Any
    grad: Any
    x_cand: Any
    val_y: Any
    val_cand: Any
    ok: Any
    n_trial: Any
    z: Any


class BaseOptimizer:
    def __init__(self, model):
        self.save_hist = True
        self.stop_mode = "iter"
        self.stop_mode = "time"
        self.n_iter = 17500
        self.time_limit = 100
        self.model = model
        self.oracle = Oracle(model)
        self.grad_best = jnp.asarray(jnp.inf, dtype=model.param.dtype)
        self.elapsed_time = 0
        self.grad_hist = []
        self.val_hist = []
        self.grad_call_hist = []
        self.val_call_hist = []
        self.grad_val_call_hist = []
        self.joint_call_hist = []
        self.oracle_call_hist = []
        self.time_hist = []
        self.acc_hist = []
        self.acc_every = 100
        self.hist_every = 1
        self.result_dir = Path("./result")

    def _savez(self, suffix, **data):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        data.setdefault("joint_call", np.asarray(self.joint_call_hist))
        data.setdefault("oracle_call", np.asarray(self.oracle_call_hist))
        np.savez(self.result_dir / f"{self.model.name}{suffix}.npz", **data)

    def _should_save_hist(self, it):
        return it % self.hist_every == 0

    def _save_call_hist(self):
        grad_calls = self.oracle.count["grad"]
        val_calls = self.oracle.count["val"]
        self.grad_call_hist.append(grad_calls)
        self.val_call_hist.append(val_calls)
        self.grad_val_call_hist.append(grad_calls + val_calls)
        self.joint_call_hist.append(self.oracle.count["joint"])
        self.oracle_call_hist.append(self.oracle.oracle_calls)

    def stop_condition(self):
        if self.stop_mode == "iter":
            return self.oracle.count["grad"] <= self.n_iter
        elif self.stop_mode == "time":
            return self.elapsed_time <= self.time_limit

    def update(self):
        pass

    @staticmethod
    def _block_until_ready(*values):
        for value in values:
            if value is not None:
                jax.block_until_ready(value)

    def _warmup_kernels(self, x, val, grad):
        """Hook for subclasses with additional JIT-compiled kernels."""
        pass

    def warmup(self):
        """Compile JAX kernels outside wall-clock timing and reset counters."""
        x = self.model.param
        val = self.oracle.loss(x, count=False)
        grad = self.oracle.grad(x, count=False)
        val_vg, grad_vg = self.oracle.value_and_grad(x, count=False)

        # Warm up common eager JAX primitives that otherwise incur a one-time
        # compilation cost inside the first timed it.
        half = jnp.asarray(0.5, dtype=x.dtype)
        tmp_add = x + grad
        tmp_sub = x - half * grad
        dot = jnp.vdot(grad, grad)
        norm = jnp.linalg.norm(grad)
        scalar = val + half * dot
        comp = scalar > val
        comp_less = val < scalar
        maximum = jnp.maximum(scalar, val)
        minimum = jnp.minimum(scalar, val)
        root = jnp.sqrt(jnp.abs(scalar) + 1.0)
        selected = jax.lax.select(comp, tmp_add, tmp_sub)
        self._block_until_ready(
            val, grad, val_vg, grad_vg, tmp_add, tmp_sub, dot, norm,
            scalar, comp, comp_less, maximum, minimum, root, selected,
        )
        self._warmup_kernels(x, val, grad)
        self.oracle.reset_count()
        self.elapsed_time = 0

class Gradient_descent(BaseOptimizer):
    def __init__(self, model):
        super().__init__(model)
        self.beta_inc = 1.1
        self.beta_dec = 0.98
        self.L_est = 0.1
        self.h = 1 / self.L_est

        loss = model.loss
        beta_inc = self.beta_inc
        beta_dec = self.beta_dec

        @jax.jit
        def backtrack(x, val, grad, L_est):
            def trial(L):
                h = 1.0 / L
                v = -h * grad
                x_cand = x + v
                val_n = loss(x_cand)
                cr = val_n - val - jnp.vdot(grad, v) - 0.5 * L * jnp.vdot(v, v)
                return h, v, x_cand, val_n, cr

            h, v, x_cand, val_n, cr = trial(L_est)
            st = GDBTState(
                L_est, h, v, x_cand, val_n, cr,
                jnp.asarray(1, dtype=jnp.int32),
            )

            def cond(st):
                return st.res > 0

            def body(st):
                L = st.L * beta_inc
                h, v, x_cand, val_n, cr = trial(L)
                return GDBTState(L, h, v, x_cand, val_n, cr, st.n_trial + 1)

            st = jax.lax.while_loop(cond, body, st)
            # Carry the next Lipschitz estimate on device.
            return GDBTState(
                st.L * beta_dec, st.h, st.v, st.x_cand, st.val_cand,
                st.res, st.n_trial,
            )

        self._gd_backtrack = backtrack

    def _warmup_kernels(self, x, val, grad):
        out = self._gd_backtrack(x, val, grad, jnp.asarray(self.L_est, dtype=x.dtype))
        self._block_until_ready(*out)

    def save_state(self, x, grad, val, save=True):
        grad_norm = jnp.linalg.norm(grad)
        better = grad_norm < self.grad_best
        self.grad_best = jnp.minimum(self.grad_best, grad_norm)
        self.model.param = jax.lax.select(better, x, self.model.param)
        if save:
            self.grad_hist.append(grad_norm)
            self.val_hist.append(val)

    def optimize(self):
        x = self.model.param
        L_est = jnp.asarray(self.L_est, dtype=x.dtype)
        val = self.oracle.loss(x)
        self._block_until_ready(val)
        it = 0
        while self.stop_condition():
            it += 1
            do_record = self._should_save_hist(it)

            tic = time.perf_counter()
            grad = self.oracle.grad(x)
            self._block_until_ready(grad)
            self.elapsed_time += time.perf_counter() - tic

            # The reported gradient is already evaluated, so charge it at
            # this point rather than recording the pre-gradient counters.
            if do_record:
                self._save_call_hist()
                self.time_hist.append(self.elapsed_time)
            self.save_state(x, grad, val, save=do_record)

            tic = time.perf_counter()
            out = self._gd_backtrack(x, val, grad, L_est)
            L_est, h, _, x_cand, val_n, _, n_trial = out
            self._block_until_ready(x_cand, val_n, L_est, n_trial)
            self.elapsed_time += time.perf_counter() - tic

            # Host-side oracle bookkeeping is excluded from optimizer timing.
            self.oracle.add_count(val=int(jax.device_get(n_trial)))
            x = x_cand
            val = val_n

        self.L_est = float(jax.device_get(L_est))
        self.h = float(jax.device_get(h))

    def run(self):
        self.warmup()
        self.optimize()
        self.grad_call_hist = np.array(self.grad_call_hist)
        self.val_call_hist = np.array(self.val_call_hist)
        self.grad_val_call_hist = np.array(self.grad_val_call_hist)
        self.joint_call_hist = np.array(self.joint_call_hist)
        self.oracle_call_hist = np.array(self.oracle_call_hist)
        self.grad_hist = np.array(self.grad_hist)
        self.val_hist = np.array(self.val_hist)
        self.time_hist = np.array(self.time_hist)
        if self.save_hist:
            self._savez('_gd', grad_call=self.grad_call_hist, val_call=self.val_call_hist, grad_val_call=self.grad_val_call_hist, grad=self.grad_hist, val=self.val_hist, time=self.time_hist)

class AdaVeloControl(BaseOptimizer):
    def __init__(self, model, R=0, alpha=0.1, hmax=1, line_search_mode="parallel", defer_R_check=False):

        super().__init__(model)
        if line_search_mode == "short":
            line_search_mode = "short_circuit"
        if line_search_mode not in {"parallel", "short_circuit"}:
            raise ValueError(
                "line_search_mode must be 'parallel' or 'short_circuit'."
            )
        self.line_search_mode = line_search_mode
        self.defer_R_check = bool(defer_R_check)
        self.result_tag = "_adavelocontrol"
        self.deferred_R_checks = 0
        # Aggregate line-search outcomes in candidate order
        # [fail at r=1, fail at r=R, fail at r=0, all pass].
        self.line_search_stats = np.zeros(4, dtype=np.int64)

        self.L_est = 1
        self.alpha = alpha
        self.beta_inc = 1.1
        self.beta_dec = 0.9
        self.R = R
        self.hmax = hmax
        self.r_hist = []
        self.v_hist = []
        self.c67 = 6 / 7
        self.hcoef = min(4 * (1 - self.R), 2)

        loss = model.loss
        grad_fn = jax.grad(loss)
        value_grad_fn = jax.value_and_grad(loss)
        R = self.R
        rho_max = max(R, 0.5)
        hmax = self.hmax
        hcoef = self.hcoef
        beta_inc = self.beta_inc
        beta_dec = self.beta_dec
        alpha = self.alpha

        def h2_from_L(L_est):
            return jnp.minimum(hcoef / L_est, hmax)

        def cand_vpre(L_est, r_b, a, a_b, v_b, grad):
            h2 = h2_from_L(L_est)
            return r_b * (2 + a) / (2 + a_b) * v_b - h2 * grad

        def cand_den(a, dtype):
            return jnp.asarray([1 + a, 2 - R + a, 2 + a], dtype=dtype)

        def cand_point(v_pre, den_i, x_b):
            v_i = v_pre / den_i
            x_i = x_b + v_i
            # Preserve the realized displacement used by the original code.
            dx_i = x_i - x_b
            return v_i, x_i, dx_i

        def eminus_res(val_cand, grad, dx, val_b, L_est):
            """Numerically ordered E^- residual used in the original code."""
            grad_term = jnp.vdot(grad, dx)
            quad_term = 0.5 * L_est * jnp.vdot(dx, dx)
            # Preserve
            #   val_cand - <grad,dx> - (val_b + L||dx||^2/2)
            # instead of algebraically regrouping into (val_cand-val_b)-... .
            lhs = jax.lax.optimization_barrier(val_cand - grad_term)
            rhs = jax.lax.optimization_barrier(val_b + quad_term)
            return jax.lax.optimization_barrier(lhs - rhs)

        def m_estimate(val1, grad1, grad_b, dx1, val_b):
            """Original evaluation order for the Hessian-Lipschitz estimate."""
            dx_norm = jnp.linalg.norm(dx1)
            sym_term = 0.5 * jnp.vdot(grad1 + grad_b, dx1)
            tmp = jax.lax.optimization_barrier(val1 - sym_term)
            num = 12.0 * jax.lax.optimization_barrier(tmp - val_b)
            return jnp.where(
                dx_norm > 0,
                jnp.maximum(num / dx_norm ** 3, 0.0),
                0.0,
            )

        def eplus_res(val_cand, grad_cand, dx, val_b, L_est):
            """Numerically ordered E^+ residual used in the original code."""
            grad_term = jnp.vdot(grad_cand, dx)
            quad_term = 0.5 * L_est * jnp.vdot(dx, dx)
            lhs = jax.lax.optimization_barrier(val_cand - grad_term)
            rhs = jax.lax.optimization_barrier(val_b + quad_term)
            return jax.lax.optimization_barrier(lhs - rhs)

        def first_fail_stats(cr):
            # Candidate order is r=1, r=R, r=0.  The final bin means all pass.
            stage = jnp.where(
                cr[0] > 0, 0,
                jnp.where(cr[1] > 0, 1, jnp.where(cr[2] > 0, 2, 3)),
            )
            return jax.nn.one_hot(stage, 4, dtype=jnp.int32)

        def cand_trial_parallel_from_vpre(L_est, a, v_pre, grad, x_b, val):
            den = cand_den(a, x_b.dtype)

            # Vectorize the three value evaluations.  Candidates are generated
            # inside the vmapped function so the line-search state never carries
            # a (3,d) candidate array.
            def candidate_value(den_i):
                _, x_i, _ = cand_point(v_pre, den_i, x_b)
                return loss(x_i)

            val_n = jax.vmap(candidate_value)(den)

            # Preserve the original E^- arithmetic ordering candidate by
            # candidate, including the realized displacement x_i - x_b.
            def candidate_residual(den_i, val_i):
                _, _, dx_i = cand_point(v_pre, den_i, x_b)
                return eminus_res(val_i, grad, dx_i, val, L_est)

            cr = jax.vmap(candidate_residual)(den, val_n)
            failed = jnp.any(cr > 0)
            stats = first_fail_stats(cr)
            return (
                val_n, cr, failed,
                jnp.asarray(3, dtype=jnp.int32), stats,
            )

        def cand_trial_short_from_vpre(L_est, a, v_pre, grad, x_b, val):
            den = cand_den(a, x_b.dtype)
            nan = jnp.asarray(jnp.nan, dtype=x_b.dtype)
            val_n0 = jnp.full((3,), nan, dtype=x_b.dtype)
            cr0 = jnp.full((3,), nan, dtype=x_b.dtype)

            def eval_cr(i, val_n, cr):
                _, x_i, dx_i = cand_point(v_pre, den[i], x_b)
                vi = loss(x_i)
                ci = eminus_res(vi, grad, dx_i, val, L_est)
                return val_n.at[i].set(vi), cr.at[i].set(ci), ci

            val_n1, cr1_arr, cr1 = eval_cr(0, val_n0, cr0)

            def fail_at_1(_):
                stats = jnp.asarray([1, 0, 0, 0], dtype=jnp.int32)
                return (
                    val_n1, cr1_arr, jnp.asarray(True),
                    jnp.asarray(1, dtype=jnp.int32), stats,
                )

            def pass_1(_):
                val_n2, cr2_arr, cr2 = eval_cr(1, val_n1, cr1_arr)

                def fail_at_R(_):
                    stats = jnp.asarray([0, 1, 0, 0], dtype=jnp.int32)
                    return (
                        val_n2, cr2_arr, jnp.asarray(True),
                        jnp.asarray(2, dtype=jnp.int32), stats,
                    )

                def pass_R(_):
                    val_n3, cr3_arr, cr3 = eval_cr(2, val_n2, cr2_arr)
                    failed = cr3 > 0
                    stats = jax.lax.select(
                        failed,
                        jnp.asarray([0, 0, 1, 0], dtype=jnp.int32),
                        jnp.asarray([0, 0, 0, 1], dtype=jnp.int32),
                    )
                    return (
                        val_n3, cr3_arr, failed,
                        jnp.asarray(3, dtype=jnp.int32), stats,
                    )

                return jax.lax.cond(cr2 > 0, fail_at_R, pass_R, operand=None)

            return jax.lax.cond(cr1 > 0, fail_at_1, pass_1, operand=None)

        def cand_trial_parallel_defer_R_from_vpre(L_est, a, v_pre, grad, x_b, val):
            """Check only r=1 and r=0 before the M-test.

            The r=R candidate is deliberately deferred until r=1 is rejected
            by the M-test.  Keep a length-three value/residual vector so the
            rest of the VC state layout remains unchanged; slot 1 is unused.
            """
            den = cand_den(a, x_b.dtype)
            inds = jnp.asarray([0, 2], dtype=jnp.int32)

            def candidate_value(i):
                _, x_i, _ = cand_point(v_pre, den[i], x_b)
                return loss(x_i)

            vals = jax.vmap(candidate_value)(inds)

            def candidate_residual(i, val_i):
                _, _, dx_i = cand_point(v_pre, den[i], x_b)
                return eminus_res(val_i, grad, dx_i, val, L_est)

            res = jax.vmap(candidate_residual)(inds, vals)
            nan = jnp.asarray(jnp.nan, dtype=x_b.dtype)
            val_n = jnp.stack((vals[0], nan, vals[1]))
            cr = jnp.stack((res[0], nan, res[1]))
            failed = jnp.logical_or(res[0] > 0, res[1] > 0)
            stage = jnp.where(res[0] > 0, 0, jnp.where(res[1] > 0, 2, 3))
            stats = jax.nn.one_hot(stage, 4, dtype=jnp.int32)
            return (
                val_n, cr, failed,
                jnp.asarray(2, dtype=jnp.int32), stats,
            )

        def cand_trial_short_defer_R_from_vpre(L_est, a, v_pre, grad, x_b, val):
            """Short-circuit pre-M check in the order r=1, then r=0."""
            den = cand_den(a, x_b.dtype)
            nan = jnp.asarray(jnp.nan, dtype=x_b.dtype)
            val_n0 = jnp.full((3,), nan, dtype=x_b.dtype)
            cr0 = jnp.full((3,), nan, dtype=x_b.dtype)

            def eval_cr(i, val_n, cr):
                _, x_i, dx_i = cand_point(v_pre, den[i], x_b)
                vi = loss(x_i)
                ci = eminus_res(vi, grad, dx_i, val, L_est)
                return val_n.at[i].set(vi), cr.at[i].set(ci), ci

            val_n1, cr1_arr, cr1 = eval_cr(0, val_n0, cr0)

            def fail_at_1(_):
                return (
                    val_n1, cr1_arr, jnp.asarray(True),
                    jnp.asarray(1, dtype=jnp.int32),
                    jnp.asarray([1, 0, 0, 0], dtype=jnp.int32),
                )

            def pass_1(_):
                val_n2, cr2_arr, cr0_res = eval_cr(2, val_n1, cr1_arr)
                failed = cr0_res > 0
                stats = jax.lax.select(
                    failed,
                    jnp.asarray([0, 0, 1, 0], dtype=jnp.int32),
                    jnp.asarray([0, 0, 0, 1], dtype=jnp.int32),
                )
                return (
                    val_n2, cr2_arr, failed,
                    jnp.asarray(2, dtype=jnp.int32), stats,
                )

            return jax.lax.cond(cr1 > 0, fail_at_1, pass_1, operand=None)

        if self.defer_R_check:
            cand_trial_from_vpre = (
                cand_trial_parallel_defer_R_from_vpre
                if self.line_search_mode == "parallel"
                else cand_trial_short_defer_R_from_vpre
            )
        else:
            cand_trial_from_vpre = (
                cand_trial_parallel_from_vpre
                if self.line_search_mode == "parallel"
                else cand_trial_short_from_vpre
            )

        def cand_trial(L_est, r_b, a, a_b, v_b, grad, x_b, val):
            v_pre = cand_vpre(L_est, r_b, a, a_b, v_b, grad)
            return cand_trial_from_vpre(L_est, a, v_pre, grad, x_b, val)

        def cand_trial_rewrite(L_est, a, grad, x_b, val):
            # For the rewritten state r_b=0, hence the old velocity vanishes
            # algebraically. Avoid constructing a length-d zero vector.
            v_pre = -h2_from_L(L_est) * grad
            return cand_trial_from_vpre(L_est, a, v_pre, grad, x_b, val)

        def phase1(L_est, L_b, r_b, a, a_b, v_b, grad, x_b, val):
            val_n, cr, failed, fev, stats = cand_trial(
                L_est, r_b, a, a_b, v_b, grad, x_b, val
            )
            st = AdaLSState(
                L_est, val_n, cr, failed,
                jnp.asarray(1, dtype=jnp.int32), fev, stats,
            )

            def cond(st):
                return jnp.logical_and(st.fail, st.L * beta_inc < L_b)

            def body(st):
                L = st.L * beta_inc
                val_n, cr, failed, fev, stats = cand_trial(
                    L, r_b, a, a_b, v_b, grad, x_b, val
                )
                return AdaLSState(
                    L, val_n, cr, failed,
                    st.n_trial + 1, st.n_val + fev, st.stats + stats,
                )

            st = jax.lax.while_loop(cond, body, st)
            return st, st.fail

        def phase2(L_est, a, grad, x_b, val):
            # The original implementation increments L once before the first
            # trial after a rewrite.
            L_est = L_est * beta_inc
            val_n, cr, failed, fev, stats = cand_trial_rewrite(
                L_est, a, grad, x_b, val
            )
            st = AdaLSState(
                L_est, val_n, cr, failed,
                jnp.asarray(1, dtype=jnp.int32), fev, stats,
            )

            def cond(st):
                return st.fail

            def body(st):
                L = st.L * beta_inc
                val_n, cr, failed, fev, stats = cand_trial_rewrite(
                    L, a, grad, x_b, val
                )
                return AdaLSState(
                    L, val_n, cr, failed,
                    st.n_trial + 1, st.n_val + fev, st.stats + stats,
                )

            return jax.lax.while_loop(cond, body, st)

        def avg_update(t, t0, x_b, xbar_num, S, x_cut, S_cut):
            tf = t.astype(x_b.dtype)
            t0f = t0.astype(x_b.dtype)
            factor = jnp.exp(alpha * ((tf - 1) ** (6 / 7) - tf ** (6 / 7))) / tf ** (1 / 7)
            xbar_num = factor * ((tf - 1) ** (1 / 7) * xbar_num + alpha * x_b)
            S = factor * ((tf - 1) ** (1 / 7) * S + alpha)

            def cut_branch(args):
                xbar_num, S, x_cut, S_cut, t0 = args
                cut_factor = (t0f / tf) ** (1 / 7) * jnp.exp(
                    alpha * (t0f ** (6 / 7) - tf ** (6 / 7))
                )
                xbar_num = xbar_num - cut_factor * x_cut
                S = S - cut_factor * S_cut
                return xbar_num, S, xbar_num, S, t

            args = (xbar_num, S, x_cut, S_cut, t0)
            xbar_num, S, x_cut, S_cut, t0 = jax.lax.cond(
                t == 2 * t0,
                cut_branch,
                lambda z: z,
                args,
            )
            return t0, xbar_num, S, x_cut, S_cut, xbar_num / S

        @jax.jit
        def full_step(
            t,
            r_b,
            a_b,
            v_b,
            x,
            x0,
            val,
            val0,
            grad,
            L_est,
            L_b,
            t0,
            xbar_num,
            S,
            x_cut,
            S_cut,
        ):
            """One complete AdaVeloControl iteration on the device.

            This fuses E^- backtracking, optional rewrite, the M-test,
            E^+ testing, r in {1, R, 0} selection, and Algorithm 3's weighted
            average. Diagnostic evaluation of the weighted average is kept
            outside this timed kernel.
            """
            dtype = x.dtype
            tf = t.astype(dtype)
            a = jnp.expm1(alpha * (tf ** (6 / 7) - (tf - 1) ** (6 / 7)))

            ls1, rewrite_prev = phase1(L_est, L_b, r_b, a, a_b, v_b, grad, x, val)
            L1, vn1 = ls1.L, ls1.val_cand
            fev1, stats1 = ls1.n_val, ls1.stats

            def rewrite_branch(_):
                x_b = x0
                val_b = val0
                grad_b = grad_fn(x_b)
                ls2 = phase2(L1, a, grad_b, x_b, val_b)
                return (
                    ls2.L, ls2.val_cand, x_b, val_b, grad_b,
                    ls2.n_trial, ls2.n_val, ls2.stats,
                    jnp.asarray(1, dtype=jnp.int32),
                )

            def no_rewrite_branch(_):
                return (
                    L1, vn1, x, val, grad,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.zeros((4,), dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                )

            (L_cur, val_n, x_b, val_b, grad_b,
             n2, fev2, stats2, n_grad_rewrite) = jax.lax.cond(
                rewrite_prev, rewrite_branch, no_rewrite_branch, operand=None
            )

            # Reconstruct candidate geometry only after line search has found
            # its final L.  The loop itself carries only scalars and the three
            # candidate values/residuals, rather than (3,d) arrays.
            def rewritten_vpre(_):
                return -h2_from_L(L_cur) * grad_b

            def continued_vpre(_):
                return cand_vpre(L_cur, r_b, a, a_b, v_b, grad_b)

            v_pre = jax.lax.cond(
                rewrite_prev, rewritten_vpre, continued_vpre, operand=None
            )
            den = cand_den(a, dtype)

            # Ordinary/replay joint evaluation. Keep the
            # existing line search; use the joint value for M and cache it
            # with the gradient when r=1 is selected. Charge the extra value.
            v1, x1, dx1 = cand_point(v_pre, den[0], x_b)
            val1_joint, grad1 = value_grad_fn(x1)
            val_n = val_n.at[0].set(val1_joint)
            v1_norm = jnp.linalg.norm(dx1)
            M = m_estimate(val_n[0], grad1, grad_b, dx1, val_b)
            threshold = (6 / 7) * rho_max * alpha / (hmax * tf ** (1 / 7))
            accept1 = v1_norm * M <= threshold

            # r=0 is carried to the next outer iteration regardless of the
            # selected branch, so materialize it eagerly.
            v0, x0_cand, _ = cand_point(v_pre, den[2], x_b)

            if self.defer_R_check:
                def choose_r1(_):
                    return (
                        jnp.asarray(1.0, dtype=dtype),
                        v1, x1, val_n[0], grad1,
                        L_cur, L_cur,
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.zeros((4,), dtype=jnp.int32),
                    )

                def choose_not_r1(_):
                    # Only now evaluate the r=R certificate.  If E^- fails,
                    # r=0 is already certified, so choose r=0 directly rather
                    # than increasing L and repeating the line search.
                    vr, xr, dxr = cand_point(v_pre, den[1], x_b)
                    val_r = loss(xr)
                    cr_minus_r = eminus_res(val_r, grad_b, dxr, val_b, L_cur)
                    cert_r = cr_minus_r <= 0

                    def r_not_certified(_):
                        grad0 = grad_fn(x0_cand)
                        # The final pre-M trial was provisionally recorded as
                        # a pass of the required pair (r=1,r=0). Reclassify it
                        # as a deferred r=R certificate failure for diagnostics.
                        stats_delta = jnp.asarray([0, 1, 0, -1], dtype=jnp.int32)
                        return (
                            jnp.asarray(0.0, dtype=dtype),
                            v0, x0_cand, val_n[2], grad0,
                            L_cur * beta_dec, L_cur,
                            jnp.asarray(1, dtype=jnp.int32),
                            jnp.asarray(1, dtype=jnp.int32),
                            stats_delta,
                        )

                    def r_certified(_):
                        grad_r = grad_fn(xr)
                        cr_plus_r = eplus_res(val_r, grad_r, dxr, val_b, L_cur)
                        accept_r = cr_plus_r <= 0

                        def choose_r(_):
                            return (
                                jnp.asarray(R, dtype=dtype),
                                vr, xr, val_r, grad_r,
                                L_cur * beta_dec, L_cur,
                                jnp.asarray(1, dtype=jnp.int32),
                                jnp.asarray(1, dtype=jnp.int32),
                                jnp.zeros((4,), dtype=jnp.int32),
                            )

                        def choose_r0(_):
                            grad0 = grad_fn(x0_cand)
                            return (
                                jnp.asarray(0.0, dtype=dtype),
                                v0, x0_cand, val_n[2], grad0,
                                L_cur * beta_dec, L_cur,
                                jnp.asarray(2, dtype=jnp.int32),
                                jnp.asarray(1, dtype=jnp.int32),
                                jnp.zeros((4,), dtype=jnp.int32),
                            )

                        return jax.lax.cond(accept_r, choose_r, choose_r0, operand=None)

                    return jax.lax.cond(
                        cert_r, r_certified, r_not_certified, operand=None
                    )

                (r, v, x_new, val_new, grad_new, L_next, L_b_next,
                 n_grad_branch, n_val_branch, stats_delta) = (
                    jax.lax.cond(accept1, choose_r1, choose_not_r1, operand=None)
                )
            else:
                def choose_r1(_):
                    return (
                        jnp.asarray(1.0, dtype=dtype),
                        v1, x1, val_n[0], grad1,
                        L_cur, L_cur,
                        jnp.asarray(0, dtype=jnp.int32),
                    )

                def choose_not_r1(_):
                    # Materialize the r=R candidate only if r=1 is rejected.
                    vr, xr, dxr = cand_point(v_pre, den[1], x_b)
                    grad_r = grad_fn(xr)
                    crn = eplus_res(val_n[1], grad_r, dxr, val_b, L_cur)
                    accept_r = crn <= 0

                    def choose_r(_):
                        return (
                            jnp.asarray(R, dtype=dtype),
                            vr, xr, val_n[1], grad_r,
                            L_cur * beta_dec, L_cur,
                            jnp.asarray(1, dtype=jnp.int32),
                        )

                    def choose_r0(_):
                        grad0 = grad_fn(x0_cand)
                        return (
                            jnp.asarray(0.0, dtype=dtype),
                            v0, x0_cand, val_n[2], grad0,
                            L_cur * beta_dec, L_cur,
                            jnp.asarray(2, dtype=jnp.int32),
                        )

                    return jax.lax.cond(accept_r, choose_r, choose_r0, operand=None)

                r, v, x_new, val_new, grad_new, L_next, L_b_next, n_grad_branch = (
                    jax.lax.cond(accept1, choose_r1, choose_not_r1, operand=None)
                )
                n_val_branch = jnp.asarray(0, dtype=jnp.int32)
                stats_delta = jnp.zeros((4,), dtype=jnp.int32)

            x0_new = x0_cand
            val0_new = val_n[2]

            # Algorithm 3 weighted average. For t=1, the official average is
            # the finalized previous iterate x_b and the recurrence state is
            # intentionally left untouched, matching the existing code.
            def weighted_first(_):
                return t0, xbar_num, S, x_cut, S_cut, x_b

            def weighted_later(_):
                return avg_update(t, t0, x_b, xbar_num, S, x_cut, S_cut)

            t0_new, xbar_num_new, S_new, x_cut_new, S_cut_new, xbar = (
                jax.lax.cond(t == 1, weighted_first, weighted_later, operand=None)
            )

            n_val = fev1 + fev2 + n_val_branch + jnp.asarray(1, dtype=jnp.int32)
            ls_stats = stats1 + stats2 + stats_delta
            # Keep the original experiment's oracle-count convention:
            # the weighted-average diagnostic gradient is not included in the
            # plotted grad-call count. It is evaluated after timing the step.
            n_grad = (
                n_grad_rewrite
                + jnp.asarray(1, dtype=jnp.int32)  # grad at r=1 candidate
                + n_grad_branch
            )

            return AdaStep(
                r=r, a=a, v=v, x=x_new, x0=x0_new, val=val_new, val0=val0_new,
                grad=grad_new, L=L_next, L_prev=L_b_next, t0=t0_new,
                xbar_num=xbar_num_new, S=S_new, x_cut=x_cut_new, S_cut=S_cut_new,
                xbar=xbar, n_val=n_val, n_grad=n_grad,
                rewrite_prev=rewrite_prev, M=M, ls_stats=ls_stats,
            )

        self._ada_full_step = full_step

        # Kernels used only by AdaVeloControlCheckpoint.  In speculative
        # mode, r=1 is overwhelmingly likely after the threshold, so the
        # r=1 value and gradient are evaluated together.  This shares the
        # forward pass and avoids a host synchronization at every iteration.
        def fused_r1_step(
            t, a_b, v_b, x, x0, val, val0, grad, L_est, L_b,
            t0, xbar_num, S, x_cut, S_cut,
        ):
            dtype = x.dtype
            tf = t.astype(dtype)
            a = jnp.expm1(alpha * (tf ** (6 / 7) - (tf - 1) ** (6 / 7)))
            one = jnp.asarray(1.0, dtype=dtype)
            v_pre = cand_vpre(L_est, one, a, a_b, v_b, grad)
            den1 = one + a
            v1, x1, dx1 = cand_point(v_pre, den1, x)

            # Evaluate this candidate exactly once. The same value feeds both
            # certificates, and the accepted (value, gradient) pair below is
            # cached for the next step. Ordinary/replay steps keep separate
            # loss-then-gradient evaluation; no extra loss(x1) is needed here.
            val1, grad1 = value_grad_fn(x1)
            cr1 = eminus_res(val1, grad, dx1, val, L_est)
            M = m_estimate(val1, grad1, grad, dx1, val)
            threshold_m = (6 / 7) * rho_max * alpha / (hmax * tf ** (1 / 7))
            accept1 = jnp.logical_and(
                cr1 <= 0,
                jnp.linalg.norm(dx1) * M <= threshold_m,
            )

            def accepted(_):
                def weighted_first(_):
                    return t0, xbar_num, S, x_cut, S_cut, x

                def weighted_later(_):
                    return avg_update(t, t0, x, xbar_num, S, x_cut, S_cut)

                t0_new, xbar_num_new, S_new, x_cut_new, S_cut_new, xbar = (
                    jax.lax.cond(t == 1, weighted_first, weighted_later, operand=None)
                )
                # x0/val0 are intentionally not materialized between full
                # checkpoints.  If speculation later fails, the whole interval
                # is rolled back; the next full checkpoint reconstructs them.
                return AdaPredictStep(
                    success=jnp.asarray(True), a=a, v=v1, x=x1, x0=x0,
                    val=val1, val0=val0, grad=grad1, L=L_est, L_prev=L_b,
                    t0=t0_new, xbar_num=xbar_num_new, S=S_new,
                    x_cut=x_cut_new, S_cut=S_cut_new, xbar=xbar,
                    n_val=jnp.asarray(1, dtype=jnp.int32),
                    n_grad=jnp.asarray(1, dtype=jnp.int32), M=M,
                    ls_stats=jnp.zeros((4,), dtype=jnp.int32),
                )

            def rejected(_):
                return AdaPredictStep(
                    success=jnp.asarray(False), a=a_b, v=v_b, x=x, x0=x0,
                    val=val, val0=val0, grad=grad, L=L_est, L_prev=L_b,
                    t0=t0, xbar_num=xbar_num, S=S, x_cut=x_cut,
                    S_cut=S_cut, xbar=x,
                    n_val=jnp.asarray(1, dtype=jnp.int32),
                    n_grad=jnp.asarray(1, dtype=jnp.int32), M=M,
                    ls_stats=jnp.zeros((4,), dtype=jnp.int32),
                )

            return jax.lax.cond(accept1, accepted, rejected, operand=None)

        @jax.jit
        def r1_batch_step(
            n_steps, t_start, a_b, v_b, x, x0, val, val0, grad, L_est, L_b,
            t0, xbar_num, S, x_cut, S_cut,
        ):
            """Run up to n_steps speculative r=1 updates in one JIT call."""
            dtype = x.dtype
            zero_i = jnp.asarray(0, dtype=jnp.int32)
            true = jnp.asarray(True)
            zero = jnp.asarray(0.0, dtype=dtype)
            stats0 = jnp.zeros((4,), dtype=jnp.int32)

            # carry: attempted, completed, alive, algorithmic state, xbar, M, counts
            carry0 = (
                zero_i, zero_i, true, a_b, v_b, x, x0, val, val0, grad,
                L_est, L_b, t0, xbar_num, S, x_cut, S_cut, x, zero,
                zero_i, zero_i, stats0,
            )

            def cond(c):
                attempted, _, alive = c[:3]
                return jnp.logical_and(attempted < n_steps, alive)

            def body(c):
                (attempted, completed, alive, a_c, v_c, x_c, x0_c, val_c,
                 val0_c, grad_c, L_c, Lb_c, t0_c, xbar_num_c, S_c, x_cut_c,
                 S_cut_c, xbar_c, M_c, n_val_c, n_grad_c, stats_c) = c
                t = t_start + attempted + 1
                step = fused_r1_step(
                    t, a_c, v_c, x_c, x0_c, val_c, val0_c, grad_c, L_c, Lb_c,
                    t0_c, xbar_num_c, S_c, x_cut_c, S_cut_c,
                )
                attempted_new = attempted + 1
                completed_new = completed + step.success.astype(jnp.int32)
                return (
                    attempted_new, completed_new, step.success, step.a, step.v,
                    step.x, step.x0, step.val, step.val0, step.grad, step.L,
                    step.L_prev, step.t0, step.xbar_num, step.S, step.x_cut,
                    step.S_cut, step.xbar, step.M, n_val_c + step.n_val,
                    n_grad_c + step.n_grad, stats_c + step.ls_stats,
                )

            c = jax.lax.while_loop(cond, body, carry0)
            (attempted, completed, success, a_f, v_f, x_f, x0_f, val_f, val0_f,
             grad_f, L_f, Lb_f, t0_f, xbar_num_f, S_f, x_cut_f, S_cut_f,
             xbar_f, M_f, n_val_f, n_grad_f, stats_f) = c
            return AdaSpecBatch(
                success=success, attempted=attempted, completed=completed,
                a=a_f, v=v_f, x=x_f, x0=x0_f, val=val_f, val0=val0_f,
                grad=grad_f, L=L_f, L_prev=Lb_f, t0=t0_f,
                xbar_num=xbar_num_f, S=S_f, x_cut=x_cut_f, S_cut=S_cut_f,
                xbar=xbar_f, n_val=n_val_f, n_grad=n_grad_f, M=M_f,
                ls_stats=stats_f,
            )

        @jax.jit
        def checkpoint_step(
            t, r_b, a_b, v_b, x, x0, val, val0, grad, L_est, L_b,
            t0, xbar_num, S, x_cut, S_cut,
        ):
            """One predictive checkpoint; reuse the joint r=1 value/gradient.

            Only the remaining required candidates receive separate value
            evaluations. Cache val1/grad1 on acceptance just as in prediction.
            """
            dtype = x.dtype
            tf = t.astype(dtype)
            a = jnp.expm1(alpha * (tf ** (6 / 7) - (tf - 1) ** (6 / 7)))
            v_pre = cand_vpre(L_est, r_b, a, a_b, v_b, grad)
            den = cand_den(a, dtype)
            v1, x1, dx1 = cand_point(v_pre, den[0], x)
            val1, grad1 = value_grad_fn(x1)
            cr1 = eminus_res(val1, grad, dx1, val, L_est)

            if self.defer_R_check:
                # Combined checkpoint + defer-R variant: a speculative
                # checkpoint needs only the certificates that can support the
                # r=1 decision.  Check r=1 and r=0, then apply the M-test.
                # If M rejects r=1, the checkpoint fails and replay uses the
                # ordinary defer-R step, where r=R is evaluated only then.
                nan = jnp.asarray(jnp.nan, dtype=dtype)

                def fail_r1(_):
                    return (
                        jnp.asarray([val1, nan, nan], dtype=dtype),
                        jnp.asarray(True),
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray([1, 0, 0, 0], dtype=jnp.int32),
                    )

                def pass_r1(_):
                    _, x0_c, dx0 = cand_point(v_pre, den[2], x)
                    val0_c = loss(x0_c)
                    cr0 = eminus_res(val0_c, grad, dx0, val, L_est)
                    stats = jax.lax.cond(
                        cr0 > 0,
                        lambda _: jnp.asarray([0, 0, 1, 0], dtype=jnp.int32),
                        lambda _: jnp.asarray([0, 0, 0, 1], dtype=jnp.int32),
                        operand=None,
                    )
                    return (
                        jnp.asarray([val1, nan, val0_c], dtype=dtype),
                        cr0 > 0,
                        jnp.asarray(2, dtype=jnp.int32),
                        stats,
                    )

                val_n, failed, n_val, stats = jax.lax.cond(
                    cr1 > 0, fail_r1, pass_r1, operand=None
                )
            elif self.line_search_mode == "parallel":
                def other_value(den_i):
                    _, x_i, _ = cand_point(v_pre, den_i, x)
                    return loss(x_i)

                val_other = jax.vmap(other_value)(den[1:])
                val_n = jnp.concatenate((jnp.reshape(val1, (1,)), val_other))

                def residual(den_i, val_i):
                    _, _, dx_i = cand_point(v_pre, den_i, x)
                    return eminus_res(val_i, grad, dx_i, val, L_est)

                cr_other = jax.vmap(residual)(den[1:], val_other)
                cr = jnp.concatenate((jnp.reshape(cr1, (1,)), cr_other))
                failed = jnp.any(cr > 0)
                stats = first_fail_stats(cr)
                n_val = jnp.asarray(3, dtype=jnp.int32)
            else:
                nan = jnp.asarray(jnp.nan, dtype=dtype)

                def fail_r1(_):
                    return (
                        jnp.asarray([val1, nan, nan], dtype=dtype),
                        jnp.asarray(True),
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray([1, 0, 0, 0], dtype=jnp.int32),
                    )

                def pass_r1(_):
                    _, xR, dxR = cand_point(v_pre, den[1], x)
                    valR = loss(xR)
                    crR = eminus_res(valR, grad, dxR, val, L_est)

                    def fail_R(_):
                        return (
                            jnp.asarray([val1, valR, nan], dtype=dtype),
                            jnp.asarray(True),
                            jnp.asarray(2, dtype=jnp.int32),
                            jnp.asarray([0, 1, 0, 0], dtype=jnp.int32),
                        )

                    def pass_R(_):
                        _, x0_c, dx0 = cand_point(v_pre, den[2], x)
                        val0_c = loss(x0_c)
                        cr0 = eminus_res(val0_c, grad, dx0, val, L_est)
                        stats = jax.lax.cond(
                            cr0 > 0,
                            lambda _: jnp.asarray([0, 0, 1, 0], dtype=jnp.int32),
                            lambda _: jnp.asarray([0, 0, 0, 1], dtype=jnp.int32),
                            operand=None,
                        )
                        return (
                            jnp.asarray([val1, valR, val0_c], dtype=dtype),
                            cr0 > 0,
                            jnp.asarray(3, dtype=jnp.int32),
                            stats,
                        )

                    return jax.lax.cond(crR > 0, fail_R, pass_R, operand=None)

                val_n, failed, n_val, stats = jax.lax.cond(
                    cr1 > 0, fail_r1, pass_r1, operand=None
                )

            M = m_estimate(val1, grad1, grad, dx1, val)
            threshold_m = (6 / 7) * rho_max * alpha / (hmax * tf ** (1 / 7))
            accept1 = jnp.logical_and(
                jnp.logical_not(failed),
                jnp.linalg.norm(dx1) * M <= threshold_m,
            )

            def accepted(_):
                _, x0_cand, _ = cand_point(v_pre, den[2], x)

                def weighted_first(_):
                    return t0, xbar_num, S, x_cut, S_cut, x

                def weighted_later(_):
                    return avg_update(t, t0, x, xbar_num, S, x_cut, S_cut)

                t0_new, xbar_num_new, S_new, x_cut_new, S_cut_new, xbar = (
                    jax.lax.cond(t == 1, weighted_first, weighted_later, operand=None)
                )
                return AdaPredictStep(
                    success=jnp.asarray(True), a=a, v=v1, x=x1, x0=x0_cand,
                    val=val1, val0=val_n[2], grad=grad1, L=L_est, L_prev=L_est,
                    t0=t0_new, xbar_num=xbar_num_new, S=S_new,
                    x_cut=x_cut_new, S_cut=S_cut_new, xbar=xbar, n_val=n_val,
                    n_grad=jnp.asarray(1, dtype=jnp.int32), M=M, ls_stats=stats,
                )

            def rejected(_):
                return AdaPredictStep(
                    success=jnp.asarray(False), a=a_b, v=v_b, x=x, x0=x0,
                    val=val, val0=val0, grad=grad, L=L_est, L_prev=L_b,
                    t0=t0, xbar_num=xbar_num, S=S, x_cut=x_cut, S_cut=S_cut,
                    xbar=x, n_val=n_val, n_grad=jnp.asarray(1, dtype=jnp.int32),
                    M=M, ls_stats=stats,
                )

            return jax.lax.cond(accept1, accepted, rejected, operand=None)

        self._ada_r1_batch_step = r1_batch_step
        self._ada_checkpoint_step = checkpoint_step

    def _warmup_kernels(self, x, val, grad):
        dtype = x.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        out = self._ada_full_step(
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=dtype),
            zero,
            jnp.zeros_like(x),
            x,
            x,
            val,
            val,
            grad,
            jnp.asarray(self.L_est, dtype=dtype),
            jnp.asarray(self.L_est, dtype=dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.zeros_like(x),
            zero,
            jnp.zeros_like(x),
            zero,
        )
        self._block_until_ready(*out)

    def optimize(self):
        dtype = self.model.param.dtype
        r = jnp.asarray(1.0, dtype=dtype)
        a = jnp.asarray(0.0, dtype=dtype)
        L_est = jnp.asarray(self.L_est, dtype=dtype)
        L_b = L_est
        x = self.model.param
        x0 = x
        v = jnp.zeros_like(x)
        sol = x

        tic = time.perf_counter()
        val, grad = self.oracle.value_and_grad(x)
        val0 = val
        self._block_until_ready(val, grad)
        self.elapsed_time += time.perf_counter() - tic

        # Algorithm 3 state for the official weighted-average output.
        t0 = jnp.asarray(1, dtype=jnp.int32)
        xbar_num = jnp.zeros_like(x)
        x_cut = jnp.zeros_like(x)
        S = jnp.asarray(0.0, dtype=dtype)
        S_cut = jnp.asarray(0.0, dtype=dtype)

        t = 0
        grad_norm_bar = jnp.linalg.norm(grad)
        while self.stop_condition():
            t += 1

            tic = time.perf_counter()
            out = self._ada_full_step(
                jnp.asarray(t, dtype=jnp.int32),
                r,
                a,
                v,
                x,
                x0,
                val,
                val0,
                grad,
                L_est,
                L_b,
                t0,
                xbar_num,
                S,
                x_cut,
                S_cut,
            )
            step = out
            r, a, v = step.r, step.a, step.v
            x, x0, val, val0, grad = step.x, step.x0, step.val, step.val0, step.grad
            L_est, L_b = step.L, step.L_prev
            t0, xbar_num, S = step.t0, step.xbar_num, step.S
            x_cut, S_cut, xbar = step.x_cut, step.S_cut, step.xbar
            n_val, n_grad = step.n_val, step.n_grad
            rewrite_prev, ls_stats = step.rewrite_prev, step.ls_stats
            # Synchronize the optimizer state before stopping the timer.
            # Diagnostic evaluations of xbar are intentionally excluded.
            self._block_until_ready(x, x0, val, grad, L_est, xbar)
            self.elapsed_time += time.perf_counter() - tic

            # The kernel bypasses Oracle's Python methods, so reproduce the
            # exact algorithmic oracle accounting on the host after completion.
            n_val_host = int(jax.device_get(n_val))
            n_grad_host = int(jax.device_get(n_grad))
            self.oracle.add_count(val=n_val_host, grad=n_grad_host, joint=1)
            self.line_search_stats += np.asarray(
                jax.device_get(ls_stats), dtype=np.int64
            )
            if self.defer_R_check:
                self.deferred_R_checks += int(float(jax.device_get(r)) < 1.0 - 1e-12)

            # If the previous iterate was rewritten as r_{t-1}=0, repair the
            # already-recorded diagnostic history. This happens after the only
            # required kernel synchronization, so it introduces no mid-step wait.
            rewrite_host = bool(jax.device_get(rewrite_prev))
            if rewrite_host and self.r_hist and self._should_save_hist(t - 1):
                self.r_hist[-1] = 0.0

            # Diagnostic evaluations and official-output tracking are excluded
            # from both oracle counts and the optimizer wall-clock timer.
            val_bar, grad_bar = self.oracle.value_and_grad(xbar, count=False)
            grad_norm_bar = jnp.linalg.norm(grad_bar)
            self._block_until_ready(val_bar, grad_norm_bar)
            if grad_norm_bar < self.grad_best:
                self.grad_best = grad_norm_bar
                sol = xbar

            if self._should_save_hist(t):
                self.r_hist.append(float(jax.device_get(r)))
                self._save_call_hist()
                self.grad_hist.append(grad_norm_bar)
                self.val_hist.append(val_bar)
                self.time_hist.append(self.elapsed_time)
                if hasattr(self.model, "accuracy") and (t % self.acc_every == 0):
                    acc = self.model.accuracy(xbar)
                    self._block_until_ready(acc)
                    self.acc_hist.append(float(acc))
                    print("Accuracy:", self.acc_hist[-1])
                else:
                    self.acc_hist.append(np.nan)

        self._block_until_ready(sol, self.grad_best)
        self.L_est = float(jax.device_get(L_est))
        self.model.param = sol

    def line_search_summary(self):
        """Return aggregate statistics for the E^- line search."""
        fail1, failR, fail0, passed = map(int, self.line_search_stats)
        trials = fail1 + failR + fail0 + passed
        if self.defer_R_check:
            # Initial E^- trials inspect only r=1 and r=0. Deferred r=R
            # checks occur only after the M-test rejects r=1.
            if self.line_search_mode == "short_circuit":
                pre_m_fev = fail1 + 2 * (trials - fail1)
            else:
                pre_m_fev = 2 * trials
            actual_fev = pre_m_fev + self.deferred_R_checks
            parallel_equivalent = 3 * trials
        else:
            actual_fev = fail1 + 2 * failR + 3 * fail0 + 3 * passed
            parallel_equivalent = 3 * trials
            if self.line_search_mode == "parallel":
                actual_fev = parallel_equivalent
        out = {
            "mode": self.line_search_mode,
            "fail_r1": fail1,
            "fail_rR": failR,
            "fail_r0": fail0,
            "pass": passed,
            "trials": trials,
            "val_evaluations": actual_fev,
            "parallel_equivalent_val_evaluations": parallel_equivalent,
            "average_vals_per_trial": actual_fev / trials if trials else 0.0,
        }
        if self.defer_R_check:
            out.update({
                "deferred_R_checks": int(self.deferred_R_checks),
                "deferred_R_failures": failR,
            })
        return out

    def run(self):
        self.warmup()
        self.optimize()
        self.grad_call_hist = np.array(self.grad_call_hist)
        self.val_call_hist = np.array(self.val_call_hist)
        self.grad_val_call_hist = np.array(self.grad_val_call_hist)
        self.joint_call_hist = np.array(self.joint_call_hist)
        self.oracle_call_hist = np.array(self.oracle_call_hist)
        self.grad_hist = np.array(self.grad_hist)
        self.val_hist = np.array(self.val_hist)
        self.time_hist = np.array(self.time_hist)
        self.r_hist = np.array(self.r_hist)
        self.acc_hist = np.array(self.acc_hist)
        if self.save_hist:
            mode_suffix = '' if self.line_search_mode == 'parallel' else '_short'
            self._savez(
                'R' + str(self.R) + 'a' + str(self.alpha) + self.result_tag + mode_suffix,
                grad_call=self.grad_call_hist,
                val_call=self.val_call_hist,
                grad_val_call=self.grad_val_call_hist,
                grad=self.grad_hist,
                val=self.val_hist,
                time=self.time_hist,
                r=self.r_hist,
                acc=self.acc_hist,
                line_search_stats=self.line_search_stats,
                line_search_mode=np.asarray(self.line_search_mode),
            )

class AdaVeloControlDeferredR(AdaVeloControl):
    """VC variant that defers the r=R E^- certificate until after the M-test.

    Each line-search trial first certifies r=1 and r=0.  If the M-test accepts
    r=1, the r=R candidate is never evaluated.  If the M-test rejects r=1,
    r=R is evaluated and certified at the final L; if that certificate fails,
    the already-certified r=0 candidate is selected directly.
    """

    def __init__(self, model, R=0, alpha=0.1, hmax=1, line_search_mode="parallel"):
        super().__init__(
            model, R=R, alpha=alpha, hmax=hmax,
            line_search_mode=line_search_mode, defer_R_check=True,
        )
        self.result_tag = "_adavelocontrol_deferR"


class AdaVeloControlCheckpoint(AdaVeloControl):
    """VC with batched speculative r=1 runs and square-spaced checkpoints.

    Ordinary VC is used until ``checkpoint_threshold`` consecutive r=1
    updates have been observed.  Thereafter, non-checkpoint r=1 updates are
    run speculatively in device-side batches.  In that mode, f(x^1) and
    grad f(x^1) are obtained by one value_and_grad call, and the unused r=R
    and r=0 candidates are skipped.  Full checks are made at streak positions
    T + j^2.  If speculation fails, the method restores the latest full
    checkpoint and replays the interval with ordinary VC.

    Joint value/gradient evaluation is also used at predictive checkpoints.
    The returned r=1 value is reused for E^-, the M-test, and the next state;
    it is not reevaluated. This policy is shared by the defer-R subclass and
    requires no additional command-line option. Initialization and weighted
    output diagnostics retain the BaseOptimizer/VC evaluation conventions.

    Wasted speculative work remains in wall-clock time and oracle counts.
    Only the algorithmic state and diagnostic history are rolled back.
    """

    def __init__(
        self,
        model,
        R=0,
        alpha=0.1,
        hmax=1,
        line_search_mode="parallel",
        checkpoint_threshold=10,
        defer_R_check=False,
    ):
        if int(checkpoint_threshold) < 1:
            raise ValueError("checkpoint_threshold must be a positive integer.")
        self.checkpoint_threshold = int(checkpoint_threshold)
        # A large square gap is split only to keep time-budget overshoot and
        # one device call bounded.  This does not create an algorithmic
        # checkpoint; it only creates a host synchronization/history endpoint.
        self.speculative_batch_max = 128
        self.speculative_attempts = 0
        self.speculative_batches = 0
        self.max_speculative_batch = 0
        self.checkpoint_checks = 0
        self.rollback_count = 0
        self.redo_iterations = 0
        self.max_rollback = 0
        super().__init__(
            model,
            R=R,
            alpha=alpha,
            hmax=hmax,
            line_search_mode=line_search_mode,
            defer_R_check=defer_R_check,
        )
        self.result_tag = "_adavelocontrol_cpT" + str(self.checkpoint_threshold)

    def _warmup_kernels(self, x, val, grad):
        super()._warmup_kernels(x, val, grad)
        dtype = x.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        out_batch = self._ada_r1_batch_step(
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            zero,
            jnp.zeros_like(x),
            x,
            x,
            val,
            val,
            grad,
            jnp.asarray(self.L_est, dtype=dtype),
            jnp.asarray(self.L_est, dtype=dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.zeros_like(x),
            zero,
            jnp.zeros_like(x),
            zero,
        )
        out_checkpoint = self._ada_checkpoint_step(
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=dtype),
            zero,
            jnp.zeros_like(x),
            x,
            x,
            val,
            val,
            grad,
            jnp.asarray(self.L_est, dtype=dtype),
            jnp.asarray(self.L_est, dtype=dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.zeros_like(x),
            zero,
            jnp.zeros_like(x),
            zero,
        )
        self._block_until_ready(*out_batch, *out_checkpoint)

    @staticmethod
    def _next_square_checkpoint(streak, threshold):
        """Smallest threshold + j^2 strictly larger than streak."""
        offset = max(int(streak) - int(threshold), 0)
        j = int(np.floor(np.sqrt(offset))) + 1
        nxt = int(threshold) + j * j
        while nxt <= streak:
            j += 1
            nxt = int(threshold) + j * j
        return nxt

    def optimize(self):
        dtype = self.model.param.dtype
        r = jnp.asarray(1.0, dtype=dtype)
        a = jnp.asarray(0.0, dtype=dtype)
        L_est = jnp.asarray(self.L_est, dtype=dtype)
        L_b = L_est
        x = self.model.param
        x0 = x
        v = jnp.zeros_like(x)
        sol = x

        tic = time.perf_counter()
        val, grad = self.oracle.value_and_grad(x)
        val0 = val
        self._block_until_ready(val, grad)
        self.elapsed_time += time.perf_counter() - tic

        t0 = jnp.asarray(1, dtype=jnp.int32)
        xbar_num = jnp.zeros_like(x)
        x_cut = jnp.zeros_like(x)
        S = jnp.asarray(0.0, dtype=dtype)
        S_cut = jnp.asarray(0.0, dtype=dtype)

        t = 0
        r1_streak = 0
        speculative = False
        checkpoint = None
        next_checkpoint_streak = None
        replay_until = None
        grad_norm_bar = jnp.linalg.norm(grad)

        hist_attrs = (
            "r_hist", "grad_call_hist", "val_call_hist", "grad_val_call_hist",
            "joint_call_hist", "oracle_call_hist",
            "grad_hist", "val_hist", "time_hist", "acc_hist",
        )

        def save_checkpoint():
            return {
                "t": t,
                "r": r,
                "a": a,
                "L_est": L_est,
                "L_b": L_b,
                "x": x,
                "x0": x0,
                "v": v,
                "val": val,
                "val0": val0,
                "grad": grad,
                "t0": t0,
                "xbar_num": xbar_num,
                "S": S,
                "x_cut": x_cut,
                "S_cut": S_cut,
                "grad_best": self.grad_best,
                "sol": sol,
                "r1_streak": r1_streak,
                "hist_lengths": {name: len(getattr(self, name)) for name in hist_attrs},
            }

        def restore_checkpoint(cp):
            nonlocal t, r, a, L_est, L_b, x, x0, v, val, val0, grad
            nonlocal t0, xbar_num, S, x_cut, S_cut, sol, r1_streak
            t = cp["t"]
            r = cp["r"]
            a = cp["a"]
            L_est = cp["L_est"]
            L_b = cp["L_b"]
            x = cp["x"]
            x0 = cp["x0"]
            v = cp["v"]
            val = cp["val"]
            val0 = cp["val0"]
            grad = cp["grad"]
            t0 = cp["t0"]
            xbar_num = cp["xbar_num"]
            S = cp["S"]
            x_cut = cp["x_cut"]
            S_cut = cp["S_cut"]
            self.grad_best = cp["grad_best"]
            sol = cp["sol"]
            r1_streak = cp["r1_streak"]
            for name, n in cp["hist_lengths"].items():
                del getattr(self, name)[n:]

        def record_committed_endpoint(xbar):
            """Record one ordinary step or one speculative batch endpoint."""
            nonlocal grad_norm_bar, sol
            val_bar, grad_bar = self.oracle.value_and_grad(xbar, count=False)
            grad_norm_bar = jnp.linalg.norm(grad_bar)
            self._block_until_ready(val_bar, grad_norm_bar)
            if grad_norm_bar < self.grad_best:
                self.grad_best = grad_norm_bar
                sol = xbar

            if self._should_save_hist(t):
                self.r_hist.append(float(jax.device_get(r)))
                self._save_call_hist()
                self.grad_hist.append(grad_norm_bar)
                self.val_hist.append(val_bar)
                self.time_hist.append(self.elapsed_time)
                if hasattr(self.model, "accuracy") and (t % self.acc_every == 0):
                    acc = self.model.accuracy(xbar)
                    self._block_until_ready(acc)
                    self.acc_hist.append(float(acc))
                    print("Accuracy:", self.acc_hist[-1])
                else:
                    self.acc_hist.append(np.nan)

        def rollback_after_failure(failed_t):
            nonlocal speculative, replay_until, next_checkpoint_streak
            rollback_len = failed_t - checkpoint["t"]
            self.rollback_count += 1
            self.redo_iterations += rollback_len
            self.max_rollback = max(self.max_rollback, rollback_len)
            restore_checkpoint(checkpoint)
            speculative = False
            replay_until = failed_t
            next_checkpoint_streak = None

        while self.stop_condition():
            next_t = t + 1

            if speculative:
                gap = int(next_checkpoint_streak - r1_streak)

                # The next update is a full square-spaced checkpoint.
                if gap <= 1:
                    self.checkpoint_checks += 1
                    tic = time.perf_counter()
                    step = self._ada_checkpoint_step(
                        jnp.asarray(next_t, dtype=jnp.int32), r, a, v, x, x0,
                        val, val0, grad, L_est, L_b, t0, xbar_num, S, x_cut, S_cut,
                    )
                    self._block_until_ready(
                        step.x, step.val, step.grad, step.L, step.xbar, step.success
                    )
                    self.elapsed_time += time.perf_counter() - tic

                    n_val_host = int(jax.device_get(step.n_val))
                    n_grad_host = int(jax.device_get(step.n_grad))
                    self.oracle.add_count(val=n_val_host, grad=n_grad_host, joint=1)
                    self.line_search_stats += np.asarray(
                        jax.device_get(step.ls_stats), dtype=np.int64
                    )

                    if not bool(jax.device_get(step.success)):
                        rollback_after_failure(next_t)
                        continue

                    t = next_t
                    r = jnp.asarray(1.0, dtype=dtype)
                    a, v = step.a, step.v
                    x, x0, val, val0, grad = (
                        step.x, step.x0, step.val, step.val0, step.grad
                    )
                    L_est, L_b = step.L, step.L_prev
                    t0, xbar_num, S = step.t0, step.xbar_num, step.S
                    x_cut, S_cut, xbar = step.x_cut, step.S_cut, step.xbar
                    r1_streak += 1
                    record_committed_endpoint(xbar)

                    checkpoint = save_checkpoint()
                    next_checkpoint_streak = self._next_square_checkpoint(
                        r1_streak, self.checkpoint_threshold
                    )
                    continue

                # Batch every non-checkpoint speculative update until just
                # before the next full checkpoint.  This is the main speedup:
                # one device synchronization for many r=1 iterations.
                n_batch = min(gap - 1, self.speculative_batch_max)
                if self.stop_mode == "iter":
                    # stop_condition uses <=, so at most this many additional
                    # gradient calls can be made before the next host check.
                    remaining = int(self.n_iter - self.oracle.count["grad"] + 1)
                    n_batch = min(n_batch, max(remaining, 1))

                tic = time.perf_counter()
                batch = self._ada_r1_batch_step(
                    jnp.asarray(n_batch, dtype=jnp.int32),
                    jnp.asarray(t, dtype=jnp.int32),
                    a, v, x, x0, val, val0, grad, L_est, L_b,
                    t0, xbar_num, S, x_cut, S_cut,
                )
                self._block_until_ready(
                    batch.x, batch.val, batch.grad, batch.L, batch.xbar,
                    batch.success, batch.attempted,
                )
                self.elapsed_time += time.perf_counter() - tic

                attempted = int(jax.device_get(batch.attempted))
                completed = int(jax.device_get(batch.completed))
                n_val_host = int(jax.device_get(batch.n_val))
                n_grad_host = int(jax.device_get(batch.n_grad))
                self.speculative_attempts += attempted
                self.speculative_batches += 1
                self.max_speculative_batch = max(self.max_speculative_batch, attempted)
                self.oracle.add_count(val=n_val_host, grad=n_grad_host, joint=attempted)
                self.line_search_stats += np.asarray(
                    jax.device_get(batch.ls_stats), dtype=np.int64
                )

                if not bool(jax.device_get(batch.success)):
                    failed_t = t + attempted
                    rollback_after_failure(failed_t)
                    continue

                t += completed
                r = jnp.asarray(1.0, dtype=dtype)
                a, v = batch.a, batch.v
                x, x0, val, val0, grad = (
                    batch.x, batch.x0, batch.val, batch.val0, batch.grad
                )
                L_est, L_b = batch.L, batch.L_prev
                t0, xbar_num, S = batch.t0, batch.xbar_num, batch.S
                x_cut, S_cut, xbar = batch.x_cut, batch.S_cut, batch.xbar
                r1_streak += completed
                # Diagnostics/history are intentionally sampled once per JIT
                # batch; evaluating every speculative xbar would erase much of
                # the wall-clock benefit of batching.
                record_committed_endpoint(xbar)
                continue

            # Ordinary VC mode, used before the threshold and during replay.
            tic = time.perf_counter()
            step = self._ada_full_step(
                jnp.asarray(next_t, dtype=jnp.int32), r, a, v, x, x0,
                val, val0, grad, L_est, L_b, t0, xbar_num, S, x_cut, S_cut,
            )
            self._block_until_ready(step.x, step.x0, step.val, step.grad, step.L, step.xbar)
            self.elapsed_time += time.perf_counter() - tic

            n_val_host = int(jax.device_get(step.n_val))
            n_grad_host = int(jax.device_get(step.n_grad))
            self.oracle.add_count(val=n_val_host, grad=n_grad_host, joint=1)
            self.line_search_stats += np.asarray(jax.device_get(step.ls_stats), dtype=np.int64)
            if self.defer_R_check:
                self.deferred_R_checks += int(float(jax.device_get(step.r)) < 1.0 - 1e-12)

            rewrite_host = bool(jax.device_get(step.rewrite_prev))
            if rewrite_host:
                r1_streak = 0
                if self.r_hist and self._should_save_hist(t):
                    self.r_hist[-1] = 0.0

            t = next_t
            r, a, v = step.r, step.a, step.v
            x, x0, val, val0, grad = step.x, step.x0, step.val, step.val0, step.grad
            L_est, L_b = step.L, step.L_prev
            t0, xbar_num, S = step.t0, step.xbar_num, step.S
            x_cut, S_cut, xbar = step.x_cut, step.S_cut, step.xbar

            r_host = float(jax.device_get(r))
            if abs(r_host - 1.0) <= 1e-12:
                r1_streak += 1
            else:
                r1_streak = 0

            record_committed_endpoint(xbar)

            if replay_until is not None:
                if r1_streak == 0:
                    replay_until = None
                elif t >= replay_until:
                    replay_until = None
                    if r1_streak >= self.checkpoint_threshold:
                        checkpoint = save_checkpoint()
                        speculative = True
                        next_checkpoint_streak = self._next_square_checkpoint(
                            r1_streak, self.checkpoint_threshold
                        )
                continue

            if r1_streak >= self.checkpoint_threshold:
                checkpoint = save_checkpoint()
                speculative = True
                next_checkpoint_streak = self._next_square_checkpoint(
                    r1_streak, self.checkpoint_threshold
                )

        self._block_until_ready(sol, self.grad_best)
        self.L_est = float(jax.device_get(L_est))
        self.model.param = sol

    def checkpoint_summary(self):
        out = {
            "checkpoint_threshold": int(self.checkpoint_threshold),
            "speculative_attempts": int(self.speculative_attempts),
            "speculative_batches": int(self.speculative_batches),
            "max_speculative_batch": int(self.max_speculative_batch),
            "checkpoint_checks": int(self.checkpoint_checks),
            "rollbacks": int(self.rollback_count),
            "redo_iterations": int(self.redo_iterations),
            "max_rollback": int(self.max_rollback),
        }
        if self.defer_R_check:
            out.update({
                "deferred_R_checks": int(self.deferred_R_checks),
                "deferred_R_failures": int(self.line_search_stats[1]),
            })
        return out

    def run(self):
        self.warmup()
        self.optimize()
        self.grad_call_hist = np.array(self.grad_call_hist)
        self.val_call_hist = np.array(self.val_call_hist)
        self.grad_val_call_hist = np.array(self.grad_val_call_hist)
        self.joint_call_hist = np.array(self.joint_call_hist)
        self.oracle_call_hist = np.array(self.oracle_call_hist)
        self.grad_hist = np.array(self.grad_hist)
        self.val_hist = np.array(self.val_hist)
        self.time_hist = np.array(self.time_hist)
        self.r_hist = np.array(self.r_hist)
        self.acc_hist = np.array(self.acc_hist)
        if self.save_hist:
            mode_suffix = "" if self.line_search_mode == "parallel" else "_short"
            self._savez(
                "R" + str(self.R) + "a" + str(self.alpha)
                + self.result_tag + mode_suffix,
                grad_call=self.grad_call_hist,
                val_call=self.val_call_hist,
                grad_val_call=self.grad_val_call_hist,
                grad=self.grad_hist,
                val=self.val_hist,
                time=self.time_hist,
                r=self.r_hist,
                acc=self.acc_hist,
                line_search_stats=self.line_search_stats,
                line_search_mode=np.asarray(self.line_search_mode),
                checkpoint_threshold=np.asarray(self.checkpoint_threshold),
                speculative_attempts=np.asarray(self.speculative_attempts),
                speculative_batches=np.asarray(self.speculative_batches),
                max_speculative_batch=np.asarray(self.max_speculative_batch),
                checkpoint_checks=np.asarray(self.checkpoint_checks),
                rollback_count=np.asarray(self.rollback_count),
                redo_iterations=np.asarray(self.redo_iterations),
                max_rollback=np.asarray(self.max_rollback),
            )


class AdaVeloControlDeferredRCheckpoint(AdaVeloControlCheckpoint):
    """Checkpoint VC whose ordinary/replay steps use deferred r=R checks.

    Before the checkpoint threshold and during rollback replay, each line-search
    trial certifies r=1 and r=0 first; r=R is evaluated only if the M-test
    rejects r=1.  In speculative mode, the batched r=1 kernel is identical to
    AdaVeloControlCheckpoint.  At square-spaced checkpoints, only r=1 and r=0
    are certified before the M-test; if M rejects r=1, speculation fails and
    replay performs the deferred r=R check.
    """

    def __init__(
        self,
        model,
        R=0,
        alpha=0.1,
        hmax=1,
        line_search_mode="parallel",
        checkpoint_threshold=10,
    ):
        super().__init__(
            model, R=R, alpha=alpha, hmax=hmax,
            line_search_mode=line_search_mode,
            checkpoint_threshold=checkpoint_threshold,
            defer_R_check=True,
        )
        self.result_tag = (
            "_adavelocontrol_deferR_cpT" + str(self.checkpoint_threshold)
        )


class AdaRestartHB(BaseOptimizer):
    def __init__(self, model):
        super().__init__(model)
        # Marumo--Takeda restarted HB parameters.
        self.L_est = 1e-3
        self.beta_inc = 2
        self.beta_dec = 0.1

        loss = model.loss
        val_grad = jax.value_and_grad(loss)
        beta_inc = self.beta_inc
        beta_dec = self.beta_dec

        @jax.jit
        def hb_step(k, h, x, xbar, v, val, grad, val_bar, grad_bar,
                    L, S, val_best, x_best, grad_best):
            """One Marumo--Takeda restarted-HB update on the device."""
            k_new = k + 1
            kf = k_new.astype(x.dtype)

            # x_k = x_{k-1} + v_k,  v_k = v_{k-1} - grad f(x_{k-1}) / L.
            v_new = v - grad / L
            x_cand = x + v_new
            v_norm = jnp.linalg.norm(v_new)
            S_new = S + v_norm**2
            val_cand, grad_cand = val_grad(x_cand)

            better_cand = val_cand < val_best
            val_best_cand = jnp.minimum(val_best, val_cand)
            x_best_cand = jax.lax.select(better_cand, x_cand, x_best)
            grad_best_cand = jax.lax.select(better_cand, grad_cand, grad_best)

            smooth_fail = (
                val_cand - val
                > jnp.vdot(grad, v_new) + 0.5 * L * v_norm**2
            )

            def init_epoch(best_x, best_val, best_grad, L_new, n_val, n_grad):
                zero = jnp.asarray(0.0, dtype=x.dtype)
                grad_norm = jnp.linalg.norm(best_grad)
                return HBStep(
                    k=jnp.asarray(0, dtype=jnp.int32),
                    h=zero,
                    x=best_x,
                    xbar=best_x,
                    v=jnp.zeros_like(x),
                    val=best_val,
                    grad=best_grad,
                    val_bar=best_val,
                    grad_bar=best_grad,
                    L=L_new,
                    S=zero,
                    val_best=best_val,
                    x_best=best_x,
                    grad_best=best_grad,
                    val_out=best_val,
                    grad_norm_out=grad_norm,
                    n_val=n_val,
                    n_grad=n_grad,
                )

            def smooth_restart_branch(_):
                # Marumo's implementation reuses the stored (f, grad f) of x_best.
                return init_epoch(
                    x_best_cand,
                    val_best_cand,
                    grad_best_cand,
                    L * beta_inc,
                    jnp.asarray(1, dtype=jnp.int32),
                    jnp.asarray(1, dtype=jnp.int32),
                )

            def smooth_ok_branch(_):
                norm_grad_bar = jnp.linalg.norm(grad_bar)
                h1 = 3 * (
                    val_cand - val - 0.5 * jnp.vdot(grad_cand + grad, v_new)
                ) / v_norm**2
                h2 = jnp.sqrt(8 / (kf * S_new)) * (
                    norm_grad_bar - L * v_norm / kf
                )
                h_new = jnp.maximum(h, jnp.maximum(h1, h2))
                restart_h = kf * (kf + 1.0) * h_new > 3 / 8 * L

                def h_restart_branch(_):
                    return init_epoch(
                        x_best_cand,
                        val_best_cand,
                        grad_best_cand,
                        L * beta_dec,
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray(1, dtype=jnp.int32),
                    )

                def continue_branch(_):
                    # Official update: xbar <- (k*xbar + x_k)/(k+1), then evaluate it.
                    xbar_new = (kf * xbar + x_cand) / (kf + 1.0)
                    val_bar_new, grad_bar_new = val_grad(xbar_new)

                    better_bar = val_bar_new < val_best_cand
                    val_best_new = jnp.minimum(val_best_cand, val_bar_new)
                    x_best_new = jax.lax.select(
                        better_bar, xbar_new, x_best_cand
                    )
                    grad_best_new = jax.lax.select(
                        better_bar, grad_bar_new, grad_best_cand
                    )

                    # Marumo's runner reports the componentwise minima over
                    # the two advertised solutions x and xbar.
                    val_out = jnp.minimum(val_cand, val_bar_new)
                    grad_norm_out = jnp.minimum(
                        jnp.linalg.norm(grad_cand),
                        jnp.linalg.norm(grad_bar_new),
                    )
                    return HBStep(
                        k=k_new,
                        h=h_new,
                        x=x_cand,
                        xbar=xbar_new,
                        v=v_new,
                        val=val_cand,
                        grad=grad_cand,
                        val_bar=val_bar_new,
                        grad_bar=grad_bar_new,
                        L=L,
                        S=S_new,
                        val_best=val_best_new,
                        x_best=x_best_new,
                        grad_best=grad_best_new,
                        val_out=val_out,
                        grad_norm_out=grad_norm_out,
                        n_val=jnp.asarray(2, dtype=jnp.int32),
                        n_grad=jnp.asarray(2, dtype=jnp.int32),
                    )

                return jax.lax.cond(
                    restart_h, h_restart_branch, continue_branch, None
                )

            return jax.lax.cond(
                smooth_fail, smooth_restart_branch, smooth_ok_branch, None
            )

        self._hb_step = hb_step

    def _warmup_kernels(self, x, val, grad):
        out = self._hb_step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=x.dtype),
            x,
            x,
            jnp.zeros_like(x),
            val,
            grad,
            val,
            grad,
            jnp.asarray(self.L_est, dtype=x.dtype),
            jnp.asarray(0.0, dtype=x.dtype),
            val,
            x,
            grad,
        )
        self._block_until_ready(*out)

    def optimize(self):
        dtype = self.model.param.dtype
        k = jnp.asarray(0, dtype=jnp.int32)
        h = jnp.asarray(0.0, dtype=dtype)
        x = self.model.param
        xbar = x
        v = jnp.zeros_like(x)
        L = jnp.asarray(self.L_est, dtype=dtype)
        S = jnp.asarray(0.0, dtype=dtype)

        tic = time.perf_counter()
        val, grad = self.oracle.value_and_grad(x)
        self._block_until_ready(val, grad)
        self.elapsed_time += time.perf_counter() - tic

        val_bar, grad_bar = val, grad
        val_best, x_best, grad_best = val, x, grad
        it = 0

        while self.stop_condition():
            tic = time.perf_counter()
            it += 1

            step = self._hb_step(
                k, h, x, xbar, v, val, grad, val_bar, grad_bar,
                L, S, val_best, x_best, grad_best
            )
            self._block_until_ready(step.grad_norm_out)
            self.elapsed_time += time.perf_counter() - tic

            k, h = step.k, step.h
            x, xbar, v = step.x, step.xbar, step.v
            val, grad = step.val, step.grad
            val_bar, grad_bar = step.val_bar, step.grad_bar
            L, S = step.L, step.S
            val_best, x_best, grad_best = (
                step.val_best, step.x_best, step.grad_best
            )

            self.oracle.add_count(
                val=int(jax.device_get(step.n_val)),
                grad=int(jax.device_get(step.n_grad)),
                joint=int(jax.device_get(step.n_grad)),
            )

            if self._should_save_hist(it):
                self._save_call_hist()
                self.grad_hist.append(step.grad_norm_out)
                self.val_hist.append(step.val_out)
                self.time_hist.append(self.elapsed_time)

        self._block_until_ready(x)
        self.L_est = float(jax.device_get(L))
        self.model.param = x

    def run(self):
        self.warmup()
        self.optimize()
        self.grad_call_hist = np.array(self.grad_call_hist)
        self.val_call_hist = np.array(self.val_call_hist)
        self.grad_val_call_hist = np.array(self.grad_val_call_hist)
        self.joint_call_hist = np.array(self.joint_call_hist)
        self.oracle_call_hist = np.array(self.oracle_call_hist)
        self.grad_hist = np.array(self.grad_hist)
        self.val_hist = np.array(self.val_hist)
        self.time_hist = np.array(self.time_hist)
        if self.save_hist:
            self._savez('_restartHB', grad_call=self.grad_call_hist, val_call=self.val_call_hist, grad_val_call=self.grad_val_call_hist, grad=self.grad_hist, val=self.val_hist, time=self.time_hist)


class AdaRestartAGD(BaseOptimizer):
    def __init__(self, model):
        super().__init__(model)
        # Keep L_init=1e-3, matching the AGD baseline used in Marumo's HB repo.
        self.L_est = 1e-3
        self.M0 = 1e-16
        self.beta_inc = 2
        self.beta_dec = 0.9
        self.inner_init()
        self.total_iter = 0

        loss = model.loss
        val_grad = jax.value_and_grad(loss)
        M0 = self.M0
        beta_inc = self.beta_inc
        beta_dec = self.beta_dec

        @jax.jit
        def agd_step(k, M, S, Z, x, y, ybar, val0, val_x, grad_x, grad_y, L):
            """One Marumo restarted-AGD update on the device."""
            k_new = k + 1
            kf = k_new.astype(x.dtype)
            theta = kf / (kf + 1.0)

            x_prev = x
            x_cand = y - grad_y / L
            dx = x_cand - x_prev
            y_cand = x_cand + theta * dx
            dx_norm = jnp.linalg.norm(dx)
            S_cand = S + dx_norm**2

            # Marumo evaluates both f and grad f at x_k before the early restart test.
            val_x_cand, grad_x_cand = val_grad(x_cand)
            restart_early = (
                val_x_cand - val0 > -0.5 * (1.0 - theta) * L * S_cand
            )

            def safe_div(a, b):
                zero = jnp.asarray(0.0, dtype=x.dtype)
                return jax.lax.cond(
                    jnp.logical_and(a > 0, b > 0),
                    lambda _: a / b,
                    lambda _: zero,
                    None,
                )

            def early_branch(_):
                zero = jnp.asarray(0.0, dtype=x.dtype)
                return AGDStep(
                    k=jnp.asarray(0, dtype=jnp.int32),
                    M=jnp.asarray(M0, dtype=x.dtype),
                    S=zero,
                    Z=jnp.asarray(1.0, dtype=x.dtype),
                    x=x_prev,
                    y=x_prev,
                    ybar=x_prev,
                    val0=val_x,
                    val_x=val_x,
                    grad_x=grad_x,
                    grad_y=grad_x,
                    L=L * beta_inc,
                    n_val=jnp.asarray(1, dtype=jnp.int32),
                    n_grad=jnp.asarray(1, dtype=jnp.int32),
                )

            def accepted_branch(_):
                val_y, grad_y_cand = val_grad(y_cand)
                dy = y_cand - x_cand

                M1 = safe_div(
                    12 * (
                        val_y
                        - val_x_cand
                        - 0.5 * jnp.vdot(grad_y_cand + grad_x_cand, dy)
                    ),
                    jnp.linalg.norm(dy) ** 3,
                )
                M2 = safe_div(
                    jnp.linalg.norm(
                        grad_y_cand
                        + theta * grad_x
                        - (1.0 + theta) * grad_x_cand
                    ),
                    theta * dx_norm**2,
                )
                M_cand = jnp.maximum(M, jnp.maximum(M1, M2))

                restart_M = (M_cand / L) ** 2 * S_cand > (1.0 - theta) ** 5
                Z_cand = 1.0 + theta * Z
                ybar_cand = (y_cand + theta * Z * ybar) / Z_cand

                def restart_M_branch(_):
                    zero = jnp.asarray(0.0, dtype=x.dtype)
                    return AGDStep(
                        k=jnp.asarray(0, dtype=jnp.int32),
                        M=jnp.asarray(M0, dtype=x.dtype),
                        S=zero,
                        Z=jnp.asarray(1.0, dtype=x.dtype),
                        x=x_cand,
                        y=x_cand,
                        ybar=x_cand,
                        val0=val_x_cand,
                        val_x=val_x_cand,
                        grad_x=grad_x_cand,
                        grad_y=grad_x_cand,
                        L=L * beta_dec,
                        n_val=jnp.asarray(2, dtype=jnp.int32),
                        n_grad=jnp.asarray(2, dtype=jnp.int32),
                    )

                def continue_branch(_):
                    return AGDStep(
                        k=k_new,
                        M=M_cand,
                        S=S_cand,
                        Z=Z_cand,
                        x=x_cand,
                        y=y_cand,
                        ybar=ybar_cand,
                        val0=val0,
                        val_x=val_x_cand,
                        grad_x=grad_x_cand,
                        grad_y=grad_y_cand,
                        L=L,
                        n_val=jnp.asarray(2, dtype=jnp.int32),
                        n_grad=jnp.asarray(2, dtype=jnp.int32),
                    )

                return jax.lax.cond(
                    restart_M, restart_M_branch, continue_branch, None
                )

            return jax.lax.cond(
                restart_early, early_branch, accepted_branch, None
            )

        self._agd_step = agd_step

    def inner_init(self):
        self.M = self.M0
        self.k = 0
        self.S = 0
        self.Z = 1

    def _warmup_kernels(self, x, val, grad):
        out = self._agd_step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(self.M0, dtype=x.dtype),
            jnp.asarray(0.0, dtype=x.dtype),
            jnp.asarray(1.0, dtype=x.dtype),
            x,
            x,
            x,
            val,
            val,
            grad,
            grad,
            jnp.asarray(self.L_est, dtype=x.dtype),
        )
        self._block_until_ready(*out)

    def _finish_step(self, ybar, val_x, grad_x, tic):
        self._block_until_ready(ybar, val_x, grad_x)
        self.elapsed_time += time.perf_counter() - tic

        # Marumo's runner evaluates ybar outside the timed update and without
        # incrementing the oracle count, then reports componentwise minima over x/ybar.
        val_bar, grad_bar = self.oracle.value_and_grad(ybar, count=False)
        grad_norm_bar = jnp.linalg.norm(grad_bar)
        grad_norm_x = jnp.linalg.norm(grad_x)
        val_out = jnp.minimum(val_x, val_bar)
        grad_norm_out = jnp.minimum(grad_norm_x, grad_norm_bar)
        self._block_until_ready(val_out, grad_norm_out)

        if self._should_save_hist(self.total_iter):
            self._save_call_hist()
            self.grad_hist.append(grad_norm_out)
            self.val_hist.append(val_out)
            self.time_hist.append(self.elapsed_time)

    def optimize(self):
        dtype = self.model.param.dtype
        x = self.model.param
        y = x
        ybar = x
        L = jnp.asarray(self.L_est, dtype=dtype)
        k = jnp.asarray(0, dtype=jnp.int32)
        M = jnp.asarray(self.M0, dtype=dtype)
        S = jnp.asarray(0.0, dtype=dtype)
        Z = jnp.asarray(1.0, dtype=dtype)

        val_x, grad_x = self.oracle.value_and_grad(x)
        val0 = val_x
        grad_y = grad_x
        self._block_until_ready(val_x, grad_x)

        while self.stop_condition():
            tic = time.perf_counter()
            self.total_iter += 1

            step = self._agd_step(
                k, M, S, Z, x, y, ybar, val0, val_x, grad_x, grad_y, L
            )
            self._block_until_ready(step.ybar)

            k, M, S, Z = step.k, step.M, step.S, step.Z
            x, y, ybar = step.x, step.y, step.ybar
            val0, val_x = step.val0, step.val_x
            grad_x, grad_y, L = step.grad_x, step.grad_y, step.L

            self.oracle.add_count(
                val=int(jax.device_get(step.n_val)),
                grad=int(jax.device_get(step.n_grad)),
                joint=int(jax.device_get(step.n_grad)),
            )
            self._finish_step(ybar, val_x, grad_x, tic)

        self._block_until_ready(x)
        self.L_est = float(jax.device_get(L))
        self.M = float(jax.device_get(M))
        self.k = int(jax.device_get(k))
        self.S = float(jax.device_get(S))
        self.Z = float(jax.device_get(Z))
        self.model.param = x

    def run(self):
        self.warmup()
        self.optimize()
        self.grad_call_hist = np.array(self.grad_call_hist)
        self.val_call_hist = np.array(self.val_call_hist)
        self.grad_val_call_hist = np.array(self.grad_val_call_hist)
        self.joint_call_hist = np.array(self.joint_call_hist)
        self.oracle_call_hist = np.array(self.oracle_call_hist)
        self.grad_hist = np.array(self.grad_hist)
        self.val_hist = np.array(self.val_hist)
        self.time_hist = np.array(self.time_hist)
        if self.save_hist:
            self._savez('_restartAGD', grad_call=self.grad_call_hist, val_call=self.val_call_hist, grad_val_call=self.grad_val_call_hist, grad=self.grad_hist, val=self.val_hist, time=self.time_hist)


class NAGsc(BaseOptimizer):
    def __init__(self, model):
        assert hasattr(model, 'mu'), "The model must have an attribute 'mu'"
        super().__init__(model)
        self.mu = model.mu
        self.L_est = 1e-3 + self.mu
        self.beta_inc = 1.1
        self.beta_dec = 0.99

        loss = model.loss
        value_and_grad = jax.value_and_grad(loss)
        mu = self.mu
        beta_inc = self.beta_inc
        beta_dec = self.beta_dec

        @jax.jit
        def line_search(x, z, A, L_est):
            def trial(L):
                q = mu / L
                As = (2*A + 1 + jnp.sqrt(4*A + 4*q*A*A + 1)) / (2*(1 - q))
                t = (As - A)*(1 + q*A) / (As + 2*q*A*As - q*A*A)
                d = (As - A) / (1 + q*As)
                y = x + t*z - t*x
                val_y, grad = value_and_grad(y)
                x_cand = y - grad / L
                val_x_cand = loss(x_cand)
                step = x_cand - y
                ok = val_x_cand - jnp.vdot(grad, step) <= val_y + 0.5 * L * jnp.vdot(step, step)
                return As, d, y, grad, x_cand, val_y, val_x_cand, ok

            As, d, y, grad, x_cand, val_y, val_x_cand, ok = trial(L_est)
            st = NAGLSState(
                L_est, As, d, y, grad, x_cand, val_y, val_x_cand, ok,
                jnp.asarray(1, dtype=jnp.int32),
            )

            def cond(st):
                return jnp.logical_not(st.ok)

            def body(st):
                L = st.L * beta_inc
                As, d, y, grad, x_cand, val_y, val_x_cand, ok = trial(L)
                return NAGLSState(
                    L, As, d, y, grad, x_cand, val_y, val_x_cand, ok, st.n_trial + 1
                )

            st = jax.lax.while_loop(cond, body, st)
            q = mu / st.L
            z_new = (1 - q*st.d)*z + q*st.d*st.y + st.d*st.x_cand - st.d*st.y
            return NAGStep(
                st.L * beta_dec, st.A, st.d, st.y, st.grad, st.x_cand, st.val_y,
                st.val_cand, st.ok, st.n_trial, z_new,
            )

        self._nag_line_search = line_search

    def _warmup_kernels(self, x, val, grad):
        out = self._nag_line_search(
            x, x, jnp.asarray(0.0, dtype=x.dtype), jnp.asarray(self.L_est, dtype=x.dtype)
        )
        self._block_until_ready(*out)

    def optimize(self):
        x = self.model.param
        z = x
        A = jnp.asarray(0.0, dtype=x.dtype)
        L_est = jnp.asarray(self.L_est, dtype=x.dtype)
        it = 0

        while self.stop_condition():
            tic = time.perf_counter()
            it += 1
            out = self._nag_line_search(x, z, A, L_est)
            L_est, As, d, y, grad, x_cand, _, _, _, n_trial, z_new = out
            self._block_until_ready(x_cand, z_new, L_est, n_trial)
            self.elapsed_time += time.perf_counter() - tic

            # Host-side oracle bookkeeping is excluded from optimizer timing.
            n_trial_host = int(jax.device_get(n_trial))
            # Each trial corresponds to grad(y), loss(y), and loss(x_cand) in the original code.
            self.oracle.add_count(val=2 * n_trial_host, grad=n_trial_host, joint=n_trial_host)

            x = x_cand
            z = z_new
            A = As

            # Diagnostics are not part of the algorithmic oracle count.
            val_x, grad_x = self.oracle.value_and_grad(x, count=False)
            grad_norm = jnp.linalg.norm(grad_x)
            self._block_until_ready(val_x, grad_norm)
            if self._should_save_hist(it):
                self._save_call_hist()
                self.grad_hist.append(grad_norm)
                self.val_hist.append(val_x)
                self.time_hist.append(self.elapsed_time)

        self.L_est = float(jax.device_get(L_est))
        self.model.param = x

    def run(self):
        self.warmup()
        self.optimize()
        self.grad_call_hist = np.array(self.grad_call_hist)
        self.val_call_hist = np.array(self.val_call_hist)
        self.grad_val_call_hist = np.array(self.grad_val_call_hist)
        self.joint_call_hist = np.array(self.joint_call_hist)
        self.oracle_call_hist = np.array(self.oracle_call_hist)
        self.grad_hist = np.array(self.grad_hist)
        self.val_hist = np.array(self.val_hist)
        self.time_hist = np.array(self.time_hist)
        if self.save_hist:
            self._savez('_nagsc', grad_call=self.grad_call_hist, val_call=self.val_call_hist, grad_val_call=self.grad_val_call_hist, grad=self.grad_hist, val=self.val_hist, time=self.time_hist)

class NAGc(BaseOptimizer):
    """Adaptive Nesterov accelerated gradient method for convex objectives.

    This is the mu=0 counterpart of NAGsc. The local smoothness estimate is
    adapted by the same backtracking rule used in NAGsc.
    """
    def __init__(self, model):
        super().__init__(model)
        self.L_est = 1e-3
        self.beta_inc = 1.1
        self.beta_dec = 0.99

        loss = model.loss
        value_and_grad = jax.value_and_grad(loss)
        beta_inc = self.beta_inc
        beta_dec = self.beta_dec

        @jax.jit
        def line_search(x, z, A, L_est):
            def trial(L):
                # mu = 0 specialization of the NAG-SC estimate-sequence step.
                As = A + 0.5 * (1.0 + jnp.sqrt(4.0 * A + 1.0))
                t = (As - A) / As
                d = As - A
                y = x + t * z - t * x
                val_y, grad = value_and_grad(y)
                x_cand = y - grad / L
                val_x_cand = loss(x_cand)
                step = x_cand - y
                ok = val_x_cand - jnp.vdot(grad, step) <= val_y + 0.5 * L * jnp.vdot(step, step)
                return As, d, y, grad, x_cand, val_y, val_x_cand, ok

            As, d, y, grad, x_cand, val_y, val_x_cand, ok = trial(L_est)
            st = NAGLSState(
                L_est, As, d, y, grad, x_cand, val_y, val_x_cand, ok,
                jnp.asarray(1, dtype=jnp.int32),
            )

            def cond(st):
                return jnp.logical_not(st.ok)

            def body(st):
                L = st.L * beta_inc
                As, d, y, grad, x_cand, val_y, val_x_cand, ok = trial(L)
                return NAGLSState(
                    L, As, d, y, grad, x_cand, val_y, val_x_cand, ok, st.n_trial + 1
                )

            st = jax.lax.while_loop(cond, body, st)
            z_new = z + st.d * (st.x_cand - st.y)
            return NAGStep(
                st.L * beta_dec, st.A, st.d, st.y, st.grad, st.x_cand, st.val_y,
                st.val_cand, st.ok, st.n_trial, z_new,
            )

        self._nag_line_search = line_search

    def _warmup_kernels(self, x, val, grad):
        out = self._nag_line_search(
            x, x, jnp.asarray(0.0, dtype=x.dtype), jnp.asarray(self.L_est, dtype=x.dtype)
        )
        self._block_until_ready(*out)

    def optimize(self):
        x = self.model.param
        z = x
        A = jnp.asarray(0.0, dtype=x.dtype)
        L_est = jnp.asarray(self.L_est, dtype=x.dtype)
        it = 0

        while self.stop_condition():
            tic = time.perf_counter()
            it += 1
            out = self._nag_line_search(x, z, A, L_est)
            L_est, As, d, y, grad, x_cand, _, _, _, n_trial, z_new = out
            self._block_until_ready(x_cand, z_new, L_est, n_trial)
            self.elapsed_time += time.perf_counter() - tic

            # Host-side oracle bookkeeping is excluded from optimizer timing.
            n_trial_host = int(jax.device_get(n_trial))
            # Each trial evaluates grad(y), loss(y), and loss(x_cand).
            self.oracle.add_count(val=2 * n_trial_host, grad=n_trial_host, joint=n_trial_host)

            x = x_cand
            z = z_new
            A = As

            # Diagnostics are excluded from both algorithmic oracle counts and timing.
            val_x, grad_x = self.oracle.value_and_grad(x, count=False)
            grad_norm = jnp.linalg.norm(grad_x)
            self._block_until_ready(val_x, grad_norm)
            if self._should_save_hist(it):
                self._save_call_hist()
                self.grad_hist.append(grad_norm)
                self.val_hist.append(val_x)
                self.time_hist.append(self.elapsed_time)

        self.L_est = float(jax.device_get(L_est))
        self.model.param = x

    def run(self):
        self.warmup()
        self.optimize()
        self.grad_call_hist = np.array(self.grad_call_hist)
        self.val_call_hist = np.array(self.val_call_hist)
        self.grad_val_call_hist = np.array(self.grad_val_call_hist)
        self.joint_call_hist = np.array(self.joint_call_hist)
        self.oracle_call_hist = np.array(self.oracle_call_hist)
        self.grad_hist = np.array(self.grad_hist)
        self.val_hist = np.array(self.val_hist)
        self.time_hist = np.array(self.time_hist)
        if self.save_hist:
            self._savez(
                '_nagc',
                grad_call=self.grad_call_hist,
                val_call=self.val_call_hist,
                grad_val_call=self.grad_val_call_hist,
                grad=self.grad_hist,
                val=self.val_hist,
                time=self.time_hist,
            )
