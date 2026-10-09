import jax.numpy as jnp
import numpy as np
import jax
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import functools
import sys
from sklearn.datasets import fetch_openml
from jax import nn, random

jax.config.update("jax_enable_x64", True)

class Oracle:
    def __init__(self, model):
        self.__loss = model.loss
        self.__grad = jax.grad(self.__loss)
        self.__value_and_grad = jax.value_and_grad(self.__loss)
        # val/grad retain their component counts; joint records their overlap.
        self.__keys = ("val", "grad", "joint")
        self.count = dict.fromkeys(self.__keys, 0)

    def reset_count(self):
        self.count = dict.fromkeys(self.__keys, 0)

    @property
    def oracle_calls(self):
        """One call for value-only, gradient-only, or joint evaluation."""
        return self.count["val"] + self.count["grad"] - self.count["joint"]

    def add_count(self, val=0, grad=0, joint=0):
        """Charge actual JIT evaluations, including rejected/rolled-back work."""
        val, grad, joint = int(val), int(grad), int(joint)
        if min(val, grad, joint) < 0 or joint > min(val, grad):
            raise ValueError("Joint calls must be a subset of both value and gradient calls.")
        self.count["val"] += int(val)
        self.count["grad"] += int(grad)
        self.count["joint"] += joint

    def loss(self, x, count=True):
        if count:
            self.count["val"] += 1
        return self.__loss(x)

    def grad(self, x, count=True):
        if count:
            self.count["grad"] += 1
        return self.__grad(x)

    def value_and_grad(self, x, count=True, count_val=None, count_grad=None):
        if count_val is None:
            count_val = count
        if count_grad is None:
            count_grad = count
        if count_val:
            self.count["val"] += 1
        if count_grad:
            self.count["grad"] += 1
        if count_val and count_grad:
            self.count["joint"] += 1
        return self.__value_and_grad(x)

class Reg_lin:
    def __init__(self, features=20):
        # Create synthetic data
        self.X, self.y, self.coef = make_regression(
            n_samples=100, n_features=features, noise=0.1, coef=True, random_state=42
        )
        self.X, self.y = jnp.array(self.X, dtype=jnp.float64), jnp.array(self.y, dtype=jnp.float64)
        self.param = jnp.zeros(self.X.shape[1], dtype=jnp.float64) 
        self.L1 = 5
        self.L2 = 5
        self.mu = 0.1
        self.name = "Reg_lin"

    def loss(self, theta):
        # Forward pass: Compute predicted y
        y_pred = self.X @ theta
        # Compute loss
        loss = jnp.mean((y_pred - self.y) ** 2)
        return loss
    
    def plot(self):
        # Plot non-zero coefficients
        plt.stem(jax.device_get(self.param), label="FISTA")
        plt.stem(self.coef, linefmt="r-", markerfmt="ro", label="True")
        plt.title("Lasso Coefficients with FISTA using PyTorch's Optimizer")
        plt.legend()
        plt.show()

    def reset(self):
        self.param = jnp.zeros(self.X.shape[1], dtype=jnp.float64)

class Quad_arith:
    def __init__(self, n=20, L=20):
        self.param = jnp.zeros(n, dtype=jnp.float64)
        self.n = n
        self.L = L
        self.mu = 1
        self.name = "Quad_arith"

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, z):
        loss = 0.5*(z - jnp.ones_like(z))**2 @ jnp.linspace(1,self.L,self.n, dtype=jnp.float64)
        return loss
    
    def reset(self):
        self.param = jnp.zeros(self.n, dtype=jnp.float64)

class HBfail:
    def __init__(self, n=20, p=20, mu=0.1):
        self.param = jnp.ones(n, dtype=jnp.float64)
        self.n = n
        self.mu = mu
        self.p = p

        # self.A_T = torch.randn(self.p,self.n, dtype=torch.float64)
        # self.b = torch.randn(self.p, dtype=torch.float64)
        # w,v = torch.linalg.eig(self.A_T@torch.t(self.A_T))
        # self.L = torch.sqrt(max(torch.real(w))).item() + self.mu
        self.L = self.n + self.mu
        self.name = "HBfail"

    def loss(self, z):
        res = 0
        # for i in range(self.p):
        #     xx = self.A_T[i]@z - self.b[i]
        #     if xx > 0:
        #         res = res + 0.5*xx**2*torch.exp(-0.001/xx)
        for i in range(self.n):
            if z[i]+1 > 0:
                res = res + (i+1)*0.5*(z[i]+1)**2*jnp.exp(-0.001/(z[i]+1))
        res += 0.5*self.mu*jnp.sum((z + jnp.ones_like(z))**2)
        return res
    
    def reset(self):
        self.param = jnp.ones(self.n, dtype=jnp.float64)


class Rosenbrock:
    def __init__(self, d, sigma_x0=1, seed=0):
        self.name = "Rosenbrock"
        key = jax.random.PRNGKey(seed)
        self.d = d
        self.x_opt = jnp.ones(d)
        self.x0 = self.x_opt + jax.random.normal(key, (d,)) * sigma_x0
        self.param = self.x0
        self.L1 = 10000
        self.L2 = 50000

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, x):
        return 100 * jnp.sum((x[1:] - x[:-1] ** 2) ** 2) + jnp.sum((x[:-1] - 1) ** 2)
    
    def reset(self):
        self.param = self.x0

class Dixon_price:
    def __init__(self, d, sigma_x0=1, seed=0):
        self.name = "Dixon_price"
        key = jax.random.PRNGKey(seed)
        self.d = d
        # self.x0 = jax.random.normal(key, (d,)) * sigma_x0
        # self.x0 = jnp.ones((d,))

        self.x_opt = jnp.power(2.0, (jnp.power(2.0, -jnp.arange(self.d)) - 1))
        self.x0 = self.x_opt + jax.random.normal(key, (d,)) * sigma_x0
        self.param = self.x0
        self.L1 = 10000
        self.L2 = 1000

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, x):
        return (x[0] - 1) ** 2 + jnp.vdot(jnp.arange(2, self.d + 1), (2 * x[1:] ** 2 - x[:-1]) ** 2)
    
    def reset(self):
        self.param = self.x0

class Powell:
    def __init__(self, d, sigma_x0=1, seed=0):
        self.name = "Powell"
        assert d % 4 == 0
        key = jax.random.PRNGKey(seed)
        self.d = d
        self.x_opt = jnp.zeros(d)
        self.x0 = self.x_opt + jax.random.normal(key, (d,)) * sigma_x0
        self.param = self.x0
        self.L1 = 200
        self.L2 = 500


    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, x):
        return (
            jnp.sum((x[::4] + 10 * x[1::4]) ** 2)
            + jnp.sum(5 * (x[2::4] - x[3::4]) ** 2)
            + jnp.sum((x[1::4] - 2 * x[2::4]) ** 4)
            + jnp.sum(10 * (x[::4] - x[3::4]) ** 4)
        )
    
    def reset(self):
        self.param = self.x0

_qing_barrier = jax.lax.optimization_barrier

@jax.custom_jvp
def _qing_square_minus_index(x, index):
    splitter = jnp.asarray(134217729., dtype=x.dtype)  # 2**27 + 1 for float64
    c = _qing_barrier(splitter * x)
    hi = _qing_barrier(c - _qing_barrier(c - x))
    lo = _qing_barrier(x - hi)
    product = _qing_barrier(x * x)
    e1 = _qing_barrier(_qing_barrier(hi * hi) - product)
    e2 = _qing_barrier(e1 + _qing_barrier(_qing_barrier(2. * hi) * lo))
    product_error = _qing_barrier(e2 + _qing_barrier(lo * lo))
    # TwoSum(product, -index), followed by the low part of the product.
    difference = _qing_barrier(product - index)
    bv = _qing_barrier(difference - product)
    av = _qing_barrier(difference - bv)
    sum_error = _qing_barrier(_qing_barrier(product - av) + _qing_barrier(-index - bv))
    return difference + _qing_barrier(sum_error + product_error)

@_qing_square_minus_index.defjvp
def __qing_square_minus_index_jvp(primals, tangents):
    x, index = primals
    dx, di = tangents
    return _qing_square_minus_index(x, index), 2. * x * dx - di

class Qing:
    def __init__(self, d, sigma_x0=1, seed=0):
        self.name = "Qing"
        key = jax.random.PRNGKey(seed)
        self.d = d
        self.x_opt = jnp.sqrt(jnp.arange(1, self.d + 1))
        self.x0 = self.x_opt + jax.random.normal(key, (d,)) * sigma_x0
        self.param = self.x0
        self.L1 = 4*d
        self.L2 = 500

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, x):
        # Evaluate the same Qing polynomial with a compensated residual.
        # The custom JVP supplies its mathematical first derivative.
        index = jnp.arange(1, self.d + 1, dtype=x.dtype)
        residual = _qing_square_minus_index(x, index)
        return jnp.sum(residual * residual)
    
    def reset(self):
        self.param = self.x0

class FashionMNIST_MLP:
    def __init__(self, widths=(1024, 1024, 512), n_samples=None, batch_size=None, seed=0, acc_eval_samples=None):
        self.name = "FMNISTMLP"

        # ---- data ----
        # ---- data (NO tensorflow) ----
        X, y = fetch_openml(
            "Fashion-MNIST",
            version=1,
            return_X_y=True,
            as_frame=False
        )

        X = X.astype(np.float64) / 255.0
        y = y.astype(np.int64)

        # train / test split（公式 split: 60000 / 10000）
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        if n_samples is not None:
            X_train = X_train[:n_samples]
            y_train = y_train[:n_samples]

        self.X = jnp.array(X_train)
        self.y = jnp.array(y_train)
        self.X_test = jnp.array(X_test)
        self.y_test = jnp.array(y_test)

        self.num_classes = 10
        self.batch_size = batch_size
        self.acc_eval_samples = acc_eval_samples  # Noneなら全test、整数ならサブサンプル

        key = random.PRNGKey(seed)

        # 固定ミニバッチ（毎回同じバッチでlossを評価）
        if batch_size is not None:
            idx = random.choice(key, self.X.shape[0], shape=(batch_size,), replace=False)
            self.Xb = self.X[idx]
            self.yb = self.y[idx]
        else:
            self.Xb, self.yb = self.X, self.y

        # ---- network shapes (flattened param) ----
        dims = [28 * 28] + list(widths) + [self.num_classes]
        self.shapes = []
        for din, dout in zip(dims[:-1], dims[1:]):
            self.shapes.append(("W", (din, dout)))
            self.shapes.append(("b", (dout,)))

        self.sizes = [int(jnp.prod(jnp.array(s))) for _, s in self.shapes]
        self.dim = int(sum(self.sizes))

        key, sub = random.split(key)
        self.param0 = 0.01 * random.normal(sub, (self.dim,), dtype=jnp.float64)
        self.param = self.param0

        self.L1 = 1e3
        self.L2 = 1e4

    def unpack(self, z):
        params = []
        i = 0
        for (_, shape), size in zip(self.shapes, self.sizes):
            params.append(z[i:i+size].reshape(shape))
            i += size
        return params

    @functools.partial(jax.jit, static_argnums=(0,))
    def _logits(self, z, X):
        params = self.unpack(z)
        h = X
        L = len(params) // 2
        for ell in range(L):
            W = params[2*ell]
            b = params[2*ell + 1]
            h = h @ W + b
            if ell < L - 1:
                h = nn.relu(h)
        return h  # logits

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, z, X):
        logits = self._logits(z, X)
        return jnp.argmax(logits, axis=1)

    def accuracy(self, z, split="test"):
        if split == "train":
            X, y = self.X, self.y
        else:
            X, y = self.X_test, self.y_test

        if self.acc_eval_samples is not None:
            # 先頭からサブサンプル（速度優先・再現性あり）
            n = min(int(self.acc_eval_samples), X.shape[0])
            X, y = X[:n], y[:n]

        pred = self.predict(z, X)
        return jnp.mean(pred == y)

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, z):
        X = self.Xb
        y = self.yb
        y_onehot = jax.nn.one_hot(y, self.num_classes)
        logits = self._logits(z, X)
        logp = nn.log_softmax(logits, axis=1)
        return -jnp.mean(jnp.sum(y_onehot * logp, axis=1))

    def reset(self):
        self.param = self.param0
