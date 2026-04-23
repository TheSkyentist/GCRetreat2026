import math

import jax
import jax.numpy as jnp
import numpy as np
from numba import njit

from bench_utils import bench, print_results

jax.config.update('jax_enable_x64', True)

# ── Config ─────────────────────────────────────────────────────────────────────
# Operation applied element-wise to every value in the array:
#
#   y = sin(x)·cos(x) + tanh(2x) − √x · e^(−3x)
#
# NumPy evaluates each ufunc eagerly, allocating a brand-new 80 MB array for
# every intermediate result (~9 allocations = ~720 MB of temporaries).
# Numba and JAX/XLA compile the entire expression into a SINGLE kernel pass —
# one read, one write, zero intermediate arrays.  That is operator fusion.
N = 10_000_000

arr_np = np.linspace(0.01, 1.0, N, dtype=np.float64)
arr_list = arr_np.tolist()
arr_jnp = jnp.array(arr_np)


# ── For loop ────────────────────────────────────────────────────────────────
# Pure CPython: each math call crosses the interpreter boundary; no SIMD;
# no vectorisation.  Expect ~6–15 s on modern hardware.
def transform_forloop():
    out = [0.0] * N
    for i, x in enumerate(arr_list):
        out[i] = (
            math.sin(x) * math.cos(x)
            + math.tanh(2.0 * x)
            - math.sqrt(x) * math.exp(-3.0 * x)
        )
    return out


# ── List comprehension ──────────────────────────────────────────────────────
# Same per-element work as the for loop; the comprehension bytecode is slightly
# tighter but the bottleneck is identical C-extension call overhead.
def transform_listcomp():
    return [
        math.sin(x) * math.cos(x)
        + math.tanh(2.0 * x)
        - math.sqrt(x) * math.exp(-3.0 * x)
        for x in arr_list
    ]


# ── map() ──────────────────────────────────────────────────────────────────
# map() avoids the list-comprehension's STORE_FAST overhead but still calls
# the same math functions one element at a time; the gain is marginal.
def transform_map():
    def f(x):
        return (
            math.sin(x) * math.cos(x)
            + math.tanh(2.0 * x)
            - math.sqrt(x) * math.exp(-3.0 * x)
        )

    return list(map(f, arr_list))


# ── NumPy  (ufuncs — eager, ~9 intermediate 80 MB arrays) ──────────────────
# Each ufunc (sin, cos, multiply, tanh, …) allocates its own output buffer.
# The data is read from RAM, transformed, written back, then read again for the
# next op — roughly 9× the memory bandwidth of a fused implementation.
def transform_numpy():
    x = arr_np
    return np.sin(x) * np.cos(x) + np.tanh(2.0 * x) - np.sqrt(x) * np.exp(-3.0 * x)


# ── Numba JIT  (single-pass, zero intermediate allocations) ────────────────
# LLVM compiles the scalar loop to vectorised machine code.  All six math ops
# are computed for one element before moving to the next — one read, one write,
# no temporaries.  Memory bandwidth usage matches the theoretical minimum.
@njit
def _numba_transform(x):
    return np.sin(x) * np.cos(x) + np.tanh(2.0 * x) - np.sqrt(x) * np.exp(-3.0 * x)


def transform_numba():
    return _numba_transform(arr_np)


# ── JAX JIT  (XLA-fused kernel, zero intermediate allocations) ─────────────
# XLA traces the Python function, recognises the chain of elementwise ops, and
# emits a single fused HLO kernel — identical fusion strategy to Numba but
# expressed at the array level rather than the scalar loop level.
# On a GPU this gap widens dramatically; on CPU it still beats NumPy ~5–15×.
@jax.jit
def _jax_transform(x):
    return jnp.sin(x) * jnp.cos(x) + jnp.tanh(2.0 * x) - jnp.sqrt(x) * jnp.exp(-3.0 * x)


def transform_jax():
    return _jax_transform(arr_jnp).block_until_ready()


# ── Warm-up JIT compilers (compilation time excluded from benchmark) ───────────
print('Warming up JIT compilers...')
_numba_transform(arr_np)
_jax_transform(arr_jnp).block_until_ready()

# ── Run benchmarks ─────────────────────────────────────────────────────────────
bench('For loop', transform_forloop)
bench('List comprehension', transform_listcomp)
bench('Map', transform_map)
bench('NumPy (stepwise)', transform_numpy)
bench('Numba JIT (fused)', transform_numba)
bench('JAX JIT (XLA fused)', transform_jax)

print_results('Chained ops on 10,000,000 floats  [y = sin·cos + tanh(2x) − √x·e^(−3x)]')
