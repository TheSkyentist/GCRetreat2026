import jax
import jax.numpy as jnp
import numpy as np
from numba import njit

from bench_utils import bench, print_results

jax.config.update('jax_enable_x64', True)

# ── Config ─────────────────────────────────────────────────────────────────────
N = 10_000_000

arr = list(range(1, N + 1))
arr_np = np.arange(1, N + 1, dtype=np.int64)
arr_jnp = jnp.arange(1, N + 1, dtype=jnp.int64)


# ── For loop ────────────────────────────────────────────────────────────────
def sum_for_loop():
    total = 0
    for x in arr:
        total += x
    return total


# ── Direct sum over existing list ──────────────────────────────────────────
def sum_range():
    return sum(arr)


# ── NumPy ──────────────────────────────────────────────────────────────────
def sum_numpy():
    return np.sum(arr_np)


# ── Numba JIT ──────────────────────────────────────────────────────────────
@njit
def _numba_sum(a):
    total = 0
    for x in a:
        total += x
    return total


def sum_numba():
    return _numba_sum(arr_np)


# ── JAX JIT ────────────────────────────────────────────────────────────────
@jax.jit
def _jax_sum(a):
    return jnp.sum(a)


def sum_jax():
    return _jax_sum(arr_jnp).block_until_ready()


# ── Math formula (Gauss: N*(N+1)/2) ────────────────────────────────────────
def sum_formula():
    return N * (N + 1) // 2


# ── Warm-up JIT compilers (compilation time excluded from benchmark) ───────────
print('Warming up JIT compilers...')
_numba_sum(arr_np)
_jax_sum(arr_jnp).block_until_ready()

# ── Run benchmarks ─────────────────────────────────────────────────────────────
bench('For loop', sum_for_loop)
bench('Direct sum', sum_range)
bench('NumPy', sum_numpy)
bench('Numba JIT', sum_numba)
bench('JAX JIT', sum_jax)
bench('Secret', sum_formula)

print_results(f'Summing 1..{N:,}')
