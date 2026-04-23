import jax
import jax.numpy as jnp
import numpy as np
from numba import njit

from bench_utils import bench, print_results

jax.config.update('jax_enable_x64', True)

# ── Config ─────────────────────────────────────────────────────────────────────
N = 256  # Matrix dimension (N×N)
# Note: Python triple-loops run N³ = 16.7 M iterations — expect a few seconds.
#       NumPy / Numba / JAX complete the same multiply in microseconds.

rng = np.random.default_rng(42)
arr_np_A = rng.random((N, N))
arr_np_B = rng.random((N, N))
list_A = arr_np_A.tolist()
list_B = arr_np_B.tolist()
arr_jnp_A = jnp.array(arr_np_A)
arr_jnp_B = jnp.array(arr_np_B)


# ── Triple for-loop (naive i,j,k) ──────────────────────────────────────────
def matmul_ijk():
    C = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += list_A[i][k] * list_B[k][j]
    return C


# ── Triple for-loop (cache-friendly i,k,j) ─────────────────────────────────
# Hoisting list_A[i][k] out of the inner loop avoids a repeated 2-D list lookup
# and improves temporal locality — the same trick C compilers apply automatically.
def matmul_ikj():
    C = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for k in range(N):
            aik = list_A[i][k]
            for j in range(N):
                C[i][j] += aik * list_B[k][j]
    return C


# ── List comprehension + zip inner products ─────────────────────────────────
# Transpose B once so each (row, col) pair is a simple zip; delegate the
# inner dot-product to the built-in sum() rather than an explicit loop.
def matmul_listcomp():
    B_T = list(zip(*list_B))
    return [[sum(a * b for a, b in zip(row, col)) for col in B_T] for row in list_A]


# ── NumPy @ operator ────────────────────────────────────────────────────────
# Delegates to an optimised BLAS DGEMM routine (OpenBLAS / MKL / Accelerate).
def matmul_numpy():
    return arr_np_A @ arr_np_B


# ── Numba JIT ──────────────────────────────────────────────────────────────
# Compiled to native machine code; uses the cache-friendly i,k,j loop order.
# Numba does NOT call BLAS — the gain over raw Python comes purely from
# eliminating interpreter overhead and enabling CPU SIMD auto-vectorisation.
@njit
def _numba_matmul(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(n):
            aik = A[i, k]
            for j in range(n):
                C[i, j] += aik * B[k, j]
    return C


def matmul_numba():
    return _numba_matmul(arr_np_A, arr_np_B)


# ── JAX JIT ────────────────────────────────────────────────────────────────
# XLA compiles the @ to an optimised kernel (BLAS on CPU, cuBLAS on GPU).
# block_until_ready() ensures asynchronous dispatch is fully complete before
# the timer stops.
@jax.jit
def _jax_matmul(A, B):
    return A @ B


def matmul_jax():
    return _jax_matmul(arr_jnp_A, arr_jnp_B).block_until_ready()


# ── Warm-up JIT compilers (compilation time excluded from benchmark) ───────────
print('Warming up JIT compilers...')
_numba_matmul(arr_np_A, arr_np_B)
_jax_matmul(arr_jnp_A, arr_jnp_B).block_until_ready()

# ── Run benchmarks ─────────────────────────────────────────────────────────────
bench('For-loop (i,j,k)', matmul_ijk)
bench('For-loop (i,k,j cached)', matmul_ikj)
bench('List comp + zip', matmul_listcomp)
bench('NumPy @', matmul_numpy)
bench('Numba JIT', matmul_numba)
bench('JAX JIT', matmul_jax)

print_results(f'Matrix multiply  {N}×{N}  (float64)')
