import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from bench_utils import bench, print_results

# ── Config ─────────────────────────────────────────────────────────────────────
N = 256
NUM_WORKERS = os.cpu_count() or 4

rng = random.Random(42)
list_A = [[rng.random() for _ in range(N)] for _ in range(N)]
list_B = [[rng.random() for _ in range(N)] for _ in range(N)]


# ── Baseline: single-threaded i,k,j ────────────────────────────────────────
def matmul_single():
    C = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for k in range(N):
            aik = list_A[i][k]
            for j in range(N):
                C[i][j] += aik * list_B[k][j]
    return C


# ── Multiprocessing: rows split across worker processes ────────────────────
# Each process receives its slice of A and all of B, computes its output rows,
# and returns them. The main process assembles the full result.
# Overhead: pickling A/B chunks + result chunks across process boundaries.
def _mp_worker(args):
    A_rows, B, row_start, n = args
    chunk = [[0.0] * n for _ in range(len(A_rows))]
    for i, row_a in enumerate(A_rows):
        for k in range(n):
            aik = row_a[k]
            for j in range(n):
                chunk[i][j] += aik * B[k][j]
    return row_start, chunk


def matmul_multiprocessing():
    chunk_size = (N + NUM_WORKERS - 1) // NUM_WORKERS
    tasks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        tasks.append((list_A[start:end], list_B, start, N))

    C = [[0.0] * N for _ in range(N)]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for row_start, chunk in pool.map(_mp_worker, tasks):
            for i, row in enumerate(chunk):
                C[row_start + i] = row
    return C


# ── Free-threading: rows split across threads, shared list-of-lists ────────
# Threads write concurrently to disjoint row ranges of a pre-allocated C.
# No locks needed: each thread exclusively owns its index range.
# With standard CPython (GIL) these threads would be serialized; with the
# free-threading build they run in parallel on separate CPU cores.
#
# Key differences from a flat array.array:
#   - List element access returns a stored PyObject pointer — no boxing/unboxing
#     of raw C doubles on every read/write, so the inner loop is cheaper.
#   - Hoisting C[i] and B[k] row references eliminates repeated 2-D lookups.
def _thread_worker(A, B, C, row_start, row_end, n):
    for i in range(row_start, row_end):
        c_row = C[i]
        a_row = A[i]
        for k in range(n):
            aik = a_row[k]
            b_row = B[k]
            for j in range(n):
                c_row[j] += aik * b_row[j]


def matmul_freethreaded():
    C = [[0.0] * N for _ in range(N)]
    chunk_size = (N + NUM_WORKERS - 1) // NUM_WORKERS

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            futures.append(
                pool.submit(_thread_worker, list_A, list_B, C, start, end, N)
            )
        for f in futures:
            f.result()

    return C


# ── Run benchmarks ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    bench('Single-threaded (i,k,j)', matmul_single)
    bench('Multiprocessing (row chunks)', matmul_multiprocessing)
    bench('Free-threading (shared array)', matmul_freethreaded)
    print_results(f'Matrix multiply  {N}×{N}  — {NUM_WORKERS} workers')
