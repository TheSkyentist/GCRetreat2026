# GCRetreat2026

Python performance benchmarks comparing naive loops, NumPy, Numba JIT, and JAX/XLA across three computational workloads, plus a free-threading benchmark using CPython 3.14's experimental no-GIL build.

## Benchmarks

| Script | Task | Backends |
|---|---|---|
| `sum.py` | Sum 1–10,000,000 | for loop, `sum()`, NumPy, Numba JIT, JAX JIT, Gauss formula |
| `matrix.py` | 256×256 matrix multiply | for loop (i,j,k), for loop (i,k,j), list comp, NumPy `@`, Numba JIT, JAX JIT |
| `batch.py` | Element-wise `sin·cos + tanh(2x) − √x·e^(−3x)` on 10M floats | for loop, list comp, `map()`, NumPy, Numba JIT (fused), JAX JIT (XLA fused) |
| `parallel.py` | 256×256 matrix multiply | single-threaded, multiprocessing, free-threaded (CPython 3.14 no-GIL) |

## Requirements

[pixi](https://pixi.sh) — manages all environments and dependencies.

> **Note:** Windows is not supported

## Running

```bash
pixi run all
```

This runs all four benchmarks in sequence. The `parallel` benchmark runs under a free-threaded CPython 3.14 build; the remaining three run under standard CPython 3.14 with JAX and Numba.

Individual benchmarks:

```bash
pixi run sum
pixi run matrix
pixi run batch
pixi run parallel
```
