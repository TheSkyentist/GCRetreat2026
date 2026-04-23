import time

results: list[tuple[str, float]] = []


def fmt_time(seconds: float) -> str:
    if seconds >= 1:
        return f'{seconds:.3f} s '
    if seconds >= 1e-3:
        return f'{seconds * 1e3:.3f} ms'
    if seconds >= 1e-6:
        return f'{seconds * 1e6:.3f} µs'
    return f'{seconds * 1e9:.3f} ns'


def bench(label: str, fn):
    t = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t
    results.append((label, elapsed))


def print_results(header: str):
    print(f'\n{header}\n' + '─' * max(60, len(header)))
    baseline_time = None
    for label, elapsed in results:
        time_str = fmt_time(elapsed)
        if baseline_time is None:
            baseline_time = elapsed
            mult_str = '(baseline)'
        else:
            mult = baseline_time / elapsed
            mult_str = (
                f'{mult:>6.1f}x faster' if mult >= 1 else f'{1 / mult:>6.1f}x slower'
            )
        print(f'{label:<35} {time_str:>12}  {mult_str}')
