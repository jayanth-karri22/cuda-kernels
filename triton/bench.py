import time
import torch


def _time(fn, args, warmup, reps):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1e3)  # ms
    return sorted(times)


def bench(fn, ref, args, warmup=10, reps=50, atol=1e-5, rtol=1e-5):
    try:
        torch.testing.assert_close(fn(*args), ref(*args), atol=atol, rtol=rtol)
        verdict = "PASS"
    except AssertionError as e:
        print(e)
        verdict = "FAIL"

    t, tr = _time(fn, args, warmup, reps), _time(ref, args, warmup, reps)
    med, ref_med = t[len(t) // 2], tr[len(tr) // 2]
    p20, p80 = t[int(0.2 * len(t))], t[int(0.8 * len(t))]

    print(f"{verdict}  median {med:.4f} ms   p20 {p20:.4f}  p80 {p80:.4f}")
    print(f"      ref    {ref_med:.4f} ms   speedup {ref_med / med:.2f}x")
    return verdict == "PASS", med


def my_add(x, y):
    return x + y


def main():
    torch.manual_seed(0)
    x, y = torch.randn(1 << 20), torch.randn(1 << 20)
    bench(my_add, torch.add, (x, y))


if __name__ == "__main__":
    main()