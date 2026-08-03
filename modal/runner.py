import modal

image = modal.Image.debian_slim().pip_install("torch", "triton")

app = modal.App("runner", image=image)


@app.function(gpu="T4", timeout=6000)
def run():
    import torch
    import triton

    torch.manual_seed(0)
    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn(1 << 20, device="cuda")

    med, p20, p80 = triton.testing.do_bench(lambda: torch.add(x, y),quantiles=[0.5, 0.2, 0.8])
    print(f"torch.add: {med:.4f} ms  (p20 {p20:.4f}, p80 {p80:.4f})")


@app.local_entrypoint()
def cli():
    run.remote()
