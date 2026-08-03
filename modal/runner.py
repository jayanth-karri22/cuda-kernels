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

    ms = triton.testing.do_bench(lambda: torch.add(x, y))
    print(f"torch.add: {ms:.4f} ms")


@app.local_entrypoint()
def cli():
    run.remote()
