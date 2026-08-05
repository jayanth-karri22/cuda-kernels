import modal 

image = modal.Image.debian_slim().pip_install("torch", "triton", "numpy", "matplotlib", "pandas")

app = modal.App("fused_softmax", image=image)

@app.function(gpu="T4", timeout=600)
def fused_softmax():
    import torch
    import triton
    import triton.language as tl #type: ignore

    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    def naive_softmax(x):
        x_max = x.max(dim=1)[0]
        z = x - x_max[:, None]

        numerator = torch.exp(z)
        denominator = numerator.sum(dim=1)

        ret = numerator / denominator[:, None]
        return ret 

@app.local_entrypoint()
def main():
    fused_softmax.remote()