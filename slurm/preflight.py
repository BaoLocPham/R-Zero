"""Run only in a GPU allocation; verify the backend's actual hardware contract."""
import importlib
import os
import torch


def main():
    for module in ("vllm", "flash_attn", "ray", "datasets", "math_verify", "stopit", "verl"):
        importlib.import_module(module)
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("No GPUs visible in this allocation")
    for i in range(count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            raise RuntimeError("R-Zero BF16/FlashAttention 2 training requires Ampere or newer; V100 is unsupported")
        x = torch.ones((32, 32), device=f"cuda:{i}", dtype=torch.bfloat16)
        assert torch.isfinite(x @ x).all()
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
    print("PASS: imports and BF16 CUDA kernels; visible GPUs:", count)
    print("Storage:", os.environ["STORAGE_PATH"])


if __name__ == "__main__":
    main()
