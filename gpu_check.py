"""Check that PyTorch can actually run on this server's GPU.

Usage (inside the venv):  python gpu_check.py
"""
import importlib.metadata as meta

import torch

print("torch       :", meta.version("torch"), "| built for CUDA", torch.version.cuda)
print("transformers:", meta.version("transformers"))
print("CUDA usable :", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("RESULT      : FAIL - PyTorch cannot see the GPU (CPU-only build or driver problem)")
    raise SystemExit(1)

major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
arch_list = torch.cuda.get_arch_list()
print("GPU         :", torch.cuda.get_device_name(0), f"({arch})")
print("arch list   :", arch_list)

try:
    x = torch.randn(512, 512, device="cuda")
    value = (x @ x).sum().item()
    print("kernel test :", "OK", round(value, 2))
except Exception as e:
    print("kernel test :", "FAILED -", e)
    print("RESULT      : FAIL - this PyTorch build has no kernels for", arch)
    print("FIX         : pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    raise SystemExit(1)

if arch not in arch_list:
    print("RESULT      : WARNING - kernels ran, but", arch, "is not in the arch list; upgrade to a cu128 build")
else:
    print("RESULT      : OK - PyTorch runs on this GPU")
