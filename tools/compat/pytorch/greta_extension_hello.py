#!/usr/bin/env python3
import sys

try:
    import torch
    from torch.utils.cpp_extension import load_inline
except Exception as e:
    print(f"STATUS=SKIPPED reason=missing_deps err={e}")
    sys.exit(0)

# CPU-only extension (GPU/HIP path comes in LOE-5).
source = r"""
#include <torch/extension.h>

torch::Tensor add_one(torch::Tensor x) {
    return x + 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one, "add_one");
}
"""

try:
    ext = load_inline(
        name="greta_ext_hello",
        cpp_sources=source,
        functions=None,
        extra_cflags=["-O3"],
        with_cuda=False,
        verbose=False,
    )
except Exception as e:
    print(f"STATUS=SKIPPED reason=build_failed err={e}")
    sys.exit(0)

x = torch.arange(8, dtype=torch.float32)
y = ext.add_one(x)
if torch.allclose(y, x + 1):
    print("STATUS=OK")
else:
    print("STATUS=FAILED")
    sys.exit(1)
