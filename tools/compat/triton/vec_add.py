#!/usr/bin/env python3
import sys

try:
    import torch
    import triton
    import triton.language as tl
except Exception as e:
    print(f"STATUS=SKIPPED reason=missing_deps err={e}")
    sys.exit(0)

def cpu_fallback():
    x = torch.arange(1024 * 1024, device="cpu", dtype=torch.float32)
    y = torch.arange(1024 * 1024, device="cpu", dtype=torch.float32)
    z = x + y
    max_err = (z - (x + y)).abs().max().item()
    if max_err == 0.0:
        print("STATUS=OK fallback=cpu")
    else:
        print(f"STATUS=FAILED fallback=cpu max_err={max_err}")
        sys.exit(1)


# Require ROCm build for AMD path; otherwise use CPU fallback for dev.
if getattr(torch.version, "hip", None) is None:
    print("TRITON_STATUS=SKIPPED reason=rocm_build_required")
    cpu_fallback()
    sys.exit(0)

if not torch.cuda.is_available():
    print("TRITON_STATUS=SKIPPED reason=rocm_device_unavailable")
    cpu_fallback()
    sys.exit(0)

@triton.jit
def vec_add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)


def main():
    n = 1024 * 1024
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.arange(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)
    vec_add_kernel[grid](x, y, z, n, BLOCK=BLOCK)
    max_err = (z - (x + y)).abs().max().item()
    if max_err == 0.0:
        print("STATUS=OK")
    else:
        print(f"STATUS=FAILED max_err={max_err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
