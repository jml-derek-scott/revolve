#!/usr/bin/env python3
"""
MPS (Metal Performance Shaders) Device Verification Script for Mac Pro.
Checks PyTorch Metal backend status on macOS with AMD GPU.
"""

import os
import platform
import sys
import time

def check_system_info():
    print("=" * 60)
    print(" 1. SYSTEM INFORMATION")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"macOS Version: {platform.mac_ver()[0]}")
    print(f"Machine / Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")


def check_pytorch_mps():
    print("\n" + "=" * 60)
    print(" 2. PYTORCH & MPS BACKEND STATUS")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("[-] PyTorch is NOT installed in this environment.")
        return False

    print(f"PyTorch Version: {torch.__version__}")
    
    is_built = torch.backends.mps.is_built()
    is_available = torch.backends.mps.is_available()
    
    print(f"torch.backends.mps.is_built():     {is_built}")
    print(f"torch.backends.mps.is_available(): {is_available}")

    if not is_built:
        print("[-] PyTorch was not built with MPS enabled.")
        return False

    if not is_available:
        print("[-] MPS is not available on this macOS system.")
        return False

    print("[+] MPS (Metal) backend is AVAILABLE and READY!")
    return True


def benchmark_tensor_computation():
    print("\n" + "=" * 60)
    print(" 3. TENSOR COMPUTATION ON MPS (AMD GPU)")
    print("=" * 60)

    import torch

    device = torch.device("mps")
    print(f"Selected Device: {device}")

    # Ensure MPS fallback environment is active for unsupported ops
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Allocation test
    size = 2048
    print(f"Allocating two {size}x{size} Float32 matrices on MPS...")
    t0 = time.perf_counter()
    x = torch.randn(size, size, device=device, dtype=torch.float32)
    y = torch.randn(size, size, device=device, dtype=torch.float32)
    torch.mps.synchronize()
    alloc_time = time.perf_counter() - t0
    print(f"[+] Allocation successful ({alloc_time * 1000:.2f} ms).")

    # Warmup
    _ = torch.matmul(x, y)
    torch.mps.synchronize()

    # Benchmark MPS
    iterations = 10
    t0 = time.perf_counter()
    for _ in range(iterations):
        z = torch.matmul(x, y)
    torch.mps.synchronize()
    mps_time = (time.perf_counter() - t0) / iterations
    print(f"[+] MPS Matmul ({size}x{size}): {mps_time * 1000:.2f} ms/iter")

    # Compare with CPU
    x_cpu = torch.randn(size, size, device="cpu", dtype=torch.float32)
    y_cpu = torch.randn(size, size, device="cpu", dtype=torch.float32)
    t0 = time.perf_counter()
    for _ in range(iterations):
        z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = (time.perf_counter() - t0) / iterations
    print(f"[+] CPU Matmul ({size}x{size}): {cpu_time * 1000:.2f} ms/iter")

    speedup = cpu_time / mps_time if mps_time > 0 else 0
    print(f"[+] MPS vs CPU Acceleration: {speedup:.2f}x speedup")

    # Value sanity check
    assert not torch.isnan(z).any(), "NaN values detected in MPS output!"
    print("[+] Numerical sanity check passed (no NaNs).")
    print("\n" + "=" * 60)
    print(" MPS VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    check_system_info()
    if check_pytorch_mps():
        benchmark_tensor_computation()
    else:
        sys.exit(1)
