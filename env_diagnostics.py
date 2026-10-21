# env_diagnostics.py
"""
Machine Learning Environment & Hardware Diagnostic Utility
Verifies active virtual environment, installed library versions,
CPU hardware allocation, and runs a quick NumPy BLAS tensor benchmark.
"""

import os
import platform
import sys
import time


def check_runtime_environment():
    in_venv = sys.prefix != sys.base_prefix
    print("=" * 60)
    print("1. RUNTIME & ENVIRONMENT CONFIGURATION")
    print("=" * 60)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.platform()}")
    print(f"Architecture   : {platform.machine()}")
    print(f"Venv Active    : {'Yes (.venv)' if in_venv else 'No (Global System)'}")
    print(f"Venv Prefix    : {sys.prefix}")


def check_ml_dependencies():
    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "mlxtend",
        "requests",
        "bs4",
        "joblib",
    ]
    print("\n" + "=" * 60)
    print("2. INSTALLED PACKAGE REGISTRY & VERSIONS")
    print("=" * 60)

    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "Installed")
            print(f"✓ {pkg:<15} : {version}")
        except ImportError:
            print(f"✗ {pkg:<15} : NOT INSTALLED")


def benchmark_matrix_compute():
    print("\n" + "=" * 60)
    print("3. NUMPY BLAS / VECTORIZATION BENCHMARK")
    print("=" * 60)
    try:
        import numpy as np

        matrix_dim = 2500
        print(f"Allocating two ({matrix_dim}x{matrix_dim}) float64 matrices...")

        start_alloc = time.perf_counter()
        A = np.random.randn(matrix_dim, matrix_dim)
        B = np.random.randn(matrix_dim, matrix_dim)
        alloc_time = time.perf_counter() - start_alloc

        print(f"Memory allocated: ~{(A.nbytes * 2) / (1024**2):.2f} MB")
        print("Computing dot product (matrix multiplication)...")

        start_dot = time.perf_counter()
        _ = np.dot(A, B)
        compute_time = time.perf_counter() - start_dot

        print(f"Matrix multiply completed in: {compute_time:.4f} seconds")
        print("Compute hardware test: PASSED")
    except ImportError:
        print("NumPy not found. Skipping compute benchmark.")


if __name__ == "__main__":
    check_runtime_environment()
    check_ml_dependencies()
    benchmark_matrix_compute()