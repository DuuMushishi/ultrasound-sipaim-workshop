import os
import sys
import subprocess
from pathlib import Path

# Get current script directory to resolve absolute paths
script_dir = Path(__file__).parent.resolve()
cu_file = script_dir / "das_kernel.cu"
dll_file = script_dir / "das_kernel.dll"

# Locate nvcc executable inside Conda environment or CUDA Toolkit
conda_bin = Path(sys.prefix) / "bin" / "nvcc.exe"
conda_scripts = Path(sys.prefix) / "Scripts" / "nvcc.exe"

if conda_bin.exists():
    nvcc_bin = str(conda_bin)
elif conda_scripts.exists():
    nvcc_bin = str(conda_scripts)
else:
    nvcc_bin = "nvcc"  # Fallback to PATH

if not dll_file.exists():
    print(f"Compiling CUDA C++ kernel to DLL using: {nvcc_bin}...")

    # Build command string with absolute file paths enclosed in quotes
    compile_cmd = f'"{nvcc_bin}" -O3 -shared "{cu_file}" -o "{dll_file}"'

    # Use shell=True to inherit full execution environment
    subprocess.run(compile_cmd, shell=True, check=True)
    print("Compilation successful! das_kernel.dll created.")