# SIPAIM 2026: Ultrasound Simulation & Reconstruction Workshop

Welcome to the hands-on workshop repository for **SIPAIM 2026**. This project demonstrates forward acoustic ultrasound simulation using k-Wave and real-time GPU-accelerated B-mode Delay-and-Sum (DAS) beamforming in Python.

---

## 🔗 Software & Prerequisites

Before setting up the repository, make sure you have the following software installed:

### Required Software & IDEs
* **[Anaconda / Miniconda](https://docs.anaconda.com/miniconda/)** – Recommended environment and package manager.
* **[Git](https://git-scm.com/downloads)** – Version control system to clone this repository.
* **[PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/)** – Recommended IDE for Python development.
* **[NVIDIA CUDA Toolkit & Drivers](https://developer.nvidia.com/cuda-downloads)** – Required for running GPU-accelerated reconstruction scripts.

### Key Libraries & Official Documentation
* **[k-Wave Python Documentation](https://k-wave-python.readthedocs.io/)** – Open-source acoustic wave propagation toolbox ported to Python.
* **[CuPy Documentation](https://docs.cupy.dev/en/stable/)** – NumPy-compatible array operations for CUDA GPU acceleration.
* **[Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)** – JIT compiler to compile pure Python functions into custom CUDA kernels.
* **[h5py Documentation](https://docs.h5py.org/en/stable/)** – Python package to store and load large HDF5 acoustic simulation datasets.

---

## 🛠️ Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ultrasound-sipaim-workshop.git
cd ultrasound-sipaim-workshop
```

### 2. Create the Conda Environment
Ensure you have Anaconda or Miniconda installed, then run:

```bash
conda env create -f environment.yml
conda activate sipaim2026
```
## 🚀 Execution Workflow

### 1. Acoustic Simulation (k-Wave)
Generates synthetic RF raw data from the phantom logo:

```bash
python heavySimulationkWaveGPU.py
```

### 2. CPU B-Mode Reconstruction
Performs Delay-and-Sum beamforming on CPU:

```bash
python reconstruct_bmode.py
```
### 3. GPU-Accelerated Reconstruction
Runs real-time beamforming using GPU matrix operations/CUDA kernels:

```bash
python reconstruct_bmode_gpu.py
```