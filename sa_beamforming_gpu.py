import os
import sys
import ctypes
import warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter
from ttictoc import tic, toc

if "CUDA_PATH" not in os.environ:
    os.environ["CUDA_PATH"] = sys.prefix
warnings.filterwarnings("ignore", category=UserWarning, module="cupy._environment")

import cupy as cp

script_dir = Path(__file__).resolve().parent
dll_path = script_dir / "sa_beamformer.dll"
h5_path = script_dir / "sipaim_synthetic_aperture_fmc.h5"

cuda_lib = ctypes.CDLL(str(dll_path))

# --- 1. Load Synthetic Aperture FMC Dataset ---
print("Loading k-Wave Synthetic Aperture FMC dataset... ", end="")
tic()
with h5py.File(h5_path, "r") as hf:
    fmc_raw = np.ascontiguousarray(hf["rf_channel_data"][:], dtype=np.float32)
    element_y_idx = hf["element_y_idx"][:]
    array_x_idx = float(hf.attrs["array_x_idx"])
    dt = float(hf.attrs["dt"])
    dx = float(hf.attrs["dx"])
    dy = float(hf.attrs["dy"])
    c0 = float(hf.attrs["c_bg"])
    f0 = float(hf.attrs["f0"])
    t0 = float(hf.attrs["t0"])
print(f"{toc():.2f} s")

n_tx, n_rx, n_samples = fmc_raw.shape
fs = 1.0 / dt

# --- 2. Setup Grids, Filter Coefficients & GPU Buffers ---
print("Setting up GPU buffers... ", end="")
tic()

elem_y_m_raw = element_y_idx * dy
elem_lat = np.ascontiguousarray(elem_y_m_raw - np.mean(elem_y_m_raw), dtype=np.float32)
elem_dep = np.ascontiguousarray(np.full_like(elem_lat, array_x_idx * dx), dtype=np.float32)

lat_grid = np.ascontiguousarray(np.linspace(-18e-3, 18e-3, 256), dtype=np.float32)
dep_grid = np.ascontiguousarray(np.linspace(6e-3, 26e-3, 512), dtype=np.float32)
n_dep, n_lat = len(dep_grid), len(lat_grid)

sos = butter(4, [0.4 * f0 / (fs / 2), 1.6 * f0 / (fs / 2)], btype="band", output="sos")
sos = np.ascontiguousarray(sos, dtype=np.float32)
n_sections = sos.shape[0]

# Allocate GPU arrays
rf_raw_gpu = cp.asarray(fmc_raw)
rf_filt_gpu = cp.empty_like(rf_raw_gpu)
sos_gpu = cp.asarray(sos)

elem_lat_gpu = cp.asarray(elem_lat)
elem_dep_gpu = cp.asarray(elem_dep)
lat_grid_gpu = cp.asarray(lat_grid)
dep_grid_gpu = cp.asarray(dep_grid)

rx_aperture_gpu = cp.zeros((n_rx, n_dep, n_lat), dtype=cp.float32)
das_rf_gpu = cp.zeros((n_dep, n_lat), dtype=cp.float32)
slsc_gpu = cp.zeros((n_dep, n_lat), dtype=cp.float32)

print(f"{toc():.2f} s")

# --- 3. Declare ctypes Signatures ---
cuda_lib.launch_biquad_filtfilt.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

cuda_lib.launch_sa_delay_and_accumulate.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
]

cuda_lib.launch_das_summation.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
]

cuda_lib.launch_slsc_coherence.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# --- 4. Execute All Four CUDA Kernels ---
f_number = 1.5
max_lag = 12
kernel_z = 8

print("Running CUDA beamforming & SLSC kernels... ", end="")
tic()

# Kernel 1: Zero-phase SOS Bandpass Filter
cuda_lib.launch_biquad_filtfilt(
    ctypes.c_void_p(rf_raw_gpu.data.ptr),
    ctypes.c_void_p(rf_filt_gpu.data.ptr),
    ctypes.c_void_p(sos_gpu.data.ptr),
    ctypes.c_int(n_tx), ctypes.c_int(n_rx), ctypes.c_int(n_samples), ctypes.c_int(n_sections)
)

# Kernel 2: Synthetic Aperture Delay + F/# Apodization + Tx Accumulation
cuda_lib.launch_sa_delay_and_accumulate(
    ctypes.c_void_p(rf_filt_gpu.data.ptr),
    ctypes.c_void_p(elem_lat_gpu.data.ptr),
    ctypes.c_void_p(elem_dep_gpu.data.ptr),
    ctypes.c_void_p(lat_grid_gpu.data.ptr),
    ctypes.c_void_p(dep_grid_gpu.data.ptr),
    ctypes.c_void_p(rx_aperture_gpu.data.ptr),
    ctypes.c_int(n_tx), ctypes.c_int(n_rx), ctypes.c_int(n_samples),
    ctypes.c_int(n_dep), ctypes.c_int(n_lat),
    ctypes.c_float(c0), ctypes.c_float(fs), ctypes.c_float(f0), ctypes.c_float(t0), ctypes.c_float(f_number)
)

# Kernel 3: DAS Aperture Summation & Normalization
cuda_lib.launch_das_summation(
    ctypes.c_void_p(rx_aperture_gpu.data.ptr),
    ctypes.c_void_p(elem_lat_gpu.data.ptr),
    ctypes.c_void_p(elem_dep_gpu.data.ptr),
    ctypes.c_void_p(lat_grid_gpu.data.ptr),
    ctypes.c_void_p(dep_grid_gpu.data.ptr),
    ctypes.c_void_p(das_rf_gpu.data.ptr),
    ctypes.c_int(n_rx), ctypes.c_int(n_dep), ctypes.c_int(n_lat), ctypes.c_float(f_number)
)

# Kernel 4: Short-Lag Spatial Coherence (SLSC)
cuda_lib.launch_slsc_coherence(
    ctypes.c_void_p(rx_aperture_gpu.data.ptr),
    ctypes.c_void_p(slsc_gpu.data.ptr),
    ctypes.c_int(n_rx), ctypes.c_int(n_dep), ctypes.c_int(n_lat),
    ctypes.c_int(max_lag), ctypes.c_int(kernel_z)
)

cp.cuda.Device().synchronize()
print(f"{toc():.4f} s")

# --- 5. Transfer Results & Post-Process DAS Envelope ---
das_rf = cp.asnumpy(das_rf_gpu)
slsc_image = cp.asnumpy(slsc_gpu)

alpha_db_cm = 0.5
depth_from_array = dep_grid - elem_dep[0]
tgc_gain = np.exp((alpha_db_cm * (f0 / 1e6) * (depth_from_array * 100.0)) / 8.686)[:, None]

das_env = np.abs(hilbert(das_rf * tgc_gain, axis=0))
das_bmode = 20.0 * np.log10(das_env / (np.max(das_env) + 1e-12) + 1e-12)

# --- 6. Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
extent = [lat_grid[0] * 1e3, lat_grid[-1] * 1e3, dep_grid[-1] * 1e3, dep_grid[0] * 1e3]

im0 = axes[0].imshow(das_bmode, cmap="gray", aspect="auto", extent=extent, vmin=-40, vmax=0)
axes[0].set_title("CUDA Synthetic Aperture DAS (f/# = 1.5)")
axes[0].set_xlabel("Lateral [mm]")
axes[0].set_ylabel("Depth [mm]")
fig.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].imshow(slsc_image, cmap="bone", aspect="auto", extent=extent, vmin=0, vmax=0.8)
axes[1].set_title("CUDA SLSC Image (f/# = 1.5, M = 12)")
axes[1].set_xlabel("Lateral [mm]")
axes[1].set_ylabel("Depth [mm]")
fig.colorbar(im1, ax=axes[1], label="Coherence")

plt.tight_layout()
plt.show()