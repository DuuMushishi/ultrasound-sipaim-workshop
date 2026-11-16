##
import os
import sys
import ctypes
import warnings
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from ttictoc import tic, toc

# Ignore CuPy CUDA_PATH warning
if "CUDA_PATH" not in os.environ:
    os.environ["CUDA_PATH"] = sys.prefix
warnings.filterwarnings("ignore", category=UserWarning, module="cupy._environment")

from scipy.signal import butter

import cupy as cp

# Set path of the CUDA dll file
dll_path = "./das_kernel.dll"
cuda_lib = ctypes.CDLL(dll_path)

print("Loading k-Wave simulation results ", end = '')
tic()

# Load HDF5 data
with h5py.File("sipaim_focused_ultrasound_rf.h5", "r") as f:
    rf_raw = np.array(f["rf_channel_data"], dtype=np.float32)  # [num_tx, num_rx, Nt]
    density_map = f["density_map"][:]
    dt = float(f.attrs["dt"])
    dx = float(f.attrs["dx"])

elapsed = toc()
print(f'Elapsed time: {elapsed:.2f} seconds')

## Setting up environment for CUDA kernel runs

print("Setting up environment for CUDA kernel runs...", end = '')
tic()

num_tx, num_rx, Nt = rf_raw.shape
Nx, Ny = density_map.shape
dy = dx
c0 = 1500.0
fs = 1.0 / dt
f0 = 5e6
t_0 = 1.0e-6

# Reconstruct coordinate grids
elem_width = 4
array_start_y = (Ny - (num_rx * elem_width)) // 2
array_row = 12

rx_y = ((array_start_y + np.arange(num_rx) * elem_width) * dy).astype(np.float32)
rx_x_coord = float(array_row * dx)

margin_trim = int(num_rx * 0.10)
grid_x_m = np.linspace(rx_x_coord, (Nx - 1) * dx, 512)
grid_y_m = np.linspace(rx_y[margin_trim], rx_y[-margin_trim], 512)
grid_y, grid_x = np.meshgrid(grid_y_m, grid_x_m)

grid_x = np.ascontiguousarray(grid_x, dtype=np.float32)
grid_y = np.ascontiguousarray(grid_y, dtype=np.float32)

# Transfer arrays to GPU memory
grid_x_gpu = cp.asarray(grid_x)
grid_y_gpu = cp.asarray(grid_y)
rx_y_gpu = cp.asarray(rx_y)
bmode_gpu = cp.zeros((512, 512), dtype=cp.float32)

# Design Butterworth bandpass filter as Second-Order Sections (SOS)
sos = butter(4, [0.4 * f0 / (fs / 2), 1.6 * f0 / (fs / 2)], btype='band', output='sos')
sos = np.ascontiguousarray(sos, dtype=np.float32)
n_sections = sos.shape[0]  # 4 sections for a 4th-order bandpass

# Transfer raw RF and filter coefficients to GPU
rf_raw_gpu = cp.asarray(rf_raw, dtype=cp.float32)
rf_filtered_gpu = cp.empty_like(rf_raw_gpu)
sos_gpu = cp.asarray(sos)

elapsed = toc()
print(f'Elapsed time: {elapsed:.2f} seconds')

## Running all CUDA kernels

print("Running all CUDA kernels...", end = '')
tic()

# Declare ctypes signature for the filter launcher
cuda_lib.launch_biquad_filtfilt.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# Run zero-phase bandpass filter on GPU
cuda_lib.launch_biquad_filtfilt(
    ctypes.c_void_p(rf_raw_gpu.data.ptr),
    ctypes.c_void_p(rf_filtered_gpu.data.ptr),
    ctypes.c_void_p(sos_gpu.data.ptr),
    ctypes.c_int(num_tx),
    ctypes.c_int(num_rx),
    ctypes.c_int(Nt),
    ctypes.c_int(n_sections)
)

# Declare ctypes signature and launch DAS beamforming kernel
cuda_lib.launch_delay_and_sum.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
]

# Now run the dela-and-sum kernel using the output of the bandpass filter
cuda_lib.launch_delay_and_sum(
    ctypes.c_void_p(rf_filtered_gpu.data.ptr),
    ctypes.c_void_p(grid_x_gpu.data.ptr),
    ctypes.c_void_p(grid_y_gpu.data.ptr),
    ctypes.c_void_p(rx_y_gpu.data.ptr),
    ctypes.c_void_p(bmode_gpu.data.ptr),
    ctypes.c_int(num_tx),
    ctypes.c_int(num_rx),
    ctypes.c_int(Nt),
    ctypes.c_float(dx),
    ctypes.c_float(c0),
    ctypes.c_float(dt),
    ctypes.c_float(t_0),
    ctypes.c_float(rx_x_coord)
)
cp.cuda.Device().synchronize()

elapsed = toc()
print(f'Elapsed time: {elapsed:.2f} seconds')

# Transfer beamformed RF array from GPU (CuPy) to CPU (NumPy)
bmode_rf = cp.asnumpy(bmode_gpu)

# Post-processing (TGC + Hilbert Envelope + Log Compression + Near-Field Mask)
alpha_db_cm = 0.5  # Attenuation coefficient in dB/(cm*MHz)
depth_m = grid_x[:, 0] - rx_x_coord
depth_mm = depth_m * 1e3
lateral_mm = grid_y[0, :] * 1e3

# Apply Time Gain Compensation (TGC)
tgc_gain = np.exp((alpha_db_cm * (f0 / 1e6) * (depth_m * 100)) / 8.686)[:, None]
bmode_tgc = bmode_rf * tgc_gain

# Envelope detection via Hilbert Transform
analytic_signal = hilbert(bmode_tgc, axis=0)
envelope = np.abs(analytic_signal)

# Log Dynamic Range Compression (-35 dB)
envelope_norm = envelope / (np.max(envelope) + 1e-12)
bmode_image = 20 * np.log10(envelope_norm + 1e-12)
bmode_image = np.clip(bmode_image, -35, 0)

# Mask out near-field transducer clutter (< 4.0 mm)
min_depth_mask = depth_mm >= 4.0
bmode_image[:np.argmax(min_depth_mask), :] = -35.0

## 8. Plot reconstructed phantom
plt.figure()
plt.imshow(
    bmode_image,
    cmap="bone",
    extent=[lateral_mm[0], lateral_mm[-1], depth_mm[-1], depth_mm[0]],
    aspect="auto",
    vmin=-40,
    vmax=0
)
plt.title("B-Mode CUDA dll DAS")
plt.xlabel("Width [mm]")
plt.ylabel("Depth [mm]")
plt.colorbar(label="Dynamic Range [dB]")
plt.tight_layout()
plt.axis("image")
plt.ylim([25, 0])
plt.show()