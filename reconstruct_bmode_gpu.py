import os
import sys
import warnings
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

# Set CUDA_PATH to active Conda environment root
if "CUDA_PATH" not in os.environ:
    os.environ["CUDA_PATH"] = sys.prefix

# Suppress cosmetic CuPy environment path warning
warnings.filterwarnings("ignore", category=UserWarning, module="cupy._environment")

import cupy as cp
from ttictoc import tic, toc

# %%==============================================================================
# 1. LOAD DATA
# ==============================================================================
h5_filename = Path("sipaim_focused_ultrasound_rf.h5")

print(f"Loading RF data from {h5_filename}...", end='')
tic()
with h5py.File(h5_filename, "r") as hf:
    rf_data = hf["rf_channel_data"][:]  # (256, 256, Nt)
    dt = float(hf.attrs["dt"])
    dx = float(hf.attrs["dx"])
    density_map = hf["density_map"][:]

num_tx, num_rx, Nt = rf_data.shape
dy = dx
c0 = 1500.0
fs = 1.0 / dt
f0 = 5e6  # 5 MHz center frequency
t_0 = 1.0e-6  # 1.0 us pulse excitation offset delay

Nx, Ny = density_map.shape
elapsed = toc()
print(f'Elapsed time: {elapsed:.2f} seconds')

# %%==============================================================================
# 2. RF SIGNAL BANDPASS FILTERING (5 MHz Center)
# ==============================================================================
print("Applying 5 MHz Bandpass Filter to raw RF channel data...", end='')
tic()
b, a = butter(4, [0.4 * f0 / (fs / 2), 1.6 * f0 / (fs / 2)], btype='band')

rf_filtered = np.zeros_like(rf_data)
for tx in range(num_tx):
    rf_filtered[tx] = filtfilt(b, a, rf_data[tx], axis=-1)

# %%==============================================================================
# 3. PHYSICAL GRID SETUP
# ==============================================================================
elem_width = 4
array_start_y = (Ny - (num_rx * elem_width)) // 2
array_row = 12

rx_y_coords = (array_start_y + np.arange(num_rx) * elem_width) * dy
rx_x_coord = array_row * dx

# Crop reconstruction grid to active aperture zone
margin_trim = int(num_rx * 0.10)  # Trim 10% edges
grid_x_m = np.linspace(rx_x_coord, (Nx - 1) * dx, 512)
grid_y_m = np.linspace(
    rx_y_coords[margin_trim],
    rx_y_coords[-margin_trim],
    512
)

GRID_Y, GRID_X = np.meshgrid(grid_y_m, grid_x_m)

# Crop ground truth density map to match reconstructed lateral zone
crop_start_y = array_start_y + margin_trim * elem_width
crop_end_y = array_start_y + (num_rx - margin_trim) * elem_width
density_cropped = density_map[:, crop_start_y:crop_end_y]

# Receive apodization weights (Uniform/Boxcar)
apod_weights = cp.ones(num_rx, dtype=cp.float32)
elapsed = toc()
print(f'Elapsed time: {elapsed:.2f} seconds')

# %%==============================================================================
# 4. GPU-ACCELERATED DELAY-AND-SUM (DAS) BEAMFORMING (Linear Interpolation)
# ==============================================================================
print("Running GPU Delay-and-Sum Beamforming via CuPy...", end='')
tic()

# Transfer arrays to GPU memory
rf_filtered_gpu = cp.asarray(rf_filtered, dtype=cp.float32)  # (num_tx, num_rx, Nt)
grid_x_gpu = cp.asarray(GRID_X, dtype=cp.float32)
grid_y_gpu = cp.asarray(GRID_Y, dtype=cp.float32)
rx_y_coords_gpu = cp.asarray(rx_y_coords, dtype=cp.float32)

bmode_rf_gpu = cp.zeros(grid_x_gpu.shape, dtype=cp.float32)

# Fully parallelized across pixels per TX
for tx_idx in range(num_tx):
    tx_y = rx_y_coords_gpu[tx_idx]
    dist_tx = cp.sqrt((grid_x_gpu - rx_x_coord) ** 2 + (grid_y_gpu - tx_y) ** 2)

    # Broadcast across all receive elements (num_rx, N_depth, N_lateral)
    dist_rx = cp.sqrt((grid_x_gpu[None, :, :] - rx_x_coord) ** 2 +
                      (grid_y_gpu[None, :, :] - rx_y_coords_gpu[:, None, None]) ** 2)

    two_way_dist = dist_rx + dist_tx[None, :, :]
    tof = (two_way_dist / c0) + t_0

    # Sub-sample fractional indices
    sample_frac = tof / dt
    idx_0 = cp.floor(sample_frac).astype(cp.int32)
    idx_1 = idx_0 + 1

    # Linear interpolation weights
    weight_1 = (sample_frac - idx_0).astype(cp.float32)
    weight_0 = 1.0 - weight_1

    # Valid mask ensuring both idx_0 and idx_1 stay within [0, Nt - 1]
    valid_mask = (idx_0 >= 0) & (idx_1 < Nt)

    for rx_i in range(num_rx):
        mask = valid_mask[rx_i]

        # Clip indices to prevent VRAM out-of-bounds access
        i0 = cp.clip(idx_0[rx_i, mask], 0, Nt - 1)
        i1 = cp.clip(idx_1[rx_i, mask], 0, Nt - 1)

        w0 = weight_0[rx_i, mask]
        w1 = weight_1[rx_i, mask]

        # Interpolate RF signal values directly on GPU
        # Interpolate RF signal values directly on GPU
        val_0 = rf_filtered_gpu[tx_idx, rx_i, i0]
        val_1 = rf_filtered_gpu[tx_idx, rx_i, i1]
        interp_signal = w0 * val_0 + w1 * val_1

        # Accumulate into B-mode image
        bmode_rf_gpu[mask] += apod_weights[rx_i] * interp_signal

# Transfer reconstructed RF tensor back to CPU
bmode_rf = cp.asnumpy(bmode_rf_gpu)
elapsed = toc()
print(f'Elapsed time: {elapsed:.2f} seconds')

# %%==============================================================================
# 5. POST-PROCESSING (TGC, Hilbert Envelope, Log Compression)
# ==============================================================================
print("Applying post-processing (TGC + Hilbert + Log Compression)...", end='')
tic()

alpha_db_cm = 0.5  # Attenuation coefficient in dB/(cm*MHz)
depth_vector_gpu = grid_x_gpu[:, 0] - rx_x_coord
tgc_gain_gpu = cp.exp((alpha_db_cm * (f0 / 1e6) * (depth_vector_gpu * 100)) / 8.686)[:, None]
bmode_tgc_gpu = bmode_rf_gpu * tgc_gain_gpu

# 2. GPU Hilbert Transform via FFT (Fixed 2D broadcasting)
N_depth = bmode_tgc_gpu.shape[0]
freqs = cp.fft.fftfreq(N_depth)[:, None]  # Added [:, None] for proper column broadcasting
h_step = cp.where(freqs > 0, 2.0, cp.where(freqs == 0, 1.0, 0.0))

analytic_signal_gpu = cp.fft.ifft(cp.fft.fft(bmode_tgc_gpu, axis=0) * h_step, axis=0)
envelope_gpu = cp.abs(analytic_signal_gpu)

# 3. GPU Dynamic Range Compression (-35 dB)
bmode_log_gpu = 20 * cp.log10(envelope_gpu / cp.max(envelope_gpu) + 1e-12)
bmode_log_gpu = cp.clip(bmode_log_gpu, -35, 0)

# Final transfer to CPU for Matplotlib visualization
bmode_log = cp.asnumpy(bmode_log_gpu)

elapsed = toc()
print(f'Elapsed time: {elapsed:.2f} seconds')

# %%==============================================================================
# 6. PLOTTING RESULTS
# ==============================================================================
depth_mm = (grid_x_m - rx_x_coord) * 1e3
lateral_mm = grid_y_m * 1e3

# Near-field clutter mask
min_depth_mask = depth_mm >= 4.0
bmode_log[:np.argmax(min_depth_mask), :] = -35.0

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Ground Truth Map (Fixed extent bounds to match reconstructed depth/lateral coordinates)
im0 = axes[0].imshow(
    density_cropped,
    extent=[lateral_mm[0], lateral_mm[-1], depth_mm[-1], depth_mm[0]],
    cmap='bone',
    aspect='auto'
)
axes[0].set_title("Ground Truth Phantom Map")
axes[0].set_ylabel("Depth (mm)")

# Reconstructed B-Mode Image
im1 = axes[1].imshow(
    bmode_log,
    extent=[lateral_mm[0], lateral_mm[-1], depth_mm[-1], depth_mm[0]],
    cmap='gray',
    aspect='auto',
    vmin=-35,
    vmax=0
)
axes[1].set_title("GPU-Accelerated B-Mode Image (Filtered RF + Uniform Sound Speed)")
axes[1].set_xlabel("Lateral Position (mm)")
axes[1].set_ylabel("Depth (mm)")

cbar = fig.colorbar(im1, ax=axes[1], orientation='vertical')
cbar.set_label('Dynamic Range (dB)')

plt.tight_layout()
plt.show()