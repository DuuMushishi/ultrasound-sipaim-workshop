import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
import cupy as cp

# %%==============================================================================
# 1. LOAD DATA
# ==============================================================================
h5_filename = Path("sipaim_focused_ultrasound_rf.h5")

print(f"Loading RF data from {h5_filename}...")
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

# %%==============================================================================
# 2. RF SIGNAL BANDPASS FILTERING (5 MHz Center)
# ==============================================================================
print("Applying 5 MHz Bandpass Filter to raw RF channel data...")
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

# %%==============================================================================
# 4. GPU-ACCELERATED DELAY-AND-SUM (DAS) BEAMFORMING
# ==============================================================================
print("Running GPU Delay-and-Sum Beamforming via CuPy...")

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

    sample_indices = cp.round(tof / dt).astype(cp.int32)
    valid_mask = (sample_indices >= 0) & (sample_indices < Nt)

    for rx_i in range(num_rx):
        mask = valid_mask[rx_i]
        bmode_rf_gpu[mask] += apod_weights[rx_i] * rf_filtered_gpu[tx_idx, rx_i, sample_indices[rx_i, mask]]

# Transfer reconstructed RF tensor back to CPU
bmode_rf = cp.asnumpy(bmode_rf_gpu)

# %%==============================================================================
# 5. POST-PROCESSING (TGC, Hilbert Envelope, Log Compression)
# ==============================================================================
print("Applying post-processing (TGC + Hilbert + Log Compression)...")

# Time Gain Compensation (TGC)
depth_vector = grid_x_m - rx_x_coord
alpha_db_cm = 0.5  # Attenuation coefficient in dB/(cm*MHz)
tgc_gain = np.exp((alpha_db_cm * (f0 / 1e6) * (depth_vector * 100)) / 8.686)[:, None]
bmode_tgc = bmode_rf * tgc_gain

# Envelope Extraction via Hilbert Transform
envelope = np.abs(hilbert(bmode_tgc, axis=0))

# Dynamic Range Log Compression (-35 dB)
bmode_log = 20 * np.log10(envelope / np.max(envelope) + 1e-12)
bmode_log = np.clip(bmode_log, -35, 0)

# %%==============================================================================
# 6. PLOTTING RESULTS
# ==============================================================================
depth_mm = (grid_x_m - rx_x_coord) * 1e3
lateral_mm = grid_y_m * 1e3

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Ground Truth Map
im0 = axes[0].imshow(
    density_cropped,
    extent=[lateral_mm[0], lateral_mm[-1], (Nx - 1) * dx * 1e3, 0],
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