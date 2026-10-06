from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, windows, butter, filtfilt

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
h5_filename = Path("sipaim_focused_ultrasound_rf.h5")

print(f"Loading RF data from {h5_filename}...")
with h5py.File(h5_filename, "r") as hf:
    rf_data = hf["rf_channel_data"][:]  # (256, 256, Nt)
    dt = hf.attrs["dt"]
    dx = hf.attrs["dx"]
    density_map = hf["density_map"][:]

num_tx, num_rx, Nt = rf_data.shape
dy = dx
c0 = 1500.0
fs = 1.0 / dt
f0 = 5e6  # 5 MHz center frequency

Nx, Ny = density_map.shape

# ==============================================================================
# 2. RF SIGNAL BANDPASS FILTERING (5 MHz Center)
# ==============================================================================
print("Applying 5 MHz Bandpass Filter to raw RF channel data...")
b, a = butter(4, [0.4 * f0 / (fs / 2), 1.6 * f0 / (fs / 2)], btype='band')

rf_filtered = np.zeros_like(rf_data)
for tx in range(num_tx):
    rf_filtered[tx] = filtfilt(b, a, rf_data[tx], axis=-1)

# ==============================================================================
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

# ==============================================================================
# 4. DELAY-AND-SUM BEAMFORMER (With Correct Focused TOF & Pulse Offset)
# ==============================================================================
print("Performing Clean Delay-and-Sum Beamforming...")

bmode_rf = np.zeros(GRID_X.shape, dtype=np.float32)

# Precompute receive distances from all hydrophones to pixel grid
rx_x_diff_sq = (GRID_X - rx_x_coord) ** 2
dist_rx_all = np.array([np.sqrt(rx_x_diff_sq + (GRID_Y - ry) ** 2) for ry in rx_y_coords])
apod_weights = windows.hann(num_rx)

# Pulse offset delay matching simulation
t_0 = 1e-6

for tx_idx in range(num_tx):
    tx_y = rx_y_coords[tx_idx]
    dist_tx = np.sqrt((GRID_X - rx_x_coord) ** 2 + (GRID_Y - tx_y) ** 2)

    # Combined Two-Way Time of Flight
    two_way_dist = dist_rx_all + dist_tx[np.newaxis, :, :]
    tof = (two_way_dist / c0) + t_0

    sample_indices = np.round(tof / dt).astype(int)
    tx_rf_matrix = rf_filtered[tx_idx, :, :]

    valid_mask = (sample_indices >= 0) & (sample_indices < Nt)

    for rx_i in range(num_rx):
        mask = valid_mask[rx_i]
        w = apod_weights[rx_i]
        bmode_rf[mask] += w * tx_rf_matrix[rx_i, sample_indices[rx_i, mask]]

# Time Gain Compensation (TGC)
depth_vector_m = grid_x_m - rx_x_coord
tgc_gain = np.exp(20.0 * depth_vector_m)[:, np.newaxis]
bmode_rf *= tgc_gain

# ==============================================================================
# 5. ENVELOPE DETECTION & DISPLAY SCALING
# ==============================================================================
print("Applying Hilbert Envelope Detection and Log Compression...")

analytic_signal = hilbert(bmode_rf, axis=0)
envelope = np.abs(analytic_signal)

# Mask top transducer element artifacts (first 3 mm)
min_depth_idx = int(0.003 / (grid_x_m[1] - grid_x_m[0]))
max_val = np.max(envelope[min_depth_idx:, :])

# Normalize envelope
envelope_norm = envelope / max_val
bmode_dB = 20 * np.log10(envelope_norm + 1e-6)
bmode_dB = np.clip(bmode_dB, -35, 0)

# Display Subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Ground Truth Phantom
axes[0].imshow(
    density_cropped,
    cmap="bone",
    extent=[grid_y_m[0] * 1e3, grid_y_m[-1] * 1e3, grid_x_m[-1] * 1e3, grid_x_m[0] * 1e3],
    aspect="auto"
)
axes[0].set_title("Ground Truth Phantom Map")
axes[0].set_ylabel("Depth (mm)")

# Reconstructed B-Mode Image
im = axes[1].imshow(
    bmode_dB,
    cmap="gray",
    extent=[grid_y_m[0] * 1e3, grid_y_m[-1] * 1e3, grid_x_m[-1] * 1e3, grid_x_m[0] * 1e3],
    aspect="auto",
    vmin=-35,
    vmax=0
)
axes[1].set_title("Reconstructed B-Mode Image (Filtered RF + Uniform Sound Speed)")
axes[1].set_xlabel("Lateral Position (mm)")
axes[1].set_ylabel("Depth (mm)")
fig.colorbar(im, ax=axes[1], label="Dynamic Range (dB)")

plt.tight_layout()
plt.show()