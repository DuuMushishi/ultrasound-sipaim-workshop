## Libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import uniform_filter1d

from pathlib import Path
import h5py
import numpy as np

## Loading data
h5_filename = Path(__file__).resolve().parent / "sipaim_synthetic_aperture_fmc.h5"
print(f"Loading: {h5_filename}")

with h5py.File(h5_filename, "r") as hf:
    fmc_raw = hf["rf_channel_data"][:]
    element_y_idx = hf["element_y_idx"][:]
    array_x_idx = float(hf.attrs["array_x_idx"])
    dt = float(hf.attrs["dt"])
    dx = float(hf.attrs["dx"])
    dy = float(hf.attrs["dy"])
    c0 = float(hf.attrs["c_bg"])
    f0 = float(hf.attrs["f0"])
    t0 = float(hf.attrs["t0"])

print(f"Loaded shape: {fmc_raw.shape}, dt: {dt}")

n_tx, n_rx, n_samples = fmc_raw.shape
fs = 1.0 / dt

# Diagnostic: check if all 256 Tx events have non-zero data
active_tx = np.where(np.max(np.abs(fmc_raw), axis=(1, 2)) > 0)[0]
print(f"Active Tx elements: {len(active_tx)} / {n_tx} (range: {active_tx[0]}..{active_tx[-1]})")

## Bandpass filter (5 MHz)
b, a = butter(4, [0.4 * f0 / (fs / 2), 1.6 * f0 / (fs / 2)], btype="band")
fmc_data = np.zeros_like(fmc_raw, dtype=np.float32)
for tx in active_tx:
    fmc_data[tx] = filtfilt(b, a, fmc_raw[tx], axis=-1).astype(np.float32)

# Element coordinates
elem_y_m_raw = element_y_idx * dy
elem_lat = elem_y_m_raw - np.mean(elem_y_m_raw)
elem_dep = np.full_like(elem_lat, array_x_idx * dx)

# Mute direct wave + top-PML reflection ringdown (8 cycles)
t_vec = np.arange(n_samples)[None, None, :] * dt
lat_dist_tx_rx = np.abs(elem_lat[:, None] - elem_lat[None, :])[:, :, None]
t_direct_end = (lat_dist_tx_rx / c0) + t0 + (8.0 / f0)
fmc_data[t_vec < t_direct_end] = 0.0

# Pixel grid
lat_grid = np.linspace(-18e-3, 18e-3, 256)
dep_grid = np.linspace(6e-3, 26e-3, 512)
n_dep, n_lat = len(dep_grid), len(lat_grid)
LAT, DEP = np.meshgrid(lat_grid, dep_grid)

d_lat = LAT[None, :, :] - elem_lat[:, None, None]
d_dep = DEP[None, :, :] - elem_dep[:, None, None]
dist_one_way = np.hypot(d_lat, d_dep).astype(np.float32)

# Dynamic Receive/Transmit F-number Apodization (f/# = 1.5)
# Eliminates diagonal edge waves and fixes left-right illumination imbalance!
f_number = 1.5
aperture_mask = (np.abs(d_lat) <= (d_dep / (2.0 * f_number))).astype(np.float32)

# Low-Memory Tx-Accumulation with F-number Apodization
rx_aperture_data = np.zeros((n_rx, n_dep, n_lat), dtype=np.float32)
rx_indices = np.arange(n_rx)[:, None, None]

for tx in active_tx:
    tau_3d = (dist_one_way[tx][None, :, :] + dist_one_way) / c0 + t0
    sample_idx = np.rint(tau_3d * fs).astype(np.int32)

    valid_mask = (sample_idx >= 0) & (sample_idx < n_samples)
    sample_idx_clipped = np.clip(sample_idx, 0, n_samples - 1)

    tx_delayed = fmc_data[tx, rx_indices, sample_idx_clipped]
    tx_delayed *= (valid_mask * aperture_mask[tx][None, :, :])
    rx_aperture_data += tx_delayed

# Apply receive f/# mask
rx_aperture_data *= aperture_mask

## DAS with Aperture Weight Normalization & TGC
# Normalize by active element count per pixel so edges don't get darker
active_weight_sum = np.sum(aperture_mask, axis=0) + 1e-6
das_rf = np.sum(rx_aperture_data, axis=0) / active_weight_sum

alpha_db_cm = 0.5
depth_from_array = dep_grid - elem_dep[0]
tgc_gain = np.exp((alpha_db_cm * (f0 / 1e6) * (depth_from_array * 100)) / 8.686)[:, None]
das_env = np.abs(hilbert(das_rf * tgc_gain, axis=0))
das_bmode = 20 * np.log10(das_env / (np.max(das_env) + 1e-12) + 1e-12)

## SLSC with F-number Masked Aperture
def compute_slsc(aperture_data, max_lag=12, kernel_z=8):
    n_rx, n_dep, n_lat = aperture_data.shape
    slsc_img = np.zeros((n_dep, n_lat), dtype=np.float32)

    for lag in range(1, max_lag + 1):
        s1 = aperture_data[:-lag]
        s2 = aperture_data[lag:]

        num = uniform_filter1d(s1 * s2, size=kernel_z, axis=1, mode="constant")
        den1 = np.maximum(uniform_filter1d(s1 ** 2, size=kernel_z, axis=1, mode="constant"), 0.0)
        den2 = np.maximum(uniform_filter1d(s2 ** 2, size=kernel_z, axis=1, mode="constant"), 0.0)

        # Only average over element pairs where both elements are inside the f/# cone
        valid_pairs = (den1 > 1e-20) & (den2 > 1e-20)
        rho = np.where(valid_pairs, num / (np.sqrt(den1 * den2) + 1e-20), 0.0)
        pair_count = np.maximum(np.sum(valid_pairs, axis=0), 1)
        slsc_img += np.sum(rho, axis=0) / pair_count

    return slsc_img / max_lag

slsc_image = compute_slsc(rx_aperture_data, max_lag=12, kernel_z=8)

## Plotting
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
extent = [lat_grid[0] * 1e3, lat_grid[-1] * 1e3, dep_grid[-1] * 1e3, dep_grid[0] * 1e3]

im0 = axes[0].imshow(das_bmode, cmap="gray", aspect="auto", extent=extent, vmin=-40, vmax=0)
axes[0].set_title("Synthetic Aperture DAS (f/# = 1.5)")
axes[0].set_xlabel("Lateral [mm]")
axes[0].set_ylabel("Depth [mm]")
fig.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].imshow(slsc_image, cmap="bone", aspect="auto", extent=extent, vmin=0, vmax=0.8)
axes[1].set_title("SLSC Image (f/# = 1.5, M = 12)")
axes[1].set_xlabel("Lateral [mm]")
axes[1].set_ylabel("Depth [mm]")
fig.colorbar(im1, ax=axes[1], label="Coherence")

plt.tight_layout()
plt.show()