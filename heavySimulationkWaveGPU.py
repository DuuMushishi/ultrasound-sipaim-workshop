import os
import sys

# Clear all user-defined global variables
for name in list(globals().keys()):
    if not name.startswith('_') and name not in sys.modules:
        del globals()[name]

def clc():
    os.system('cls' if os.name == 'nt' else 'clear')

clc()

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

plt.close('all')

# %%==============================================================================
# 1. GRID & TRANSDUCER ARRAY GEOMETRY CONFIGURATION
# ==============================================================================
Nx, Ny = 1024, 2048
dx = dy = 4e-5  # 40 um spatial step (~4.1 cm x 8.2 cm)
kgrid = kWaveGrid([Nx, Ny], [dx, dy])

f0 = 5e6  # 5 MHz center frequency
dt = 5e-9
Nt = 11000  # Extended time steps to capture full 40mm depth
kgrid.setTime(Nt, dt)
t = kgrid.t_array.flatten()

# Transducer Array Geometry
num_elements = 256
elem_width = 4
array_start_y = (Ny - (num_elements * elem_width)) // 2
array_row = 12

# %%==============================================================================
# 2. IMAGE PHANTOM GENERATOR (Speckle-Based Scatterer Map)
# ==============================================================================
image_path = "SIPAIM_2026_logo_blanco_small.png"
c_bg, rho_bg = 1500.0, 1000.0  # Homogeneous sound speed

if os.path.exists(image_path):
    img = Image.open(image_path).convert("RGBA")

    # Fit logo to 75% of transducer active width
    array_total_width = num_elements * elem_width  # 1024 points
    target_ny = int(array_total_width * 0.75)
    target_nx = int(target_ny * (img.height / img.width))

    img_resized = img.resize((target_ny, target_nx), Image.Resampling.LANCZOS)
    img_arr = np.array(img_resized)

    alpha = img_arr[:, :, 3] / 255.0
    grayscale = (0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]) / 255.0
    logo_mask = (grayscale * alpha) > 0.15

    pad_x_start = 280  # Depth ~11.2 mm
    pad_y_start = array_start_y + (array_total_width - target_ny) // 2

    padded_mask = np.zeros((Nx, Ny), dtype=bool)
    padded_mask[pad_x_start: pad_x_start + target_nx, pad_y_start: pad_y_start + target_ny] = logo_mask
else:
    print(f"Warning: {image_path} not found. Running with empty mask.")
    padded_mask = np.zeros((Nx, Ny), dtype=bool)

# Density map baseline
np.random.seed(42)
density_map = np.full((Nx, Ny), rho_bg, dtype=np.float32)

# Background tissue speckle (low amplitude random noise)
bg_scatter = np.random.normal(0, 2.0, size=(Nx, Ny))
density_map += bg_scatter

# Hyperechoic logo speckle (zero-mean noise prevents phase distortion/shadowing)
logo_scatter = np.random.normal(0, 15.0, size=(Nx, Ny))
density_map[padded_mask] += logo_scatter[padded_mask]

# Homogeneous sound speed map
sound_speed_map = np.full((Nx, Ny), c_bg, dtype=np.float32)

medium = kWaveMedium(
    sound_speed=sound_speed_map,
    density=density_map,
    alpha_coeff=0.5,
    alpha_power=1.5
)

# ==============================================================================
# VISUALIZATION
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(padded_mask, cmap="bone", aspect="auto")
axes[0].set_title("Padded Mask (Centered 75% FOV)")
axes[0].set_xlabel("Y Grid Units")
axes[0].set_ylabel("X Depth Grid Units")
fig.colorbar(im0, ax=axes[0], label="Binary Mask")

im1 = axes[1].imshow(density_map, cmap="bone", aspect="auto")
axes[1].set_title("Density Map (kg/m³)")
axes[1].set_xlabel("Y Grid Units")
fig.colorbar(im1, ax=axes[1], label="Density")

im2 = axes[2].imshow(sound_speed_map, cmap="bone", aspect="auto")
axes[2].set_title("Sound Speed Map (m/s)")
axes[2].set_xlabel("Y Grid Units")
fig.colorbar(im2, ax=axes[2], label="Speed (m/s)")

plt.tight_layout()
plt.show()

# %%==============================================================================
# 3. TRANSDUCER MASK SETUP
# ==============================================================================
sensor_mask = np.zeros((Nx, Ny), dtype=bool)
sensor_mask[array_row, array_start_y : array_start_y + num_elements * elem_width : elem_width] = True

sensor = kSensor()
sensor.mask = sensor_mask

element_y_coords = np.where(sensor_mask[array_row, :])[0]

# %%==============================================================================
# 4. FOCUSED ULTRASOUND SCANNING SEQUENCE
# ==============================================================================
active_elements = 32
focal_depth_m = 500 * dx  # Focus around ~20 mm depth

h5_filename = "sipaim_focused_ultrasound_rf.h5"

sim_options = SimulationOptions()
sim_options.save_to_disk = True

exec_options = SimulationExecutionOptions()
exec_options.is_gpu_simulation = True

print(f"Creating output HDF5 container: {h5_filename}")
with h5py.File(h5_filename, "w") as hf:
    hf.create_dataset(
        "rf_channel_data",
        shape=(num_elements, num_elements, Nt),
        dtype=np.float32,
        compression="gzip"
    )
    hf.create_dataset("density_map", data=density_map, compression="gzip")
    hf.create_dataset("sound_speed_map", data=sound_speed_map, compression="gzip")
    hf.attrs["dt"] = dt
    hf.attrs["dx"] = dx

print(f"Starting 256-Rayline GPU Simulation Sequence (Nt={Nt})...")

for tx_idx in range(num_elements):
    # Fixed 32-element aperture clamping across array edges
    sub_start = int(np.clip(tx_idx - active_elements // 2, 0, num_elements - active_elements))
    sub_end = sub_start + active_elements
    sub_indices = np.arange(sub_start, sub_end)

    sub_y_coords = element_y_coords[sub_indices]
    focal_point_y = element_y_coords[tx_idx]

    distances = np.sqrt(((sub_y_coords - focal_point_y) * dy) ** 2 + focal_depth_m ** 2)
    delays = (np.max(distances) - distances) / c_bg

    p_mask = np.zeros((Nx, Ny), dtype=bool)
    for y_c in sub_y_coords:
        p_mask[array_row, y_c] = True

    source = kSource()
    source.p_mask = p_mask

    source_p = np.zeros((len(sub_indices), Nt))
    for i, delay in enumerate(delays):
        t_shifted = t - delay - 1e-6
        pulse = np.sin(2 * np.pi * f0 * t_shifted) * np.exp(-(t_shifted ** 2) / (2 * 0.2e-6 ** 2))
        pulse[t < (delay + 1e-6)] = 0  # Fixed zeroing condition
        source_p[i, :] = pulse

    source.p = source_p

    sensor_data = kspaceFirstOrder2D(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=sim_options,
        execution_options=exec_options
    )

    with h5py.File(h5_filename, "a") as hf:
        hf["rf_channel_data"][tx_idx, :, :] = sensor_data['p'].T

    print(f"Completed Rayline {tx_idx + 1}/256")

print("Simulation complete.")