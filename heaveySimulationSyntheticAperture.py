## Libraries
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from ttictoc import tic, toc

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

## Grid configuration
Nx, Ny = 1024, 1280  # 1024 active array points + 128 padding per side (~4.1 cm x 5.12 cm)
dx = dy = 4e-5       # 40 um spatial step
kgrid = kWaveGrid([Nx, Ny], [dx, dy])

f0 = 5e6             # 5 MHz center frequency
dt = 10e-9           # 100 MHz sampling (CFL = 0.375, stable for power-law absorption)
Nt = 5500            # 110 us total window
kgrid.setTime(Nt, dt)
t = kgrid.t_array.flatten()

# Transducer Array Geometry
num_elements = 256
elem_width = 4
array_start_y = (Ny - (num_elements * elem_width)) // 2
array_row = 12

## Image phantom generator, modifying sound speed map and density map
script_dir = Path(__file__).resolve().parent
image_path = script_dir / "SIPAIM_2026_logo_blanco_small.png"
c_bg, rho_bg = 1500.0, 1000.0  # Homogeneous sound speed

if image_path.exists():
    img = Image.open(image_path).convert("RGBA")

    # Fit logo to 75% of transducer active width
    array_total_width = num_elements * elem_width  # 1024 points
    target_ny = int(array_total_width * 0.75)
    target_nx = int(target_ny * (img.height / img.width))

    img_resized = img.resize((target_ny, target_nx), Image.Resampling.LANCZOS)
    img_arr = np.array(img_resized)

    alpha = img_arr[:, :, 3] / 255.0
    grayscale = (
        0.299 * img_arr[:, :, 0]
        + 0.587 * img_arr[:, :, 1]
        + 0.114 * img_arr[:, :, 2]
    ) / 255.0
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

# Background tissue speckle
bg_scatter = np.random.normal(0, 2.0, size=(Nx, Ny))
density_map += bg_scatter

# Hyperechoic logo speckle with zero-mean noise
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

## Transducer mask setup
sensor_mask = np.zeros((Nx, Ny), dtype=bool)
sensor_mask[array_row, array_start_y: array_start_y + num_elements * elem_width: elem_width] = True

sensor = kSensor()
sensor.mask = sensor_mask

element_y_coords = np.where(sensor_mask[array_row, :])[0]

## Synthetic Aperture (Full matrix capture) scanning sequence
h5_filename = script_dir / "sipaim_synthetic_aperture_fmc.h5"

# Pre-compute single-element excitation pulse (t0 = 1.0 us offset)
t0 = 1e-6
t_shifted = t - t0
single_pulse = np.sin(2 * np.pi * f0 * t_shifted) * np.exp(-(t_shifted ** 2) / (2 * 0.2e-6 ** 2))
single_pulse[t < t0] = 0.0

# Create HDF5 container if it doesn't exist, or resume if it does
if not h5_filename.exists():
    print(f"Creating new HDF5 container: {h5_filename}")
    with h5py.File(h5_filename, "w") as hf:
        hf.create_dataset(
            "rf_channel_data",
            shape=(num_elements, num_elements, Nt),
            chunks=(1, num_elements, Nt),
            dtype=np.float32
        )
        hf.create_dataset("density_map", data=density_map, compression="gzip")
        hf.create_dataset("sound_speed_map", data=sound_speed_map, compression="gzip")

        hf.create_dataset("element_y_idx", data=element_y_coords)
        hf.attrs["array_x_idx"] = array_row
        hf.attrs["dt"] = dt
        hf.attrs["dx"] = dx
        hf.attrs["dy"] = dy
        hf.attrs["c_bg"] = c_bg
        hf.attrs["f0"] = f0
        hf.attrs["t0"] = t0
else:
    print(f"Found existing HDF5 container: {h5_filename} (checking completed Tx elements...)")

# Detect which Tx elements are already simulated
with h5py.File(h5_filename, "r") as hf:
    if hf["rf_channel_data"].shape != (num_elements, num_elements, Nt):
        raise ValueError(
            f"Existing file has shape {hf['rf_channel_data'].shape}, expected {(num_elements, num_elements, Nt)}. "
            "Delete or rename the old .h5 file first."
        )
    raw = hf["rf_channel_data"][:]
    completed_mask = np.isfinite(raw).all(axis=(1, 2)) & (np.max(np.abs(raw), axis=(1, 2)) > 0)

pending_tx = np.where(~completed_mask)[0]
print(f"Completed: {np.sum(completed_mask)}/{num_elements} | Remaining: {len(pending_tx)} Tx elements")

exec_options = SimulationExecutionOptions()
exec_options.is_gpu_simulation = True
exec_options.show_sim_log = False

BATCH_SIZE = 4  # Workaround, TODO, avoid this: Python process every 4 Tx avoids k-wave-python's 4-call state limit

if len(pending_tx) > 0:
    batch = pending_tx[:BATCH_SIZE]
    print(f"Processing batch: Tx {batch[0] + 1}..{batch[-1] + 1}")

    for tx_idx in batch:
        tic()
        p_mask = np.zeros((Nx, Ny), dtype=bool)
        p_mask[array_row, element_y_coords[tx_idx]] = True

        source = kSource()
        source.p_mask = p_mask
        source.p = single_pulse.reshape(1, Nt)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            sim_options = SimulationOptions()
            sim_options.save_to_disk = True
            sim_options.data_path = tmpdir

            sensor_data = kspaceFirstOrder2D(
                kgrid=kgrid,
                medium=medium,
                source=source,
                sensor=sensor,
                simulation_options=sim_options,
                execution_options=exec_options
            )

        p_data = sensor_data['p'].T
        if not np.isfinite(p_data).all() or np.max(np.abs(p_data)) == 0.0:
            raise RuntimeError(f"Aborted at Tx {tx_idx + 1}/{num_elements}: invalid or zero output.")

        with h5py.File(h5_filename, "a") as hf:
            hf["rf_channel_data"][tx_idx, :, :] = p_data

        print(f"Completed Tx Element {tx_idx + 1}/{num_elements} ({toc():.2f} s)")

    # Spawn a fresh Python process for the next batch and exit cleanly
    if len(pending_tx) > BATCH_SIZE:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())])
        sys.exit(result.returncode)

print("Synthetic Aperture FMC Simulation complete (256/256)!")