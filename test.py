import h5py, numpy as np
with h5py.File("sipaim_synthetic_aperture_fmc.h5", "r") as hf:
    d = hf["rf_channel_data"][:]
    print("Shape:", d.shape, "Min:", np.nanmin(d), "Max:", np.nanmax(d), "NaNs:", np.isnan(d).sum())
    print("Attributes:", dict(hf.attrs))