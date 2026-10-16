from numba import cuda
import math


@cuda.jit
def das_kernel_numba(rf_data, grid_x, grid_y, rx_y_coords, bmode_out, dt, c0, t_0, rx_x_coord):
    pixel_idx = cuda.grid(1)
    if pixel_idx < grid_x.size:
        x = grid_x[pixel_idx]
        y = grid_y[pixel_idx]
        val = 0.0

        for tx in range(rf_data.shape[0]):
            tx_y = rx_y_coords[tx]
            dist_tx = math.sqrt((x - rx_x_coord) ** 2 + (y - tx_y) ** 2)

            for rx in range(rf_data.shape[1]):
                rx_y = rx_y_coords[rx]
                dist_rx = math.sqrt((x - rx_x_coord) ** 2 + (y - rx_y) ** 2)

                sample_idx = int(round((((dist_tx + dist_rx) / c0) + t_0) / dt))
                if 0 <= sample_idx < rf_data.shape[2]:
                    val += rf_data[tx, rx, sample_idx]

        bmode_out[pixel_idx] = val