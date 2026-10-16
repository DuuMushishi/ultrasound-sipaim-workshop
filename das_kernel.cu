// das_kernel.cu
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void delay_and_sum_kernel(
    const float* __restrict__ rf_data,     // (num_tx, num_rx, Nt)
    const float* __restrict__ grid_x,       // (N_depth, N_lateral)
    const float* __restrict__ grid_y,       // (N_depth, N_lateral)
    const float* __restrict__ rx_y_coords,  // (num_rx,)
    float* __restrict__ bmode_rf,           // (N_depth, N_lateral)
    int num_tx, int num_rx, int Nt,
    float dx, float c0, float dt, float t_0, float rx_x_coord)
{
    int depth_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lat_idx   = blockIdx.y * blockDim.y + threadIdx.y;

    if (depth_idx >= 512 || lat_idx >= 512) return;

    int pixel_idx = depth_idx * 512 + lat_idx;
    float x_p = grid_x[pixel_idx];
    float y_p = grid_y[pixel_idx];

    float pixel_sum = 0.0f;

    for (int tx = 0; tx < num_tx; ++tx) {
        float tx_y = rx_y_coords[tx];
        float dist_tx = sqrtf((x_p - rx_x_coord) * (x_p - rx_x_coord) + (y_p - tx_y) * (y_p - tx_y));

        for (int rx = 0; rx < num_rx; ++rx) {
            float rx_y = rx_y_coords[rx];
            float dist_rx = sqrtf((x_p - rx_x_coord) * (x_p - rx_x_coord) + (y_p - rx_y) * (y_p - rx_y));

            float tof = (dist_tx + dist_rx) / c0 + t_0;
            float sample_frac = tof / dt;

            int i0 = (int)floorf(sample_frac);
            int i1 = i0 + 1;

            if (i0 >= 0 && i1 < Nt) {
                float w1 = sample_frac - i0;
                float w0 = 1.0f - w1;

                int rf_idx_0 = (tx * num_rx + rx) * Nt + i0;
                int rf_idx_1 = (tx * num_rx + rx) * Nt + i1;

                pixel_sum += w0 * rf_data[rf_idx_0] + w1 * rf_data[rf_idx_1];
            }
        }
    }
    bmode_rf[pixel_idx] = pixel_sum;
}