import cupy as cp

das_kernel_code = r'''
extern "C" __global__
void delay_and_sum_kernel(
    const float* rf_data,     // Shape: [num_tx, num_rx, Nt]
    const float* grid_x,      // Shape: [num_pixels]
    const float* grid_y,
    const float* rx_y_coords,
    float* bmode_out,         // Shape: [num_pixels]
    int num_tx, int num_rx, int Nt,
    float dt, float dx, float c0, float t_0, float rx_x_coord
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float x = grid_x[idx];
    float y = grid_y[idx];
    float sum_val = 0.0f;

    for (int tx = 0; tx < num_tx; ++tx) {
        float tx_y = rx_y_coords[tx];
        float dist_tx = sqrtf((x - rx_x_coord)*(x - rx_x_coord) + (y - tx_y)*(y - tx_y));

        for (int rx = 0; rx < num_rx; ++rx) {
            float rx_y = rx_y_coords[rx];
            float dist_rx = sqrtf((x - rx_x_coord)*(x - rx_x_coord) + (y - rx_y)*(y - rx_y));

            float tof = ((dist_tx + dist_rx) / c0) + t_0;
            int sample_idx = __float2int_rn(tof / dt);

            if (sample_idx >= 0 && sample_idx < Nt) {
                // Access 3D flattened tensor
                int rf_idx = tx * (num_rx * Nt) + rx * Nt + sample_idx;
                sum_val += rf_data[rf_idx];
            }
        }
    }
    bmode_out[idx] = sum_val;
}
'''

das_kernel = cp.RawKernel(das_kernel_code, 'delay_and_sum_kernel')

# Launch configuration
num_pixels = GRID_X.size
threads_per_block = 256
blocks = (num_pixels + threads_per_block - 1) // threads_per_block

# Execute kernel directly on GPU
das_kernel((blocks,), (threads_per_block,), (
    rf_gpu, grid_x_gpu, grid_y_gpu, rx_y_gpu, bmode_gpu,
    num_tx, num_rx, Nt, dt, dx, c0, t_0, rx_x_coord
))