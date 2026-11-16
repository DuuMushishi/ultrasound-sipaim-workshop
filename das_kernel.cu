// das_kernel.cu
#include <cuda_runtime.h>
#include <math.h>

#define MAX_SECTIONS 8

// Helper: Run one sample through N cascaded biquad (SOS) sections (Transposed Direct Form II)
__device__ __forceinline__ float biquad_step(
    float x,
    const float* __restrict__ sos, // [n_sections * 6]: b0, b1, b2, a0, a1, a2
    float* z1,
    float* z2,
    int n_sections)
{
    float y = x;
    #pragma unroll
    for (int s = 0; s < MAX_SECTIONS; ++s) {
        if (s >= n_sections) break;
        const float* sec = sos + s * 6;
        float b0 = sec[0], b1 = sec[1], b2 = sec[2];
        float a1 = sec[4], a2 = sec[5]; // sec[3] (a0) is always 1.0 in SciPy SOS

        float out = b0 * y + z1[s];
        z1[s] = b1 * y - a1 * out + z2[s];
        z2[s] = b2 * y - a2 * out;
        y = out;
    }
    return y;
}

__global__ void biquad_filtfilt_kernel(
    const float* __restrict__ rf_raw,      // (num_tx, num_rx, Nt)
    float* __restrict__ rf_filtered,       // (num_tx, num_rx, Nt)
    const float* __restrict__ sos,         // (n_sections, 6)
    int num_traces,                        // num_tx * num_rx
    int Nt,
    int n_sections,
    int padlen)
{
    int trace_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (trace_idx >= num_traces) return;

    const float* x = rf_raw + (size_t)trace_idx * Nt;
    float* y = rf_filtered + (size_t)trace_idx * Nt;

    // Clamp padlen safely if Nt is very short
    int eff_pad = (padlen < Nt - 1) ? padlen : (Nt - 1);

    // ---------------------------------------------------------
    // 1. FORWARD PASS (with odd reflection edge padding)
    // ---------------------------------------------------------
    float z1[MAX_SECTIONS] = {0.0f};
    float z2[MAX_SECTIONS] = {0.0f};

    float x0 = x[0];
    for (int i = eff_pad; i >= 1; --i) {
        float val = 2.0f * x0 - x[i];
        biquad_step(val, sos, z1, z2, n_sections);
    }

    for (int t = 0; t < Nt; ++t) {
        y[t] = biquad_step(x[t], sos, z1, z2, n_sections);
    }

    float x_end = x[Nt - 1];
    for (int i = 1; i <= eff_pad; ++i) {
        float val = 2.0f * x_end - x[Nt - 1 - i];
        biquad_step(val, sos, z1, z2, n_sections);
    }

    // ---------------------------------------------------------
    // 2. BACKWARD PASS (zero-phase reversal)
    // ---------------------------------------------------------
    #pragma unroll
    for (int s = 0; s < MAX_SECTIONS; ++s) {
        z1[s] = 0.0f;
        z2[s] = 0.0f;
    }

    // Note: For exact SciPy parity, the backward pass warms up on the reversed
    // right-padded tail; warming up on reflected y[Nt-1] closely matches SciPy
    float y_end = y[Nt - 1];
    for (int i = eff_pad; i >= 1; --i) {
        float val = 2.0f * y_end - y[Nt - 1 - i];
        biquad_step(val, sos, z1, z2, n_sections);
    }

    for (int t = Nt - 1; t >= 0; --t) {
        y[t] = biquad_step(y[t], sos, z1, z2, n_sections);
    }
}

extern "C" __declspec(dllexport) void launch_biquad_filtfilt(
    const float* rf_raw,
    float* rf_filtered,
    const float* sos,
    int num_tx,
    int num_rx,
    int Nt,
    int n_sections)
{
    int num_traces = num_tx * num_rx;
    int padlen = 3 * (2 * n_sections + 1); // Matches SciPy sosfiltfilt default padlen

    int threads = 256;
    int blocks = (num_traces + threads - 1) / threads;

    biquad_filtfilt_kernel<<<blocks, threads>>>(
        rf_raw, rf_filtered, sos, num_traces, Nt, n_sections, padlen
    );
    cudaDeviceSynchronize();
}

// --- Existing DAS Kernel ---
__global__ void delay_and_sum_kernel(
    const float* __restrict__ rf_data,
    const float* __restrict__ grid_x,
    const float* __restrict__ grid_y,
    const float* __restrict__ rx_y_coords,
    float* __restrict__ bmode_rf,
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

extern "C" __declspec(dllexport) void launch_delay_and_sum(
    const float* rf_data,
    const float* grid_x,
    const float* grid_y,
    const float* rx_y_coords,
    float* bmode_rf,
    int num_tx, int num_rx, int Nt,
    float dx, float c0, float dt, float t_0, float rx_x_coord)
{
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (512 + threads_per_block.x - 1) / threads_per_block.x,
        (512 + threads_per_block.y - 1) / threads_per_block.y
    );

    delay_and_sum_kernel<<<blocks_per_grid, threads_per_block>>>(
        rf_data, grid_x, grid_y, rx_y_coords, bmode_rf,
        num_tx, num_rx, Nt, dx, c0, dt, t_0, rx_x_coord
    );
    cudaDeviceSynchronize();
}