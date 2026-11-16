#include <cuda_runtime.h>
#include <math.h>

#define MAX_SECTIONS 8

// ============================================================================
// 1. ZERO-PHASE SOS BIQUAD BANDPASS FILTER (Forward + Backward Pass)
// ============================================================================
__global__ void biquad_filtfilt_kernel(
    const float* __restrict__ rf_in,
    float* __restrict__ rf_out,
    const float* __restrict__ sos,
    int n_tx,
    int n_rx,
    int n_samples,
    int n_sections
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = n_tx * n_rx;
    if (ch >= total_channels) return;

    const float* x = rf_in + (size_t)ch * n_samples;
    float* y = rf_out + (size_t)ch * n_samples;

    // Copy input to output buffer for in-place cascade
    for (int i = 0; i < n_samples; ++i) {
        y[i] = x[i];
    }

    // Forward pass through SOS sections
    for (int s = 0; s < n_sections; ++s) {
        float b0 = sos[s * 6 + 0];
        float b1 = sos[s * 6 + 1];
        float b2 = sos[s * 6 + 2];
        float a1 = sos[s * 6 + 4];
        float a2 = sos[s * 6 + 5];

        float z1 = 0.0f, z2 = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            float in_val = y[i];
            float out_val = b0 * in_val + z1;
            z1 = b1 * in_val - a1 * out_val + z2;
            z2 = b2 * in_val - a2 * out_val;
            y[i] = out_val;
        }
    }

    // Backward pass through SOS sections (zero-phase)
    for (int s = 0; s < n_sections; ++s) {
        float b0 = sos[s * 6 + 0];
        float b1 = sos[s * 6 + 1];
        float b2 = sos[s * 6 + 2];
        float a1 = sos[s * 6 + 4];
        float a2 = sos[s * 6 + 5];

        float z1 = 0.0f, z2 = 0.0f;
        for (int i = n_samples - 1; i >= 0; --i) {
            float in_val = y[i];
            float out_val = b0 * in_val + z1;
            z1 = b1 * in_val - a1 * out_val + z2;
            z2 = b2 * in_val - a2 * out_val;
            y[i] = out_val;
        }
    }
}

// ============================================================================
// 2. SYNTHETIC APERTURE DELAY + F/# APODIZATION + TX ACCUMULATION
// ============================================================================
__global__ void sa_delay_and_accumulate_kernel(
    const float* __restrict__ rf_filtered,   // [n_tx, n_rx, n_samples]
    const float* __restrict__ elem_lat,      // [n_rx] lateral coordinates (m)
    const float* __restrict__ elem_dep,      // [n_rx] depth coordinates (m)
    const float* __restrict__ lat_grid,      // [n_lat]
    const float* __restrict__ dep_grid,      // [n_dep]
    float* __restrict__ rx_aperture_out,     // [n_rx, n_dep, n_lat]
    int n_tx,
    int n_rx,
    int n_samples,
    int n_dep,
    int n_lat,
    float c0,
    float fs,
    float f0,
    float t0,
    float f_number
) {
    int lat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dep_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int rx_idx  = blockIdx.z;

    if (lat_idx >= n_lat || dep_idx >= n_dep || rx_idx >= n_rx) return;

    float pix_lat = lat_grid[lat_idx];
    float pix_dep = dep_grid[dep_idx];

    float rx_lat = elem_lat[rx_idx];
    float rx_dep = elem_dep[rx_idx];

    float d_lat_rx = pix_lat - rx_lat;
    float d_dep_rx = pix_dep - rx_dep;

    // Receive f/# aperture check
    if (fabsf(d_lat_rx) > (d_dep_rx / (2.0f * f_number))) {
        rx_aperture_out[((size_t)rx_idx * n_dep + dep_idx) * n_lat + lat_idx] = 0.0f;
        return;
    }

    float dist_rx = hypotf(d_lat_rx, d_dep_rx);
    float ringdown_sec = t0 + (8.0f / f0);
    float acc = 0.0f;

    // Accumulate over all Tx elements
    for (int tx_idx = 0; tx_idx < n_tx; ++tx_idx) {
        float tx_lat = elem_lat[tx_idx];
        float tx_dep = elem_dep[tx_idx];

        float d_lat_tx = pix_lat - tx_lat;
        float d_dep_tx = pix_dep - tx_dep;

        // Transmit f/# aperture check
        if (fabsf(d_lat_tx) > (d_dep_tx / (2.0f * f_number))) {
            continue;
        }

        float dist_tx = hypotf(d_lat_tx, d_dep_tx);
        float tau = (dist_tx + dist_rx) / c0 + t0;

        // Direct-wave + top-PML ringdown muting
        float t_direct_end = (fabsf(tx_lat - rx_lat) / c0) + ringdown_sec;
        if (tau < t_direct_end) {
            continue;
        }

        float sample_pos = tau * fs;
        int idx0 = (int)floorf(sample_pos);
        if (idx0 >= 0 && idx0 < n_samples - 1) {
            float frac = sample_pos - (float)idx0;
            const float* trace = rf_filtered + ((size_t)tx_idx * n_rx + rx_idx) * n_samples;
            float val = (1.0f - frac) * trace[idx0] + frac * trace[idx0 + 1];
            acc += val;
        }
    }

    rx_aperture_out[((size_t)rx_idx * n_dep + dep_idx) * n_lat + lat_idx] = acc;
}

// ============================================================================
// 3. DAS SUMMATION WITH ACTIVE APERTURE WEIGHT NORMALIZATION
// ============================================================================
__global__ void das_summation_kernel(
    const float* __restrict__ rx_aperture,   // [n_rx, n_dep, n_lat]
    const float* __restrict__ elem_lat,      // [n_rx]
    const float* __restrict__ elem_dep,      // [n_rx]
    const float* __restrict__ lat_grid,      // [n_lat]
    const float* __restrict__ dep_grid,      // [n_dep]
    float* __restrict__ das_rf_out,          // [n_dep, n_lat]
    int n_rx,
    int n_dep,
    int n_lat,
    float f_number
) {
    int lat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dep_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (lat_idx >= n_lat || dep_idx >= n_dep) return;

    float pix_lat = lat_grid[lat_idx];
    float pix_dep = dep_grid[dep_idx];
    size_t pix_offset = (size_t)dep_idx * n_lat + lat_idx;
    size_t plane_stride = (size_t)n_dep * n_lat;

    float sum_val = 0.0f;
    float weight_sum = 0.0f;

    for (int rx_idx = 0; rx_idx < n_rx; ++rx_idx) {
        float d_lat = fabsf(pix_lat - elem_lat[rx_idx]);
        float d_dep = pix_dep - elem_dep[rx_idx];
        if (d_lat <= (d_dep / (2.0f * f_number))) {
            sum_val += rx_aperture[(size_t)rx_idx * plane_stride + pix_offset];
            weight_sum += 1.0f;
        }
    }

    das_rf_out[pix_offset] = sum_val / (weight_sum + 1e-6f);
}

// ============================================================================
// 4. SHORT-LAG SPATIAL COHERENCE (SLSC) KERNEL
// ============================================================================
__global__ void slsc_coherence_kernel(
    const float* __restrict__ rx_aperture,   // [n_rx, n_dep, n_lat]
    float* __restrict__ slsc_out,            // [n_dep, n_lat]
    int n_rx,
    int n_dep,
    int n_lat,
    int max_lag,
    int kernel_z
) {
    int lat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dep_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (lat_idx >= n_lat || dep_idx >= n_dep) return;

    int half_k = kernel_z / 2;
    int z_start = max(0, dep_idx - half_k);
    int z_end   = min(n_dep - 1, dep_idx + (kernel_z - half_k - 1));
    size_t plane_stride = (size_t)n_dep * n_lat;

    float total_lag_coherence = 0.0f;

    for (int lag = 1; lag <= max_lag; ++lag) {
        float rho_sum = 0.0f;
        int valid_pairs = 0;

        for (int rx = 0; rx < n_rx - lag; ++rx) {
            float num = 0.0f;
            float den1 = 0.0f;
            float den2 = 0.0f;

            const float* base1 = rx_aperture + (size_t)rx * plane_stride + lat_idx;
            const float* base2 = rx_aperture + (size_t)(rx + lag) * plane_stride + lat_idx;

            for (int z = z_start; z <= z_end; ++z) {
                float v1 = base1[(size_t)z * n_lat];
                float v2 = base2[(size_t)z * n_lat];
                num  += v1 * v2;
                den1 += v1 * v1;
                den2 += v2 * v2;
            }

            if (den1 > 1e-20f && den2 > 1e-20f) {
                rho_sum += num / (sqrtf(den1 * den2) + 1e-20f);
                valid_pairs++;
            }
        }

        if (valid_pairs > 0) {
            total_lag_coherence += rho_sum / (float)valid_pairs;
        }
    }

    slsc_out[(size_t)dep_idx * n_lat + lat_idx] = total_lag_coherence / (float)max_lag;
}

// ============================================================================
// C-ABI DLL EXPORT WRAPPERS
// ============================================================================
extern "C" {

__declspec(dllexport) void launch_biquad_filtfilt(
    const float* rf_in, float* rf_out, const float* sos,
    int n_tx, int n_rx, int n_samples, int n_sections
) {
    int total = n_tx * n_rx;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    biquad_filtfilt_kernel<<<blocks, threads>>>(rf_in, rf_out, sos, n_tx, n_rx, n_samples, n_sections);
}

__declspec(dllexport) void launch_sa_delay_and_accumulate(
    const float* rf_filtered, const float* elem_lat, const float* elem_dep,
    const float* lat_grid, const float* dep_grid, float* rx_aperture_out,
    int n_tx, int n_rx, int n_samples, int n_dep, int n_lat,
    float c0, float fs, float f0, float t0, float f_number
) {
    dim3 threads(16, 16, 1);
    dim3 blocks((n_lat + 15) / 16, (n_dep + 15) / 16, n_rx);
    sa_delay_and_accumulate_kernel<<<blocks, threads>>>(
        rf_filtered, elem_lat, elem_dep, lat_grid, dep_grid, rx_aperture_out,
        n_tx, n_rx, n_samples, n_dep, n_lat, c0, fs, f0, t0, f_number
    );
}

__declspec(dllexport) void launch_das_summation(
    const float* rx_aperture, const float* elem_lat, const float* elem_dep,
    const float* lat_grid, const float* dep_grid, float* das_rf_out,
    int n_rx, int n_dep, int n_lat, float f_number
) {
    dim3 threads(16, 16);
    dim3 blocks((n_lat + 15) / 16, (n_dep + 15) / 16);
    das_summation_kernel<<<blocks, threads>>>(
        rx_aperture, elem_lat, elem_dep, lat_grid, dep_grid, das_rf_out,
        n_rx, n_dep, n_lat, f_number
    );
}

__declspec(dllexport) void launch_slsc_coherence(
    const float* rx_aperture, float* slsc_out,
    int n_rx, int n_dep, int n_lat, int max_lag, int kernel_z
) {
    dim3 threads(16, 16);
    dim3 blocks((n_lat + 15) / 16, (n_dep + 15) / 16);
    slsc_coherence_kernel<<<blocks, threads>>>(
        rx_aperture, slsc_out, n_rx, n_dep, n_lat, max_lag, kernel_z
    );
}

} // extern "C"