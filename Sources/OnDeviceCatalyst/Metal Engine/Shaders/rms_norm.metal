//
//  rms_norm.metal
//  OnDeviceCatalyst
//
//  RMS Normalization: output = x * rsqrt(mean(x^2) + eps) * weight
//

#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"

/// RMS normalization with weight scaling.
/// One threadgroup processes one vector.
kernel void rms_norm(
    const device float* input       [[buffer(0)]],  // [size]
    device float*       output      [[buffer(1)]],  // [size]
    const device float* weight      [[buffer(2)]],  // [size] gamma
    constant NormParams& params     [[buffer(3)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint size = params.size;

    // Step 1: Compute sum of squares (each thread handles a stride)
    float sum_sq = 0.0f;
    for (uint i = tid; i < size; i += tg_size) {
        float val = input[i];
        sum_sq += val * val;
    }

    // Reduce across simdgroup
    sum_sq = simd_sum(sum_sq);

    // For larger threadgroups, we need threadgroup-level reduction
    // Using threadgroup memory for multi-simdgroup reduction
    threadgroup float shared_sum[32]; // max 32 simdgroups
    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;

    if (simd_lane == 0) {
        shared_sum[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first simdgroup
    if (simd_group == 0) {
        uint num_simdgroups = (tg_size + 31) / 32;
        float s = (simd_lane < num_simdgroups) ? shared_sum[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) {
            shared_sum[0] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = shared_sum[0];

    // Step 2: Compute RMS scale factor
    float rms = rsqrt(total_sum_sq / float(size) + params.eps);

    // Step 3: Normalize and scale
    for (uint i = tid; i < size; i += tg_size) {
        output[i] = input[i] * rms * weight[i];
    }
}

/// RMS norm without weight (just normalize).
kernel void rms_norm_no_weight(
    const device float* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    constant NormParams& params     [[buffer(2)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint size = params.size;

    float sum_sq = 0.0f;
    for (uint i = tid; i < size; i += tg_size) {
        float val = input[i];
        sum_sq += val * val;
    }

    sum_sq = simd_sum(sum_sq);

    threadgroup float shared_sum[32];
    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;

    if (simd_lane == 0) {
        shared_sum[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        uint num_simdgroups = (tg_size + 31) / 32;
        float s = (simd_lane < num_simdgroups) ? shared_sum[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) {
            shared_sum[0] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = rsqrt(shared_sum[0] / float(size) + params.eps);

    for (uint i = tid; i < size; i += tg_size) {
        output[i] = input[i] * rms;
    }
}
