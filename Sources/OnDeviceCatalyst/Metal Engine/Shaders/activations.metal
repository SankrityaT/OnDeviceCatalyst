//
//  activations.metal
//  OnDeviceCatalyst
//
//  Activation functions and elementwise operations.
//

#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"

/// Fused SiLU gate activation with elementwise multiply.
/// output[i] = silu(gate[i]) * up[i]
/// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// This fuses two operations into one kernel, saving one memory round-trip.
kernel void silu_mul(
    const device float* gate        [[buffer(0)]],  // [count]
    const device float* up          [[buffer(1)]],  // [count]
    device float*       output      [[buffer(2)]],  // [count]
    constant ElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.count) return;

    float x = gate[tid];
    float silu_x = x / (1.0f + exp(-x));
    output[tid] = silu_x * up[tid];
}

/// Residual connection: output = x + residual
kernel void residual_add(
    const device float* x           [[buffer(0)]],  // [count]
    const device float* residual    [[buffer(1)]],  // [count]
    device float*       output      [[buffer(2)]],  // [count]
    constant ElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.count) return;
    output[tid] = x[tid] + residual[tid];
}

/// In-place residual add: x += residual
kernel void residual_add_inplace(
    device float*       x           [[buffer(0)]],  // [count], modified in-place
    const device float* residual    [[buffer(1)]],  // [count]
    constant ElementwiseParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.count) return;
    x[tid] += residual[tid];
}

/// Copy a buffer.
kernel void copy_buffer(
    const device float* src         [[buffer(0)]],
    device float*       dst         [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.count) return;
    dst[tid] = src[tid];
}

/// Copy K or V scratch buffer into the KV cache at a given position.
/// src: [numKVHeads * headDim], dst: [maxSeqLen * numKVHeads * headDim]
kernel void kv_cache_copy(
    const device float* src         [[buffer(0)]],
    device float*       dst         [[buffer(1)]],
    constant KVCacheParams& params  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint count = params.num_kv_heads * params.head_dim;
    if (tid >= count) return;
    uint offset = params.position * count;
    dst[offset + tid] = src[tid];
}
