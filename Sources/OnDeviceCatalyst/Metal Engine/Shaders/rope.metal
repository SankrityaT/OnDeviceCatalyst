//
//  rope.metal
//  OnDeviceCatalyst
//
//  Rotary Position Embedding (RoPE).
//  Applied to Q and K vectors before attention computation.
//

#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"

/// Apply RoPE to Q or K vectors in-place (single token).
/// Each thread handles one pair of dimensions for one head.
///
/// Input layout: [num_heads, head_dim] stored contiguously.
/// Operates on dimension pairs (2i, 2i+1).
kernel void rope_apply(
    device float*        qk         [[buffer(0)]],  // [num_heads, head_dim]
    constant RoPEParams& params     [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint head_dim = params.head_dim;
    const uint num_heads = params.num_heads;
    const uint half_dim = head_dim / 2;
    const uint total_pairs = num_heads * half_dim;

    if (tid >= total_pairs) return;

    uint head = tid / half_dim;
    uint pair_idx = tid % half_dim;

    // Compute rotation angle
    float freq = 1.0f / pow(params.freq_base, float(2 * pair_idx) / float(head_dim));
    freq *= params.freq_scale;
    float angle = float(params.position) * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Rotate the pair
    uint base = head * head_dim + pair_idx * 2;
    float x0 = qk[base];
    float x1 = qk[base + 1];

    qk[base]     = x0 * cos_val - x1 * sin_val;
    qk[base + 1] = x0 * sin_val + x1 * cos_val;
}

/// Batch RoPE — apply RoPE to multiple tokens at different positions.
/// qk layout: [batch_size, num_heads, head_dim]
/// positions: [batch_size] — absolute position for each token
kernel void batch_rope_apply(
    device float*        qk         [[buffer(0)]],  // [batch_size, num_heads, head_dim]
    const device uint*   positions  [[buffer(1)]],  // [batch_size]
    constant RoPEParams& params     [[buffer(2)]],  // num_heads, head_dim, freq_base, freq_scale used
    constant uint&       batch_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint head_dim = params.head_dim;
    const uint num_heads = params.num_heads;
    const uint half_dim = head_dim / 2;
    const uint pairs_per_token = num_heads * half_dim;
    const uint total = batch_size * pairs_per_token;

    if (tid >= total) return;

    uint batch_idx = tid / pairs_per_token;
    uint within_token = tid % pairs_per_token;
    uint head = within_token / half_dim;
    uint pair_idx = within_token % half_dim;

    float freq = 1.0f / pow(params.freq_base, float(2 * pair_idx) / float(head_dim));
    freq *= params.freq_scale;
    float angle = float(positions[batch_idx]) * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    uint base = (batch_idx * num_heads + head) * head_dim + pair_idx * 2;
    float x0 = qk[base];
    float x1 = qk[base + 1];

    qk[base]     = x0 * cos_val - x1 * sin_val;
    qk[base + 1] = x0 * sin_val + x1 * cos_val;
}
