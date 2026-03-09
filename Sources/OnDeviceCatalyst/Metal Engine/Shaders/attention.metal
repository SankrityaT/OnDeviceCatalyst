//
//  attention.metal
//  OnDeviceCatalyst
//
//  Grouped Query Attention (GQA) for the decode step.
//  Uses online softmax (single pass over K/V cache) for correct results at any seq_len.
//

#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"

// Maximum head dimension supported (128 covers LLaMA, Qwen, Mistral, etc.)
constant constexpr uint MAX_HEAD_DIM = 128;

/// Online softmax state for merge-reduction across threads/simdgroups.
struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;
    float v_accum[MAX_HEAD_DIM];
};

/// Single-token attention decode using online softmax.
///
/// Each threadgroup handles one query head. Threads split sequence positions.
/// Single pass over K/V cache: computes Q·K scores and accumulates weighted V
/// simultaneously using the online softmax algorithm, then reduces across threads.
kernel void gqa_attention_decode(
    const device float* q           [[buffer(0)]],  // [num_heads, head_dim]
    const device float* k_cache     [[buffer(1)]],  // [max_seq_len, num_kv_heads, head_dim]
    const device float* v_cache     [[buffer(2)]],  // [max_seq_len, num_kv_heads, head_dim]
    device float*       output      [[buffer(3)]],  // [num_heads, head_dim]
    constant AttentionParams& params [[buffer(4)]],
    uint head_id [[threadgroup_position_in_grid]],   // which query head
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (head_id >= params.num_heads) return;

    const uint head_dim = params.head_dim;
    const uint seq_len = params.seq_len;
    const uint num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;

    // Which KV head this query head maps to (for GQA)
    uint kv_head = head_id * num_kv_heads / params.num_heads;

    // Pointer to this head's query vector
    const device float* q_head = q + head_id * head_dim;

    // --- Online Softmax: single pass over K/V cache ---
    // Each thread maintains its own (max, sum, v_accum) state
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    float v_accum[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        v_accum[d] = 0.0f;
    }

    for (uint pos = tid; pos < seq_len; pos += tg_size) {
        const device float* k_vec = k_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;

        // Compute Q·K score
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += q_head[d] * k_vec[d];
        }
        score *= scale;

        // Online softmax update
        float old_max = thread_max;
        thread_max = max(thread_max, score);
        float correction = exp(old_max - thread_max);
        thread_sum = thread_sum * correction + exp(score - thread_max);

        // Update V accumulator with correction
        const device float* v_vec = v_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;
        float weight = exp(score - thread_max);
        for (uint d = 0; d < head_dim; d++) {
            v_accum[d] = v_accum[d] * correction + weight * v_vec[d];
        }
    }

    // --- Cross-thread reduction via online softmax merge ---

    // Step 1: Simd-level merge (within 32-thread simdgroup)
    // Use simd_shuffle to merge pairs: lane 0+1, 2+3, ... then 0+2, ... etc.
    for (uint offset = 1; offset < min(32u, tg_size); offset *= 2) {
        float other_max = simd_shuffle_xor(thread_max, offset);
        float other_sum = simd_shuffle_xor(thread_sum, offset);

        float merged_max = max(thread_max, other_max);
        float wa = exp(thread_max - merged_max);
        float wb = exp(other_max - merged_max);

        for (uint d = 0; d < head_dim; d++) {
            float other_v = simd_shuffle_xor(v_accum[d], offset);
            v_accum[d] = v_accum[d] * wa + other_v * wb;
        }

        thread_max = merged_max;
        thread_sum = thread_sum * wa + other_sum * wb;
    }

    // Step 2: Cross-simdgroup merge via threadgroup memory
    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;
    uint n_sg = (tg_size + 31) / 32;

    // Shared memory for cross-simdgroup reduction
    threadgroup float shared_max[8];
    threadgroup float shared_sum[8];
    threadgroup float shared_v[8 * MAX_HEAD_DIM];  // [8][MAX_HEAD_DIM]

    if (n_sg > 1) {
        // Each simdgroup leader writes its state
        if (simd_lane == 0) {
            shared_max[simd_group] = thread_max;
            shared_sum[simd_group] = thread_sum;
            for (uint d = 0; d < head_dim; d++) {
                shared_v[simd_group * MAX_HEAD_DIM + d] = v_accum[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // First simdgroup's lane 0 does final merge
        if (tid == 0) {
            float final_max = shared_max[0];
            float final_sum = shared_sum[0];
            // v_accum already has simdgroup 0's values

            for (uint sg = 1; sg < n_sg; sg++) {
                float other_max = shared_max[sg];
                float merged_max = max(final_max, other_max);
                float wa = exp(final_max - merged_max);
                float wb = exp(other_max - merged_max);

                for (uint d = 0; d < head_dim; d++) {
                    v_accum[d] = v_accum[d] * wa + shared_v[sg * MAX_HEAD_DIM + d] * wb;
                }

                final_max = merged_max;
                final_sum = final_sum * wa + shared_sum[sg] * wb;
            }

            thread_max = final_max;
            thread_sum = final_sum;
        }
    }

    // --- Write output ---
    if (tid == 0) {
        device float* out_head = output + head_id * head_dim;
        float inv_sum = (thread_sum > 0.0f) ? (1.0f / thread_sum) : 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            out_head[d] = v_accum[d] * inv_sum;
        }
    }
}

// MARK: - Prefill Attention (Causal Masked)

/// Prefill attention for batch processing multiple query tokens.
/// One threadgroup per (query_head, query_position).
/// Each query attends to positions [0, query_pos] only (causal mask).
kernel void prefill_attention(
    const device float* q           [[buffer(0)]],  // [batch_size, num_heads, head_dim]
    const device float* k_cache     [[buffer(1)]],  // [max_seq_len, num_kv_heads, head_dim]
    const device float* v_cache     [[buffer(2)]],  // [max_seq_len, num_kv_heads, head_dim]
    device float*       output      [[buffer(3)]],  // [batch_size, num_heads, head_dim]
    constant AttentionParams& params [[buffer(4)]],
    constant uint&      start_pos   [[buffer(5)]],  // start position in KV cache
    constant uint&      batch_size  [[buffer(6)]],
    uint2 gid    [[threadgroup_position_in_grid]],
    uint2 tid2   [[thread_position_in_threadgroup]],
    uint2 tg2    [[threads_per_threadgroup]]
) {
    uint head_id = gid.x;
    uint batch_idx = gid.y;
    uint tid = tid2.x;
    uint tg_size = tg2.x;

    if (head_id >= params.num_heads || batch_idx >= batch_size) return;

    const uint head_dim = params.head_dim;
    const uint num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;
    uint kv_head = head_id * num_kv_heads / params.num_heads;

    // This query token's position and the causal limit
    uint query_pos = start_pos + batch_idx;
    uint causal_len = query_pos + 1;  // attend to [0, query_pos]

    // Pointer to this query vector
    const device float* q_head = q + (batch_idx * params.num_heads + head_id) * head_dim;

    // Online softmax over causal positions
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    float v_accum[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; d++) {
        v_accum[d] = 0.0f;
    }

    for (uint pos = tid; pos < causal_len; pos += tg_size) {
        const device float* k_vec = k_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;

        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += q_head[d] * k_vec[d];
        }
        score *= scale;

        float old_max = thread_max;
        thread_max = max(thread_max, score);
        float correction = exp(old_max - thread_max);
        thread_sum = thread_sum * correction + exp(score - thread_max);

        const device float* v_vec = v_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;
        float weight = exp(score - thread_max);
        for (uint d = 0; d < head_dim; d++) {
            v_accum[d] = v_accum[d] * correction + weight * v_vec[d];
        }
    }

    // Cross-thread reduction (same as decode)
    for (uint offset = 1; offset < min(32u, tg_size); offset *= 2) {
        float other_max = simd_shuffle_xor(thread_max, offset);
        float other_sum = simd_shuffle_xor(thread_sum, offset);

        float merged_max = max(thread_max, other_max);
        float wa = exp(thread_max - merged_max);
        float wb = exp(other_max - merged_max);

        for (uint d = 0; d < head_dim; d++) {
            float other_v = simd_shuffle_xor(v_accum[d], offset);
            v_accum[d] = v_accum[d] * wa + other_v * wb;
        }

        thread_max = merged_max;
        thread_sum = thread_sum * wa + other_sum * wb;
    }

    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;
    uint n_sg = (tg_size + 31) / 32;

    threadgroup float shared_max[8];
    threadgroup float shared_sum[8];
    threadgroup float shared_v[8 * MAX_HEAD_DIM];

    if (n_sg > 1) {
        if (simd_lane == 0) {
            shared_max[simd_group] = thread_max;
            shared_sum[simd_group] = thread_sum;
            for (uint d = 0; d < head_dim; d++) {
                shared_v[simd_group * MAX_HEAD_DIM + d] = v_accum[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float final_max = shared_max[0];
            float final_sum = shared_sum[0];

            for (uint sg = 1; sg < n_sg; sg++) {
                float other_max = shared_max[sg];
                float merged_max = max(final_max, other_max);
                float wa = exp(final_max - merged_max);
                float wb = exp(other_max - merged_max);

                for (uint d = 0; d < head_dim; d++) {
                    v_accum[d] = v_accum[d] * wa + shared_v[sg * MAX_HEAD_DIM + d] * wb;
                }

                final_max = merged_max;
                final_sum = final_sum * wa + shared_sum[sg] * wb;
            }

            thread_max = final_max;
            thread_sum = final_sum;
        }
    }

    if (tid == 0) {
        device float* out_head = output + (batch_idx * params.num_heads + head_id) * head_dim;
        float inv_sum = (thread_sum > 0.0f) ? (1.0f / thread_sum) : 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            out_head[d] = v_accum[d] * inv_sum;
        }
    }
}
