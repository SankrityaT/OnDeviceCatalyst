//
//  embedding.metal
//  OnDeviceCatalyst
//
//  Token embedding lookup — copies rows from the embedding table.
//

#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"

/// Look up a token embedding from the embedding table.
/// Copies embedding_table[token_id] to the output buffer.
kernel void embedding_lookup(
    const device float* embedding_table [[buffer(0)]],  // [vocab_size, hidden_size]
    device float*       output          [[buffer(1)]],  // [hidden_size]
    constant EmbeddingParams& params    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.hidden_size) return;

    uint offset = params.token_id * params.hidden_size + tid;
    output[tid] = embedding_table[offset];
}

/// Look up from an f16 embedding table.
kernel void embedding_lookup_f16(
    const device half*  embedding_table [[buffer(0)]],
    device float*       output          [[buffer(1)]],
    constant EmbeddingParams& params    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.hidden_size) return;

    uint offset = params.token_id * params.hidden_size + tid;
    output[tid] = float(embedding_table[offset]);
}

/// Batch embedding lookup — copies multiple rows from the embedding table.
/// output: [batch_size, hidden_size], token_ids: [batch_size]
kernel void batch_embedding_lookup(
    const device float* embedding_table [[buffer(0)]],  // [vocab_size, hidden_size]
    device float*       output          [[buffer(1)]],  // [batch_size, hidden_size]
    const device uint*  token_ids       [[buffer(2)]],  // [batch_size]
    constant uint&      hidden_size     [[buffer(3)]],
    constant uint&      batch_size      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = batch_size * hidden_size;
    if (tid >= total) return;

    uint batch_idx = tid / hidden_size;
    uint dim_idx = tid % hidden_size;
    uint token_id = token_ids[batch_idx];

    output[tid] = embedding_table[token_id * hidden_size + dim_idx];
}

/// Batch embedding lookup from f16 table.
kernel void batch_embedding_lookup_f16(
    const device half*  embedding_table [[buffer(0)]],
    device float*       output          [[buffer(1)]],
    const device uint*  token_ids       [[buffer(2)]],
    constant uint&      hidden_size     [[buffer(3)]],
    constant uint&      batch_size      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = batch_size * hidden_size;
    if (tid >= total) return;

    uint batch_idx = tid / hidden_size;
    uint dim_idx = tid % hidden_size;
    uint token_id = token_ids[batch_idx];

    output[tid] = float(embedding_table[token_id * hidden_size + dim_idx]);
}
