//
//  matvec.metal
//  OnDeviceCatalyst
//
//  Quantized matrix-vector multiplication — the critical kernel for decode speed.
//  During decode, we multiply a [rows, cols] weight matrix by a [cols, 1] input vector.
//  This is memory-bandwidth bound: the entire weight matrix must be read for each token.
//
//  Strategy: Each threadgroup computes one output row. Threads within the group
//  cooperatively read and dequantize blocks along the row, accumulate partial
//  dot products, then reduce with simd_sum + cross-simdgroup reduction.
//

#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"
#include "dequantize.h"

// MARK: - Q8_0 Matrix-Vector Multiply

/// Each threadgroup computes one output element.
/// Threads within the group split the row across themselves.
kernel void matvec_q8_0(
    const device uchar* weights     [[buffer(0)]],  // quantized weight matrix
    const device float* input       [[buffer(1)]],  // input vector [cols]
    device float*       output      [[buffer(2)]],  // output vector [rows]
    constant MatVecParams& params   [[buffer(3)]],
    uint row_id  [[threadgroup_position_in_grid]],   // which output row
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row_id >= params.rows) return;

    const uint cols = params.cols;
    const uint blocks_per_row = params.blocks_per_row;

    // Pointer to this row's weight blocks
    const device uchar* row_data = weights + row_id * blocks_per_row * Q8_0_BYTES_PER_BLOCK;

    float sum = 0.0f;

    // Each thread processes a stride of blocks
    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        const device uchar* block = row_data + block_idx * Q8_0_BYTES_PER_BLOCK;
        float d = fp16_to_float(((const device ushort*)block)[0]);
        const device int8_t* qs = (const device int8_t*)(block + 2);

        uint base_col = block_idx * Q8_0_BLOCK_SIZE;

        // Unrolled dot product within this block
        float block_sum = 0.0f;
        for (int j = 0; j < Q8_0_BLOCK_SIZE && (base_col + j) < cols; j++) {
            block_sum += float(qs[j]) * input[base_col + j];
        }
        sum += d * block_sum;
    }

    // Cross-simdgroup reduction
    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_lane = simd_lane_id();
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[row_id] = s;
    }
}

// MARK: - Q4_0 Matrix-Vector Multiply

kernel void matvec_q4_0(
    const device uchar* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatVecParams& params   [[buffer(3)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row_id >= params.rows) return;

    const uint blocks_per_row = params.blocks_per_row;
    const device uchar* row_data = weights + row_id * blocks_per_row * Q4_0_BYTES_PER_BLOCK;

    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        const device uchar* block = row_data + block_idx * Q4_0_BYTES_PER_BLOCK;
        float d = fp16_to_float(((const device ushort*)block)[0]);

        uint base_col = block_idx * Q4_0_BLOCK_SIZE;
        float block_sum = 0.0f;

        for (int j = 0; j < Q4_0_BLOCK_SIZE && (base_col + j) < params.cols; j++) {
            int byte_idx = j / 2;
            uchar byte_val = block[2 + byte_idx];
            int nibble = (j & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
            float w = d * (float(nibble) - 8.0f);
            block_sum += w * input[base_col + j];
        }
        sum += block_sum;
    }

    // Cross-simdgroup reduction
    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_lane = simd_lane_id();
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[row_id] = s;
    }
}

// MARK: - Q4_K Matrix-Vector Multiply

kernel void matvec_q4_k(
    const device uchar* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatVecParams& params   [[buffer(3)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row_id >= params.rows) return;

    const uint blocks_per_row = params.blocks_per_row;
    const device uchar* row_data = weights + row_id * blocks_per_row * Q4_K_BYTES_PER_BLOCK;

    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        const device uchar* superblock = row_data + block_idx * Q4_K_BYTES_PER_BLOCK;
        uint base_col = block_idx * Q4_K_BLOCK_SIZE;

        for (int j = 0; j < Q4_K_BLOCK_SIZE && (base_col + j) < params.cols; j++) {
            float w = dequant_q4_k(superblock, j);
            sum += w * input[base_col + j];
        }
    }

    // Cross-simdgroup reduction
    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_lane = simd_lane_id();
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[row_id] = s;
    }
}

// MARK: - Q6_K Matrix-Vector Multiply

kernel void matvec_q6_k(
    const device uchar* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatVecParams& params   [[buffer(3)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row_id >= params.rows) return;

    const uint blocks_per_row = params.blocks_per_row;
    const device uchar* row_data = weights + row_id * blocks_per_row * Q6_K_BYTES_PER_BLOCK;

    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        const device uchar* superblock = row_data + block_idx * Q6_K_BYTES_PER_BLOCK;
        uint base_col = block_idx * Q6_K_BLOCK_SIZE;

        for (int j = 0; j < Q6_K_BLOCK_SIZE && (base_col + j) < params.cols; j++) {
            float w = dequant_q6_k(superblock, j);
            sum += w * input[base_col + j];
        }
    }

    // Cross-simdgroup reduction
    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_lane = simd_lane_id();
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[row_id] = s;
    }
}

// MARK: - F16 Matrix-Vector Multiply

kernel void matvec_f16(
    const device half*  weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatVecParams& params   [[buffer(3)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row_id >= params.rows) return;

    const uint cols = params.cols;
    const device half* row_data = weights + row_id * cols;

    float sum = 0.0f;
    for (uint j = tid; j < cols; j += tg_size) {
        sum += float(row_data[j]) * input[j];
    }

    // Cross-simdgroup reduction
    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_lane = simd_lane_id();
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[row_id] = s;
    }
}

// MARK: - F32 Matrix-Vector Multiply

kernel void matvec_f32(
    const device float* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatVecParams& params   [[buffer(3)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row_id >= params.rows) return;

    const uint cols = params.cols;
    const device float* row_data = weights + row_id * cols;

    float sum = 0.0f;
    for (uint j = tid; j < cols; j += tg_size) {
        sum += row_data[j] * input[j];
    }

    // Cross-simdgroup reduction
    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_lane = simd_lane_id();
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[row_id] = s;
    }
}
