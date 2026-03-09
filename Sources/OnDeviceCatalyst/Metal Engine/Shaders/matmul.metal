//
//  matmul.metal
//  OnDeviceCatalyst
//
//  Batched matrix-matrix multiply for prompt prefill.
//  Y[batch, row] = sum_col(W[row, col] * X[batch, col])
//
//  Strategy: "batched matvec" — each threadgroup handles one (row, batch_tile).
//  Threads within the group split cols. Multiple batch tokens processed per row dispatch.
//

#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"
#include "dequantize.h"

// MatMulParams is defined in shader_types.h

// MARK: - Q8_0 Batched Matmul

kernel void matmul_q8_0(
    const device uchar* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatMulParams& params   [[buffer(3)]],
    uint2 gid    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint row_id = gid.x;
    uint batch_idx = gid.y;
    if (row_id >= params.rows || batch_idx >= params.batch_size) return;

    const uint cols = params.cols;
    const uint blocks_per_row = params.blocks_per_row;
    const device uchar* row_data = weights + row_id * blocks_per_row * Q8_0_BYTES_PER_BLOCK;
    const device float* x = input + batch_idx * cols;

    float sum = 0.0f;
    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        const device uchar* block = row_data + block_idx * Q8_0_BYTES_PER_BLOCK;
        float d = fp16_to_float(((const device ushort*)block)[0]);
        const device int8_t* qs = (const device int8_t*)(block + 2);
        uint base_col = block_idx * Q8_0_BLOCK_SIZE;

        float block_sum = 0.0f;
        for (int j = 0; j < Q8_0_BLOCK_SIZE && (base_col + j) < cols; j++) {
            block_sum += float(qs[j]) * x[base_col + j];
        }
        sum += d * block_sum;
    }

    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[batch_idx * params.rows + row_id] = s;
    }
}

// MARK: - Q4_0 Batched Matmul

kernel void matmul_q4_0(
    const device uchar* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatMulParams& params   [[buffer(3)]],
    uint2 gid    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint row_id = gid.x;
    uint batch_idx = gid.y;
    if (row_id >= params.rows || batch_idx >= params.batch_size) return;

    const uint blocks_per_row = params.blocks_per_row;
    const device uchar* row_data = weights + row_id * blocks_per_row * Q4_0_BYTES_PER_BLOCK;
    const device float* x = input + batch_idx * params.cols;

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
            block_sum += w * x[base_col + j];
        }
        sum += block_sum;
    }

    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[batch_idx * params.rows + row_id] = s;
    }
}

// MARK: - Q4_K Batched Matmul

kernel void matmul_q4_k(
    const device uchar* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatMulParams& params   [[buffer(3)]],
    uint2 gid    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint row_id = gid.x;
    uint batch_idx = gid.y;
    if (row_id >= params.rows || batch_idx >= params.batch_size) return;

    const uint blocks_per_row = params.blocks_per_row;
    const device uchar* row_data = weights + row_id * blocks_per_row * Q4_K_BYTES_PER_BLOCK;
    const device float* x = input + batch_idx * params.cols;

    float sum = 0.0f;
    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        const device uchar* superblock = row_data + block_idx * Q4_K_BYTES_PER_BLOCK;
        uint base_col = block_idx * Q4_K_BLOCK_SIZE;

        for (int j = 0; j < Q4_K_BLOCK_SIZE && (base_col + j) < params.cols; j++) {
            float w = dequant_q4_k(superblock, j);
            sum += w * x[base_col + j];
        }
    }

    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[batch_idx * params.rows + row_id] = s;
    }
}

// MARK: - Q6_K Batched Matmul

kernel void matmul_q6_k(
    const device uchar* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatMulParams& params   [[buffer(3)]],
    uint2 gid    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint row_id = gid.x;
    uint batch_idx = gid.y;
    if (row_id >= params.rows || batch_idx >= params.batch_size) return;

    const uint blocks_per_row = params.blocks_per_row;
    const device uchar* row_data = weights + row_id * blocks_per_row * Q6_K_BYTES_PER_BLOCK;
    const device float* x = input + batch_idx * params.cols;

    float sum = 0.0f;
    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        const device uchar* superblock = row_data + block_idx * Q6_K_BYTES_PER_BLOCK;
        uint base_col = block_idx * Q6_K_BLOCK_SIZE;

        for (int j = 0; j < Q6_K_BLOCK_SIZE && (base_col + j) < params.cols; j++) {
            float w = dequant_q6_k(superblock, j);
            sum += w * x[base_col + j];
        }
    }

    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[batch_idx * params.rows + row_id] = s;
    }
}

// MARK: - F16 Batched Matmul

kernel void matmul_f16(
    const device half*  weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatMulParams& params   [[buffer(3)]],
    uint2 gid    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint row_id = gid.x;
    uint batch_idx = gid.y;
    if (row_id >= params.rows || batch_idx >= params.batch_size) return;

    const uint cols = params.cols;
    const device half* row_data = weights + row_id * cols;
    const device float* x = input + batch_idx * cols;

    float sum = 0.0f;
    for (uint j = tid; j < cols; j += tg_size) {
        sum += float(row_data[j]) * x[j];
    }

    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[batch_idx * params.rows + row_id] = s;
    }
}

// MARK: - F32 Batched Matmul

kernel void matmul_f32(
    const device float* weights     [[buffer(0)]],
    const device float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant MatMulParams& params   [[buffer(3)]],
    uint2 gid    [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]]
) {
    uint row_id = gid.x;
    uint batch_idx = gid.y;
    if (row_id >= params.rows || batch_idx >= params.batch_size) return;

    const uint cols = params.cols;
    const device float* row_data = weights + row_id * cols;
    const device float* x = input + batch_idx * cols;

    float sum = 0.0f;
    for (uint j = tid; j < cols; j += tg_size) {
        sum += row_data[j] * x[j];
    }

    sum = simd_sum(sum);
    threadgroup float shared_partial[8];
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        uint n_sg = (tg_size + 31) / 32;
        float s = (simd_lane < n_sg) ? shared_partial[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) output[batch_idx * params.rows + row_id] = s;
    }
}
