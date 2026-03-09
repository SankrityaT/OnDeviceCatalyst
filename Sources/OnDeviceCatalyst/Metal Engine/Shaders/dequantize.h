//
//  dequantize.h
//  OnDeviceCatalyst
//
//  Inline dequantization functions for quantized weight formats.
//  Called from the main compute kernels.
//

#ifndef DEQUANTIZE_H
#define DEQUANTIZE_H

#include <metal_stdlib>
using namespace metal;

// MARK: - Half-precision helpers

/// Interpret a uint16 as a half-precision float.
inline float fp16_to_float(ushort bits) {
    return float(as_type<half>(bits));
}

// MARK: - Q4_0 dequantization
// Block: 32 values = 2-byte fp16 scale + 16 bytes of 4-bit values
// Each byte stores 2 values: low nibble first, high nibble second.
// Values are unsigned 0-15, centered by subtracting 8.

constant int Q4_0_BLOCK_SIZE = 32;
constant int Q4_0_BYTES_PER_BLOCK = 18;

inline float dequant_q4_0(const device uchar* block, int idx) {
    float d = fp16_to_float(((const device ushort*)block)[0]);
    int byte_idx = idx / 2;
    uchar byte_val = block[2 + byte_idx];
    int nibble = (idx & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
    return d * (float(nibble) - 8.0f);
}

// MARK: - Q8_0 dequantization
// Block: 32 values = 2-byte fp16 scale + 32 int8 values

constant int Q8_0_BLOCK_SIZE = 32;
constant int Q8_0_BYTES_PER_BLOCK = 34;

inline float dequant_q8_0(const device uchar* block, int idx) {
    float d = fp16_to_float(((const device ushort*)block)[0]);
    int8_t val = ((const device int8_t*)(block + 2))[idx];
    return d * float(val);
}

// MARK: - Q4_K dequantization
// Super-block: 256 values = 2 fp16 (d, dmin) + 12 bytes scales + 128 bytes quants
// Sub-blocks of 32 values each (8 sub-blocks per super-block).
// Each sub-block has a 6-bit scale and 6-bit min packed into the scales array.

constant int Q4_K_BLOCK_SIZE = 256;
constant int Q4_K_BYTES_PER_BLOCK = 144;

inline float dequant_q4_k(const device uchar* superblock, int idx) {
    float d    = fp16_to_float(((const device ushort*)superblock)[0]);
    float dmin = fp16_to_float(((const device ushort*)superblock)[1]);

    const device uchar* scales_data = superblock + 4;
    const device uchar* quants = superblock + 16;

    int sub_block = idx / 32;
    int sub_idx = idx % 32;

    // Extract 6-bit scale and min for this sub-block.
    // The scales array packs 8 sub-block scales and 8 sub-block mins
    // into 12 bytes using a specific bit layout.
    float sc, m;
    if (sub_block < 4) {
        sc = float(scales_data[sub_block] & 0x3F);
        m  = float(scales_data[sub_block + 4] & 0x3F);
    } else {
        sc = float((scales_data[sub_block + 4] & 0x0F) | ((scales_data[sub_block - 4] >> 6) << 4));
        m  = float((scales_data[sub_block + 4] >> 4)   | ((scales_data[sub_block]     >> 6) << 4));
    }

    // Dequantize the nibble
    int byte_idx = (sub_block * 32 + sub_idx) / 2;
    uchar byte_val = quants[byte_idx];
    int nibble = (sub_idx & 1) ? (byte_val >> 4) : (byte_val & 0x0F);

    return d * sc * float(nibble) - dmin * m;
}

// MARK: - Q6_K dequantization
// Super-block: 256 values = 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d

constant int Q6_K_BLOCK_SIZE = 256;
constant int Q6_K_BYTES_PER_BLOCK = 210;

inline float dequant_q6_k(const device uchar* superblock, int idx) {
    const device uchar* ql = superblock;
    const device uchar* qh = superblock + 128;
    const device int8_t* scales = (const device int8_t*)(superblock + 192);
    float d = fp16_to_float(((const device ushort*)(superblock + 208))[0]);

    int sub_block = idx / 16;
    int sub_idx = idx % 16;

    // Lower 4 bits from ql
    int ql_idx = (sub_block / 2) * 32 + (sub_block & 1) * 16 + sub_idx;
    uchar ql_byte = ql[ql_idx / 2];
    int q_lo = (ql_idx & 1) ? (ql_byte >> 4) : (ql_byte & 0x0F);

    // Upper 2 bits from qh
    int qh_idx = (sub_block / 4) * 32 + (sub_block & 1) * 16 + sub_idx;
    uchar qh_byte = qh[qh_idx / 2];
    int qh_val = (qh_idx & 1) ? (qh_byte >> 4) : (qh_byte & 0x0F);
    int shift = ((sub_block / 2) & 1) * 2;
    int q_hi = (qh_val >> shift) & 0x03;

    int q = q_lo | (q_hi << 4);
    float sc = float(scales[sub_block]);

    return d * sc * (float(q) - 32.0f);
}

// MARK: - F16 dequantization (trivial)

inline float dequant_f16(const device uchar* data, int idx) {
    return float(((const device half*)data)[idx]);
}

// MARK: - F32 (no dequantization needed)

inline float dequant_f32(const device uchar* data, int idx) {
    return ((const device float*)data)[idx];
}

#endif // DEQUANTIZE_H
