//
//  shader_types.h
//  OnDeviceCatalyst
//
//  Shared data structures between Swift and Metal shaders.
//

#ifndef SHADER_TYPES_H
#define SHADER_TYPES_H

#include <simd/simd.h>

// MARK: - Quantization block structures

// Q4_0: 32 values per block = 18 bytes
// Layout: fp16 scale (2 bytes) + 16 bytes of nibbles (32 × 4-bit)
struct BlockQ4_0 {
    ushort scale;       // fp16 scale factor (d)
    uchar  quants[16];  // 4-bit quantized values, 2 per byte
};

// Q8_0: 32 values per block = 34 bytes
// Layout: fp16 scale (2 bytes) + 32 int8 values
struct BlockQ8_0 {
    ushort scale;       // fp16 scale factor (d)
    char   quants[32];  // 8-bit quantized values
};

// Q4_K: 256 values per super-block = 144 bytes
// Layout: fp16 d (2) + fp16 dmin (2) + 12 bytes scales + 128 bytes quants
struct BlockQ4_K {
    ushort d;           // super-block scale
    ushort dmin;        // super-block minimum
    uchar  scales[12];  // 6-bit sub-block scales and mins
    uchar  quants[128]; // 4-bit quantized values
};

// Q6_K: 256 values per super-block = 210 bytes
struct BlockQ6_K {
    uchar  ql[128];     // lower 4 bits of 6-bit quants
    uchar  qh[64];      // upper 2 bits of 6-bit quants
    char   scales[16];  // int8 scales
    ushort d;           // fp16 super-block scale
};

// MARK: - Kernel parameter structures

struct MatVecParams {
    uint rows;          // output dimension
    uint cols;          // input dimension
    uint quant_type;    // GGMLType raw value
    uint block_size;    // elements per quant block
    uint blocks_per_row; // number of quant blocks per row
};

struct NormParams {
    uint  size;         // vector dimension
    float eps;          // normalization epsilon
};

struct RoPEParams {
    uint  head_dim;     // dimension per head
    uint  num_heads;    // number of heads to process
    uint  position;     // absolute position in sequence
    float freq_base;    // base frequency (usually 10000.0)
    float freq_scale;   // frequency scaling factor (1.0 for standard)
};

struct AttentionParams {
    uint  num_heads;      // number of query heads
    uint  num_kv_heads;   // number of key-value heads (for GQA)
    uint  head_dim;       // dimension per head
    uint  seq_len;        // current sequence length (for K/V cache)
    float scale;          // attention scale (1/sqrt(head_dim))
};

struct EmbeddingParams {
    uint  vocab_size;     // vocabulary size
    uint  hidden_size;    // embedding dimension
    uint  token_id;       // token to look up
};

struct ElementwiseParams {
    uint count;           // number of elements
};

struct KVCacheParams {
    uint  num_kv_heads;
    uint  head_dim;
    uint  position;       // where to write in the cache
    uint  max_seq_len;    // max cache size
};

struct MatMulParams {
    uint  rows;           // output dimension
    uint  cols;           // input dimension
    uint  batch_size;     // number of tokens
    uint  blocks_per_row; // for quantized formats
};

#endif // SHADER_TYPES_H
