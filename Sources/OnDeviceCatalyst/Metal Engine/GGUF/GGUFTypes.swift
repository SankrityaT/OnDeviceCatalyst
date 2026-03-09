//
//  GGUFTypes.swift
//  OnDeviceCatalyst
//
//  Data types for the GGUF file format (v3).
//  Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//

import Foundation

// MARK: - Quantization Types

/// GGML tensor quantization types matching the GGUF spec.
public enum GGMLType: UInt32, CaseIterable {
    case f32      = 0
    case f16      = 1
    case q4_0     = 2
    case q4_1     = 3
    case q5_0     = 6
    case q5_1     = 7
    case q8_0     = 8
    case q8_1     = 9
    case q2_K     = 10
    case q3_K     = 11
    case q4_K     = 12
    case q5_K     = 13
    case q6_K     = 14
    case iq2_xxs  = 16
    case iq2_xs   = 17
    case iq3_xxs  = 18
    case iq1_s    = 19
    case iq4_nl   = 20
    case iq3_s    = 21
    case iq2_s    = 22
    case iq4_xs   = 23
    case i8       = 24
    case i16      = 25
    case i32      = 26
    case i64      = 27
    case f64      = 28
    case iq1_m    = 29
    case bf16     = 30

    /// Number of elements per quantization block.
    public var blockSize: Int {
        switch self {
        case .f32, .f16, .bf16, .f64: return 1
        case .i8, .i16, .i32, .i64:  return 1
        case .q4_0, .q4_1:           return 32
        case .q5_0, .q5_1:           return 32
        case .q8_0, .q8_1:           return 32
        case .q2_K:                  return 256
        case .q3_K:                  return 256
        case .q4_K:                  return 256
        case .q5_K:                  return 256
        case .q6_K:                  return 256
        case .iq2_xxs, .iq2_xs, .iq2_s: return 256
        case .iq3_xxs, .iq3_s:      return 256
        case .iq1_s, .iq1_m:        return 256
        case .iq4_nl, .iq4_xs:      return 32
        }
    }

    /// Byte size of one quantization block.
    public var bytesPerBlock: Int {
        switch self {
        case .f32:    return 4
        case .f16:    return 2
        case .bf16:   return 2
        case .f64:    return 8
        case .i8:     return 1
        case .i16:    return 2
        case .i32:    return 4
        case .i64:    return 8
        case .q4_0:   return 18   // 2 (scale) + 16 (4-bit × 32 / 8)
        case .q4_1:   return 20   // 2 (scale) + 2 (min) + 16
        case .q5_0:   return 22   // 2 + 4 (high bits) + 16
        case .q5_1:   return 24   // 2 + 2 + 4 + 16
        case .q8_0:   return 34   // 2 (scale) + 32 (int8 × 32)
        case .q8_1:   return 36   // 2 + 2 + 32
        case .q2_K:   return 84
        case .q3_K:   return 110
        case .q4_K:   return 144  // super-block: scales + mins + quants for 256 values
        case .q5_K:   return 176
        case .q6_K:   return 210
        case .iq2_xxs: return 66
        case .iq2_xs:  return 74
        case .iq2_s:   return 82
        case .iq3_xxs: return 98
        case .iq3_s:   return 110
        case .iq1_s:   return 50
        case .iq1_m:   return 56
        case .iq4_nl:  return 18
        case .iq4_xs:  return 36
        }
    }

    /// Whether this type is supported by our Metal shaders (initially).
    public var isSupported: Bool {
        switch self {
        case .f32, .f16, .q4_0, .q8_0, .q4_K, .q6_K:
            return true
        default:
            return false
        }
    }
}

// MARK: - GGUF Metadata Value Types

/// Value types that can appear in GGUF metadata key-value pairs.
public enum GGUFMetadataValueType: UInt32 {
    case uint8   = 0
    case int8    = 1
    case uint16  = 2
    case int16   = 3
    case uint32  = 4
    case int32   = 5
    case float32 = 6
    case bool    = 7
    case string  = 8
    case array   = 9
    case uint64  = 10
    case int64   = 11
    case float64 = 12
}

/// A parsed GGUF metadata value.
public enum GGUFValue {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case bool(Bool)
    case string(String)
    case array([GGUFValue])
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)

    /// Get as UInt32 (with coercion from smaller integer types).
    public var asUInt32: UInt32? {
        switch self {
        case .uint32(let v): return v
        case .uint64(let v): return UInt32(v)
        case .int32(let v):  return UInt32(v)
        case .uint16(let v): return UInt32(v)
        case .uint8(let v):  return UInt32(v)
        default: return nil
        }
    }

    /// Get as Int.
    public var asInt: Int? {
        switch self {
        case .uint32(let v): return Int(v)
        case .int32(let v):  return Int(v)
        case .uint64(let v): return Int(v)
        case .int64(let v):  return Int(v)
        case .uint16(let v): return Int(v)
        case .int16(let v):  return Int(v)
        case .uint8(let v):  return Int(v)
        case .int8(let v):   return Int(v)
        default: return nil
        }
    }

    /// Get as Float.
    public var asFloat: Float? {
        switch self {
        case .float32(let v): return v
        case .float64(let v): return Float(v)
        default: return nil
        }
    }

    /// Get as String.
    public var asString: String? {
        switch self {
        case .string(let v): return v
        default: return nil
        }
    }

    /// Get as Bool.
    public var asBool: Bool? {
        switch self {
        case .bool(let v): return v
        default: return nil
        }
    }

    /// Get as array of strings.
    public var asStringArray: [String]? {
        switch self {
        case .array(let arr):
            return arr.compactMap { $0.asString }
        default:
            return nil
        }
    }

    /// Get as array of floats.
    public var asFloatArray: [Float]? {
        switch self {
        case .array(let arr):
            return arr.compactMap { $0.asFloat }
        default:
            return nil
        }
    }
}

// MARK: - Tensor Info

/// Describes one tensor in the GGUF file.
public struct GGUFTensorInfo {
    /// Tensor name (e.g., "blk.0.attn_q.weight").
    public let name: String

    /// Tensor dimensions (e.g., [4096, 4096] for a square weight matrix).
    public let dimensions: [Int]

    /// Quantization type.
    public let type: GGMLType

    /// Byte offset from the start of the data section.
    public let offset: UInt64

    /// Total number of elements in the tensor.
    public var elementCount: Int {
        dimensions.reduce(1, *)
    }

    /// Total byte size of the tensor data.
    public var byteSize: Int {
        let blocks = (elementCount + type.blockSize - 1) / type.blockSize
        return blocks * type.bytesPerBlock
    }
}

// MARK: - Transformer Config

/// Model hyperparameters extracted from GGUF metadata.
public struct TransformerConfig {
    /// Architecture name (e.g., "llama", "qwen2")
    public let architecture: String

    /// Vocabulary size
    public let vocabSize: Int

    /// Hidden dimension (embedding length)
    public let hiddenSize: Int

    /// Number of transformer layers
    public let numLayers: Int

    /// Number of attention heads
    public let numHeads: Int

    /// Number of key-value heads (for GQA; equals numHeads for MHA)
    public let numKVHeads: Int

    /// FFN intermediate dimension
    public let intermediateSize: Int

    /// RMS normalization epsilon
    public let rmsNormEps: Float

    /// RoPE frequency base
    public let ropeFreqBase: Float

    /// Computed head dimension
    public var headDim: Int {
        hiddenSize / numHeads
    }

    /// Number of query heads per KV head group (for GQA)
    public var numQueryGroups: Int {
        numHeads / numKVHeads
    }

    /// Attention scaling factor (1 / sqrt(headDim))
    public var attentionScale: Float {
        1.0 / sqrt(Float(headDim))
    }
}

// MARK: - Parsed GGUF File

/// Complete parsed representation of a GGUF file.
public struct GGUFFile {
    /// GGUF format version.
    public let version: UInt32

    /// All metadata key-value pairs.
    public let metadata: [String: GGUFValue]

    /// All tensor descriptors.
    public let tensors: [GGUFTensorInfo]

    /// Byte offset where the tensor data section starts.
    public let dataOffset: UInt64

    /// Extracted transformer config (nil if metadata is incomplete).
    public let config: TransformerConfig?
}
