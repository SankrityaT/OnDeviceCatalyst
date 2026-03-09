//
//  GGUFParser.swift
//  OnDeviceCatalyst
//
//  Parses GGUF v3 model files to extract metadata and tensor descriptors.
//  This enables loading models without depending on llama.cpp.
//

import Foundation

/// Parses GGUF model files.
public final class GGUFParser {

    /// Expected GGUF magic bytes.
    private static let magic: UInt32 = 0x46554747 // "GGUF" in little-endian

    /// Default alignment for the data section.
    private static let defaultAlignment: Int = 32

    // MARK: - Public API

    /// Parse a GGUF file and return its contents.
    ///
    /// This reads the header, metadata, and tensor descriptors but does NOT
    /// load the actual tensor data into memory. Use `ModelWeights` for that.
    public static func parse(path: String) throws -> GGUFFile {
        guard FileManager.default.fileExists(atPath: path) else {
            throw CatalystError.modelFileNotFound(path: path)
        }

        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe) else {
            throw CatalystError.modelFileCorrupted(path: path, reason: "Failed to memory-map file")
        }

        var reader = BinaryReader(data: data)

        // 1. Parse header
        let fileMagic = try reader.readUInt32()
        guard fileMagic == magic else {
            throw CatalystError.modelFileCorrupted(
                path: path,
                reason: "Invalid GGUF magic: expected 0x46554747, got 0x\(String(fileMagic, radix: 16))"
            )
        }

        let version = try reader.readUInt32()
        guard version >= 2 && version <= 3 else {
            throw CatalystError.modelFileCorrupted(
                path: path,
                reason: "Unsupported GGUF version: \(version) (expected 2 or 3)"
            )
        }

        let tensorCount = try reader.readUInt64()
        let metadataKVCount = try reader.readUInt64()

        // 2. Parse metadata
        var metadata: [String: GGUFValue] = [:]
        for _ in 0..<metadataKVCount {
            let key = try reader.readGGUFString()
            let value = try reader.readGGUFValue()
            metadata[key] = value
        }

        // 3. Parse tensor descriptors
        var tensors: [GGUFTensorInfo] = []
        for _ in 0..<tensorCount {
            let tensor = try reader.readTensorInfo()
            tensors.append(tensor)
        }

        // 4. Calculate data offset (aligned)
        let alignment = metadata["general.alignment"]?.asInt ?? defaultAlignment
        let currentOffset = reader.offset
        let dataOffset = alignUp(currentOffset, to: alignment)

        // 5. Extract transformer config
        let config = extractConfig(from: metadata)

        return GGUFFile(
            version: version,
            metadata: metadata,
            tensors: tensors,
            dataOffset: UInt64(dataOffset),
            config: config
        )
    }

    // MARK: - Config Extraction

    /// Extract transformer hyperparameters from GGUF metadata.
    private static func extractConfig(from metadata: [String: GGUFValue]) -> TransformerConfig? {
        // Determine architecture
        guard let arch = metadata["general.architecture"]?.asString else {
            return nil
        }

        // Read hyperparameters using architecture-prefixed keys
        guard let vocabSize = metadata["\(arch).vocab_size"]?.asInt
                ?? metadata["tokenizer.ggml.tokens"]?.asStringArray.map({ $0.count }),
              let hiddenSize = metadata["\(arch).embedding_length"]?.asInt,
              let numLayers = metadata["\(arch).block_count"]?.asInt,
              let numHeads = metadata["\(arch).attention.head_count"]?.asInt else {
            return nil
        }

        let numKVHeads = metadata["\(arch).attention.head_count_kv"]?.asInt ?? numHeads
        let intermediateSize = metadata["\(arch).feed_forward_length"]?.asInt ?? (hiddenSize * 4)
        let rmsNormEps = metadata["\(arch).attention.layer_norm_rms_epsilon"]?.asFloat ?? 1e-5
        let ropeFreqBase = metadata["\(arch).rope.freq_base"]?.asFloat ?? 10000.0

        return TransformerConfig(
            architecture: arch,
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            intermediateSize: intermediateSize,
            rmsNormEps: rmsNormEps,
            ropeFreqBase: ropeFreqBase
        )
    }

    // MARK: - Helpers

    private static func alignUp(_ offset: Int, to alignment: Int) -> Int {
        return (offset + alignment - 1) / alignment * alignment
    }
}

// MARK: - Binary Reader

/// Sequential binary reader for parsing GGUF files.
internal struct BinaryReader {
    let data: Data
    private(set) var offset: Int = 0

    init(data: Data) {
        self.data = data
    }

    // MARK: - Primitives

    mutating func readUInt8() throws -> UInt8 {
        guard offset + 1 <= data.count else { throw readerError("UInt8") }
        let value = data[offset]
        offset += 1
        return value
    }

    mutating func readInt8() throws -> Int8 {
        return Int8(bitPattern: try readUInt8())
    }

    mutating func readUInt16() throws -> UInt16 {
        return try readRaw(UInt16.self)
    }

    mutating func readInt16() throws -> Int16 {
        return try readRaw(Int16.self)
    }

    mutating func readUInt32() throws -> UInt32 {
        return try readRaw(UInt32.self)
    }

    mutating func readInt32() throws -> Int32 {
        return try readRaw(Int32.self)
    }

    mutating func readUInt64() throws -> UInt64 {
        return try readRaw(UInt64.self)
    }

    mutating func readInt64() throws -> Int64 {
        return try readRaw(Int64.self)
    }

    mutating func readFloat32() throws -> Float {
        return try readRaw(Float.self)
    }

    mutating func readFloat64() throws -> Double {
        return try readRaw(Double.self)
    }

    mutating func readBool() throws -> Bool {
        let byte = try readUInt8()
        return byte != 0
    }

    private mutating func readRaw<T>(_ type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        guard offset + size <= data.count else { throw readerError(String(describing: T.self)) }

        let value = data[offset..<offset + size].withUnsafeBytes { ptr in
            ptr.loadUnaligned(as: T.self)
        }
        offset += size
        return value
    }

    // MARK: - GGUF Strings

    /// Read a GGUF string (uint64 length + UTF-8 bytes, no null terminator).
    mutating func readGGUFString() throws -> String {
        let length = try readUInt64()
        guard length < 1_000_000 else { throw readerError("string (length \(length) too large)") }

        let byteCount = Int(length)
        guard offset + byteCount <= data.count else { throw readerError("string data") }

        let bytes = data[offset..<offset + byteCount]
        offset += byteCount

        guard let str = String(data: bytes, encoding: .utf8) else {
            throw readerError("string (invalid UTF-8)")
        }
        return str
    }

    // MARK: - GGUF Values

    /// Read a typed GGUF metadata value.
    mutating func readGGUFValue() throws -> GGUFValue {
        let rawType = try readUInt32()
        guard let valueType = GGUFMetadataValueType(rawValue: rawType) else {
            throw readerError("unknown value type \(rawType)")
        }
        return try readValue(ofType: valueType)
    }

    private mutating func readValue(ofType type: GGUFMetadataValueType) throws -> GGUFValue {
        switch type {
        case .uint8:   return .uint8(try readUInt8())
        case .int8:    return .int8(try readInt8())
        case .uint16:  return .uint16(try readUInt16())
        case .int16:   return .int16(try readInt16())
        case .uint32:  return .uint32(try readUInt32())
        case .int32:   return .int32(try readInt32())
        case .float32: return .float32(try readFloat32())
        case .bool:    return .bool(try readBool())
        case .string:  return .string(try readGGUFString())
        case .uint64:  return .uint64(try readUInt64())
        case .int64:   return .int64(try readInt64())
        case .float64: return .float64(try readFloat64())
        case .array:
            let elementTypeRaw = try readUInt32()
            guard let elementType = GGUFMetadataValueType(rawValue: elementTypeRaw) else {
                throw readerError("unknown array element type \(elementTypeRaw)")
            }
            let count = try readUInt64()
            guard count < 10_000_000 else { throw readerError("array too large (\(count))") }

            var elements: [GGUFValue] = []
            elements.reserveCapacity(Int(count))
            for _ in 0..<count {
                elements.append(try readValue(ofType: elementType))
            }
            return .array(elements)
        }
    }

    // MARK: - Tensor Info

    /// Read a GGUF tensor descriptor.
    mutating func readTensorInfo() throws -> GGUFTensorInfo {
        let name = try readGGUFString()
        let nDims = try readUInt32()

        var dims: [Int] = []
        for _ in 0..<nDims {
            let dim = try readUInt64()
            dims.append(Int(dim))
        }

        let rawType = try readUInt32()
        guard let type = GGMLType(rawValue: rawType) else {
            throw readerError("unknown tensor type \(rawType) for '\(name)'")
        }

        let offset = try readUInt64()

        return GGUFTensorInfo(
            name: name,
            dimensions: dims,
            type: type,
            offset: offset
        )
    }

    // MARK: - Error

    private func readerError(_ context: String) -> CatalystError {
        return CatalystError.modelFileCorrupted(
            path: "<binary>",
            reason: "Failed to read \(context) at offset \(offset)"
        )
    }
}
