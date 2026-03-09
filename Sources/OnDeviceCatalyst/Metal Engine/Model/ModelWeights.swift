//
//  ModelWeights.swift
//  OnDeviceCatalyst
//
//  Zero-copy weight loading: mmap the GGUF file and create MTLBuffers
//  that point directly into the mapped memory (unified memory, no copies).
//

import Foundation
import Metal

/// Loads model weights from a GGUF file with zero-copy via mmap + MTLBuffer.
public final class ModelWeights {

    /// Memory-mapped file data.
    private var mappedData: UnsafeMutableRawPointer?
    private var mappedSize: Int = 0
    private var fileDescriptor: Int32 = -1

    /// Metal device for buffer creation.
    private let device: MTLDevice

    /// Byte offset where tensor data begins in the file.
    private let dataOffset: UInt64

    /// Tensor registry: name → (buffer, offset within buffer, shape, type)
    public private(set) var tensors: [String: TensorRef] = [:]

    /// Reference to a loaded tensor.
    public struct TensorRef {
        public let buffer: MTLBuffer
        public let offset: Int
        public let dimensions: [Int]
        public let type: GGMLType
        public let byteSize: Int

        public var elementCount: Int {
            dimensions.reduce(1, *)
        }
    }

    // MARK: - Init

    /// Load weights from a parsed GGUF file.
    ///
    /// This mmaps the file and creates MTLBuffer wrappers that point directly
    /// into the mapped memory — zero copies on Apple Silicon unified memory.
    public init(path: String, gguf: GGUFFile, device: MTLDevice) throws {
        self.device = device
        self.dataOffset = gguf.dataOffset

        // Open and mmap the file
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw CatalystError.modelFileNotFound(path: path)
        }
        self.fileDescriptor = fd

        // Get file size
        var stat = stat()
        guard fstat(fd, &stat) == 0 else {
            close(fd)
            throw CatalystError.modelFileCorrupted(path: path, reason: "Failed to stat file")
        }
        let fileSize = Int(stat.st_size)
        self.mappedSize = fileSize

        // Memory-map the file (read-only)
        let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0)
        guard mapped != MAP_FAILED else {
            close(fd)
            throw CatalystError.modelFileCorrupted(path: path, reason: "Failed to mmap file")
        }
        self.mappedData = mapped

        // Create tensor registry
        for tensorInfo in gguf.tensors {
            let absoluteOffset = Int(dataOffset) + Int(tensorInfo.offset)
            let byteSize = tensorInfo.byteSize

            guard absoluteOffset + byteSize <= fileSize else {
                print("Warning: Tensor '\(tensorInfo.name)' extends beyond file bounds, skipping")
                continue
            }

            // Create an MTLBuffer that wraps the mmap'd region (zero-copy)
            let tensorPtr = mapped!.advanced(by: absoluteOffset)

            // Page-align the buffer for Metal
            let pageSize = Int(getpagesize())
            let alignedOffset = (absoluteOffset / pageSize) * pageSize
            let offsetDelta = absoluteOffset - alignedOffset
            let alignedSize = byteSize + offsetDelta
            let alignedPtr = mapped!.advanced(by: alignedOffset)

            guard let buffer = device.makeBuffer(
                bytesNoCopy: alignedPtr,
                length: alignedSize,
                options: .storageModeShared,
                deallocator: nil
            ) else {
                print("Warning: Failed to create MTLBuffer for tensor '\(tensorInfo.name)', skipping")
                continue
            }

            tensors[tensorInfo.name] = TensorRef(
                buffer: buffer,
                offset: offsetDelta,
                dimensions: tensorInfo.dimensions,
                type: tensorInfo.type,
                byteSize: byteSize
            )
        }

        print("ModelWeights: Loaded \(tensors.count) tensors via zero-copy mmap")
    }

    deinit {
        if let mapped = mappedData {
            munmap(mapped, mappedSize)
        }
        if fileDescriptor >= 0 {
            close(fileDescriptor)
        }
    }

    // MARK: - Tensor Access

    /// Get a tensor by name.
    public func tensor(_ name: String) -> TensorRef? {
        return tensors[name]
    }

    /// Get a tensor by name, throwing if not found.
    public func requireTensor(_ name: String) throws -> TensorRef {
        guard let ref = tensors[name] else {
            throw CatalystError.unknown(details: "Required tensor '\(name)' not found in model")
        }
        return ref
    }

    // MARK: - Standard GGUF Tensor Names

    /// Token embedding table.
    public var tokenEmbedding: TensorRef? {
        return tensor("token_embd.weight")
    }

    /// Output normalization weight.
    public var outputNorm: TensorRef? {
        return tensor("output_norm.weight")
    }

    /// Output (lm_head) projection weight.
    public var outputWeight: TensorRef? {
        return tensor("output.weight")
    }

    /// Attention norm for layer i.
    public func attnNorm(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).attn_norm.weight")
    }

    /// FFN norm for layer i.
    public func ffnNorm(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).ffn_norm.weight")
    }

    /// Attention Q weight for layer i.
    public func attnQ(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).attn_q.weight")
    }

    /// Attention K weight for layer i.
    public func attnK(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).attn_k.weight")
    }

    /// Attention V weight for layer i.
    public func attnV(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).attn_v.weight")
    }

    /// Attention output projection for layer i.
    public func attnOutput(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).attn_output.weight")
    }

    /// FFN gate projection for layer i.
    public func ffnGate(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).ffn_gate.weight")
    }

    /// FFN up projection for layer i.
    public func ffnUp(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).ffn_up.weight")
    }

    /// FFN down projection for layer i.
    public func ffnDown(layer: Int) -> TensorRef? {
        return tensor("blk.\(layer).ffn_down.weight")
    }
}
