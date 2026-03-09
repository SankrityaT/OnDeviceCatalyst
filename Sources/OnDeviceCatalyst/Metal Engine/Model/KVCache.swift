//
//  KVCache.swift
//  OnDeviceCatalyst
//
//  Per-layer key-value cache as MTLBuffers for the attention mechanism.
//

import Foundation
import Metal

/// Manages the KV cache for all transformer layers.
public final class KVCache {

    /// Per-layer K and V buffers.
    public struct LayerCache {
        public let keyBuffer: MTLBuffer    // [maxSeqLen, numKVHeads, headDim]
        public let valueBuffer: MTLBuffer  // [maxSeqLen, numKVHeads, headDim]
    }

    /// Cache for each layer.
    public private(set) var layers: [LayerCache]

    /// Current sequence length (number of tokens cached).
    public private(set) var currentLength: Int = 0

    /// Maximum sequence length the cache can hold.
    public let maxSeqLen: Int

    /// Number of KV heads per layer.
    public let numKVHeads: Int

    /// Dimension per head.
    public let headDim: Int

    /// Bytes per position per layer (for one K or V).
    public var bytesPerPosition: Int {
        numKVHeads * headDim * MemoryLayout<Float>.size
    }

    /// Total bytes allocated across all layers (K + V).
    public var totalAllocatedBytes: Int {
        layers.count * 2 * maxSeqLen * bytesPerPosition
    }

    // MARK: - Init

    /// Allocate KV cache for all layers.
    public init(
        device: MTLDevice,
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int
    ) throws {
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.maxSeqLen = maxSeqLen

        let bufferSize = maxSeqLen * numKVHeads * headDim * MemoryLayout<Float>.size

        var layers: [LayerCache] = []
        for i in 0..<numLayers {
            guard let kBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared),
                  let vBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
                throw CatalystError.unknown(
                    details: "Failed to allocate KV cache for layer \(i) (\(bufferSize) bytes)"
                )
            }
            layers.append(LayerCache(keyBuffer: kBuf, valueBuffer: vBuf))
        }
        self.layers = layers

        let totalMB = Double(numLayers * 2 * bufferSize) / (1024 * 1024)
        print("KVCache: Allocated \(numLayers) layers × \(maxSeqLen) positions = \(String(format: "%.1f", totalMB)) MB")
    }

    // MARK: - Operations

    /// Record that a new token has been added at the current position.
    public func advancePosition() {
        currentLength = min(currentLength + 1, maxSeqLen)
    }

    /// Clear the entire cache.
    public func clear() {
        currentLength = 0
        for layer in layers {
            memset(layer.keyBuffer.contents(), 0, layer.keyBuffer.length)
            memset(layer.valueBuffer.contents(), 0, layer.valueBuffer.length)
        }
    }

    /// Remove tokens from position range [startPos, endPos).
    /// Shifts remaining tokens down to fill the gap.
    public func removeTokens(startPos: Int, endPos: Int) {
        guard startPos < endPos && startPos < currentLength else { return }

        let clampedEnd = min(endPos, currentLength)
        let tokensToShift = currentLength - clampedEnd

        if tokensToShift > 0 {
            let bytesPerPos = bytesPerPosition
            for layer in layers {
                let kSrc = layer.keyBuffer.contents().advanced(by: clampedEnd * bytesPerPos)
                let kDst = layer.keyBuffer.contents().advanced(by: startPos * bytesPerPos)
                memmove(kDst, kSrc, tokensToShift * bytesPerPos)

                let vSrc = layer.valueBuffer.contents().advanced(by: clampedEnd * bytesPerPos)
                let vDst = layer.valueBuffer.contents().advanced(by: startPos * bytesPerPos)
                memmove(vDst, vSrc, tokensToShift * bytesPerPos)
            }
        }

        currentLength = startPos + tokensToShift
    }

    /// Check if the cache can accommodate additional tokens.
    public func canAccommodate(additionalTokens: Int) -> Bool {
        return currentLength + additionalTokens <= maxSeqLen
    }

    /// Evict the oldest fraction of cached tokens to free memory.
    /// Shifts remaining tokens down to fill the gap.
    public func evictOldest(fraction: Float) {
        guard currentLength > 0 else { return }
        let tokensToEvict = max(1, Int(Float(currentLength) * fraction))
        let evictEnd = min(tokensToEvict, currentLength)

        print("KVCache: Evicting oldest \(evictEnd) of \(currentLength) cached tokens")
        removeTokens(startPos: 0, endPos: evictEnd)
    }

    /// Write K and V vectors for a specific layer and position.
    /// This is used by the CPU side; the GPU writes directly via compute shaders.
    public func writeKV(layer: Int, position: Int, key: UnsafePointer<Float>, value: UnsafePointer<Float>) {
        let bytesPerPos = bytesPerPosition
        let offset = position * bytesPerPos

        let kDst = layers[layer].keyBuffer.contents().advanced(by: offset)
        memcpy(kDst, key, bytesPerPos)

        let vDst = layers[layer].valueBuffer.contents().advanced(by: offset)
        memcpy(vDst, value, bytesPerPos)
    }
}
