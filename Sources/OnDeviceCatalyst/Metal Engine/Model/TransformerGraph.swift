//
//  TransformerGraph.swift
//  OnDeviceCatalyst
//
//  Orchestrates the full transformer forward pass on the GPU.
//  Single command buffer per token (decode) or per prompt (prefill).
//

import Foundation
import Metal

/// Executes the transformer forward pass using Metal compute shaders.
public final class TransformerGraph {

    private let engine: MetalComputeEngine
    private let weights: ModelWeights
    private let config: TransformerConfig
    private let kvCache: KVCache

    // MARK: - Scratch Buffers (Decode)

    private let residual: MTLBuffer      // [hiddenSize]
    private let normed: MTLBuffer        // [hiddenSize]
    private let qBuf: MTLBuffer          // [numHeads * headDim]
    private let kBuf: MTLBuffer          // [numKVHeads * headDim]
    private let vBuf: MTLBuffer          // [numKVHeads * headDim]
    private let attnOut: MTLBuffer       // [numHeads * headDim]
    private let attnProj: MTLBuffer      // [hiddenSize]
    private let gateBuf: MTLBuffer       // [intermediateSize]
    private let upBuf: MTLBuffer         // [intermediateSize]
    private let ffnIntermediate: MTLBuffer // [intermediateSize]
    private let ffnOut: MTLBuffer        // [hiddenSize]
    private let logitsBuf: MTLBuffer     // [vocabSize]

    /// Expose the final hidden state (pre-logit norm output) for embedding extraction.
    public var lastHiddenState: MTLBuffer { normed }

    // MARK: - Init

    public init(
        engine: MetalComputeEngine,
        weights: ModelWeights,
        config: TransformerConfig,
        kvCache: KVCache
    ) throws {
        self.engine = engine
        self.weights = weights
        self.config = config
        self.kvCache = kvCache

        let h = config.hiddenSize
        let ff = config.intermediateSize
        let v = config.vocabSize
        let floatSize = MemoryLayout<Float>.size

        guard let residual = engine.makeBuffer(length: h * floatSize),
              let normed = engine.makeBuffer(length: h * floatSize),
              let qBuf = engine.makeBuffer(length: config.numHeads * config.headDim * floatSize),
              let kBuf = engine.makeBuffer(length: config.numKVHeads * config.headDim * floatSize),
              let vBuf = engine.makeBuffer(length: config.numKVHeads * config.headDim * floatSize),
              let attnOut = engine.makeBuffer(length: config.numHeads * config.headDim * floatSize),
              let attnProj = engine.makeBuffer(length: h * floatSize),
              let gateBuf = engine.makeBuffer(length: ff * floatSize),
              let upBuf = engine.makeBuffer(length: ff * floatSize),
              let ffnIntermediate = engine.makeBuffer(length: ff * floatSize),
              let ffnOut = engine.makeBuffer(length: h * floatSize),
              let logitsBuf = engine.makeBuffer(length: v * floatSize) else {
            throw CatalystError.unknown(details: "Failed to allocate scratch buffers for transformer")
        }

        self.residual = residual
        self.normed = normed
        self.qBuf = qBuf
        self.kBuf = kBuf
        self.vBuf = vBuf
        self.attnOut = attnOut
        self.attnProj = attnProj
        self.gateBuf = gateBuf
        self.upBuf = upBuf
        self.ffnIntermediate = ffnIntermediate
        self.ffnOut = ffnOut
        self.logitsBuf = logitsBuf
    }

    // MARK: - Single-Token Decode (6.1: single command buffer)

    /// Run a single-token decode step in a single command buffer.
    /// Returns pointer to the logits buffer (vocab-sized float array).
    public func decodeToken(tokenId: Int, position: Int) throws -> UnsafeMutablePointer<Float> {
        guard let cmdBuffer = engine.commandQueue.makeCommandBuffer() else {
            throw CatalystError.unknown(details: "Failed to create Metal command buffer")
        }
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw CatalystError.unknown(details: "Failed to create Metal compute encoder")
        }

        // Step 1: Token embedding lookup → residual
        if let emb = weights.tokenEmbedding {
            engine.encodeEmbedding(
                encoder: encoder,
                table: emb.buffer,
                tableOffset: emb.offset,
                output: residual,
                tokenId: tokenId,
                hiddenSize: config.hiddenSize,
                isF16: emb.type == .f16
            )
        }

        // Step 2: Process each transformer layer in single command buffer
        for layer in 0..<config.numLayers {
            try encodeDecoderLayer(encoder: encoder, layer: layer, position: position)
        }

        // Step 3: Final RMS norm + output projection → logits
        try encodeFinalProjection(encoder: encoder)

        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if let error = cmdBuffer.error {
            throw CatalystError.unknown(details: "Metal command buffer failed: \(error.localizedDescription)")
        }

        kvCache.advancePosition()

        return logitsBuf.contents().assumingMemoryBound(to: Float.self)
    }

    // MARK: - Batch Prefill (6.4)

    /// Process multiple prompt tokens in a single pass using batch matmul.
    /// Returns pointer to the logits buffer for the last token.
    public func prefillTokens(tokenIds: [Int], startPosition: Int) throws -> UnsafeMutablePointer<Float> {
        let batchSize = tokenIds.count
        guard batchSize > 0 else {
            throw CatalystError.unknown(details: "Empty token list for prefill")
        }

        if batchSize == 1 {
            return try decodeToken(tokenId: tokenIds[0], position: startPosition)
        }

        let h = config.hiddenSize
        let ff = config.intermediateSize
        let floatSize = MemoryLayout<Float>.size
        let qDim = config.numHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim

        // Allocate batch scratch buffers
        guard let batchResidual = engine.makeBuffer(length: batchSize * h * floatSize),
              let batchNormed = engine.makeBuffer(length: batchSize * h * floatSize),
              let batchQ = engine.makeBuffer(length: batchSize * qDim * floatSize),
              let batchK = engine.makeBuffer(length: batchSize * kvDim * floatSize),
              let batchV = engine.makeBuffer(length: batchSize * kvDim * floatSize),
              let batchAttnOut = engine.makeBuffer(length: batchSize * qDim * floatSize),
              let batchAttnProj = engine.makeBuffer(length: batchSize * h * floatSize),
              let batchGate = engine.makeBuffer(length: batchSize * ff * floatSize),
              let batchUp = engine.makeBuffer(length: batchSize * ff * floatSize),
              let batchFFNIntermediate = engine.makeBuffer(length: batchSize * ff * floatSize),
              let batchFFNOut = engine.makeBuffer(length: batchSize * h * floatSize),
              let tokenIdsBuf = engine.makeBuffer(length: batchSize * MemoryLayout<UInt32>.size),
              let positionsBuf = engine.makeBuffer(length: batchSize * MemoryLayout<UInt32>.size) else {
            throw CatalystError.unknown(details: "Failed to allocate batch prefill buffers")
        }

        // Fill token IDs and positions buffers
        let tokenIdsPtr = tokenIdsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let positionsPtr = positionsBuf.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<batchSize {
            tokenIdsPtr[i] = UInt32(tokenIds[i])
            positionsPtr[i] = UInt32(startPosition + i)
        }

        guard let cmdBuffer = engine.commandQueue.makeCommandBuffer() else {
            throw CatalystError.unknown(details: "Failed to create Metal command buffer")
        }
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw CatalystError.unknown(details: "Failed to create Metal compute encoder")
        }

        // Step 1: Batch embedding lookup
        if let emb = weights.tokenEmbedding {
            engine.encodeBatchEmbedding(
                encoder: encoder,
                table: emb.buffer,
                tableOffset: emb.offset,
                output: batchResidual,
                tokenIds: tokenIdsBuf,
                hiddenSize: h,
                batchSize: batchSize,
                isF16: emb.type == .f16
            )
        }

        // Step 2: Process each transformer layer
        for layer in 0..<config.numLayers {
            // Attention RMSNorm (per token)
            guard let attnNorm = weights.attnNorm(layer: layer) else {
                throw CatalystError.unknown(details: "Missing attn_norm weight for layer \(layer)")
            }
            for b in 0..<batchSize {
                engine.encodeBatchRMSNorm(
                    encoder: encoder,
                    input: batchResidual, inputOffset: b * h * floatSize,
                    output: batchNormed, outputOffset: b * h * floatSize,
                    weight: attnNorm.buffer, weightOffset: attnNorm.offset,
                    size: h, eps: config.rmsNormEps
                )
            }

            // Batch Q/K/V projections
            guard let wQ = weights.attnQ(layer: layer),
                  let wK = weights.attnK(layer: layer),
                  let wV = weights.attnV(layer: layer) else {
                throw CatalystError.unknown(details: "Missing Q/K/V weights for layer \(layer)")
            }

            engine.encodeMatmul(encoder: encoder, weights: wQ.buffer, weightsOffset: wQ.offset,
                               input: batchNormed, output: batchQ,
                               rows: qDim, cols: h, batchSize: batchSize, quantType: wQ.type)
            engine.encodeMatmul(encoder: encoder, weights: wK.buffer, weightsOffset: wK.offset,
                               input: batchNormed, output: batchK,
                               rows: kvDim, cols: h, batchSize: batchSize, quantType: wK.type)
            engine.encodeMatmul(encoder: encoder, weights: wV.buffer, weightsOffset: wV.offset,
                               input: batchNormed, output: batchV,
                               rows: kvDim, cols: h, batchSize: batchSize, quantType: wV.type)

            // Batch RoPE
            engine.encodeBatchRoPE(encoder: encoder, qk: batchQ, positions: positionsBuf,
                                  headDim: config.headDim, numHeads: config.numHeads,
                                  batchSize: batchSize, freqBase: config.ropeFreqBase)
            engine.encodeBatchRoPE(encoder: encoder, qk: batchK, positions: positionsBuf,
                                  headDim: config.headDim, numHeads: config.numKVHeads,
                                  batchSize: batchSize, freqBase: config.ropeFreqBase)

            // KV cache copy for each token in batch (with src offset)
            for b in 0..<batchSize {
                let kvOffset = b * kvDim * floatSize
                engine.encodeKVCacheCopy(
                    encoder: encoder,
                    src: batchK, srcOffset: kvOffset,
                    dst: kvCache.layers[layer].keyBuffer,
                    numKVHeads: config.numKVHeads, headDim: config.headDim,
                    position: startPosition + b, maxSeqLen: kvCache.maxSeqLen
                )
                engine.encodeKVCacheCopy(
                    encoder: encoder,
                    src: batchV, srcOffset: kvOffset,
                    dst: kvCache.layers[layer].valueBuffer,
                    numKVHeads: config.numKVHeads, headDim: config.headDim,
                    position: startPosition + b, maxSeqLen: kvCache.maxSeqLen
                )
            }

            encoder.memoryBarrier(scope: .buffers)

            // Prefill attention (causal masked)
            engine.encodePrefillAttention(
                encoder: encoder,
                q: batchQ,
                kCache: kvCache.layers[layer].keyBuffer,
                vCache: kvCache.layers[layer].valueBuffer,
                output: batchAttnOut,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                headDim: config.headDim,
                startPos: startPosition,
                batchSize: batchSize
            )

            // Batch output projection
            guard let wO = weights.attnOutput(layer: layer) else {
                throw CatalystError.unknown(details: "Missing attn_output weight for layer \(layer)")
            }
            engine.encodeMatmul(encoder: encoder, weights: wO.buffer, weightsOffset: wO.offset,
                               input: batchAttnOut, output: batchAttnProj,
                               rows: h, cols: qDim, batchSize: batchSize, quantType: wO.type)

            // Batch residual add (attention)
            engine.encodeResidualAdd(encoder: encoder, x: batchResidual, residual: batchAttnProj,
                                    count: batchSize * h)

            // FFN RMSNorm (per token)
            guard let ffnNorm = weights.ffnNorm(layer: layer) else {
                throw CatalystError.unknown(details: "Missing ffn_norm weight for layer \(layer)")
            }
            for b in 0..<batchSize {
                engine.encodeBatchRMSNorm(
                    encoder: encoder,
                    input: batchResidual, inputOffset: b * h * floatSize,
                    output: batchNormed, outputOffset: b * h * floatSize,
                    weight: ffnNorm.buffer, weightOffset: ffnNorm.offset,
                    size: h, eps: config.rmsNormEps
                )
            }

            // Batch FFN
            guard let wGate = weights.ffnGate(layer: layer),
                  let wUp = weights.ffnUp(layer: layer),
                  let wDown = weights.ffnDown(layer: layer) else {
                throw CatalystError.unknown(details: "Missing FFN weights for layer \(layer)")
            }

            engine.encodeMatmul(encoder: encoder, weights: wGate.buffer, weightsOffset: wGate.offset,
                               input: batchNormed, output: batchGate,
                               rows: ff, cols: h, batchSize: batchSize, quantType: wGate.type)
            engine.encodeMatmul(encoder: encoder, weights: wUp.buffer, weightsOffset: wUp.offset,
                               input: batchNormed, output: batchUp,
                               rows: ff, cols: h, batchSize: batchSize, quantType: wUp.type)

            engine.encodeSiLUMul(encoder: encoder, gate: batchGate, up: batchUp,
                                output: batchFFNIntermediate, count: batchSize * ff)

            engine.encodeMatmul(encoder: encoder, weights: wDown.buffer, weightsOffset: wDown.offset,
                               input: batchFFNIntermediate, output: batchFFNOut,
                               rows: h, cols: ff, batchSize: batchSize, quantType: wDown.type)

            // Batch residual add (FFN)
            engine.encodeResidualAdd(encoder: encoder, x: batchResidual, residual: batchFFNOut,
                                    count: batchSize * h)
        }

        // Step 3: Final norm + logits (last token only)
        let lastTokenOffset = (batchSize - 1) * h * floatSize
        if let normWeight = weights.outputNorm {
            engine.encodeBatchRMSNorm(
                encoder: encoder,
                input: batchResidual, inputOffset: lastTokenOffset,
                output: normed, outputOffset: 0,
                weight: normWeight.buffer, weightOffset: normWeight.offset,
                size: h, eps: config.rmsNormEps
            )
        }

        if let outWeight = weights.outputWeight {
            engine.encodeMatvec(
                encoder: encoder,
                weights: outWeight.buffer,
                weightsOffset: outWeight.offset,
                input: normed,
                output: logitsBuf,
                rows: config.vocabSize,
                cols: h,
                quantType: outWeight.type
            )
        }

        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if let error = cmdBuffer.error {
            throw CatalystError.unknown(details: "Metal command buffer failed: \(error.localizedDescription)")
        }

        for _ in 0..<batchSize {
            kvCache.advancePosition()
        }

        return logitsBuf.contents().assumingMemoryBound(to: Float.self)
    }

    /// Get the hidden size for embedding extraction.
    public var hiddenSize: Int { config.hiddenSize }

    // MARK: - Decode Layer Helpers

    /// Encode a complete decoder layer (attention + FFN) for single-token decode.
    private func encodeDecoderLayer(
        encoder: MTLComputeCommandEncoder,
        layer: Int,
        position: Int
    ) throws {
        // Attention RMSNorm
        guard let attnNorm = weights.attnNorm(layer: layer) else {
            throw CatalystError.unknown(details: "Missing attn_norm weight for layer \(layer)")
        }
        engine.encodeRMSNorm(
            encoder: encoder,
            input: residual, output: normed,
            weight: attnNorm.buffer, weightOffset: attnNorm.offset,
            size: config.hiddenSize, eps: config.rmsNormEps
        )

        // Q/K/V projections
        guard let wQ = weights.attnQ(layer: layer),
              let wK = weights.attnK(layer: layer),
              let wV = weights.attnV(layer: layer) else {
            throw CatalystError.unknown(details: "Missing Q/K/V weights for layer \(layer)")
        }

        engine.encodeMatvec(encoder: encoder, weights: wQ.buffer, weightsOffset: wQ.offset,
                           input: normed, output: qBuf,
                           rows: config.numHeads * config.headDim, cols: config.hiddenSize,
                           quantType: wQ.type)
        engine.encodeMatvec(encoder: encoder, weights: wK.buffer, weightsOffset: wK.offset,
                           input: normed, output: kBuf,
                           rows: config.numKVHeads * config.headDim, cols: config.hiddenSize,
                           quantType: wK.type)
        engine.encodeMatvec(encoder: encoder, weights: wV.buffer, weightsOffset: wV.offset,
                           input: normed, output: vBuf,
                           rows: config.numKVHeads * config.headDim, cols: config.hiddenSize,
                           quantType: wV.type)

        // RoPE
        engine.encodeRoPE(encoder: encoder, qk: qBuf,
                         headDim: config.headDim, numHeads: config.numHeads,
                         position: position, freqBase: config.ropeFreqBase)
        engine.encodeRoPE(encoder: encoder, qk: kBuf,
                         headDim: config.headDim, numHeads: config.numKVHeads,
                         position: position, freqBase: config.ropeFreqBase)

        // GPU-side KV cache copy
        engine.encodeKVCacheCopy(
            encoder: encoder, src: kBuf, dst: kvCache.layers[layer].keyBuffer,
            numKVHeads: config.numKVHeads, headDim: config.headDim,
            position: position, maxSeqLen: kvCache.maxSeqLen
        )
        engine.encodeKVCacheCopy(
            encoder: encoder, src: vBuf, dst: kvCache.layers[layer].valueBuffer,
            numKVHeads: config.numKVHeads, headDim: config.headDim,
            position: position, maxSeqLen: kvCache.maxSeqLen
        )

        // Memory barrier: KV writes must complete before attention reads
        encoder.memoryBarrier(scope: .buffers)

        // GQA Attention
        engine.encodeAttention(
            encoder: encoder,
            q: qBuf,
            kCache: kvCache.layers[layer].keyBuffer,
            vCache: kvCache.layers[layer].valueBuffer,
            output: attnOut,
            numHeads: config.numHeads,
            numKVHeads: config.numKVHeads,
            headDim: config.headDim,
            seqLen: kvCache.currentLength + 1
        )

        // Output projection
        guard let wO = weights.attnOutput(layer: layer) else {
            throw CatalystError.unknown(details: "Missing attn_output weight for layer \(layer)")
        }
        engine.encodeMatvec(encoder: encoder, weights: wO.buffer, weightsOffset: wO.offset,
                           input: attnOut, output: attnProj,
                           rows: config.hiddenSize, cols: config.numHeads * config.headDim,
                           quantType: wO.type)

        // Residual connection
        engine.encodeResidualAdd(encoder: encoder, x: residual, residual: attnProj, count: config.hiddenSize)

        // FFN block
        guard let ffnNorm = weights.ffnNorm(layer: layer) else {
            throw CatalystError.unknown(details: "Missing ffn_norm weight for layer \(layer)")
        }
        engine.encodeRMSNorm(
            encoder: encoder,
            input: residual, output: normed,
            weight: ffnNorm.buffer, weightOffset: ffnNorm.offset,
            size: config.hiddenSize, eps: config.rmsNormEps
        )

        guard let wGate = weights.ffnGate(layer: layer),
              let wUp = weights.ffnUp(layer: layer),
              let wDown = weights.ffnDown(layer: layer) else {
            throw CatalystError.unknown(details: "Missing FFN weights for layer \(layer)")
        }

        engine.encodeMatvec(encoder: encoder, weights: wGate.buffer, weightsOffset: wGate.offset,
                           input: normed, output: gateBuf,
                           rows: config.intermediateSize, cols: config.hiddenSize,
                           quantType: wGate.type)
        engine.encodeMatvec(encoder: encoder, weights: wUp.buffer, weightsOffset: wUp.offset,
                           input: normed, output: upBuf,
                           rows: config.intermediateSize, cols: config.hiddenSize,
                           quantType: wUp.type)

        engine.encodeSiLUMul(encoder: encoder, gate: gateBuf, up: upBuf,
                            output: ffnIntermediate, count: config.intermediateSize)

        engine.encodeMatvec(encoder: encoder, weights: wDown.buffer, weightsOffset: wDown.offset,
                           input: ffnIntermediate, output: ffnOut,
                           rows: config.hiddenSize, cols: config.intermediateSize,
                           quantType: wDown.type)

        engine.encodeResidualAdd(encoder: encoder, x: residual, residual: ffnOut, count: config.hiddenSize)
    }

    /// Encode final RMS norm + output projection to logits.
    private func encodeFinalProjection(encoder: MTLComputeCommandEncoder) throws {
        if let normWeight = weights.outputNorm {
            engine.encodeRMSNorm(
                encoder: encoder,
                input: residual, output: normed,
                weight: normWeight.buffer, weightOffset: normWeight.offset,
                size: config.hiddenSize, eps: config.rmsNormEps
            )
        }

        if let outWeight = weights.outputWeight {
            engine.encodeMatvec(
                encoder: encoder,
                weights: outWeight.buffer,
                weightsOffset: outWeight.offset,
                input: normed,
                output: logitsBuf,
                rows: config.vocabSize,
                cols: config.hiddenSize,
                quantType: outWeight.type
            )
        }
    }
}
