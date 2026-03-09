//
//  MetalComputeEngine.swift
//  OnDeviceCatalyst
//
//  Manages Metal device, command queue, pipeline states, and shader dispatch.
//

import Foundation
import Metal

/// Manages Metal compute resources and dispatches shader kernels.
public final class MetalComputeEngine {

    // MARK: - Metal Objects

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    // MARK: - Pipeline States

    // Matvec kernels (one per quant type)
    private let matvecQ8_0: MTLComputePipelineState
    private let matvecQ4_0: MTLComputePipelineState
    private let matvecQ4_K: MTLComputePipelineState
    private let matvecQ6_K: MTLComputePipelineState
    private let matvecF16: MTLComputePipelineState
    private let matvecF32: MTLComputePipelineState

    // Matmul (batched) kernels
    private let matmulQ8_0: MTLComputePipelineState
    private let matmulQ4_0: MTLComputePipelineState
    private let matmulQ4_K: MTLComputePipelineState
    private let matmulQ6_K: MTLComputePipelineState
    private let matmulF16: MTLComputePipelineState
    private let matmulF32: MTLComputePipelineState

    // Normalization
    private let rmsNormPipeline: MTLComputePipelineState

    // RoPE
    private let ropePipeline: MTLComputePipelineState
    private let batchRopePipeline: MTLComputePipelineState

    // Attention
    private let attentionPipeline: MTLComputePipelineState
    private let prefillAttentionPipeline: MTLComputePipelineState

    // Activations
    private let siluMulPipeline: MTLComputePipelineState
    private let residualAddPipeline: MTLComputePipelineState
    private let residualAddInplacePipeline: MTLComputePipelineState
    private let copyBufferPipeline: MTLComputePipelineState
    private let kvCacheCopyPipeline: MTLComputePipelineState

    // Embedding
    private let embeddingPipeline: MTLComputePipelineState
    private let embeddingF16Pipeline: MTLComputePipelineState
    private let batchEmbeddingPipeline: MTLComputePipelineState
    private let batchEmbeddingF16Pipeline: MTLComputePipelineState

    // MARK: - Threadgroup Configuration (6.6)

    /// Threadgroup size configuration, auto-tuned per device.
    public struct ThreadgroupConfig {
        public var matvec: Int
        public var attention: Int
        public var rmsNorm: Int
        public var elementwise: Int
    }

    /// Current threadgroup sizes (auto-tuned or default).
    public private(set) var threadgroupConfig: ThreadgroupConfig

    // MARK: - Init

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CatalystError.unknown(details: "Metal is not available on this device")
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw CatalystError.unknown(details: "Failed to create Metal command queue")
        }
        self.commandQueue = queue

        // Load shader library
        guard let lib = device.makeDefaultLibrary() else {
            throw CatalystError.unknown(details: "Failed to load Metal shader library. Ensure .metal files are in the target.")
        }
        self.library = lib

        // Create pipeline states for each kernel
        func makePipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = lib.makeFunction(name: name) else {
                throw CatalystError.unknown(details: "Metal function '\(name)' not found in shader library")
            }
            return try device.makeComputePipelineState(function: function)
        }

        matvecQ8_0 = try makePipeline("matvec_q8_0")
        matvecQ4_0 = try makePipeline("matvec_q4_0")
        matvecQ4_K = try makePipeline("matvec_q4_k")
        matvecQ6_K = try makePipeline("matvec_q6_k")
        matvecF16 = try makePipeline("matvec_f16")
        matvecF32 = try makePipeline("matvec_f32")

        matmulQ8_0 = try makePipeline("matmul_q8_0")
        matmulQ4_0 = try makePipeline("matmul_q4_0")
        matmulQ4_K = try makePipeline("matmul_q4_k")
        matmulQ6_K = try makePipeline("matmul_q6_k")
        matmulF16 = try makePipeline("matmul_f16")
        matmulF32 = try makePipeline("matmul_f32")

        rmsNormPipeline = try makePipeline("rms_norm")
        ropePipeline = try makePipeline("rope_apply")
        batchRopePipeline = try makePipeline("batch_rope_apply")
        attentionPipeline = try makePipeline("gqa_attention_decode")
        prefillAttentionPipeline = try makePipeline("prefill_attention")
        siluMulPipeline = try makePipeline("silu_mul")
        residualAddPipeline = try makePipeline("residual_add")
        residualAddInplacePipeline = try makePipeline("residual_add_inplace")
        copyBufferPipeline = try makePipeline("copy_buffer")
        kvCacheCopyPipeline = try makePipeline("kv_cache_copy")
        embeddingPipeline = try makePipeline("embedding_lookup")
        embeddingF16Pipeline = try makePipeline("embedding_lookup_f16")
        batchEmbeddingPipeline = try makePipeline("batch_embedding_lookup")
        batchEmbeddingF16Pipeline = try makePipeline("batch_embedding_lookup_f16")

        // Set default threadgroup config based on device
        threadgroupConfig = MetalComputeEngine.defaultConfig(for: device)

        print("MetalComputeEngine: Initialized with device '\(device.name)', matvec tg=\(threadgroupConfig.matvec)")
    }

    // MARK: - Threadgroup Auto-Tuning (6.6)

    /// Default threadgroup sizes based on device name.
    private static func defaultConfig(for device: MTLDevice) -> ThreadgroupConfig {
        let name = device.name.lowercased()

        let matvec: Int
        if name.contains("a15") || name.contains("a14") {
            matvec = 64
        } else if name.contains("a16") || name.contains("a17") || name.contains("m1") || name.contains("m2") || name.contains("m3") || name.contains("m4") {
            matvec = 128
        } else {
            matvec = 64  // safe default
        }

        return ThreadgroupConfig(
            matvec: matvec,
            attention: 256,
            rmsNorm: 256,
            elementwise: 256
        )
    }

    /// Micro-benchmark matvec at different threadgroup sizes and pick the fastest.
    /// Call after model loading for optimal tuning. Adds < 100ms to warmup.
    public func tuneThreadgroupSizes(testBuffer: MTLBuffer, inputBuffer: MTLBuffer, outputBuffer: MTLBuffer, rows: Int, cols: Int, quantType: GGMLType) {
        let candidates = [32, 64, 128, 256]
        var bestSize = threadgroupConfig.matvec
        var bestTime: Double = .infinity

        let pipeline = matvecPipeline(for: quantType)

        for tgSize in candidates {
            if tgSize > pipeline.maxTotalThreadsPerThreadgroup { continue }

            guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuffer.makeComputeCommandEncoder() else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(testBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)

            var params = MatVecParams(
                rows: UInt32(rows),
                cols: UInt32(cols),
                quant_type: quantType.rawValue,
                block_size: UInt32(quantType.blockSize),
                blocks_per_row: UInt32((cols + quantType.blockSize - 1) / quantType.blockSize)
            )
            encoder.setBytes(&params, length: MemoryLayout<MatVecParams>.size, index: 3)

            encoder.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
            )
            encoder.endEncoding()
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()

            let elapsed = cmdBuffer.gpuEndTime - cmdBuffer.gpuStartTime
            if elapsed < bestTime && elapsed > 0 {
                bestTime = elapsed
                bestSize = tgSize
            }
        }

        threadgroupConfig.matvec = bestSize
        print("MetalComputeEngine: Auto-tuned matvec threadgroup size to \(bestSize)")
    }

    // MARK: - Buffer Creation

    /// Create a shared-mode buffer (CPU + GPU accessible on unified memory).
    public func makeBuffer(length: Int) -> MTLBuffer? {
        return device.makeBuffer(length: length, options: .storageModeShared)
    }

    /// Create a buffer wrapping existing memory (zero-copy).
    public func makeBuffer(bytesNoCopy pointer: UnsafeMutableRawPointer, length: Int) -> MTLBuffer? {
        return device.makeBuffer(
            bytesNoCopy: pointer,
            length: length,
            options: .storageModeShared,
            deallocator: nil
        )
    }

    // MARK: - Dispatch Helpers

    /// Select the matvec pipeline for a given quantization type.
    public func matvecPipeline(for quantType: GGMLType) -> MTLComputePipelineState {
        switch quantType {
        case .q8_0: return matvecQ8_0
        case .q4_0: return matvecQ4_0
        case .q4_K: return matvecQ4_K
        case .q6_K: return matvecQ6_K
        case .f16:  return matvecF16
        case .f32:  return matvecF32
        default:    return matvecQ8_0
        }
    }

    /// Select the matmul pipeline for a given quantization type.
    public func matmulPipeline(for quantType: GGMLType) -> MTLComputePipelineState {
        switch quantType {
        case .q8_0: return matmulQ8_0
        case .q4_0: return matmulQ4_0
        case .q4_K: return matmulQ4_K
        case .q6_K: return matmulQ6_K
        case .f16:  return matmulF16
        case .f32:  return matmulF32
        default:    return matmulQ8_0
        }
    }

    /// Optimal threadgroup size for matvec (one threadgroup per output row).
    public var matvecThreadgroupSize: Int {
        return threadgroupConfig.matvec
    }

    // MARK: - Encode Operations

    /// Encode a matrix-vector multiplication.
    public func encodeMatvec(
        encoder: MTLComputeCommandEncoder,
        weights: MTLBuffer,
        weightsOffset: Int,
        input: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        cols: Int,
        quantType: GGMLType
    ) {
        let pipeline = matvecPipeline(for: quantType)
        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var params = MatVecParams(
            rows: UInt32(rows),
            cols: UInt32(cols),
            quant_type: quantType.rawValue,
            block_size: UInt32(quantType.blockSize),
            blocks_per_row: UInt32((cols + quantType.blockSize - 1) / quantType.blockSize)
        )
        encoder.setBytes(&params, length: MemoryLayout<MatVecParams>.size, index: 3)

        let tgSize = min(matvecThreadgroupSize, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode a batched matrix-matrix multiplication (for prefill).
    public func encodeMatmul(
        encoder: MTLComputeCommandEncoder,
        weights: MTLBuffer,
        weightsOffset: Int,
        input: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        cols: Int,
        batchSize: Int,
        quantType: GGMLType
    ) {
        let pipeline = matmulPipeline(for: quantType)
        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var params = MatMulParams(
            rows: UInt32(rows),
            cols: UInt32(cols),
            batch_size: UInt32(batchSize),
            blocks_per_row: UInt32((cols + quantType.blockSize - 1) / quantType.blockSize)
        )
        encoder.setBytes(&params, length: MemoryLayout<MatMulParams>.size, index: 3)

        let tgSize = min(matvecThreadgroupSize, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode RMS normalization.
    public func encodeRMSNorm(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        weight: MTLBuffer,
        weightOffset: Int,
        size: Int,
        eps: Float
    ) {
        encoder.setComputePipelineState(rmsNormPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)

        var params = NormParams(size: UInt32(size), eps: eps)
        encoder.setBytes(&params, length: MemoryLayout<NormParams>.size, index: 3)

        let tgSize = min(threadgroupConfig.rmsNorm, rmsNormPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode batch RMS normalization (one vector at a time but with offset).
    public func encodeBatchRMSNorm(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        inputOffset: Int,
        output: MTLBuffer,
        outputOffset: Int,
        weight: MTLBuffer,
        weightOffset: Int,
        size: Int,
        eps: Float
    ) {
        encoder.setComputePipelineState(rmsNormPipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)

        var params = NormParams(size: UInt32(size), eps: eps)
        encoder.setBytes(&params, length: MemoryLayout<NormParams>.size, index: 3)

        let tgSize = min(threadgroupConfig.rmsNorm, rmsNormPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode RoPE application (single token).
    public func encodeRoPE(
        encoder: MTLComputeCommandEncoder,
        qk: MTLBuffer,
        headDim: Int,
        numHeads: Int,
        position: Int,
        freqBase: Float,
        freqScale: Float = 1.0
    ) {
        encoder.setComputePipelineState(ropePipeline)
        encoder.setBuffer(qk, offset: 0, index: 0)

        var params = RoPEParams(
            head_dim: UInt32(headDim),
            num_heads: UInt32(numHeads),
            position: UInt32(position),
            freq_base: freqBase,
            freq_scale: freqScale
        )
        encoder.setBytes(&params, length: MemoryLayout<RoPEParams>.size, index: 1)

        let totalPairs = numHeads * (headDim / 2)
        let tgSize = min(256, ropePipeline.maxTotalThreadsPerThreadgroup)
        let numGroups = (totalPairs + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode batch RoPE (multiple tokens with different positions).
    public func encodeBatchRoPE(
        encoder: MTLComputeCommandEncoder,
        qk: MTLBuffer,
        positions: MTLBuffer,
        headDim: Int,
        numHeads: Int,
        batchSize: Int,
        freqBase: Float,
        freqScale: Float = 1.0
    ) {
        encoder.setComputePipelineState(batchRopePipeline)
        encoder.setBuffer(qk, offset: 0, index: 0)
        encoder.setBuffer(positions, offset: 0, index: 1)

        var params = RoPEParams(
            head_dim: UInt32(headDim),
            num_heads: UInt32(numHeads),
            position: 0,  // not used in batch version
            freq_base: freqBase,
            freq_scale: freqScale
        )
        encoder.setBytes(&params, length: MemoryLayout<RoPEParams>.size, index: 2)

        var bs = UInt32(batchSize)
        encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 3)

        let totalPairs = batchSize * numHeads * (headDim / 2)
        let tgSize = min(256, batchRopePipeline.maxTotalThreadsPerThreadgroup)
        let numGroups = (totalPairs + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode GQA attention decode.
    public func encodeAttention(
        encoder: MTLComputeCommandEncoder,
        q: MTLBuffer,
        kCache: MTLBuffer,
        vCache: MTLBuffer,
        output: MTLBuffer,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        seqLen: Int
    ) {
        encoder.setComputePipelineState(attentionPipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(kCache, offset: 0, index: 1)
        encoder.setBuffer(vCache, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

        var params = AttentionParams(
            num_heads: UInt32(numHeads),
            num_kv_heads: UInt32(numKVHeads),
            head_dim: UInt32(headDim),
            seq_len: UInt32(seqLen),
            scale: 1.0 / sqrt(Float(headDim))
        )
        encoder.setBytes(&params, length: MemoryLayout<AttentionParams>.size, index: 4)

        let tgSize = min(threadgroupConfig.attention, attentionPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode prefill attention with causal mask.
    public func encodePrefillAttention(
        encoder: MTLComputeCommandEncoder,
        q: MTLBuffer,
        kCache: MTLBuffer,
        vCache: MTLBuffer,
        output: MTLBuffer,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        startPos: Int,
        batchSize: Int
    ) {
        encoder.setComputePipelineState(prefillAttentionPipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(kCache, offset: 0, index: 1)
        encoder.setBuffer(vCache, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)

        var params = AttentionParams(
            num_heads: UInt32(numHeads),
            num_kv_heads: UInt32(numKVHeads),
            head_dim: UInt32(headDim),
            seq_len: 0,  // not used in prefill (causal_len computed per query)
            scale: 1.0 / sqrt(Float(headDim))
        )
        encoder.setBytes(&params, length: MemoryLayout<AttentionParams>.size, index: 4)

        var sp = UInt32(startPos)
        encoder.setBytes(&sp, length: MemoryLayout<UInt32>.size, index: 5)

        var bs = UInt32(batchSize)
        encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 6)

        let tgSize = min(threadgroupConfig.attention, prefillAttentionPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode fused SiLU * elementwise multiply.
    public func encodeSiLUMul(
        encoder: MTLComputeCommandEncoder,
        gate: MTLBuffer,
        up: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) {
        encoder.setComputePipelineState(siluMulPipeline)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var params = ElementwiseParams(count: UInt32(count))
        encoder.setBytes(&params, length: MemoryLayout<ElementwiseParams>.size, index: 3)

        let tgSize = min(threadgroupConfig.elementwise, siluMulPipeline.maxTotalThreadsPerThreadgroup)
        let numGroups = (count + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode in-place residual add.
    public func encodeResidualAdd(
        encoder: MTLComputeCommandEncoder,
        x: MTLBuffer,
        xOffset: Int = 0,
        residual: MTLBuffer,
        residualOffset: Int = 0,
        count: Int
    ) {
        encoder.setComputePipelineState(residualAddInplacePipeline)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(residual, offset: residualOffset, index: 1)

        var params = ElementwiseParams(count: UInt32(count))
        encoder.setBytes(&params, length: MemoryLayout<ElementwiseParams>.size, index: 2)

        let tgSize = min(threadgroupConfig.elementwise, residualAddInplacePipeline.maxTotalThreadsPerThreadgroup)
        let numGroups = (count + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode KV cache copy (GPU-side, replaces CPU memcpy).
    public func encodeKVCacheCopy(
        encoder: MTLComputeCommandEncoder,
        src: MTLBuffer,
        srcOffset: Int = 0,
        dst: MTLBuffer,
        numKVHeads: Int,
        headDim: Int,
        position: Int,
        maxSeqLen: Int
    ) {
        encoder.setComputePipelineState(kvCacheCopyPipeline)
        encoder.setBuffer(src, offset: srcOffset, index: 0)
        encoder.setBuffer(dst, offset: 0, index: 1)

        var params = KVCacheParams(
            num_kv_heads: UInt32(numKVHeads),
            head_dim: UInt32(headDim),
            position: UInt32(position),
            max_seq_len: UInt32(maxSeqLen)
        )
        encoder.setBytes(&params, length: MemoryLayout<KVCacheParams>.size, index: 2)

        let count = numKVHeads * headDim
        let tgSize = min(threadgroupConfig.elementwise, kvCacheCopyPipeline.maxTotalThreadsPerThreadgroup)
        let numGroups = (count + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode embedding lookup (single token).
    public func encodeEmbedding(
        encoder: MTLComputeCommandEncoder,
        table: MTLBuffer,
        tableOffset: Int,
        output: MTLBuffer,
        tokenId: Int,
        hiddenSize: Int,
        isF16: Bool
    ) {
        let pipeline = isF16 ? embeddingF16Pipeline : embeddingPipeline
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)

        var params = EmbeddingParams(
            vocab_size: 0,
            hidden_size: UInt32(hiddenSize),
            token_id: UInt32(tokenId)
        )
        encoder.setBytes(&params, length: MemoryLayout<EmbeddingParams>.size, index: 2)

        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let numGroups = (hiddenSize + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Encode batch embedding lookup.
    public func encodeBatchEmbedding(
        encoder: MTLComputeCommandEncoder,
        table: MTLBuffer,
        tableOffset: Int,
        output: MTLBuffer,
        tokenIds: MTLBuffer,
        hiddenSize: Int,
        batchSize: Int,
        isF16: Bool
    ) {
        let pipeline = isF16 ? batchEmbeddingF16Pipeline : batchEmbeddingPipeline
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(tokenIds, offset: 0, index: 2)

        var hs = UInt32(hiddenSize)
        encoder.setBytes(&hs, length: MemoryLayout<UInt32>.size, index: 3)

        var bs = UInt32(batchSize)
        encoder.setBytes(&bs, length: MemoryLayout<UInt32>.size, index: 4)

        let total = batchSize * hiddenSize
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let numGroups = (total + tgSize - 1) / tgSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    // MARK: - C Struct Wrappers (matching shader_types.h)

    struct MatVecParams {
        var rows: UInt32
        var cols: UInt32
        var quant_type: UInt32
        var block_size: UInt32
        var blocks_per_row: UInt32
    }

    struct MatMulParams {
        var rows: UInt32
        var cols: UInt32
        var batch_size: UInt32
        var blocks_per_row: UInt32
    }

    struct NormParams {
        var size: UInt32
        var eps: Float
    }

    struct RoPEParams {
        var head_dim: UInt32
        var num_heads: UInt32
        var position: UInt32
        var freq_base: Float
        var freq_scale: Float
    }

    struct AttentionParams {
        var num_heads: UInt32
        var num_kv_heads: UInt32
        var head_dim: UInt32
        var seq_len: UInt32
        var scale: Float
    }

    struct EmbeddingParams {
        var vocab_size: UInt32
        var hidden_size: UInt32
        var token_id: UInt32
    }

    struct ElementwiseParams {
        var count: UInt32
    }

    struct KVCacheParams {
        var num_kv_heads: UInt32
        var head_dim: UInt32
        var position: UInt32
        var max_seq_len: UInt32
    }
}
