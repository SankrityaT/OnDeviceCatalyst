//
//  MetalBackend.swift
//  OnDeviceCatalyst
//
//  InferenceBackend implementation using our custom Metal inference engine.
//  Replaces llama.cpp with direct GPU execution via Metal compute shaders.
//

import Foundation
import Metal
#if canImport(Dispatch)
import Dispatch
#endif

/// Custom Metal-based inference backend for maximum decode speed on Apple Silicon.
public final class MetalBackend: InferenceBackend {

    // MARK: - Components

    private var computeEngine: MetalComputeEngine?
    private var modelWeights: ModelWeights?
    private var tokenizer: GGUFTokenizer?
    private var transformerGraph: TransformerGraph?
    private var kvCache: KVCache?
    private var modelConfig: TransformerConfig?

    /// The last computed logits pointer (valid until next decode call).
    private var lastLogits: UnsafeMutablePointer<Float>?

    /// Memory pressure observer.
    private var memoryPressureSource: DispatchSourceMemoryPressure?

    // MARK: - Lifecycle

    public func loadModel(path: String, settings: InstanceSettings) throws {
        print("MetalBackend: Parsing GGUF file...")
        let gguf = try GGUFParser.parse(path: path)

        guard let config = gguf.config else {
            throw CatalystError.modelLoadingFailed(
                details: "Failed to extract model config from GGUF metadata"
            )
        }
        self.modelConfig = config

        print("MetalBackend: Model config - \(config.architecture), " +
              "\(config.numLayers) layers, \(config.hiddenSize) hidden, " +
              "\(config.numHeads) heads, \(config.numKVHeads) KV heads")

        // Initialize tokenizer
        print("MetalBackend: Loading tokenizer...")
        tokenizer = try GGUFTokenizer(metadata: gguf.metadata)

        // Initialize Metal compute engine
        print("MetalBackend: Initializing Metal compute engine...")
        let engine = try MetalComputeEngine()
        self.computeEngine = engine

        // Load weights with zero-copy mmap
        print("MetalBackend: Loading weights (zero-copy mmap)...")
        modelWeights = try ModelWeights(path: path, gguf: gguf, device: engine.device)

        print("MetalBackend: Model loaded successfully")
    }

    public func createContext(settings: InstanceSettings) throws {
        guard let engine = computeEngine, let config = modelConfig else {
            throw CatalystError.engineNotInitialized
        }

        var maxSeqLen = Int(settings.contextLength)

        // Check available memory before allocating KV cache
        let kvBytesPerToken = config.numLayers * 2 * config.numKVHeads * config.headDim * MemoryLayout<Float>.size
        let requestedBytes = maxSeqLen * kvBytesPerToken
        let availableMemory = MetalBackend.availableMemoryBytes()

        if requestedBytes > Int(Double(availableMemory) * 0.5) {
            // Reduce context length to fit within 50% of available memory
            let reducedSeqLen = max(512, Int(Double(availableMemory) * 0.5) / kvBytesPerToken)
            print("MetalBackend: Reducing context length from \(maxSeqLen) to \(reducedSeqLen) due to memory constraints")
            maxSeqLen = reducedSeqLen
        }

        // Allocate KV cache
        print("MetalBackend: Allocating KV cache (max \(maxSeqLen) tokens)...")
        let cache = try KVCache(
            device: engine.device,
            numLayers: config.numLayers,
            numKVHeads: config.numKVHeads,
            headDim: config.headDim,
            maxSeqLen: maxSeqLen
        )
        self.kvCache = cache

        // Create transformer graph
        guard let weights = modelWeights else {
            throw CatalystError.engineNotInitialized
        }

        print("MetalBackend: Building transformer graph...")
        transformerGraph = try TransformerGraph(
            engine: engine,
            weights: weights,
            config: config,
            kvCache: cache
        )

        // Set up memory pressure monitoring
        setupMemoryPressureMonitoring()

        print("MetalBackend: Context created successfully")
    }

    public func warmup() throws {
        guard transformerGraph != nil, let tokenizer = tokenizer else {
            throw CatalystError.engineNotInitialized
        }

        let tokens = tokenizer.tokenize(text: "Hello", addBos: true)
        guard !tokens.isEmpty else { return }

        // Run one decode step to warm up GPU caches
        lastLogits = try transformerGraph?.decodeToken(tokenId: Int(tokens[0]), position: 0)
        clearKVCache()
    }

    public func shutdown() {
        memoryPressureSource?.cancel()
        memoryPressureSource = nil
        transformerGraph = nil
        kvCache?.clear()
        kvCache = nil
        modelWeights = nil
        tokenizer = nil
        computeEngine = nil
        modelConfig = nil
        lastLogits = nil
        print("MetalBackend: Shut down")
    }

    deinit {
        shutdown()
    }

    // MARK: - Model Metadata

    public var vocabularySize: Int32 {
        return tokenizer?.vocabularySize ?? 0
    }

    public var embeddingSize: Int32 {
        return Int32(modelConfig?.hiddenSize ?? 0)
    }

    public func isEndOfGeneration(token: Int32) -> Bool {
        return tokenizer?.isEndOfGeneration(token: token) ?? false
    }

    // MARK: - Tokenization

    public func tokenize(text: String, addBos: Bool, parseSpecial: Bool = true) throws -> [Int32] {
        guard let tokenizer = tokenizer else {
            throw CatalystError.engineNotInitialized
        }
        return tokenizer.tokenize(text: text, addBos: addBos)
    }

    public func detokenize(token: Int32) -> String {
        return tokenizer?.detokenize(token: token) ?? ""
    }

    // MARK: - Prompt Processing

    public func processTokens(_ tokens: [Int32], positions: [Int32], generateLogitsAtLast: Bool) throws {
        guard let graph = transformerGraph else {
            throw CatalystError.engineNotInitialized
        }

        // Use batch prefill for multiple tokens
        if tokens.count > 1 {
            let tokenInts = tokens.map { Int($0) }
            let startPos = Int(positions[0])
            lastLogits = try graph.prefillTokens(tokenIds: tokenInts, startPosition: startPos)
        } else if let token = tokens.first, let pos = positions.first {
            lastLogits = try graph.decodeToken(tokenId: Int(token), position: Int(pos))
        }
    }

    // MARK: - Decode Step

    public func decodeToken(_ token: Int32, position: Int32) throws {
        guard let graph = transformerGraph else {
            throw CatalystError.engineNotInitialized
        }
        lastLogits = try graph.decodeToken(tokenId: Int(token), position: Int(position))
    }

    // MARK: - Logits

    public func getLogits() -> UnsafeMutablePointer<Float>? {
        return lastLogits
    }

    // MARK: - KV Cache

    public func clearKVCache() {
        kvCache?.clear()
    }

    public func removeKVCacheTokens(sequenceId: Int32, startPos: Int32, endPos: Int32) {
        kvCache?.removeTokens(startPos: Int(startPos), endPos: Int(endPos))
    }

    // MARK: - Embeddings

    public func processTokensForEmbedding(_ tokens: [Int32]) throws {
        let positions = tokens.indices.map { Int32($0) }
        try processTokens(tokens, positions: positions, generateLogitsAtLast: true)
    }

    public func getEmbeddings() -> UnsafeMutablePointer<Float>? {
        guard let graph = transformerGraph else { return nil }
        return graph.lastHiddenState.contents().assumingMemoryBound(to: Float.self)
    }

    // MARK: - State Persistence

    public var stateSize: Int {
        return 0
    }

    public func saveState(to buffer: UnsafeMutablePointer<UInt8>, size: Int) -> Int {
        return 0
    }

    public func loadState(from buffer: UnsafePointer<UInt8>, size: Int) -> Int {
        return 0
    }

    // MARK: - Memory Pressure Handling (6.5)

    private func setupMemoryPressureMonitoring() {
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .main)
        source.setEventHandler { [weak self] in
            guard let self = self else { return }
            let event = source.data
            if event.contains(.critical) {
                print("MetalBackend: CRITICAL memory pressure — evicting 25% of KV cache")
                self.kvCache?.evictOldest(fraction: 0.25)
            } else if event.contains(.warning) {
                print("MetalBackend: Memory pressure warning")
            }
        }
        source.resume()
        self.memoryPressureSource = source
    }

    /// Get approximate available memory in bytes.
    private static func availableMemoryBytes() -> Int {
        let totalPhysical = ProcessInfo.processInfo.physicalMemory
        // Use task_info to get current memory usage
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            let usedBytes = Int(info.phys_footprint)
            return max(0, Int(totalPhysical) - usedBytes)
        }
        // Fallback: assume 60% is available
        return Int(Double(totalPhysical) * 0.6)
    }
}
