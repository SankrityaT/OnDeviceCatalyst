//
//  LlamaBridge.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//
//  Safe Swift wrapper around llama.cpp C API
//  Updated for llama.cpp b7870 XCFramework
//

import Foundation
import llama

// MARK: - C Type Aliases
public typealias CModel = OpaquePointer
public typealias CContext = OpaquePointer
public typealias CBatch = llama_batch
public typealias CToken = llama_token
public typealias CSampler = UnsafeMutablePointer<llama_sampler>

/// Safe bridge to llama.cpp C API with comprehensive error handling
public enum LlamaBridge {

    // MARK: - Backend Management

    /// Initialize the llama.cpp backend (call once at app start)
    public static func initializeBackend() {
        llama_backend_init()
        print("Catalyst: llama.cpp backend initialized (b7870)")
    }

    /// Free the llama.cpp backend (call at app shutdown)
    public static func freeBackend() {
        llama_backend_free()
        print("Catalyst: llama.cpp backend freed")
    }

    // MARK: - Model Loading

    /// Loads a model from file with comprehensive validation
    public static func loadModel(path: String, settings: InstanceSettings) throws -> CModel {
        // Pre-flight validation
        try validateModelFile(path: path)

        // Configure model parameters
        var params = llama_model_default_params()
        params.n_gpu_layers = settings.gpuLayers
        params.use_mmap = settings.enableMemoryMapping
        params.use_mlock = settings.enableMemoryLocking

        print("Catalyst: Loading model from \(path)")
        print("Catalyst: GPU layers: \(settings.gpuLayers), mmap: \(settings.enableMemoryMapping), mlock: \(settings.enableMemoryLocking)")

        // Load model (deprecated function works with Swift binding)
        guard let model = llama_load_model_from_file(path, params) else {
            throw createModelLoadingError(path: path, settings: settings)
        }

        print("Catalyst: Model loaded successfully")
        return model
    }

    /// Creates appropriate error for model loading failure
    private static func createModelLoadingError(path: String, settings: InstanceSettings) -> CatalystError {
        let filename = (path as NSString).lastPathComponent.lowercased()

        if filename.contains("qwen2.5") || filename.contains("qwen25") {
            return .architectureUnsupported(
                architecture: "qwen2.5",
                suggestion: "Qwen 2.5 is supported. Try reducing GPU layers or using CPU-only mode if loading fails."
            )
        } else if filename.contains("llama3") || filename.contains("llama-3") {
            return .architectureUnsupported(
                architecture: "llama3.x",
                suggestion: "Llama 3.x is supported. Try reducing context length or GPU layers."
            )
        } else if filename.contains("phi") {
            return .architectureUnsupported(
                architecture: "phi",
                suggestion: "Phi models are supported. Try reducing GPU layers."
            )
        } else if filename.contains("deepseek") {
            return .architectureUnsupported(
                architecture: "deepseek",
                suggestion: "DeepSeek is supported. This may be a large model - ensure sufficient memory."
            )
        }

        var details = "Failed to load model. This could be due to:"
        details += "\n• Unsupported model architecture"
        details += "\n• Corrupted model file"
        details += "\n• Insufficient memory"
        details += "\n• Incompatible quantization format"

        if settings.gpuLayers > 0 {
            details += "\n\nTry setting GPU layers to 0 for CPU-only mode."
        }

        return .modelLoadingFailed(details: details)
    }

    /// Validates model file before loading
    private static func validateModelFile(path: String) throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw CatalystError.modelFileNotFound(path: path)
        }

        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: path)
            guard let fileSize = attributes[.size] as? UInt64, fileSize >= 1_048_576 else {
                throw CatalystError.modelFileCorrupted(
                    path: path,
                    reason: "File is too small to be a valid model (< 1MB)"
                )
            }
        } catch {
            throw CatalystError.modelFileNotFound(path: path)
        }

        guard let fileHandle = FileHandle(forReadingAtPath: path) else {
            throw CatalystError.modelFileNotFound(path: path)
        }

        defer { fileHandle.closeFile() }

        let magicBytes = fileHandle.readData(ofLength: 4)
        let expectedMagic = "GGUF".data(using: .ascii)!

        guard magicBytes.count >= 4 && magicBytes == expectedMagic else {
            throw CatalystError.modelFileCorrupted(
                path: path,
                reason: "Invalid GGUF format - missing or incorrect magic header"
            )
        }
    }

    /// Safely free a model
    public static func freeModel(_ model: CModel) {
        llama_free_model(model)
        print("Catalyst: Model freed")
    }

    // MARK: - Context Management

    /// Creates inference context with error handling
    public static func createContext(model: CModel, settings: InstanceSettings) throws -> CContext {
        var params = llama_context_default_params()
        params.n_ctx = settings.contextLength
        params.n_batch = settings.batchSize
        params.n_ubatch = settings.batchSize
        params.n_threads = settings.cpuThreads
        params.n_threads_batch = settings.cpuThreads
        params.embeddings = false

        print("Catalyst: Creating context - ctx:\(settings.contextLength), batch:\(settings.batchSize), threads:\(settings.cpuThreads)")

        guard let context = llama_new_context_with_model(model, params) else {
            throw CatalystError.contextCreationFailed(
                details: "llama_new_context_with_model failed. Try reducing context length, batch size, or GPU layers."
            )
        }

        let actualCtx = llama_n_ctx(context)
        if actualCtx != settings.contextLength {
            print("Catalyst: Context length adjusted from \(settings.contextLength) to \(actualCtx)")
        }

        print("Catalyst: Context created successfully")
        return context
    }

    /// Safely free a context
    public static func freeContext(_ context: CContext) {
        llama_free(context)
        print("Catalyst: Context freed")
    }

    // MARK: - Model Information

    /// Get maximum context length supported by model
    public static func getMaxContextLength(_ context: CContext) -> UInt32 {
        return llama_n_ctx(context)
    }

    /// Get model vocabulary size
    public static func getVocabularySize(_ model: CModel) -> Int32 {
        guard let vocab = llama_model_get_vocab(model) else { return 0 }
        return llama_vocab_n_tokens(vocab)
    }

    /// Get special tokens
    public static func getBosToken(_ model: CModel) -> CToken {
        guard let vocab = llama_model_get_vocab(model) else { return 0 }
        return llama_vocab_bos(vocab)
    }

    public static func getEosToken(_ model: CModel) -> CToken {
        guard let vocab = llama_model_get_vocab(model) else { return 0 }
        return llama_vocab_eos(vocab)
    }

    /// Get model embedding size (hidden dimension)
    public static func getEmbeddingSize(_ model: CModel) -> Int32 {
        return llama_n_embd(model)
    }

    /// Check if token indicates end of generation
    public static func isEndOfGeneration(_ model: CModel, token: CToken) -> Bool {
        guard let vocab = llama_model_get_vocab(model) else { return false }
        return token == llama_vocab_eos(vocab) || llama_vocab_is_eog(vocab, token)
    }

    // MARK: - Tokenization

    /// Tokenize text with proper error handling
    public static func tokenize(
        text: String,
        model: CModel,
        addBos: Bool = true,
        parseSpecial: Bool = true
    ) throws -> [CToken] {
        // Get vocab from model (new API in llama.cpp b7870+)
        guard let vocab = llama_model_get_vocab(model) else {
            throw CatalystError.tokenizationFailed(
                text: String(text.prefix(100)) + (text.count > 100 ? "..." : ""),
                reason: "Failed to get vocabulary from model"
            )
        }

        let utf8Count = text.utf8.count
        let maxTokens = utf8Count + (addBos ? 1 : 0) + 1
        var tokens = Array<CToken>(repeating: 0, count: maxTokens)

        let tokenCount = llama_tokenize(
            vocab,
            text,
            Int32(utf8Count),
            &tokens,
            Int32(maxTokens),
            addBos,
            parseSpecial
        )

        guard tokenCount >= 0 else {
            throw CatalystError.tokenizationFailed(
                text: String(text.prefix(100)) + (text.count > 100 ? "..." : ""),
                reason: "llama_tokenize returned \(tokenCount). Text length: \(utf8Count), max tokens: \(maxTokens)"
            )
        }

        return Array(tokens.prefix(Int(tokenCount)))
    }

    /// Convert token back to text
    public static func detokenize(token: CToken, model: CModel) -> String {
        // Get vocab from model (new API in llama.cpp b7870+)
        guard let vocab = llama_model_get_vocab(model) else {
            return ""
        }

        let bufferSize = 128
        var buffer = Array<CChar>(repeating: 0, count: bufferSize)

        let charCount = llama_token_to_piece(
            vocab,
            token,
            &buffer,
            Int32(bufferSize),
            0,
            false
        )

        guard charCount > 0 else {
            return ""
        }

        let validChars = buffer.prefix(Int(charCount)).map { UInt8(bitPattern: $0) }
        return String(decoding: validChars, as: UTF8.self)
    }

    // MARK: - Batch Processing

    /// Create a new batch for token processing
    public static func createBatch(maxTokens: UInt32, embeddingSize: Int32 = 0, numSequences: Int32 = 1) throws -> CBatch {
        let batch = llama_batch_init(Int32(maxTokens), embeddingSize, numSequences)
        return batch
    }

    /// Process tokens directly using llama_batch_get_one (simpler, safer approach)
    public static func processTokensDirect(context: CContext, tokens: [CToken]) throws {
        var tokensCopy = tokens
        let batch = llama_batch_get_one(&tokensCopy, Int32(tokens.count))
        let result = llama_decode(context, batch)

        guard result == 0 else {
            throw CatalystError.batchProcessingFailed(details: "llama_decode returned \(result)")
        }
    }

    /// Free a batch
    public static func freeBatch(_ batch: CBatch) {
        llama_batch_free(batch)
    }

    /// Clear batch for reuse
    public static func clearBatch(_ batch: inout CBatch) {
        batch.n_tokens = 0
    }

    /// Add token to batch
    public static func addTokenToBatch(
        batch: inout CBatch,
        token: CToken,
        position: Int32,
        sequenceId: Int32 = 0,
        generateLogits: Bool = false
    ) {
        let index = Int(batch.n_tokens)

        batch.token?[index] = token
        batch.pos?[index] = llama_pos(position)
        batch.n_seq_id?[index] = 1

        if let seqIdPtr = batch.seq_id?[index] {
            seqIdPtr.pointee = llama_seq_id(sequenceId)
        }

        batch.logits?[index] = generateLogits ? 1 : 0
        batch.n_tokens += 1
    }

    /// Process batch through model
    public static func processBatch(context: CContext, batch: CBatch) throws {
        let result = llama_decode(context, batch)

        guard result == 0 else {
            var errorDetails = "llama_decode returned \(result)"

            if result == 1 {
                errorDetails += " (could not find a KV slot for the batch)"
            } else if result == 2 {
                errorDetails += " (aborted - processed ubatches remain in memory)"
            } else if result < 0 {
                errorDetails += " (invalid input batch)"
            }

            throw CatalystError.batchProcessingFailed(details: errorDetails)
        }
    }

    // MARK: - Sampling

    /// Get logits from last processed batch
    public static func getLogits(context: CContext, batchIndex: Int32 = -1) -> UnsafeMutablePointer<Float>? {
        return llama_get_logits_ith(context, batchIndex)
    }

    /// Get embeddings from last processed batch
    public static func getEmbeddings(context: CContext) -> UnsafeMutablePointer<Float>? {
        return llama_get_embeddings(context)
    }

    /// Get embeddings for a specific sequence
    public static func getEmbeddingsSeq(context: CContext, sequenceId: Int32) -> UnsafeMutablePointer<Float>? {
        return llama_get_embeddings_seq(context, llama_seq_id(sequenceId))
    }

    /// Sample token using greedy strategy
    public static func sampleGreedy(model: CModel, context: CContext, logits: UnsafeMutablePointer<Float>) -> CToken {
        let vocabSize = getVocabularySize(model)
        var bestToken: CToken = 0
        var bestLogit: Float = -Float.infinity

        for i in 0..<Int(vocabSize) {
            if logits[i] > bestLogit {
                bestLogit = logits[i]
                bestToken = CToken(i)
            }
        }

        return bestToken
    }

    /// Sample token using temperature
    public static func sampleWithConfig(
        model: CModel,
        context: CContext,
        logits: UnsafeMutablePointer<Float>,
        config: PredictionConfig
    ) throws -> CToken {
        if config.temperature <= 0.0 {
            return sampleGreedy(model: model, context: context, logits: logits)
        }

        let vocabSize = Int(getVocabularySize(model))
        var probabilities = Array<Float>(repeating: 0.0, count: vocabSize)
        var maxLogit: Float = -Float.infinity

        for i in 0..<vocabSize {
            maxLogit = max(maxLogit, logits[i])
        }

        var sumExp: Float = 0.0
        for i in 0..<vocabSize {
            let scaledLogit = (logits[i] - maxLogit) / config.temperature
            let expValue = exp(scaledLogit)
            probabilities[i] = expValue
            sumExp += expValue
        }

        for i in 0..<vocabSize {
            probabilities[i] /= sumExp
        }

        let randomValue = Float.random(in: 0.0..<1.0)
        var cumulativeProb: Float = 0.0

        for i in 0..<vocabSize {
            cumulativeProb += probabilities[i]
            if randomValue <= cumulativeProb {
                return CToken(i)
            }
        }

        return CToken(vocabSize - 1)
    }

    // MARK: - KV Cache Management
    // Note: KV cache direct access functions are not exposed in the XCFramework binary.
    // Context reset is handled through re-initialization when needed.

    /// Clear the entire KV cache (memory) for fresh generation
    public static func clearKVCache(_ context: CContext) {
        guard let memory = llama_get_memory(context) else {
            print("LlamaBridge: Failed to get memory handle for KV cache clear")
            return
        }
        // Use llama_memory_seq_rm with full range to clear the sequence and reset position tracking
        // -1 for p0 and p1 means "all positions"
        _ = llama_memory_seq_rm(memory, 0, -1, -1)
        print("LlamaBridge: KV cache cleared and position tracking reset for seq 0")
    }

    /// Remove tokens from KV cache in a range
    public static func removeKVCacheTokens(
        context: CContext,
        sequenceId: Int32 = 0,
        startPos: Int32,
        endPos: Int32
    ) {
        guard let memory = llama_get_memory(context) else {
            print("LlamaBridge: Failed to get memory handle for token removal")
            return
        }
        _ = llama_memory_seq_rm(memory, sequenceId, startPos, endPos)
        print("LlamaBridge: Removed tokens from pos \(startPos) to \(endPos)")
    }

    /// Copy sequence in KV cache
    public static func copyKVCacheSequence(
        context: CContext,
        sourceSeqId: Int32,
        destSeqId: Int32,
        startPos: Int32,
        endPos: Int32
    ) {
        guard let memory = llama_get_memory(context) else {
            print("LlamaBridge: Failed to get memory handle for sequence copy")
            return
        }
        llama_memory_seq_cp(memory, sourceSeqId, destSeqId, startPos, endPos)
        print("LlamaBridge: Copied sequence \(sourceSeqId) to \(destSeqId)")
    }

    /// Get the maximum position in KV cache for a sequence
    public static func getKVCacheMaxPos(_ context: CContext, sequenceId: Int32 = 0) -> Int32 {
        guard let memory = llama_get_memory(context) else {
            return -1
        }
        return llama_memory_seq_pos_max(memory, sequenceId)
    }

    /// Get the minimum position in KV cache for a sequence
    public static func getKVCacheMinPos(_ context: CContext, sequenceId: Int32 = 0) -> Int32 {
        guard let memory = llama_get_memory(context) else {
            return -1
        }
        return llama_memory_seq_pos_min(memory, sequenceId)
    }

    // MARK: - Thread Management

    /// Set thread counts for processing
    public static func setThreadCounts(
        context: CContext,
        mainThreads: Int32,
        batchThreads: Int32
    ) {
        llama_set_n_threads(context, mainThreads, batchThreads)
    }

    // MARK: - State Management

    /// Get the size needed for state save
    public static func getStateSize(_ context: CContext) -> Int {
        return llama_state_get_size(context)
    }

    /// Save state to data buffer
    public static func saveState(_ context: CContext, to buffer: UnsafeMutablePointer<UInt8>, size: Int) -> Int {
        return llama_state_get_data(context, buffer, size)
    }

    /// Load state from data buffer
    public static func loadState(_ context: CContext, from buffer: UnsafePointer<UInt8>, size: Int) -> Int {
        return llama_state_set_data(context, buffer, size)
    }

    // MARK: - Sampler Chain API

    /// Create a sampler chain
    public static func createSamplerChain(noPerf: Bool = false) -> CSampler? {
        var params = llama_sampler_chain_default_params()
        params.no_perf = noPerf
        return llama_sampler_chain_init(params)
    }

    /// Add a sampler to the chain
    public static func addSamplerToChain(_ chain: CSampler, sampler: CSampler) {
        llama_sampler_chain_add(chain, sampler)
    }

    /// Sample a token using the sampler chain
    public static func sampleWithChain(_ chain: CSampler, context: CContext, batchIndex: Int32 = -1) -> CToken {
        return llama_sampler_sample(chain, context, batchIndex)
    }

    /// Accept a token (update sampler state)
    public static func acceptToken(_ sampler: CSampler, token: CToken) {
        llama_sampler_accept(sampler, token)
    }

    /// Reset sampler state
    public static func resetSampler(_ sampler: CSampler) {
        llama_sampler_reset(sampler)
    }

    /// Free a sampler
    public static func freeSampler(_ sampler: CSampler) {
        llama_sampler_free(sampler)
    }

    // MARK: - Built-in Samplers

    /// Create greedy sampler
    public static func createGreedySampler() -> CSampler? {
        return llama_sampler_init_greedy()
    }

    /// Create distribution sampler with seed
    public static func createDistSampler(seed: UInt32) -> CSampler? {
        return llama_sampler_init_dist(seed)
    }

    /// Create top-k sampler
    public static func createTopKSampler(k: Int32) -> CSampler? {
        return llama_sampler_init_top_k(k)
    }

    /// Create top-p (nucleus) sampler
    public static func createTopPSampler(p: Float, minKeep: Int = 1) -> CSampler? {
        return llama_sampler_init_top_p(p, minKeep)
    }

    /// Create min-p sampler
    public static func createMinPSampler(p: Float, minKeep: Int = 1) -> CSampler? {
        return llama_sampler_init_min_p(p, minKeep)
    }

    /// Create typical-p sampler
    public static func createTypicalSampler(p: Float, minKeep: Int = 1) -> CSampler? {
        return llama_sampler_init_typical(p, minKeep)
    }

    /// Create temperature sampler
    public static func createTempSampler(temp: Float) -> CSampler? {
        return llama_sampler_init_temp(temp)
    }

    /// Create dynamic temperature sampler
    public static func createTempExtSampler(temp: Float, delta: Float, exponent: Float) -> CSampler? {
        return llama_sampler_init_temp_ext(temp, delta, exponent)
    }

    /// Create XTC sampler
    public static func createXTCSampler(p: Float, t: Float, minKeep: Int = 1, seed: UInt32) -> CSampler? {
        return llama_sampler_init_xtc(p, t, minKeep, seed)
    }

    /// Create Mirostat sampler
    public static func createMirostatSampler(nVocab: Int32, seed: UInt32, tau: Float, eta: Float, m: Int32) -> CSampler? {
        return llama_sampler_init_mirostat(nVocab, seed, tau, eta, m)
    }

    /// Create Mirostat v2 sampler
    public static func createMirostatV2Sampler(seed: UInt32, tau: Float, eta: Float) -> CSampler? {
        return llama_sampler_init_mirostat_v2(seed, tau, eta)
    }

    /// Create repetition penalty sampler
    /// - Parameters:
    ///   - lastN: How many tokens to look back for penalties (0 = disable, -1 = context size)
    ///   - repeatPenalty: Penalty for repeating tokens (1.0 = disabled)
    ///   - freqPenalty: Penalty based on frequency (0.0 = disabled)
    ///   - presencePenalty: Penalty based on presence (0.0 = disabled)
    public static func createPenaltiesSampler(
        lastN: Int32,
        repeatPenalty: Float,
        freqPenalty: Float,
        presencePenalty: Float
    ) -> CSampler? {
        return llama_sampler_init_penalties(
            lastN,              // penalty_last_n
            repeatPenalty,      // penalty_repeat
            freqPenalty,        // penalty_freq
            presencePenalty     // penalty_present
        )
    }

    // MARK: - Grammar Sampler

    /// Create a grammar-based sampler for constrained generation
    /// - Parameters:
    ///   - model: The model to use for grammar parsing
    ///   - grammarStr: The grammar string in GBNF format
    ///   - rootRule: The root rule name (default: "root")
    /// - Returns: A grammar sampler or nil if creation failed
    public static func createGrammarSampler(
        model: CModel,
        grammarStr: String,
        rootRule: String = "root"
    ) -> CSampler? {
        return grammarStr.withCString { grammarPtr in
            rootRule.withCString { rootPtr in
                return llama_sampler_init_grammar(model, grammarPtr, rootPtr)
            }
        }
    }
}
