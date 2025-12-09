//
//  LlamaBridge.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//
//
//  LlamaBridge.swift
//  OnDeviceCatalyst
//
//  Safe Swift wrapper around llama.cpp C API
//

import Foundation
import llama

// MARK: - C Type Aliases
public typealias CModel = OpaquePointer
public typealias CContext = OpaquePointer
public typealias CBatch = llama_batch
public typealias CToken = llama_token

/// Safe bridge to llama.cpp C API with comprehensive error handling
public enum LlamaBridge {
    
    // MARK: - Backend Management
    
    /// Initialize the llama.cpp backend (call once at app start)
    public static func initializeBackend() {
        llama_backend_init()
        print("Catalyst: llama.cpp backend initialized")
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
        
        // Attempt model loading
        guard let model = llama_load_model_from_file(path, params) else {
            throw createModelLoadingError(path: path, settings: settings)
        }
        
        print("Catalyst: Model loaded successfully")
        return model
    }
    
    /// Creates appropriate error for model loading failure
    private static func createModelLoadingError(path: String, settings: InstanceSettings) -> CatalystError {
        let filename = (path as NSString).lastPathComponent.lowercased()
        
        // Check for architecture-specific issues
        if filename.contains("qwen2.5") || filename.contains("qwen25") {
            return .architectureUnsupported(
                architecture: "qwen2.5",
                suggestion: "Qwen 2.5 requires llama.cpp built after October 2024. Try reducing GPU layers or using CPU-only mode."
            )
        } else if filename.contains("llama3.1") || filename.contains("llama-3.1") {
            return .architectureUnsupported(
                architecture: "llama3.1",
                suggestion: "Llama 3.1 requires recent llama.cpp version. Try reducing context length or GPU layers."
            )
        } else if filename.contains("phi3.5") {
            return .architectureUnsupported(
                architecture: "phi3.5",
                suggestion: "Phi 3.5 requires llama.cpp from September 2024 or later."
            )
        }
        
        // Generic loading failure
        var details = "Failed to load model. This could be due to:"
        details += "\n• Unsupported model architecture"
        details += "\n• Corrupted model file"
        details += "\n• Insufficient memory"
        details += "\n• Incompatible llama.cpp version"
        
        if settings.gpuLayers > 0 {
            details += "\n\nTry setting GPU layers to 0 for CPU-only mode."
        }
        
        return .modelLoadingFailed(details: details)
    }
    
    /// Validates model file before loading
    private static func validateModelFile(path: String) throws {
        // Check file exists
        guard FileManager.default.fileExists(atPath: path) else {
            throw CatalystError.modelFileNotFound(path: path)
        }
        
        // Check file size
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
        
        // Validate GGUF magic header
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
        params.flash_attn = settings.useFlashAttention
        params.n_threads = settings.cpuThreads
        params.n_threads_batch = settings.cpuThreads
        
        print("Catalyst: Creating context - ctx:\(settings.contextLength), batch:\(settings.batchSize), threads:\(settings.cpuThreads)")
        
        guard let context = llama_new_context_with_model(model, params) else {
            throw CatalystError.contextCreationFailed(
                details: "llama_new_context_with_model failed. Try reducing context length, batch size, or GPU layers."
            )
        }
        
        // Validate context was created with expected parameters
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
        return llama_n_vocab(model)
    }
    
    /// Get special tokens
    public static func getBosToken(_ model: CModel) -> CToken {
        return llama_token_bos(model)
    }
    
    public static func getEosToken(_ model: CModel) -> CToken {
        return llama_token_eos(model)
    }

    /// Get model embedding size (hidden dimension)
    public static func getEmbeddingSize(_ model: CModel) -> Int32 {
        return llama_n_embd(model)
    }
    
    /// Check if token indicates end of generation
    public static func isEndOfGeneration(_ model: CModel, token: CToken) -> Bool {
        return token == llama_token_eos(model) || llama_token_is_eog(model, token)
    }
    
    // MARK: - Tokenization
    
    /// Tokenize text with proper error handling
    public static func tokenize(
        text: String,
        model: CModel,
        addBos: Bool = true,
        parseSpecial: Bool = true
    ) throws -> [CToken] {
        let utf8Count = text.utf8.count
        let maxTokens = utf8Count + (addBos ? 1 : 0) + 1
        var tokens = Array<CToken>(repeating: 0, count: maxTokens)
        
        let tokenCount = llama_tokenize(
            model,
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
        let bufferSize = 128
        var buffer = Array<CChar>(repeating: 0, count: bufferSize)
        
        let charCount = llama_token_to_piece(
            model,
            token,
            &buffer,
            Int32(bufferSize),
            0,
            false
        )
        
        guard charCount > 0 else {
            return "" // Empty token or error
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
        
        // All batch arrays may be nil for embedding batches - guard each access
        batch.token?[index] = token
        batch.pos?[index] = llama_pos(position)
        batch.n_seq_id?[index] = 1
        
        // seq_id may be nil for embedding batches - only set if available
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
            
            // Provide specific error context
            if result == 1 {
                errorDetails += " (could not find a KV slot for the batch)"
            } else if result < 0 {
                errorDetails += " (negative return indicates serious error)"
            }
            
            throw CatalystError.batchProcessingFailed(details: errorDetails)
        }
    }
    
    // MARK: - Sampling
    
    /// Get logits from last processed batch
    public static func getLogits(context: CContext, batchIndex: Int32 = -1) -> UnsafeMutablePointer<Float>? {
        return llama_get_logits_ith(context, batchIndex)
    }

    /// Get embeddings from last processed batch (when embeddingSize > 0 in batch)
    public static func getEmbeddings(context: CContext) -> UnsafeMutablePointer<Float>? {
        return llama_get_embeddings(context)
    }
    
    /// Sample token using greedy strategy
    public static func sampleGreedy(model: CModel, context: CContext, logits: UnsafeMutablePointer<Float>) -> CToken {
        let vocabSize = llama_n_vocab(model)
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
    
    /// Sample token using temperature and top-p
    public static func sampleWithConfig(
        model: CModel,
        context: CContext,
        logits: UnsafeMutablePointer<Float>,
        config: PredictionConfig
    ) throws -> CToken {
        // For now, implement simple temperature scaling
        // More sophisticated sampling will be added in SamplingEngine
        
        if config.temperature <= 0.0 {
            return sampleGreedy(model: model, context: context, logits: logits)
        }
        
        // Apply temperature scaling
        let vocabSize = Int(llama_n_vocab(model))
        var probabilities = Array<Float>(repeating: 0.0, count: vocabSize)
        var maxLogit: Float = -Float.infinity
        
        // Find max logit for numerical stability
        for i in 0..<vocabSize {
            maxLogit = max(maxLogit, logits[i])
        }
        
        // Apply temperature and compute probabilities
        var sumExp: Float = 0.0
        for i in 0..<vocabSize {
            let scaledLogit = (logits[i] - maxLogit) / config.temperature
            let expValue = exp(scaledLogit)
            probabilities[i] = expValue
            sumExp += expValue
        }
        
        // Normalize probabilities
        for i in 0..<vocabSize {
            probabilities[i] /= sumExp
        }
        
        // Sample from distribution
        let randomValue = Float.random(in: 0.0..<1.0)
        var cumulativeProb: Float = 0.0
        
        for i in 0..<vocabSize {
            cumulativeProb += probabilities[i]
            if randomValue <= cumulativeProb {
                return CToken(i)
            }
        }
        
        // Fallback to last token if rounding errors occur
        return CToken(vocabSize - 1)
    }
    
    // MARK: - KV Cache Management
    
    /// Clear the entire KV cache
    public static func clearKVCache(_ context: CContext) {
        llama_kv_cache_clear(context)
    }
    
    /// Remove tokens from KV cache in a range
    public static func removeKVCacheTokens(
        context: CContext,
        sequenceId: Int32 = 0,
        startPos: Int32,
        endPos: Int32
    ) {
        llama_kv_cache_seq_rm(
            context,
            llama_seq_id(sequenceId),
            llama_pos(startPos),
            llama_pos(endPos)
        )
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
}
