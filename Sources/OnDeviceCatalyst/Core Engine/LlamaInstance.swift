//
//  LlamaInstanceActor.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/29/25.
//

//
//  LlamaInstance.swift
//  OnDeviceCatalyst
//
//  Main model instance managing the complete lifecycle of model operations
//

import Foundation
import llama

/// Main model instance handling model lifecycle, inference, and streaming generation
public class LlamaInstance {
    
    // MARK: - Properties
    
    public let profile: ModelProfile
    public private(set) var settings: InstanceSettings
    public private(set) var predictionConfig: PredictionConfig
    
    internal var cModel: CModel?
    internal var cContext: CContext?
    private var cBatch: CBatch?
    private var samplingEngine: SamplingEngine?
    
    /// Expose model for internal use (e.g., embedding extraction)
    internal var model: CModel? { cModel }
    
    /// Expose context for internal use (e.g., embedding extraction)
    internal var context: CContext? { cContext }
    
    private let promptFormatter: StandardPromptFormatter
    private var stopSequenceHandler: StopSequenceHandler
    
    private var contextTokens: [CToken] = []
    private var isGenerating: Bool = false
    private var shouldInterrupt: Bool = false
    
    private var loadingContinuation: AsyncStream<LoadProgress>.Continuation?
    
    // MARK: - Initialization
    
    public init(
        profile: ModelProfile,
        settings: InstanceSettings,
        predictionConfig: PredictionConfig,
        formatter: StandardPromptFormatter? = nil,
        customStopSequences: [String] = []
    ) {
        self.profile = profile
        self.settings = settings
        self.predictionConfig = predictionConfig
        self.promptFormatter = formatter ?? StandardPromptFormatter()
        self.stopSequenceHandler = StopSequenceHandler(
            architecture: profile.architecture,
            customStopSequences: customStopSequences
        )
    }
    
    deinit {
        cleanup()
    }
    
    // MARK: - Lifecycle Management
    
    /// Current state of the instance
    public var isReady: Bool {
        return cModel != nil && cContext != nil && cBatch != nil && samplingEngine != nil
    }
    
    /// Loads and initializes the model
    public func initialize() -> AsyncStream<LoadProgress> {
        return AsyncStream { continuation in
            loadingContinuation = continuation
            
            Task {
                await self.performInitialization()
            }
        }
    }
    
    private func performInitialization() async {
        guard !isReady else {
            publishProgress(.ready("Model already initialized"))
            return
        }
        
        do {
            publishProgress(.preparing("Validating model file"))
            try profile.validateModel()
            
            publishProgress(.loading("Initializing llama.cpp backend"))
            LlamaBridge.initializeBackend()
            
            publishProgress(.loading("Loading model from \(profile.filePath)"))
            cModel = try LlamaBridge.loadModel(path: profile.filePath, settings: settings)
            
            guard let model = cModel else {
                throw CatalystError.modelLoadingFailed(details: "Model pointer is nil after loading")
            }
            
            publishProgress(.loading("Creating inference context"))
            cContext = try LlamaBridge.createContext(model: model, settings: settings)
            
            guard let context = cContext else {
                throw CatalystError.contextCreationFailed(details: "Context pointer is nil after creation")
            }
            
            publishProgress(.loading("Setting up batch processing"))
            cBatch = try LlamaBridge.createBatch(maxTokens: settings.batchSize)
            
            publishProgress(.loading("Initializing sampling engine"))
            samplingEngine = SamplingEngine(model: model, context: context)
            
            // Skip warmup for encoder-only models (BERT, etc.) - they don't do generation
            if !profile.architecture.isEncoderOnly {
                publishProgress(.loading("Performing warmup inference"))
                try await performWarmup()
            } else {
                publishProgress(.loading("Encoder model ready (no warmup needed)"))
            }
            
            publishProgress(.ready("Model ready for inference"))
            
        } catch let error as CatalystError {
            await handleInitializationError(error)
        } catch {
            await handleInitializationError(.unknown(details: error.localizedDescription))
        }
    }
    
    private func performWarmup() async throws {
        guard let model = cModel, let context = cContext, let batch = cBatch else {
            throw CatalystError.engineNotInitialized
        }
        
        // Tokenize a simple warmup prompt
        let warmupText = "Hello"
        let tokens = try LlamaBridge.tokenize(text: warmupText, model: model, addBos: true)
        
        guard !tokens.isEmpty else {
            throw CatalystError.tokenizationFailed(text: warmupText, reason: "No tokens generated")
        }
        
        // Create a temporary batch for warmup
        var warmupBatch = batch
        LlamaBridge.clearBatch(&warmupBatch)
        LlamaBridge.addTokenToBatch(
            batch: &warmupBatch,
            token: tokens[0],
            position: 0,
            sequenceId: 0,
            generateLogits: true
        )
        
        try LlamaBridge.processBatch(context: context, batch: warmupBatch)
        
        // Clear context for actual use
        LlamaBridge.clearKVCache(context)
        contextTokens.removeAll()
        
        // Don't need to update stored batch since this was just warmup
    }

    // MARK: - Embeddings

    /// Compute a single embedding vector for the given text using the current model/context.
    ///
    /// This is a synchronous helper intended for use by higher-level components like
    /// memory systems (e.g. SwiftMem). It should not be called while generation is
    /// in progress.
    ///
    /// For decoder-only models like Qwen2.5, we use mean pooling over the last token's
    /// logits as a pseudo-embedding. This is not as good as a dedicated embedding model
    /// but provides reasonable semantic similarity for memory retrieval.
    public func embed(text: String) throws -> [Float] {
        guard isReady else {
            throw CatalystError.engineNotInitialized
        }
        guard !isGenerating else {
            throw CatalystError.generationFailed(details: "Cannot compute embeddings while generation is in progress")
        }
        guard let model = cModel, let context = cContext else {
            throw CatalystError.engineNotInitialized
        }

        let isEncoder = profile.architecture.isEncoderOnly
        
        // Tokenize input - BERT models handle their own special tokens via tokenizer
        let tokens = try LlamaBridge.tokenize(
            text: text,
            model: model,
            addBos: !isEncoder && profile.architecture.requiresSpecialTokens
        )
        guard !tokens.isEmpty else {
            throw CatalystError.tokenizationFailed(text: text, reason: "No tokens generated for embedding")
        }

        let embSize = LlamaBridge.getEmbeddingSize(model)
        
        // Create a temporary batch configured for embeddings
        var embBatch = try LlamaBridge.createBatch(
            maxTokens: UInt32(tokens.count),
            embeddingSize: embSize,
            numSequences: 1
        )
        defer { LlamaBridge.freeBatch(embBatch) }
        
        LlamaBridge.clearBatch(&embBatch)
        
        var position: Int32 = 0
        for token in tokens {
            LlamaBridge.addTokenToBatch(
                batch: &embBatch,
                token: token,
                position: position,
                sequenceId: 0,
                generateLogits: false  // Embeddings don't need logits
            )
            position += 1
        }

        // Run the batch to compute embeddings
        try LlamaBridge.processBatch(context: context, batch: embBatch)

        // Get pooled embeddings
        if let embPtr = LlamaBridge.getEmbeddings(context: context) {
            let embCount = Int(embSize)
            let buffer = UnsafeBufferPointer(start: embPtr, count: embCount)
            var embedding = Array(buffer)
            
            // L2 normalize
            let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
            if norm > 0 {
                embedding = embedding.map { $0 / norm }
            }
            
            LlamaBridge.clearKVCache(context)
            return embedding
        }
        
        throw CatalystError.generationFailed(details: "Failed to obtain embeddings from llama context")
    }
    
    private func handleInitializationError(_ error: CatalystError) async {
        print("Catalyst: Initialization failed - \(error.localizedDescription)")
        
        // Attempt cleanup
        cleanup()
        
        // Try fallback initialization if error is recoverable
        if error.isRecoverable {
            await attemptFallbackInitialization(originalError: error)
        } else {
            publishProgress(.failed(error.localizedDescription))
        }
    }
    
    private func attemptFallbackInitialization(originalError: CatalystError) async {
        print("Catalyst: Attempting fallback initialization")
        
        // Try with reduced settings
        var fallbackSettings = settings
        fallbackSettings.contextLength = min(settings.contextLength, 2048)
        fallbackSettings.batchSize = min(settings.batchSize, 256)
        fallbackSettings.gpuLayers = 0 // CPU only
        
        do {
            publishProgress(.loading("Retrying with fallback settings"))
            
            // Retry with fallback settings
            cModel = try LlamaBridge.loadModel(path: profile.filePath, settings: fallbackSettings)
            guard let model = cModel else {
                throw CatalystError.modelLoadingFailed(details: "Fallback model loading failed")
            }
            
            cContext = try LlamaBridge.createContext(model: model, settings: fallbackSettings)
            guard let context = cContext else {
                throw CatalystError.contextCreationFailed(details: "Fallback context creation failed")
            }
            
            cBatch = try LlamaBridge.createBatch(maxTokens: fallbackSettings.batchSize)
            samplingEngine = SamplingEngine(model: model, context: context)
            
            // Update settings to working configuration
            settings = fallbackSettings
            
            try await performWarmup()
            
            publishProgress(.ready("Model ready with fallback settings (CPU-only, reduced context)"))
            
        } catch {
            publishProgress(.failed("Both primary and fallback initialization failed: \(error.localizedDescription)"))
        }
    }
    
    /// Safely shutdown the instance
    public func shutdown() async {
        shouldInterrupt = true
        cleanup()
    }
    
    private func cleanup() {
        if let batch = cBatch {
            LlamaBridge.freeBatch(batch)
            cBatch = nil
        }
        
        if let context = cContext {
            LlamaBridge.freeContext(context)
            cContext = nil
        }
        
        if let model = cModel {
            LlamaBridge.freeModel(model)
            cModel = nil
        }
        
        samplingEngine = nil
        contextTokens.removeAll()
        isGenerating = false
        shouldInterrupt = false
        
        loadingContinuation?.finish()
        loadingContinuation = nil
    }
    
    // MARK: - Generation
    
    /// Generate streaming response for a conversation
    public func generate(
        conversation: [Turn],
        systemPrompt: String? = nil,
        overrideConfig: PredictionConfig? = nil
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        
        print("LlamaInstance.generate: Creating AsyncThrowingStream")
        print("LlamaInstance.generate: Conversation has \(conversation.count) turns")
        
        return AsyncThrowingStream { continuation in
            print("LlamaInstance.generate: Inside AsyncThrowingStream closure")
            Task {
                print("LlamaInstance.generate: Starting Task")
                do {
                    print("LlamaInstance.generate: Calling performGeneration...")
                    try await self.performGeneration(
                        conversation: conversation,
                        systemPrompt: systemPrompt,
                        config: overrideConfig ?? self.predictionConfig,
                        continuation: continuation
                    )
                    print("LlamaInstance.generate: performGeneration completed normally")
                } catch {
                    print("LlamaInstance.generate: ERROR in performGeneration: \(error)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    private func performGeneration(
        conversation: [Turn],
        systemPrompt: String?,
        config: PredictionConfig,
        continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws {
        
        print("LlamaInstance.performGeneration: Starting")
        print("LlamaInstance.performGeneration: Ready status: \(isReady)")
        print("LlamaInstance.performGeneration: Currently generating: \(isGenerating)")
        
        guard isReady else {
            print("LlamaInstance.performGeneration: ERROR - Engine not initialized")
            throw CatalystError.engineNotInitialized
        }
        
        guard !isGenerating else {
            print("LlamaInstance.performGeneration: ERROR - Generation already in progress")
            throw CatalystError.generationFailed(details: "Generation already in progress")
        }
        
        isGenerating = true
        shouldInterrupt = false
        defer { 
            print("LlamaInstance.performGeneration: Cleaning up, setting isGenerating = false")
            isGenerating = false 
        }
        
        let startTime = Date()
        var tokensGenerated = 0
        
        do {
            print("LlamaInstance.performGeneration: Formatting prompt...")
            // Format prompt
            let prompt = promptFormatter.formatPrompt(
                turns: conversation,
                systemPrompt: systemPrompt,
                architecture: profile.architecture
            )
            print("LlamaInstance.performGeneration: Formatted prompt length: \(prompt.count) characters")
            print("LlamaInstance.performGeneration: Prompt preview: '\(String(prompt.prefix(100)))...'")
            
            // **QUESTION 1 ANSWER: Check tokenization**
            print("LlamaInstance.performGeneration: Starting tokenization...")
            guard let model = cModel else {
                print("LlamaInstance.performGeneration: ERROR - No model available for tokenization")
                throw CatalystError.engineNotInitialized
            }
            
            // Tokenize prompt
            let promptTokens = try LlamaBridge.tokenize(
                text: prompt,
                model: cModel!,
                addBos: profile.architecture.requiresSpecialTokens
            )
            
            print("LlamaInstance.performGeneration: ✅ TOKENIZATION SUCCESS - \(promptTokens.count) tokens")
            print("LlamaInstance.performGeneration: First 10 tokens: \(Array(promptTokens.prefix(10)))")
            
            // Check context limits
            let availableContext = Int(settings.contextLength) - promptTokens.count
            print("LlamaInstance.performGeneration: Available context: \(availableContext) tokens")
            
            guard availableContext > 0 else {
                print("LlamaInstance.performGeneration: ERROR - Context window exceeded")
                throw CatalystError.contextWindowExceeded(
                    tokenCount: promptTokens.count,
                    limit: Int(settings.contextLength)
                )
            }
            
            print("LlamaInstance.performGeneration: Processing prompt tokens...")
            // Process prompt tokens
            try await processPromptTokens(promptTokens)
            print("LlamaInstance.performGeneration: ✅ Prompt tokens processed successfully")
            
            // Generate tokens
            let maxNewTokens = config.effectiveMaxTokens(
                contextLength: settings.contextLength,
                promptTokens: promptTokens.count
            )
            print("LlamaInstance.performGeneration: Max new tokens to generate: \(maxNewTokens)")
            
            print("LlamaInstance.performGeneration: 🚀 Starting token generation...")
            tokensGenerated = try await generateTokens(
                maxTokens: maxNewTokens,
                config: config,
                continuation: continuation
            )
            print("LlamaInstance.performGeneration: ✅ Token generation completed: \(tokensGenerated) tokens")
            
            // Send completion
            let duration = Date().timeIntervalSince(startTime)
            let metadata = ResponseMetadata(
                completionReason: .natural,
                tokensGenerated: tokensGenerated,
                tokensPerSecond: Double(tokensGenerated) / duration,
                generationTimeMs: Int64(duration * 1000),
                promptTokens: promptTokens.count,
                totalTokens: promptTokens.count + tokensGenerated
            )
            
            let completionChunk = StreamChunk.completion(reason: .natural, metadata: metadata)
            continuation.yield(completionChunk)
            continuation.finish()
            
        } catch {
            let errorReason = CompletionReason.error(error.localizedDescription)
            let errorChunk = StreamChunk.completion(reason: errorReason)
            continuation.yield(errorChunk)
            continuation.finish(throwing: error)
        }
    }
    
    private func processPromptTokens(_ tokens: [CToken]) async throws {
        guard let context = cContext, var batch = cBatch else {
            throw CatalystError.engineNotInitialized
        }
        
        // Find common prefix with existing context
        let commonPrefixLength = findCommonPrefixLength(tokens)
        
        // Remove divergent tokens from KV cache if needed
        if commonPrefixLength < contextTokens.count {
            LlamaBridge.removeKVCacheTokens(
                context: context,
                sequenceId: 0,
                startPos: Int32(commonPrefixLength),
                endPos: Int32(contextTokens.count)
            )
        }
        
        // Process new tokens
        let newTokens = Array(tokens[commonPrefixLength...])
        var position = Int32(commonPrefixLength)
        
        var tokenIndex = 0
        while tokenIndex < newTokens.count {
            guard !shouldInterrupt else {
                throw CatalystError.operationCancelled
            }
            
            LlamaBridge.clearBatch(&batch)
            
            // Fill batch
            let batchEnd = min(tokenIndex + Int(settings.batchSize), newTokens.count)
            for i in tokenIndex..<batchEnd {
                let generateLogits = (i == batchEnd - 1) // Only generate logits for last token
                LlamaBridge.addTokenToBatch(
                    batch: &batch,
                    token: newTokens[i],
                    position: position + Int32(i - tokenIndex),
                    sequenceId: 0,
                    generateLogits: generateLogits
                )
            }
            
            // Process batch
            try LlamaBridge.processBatch(context: context, batch: batch)
            
            position += Int32(batchEnd - tokenIndex)
            tokenIndex = batchEnd
        }
        
        contextTokens = tokens
    }
    
    private func generateTokens(
        maxTokens: Int,
        config: PredictionConfig,
        continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws -> Int {
        
        guard let model = cModel, let context = cContext, var batch = cBatch,
              let sampler = samplingEngine else {
            throw CatalystError.engineNotInitialized
        }
        
        let processor = StreamProcessor(stopHandler: stopSequenceHandler)
        var generatedCount = 0
        
        // Start performance monitoring
        PerformanceMonitor.startInferenceMonitoring()
        
        print("Catalyst: Starting generation with \(contextTokens.count) context tokens")
        print("Catalyst: Max tokens: \(maxTokens)")
        
        // Process an empty batch to prepare for logits generation
        LlamaBridge.clearBatch(&batch)
        LlamaBridge.addTokenToBatch(
            batch: &batch,
            token: contextTokens.last ?? 0,
            position: Int32(contextTokens.count - 1),
            sequenceId: 0,
            generateLogits: true
        )
        
        print("Catalyst: Processing initial batch...")
        try LlamaBridge.processBatch(context: context, batch: batch)
        print("Catalyst: Initial batch processed, starting generation loop")
        
        while generatedCount < maxTokens {
            guard !shouldInterrupt else {
                let chunk = StreamChunk.completion(reason: .userCancelled)
                continuation.yield(chunk)
                return generatedCount
            }
            
            // Get logits and sample next token
            guard let logits = LlamaBridge.getLogits(context: context, batchIndex: -1) else {
                print("Catalyst: ERROR - Failed to get logits at generation step \(generatedCount)")
                throw CatalystError.generationFailed(details: "Failed to get logits")
            }
            
            print("Catalyst: Got logits for token \(generatedCount), sampling...")
            
            let recentTokens = Array(contextTokens.suffix(Int(config.repetitionPenaltyRange)))
            let sampledToken = try sampler.sampleToken(
                logits: logits,
                config: config,
                recentTokens: recentTokens
            )
            
            print("Catalyst: Sampled token: \(sampledToken)")
            
            // Check for end of generation
            if LlamaBridge.isEndOfGeneration(model, token: sampledToken) {
                print("Catalyst: End of generation token detected")
                let chunk = StreamChunk.completion(reason: .natural)
                continuation.yield(chunk)
                return generatedCount
            }
            
            // Convert token to text
            let tokenText = LlamaBridge.detokenize(token: sampledToken, model: model)
            
            // Debug: Log token generation
            print("Catalyst: Generated token \(sampledToken) -> '\(tokenText)'")
            
            // Record token for performance monitoring
            PerformanceMonitor.recordToken()
            
            // Process through stop sequence handler
            let chunks = processor.processToken(tokenText)
            print("Catalyst: Processed \(chunks.count) chunks from token")
            
            for chunk in chunks {
                print("Catalyst: Yielding chunk: \(chunk)")
                continuation.yield(chunk)
                
                if chunk.isComplete {
                    print("Catalyst: Generation complete")
                    // End performance monitoring
                    let _ = PerformanceMonitor.endInferenceMonitoring(settings: settings)
                    return generatedCount
                }
            }
            
            // Add token to context
            contextTokens.append(sampledToken)
            generatedCount += 1
            
            // Check context limits
            if contextTokens.count >= settings.contextLength {
                let chunk = StreamChunk.completion(reason: .contextWindowFull)
                continuation.yield(chunk)
                return generatedCount
            }
            
            // Prepare next iteration
            LlamaBridge.clearBatch(&batch)
            LlamaBridge.addTokenToBatch(
                batch: &batch,
                token: sampledToken,
                position: Int32(contextTokens.count - 1),
                sequenceId: 0,
                generateLogits: true
            )
            
            try LlamaBridge.processBatch(context: context, batch: batch)
            
            // Minimal yielding for maximum speed
            if generatedCount % 50 == 0 {
                await Task.yield()
            }
        }
        
        // Max tokens reached
        let chunk = StreamChunk.completion(reason: .maxTokensReached)
        continuation.yield(chunk)
        
        // End performance monitoring
        let _ = PerformanceMonitor.endInferenceMonitoring(settings: settings)
        
        return generatedCount
    }
    
    private func findCommonPrefixLength(_ newTokens: [CToken]) -> Int {
        var commonLength = 0
        let maxComparable = min(contextTokens.count, newTokens.count)
        
        for i in 0..<maxComparable {
            if contextTokens[i] == newTokens[i] {
                commonLength += 1
            } else {
                break
            }
        }
        
        return commonLength
    }
    
    // MARK: - Control
    
    /// Interrupt current generation
    public func interrupt() {
        shouldInterrupt = true
    }
    
    /// Update prediction configuration
    public func updateConfig(_ newConfig: PredictionConfig) {
        predictionConfig = newConfig
    }
    
    /// Clear conversation context
    public func clearContext() {
        guard let context = cContext else { return }
        
        LlamaBridge.clearKVCache(context)
        contextTokens.removeAll()
        print("Catalyst: Context cleared")
    }
    
    // MARK: - Private Helpers
    
    private func publishProgress(_ progress: LoadProgress) {
        loadingContinuation?.yield(progress)
        
        if case .ready = progress, case .failed = progress {
            loadingContinuation?.finish()
            loadingContinuation = nil
        }
    }
}

// MARK: - Load Progress

public enum LoadProgress {
    case preparing(String)
    case loading(String)
    case ready(String)
    case failed(String)
    
    public var isComplete: Bool {
        switch self {
        case .ready, .failed:
            return true
        case .preparing, .loading:
            return false
        }
    }
    
    public var message: String {
        switch self {
        case .preparing(let msg), .loading(let msg), .ready(let msg), .failed(let msg):
            return msg
        }
    }
}
