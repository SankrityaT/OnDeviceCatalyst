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

/// Main model instance handling model lifecycle, inference, and streaming generation
public class LlamaInstance {

    // MARK: - Properties

    public let profile: ModelProfile
    public private(set) var settings: InstanceSettings
    public private(set) var predictionConfig: PredictionConfig

    /// The inference backend (llama.cpp or Metal)
    internal var backend: InferenceBackend?
    private var samplingEngine: SamplingEngine?

    private let promptFormatter: StandardPromptFormatter
    private var stopSequenceHandler: StopSequenceHandler

    private var contextTokens: [Int32] = []
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
        return backend != nil && samplingEngine != nil
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

            // Create the appropriate backend
            let newBackend: InferenceBackend
            switch settings.backendType {
            case .llamaCpp:
                publishProgress(.loading("Initializing llama.cpp backend"))
                newBackend = LlamaCppBackend()
            case .metal:
                publishProgress(.loading("Initializing Metal inference engine"))
                newBackend = MetalBackend()
            }

            publishProgress(.loading("Loading model from \(profile.filePath)"))
            try newBackend.loadModel(path: profile.filePath, settings: settings)

            publishProgress(.loading("Creating inference context"))
            try newBackend.createContext(settings: settings)

            self.backend = newBackend

            publishProgress(.loading("Initializing sampling engine"))
            samplingEngine = SamplingEngine(vocabularySize: newBackend.vocabularySize)

            // Skip warmup for encoder-only models (BERT, etc.) - they don't do generation
            if !profile.architecture.isEncoderOnly {
                publishProgress(.loading("Performing warmup inference"))
                try newBackend.warmup()
                newBackend.clearKVCache()
                contextTokens.removeAll()
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

    // MARK: - Embeddings

    /// Compute a single embedding vector for the given text using the current model/context.
    public func embed(text: String) throws -> [Float] {
        guard isReady else {
            throw CatalystError.engineNotInitialized
        }
        guard !isGenerating else {
            throw CatalystError.generationFailed(details: "Cannot compute embeddings while generation is in progress")
        }
        guard let backend = backend else {
            throw CatalystError.engineNotInitialized
        }

        let isEncoder = profile.architecture.isEncoderOnly

        // Tokenize input - BERT models handle their own special tokens via tokenizer
        let tokens = try backend.tokenize(
            text: text,
            addBos: !isEncoder && profile.architecture.requiresSpecialTokens,
            parseSpecial: true
        )
        guard !tokens.isEmpty else {
            throw CatalystError.tokenizationFailed(text: text, reason: "No tokens generated for embedding")
        }

        let embSize = backend.embeddingSize

        // Process tokens for embedding extraction
        try backend.processTokensForEmbedding(tokens)

        // Get pooled embeddings
        if let embPtr = backend.getEmbeddings() {
            let embCount = Int(embSize)
            let buffer = UnsafeBufferPointer(start: embPtr, count: embCount)
            var embedding = Array(buffer)

            // L2 normalize
            let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
            if norm > 0 {
                embedding = embedding.map { $0 / norm }
            }

            backend.clearKVCache()
            return embedding
        }

        throw CatalystError.generationFailed(details: "Failed to obtain embeddings from context")
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

            let newBackend = LlamaCppBackend()
            try newBackend.loadModel(path: profile.filePath, settings: fallbackSettings)
            try newBackend.createContext(settings: fallbackSettings)

            self.backend = newBackend
            samplingEngine = SamplingEngine(vocabularySize: newBackend.vocabularySize)

            // Update settings to working configuration
            settings = fallbackSettings

            try newBackend.warmup()
            newBackend.clearKVCache()
            contextTokens.removeAll()

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
        backend?.shutdown()
        backend = nil

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

        guard let backend = backend else {
            throw CatalystError.engineNotInitialized
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

            // Tokenize prompt
            print("LlamaInstance.performGeneration: Starting tokenization...")
            let promptTokens = try backend.tokenize(
                text: prompt,
                addBos: profile.architecture.requiresSpecialTokens,
                parseSpecial: true
            )

            print("LlamaInstance.performGeneration: Tokenization success - \(promptTokens.count) tokens")
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
            try await processPromptTokens(promptTokens, backend: backend)
            print("LlamaInstance.performGeneration: Prompt tokens processed successfully")

            // Generate tokens
            let maxNewTokens = config.effectiveMaxTokens(
                contextLength: settings.contextLength,
                promptTokens: promptTokens.count
            )
            print("LlamaInstance.performGeneration: Max new tokens to generate: \(maxNewTokens)")

            print("LlamaInstance.performGeneration: Starting token generation...")
            tokensGenerated = try await generateTokens(
                maxTokens: maxNewTokens,
                config: config,
                backend: backend,
                continuation: continuation
            )
            print("LlamaInstance.performGeneration: Token generation completed: \(tokensGenerated) tokens")

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

    private func processPromptTokens(_ tokens: [Int32], backend: InferenceBackend) async throws {
        // Find common prefix with existing context
        let commonPrefixLength = findCommonPrefixLength(tokens)

        // Remove divergent tokens from KV cache if needed
        if commonPrefixLength < contextTokens.count {
            backend.removeKVCacheTokens(
                sequenceId: 0,
                startPos: Int32(commonPrefixLength),
                endPos: Int32(contextTokens.count)
            )
        }

        // Process new tokens
        let newTokens = Array(tokens[commonPrefixLength...])
        guard !newTokens.isEmpty else {
            contextTokens = tokens
            return
        }

        let startPosition = Int32(commonPrefixLength)
        let positions = newTokens.indices.map { startPosition + Int32($0) }

        // Process in batches
        var tokenIndex = 0
        while tokenIndex < newTokens.count {
            guard !shouldInterrupt else {
                throw CatalystError.operationCancelled
            }

            let batchEnd = min(tokenIndex + Int(settings.batchSize), newTokens.count)
            let batchTokens = Array(newTokens[tokenIndex..<batchEnd])
            let batchPositions = Array(positions[tokenIndex..<batchEnd])
            let isLastBatch = (batchEnd == newTokens.count)

            try backend.processTokens(
                batchTokens,
                positions: batchPositions,
                generateLogitsAtLast: isLastBatch
            )

            tokenIndex = batchEnd
        }

        contextTokens = tokens
    }

    private func generateTokens(
        maxTokens: Int,
        config: PredictionConfig,
        backend: InferenceBackend,
        continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws -> Int {

        guard let sampler = samplingEngine else {
            throw CatalystError.engineNotInitialized
        }

        let processor = StreamProcessor(stopHandler: stopSequenceHandler)
        var generatedCount = 0

        // Start performance monitoring
        PerformanceMonitor.startInferenceMonitoring()

        print("Catalyst: Starting generation with \(contextTokens.count) context tokens")
        print("Catalyst: Max tokens: \(maxTokens)")
        print("Catalyst: Processing initial batch...")

        while generatedCount < maxTokens {
            guard !shouldInterrupt else {
                let chunk = StreamChunk.completion(reason: .userCancelled)
                continuation.yield(chunk)
                return generatedCount
            }

            // Get logits and sample next token
            guard let logits = backend.getLogits() else {
                print("Catalyst: ERROR - Failed to get logits at generation step \(generatedCount)")
                throw CatalystError.generationFailed(details: "Failed to get logits")
            }

            let recentTokens = Array(contextTokens.suffix(Int(config.repetitionPenaltyRange)))
            let sampledToken = try sampler.sampleToken(
                logits: logits,
                config: config,
                recentTokens: recentTokens
            )

            // Check for end of generation
            if backend.isEndOfGeneration(token: sampledToken) {
                print("Catalyst: End of generation token detected")
                let chunk = StreamChunk.completion(reason: .natural)
                continuation.yield(chunk)
                return generatedCount
            }

            // Convert token to text
            let tokenText = backend.detokenize(token: sampledToken)

            // Record token for performance monitoring
            PerformanceMonitor.recordToken()

            // Process through stop sequence handler
            let chunks = processor.processToken(tokenText)

            for chunk in chunks {
                continuation.yield(chunk)

                if chunk.isComplete {
                    print("Catalyst: Generation complete")
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

            // Decode next token
            try backend.decodeToken(
                sampledToken,
                position: Int32(contextTokens.count - 1)
            )

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

    private func findCommonPrefixLength(_ newTokens: [Int32]) -> Int {
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
        backend?.clearKVCache()
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
