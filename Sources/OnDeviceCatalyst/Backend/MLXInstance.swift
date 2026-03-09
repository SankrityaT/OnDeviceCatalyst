//
//  MLXInstance.swift
//  OnDeviceCatalyst
//
//  MLX-based model instance for hybrid architectures (Qwen 3.5, etc.)
//  that lack full Metal kernel support in llama.cpp.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Model instance powered by Apple's MLX framework.
///
/// Drop-in alternative to `LlamaInstance` for models whose architectures
/// are fully supported by MLX (e.g. Qwen 3.5 with Gated Delta Net layers).
public class MLXInstance {

    // MARK: - Properties

    public let profile: ModelProfile
    public private(set) var settings: InstanceSettings
    public private(set) var predictionConfig: PredictionConfig

    private var container: ModelContainer?
    private var shouldInterrupt = false
    private var isGenerating = false
    private var loadingContinuation: AsyncStream<LoadProgress>.Continuation?

    /// The HuggingFace model ID used by MLX (e.g. "mlx-community/Qwen3.5-4B-4bit")
    public let mlxModelId: String

    // MARK: - Initialization

    public init(
        profile: ModelProfile,
        settings: InstanceSettings,
        predictionConfig: PredictionConfig,
        mlxModelId: String
    ) {
        self.profile = profile
        self.settings = settings
        self.predictionConfig = predictionConfig
        self.mlxModelId = mlxModelId
    }

    deinit {
        cleanup()
    }

    // MARK: - Lifecycle

    public var isReady: Bool {
        container != nil
    }

    /// Load the MLX model and return a progress stream.
    public func initialize() -> AsyncStream<LoadProgress> {
        AsyncStream { continuation in
            loadingContinuation = continuation
            Task {
                await performInitialization()
            }
        }
    }

    private func performInitialization() async {
        guard !isReady else {
            publishProgress(.ready("MLX model already initialized"))
            return
        }

        do {
            publishProgress(.preparing("Configuring MLX model"))

            let configuration = ModelConfiguration(
                id: mlxModelId,
                defaultPrompt: "Hello"
            )

            publishProgress(.loading("Downloading / loading MLX model weights"))

            // Set GPU cache limit based on device
            #if os(iOS)
            MLX.GPU.set(cacheLimit: 512 * 1024 * 1024) // 512 MB on iOS
            #else
            MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024) // 1 GB on macOS
            #endif

            let loadedContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            ) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                Task { @MainActor in
                    // Progress updates during download
                    print("Catalyst MLX: Loading \(pct)%")
                }
            }

            self.container = loadedContainer
            publishProgress(.ready("MLX model ready for inference"))

        } catch {
            publishProgress(.failed("MLX initialization failed: \(error.localizedDescription)"))
        }
    }

    // MARK: - Generation

    /// Stream a response for the given conversation, matching LlamaInstance's interface.
    public func generate(
        conversation: [Turn],
        systemPrompt: String? = nil,
        overrideConfig: PredictionConfig? = nil
    ) -> AsyncThrowingStream<StreamChunk, Error> {

        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await performGeneration(
                        conversation: conversation,
                        systemPrompt: systemPrompt,
                        config: overrideConfig ?? predictionConfig,
                        continuation: continuation
                    )
                } catch {
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

        guard let container else {
            throw CatalystError.engineNotInitialized
        }
        guard !isGenerating else {
            throw CatalystError.generationFailed(details: "Generation already in progress")
        }

        isGenerating = true
        shouldInterrupt = false
        defer { isGenerating = false }

        let startTime = Date()

        // Build messages array for MLX chat template
        var messages: [[String: String]] = []
        if let sys = systemPrompt {
            messages.append(["role": "system", "content": sys])
        }
        for turn in conversation {
            messages.append(["role": turn.role.rawValue, "content": turn.content])
        }

        // Prepare input using chat template
        let userInput = UserInput(messages: messages)
        let lmInput = try await container.prepare(input: userInput)

        // Map PredictionConfig → GenerateParameters
        let params = GenerateParameters(
            maxTokens: config.maxTokens > 0 ? config.maxTokens : 1024,
            temperature: config.temperature,
            topP: config.topP,
            repetitionPenalty: config.repetitionPenalty > 1.0 ? config.repetitionPenalty : nil,
            repetitionContextSize: Int(config.repetitionPenaltyRange)
        )

        // Stream tokens
        let stream = try await container.generate(input: lmInput, parameters: params)
        var tokensGenerated = 0
        var promptTokens = 0

        for await generation in stream {
            if shouldInterrupt {
                let chunk = StreamChunk.completion(reason: .userCancelled)
                continuation.yield(chunk)
                continuation.finish()
                return
            }

            switch generation {
            case .chunk(let text):
                tokensGenerated += 1
                continuation.yield(.content(text))

            case .info(let info):
                promptTokens = info.promptTokenCount
                let duration = Date().timeIntervalSince(startTime)
                let metadata = ResponseMetadata(
                    completionReason: .natural,
                    tokensGenerated: info.generationTokenCount,
                    tokensPerSecond: info.tokensPerSecond,
                    generationTimeMs: Int64(duration * 1000),
                    promptTokens: info.promptTokenCount,
                    totalTokens: info.promptTokenCount + info.generationTokenCount
                )
                continuation.yield(.completion(reason: .natural, metadata: metadata))

            case .toolCall:
                // Tool calls from MLX are handled at a higher level
                break
            }
        }

        continuation.finish()
    }

    // MARK: - Control

    public func interrupt() {
        shouldInterrupt = true
    }

    public func updateConfig(_ newConfig: PredictionConfig) {
        predictionConfig = newConfig
    }

    public func clearContext() {
        // MLX manages its own KV cache internally
        print("Catalyst MLX: Context cleared")
    }

    public func shutdown() async {
        shouldInterrupt = true
        cleanup()
    }

    private func cleanup() {
        container = nil
        isGenerating = false
        shouldInterrupt = false
        loadingContinuation?.finish()
        loadingContinuation = nil
    }

    // MARK: - Helpers

    private func publishProgress(_ progress: LoadProgress) {
        loadingContinuation?.yield(progress)
        if progress.isComplete {
            loadingContinuation?.finish()
            loadingContinuation = nil
        }
    }
}
