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

            // Aggressively limit GPU memory on iOS to avoid OOM kills
            #if os(iOS)
            MLX.GPU.set(cacheLimit: 256 * 1024 * 1024)  // 256 MB cache
            MLX.GPU.set(memoryLimit: 3 * 1024 * 1024 * 1024, relaxed: true) // 3 GB soft limit
            #else
            MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024)
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
        overrideConfig: PredictionConfig? = nil,
        tools: [CatalystTool]? = nil
    ) -> AsyncThrowingStream<StreamChunk, Error> {

        AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.isGenerating = false
            }
            Task {
                do {
                    try await performGeneration(
                        conversation: conversation,
                        systemPrompt: systemPrompt,
                        config: overrideConfig ?? predictionConfig,
                        tools: tools,
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
        tools: [CatalystTool]?,
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

        // Convert CatalystTool → ToolSpec (OpenAI function calling format)
        let toolSpecs: [[String: any Sendable]]? = tools?.map { tool in
            var properties: [String: Any] = [:]
            var required: [String] = []
            for param in tool.parameters {
                var prop: [String: Any] = [
                    "type": param.type,
                    "description": param.description
                ]
                if let enumValues = param.enumValues {
                    prop["enum"] = enumValues
                }
                properties[param.name] = prop
                if param.required {
                    required.append(param.name)
                }
            }
            return [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": [
                        "type": "object",
                        "properties": properties,
                        "required": required
                    ] as [String: Any]
                ] as [String: Any]
            ] as [String: any Sendable]
        }

        // Prepare input using chat template
        let userInput = UserInput(messages: messages, tools: toolSpecs)
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

            case .toolCall(let call):
                let args = call.function.arguments.mapValues { jsonValue in
                    AnyCodable(Self.jsonValueToAny(jsonValue))
                }
                let catalystCall = CatalystToolCall(
                    name: call.function.name,
                    arguments: args
                )
                continuation.yield(.completionWithToolCalls(
                    reason: .natural,
                    toolCalls: [catalystCall]
                ))
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

    private static func jsonValueToAny(_ value: JSONValue) -> Any {
        switch value {
        case .null: return NSNull()
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let arr): return arr.map { jsonValueToAny($0) }
        case .object(let obj): return obj.mapValues { jsonValueToAny($0) }
        }
    }

    private func publishProgress(_ progress: LoadProgress) {
        loadingContinuation?.yield(progress)
        if progress.isComplete {
            loadingContinuation?.finish()
            loadingContinuation = nil
        }
    }
}
