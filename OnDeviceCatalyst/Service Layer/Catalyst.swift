//
//  Catalyst.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/29/25.
//

//
//  Catalyst.swift
//  OnDeviceCatalyst
//
//  Main service interface for the Catalyst AI wrapper
//

import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Result type for safe model loading operations
public enum LoadResult {
    case success(LlamaInstance)
    case failure(CatalystError)
    
    /// Returns true if the load was successful
    public var isSuccess: Bool {
        switch self {
        case .success: return true
        case .failure: return false
        }
    }
    
    /// Returns the error if load failed, nil if successful
    public var error: CatalystError? {
        switch self {
        case .success: return nil
        case .failure(let error): return error
        }
    }
    
    /// Returns the instance if load was successful, nil if failed
    public var instance: LlamaInstance? {
        switch self {
        case .success(let instance): return instance
        case .failure: return nil
        }
    }
}

/// Main service class for the Catalyst AI wrapper
public class Catalyst {
    public static let shared = Catalyst()
    
    private var activeInstances: [String: LlamaInstance] = [:]
    private var instanceRefCounts: [String: Int] = [:]
    private let modelCache = ModelCache.shared
    private let queue = DispatchQueue(label: "com.catalyst.service", attributes: .concurrent)
    
    private init() {
        print("Catalyst: Service initialized")
        setupNotificationObservers()
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    // MARK: - Configuration
    
    /// Configure caching behavior
    public func configureCaching(with settings: CacheSettings) {
        modelCache.configure(with: settings)
    }
    
    // MARK: - Instance Management
    
    /// Get or create a model instance with caching support
    public func instance(
        for profile: ModelProfile,
        settings: InstanceSettings = .balanced,
        predictionConfig: PredictionConfig = .balanced,
        formatter: StandardPromptFormatter? = nil,
        customStopSequences: [String] = []
    ) async -> (instance: LlamaInstance, loadStream: AsyncStream<LoadProgress>) {
        
        let instanceId = profile.id
        
        // Check active instances first
        if let existingInstance = getActiveInstance(instanceId) {
            incrementRefCount(instanceId)
            let stream = AsyncStream<LoadProgress> { continuation in
                continuation.yield(.ready("Retrieved from active instances"))
                continuation.finish()
            }
            return (existingInstance, stream)
        }
        
        // Check cache
        modelCache.recordRequest(hit: false)
        if let cachedInstance = modelCache.getInstance(for: profile, with: settings) {
            modelCache.recordRequest(hit: true)
            storeActiveInstance(cachedInstance, for: instanceId)
            
            let stream = AsyncStream<LoadProgress> { continuation in
                continuation.yield(.ready("Retrieved from cache"))
                continuation.finish()
            }
            return (cachedInstance, stream)
        }
        
        // Create new instance
        let newInstance = LlamaInstance(
            profile: profile,
            settings: settings.optimizedFor(profile.architecture),
            predictionConfig: predictionConfig.optimizedFor(profile.architecture),
            formatter: formatter,
            customStopSequences: customStopSequences
        )
        
        let loadStream = newInstance.initialize()
        storeActiveInstance(newInstance, for: instanceId)
        
        return (newInstance, loadStream)
    }
    
    /// Safe model loading with comprehensive error handling
    public static func loadModelSafely(
        profile: ModelProfile,
        settings: InstanceSettings = .balanced,
        predictionConfig: PredictionConfig = .balanced,
        formatter: StandardPromptFormatter? = nil,
        customStopSequences: [String] = []
    ) async -> LoadResult {
        
        do {
            // Pre-validate the model
            try profile.validateModel()
            
            let (instance, stream) = await Catalyst.shared.instance(
                for: profile,
                settings: settings,
                predictionConfig: predictionConfig,
                formatter: formatter,
                customStopSequences: customStopSequences
            )
            
            // Collect the load results
            var finalProgress: LoadProgress?
            for await progress in stream {
                finalProgress = progress
                if progress.isComplete {
                    break
                }
            }
            
            if let progress = finalProgress {
                switch progress {
                case .ready:
                    return .success(instance)
                case .failed(let message):
                    return .failure(.modelLoadingFailed(details: message))
                default:
                    return .failure(.unknown(details: "Loading incomplete"))
                }
            } else {
                return .failure(.unknown(details: "No load progress received"))
            }
            
        } catch let error as CatalystError {
            return .failure(error)
        } catch {
            return .failure(.unknown(details: error.localizedDescription))
        }
    }
    
    // MARK: - High-Level Generation Methods
    
    /// Generate response for a conversation with a model
    public func generate(
        conversation: [Turn],
        systemPrompt: String,
        using profile: ModelProfile,
        settings: InstanceSettings = .balanced,
        predictionConfig: PredictionConfig = .balanced,
        customStopSequences: [String] = []
    ) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        
        print("Catalyst.generate: Starting with \(conversation.count) turns")
        print("Catalyst.generate: System prompt: '\(systemPrompt)'")
        print("Catalyst.generate: Profile: \(profile.name)")
        
        let (instance, loadStream) = await self.instance(
            for: profile,
            settings: settings,
            predictionConfig: predictionConfig,
            customStopSequences: customStopSequences
        )
        
        print("Catalyst.generate: Got instance, ready status: \(instance.isReady)")
        
        // Wait for model to be ready
        if !instance.isReady {
            print("Catalyst.generate: Waiting for model to load...")
            for await progress in loadStream {
                print("Catalyst: Loading \(profile.id) - \(progress.message)")
                if progress.isComplete {
                    if case .failed(let message) = progress {
                        print("Catalyst.generate: Model loading failed: \(message)")
                        await releaseInstance(for: profile.id, forceShutdown: true)
                        throw CatalystError.modelLoadingFailed(details: message)
                    }
                    print("Catalyst.generate: Model loading completed successfully")
                    break
                }
            }
        } else {
            print("Catalyst.generate: Model already ready, proceeding to generation")
        }
        
        print("Catalyst.generate: Calling instance.generate...")
        let stream = instance.generate(
            conversation: conversation,
            systemPrompt: systemPrompt,
            overrideConfig: predictionConfig
        )
        print("Catalyst.generate: Stream created, returning to caller")
        
        return stream
    }
    
    /// Simple text completion method
    public func complete(
        prompt: String,
        systemPrompt: String = "You are a helpful AI assistant.",
        using profile: ModelProfile,
        settings: InstanceSettings = .balanced,
        predictionConfig: PredictionConfig = .balanced
    ) async throws -> String {
        
        let conversation = [Turn.user(prompt)]
        let stream = try await generate(
            conversation: conversation,
            systemPrompt: systemPrompt,
            using: profile,
            settings: settings,
            predictionConfig: predictionConfig
        )
        
        var response = ""
        for try await chunk in stream {
            response += chunk.content
            if chunk.isComplete {
                break
            }
        }
        
        return response
    }
    
    /// Generate with conversation management
    public func generateWithConversation(
        _ conversation: inout Conversation,
        userMessage: String,
        using profile: ModelProfile,
        settings: InstanceSettings = .balanced,
        predictionConfig: PredictionConfig = .balanced
    ) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        
        // Add user turn
        let userTurn = Turn.user(userMessage)
        conversation.addTurn(userTurn)
        
        // Capture conversation state
        let conversationTurns = conversation.turns
        let systemPrompt = conversation.systemPrompt ?? "You are a helpful AI assistant."
        
        // Generate response
        let stream = try await generate(
            conversation: conversationTurns,
            systemPrompt: systemPrompt,
            using: profile,
            settings: settings,
            predictionConfig: predictionConfig
        )
        
        // Return a stream that collects response for later conversation update
        return AsyncThrowingStream { continuation in
            Task {
                var assistantResponse = ""
                var completionMetadata: ResponseMetadata?
                
                do {
                    for try await chunk in stream {
                        assistantResponse += chunk.content
                        
                        if chunk.isComplete {
                            completionMetadata = chunk.metadata?.responseMetadata
                        }
                        
                        continuation.yield(chunk)
                        
                        if chunk.isComplete {
                            break
                        }
                    }
                    
                    // Note: Conversation will need to be updated manually after this stream completes
                    // since we can't capture the inout parameter
                    
                    continuation.finish()
                    
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Helper method to add assistant response to conversation after generation
    public func addAssistantResponse(
        _ response: String,
        metadata: ResponseMetadata?,
        to conversation: inout Conversation,
        modelName: String
    ) {
        let assistantMetadata = TurnMetadata(
            tokensGenerated: metadata?.tokensGenerated,
            generationTimeMs: metadata?.generationTimeMs,
            tokensPerSecond: metadata?.tokensPerSecond,
            completionReason: metadata?.completionReason,
            modelUsed: modelName
        )
        
        let assistantTurn = Turn.assistant(response, metadata: assistantMetadata)
        conversation.addTurn(assistantTurn)
    }
    
    // MARK: - Instance Lifecycle
    
    /// Release an instance (decrements reference count)
    public func releaseInstance(for profileId: String, forceShutdown: Bool = false) async {
        await queue.sync(flags: .barrier) {
            guard let instance = activeInstances[profileId] else { return }
            
            let currentRefCount = instanceRefCounts[profileId, default: 1]
            instanceRefCounts[profileId] = currentRefCount - 1
            
            if forceShutdown || instanceRefCounts[profileId, default: 0] <= 0 {
                // Cache the instance before shutting down
                if !forceShutdown && instance.isReady {
                    let profile = instance.profile
                    let settings = instance.settings
                    modelCache.storeInstance(instance, for: profile, with: settings)
                }
                
                Task {
                    await instance.shutdown()
                }
                
                activeInstances.removeValue(forKey: profileId)
                instanceRefCounts.removeValue(forKey: profileId)
                
                print("Catalyst: Released instance for \(profileId)")
            } else {
                print("Catalyst: Decreased ref count for \(profileId) to \(instanceRefCounts[profileId]!)")
            }
        }
    }
    
    /// Shutdown all active instances
    public func shutdownAll() async {
        let instancesToShutdown = queue.sync { Array(activeInstances.values) }
        
        for instance in instancesToShutdown {
            await instance.shutdown()
        }
        
        queue.sync(flags: .barrier) {
            activeInstances.removeAll()
            instanceRefCounts.removeAll()
        }
        
        modelCache.clearAll()
        print("Catalyst: All instances shut down")
    }
    
    /// Interrupt generation for a specific model
    public func interruptGeneration(for profileId: String) {
        queue.sync {
            activeInstances[profileId]?.interrupt()
        }
    }
    
    /// Interrupt all active generations
    public func interruptAllGenerations() {
        queue.sync {
            for instance in activeInstances.values {
                instance.interrupt()
            }
        }
    }
    
    // MARK: - Cache Management
    
    /// Clear model cache
    public func clearCache() {
        modelCache.clearAll()
    }
    
    /// Get cache statistics
    public var cacheStatistics: CacheStatistics {
        return modelCache.statistics
    }
    
    // MARK: - Private Helpers
    
    private func getActiveInstance(_ instanceId: String) -> LlamaInstance? {
        return queue.sync {
            return activeInstances[instanceId]
        }
    }
    
    private func storeActiveInstance(_ instance: LlamaInstance, for instanceId: String) {
        queue.sync(flags: .barrier) {
            activeInstances[instanceId] = instance
            incrementRefCount(instanceId)
        }
    }
    
    private func incrementRefCount(_ instanceId: String) {
        instanceRefCounts[instanceId, default: 0] += 1
    }
    
    // MARK: - System Integration
    
    private func setupNotificationObservers() {
        #if os(iOS)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task {
                await self?.handleAppBackground()
            }
        }

        NotificationCenter.default.addObserver(
            forName: UIApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task {
                await self?.handleAppTermination()
            }
        }
        #endif
    }
    
    private func handleAppBackground() async {
        // Cache active instances when app goes to background
        let activeInstancesList = queue.sync { Array(activeInstances) }
        
        for (_, instance) in activeInstancesList {
            let profile = instance.profile
            let settings = instance.settings
            modelCache.storeInstance(instance, for: profile, with: settings)
        }
        
        print("Catalyst: App backgrounded, cached active instances")
    }
    
    private func handleAppTermination() async {
        await shutdownAll()
        print("Catalyst: App terminating, cleaned up all instances")
    }
}

// MARK: - Statistics

extension Catalyst {
    /// Get service statistics
    public var statistics: ServiceStatistics {
        return queue.sync {
            ServiceStatistics(
                activeInstanceCount: activeInstances.count,
                totalRefCount: instanceRefCounts.values.reduce(0, +),
                cacheStats: modelCache.statistics
            )
        }
    }
}

/// Overall service statistics
public struct ServiceStatistics {
    public let activeInstanceCount: Int
    public let totalRefCount: Int
    public let cacheStats: CacheStatistics
    
    public var summary: String {
        return "Service: \(activeInstanceCount) active instances, \(totalRefCount) total refs. \(cacheStats.summary)"
    }
}
