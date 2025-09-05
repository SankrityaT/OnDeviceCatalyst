//
//  CompletionReason.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//


import Foundation

/// Indicates why text generation stopped and provides context for next actions
public enum CompletionReason: Equatable, Codable, Hashable {
    case natural                        // Model hit EOS token naturally
    case maxTokensReached              // Hit the configured max token limit
    case contextWindowFull             // Ran out of context space
    case userCancelled                 // User interrupted generation
    case stopSequenceFound(String)    // Hit a custom stop sequence
    case error(String)                 // Generation failed with error
    case thermalThrottling             // Device thermal protection activated
    case memoryPressure                // System low memory condition
    
    /// Indicates if generation could theoretically continue
    public var canContinue: Bool {
        switch self {
        case .natural, .userCancelled, .error:
            return false
        case .maxTokensReached, .contextWindowFull, .stopSequenceFound, .thermalThrottling, .memoryPressure:
            return true
        }
    }
    
    /// Indicates if the completion reason suggests user action might be needed
    public var requiresUserAction: Bool {
        switch self {
        case .contextWindowFull, .thermalThrottling, .memoryPressure:
            return true
        case .natural, .maxTokensReached, .userCancelled, .stopSequenceFound, .error:
            return false
        }
    }
    
    /// Human-readable description of why generation stopped
    public var description: String {
        switch self {
        case .natural:
            return "Generation completed naturally"
        case .maxTokensReached:
            return "Reached maximum token limit"
        case .contextWindowFull:
            return "Context window is full"
        case .userCancelled:
            return "Cancelled by user"
        case .stopSequenceFound(let sequence):
            return "Found stop sequence: '\(sequence)'"
        case .error(let message):
            return "Generation error: \(message)"
        case .thermalThrottling:
            return "Paused due to device thermal protection"
        case .memoryPressure:
            return "Paused due to low memory"
        }
    }
    
    /// Suggested action the user or app could take
    public var suggestedAction: String? {
        switch self {
        case .maxTokensReached:
            return "Continue generation or increase max token limit"
        case .contextWindowFull:
            return "Clear conversation history or use a model with larger context"
        case .stopSequenceFound:
            return "Remove stop sequence to continue generation"
        case .thermalThrottling:
            return "Wait for device to cool down before continuing"
        case .memoryPressure:
            return "Close other apps or reduce context length"
        case .error(let message) where message.contains("memory"):
            return "Reduce batch size or context length"
        case .natural, .userCancelled, .error:
            return nil
        }
    }
    
    /// Priority level for handling this completion reason
    public var priority: CompletionPriority {
        switch self {
        case .error:
            return .high
        case .thermalThrottling, .memoryPressure:
            return .medium
        case .contextWindowFull, .maxTokensReached:
            return .low
        case .natural, .userCancelled, .stopSequenceFound:
            return .none
        }
    }
}

/// Priority levels for handling completion reasons
public enum CompletionPriority: Int, Codable {
    case none = 0      // No action needed
    case low = 1       // User might want to take action
    case medium = 2    // Action recommended
    case high = 3      // Action required
}

/// Complete response metadata including generation statistics
public struct ResponseMetadata: Codable {
    public let completionReason: CompletionReason
    public let tokensGenerated: Int
    public let tokensPerSecond: Double?
    public let generationTimeMs: Int64
    public let promptTokens: Int
    public let totalTokens: Int
    public let stopSequence: String?
    
    public init(
        completionReason: CompletionReason,
        tokensGenerated: Int,
        tokensPerSecond: Double? = nil,
        generationTimeMs: Int64,
        promptTokens: Int,
        totalTokens: Int,
        stopSequence: String? = nil
    ) {
        self.completionReason = completionReason
        self.tokensGenerated = tokensGenerated
        self.tokensPerSecond = tokensPerSecond
        self.generationTimeMs = generationTimeMs
        self.promptTokens = promptTokens
        self.totalTokens = totalTokens
        self.stopSequence = stopSequence
    }
    
    /// Calculates tokens per second if not provided
    public var effectiveTokensPerSecond: Double {
        if let tps = tokensPerSecond {
            return tps
        }
        
        let timeInSeconds = Double(generationTimeMs) / 1000.0
        return timeInSeconds > 0 ? Double(tokensGenerated) / timeInSeconds : 0.0
    }
    
    /// Returns a performance assessment
    public var performanceLevel: PerformanceLevel {
        let tps = effectiveTokensPerSecond
        
        if tps >= 20.0 {
            return .excellent
        } else if tps >= 10.0 {
            return .good
        } else if tps >= 5.0 {
            return .acceptable
        } else {
            return .slow
        }
    }
    
    /// Returns usage statistics summary
    public var usageSummary: String {
        let tps = String(format: "%.1f", effectiveTokensPerSecond)
        return "Generated \(tokensGenerated) tokens in \(generationTimeMs)ms (\(tps) tok/s)"
    }
}

/// Performance assessment levels
public enum PerformanceLevel: String, Codable, CaseIterable {
    case excellent = "Excellent"
    case good = "Good"
    case acceptable = "Acceptable"
    case slow = "Slow"
    
    public var color: String {
        switch self {
        case .excellent: return "green"
        case .good: return "blue"
        case .acceptable: return "orange"
        case .slow: return "red"
        }
    }
}

/// Streaming response that includes both content and metadata
public struct StreamResponse: Codable {
    public let content: String
    public let isComplete: Bool
    public let metadata: ResponseMetadata?
    
    public init(content: String, isComplete: Bool = false, metadata: ResponseMetadata? = nil) {
        self.content = content
        self.isComplete = isComplete
        self.metadata = metadata
    }
    
    /// Creates a completion response with metadata
    public static func completion(
        reason: CompletionReason,
        tokensGenerated: Int,
        generationTimeMs: Int64,
        promptTokens: Int
    ) -> StreamResponse {
        let metadata = ResponseMetadata(
            completionReason: reason,
            tokensGenerated: tokensGenerated,
            generationTimeMs: generationTimeMs,
            promptTokens: promptTokens,
            totalTokens: promptTokens + tokensGenerated
        )
        
        return StreamResponse(
            content: "",
            isComplete: true,
            metadata: metadata
        )
    }
}
