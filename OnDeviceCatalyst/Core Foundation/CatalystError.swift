//
//  CatalystError.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//

import Foundation

public enum CatalystError: Error, LocalizedError, Equatable {
    case modelFileNotFound(path: String)
    case modelFileCorrupted(path: String, reason: String)
    case modelLoadingFailed(details: String)
    case contextCreationFailed(details: String)
    case batchProcessingFailed(details: String)
    case tokenizationFailed(text: String, reason: String)
    case generationFailed(details: String)
    case engineNotInitialized
    case operationCancelled
    case resourceExhausted(resource: String)
    case configurationInvalid(parameter: String, reason: String)
    case architectureUnsupported(architecture: String, suggestion: String)
    case memoryInsufficient(required: String, available: String)
    case contextWindowExceeded(tokenCount: Int, limit: Int)
    case samplingFailed(method: String, reason: String)
    case cacheOperationFailed(operation: String, reason: String)
    case networkResourceUnavailable(details: String)
    case unknown(details: String)

    public var errorDescription: String? {
        switch self {
        case .modelFileNotFound(let path):
            return "Model file not found at path: \(path). Verify the file exists and is accessible."
            
        case .modelFileCorrupted(let path, let reason):
            return "Model file at \(path) appears corrupted: \(reason). Try re-downloading the model."
            
        case .modelLoadingFailed(let details):
            return "Failed to load model: \(details)"
            
        case .contextCreationFailed(let details):
            return "Failed to create inference context: \(details)"
            
        case .batchProcessingFailed(let details):
            return "Batch processing failed: \(details). This may indicate memory issues or model corruption."
            
        case .tokenizationFailed(let text, let reason):
            return "Failed to tokenize text (length: \(text.count)): \(reason)"
            
        case .generationFailed(let details):
            return "Text generation failed: \(details)"
            
        case .engineNotInitialized:
            return "Catalyst engine is not initialized. Ensure the model is loaded before generating."
            
        case .operationCancelled:
            return "Operation was cancelled by user request."
            
        case .resourceExhausted(let resource):
            return "Resource exhausted: \(resource). Try reducing batch size or context length."
            
        case .configurationInvalid(let parameter, let reason):
            return "Invalid configuration for \(parameter): \(reason)"
            
        case .architectureUnsupported(let architecture, let suggestion):
            return "Unsupported model architecture '\(architecture)'. \(suggestion)"
            
        case .memoryInsufficient(let required, let available):
            return "Insufficient memory: requires \(required), available \(available)."
            
        case .contextWindowExceeded(let tokenCount, let limit):
            return "Input exceeds context window: \(tokenCount) tokens > \(limit) limit."
            
        case .samplingFailed(let method, let reason):
            return "Sampling method '\(method)' failed: \(reason)"
            
        case .cacheOperationFailed(let operation, let reason):
            return "Cache \(operation) failed: \(reason)"
            
        case .networkResourceUnavailable(let details):
            return "Network resource unavailable: \(details)"
            
        case .unknown(let details):
            return "Unknown error occurred: \(details)"
        }
    }
    
    /// Indicates if this error might be resolved by retrying with different parameters
    public var isRecoverable: Bool {
        switch self {
        case .architectureUnsupported, .contextCreationFailed, .memoryInsufficient, .contextWindowExceeded:
            return true
        case .modelFileNotFound, .modelFileCorrupted, .configurationInvalid:
            return false
        case .resourceExhausted, .samplingFailed, .batchProcessingFailed:
            return true
        case .engineNotInitialized, .tokenizationFailed:
            return false
        default:
            return false
        }
    }
    
    /// Provides actionable recovery suggestions for the error
    public var recoverySuggestion: String? {
        switch self {
        case .architectureUnsupported(_, let suggestion):
            return suggestion
            
        case .contextCreationFailed:
            return "Try reducing context length, disabling GPU acceleration, or using fewer GPU layers."
            
        case .memoryInsufficient:
            return "Reduce context length, batch size, or close other memory-intensive apps."
            
        case .contextWindowExceeded(let tokenCount, let limit):
            return "Reduce input length by \(tokenCount - limit) tokens or increase context window size."
            
        case .resourceExhausted(let resource):
            if resource.lowercased().contains("memory") {
                return "Reduce batch size, context length, or GPU layers allocated."
            } else if resource.lowercased().contains("context") {
                return "Shorten conversation history or increase context window."
            }
            return "Reduce resource usage in model configuration."
            
        case .samplingFailed(let method, _):
            return "Try using a different sampling method or check sampling parameters for \(method)."
            
        case .batchProcessingFailed:
            return "Reduce batch size or check model compatibility."
            
        case .modelLoadingFailed:
            return "Verify model file integrity and llama.cpp version compatibility."
            
        case .cacheOperationFailed:
            return "Clear cache and retry, or disable caching temporarily."
            
        default:
            return nil
        }
    }
    
    /// Returns the category of error for logging and analytics
    public var category: ErrorCategory {
        switch self {
        case .modelFileNotFound, .modelFileCorrupted:
            return .modelFile
        case .modelLoadingFailed, .architectureUnsupported:
            return .modelLoading
        case .contextCreationFailed, .contextWindowExceeded:
            return .context
        case .tokenizationFailed, .samplingFailed:
            return .processing
        case .memoryInsufficient, .resourceExhausted:
            return .resources
        case .configurationInvalid:
            return .configuration
        case .cacheOperationFailed:
            return .cache
        case .networkResourceUnavailable:
            return .network
        default:
            return .runtime
        }
    }
}

public enum ErrorCategory: String, CaseIterable {
    case modelFile = "model_file"
    case modelLoading = "model_loading"
    case context = "context"
    case processing = "processing"
    case resources = "resources"
    case configuration = "configuration"
    case cache = "cache"
    case network = "network"
    case runtime = "runtime"
}

extension CatalystError {
    /// Creates an error with enhanced context for debugging
    public static func withContext(_ baseError: CatalystError, context: String) -> CatalystError {
        switch baseError {
        case .modelLoadingFailed(let details):
            return .modelLoadingFailed(details: "\(context): \(details)")
        case .generationFailed(let details):
            return .generationFailed(details: "\(context): \(details)")
        case .unknown(let details):
            return .unknown(details: "\(context): \(details)")
        default:
            return baseError
        }
    }
}
