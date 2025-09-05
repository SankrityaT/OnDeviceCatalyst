//
//  PredictionConfig.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//

import Foundation

/// Configuration for text generation behavior and sampling methods
public struct PredictionConfig: Codable, Hashable {
    
    // MARK: - Core Sampling Parameters
    public var temperature: Float
    public var topK: Int32
    public var topP: Float
    public var minP: Float
    public var typicalP: Float
    
    // MARK: - Repetition Control
    public var repetitionPenalty: Float
    public var repetitionPenaltyRange: Int32
    public var frequencyPenalty: Float
    public var presencePenalty: Float
    
    // MARK: - Advanced Sampling
    public var mirostatMode: Int32       // 0=disabled, 1=v1, 2=v2
    public var mirostatTau: Float        // Target entropy
    public var mirostatEta: Float        // Learning rate
    
    // MARK: - Output Control
    public var maxTokens: Int            // -1 for unlimited
    public var stopSequences: [String]
    
    /// Initialize with comprehensive sampling parameters
    public init(
        temperature: Float = 0.7,
        topK: Int32 = 40,
        topP: Float = 0.9,
        minP: Float = 0.05,
        typicalP: Float = 1.0,
        repetitionPenalty: Float = 1.1,
        repetitionPenaltyRange: Int32 = 64,
        frequencyPenalty: Float = 0.0,
        presencePenalty: Float = 0.0,
        mirostatMode: Int32 = 0,
        mirostatTau: Float = 5.0,
        mirostatEta: Float = 0.1,
        maxTokens: Int = -1,
        stopSequences: [String] = []
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.typicalP = typicalP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionPenaltyRange = repetitionPenaltyRange
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.mirostatMode = mirostatMode
        self.mirostatTau = mirostatTau
        self.mirostatEta = mirostatEta
        self.maxTokens = maxTokens
        self.stopSequences = stopSequences
    }
    
    // MARK: - Preset Configurations
    
    /// Balanced settings suitable for general conversation
    public static var balanced: PredictionConfig {
        PredictionConfig()
    }
    
    /// Creative settings for storytelling and creative writing
    public static var creative: PredictionConfig {
        PredictionConfig(
            temperature: 0.9,
            topK: 0,           // Disable top-K for more variety
            topP: 0.95,
            minP: 0.02,        // Lower threshold for creativity
            repetitionPenalty: 1.05 // Lighter penalty for creative flow
        )
    }
    
    /// Speed-optimized settings for fast responses
    public static var speed: PredictionConfig {
        PredictionConfig(
            temperature: 0.3,
            topK: 5,           // Very limited for speed
            topP: 0.6,
            minP: 0.2,         // Higher threshold for faster decisions
            repetitionPenalty: 1.05, // Lower penalty for speed
            repetitionPenaltyRange: 32, // Shorter range for speed
            maxTokens: 150     // Limit response length
        )
    }
    
    /// Deterministic settings for consistent outputs
    public static var deterministic: PredictionConfig {
        PredictionConfig(
            temperature: 0.0,  // Greedy sampling
            topK: 1,
            topP: 1.0,
            repetitionPenalty: 1.0
        )
    }
    
    /// Mirostat v2 settings for dynamic sampling
    public static var mirostat: PredictionConfig {
        PredictionConfig(
            temperature: 1.0,
            topK: 0,
            topP: 1.0,
            mirostatMode: 2,
            mirostatTau: 5.0,
            mirostatEta: 0.1
        )
    }
    
    // MARK: - Validation
    
    /// Validates that all parameters are within acceptable ranges
    public func validate() throws {
        // Temperature validation
        guard temperature >= 0.0 && temperature <= 2.0 else {
            throw CatalystError.configurationInvalid(
                parameter: "temperature",
                reason: "Temperature must be between 0.0 and 2.0"
            )
        }
        
        // Top-K validation
        guard topK >= 0 else {
            throw CatalystError.configurationInvalid(
                parameter: "topK",
                reason: "Top-K must be non-negative (0 disables top-K)"
            )
        }
        
        // Top-P validation
        guard topP >= 0.0 && topP <= 1.0 else {
            throw CatalystError.configurationInvalid(
                parameter: "topP",
                reason: "Top-P must be between 0.0 and 1.0"
            )
        }
        
        // Min-P validation
        guard minP >= 0.0 && minP <= 1.0 else {
            throw CatalystError.configurationInvalid(
                parameter: "minP",
                reason: "Min-P must be between 0.0 and 1.0"
            )
        }
        
        // Typical-P validation
        guard typicalP >= 0.0 && typicalP <= 1.0 else {
            throw CatalystError.configurationInvalid(
                parameter: "typicalP",
                reason: "Typical-P must be between 0.0 and 1.0"
            )
        }
        
        // Repetition penalty validation
        guard repetitionPenalty >= 0.0 && repetitionPenalty <= 2.0 else {
            throw CatalystError.configurationInvalid(
                parameter: "repetitionPenalty",
                reason: "Repetition penalty must be between 0.0 and 2.0"
            )
        }
        
        // Penalty range validation
        guard repetitionPenaltyRange >= 0 else {
            throw CatalystError.configurationInvalid(
                parameter: "repetitionPenaltyRange",
                reason: "Repetition penalty range must be non-negative"
            )
        }
        
        // Frequency and presence penalty validation
        guard frequencyPenalty >= -2.0 && frequencyPenalty <= 2.0 else {
            throw CatalystError.configurationInvalid(
                parameter: "frequencyPenalty",
                reason: "Frequency penalty must be between -2.0 and 2.0"
            )
        }
        
        guard presencePenalty >= -2.0 && presencePenalty <= 2.0 else {
            throw CatalystError.configurationInvalid(
                parameter: "presencePenalty",
                reason: "Presence penalty must be between -2.0 and 2.0"
            )
        }
        
        // Mirostat validation
        guard mirostatMode >= 0 && mirostatMode <= 2 else {
            throw CatalystError.configurationInvalid(
                parameter: "mirostatMode",
                reason: "Mirostat mode must be 0 (disabled), 1 (v1), or 2 (v2)"
            )
        }
        
        if mirostatMode > 0 {
            guard mirostatTau > 0.0 else {
                throw CatalystError.configurationInvalid(
                    parameter: "mirostatTau",
                    reason: "Mirostat tau must be positive when Mirostat is enabled"
                )
            }
            
            guard mirostatEta > 0.0 && mirostatEta <= 1.0 else {
                throw CatalystError.configurationInvalid(
                    parameter: "mirostatEta",
                    reason: "Mirostat eta must be between 0.0 and 1.0"
                )
            }
        }
        
        // Max tokens validation
        guard maxTokens == -1 || maxTokens > 0 else {
            throw CatalystError.configurationInvalid(
                parameter: "maxTokens",
                reason: "Max tokens must be positive or -1 for unlimited"
            )
        }
    }
    
    /// Returns a copy optimized for the given model architecture
    public func optimizedFor(_ architecture: ModelArchitecture) -> PredictionConfig {
        var optimized = self
        
        switch architecture {
        case .codeLlama, .codeQwen, .deepSeekCoder:
            // Code models benefit from more deterministic generation
            optimized.temperature = min(temperature, 0.3)
            optimized.topK = topK > 0 ? min(topK, 20) : 20
            optimized.repetitionPenalty = max(repetitionPenalty, 1.1)
            
        case .llama3, .llama31:
            // Llama 3 models handle higher creativity well
            if temperature < 0.1 {
                optimized.temperature = 0.7
            }
            
        case .qwen2, .qwen25:
            // Qwen models work well with typical-P sampling
            optimized.typicalP = min(typicalP, 0.95)
            
        case .unknown:
            // Conservative settings for unknown architectures
            optimized.temperature = min(temperature, 0.8)
            optimized.topP = min(topP, 0.9)
            
        default:
            break
        }
        
        return optimized
    }
    
    /// Calculates effective max tokens given context constraints
    public func effectiveMaxTokens(contextLength: UInt32, promptTokens: Int) -> Int {
        let availableTokens = Int(contextLength) - promptTokens
        
        if maxTokens == -1 {
            return max(0, availableTokens)
        } else {
            return min(maxTokens, max(0, availableTokens))
        }
    }
    
    /// Returns a description of the sampling strategy being used
    public var samplingStrategy: String {
        if temperature == 0.0 {
            return "Greedy (deterministic)"
        } else if mirostatMode > 0 {
            return "Mirostat v\(mirostatMode)"
        } else if topK > 0 && topP < 1.0 {
            return "Top-K (\(topK)) + Top-P (\(topP))"
        } else if topK > 0 {
            return "Top-K (\(topK))"
        } else if topP < 1.0 {
            return "Top-P (\(topP))"
        } else {
            return "Temperature (\(temperature))"
        }
    }
}
