//
//  ConfigurationPresets.swift
//  OnDeviceCatalyst
//
//  Flexible configuration presets for different use cases
//

import Foundation

public struct ConfigurationPresets {
    
    // MARK: - Coaching App Configurations
    
    /// Optimized for coaching apps with complex prompts and long conversations
    public static func coachingApp(
        maxContextTokens: Int = 8192,
        expectedPromptSize: Int = 1000
    ) -> (InstanceSettings, PredictionConfig) {
        
        let contextLength = SafetyManager.safeContextLength(
            for: expectedPromptSize,
            maxDesired: UInt32(maxContextTokens)
        )
        
        let settings = InstanceSettings(
            contextLength: contextLength,
            batchSize: SafetyManager.optimalBatchSize(contextLength: contextLength),
            gpuLayers: SafetyManager.isMemoryUsageSafe() ? 24 : 16,
            cpuThreads: 4,
            enableMemoryMapping: true,
            enableMemoryLocking: false, // Disabled to prevent crashes
            useFlashAttention: true
        )
        
        let predictionConfig = PredictionConfig(
            temperature: 0.7,
            topK: 40,
            topP: 0.9,
            repetitionPenalty: 1.1,
            maxTokens: 512 // Reasonable limit for coaching responses
        )
        
        return (SafetyManager.adjustSettingsForSafety(settings), predictionConfig)
    }
    
    // MARK: - Production App Configurations
    
    /// Ultra-safe settings for production apps
    public static func production(
        prioritizeStability: Bool = true
    ) -> (InstanceSettings, PredictionConfig) {
        
        let settings = InstanceSettings(
            contextLength: prioritizeStability ? 2048 : 4096,
            batchSize: prioritizeStability ? 128 : 256,
            gpuLayers: prioritizeStability ? 8 : 16,
            cpuThreads: 3,
            enableMemoryMapping: true,
            enableMemoryLocking: false,
            useFlashAttention: false
        )
        
        let predictionConfig = PredictionConfig(
            temperature: 0.6,
            topK: 30,
            topP: 0.85,
            repetitionPenalty: 1.15,
            maxTokens: 256
        )
        
        return (SafetyManager.adjustSettingsForSafety(settings), predictionConfig)
    }
    
    // MARK: - Development Configurations
    
    /// Safe settings for development and testing
    public static func development() -> (InstanceSettings, PredictionConfig) {
        
        let settings = InstanceSettings(
            contextLength: 1024,
            batchSize: 64,
            gpuLayers: 4,
            cpuThreads: 2,
            enableMemoryMapping: true,
            enableMemoryLocking: false,
            useFlashAttention: false
        )
        
        let predictionConfig = PredictionConfig(
            temperature: 0.8,
            topK: 50,
            topP: 0.95,
            repetitionPenalty: 1.05,
            maxTokens: 128
        )
        
        return (SafetyManager.adjustSettingsForSafety(settings), predictionConfig)
    }
    
    // MARK: - Custom Configuration Builder
    
    /// Build custom configuration with automatic safety adjustments
    public static func custom(
        contextLength: UInt32,
        prioritizeSpeed: Bool = false,
        prioritizeStability: Bool = false,
        maxResponseTokens: Int = 512
    ) -> (InstanceSettings, PredictionConfig) {
        
        var settings = InstanceSettings(
            contextLength: contextLength,
            batchSize: SafetyManager.optimalBatchSize(contextLength: contextLength),
            gpuLayers: prioritizeSpeed ? 32 : (prioritizeStability ? 8 : 16),
            cpuThreads: prioritizeSpeed ? 6 : (prioritizeStability ? 2 : 4),
            enableMemoryMapping: true,
            enableMemoryLocking: false,
            useFlashAttention: prioritizeSpeed && !prioritizeStability
        )
        
        let predictionConfig = PredictionConfig(
            temperature: prioritizeStability ? 0.5 : 0.7,
            topK: prioritizeStability ? 20 : 40,
            topP: prioritizeStability ? 0.8 : 0.9,
            repetitionPenalty: prioritizeStability ? 1.2 : 1.1,
            maxTokens: maxResponseTokens
        )
        
        return (SafetyManager.adjustSettingsForSafety(settings), predictionConfig)
    }
}

// MARK: - Usage Examples and Documentation

/*
 Usage Examples for your coaching app package:
 
 1. Basic coaching app:
 let (settings, config) = ConfigurationPresets.coachingApp()
 
 2. Coaching app with large context:
 let (settings, config) = ConfigurationPresets.coachingApp(
     maxContextTokens: 12000,
     expectedPromptSize: 2000
 )
 
 3. Production deployment:
 let (settings, config) = ConfigurationPresets.production(prioritizeStability: true)
 
 4. Custom configuration:
 let (settings, config) = ConfigurationPresets.custom(
     contextLength: 6144,
     prioritizeSpeed: false,
     prioritizeStability: true,
     maxResponseTokens: 1024
 )
 
 All configurations automatically adjust for:
 - Device capabilities
 - Current memory usage
 - Thermal conditions
 - Safety limits
 */
