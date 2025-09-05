//
//  SimulatorSupport.swift
//  OnDeviceCatalyst
//
//  Simulator-specific configurations and workarounds
//

import Foundation

public struct SimulatorSupport {
    
    /// Get simulator-safe settings
    public static func simulatorSafeSettings() -> InstanceSettings {
        return InstanceSettings(
            contextLength: 1024,      // Small context for simulator
            batchSize: 64,            // Small batch size
            gpuLayers: 0,             // CPU-only - no Metal on simulator
            cpuThreads: 2,            // Limited threads
            enableMemoryMapping: true,
            enableMemoryLocking: false, // Never enable on simulator
            useFlashAttention: false   // Disable advanced features
        )
    }
    
    /// Check if running on simulator
    public static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }
    
    /// Get appropriate settings based on environment
    public static func environmentAwareSettings(_ baseSettings: InstanceSettings) -> InstanceSettings {
        if isSimulator {
            return simulatorSafeSettings()
        }
        return baseSettings
    }
}

// MARK: - Configuration Extensions

extension InstanceSettings {
    
    /// Simulator-safe preset
    public static var simulator: InstanceSettings {
        return SimulatorSupport.simulatorSafeSettings()
    }
}

extension ConfigurationPresets {
    
    /// Get environment-aware configuration
    public static func environmentAware(
        basePreset: () -> (InstanceSettings, PredictionConfig) = { coachingApp() }
    ) -> (InstanceSettings, PredictionConfig) {
        
        let (settings, config) = basePreset()
        
        if SimulatorSupport.isSimulator {
            // Use simulator-safe settings but keep the prediction config
            return (SimulatorSupport.simulatorSafeSettings(), config)
        }
        
        return (settings, config)
    }
}
