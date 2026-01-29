//
//  SafetyManager.swift
//  OnDeviceCatalyst
//
//  Safety mechanisms for preventing crashes and resource exhaustion
//

import Foundation
#if canImport(UIKit)
import UIKit
#endif

public class SafetyManager {
    
    // MARK: - Memory Safety
    
    /// Maximum safe memory usage percentage
    private static let maxMemoryUsagePercent: Double = 0.7
    
    /// Check if current memory usage is safe
    public static func isMemoryUsageSafe() -> Bool {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let usedMemory = getUsedMemory()
        let usagePercent = Double(usedMemory) / Double(totalMemory)
        
        return usagePercent < maxMemoryUsagePercent
    }
    
    /// Get current memory usage in bytes
    private static func getUsedMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? UInt64(info.resident_size) : 0
    }
    
    // MARK: - Resource Monitoring
    
    /// Monitor and adjust settings based on system state
    public static func adjustSettingsForSafety(_ settings: InstanceSettings) -> InstanceSettings {
        var safeSettings = settings
        
        // Check memory pressure
        if !isMemoryUsageSafe() {
            safeSettings.contextLength = min(safeSettings.contextLength, 2048)
            safeSettings.batchSize = min(safeSettings.batchSize, 256)
            safeSettings.gpuLayers = min(safeSettings.gpuLayers, 16)
        }
        
        // Check thermal state
        if DeviceOptimizer.isUnderThermalPressure() {
            safeSettings = DeviceOptimizer.thermalAdjustedSettings(safeSettings)
        }
        
        // Disable risky features on older devices
        if !isHighPerformanceDevice() {
            safeSettings.enableMemoryLocking = false
            safeSettings.useFlashAttention = false
            safeSettings.gpuLayers = min(safeSettings.gpuLayers, 8)
        }
        
        return safeSettings
    }
    
    /// Check if device can handle high-performance settings
    private static func isHighPerformanceDevice() -> Bool {
        #if targetEnvironment(simulator)
        return false // Simulator has limited capabilities
        #else
        var systemInfo = utsname()
        uname(&systemInfo)
        let deviceIdentifier = String(bytes: Data(bytes: &systemInfo.machine, count: Int(_SYS_NAMELEN)), encoding: .ascii)?.trimmingCharacters(in: .controlCharacters) ?? "Unknown"
        
        let id = deviceIdentifier.lowercased()
        return id.contains("iphone16") || id.contains("iphone15") || id.contains("iphone14")
        #endif
    }
    
    // MARK: - Context Management
    
    /// Calculate safe context length for given prompt size
    public static func safeContextLength(for promptTokens: Int, maxDesired: UInt32 = 8192) -> UInt32 {
        let baseContext = UInt32(promptTokens * 2) // Double the prompt size
        let deviceLimit = DeviceOptimizer.recommendedContextLength()
        let memoryLimit = isMemoryUsageSafe() ? maxDesired : maxDesired / 2
        
        return min(baseContext, min(deviceLimit, memoryLimit))
    }
    
    // MARK: - Batch Size Optimization
    
    /// Calculate optimal batch size for current conditions
    public static func optimalBatchSize(contextLength: UInt32) -> UInt32 {
        let baseBatch = DeviceOptimizer.recommendedBatchSize()
        
        // Reduce batch size for large contexts
        if contextLength > 4096 {
            return min(baseBatch, 256)
        } else if contextLength > 2048 {
            return min(baseBatch, 384)
        }
        
        return baseBatch
    }
}

// MARK: - Configuration Presets for Different Use Cases

extension InstanceSettings {
    
    /// Ultra-fast settings for coaching apps with large prompts
    public static var coaching: InstanceSettings {
        let baseSettings = InstanceSettings(
            contextLength: 2048,    // Smaller context for speed
            batchSize: 2048,        // Massive batch for maximum speed
            gpuLayers: 99,          // Max GPU acceleration
            cpuThreads: 8,          // Maximum threads
            enableMemoryMapping: true,
            enableMemoryLocking: false, // Keep disabled to prevent crashes
            useFlashAttention: true
        )
        
        return baseSettings
    }
    
    /// Ultra-safe settings for production apps
    public static var production: InstanceSettings {
        let baseSettings = InstanceSettings(
            contextLength: 4096,
            batchSize: 256,
            gpuLayers: 16,
            cpuThreads: 3,
            enableMemoryMapping: true,
            enableMemoryLocking: false,
            useFlashAttention: false
        )
        
        return SafetyManager.adjustSettingsForSafety(baseSettings)
    }
    
    /// Development settings with extra safety checks
    public static var development: InstanceSettings {
        let baseSettings = InstanceSettings(
            contextLength: 2048,
            batchSize: 128,
            gpuLayers: 8,
            cpuThreads: 2,
            enableMemoryMapping: true,
            enableMemoryLocking: false,
            useFlashAttention: false
        )
        
        return SafetyManager.adjustSettingsForSafety(baseSettings)
    }
}
