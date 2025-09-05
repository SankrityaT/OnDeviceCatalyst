//
//  DeviceOptimizer.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//


import Foundation
import UIKit

public enum DeviceOptimizer {
    
    // MARK: - GPU Optimization
    
    /// Returns the recommended number of GPU layers based on device capabilities
    public static func recommendedGpuLayers() -> Int32 {
        #if targetEnvironment(simulator)
        return 0  // No GPU acceleration on simulator - Metal backend causes vector errors
        #else
        
        let device = currentDevice()
        
        switch device.tier {
        case .high:
            return 99  // Let llama.cpp auto-determine optimal layers
        case .medium:
            return 20  // Conservative GPU usage for mid-tier devices
        case .low:
            return 0   // CPU-only for older devices
        }
        
        #endif
    }
    
    // MARK: - CPU Optimization
    
    /// Returns the recommended number of CPU threads
    public static func recommendedCpuThreads() -> Int32 {
        let totalCores = ProcessInfo.processInfo.processorCount
        
        // Reserve cores for system and UI
        let reservedCores = max(1, totalCores / 4)
        let availableCores = max(1, totalCores - reservedCores)
        
        #if os(iOS)
        // iOS: Be more conservative to avoid thermal throttling
        return Int32(min(availableCores, 4))
        #else
        // macOS: Can use more cores
        return Int32(min(availableCores, 8))
        #endif
    }
    
    // MARK: - Memory Optimization
    
    /// Returns recommended context length based on available memory
    public static func recommendedContextLength() -> UInt32 {
        let device = currentDevice()
        let availableMemory = getAvailableMemory()
        
        // Estimate memory per token (rough approximation)
        let memoryPerToken: UInt64 = 8 // bytes per token in KV cache
        
        // Reserve 1GB for system and model
        let reservedMemory: UInt64 = 1024 * 1024 * 1024
        let usableMemory = availableMemory > reservedMemory ? availableMemory - reservedMemory : availableMemory / 2
        
        // Calculate max context based on memory
        let maxContextFromMemory = min(UInt32(usableMemory / memoryPerToken), 32768)
        
        // Apply device-specific limits
        switch device.tier {
        case .high:
            return min(maxContextFromMemory, 16384)
        case .medium:
            return min(maxContextFromMemory, 8192)
        case .low:
            return min(maxContextFromMemory, 2048)
        }
    }
    
    /// Returns recommended batch size based on device performance
    public static func recommendedBatchSize() -> UInt32 {
        let device = currentDevice()
        
        switch device.tier {
        case .high:
            return 512
        case .medium:
            return 256
        case .low:
            return 128
        }
    }
    
    // MARK: - Device Detection
    
    /// Gets current device information
    private static func currentDevice() -> DeviceInfo {
        #if os(iOS)
        
        let device = UIDevice.current
        let model = device.model
        let systemVersion = device.systemVersion
        
        // Get device identifier
        var systemInfo = utsname()
        uname(&systemInfo)
        let deviceIdentifier = String(bytes: Data(bytes: &systemInfo.machine, count: Int(_SYS_NAMELEN)), encoding: .ascii)?.trimmingCharacters(in: .controlCharacters) ?? "Unknown"
        
        let tier = classifyDeviceTier(identifier: deviceIdentifier)
        
        return DeviceInfo(
            identifier: deviceIdentifier,
            model: model,
            systemVersion: systemVersion,
            tier: tier
        )
        
        #else
        
        // macOS - generally high performance
        return DeviceInfo(
            identifier: "Mac",
            model: "macOS",
            systemVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            tier: .high
        )
        
        #endif
    }
    
    /// Classifies device performance tier based on identifier
    private static func classifyDeviceTier(identifier: String) -> DeviceTier {
        let id = identifier.lowercased()
        
        // High-performance devices (iPhone 13+, iPad Pro M1+)
        if id.contains("iphone16") || id.contains("iphone15") || id.contains("iphone14") || id.contains("iphone13") ||
           id.contains("ipad13") || id.contains("ipad14") || // iPad Pro M1/M2
           id.contains("mac") {
            return .high
        }
        
        // Medium-performance devices (iPhone 11-12, iPad Air 4+)
        if id.contains("iphone12") || id.contains("iphone11") ||
           id.contains("ipad11") || id.contains("ipad12") || // iPad Air 4/5
           id.contains("ipad8") || id.contains("ipad9") {    // iPad 8th/9th gen
            return .medium
        }
        
        // Low-performance devices (older iPhones, base iPads)
        return .low
    }
    
    // MARK: - Memory Detection
    
    /// Gets available system memory in bytes (safer implementation)
    private static func getAvailableMemory() -> UInt64 {
        // Use ProcessInfo instead of direct mach calls to prevent crashes
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        
        // Conservative estimate: assume 70% is available
        return UInt64(Double(physicalMemory) * 0.7)
    }
    
    // MARK: - Thermal Management
    
    /// Checks if device is under thermal pressure
    public static func isUnderThermalPressure() -> Bool {
        #if os(iOS)
        return ProcessInfo.processInfo.thermalState == .serious || ProcessInfo.processInfo.thermalState == .critical
        #else
        return false // macOS doesn't expose thermal state as easily
        #endif
    }
    
    /// Returns settings adjusted for thermal conditions
    public static func thermalAdjustedSettings(_ settings: InstanceSettings) -> InstanceSettings {
        guard isUnderThermalPressure() else { return settings }
        
        var adjusted = settings
        
        // Reduce GPU usage under thermal pressure
        adjusted.gpuLayers = min(adjusted.gpuLayers, Int32(adjusted.gpuLayers / 2))
        
        // Reduce thread count
        adjusted.cpuThreads = max(1, Int32(adjusted.cpuThreads / 2))
        
        // Reduce batch size for less intensive processing
        adjusted.batchSize = max(32, adjusted.batchSize / 2)
        
        return adjusted
    }
}

// MARK: - Supporting Types

private struct DeviceInfo {
    let identifier: String
    let model: String
    let systemVersion: String
    let tier: DeviceTier
}

private enum DeviceTier {
    case high
    case medium
    case low
}
