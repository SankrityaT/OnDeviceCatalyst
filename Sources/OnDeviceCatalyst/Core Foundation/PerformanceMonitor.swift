//
//  PerformanceMonitor.swift
//  OnDeviceCatalyst
//
//  Performance monitoring and metrics collection
//

import Foundation
import os.log

public class PerformanceMonitor {
    
    private static let logger = Logger(subsystem: "OnDeviceCatalyst", category: "Performance")
    
    // MARK: - Metrics Storage
    
    public struct InferenceMetrics {
        public let tokensPerSecond: Double
        public let totalTokens: Int
        public let totalTime: TimeInterval
        public let memoryUsage: UInt64
        public let gpuLayers: Int32
        public let batchSize: UInt32
        public let contextLength: UInt32
        public let timestamp: Date
        
        public var description: String {
            """
            === Inference Metrics ===
            Tokens/sec: \(String(format: "%.2f", tokensPerSecond))
            Total tokens: \(totalTokens)
            Total time: \(String(format: "%.2f", totalTime))s
            Memory usage: \(ByteCountFormatter.string(fromByteCount: Int64(memoryUsage), countStyle: .memory))
            GPU layers: \(gpuLayers)
            Batch size: \(batchSize)
            Context length: \(contextLength)
            Timestamp: \(timestamp)
            """
        }
    }
    
    public struct ModelLoadMetrics {
        public let loadTime: TimeInterval
        public let modelSize: UInt64
        public let memoryAfterLoad: UInt64
        public let settings: InstanceSettings
        public let timestamp: Date
        
        public var description: String {
            """
            === Model Load Metrics ===
            Load time: \(String(format: "%.2f", loadTime))s
            Model size: \(ByteCountFormatter.string(fromByteCount: Int64(modelSize), countStyle: .file))
            Memory after load: \(ByteCountFormatter.string(fromByteCount: Int64(memoryAfterLoad), countStyle: .memory))
            GPU layers: \(settings.gpuLayers)
            Context length: \(settings.contextLength)
            Batch size: \(settings.batchSize)
            Timestamp: \(timestamp)
            """
        }
    }
    
    // MARK: - Monitoring
    
    private static var inferenceStartTime: Date?
    private static var tokenCount = 0
    
    /// Start monitoring inference performance
    public static func startInferenceMonitoring() {
        inferenceStartTime = Date()
        tokenCount = 0
        logger.info("Started inference monitoring")
    }
    
    /// Record a generated token
    public static func recordToken() {
        tokenCount += 1
    }
    
    /// End monitoring and return metrics
    public static func endInferenceMonitoring(settings: InstanceSettings) -> InferenceMetrics? {
        guard let startTime = inferenceStartTime else { return nil }
        
        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = totalTime > 0 ? Double(tokenCount) / totalTime : 0
        let memoryUsage = getCurrentMemoryUsage()
        
        let metrics = InferenceMetrics(
            tokensPerSecond: tokensPerSecond,
            totalTokens: tokenCount,
            totalTime: totalTime,
            memoryUsage: memoryUsage,
            gpuLayers: settings.gpuLayers,
            batchSize: settings.batchSize,
            contextLength: settings.contextLength,
            timestamp: endTime
        )
        
        logger.info("\(metrics.description)")
        print(metrics.description)
        
        // Reset for next measurement
        inferenceStartTime = nil
        tokenCount = 0
        
        return metrics
    }
    
    /// Monitor model loading performance
    public static func monitorModelLoad<T>(
        modelSize: UInt64,
        settings: InstanceSettings,
        operation: () throws -> T
    ) throws -> (result: T, metrics: ModelLoadMetrics) {
        
        let startTime = Date()
        let result = try operation()
        let endTime = Date()
        
        let loadTime = endTime.timeIntervalSince(startTime)
        let memoryAfterLoad = getCurrentMemoryUsage()
        
        let metrics = ModelLoadMetrics(
            loadTime: loadTime,
            modelSize: modelSize,
            memoryAfterLoad: memoryAfterLoad,
            settings: settings,
            timestamp: endTime
        )
        
        logger.info("\(metrics.description)")
        print(metrics.description)
        
        return (result, metrics)
    }
    
    // MARK: - Memory Monitoring
    
    /// Get current memory usage in bytes
    public static func getCurrentMemoryUsage() -> UInt64 {
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
    
    /// Log current system state
    public static func logSystemState() {
        let memoryUsage = getCurrentMemoryUsage()
        let thermalState = ProcessInfo.processInfo.thermalState
        let processorCount = ProcessInfo.processInfo.processorCount
        
        let systemInfo = """
        === System State ===
        Memory usage: \(ByteCountFormatter.string(fromByteCount: Int64(memoryUsage), countStyle: .memory))
        Thermal state: \(thermalState.rawValue)
        Processor count: \(processorCount)
        """
        
        logger.info("\(systemInfo)")
        print(systemInfo)
    }
    
    // MARK: - Benchmarking
    
    /// Run a simple benchmark
    public static func benchmark(settings: InstanceSettings, iterations: Int = 5) {
        print("=== Performance Benchmark ===")
        print("Settings: GPU layers: \(settings.gpuLayers), Batch: \(settings.batchSize), Context: \(settings.contextLength)")
        
        logSystemState()
        
        // This would need to be integrated with actual inference calls
        print("Run benchmark with \(iterations) iterations using these settings")
    }
}
