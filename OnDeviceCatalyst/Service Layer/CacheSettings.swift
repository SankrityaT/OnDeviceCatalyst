//
//  CacheSettings.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/29/25.
//

import Foundation
import SwiftUI

/// Settings for controlling model caching behavior
public struct CacheSettings: Codable {
    public var maxCachedInstances: Int
    public var enablePersistentCache: Bool
    public var cacheDirectory: String?
    public var memoryThresholdMB: Int
    
    public init(
        maxCachedInstances: Int = 2,
        enablePersistentCache: Bool = false, // Disabled by default for simplicity
        cacheDirectory: String? = nil,
        memoryThresholdMB: Int = 1024
    ) {
        self.maxCachedInstances = maxCachedInstances
        self.enablePersistentCache = enablePersistentCache
        self.cacheDirectory = cacheDirectory
        self.memoryThresholdMB = memoryThresholdMB
    }
    
    public static var `default`: CacheSettings {
        CacheSettings()
    }
}

/// Metadata about a cached model instance
public struct CachedInstanceMetadata: Codable {
    public let profileId: String
    public let modelPath: String
    public let architecture: ModelArchitecture
    public let settings: InstanceSettings
    public let lastAccessed: Date
    public let memoryUsageMB: Int
    
    public init(from profile: ModelProfile, settings: InstanceSettings, memoryUsageMB: Int) {
        self.profileId = profile.id
        self.modelPath = profile.filePath
        self.architecture = profile.architecture
        self.settings = settings
        self.lastAccessed = Date()
        self.memoryUsageMB = memoryUsageMB
    }
}

/// Manages caching of model instances for performance optimization
public class ModelCache {
    public static let shared = ModelCache()
    
    private var cacheSettings: CacheSettings
    private var cachedInstances: [String: LlamaInstance] = [:]
    private var instanceMetadata: [String: CachedInstanceMetadata] = [:]
    private var accessTimes: [String: Date] = [:]
    
    private let queue = DispatchQueue(label: "com.catalyst.modelcache", attributes: .concurrent)
    
    private init() {
        self.cacheSettings = .default
        setupMemoryWarningObserver()
    }
    
    // MARK: - Configuration
    
    /// Configure caching behavior
    public func configure(with settings: CacheSettings) {
        queue.async(flags: .barrier) {
            self.cacheSettings = settings
            
            // Trim cache if new limit is lower
            self.trimCacheToLimit()
        }
    }
    
    // MARK: - Cache Operations
    
    /// Check if an instance is cached
    public func hasInstance(for profile: ModelProfile, with settings: InstanceSettings) -> Bool {
        let cacheKey = generateCacheKey(profile: profile, settings: settings)
        
        return queue.sync {
            return cachedInstances[cacheKey] != nil
        }
    }
    
    /// Get cached instance if available
    public func getInstance(for profile: ModelProfile, with settings: InstanceSettings) -> LlamaInstance? {
        let cacheKey = generateCacheKey(profile: profile, settings: settings)
        
        return queue.sync {
            guard let instance = cachedInstances[cacheKey] else {
                return nil
            }
            
            // Update access time
            accessTimes[cacheKey] = Date()
            
            print("Catalyst: Retrieved cached instance for \(profile.id)")
            return instance
        }
    }
    
    /// Store instance in cache
    public func storeInstance(
        _ instance: LlamaInstance,
        for profile: ModelProfile,
        with settings: InstanceSettings
    ) {
        let cacheKey = generateCacheKey(profile: profile, settings: settings)
        let estimatedMemoryMB = Int(settings.estimatedMemoryUsage / (1024 * 1024))
        
        queue.async(flags: .barrier) {
            // Check memory constraints
            if self.getTotalCachedMemoryMB() + estimatedMemoryMB > self.cacheSettings.memoryThresholdMB {
                self.evictLeastRecentlyUsed()
            }
            
            // Store instance
            self.cachedInstances[cacheKey] = instance
            self.instanceMetadata[cacheKey] = CachedInstanceMetadata(
                from: profile,
                settings: settings,
                memoryUsageMB: estimatedMemoryMB
            )
            self.accessTimes[cacheKey] = Date()
            
            // Trim cache if over limit
            self.trimCacheToLimit()
            
            print("Catalyst: Cached instance for \(profile.id) (~\(estimatedMemoryMB)MB)")
        }
    }
    
    /// Remove specific instance from cache
    public func removeInstance(for profile: ModelProfile, with settings: InstanceSettings) {
        let cacheKey = generateCacheKey(profile: profile, settings: settings)
        
        queue.async(flags: .barrier) {
            if let instance = self.cachedInstances.removeValue(forKey: cacheKey) {
                Task {
                    await instance.shutdown()
                }
                
                self.instanceMetadata.removeValue(forKey: cacheKey)
                self.accessTimes.removeValue(forKey: cacheKey)
                
                print("Catalyst: Removed cached instance for \(profile.id)")
            }
        }
    }
    
    /// Clear all cached instances
    public func clearAll() {
        queue.async(flags: .barrier) {
            for instance in self.cachedInstances.values {
                Task {
                    await instance.shutdown()
                }
            }
            
            self.cachedInstances.removeAll()
            self.instanceMetadata.removeAll()
            self.accessTimes.removeAll()
            
            print("Catalyst: Cleared all cached instances")
        }
    }
    
    // MARK: - Cache Management
    
    private func generateCacheKey(profile: ModelProfile, settings: InstanceSettings) -> String {
        let settingsHash = "\(settings.contextLength)_\(settings.gpuLayers)_\(settings.useFlashAttention)"
        return "\(profile.id)_\(settingsHash)"
    }
    
    private func trimCacheToLimit() {
        while cachedInstances.count > cacheSettings.maxCachedInstances {
            evictLeastRecentlyUsed()
        }
    }
    
    private func evictLeastRecentlyUsed() {
        guard let oldestKey = accessTimes.min(by: { $0.value < $1.value })?.key else {
            return
        }
        
        if let instance = cachedInstances.removeValue(forKey: oldestKey) {
            Task {
                await instance.shutdown()
            }
            
            instanceMetadata.removeValue(forKey: oldestKey)
            accessTimes.removeValue(forKey: oldestKey)
            
            print("Catalyst: Evicted LRU cached instance: \(oldestKey)")
        }
    }
    
    private func getTotalCachedMemoryMB() -> Int {
        return instanceMetadata.values.reduce(0) { $0 + $1.memoryUsageMB }
    }
    
    // MARK: - Memory Management
    
    private func setupMemoryWarningObserver() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            print("Catalyst: Received memory warning, clearing cache")
            self?.handleMemoryWarning()
        }
    }
    
    private func handleMemoryWarning() {
        queue.async(flags: .barrier) {
            // Evict half the cache on memory warning
            let targetCount = max(0, self.cachedInstances.count / 2)
            
            while self.cachedInstances.count > targetCount {
                self.evictLeastRecentlyUsed()
            }
        }
    }
    
    // MARK: - Statistics
    
    /// Get cache statistics
    public var statistics: CacheStatistics {
        return queue.sync {
            let totalMemoryMB = getTotalCachedMemoryMB()
            let instanceCount = cachedInstances.count
            
            return CacheStatistics(
                cachedInstanceCount: instanceCount,
                totalMemoryUsageMB: totalMemoryMB,
                maxInstances: cacheSettings.maxCachedInstances,
                memoryThresholdMB: cacheSettings.memoryThresholdMB,
                hitRate: calculateHitRate()
            )
        }
    }
    
    private var totalRequests: Int = 0
    private var cacheHits: Int = 0
    
    private func calculateHitRate() -> Double {
        return totalRequests > 0 ? Double(cacheHits) / Double(totalRequests) : 0.0
    }
    
    // For tracking hit rate
    internal func recordRequest(hit: Bool) {
        queue.async(flags: .barrier) {
            self.totalRequests += 1
            if hit {
                self.cacheHits += 1
            }
        }
    }
}

/// Cache performance statistics
public struct CacheStatistics {
    public let cachedInstanceCount: Int
    public let totalMemoryUsageMB: Int
    public let maxInstances: Int
    public let memoryThresholdMB: Int
    public let hitRate: Double
    
    public var utilizationRate: Double {
        return maxInstances > 0 ? Double(cachedInstanceCount) / Double(maxInstances) : 0.0
    }
    
    public var memoryUtilizationRate: Double {
        return memoryThresholdMB > 0 ? Double(totalMemoryUsageMB) / Double(memoryThresholdMB) : 0.0
    }
    
    public var summary: String {
        return "Cache: \(cachedInstanceCount)/\(maxInstances) instances, \(totalMemoryUsageMB)MB/\(memoryThresholdMB)MB, \(String(format: "%.1f", hitRate * 100))% hit rate"
    }
}
