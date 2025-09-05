//
//  ModelProfile.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//
//
//  ModelProfile.swift
//  OnDeviceCatalyst
//
//  Model configuration and validation for Catalyst AI wrapper
//

import Foundation

/// Represents a complete model configuration with validation
public struct ModelProfile: Hashable, Codable, Identifiable {
    public let id: String
    public let name: String
    public let filePath: String
    public let architecture: ModelArchitecture
    public let fileSize: UInt64?
    public let checksum: String?
    public let createdAt: Date
    
    /// Initialize a model profile with automatic architecture detection
    public init(
        filePath: String,
        id: String? = nil,
        name: String? = nil,
        architecture: ModelArchitecture? = nil,
        checksum: String? = nil
    ) throws {
        self.filePath = filePath
        self.architecture = architecture ?? ModelArchitecture.detectFromPath(filePath)
        self.name = name ?? Self.extractModelName(from: filePath)
        self.id = id ?? Self.generateID(from: filePath, architecture: self.architecture)
        self.checksum = checksum
        self.createdAt = Date()
        
        // Get file size during initialization
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: filePath)
            self.fileSize = attributes[.size] as? UInt64
        } catch {
            self.fileSize = nil
            throw CatalystError.modelFileNotFound(path: filePath)
        }
        
        // Validate the model file
        try self.validateModel()
    }
    
    /// Create a profile with explicit architecture (for fallback scenarios)
    public static func withFallback(
        filePath: String,
        primaryArchitecture: ModelArchitecture,
        name: String? = nil
    ) throws -> ModelProfile {
        do {
            return try ModelProfile(filePath: filePath, name: name, architecture: primaryArchitecture)
        } catch let error as CatalystError {
            // If primary architecture fails, try fallback
            let fallbackArch = primaryArchitecture.getFallbackArchitecture()
            print("Catalyst: Primary architecture \(primaryArchitecture.rawValue) failed, trying fallback \(fallbackArch.rawValue)")
            
            do {
                return try ModelProfile(filePath: filePath, name: name, architecture: fallbackArch)
            } catch {
                if let catalystError = error as? CatalystError {
                    throw CatalystError.withContext(catalystError, context: "Both primary and fallback architecture initialization failed")
                } else {
                    throw CatalystError.unknown(details: "Both primary and fallback architecture initialization failed: \(error.localizedDescription)")
                }
            }
        }
    }
    
    /// Validates the model file exists and is a valid GGUF format
    public func validateModel() throws {
        // Check file existence
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw CatalystError.modelFileNotFound(path: filePath)
        }
        
        // Check minimum file size (valid models should be at least 1MB)
        if let size = fileSize, size < 1_048_576 {
            throw CatalystError.modelFileCorrupted(
                path: filePath,
                reason: "File size (\(size) bytes) is too small for a valid model"
            )
        }
        
        // Validate GGUF magic header
        try validateGGUFFormat()
        
        // Validate architecture compatibility
        try architecture.validateCompatibility()
    }
    
    /// Validates GGUF file format by checking magic bytes
    private func validateGGUFFormat() throws {
        guard let fileHandle = FileHandle(forReadingAtPath: filePath) else {
            throw CatalystError.modelFileNotFound(path: filePath)
        }
        
        defer { fileHandle.closeFile() }
        
        do {
            let magicBytes = fileHandle.readData(ofLength: 4)
            let expectedMagic = "GGUF".data(using: .ascii)!
            
            if magicBytes.count < 4 {
                throw CatalystError.modelFileCorrupted(
                    path: filePath,
                    reason: "File too short to contain GGUF header"
                )
            }
            
            if magicBytes != expectedMagic {
                let actualMagic = String(data: magicBytes, encoding: .ascii) ?? "unknown"
                throw CatalystError.modelFileCorrupted(
                    path: filePath,
                    reason: "Invalid GGUF magic bytes. Expected 'GGUF', found '\(actualMagic)'"
                )
            }
        } catch let error as CatalystError {
            throw error
        } catch {
            throw CatalystError.modelFileCorrupted(
                path: filePath,
                reason: "Could not read file header: \(error.localizedDescription)"
            )
        }
    }
    
    /// Generate a unique ID for the model profile
    private static func generateID(from path: String, architecture: ModelArchitecture) -> String {
        let filename = (path as NSString).lastPathComponent
        let pathHash = String(path.hashValue)
        return "\(architecture.rawValue)_\(filename)_\(pathHash)".replacingOccurrences(of: "/", with: "_")
    }
    
    /// Extract a human-readable name from the file path
    private static func extractModelName(from path: String) -> String {
        let filename = (path as NSString).lastPathComponent
        let nameWithoutExtension = (filename as NSString).deletingPathExtension
        
        // Clean up common model naming patterns
        return nameWithoutExtension
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: ".gguf", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .capitalized
    }
    
    /// Returns recommended settings for this specific model
    public func recommendedSettings() -> (contextLength: UInt32, batchSize: UInt32, gpuLayers: Int32) {
        return architecture.recommendedSettings()
    }
    
    /// Returns file size in a human-readable format
    public var fileSizeFormatted: String {
        guard let size = fileSize else { return "Unknown" }
        
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(size))
    }
    
    /// Returns a summary of the model's capabilities
    public var capabilities: ModelCapabilities {
        return ModelCapabilities(
            supportsSystemPrompts: architecture.supportsSystemPrompts,
            maxContextLength: architecture.defaultContextWindow,
            requiresSpecialTokens: architecture.requiresSpecialTokens,
            compatibilityLevel: getCompatibilityLevel()
        )
    }
    
    private func getCompatibilityLevel() -> CompatibilityLevel {
        switch architecture {
        case .llama2, .llama3, .mistral, .mistralInstruct:
            return .high
        case .llama31, .qwen2, .qwen25, .phi3:
            return .medium
        case .unknown, .commandR:
            return .low
        default:
            return .medium
        }
    }
    
    /// Generates a cache key for this model configuration
    public func cacheKey(contextLength: UInt32, gpuLayers: Int32, useFlashAttention: Bool) -> String {
        let settingsHash = "\(contextLength)_\(gpuLayers)_\(useFlashAttention)"
        return "\(id)_\(settingsHash)"
    }
}

/// Describes model capabilities for UI and decision making
public struct ModelCapabilities {
    public let supportsSystemPrompts: Bool
    public let maxContextLength: UInt32
    public let requiresSpecialTokens: Bool
    public let compatibilityLevel: CompatibilityLevel
}

public enum CompatibilityLevel: String, CaseIterable {
    case high = "High"
    case medium = "Medium"
    case low = "Low"
    
    public var description: String {
        switch self {
        case .high: return "Excellent compatibility with all features"
        case .medium: return "Good compatibility with most features"
        case .low: return "Basic compatibility, some features may not work"
        }
    }
}
