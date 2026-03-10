//
//  ModelDownloader.swift
//  OnDeviceCatalyst
//
//  Auto-downloads GGUF models from HuggingFace for frictionless setup
//

import Foundation

// MARK: - Model Presets

/// Pre-configured GGUF models available for auto-download from HuggingFace
public enum CatalystModel: String, CaseIterable, Codable {

    // MARK: - Embedding Models
    case nomicEmbedV1_5     = "nomic-embed-text-v1.5"
    case gteQwen2_1_5B      = "gte-Qwen2-1.5B-instruct"

    // MARK: - Completion Models (Small — phones)
    case qwen25_0_5B        = "Qwen2.5-0.5B-Instruct"
    case qwen25_1_5B        = "Qwen2.5-1.5B-Instruct"
    case phi4_mini           = "Phi-4-mini-instruct"

    // MARK: - Completion Models (Medium — tablets/Macs)
    case qwen25_3B          = "Qwen2.5-3B-Instruct"
    case llama32_3B         = "Llama-3.2-3B-Instruct"

    /// HuggingFace download URL for the GGUF file
    public var downloadURL: URL {
        switch self {
        case .nomicEmbedV1_5:
            return URL(string: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf")!
        case .gteQwen2_1_5B:
            return URL(string: "https://huggingface.co/mav23/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-Q8_0.gguf")!
        case .qwen25_0_5B:
            return URL(string: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf")!
        case .qwen25_1_5B:
            return URL(string: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf")!
        case .phi4_mini:
            return URL(string: "https://huggingface.co/bartowski/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf")!
        case .qwen25_3B:
            return URL(string: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf")!
        case .llama32_3B:
            return URL(string: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf")!
        }
    }

    /// Expected filename on disk
    public var filename: String {
        return downloadURL.lastPathComponent
    }

    /// Model architecture for OnDeviceCatalyst
    public var architecture: ModelArchitecture {
        switch self {
        case .nomicEmbedV1_5:       return .nomic
        case .gteQwen2_1_5B:        return .qwen25
        case .qwen25_0_5B:          return .qwen25
        case .qwen25_1_5B:          return .qwen25
        case .phi4_mini:            return .phi4
        case .qwen25_3B:            return .qwen25
        case .llama32_3B:           return .llama32
        }
    }

    /// Embedding dimensions (nil for completion models)
    public var embeddingDimensions: Int? {
        switch self {
        case .nomicEmbedV1_5:       return 768
        case .gteQwen2_1_5B:        return 1536
        default:                    return nil
        }
    }

    /// Whether this is an embedding model
    public var isEmbeddingModel: Bool {
        return embeddingDimensions != nil
    }

    /// Approximate download size in MB
    public var approximateSizeMB: Int {
        switch self {
        case .nomicEmbedV1_5:       return 550
        case .gteQwen2_1_5B:        return 1600
        case .qwen25_0_5B:          return 530
        case .qwen25_1_5B:          return 1600
        case .phi4_mini:            return 2400
        case .qwen25_3B:            return 2000
        case .llama32_3B:           return 2000
        }
    }

    /// Human-readable description
    public var displayName: String {
        switch self {
        case .nomicEmbedV1_5:       return "Nomic Embed v1.5 (768d, ~550MB)"
        case .gteQwen2_1_5B:        return "GTE-Qwen2 1.5B (1536d, ~1.6GB)"
        case .qwen25_0_5B:          return "Qwen 2.5 0.5B (small, ~530MB)"
        case .qwen25_1_5B:          return "Qwen 2.5 1.5B (balanced, ~1.6GB)"
        case .phi4_mini:            return "Phi-4 Mini (strong, ~2.4GB)"
        case .qwen25_3B:            return "Qwen 2.5 3B (powerful, ~2GB)"
        case .llama32_3B:           return "Llama 3.2 3B (powerful, ~2GB)"
        }
    }
}

// MARK: - Download Progress

/// Progress updates for model downloads
public enum DownloadProgress: Sendable {
    case starting(model: String)
    case downloading(model: String, progress: Double, downloadedMB: Int, totalMB: Int)
    case verifying(model: String)
    case completed(model: String, path: String)
    case failed(model: String, error: String)
    case alreadyCached(model: String, path: String)

    public var isComplete: Bool {
        switch self {
        case .completed, .failed, .alreadyCached: return true
        default: return false
        }
    }

    public var localPath: String? {
        switch self {
        case .completed(_, let path), .alreadyCached(_, let path): return path
        default: return nil
        }
    }
}

// MARK: - Model Downloader

/// Downloads and caches GGUF models from HuggingFace
/// Thread-safe, supports progress reporting, caches locally
public actor ModelDownloader {

    public static let shared = ModelDownloader()

    /// Directory where models are cached
    private let cacheDirectory: URL

    private init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        self.cacheDirectory = appSupport.appendingPathComponent("OnDeviceCatalyst/Models", isDirectory: true)
    }

    // MARK: - Public API

    /// Get a model's local path, downloading if needed. Returns an AsyncStream of progress updates.
    public func ensure(_ model: CatalystModel) -> AsyncStream<DownloadProgress> {
        return AsyncStream { continuation in
            Task {
                // Check cache first
                if let cachedPath = self.cachedPath(for: model) {
                    continuation.yield(.alreadyCached(model: model.displayName, path: cachedPath))
                    continuation.finish()
                    return
                }

                // Download
                continuation.yield(.starting(model: model.displayName))

                do {
                    let path = try await self.download(model: model) { progress, downloadedMB, totalMB in
                        continuation.yield(.downloading(
                            model: model.displayName,
                            progress: progress,
                            downloadedMB: downloadedMB,
                            totalMB: totalMB
                        ))
                    }

                    continuation.yield(.verifying(model: model.displayName))

                    // Validate GGUF header
                    try self.validateGGUF(at: path)

                    continuation.yield(.completed(model: model.displayName, path: path))
                } catch {
                    continuation.yield(.failed(model: model.displayName, error: error.localizedDescription))
                }

                continuation.finish()
            }
        }
    }

    /// Get a model's local path synchronously. Returns nil if not downloaded yet.
    public func cachedPath(for model: CatalystModel) -> String? {
        let filePath = cacheDirectory.appendingPathComponent(model.filename).path
        if FileManager.default.fileExists(atPath: filePath) {
            return filePath
        }
        return nil
    }

    /// Get a model and return just the path (blocks until download completes)
    public func resolve(_ model: CatalystModel) async throws -> String {
        // Check cache
        if let path = cachedPath(for: model) {
            return path
        }

        // Download
        let path = try await download(model: model, onProgress: nil)
        try validateGGUF(at: path)
        return path
    }

    /// Create a ModelProfile for a preset model (downloads if needed)
    public func profile(for model: CatalystModel) async throws -> ModelProfile {
        let path = try await resolve(model)
        return try ModelProfile(
            filePath: path,
            name: model.rawValue,
            architecture: model.architecture
        )
    }

    /// Delete a cached model to free disk space
    public func deleteCache(for model: CatalystModel) throws {
        let filePath = cacheDirectory.appendingPathComponent(model.filename)
        if FileManager.default.fileExists(atPath: filePath.path) {
            try FileManager.default.removeItem(at: filePath)
            print("🗑️ [ModelDownloader] Deleted cached model: \(model.rawValue)")
        }
    }

    /// Delete all cached models
    public func clearAllCache() throws {
        if FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.removeItem(at: cacheDirectory)
            print("🗑️ [ModelDownloader] Cleared all cached models")
        }
    }

    /// List all cached models with their sizes
    public func listCachedModels() -> [(model: CatalystModel, sizeMB: Int)] {
        var cached: [(CatalystModel, Int)] = []
        for model in CatalystModel.allCases {
            let filePath = cacheDirectory.appendingPathComponent(model.filename).path
            if FileManager.default.fileExists(atPath: filePath),
               let attrs = try? FileManager.default.attributesOfItem(atPath: filePath),
               let size = attrs[.size] as? UInt64 {
                cached.append((model, Int(size / 1_048_576)))
            }
        }
        return cached
    }

    // MARK: - Private

    private func download(
        model: CatalystModel,
        onProgress: ((Double, Int, Int) -> Void)?
    ) async throws -> String {
        // Ensure cache directory exists
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        let destinationURL = cacheDirectory.appendingPathComponent(model.filename)
        let tempURL = destinationURL.appendingPathExtension("downloading")

        // Clean up any previous partial download
        try? FileManager.default.removeItem(at: tempURL)

        // Download with URLSession
        let (asyncBytes, response) = try await URLSession.shared.bytes(from: model.downloadURL)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw CatalystError.networkResourceUnavailable(
                details: "Failed to download \(model.rawValue): HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)"
            )
        }

        let totalBytes = Int(httpResponse.expectedContentLength)
        let totalMB = max(totalBytes / 1_048_576, model.approximateSizeMB)

        // Stream to disk
        let fileHandle = try FileHandle(forWritingTo: {
            FileManager.default.createFile(atPath: tempURL.path, contents: nil)
            return tempURL
        }())
        defer { try? fileHandle.close() }

        var downloadedBytes = 0
        var lastReportedProgress: Double = 0

        for try await byte in asyncBytes {
            fileHandle.write(Data([byte]))
            downloadedBytes += 1

            // Report progress every ~1%
            let progress = totalBytes > 0 ? Double(downloadedBytes) / Double(totalBytes) : 0
            if progress - lastReportedProgress >= 0.01 {
                lastReportedProgress = progress
                onProgress?(progress, downloadedBytes / 1_048_576, totalMB)
            }
        }

        // Move temp file to final location
        try? FileManager.default.removeItem(at: destinationURL)
        try FileManager.default.moveItem(at: tempURL, to: destinationURL)

        print("✅ [ModelDownloader] Downloaded \(model.rawValue) (\(downloadedBytes / 1_048_576) MB)")
        return destinationURL.path
    }

    private func validateGGUF(at path: String) throws {
        guard let handle = FileHandle(forReadingAtPath: path) else {
            throw CatalystError.modelFileNotFound(path: path)
        }
        defer { handle.closeFile() }

        let magic = handle.readData(ofLength: 4)
        guard magic == "GGUF".data(using: .ascii) else {
            // Delete corrupt file
            try? FileManager.default.removeItem(atPath: path)
            throw CatalystError.modelFileCorrupted(
                path: path,
                reason: "Invalid GGUF header — file may be corrupted or incomplete"
            )
        }
    }
}
