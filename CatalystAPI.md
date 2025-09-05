# Catalyst AI Engine - Public API

## Quick Start

```swift
import OnDeviceCatalyst

// 1. Create model profile
let modelProfile = try ModelProfile(
    filePath: Bundle.main.path(forResource: "qwen2.5-3b-instruct-q4_k_m", ofType: "gguf")!,
    name: "My AI Assistant"
)

// 2. Generate response
let stream = try await Catalyst.shared.generate(
    conversation: [Turn.user("Hello, how are you?")],
    systemPrompt: "You are a helpful AI assistant.",
    using: modelProfile,
    settings: .iphone16ProMax,
    predictionConfig: .speed
)

// 3. Process streaming response
for try await chunk in stream {
    print(chunk.content, terminator: "")
    if chunk.isComplete {
        break
    }
}
```

## Core API Methods

### 1. Simple Generation (Recommended)
```swift
func generate(
    conversation: [Turn],
    systemPrompt: String,
    using profile: ModelProfile,
    settings: InstanceSettings = .balanced,
    predictionConfig: PredictionConfig = .balanced
) async throws -> AsyncThrowingStream<StreamChunk, Error>
```

### 2. Conversation Management
```swift
func generateWithConversation(
    _ conversation: inout Conversation,
    userMessage: String,
    using profile: ModelProfile,
    settings: InstanceSettings = .balanced,
    predictionConfig: PredictionConfig = .balanced
) async throws -> AsyncThrowingStream<StreamChunk, Error>
```

### 3. Non-Streaming Generation
```swift
func generateComplete(
    conversation: [Turn],
    systemPrompt: String,
    using profile: ModelProfile,
    settings: InstanceSettings = .balanced,
    predictionConfig: PredictionConfig = .balanced
) async throws -> String
```

## Configuration Presets

### Device-Optimized Settings
```swift
.iphone16ProMax    // 2048 context, 256 batch, 25 GPU layers
.balanced          // Default balanced settings
.memoryEfficient   // CPU-only, minimal memory
.highCapacity      // Maximum context length
```

### Prediction Configs
```swift
.speed      // Fast responses, lower quality
.balanced   // Good balance of speed and quality
.quality    // Best quality, slower responses
```

## Example: Chat App Integration

```swift
class ChatService {
    private let modelProfile: ModelProfile
    private var conversation = Conversation(systemPrompt: "You are a helpful assistant.")
    
    init() throws {
        self.modelProfile = try ModelProfile(
            filePath: Bundle.main.path(forResource: "qwen2.5-3b-instruct-q4_k_m", ofType: "gguf")!
        )
    }
    
    func sendMessage(_ message: String) async throws -> AsyncThrowingStream<String, Error> {
        let stream = try await Catalyst.shared.generateWithConversation(
            &conversation,
            userMessage: message,
            using: modelProfile,
            settings: .iphone16ProMax,
            predictionConfig: .speed
        )
        
        return AsyncThrowingStream { continuation in
            Task {
                for try await chunk in stream {
                    continuation.yield(chunk.content)
                    if chunk.isComplete {
                        continuation.finish()
                        return
                    }
                }
            }
        }
    }
}
```

## Error Handling

```swift
do {
    let stream = try await Catalyst.shared.generate(...)
    // Process stream
} catch let error as CatalystError {
    switch error {
    case .modelFileNotFound(let path):
        print("Model not found: \(path)")
    case .engineNotInitialized:
        print("Engine failed to initialize")
    case .generationFailed(let reason):
        print("Generation failed: \(reason)")
    default:
        print("Other error: \(error)")
    }
}
```

## Required Files in Your Project

1. **Model file**: `qwen2.5-3b-instruct-q4_k_m.gguf` (~2GB)
2. **Framework**: OnDeviceCatalyst framework
3. **Bundle**: Add model to app bundle

## Performance Notes

- **iPhone 16 Pro Max**: Use `.iphone16ProMax` settings for optimal GPU acceleration
- **Older devices**: Use `.memoryEfficient` for CPU-only operation
- **Model size**: 3B models are the sweet spot for mobile (2GB vs 4GB for 7B)
- **Streaming**: Always use streaming for real-time UI updates
