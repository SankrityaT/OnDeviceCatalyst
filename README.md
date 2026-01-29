# OnDeviceCatalyst

A Swift framework for running LLMs locally on iOS and macOS using llama.cpp. Full GPU acceleration with Metal, streaming generation, and tool calling support.

## Features

- **Full GPU Acceleration** - Metal backend for fast inference on Apple Silicon
- **Latest Model Support** - Qwen3, Llama 3.2/3.3, Gemma 3, Phi-4, DeepSeek V3
- **Streaming Generation** - Real-time token streaming with async/await
- **Tool Calling** - Built-in function calling support (Qwen3 native + JSON format)
- **Automatic Optimization** - Device-aware GPU/CPU settings
- **Simple API** - Clean, plug-and-play interface

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/SankrityaT/OnDeviceCatalyst.git", from: "1.0.0")
]
```

Or in Xcode: **File → Add Package Dependencies** → Enter the repository URL.

---

## Quick Start

### 1. Basic Text Generation

```swift
import OnDeviceCatalyst

// Create a model profile
let profile = try ModelProfile(
    filePath: modelPath,
    name: "Qwen3-4B",
    architecture: .qwen3
)

// Generate a response
let response = try await Catalyst.shared.complete(
    prompt: "What is the capital of France?",
    using: profile
)
print(response)
```

### 2. Streaming Generation

```swift
let conversation = [Turn.user("Tell me a story")]

let stream = try await Catalyst.shared.generate(
    conversation: conversation,
    systemPrompt: "You are a creative storyteller.",
    using: profile
)

for try await chunk in stream {
    print(chunk.content, terminator: "")
    if chunk.isComplete { break }
}
```

### 3. Multi-turn Conversation

```swift
var conversation = Conversation(
    title: "Chat",
    systemPrompt: "You are a helpful assistant."
)

// First message
let stream = try await Catalyst.shared.generateWithConversation(
    &conversation,
    userMessage: "Hi! What's your name?",
    using: profile
)

var response = ""
for try await chunk in stream {
    response += chunk.content
    if chunk.isComplete { break }
}

// Save response to conversation history
Catalyst.shared.addAssistantResponse(response, metadata: nil, to: &conversation, modelName: "Qwen3")

// Continue the conversation (context maintained automatically)
let stream2 = try await Catalyst.shared.generateWithConversation(
    &conversation,
    userMessage: "What did I just ask?",
    using: profile
)
```

### 4. Tool Calling

```swift
// Define available tools
let tools = [
    CatalystTool(
        name: "switch_coach",
        description: "Switch to a different coach when their expertise is needed",
        parameters: [
            CatalystToolParameter(
                name: "coach_name",
                description: "Name of the coach",
                enumValues: ["Fitness Coach", "Career Coach", "Life Coach"]
            ),
            CatalystToolParameter(
                name: "reason",
                description: "Why this coach would help"
            )
        ]
    ),
    CatalystTool(
        name: "get_weather",
        description: "Get current weather for a location",
        parameters: [
            CatalystToolParameter(name: "location", description: "City name")
        ]
    )
]

// Generate with tools
let (text, toolCalls) = try await Catalyst.shared.completeWithTools(
    prompt: "I want to get in shape. Can you help?",
    systemPrompt: "You are a helpful assistant.",
    tools: tools,
    using: profile
)

// Handle tool calls
for call in toolCalls {
    print("Tool: \(call.name)")

    if call.name == "switch_coach" {
        let coachName: String? = call.argument("coach_name")
        let reason: String? = call.argument("reason")
        // Execute the switch...
    }
}

// Or display the text response
print(text)
```

### 5. Custom Settings

```swift
// Custom instance settings
let settings = InstanceSettings(
    contextLength: 8192,
    batchSize: 512,
    gpuLayers: 99,           // 99 = all layers on GPU
    cpuThreads: 8,
    useFlashAttention: true
)

// Sampling configuration
let config = PredictionConfig(
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    maxTokens: 2048
)

let stream = try await Catalyst.shared.generate(
    conversation: conversation,
    systemPrompt: systemPrompt,
    using: profile,
    settings: settings,
    predictionConfig: config
)
```

---

## Supported Models

| Architecture | Example Models | Notes |
|--------------|----------------|-------|
| `.qwen3` | Qwen3-4B, Qwen3-8B | Native tool calling, thinking mode auto-disabled |
| `.qwen25` | Qwen 2.5 0.5B-72B | Great for coding |
| `.llama32` | Llama 3.2 1B-3B | Meta's compact models |
| `.llama33` | Llama 3.3 70B | Latest Llama |
| `.gemma3` | Gemma 3 | Google's open model |
| `.phi4` | Phi-4 | Microsoft's efficient model |
| `.deepSeekV3` | DeepSeek V3 | Strong reasoning |
| `.mistral` | Mistral 7B | Fast and capable |

### Recommended Models for iOS

| Model | Size | Best For |
|-------|------|----------|
| Qwen3-4B-Q4_K_M | ~2.7GB | Best quality/speed balance |
| Qwen2.5-3B-Q4_K_M | ~2.0GB | Coding, general use |
| Llama-3.2-3B-Q4_K_M | ~1.8GB | Fast responses |

### Download URLs

```
Qwen3-4B:     https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
Qwen2.5-3B:   https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
Llama-3.2-3B: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

## Presets

### Instance Settings

```swift
InstanceSettings.balanced          // Default, works on most devices
InstanceSettings.coaching(maxContextTokens: 8192)  // Optimized for chat apps
InstanceSettings.embedding()       // For embedding models (BERT, etc.)
InstanceSettings.memoryEfficient   // CPU-only, low memory
InstanceSettings.iphone16ProMax    // iPhone 16 Pro optimized
```

### Prediction Config

```swift
PredictionConfig.balanced      // Good default (temp=0.7)
PredictionConfig.creative      // More varied (temp=0.9)
PredictionConfig.deterministic // Consistent output (temp=0)
PredictionConfig.mirostat      // Adaptive entropy
PredictionConfig.speed         // Fast generation
```

---

## Device Optimization

OnDeviceCatalyst automatically detects your device and optimizes settings:

| Device | GPU Layers | Context | Performance |
|--------|------------|---------|-------------|
| iPhone 16 Pro | 99 (full) | 8192 | ~15-25 tok/s |
| iPhone 15 Pro | 99 (full) | 4096 | ~10-20 tok/s |
| iPhone 14 Pro | 99 (full) | 4096 | ~8-15 tok/s |
| iPhone 13 | 20 | 4096 | ~5-10 tok/s |
| iPhone 12 | 20 | 2048 | ~3-8 tok/s |
| Older iPhones | 0 (CPU) | 2048 | ~1-3 tok/s |
| Mac (M1/M2/M3) | 99 (full) | 16384 | ~30-60 tok/s |

---

## API Reference

### Catalyst (Main Service)

```swift
// Singleton
Catalyst.shared

// Simple completion
func complete(prompt:systemPrompt:using:settings:predictionConfig:) async throws -> String

// Streaming
func generate(conversation:systemPrompt:using:settings:predictionConfig:) async throws -> AsyncThrowingStream<StreamChunk, Error>

// With tools
func generateWithTools(conversation:systemPrompt:tools:using:) async throws -> AsyncThrowingStream<StreamChunk, Error>
func completeWithTools(prompt:systemPrompt:tools:using:) async throws -> (text: String, toolCalls: [CatalystToolCall])

// Conversation management
func generateWithConversation(_:userMessage:using:) async throws -> AsyncThrowingStream<StreamChunk, Error>
func addAssistantResponse(_:metadata:to:modelName:)

// Control
func interruptGeneration(for profileId: String)
func releaseInstance(for profileId: String)
func shutdownAll() async
```

### Tool Support

```swift
// Define a tool
let tool = CatalystTool(
    name: "function_name",
    description: "What this function does",
    parameters: [
        CatalystToolParameter(
            name: "param1",
            type: "string",
            description: "Parameter description",
            required: true,
            enumValues: ["option1", "option2"]  // optional
        )
    ]
)

// Parse tool calls from any response
let (cleanText, toolCalls) = ToolCallParser.parse(from: responseText)

// Access arguments with type safety
let value: String? = toolCall.argument("param1")
```

---

## Error Handling

```swift
do {
    let response = try await Catalyst.shared.complete(prompt: "Hello", using: profile)
} catch let error as CatalystError {
    switch error {
    case .modelFileNotFound(let path):
        print("Model not found: \(path)")
    case .modelLoadingFailed(let details):
        print("Load failed: \(details)")
    case .contextWindowExceeded(let count, let limit):
        print("Context too long: \(count)/\(limit)")
    case .generationFailed(let details):
        print("Generation error: \(details)")
    case .operationCancelled:
        print("Cancelled by user")
    default:
        print(error.localizedDescription)
    }
}
```

---

## Requirements

- iOS 16.0+ / macOS 13.0+
- Xcode 15.0+
- Swift 5.9+

## License

MIT License

## Contributing

Issues and PRs welcome!
