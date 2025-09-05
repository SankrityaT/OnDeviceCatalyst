# OnDeviceCatalyst

A high-performance Swift package for running large language models on-device using llama.cpp, optimized for iOS devices.

## Features

- 🚀 **GPU Acceleration** - Optimized Metal kernels for Apple Silicon
- 📱 **Mobile Optimized** - Tuned for iPhone 13-16 series with 3B model support
- 🔄 **Streaming Generation** - Real-time token streaming for responsive UIs
- 💾 **Memory Efficient** - Smart caching and memory management
- 🛡️ **Production Ready** - Comprehensive error handling and safety features

## Quick Start

### Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/SankrityaT/OnDeviceCatalyst", from: "1.0.0")
]
```

### Basic Usage

```swift
import OnDeviceCatalyst

// 1. Create model profile
let modelProfile = try ModelProfile(
    filePath: Bundle.main.path(forResource: "qwen2.5-3b-instruct-q4_k_m", ofType: "gguf")!,
    name: "My AI Assistant"
)

// 2. Generate streaming response
let stream = try await Catalyst.shared.generate(
    conversation: [Turn.user("Hello, how are you?")],
    systemPrompt: "You are a helpful AI assistant.",
    using: modelProfile,
    settings: .iphone16ProMax,
    predictionConfig: .speed
)

// 3. Process response
for try await chunk in stream {
    print(chunk.content, terminator: "")
    if chunk.isComplete { break }
}
```

## Recommended Models

For optimal performance on iOS devices:

- **Qwen2.5-3B-Instruct-Q4_K_M** (~2.0GB) - Best balance of quality and speed
- **Llama-3.2-3B-Instruct-Q4_0** (~1.8GB) - Faster, slightly lower quality

## Device Optimization

### iPhone 16 Pro Max
```swift
.iphone16ProMax  // 2048 context, 256 batch, 25 GPU layers
```

### Older Devices
```swift
.memoryEfficient  // CPU-only, minimal memory usage
```

## Performance

| Device | Model Size | Speed | GPU Layers |
|--------|------------|-------|------------|
| iPhone 16 Pro Max | 3B Q4_K_M | 2-5 t/s | 25 |
| iPhone 15 Pro | 3B Q4_0 | 1-3 t/s | 20 |
| iPhone 14 | 1.5B Q4_0 | 3-6 t/s | 15 |

## Requirements

- iOS 16.0+ / macOS 13.0+
- Model file downloaded separately (see below)
- ~2GB free storage for 3B models

## Model Setup

Download the recommended model file and add it to your app bundle:

**Recommended Model:** `qwen2.5-3b-instruct-q4_k_m.gguf` (~2.0GB)
- Download from: [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
- Add to your Xcode project bundle
- Reference in code: `Bundle.main.path(forResource: "qwen2.5-3b-instruct-q4_k_m", ofType: "gguf")`

## Documentation

See [CatalystAPI.md](CatalystAPI.md) for complete API documentation and examples.

## License

MIT License - see LICENSE file for details.
