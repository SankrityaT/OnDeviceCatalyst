# Architecture

## Current v2 source of truth

The Swift package in `Sources/OnDeviceCatalyst` is authoritative. The top-level
`OnDeviceCatalyst` directory is a legacy SwiftUI application containing older
copies of library files.

```text
Application
    |
    v
Catalyst service facade
    |
    +-- LlamaInstance
    |      +-- LlamaCppBackend -> LlamaBridge -> llama XCFramework
    |      +-- MetalBackend -> GGUF parser/tokenizer -> Metal kernels
    |
    +-- MLXInstance -> MLX Swift LM

Supporting services
    +-- chat, prompt formatting, streams, and stop sequences
    +-- tools and tool-call parsing
    +-- model download and instance cache
    +-- settings, device optimization, safety, and metrics
```

Known v2 debt includes duplicated sources, an iOS-only published llama artifact
despite declared macOS support, invalid stream lifecycle behavior, unsafe cache
ownership, unhandled Metal shader resources, limited tests, and concurrency
state that predates Swift 6.

ODC-0002 captures the reproducible baseline. ODC-0004 captures behavior before
runtime repair.

## V3 constraints already approved

- The public core remains independent of a concrete backend.
- The core compiles for iOS and iPadOS 17-26 and macOS 14-26.
- Llama.cpp, MLX, and Apple system-model integration are optional products.
- Heavy backend dependencies do not resolve for core-only consumers.
- Backend C or C++ types never cross the public core API.
- Lifecycle state moves to explicit Swift 6 actor ownership.
- Streams have exactly one documented terminal event and cancellation reaches
  underlying work.
- Canonical backend or tokenizer chat templates replace filename-based manual
  claims where available.
- Local model files never require the downloader service.

Specific package names, public types, migration behavior, and backend selection
remain blocked until their ODC-0100 series specs are approved.

## Apple-native integration boundary

The shipping Xcode 26 SDK exposes Apple Foundation Models on iOS 26 through
`SystemLanguageModel` and `LanguageModelSession`. V3 may wrap that system model
as an optional Catalyst backend while preserving its own older-OS API.

Preview-only custom-provider and future Core AI APIs are research inputs only.
They cannot be product dependencies until a future stable-SDK spec is reviewed.

Model delivery must support local files and ordinary application downloads.
ODC-0206 separately evaluates Apple Background Assets for App Store delivery.
