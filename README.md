# OnDeviceCatalyst

OnDeviceCatalyst is an Apple-first Swift package for private, efficient
on-device language-model inference.

> [!IMPORTANT]
> The project is in a public revival. The tagged v2 implementation remains
> available for experimentation, but it has known packaging, lifecycle, and
> test gaps and is not currently presented as production-stable. The v3 program
> is spec-driven, and runtime changes do not begin until their specifications
> are reviewed and approved.

## North star

Build a free package that Apple-platform developers can trust for local LLM
generation and embeddings, while supporting a separate research program that
can produce genuinely new inference results rather than repeating existing
runtimes.

The first product-grade v3 release is scoped to:

- iOS and iPadOS 17 through 26.
- macOS 14 through 26.
- Swift 6 concurrency and explicit lifecycle ownership.
- Optional llama.cpp and MLX backends behind one behavioral contract.
- Streaming, structured output, tool calling, and embeddings.
- An optional iOS 26 Apple system-model backend.
- Model delivery that works with local files, ordinary downloads, and an
  optional Apple Background Assets integration.

Preview-only platform APIs are research inputs, not release requirements.

## Current repository

`Sources/OnDeviceCatalyst` is the Swift package source of truth. The top-level
`OnDeviceCatalyst` directory is a legacy demonstration app containing older
source copies. See [the architecture guide](docs/ARCHITECTURE.md) before making
structural changes.

The current v2 package contains:

- llama.cpp and MLX inference paths.
- An experimental custom Metal backend.
- Conversation and prompt-formatting types.
- Streaming and stop-sequence processing.
- Tool-call parsing and prompt formatting.
- Embedding extraction.
- Model downloading, caching, device settings, and performance utilities.

Known issues and the recovery sequence are tracked in
[ROADMAP.md](ROADMAP.md) and [Tickets.md](Tickets.md).

## Work on the project

Every change starts from a ticket and reviewed specification. Begin with:

1. [AGENTS.md](AGENTS.md) for continuity and execution gates.
2. [ROADMAP.md](ROADMAP.md) for product direction and current state.
3. [Tickets.md](Tickets.md) for the canonical work ledger.
4. [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and DCO requirements.

Major decisions are preserved in [architecture decision records](docs/decisions).
Public benchmark methods and released research live in [docs/research](docs/research).

## Build the current package

Resolve dependencies:

```sh
swift package resolve
```

Build for an iOS simulator using the installed Xcode SDK:

```sh
CATALYST_SDK_PATH="$(xcrun --sdk iphonesimulator --show-sdk-path)"
swift build \
  --triple arm64-apple-ios17.0-simulator \
  --sdk "$CATALYST_SDK_PATH"
```

The v2.0.4 binary artifact lacks a macOS slice even though the package declares
macOS support. ODC-0002 owns reproduction and documentation of the complete v2
baseline before that behavior is changed.

## License

OnDeviceCatalyst is licensed under the [Apache License 2.0](LICENSE). Public
contributions require Developer Certificate of Origin sign-off.
