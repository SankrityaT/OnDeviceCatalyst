// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// ============================================================================
// CONFIGURATION - Update these values before releasing
// ============================================================================

// Checksum for the xcframework zip
// Compute with: swift package compute-checksum llama.xcframework.zip
// UPDATE THIS after uploading to GitHub releases!
let xcframeworkChecksum = "a234cff03412d7df5ad0727e5a8cd3a4d58b5a6a90b695fb2e91d32e5092b5ac"

// ============================================================================

let package = Package(
    name: "OnDeviceCatalyst",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "OnDeviceCatalyst",
            targets: ["OnDeviceCatalyst"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm/", branch: "main"),
    ],
    targets: [
        // llama.cpp XCFramework with full model support:
        // Qwen3, Gemma3, Llama 3.2/3.3, Phi-4, DeepSeek V3, and more
        .binaryTarget(
            name: "llama",
            url: "https://github.com/SankrityaT/OnDeviceCatalyst/releases/download/v2.0.1/llama.xcframework.zip",
            checksum: xcframeworkChecksum
        ),
        .target(
            name: "OnDeviceCatalyst",
            dependencies: [
                "llama",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/OnDeviceCatalyst",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation")
            ]
        ),
        .testTarget(
            name: "OnDeviceCatalystTests",
            dependencies: ["OnDeviceCatalyst"],
            path: "Tests/OnDeviceCatalystTests"
        ),
    ]
)
