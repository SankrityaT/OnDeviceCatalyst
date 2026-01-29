// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// ============================================================================
// CONFIGURATION - Update these values before releasing
// ============================================================================

// Checksum for the xcframework zip
// Compute with: swift package compute-checksum llama.xcframework.zip
// UPDATE THIS after uploading to GitHub releases!
let xcframeworkChecksum = "8a3bb0e2d153ad68438704636b94ffe613fc8e953beb7aefa23a9ad5b8768071"

// ============================================================================

let package = Package(
    name: "OnDeviceCatalyst",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "OnDeviceCatalyst",
            targets: ["OnDeviceCatalyst"]
        ),
    ],
    dependencies: [],
    targets: [
        // llama.cpp XCFramework with full model support:
        // Qwen3, Gemma3, Llama 3.2/3.3, Phi-4, DeepSeek V3, and more
        .binaryTarget(
            name: "llama",
            url: "https://github.com/SankrityaT/OnDeviceCatalyst/releases/download/v1.0.0/llama.xcframework.zip",
            checksum: xcframeworkChecksum
        ),
        .target(
            name: "OnDeviceCatalyst",
            dependencies: ["llama"],
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
