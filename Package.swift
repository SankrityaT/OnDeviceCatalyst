// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// ============================================================================
// CONFIGURATION - Update these values before releasing
// ============================================================================

// Checksum for the xcframework zip
// Compute with: swift package compute-checksum llama.xcframework.zip
// UPDATE THIS after uploading to GitHub releases!
let xcframeworkChecksum = "64bec1e522513f2b4d705868664f9d654ae0f5a097c8a9a8b89fd74c40542642"

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
        // llama.cpp XCFramework — headers only, no modulemap (avoids collision
        // with other xcframeworks like sentencepiece). Module defined by CLlama.
        .binaryTarget(
            name: "llama_binary",
            url: "https://github.com/SankrityaT/OnDeviceCatalyst/releases/download/v2.0.2/llama.xcframework.zip",
            checksum: xcframeworkChecksum
        ),
        // C wrapper that provides the `llama` module via its own modulemap.
        // This avoids "Multiple commands produce module.modulemap" when another
        // SPM package also ships a binary xcframework with a modulemap.
        .target(
            name: "CLlama",
            dependencies: ["llama_binary"],
            path: "Sources/CLlama"
        ),
        .target(
            name: "OnDeviceCatalyst",
            dependencies: [
                "CLlama",
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
