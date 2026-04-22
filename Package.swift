// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// ============================================================================
// CONFIGURATION - Update these values before releasing
// ============================================================================

// Checksum for the xcframework zip
// Compute with: swift package compute-checksum llama.xcframework.zip
// UPDATE THIS after uploading to GitHub releases!
let xcframeworkChecksum = "741c3b584228c290c06bfbced9db161c3e7cb920c85d4d8df4ec54e8188a4e39"

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
        .package(url: "https://github.com/ml-explore/mlx-swift-lm/", exact: "2.29.3"),
    ],
    targets: [
        // llama.cpp XCFramework — headers + modulemap in Headers/llama/ subdirectory
        // to avoid "Multiple commands produce module.modulemap" collision with other
        // xcframeworks (e.g. sentencepiece). Clang finds it via llama/module.modulemap.
        // Includes an arm64-simulator stub slice so consumers can build for the iOS
        // Simulator (all llama usage is guarded #if !targetEnvironment(simulator)).
        .binaryTarget(
            name: "llama",
            url: "https://github.com/SankrityaT/OnDeviceCatalyst/releases/download/v2.0.4/llama.xcframework.zip",
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
