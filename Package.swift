// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// ============================================================================
// CONFIGURATION - Update these values before releasing
// ============================================================================

// Checksum for the xcframework zip
// Compute with: swift package compute-checksum llama.xcframework.zip
// UPDATE THIS after uploading to GitHub releases!
let xcframeworkChecksum = "13e6fa4b7f91c708ba405ffc2ab9d9118cd137516d12b1fa6a221d889ee803bd"

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
        // llama.cpp XCFramework — headers + modulemap in Headers/llama/ subdirectory
        // to avoid "Multiple commands produce module.modulemap" collision with other
        // xcframeworks (e.g. sentencepiece). Clang finds it via llama/module.modulemap.
        .binaryTarget(
            name: "llama",
            url: "https://github.com/SankrityaT/OnDeviceCatalyst/releases/download/v2.0.3/llama.xcframework.zip",
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
