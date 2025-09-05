// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

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
    dependencies: [
        .package(
            url: "https://github.com/ggerganov/llama.cpp",
            revision: "9b75f03cd2ec9cc482084049d87a0f08f9f01517"
        )
    ],
    targets: [
        .target(
            name: "OnDeviceCatalyst",
            dependencies: [
                .product(name: "llama", package: "llama.cpp")
            ],
            path: "Sources/OnDeviceCatalyst",
            resources: [
                .process("Resources")
            ],
            cSettings: [
                .headerSearchPath("include"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_ACCELERATE"),
                .unsafeFlags(["-fmodules", "-fcxx-modules"])
            ],
            cxxSettings: [
                .headerSearchPath("include"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_ACCELERATE"),
                .unsafeFlags(["-fmodules", "-fcxx-modules"])
            ],
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
