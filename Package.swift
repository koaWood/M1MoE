// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "M1MoE",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "M1MoE", targets: ["M1MoE"]),
        .executable(name: "bench", targets: ["bench"]),
        .executable(name: "chat",  targets: ["chat"]),
    ],
    targets: [
        .target(
            name: "M1MoE",
            path: "Sources/M1MoE",
            resources: [.copy("Shaders.metal")]
        ),
        .executableTarget(name: "bench", dependencies: ["M1MoE"], path: "Sources/bench"),
        .executableTarget(name: "chat",  dependencies: ["M1MoE"], path: "Sources/chat"),
    ]
)
