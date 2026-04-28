// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Switchcraft",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "Switchcraft", targets: ["Switchcraft"]),
        .library(name: "SwitchcraftSQLite", targets: ["SwitchcraftSQLite"]),
    ],
    targets: [
        .target(
            name: "SwitchcraftCore",
            path: "Sources/SwitchcraftCore"
        ),
        .target(
            name: "Switchcraft",
            dependencies: ["SwitchcraftCore"],
            path: "Sources/Switchcraft"
        ),
        .target(
            name: "SwitchcraftSQLite",
            dependencies: ["SwitchcraftCore"],
            path: "Sources/SwitchcraftSQLite"
        ),
        .testTarget(
            name: "SwitchcraftTests",
            dependencies: ["Switchcraft", "SwitchcraftSQLite", "SwitchcraftCore"],
            path: "Tests/SwitchcraftTests"
        ),
    ]
)
