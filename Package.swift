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
        .library(name: "SwitchcraftStorageTesting", targets: ["SwitchcraftStorageTesting"]),
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
        .target(
            name: "SwitchcraftStorageTesting",
            dependencies: ["SwitchcraftCore"],
            path: "Sources/SwitchcraftStorageTesting"
        ),
        .testTarget(
            name: "SwitchcraftTests",
            dependencies: [
                "Switchcraft",
                "SwitchcraftSQLite",
                "SwitchcraftCore",
                "SwitchcraftStorageTesting",
            ],
            path: "Tests/SwitchcraftTests",
            resources: [.copy("../Fixtures")]
        ),
    ]
)
