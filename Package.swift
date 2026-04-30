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
        // Primary consumer product: SwitchcraftStore actor + Embedder protocol
        // + StoreConfig. Re-exports SwitchcraftCore. Stable public API surface
        // per ADR 009.
        .library(name: "Switchcraft", targets: ["Switchcraft"]),

        // SQLite + FTS5 storage backend. Adopters import this alongside
        // Switchcraft to use the `SwitchcraftStore.sqlite(...)` factory.
        .library(name: "SwitchcraftSQLite", targets: ["SwitchcraftSQLite"]),

        // CoreML embedder (T5CoreMLEmbedder) backed by google/xtr-base-en.
        // Requires the .mlpackage asset built via
        // `scripts/convert-xtr-to-coreml.py` (see ADR 010).
        .library(name: "SwitchcraftCoreML", targets: ["SwitchcraftCoreML"]),

        // Test-support only: a reusable conformance suite for adopters
        // writing custom SwitchcraftStorage backends. Not intended for
        // end-user apps. Whether this stays a top-level product or moves
        // behind an SPM trait is deferred (see issue #41).
        .library(name: "SwitchcraftStorageTesting", targets: ["SwitchcraftStorageTesting"]),

        // SwitchcraftCore is intentionally NOT a top-level product per
        // ADR 009(i). It is an internal target re-exported by
        // `Switchcraft`; depending on it directly is unsupported.
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
            dependencies: ["SwitchcraftCore", "Switchcraft"],
            path: "Sources/SwitchcraftSQLite"
        ),
        .target(
            name: "SwitchcraftCoreML",
            dependencies: ["SwitchcraftCore"],
            path: "Sources/SwitchcraftCoreML",
            linkerSettings: [
                .linkedFramework(
                    "CoreML",
                    .when(platforms: [.macOS, .iOS, .visionOS])
                ),
            ]
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
                "SwitchcraftCoreML",
                "SwitchcraftCore",
                "SwitchcraftStorageTesting",
            ],
            path: "Tests/SwitchcraftTests",
            resources: [.copy("../Fixtures")]
        ),
    ]
)
