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
        // Standalone Metal-matmul prototype for the Phase 2 custom-Metal-kernels
        // feasibility investigation (issue #49). Intentionally NOT in `products`
        // — only the matching test target consumes it. Library is fully isolated
        // (zero Switchcraft dependencies) so it can be deleted or re-homed without
        // ripple. See `docs/investigations/metal-matmul-feasibility.md`.
        .target(
            name: "SwitchcraftMetalProto",
            path: "Sources/SwitchcraftMetalProto",
            resources: [
                .process("MetalMatmul.metal"),
            ],
            linkerSettings: [
                .linkedFramework(
                    "Metal",
                    .when(platforms: [.macOS, .iOS, .visionOS])
                ),
                .linkedFramework(
                    "MetalPerformanceShaders",
                    .when(platforms: [.macOS, .iOS, .visionOS])
                ),
            ]
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
        // Test target for the Metal-matmul prototype. Gated at suite level by
        // env vars (correctness: SWITCHCRAFT_METAL_PROTO=1, benchmarks:
        // SWITCHCRAFT_METAL_PROTO_BENCH=1 + release build) so the default
        // `swift test` invocation skips the entire suite cleanly.
        .testTarget(
            name: "SwitchcraftMetalProtoTests",
            dependencies: [
                "SwitchcraftMetalProto",
                "SwitchcraftCore",
            ],
            path: "Tests/SwitchcraftMetalProtoTests"
        ),
    ]
)
