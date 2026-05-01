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

        // Phase 2 Metal embedder support — GGUF reader + (in subsequent
        // sub-issues) Q4_K matmul / RMSNorm / SDPA kernels and the
        // T5MetalEmbedder. Requires the Q4-quantised GGUF asset gated by
        // SWITCHCRAFT_XTR_GGUF (see ADR 016 and ADR 010(j)).
        .library(name: "SwitchcraftMetal", targets: ["SwitchcraftMetal"]),

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
            path: "Sources/SwitchcraftCore",
            resources: [
                // SwiftPM 6's `.process(...)` ships `.metal` files as raw
                // source rather than as a precompiled `.metallib`; the
                // `MetalContext` runtime-source fallback handles that case.
                // See ADR 015 for the build-strategy rationale.
                .process("Metal/Shaders"),
            ],
            linkerSettings: [
                .linkedFramework(
                    "Metal",
                    .when(platforms: [.macOS, .iOS, .visionOS])
                ),
                // MetalPerformanceShaders is intentionally NOT linked here —
                // it is deferred to the first sub-issue that actually uses
                // an MPS primitive (ADR 015 §"Build / packaging").
            ]
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
        // Phase 2 Metal embedder (umbrella issue #57). Sub-issue #59
        // lands the GGUF reader + Q4_K CPU dequantisation reference and
        // the @_spi(SwitchcraftMetal) hatch on MetalContext that lets
        // this target reuse SwitchcraftCore's pipeline cache without
        // re-foundation. Future sub-issues (#60–#64) add the kernels and
        // T5MetalEmbedder here. See ADR 015 (Metal context) and ADR 016
        // (GGUF asset distribution).
        .target(
            name: "SwitchcraftMetal",
            // SwitchcraftCoreML is depended on solely for `SlidingWindow`
            // (pure-Swift sliding-window planner/merger reused per ADR
            // 011). CoreML the framework is not actually used here.
            dependencies: ["SwitchcraftCore", "SwitchcraftCoreML"],
            path: "Sources/SwitchcraftMetal",
            resources: [
                // Per ADR 015 §(e), each target that ships Metal shaders
                // owns its own bundle; the kernels here are registered
                // with `MetalContext` lazily at first kernel-init time.
                // SwiftPM 6's `.process(...)` ships `.metal` raw and
                // the runtime-source-compile fallback in `MetalContext`
                // handles that case.
                .process("Shaders"),
            ],
            linkerSettings: [
                .linkedFramework(
                    "Metal",
                    .when(platforms: [.macOS, .iOS, .visionOS])
                ),
                // MetalPerformanceShaders deliberately omitted — Phase 2
                // ships custom kernels only (ADR 015 §"Build / packaging").
            ]
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
        // Tests for SwitchcraftMetal (issue #59 onwards). Header-parsing
        // and Q4_K decode tests run unconditionally on in-memory
        // fixtures; round-trip parity tests against the upstream Q4_K
        // GGUF are gated on the SWITCHCRAFT_XTR_GGUF env var (ADR 016).
        .testTarget(
            name: "SwitchcraftMetalTests",
            dependencies: [
                "SwitchcraftMetal",
                "SwitchcraftCore",
            ],
            path: "Tests/SwitchcraftMetalTests",
            resources: [.copy("../Fixtures")]
        ),
    ]
)
