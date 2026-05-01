// SPDX-License-Identifier: Apache-2.0
//
// Metal context for the Phase 2 search-path kernels (umbrella #50,
// scaffolding #51).
//
// Responsibilities
// ----------------
//
// - Owns the process-wide `MTLDevice` + `MTLCommandQueue` + `MTLLibrary`
//   so individual kernel call sites don't pay the (non-trivial) Metal
//   construction cost on every dispatch.
// - Caches `MTLComputePipelineState` per kernel function name, guarded by
//   an `NSLock` so concurrent search threads can resolve pipelines safely.
// - Provides a single `MetalContext.shared` lazy global that returns
//   `nil` when Metal is unavailable or the `SWITCHCRAFT_FORCE_ACCELERATE`
//   env var is set, so call sites can branch onto the Accelerate path.
//
// Sendable policy
// ---------------
//
// `MetalContext` is `@unchecked Sendable`. Justification:
//
//   1. `MTLDevice` and `MTLCommandQueue` are documented thread-safe by
//      Apple ("Metal best practices: Resource synchronization") — they
//      can be shared across threads without external synchronization.
//   2. `MTLLibrary` is **not** documented as thread-safe. We resolve the
//      library exactly once during `init` (single-threaded by definition,
//      because `static let` enforces this) and never mutate it
//      afterwards; reads from a fully-initialised, never-mutated library
//      are safe.
//   3. The pipeline-state cache is a mutable `[String: ...]` dictionary.
//      All access goes through `pipeline(for:)`, which holds an
//      `NSLock` for the read-cache-or-build-and-store critical section.
//      `MTLComputePipelineState` itself is documented thread-safe so
//      sharing the cached value across threads after release is safe.
//
// `actor` isolation was rejected as the alternative because forcing
// `await` at every kernel dispatch site would conflict with the
// synchronous-core tenet (CLAUDE.md, ADR 009) — `SearchEngine`'s hot
// paths are synchronous and cannot easily switch to async-only Metal
// dispatch without ripple effects through the engine.
//
// Library load strategy
// ---------------------
//
// SwiftPM 6's `.process(...)` resource declaration empirically ships
// `.metal` files as raw source rather than as a compiled `.metallib`
// (verified during the issue #49 matmul investigation, see
// `docs/investigations/metal-matmul-feasibility.md`). To handle both
// cases robustly we try `device.makeDefaultLibrary(bundle:)` first; on
// failure we fall back to `device.makeLibrary(source:)` reading the
// `.metal` file as a bundle resource string. The fallback runs once at
// `init`; pipeline-state caching keeps the per-call overhead at zero.
//
// See `adrs/015-metal-context-and-dispatch.md` for the full rationale.

#if canImport(Metal)

import Foundation
import Metal
import os

@_spi(SwitchcraftMetal)
public enum MetalContextError: Error, CustomStringConvertible {
    case noDeviceAvailable
    case commandQueueCreationFailed
    case libraryLoadFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)

    public var description: String {
        switch self {
        case .noDeviceAvailable:
            return "MTLCreateSystemDefaultDevice() returned nil — no Metal-capable GPU"
        case .commandQueueCreationFailed:
            return "device.makeCommandQueue() returned nil"
        case .libraryLoadFailed(let m): return "Metal library load failed: \(m)"
        case .functionNotFound(let n): return "Metal function not found: \(n)"
        case .pipelineCreationFailed(let m): return "Metal pipeline creation failed: \(m)"
        }
    }
}

/// Process-wide Metal context for the search-path kernels.
///
/// Visibility is **`@_spi(SwitchcraftMetal) public`**, not full `public`,
/// so the type does not appear in the stable Switchcraft public API
/// surface (ADR 009). The SPI hatch lets the sibling `SwitchcraftMetal`
/// target reuse this pipeline cache without re-implementing it; consumers
/// outside the package cannot reach the type. Internal `SwitchcraftCore`
/// call-sites continue to use the default-internal access. See ADR 015
/// §"Sendable policy" and ADR 016 §"`@_spi(SwitchcraftMetal)` import
/// pattern".
@_spi(SwitchcraftMetal)
public final class MetalContext: @unchecked Sendable {
    /// Env var that forces the Accelerate fallback path even when a
    /// Metal device is available. Read once at first `shared` access.
    /// Documented in README.md.
    public static let forceAccelerateEnvVar = "SWITCHCRAFT_FORCE_ACCELERATE"

    /// Bundle-resource basename for the placeholder shader source. Used
    /// by the runtime-compile fallback.
    private static let shaderResourceName = "MetalCoreShaders"

    /// Default subsystem for `os.Logger` instances scoped to this module.
    public static let logSubsystem = "com.switchcraft.core"

    /// Lazy process-wide singleton. `nil` when:
    ///
    ///   - `MTLCreateSystemDefaultDevice()` returns `nil` (no GPU), or
    ///   - `SWITCHCRAFT_FORCE_ACCELERATE` is set to a truthy value, or
    ///   - the Metal library can't be loaded (e.g. shader compile failed).
    ///
    /// `static let` guarantees the initializer runs exactly once across
    /// the process, satisfying the `MTLLibrary` "no concurrent
    /// initialization" constraint.
    public static let shared: MetalContext? = {
        if isForceAccelerateSet() {
            return nil
        }
        do {
            return try MetalContext()
        } catch {
            // Fallback policy: any Metal failure means we use the
            // Accelerate path. Logged at debug level so test/diagnostic
            // runs can confirm the fallback fired.
            let log = Logger(subsystem: MetalContext.logSubsystem, category: "Metal")
            log.debug("MetalContext init failed; falling back to Accelerate. Error: \(String(describing: error), privacy: .public)")
            return nil
        }
    }()

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary

    /// Pipeline cache keyed by Metal function name. Guarded by `cacheLock`.
    /// The cached `MTLComputePipelineState` values are themselves
    /// thread-safe, so callers can hold them past the lock release.
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let cacheLock = NSLock()

    /// Designated initializer.
    ///
    /// Exposed via `@_spi(SwitchcraftMetal) public` so the sibling
    /// `SwitchcraftMetal` target (and tests with the matching SPI
    /// import) can construct a context directly when necessary. Most
    /// consumers should still use `MetalContext.shared` to honor the
    /// single-shared-instance discipline — Metal context creation is
    /// non-trivial (issue #51 spec) and `static let shared` is what
    /// guarantees the `MTLLibrary` is initialised exactly once across
    /// the process.
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalContextError.noDeviceAvailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalContextError.commandQueueCreationFailed
        }
        self.device = device
        self.queue = queue
        self.library = try Self.loadLibrary(device: device)
    }

    /// Resolve (or build and cache) the pipeline state for a kernel by
    /// function name. Thread-safe via `cacheLock`.
    ///
    /// The lock is held across pipeline construction so concurrent
    /// callers asking for the same function can't each build their own
    /// state and race the writeback (which would leave one caller with
    /// an orphan state). Pipeline construction is a one-time cost per
    /// function name; subsequent lookups hit the dictionary read fast
    /// path. `MTLLibrary.makeFunction` is undocumented for thread
    /// safety, which is the second reason the lock spans the whole
    /// critical section (see ADR 015 §"Sendable policy").
    public func pipeline(for functionName: String) throws -> MTLComputePipelineState {
        cacheLock.lock()
        defer { cacheLock.unlock() }

        if let cached = pipelines[functionName] {
            return cached
        }
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalContextError.functionNotFound(functionName)
        }
        let state: MTLComputePipelineState
        do {
            state = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalContextError.pipelineCreationFailed(String(describing: error))
        }
        pipelines[functionName] = state
        return state
    }

    // MARK: - Private

    /// Load the Metal library. Tries the precompiled `default.metallib`
    /// first; falls back to runtime source compilation when SwiftPM
    /// has shipped the `.metal` file as raw source (see header comment).
    private static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }
        guard
            let url = Bundle.module.url(
                forResource: shaderResourceName,
                withExtension: "metal"
            ),
            let source = try? String(contentsOf: url, encoding: .utf8)
        else {
            throw MetalContextError.libraryLoadFailed(
                "\(shaderResourceName).metal not found in module bundle"
            )
        }
        do {
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalContextError.libraryLoadFailed(String(describing: error))
        }
    }

    /// Truthy-env-var check matching the existing
    /// `Tests/SwitchcraftMetalProtoTests/Support/BenchmarkInputs.swift`
    /// `MetalProtoGate` shape: any non-empty value other than `"0"` counts.
    private static func isForceAccelerateSet() -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[forceAccelerateEnvVar] else {
            return false
        }
        return !raw.isEmpty && raw != "0"
    }
}

#endif // canImport(Metal)
