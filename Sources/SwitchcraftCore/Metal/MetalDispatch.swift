// SPDX-License-Identifier: Apache-2.0
//
// Transparent Metal-or-Accelerate dispatch helper. Search-path call
// sites express their work as a pair of equivalent closures — one that
// runs through Metal, one that runs on the existing Accelerate path —
// and this helper picks the Metal path when available, falling back
// silently on `nil` context, thrown errors, or command-buffer failures.
//
// Metal is treated as an optimization, not a correctness requirement.
// Failures are logged at debug level (so test/diagnostic runs can
// confirm the fallback fired) but never propagated to the caller. The
// fallback closure must produce results equivalent to the Metal path
// within the kernel's documented tolerance.
//
// Subsequent kernel sub-issues will adopt this helper at each call
// site (`SearchEngine.matmulQueryTimesRowMajorTranspose`, `KMeans`,
// residual scoring). The scaffolding sub-issue (#51) ships only the
// helper — no production call sites are wired up yet.
//
// See `adrs/015-metal-context-and-dispatch.md` for the rationale.

#if canImport(Metal)

import Foundation
import Metal
import os

/// Run `metal` with the shared `MetalContext` if one is available;
/// otherwise (no Metal, env var override, or `metal` throws) silently
/// run `fallback`.
///
/// `metal` may inspect the context, encode work onto a command buffer,
/// and call `commit()` + `waitUntilCompleted()`. Throwing or returning a
/// command-buffer error is acceptable — the helper catches and routes to
/// `fallback`.
///
/// Both closures must be pure with respect to their inputs (no observable
/// side effects beyond producing the result), since exactly one of them
/// runs.
func runMetalOrFallback<T>(
    metal: (MetalContext) throws -> T,
    fallback: () -> T
) -> T {
    guard let ctx = MetalContext.shared else {
        return fallback()
    }
    do {
        return try metal(ctx)
    } catch {
        let log = Logger(subsystem: MetalContext.logSubsystem, category: "MetalDispatch")
        log.debug("Metal dispatch threw; using Accelerate. Error: \(String(describing: error), privacy: .public)")
        return fallback()
    }
}

/// Synchronous command-buffer commit + wait, with a documented error
/// channel. Returns `nil` on success and a non-nil `Error` when the
/// command buffer reports a failure post-completion. Kernel sites should
/// treat any non-nil return as "fall back to Accelerate".
///
/// Centralised here so the same pattern doesn't need to be re-derived in
/// every kernel — a small but error-prone piece of boilerplate.
func commitAndWait(_ commandBuffer: MTLCommandBuffer) -> Error? {
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.error
}

#endif // canImport(Metal)
