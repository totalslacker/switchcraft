// SPDX-License-Identifier: Apache-2.0
//
// One-shot helper that registers `SwitchcraftMetal`'s `Bundle.module`
// with the shared `MetalContext` so this target's MSL kernels are
// reachable via `MetalContext.pipeline(for:)`. Issue #60 (the first
// kernel landing in this target) is the first caller; subsequent
// kernel sub-issues reuse the same helper.
//
// Why a separate file
// -------------------
//
// Swift has no portable module-init hook, so registration happens
// lazily at the first kernel-wrapper init. The helper is idempotent
// (matching `MetalContext.register(bundle:resourceName:)`) — calling
// it from multiple kernel wrappers in the same process is safe and
// cheap. Keeping it in its own file makes the contract obvious to
// future kernel authors: call `registerSwitchcraftMetalShaders(...)`
// from your wrapper's `init`, and your `.metal` file lands in the
// shared pipeline cache.

#if canImport(Metal)

import Foundation
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// Resource basename of the `SwitchcraftMetal` MSL source. Single
/// `.metal` file at this stage; future kernels can either share this
/// file or add new ones (each new file gets its own basename and a
/// matching `register(bundle:resourceName:)` call).
///
/// Matches the file name (without extension) of the runtime-source-
/// compile fallback target — i.e. `Sources/SwitchcraftMetal/Shaders/Q4KMatMul.metal`.
@_spi(SwitchcraftMetal)
public let switchcraftMetalShaderResourceName = "Q4KMatMul"

/// Register `SwitchcraftMetal`'s `Bundle.module` with `context` so its
/// `.metal` shaders are reachable via `pipeline(for:)`. Idempotent.
@_spi(SwitchcraftMetal)
public func registerSwitchcraftMetalShaders(with context: MetalContext) throws {
    try context.register(
        bundle: Bundle.module,
        resourceName: switchcraftMetalShaderResourceName
    )
}

#endif // canImport(Metal)
