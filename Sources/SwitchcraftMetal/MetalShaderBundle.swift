// SPDX-License-Identifier: Apache-2.0
//
// One-shot helper that registers `SwitchcraftMetal`'s `Bundle.module`
// with the shared `MetalContext` so this target's MSL kernels are
// reachable via `MetalContext.pipeline(for:)`. Issue #60 (the first
// kernel landing in this target) added this helper for `Q4KMatMul`;
// issue #61 generalises it to handle multiple `.metal` source files
// in the same bundle.
//
// Why a separate file
// -------------------
//
// Swift has no portable module-init hook, so registration happens
// lazily at the first kernel-wrapper init. The helper is idempotent
// (`MetalContext.register(bundle:resourceName:)` keys on
// `(bundleURL, resourceName)`) — calling it from multiple kernel
// wrappers in the same process is safe and cheap. Keeping it in its
// own file makes the contract obvious to future kernel authors: add
// your `.metal` basename to `switchcraftMetalShaderResourceNames`,
// call `registerSwitchcraftMetalShaders(...)` from your wrapper's
// `init`, and your kernel lands in the shared pipeline cache.

#if canImport(Metal)

import Foundation
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// Resource basenames of the `SwitchcraftMetal` MSL sources that ship
/// in `Bundle.module`. Each entry must match the file name (without
/// extension) of a `.metal` file under
/// `Sources/SwitchcraftMetal/Shaders/`. The runtime-source-compile
/// fallback path consults these names one at a time; each name produces
/// its own `MTLLibrary` registered with the shared `MetalContext`.
///
/// New kernel sub-issues append their `.metal` basename here.
@_spi(SwitchcraftMetal)
public let switchcraftMetalShaderResourceNames: [String] = [
    "Q4KMatMul",   // issue #60
    "RMSNorm",     // issue #61
    "Softmax",     // issue #62
    "ResidualAdd", // issue #63
    "GatedGELU",   // issue #63
    "L2Norm",      // issue #63
]

/// Register `SwitchcraftMetal`'s `Bundle.module` with `context` so its
/// `.metal` shaders are reachable via `pipeline(for:)`. Idempotent —
/// each `(bundle, resourceName)` pair registers exactly once.
@_spi(SwitchcraftMetal)
public func registerSwitchcraftMetalShaders(with context: MetalContext) throws {
    for resourceName in switchcraftMetalShaderResourceNames {
        try context.register(
            bundle: Bundle.module,
            resourceName: resourceName
        )
    }
}

#endif // canImport(Metal)
