// SPDX-License-Identifier: Apache-2.0
//
// Bundle-registration smoke tests for `SwitchcraftMetal/Bundle.module`.
// Issue #61 introduced a second `.metal` file in this bundle
// (`RMSNorm.metal`, alongside the existing `Q4KMatMul.metal` from
// #60) and changed `MetalContext.register(bundle:resourceName:)`'s
// idempotency key from `bundleURL` alone to `(bundleURL, resourceName)`
// so multi-resource bundles work. These tests exercise the positive
// path on the host: both kernels resolve to working pipelines against
// a single shared `MetalContext`.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

@Suite("SwitchcraftMetal bundle registration",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct MetalShaderBundleTests {
    @Test("registerSwitchcraftMetalShaders loads every catalogued .metal file")
    func registerLoadsEveryShaderResource() throws {
        let ctx = try #require(MetalContext.shared)
        // Idempotent — running twice in this test process is fine.
        try registerSwitchcraftMetalShaders(with: ctx)

        // Every catalogued resource exposes at least one kernel that
        // must resolve. We probe both target kernels here so a future
        // accidental drop of a `.metal` file from
        // `switchcraftMetalShaderResourceNames` trips this test rather
        // than the kernel-specific suites (which would hide the bundle-
        // registration bug behind kernel test failures).
        _ = try ctx.pipeline(for: Q4KMatMulKernel.kernelFunctionName)
        _ = try ctx.pipeline(for: RMSNormKernel.kernelFunctionName)
    }

    @Test("Q4KMatMulKernel and RMSNormKernel coexist on a single MetalContext")
    func bothKernelsCoexist() throws {
        let ctx = try #require(MetalContext.shared)
        let q4 = try Q4KMatMulKernel(context: ctx)
        let rn = try RMSNormKernel(context: ctx)
        // Surface the wrapper-side function-name constants so a future
        // rename can't silently break `init(context:)`.
        _ = q4
        _ = rn
        #expect(Q4KMatMulKernel.kernelFunctionName == "kernel_mul_mm_q4_K_f32")
        #expect(RMSNormKernel.kernelFunctionName == "kernel_rms_norm_mul_f32")
    }
}

#endif // canImport(Metal)
