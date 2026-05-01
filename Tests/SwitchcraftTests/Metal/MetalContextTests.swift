// SPDX-License-Identifier: Apache-2.0
//
// End-to-end wiring smoke tests for the search-path Metal scaffolding
// (issue #51). Validates that:
//
//   1. `MetalContext.shared` resolves to a usable device + queue +
//      library on a Metal-capable host.
//   2. The placeholder `identity_fp32` shader compiles, loads, and
//      dispatches successfully — i.e. the SwiftPM Metal build pipeline
//      (whether `.process(...)` produced a `.metallib` or shipped raw
//      source for runtime compilation) is intact end-to-end.
//   3. The pipeline-state cache returns the same object on repeat
//      lookups (lock-guarded read-or-build).
//
// These tests deliberately avoid testing performance — that lives in
// `Tests/SwitchcraftTests/Performance/MetalPerformanceTests.swift`.
// The shared-instance `MetalContext.shared` is read-only here; we
// don't need a force-Accelerate test in this file because the env var
// is sniffed once at first `shared` access and would interfere with
// any other test reading `shared` in the same process. Manual
// verification of the env var is documented in the PR description.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) @testable import SwitchcraftCore

@Suite("MetalContext smoke tests",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE is set)"))
struct MetalContextTests {
    @Test("shared instance resolves to a device + queue + library")
    func sharedInstanceResolves() throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        // `device` and `queue` are non-optional value types in the
        // context API, so reaching them confirms they constructed.
        _ = ctx.device
        _ = ctx.queue
        _ = ctx.library
    }

    @Test("identity_fp32 dispatch returns input unchanged")
    func identityDispatchEndToEnd() throws {
        let ctx = try #require(MetalContext.shared)

        let input: [Float] = (0..<256).map { Float($0) * 0.5 - 32.0 }
        let output = try runIdentityKernel(ctx: ctx, input: input)

        #expect(output.count == input.count)
        // The placeholder kernel is a bit-exact copy; any drift is a
        // wiring bug, not a precision issue.
        #expect(output == input,
                "identity_fp32 produced different output than input")
    }

    @Test("pipeline cache returns the same state on repeat lookups")
    func pipelineCacheReturnsSameStateOnRepeatLookup() throws {
        let ctx = try #require(MetalContext.shared)
        let first = try ctx.pipeline(for: "identity_fp32")
        let second = try ctx.pipeline(for: "identity_fp32")
        #expect(first === second,
                "pipeline cache returned a fresh state on the second lookup")
    }

    @Test("unknown function name throws functionNotFound")
    func unknownFunctionNameThrows() throws {
        let ctx = try #require(MetalContext.shared)
        #expect(throws: MetalContextError.self) {
            _ = try ctx.pipeline(for: "kernel_that_does_not_exist")
        }
    }

    @Test("register(bundle:resourceName:) error path leaves context state intact")
    func registerBundleErrorPathLeavesContextIntact() throws {
        // Use a fresh context so the registration failure can't leak
        // into the shared singleton.
        let ctx = try MetalContext()

        // SwitchcraftTests' own `Bundle.module` has no `.metal`
        // resource, so both library-load paths fail and `register`
        // throws `libraryLoadFailed`. The registration must NOT
        // mutate the libraries list on the failure path (no
        // half-registered state); the previously-resolvable default-
        // bundle kernels must continue to resolve.
        #expect(throws: MetalContextError.self) {
            try ctx.register(
                bundle: Bundle.module,
                resourceName: "no_such_shader_file"
            )
        }

        // Default-library kernels still resolve after the failure.
        let pipeline = try ctx.pipeline(for: "identity_fp32")
        #expect(pipeline.threadExecutionWidth >= 1)

        // The failed registration didn't pollute the registered-
        // bundle set: a second attempt with the same bundle still
        // throws (i.e. it wasn't silently marked registered).
        #expect(throws: MetalContextError.self) {
            try ctx.register(
                bundle: Bundle.module,
                resourceName: "no_such_shader_file"
            )
        }

        // Pipeline cache hits on repeat lookup (same `===` instance).
        let again = try ctx.pipeline(for: "identity_fp32")
        #expect(pipeline === again,
                "pipeline cache returned a fresh state on the second lookup after registration failure")
    }

    // MARK: - Helpers

    /// Run the placeholder `identity_fp32` shader synchronously. Kept
    /// inline here (rather than promoted into `MetalKernelTestSupport`)
    /// because subsequent kernel sub-issues will add their own dispatch
    /// helpers and there is no shared shape to extract yet.
    private func runIdentityKernel(ctx: MetalContext, input: [Float]) throws -> [Float] {
        let pipeline = try ctx.pipeline(for: "identity_fp32")
        let count = input.count

        let inBuf: MTLBuffer = try input.withUnsafeBufferPointer { ptr in
            guard
                let base = ptr.baseAddress,
                let buf = ctx.device.makeBuffer(
                    bytes: base,
                    length: count * MemoryLayout<Float>.size,
                    options: .storageModeShared
                )
            else {
                throw MetalContextError.pipelineCreationFailed("input buffer alloc returned nil")
            }
            return buf
        }
        guard let outBuf = ctx.device.makeBuffer(
            length: count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw MetalContextError.pipelineCreationFailed("output buffer alloc returned nil")
        }

        guard
            let cmd = ctx.queue.makeCommandBuffer(),
            let enc = cmd.makeComputeCommandEncoder()
        else {
            throw MetalContextError.pipelineCreationFailed(
                "makeCommandBuffer/Encoder returned nil"
            )
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var c = UInt32(count)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 2)

        let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (count + 63) / 64,
            height: 1,
            depth: 1
        )
        enc.dispatchThreadgroups(
            threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw MetalContextError.pipelineCreationFailed("command buffer error: \(err)")
        }

        let raw = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }
}

#endif // canImport(Metal)
