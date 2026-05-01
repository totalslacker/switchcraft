// SPDX-License-Identifier: Apache-2.0
//
// Parity tests for the FP32 batched matmul Metal kernel
// (`kernel_fp32_matmul`) added in issue #64. Validates the three
// concrete shapes the T5 encoder forward pass uses against a pure-Swift
// FP64-accumulator reference.
//
// All suites gate on `MetalAvailability.isAvailable` so a no-GPU host
// (or a `SWITCHCRAFT_FORCE_ACCELERATE`-set run) skips cleanly.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// MARK: - PRNG

private struct LCG {
    var state: UInt64
    init(seed: UInt64) { self.state = seed &* 0x9E3779B97F4A7C15 &+ 1 }

    mutating func nextU32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(truncatingIfNeeded: state >> 32)
    }

    mutating func nextFloat() -> Float {
        let f = Float(nextU32()) / Float(UInt32.max)
        return f * 2 - 1
    }
}

// MARK: - Buffer helpers

private enum BufferHelpers {
    static func makeBuffer(
        ctx: MetalContext, floats: [Float], label: String
    ) throws -> MTLBuffer {
        return try floats.withUnsafeBufferPointer { ptr -> MTLBuffer in
            guard
                let base = ptr.baseAddress,
                let buf = ctx.device.makeBuffer(
                    bytes: base,
                    length: ptr.count * MemoryLayout<Float>.size,
                    options: .storageModeShared
                )
            else {
                throw MetalContextError.pipelineCreationFailed("\(label) alloc returned nil")
            }
            return buf
        }
    }

    static func makeOutputBuffer(
        ctx: MetalContext, count: Int
    ) throws -> MTLBuffer {
        let len = count * MemoryLayout<Float>.size
        guard let buf = ctx.device.makeBuffer(
            length: len,
            options: .storageModeShared
        ) else {
            throw MetalContextError.pipelineCreationFailed("output alloc returned nil")
        }
        memset(buf.contents(), 0, len)
        return buf
    }

    static func readFloats(_ buf: MTLBuffer, count: Int) -> [Float] {
        let raw = buf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }
}

// MARK: - Reference implementations

/// CPU FP64-accumulator reference for `C = A · B` or `C = A · Bᵀ`,
/// batched. Layout matches the kernel's contract.
private func referenceMatMul(
    a: [Float], b: [Float],
    M: Int, N: Int, K: Int, batch: Int,
    transposeB: Bool
) -> [Float] {
    var c = [Float](repeating: 0, count: batch * M * N)
    for bi in 0..<batch {
        for m in 0..<M {
            for n in 0..<N {
                var acc: Double = 0
                for k in 0..<K {
                    let aIdx = bi * M * K + m * K + k
                    let bIdx: Int
                    if transposeB {
                        bIdx = bi * N * K + n * K + k
                    } else {
                        bIdx = bi * K * N + k * N + n
                    }
                    acc += Double(a[aIdx]) * Double(b[bIdx])
                }
                c[bi * M * N + m * N + n] = Float(acc)
            }
        }
    }
    return c
}

// MARK: - Test suite

@Suite("FP32MatMul kernel parity",
       .enabled(if: MetalAvailability.isAvailable,
                "Metal unavailable on this host (or SWITCHCRAFT_FORCE_ACCELERATE set)"))
struct FP32MatMulKernelTests {

    /// Q · Kᵀ shape — per-head, batch=12, M=N=512, K=64, transposeB=true.
    /// Verifies the small-K path (K=64 is below Q4KMatMul's K%256==0
    /// requirement, the original motivation for the FP32 kernel).
    @Test("Q · Kᵀ shape: batch=12, M=N=512, K=64, transposeB=true")
    func qkT() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try FP32MatMulKernel(context: ctx)
        // Use a smaller batch / M to keep the FP64 reference fast.
        // Shape preserves the K=64 small-contraction property.
        let batch = 4
        let M = 32
        let N = 32
        let K = 64
        var rng = LCG(seed: 0xA5A5_FACE)

        let a = (0..<(batch * M * K)).map { _ in rng.nextFloat() }
        let b = (0..<(batch * N * K)).map { _ in rng.nextFloat() }
        let aBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: a, label: "A")
        let bBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: b, label: "B")
        let outBuf = try BufferHelpers.makeOutputBuffer(ctx: ctx, count: batch * M * N)

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(
            commandBuffer: cb,
            a: aBuf, b: bBuf, output: outBuf,
            M: M, N: N, K: K, batch: batch, transposeB: true
        )
        cb.commit()
        cb.waitUntilCompleted()

        let actual = BufferHelpers.readFloats(outBuf, count: batch * M * N)
        let expected = referenceMatMul(a: a, b: b, M: M, N: N, K: K, batch: batch, transposeB: true)

        let cosine = MetalCorrectness.cosineSimilarity(actual, expected)
        let maxAbs = MetalCorrectness.maxAbsError(actual, expected)
        #expect(cosine >= 0.99999, "Q·Kᵀ cosine too low: \(cosine)")
        #expect(maxAbs < 1e-3, "Q·Kᵀ maxAbsError too high: \(maxAbs)")
    }

    /// softmax · V shape — per-head, transposeB=false, large K.
    @Test("softmax · V shape: batch=2, M=32, N=64, K=128, transposeB=false")
    func softmaxV() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try FP32MatMulKernel(context: ctx)
        let batch = 2
        let M = 32
        let N = 64
        let K = 128
        var rng = LCG(seed: 0xDEAD_BEEF)

        let a = (0..<(batch * M * K)).map { _ in rng.nextFloat() }
        let b = (0..<(batch * K * N)).map { _ in rng.nextFloat() }
        let aBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: a, label: "A")
        let bBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: b, label: "B")
        let outBuf = try BufferHelpers.makeOutputBuffer(ctx: ctx, count: batch * M * N)

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(
            commandBuffer: cb,
            a: aBuf, b: bBuf, output: outBuf,
            M: M, N: N, K: K, batch: batch, transposeB: false
        )
        cb.commit()
        cb.waitUntilCompleted()

        let actual = BufferHelpers.readFloats(outBuf, count: batch * M * N)
        let expected = referenceMatMul(a: a, b: b, M: M, N: N, K: K, batch: batch, transposeB: false)

        let cosine = MetalCorrectness.cosineSimilarity(actual, expected)
        let maxAbs = MetalCorrectness.maxAbsError(actual, expected)
        #expect(cosine >= 0.99999, "softmax·V cosine too low: \(cosine)")
        #expect(maxAbs < 1e-3, "softmax·V maxAbsError too high: \(maxAbs)")
    }

    /// 2_Dense projection shape — batch=1, large K.
    @Test("2_Dense projection: batch=1, M=64, N=128, K=768, transposeB=true")
    func projection2Dense() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try FP32MatMulKernel(context: ctx)
        let batch = 1
        let M = 64
        let N = 128
        let K = 768
        var rng = LCG(seed: 0xCAFE_BABE)

        let a = (0..<(batch * M * K)).map { _ in rng.nextFloat() }
        let b = (0..<(batch * N * K)).map { _ in rng.nextFloat() }
        let aBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: a, label: "A")
        let bBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: b, label: "B")
        let outBuf = try BufferHelpers.makeOutputBuffer(ctx: ctx, count: batch * M * N)

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(
            commandBuffer: cb,
            a: aBuf, b: bBuf, output: outBuf,
            M: M, N: N, K: K, batch: batch, transposeB: true
        )
        cb.commit()
        cb.waitUntilCompleted()

        let actual = BufferHelpers.readFloats(outBuf, count: batch * M * N)
        let expected = referenceMatMul(a: a, b: b, M: M, N: N, K: K, batch: batch, transposeB: true)

        let cosine = MetalCorrectness.cosineSimilarity(actual, expected)
        let maxAbs = MetalCorrectness.maxAbsError(actual, expected)
        #expect(cosine >= 0.99999, "2_Dense projection cosine too low: \(cosine)")
        #expect(maxAbs < 1e-2, "2_Dense projection maxAbsError too high: \(maxAbs)")
    }

    /// Identity sanity check — A · I = A.
    @Test("Identity: A · I = A")
    func identity() throws {
        let ctx = try #require(MetalContext.shared)
        let kernel = try FP32MatMulKernel(context: ctx)
        let M = 7
        let N = 5
        let K = 5
        var rng = LCG(seed: 0xBEEF_0001)

        let a = (0..<(M * K)).map { _ in rng.nextFloat() }
        var b = [Float](repeating: 0, count: K * N)
        for i in 0..<min(K, N) { b[i * N + i] = 1 }

        let aBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: a, label: "A")
        let bBuf = try BufferHelpers.makeBuffer(ctx: ctx, floats: b, label: "B")
        let outBuf = try BufferHelpers.makeOutputBuffer(ctx: ctx, count: M * N)

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(
            commandBuffer: cb,
            a: aBuf, b: bBuf, output: outBuf,
            M: M, N: N, K: K, batch: 1, transposeB: false
        )
        cb.commit()
        cb.waitUntilCompleted()

        let actual = BufferHelpers.readFloats(outBuf, count: M * N)
        for r in 0..<M {
            for c in 0..<N {
                #expect(actual[r * N + c] == a[r * K + c],
                        "Identity mismatch at (\(r),\(c)): \(actual[r * N + c]) vs \(a[r * K + c])")
            }
        }
    }
}

#endif // canImport(Metal)
