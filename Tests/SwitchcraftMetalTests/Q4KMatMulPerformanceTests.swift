// SPDX-License-Identifier: Apache-2.0
//
// Informational performance benchmark for `Q4KMatMulKernel` on the
// `ffn_wi_1` shape (M=512, K=768, N=3072) — the largest of the T5-base
// Q4_K matmuls and a representative hot path. Issue #60 §"Performance
// tests (informational)".
//
// No latency floor / pass-fail gate — that lands with the umbrella's
// perf-gate sub-issue (TBD per issue #57). Today this case just
// reports p50/p95 per ADR 012 so a future ratchet has a baseline.
//
// Release-only: `swift test -c release --filter Q4KMatMulPerformanceTests`.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

@Suite("Q4KMatMul performance",
       .serialized,
       .enabled(if: isReleaseBuildMetal && MetalAvailability.isAvailable,
                "Release-only suite; also requires a Metal-capable host (and no SWITCHCRAFT_FORCE_ACCELERATE)"))
struct Q4KMatMulPerformanceTests {

    @Test("ffn_wi_1 (M=512, K=768, N=3072) reports p50/p95")
    func ffnWi1Percentiles() throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        let kernel = try Q4KMatMulKernel(context: ctx)

        let M = 512, K = 768, N = 3072
        let blocksPerRow = K / 256

        // Build a dummy weight buffer of the right size — for a perf
        // benchmark the actual values don't matter, only the dispatch
        // shape does. Use a single zero super-block replicated across
        // rows so build cost stays small.
        let zeroBlock = Q4KDecodeTests.makeBlock(
            d: 0, dmin: 0,
            subScales: [UInt8](repeating: 0, count: 8),
            subMins: [UInt8](repeating: 0, count: 8),
            weightsLow: [UInt8](repeating: 0, count: 128),
            weightsHigh: [UInt8](repeating: 0, count: 128)
        )
        var weightBytes = Data()
        weightBytes.reserveCapacity(N * blocksPerRow * 144)
        for _ in 0..<(N * blocksPerRow) { weightBytes.append(zeroBlock) }

        let weightBuf: MTLBuffer = try #require(
            weightBytes.withUnsafeBytes { raw in
                raw.baseAddress.flatMap {
                    ctx.device.makeBuffer(
                        bytes: $0,
                        length: raw.count,
                        options: .storageModeShared
                    )
                }
            },
            "Failed to allocate weight buffer for ffn_wi_1 perf test"
        )

        let activations = [Float](repeating: 0, count: M * K)
        let actBuf: MTLBuffer = try #require(
            activations.withUnsafeBufferPointer { ptr in
                ptr.baseAddress.flatMap {
                    ctx.device.makeBuffer(
                        bytes: $0,
                        length: ptr.count * MemoryLayout<Float>.size,
                        options: .storageModeShared
                    )
                }
            },
            "Failed to allocate activation buffer for ffn_wi_1 perf test"
        )
        let outBuf: MTLBuffer = try #require(
            ctx.device.makeBuffer(
                length: M * N * MemoryLayout<Float>.size,
                options: .storageModeShared
            ),
            "Failed to allocate output buffer for ffn_wi_1 perf test"
        )

        // Warm-up: shader-binding cost on first dispatch is the
        // canonical Metal trap. One throwaway dispatch absorbs it.
        do {
            let cmd = try #require(
                ctx.queue.makeCommandBuffer(),
                "makeCommandBuffer returned nil during warm-up"
            )
            kernel.encode(
                commandBuffer: cmd,
                weight: weightBuf, weightShape: (N, K),
                input: actBuf, output: outBuf, M: M
            )
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let iterations = 50
        var samplesNs: [UInt64] = []
        samplesNs.reserveCapacity(iterations)
        let clock = ContinuousClock()
        for _ in 0..<iterations {
            let start = clock.now
            let cmd = try #require(
                ctx.queue.makeCommandBuffer(),
                "makeCommandBuffer returned nil mid-benchmark"
            )
            kernel.encode(
                commandBuffer: cmd,
                weight: weightBuf, weightShape: (N, K),
                input: actBuf, output: outBuf, M: M
            )
            cmd.commit()
            cmd.waitUntilCompleted()
            let elapsed = clock.now - start
            samplesNs.append(MetalPerf.nanoseconds(elapsed))
        }

        samplesNs.sort()
        let p50ns = samplesNs[MetalPerf.percentileIndex(0.50, count: iterations)]
        let p95ns = samplesNs[MetalPerf.percentileIndex(0.95, count: iterations)]
        let p50us = Double(p50ns) / 1_000.0
        let p95us = Double(p95ns) / 1_000.0
        print("[Q4KMatMul perf ffn_wi_1 M=512 K=768 N=3072] p50=\(p50us) µs, p95=\(p95us) µs")

        #expect(p50ns > 0, "p50 was 0 ns — the timer didn't fire")
        #expect(p95ns >= p50ns,
                "p95 (\(p95ns) ns) less than p50 (\(p50ns) ns)")
    }
}

#endif // canImport(Metal)
