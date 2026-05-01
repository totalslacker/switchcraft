// SPDX-License-Identifier: Apache-2.0
//
// Performance regression suite for the Phase 2 search-path Metal
// kernels (umbrella #50). This is the production-shaped sibling of
// `PerformanceTests`: same release-only gating, same `static let
// buildOnce` shared-fixture pattern, same percentile-reporting helpers,
// but at `MockEmbedder(dims: 128)` (Witchcraft's projection dimension)
// rather than the dims=32 ceiling-conscious shape ADR 012's existing
// suite uses. Both run.
//
// Corpus tuning
// -------------
//
//     5_000 docs × ~10 tokens × 128 dims × 4 bytes ≈ 25 MB
//
// at the per-doc float ledger, comfortably under the 300 MB peak-RSS
// ceiling from ADR 012 even after `DocumentRecord` / `ChunkRecord`
// overhead, the BM25 index, and the LSM generations produced by
// `IndexerConfig.production`. If a host hits the ceiling locally,
// reduce to 2_500 docs rather than tightening other tests.
//
// Sub-issue scope (#51 scaffolding)
// ---------------------------------
//
// This file ships with **no kernel test cases yet**. Sub-issues #2–#4
// of the umbrella (Q4 dequant + matmul, centroid similarity, MaxSim)
// each drop their kernel test in here with a concrete pass/fail floor
// (≥ 1.3× speedup vs Accelerate on the same corpus, plus a hard p95
// cap aligned with the umbrella's ≤ 22 ms search end-to-end target).
//
// What this file *does* validate today is the wiring: the suite
// builds the corpus once, the perf harness runs, percentile arithmetic
// returns sensible numbers, and the suite skips cleanly on hosts
// without Metal (or with `SWITCHCRAFT_FORCE_ACCELERATE` set).
//
// Run with:
//
//     swift test -c release --filter MetalPerformanceTests

#if canImport(Metal)

import Foundation
import Metal
import Testing
@testable import Switchcraft
@_spi(SwitchcraftMetal) @testable import SwitchcraftCore

/// `true` when compiled with optimisations (`swift test -c release`).
/// Mirrors the `assert`-trick used in `PerformanceTests.swift`. Defined
/// privately here so this file is self-contained — `PerformanceTests`'
/// copy is `private` and not visible from this module-level scope.
private let isReleaseBuildMetal: Bool = {
    var debug = false
    assert({ debug = true; return true }())
    return !debug
}()

@Suite("MetalPerformanceTests", .serialized,
       .enabled(if: isReleaseBuildMetal && MetalAvailability.isAvailable,
                "Release-only suite; also requires a Metal-capable host (and no SWITCHCRAFT_FORCE_ACCELERATE)"))
struct MetalPerformanceTests {

    // MARK: - Fixture

    fileprivate struct Fixture: Sendable {
        let store: SwitchcraftStore
        let storage: InMemoryStorage
    }

    /// Witchcraft's projection dimension. The headline difference vs
    /// `PerformanceTests` (dims=32). At dims=128 the in-memory float
    /// ledger is ~4× larger but still well inside the 300 MB ceiling
    /// (see header comment).
    fileprivate static let embedDims: Int = 128

    /// Number of documents in the shared corpus. Kept at the same
    /// 5,000 as `PerformanceTests` so the two suites are comparable
    /// modulo dims.
    fileprivate static let corpusSize: Int = 5_000

    fileprivate static let buildOnce: Task<Fixture, Error> = Task {
        try await buildCorpus(size: Self.corpusSize)
    }

    fileprivate static func buildCorpus(size: Int) async throws -> Fixture {
        let storage = InMemoryStorage()
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: Self.embedDims),
            config: .default
        )
        for i in 0..<size {
            try await store.add(id: "doc-\(i)", body: Self.body(i))
        }
        try await store.index()
        return Fixture(store: store, storage: storage)
    }

    fileprivate static func body(_ i: Int) -> String {
        "document \(i) about apples and oranges and grapes and pears"
    }

    fileprivate static let probeQuery: String = "apples"

    // MARK: - Tests

    /// Wiring smoke test: build the corpus, run search 50 times against
    /// the (still Accelerate-backed) production path, and report p50/p95.
    /// No floor — sub-issues #2–#4 add per-kernel floors when their
    /// kernels land. The point of this case is to confirm the dims=128
    /// suite wires together and stays under the RSS ceiling.
    @Test("baseline search wiring at dims=128 reports sensible p50/p95")
    func baselineSearchWiringReportsPercentiles() async throws {
        let fixture = try await Self.buildOnce.value
        let store = fixture.store

        // Warm-up: shader-compile cost on first kernel touch is the
        // canonical Metal trap. This warm-up will absorb it for the
        // future kernel cases; here it just exercises the same shape.
        _ = try await store.search(query: Self.probeQuery, topK: 10)

        let iterations = 50
        var samplesNs: [UInt64] = []
        samplesNs.reserveCapacity(iterations)

        let clock = ContinuousClock()
        for _ in 0..<iterations {
            let start = clock.now
            _ = try await store.search(query: Self.probeQuery, topK: 10)
            let elapsed = clock.now - start
            samplesNs.append(Self.nanoseconds(elapsed))
        }

        samplesNs.sort()
        let p50ns = samplesNs[Self.percentileIndex(0.50, count: iterations)]
        let p95ns = samplesNs[Self.percentileIndex(0.95, count: iterations)]

        // Sanity: percentiles are non-zero and p95 ≥ p50. No latency
        // floor here — that's a per-kernel concern.
        #expect(p50ns > 0, "p50 was 0 ns — the timer didn't fire")
        #expect(p95ns >= p50ns,
                "p95 (\(p95ns) ns) less than p50 (\(p50ns) ns)")
    }

    /// Wiring smoke test for the placeholder kernel itself: dispatch
    /// `identity_fp32` 50 times and report the per-call latency. This
    /// is what sub-issues #2–#4 will replace with real kernel calls and
    /// per-kernel floors. Today it just confirms the Metal dispatch
    /// path runs without errors at the same percentile-reporting shape
    /// the production tests will use.
    @Test("placeholder kernel dispatch wiring reports p50/p95")
    func placeholderKernelDispatchWiring() throws {
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        let pipeline = try ctx.pipeline(for: "identity_fp32")

        let count = 4096
        var input = [Float](repeating: 0, count: count)
        for i in 0..<count { input[i] = Float(i) * 0.001 }

        // Warm-up dispatch: absorbs first-touch shader-binding cost so
        // it doesn't pollute the p95 reading.
        _ = try Self.dispatchIdentity(
            ctx: ctx, pipeline: pipeline, input: input
        )

        let iterations = 50
        var samplesNs: [UInt64] = []
        samplesNs.reserveCapacity(iterations)
        let clock = ContinuousClock()
        for _ in 0..<iterations {
            let start = clock.now
            _ = try Self.dispatchIdentity(
                ctx: ctx, pipeline: pipeline, input: input
            )
            let elapsed = clock.now - start
            samplesNs.append(Self.nanoseconds(elapsed))
        }

        samplesNs.sort()
        let p50ns = samplesNs[Self.percentileIndex(0.50, count: iterations)]
        let p95ns = samplesNs[Self.percentileIndex(0.95, count: iterations)]

        #expect(p50ns > 0, "p50 was 0 ns — the timer didn't fire")
        #expect(p95ns >= p50ns,
                "p95 (\(p95ns) ns) less than p50 (\(p50ns) ns)")
    }

    // MARK: - Helpers

    private static func dispatchIdentity(
        ctx: MetalContext,
        pipeline: MTLComputePipelineState,
        input: [Float]
    ) throws -> [Float] {
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

    fileprivate static func nanoseconds(_ duration: ContinuousClock.Duration) -> UInt64 {
        let parts = duration.components
        let secNs = UInt64(max(parts.seconds, 0)) * 1_000_000_000
        let attoNs = UInt64(max(parts.attoseconds, 0)) / 1_000_000_000
        return secNs &+ attoNs
    }

    fileprivate static func percentileIndex(_ p: Double, count: Int) -> Int {
        precondition(count > 0, "percentileIndex requires count > 0")
        let raw = Int((p * Double(count)).rounded(.up)) - 1
        return min(max(raw, 0), count - 1)
    }
}

#endif // canImport(Metal)
