// SPDX-License-Identifier: Apache-2.0
//
// Compaction memory regression suite (issue #112 / ADR 027).
//
// # What this tests
//
// Before the fix in issue #112, cascade compaction of three input segments
// totalling ~307 MB allocated >100 GB of heap — a ~120× amplification
// driven by a single O(m × k) = O(m^1.5) scores matrix in
// KMeans.assignAll(). This suite exercises a cascade with ≥3 input
// segments and ≥600 000 total embeddings (dims=128) and asserts that
// peak physical-memory growth stays within bounds.
//
// # Memory bounds
//
// Hard gate (fails the build if violated):
//   peak phys_footprint delta < 32 GB (absolute OOM fence)
//
// Soft target (prints a warning but does not fail):
//   peak phys_footprint delta ≤ 4 × (ledger bytes) = 4 × (m × dims × 4)
//   This represents the expected k-means scratch + output buffer overhead.
//
// # How to run
//
//   SWITCHCRAFT_COMPACT_REGRESSION=1 swift test \
//     --filter CompactionMemoryRegression
//
// Requires ≥ 8 GB free RAM. Expected elapsed time: 30–90 seconds.
// macOS-only (uses task_vm_info.phys_footprint). On other platforms the
// test prints "[SKIP]" and returns without assertion.

import Foundation
import Testing
@testable import SwitchcraftCore

#if os(macOS)
import Darwin.Mach
#endif

private let regressionEnabled =
    ProcessInfo.processInfo.environment["SWITCHCRAFT_COMPACT_REGRESSION"] != nil

@Suite("CompactionMemoryRegression",
       .enabled(if: regressionEnabled,
                "Set SWITCHCRAFT_COMPACT_REGRESSION=1 to run memory regression suite"))
struct CompactionMemoryRegressionTests {

    // MARK: - Memory measurement

#if os(macOS)
    /// Returns current `phys_footprint` in bytes via task_vm_info.
    /// Returns 0 on failure (graceful fallback — the test will still exercise
    /// compaction correctness even if the OS query fails).
    static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return info.phys_footprint
    }
#else
    static func physFootprint() -> UInt64 { 0 }
#endif

    // MARK: - Corpus construction helpers

    /// Generate `count` L2-normalised random 128-dim embeddings using
    /// SplitMix64 (fast, reproducible). Returns a flat `[Float]` row-major
    /// matrix suitable for `Indexer.add(chunkID:embeddings:dims:)`.
    static func randomEmbeddings(count: Int, dims: Int, rng: inout SplitMix64) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(count * dims)
        for _ in 0..<count {
            var v = [Float](repeating: 0, count: dims)
            var sumSq: Float = 0
            for j in 0..<dims {
                let bits = UInt32(truncatingIfNeeded: rng.next())
                let f = Float(Int32(bitPattern: bits)) / Float(Int32.max)
                v[j] = f
                sumSq += f * f
            }
            let norm = sumSq > 0 ? 1 / sumSq.squareRoot() : 1
            for j in 0..<dims { v[j] *= norm }
            out.append(contentsOf: v)
        }
        return out
    }

    // MARK: - Regression test

    /// Cascade compaction with ≥3 input segments and ≥600 000 total
    /// embeddings must complete within the memory bounds above.
    ///
    /// Corpus construction:
    ///   Config: l0Capacity=100_000, lsmFanout=2, dims=128.
    ///   Level capacities: L0=200K, L1=400K, L2=800K.
    ///   Six flushes of 100K embeddings each trigger the following cascades:
    ///     Flush 3 → L1 segment (2 input segs, 300K total).
    ///     Flush 6 → L2 segment (3 input segs: L0=200K, L1=300K, pending=100K).
    ///   The flush-6 cascade is the one this test exercises: 3 input segments,
    ///   600K total embeddings, k ≈ 16×√600K ≈ 12 395.
    @Test("cascade compaction (3 input segs, 600K embeddings) stays within memory bounds")
    func cascadeCompactionMemoryBounds() async throws {
#if !os(macOS)
        print("[SKIP] CompactionMemoryRegressionTests requires macOS (task_vm_info).")
        return
#endif

        let dims = 128
        let batchSize = 100_000
        let batchCount = 6

        // Config: l0Capacity=100K, lsmFanout=2 → cascade to L2 on flush 6.
        let config = IndexerConfig(
            l0Capacity: batchSize,
            lsmFanout: 2,
            kmeansIterations: 5,
            kCoefficient: 16.0,
            seed: 42
        )
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: config)

        var rng = SplitMix64(seed: 0x5CA1AB1E_CAFE_F00D)

        // Build and flush the first 5 batches (these create L0 and L1 segments).
        // Flush 1-2 → L0 (merged). Flush 3 → L1 (2 input segs). Flush 4-5 → L0.
        print("[CompactionMemoryRegression] Building corpus: \(batchCount) × \(batchSize) embeddings …")
        for batch in 0..<(batchCount - 1) {
            let embeddings = Self.randomEmbeddings(count: batchSize, dims: dims, rng: &rng)
            let chunkID = Int64(batch * batchSize + 1)
            try await indexer.add(chunkID: chunkID, embeddings: embeddings, dims: dims)
            try await indexer.flush()
        }

        // Verify we have both L0 and L1 segments before the critical flush.
        let gensBeforeFlush6 = try await storage.generations()
        let levelsBeforeFlush6 = Set(gensBeforeFlush6.map { $0.level })
        #expect(
            levelsBeforeFlush6.contains(0) && levelsBeforeFlush6.contains(1),
            "Expected both L0 and L1 segments before flush 6, got levels: \(levelsBeforeFlush6.sorted())"
        )

        // Add the 6th batch (the cascade trigger). Don't flush yet.
        let lastBatch = Self.randomEmbeddings(count: batchSize, dims: dims, rng: &rng)
        let lastChunkID = Int64((batchCount - 1) * batchSize + 1)
        try await indexer.add(chunkID: lastChunkID, embeddings: lastBatch, dims: dims)

        // Start memory sampling before the cascade flush.
        let baseline = Self.physFootprint()
        let tracker = PeakTracker(initial: baseline)

        // Background sampler: polls phys_footprint every 250 ms while flush runs.
        let sampler = Task {
            while await !tracker.isDone {
                let fp = Self.physFootprint()
                await tracker.update(fp)
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
        }

        // Run the cascade flush (flush 6 — the fix must prevent OOM here).
        print("[CompactionMemoryRegression] Starting cascade flush (3 input segments) …")
        do {
            try await indexer.flush()
        } catch {
            await tracker.markDone()
            sampler.cancel()
            Issue.record("Cascade compaction threw: \(error) — expected clean completion")
            return
        }

        await tracker.markDone()
        sampler.cancel()

        // Also sample immediately after flush (may catch late-release pages).
        let postFlush = Self.physFootprint()
        await tracker.update(postFlush)

        // Confirm a single L2 generation now exists.
        let gensAfterFlush = try await storage.generations()
        #expect(
            gensAfterFlush.count == 1 && gensAfterFlush[0].level == 2,
            "Expected single L2 generation after cascade; got \(gensAfterFlush.map { "L\($0.level)=\($0.numEmbeddings)" })"
        )

        let peak = await tracker.peak
        let peakDeltaBytes = peak > baseline ? peak - baseline : 0
        let totalEmbeddings = batchSize * batchCount
        let ledgerBytes = UInt64(totalEmbeddings) * UInt64(dims) * 4

        print("[CompactionMemoryRegression] baseline=\(baseline >> 20) MB  peak=\(peak >> 20) MB  delta=\(peakDeltaBytes >> 20) MB  ledger=\(ledgerBytes >> 20) MB")

        // Hard gate: peak delta must be < 32 GB (OOM fence).
        let hardGateBytes: UInt64 = 32 * 1024 * 1024 * 1024
        #expect(
            peakDeltaBytes < hardGateBytes,
            "HARD GATE FAILED: peak phys_footprint delta \(peakDeltaBytes >> 20) MB ≥ 32 GB limit"
        )

        // Soft target: peak delta ≤ 4 × ledger bytes. Warn but don't fail.
        let softTargetBytes = ledgerBytes * 4
        if peakDeltaBytes > softTargetBytes {
            print("""
            [CompactionMemoryRegression] WARNING: soft target missed. \
            peak_delta=\(peakDeltaBytes >> 20) MB > 4×ledger=\(softTargetBytes >> 20) MB. \
            The fix reduces worst-case amplification but the soft target may not be \
            achievable on all configurations. See ADR 027 §soft-target.
            """)
        } else {
            print("[CompactionMemoryRegression] Soft target passed: \(peakDeltaBytes >> 20) MB ≤ \(softTargetBytes >> 20) MB")
        }
    }
}

// MARK: - Actor helpers

/// Actor-isolated mutable cell for accumulating the peak footprint value
/// across concurrent reads/writes from the sampler Task.
private actor PeakTracker {
    var peak: UInt64
    var done: Bool = false
    init(initial: UInt64) { peak = initial }
    func update(_ candidate: UInt64) { if candidate > peak { peak = candidate } }
    func markDone() { done = true }
    var isDone: Bool { done }
}
