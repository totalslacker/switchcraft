// SPDX-License-Identifier: Apache-2.0
//
// Test-side loader for the committed PyTorch FP32 reference embeddings
// (`Tests/Fixtures/xtr-base-en.embeddings.{json,bin}`). Used by the
// `T5MetalEmbedderTests` cosine-similarity parity gate (issue #64).
//
// This is a duplicate of `Tests/SwitchcraftTests/Support/CoreMLAsset.swift
// :CoreMLEmbeddingFixtures` per the issue #64 plan §"Key Decisions" —
// `SwitchcraftMetalTests` doesn't depend on `SwitchcraftTests` so the
// loader needs to live here too. If a third consumer arrives, extract a
// shared `MetalTestSupport` target both depend on.

import Foundation

enum CoreMLEmbeddingFixtures {
    static let indexResourceName = "xtr-base-en.embeddings"
    static let blobResourceName = "xtr-base-en.embeddings"

    /// JSON sidecar matching the conversion script's output:
    /// ```
    /// { "model": "google/xtr-base-en@<sha>", "dims": 128,
    ///   "windowSize": 512, "stride": 256, "minNorm": 1.0,
    ///   "fixtures": [
    ///     { "name": "...", "input": "...", "rows": <m>,
    ///       "byteOffset": <o>, "byteLength": <m * dims * 4> }
    ///   ] }
    /// ```
    struct Index: Decodable {
        let model: String
        let dims: Int
        let windowSize: Int
        let stride: Int
        let minNorm: Float
        let fixtures: [Fixture]

        struct Fixture: Decodable {
            let name: String
            let input: String
            let rows: Int
            let byteOffset: Int
            let byteLength: Int
        }
    }

    static var isAvailable: Bool {
        guard
            Bundle.module.url(
                forResource: indexResourceName,
                withExtension: "json",
                subdirectory: "Fixtures"
            ) != nil,
            Bundle.module.url(
                forResource: blobResourceName,
                withExtension: "bin",
                subdirectory: "Fixtures"
            ) != nil
        else { return false }
        return true
    }

    static func loadIndex() throws -> Index {
        guard
            let url = Bundle.module.url(
                forResource: indexResourceName,
                withExtension: "json",
                subdirectory: "Fixtures"
            )
        else {
            throw FixtureError.indexNotFound
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Index.self, from: data)
    }

    static func loadBlob() throws -> Data {
        guard
            let url = Bundle.module.url(
                forResource: blobResourceName,
                withExtension: "bin",
                subdirectory: "Fixtures"
            )
        else {
            throw FixtureError.blobNotFound
        }
        return try Data(contentsOf: url)
    }

    /// Slice rows for a single fixture out of the blob.
    static func rows(for fixture: Index.Fixture, blob: Data, dims: Int) -> [[Float]] {
        precondition(
            fixture.byteLength == fixture.rows * dims * MemoryLayout<Float>.size,
            "Fixture byte length inconsistent with row count and dims"
        )
        var rows: [[Float]] = []
        rows.reserveCapacity(fixture.rows)
        for r in 0..<fixture.rows {
            let start = fixture.byteOffset + r * dims * MemoryLayout<Float>.size
            let end = start + dims * MemoryLayout<Float>.size
            let slice = blob[start..<end]
            var row = [Float](repeating: 0, count: dims)
            _ = row.withUnsafeMutableBytes { dst in
                slice.copyBytes(to: dst)
            }
            rows.append(row)
        }
        return rows
    }

    enum FixtureError: Error, CustomStringConvertible {
        case indexNotFound
        case blobNotFound

        var description: String {
            switch self {
            case .indexNotFound: return "xtr-base-en.embeddings.json missing from test bundle"
            case .blobNotFound: return "xtr-base-en.embeddings.bin missing from test bundle"
            }
        }
    }
}

/// Resolves the committed `xtr-base-en.tokenizer.json` fixture used by
/// `T5MetalEmbedderTests`. Mirrors the loader pattern in
/// `SwitchcraftTests` but lives here because `SwitchcraftMetalTests`
/// doesn't depend on the consumer-side test target.
enum TokenizerFixture {
    /// Path to the committed `xtr-base-en.tokenizer.json` in the test
    /// bundle. Returns `nil` only if the file is missing from the
    /// fixtures (worktree wasn't checked out cleanly).
    static var url: URL? {
        Bundle.module.url(
            forResource: "xtr-base-en.tokenizer",
            withExtension: "json",
            subdirectory: "Fixtures"
        )
    }
}
