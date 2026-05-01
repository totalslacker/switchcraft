// SPDX-License-Identifier: Apache-2.0
//
// Witchcraft per-token embeddings reference loader for the SwitchcraftMetal
// cross-stack parity test (`CrossStackEmbeddingParityMetalTests`). Issue #65.
//
// **Duplicate of `Tests/SwitchcraftTests/Support/ReferenceFixtures.swift`'s
// `ReferenceEmbeddingsFixture`** per the issue #65 plan §"Key Decisions" —
// `SwitchcraftMetalTests` does not depend on `SwitchcraftTests`, so the
// loader needs to live here too. If a third consumer arrives, extract a
// shared `MetalTestSupport` target both depend on.
//
// Provenance, regeneration policy, and per-fixture tolerances are
// documented in `adrs/013-reference-fixture-provenance.md`. The blob is
// produced via `scripts/witchcraft-fixture-export.patch` against a pinned
// Witchcraft commit and is **not committed** to the repository — fresh
// checkouts must regenerate it locally before the cross-stack parity
// suite can run. The `isAvailable` gate keeps the suite skipped cleanly
// when the fixture is absent.

import Foundation

enum ReferenceEmbeddingsFixtureMetal {
    static let indexResourceName = "reference_embeddings"
    static let blobResourceName = "reference_embeddings"

    struct Index: Decodable {
        let dims: Int
        let fixtures: [Fixture]
        let provenance: Provenance

        struct Fixture: Decodable {
            let name: String
            let input: String
            let rows: Int
            let byteOffset: Int
            let byteLength: Int
        }

        struct Provenance: Decodable {
            let witchcraftCommit: String
            let generatedAt: String?
            let source: String?
            let computeUnits: String?
        }
    }

    static var isAvailable: Bool {
        Bundle.module.url(
            forResource: indexResourceName,
            withExtension: "json",
            subdirectory: "Fixtures"
        ) != nil
            && Bundle.module.url(
                forResource: blobResourceName,
                withExtension: "bin",
                subdirectory: "Fixtures"
            ) != nil
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

    static func rows(for fixture: Index.Fixture, blob: Data, dims: Int) -> [[Float]] {
        precondition(
            fixture.byteLength == fixture.rows * dims * MemoryLayout<Float>.size,
            "reference_embeddings: byte length inconsistent with rows × dims"
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
            case .indexNotFound:
                return "reference_embeddings.json missing from test bundle — regenerate via scripts/witchcraft-fixture-export.patch (ADR 013)"
            case .blobNotFound:
                return "reference_embeddings.bin missing from test bundle — regenerate via scripts/witchcraft-fixture-export.patch (ADR 013)"
            }
        }
    }
}
