// SPDX-License-Identifier: Apache-2.0
//
// Per-layer encoder hidden-state fixture loader for Metal bisection tests.
// Issue #75 — T5MetalEmbedder end-to-end parity gate.
//
// Provides access to per-layer PyTorch FP32 hidden states produced by
// `scripts/dump-pytorch-hidden-states.py`. These allow in-encoder bisection:
// compare Metal hidden states layer-by-layer against the PyTorch reference to
// isolate which encoder layer (or sub-operation) first diverges.
//
// Files: Tests/Fixtures/hidden_states.{bin,json}
// Generation: `python3 scripts/dump-pytorch-hidden-states.py --out Tests/Fixtures`
// Schema: flat JSON array of FixtureEntry, companion blob of raw FP32.
//
// When the fixture files are absent the loader's `isAvailable` gate returns false
// and any test that depends on these fixtures must skip cleanly.

import Foundation

enum HiddenStatesFixtureMetal {
    static let indexResourceName = "hidden_states"
    static let blobResourceName  = "hidden_states"

    struct FixtureEntry: Decodable {
        let name: String        // e.g. "short_layer_0"
        let byteOffset: Int
        let byteLength: Int
        let rows: Int           // number of real tokens
        let cols: Int           // always 768 (dModel)
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

    static func loadIndex() throws -> [FixtureEntry] {
        guard
            let url = Bundle.module.url(
                forResource: indexResourceName,
                withExtension: "json",
                subdirectory: "Fixtures"
            )
        else { throw FixtureError.indexNotFound }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([FixtureEntry].self, from: data)
    }

    static func loadBlob() throws -> Data {
        guard
            let url = Bundle.module.url(
                forResource: blobResourceName,
                withExtension: "bin",
                subdirectory: "Fixtures"
            )
        else { throw FixtureError.blobNotFound }
        return try Data(contentsOf: url)
    }

    /// Returns the rows of a fixture entry as a `[Float]` slice from the blob.
    static func rows(for entry: FixtureEntry, blob: Data) throws -> [[Float]] {
        let floatCount = entry.rows * entry.cols
        let expectedBytes = floatCount * MemoryLayout<Float>.size
        guard
            entry.byteOffset + expectedBytes <= blob.count,
            entry.byteLength == expectedBytes
        else {
            throw FixtureError.byteLengthMismatch(
                expected: expectedBytes, actual: entry.byteLength, name: entry.name
            )
        }
        let slice = blob[entry.byteOffset ..< entry.byteOffset + expectedBytes]
        return slice.withUnsafeBytes { ptr in
            let floats = ptr.bindMemory(to: Float.self)
            return (0 ..< entry.rows).map { r in
                let base = r * entry.cols
                return Array(floats[base ..< base + entry.cols])
            }
        }
    }

    enum FixtureError: Error, CustomStringConvertible {
        case indexNotFound
        case blobNotFound
        case byteLengthMismatch(expected: Int, actual: Int, name: String)

        var description: String {
            switch self {
            case .indexNotFound:
                return "hidden_states.json not found in Fixtures/ — run scripts/dump-pytorch-hidden-states.py"
            case .blobNotFound:
                return "hidden_states.bin not found in Fixtures/ — run scripts/dump-pytorch-hidden-states.py"
            case .byteLengthMismatch(let expected, let actual, let name):
                return "hidden_states.json entry '\(name)': byteLength \(actual) != expected \(expected)"
            }
        }
    }
}
