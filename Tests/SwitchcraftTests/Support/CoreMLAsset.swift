import Foundation

/// Resolves the canonical XTR-base-en `.mlpackage` for asset-gated tests.
///
/// The model asset (~80 MB) is **not committed** to the repository. To
/// exercise the integration tests:
///
///   1. Run `python3 scripts/convert-xtr-to-coreml.py …` to produce the
///      `.mlpackage` (see README.md → Getting Started).
///   2. Set `SWITCHCRAFT_XTR_MLPACKAGE=<absolute-path-to-mlpackage>`.
///   3. `swift test`.
///
/// When the env var is unset or points to a non-existent path, the
/// asset-gated tests skip cleanly via Swift Testing's `.enabled(if:)`
/// trait. Fresh checkouts stay green.
enum CoreMLAsset {
    /// Environment variable consulted by tests. Documented in README.md.
    static let envVar = "SWITCHCRAFT_XTR_MLPACKAGE"

    /// `true` iff the env var is set and resolves to an existing path.
    static var isAvailable: Bool {
        url != nil
    }

    /// Resolved URL of the `.mlpackage` (or `.mlmodelc`), or `nil` when
    /// the env var is unset / the path does not exist.
    static var url: URL? {
        guard
            let raw = ProcessInfo.processInfo.environment[envVar],
            !raw.isEmpty
        else {
            return nil
        }
        let url = URL(fileURLWithPath: raw)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return url
    }
}

/// Resolves the FP32 reference embeddings + index file produced by the
/// conversion script. Used by `T5CoreMLEmbedderTests` to verify that the
/// CoreML output matches the PyTorch reference within cosine-similarity
/// tolerance.
///
/// The fixture files are committed under `Tests/Fixtures/`; tests that
/// require them check `isAvailable` so a worktree without the
/// regenerated fixtures still skips cleanly rather than failing.
enum CoreMLEmbeddingFixtures {
    static let indexResourceName = "xtr-base-en.embeddings"
    static let blobResourceName = "xtr-base-en.embeddings"

    /// Companion JSON index. Layout produced by the conversion script:
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

    /// `true` iff both the JSON index and binary blob can be loaded
    /// from the test bundle.
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

    /// Load and decode the index file from the test bundle.
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

    /// Load the raw FP32 little-endian blob from the test bundle.
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
