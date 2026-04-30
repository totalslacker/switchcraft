import Foundation

/// Loaders for the three Witchcraft-produced reference fixtures committed
/// under `Tests/Fixtures/`:
///
///   * `reference_centroids.{bin,json}` — bucket-table K-means centroids
///     (FP32 LE, dims-bound) emitted by indexing the 33-fact corpus.
///   * `reference_residuals.{bin,json}` — deterministic input vectors
///     fed through Witchcraft's `to_q4_bytes` / `from_companded_q4_bytes`,
///     plus the dequantised float blob.
///   * `reference_embeddings.{bin,json}` — Witchcraft GGUF per-token
///     embeddings for inputs shared with `xtr-base-en.embeddings.json`.
///
/// All three follow the existing `xtr-base-en.embeddings.{bin,json}`
/// convention: a raw FP32-LE blob plus a companion JSON index.
/// Fixtures load via `Bundle.module`, with each loader exposing an
/// `isAvailable` flag so tests can skip cleanly on a fresh checkout
/// where the fixtures haven't been regenerated yet.
///
/// Provenance, regeneration policy, and per-fixture tolerances are
/// documented in `adrs/013-reference-fixture-provenance.md`.

// MARK: - reference_centroids

enum ReferenceCentroidsFixture {
    static let indexResourceName = "reference_centroids"
    static let blobResourceName = "reference_centroids"

    struct Index: Decodable {
        let dims: Int
        let corpus: String?
        let inputs: Region
        let centroids: Region
        let provenance: Provenance

        struct Region: Decodable {
            let rows: Int?
            let clusters: Int?
            let byteOffset: Int
            let byteLength: Int
        }

        struct Provenance: Decodable {
            let witchcraftCommit: String
            let generatedAt: String?
            let source: String?
            let k: Int?
            let tPrime: Int?
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
            throw ReferenceFixtureError.indexNotFound(indexResourceName)
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
            throw ReferenceFixtureError.blobNotFound(blobResourceName)
        }
        return try Data(contentsOf: url)
    }

    /// Decode the k-means input embeddings as `rows * dims` row-major FP32.
    static func inputs(index: Index, blob: Data) -> [Float] {
        guard let rows = index.inputs.rows else {
            preconditionFailure("reference_centroids: inputs region missing rows")
        }
        precondition(
            index.inputs.byteLength == rows * index.dims * MemoryLayout<Float>.size,
            "reference_centroids: inputs byte length inconsistent with rows × dims"
        )
        return readFloats(
            blob: blob,
            byteOffset: index.inputs.byteOffset,
            byteLength: index.inputs.byteLength
        )
    }

    /// Decode the bucket centroids as `clusters * dims` row-major FP32.
    static func centroids(index: Index, blob: Data) -> [Float] {
        guard let clusters = index.centroids.clusters else {
            preconditionFailure("reference_centroids: centroids region missing clusters")
        }
        precondition(
            index.centroids.byteLength == clusters * index.dims * MemoryLayout<Float>.size,
            "reference_centroids: centroids byte length inconsistent with clusters × dims"
        )
        return readFloats(
            blob: blob,
            byteOffset: index.centroids.byteOffset,
            byteLength: index.centroids.byteLength
        )
    }

    private static func readFloats(blob: Data, byteOffset: Int, byteLength: Int) -> [Float] {
        let slice = blob[byteOffset..<(byteOffset + byteLength)]
        var floats = [Float](repeating: 0, count: byteLength / MemoryLayout<Float>.size)
        _ = floats.withUnsafeMutableBytes { dst in
            slice.copyBytes(to: dst)
        }
        return floats
    }
}

// MARK: - reference_residuals

enum ReferenceResidualsFixture {
    static let indexResourceName = "reference_residuals"
    static let blobResourceName = "reference_residuals"

    struct Index: Decodable {
        let dims: Int
        let vectors: [Vector]
        let provenance: Provenance

        struct Vector: Decodable {
            let name: String
            let dims: Int
            /// Raw input vector bytes (FP32 LE) hex-encoded.
            let inputFloatsHex: String
            /// Witchcraft-encoded Q4 bytes hex-encoded.
            let q4BytesHex: String
            let byteOffset: Int
            let byteLength: Int
        }

        struct Provenance: Decodable {
            let witchcraftCommit: String
            let generatedAt: String?
            let source: String?
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
            throw ReferenceFixtureError.indexNotFound(indexResourceName)
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
            throw ReferenceFixtureError.blobNotFound(blobResourceName)
        }
        return try Data(contentsOf: url)
    }

    /// Decode the dequantised-floats slice for one vector entry.
    static func dequantisedFloats(for entry: Index.Vector, blob: Data) -> [Float] {
        precondition(
            entry.byteLength == entry.dims * MemoryLayout<Float>.size,
            "reference_residuals: byte length inconsistent with dims"
        )
        let slice = blob[entry.byteOffset..<(entry.byteOffset + entry.byteLength)]
        var floats = [Float](repeating: 0, count: entry.dims)
        _ = floats.withUnsafeMutableBytes { dst in
            slice.copyBytes(to: dst)
        }
        return floats
    }
}

// MARK: - reference_embeddings

enum ReferenceEmbeddingsFixture {
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
            throw ReferenceFixtureError.indexNotFound(indexResourceName)
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
            throw ReferenceFixtureError.blobNotFound(blobResourceName)
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
}

// MARK: - Hex helpers

/// Decode a lowercase or mixed-case hex string into raw bytes. Returns
/// `nil` for malformed input. Used by the residual parity test to
/// expand `q4BytesHex` / `inputFloatsHex` from the fixture index.
enum ReferenceFixtureHex {
    static func decode(_ hex: String) -> Data? {
        let chars = Array(hex.utf8)
        guard chars.count % 2 == 0 else { return nil }
        var data = Data(capacity: chars.count / 2)
        for i in stride(from: 0, to: chars.count, by: 2) {
            guard
                let high = nibble(chars[i]),
                let low = nibble(chars[i + 1])
            else { return nil }
            data.append((high << 4) | low)
        }
        return data
    }

    private static func nibble(_ b: UInt8) -> UInt8? {
        switch b {
        case 0x30...0x39: return b - 0x30
        case 0x41...0x46: return b - 0x41 + 10
        case 0x61...0x66: return b - 0x61 + 10
        default: return nil
        }
    }
}

/// Decode a hex string of FP32-LE bytes back into a `[Float]` of the
/// expected dim count. Trips a precondition on malformed input — only
/// invoked by tests over data we own.
extension ReferenceFixtureHex {
    static func decodeFloats(_ hex: String, count: Int) -> [Float] {
        guard let data = decode(hex) else {
            preconditionFailure("malformed hex string")
        }
        precondition(
            data.count == count * MemoryLayout<Float>.size,
            "hex byte length \(data.count) does not match \(count) FP32 values"
        )
        var out = [Float](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst)
        }
        return out
    }
}

// MARK: - Errors

enum ReferenceFixtureError: Error, CustomStringConvertible {
    case indexNotFound(String)
    case blobNotFound(String)

    var description: String {
        switch self {
        case .indexNotFound(let name):
            return "\(name).json missing from test bundle"
        case .blobNotFound(let name):
            return "\(name).bin missing from test bundle"
        }
    }
}
