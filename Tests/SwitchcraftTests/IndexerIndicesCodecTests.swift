import Foundation
import Testing
@testable import SwitchcraftCore

@Suite("Indexer Indices Codec")
struct IndexerIndicesCodecTests {

    // MARK: - Round-trip

    @Test("encode → decode round-trips an empty list")
    func roundTripEmpty() throws {
        let pairs: [IndexPair] = []
        let blob = IndicesCodec.encode(pairs)
        let decoded = try IndicesCodec.decode(blob)
        #expect(decoded.isEmpty)
    }

    @Test("encode → decode round-trips a single pair")
    func roundTripSingle() throws {
        let pairs = [IndexPair(chunkID: 7, tokenOffset: 3)]
        let blob = IndicesCodec.encode(pairs)
        let decoded = try IndicesCodec.decode(blob)
        #expect(decoded == pairs)
    }

    @Test("encode → decode round-trips ascending majors with zero minors")
    func roundTripAscendingMajors() throws {
        let pairs = (1...32).map { IndexPair(chunkID: UInt32($0), tokenOffset: 0) }
        let blob = IndicesCodec.encode(pairs)
        let decoded = try IndicesCodec.decode(blob)
        #expect(decoded == pairs)
    }

    @Test("encode → decode round-trips a single major with many minors")
    func roundTripSameMajor() throws {
        let pairs = (0..<128).map { IndexPair(chunkID: 5, tokenOffset: UInt32($0)) }
        let blob = IndicesCodec.encode(pairs)
        let decoded = try IndicesCodec.decode(blob)
        #expect(decoded == pairs)
    }

    @Test("encode → decode round-trips mixed major/minor patterns")
    func roundTripMixed() throws {
        let pairs: [IndexPair] = [
            IndexPair(chunkID: 1, tokenOffset: 0),
            IndexPair(chunkID: 1, tokenOffset: 1),
            IndexPair(chunkID: 2, tokenOffset: 0),
            IndexPair(chunkID: 2, tokenOffset: 5),
            IndexPair(chunkID: 3, tokenOffset: 0),
            IndexPair(chunkID: 10, tokenOffset: 0),
            IndexPair(chunkID: 10, tokenOffset: 100),
        ]
        let blob = IndicesCodec.encode(pairs)
        let decoded = try IndicesCodec.decode(blob)
        #expect(decoded == pairs)
    }

    @Test("encode → decode round-trips many chunks × many tokens")
    func roundTripDense() throws {
        var pairs = [IndexPair]()
        for c in 1..<33 {
            for t in 0..<16 {
                pairs.append(IndexPair(chunkID: UInt32(c), tokenOffset: UInt32(t)))
            }
        }
        let blob = IndicesCodec.encode(pairs)
        let decoded = try IndicesCodec.decode(blob)
        #expect(decoded == pairs)
    }

    // MARK: - Decode-side parity with Rust lz4_flex / compress_keys

    @Test("decode accepts blobs produced by upstream Witchcraft compress_keys")
    func parityWithRust() throws {
        let url = try #require(
            Bundle.module.url(
                forResource: "sample-pairs",
                withExtension: "json",
                subdirectory: "Fixtures/indices"
            ),
            "missing Tests/Fixtures/indices/sample-pairs.json"
        )
        let data = try Data(contentsOf: url)
        let fixtures = try JSONDecoder().decode([IndicesFixture].self, from: data)
        #expect(fixtures.count >= 6, "fixture file looks empty")

        for fixture in fixtures where fixture.label != "_sentinel" {
            let expected = fixture.pairs.map {
                IndexPair(chunkID: $0[0], tokenOffset: $0[1])
            }
            let blob = Data(hexString: fixture.rustCompressedHex)
            let decoded = try IndicesCodec.decode(blob)
            #expect(
                decoded == expected,
                "fixture '\(fixture.label)' did not decode to its pair list"
            )
        }
    }

    private struct IndicesFixture: Decodable {
        let label: String
        let pairs: [[UInt32]]
        let rustCompressedHex: String

        enum CodingKeys: String, CodingKey {
            case label
            case pairs
            case rustCompressedHex = "rust_compressed_hex"
        }
    }
}
