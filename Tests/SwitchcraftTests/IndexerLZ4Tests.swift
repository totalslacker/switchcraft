import Foundation
import Testing
@testable import SwitchcraftCore

@Suite("Indexer LZ4 Codec")
struct IndexerLZ4Tests {

    // MARK: - Round-trip

    @Test("compressPrependSize → decompressPrependSize round-trips empty")
    func roundTripEmpty() throws {
        let blob = LZ4.compressPrependSize(Data())
        // 4-byte LE size header == 0, no body.
        #expect(Array(blob) == [0x00, 0x00, 0x00, 0x00])
        let decoded = try LZ4.decompressPrependSize(blob)
        #expect(decoded == Data())
    }

    @Test("compressPrependSize → decompressPrependSize round-trips small inputs")
    func roundTripSmall() throws {
        let cases: [Data] = [
            Data("Hello, world!".utf8),
            Data(repeating: 0, count: 64),
            Data((0..<256).map { UInt8($0) }),
            Data("abc".utf8) + Data("abc".utf8) + Data("abc".utf8) + Data("abc".utf8),
        ]
        for input in cases {
            let blob = LZ4.compressPrependSize(input)
            let decoded = try LZ4.decompressPrependSize(blob)
            #expect(decoded == input, "round-trip mismatch for \(input.count)-byte input")
        }
    }

    @Test("compressPrependSize → decompressPrependSize round-trips 8 KiB random data")
    func roundTripRandom() throws {
        var rng = SplitMix64(seed: 0xDEAD_BEEF_CAFE_F00D)
        var bytes = [UInt8]()
        bytes.reserveCapacity(8 * 1024)
        for _ in 0..<8192 {
            bytes.append(UInt8(rng.next() & 0xFF))
        }
        let input = Data(bytes)
        let blob = LZ4.compressPrependSize(input)
        let decoded = try LZ4.decompressPrependSize(blob)
        #expect(decoded == input)
    }

    // MARK: - Header / error paths

    @Test("decompressPrependSize rejects buffers shorter than the size header")
    func decodeShortHeader() {
        let bad = Data([0x01, 0x02])
        #expect(throws: LZ4.Error.headerTooShort) {
            _ = try LZ4.decompressPrependSize(bad)
        }
    }

    @Test("decompressPrependSize honours the little-endian size prefix")
    func decodeSizePrefix() throws {
        // Header is little-endian, so 0x05 0x00 0x00 0x00 == 5 bytes expected.
        // Body is "Hello" as a single literal sequence: token 0x50 then 5 literals.
        let blob = Data([0x05, 0x00, 0x00, 0x00, 0x50, 0x48, 0x65, 0x6C, 0x6C, 0x6F])
        let decoded = try LZ4.decompressPrependSize(blob)
        #expect(decoded == Data("Hello".utf8))
    }

    // MARK: - Parity with lz4_flex

    @Test("decompressPrependSize accepts blobs produced by Rust lz4_flex")
    func parityWithRustLZ4Flex() throws {
        let url = try #require(
            Bundle.module.url(
                forResource: "sample-roundtrip",
                withExtension: "json",
                subdirectory: "Fixtures/lz4"
            ),
            "missing Tests/Fixtures/lz4/sample-roundtrip.json"
        )
        let data = try Data(contentsOf: url)
        let fixtures = try JSONDecoder().decode([LZ4Fixture].self, from: data)
        // Should at least cover a few non-trivial cases plus the sentinel.
        #expect(fixtures.count >= 5, "fixture file looks empty")

        for fixture in fixtures where fixture.label != "_sentinel" {
            let input = Data(hexString: fixture.inputHex)
            let rustBlob = Data(hexString: fixture.lz4FlexOutputHex)
            let decoded = try LZ4.decompressPrependSize(rustBlob)
            #expect(
                decoded == input,
                "fixture '\(fixture.label)' did not decode to its input"
            )
        }
    }

    // MARK: - Fixture model

    private struct LZ4Fixture: Decodable {
        let label: String
        let inputHex: String
        let lz4FlexOutputHex: String

        enum CodingKeys: String, CodingKey {
            case label
            case inputHex = "input_hex"
            case lz4FlexOutputHex = "lz4_flex_output_hex"
        }
    }
}

// MARK: - Hex helper

extension Data {
    fileprivate init(hexString: String) {
        var out = Data()
        out.reserveCapacity(hexString.count / 2)
        var index = hexString.startIndex
        while index < hexString.endIndex {
            let next = hexString.index(index, offsetBy: 2)
            if let byte = UInt8(hexString[index..<next], radix: 16) {
                out.append(byte)
            }
            index = next
        }
        self = out
    }
}
