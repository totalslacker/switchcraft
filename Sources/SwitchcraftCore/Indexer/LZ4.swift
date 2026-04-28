import Foundation
import Compression

/// LZ4 block codec compatible with the upstream Rust `lz4_flex` crate's
/// `compress_prepend_size` / `decompress_size_prepended` framing.
///
/// Wire format (parity with `lz4_flex::block`):
///
///     [u32 LE uncompressed size][raw LZ4 block bytes]
///
/// Encoded buffers are written using Apple's `Compression.framework` with
/// `COMPRESSION_LZ4_RAW` (the "raw" variant produces an LZ4 block with no
/// outer framing — matching `lz4_flex`'s block layer). The uncompressed
/// size header is written by us as a 4-byte little-endian `UInt32`.
///
/// Decode is the parity gate: it must accept any blob produced by
/// `lz4_flex::compress_prepend_size`. Encode round-trips through `decode`
/// but does not have to be byte-identical to `lz4_flex`'s encoder; multiple
/// valid LZ4 compressions exist for the same input.
public enum LZ4 {

    public enum Error: Swift.Error, Sendable, Equatable {
        case headerTooShort
        case decompressionFailed
        case sizeMismatch(expected: Int, actual: Int)
    }

    /// Encode `input` to an `lz4_flex`-compatible blob:
    ///
    ///     [u32 LE uncompressed size][raw LZ4 block]
    public static func compressPrependSize(_ input: Data) -> Data {
        precondition(
            input.count <= Int(UInt32.max),
            "LZ4.compressPrependSize cannot frame inputs larger than UInt32.max bytes (got \(input.count))"
        )
        var out = Data()
        out.reserveCapacity(4 + input.count)

        let size = UInt32(input.count).littleEndian
        withUnsafeBytes(of: size) { out.append(contentsOf: $0) }

        guard !input.isEmpty else { return out }

        // LZ4 worst-case expansion is small but non-zero; allocate a
        // generous destination buffer (input.count + input.count/255 + 16).
        let destCapacity = input.count + (input.count / 255) + 16
        let dest = UnsafeMutablePointer<UInt8>.allocate(capacity: destCapacity)
        defer { dest.deallocate() }

        let written: Int = input.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int in
            guard let src = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return 0
            }
            return compression_encode_buffer(
                dest, destCapacity,
                src, input.count,
                nil,
                COMPRESSION_LZ4_RAW
            )
        }

        if written == 0 {
            // Apple's COMPRESSION_LZ4_RAW returns 0 when the input is
            // incompressible. Emit a hand-rolled all-literals LZ4 block —
            // the standard decoder (lz4_flex and Apple's own decoder both)
            // accepts a single sequence whose match part is absent because
            // it is the last sequence of the block.
            out.append(literalOnlyBlock(input))
            return out
        }

        out.append(dest, count: written)
        return out
    }

    /// Produce an LZ4 block consisting of a single all-literals sequence.
    /// The last sequence in an LZ4 block omits the match section, so this
    /// is valid for the entire input.
    private static func literalOnlyBlock(_ input: Data) -> Data {
        var out = Data()
        let n = input.count
        let literalLenInToken = min(n, 15)
        let token = UInt8(literalLenInToken << 4)
        out.append(token)
        if n >= 15 {
            var remainder = n - 15
            while remainder >= 255 {
                out.append(0xFF)
                remainder -= 255
            }
            out.append(UInt8(remainder))
        }
        out.append(input)
        return out
    }

    /// Decode an `lz4_flex`-compatible blob: read the 4-byte little-endian
    /// uncompressed-size header, then decompress the remaining bytes as a
    /// raw LZ4 block of exactly that length.
    public static func decompressPrependSize(_ blob: Data) throws -> Data {
        guard blob.count >= 4 else {
            throw Error.headerTooShort
        }

        let expectedSize = blob.withUnsafeBytes { raw -> UInt32 in
            var value: UInt32 = 0
            // Build the value byte-by-byte so endianness is explicit and
            // safe for unaligned reads.
            value |= UInt32(raw[0])
            value |= UInt32(raw[1]) << 8
            value |= UInt32(raw[2]) << 16
            value |= UInt32(raw[3]) << 24
            return value
        }
        let expected = Int(expectedSize)

        if expected == 0 {
            // Intentionally lenient: `lz4_flex::compress_prepend_size`
            // can emit a trailing zero-byte token after the 4-byte size
            // header for empty inputs (see committed parity fixture
            // `Tests/Fixtures/indices/sample-pairs.json` "empty" case).
            // Tightening to `blob.count == 4` would break decode-side
            // parity, which is this codec's primary contract. Any
            // trailing bytes are ignored when the declared size is 0.
            return Data()
        }

        let body = blob.subdata(in: 4..<blob.count)
        let dest = UnsafeMutablePointer<UInt8>.allocate(capacity: expected)
        defer { dest.deallocate() }

        let written: Int = body.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int in
            guard let src = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return 0
            }
            return compression_decode_buffer(
                dest, expected,
                src, body.count,
                nil,
                COMPRESSION_LZ4_RAW
            )
        }

        guard written > 0 else {
            throw Error.decompressionFailed
        }
        guard written == expected else {
            throw Error.sizeMismatch(expected: expected, actual: written)
        }

        return Data(bytes: dest, count: written)
    }
}
