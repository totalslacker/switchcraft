import Foundation

/// One entry in a bucket's `indices` blob: a `(chunkID, tokenOffset)` tuple
/// pointing back at one token embedding in `chunk.embeddings`.
public struct IndexPair: Sendable, Hashable {
    public var chunkID: UInt32
    public var tokenOffset: UInt32

    public init(chunkID: UInt32, tokenOffset: UInt32) {
        self.chunkID = chunkID
        self.tokenOffset = tokenOffset
    }
}

/// Wire-format codec for `BucketRecord.indices`, matching upstream
/// Witchcraft's `compress_keys` / `decompress_keys` exactly.
///
/// Algorithm:
///
///  1. Write the first pair as two raw little-endian `UInt32`s (8 bytes).
///  2. For each subsequent pair, with `base` = the previous pair:
///     - `dMajor = pair.chunkID &- base.chunkID`
///     - `dMinor = (dMajor == 0) ? pair.tokenOffset &- base.tokenOffset
///                              : pair.tokenOffset`
///     - Write `(dMajor, dMinor)` as two little-endian `UInt32`s.
///  3. Wrap the resulting byte stream in `LZ4.compressPrependSize`.
///
/// Decoding inverts step (1)/(2) using `wrapping_add` semantics so that
/// the upstream Rust round-trip property holds: encoded bytes from
/// `lz4_flex` decompress to the exact same `(chunkID, tokenOffset)`
/// sequence regardless of whether the deltas underflowed.
///
/// Endianness is **explicit little-endian** in both directions. Apple
/// platforms are LE today, but encoding the bytes by hand makes any
/// future big-endian backend continue to read fixtures that were
/// originally produced on an LE host.
public enum IndicesCodec {

    public enum Error: Swift.Error, Sendable, Equatable {
        case truncatedPayload(byteCount: Int)
        case lz4(LZ4.Error)
    }

    public static func encode(_ pairs: [IndexPair]) -> Data {
        var bytes = Data()
        bytes.reserveCapacity(pairs.count * 8)

        guard let first = pairs.first else {
            return LZ4.compressPrependSize(bytes)
        }
        appendU32LE(first.chunkID, into: &bytes)
        appendU32LE(first.tokenOffset, into: &bytes)

        var base = first
        for pair in pairs.dropFirst() {
            let deltaMajor = pair.chunkID &- base.chunkID
            let deltaMinor: UInt32
            if deltaMajor == 0 {
                deltaMinor = pair.tokenOffset &- base.tokenOffset
            } else {
                deltaMinor = pair.tokenOffset
            }
            appendU32LE(deltaMajor, into: &bytes)
            appendU32LE(deltaMinor, into: &bytes)
            base = pair
        }

        return LZ4.compressPrependSize(bytes)
    }

    public static func decode(_ blob: Data) throws -> [IndexPair] {
        let raw: Data
        do {
            raw = try LZ4.decompressPrependSize(blob)
        } catch let err as LZ4.Error {
            throw Error.lz4(err)
        }

        if raw.isEmpty { return [] }
        guard raw.count % 8 == 0 else {
            throw Error.truncatedPayload(byteCount: raw.count)
        }

        let count = raw.count / 8
        var out = [IndexPair]()
        out.reserveCapacity(count)

        let firstChunk = readU32LE(raw, at: 0)
        let firstOffset = readU32LE(raw, at: 4)
        var base = IndexPair(chunkID: firstChunk, tokenOffset: firstOffset)
        out.append(base)

        for i in 1..<count {
            let off = i * 8
            let dMajor = readU32LE(raw, at: off)
            let dMinor = readU32LE(raw, at: off + 4)
            let major = base.chunkID &+ dMajor
            let minor: UInt32
            if dMajor == 0 {
                minor = base.tokenOffset &+ dMinor
            } else {
                minor = dMinor
            }
            base = IndexPair(chunkID: major, tokenOffset: minor)
            out.append(base)
        }

        return out
    }

    // MARK: - Endianness helpers

    @inline(__always)
    private static func appendU32LE(_ value: UInt32, into data: inout Data) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8((value >> 16) & 0xFF))
        data.append(UInt8((value >> 24) & 0xFF))
    }

    @inline(__always)
    private static func readU32LE(_ data: Data, at offset: Int) -> UInt32 {
        let base = data.startIndex + offset
        return UInt32(data[base])
            | (UInt32(data[base + 1]) << 8)
            | (UInt32(data[base + 2]) << 16)
            | (UInt32(data[base + 3]) << 24)
    }
}
