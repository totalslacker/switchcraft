// SPDX-License-Identifier: Apache-2.0
//
// GGUF v2/v3 reader for the Phase 2 Metal embedder (issue #59).
//
// Parses the header, KV metadata block, and tensor-info block of a GGUF
// v2 or v3 file, then lazily uploads requested tensor bytes to `MTLBuffer`s.
//
// On-disk format reference: `ggml/docs/gguf.md`. The v2/v3 layout is:
//
//     magic   : 4 bytes "GGUF"
//     version : u32 (must be 2 or 3)
//     n_tensors: u64
//     n_kv     : u64
//     kv_pairs : { string key, u32 value_type, value } × n_kv
//     tensors  : { string name, u32 n_dims, u64 dims[n_dims],
//                  u32 type, u64 data_offset } × n_tensors
//     padding  : up to `alignment` (default 32, override via
//                `general.alignment` KV)
//     data     : tensor bytes, each tensor at `data_block_start +
//                tensor.data_offset`
//
// Strings are u64 byte-length + UTF-8 bytes (no terminator). Numeric
// fields are little-endian. The GGUF spec is little-endian-only and the
// reader does not attempt big-endian fallback.
//
// Implementation notes
// --------------------
//
// - File contents are memory-mapped via `Data(contentsOf:options: .mappedIfSafe)`.
//   ~80 MB resident is well under ADR 012's 300 MB peak-RSS ceiling.
// - All 13 GGUF value types are *parsed* (so unknown KVs do not desync
//   the file offset), but only `string`/`int64`/`uint64`/`float32`/
//   `float64`/`bool` are surfaced via `metadata`. Arrays and small-width
//   integer values are consumed and discarded. See ADR 016 §"Metadata
//   KV surface".
// - Tensor uploads are lazy + cached. First call to `tensor(_:)` for a
//   given name allocates an `MTLBuffer` and copies the on-disk bytes
//   in; subsequent calls return the cached `GGUFTensor`. Buffers use
//   `.storageModeShared` (ADR 016 §"MTLBuffer storage mode").

#if canImport(Metal)

import Foundation
import Metal

/// Pure-Swift GGUF v2/v3 reader.
///
/// Construct with a file URL + `MTLDevice`; parse runs synchronously
/// inside `init`. `tensor(_:)` and `tensorNames()` are async because
/// the reader is an `actor` (state — the cache and parsed tensor info —
/// is mutated across calls).
///
/// GGUF v2 and v3 are structurally identical for the dtypes Switchcraft
/// consumes (Q4_K, F32, F16). v2 is accepted because Witchcraft's
/// `quantize-tool` at the pinned Candle rev `5bd5618` emits GGUF v2.
/// See ADR 016 and Switchcraft issue #74.
@_spi(SwitchcraftMetal)
public actor GGUFReader {
    /// First four bytes of every GGUF file: ASCII "GGUF" little-endian.
    public static let magic: UInt32 = 0x4655_4747  // 'F' 'U' 'G' 'G' in LE byte order
    /// Default tensor-data alignment when `general.alignment` is absent.
    public static let defaultAlignment: Int = 32

    /// The GGUF version loaded from this file. Always 2 or 3 for any
    /// file accepted by this reader. Exposed for debuggability — a future
    /// structural divergence between v2 and v3 would be visible here
    /// rather than silently tolerated.
    public nonisolated let loadedVersion: UInt32

    /// Description of one tensor as parsed from the tensor-info block.
    /// The MTLBuffer is allocated on demand and held in `cache`.
    private struct TensorInfo {
        let name: String
        let dtype: GGUFDType
        let shape: [Int]
        let elementCount: Int
        /// Byte offset relative to `dataBlockStart`, as recorded in the
        /// tensor-info block.
        let relativeOffset: Int
    }

    /// Backing file held mapped for the reader's lifetime so `tensor(_:)`
    /// can read raw bytes without re-opening.
    private let data: Data
    /// Metal device used to allocate buffers.
    private let device: MTLDevice
    /// Parsed metadata (typed subset only).
    private let _metadata: [String: GGUFValue]
    /// Tensor info table indexed by name.
    private let tensorInfo: [String: TensorInfo]
    /// Tensor names in declaration order.
    private let names: [String]
    /// Absolute file offset where the tensor-data block begins (after
    /// alignment padding past the tensor-info block).
    private let dataBlockStart: Int
    /// Tensor cache populated lazily by `tensor(_:)`.
    private var cache: [String: GGUFTensor] = [:]

    /// Open and parse a GGUF v2 or v3 file.
    public init(url: URL, device: MTLDevice) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw GGUFError.ioError("failed to open \(url.path): \(error)")
        }

        var cursor = 0
        // ----- Header -----
        let magic: UInt32 = try Self.readUInt32(data, cursor: &cursor)
        guard magic == Self.magic else { throw GGUFError.invalidMagic }
        let version: UInt32 = try Self.readUInt32(data, cursor: &cursor)
        guard version == 2 || version == 3 else {
            throw GGUFError.unsupportedVersion(found: version)
        }
        self.loadedVersion = version
        let nTensors: UInt64 = try Self.readUInt64(data, cursor: &cursor)
        let nKV: UInt64 = try Self.readUInt64(data, cursor: &cursor)

        // ----- KV block -----
        var metadata: [String: GGUFValue] = [:]
        for _ in 0..<nKV {
            let key = try Self.readString(data, cursor: &cursor)
            let valueTypeRaw: UInt32 = try Self.readUInt32(data, cursor: &cursor)
            let value = try Self.readValue(
                data,
                cursor: &cursor,
                valueTypeRaw: valueTypeRaw
            )
            // Only surface the typed-subset variants.
            if let exposed = value {
                metadata[key] = exposed
            }
        }

        // Resolve alignment override before parsing tensor info — the
        // alignment governs the data-block start, not the tensor-info
        // block, but resolve it once here. Validate that the value is
        // positive and a power of two; ggml writes 32 by default and
        // round-tripping anything else through `% alignment` would
        // either trap (alignment == 0) or produce a misaligned data
        // block start (non-power-of-two).
        let alignment: Int
        if case .uint64(let n) = metadata["general.alignment"] {
            guard n >= 1, n <= UInt64(Int.max) else {
                throw GGUFError.invalidHeader(
                    reason: "general.alignment=\(n) is not a positive Int-representable value"
                )
            }
            let v = Int(n)
            // Power-of-two check: GGUF spec defines alignment as
            // a power of two; reject anything else so a corrupt KV
            // doesn't quietly produce a misaligned tensor-data block.
            guard v & (v - 1) == 0 else {
                throw GGUFError.invalidHeader(
                    reason: "general.alignment=\(v) is not a power of two"
                )
            }
            alignment = v
        } else {
            alignment = Self.defaultAlignment
        }

        // ----- Tensor info block -----
        var tensorInfo: [String: TensorInfo] = [:]
        var orderedNames: [String] = []
        orderedNames.reserveCapacity(Int(nTensors))
        for _ in 0..<nTensors {
            let name = try Self.readString(data, cursor: &cursor)
            let nDims: UInt32 = try Self.readUInt32(data, cursor: &cursor)
            var shape: [Int] = []
            shape.reserveCapacity(Int(nDims))
            var elementCount = 1
            for _ in 0..<nDims {
                let dim: UInt64 = try Self.readUInt64(data, cursor: &cursor)
                // Guard the u64 → Int conversion and the multiplicative
                // accumulation so a corrupt or hostile header becomes a
                // typed error instead of a trap.
                guard dim <= UInt64(Int.max) else {
                    throw GGUFError.malformedTensor(
                        name: name,
                        reason: "dim \(dim) does not fit in Int"
                    )
                }
                let dimInt = Int(dim)
                shape.append(dimInt)
                let (product, overflow) = elementCount.multipliedReportingOverflow(by: dimInt)
                guard !overflow else {
                    throw GGUFError.malformedTensor(
                        name: name,
                        reason: "element count overflow on dim \(dimInt) (running product so far: \(elementCount))"
                    )
                }
                elementCount = product
            }
            let typeRaw: UInt32 = try Self.readUInt32(data, cursor: &cursor)
            guard let dtype = GGUFDType.from(rawGGMLType: typeRaw) else {
                throw GGUFError.unsupportedDType(name: name, raw: typeRaw)
            }
            // Q4_K element counts must be a whole number of super-blocks.
            if dtype == .q4k && elementCount % Q4KDecode.elementsPerSuperBlock != 0 {
                throw GGUFError.malformedTensor(
                    name: name,
                    reason: "Q4_K element count \(elementCount) is not a multiple of 256"
                )
            }
            let relativeOffset: UInt64 = try Self.readUInt64(data, cursor: &cursor)
            guard relativeOffset <= UInt64(Int.max) else {
                throw GGUFError.malformedTensor(
                    name: name,
                    reason: "relative offset \(relativeOffset) does not fit in Int"
                )
            }
            let info = TensorInfo(
                name: name,
                dtype: dtype,
                shape: shape,
                elementCount: elementCount,
                relativeOffset: Int(relativeOffset)
            )
            tensorInfo[name] = info
            orderedNames.append(name)
        }

        // Pad cursor up to the next alignment boundary; that's where
        // tensor data begins. When the file declares zero tensors the
        // data block may legitimately be absent — only enforce the
        // bounds check when at least one tensor is declared (per-tensor
        // bounds checks in `tensor(_:)` cover individual reads).
        let dataBlockStart = Self.alignUp(cursor, to: alignment)
        if !orderedNames.isEmpty {
            guard dataBlockStart <= data.count else {
                throw GGUFError.truncated(at: cursor, needed: dataBlockStart - data.count)
            }
        }

        self.data = data
        self.device = device
        self._metadata = metadata
        self.tensorInfo = tensorInfo
        self.names = orderedNames
        self.dataBlockStart = dataBlockStart
    }

    /// Typed-subset metadata KVs from the file. Unsupported value types
    /// (arrays, narrow ints) are consumed during parsing but not
    /// surfaced here.
    public var metadata: [String: GGUFValue] { _metadata }

    /// Tensor names in declaration order.
    public func tensorNames() -> [String] { names }

    /// Resolve a tensor by name, allocating an `MTLBuffer` and copying
    /// its on-disk bytes on first access. Subsequent calls return the
    /// cached instance.
    public func tensor(_ name: String) throws -> GGUFTensor {
        if let cached = cache[name] {
            return cached
        }
        guard let info = tensorInfo[name] else {
            throw GGUFError.tensorNotFound(name: name)
        }
        let byteCount = info.dtype.storageBytes(for: info.elementCount)
        let (absoluteOffset, ovAbs) = dataBlockStart.addingReportingOverflow(info.relativeOffset)
        guard !ovAbs else {
            throw GGUFError.malformedTensor(
                name: name,
                reason: "absolute data offset overflow (dataBlockStart=\(dataBlockStart), relative=\(info.relativeOffset))"
            )
        }
        let (endOffset, ovEnd) = absoluteOffset.addingReportingOverflow(byteCount)
        guard !ovEnd, endOffset <= data.count else {
            throw GGUFError.malformedTensor(
                name: name,
                reason: "data range [\(absoluteOffset), \(absoluteOffset)+\(byteCount)) extends past EOF (\(data.count))"
            )
        }
        let buffer: MTLBuffer = try data.withUnsafeBytes { raw -> MTLBuffer in
            let src = raw.baseAddress!.advanced(by: absoluteOffset)
            guard let buf = device.makeBuffer(
                bytes: src,
                length: byteCount,
                options: .storageModeShared
            ) else {
                throw GGUFError.bufferAllocationFailed(name: name, bytes: byteCount)
            }
            buf.label = "GGUF:\(name)"
            return buf
        }
        let tensor = GGUFTensor(
            name: info.name,
            dtype: info.dtype,
            shape: info.shape,
            buffer: buffer,
            elementCount: info.elementCount
        )
        cache[name] = tensor
        return tensor
    }

    // MARK: - Low-level byte readers

    @inline(__always)
    private static func need(_ data: Data, cursor: Int, bytes: Int) throws {
        let (end, overflow) = cursor.addingReportingOverflow(bytes)
        if overflow || end > data.count {
            // Report the *missing* byte count, not the requested size, so
            // diagnostics describe the shortfall rather than the read.
            let shortfall: Int
            if overflow {
                shortfall = .max
            } else {
                shortfall = end - data.count
            }
            throw GGUFError.truncated(at: cursor, needed: shortfall)
        }
    }

    private static func readUInt8(_ data: Data, cursor: inout Int) throws -> UInt8 {
        try need(data, cursor: cursor, bytes: 1)
        let v = data[data.startIndex + cursor]
        cursor += 1
        return v
    }

    private static func readUInt16(_ data: Data, cursor: inout Int) throws -> UInt16 {
        try need(data, cursor: cursor, bytes: 2)
        let lo = UInt16(data[data.startIndex + cursor])
        let hi = UInt16(data[data.startIndex + cursor + 1])
        cursor += 2
        return lo | (hi << 8)
    }

    private static func readUInt32(_ data: Data, cursor: inout Int) throws -> UInt32 {
        try need(data, cursor: cursor, bytes: 4)
        var v: UInt32 = 0
        for i in 0..<4 {
            v |= UInt32(data[data.startIndex + cursor + i]) << (8 * i)
        }
        cursor += 4
        return v
    }

    private static func readUInt64(_ data: Data, cursor: inout Int) throws -> UInt64 {
        try need(data, cursor: cursor, bytes: 8)
        var v: UInt64 = 0
        for i in 0..<8 {
            v |= UInt64(data[data.startIndex + cursor + i]) << (8 * i)
        }
        cursor += 8
        return v
    }

    private static func readFloat32(_ data: Data, cursor: inout Int) throws -> Float {
        let bits = try readUInt32(data, cursor: &cursor)
        return Float(bitPattern: bits)
    }

    private static func readFloat64(_ data: Data, cursor: inout Int) throws -> Double {
        let bits = try readUInt64(data, cursor: &cursor)
        return Double(bitPattern: bits)
    }

    private static func readString(_ data: Data, cursor: inout Int) throws -> String {
        let lenStart = cursor
        let len = try readUInt64(data, cursor: &cursor)
        // Guard against u64 → Int truncation on 64-bit (and especially
        // 32-bit) hosts so a malformed length field becomes a typed
        // error instead of a trap.
        guard len <= UInt64(Int.max) else {
            throw GGUFError.malformedString(at: lenStart)
        }
        let lenInt = Int(len)
        try need(data, cursor: cursor, bytes: lenInt)
        let start = data.startIndex + cursor
        let slice = data[start..<(start + lenInt)]
        cursor += lenInt
        guard let s = String(data: slice, encoding: .utf8) else {
            throw GGUFError.malformedString(at: cursor - lenInt)
        }
        return s
    }

    /// Read one KV value of the given GGUF type. Returns `nil` for
    /// value types we deliberately do not surface (narrow ints, arrays);
    /// the cursor is advanced regardless so the file offset stays
    /// consistent.
    private static func readValue(
        _ data: Data,
        cursor: inout Int,
        valueTypeRaw: UInt32
    ) throws -> GGUFValue? {
        switch valueTypeRaw {
        case 0:  // u8
            _ = try readUInt8(data, cursor: &cursor)
            return nil
        case 1:  // i8
            _ = try readUInt8(data, cursor: &cursor)
            return nil
        case 2:  // u16
            _ = try readUInt16(data, cursor: &cursor)
            return nil
        case 3:  // i16
            _ = try readUInt16(data, cursor: &cursor)
            return nil
        case 4:  // u32
            let v = try readUInt32(data, cursor: &cursor)
            return .uint64(UInt64(v))
        case 5:  // i32
            let v = try readUInt32(data, cursor: &cursor)
            return .int64(Int64(Int32(bitPattern: v)))
        case 6:  // f32
            let v = try readFloat32(data, cursor: &cursor)
            return .float32(v)
        case 7:  // bool
            let v = try readUInt8(data, cursor: &cursor)
            return .bool(v != 0)
        case 8:  // string
            let s = try readString(data, cursor: &cursor)
            return .string(s)
        case 9:  // array
            let elemType = try readUInt32(data, cursor: &cursor)
            let count = try readUInt64(data, cursor: &cursor)
            // Recursively consume elements without surfacing them.
            for _ in 0..<count {
                _ = try readValue(data, cursor: &cursor, valueTypeRaw: elemType)
            }
            return nil
        case 10:  // u64
            let v = try readUInt64(data, cursor: &cursor)
            return .uint64(v)
        case 11:  // i64
            let v = try readUInt64(data, cursor: &cursor)
            return .int64(Int64(bitPattern: v))
        case 12:  // f64
            let v = try readFloat64(data, cursor: &cursor)
            return .float64(v)
        default:
            throw GGUFError.unknownValueType(raw: valueTypeRaw, at: cursor)
        }
    }

    @inline(__always)
    private static func alignUp(_ value: Int, to alignment: Int) -> Int {
        let r = value % alignment
        return r == 0 ? value : value + (alignment - r)
    }
}

#endif // canImport(Metal)
