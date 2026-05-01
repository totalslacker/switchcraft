// SPDX-License-Identifier: Apache-2.0
//
// Public types for the SwitchcraftMetal GGUF reader (issue #59).
//
// The reader parses GGUF v3 (`ggml/docs/gguf.md`) and surfaces the
// subset Switchcraft consumes: weight tensors in `Q4_K`, `F32`, and
// `F16`, plus a typed subset of metadata KV values (string / int /
// float / bool — arrays and small integer widths are parsed-but-not-
// exposed; see ADR 016 §"Metadata KV surface" for the rationale).
//
// All types are `Sendable` so they can cross actor boundaries (the
// reader itself is an actor).

#if canImport(Metal)

import Foundation
import Metal

/// GGUF tensor data type. Only the dtypes the Switchcraft Phase 2 Metal
/// embedder consumes are surfaced; other ggml dtypes (Q5_K, Q6_K, etc.)
/// throw a descriptive ``GGUFError/unsupportedDType`` on read.
@_spi(SwitchcraftMetal)
public enum GGUFDType: Sendable, Equatable {
    /// 4-bit K-quant — 256-element super-blocks, FP16 super-block scale +
    /// FP16 super-block min, 12 bytes of packed 6-bit sub-scales/mins,
    /// 128 bytes of packed 4-bit weights = 144 bytes per 256 weights.
    /// Format pinned to the catalogue (`docs/porting/ggml-t5.md` §Q4_K).
    case q4k
    /// 32-bit IEEE-754 float, little-endian.
    case f32
    /// 16-bit IEEE-754 half-float (binary16), little-endian.
    case f16

    /// Raw ggml `enum ggml_type` integer values.
    /// `GGML_TYPE_F32 = 0`, `GGML_TYPE_F16 = 1`, `GGML_TYPE_Q4_K = 12`
    /// (per `ggml/include/ggml.h` at the catalogue's pinned commit).
    static func from(rawGGMLType raw: UInt32) -> GGUFDType? {
        switch raw {
        case 0: return .f32
        case 1: return .f16
        case 12: return .q4k
        default: return nil
        }
    }

    /// Bytes of storage required for `elementCount` elements of this
    /// dtype, given a Q4_K super-block size of 256.
    func storageBytes(for elementCount: Int) -> Int {
        switch self {
        case .f32: return elementCount * 4
        case .f16: return elementCount * 2
        case .q4k:
            precondition(
                elementCount % 256 == 0,
                "Q4_K element count must be a multiple of 256 (got \(elementCount))"
            )
            return (elementCount / 256) * 144
        }
    }
}

/// A single tensor loaded from a GGUF file.
///
/// The `buffer` holds the raw on-disk bytes for the tensor — for `Q4_K`
/// that means the packed super-blocks (no eager dequantisation), for
/// `F32` / `F16` it is the as-on-disk little-endian float array. The
/// matmul kernel landing in sub-issue #60 is responsible for reading
/// from this buffer in the layout the catalogue documents.
@_spi(SwitchcraftMetal)
public struct GGUFTensor: @unchecked Sendable {
    public let name: String
    public let dtype: GGUFDType
    /// Tensor shape with the **GGUF on-disk convention**: dims are
    /// stored fastest-varying-first (i.e. for a row-major MxN matrix
    /// the on-disk shape is `[N, M]`). Callers that want the row-major
    /// shape should reverse this array. Preserved here to match the
    /// upstream tooling output exactly.
    public let shape: [Int]
    /// `MTLBuffer` is documented thread-safe by Apple (see ADR 015
    /// §"Sendable policy"); the Swift annotation is missing because
    /// `Metal.framework` predates Swift concurrency. The struct stores
    /// only `let` properties so `@unchecked Sendable` is justified.
    public let buffer: MTLBuffer
    public let elementCount: Int
}

/// Typed subset of GGUF KV values exposed by the reader.
///
/// GGUF v3 has 13 value types; the reader parses all of them
/// (so unknown KVs do not desync the file offset) but surfaces only the
/// small set that has a clear use in Switchcraft. Adding more variants
/// is non-breaking. See ADR 016 §"Metadata KV surface".
@_spi(SwitchcraftMetal)
public enum GGUFValue: Sendable, Equatable {
    case string(String)
    case int64(Int64)
    case uint64(UInt64)
    case float32(Float)
    case float64(Double)
    case bool(Bool)
}

/// Errors thrown by the GGUF reader.
@_spi(SwitchcraftMetal)
public enum GGUFError: Error, CustomStringConvertible, Equatable {
    /// The file's first four bytes are not `"GGUF"`.
    case invalidMagic
    /// The file's version field is not 3. Found version is reported.
    case unsupportedVersion(found: UInt32)
    /// A read attempted to consume more bytes than the file contains.
    case truncated(at: Int, needed: Int)
    /// A KV key or value-type byte sequence was not valid UTF-8 or the
    /// declared length exceeded the file size.
    case malformedString(at: Int)
    /// A KV value type or array element type was outside the documented
    /// range `[0, 12]`.
    case unknownValueType(raw: UInt32, at: Int)
    /// A tensor declared a ggml dtype the reader does not support.
    /// Includes the offending tensor name so the message is actionable.
    case unsupportedDType(name: String, raw: UInt32)
    /// A tensor's declared element count is not consistent with its
    /// dtype's storage requirements (e.g. Q4_K count not a multiple of
    /// 256), or the tensor data offset would read past EOF.
    case malformedTensor(name: String, reason: String)
    /// File-level fields (`general.alignment`, tensor shape dims, KV
    /// counts) overflowed or contained an invalid value.
    case invalidHeader(reason: String)
    /// Failed to allocate an `MTLBuffer` for a tensor.
    case bufferAllocationFailed(name: String, bytes: Int)
    /// The named tensor is not present in the file.
    case tensorNotFound(name: String)
    /// The file could not be opened or read from disk.
    case ioError(String)

    public var description: String {
        switch self {
        case .invalidMagic:
            return "GGUF: bad magic — first 4 bytes are not 'GGUF'"
        case .unsupportedVersion(let v):
            return "GGUF: unsupported version \(v) — only v3 is supported"
        case .truncated(let at, let needed):
            return "GGUF: truncated read at offset \(at), needed \(needed) more bytes"
        case .malformedString(let at):
            return "GGUF: malformed string at offset \(at)"
        case .unknownValueType(let raw, let at):
            return "GGUF: unknown value type \(raw) at offset \(at)"
        case .unsupportedDType(let name, let raw):
            return "GGUF: unsupported tensor dtype \(raw) for tensor '\(name)' — only F32 (0), F16 (1), Q4_K (12) are supported"
        case .malformedTensor(let name, let reason):
            return "GGUF: malformed tensor '\(name)' — \(reason)"
        case .invalidHeader(let reason):
            return "GGUF: invalid header — \(reason)"
        case .bufferAllocationFailed(let name, let bytes):
            return "GGUF: MTLBuffer allocation failed for tensor '\(name)' (\(bytes) bytes)"
        case .tensorNotFound(let name):
            return "GGUF: tensor '\(name)' not found"
        case .ioError(let msg):
            return "GGUF: I/O error — \(msg)"
        }
    }
}

#endif // canImport(Metal)
