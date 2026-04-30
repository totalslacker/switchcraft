// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Q4 codec for residual vectors stored in bucket records.
///
/// Pipeline (port of `to_q4_bytes` / `from_companded_q4_bytes` in upstream
/// `packops.rs`):
///
///     encode: residual ── μ-law compand ── quantize(4 bits) ── pack 2/byte ──▶ Data
///     decode: Data ── unpack ── 16-entry dequant table ──▶ residual
///
/// Companding is the standard μ-law transform with `companderParam = 255`
/// applied to inputs scaled by 4. Residuals of normalised embeddings rarely
/// exceed roughly ±0.26, so the ×4 scale stretches them across the full
/// `[-1, 1]` range before quantization.
///
/// Round-trip is lossy by construction (4 bits = 16 levels) but companding
/// keeps quantization error small near zero, where most residuals live.
public enum Q4Codec {

    /// Pre-scale applied before companding. Residuals are typically in
    /// `[-0.26, 0.26]`; multiplying by 4 maps them onto roughly `[-1, 1]`.
    public static let scaleParam: Float = 4.0

    /// μ-law parameter. With `companderParam = 255`, companding maps
    /// `[-1, 1] → [-1, 1]` non-linearly, expanding the small-magnitude
    /// region.
    public static let companderParam: Float = 255.0

    // MARK: - Compander

    /// μ-law compand a single value. `sign(x) * log(1 + μ|4x|) / log(1 + μ)`.
    public static func compand(_ x: Float) -> Float {
        let scaled = x * scaleParam
        let magnitude = abs(scaled)
        let signF: Float = scaled < 0 ? -1 : (scaled > 0 ? 1 : 0)
        // Compute the denominator in f64 to match the upstream layout.
        let invDenominator = Float(1.0 / Foundation.log(1.0 + Double(companderParam)))
        return signF * log(1.0 + companderParam * magnitude) * invDenominator
    }

    /// Inverse of `compand`: `sign(y) * ((1+μ)^|y| - 1) / μ / 4`.
    public static func invCompand(_ y: Float) -> Float {
        let magnitude = abs(y)
        let signF: Float = y < 0 ? -1 : (y > 0 ? 1 : 0)
        let scaled = signF * (pow(1.0 + companderParam, magnitude) - 1.0) / companderParam
        return scaled / scaleParam
    }

    // MARK: - Quantize / Dequantize

    /// Map a value in `[-1, 1]` to an integer level in `[0, 2^bits - 1]`.
    /// Uses round-half-away-from-zero, matching Rust's `f32::round`.
    public static func quantize(_ x: Float, bits: Int) -> Int {
        let qmax = Float((1 << bits) - 1)
        let scale = qmax / 2.0
        let zp = qmax / 2.0
        let q = (x * scale + zp).rounded()
        return Int(min(max(q, 0), qmax))
    }

    /// Inverse of `quantize`. Maps `[0, 2^bits - 1]` to `[-1, 1]`.
    public static func dequantize(_ q: Int, bits: Int) -> Float {
        let qmax = Float((1 << bits) - 1)
        let scale: Float = 2.0 / qmax
        let zp = qmax / 2.0
        return (Float(q) - zp) * scale
    }

    // MARK: - Decode table

    /// 16-entry table mapping a Q4 nibble (0..15) directly to its
    /// dequantized + decompanded float, used on the decode hot path.
    public static let dequantTable: [Float] = {
        (0..<16).map { invCompand(dequantize($0, bits: 4)) }
    }()

    // MARK: - Pack / Unpack

    /// Pack a sequence of values already in `[0, 15]` into bytes — two
    /// nibbles per byte, high nibble first.
    ///
    /// Precondition: `values.count` must be even.
    public static func packQ4(_ values: [Int]) -> Data {
        precondition(values.count % 2 == 0, "Q4 packing requires an even number of values")
        var out = Data(count: values.count / 2)
        for i in stride(from: 0, to: values.count, by: 2) {
            let high = UInt8(values[i] & 0x0F)
            let low = UInt8(values[i + 1] & 0x0F)
            out[i / 2] = (high << 4) | low
        }
        return out
    }

    /// Unpack and dequantize a Q4 byte buffer using `dequantTable`.
    /// Output length is `bytes.count * 2`.
    public static func unpackQ4(_ bytes: Data) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(bytes.count * 2)
        let table = dequantTable
        for byte in bytes {
            let high = Int((byte >> 4) & 0x0F)
            let low = Int(byte & 0x0F)
            out.append(table[high])
            out.append(table[low])
        }
        return out
    }

    // MARK: - End-to-end

    /// Encode residual values: compand → quantize(4) → pack.
    /// Output length is `values.count / 2` bytes.
    ///
    /// Precondition: `values.count` must be even.
    public static func encodeResiduals(_ values: [Float]) -> Data {
        precondition(values.count % 2 == 0, "Q4 packing requires an even number of values")
        var levels = [Int]()
        levels.reserveCapacity(values.count)
        for v in values {
            levels.append(quantize(compand(v), bits: 4))
        }
        return packQ4(levels)
    }

    /// Decode residuals from a Q4 byte buffer.
    public static func decodeResiduals(_ bytes: Data) -> [Float] {
        unpackQ4(bytes)
    }
}
