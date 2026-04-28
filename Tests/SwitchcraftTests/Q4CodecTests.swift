import Foundation
import Testing
import SwitchcraftCore

@Suite("Q4 Residual Codec")
struct Q4CodecTests {

    // MARK: - Pack / Unpack

    @Test("packQ4 places the first value in the high nibble")
    func packLayout() {
        let bytes = Q4Codec.packQ4([1, 2, 3, 4])
        #expect(Array(bytes) == [0x12, 0x34])
    }

    @Test("packQ4 masks values to the low 4 bits")
    func packMask() {
        // 0x1F gets masked to 0x0F; 0xA0 gets masked to 0x00.
        let bytes = Q4Codec.packQ4([0x1F, 0xA0])
        #expect(Array(bytes) == [0xF0])
    }

    @Test("packQ4 → unpackQ4 round-trips through the dequant table")
    func packUnpackRoundTrip() {
        let levels = Array(0..<16) // 0,1,2,...,15
        let packed = Q4Codec.packQ4(levels)
        #expect(packed.count == levels.count / 2)

        let unpacked = Q4Codec.unpackQ4(packed)
        // Each level should map to its dequant table entry.
        for (i, value) in unpacked.enumerated() {
            #expect(value == Q4Codec.dequantTable[i])
        }
    }

    // MARK: - Quantize / Dequantize (linear)

    @Test("quantize maps the [-1, 1] endpoints to extremes")
    func quantizeEndpoints() {
        #expect(Q4Codec.quantize(-1.0, bits: 4) == 0)
        #expect(Q4Codec.quantize(1.0, bits: 4) == 15)
    }

    @Test("quantize clamps out-of-range inputs")
    func quantizeClamps() {
        #expect(Q4Codec.quantize(-2.0, bits: 4) == 0)
        #expect(Q4Codec.quantize(2.0, bits: 4) == 15)
    }

    @Test("dequantize is the inverse of quantize at integer levels")
    func dequantizeInverse() {
        for q in 0...15 {
            let x = Q4Codec.dequantize(q, bits: 4)
            #expect(Q4Codec.quantize(x, bits: 4) == q)
        }
    }

    // MARK: - Compander

    @Test("compand fixes the origin and the unit endpoints")
    func companderFixedPoints() {
        #expect(Q4Codec.compand(0.0) == 0.0)
        // x = 0.25 → scaled = 1 → log(1+255)/log(256) = 1.
        let high = Q4Codec.compand(0.25)
        #expect(abs(high - 1.0) < 1e-5)
        let low = Q4Codec.compand(-0.25)
        #expect(abs(low + 1.0) < 1e-5)
    }

    @Test("invCompand undoes compand within FP tolerance")
    func companderRoundTrip() {
        // Sample residual-shaped values in the typical operating range.
        let samples: [Float] = stride(from: -0.25, through: 0.25, by: 0.01).map(Float.init)
        var maxErr: Float = 0
        for x in samples {
            let y = Q4Codec.invCompand(Q4Codec.compand(x))
            maxErr = max(maxErr, abs(x - y))
        }
        #expect(maxErr < 1e-5, "max compand round-trip error: \(maxErr)")
    }

    // MARK: - Decode table

    @Test("dequant table is monotone and roughly symmetric")
    func decodeTableShape() {
        let table = Q4Codec.dequantTable
        #expect(table.count == 16)
        // Strictly increasing: small-magnitude region is densely packed near zero.
        for i in 1..<16 {
            #expect(table[i] > table[i - 1])
        }
        // Endpoints sit near ±0.25 (the original ×4 input range).
        #expect(abs(table[0] + 0.25) < 1e-4)
        #expect(abs(table[15] - 0.25) < 1e-4)
        // Around the midpoint, values are small.
        #expect(abs(table[7]) < 0.05)
        #expect(abs(table[8]) < 0.05)
    }

    // MARK: - End-to-end residual round-trip

    @Test("encodeResiduals output is half the input length")
    func encodeSizing() {
        let values = [Float](repeating: 0.0, count: 128)
        let encoded = Q4Codec.encodeResiduals(values)
        #expect(encoded.count == 64)
    }

    @Test("residual round-trip MSE is small for typical inputs")
    func residualRoundTripMSE() {
        // Deterministic pseudo-random residuals in the [-0.25, 0.25] range.
        var rng = SystemRandomNumberGenerator()
        var values = [Float]()
        for _ in 0..<2048 {
            let u = Float(rng.next()) / Float(UInt64.max)  // [0, 1]
            values.append((u - 0.5) * 0.5)                 // ~[-0.25, 0.25]
        }

        let encoded = Q4Codec.encodeResiduals(values)
        let decoded = Q4Codec.decodeResiduals(encoded)

        #expect(decoded.count == values.count)

        var sse: Double = 0
        for i in 0..<values.count {
            let d = Double(values[i] - decoded[i])
            sse += d * d
        }
        let mse = sse / Double(values.count)
        // Companded Q4 should keep MSE well under 0.001 for residuals
        // distributed in [-0.25, 0.25].
        #expect(mse < 1e-3, "round-trip MSE: \(mse)")
    }

    @Test("residual round-trip produces values bounded by the input scale")
    func residualBounded() {
        let values: [Float] = (0..<128).map { i in
            let t = Float(i) / 127.0          // 0..1
            return (t - 0.5) * 0.5            // -0.25..0.25
        }
        let decoded = Q4Codec.decodeResiduals(Q4Codec.encodeResiduals(values))
        for v in decoded {
            #expect(v >= -0.26 && v <= 0.26, "decoded out of range: \(v)")
        }
    }

    // MARK: - Determinism

    @Test("encoding the same input twice yields the same bytes")
    func deterministic() {
        let values: [Float] = (0..<256).map { Float($0) / 1024.0 - 0.125 }
        let a = Q4Codec.encodeResiduals(values)
        let b = Q4Codec.encodeResiduals(values)
        #expect(a == b)
    }
}
