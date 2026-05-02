// SPDX-License-Identifier: Apache-2.0
//
// Tests for the SwitchcraftMetal GGUF reader (issue #59).
//
// Two layers of coverage:
//
//   1. **Header / KV / tensor-info parsing** — hand-crafted in-memory
//      GGUF v3 byte buffers exercise the magic check, version field,
//      KV value-type handling, tensor info parsing, alignment override,
//      and the unsupported-dtype error path. These run unconditionally.
//
//   2. **Round-trip parity** — env-gated on `SWITCHCRAFT_XTR_GGUF`.
//      Reads the published Q4_K asset, CPU-dequantises three sampled
//      tensors via `Q4KDecode`, and (when a reference FP32 dump is
//      provided via `SWITCHCRAFT_XTR_GGUF_FP32_REF`) compares the
//      decoded values bit-for-bit against the reference. Without the
//      reference dump the test falls back to a structural sanity
//      check (correct shape and finite values) so the suite stays
//      useful even if the optional reference fixture is absent.
//
// See the issue body and ADR 016 for the bit-equal-decode rationale.

#if canImport(Metal)

import Foundation
import Metal
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// MARK: - GGUF v3 byte-buffer builder

/// Tiny GGUF v3 writer used by the in-memory parsing tests.
///
/// Only encodes the value types the tests actually need
/// (`string`, `uint32`, `uint64`, `bool`); add cases here if a future
/// test needs a different metadata type.
struct GGUFFixture {
    var version: UInt32 = 3
    var alignment: Int = 32
    var metadata: [(String, GGUFFixtureValue)] = []
    var tensors: [GGUFFixtureTensor] = []

    enum GGUFFixtureValue {
        case string(String)
        case uint32(UInt32)
        case uint64(UInt64)
        case bool(Bool)
    }

    struct GGUFFixtureTensor {
        let name: String
        /// On-disk shape (fastest-varying-first). Element count is the
        /// product of the dims.
        let shape: [UInt64]
        /// Raw ggml type integer (0=F32, 1=F16, 12=Q4_K, etc.).
        let typeRaw: UInt32
        /// Raw bytes for the tensor (must match the dtype's storage
        /// requirement; the writer does not check, deliberately, so a
        /// fixture can encode an "unsupported dtype" tensor without
        /// triggering an internal precondition).
        let bytes: Data
    }

    func build() -> Data {
        var out = Data()
        // Magic + version + counts.
        appendU32(0x4655_4747, to: &out)  // 'GGUF' little-endian
        appendU32(version, to: &out)
        appendU64(UInt64(tensors.count), to: &out)
        appendU64(UInt64(metadata.count), to: &out)

        // KV block.
        for (key, value) in metadata {
            appendString(key, to: &out)
            switch value {
            case .string(let s):
                appendU32(8, to: &out)
                appendString(s, to: &out)
            case .uint32(let v):
                appendU32(4, to: &out)
                appendU32(v, to: &out)
            case .uint64(let v):
                appendU32(10, to: &out)
                appendU64(v, to: &out)
            case .bool(let b):
                appendU32(7, to: &out)
                out.append(b ? 1 : 0)
            }
        }

        // Tensor info block — write a placeholder offset for each
        // tensor so we can patch in the real offset once we know where
        // the data block starts.
        var offsetPatchSlots: [Int] = []
        for t in tensors {
            appendString(t.name, to: &out)
            appendU32(UInt32(t.shape.count), to: &out)
            for d in t.shape { appendU64(d, to: &out) }
            appendU32(t.typeRaw, to: &out)
            offsetPatchSlots.append(out.count)
            appendU64(0, to: &out)  // placeholder
        }

        // Pad to alignment.
        let infoEnd = out.count
        let padded = alignUp(infoEnd, to: alignment)
        if padded > infoEnd {
            out.append(Data(count: padded - infoEnd))
        }

        // Tensor data block — pack each tensor at its current offset
        // (relative to `padded`) and patch the offset slot.
        var relOffset = 0
        for (idx, t) in tensors.enumerated() {
            // ggml conventionally pads each tensor to alignment too,
            // but the relative offset only needs to be alignment-
            // aligned; the data block start is already aligned, so
            // bumping `relOffset` to a multiple of alignment satisfies
            // both constraints simultaneously.
            let alignedRel = alignUp(relOffset, to: alignment)
            if alignedRel > relOffset {
                out.append(Data(count: alignedRel - relOffset))
                relOffset = alignedRel
            }
            // Patch in the real offset.
            let slot = offsetPatchSlots[idx]
            patchU64(UInt64(relOffset), at: slot, in: &out)
            out.append(t.bytes)
            relOffset += t.bytes.count
        }
        return out
    }

    private func appendU32(_ v: UInt32, to data: inout Data) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    private func appendU64(_ v: UInt64, to data: inout Data) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    private func appendString(_ s: String, to data: inout Data) {
        let bytes = Array(s.utf8)
        appendU64(UInt64(bytes.count), to: &data)
        data.append(contentsOf: bytes)
    }

    private func patchU64(_ v: UInt64, at offset: Int, in data: inout Data) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { src in
            for i in 0..<8 {
                data[offset + i] = src[i]
            }
        }
    }

    private func alignUp(_ v: Int, to a: Int) -> Int {
        let r = v % a
        return r == 0 ? v : v + (a - r)
    }
}

/// Write a fixture to a temp file and return its URL. The caller is
/// responsible for cleanup; tests use a per-test directory under
/// `FileManager.default.temporaryDirectory` so leftover files are easy
/// to identify.
func writeFixture(_ fixture: GGUFFixture, named name: String = "fixture.gguf") throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("switchcraft-gguf-tests", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent("\(UUID().uuidString)-\(name)")
    try fixture.build().write(to: url)
    return url
}

// MARK: - Test fixtures shared between cases

/// Builds a 256-element zero-valued Q4_K block (144 bytes of zeros). All
/// FP16 fields, all 6-bit sub-scales, and all 4-bit weights are zero,
/// so the dequantised tensor is all zeros — useful for round-trip and
/// shape tests that don't care about specific values.
private func zeroQ4KBlock() -> Data {
    Data(count: 144)
}

private func device() throws -> MTLDevice {
    guard let dev = MTLCreateSystemDefaultDevice() else {
        Issue.record("MTLCreateSystemDefaultDevice() returned nil")
        throw GGUFError.ioError("no Metal device")
    }
    return dev
}

// MARK: - Header / parse tests (unconditional)

@Suite("GGUFReader header & KV parsing")
struct GGUFReaderHeaderTests {

    @Test("zero-tensor file with assorted KV types")
    func zeroTensors() async throws {
        var fixture = GGUFFixture()
        fixture.metadata = [
            ("general.architecture", .string("t5")),
            ("general.alignment", .uint32(32)),
            ("t5.context_length", .uint64(512)),
            ("t5.use_parallel_residual", .bool(false)),
        ]
        let url = try writeFixture(fixture)
        let dev = try device()
        let reader = try GGUFReader(url: url, device: dev)
        let names = await reader.tensorNames()
        #expect(names.isEmpty)
        let md = await reader.metadata
        #expect(md["general.architecture"] == .string("t5"))
        #expect(md["t5.context_length"] == .uint64(512))
        #expect(md["t5.use_parallel_residual"] == .bool(false))
        // u32 values surface as uint64.
        #expect(md["general.alignment"] == .uint64(32))
        try? FileManager.default.removeItem(at: url)
    }

    @Test("single F32 tensor round-trips raw bytes")
    func singleF32Tensor() async throws {
        var fixture = GGUFFixture()
        // 4 floats = 16 bytes.
        let values: [Float] = [1.0, -2.5, 3.14, 0.0]
        let bytes = values.withUnsafeBufferPointer { Data(buffer: $0) }
        fixture.tensors = [
            .init(name: "weights", shape: [4], typeRaw: 0, bytes: bytes)
        ]
        let url = try writeFixture(fixture)
        let dev = try device()
        let reader = try GGUFReader(url: url, device: dev)
        let names = await reader.tensorNames()
        #expect(names == ["weights"])
        let tensor = try await reader.tensor("weights")
        #expect(tensor.dtype == .f32)
        #expect(tensor.shape == [4])
        #expect(tensor.elementCount == 4)
        // Read back from the MTLBuffer.
        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: 4)
        let roundTrip = Array(UnsafeBufferPointer(start: ptr, count: 4))
        #expect(roundTrip == values)
        try? FileManager.default.removeItem(at: url)
    }

    @Test("mixed F32 / F16 / Q4_K tensors parse and resolve")
    func mixedDtypes() async throws {
        var fixture = GGUFFixture()
        // F32 tensor: 8 floats.
        let f32: [Float] = (0..<8).map { Float($0) }
        let f32Bytes = f32.withUnsafeBufferPointer { Data(buffer: $0) }
        // F16 tensor: 4 halves.
        let f16: [Float16] = [Float16(0.5), Float16(1.0), Float16(-0.5), Float16(2.0)]
        let f16Bytes = f16.withUnsafeBufferPointer { Data(buffer: $0) }
        // Q4_K tensor: 1 super-block (256 elements, 144 bytes).
        let q4kBytes = zeroQ4KBlock()

        fixture.tensors = [
            .init(name: "a_f32", shape: [8], typeRaw: 0, bytes: f32Bytes),
            .init(name: "b_f16", shape: [4], typeRaw: 1, bytes: f16Bytes),
            .init(name: "c_q4k", shape: [256], typeRaw: 12, bytes: q4kBytes),
        ]
        let url = try writeFixture(fixture)
        let dev = try device()
        let reader = try GGUFReader(url: url, device: dev)

        let names = await reader.tensorNames()
        #expect(names == ["a_f32", "b_f16", "c_q4k"])

        let a = try await reader.tensor("a_f32")
        #expect(a.dtype == .f32)
        #expect(a.elementCount == 8)
        #expect(a.buffer.length == 32)

        let b = try await reader.tensor("b_f16")
        #expect(b.dtype == .f16)
        #expect(b.elementCount == 4)
        #expect(b.buffer.length == 8)

        let c = try await reader.tensor("c_q4k")
        #expect(c.dtype == .q4k)
        #expect(c.elementCount == 256)
        #expect(c.buffer.length == 144)

        // Cache hit on second call.
        let aAgain = try await reader.tensor("a_f32")
        #expect(aAgain.buffer === a.buffer)

        try? FileManager.default.removeItem(at: url)
    }

    @Test("unsupported dtype throws a descriptive error naming the tensor")
    func unsupportedDType() async throws {
        var fixture = GGUFFixture()
        // Q5_K = ggml type 13. Bytes are arbitrary (the reader rejects
        // before it reaches the data block).
        fixture.tensors = [
            .init(name: "bogus", shape: [256], typeRaw: 13, bytes: Data(count: 176))
        ]
        let url = try writeFixture(fixture)
        let dev = try device()
        do {
            _ = try GGUFReader(url: url, device: dev)
            Issue.record("expected unsupportedDType error, init succeeded")
        } catch let GGUFError.unsupportedDType(name, raw) {
            #expect(name == "bogus")
            #expect(raw == 13)
        } catch {
            Issue.record("expected unsupportedDType, got \(error)")
        }
        try? FileManager.default.removeItem(at: url)
    }

    @Test("invalid magic is rejected")
    func invalidMagic() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("switchcraft-gguf-tests", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("\(UUID().uuidString)-bad-magic.gguf")
        try Data([0x00, 0x00, 0x00, 0x00,
                  0x03, 0x00, 0x00, 0x00,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0]).write(to: url)
        let dev = try device()
        do {
            _ = try GGUFReader(url: url, device: dev)
            Issue.record("expected invalidMagic, init succeeded")
        } catch GGUFError.invalidMagic {
            // pass
        } catch {
            Issue.record("expected invalidMagic, got \(error)")
        }
        try? FileManager.default.removeItem(at: url)
    }

    @Test("unsupported version is rejected (v1)")
    func unsupportedVersion() async throws {
        var fixture = GGUFFixture()
        fixture.version = 1  // v1 predates v2/v3; must be rejected
        let url = try writeFixture(fixture)
        let dev = try device()
        do {
            _ = try GGUFReader(url: url, device: dev)
            Issue.record("expected unsupportedVersion, init succeeded")
        } catch let GGUFError.unsupportedVersion(found) {
            #expect(found == 1)
        } catch {
            Issue.record("expected unsupportedVersion, got \(error)")
        }
        try? FileManager.default.removeItem(at: url)
    }

    @Test("GGUF v2 file is accepted and loadedVersion is 2")
    func ggufV2Accepted() async throws {
        var fixture = GGUFFixture()
        fixture.version = 2
        let url = try writeFixture(fixture)
        let dev = try device()
        do {
            let reader = try GGUFReader(url: url, device: dev)
            let v = await reader.loadedVersion
            #expect(v == 2)
        } catch {
            Issue.record("expected GGUF v2 to be accepted, got \(error)")
        }
        try? FileManager.default.removeItem(at: url)
    }

    @Test("GGUF v3 file is accepted and loadedVersion is 3")
    func ggufV3Accepted() async throws {
        var fixture = GGUFFixture()
        fixture.version = 3  // default, but explicit here for clarity
        let url = try writeFixture(fixture)
        let dev = try device()
        do {
            let reader = try GGUFReader(url: url, device: dev)
            let v = await reader.loadedVersion
            #expect(v == 3)
        } catch {
            Issue.record("expected GGUF v3 to be accepted, got \(error)")
        }
        try? FileManager.default.removeItem(at: url)
    }

    @Test("array KVs are parsed and skipped without desyncing the file")
    func arrayKVHandling() async throws {
        // Build manually to include an array KV (which the high-level
        // GGUFFixture builder doesn't expose).
        var raw = Data()
        // Magic + version + counts.
        var le32: UInt32 = 0x4655_4747
        withUnsafeBytes(of: &le32) { raw.append(contentsOf: $0) }
        le32 = 3
        withUnsafeBytes(of: &le32) { raw.append(contentsOf: $0) }
        var le64: UInt64 = 0  // n_tensors
        withUnsafeBytes(of: &le64) { raw.append(contentsOf: $0) }
        le64 = 2  // n_kv
        withUnsafeBytes(of: &le64) { raw.append(contentsOf: $0) }

        // KV 1: array of 3 strings.
        let key1 = "tokens"
        le64 = UInt64(key1.utf8.count)
        withUnsafeBytes(of: &le64) { raw.append(contentsOf: $0) }
        raw.append(contentsOf: Array(key1.utf8))
        le32 = 9  // array
        withUnsafeBytes(of: &le32) { raw.append(contentsOf: $0) }
        le32 = 8  // element type = string
        withUnsafeBytes(of: &le32) { raw.append(contentsOf: $0) }
        le64 = 3  // count
        withUnsafeBytes(of: &le64) { raw.append(contentsOf: $0) }
        for s in ["a", "bb", "ccc"] {
            le64 = UInt64(s.utf8.count)
            withUnsafeBytes(of: &le64) { raw.append(contentsOf: $0) }
            raw.append(contentsOf: Array(s.utf8))
        }
        // KV 2: a string we expect to see — confirms the array KV
        // didn't desync the cursor.
        let key2 = "general.architecture"
        le64 = UInt64(key2.utf8.count)
        withUnsafeBytes(of: &le64) { raw.append(contentsOf: $0) }
        raw.append(contentsOf: Array(key2.utf8))
        le32 = 8  // string
        withUnsafeBytes(of: &le32) { raw.append(contentsOf: $0) }
        let val = "t5"
        le64 = UInt64(val.utf8.count)
        withUnsafeBytes(of: &le64) { raw.append(contentsOf: $0) }
        raw.append(contentsOf: Array(val.utf8))

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("switchcraft-gguf-tests", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("\(UUID().uuidString)-array.gguf")
        try raw.write(to: url)
        let dev = try device()
        let reader = try GGUFReader(url: url, device: dev)
        let md = await reader.metadata
        // Array KV is consumed but not surfaced.
        #expect(md["tokens"] == nil)
        // String KV after the array round-trips correctly.
        #expect(md["general.architecture"] == .string("t5"))
        try? FileManager.default.removeItem(at: url)
    }

    @Test("missing tensor throws tensorNotFound")
    func tensorNotFound() async throws {
        var fixture = GGUFFixture()
        fixture.tensors = [
            .init(name: "real", shape: [4], typeRaw: 0,
                  bytes: Data(count: 16))
        ]
        let url = try writeFixture(fixture)
        let dev = try device()
        let reader = try GGUFReader(url: url, device: dev)
        do {
            _ = try await reader.tensor("ghost")
            Issue.record("expected tensorNotFound")
        } catch let GGUFError.tensorNotFound(name) {
            #expect(name == "ghost")
        } catch {
            Issue.record("expected tensorNotFound, got \(error)")
        }
        try? FileManager.default.removeItem(at: url)
    }
}

// MARK: - SPI smoke test

@Suite("SwitchcraftMetal SPI smoke")
struct SwitchcraftMetalSPISmokeTests {
    @Test("MetalContext.shared is reachable through @_spi import",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil,
                   "Metal unavailable on this host"))
    func metalContextReachable() throws {
        // The SPI hatch on MetalContext is what lets SwitchcraftMetal
        // (and these tests) reach the type. If this stops compiling,
        // sub-issues #60–#64 will all break — keep the test as a tripwire.
        let ctx = try #require(MetalContext.shared,
                               "MetalContext.shared was nil on a Metal-capable host")
        _ = ctx.device
        _ = ctx.queue
        _ = ctx.library
    }
}

// MARK: - Round-trip parity (env-gated)

@Suite("GGUFReader round-trip parity",
       .enabled(if: GGUFAsset.isAvailable,
                "Set SWITCHCRAFT_XTR_GGUF=<path> to enable round-trip parity tests"))
struct GGUFReaderParityTests {

    /// The three tensors the issue body calls out for sampling.
    static let sampledTensorNames = [
        "encoder.embed_tokens.weight",
        "encoder.block.0.layer.0.SelfAttention.q.weight",
        "encoder.final_layer_norm.weight",
    ]

    @Test("Q4_K tensors decode to bit-equal FP32 vs reference dump",
          arguments: sampledTensorNames)
    func bitEqualParity(name: String) async throws {
        let url = try #require(GGUFAsset.url, "GGUFAsset.url was nil")
        let dev = try #require(MTLCreateSystemDefaultDevice(),
                               "Metal unavailable")
        let reader = try GGUFReader(url: url, device: dev)
        let availableNames = Set(await reader.tensorNames())
        // The asset shape is fixed once we have it locally — but be
        // gentle if a reader's local copy uses slightly different
        // tensor naming (e.g. an alternate quantizer's prefix). Skip
        // the row in that case rather than failing.
        try #require(
            availableNames.contains(name),
            "Tensor '\(name)' not present in this GGUF. Is SWITCHCRAFT_XTR_GGUF pointing at the canonical xtr-base-en asset?"
        )
        let tensor = try await reader.tensor(name)

        // CPU dequantise (or read directly for F32 / F16).
        let decoded: [Float]
        switch tensor.dtype {
        case .q4k:
            let raw = UnsafeRawBufferPointer(
                start: tensor.buffer.contents(),
                count: tensor.buffer.length
            )
            var out = [Float](repeating: 0, count: tensor.elementCount)
            out.withUnsafeMutableBufferPointer { dst in
                Q4KDecode.dequantise(blocks: raw, into: dst)
            }
            decoded = out
        case .f32:
            let p = tensor.buffer.contents().bindMemory(
                to: Float.self, capacity: tensor.elementCount
            )
            decoded = Array(UnsafeBufferPointer(start: p, count: tensor.elementCount))
        case .f16:
            let p = tensor.buffer.contents().bindMemory(
                to: Float16.self, capacity: tensor.elementCount
            )
            decoded = (0..<tensor.elementCount).map { Float(p[$0]) }
        }

        // Structural sanity that every reader run can perform.
        #expect(decoded.count == tensor.elementCount)
        for v in decoded { #expect(v.isFinite) }

        // Bit-equal parity vs reference dump, when one is provided.
        // The dump path is optional because not every contributor will
        // have produced one yet (the dumper script is documented in
        // ADR 016 but not yet committed). When the env var is unset,
        // the structural-sanity above is the gate.
        guard
            let indexURL = GGUFFP32Reference.indexURL,
            let blobURL = GGUFFP32Reference.blobURL
        else {
            return
        }
        struct Sidecar: Decodable {
            let tensors: [Entry]
            struct Entry: Decodable {
                let name: String
                let shape: [Int]
                let byteOffset: Int
                let byteLength: Int
            }
        }
        let sidecar = try JSONDecoder().decode(
            Sidecar.self, from: try Data(contentsOf: indexURL)
        )
        guard let entry = sidecar.tensors.first(where: { $0.name == name }) else {
            // Missing entry — treat as "no reference for this tensor"
            // and stop short, same as if the dump file was absent.
            return
        }
        #expect(entry.byteLength == tensor.elementCount * 4,
                "reference dump length mismatch for \(name)")
        let blob = try Data(contentsOf: blobURL)
        var reference = [Float](repeating: 0, count: tensor.elementCount)
        reference.withUnsafeMutableBytes { dst in
            blob.copyBytes(
                to: dst.bindMemory(to: UInt8.self),
                from: entry.byteOffset..<(entry.byteOffset + entry.byteLength)
            )
        }
        // Bit-equal: the catalogue documents that `dequantize_row_q4_K`
        // is a deterministic FP32 multiply/add sequence with no FMA
        // fusion, so the Swift port should reproduce the exact bit
        // pattern. If this assertion ever fails in practice, the
        // first-line investigation is whether the reference dump used
        // a different op order (e.g. ggml's sgemm path); only after
        // ruling that out should we consider relaxing this gate to
        // `≤ 1 ULP relative` — see the ADR 016 §"Bit-equal Q4_K
        // decode" rationale.
        var firstMismatch: Int? = nil
        for i in 0..<tensor.elementCount {
            if decoded[i].bitPattern != reference[i].bitPattern {
                firstMismatch = i
                break
            }
        }
        #expect(firstMismatch == nil,
                "tensor \(name): first bit-mismatch at element \(firstMismatch ?? -1)")
    }
}

#endif // canImport(Metal)
