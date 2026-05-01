// SPDX-License-Identifier: Apache-2.0
//
// Swift orchestration for the FP32 custom-Metal GEMM kernel. Issue #49
// scope: hand a `[Float]` row-major `A` and `B` to `matmul(...)`, get a
// `[Float]` row-major `C` back. Pipeline-state objects are cached per
// `(TG_M, TG_N)` tile shape so the threadgroup-sweep test suite doesn't pay
// a recompile cost on every iteration.
//
// All buffers use `.storageModeShared` — Apple Silicon's unified memory
// means there is no host/device copy and the readback is just a CPU pointer
// dereference once the command buffer completes. On the discrete-GPU
// platforms this prototype does not target (Intel Macs, x86 simulators)
// shared-mode buffers fall back to managed allocations; not relevant for
// this investigation.

#if canImport(Metal)

import Foundation
import Metal

/// Tile shape used to instantiate a `gemm_fp32` pipeline. One thread per
/// output element; `tgM × tgN` becomes the `MTLSize`'s thread-per-threadgroup
/// dimensions. The sweep test suite enumerates a list of these.
public struct MetalTile: Hashable, Sendable, Codable {
    public let tgM: Int
    public let tgN: Int

    public init(_ tgM: Int, _ tgN: Int) {
        self.tgM = tgM
        self.tgN = tgN
    }
}

public enum MetalMatmulError: Error, CustomStringConvertible {
    case noDeviceAvailable
    case libraryLoadFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .noDeviceAvailable:
            return "MTLCreateSystemDefaultDevice() returned nil — no Metal-capable GPU"
        case .libraryLoadFailed(let m): return "Metal library load failed: \(m)"
        case .functionNotFound(let n): return "Metal function not found: \(n)"
        case .pipelineCreationFailed(let m): return "Metal pipeline creation failed: \(m)"
        case .bufferAllocationFailed: return "MTLBuffer allocation returned nil"
        case .shapeMismatch(let m): return "shape mismatch: \(m)"
        }
    }
}

/// One-stop orchestrator for the prototype kernel. Construct once, call
/// `matmul(...)` many times. All public methods are non-blocking on the
/// Swift side except for the `waitUntilCompleted()` inside `matmul(...)`,
/// which is intentional for the benchmark harness's per-iteration timing.
public final class MetalMatmul: @unchecked Sendable {
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let library: MTLLibrary

    /// Pipeline cache keyed by `(TG_M, TG_N)`. Building the pipeline involves
    /// resolving function constants and compiling — non-trivial cost we don't
    /// want on the benchmark hot path.
    private var pipelines: [MetalTile: MTLComputePipelineState] = [:]
    private let cacheLock = NSLock()

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalMatmulError.noDeviceAvailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalMatmulError.libraryLoadFailed("makeCommandQueue() returned nil")
        }
        self.device = device
        self.queue = queue

        // Try the precompiled `default.metallib` first (Xcode-driven builds
        // produce one). SwiftPM 6 currently ships `.metal` files declared as
        // `.process()` resources as raw source rather than as a compiled
        // metallib, so we fall back to runtime source compilation when the
        // default library lookup fails. The fallback happens once at init —
        // pipeline-state caching keeps the per-call overhead at zero.
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            self.library = lib
        } else {
            guard
                let url = Bundle.module.url(forResource: "MetalMatmul", withExtension: "metal"),
                let source = try? String(contentsOf: url, encoding: .utf8)
            else {
                throw MetalMatmulError.libraryLoadFailed("MetalMatmul.metal not found in bundle")
            }
            do {
                self.library = try device.makeLibrary(source: source, options: nil)
            } catch {
                throw MetalMatmulError.libraryLoadFailed(String(describing: error))
            }
        }
    }

    /// Resolve (or build and cache) the pipeline state for the given tile.
    public func pipeline(for tile: MetalTile) throws -> MTLComputePipelineState {
        cacheLock.lock()
        if let cached = pipelines[tile] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let constants = MTLFunctionConstantValues()
        var tgM = UInt32(tile.tgM)
        var tgN = UInt32(tile.tgN)
        constants.setConstantValue(&tgM, type: .uint, index: 0)
        constants.setConstantValue(&tgN, type: .uint, index: 1)

        let function: MTLFunction
        do {
            function = try library.makeFunction(name: "gemm_fp32", constantValues: constants)
        } catch {
            throw MetalMatmulError.functionNotFound("gemm_fp32: \(error)")
        }

        let state: MTLComputePipelineState
        do {
            state = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalMatmulError.pipelineCreationFailed(String(describing: error))
        }

        cacheLock.lock()
        pipelines[tile] = state
        cacheLock.unlock()
        return state
    }

    /// Compute `C = A · B`. `A` is row-major `M × K`, `B` is row-major `K × N`,
    /// the returned `[Float]` is row-major `M × N`. Synchronous: blocks on the
    /// command buffer.
    ///
    /// `tile` selects the threadgroup configuration (defaults to 16 × 16, a
    /// reasonable starting point on Apple Silicon's 32-wide SIMD groups).
    public func matmul(
        a: [Float],
        aShape: (Int, Int),
        b: [Float],
        bShape: (Int, Int),
        tile: MetalTile = MetalTile(16, 16)
    ) throws -> [Float] {
        let (m, kA) = aShape
        let (kB, n) = bShape
        guard kA == kB else {
            throw MetalMatmulError.shapeMismatch("A.K=\(kA) != B.K=\(kB)")
        }
        guard a.count == m * kA else {
            throw MetalMatmulError.shapeMismatch("a.count=\(a.count), expected \(m * kA)")
        }
        guard b.count == kB * n else {
            throw MetalMatmulError.shapeMismatch("b.count=\(b.count), expected \(kB * n)")
        }
        let k = kA

        let pipeline = try self.pipeline(for: tile)

        let aBuf = try makeBuffer(from: a)
        let bBuf = try makeBuffer(from: b)
        guard let cBuf = device.makeBuffer(
            length: m * n * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw MetalMatmulError.bufferAllocationFailed
        }

        guard
            let cmd = queue.makeCommandBuffer(),
            let enc = cmd.makeComputeCommandEncoder()
        else {
            throw MetalMatmulError.pipelineCreationFailed("makeCommandBuffer/Encoder returned nil")
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)
        var mU = UInt32(m), kU = UInt32(k), nU = UInt32(n)
        enc.setBytes(&mU, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&kU, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&nU, length: MemoryLayout<UInt32>.size, index: 5)

        let threadsPerThreadgroup = MTLSize(width: tile.tgN, height: tile.tgM, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (n + tile.tgN - 1) / tile.tgN,
            height: (m + tile.tgM - 1) / tile.tgM,
            depth: 1
        )
        enc.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        if let err = cmd.error {
            throw MetalMatmulError.pipelineCreationFailed("command buffer error: \(err)")
        }

        return readback(buffer: cBuf, count: m * n)
    }

    private func makeBuffer(from values: [Float]) throws -> MTLBuffer {
        let bytes = values.count * MemoryLayout<Float>.size
        guard let buf = values.withUnsafeBufferPointer({ ptr -> MTLBuffer? in
            guard let base = ptr.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: bytes, options: .storageModeShared)
        }) else {
            throw MetalMatmulError.bufferAllocationFailed
        }
        return buf
    }

    private func readback(buffer: MTLBuffer, count: Int) -> [Float] {
        let raw = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }
}

#endif // canImport(Metal)
