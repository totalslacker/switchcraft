// SPDX-License-Identifier: Apache-2.0
//
// Thin `MPSMatrixMultiplication` wrapper for the issue #49 prototype.
// Mirrors `MetalMatmul`'s API shape so the benchmark harness can swap
// backends without per-shape branching.
//
// MPS row-byte alignment requirement (16-byte multiple) is satisfied for all
// four T5-base shapes by construction: with FP32 floats the tightest stride
// is `768 × 4 = 3072 B` and the smallest `128 × 4 = 512 B`, both exact
// multiples of 16. No padding workaround is needed.

#if canImport(Metal) && canImport(MetalPerformanceShaders)

import Foundation
import Metal
import MetalPerformanceShaders

public final class MPSMatmul: @unchecked Sendable {
    public let device: MTLDevice
    private let queue: MTLCommandQueue

    /// Cache key: `(M, K, N)`. `MPSMatrixMultiplication` is shape-bound, so we
    /// cache one per benchmark shape to keep per-iteration allocation out of
    /// the hot path. Without this, each iteration paid for a fresh
    /// `MPSMatrixMultiplication` object — biasing the MPS-vs-cblas comparison.
    private struct OpKey: Hashable {
        let m: Int
        let k: Int
        let n: Int
    }
    private var ops: [OpKey: MPSMatrixMultiplication] = [:]
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
    }

    private func operation(m: Int, k: Int, n: Int) -> MPSMatrixMultiplication {
        let key = OpKey(m: m, k: k, n: n)
        cacheLock.lock()
        if let cached = ops[key] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()
        let mm = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: m,
            resultColumns: n,
            interiorColumns: k,
            alpha: 1.0,
            beta: 0.0
        )
        cacheLock.lock()
        ops[key] = mm
        cacheLock.unlock()
        return mm
    }

    /// Compute `C = A · B` via `MPSMatrixMultiplication`. Synchronous; blocks
    /// on the command buffer to match `MetalMatmul.matmul(...)`.
    public func matmul(
        a: [Float],
        aShape: (Int, Int),
        b: [Float],
        bShape: (Int, Int)
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

        let aRowBytes = k * MemoryLayout<Float>.size
        let bRowBytes = n * MemoryLayout<Float>.size
        let cRowBytes = n * MemoryLayout<Float>.size

        guard
            let aBuf = makeBuffer(from: a),
            let bBuf = makeBuffer(from: b),
            let cBuf = device.makeBuffer(length: m * cRowBytes, options: .storageModeShared)
        else {
            throw MetalMatmulError.bufferAllocationFailed
        }

        let aDesc = MPSMatrixDescriptor(rows: m, columns: k,
                                        rowBytes: aRowBytes, dataType: .float32)
        let bDesc = MPSMatrixDescriptor(rows: k, columns: n,
                                        rowBytes: bRowBytes, dataType: .float32)
        let cDesc = MPSMatrixDescriptor(rows: m, columns: n,
                                        rowBytes: cRowBytes, dataType: .float32)

        let aMat = MPSMatrix(buffer: aBuf, descriptor: aDesc)
        let bMat = MPSMatrix(buffer: bBuf, descriptor: bDesc)
        let cMat = MPSMatrix(buffer: cBuf, descriptor: cDesc)

        let mm = operation(m: m, k: k, n: n)

        guard let cmd = queue.makeCommandBuffer() else {
            throw MetalMatmulError.pipelineCreationFailed("makeCommandBuffer returned nil")
        }
        mm.encode(commandBuffer: cmd, leftMatrix: aMat, rightMatrix: bMat, resultMatrix: cMat)
        cmd.commit()
        cmd.waitUntilCompleted()

        if let err = cmd.error {
            throw MetalMatmulError.pipelineCreationFailed("MPS command buffer error: \(err)")
        }

        let raw = cBuf.contents().bindMemory(to: Float.self, capacity: m * n)
        return Array(UnsafeBufferPointer(start: raw, count: m * n))
    }

    private func makeBuffer(from values: [Float]) -> MTLBuffer? {
        let bytes = values.count * MemoryLayout<Float>.size
        return values.withUnsafeBufferPointer { ptr -> MTLBuffer? in
            guard let base = ptr.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: bytes, options: .storageModeShared)
        }
    }
}

#endif
