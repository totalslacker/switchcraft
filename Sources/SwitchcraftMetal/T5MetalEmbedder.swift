// SPDX-License-Identifier: Apache-2.0
//
// `Embedder` conformance backed by a Q4_K-quantised GGUF asset of
// `google/xtr-base-en` plus the SwitchcraftMetal kernel set (issues
// #60–#63 + #64's FP32 matmul). Issue #64.
//
// Forward-pass orchestration mirrors the upstream T5 v1.1 encoder
// graph (`llama.cpp/src/llama-model.cpp` LLM_ARCH_T5 branch +
// `candle-transformers/src/models/t5.rs::T5EncoderModel`):
//
//   tokens → embed-table gather → 12 × {pre-attn RMSNorm → Q/K/V
//   projections → per-head Q·Kᵀ → +relpos bias → softmax →
//   per-head softmax·V → merge heads → O projection → residual;
//   pre-FFN RMSNorm → wi_0 → wi_1 → gated-GELU → wo → residual} →
//   final RMSNorm → 2_Dense projection (768→128) → L2 norm.
//
// Per-head Q/K/V slicing is handled by the FP32MatMul kernel's row-
// stride parameters — Q/K/V stay in interleaved `[M, D]` layout and
// each head reads its `dH`-wide column window via `aStrideBatch=dH`,
// `aRowStride=D`. softmax·V's output is similarly merged back into
// `[M, D]` interleaved layout via the kernel's `cStrideBatch`/
// `cRowStride` so the O projection sees a contiguous `[M, D]` input
// without a separate transpose / merge pass.
//
// Sliding-window decomposition + MIN_NORM filtering reuses
// `SwitchcraftCoreML.SlidingWindow` per ADR 011. The embedder commits
// + waits once per window and reads the projected output back to host
// memory for the merge.
//
// See ADR 017 for the per-op precision matrix this embedder
// implements (informational matrix → normative ADR per the issue
// spec).

#if canImport(Metal)

import Foundation
import Metal
import SwitchcraftCore
import SwitchcraftCoreML
@_spi(SwitchcraftMetal) import SwitchcraftCore

/// `Embedder` conformance running the T5-base v1.1 encoder + 2_Dense
/// projection + per-token L2 norm on Apple Metal.
///
/// Construction loads the GGUF asset eagerly (~80 MB resident,
/// dominated by the Q4_K weight tensors), pre-builds the
/// relative-position bucket table, allocates per-encode workspace
/// buffers, and constructs each kernel wrapper. `init` throws
/// `T5MetalEmbedderError.metalUnavailable` when `MetalContext.shared`
/// is `nil` (no Metal device or `SWITCHCRAFT_FORCE_ACCELERATE` is set);
/// callers that want a CoreML fallback catch the throw and instantiate
/// `T5CoreMLEmbedder` instead.
///
/// Determinism: every kernel is deterministic for fixed input on the
/// same device, and the per-window `MTLCommandBuffer` execution order
/// is fully serialised, so identical inputs produce bit-equal outputs
/// across calls (per the `Embedder` contract).
@_spi(SwitchcraftMetal)
public actor T5MetalEmbedder: Embedder {

    // MARK: - Public API

    public nonisolated let dims: Int
    public nonisolated let modelIdentifier: String
    public nonisolated let minNorm: Float
    public nonisolated let windowSize: Int
    public nonisolated let stride: Int

    // MARK: - Architecture constants
    //
    // Pinned to `google/xtr-base-en/config.json` (T5 v1.1 derivative;
    // see `docs/porting/ggml-t5.md` §"FFN activation determination").

    private let dModel: Int = 768
    private let nLayers: Int = 12
    private let nHeads: Int = 12
    private let dHead: Int = 64
    private let dFF: Int = 2048
    private let layerNormEps: Float = 1e-6

    // MARK: - Stored state

    private let tokenizer: Tokenizer
    private let context: MetalContext

    private struct LayerWeights {
        let preAttnNormGain: MTLBuffer    // FP32 [D]
        let qWeight: MTLBuffer            // Q4_K [D, D] (PyTorch [D out, D in])
        let kWeight: MTLBuffer
        let vWeight: MTLBuffer
        let oWeight: MTLBuffer
        let preFFNNormGain: MTLBuffer     // FP32 [D]
        let wi0Weight: MTLBuffer          // Q4_K [dFF, D]
        let wi1Weight: MTLBuffer          // Q4_K [dFF, D]
        let woWeight: MTLBuffer           // Q4_K [D, dFF]
    }
    private let layers: [LayerWeights]

    /// Token-embedding table (raw Q4_K bytes; CPU-dequantised per
    /// window for the gather, ~1.5 MB per window).
    private let embedTokensBytes: Data
    private let vocabSize: Int

    private let finalNormGain: MTLBuffer
    private let projWeightBuffer: MTLBuffer

    private let bucketTable: [Int32]
    private let relposBiasWeights: [Float]

    // Kernels
    private let q4kMatMul: Q4KMatMulKernel
    private let rmsNormKernel: RMSNormKernel
    private let softmaxKernel: SoftmaxKernel
    private let fp32MatMulKernel: FP32MatMulKernel
    private let gatedGELUKernel: GatedGELUKernel
    private let residualAddKernel: ResidualAddKernel
    private let l2NormKernel: L2NormKernel

    // Workspace
    private let residualBuf: MTLBuffer
    private let xNormBuf: MTLBuffer
    private let qBuf: MTLBuffer
    private let kBuf: MTLBuffer
    private let vBuf: MTLBuffer
    private let scoresBuf: MTLBuffer
    private let softmaxOutBuf: MTLBuffer
    private let relposBiasBuf: MTLBuffer
    // Scratch buffer for attention masking when tokenCount < windowSize.
    // CPU-filled once per window: copy relposBiasBuf then write -1e9 for
    // all column positions n >= tokenCount so padding tokens receive zero
    // attention weight (mirrors PyTorch's `attention_mask=(ids!=PAD).long()`).
    private let maskedRelposBiasBuf: MTLBuffer
    private let attnMergedBuf: MTLBuffer
    private let oProjBuf: MTLBuffer
    private let ffnGateBuf: MTLBuffer
    private let ffnUpBuf: MTLBuffer
    private let ffnActBuf: MTLBuffer
    private let ffnDownBuf: MTLBuffer
    private let projOutBuf: MTLBuffer
    private let l2OutBuf: MTLBuffer
    private let embedBuf: MTLBuffer

    // MARK: - Init

    public init(
        modelURL: URL,
        tokenizer: Tokenizer,
        dims dimsParam: Int = 128,
        windowSize windowSizeParam: Int = 512,
        stride strideParam: Int = 256,
        minNorm minNormParam: Float = 1.0,
        modelIdentifier modelIdentifierParam: String = "google/xtr-base-en@v1+gguf"
    ) async throws {
        precondition(dimsParam > 0 && dimsParam % 2 == 0,
                     "dims must be positive and even (Q4 codec packs two nibbles per byte)")
        precondition(windowSizeParam > 0)
        precondition(strideParam > 0 && strideParam <= windowSizeParam)

        guard let context = MetalContext.shared else {
            throw T5MetalEmbedderError.metalUnavailable
        }

        let reader = try GGUFReader(url: modelURL, device: context.device)
        let tensorNames = Set(await reader.tensorNames())

        // Resolve the 2_Dense projection tensor: try a handful of names
        // produced by different conversion pipelines (Witchcraft's
        // `quantize-tool` vs. llama.cpp's `convert-*-to-gguf.py`). The
        // typed error includes the candidates so a name mismatch is
        // immediately actionable for the operator.
        let projCandidates = [
            "2_Dense.linear.weight",
            "linear.weight",        // Witchcraft quantize-tool at pinned Candle rev 5bd5618
            "2_Dense.weight",
            "dense.weight",
            "projection.weight",
            "output_projection.weight",
        ]
        guard let projName = projCandidates.first(where: { tensorNames.contains($0) }) else {
            throw T5MetalEmbedderError.tensorMissing(
                "2_Dense projection (tried: \(projCandidates.joined(separator: ", ")))"
            )
        }

        // ----- Token embedding table (Q4_K; kept as raw bytes) -----
        let embedTensor = try await Self.requireTensor(
            reader, name: Self.tensorEmbedTokens, available: tensorNames
        )
        try Self.requireDtype(embedTensor, .q4k)
        // GGUF on-disk shape is fastest-varying-first: [d_model, vocab_size]
        // for PyTorch's [vocab_size, d_model] embedding-table layout.
        guard embedTensor.shape.count == 2 else {
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(Self.tensorEmbedTokens) rank \(embedTensor.shape.count) ≠ 2"
            )
        }
        let embedDModel = embedTensor.shape[0]
        let embedVocabSize = embedTensor.shape[1]
        guard embedDModel == dModel else {
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(Self.tensorEmbedTokens) inner dim \(embedDModel) ≠ d_model \(dModel)"
            )
        }
        let embedBytes = Data(
            bytes: embedTensor.buffer.contents(),
            count: embedTensor.buffer.length
        )

        // ----- Per-layer weights -----
        var layerWeights: [LayerWeights] = []
        layerWeights.reserveCapacity(nLayers)
        for ℓ in 0..<nLayers {
            let preAttnNorm = try await Self.requireFP32Vector(
                reader, name: Self.preAttnNormName(layer: ℓ),
                length: dModel, available: tensorNames
            )
            let q = try await Self.requireQ4KMatrix(
                reader, name: Self.qName(layer: ℓ), N: dModel, K: dModel,
                available: tensorNames
            )
            let k = try await Self.requireQ4KMatrix(
                reader, name: Self.kName(layer: ℓ), N: dModel, K: dModel,
                available: tensorNames
            )
            let v = try await Self.requireQ4KMatrix(
                reader, name: Self.vName(layer: ℓ), N: dModel, K: dModel,
                available: tensorNames
            )
            let o = try await Self.requireQ4KMatrix(
                reader, name: Self.oName(layer: ℓ), N: dModel, K: dModel,
                available: tensorNames
            )
            let preFFNNorm = try await Self.requireFP32Vector(
                reader, name: Self.preFFNNormName(layer: ℓ),
                length: dModel, available: tensorNames
            )
            let wi0 = try await Self.requireQ4KMatrix(
                reader, name: Self.wi0Name(layer: ℓ), N: dFF, K: dModel,
                available: tensorNames
            )
            let wi1 = try await Self.requireQ4KMatrix(
                reader, name: Self.wi1Name(layer: ℓ), N: dFF, K: dModel,
                available: tensorNames
            )
            let wo = try await Self.requireQ4KMatrix(
                reader, name: Self.woName(layer: ℓ), N: dModel, K: dFF,
                available: tensorNames
            )
            layerWeights.append(LayerWeights(
                preAttnNormGain: preAttnNorm.buffer,
                qWeight: q.buffer, kWeight: k.buffer,
                vWeight: v.buffer, oWeight: o.buffer,
                preFFNNormGain: preFFNNorm.buffer,
                wi0Weight: wi0.buffer, wi1Weight: wi1.buffer,
                woWeight: wo.buffer
            ))
        }

        // ----- Final encoder RMSNorm -----
        let finalNormTensor = try await Self.requireFP32Vector(
            reader, name: Self.tensorFinalNorm, length: dModel,
            available: tensorNames
        )
        let finalNorm = finalNormTensor.buffer

        // ----- Relative-position bias weights (FP32 [num_buckets, num_heads]) -----
        let relposTensor = try await Self.requireTensor(
            reader, name: Self.tensorRelposBias, available: tensorNames
        )
        try Self.requireDtype(relposTensor, .f32)
        let relposExpected = T5RelativePositionBias.numBuckets * nHeads
        guard relposTensor.elementCount == relposExpected else {
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(Self.tensorRelposBias) elements \(relposTensor.elementCount) ≠ expected \(relposExpected)"
            )
        }
        let relposPtr = relposTensor.buffer.contents()
            .bindMemory(to: Float.self, capacity: relposTensor.elementCount)
        let relposWeights = Array(UnsafeBufferPointer(
            start: relposPtr, count: relposTensor.elementCount
        ))

        // ----- 2_Dense projection (FP16 → widened to FP32) -----
        let projTensor = try await reader.tensor(projName)
        let projExpected = dimsParam * dModel
        guard projTensor.elementCount == projExpected else {
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(projName) elements \(projTensor.elementCount) ≠ expected \(projExpected) (dims × dModel)"
            )
        }
        // GGUF on-disk shape is fastest-varying-first: PyTorch's
        // [out_features=dims, in_features=dModel] is stored as
        // [dModel, dims]. Validate explicitly so a transposed asset
        // throws rather than silently producing wrong embeddings.
        guard projTensor.shape.count == 2,
              projTensor.shape[0] == dModel,
              projTensor.shape[1] == dimsParam else {
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(projName) on-disk shape \(projTensor.shape) ≠ expected [\(dModel), \(dimsParam)] (fastest-varying-first)"
            )
        }
        let projWeights: [Float]
        switch projTensor.dtype {
        case .f16:
            let h16 = projTensor.buffer.contents()
                .bindMemory(to: Float16.self, capacity: projTensor.elementCount)
            var widened = [Float](repeating: 0, count: projTensor.elementCount)
            for i in 0..<projTensor.elementCount {
                widened[i] = Float(h16[i])
            }
            projWeights = widened
        case .f32:
            let f32 = projTensor.buffer.contents()
                .bindMemory(to: Float.self, capacity: projTensor.elementCount)
            projWeights = Array(UnsafeBufferPointer(
                start: f32, count: projTensor.elementCount
            ))
        case .q4k:
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(projName) is Q4_K but the precision-routing ADR (017) requires FP16/FP32"
            )
        }
        guard let projBuf = context.device.makeBuffer(
            bytes: projWeights,
            length: projWeights.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw T5MetalEmbedderError.bufferAllocationFailed("projection weight")
        }
        projBuf.label = "T5MetalEmbedder.projWeight"

        // ----- Bucket table -----
        let bucketTable = T5RelativePositionBias.bucketTable(seqLen: windowSizeParam)

        // ----- Kernel wrappers -----
        let q4kMM = try Q4KMatMulKernel(context: context)
        let rms = try RMSNormKernel(context: context)
        let sm = try SoftmaxKernel(context: context)
        let fp32MM = try FP32MatMulKernel(context: context)
        let ggelu = try GatedGELUKernel(context: context)
        let radd = try ResidualAddKernel(context: context)
        let l2 = try L2NormKernel(context: context)

        // ----- Workspace allocation -----
        // A capturing closure rather than a nested `func`: Swift treats
        // a nested `func` inside an actor `init` as implicitly capturing
        // `self`, which trips the DI analysis on subsequent stored-
        // property reads. The closure form avoids the implicit capture.
        let floatBytes = MemoryLayout<Float>.size
        let device = context.device
        let alloc: (Int, String) throws -> MTLBuffer = { count, label in
            guard let buf = device.makeBuffer(
                length: count * floatBytes,
                options: .storageModeShared
            ) else {
                throw T5MetalEmbedderError.bufferAllocationFailed(label)
            }
            buf.label = "T5MetalEmbedder.\(label)"
            return buf
        }

        // Use local snapshots of the architecture constants (the
        // stored properties have default initialisers but Swift's
        // definite-initialisation analysis flags them as not-yet-init
        // when other stored properties — like `dims` from the
        // parameter — are referenced in the same expression).
        let M = windowSizeParam
        let D = 768           // dModel
        let H = 12            // nHeads
        let dFFLocal = 2048   // dFF
        let dProj = dimsParam

        let residualBuf = try alloc(M * D, "residual")
        let xNormBuf = try alloc(M * D, "xNorm")
        let qBuf = try alloc(M * D, "q")
        let kBuf = try alloc(M * D, "k")
        let vBuf = try alloc(M * D, "v")
        let scoresBuf = try alloc(H * M * M, "scores")
        let softmaxOutBuf = try alloc(H * M * M, "softmaxOut")
        let relposBiasBuf = try alloc(H * M * M, "relposBias")
        let maskedRelposBiasBuf = try alloc(H * M * M, "maskedRelposBias")
        let attnMergedBuf = try alloc(M * D, "attnMerged")
        let oProjBuf = try alloc(M * D, "oProj")
        let ffnGateBuf = try alloc(M * dFFLocal, "ffnGate")
        let ffnUpBuf = try alloc(M * dFFLocal, "ffnUp")
        let ffnActBuf = try alloc(M * dFFLocal, "ffnAct")
        let ffnDownBuf = try alloc(M * D, "ffnDown")
        let projOutBuf = try alloc(M * dProj, "projOut")
        let l2OutBuf = try alloc(M * dProj, "l2Out")
        let embedBuf = try alloc(M * D, "embed")

        // ----- Precompute relpos bias into relposBiasBuf -----
        // The bias depends only on (bucketTable, weights, numHeads,
        // seqLen=windowSize), all fixed at init — so build once here
        // instead of on every encode (~12 MB memcpy avoided per call).
        let biasMatrix = T5RelativePositionBias.buildBiasMatrix(
            bucketTable: bucketTable,
            weights: relposWeights,
            numHeads: H,
            seqLen: M
        )
        _ = biasMatrix.withUnsafeBufferPointer { src -> Int in
            memcpy(
                relposBiasBuf.contents(),
                src.baseAddress!,
                src.count * MemoryLayout<Float>.size
            )
            return src.count
        }

        // ----- Stored state assignment -----
        self.dims = dimsParam
        self.modelIdentifier = modelIdentifierParam
        self.minNorm = minNormParam
        self.windowSize = windowSizeParam
        self.stride = strideParam
        self.tokenizer = tokenizer
        self.context = context
        self.layers = layerWeights
        self.embedTokensBytes = embedBytes
        self.vocabSize = embedVocabSize
        self.finalNormGain = finalNorm
        self.projWeightBuffer = projBuf
        self.bucketTable = bucketTable
        self.relposBiasWeights = relposWeights
        self.q4kMatMul = q4kMM
        self.rmsNormKernel = rms
        self.softmaxKernel = sm
        self.fp32MatMulKernel = fp32MM
        self.gatedGELUKernel = ggelu
        self.residualAddKernel = radd
        self.l2NormKernel = l2
        self.residualBuf = residualBuf
        self.xNormBuf = xNormBuf
        self.qBuf = qBuf
        self.kBuf = kBuf
        self.vBuf = vBuf
        self.scoresBuf = scoresBuf
        self.softmaxOutBuf = softmaxOutBuf
        self.relposBiasBuf = relposBiasBuf
        self.maskedRelposBiasBuf = maskedRelposBiasBuf
        self.attnMergedBuf = attnMergedBuf
        self.oProjBuf = oProjBuf
        self.ffnGateBuf = ffnGateBuf
        self.ffnUpBuf = ffnUpBuf
        self.ffnActBuf = ffnActBuf
        self.ffnDownBuf = ffnDownBuf
        self.projOutBuf = projOutBuf
        self.l2OutBuf = l2OutBuf
        self.embedBuf = embedBuf
    }

    // MARK: - Embedder

    public func encode(_ text: String) async throws -> [Float] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return [] }

        let tokens = try tokenizer.encode(text, addSpecialTokens: true)
        if tokens.isEmpty { return [] }

        let starts = SlidingWindow.plan(
            tokenCount: tokens.count,
            windowSize: windowSize,
            stride: stride
        )

        // The relpos bias is precomputed into `relposBiasBuf` at
        // `init` time — it is fixed for a given `windowSize`.

        var perWindowNormalised: [[Float]] = []
        var perWindowRawNorms: [[Float]] = []
        perWindowNormalised.reserveCapacity(starts.count)
        perWindowRawNorms.reserveCapacity(starts.count)

        for start in starts {
            let lo = start
            let hi = min(start + windowSize, tokens.count)
            let slice = Array(tokens[lo..<hi])
            let (normalised, rawNorms) = try encodeWindow(tokens: slice)
            perWindowNormalised.append(normalised)
            perWindowRawNorms.append(rawNorms)
        }

        let merged = SlidingWindow.merge(
            windowStarts: starts,
            windowSize: windowSize,
            tokenCount: tokens.count,
            dims: dims,
            windowNormalised: perWindowNormalised,
            windowRawNorms: perWindowRawNorms,
            minNorm: minNorm
        )
        return merged.embeddings
    }

    // MARK: - Per-window forward pass

    private func encodeWindow(
        tokens: [Int32]
    ) throws -> (normalised: [Float], rawNorms: [Float]) {
        precondition(tokens.count <= windowSize)

        // 1. Token embedding gather (CPU dequant, FP32 upload).
        gatherTokenEmbeddings(tokens: tokens)

        // BISECT[1]: log embed table output before GPU work.
        bisectLog(embedBuf, count: tokens.count * dModel,
                  label: "embedBuf[after gather, \(tokens.count) tokens]")

        // Initialise residual stream to gathered embeddings.
        memcpy(
            residualBuf.contents(),
            embedBuf.contents(),
            windowSize * dModel * MemoryLayout<Float>.size
        )

        // Compute the softmax bias buffer for this window: apply -1e9 at
        // pad positions when the input is shorter than windowSize so that
        // attention doesn't leak to the PAD token embeddings.
        let attnBiasBuf = tokens.count < windowSize
            ? applyAttentionMask(tokenCount: tokens.count)
            : relposBiasBuf

        // 2. Encode each layer in its own command buffer for per-layer bisection.
        // GatedGELU NaN fix verified. Now bisect layer 0 intra-layer to
        // investigate cosine ~0.81-0.88 (second root cause, Task 4).
        for ℓ in 0..<nLayers {
            if ℓ == 0 {
                // Intra-layer bisection for layer 0 (investigate cosine error).
                try bisectLayer2(layer: layers[ℓ], tokenCount: tokens.count, biasBuf: attnBiasBuf)
            } else {
                guard let cbL = context.queue.makeCommandBuffer() else {
                    throw T5MetalEmbedderError.commandBufferCreationFailed
                }
                cbL.label = "T5MetalEmbedder.encodeWindow.layer\(ℓ)"
                encodeAttentionSubLayer(commandBuffer: cbL, layer: layers[ℓ], biasBuf: attnBiasBuf)
                encodeFFNSubLayer(commandBuffer: cbL, layer: layers[ℓ])
                cbL.commit(); cbL.waitUntilCompleted()
                if let err = cbL.error {
                    throw T5MetalEmbedderError.commandBufferFailed("layer \(ℓ): \(err)")
                }
                bisectLog(residualBuf, count: tokens.count * dModel,
                          label: "residualBuf[after layer \(ℓ)]")
            }
        }

        // 3–5. Final RMSNorm + 2_Dense projection + L2 norm.
        guard let cb2 = context.queue.makeCommandBuffer() else {
            throw T5MetalEmbedderError.commandBufferCreationFailed
        }
        cb2.label = "T5MetalEmbedder.encodeWindow.projL2"

        rmsNormKernel.encode(
            commandBuffer: cb2,
            input: residualBuf,
            gain: finalNormGain,
            output: xNormBuf,
            M: windowSize, D: dModel,
            eps: layerNormEps
        )
        fp32MatMulKernel.encode(
            commandBuffer: cb2,
            a: xNormBuf,
            b: projWeightBuffer,
            output: projOutBuf,
            M: windowSize, N: dims, K: dModel,
            batch: 1, transposeB: true
        )
        l2NormKernel.encode(
            commandBuffer: cb2,
            input: projOutBuf,
            output: l2OutBuf,
            rows: windowSize, cols: dims,
            epsilon: 1e-12
        )
        cb2.commit(); cb2.waitUntilCompleted()
        if let err = cb2.error {
            throw T5MetalEmbedderError.commandBufferFailed("proj+l2: \(err)")
        }

        // BISECT[4]: xNormBuf (after final RMSNorm, before 2_Dense).
        bisectLog(xNormBuf, count: tokens.count * dModel,
                  label: "xNormBuf[after final RMSNorm]")

        // BISECT[5]: projOutBuf (after 2_Dense, before L2).
        bisectLog(projOutBuf, count: tokens.count * dims,
                  label: "projOutBuf[after 2_Dense]")

        // 6. Read back projOut + l2Out and compute pre-L2 row norms.
        let proj = readFloats(projOutBuf, count: windowSize * dims)
        let normalised = readFloats(l2OutBuf, count: windowSize * dims)
        var rawNorms = [Float](repeating: 0, count: windowSize)
        for r in 0..<windowSize {
            var sumSq: Double = 0
            let base = r * dims
            for c in 0..<dims {
                let v = Double(proj[base + c])
                sumSq += v * v
            }
            rawNorms[r] = Float(sumSq.squareRoot())
        }

        // BISECT[6]: summary of raw norms and how many survive the minNorm gate.
        let keepCount = rawNorms.filter { $0 >= minNorm }.count
        let normMin = rawNorms.min() ?? 0
        let normMax = rawNorms.max() ?? 0
        print("[T5Bisect] rawNorms: min=\(normMin) max=\(normMax) "
            + "keepCount=\(keepCount)/\(windowSize) (minNorm=\(minNorm))")

        return (normalised, rawNorms)
    }

    // Intra-layer bisection for layer 2 (first NaN layer). Each sub-operation
    // runs in its own command buffer so we can log the intermediate buffer values
    // after each step. Temporary scaffolding; removed in Task 8.
    private func bisectLayer2(layer: LayerWeights, tokenCount: Int, biasBuf: MTLBuffer) throws {
        let M = windowSize
        let D = dModel
        let H = nHeads
        let dH = dHead
        let dFF_ = dFF

        func run(_ label: String, _ body: (MTLCommandBuffer) -> Void) throws {
            guard let cb = context.queue.makeCommandBuffer() else {
                throw T5MetalEmbedderError.commandBufferCreationFailed
            }
            cb.label = label
            body(cb)
            cb.commit(); cb.waitUntilCompleted()
            if let err = cb.error {
                throw T5MetalEmbedderError.commandBufferFailed("\(label): \(err)")
            }
        }

        // Attn sub-layer: pre-norm
        try run("L2.attn.rmsNorm") { cb in
            rmsNormKernel.encode(commandBuffer: cb, input: residualBuf, gain: layer.preAttnNormGain,
                                 output: xNormBuf, M: M, D: D, eps: layerNormEps)
        }
        bisectLog(xNormBuf, count: tokenCount * D, label: "L2.xNorm[after preAttnNorm]")

        // Q projection
        try run("L2.Q") { cb in
            q4kMatMul.encode(commandBuffer: cb, weight: layer.qWeight, weightShape: (D, D),
                             input: xNormBuf, output: qBuf, M: M)
        }
        bisectLog(qBuf, count: tokenCount * D, label: "L2.qBuf[after Q proj]")

        // K, V projections
        try run("L2.KV") { cb in
            q4kMatMul.encode(commandBuffer: cb, weight: layer.kWeight, weightShape: (D, D),
                             input: xNormBuf, output: kBuf, M: M)
            q4kMatMul.encode(commandBuffer: cb, weight: layer.vWeight, weightShape: (D, D),
                             input: xNormBuf, output: vBuf, M: M)
        }
        bisectLog(kBuf, count: tokenCount * D, label: "L2.kBuf[after K proj]")

        // Q·Kᵀ scores
        try run("L2.scores") { cb in
            fp32MatMulKernel.encode(commandBuffer: cb, a: qBuf, b: kBuf, output: scoresBuf,
                                    M: M, N: M, K: dH, batch: H, transposeB: true,
                                    aStrideBatch: dH, bStrideBatch: dH,
                                    cStrideBatch: M * M, aRowStride: D, bRowStride: D, cRowStride: M)
        }
        bisectLog(scoresBuf, count: tokenCount * M, label: "L2.scoresBuf[after Q·Kᵀ, head 0 rows 0-7]")

        // Softmax + relpos bias (with attention mask for pad positions).
        try run("L2.softmax") { cb in
            softmaxKernel.encode(commandBuffer: cb, input: scoresBuf, bias: biasBuf,
                                 output: softmaxOutBuf, B: H * M, N: M, scale: 1.0)
        }
        bisectLog(softmaxOutBuf, count: tokenCount * M, label: "L2.softmaxOut[after softmax]")

        // softmax·V → attnMerged
        try run("L2.attnMerge") { cb in
            fp32MatMulKernel.encode(commandBuffer: cb, a: softmaxOutBuf, b: vBuf, output: attnMergedBuf,
                                    M: M, N: dH, K: M, batch: H, transposeB: false,
                                    aStrideBatch: M * M, bStrideBatch: dH,
                                    cStrideBatch: dH, aRowStride: M, bRowStride: D, cRowStride: D)
        }
        bisectLog(attnMergedBuf, count: tokenCount * D, label: "L2.attnMergedBuf[after softmax·V]")

        // O projection
        try run("L2.O") { cb in
            q4kMatMul.encode(commandBuffer: cb, weight: layer.oWeight, weightShape: (D, D),
                             input: attnMergedBuf, output: oProjBuf, M: M)
        }
        bisectLog(oProjBuf, count: tokenCount * D, label: "L2.oProjBuf[after O proj]")

        // Residual add (attn)
        try run("L2.residualAdd1") { cb in
            residualAddKernel.encode(commandBuffer: cb, a: residualBuf, b: oProjBuf, output: residualBuf, count: M * D)
        }
        bisectLog(residualBuf, count: tokenCount * D, label: "L2.residualBuf[after attn residual add]")

        // FFN sub-layer: pre-norm
        try run("L2.ffn.rmsNorm") { cb in
            rmsNormKernel.encode(commandBuffer: cb, input: residualBuf, gain: layer.preFFNNormGain,
                                 output: xNormBuf, M: M, D: D, eps: layerNormEps)
        }
        bisectLog(xNormBuf, count: tokenCount * D, label: "L2.xNorm[after preFFNNorm]")

        // wi_0, wi_1 projections
        try run("L2.wi") { cb in
            q4kMatMul.encode(commandBuffer: cb, weight: layer.wi0Weight, weightShape: (dFF_, D),
                             input: xNormBuf, output: ffnGateBuf, M: M)
            q4kMatMul.encode(commandBuffer: cb, weight: layer.wi1Weight, weightShape: (dFF_, D),
                             input: xNormBuf, output: ffnUpBuf, M: M)
        }
        bisectLog(ffnGateBuf, count: tokenCount * dFF_, label: "L2.ffnGateBuf[after wi_0]")
        bisectLog(ffnUpBuf, count: tokenCount * dFF_, label: "L2.ffnUpBuf[after wi_1]")

        // Spotlight: directly read elements around the known NaN position (token 6, dim 324 → index 12932).
        do {
            let nanIdx = 12932
            let gateRaw = ffnGateBuf.contents().bindMemory(to: Float.self, capacity: tokenCount * dFF_)
            let upRaw   = ffnUpBuf.contents().bindMemory(to: Float.self, capacity: tokenCount * dFF_)
            let lo = max(0, nanIdx - 3); let hi = min(tokenCount * dFF_ - 1, nanIdx + 3)
            let gateCtx = (lo...hi).map { "[\($0)]=\(String(format: "%.4f", gateRaw[$0]))" }.joined(separator: " ")
            let upCtx   = (lo...hi).map { "[\($0)]=\(String(format: "%.4f", upRaw[$0]))"   }.joined(separator: " ")
            print("[T5Bisect] SPOTLIGHT gate around \(nanIdx): \(gateCtx)")
            print("[T5Bisect] SPOTLIGHT up around \(nanIdx): \(upCtx)")
            // Also scan all of row 6 (indices 12288..14335) for any NaN/Inf.
            let row6Start = 6 * dFF_; let row6End = row6Start + dFF_ - 1
            var row6NaN: Int? = nil; var row6Inf: Int? = nil
            for i in row6Start...row6End {
                if gateRaw[i].isNaN && row6NaN == nil { row6NaN = i }
                if gateRaw[i].isInfinite && row6Inf == nil { row6Inf = i }
            }
            print("[T5Bisect] SPOTLIGHT ffnGateBuf row6 (\(row6Start)..\(row6End)): firstNaN=\(row6NaN.map{"\($0)"}  ?? "none") firstInf=\(row6Inf.map{"\($0)"} ?? "none")")
            row6NaN = nil; row6Inf = nil
            for i in row6Start...row6End {
                if upRaw[i].isNaN && row6NaN == nil { row6NaN = i }
                if upRaw[i].isInfinite && row6Inf == nil { row6Inf = i }
            }
            print("[T5Bisect] SPOTLIGHT ffnUpBuf row6 (\(row6Start)..\(row6End)): firstNaN=\(row6NaN.map{"\($0)"} ?? "none") firstInf=\(row6Inf.map{"\($0)"} ?? "none")")
        }

        // GatedGELU
        try run("L2.gatedGELU") { cb in
            gatedGELUKernel.encode(commandBuffer: cb, gate: ffnGateBuf, up: ffnUpBuf, output: ffnActBuf, count: M * dFF_)
        }
        bisectLog(ffnActBuf, count: tokenCount * dFF_, label: "L2.ffnActBuf[after gated-GELU]")

        // wo projection
        try run("L2.wo") { cb in
            q4kMatMul.encode(commandBuffer: cb, weight: layer.woWeight, weightShape: (D, dFF_),
                             input: ffnActBuf, output: ffnDownBuf, M: M)
        }
        bisectLog(ffnDownBuf, count: tokenCount * D, label: "L2.ffnDownBuf[after wo proj]")

        // Residual add (FFN)
        try run("L2.residualAdd2") { cb in
            residualAddKernel.encode(commandBuffer: cb, a: residualBuf, b: ffnDownBuf, output: residualBuf, count: M * D)
        }
        bisectLog(residualBuf, count: tokenCount * D, label: "L2.residualBuf[after FFN residual add = layer 2 done]")
    }

    // BISECT helper — log first-8 + full-buffer NaN scan of a buffer.
    // Temporary scaffolding; removed in Task 8.
    private func bisectLog(_ buf: MTLBuffer, count: Int, label: String) {
        guard count > 0 else {
            print("[T5Bisect] \(label): count=0")
            return
        }
        let raw = buf.contents().bindMemory(to: Float.self, capacity: count)
        let sample = min(count, 8)
        let vals = (0..<sample).map { raw[$0] }
        let first8Finite = vals.allSatisfy { $0.isFinite }
        // Scan full buffer for first NaN/Inf index.
        var firstNaNIdx: Int? = nil
        var firstInfIdx: Int? = nil
        var globalMaxAbs: Float = 0
        for i in 0..<count {
            let v = raw[i]
            if v.isNaN && firstNaNIdx == nil { firstNaNIdx = i }
            if v.isInfinite && firstInfIdx == nil { firstInfIdx = i }
            let a = abs(v)
            if a.isFinite && a > globalMaxAbs { globalMaxAbs = a }
        }
        let anyNaN = firstNaNIdx != nil
        let anyInf = firstInfIdx != nil
        let allFinite = !anyNaN && !anyInf
        let mean = vals.reduce(Float(0), +) / Float(vals.count)
        print("[T5Bisect] \(label): "
            + "first8=\(vals.map { String(format: "%.4f", $0) }) "
            + "allFinite=\(allFinite) "
            + (anyNaN ? "firstNaN@\(firstNaNIdx!) " : "")
            + (anyInf ? "firstInf@\(firstInfIdx!) " : "")
            + "globalMaxAbs=\(String(format: "%.4f", globalMaxAbs)) "
            + "mean=\(String(format: "%.6f", mean))")
        // If NaN found, log the surrounding context (10 elements either side).
        if let nanIdx = firstNaNIdx {
            let lo = max(0, nanIdx - 2)
            let hi = min(count - 1, nanIdx + 5)
            let ctx = (lo...hi).map { i -> String in
                let v = raw[i]
                return "[\(i)]=\(v.isNaN ? "NaN" : String(format: "%.4f", v))"
            }
            print("[T5Bisect] \(label) NaN context: \(ctx.joined(separator: " "))")
        }
    }

    /// One attention sub-layer. Reads/writes `residualBuf`. Uses the
    /// per-window workspace buffers for intermediates.
    ///
    /// Per-head Q/K/V slicing relies on `FP32MatMulKernel`'s row-stride
    /// arguments: q/k/v stay in interleaved `[M, D=H*dH]` layout and
    /// each head reads its `dH`-wide slice via `aStrideBatch=dH`,
    /// `aRowStride=D`. softmax·V's output is similarly merged back into
    /// `[M, D]` interleaved layout via `cStrideBatch=dH`, `cRowStride=D`,
    /// so the O projection sees a contiguous `[M, D]` input.
    private func encodeAttentionSubLayer(
        commandBuffer cb: MTLCommandBuffer,
        layer: LayerWeights,
        biasBuf: MTLBuffer
    ) {
        let M = windowSize
        let D = dModel
        let H = nHeads
        let dH = dHead

        // Pre-attn RMSNorm: residual → xNorm.
        rmsNormKernel.encode(
            commandBuffer: cb,
            input: residualBuf, gain: layer.preAttnNormGain,
            output: xNormBuf, M: M, D: D, eps: layerNormEps
        )

        // Q/K/V Q4_K projections — output is [M, D] row-major.
        // Q4KMatMul wrapper takes weightShape (N, K). Our weights are
        // [D, D] (PyTorch `[out=D, in=D]`), so (N=D, K=D).
        q4kMatMul.encode(
            commandBuffer: cb,
            weight: layer.qWeight, weightShape: (D, D),
            input: xNormBuf, output: qBuf, M: M
        )
        q4kMatMul.encode(
            commandBuffer: cb,
            weight: layer.kWeight, weightShape: (D, D),
            input: xNormBuf, output: kBuf, M: M
        )
        q4kMatMul.encode(
            commandBuffer: cb,
            weight: layer.vWeight, weightShape: (D, D),
            input: xNormBuf, output: vBuf, M: M
        )

        // Per-head Q · Kᵀ. Interleaved-buffer slicing via row strides.
        // T5 v1.1 attention does NOT scale Q by 1/√d_k (the softmax is
        // over un-scaled logits; T5's relpos bias substitutes for the
        // sinusoidal/scale combo).
        fp32MatMulKernel.encode(
            commandBuffer: cb,
            a: qBuf, b: kBuf, output: scoresBuf,
            M: M, N: M, K: dH, batch: H, transposeB: true,
            aStrideBatch: dH, bStrideBatch: dH,
            cStrideBatch: M * M,
            aRowStride: D, bRowStride: D,
            cRowStride: M
        )

        // Softmax with relpos bias (plus attention mask for pad positions).
        // Flatten (H, M, M) → (B=H*M, N=M). Caller supplies biasBuf which
        // is either the raw relposBiasBuf (full-window input) or the
        // maskedRelposBiasBuf with -1e9 for pad columns (short inputs).
        softmaxKernel.encode(
            commandBuffer: cb,
            input: scoresBuf,
            bias: biasBuf,
            output: softmaxOutBuf,
            B: H * M, N: M, scale: 1.0
        )

        // softmax · V — write directly into attnMergedBuf [M, D]
        // interleaved layout so the O projection sees a contiguous
        // input.
        fp32MatMulKernel.encode(
            commandBuffer: cb,
            a: softmaxOutBuf, b: vBuf, output: attnMergedBuf,
            M: M, N: dH, K: M, batch: H, transposeB: false,
            aStrideBatch: M * M, bStrideBatch: dH,
            cStrideBatch: dH,
            aRowStride: M, bRowStride: D,
            cRowStride: D
        )

        // O projection.
        q4kMatMul.encode(
            commandBuffer: cb,
            weight: layer.oWeight, weightShape: (D, D),
            input: attnMergedBuf, output: oProjBuf, M: M
        )

        // Residual add: residual = residual + oProj.
        residualAddKernel.encode(
            commandBuffer: cb,
            a: residualBuf, b: oProjBuf, output: residualBuf,
            count: M * D
        )
    }

    /// One FFN sub-layer (pre-FFN RMSNorm → wi_0/wi_1 → gated-GELU →
    /// wo → residual). Reads/writes `residualBuf`.
    private func encodeFFNSubLayer(
        commandBuffer cb: MTLCommandBuffer,
        layer: LayerWeights
    ) {
        let M = windowSize
        let D = dModel
        let dFF_ = dFF

        // Pre-FFN RMSNorm: residual → xNorm.
        rmsNormKernel.encode(
            commandBuffer: cb,
            input: residualBuf, gain: layer.preFFNNormGain,
            output: xNormBuf, M: M, D: D, eps: layerNormEps
        )

        // wi_0, wi_1 — Q4_K matmuls, weights [dFF, D].
        q4kMatMul.encode(
            commandBuffer: cb,
            weight: layer.wi0Weight, weightShape: (dFF_, D),
            input: xNormBuf, output: ffnGateBuf, M: M
        )
        q4kMatMul.encode(
            commandBuffer: cb,
            weight: layer.wi1Weight, weightShape: (dFF_, D),
            input: xNormBuf, output: ffnUpBuf, M: M
        )

        // Gated-GELU: ffnAct = gelu_new(gate) * up.
        gatedGELUKernel.encode(
            commandBuffer: cb,
            gate: ffnGateBuf, up: ffnUpBuf, output: ffnActBuf,
            count: M * dFF_
        )

        // wo — Q4_K matmul, weight [D, dFF].
        q4kMatMul.encode(
            commandBuffer: cb,
            weight: layer.woWeight, weightShape: (D, dFF_),
            input: ffnActBuf, output: ffnDownBuf, M: M
        )

        // Residual add.
        residualAddKernel.encode(
            commandBuffer: cb,
            a: residualBuf, b: ffnDownBuf, output: residualBuf,
            count: M * D
        )
    }

    // MARK: - Attention mask

    /// Copy `relposBiasBuf` into `maskedRelposBiasBuf` and write -1e9 into
    /// every column position n >= `tokenCount` for all (head, query-row)
    /// pairs. The result is returned as the bias argument for
    /// `encodeAttentionSubLayer` when `tokenCount < windowSize`.
    ///
    /// This mirrors PyTorch's `attention_mask = (input_ids != PAD_ID).long()`
    /// which the reference embedder in `scripts/convert-xtr-to-coreml.py`
    /// applies before every T5EncoderModel call. Without this masking,
    /// softmax distributes ~(windowSize-tokenCount)/windowSize of each
    /// query token's attention weight to padding positions, degrading
    /// cosine similarity (issue #75, second root cause).
    private func applyAttentionMask(tokenCount: Int) -> MTLBuffer {
        let M = windowSize
        let H = nHeads
        let total = H * M * M
        memcpy(maskedRelposBiasBuf.contents(), relposBiasBuf.contents(),
               total * MemoryLayout<Float>.size)
        let ptr = maskedRelposBiasBuf.contents()
            .bindMemory(to: Float.self, capacity: total)
        let maskVal: Float = -1e9
        for h in 0..<H {
            for m in 0..<M {
                let rowBase = h * M * M + m * M
                for n in tokenCount..<M {
                    ptr[rowBase + n] = maskVal
                }
            }
        }
        return maskedRelposBiasBuf
    }

    // MARK: - Token embedding gather

    /// CPU dequantise the rows of the Q4_K embed-tokens table for the
    /// given `tokens` and copy the FP32 result into `embedBuf`. Slots
    /// beyond `tokens.count` are filled with the pad-token row
    /// (`tokenID == 0`) to match the CoreML path, which pads `input_ids`
    /// with the pad-token ID — keeping attention behaviour identical
    /// across the two embedders for short inputs.
    ///
    /// Cost is bounded by `windowSize × (dModel / 256) × 144 ≈ 1.5 MB`
    /// of source bytes per window; the Swift `Q4KDecode.dequantise`
    /// path is the canonical CPU reference (issue #59).
    private func gatherTokenEmbeddings(tokens: [Int32]) {
        let M = windowSize
        let D = dModel
        let bytesPerRow = (D / Q4KDecode.elementsPerSuperBlock)
            * Q4KDecode.bytesPerSuperBlock
        let dst = embedBuf.contents()
            .bindMemory(to: Float.self, capacity: M * D)

        embedTokensBytes.withUnsafeBytes { rawPtr in
            let base = rawPtr.bindMemory(to: UInt8.self).baseAddress!
            for slot in 0..<M {
                let rawID = slot < tokens.count ? Int(tokens[slot]) : 0
                let tokenID = (rawID >= 0 && rawID < vocabSize) ? rawID : 0
                let rowOffset = tokenID * bytesPerRow
                let rowPtr = base.advanced(by: rowOffset)
                let blockBuf = UnsafeRawBufferPointer(
                    start: rowPtr, count: bytesPerRow
                )
                let dstStart = dst.advanced(by: slot * D)
                let dstBuf = UnsafeMutableBufferPointer(
                    start: dstStart, count: D
                )
                Q4KDecode.dequantise(blocks: blockBuf, into: dstBuf)
            }
        }
    }

    // MARK: - Buffer helpers

    private func readFloats(_ buf: MTLBuffer, count: Int) -> [Float] {
        let raw = buf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: raw, count: count))
    }
}

// MARK: - Errors

@_spi(SwitchcraftMetal)
public enum T5MetalEmbedderError: Error, Sendable, Equatable {
    /// `MetalContext.shared` was `nil` at init. Caller can fall back to
    /// `T5CoreMLEmbedder` per ADR 015's nil-fallback policy.
    case metalUnavailable
    /// A required GGUF tensor was not present in the asset.
    case tensorMissing(String)
    /// A GGUF tensor's shape or dtype did not match the embedder's
    /// expectations.
    case tensorShapeMismatch(String)
    /// `MTLDevice.makeBuffer` returned `nil` for a workspace allocation.
    case bufferAllocationFailed(String)
    /// `MTLCommandQueue.makeCommandBuffer` returned `nil`.
    case commandBufferCreationFailed
    /// The command buffer completed with an error.
    case commandBufferFailed(String)
}

// MARK: - Tensor name resolution + validation helpers

extension T5MetalEmbedder {

    fileprivate static let tensorEmbedTokens = "encoder.embed_tokens.weight"
    fileprivate static let tensorFinalNorm = "encoder.final_layer_norm.weight"
    fileprivate static let tensorRelposBias =
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"

    fileprivate static func preAttnNormName(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.0.layer_norm.weight"
    }
    fileprivate static func qName(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.0.SelfAttention.q.weight"
    }
    fileprivate static func kName(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.0.SelfAttention.k.weight"
    }
    fileprivate static func vName(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.0.SelfAttention.v.weight"
    }
    fileprivate static func oName(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.0.SelfAttention.o.weight"
    }
    fileprivate static func preFFNNormName(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.1.layer_norm.weight"
    }
    fileprivate static func wi0Name(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.1.DenseReluDense.wi_0.weight"
    }
    fileprivate static func wi1Name(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.1.DenseReluDense.wi_1.weight"
    }
    fileprivate static func woName(layer ℓ: Int) -> String {
        "encoder.block.\(ℓ).layer.1.DenseReluDense.wo.weight"
    }

    fileprivate static func requireTensor(
        _ reader: GGUFReader, name: String, available: Set<String>
    ) async throws -> GGUFTensor {
        guard available.contains(name) else {
            throw T5MetalEmbedderError.tensorMissing(name)
        }
        return try await reader.tensor(name)
    }

    fileprivate static func requireDtype(
        _ tensor: GGUFTensor, _ expected: GGUFDType
    ) throws {
        if tensor.dtype != expected {
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(tensor.name) dtype \(tensor.dtype) ≠ expected \(expected)"
            )
        }
    }

    // Returns `GGUFTensor` (which is `@unchecked Sendable`) rather than
    // `MTLBuffer` so the result can cross the actor-isolated `init`'s
    // `await` boundaries; the caller pulls `.buffer` synchronously.
    fileprivate static func requireFP32Vector(
        _ reader: GGUFReader, name: String, length: Int,
        available: Set<String>
    ) async throws -> GGUFTensor {
        let tensor = try await requireTensor(reader, name: name, available: available)
        try requireDtype(tensor, .f32)
        guard tensor.elementCount == length else {
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(name) elements \(tensor.elementCount) ≠ expected \(length)"
            )
        }
        return tensor
    }

    fileprivate static func requireQ4KMatrix(
        _ reader: GGUFReader, name: String, N: Int, K: Int,
        available: Set<String>
    ) async throws -> GGUFTensor {
        let tensor = try await requireTensor(reader, name: name, available: available)
        try requireDtype(tensor, .q4k)
        // GGUF on-disk fastest-varying-first: shape == [K, N].
        guard tensor.shape.count == 2,
              tensor.shape[0] == K,
              tensor.shape[1] == N else {
            throw T5MetalEmbedderError.tensorShapeMismatch(
                "\(name) on-disk shape \(tensor.shape) ≠ expected [\(K), \(N)] (fastest-varying-first)"
            )
        }
        return tensor
    }
}

#endif // canImport(Metal)
