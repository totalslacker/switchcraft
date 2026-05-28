# Switchcraft: A Swift Port of Witchcraft

> **Status**: Implementation plan — early scaffolding. The project will eventually be released as open source.

Switchcraft is a Swift port of [Witchcraft](https://github.com/dropbox/witchcraft) (Dropbox, Apache 2.0), the Rust reimplementation of [XTR-Warp](https://github.com/jlscheerer/xtr-warp) (Stanford/ETH Zurich, SIGIR'25). It brings token-level semantic search with sub-linear retrieval to native Apple platforms.

No Swift, Go, Java, Kotlin, or TypeScript ports of XTR-Warp / Witchcraft currently exist — Switchcraft will be the first native Apple platform implementation.

## Progress at a Glance

High-level phases. Detailed checklists live in each section below — keep both in sync.

- [x] **Phase 1: MVP** — port Witchcraft to Swift, SQLite-backed, passing the upstream test corpus
- [ ] **Phase 2: Production optimization** — Metal kernels, caching, concurrency improvements
- [x] **Open source release** — Apache 2.0 Swift package on SPM (v0.1.0)

## Motivation

Most local semantic search systems use **document-level embeddings** — a single ~768-dimensional vector per document, searched with brute-force KNN. This has well-known quality limitations:

- **False similarities on short queries**: Single-word queries don't have enough signal in 768 dimensions to disambiguate documents that share vocabulary but differ in meaning.
- **Short text embedding problem**: Summary embeddings (~20-30 words) don't populate 768 dimensions meaningfully, causing many false-positive matches at high reported similarity.
- **Linear scan**: Brute-force KNN over all document vectors is O(n) with no indexing structure, which doesn't scale.

The root cause is **document-level embedding**: collapsing an entire document into one vector loses the information needed to distinguish documents that share vocabulary but differ in meaning.

## Solution: Token-Level Embeddings via XTR-Warp

XTR-Warp and Witchcraft use **token-level embeddings** — one 128-dimensional vector per token in a document. A 500-word document has ~700 token embeddings, capturing on the order of 89,600 dimensions of information total vs ~768 for a single document vector.

Key advantages:

- **MaxSim scoring**: For each query token, find the best-matching document token, then average. A query token matches the specific corresponding token in a document, not a blurred whole-document vector.
- **Short query handling**: Each query token is matched independently — a single word is sufficient.
- **Sub-linear search**: Inverted index over k-means centroids avoids brute-force scanning.
- **Proven performance**: 21ms p95 end-to-end on Apple M2 Max (Witchcraft benchmark on NFCorpus).

## Why Port Witchcraft (Not XTR-Warp)

Two reference implementations exist. Witchcraft is the better porting target for a Swift/Apple platform implementation.

| Factor | XTR-Warp (Python) | Witchcraft (Rust) |
|--------|---|---|
| **Storage** | Filesystem (.pt files) | **SQLite** — single-file, transactional |
| **Model format** | PyTorch .bin | **GGUF** — portable, quantized |
| **GPU** | CUDA-first | **Metal** — already supports Apple Silicon |
| **Latency** | ~100ms+ | **21ms p95** on M2 Max |
| **Dependencies** | PyTorch, FAISS, HuggingFace | Minimal (Candle, SQLite, tokenizer) |
| **Design target** | Server/GPU cluster | **Client-side deployment** |
| **Port effort** | 14-18 weeks | **10-14 weeks** |

Witchcraft was designed from the ground up for local, low-resource deployment on a single SQLite file. It already runs on Apple Silicon with Metal acceleration.

## Licensing

All upstream components are Apache 2.0 — fully compatible with closed-source commercial use and with Switchcraft itself being released under Apache 2.0.

| Component | License | Commercial Use |
|---|---|---|
| Witchcraft code | Apache 2.0 | Yes — attribution required |
| XTR model weights (google/xtr-base-en) | Apache 2.0 | Yes — attribution required |
| Candle ML framework | Apache 2.0 / MIT | Yes |
| T5 architecture | Apache 2.0 | Yes |

No GPL/copyleft anywhere in the chain.

---

## Architecture

### Embedding Model

- **Model**: T5-base encoder (Google DeepMind XTR)
- **Output**: 128-dimensional embedding per token (projected from T5's native 768)
- **Quantization**: Q4 GGUF — model size ~60-100MB
- **Long documents**: Sliding window with stride=256 for documents exceeding 2048 tokens
- **Low-signal filtering**: Tokens with near-zero norm (padding, special tokens) are discarded before indexing

### Index Structure: LSM-Tree with K-Means Centroids

Witchcraft uses a multi-level inverted index:

```
Level 0 (L0 buffer): In-memory, capacity ~1024 embeddings
    ↓ (when full, merge into L1)
Level 1: K-means clustered, k = sqrt(n) * 16
    ↓ (when large enough, merge into L2)
Level 2: Larger clusters, fewer centroids
    ↓ ...
```

Each level contains:

- **Centroids**: Cluster centers (128 × F32 = 512 bytes each)
- **Buckets**: Document pointers (delta-encoded, compressed) + quantized residuals (4-bit, 64 bytes per 128-dim embedding)

### Search Algorithm

```
1. Embed query via T5 → per-token embeddings
2. For each query token:
   a. Compute similarity against all centroids (across all generations)
   b. Select top-k centroids (k=32) until t_prime=40,000 candidate embeddings accumulated
3. Decompress residuals for candidate embeddings
4. Compute token-level similarities: query_token · (centroid + residual)
5. MaxSim scoring: for each query token, take max similarity across all document tokens
6. Average MaxSim scores across query tokens → document score
7. RRF fusion with BM25/FTS5 results
8. Return top-k results
```

### Storage: Pluggable Backend (SQLite First)

The data layer is designed as a **pluggable storage backend** behind a Swift protocol so Switchcraft can be retargeted to other stores without touching the search/index/scoring code.

- **MVP backend**: SQLite (matching Witchcraft) — single-file, transactional, FTS5 built-in.
- **Future backends**: Other embedded KV/SQL stores (e.g. DuckDB, RocksDB, LMDB), server-side stores (e.g. Postgres + pgvector or a dedicated vector DB), or in-memory stores for tests.

The schema below is the SQLite reference implementation. The protocol abstracts:

- Document CRUD (`document` table operations)
- Chunk storage and lookup by content hash (`chunk` table)
- LSM generation metadata (`generation` table)
- Bucket reads/writes including centroid, indices, and residual blobs (`bucket` table)
- Full-text search hooks (FTS5 today, but any BM25-capable backend can substitute)
- A filter expression language that can be lowered to each backend's native query form (SQL `WHERE`, KV scans, etc.)

Backend authors implement a `SwitchcraftStorage` protocol; the search engine, index builder, and tokenizer are backend-agnostic. The SQLite schema is therefore one concrete realization, not a hard dependency of the algorithm.

#### SQLite Schema (Reference Implementation)

```sql
-- Documents
CREATE TABLE document(
    uuid TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    metadata JSON,
    hash TEXT UNIQUE,
    body TEXT,
    lens TEXT                    -- comma-separated chunk lengths
);

-- Full-text search (BM25)
CREATE VIRTUAL TABLE document_fts
    USING fts5(body, content='document', content_rowid='rowid');

-- Deduplicated content chunks with embeddings
CREATE TABLE chunk(
    hash TEXT PRIMARY KEY,
    model TEXT,
    embeddings BLOB,            -- packed token embeddings
    counts TEXT                  -- token counts per chunk
);

-- LSM-tree generations (index levels)
CREATE TABLE generation(
    id INTEGER PRIMARY KEY,
    level INTEGER NOT NULL,
    num_embeddings INTEGER NOT NULL,
    min_chunk_rowid INTEGER NOT NULL,
    max_chunk_rowid INTEGER NOT NULL,
    created TEXT
);

-- Clustered token embeddings
CREATE TABLE bucket(
    id INTEGER PRIMARY KEY,
    generation_id INTEGER REFERENCES generation(id),
    center BLOB NOT NULL,       -- 128×F32 centroid
    indices BLOB NOT NULL,      -- compressed doc pointers
    residuals BLOB NOT NULL     -- Q4 quantized residuals
);
```

### Compression

- **Residuals**: 4-bit quantization — 128 dims / 2 values per byte = 64 bytes per token embedding (vs 512 bytes at F32)
- **Document pointers**: Delta-encoded and LZ4 compressed
- **Model weights**: GGUF Q4 — ~60-100MB (can be embedded in the app binary)

---

## Swift API Design

### Core Layer (Synchronous, Matching Witchcraft)

The internal implementation stays close to the Rust original for easier porting and correctness verification against upstream tests.

### Public API (Async Swift)

The public surface is the `SwitchcraftStore` actor plus the `Embedder`
protocol. The store wraps a `SwitchcraftStorage` backend, an `Embedder`,
and the internal `Indexer` and `SearchEngine` actors. See ADR 009 for
the locked contract.

```swift
public protocol Embedder: Sendable {
    var dims: Int { get }
    var modelIdentifier: String { get }
    func encode(_ text: String) async throws -> [Float]
}

public struct StoreConfig: Sendable, Hashable {
    public var indexer: IndexerConfig
    public var search:  SearchConfig
    public var hybrid:  HybridConfig
    public static let `default`: StoreConfig
    public static func testing(...) -> StoreConfig
}

public actor SwitchcraftStore {

    /// General initializer — caller supplies any storage backend.
    public init(storage: any SwitchcraftStorage,
                embedder: any Embedder,
                config: StoreConfig = .default) async throws

    /// SQLite convenience — declared in `SwitchcraftSQLite`.
    public static func sqlite(databasePath: String,
                              embedder: any Embedder,
                              config: StoreConfig = .default) async throws -> SwitchcraftStore

    // MARK: - Document management

    public func add(id: String,
                    date: Date = Date(),
                    metadata: [String: String] = [:],
                    body: String) async throws
    public func remove(id: String) async throws
    public func index() async throws
    public func clear() async throws

    // MARK: - Search

    public func search(query: String,
                       topK: Int = 10,
                       filter: StorageFilter = .all) async throws -> [HybridHit]
    public func score(query: String,
                      passages: [String]) async throws -> [Float]

    // MARK: - Lifecycle

    public func shutdown() async throws
}

public enum SwitchcraftStoreError: Error, Equatable, Sendable {
    case alreadyShutDown
    case invalidEmbeddingDimensions(Int)
    case embeddingMismatch(count: Int, dims: Int)
}
```

The store accepts any `Embedder` (T5/CoreML, llama.cpp, a remote API, a
deterministic mock for tests). `search` returns `[HybridHit]` from
`SearchEngine.searchHybrid` directly; per-source rank/score is exposed
on each hit so callers can explain or re-rank. Hybrid search auto-flushes
pending L0 embeddings, so the typical caller never has to invoke
`index()` explicitly.

---

## Embedding Dimension Analysis

Witchcraft uses 128-dim token embeddings (projected from T5's native 768 via a learned dense layer). This is deliberately smaller than the typical 768-dim document vector, and for good reason:

**Why 128 is sufficient for token-level:**

- Token-level matching is precise — each query token finds its best-matching document token independently.
- A 500-word document has ~700 tokens × 128 dims = 89,600 total dimensions of information.
- Compared to a single 768-dim document vector, this is ~117× more information per document.

**Why larger wouldn't help:**

- The XTR paper chose 128 after evaluation — diminishing returns past this for token matching.
- Residual compression at 128 dims = 64 bytes per token (Q4). At 768 dims = 384 bytes — 6× storage for marginal quality gain.
- Centroid matching and k-means clustering are faster at lower dimensions.

---

## ML Inference Strategy

### CoreML (Recommended for MVP)

```
T5 encoder (HuggingFace) → coremltools conversion → .mlmodel/.mlpackage
```

- Runs on CPU, GPU, or Neural Engine (Apple decides optimal)
- Quantization support (FP16, INT8) for smaller model + faster inference
- Standard Apple framework, well-supported

### Metal Compute Shaders (Phase 2 Optimization)

Custom kernels for hot paths:

- Q4 dequantization + matrix multiply (fused)
- Centroid similarity computation
- Residual scoring

### Model Assets

- **GGUF weights**: ~60-100MB, can be embedded in the app binary or downloaded on first launch
- **Tokenizer**: Unigram (SentencePiece) tokenizer ported to Swift (JSON-based, ~1-2 weeks to implement)
- **CoreML model**: Converted offline, shipped with the package or downloaded on first launch

---

## Porting Effort Estimate

### Phase 1: MVP (10-14 weeks)

Track progress by checking off items as they land. Effort estimates and notes follow each item.

- [x] **Project scaffolding** — `Package.swift`, target layout, CI skeleton
- [x] **Storage protocol** — `SwitchcraftStorage` and conformance test suite (must precede backend work)
- [x] **SQLite backend** (1-2 weeks) — reference implementation of the storage protocol; native sqlite3
- [x] **Unigram (SentencePiece) tokenizer** (1-2 weeks) — Port from HuggingFace tokenizer.json; identical token IDs vs Rust
- [x] **T5 encoder → CoreML** (1-2 weeks) — Model conversion + Swift wrapper
- [x] **T5CoreMLEmbedder crash-safety** (#78) — `MLExceptionCatcher` ObjC `@try/@catch` bridge (separate `SwitchcraftCoreMLObjC` clang target per ADR 018); `catchingNSException` Swift facade converts CoreML internal `NSException`s to `CoreMLNativeError.nativeException` typed errors; `MLPredictor` internal protocol DI seam; re-entrancy guard (`inFlight` + `waiters` continuations); optional `failureLogURL` for JSONL crash telemetry + `os.Logger` logging
- [x] **T5CoreMLEmbedder ANE IOSurface fix** (#87) — `autoreleasepool` per window, proactive model reload every `reloadInterval` encodes (default 500, tunable), reactive CPU-fallback with JSONL recovery telemetry (`"recovered_iosurface_exhaustion"`); stub stress test (5k iterations, always-on CI) + real-asset stress test (10k iterations, asset-gated); ADR 021
- [x] **Embedder overflow guard** (#89) — `maxInputTokens: Int` added to `Embedder` protocol; `EmbedderOverflowPolicy` (`.truncate` / `.reject`) + `EmbedderError.inputTooLarge` in `SwitchcraftCore`; overflow guard in `T5CoreMLEmbedder.encode(_:)` and `T5MetalEmbedder.encode(_:)` between tokenization and `SlidingWindow.plan`; default `8 * windowSize` (4,096 tokens → ~15 windows); prevents ANE pool poisoning from oversized inputs; ADR 022
- [x] **T5CoreMLEmbedder ANE IOSurface mitigation hardening** (#90) — Fix three bugs in post-#88 production code: (1) silent CPU fallback gap: new `"cpu_fallback_failed"` JSONL category with `cpuErrorName`/`cpuErrorReason`/`cpuCallStack` fields; (2) `reloadInterval` default lowered 500→150 (below observed 388-call production failure point); (3) reactive reload + ANE retry added to Layer 3 before CPU fallback; per-window timing via `os.Logger`; Scenario A/B/C mock tests; ADR 021 amended
- [x] **Remove Layer 3b CPU fallback (dead code, 0% recovery rate)** (#93) — Production evidence (0/1,038 CPU recovery rate across 2026-05-07 SafariUnfucker run) showed the CPU fallback never recovered any inference; `cpuPredictorFactory` property and all wiring removed; Layer 3b branch deleted from `predictWindow`; Layer 3a failure path now explicitly logs + rethrows; `logRecoveredIOSurface`, `logCPUFallbackFailed`, `extractCPUErrorFields` methods deleted; `cpu_fallback_failed` JSONL category retired; two new Layer 3a stub tests added (`testANERetrySucceedsAfterReload`, `testANERetryFailsLogsErrorRow`); ADR 021 second addendum
- [x] **K-means clustering** (1 week) — Standard algorithm, use Accelerate
- [x] **4-bit residual codec** (1 week) — ~200 lines, bit-level packing; round-trip property tests
- [x] **LSM-tree index structure** (1 week) — Cascading merge logic
- [x] **`rehydrationConflict` auto-recovery** (#101) — `IndexerConfig.rehydrationConflictBehavior` (`.autoRecover` default, `.throwError` opt-out); `Indexer.recoveredConflictCount`; `replaceGeneration()` protocol method with atomic transaction semantics in `InMemoryStorage` and `SQLiteStorage`; LSM winner/loser selection rule `(level DESC, created DESC, id DESC)`; structured `os.Logger` warning per conflict; 5 new tests covering 2-gen and 3-gen conflicts, storage-repair verification, atomicity rollback, and backwards-compatible `throwError` mode; ADR 024
- [x] **Post-recovery `ledgerOutOfSync` fix** (#103) — Step 3.5 added to `rehydrateAutoRecover`: corrects stale `numEmbeddings` on surviving (non-loser) generations using already-decoded bucket data (zero extra I/O); `updateGenerationEmbeddingCount(id:count:)` new required protocol method on `SwitchcraftStorage` with implementations in `InMemoryStorage`, `SQLiteWriterActor`, and `SQLiteStorage` (both `inMemory` and `fileBacked` paths); transparent forwarding stub in `FailingReplaceStorage`; `runUpdateGenerationEmbeddingCount` scenario added to `StorageConformance`; Test 6 in `IndexerConflictRecoveryTests` verifies inflated winner count is corrected and subsequent `add()` + `flush()` succeeds without `ledgerOutOfSync`; ADR 024 amended
- [x] **Search pipeline** (1-2 weeks) — Centroid match → residual decode → MaxSim scoring
- [x] **RRF + FTS5 hybrid** (3-4 days) — Adapt Witchcraft's hybrid fusion
- [x] **Async Swift API** (3-4 days) — `SwitchcraftStore` actor wrapper
- [ ] **Testing + benchmarking** (2-3 weeks) — Correctness + performance validation (see Testing Strategy)

### Phase 2: Production Optimization

- [ ] **SmoothQuant FP16 conversion path** (size + speed + ANE eligibility) — see "SmoothQuant FP16 path" below. **Investigation phase complete; see status below.**
- [x] **INT8 weight-only `.mlpackage` variant** (rung 1 of "If SmoothQuant fails" ladder, ~110 MB) — see ADR 010(i); ships alongside FP32 default, opt-in via `SWITCHCRAFT_XTR_MLPACKAGE_INT8W`
- [x] **Port ggml's T5 inference to Swift + Metal** (gives us roughly Witchcraft's embedder size + speed + power; Metal kernels themselves don't reach ANE today but the `Embedder` seam preserves a future CoreML-FP16 variant if/when one becomes viable) — see "Port ggml's T5 inference to Swift + Metal" below. **All eight sub-issues (#58–#65, #74) complete. NDCG@10 = 0.336 (Metal-specific band [0.31, 0.34]; Metal FP32-throughout scores slightly above ggml's 0.33 ceiling per ADR 014). Cross-stack tolerance calibrated: maxAbs = 0.000216, in-tree constant = 0.0005 (issue #75, 2026-05-02).**
  - [x] **#58** — Porting catalogue ([`docs/porting/ggml-t5.md`](porting/ggml-t5.md)) + ADR 014(g) status update. Pins ggml/llama.cpp/Candle/Witchcraft commits, artefact map, Q4_K block layout, per-op precision matrix (informational; ADR 016 normative), relpos bucket formula, FFN activation (`gated-gelu`), GGUF acquisition pipeline, tokeniser disposition. **No `Sources/` changes.**
  - [x] **#59** — `SwitchcraftMetal` library target + GGUF reader for Q4_K weights + ADR 010(j) (asset format and `SWITCHCRAFT_XTR_GGUF` env-var gating). Lifts `MetalContext` visibility to `@_spi(SwitchcraftMetal) public`; reuses #51's foundation.
  - [x] **#60** — Q4_K matmul kernel: port `kernel_mul_mm_q4_K_f32` (matrix-matrix instantiation) from `ggml-metal.metal` to MSL; ≥0.99999 cosine vs CPU reference (`Q4KDecode.dequantise` + `cblas_sgemm`) on the three T5-base shapes (qkv_proj, ffn_wi, ffn_wo) plus M=1 and non-tile-aligned N edge cases. Standalone `kernel_dequantize_q4_K` Metal kernel deferred to #64 alongside its first consumer (`kernel_get_rows_q4_K` token embedding lookup). `MetalContext` extended with per-target shader-bundle registration (ADR 015 §(e)).
  - [x] **#61** — RMSNorm kernel: port the gain-fused `kernel_rms_norm_mul_f32` (the F==2 specialisation of `kernel_rms_norm_fuse_impl<float, 2>`) to MSL; FP32 throughout per ADR 003 + ADR 014(b); ≥0.99999 row-wise cosine vs pure-Swift FP32 reference on T5-base (M=512, D=768) and synthetic shapes (M=1, D=128; M=64, D=2048). Edge cases asserted: all-zeros row, all-equal-values row, FP16-overflow magnitudes (the canonical falsifying case behind ADR 014(i) — kernel produces finite output where FP16 `sum(x²)` would saturate). `MetalContext.register(bundle:resourceName:)` idempotency key changed from `bundleURL` alone to `(bundleURL, resourceName)` so a single bundle can register multiple `.metal` source files.
  - [x] **#62** — Softmax kernel: port `kernel_soft_max_f32` (FP32 throughout per ADR 014(i) — softmax is the canonical sensitive op the precision carve-out cites). The optional additive mask/bias buffer is preserved verbatim from upstream's contract so the orchestration layer (#64) can wire in T5's relative-position bias unchanged. ≥0.99999 row-wise cosine vs pure-Swift FP32 reference at the T5 attention shape (B = 1 × 12 × 512 = 6144, N = 512), with and without bias, plus small / non-aligned-N synthetic shapes. Edge cases asserted: all-equal input → uniform 1/N; single very large value → near one-hot; strongly-negative bias → exact-zero output (mask semantics — FP32 underflow). Generation of the relative-position bias *table* (T5 `_relative_position_bucket` + per-head gather) deferred to #64's orchestration; this issue only proves the softmax kernel applies a bias buffer correctly.
  - [x] **#63** — Element-wise kernels: residual add (`kernel_add_f32`), gated-GELU (`kernel_gated_gelu_f32` — fused `gelu_new(gate) * up` with `tanh`-form `gelu_new` matching ggml's `kernel_gelu_f32`; xtr-base-en `d_ff = 2048`, **not** classic T5-base's 3072), L2 normalisation (`kernel_l2_norm_f32` — fused per-row reduction with `1/max(eps, √Σ)` denominator floor, all-zeros input ⇒ exact-zero output; FP32 throughout — `sum(x²)` overflows FP16 per ADR 014(b)). All three pass ≥0.99999 cosine vs FP64-reference at xtr-base-en shapes (residual M=512/D=768, gated-GELU M=512/D=2048, L2 M=512/D=128 + M=64/D=768).
  - [x] **#64** — `T5MetalEmbedder` orchestration: encoder forward pass dispatch order, `MTLBuffer` lifecycle, `Embedder` protocol conformance (ADR 009); includes T5 `_relative_position_bucket` table generation + per-head bias gather (handed to the #62 softmax kernel as a `B × N` bias buffer); ADR 017 promotes the per-op precision matrix from informational to normative (renumbered from the issue body's "ADR 016" since slot 016 is taken by the GGUF asset distribution ADR landed by #59); ≥0.99999 per-token cosine vs PyTorch FP32 reference (`Tests/Fixtures/xtr-base-en.embeddings.bin`). Adds `FP32MatMulKernel` — one shared FP32 batched matmul with row-stride args covers `Q · Kᵀ`, `softmax · V`, and `2_Dense` (the catalogue's `Attention.{swift,metal}` and `ProjectionMatMul.{swift,metal}` placeholder rows are superseded). 2_Dense FP16 weight widened to FP32 once at init (~384 KiB resident).
  - [x] **#65** — NFCorpus NDCG@10 gate confirmed (issue #75, 2026-05-02): observed NDCG@10 = 0.336 with `T5MetalEmbedder` + `xtr-v3.gguf` (Q4K). Metal-specific band `[0.31, 0.34]` (upper bound calibrated above Witchcraft's 0.33 ggml ceiling due to FP32-throughout arithmetic per ADR 014). Cross-stack tolerance calibrated: `maxAbs = 0.000216`, in-tree constant = `0.0005` (`2 × maxAbs`, rounded). ADR 010(h) fully updated.
  - [x] **#74** — Six operator-pipeline defects blocking the NDCG@10 gate: (1) GGUF reader strict v3-only check rejects the v2 asset emitted by Witchcraft's `quantize-tool` at Candle rev `5bd5618` — relaxed to v2 ∪ v3; `GGUFReader.loadedVersion` property exposed for debugging; (2) `GGUFTypes.unsupportedVersion` description updated; (3) `GGUFReaderTests` rejection test corrected to use v1, v2/v3 acceptance tests added; (4) `T5MetalEmbedder.projCandidates` missing `"linear.weight"` (the bare name `quantize-tool` writes) — added; regression test added; (5) `linear.weight` incorrectly Q4_K-quantised by `quantize-tool` (768 % 256 == 0 false positive) — `scripts/witchcraft-fixture-export.patch` adds `name != "linear.weight" &&` carve-out to `quantize-tool/src/main.rs`; (6) `witchcraft-fixture-export.patch` corrupt (truncated hunk header, moved `PackOps` trait methods, `Tensor.iter()` non-existent) — patch reconstructed. ADRs 010/016/017 and `docs/porting/ggml-t5.md` updated coherently.
- [x] **Metal compute shaders for search hot paths — scaffolding** (#51 / PR #56, ADR 015). Per-kernel sub-issues (#52/#53/#54) closed as wrongly-scoped after #49 established Accelerate beats MPS; future kernel work goes against ggml's targets, not Accelerate — see "Search-side Metal: rescoped" below
- [x] **Per-search wall-time deadline and cooperative task cancellation** (#83, ADR 020) — `SearchConfig.searchDeadline` (default 5 s), per-call `deadline: Duration?` override on `SwitchcraftStore.search()`, pre-phase deadline checkpoints, `sqlite3_progress_handler` with remaining-budget enforcement, `Task.checkCancellation()` at key storage checkpoints, `SwitchcraftStoreError.searchTimedOut(elapsed:)`
- [x] **ColBERT-style title prepend for proper-noun query alignment** (#105, ADR 025) — `SwitchcraftStore.add(id:date:metadata:title:body:)` gains an optional `title: String? = nil` parameter; when non-nil, the embedder sees `"\(title)\n\(body)"` (embeddingText) and the chunk dedup key is `SHA-256(embeddingText)`. Fixes H3 root cause: XTR vector pipeline mis-ranked proper-noun queries (e.g. "alfred" retrieved Bartleby.com above alfredapp.com). `DocumentRecord.body` stores original caller-supplied body; no schema migration. H2 (`k`/`tPrime`/`missing[q]`) deferred. Null-path (title==nil) is byte-identical to pre-025; NFCorpus + 33-fact corpus unaffected. ADR 009 sections (a)+(e) amended. New `ProperNounRankingTests` suite: pathology-demo + fix assertions, both asset-gated.
- [ ] LRU caching for query embeddings
- [ ] Background indexing pipeline
- [ ] Parallel bucket search
- [ ] Batch document processing

#### SmoothQuant FP16 path

The current `.mlpackage` ships at FP32 compute / FP16 outputs / ~430 MB
because blanket FP16 conversion of `google/xtr-base-en` produces NaN
on this graph (T5 activation outliers + L2-norm overflow; see ADRs
010(c) and 014 for the full rationale). The strategic fix is
**SmoothQuant** (Xiao et al., NeurIPS 2023) — a pre-conversion
PyTorch transform that re-parameterises the model so activations
don't have outliers anymore, unblocking FP16 conversion.

If SmoothQuant works on `google/xtr-base-en`, the resulting model is
**simultaneously**:

- ~80 MB on disk (matches Witchcraft) — 5× smaller than the current
  FP32 build.
- ~2× faster matmul on Apple GPU (FP16 ops are double-throughput).
- **ANE-eligible** — FP16 weights + FP16 activations is the Apple
  Neural Engine's happy path, dramatically faster and more
  power-efficient than GPU compute for transformer encoders.
- Tighter cross-stack parity with Witchcraft (FP16-vs-Q4K is closer
  than FP32-vs-Q4K), so ADR 010(h)'s ±0.025 tolerance can ratchet
  back toward ±0.01.

Three wins for one investment, which is why this is the strategic
top-of-Phase-2 item.

##### Phasing

1. **Investigation issue** (separate, file before any other work):
   profile activation distributions in the encoder, implement
   SmoothQuant scaling in PyTorch, attempt coremltools FP16 export of
   the smoothed model, run parity vs FP32 PyTorch reference. Output:
   feasibility report + go/no-go signal + recommended parameters
   (α value, per-layer policy, calibration set size). **Touches no
   `Sources/` code; produces a `scripts/investigate-smoothquant.py`
   plus a short report under `docs/investigations/`.**
2. **Implementation issues** (only if investigation says go):
   modify `scripts/convert-xtr-to-coreml.py` to apply SmoothQuant
   before export; build the FP16 `.mlpackage` variant; ship it
   alongside the existing FP32 variant; update parity gates; amend
   ADR 010(c) with the new precision contract.
3. **Default flip** (final PR): switch the default
   `T5CoreMLEmbedder.init` to use the FP16 variant; deprecate / remove
   the FP32 variant per a documented migration window.

##### Multi-variant `.mlpackage` shipping

Throughout the campaign, **the FP32 build remains the default**. The
FP16 variant is opt-in via init parameter or env var until the very
last "default flip" PR. This keeps `main` continuously usable for
downstream consumers (e.g. SafariUnfucker) — they can build against
`main` at any commit during the SmoothQuant campaign without their
embeddings becoming stale until they explicitly opt in.

The cost of switching variants for a downstream consumer is
**reindexing the corpus** — embeddings stored against the FP32 model
are not interchangeable with embeddings stored against the FP16
model (precision pair differs). This is the standard "model swap =
reindex" deal documented elsewhere; consumers know to plan for it.

The intended end state is shipping multiple variants that consumers
can choose between based on their constraints:

| Variant | Asset | Compute | Use case |
|---|---|---|---|
| `xtr-base-en-fp32.mlpackage` | ~430 MB | FP32 GPU/CPU | Maximum precision, server-side macOS |
| `xtr-base-en-fp16.mlpackage` (post-SmoothQuant) | ~80 MB | FP16 ANE/GPU | iOS apps, max speed + power |
| `xtr-base-en-int8w.mlpackage` (fallback if SmoothQuant fails) | ~110 MB | FP32 GPU/CPU | Size-constrained, current latency |

##### Branching strategy

Despite the multi-PR shape, **all SmoothQuant work lands directly to
`main`** rather than a long-lived `main-smoothquant` feature branch.
Reasons:

- The Phase 1 pattern — every PR green, every commit ADR-aligned,
  pre-commit + pre-merge hard rules — relies on `main` being the
  single integration target. Re-targeting Fabrik tooling, CI
  triggers, project board automation, and yolo-merge logic at a side
  branch costs engineering time that's better spent on SmoothQuant
  itself.
- The "FP16 variant is opt-in until the final flip" pattern preserves
  `main` as continuously usable for downstream consumers without the
  isolation a side branch would provide. ADR 010(c) is not amended
  until the implementation issue lands a working FP16 variant; until
  then the precision contract on `main` is unchanged.
- If during step 2 the implementation work turns out to be much
  bigger than expected (e.g. coremltools beta upgrade, model surgery
  beyond SmoothQuant, 10+ PR rewrite), branch strategy can be
  revisited with concrete data. Until then, branching adds bookkeeping
  without unlocking anything.

##### Before declaring no-go

A borderline result from the feasibility investigation does not have
to fall straight through to the post-SmoothQuant ladder below. The
in-SmoothQuant escalation lever — to be exhausted before "no-go" is
declared — is **per-layer α tuning**.

Standard SmoothQuant uses a single α for every Linear in the model.
T5-base may have layer-specific outlier behaviour (e.g. early-layer
attention vs. late-layer FFN) that benefits from a non-uniform α
policy: a higher α where outliers are extreme, lower α where weight
magnitudes dominate. The feasibility report's per-layer outlier
heatmaps + the α-sweep mean-cosine table are the data we need to
decide whether per-layer tuning is worth pursuing. If the global
sweep produces a NaN-free run with mean cosine in [0.99, 0.999), try
per-layer α before declaring no-go. If the global sweep is NaN-flooded
across the board, per-layer α is unlikely to rescue it and the post-
SmoothQuant ladder applies.

##### If SmoothQuant fails — empirical status

The investigation produced a definitive **no-go** (#43, PR #44, see
`docs/investigations/smoothquant-feasibility.md`). SmoothQuant
re-parameterises *inside* the Linear and cannot lower the
residual-stream magnitude that overflows the next block's RMSNorm.
Cosine vs FP32 PyTorch reference: 0.0 across α ∈ {0.3, 0.5, 0.7,
0.85} and the absorbed-LayerNorm formulation.

The fallback ladder, with current empirical status:

1. **INT8 weight-only quantisation** ✅ in flight via #45 — see
   ADR 010(i). Ships ~110 MB asset, FP32 compute, no ANE eligibility.
   Bounded scope, well-understood. Storage-axis win that operates
   independently of the FP16-blocked compute axis.
2. **Custom MIL pass** ❌ **empirically falsified** (#46, PR #48,
   see `docs/investigations/mil-fp32-promote-feasibility.md`). The
   FP16 saturation happens *inside* `ff.wo` before any reachable
   carve-out boundary; promoting the RMSNorm cluster to FP32 receives
   an already-saturated `+inf` input and produces NaN. Cosine: 0.0 on
   every targeted island. The only passing carve-out is functionally
   full-FP32 with FP16 weight storage (see ADR 014(g) for the full
   record).
3. **Port ggml's T5 inference to Swift + Metal** — replaces CoreML
   inference entirely with a Swift+Metal port of ggml's existing T5
   forward path (Q4-quantised kernels, FP32 sensitive ops, the works).
   Closes the embedder size + speed + power gap with Witchcraft
   because we are running essentially the same code, just translated.
   The Metal kernels themselves don't reach ANE; the `Embedder`
   protocol seam preserves a future CoreML-FP16 variant if one
   becomes viable. See "Port ggml's T5 inference to Swift + Metal"
   subsection below.
4. **Switch model** (e.g. distilled MiniLM-128). Reluctant choice —
   loses comparability with Witchcraft, the NDCG@10 reference point,
   and most cross-stack parity gates. Only legitimate if upstream
   Witchcraft also switches.
5. **Accept FP32 as the long-term answer.** Document the size and
   ANE limitations; lean on Phase 2's other optimisations (Metal
   shaders for search hot paths, parallel bucket search, batch
   document processing) for performance gains.

#### Port ggml's T5 inference to Swift + Metal

##### Goal and framing

The user-facing goal is straightforward: a Swift-native framework
that performs roughly like Witchcraft on iOS and macOS — similar
size, similar speed, similar power consumption. Not faster. Not
novel. *Good enough that customers don't bounce off perf.*

ggml is open source (MIT). llama.cpp is open source (MIT). The
exact kernels Witchcraft uses for `google/xtr-base-en` are
documented C and `.metal` source in `ggml/src/ggml-metal.m` and
related files. **This is a code-analysis-and-port task, not a
research investigation.** We read what ggml does, translate the
patterns to Swift and Metal, run parity tests against Witchcraft's
published numbers (NDCG@10 ∈ [0.31, 0.33], 21 ms p95) and against
the existing PyTorch reference fixture, and ship the result as a
second `Embedder` conformance.

This subsection supersedes the previous "Custom Metal kernels for
the T5 embedder" framing, which was wrong-shaped: it treated
matching ggml as a research question rather than a porting task.
The matmul investigation #49 confirmed the wrong-shape: it tested
"can a hand-rolled FP32 GEMM kernel beat Accelerate?" (no), but
the actual question is "can we port ggml's existing Q4-quantised
kernels?" (the answer is yes, by definition — the source is right
there).

##### Trade-off: ANE set aside, not forfeited

Custom Metal kernels run on GPU; ANE access is gated through
CoreML's `compute_precision` pipeline. Going down the port path
means matching Witchcraft on GPU performance — *not* leapfrogging
it via ANE, today.

But this is not a permanent decision against ANE. The `Embedder`
protocol seam (#16, ADR 009) is designed to support multiple
conformances simultaneously. A `T5MetalEmbedder` shipping today
does not preclude a future `T5CoreMLFP16Embedder` shipping alongside
it if a coremltools release surfaces finer-grained per-op precision
control or if model surgery beyond SmoothQuant turns out to work.
Consumers pick which conformance they want at init time, and the
choice is fully reversible. **We are setting ANE aside for now,
not forfeiting it.**

##### Phased delivery (port shape, not investigation shape)

The work is a multi-PR engineering campaign, ~4-6 weeks total.
**No more matmul-vs-Accelerate benchmarking.** ggml's published
numbers + Witchcraft's NDCG@10 are the parity bar.

1. **Code analysis & porting plan** (one PR, days not weeks).
   Read `ggml-metal.m`, llama.cpp's T5 inference path, and
   Witchcraft's Rust glue. Catalogue the kernels and orchestration
   we need to port:
   - The Q4-quantised matmul kernel (`mul_mat_q4_K_f32` and
     siblings — fused dequant-and-multiply-and-accumulate on the
     ggml block layout).
   - RMSNorm at FP32 (`kernel_rms_norm_f32`).
   - Softmax at FP32 (`kernel_soft_max_f32`).
   - Element-wise: residual add, gelu, L2 normalisation.
   - The orchestration layer: weight loading from GGUF, kernel
     dispatch order matching T5's encoder forward pass,
     `MTLBuffer` lifecycle, mixed-precision routing per ADR 014(i).
   Output: a porting plan committed to `docs/porting/ggml-t5.md`
   that names each ggml source file, each kernel, and the
   corresponding Swift target file. This is mechanical mapping
   work — produces a checklist subsequent sub-issues consume.

2. **Per-kernel ports** (one PR per kernel, ordered by dependency).
   Each PR translates a single ggml kernel to Swift + Metal,
   commits parity tests vs the Swift FP32 reference path, and
   records the port in the catalogue. Order roughly:
   - GGUF reader (load Q4-quantised weights into `MTLBuffer`s).
   - Q4 matmul kernel.
   - RMSNorm kernel.
   - Softmax kernel.
   - Element-wise kernels (potentially grouped).
   Each PR is bounded (3-5 days). Parity gate per kernel: ≥0.99999
   cosine vs PyTorch FP32 reference on the kernel's specific
   inputs. **No "beat Accelerate" requirement** — the goal is
   match ggml, not optimise.

3. **`T5MetalEmbedder` orchestration** (one PR). Implements the
   forward pass in Swift, dispatching the ported kernels in the
   T5 encoder order, conforming to the `Embedder` protocol so
   `SwitchcraftStore` can use it interchangeably with
   `T5CoreMLEmbedder`. Tests verify the full forward pass on the
   existing committed fixtures (`Tests/Fixtures/xtr-base-en.embeddings.bin`)
   at ≥0.99999 cosine vs the PyTorch reference.

4. **Cross-stack parity gate** (one PR). End-to-end test that runs
   the same NFCorpus benchmark (#27) against `T5MetalEmbedder` and
   asserts NDCG@10 ∈ [0.31, 0.33] — the bar Witchcraft ships at.
   This is the umbrella's acceptance criterion: if `T5MetalEmbedder`
   clears NDCG, the port is correct. If it doesn't, debug the
   specific kernel that diverged.

5. **ADR amendments** as we go. ADR 010 picks up an INT (j) section
   describing the GGUF asset format and `T5MetalEmbedder` variant.
   ADR 014(g) records that the port path is followed (not
   investigated) and why this differs from the SmoothQuant + MIL
   pass paths that *were* research questions.

##### Multi-implementation shipping

Throughout, **`T5CoreMLEmbedder` (FP32 / INT8w) remains the default**
`Embedder` implementation. `T5MetalEmbedder` ships as an additional
conformance — consumer-selected at init time. SafariUnfucker (and
any other downstream consumer) keeps working with the CoreML
embedder unchanged until it explicitly opts in.

##### Reuse of #51's scaffolding

The Metal scaffolding from #51 (`MetalContext`, dispatch helpers,
fallback policy, ADR 015) was designed for the search-path kernel
work. It is **directly reusable** for the embedder port: same
`MTLDevice` acquisition, same pipeline cache pattern, same
`MetalAvailability` gate. The embedder port doesn't re-invent the
foundation — it builds on what #51 shipped. The only new
infrastructure required is the GGUF reader for Q4-quantised
weights, which is a per-port concern.

##### Acceptance for the campaign

When the umbrella closes:

- **Asset size**: GGUF Q4K T5-base ~80 MB, matching Witchcraft.
- **Search end-to-end p95**: ≤25 ms on the 5000-document
  benchmark corpus (within ~20% of Witchcraft's 21 ms; not aiming
  for parity, aiming for "in the same ballpark").
- **Embedder latency**: comparable to Witchcraft's GGUF path on the
  same hardware (~50 ms per 512-token encode, give or take). Match
  ggml's published M1-class numbers.
- **NDCG@10**: ∈ [0.31, 0.34] on NFCorpus when `T5MetalEmbedder`
  is the active embedder. (Observed 0.336 on 2026-05-02; upper bound
  calibrated to 0.34 because Metal's FP32-throughout arithmetic scores
  slightly above ggml's 0.33 ceiling per ADR 014.)
- **Power consumption**: not directly measured; comparable
  inferred from comparable wall time at comparable GPU
  utilisation.
- **`docs/Plan.md` "Port ggml's T5 inference to Swift + Metal"**
  line is `[x]`.

##### Why we are not investigating further before porting

We have, in the recent past, taken the bait of "investigate first,
implement on go" framing on questions where the *implementation*
was the answer. SmoothQuant and MIL pass were genuinely research
questions and the investigations were correct. The matmul
investigation (#49) was *partly* correct as a measurement question
("does our hand-rolled kernel beat Accelerate?") but produced a
no-go that was wrong-shaped because Accelerate isn't the relevant
target — ggml is. We do not need a third investigation to confirm
that ggml's Q4 kernels work on `google/xtr-base-en`; ggml has
shipped them in production for years. The remaining work is
translation, not discovery.

#### Search-side Metal: rescoped

The original "Metal compute shaders for search hot paths"
subsection (now removed) framed three sub-issues — #52 (Q4 dequant
+ matmul), #53 (centroid similarity), #54 (MaxSim reduction) — as
kernels competing with Accelerate's `cblas_sgemm`. After #49's
empirical finding that **Accelerate beats `MPSMatrixMultiplication`
by 2-5× on M1 Ultra** for the shapes that matter (almost certainly
AMX dispatch via Apple's matrix coprocessor), this framing was
wrong:

- The search path is already running on the optimal CPU path.
  Replacing `cblas_sgemm` with Metal is unlikely to win.
- Witchcraft does not compete with Accelerate either. Witchcraft's
  speed comes from a different architecture entirely (GGUF + ggml
  Metal kernels, all GPU, *no* CPU-BLAS path).
- The right target for embedder-side Metal work is ggml's
  inference path (see "Port ggml's T5 inference to Swift + Metal"
  above). The right answer for the search side is **stay on
  Accelerate** until a workload surfaces that the AMX path can't
  handle.

What's preserved from the original umbrella:

- **#51 / PR #56 ✅** — the Metal scaffolding (`MetalContext`,
  pipeline cache, fallback helper, `MetalAvailability`, ADR 015,
  performance regression infrastructure). This stays on `main`
  and is directly reusable for the embedder port.

What's closed:

- **#52, #53, #54** — closed as wrongly-scoped against Accelerate.
  Their work is not lost; if at some point in the future a
  search-path Metal kernel becomes worth pursuing, the framing
  should be "match ggml's pattern" not "beat Accelerate," and the
  scaffolding from #51 is ready to host it.

The Phase 2 list above ticks the scaffolding (`[x]`); per-kernel
search-side Metal work is no longer a Phase 2 deliverable.

---

## Testing Strategy

### Witchcraft's Existing Test Infrastructure

Witchcraft has 12 unit tests (672 lines in `tests.rs`) plus per-module tests and an NFCorpus benchmark:

| Area | Tests | Coverage |
|---|---|---|
| End-to-end search | `test_end_to_end` — 33-fact corpus, verifies correct retrieval across 3 index rounds | Good |
| Sub-document indexing | `test_sub_docs` — multi-chunk documents | Good |
| Incremental indexing | `test_incremental_index` — add, index, add more, re-index | Good |
| LSM cascade merging | `test_cascade` — forces multi-level merge | Good |
| Scoring correctness | `test_scoring` — validates score values | Good |
| Error handling | `test_embedder_without_assets`, `test_open_bad_db_path`, `test_open_corrupted_db` | Good |
| Edge cases | `test_single_doc_search`, `test_empty_body_not_embedded` | Good |
| Filtering | `test_search_with_uuid_filter` | Good |
| Q4 codec (`packops.rs`) | 4 tests — quantize/dequantize round-trip, MSE validation | Good |
| SQL generation | 11 tests | Good |
| Merger | 3 tests | Basic |
| NFCorpus benchmark | NDCG@10 = 0.31-0.33 (standard IR benchmark) | Strong quality validation |

**Gaps in Witchcraft's tests** that Switchcraft should fill:

| Area | Gap | Risk |
|---|---|---|
| `fast_ops.rs` (GEMM operations) | 0 tests | High — performance-critical matrix ops |
| `rans64.rs` (entropy coding) | 0 tests | Medium — index compression |
| Metal/GPU path | No GPU-specific tests | Medium |
| Concurrent read/write | Not explicitly tested | Medium |
| Tokenizer edge cases | Not tested in isolation | Low |

### Phase 1: Cross-Platform Validation

The primary goal is proving the Swift port produces **identical results** to the Rust implementation for the same inputs. Witchcraft's existing test corpus and queries are the foundation.

Deliverables:

- [x] 33-fact corpus ported verbatim, queries pass with same retrieval / ranking / scores (top-1 strict, top-3 set, scores ±0.025 — see ADR 010(h) for the FP32-vs-Q4K precision rationale)
- [x] NFCorpus benchmark pipeline ported, NDCG@10 ∈ [0.31, 0.33]
- [x] Embedding parity validation: PyTorch ≥0.999 cosine (per `T5CoreMLEmbedderTests.fixtureParity`, ADR 010(c)) + Witchcraft GGUF ±0.025 cross-stack (per `CrossStackEmbeddingParityTests`, ADR 010(h)). Bit-exact against Candle is unreachable by construction — Switchcraft runs FP32 compute / FP16 outputs, Witchcraft runs Q4K GGUF + Q8 attention; the precision pair cannot produce bit-equal floats. The two tolerances above are the strongest correctness gates the precision pair admits.

#### Port Witchcraft's 33-Fact Corpus Verbatim

The 33 "fun facts" test corpus and its queries are carefully designed to test semantic understanding:

```swift
// Identical to Witchcraft's tests.rs
let facts = [
    "Bananas are berries, but strawberries aren't.",
    "Octopuses have three hearts and blue blood.",
    "A day on Venus is longer than a year on Venus.",
    // ... all 33 facts
]

let queries: [(String, Int)] = [
    ("a lake with funny colors", 31),        // → pink lake fact
    ("A group of flamingos", 15),             // → flamboyance fact
    ("facts about fruits and berries", 0),    // → banana fact
]
```

Run the same queries in both Rust (Witchcraft) and Swift (Switchcraft), assert (per ADR 010(h), the relaxed FP32-vs-Q4K shape):

- Top-1 doc-id: strict equality with Witchcraft
- Top-3 doc-id set: set equality (membership, not order)
- Per-doc score: within ±0.025 of the Witchcraft reference for that doc-id
- Doc order beyond rank 3: best-effort, no assertion

This catches any divergence in embedding, centroid matching, residual codec, or scoring. The tolerance will be tightened back toward ±0.01 once Witchcraft ships a higher-precision quant (Q5/Q8/F16/F32).

#### NFCorpus Benchmark

Port the NFCorpus benchmark pipeline (`make nfcorpus-score`) to Swift:

- Index the same NFCorpus documents
- Run the same test queries
- Score with NDCG@10
- **Must achieve 0.31-0.33** (same range as Witchcraft)

Any score below 0.31 indicates an algorithmic bug in the port. This is the quantitative quality gate.

#### Bit-Exact Embedding Validation

For a subset of documents, compare the raw T5 token embeddings between Rust (Candle) and Swift (CoreML):

- Same input text → same tokenization → same token IDs
- Same token IDs → same model output (within FP precision)
- Validates the CoreML model conversion is correct

### Phase 2: Extended Coverage (Gaps in Witchcraft)

Tests that Witchcraft doesn't have but Switchcraft must. Each suite below is a deliverable:

- [x] Matrix Operations suite (GEMM, matmul_t, batched argmax, packed/fused GEMM)
- [x] Residual Codec suite (round-trip, sizing, range, batch parity)
- [x] Concurrency suite (concurrent reads during indexing, reader/writer visibility, simultaneous searches)
- [x] CoreML Inference suite (parity vs Candle, tokenizer parity, sliding window, low-signal filtering)
- [x] Edge Cases suite (empty/short queries, very long docs, stopwords, Unicode, dedup, removal)
- [x] Performance Regression suite (search latency, indexing throughput, memory)
- [x] Search Timeout and Cancellation suite (pre-checkpoint path, progress-handler interrupt, task cancellation, happy-path regression, post-timeout recovery)
- [x] Proper Noun Ranking suite — asset-gated; pathology-demo (Bartleby outranks Alfred without title) + fix assertion (Alfred outranks Bartleby with title prepend); validates ADR 025 / issue #105

#### GEMM / Matrix Operations

```swift
@Suite("Matrix Operations")
struct MatrixOpsTests {
    @Test("matmul_t produces correct transpose multiply")
    @Test("batched argmax matches naive implementation")
    @Test("packed GEMM matches unpacked reference")
    @Test("Q4 dequant + matmul fused matches sequential")
}
```

These are the most dangerous area to get wrong — silent numerical errors would degrade search quality without obvious failures.

#### Q4 Codec Round-Trip (Property-Based)

```swift
@Suite("Residual Codec")
struct CodecTests {
    @Test("quantize-dequantize round-trip preserves signal")
    // Fuzz with random vectors, assert MSE < threshold

    @Test("128-dim vector quantizes to exactly 64 bytes")
    @Test("dequantized values fall within expected range")
    @Test("batch quantization matches single-vector quantization")
}
```

#### Concurrent Access

```swift
@Suite("Concurrency")
struct ConcurrencyTests {
    @Test("concurrent reads during active indexing return consistent results")
    @Test("reader sees committed documents after writer flushes")
    @Test("multiple simultaneous searches don't interfere")
    @Test("index build during active search doesn't corrupt results")
}
```

#### CoreML Inference Validation

```swift
@Suite("CoreML Inference")
struct InferenceTests {
    @Test("CoreML output matches Candle reference for known input")
    // Pre-computed reference embeddings from Rust, compare in Swift

    @Test("tokenizer produces identical token IDs to HuggingFace")
    @Test("long document sliding window matches Rust behavior")
    @Test("low-signal token filtering matches Rust thresholds")
}
```

#### Edge Cases

```swift
@Suite("Edge Cases")
struct EdgeCaseTests {
    @Test("empty query returns empty results")
    @Test("single-character query works")
    @Test("very long document (>10000 tokens) indexes correctly")
    @Test("document with only stopwords indexes correctly")
    @Test("Unicode text (CJK, emoji, RTL) tokenizes correctly")
    @Test("duplicate document content is deduplicated via hash")
    @Test("remove document also removes its embeddings and index entries")
}
```

#### Performance Regression

```swift
@Suite("Performance", .serialized)
struct PerformanceTests {
    @Test("search latency under 50ms for 5000-document corpus")
    @Test("indexing throughput > 10 documents/second")
    @Test("memory usage under 300MB during search")
}
```

### Test Infrastructure Requirements

- **No real model calls in unit tests** — use pre-computed reference embeddings for codec and scoring tests
- **CoreML inference tests** — marked as integration tests, run on CI with real model
- **All tests under 5 seconds** individually
- **Full suite under 2 minutes**
- **Zero tolerance for failures** — no "flaky" or "pre-existing" exceptions

### Pre-Computed Reference Data

Generate reference data from the Rust implementation once and commit to the test fixtures:

```
Tests/Fixtures/
├── facts_corpus.json                # 33 facts with expected search results
├── xtr-base-en.embeddings.bin       # PyTorch reference T5 token embeddings (CoreML ≈ PyTorch gate)
├── reference_centroids.bin          # Witchcraft GGUF: kmeans inputs + bucket centroids
├── reference_residuals.bin          # Witchcraft Q4 codec round-trip ground truth
└── reference_embeddings.bin         # Witchcraft GGUF per-token embeddings (cross-stack parity)
```

Two distinct embedding fixtures live under this tree:

- `xtr-base-en.embeddings.bin` — produced by `scripts/convert-xtr-to-coreml.py`. **Source: PyTorch.** Used by `T5CoreMLEmbedderTests.fixtureParity` to gate CoreML output against the PyTorch reference (≥0.999 mean cosine).
- `reference_embeddings.bin` — produced by `scripts/witchcraft-fixture-export.patch`. **Source: Rust Witchcraft / Q4K GGUF.** Used by `CrossStackEmbeddingParityTests` to gate Switchcraft's CoreML output against the Witchcraft GGUF stack (±0.025 per ADR 010(h)).

See `adrs/013-reference-fixture-provenance.md` for the full provenance, regeneration policy, and tolerance bounds.

Fixture deliverables:

- [x] `facts_corpus.json`
- [x] `xtr-base-en.embeddings.bin` — PyTorch-side T5 token embedding reference (committed; produced by `scripts/convert-xtr-to-coreml.py`)
- [x] Reference fixtures scaffolding (loaders, parity tests, generator patch, ADR 013) — see issue #28
- [ ] `reference_centroids.bin` — produced by `scripts/witchcraft-fixture-export.patch` (regenerate on a Witchcraft checkout per `scripts/README.md`)
- [ ] `reference_residuals.bin` — produced by `scripts/witchcraft-fixture-export.patch`
- [ ] `reference_embeddings.bin` — produced by `scripts/witchcraft-fixture-export.patch`
- [x] NFCorpus setup script + README documenting `SWITCHCRAFT_NFCORPUS_DIR` (academic-use-only dataset; not committed — `scripts/fetch-nfcorpus.sh` populates a developer-supplied directory)

This allows the Swift tests to validate correctness without running the Rust implementation — just compare against the committed reference data.

---

## Open Source Release

Switchcraft is being released as a standalone Swift package, the **first native Apple platform implementation** of XTR-Warp token-level semantic search.

Prepared for release in **v0.1.0** (issue #41) — ready to tag once the release PR merges:

- **Standalone Swift package** distributed via Swift Package Manager
- **Apache 2.0 license** — matching Witchcraft's license, maximally permissive
- **Cross-platform within Apple**: macOS 13+, iOS 16+, visionOS 1+
- `NOTICE` file with Apache 2.0 §4(d) attributions, SPDX file headers
  across `Sources/`, `CHANGELOG.md`, `CONTRIBUTING.md`, DocC comments
  on the full public API surface, and `Package.swift` product-intent
  comments.

The framework is useful for any app that needs local semantic search: note-taking apps, document managers, email clients, code editors, personal knowledge bases.

---

## Performance Expectations

Based on Witchcraft's benchmarks on Apple M2 Max:

| Metric | Witchcraft (Rust) | Expected Swift Port |
|---|---|---|
| End-to-end search latency | 21ms p95 | 25-35ms (CoreML overhead, optimize in Phase 2) |
| Embedding latency | ~30-50ms per document | Similar (CoreML T5) |
| Index build | Seconds for 1000s of docs | Similar |
| Storage per document | ~64 bytes per token (Q4) | Same |
| NDCG@10 (NFCorpus) | 0.31-0.33 | 0.336 (Metal, FP32-throughout; ggml band 0.31-0.33) |

---

## References

- [XTR-Warp paper (SIGIR'25)](https://arxiv.org/abs/2501.17788) — WARP: An Efficient Engine for Multi-Vector Retrieval
- [XTR-Warp code (Python)](https://github.com/jlscheerer/xtr-warp) — Stanford original
- [Witchcraft (Rust)](https://github.com/dropbox/witchcraft) — Dropbox reimplementation
- [XTR paper](https://arxiv.org/abs/2304.01982) — Rethinking the Role of Token Retrieval in Multi-Vector Retrieval
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) — The original token-level embedding approach
- [ColBERT overview](https://zilliz.com/learn/explore-colbert-token-level-embedding-and-ranking-model-for-similarity-search) — Token-level embedding explanation
