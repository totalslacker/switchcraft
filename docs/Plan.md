# Switchcraft: A Swift Port of Witchcraft

> **Status**: Implementation plan — early scaffolding. The project will eventually be released as open source.

Switchcraft is a Swift port of [Witchcraft](https://github.com/dropbox/witchcraft) (Dropbox, Apache 2.0), the Rust reimplementation of [XTR-Warp](https://github.com/jlscheerer/xtr-warp) (Stanford/ETH Zurich, SIGIR'25). It brings token-level semantic search with sub-linear retrieval to native Apple platforms.

No Swift, Go, Java, Kotlin, or TypeScript ports of XTR-Warp / Witchcraft currently exist — Switchcraft will be the first native Apple platform implementation.

## Progress at a Glance

High-level phases. Detailed checklists live in each section below — keep both in sync.

- [ ] **Phase 1: MVP** — port Witchcraft to Swift, SQLite-backed, passing the upstream test corpus
- [ ] **Phase 2: Production optimization** — Metal kernels, caching, concurrency improvements
- [ ] **Open source release** — Apache 2.0 Swift package on SPM

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
- [x] **K-means clustering** (1 week) — Standard algorithm, use Accelerate
- [x] **4-bit residual codec** (1 week) — ~200 lines, bit-level packing; round-trip property tests
- [x] **LSM-tree index structure** (1 week) — Cascading merge logic
- [x] **Search pipeline** (1-2 weeks) — Centroid match → residual decode → MaxSim scoring
- [x] **RRF + FTS5 hybrid** (3-4 days) — Adapt Witchcraft's hybrid fusion
- [x] **Async Swift API** (3-4 days) — `SwitchcraftStore` actor wrapper
- [ ] **Testing + benchmarking** (2-3 weeks) — Correctness + performance validation (see Testing Strategy)

### Phase 2: Production Optimization

- [ ] Metal compute shaders for hot paths (Q4 dequant + matmul, centroid similarity, residual scoring)
- [ ] LRU caching for query embeddings
- [ ] Background indexing pipeline
- [ ] Parallel bucket search
- [ ] Batch document processing

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

- [ ] 33-fact corpus ported verbatim, queries pass with same retrieval / ranking / scores (±0.01)
- [ ] NFCorpus benchmark pipeline ported, NDCG@10 ∈ [0.31, 0.33]
- [ ] Bit-exact embedding validation against Candle reference outputs

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

Run the same queries in both Rust (Witchcraft) and Swift (Switchcraft), assert:

- Same documents retrieved
- Same ranking order
- Scores within tolerance (±0.01 for floating point differences)

This catches any divergence in embedding, centroid matching, residual codec, or scoring.

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

- [ ] Matrix Operations suite (GEMM, matmul_t, batched argmax, packed/fused GEMM)
- [x] Residual Codec suite (round-trip, sizing, range, batch parity)
- [ ] Concurrency suite (concurrent reads during indexing, reader/writer visibility, simultaneous searches)
- [ ] CoreML Inference suite (parity vs Candle, tokenizer parity, sliding window, low-signal filtering)
- [ ] Edge Cases suite (empty/short queries, very long docs, stopwords, Unicode, dedup, removal)
- [ ] Performance Regression suite (search latency, indexing throughput, memory)

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
├── facts_corpus.json           # 33 facts with expected search results
├── reference_embeddings.bin    # T5 token embeddings for known inputs
├── reference_centroids.bin     # K-means centroids for known corpus
├── reference_residuals.bin     # Q4 residuals for codec validation
├── nfcorpus_queries.tsv        # NFCorpus test queries
└── nfcorpus_expected.json      # Expected NDCG@10 scores
```

Fixture deliverables:

- [ ] `facts_corpus.json`
- [x] `reference_embeddings.bin` (committed as `xtr-base-en.embeddings.bin` + index JSON; produced by `scripts/convert-xtr-to-coreml.py`)
- [ ] `reference_centroids.bin`
- [ ] `reference_residuals.bin`
- [ ] `nfcorpus_queries.tsv`
- [ ] `nfcorpus_expected.json`

This allows the Swift tests to validate correctness without running the Rust implementation — just compare against the committed reference data.

---

## Open Source Release

Switchcraft is intended to be released as a standalone Swift package, the **first native Apple platform implementation** of XTR-Warp token-level semantic search.

Release plan:

- **Standalone Swift package** distributed via Swift Package Manager
- **Apache 2.0 license** — matching Witchcraft's license, maximally permissive
- **Cross-platform within Apple**: macOS, iOS, iPadOS, visionOS

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
| NDCG@10 (NFCorpus) | 0.31-0.33 | Same (identical algorithm) |

---

## References

- [XTR-Warp paper (SIGIR'25)](https://arxiv.org/abs/2501.17788) — WARP: An Efficient Engine for Multi-Vector Retrieval
- [XTR-Warp code (Python)](https://github.com/jlscheerer/xtr-warp) — Stanford original
- [Witchcraft (Rust)](https://github.com/dropbox/witchcraft) — Dropbox reimplementation
- [XTR paper](https://arxiv.org/abs/2304.01982) — Rethinking the Role of Token Retrieval in Multi-Vector Retrieval
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) — The original token-level embedding approach
- [ColBERT overview](https://zilliz.com/learn/explore-colbert-token-level-embedding-and-ranking-model-for-similarity-search) — Token-level embedding explanation
