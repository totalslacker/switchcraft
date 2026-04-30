# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Unreleased

First public release. Phase 1 MVP feature-complete: a Swift port of
[Witchcraft](https://github.com/dropbox/witchcraft) (Apache 2.0) with the
XTR-Warp algorithm running natively on Apple platforms.

### Added

- **Tokenizer** (`SwitchcraftCore/Tokenizer/`): Swift port of the HuggingFace
  `tokenizers` Unigram (SentencePiece) pipeline for `google/xtr-base-en`.
  Loads `tokenizer.json` and produces token IDs that match the upstream
  Rust crate bit-exactly. Includes Precompiled normalizer with double-array
  trie, Metaspace pre-tokenizer, Viterbi-based Unigram model, added-tokens
  pre-segmentation, and TemplateProcessing post-processor.
- **Embedder protocol** (`SwitchcraftCore/Embedding/`): `Embedder` protocol
  with `dims`, `modelIdentifier`, and `encode(_:)` async API.
- **CoreML embedder** (`SwitchcraftCoreML`): `T5CoreMLEmbedder` actor wraps a
  CoreML-converted `google/xtr-base-en` T5 encoder + 768→128 projection
  layer. Sliding-window inference (window 512, stride 256) for long inputs;
  pre-normalisation L2-norm filter (`MIN_NORM = 1.0`) drops low-signal
  positions. Asset built locally via `scripts/convert-xtr-to-coreml.py`
  (see ADR 010, ADR 011).
- **Indexer** (`SwitchcraftCore/Indexer/`): LSM-tree indexer with cascade
  policy matching upstream Witchcraft (`L0_CAPACITY=1024`, `LSM_FANOUT=16`).
  Spherical k-means clustering (`KMeans`) for centroid training. Q4 residual
  codec for vector compression. Index-key compression (`IndicesCodec`)
  bit-exact with Witchcraft's `compress_keys` / `decompress_keys`. See
  ADRs 004, 005.
- **Search engine** (`SwitchcraftCore/Search/`): `SearchEngine.search`
  (`match_centroids` parity, `k=32`, `tPrime=40_000`) and
  `SearchEngine.searchHybrid` (vector + BM25 RRF fusion, `rrfK=60`,
  `perSourceBudget=50`). See ADRs 006, 007, 008.
- **Hybrid fusion**: equal-weight Reciprocal Rank Fusion combining the
  vector and FTS pipelines into a single `[HybridHit]` ranking.
- **Public API** (`Switchcraft`): `SwitchcraftStore` actor as the primary
  consumer surface. Async `add(id:body:)`, `remove(id:)`, `index()`,
  `clear()`, `search(query:topK:filter:)`, `score(query:passages:)`,
  `shutdown()`. See ADR 009.
- **Storage** (`SwitchcraftCore/Storage/`): `SwitchcraftStorage` protocol
  with the full document/chunk/generation/bucket/FTS surface, plus a
  `StorageFilter` expression language. Reference `InMemoryStorage`
  implementation.
- **SQLite backend** (`SwitchcraftSQLite`): `SQLiteStorage` actor with
  FTS5 full-text search; `SwitchcraftStore.sqlite(...)` factory.
- **Conformance test suite** (`SwitchcraftStorageTesting`):
  `StorageConformance.runAll(makeStorage:)` covers documents, chunks,
  generations, buckets, FTS, filters, and round-trip determinism for
  any backend implementation.
- **Cross-stack reference fixtures** (`Tests/Fixtures`): Candle-derived
  reference embeddings for cross-implementation parity validation. See
  ADRs 013, 014.
- **Architecture decision records**: ADRs 001–014 covering tokenizer
  architecture, double-array normalizer, typing requirements, LSM cascade
  policy, bucket indices encoding, search constants, search vs. index
  responsibility, hybrid fusion, public API shape, embedder + asset
  distribution, sliding-window long-input strategy, performance regression
  thresholds, reference-fixture provenance, and precision asymmetry.
- **Open-source release plumbing**: `NOTICE` file (Apache 2.0 §4(d)),
  SPDX file headers across `Sources/`, `CHANGELOG.md`, `CONTRIBUTING.md`,
  and DocC comments on the full public API surface.

### Notes

- Pre-1.0. The public API may change between minor versions until v1.0.
- CI runs `swift test` and `swift test -c release` on macOS only;
  iOS/visionOS are listed in `Package.swift` but not exercised in CI yet.
- `MockEmbedder` is intentionally test-target-only (ADR 009(j)). Adopters
  who need a deterministic embedder for their own tests should vendor it
  or write their own.

[Unreleased]: https://github.com/totalslacker/switchcraft/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/totalslacker/switchcraft/releases/tag/v0.1.0
