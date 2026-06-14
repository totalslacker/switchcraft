# Architecture Decision Records

ADRs document significant design decisions for Switchcraft. Each file follows the naming convention `NNN-title.md` (3-digit, zero-padded).

**Convention**: Future PRs that touch parity-relevant arithmetic (cross-implementation bit-exact matching with upstream Rust) must cite ADR 003 in the commit body or PR description.

## Index

- [001 — Tokenizer from scratch](001-tokenizer-from-scratch.md)
- [002 — DoubleArray normalizer](002-doublearray-normalizer.md)
- [003 — Typing requirements](003-typing-requirements.md)
- [004 — LSM cascade policy](004-lsm-cascade-policy.md)
- [005 — Bucket indices encoding](005-bucket-indices-encoding.md)
- [006 — Search constants](006-search-constants.md)
- [007 — Search vs index responsibility](007-search-vs-index-responsibility.md)
- [008 — Hybrid fusion](008-hybrid-fusion.md)
- [009 — Public API shape](009-public-api-shape.md)
- [010 — Embedder model and asset distribution](010-embedder-model-and-asset-distribution.md)
- [011 — Sliding window long-input strategy](011-sliding-window-long-input-strategy.md)
- [012 — Performance regression thresholds](012-performance-regression-thresholds.md)
- [013 — Reference fixture provenance](013-reference-fixture-provenance.md)
- [014 — Precision asymmetry cross-stack](014-precision-asymmetry-cross-stack.md)
- [015 — Metal context and dispatch](015-metal-context-and-dispatch.md)
- [016 — GGUF asset distribution](016-gguf-asset-distribution.md)
- [017 — Per-op precision routing in T5MetalEmbedder](017-per-op-precision-routing.md)
- [018 — Separate ObjC clang target for the CoreML exception bridge](018-objc-clang-target-for-exception-bridge.md)
- [019 — SQLiteStorage writer + reader actor split](019-sqlite-writer-reader-split.md)
- [020 — Search timeout and cancellation design](020-search-timeout-and-cancellation.md)
- [021 — ANE IOSurface pool exhaustion mitigation](021-ane-iosurface-pool-exhaustion-mitigation.md)
- [022 — Embedder overflow guard (`maxInputTokens` and `EmbedderOverflowPolicy`)](022-embedder-overflow-guard.md)
- [023 — Delete WAL concurrency microbench (`performanceAssertionTest`)](023-wal-microbench-deletion.md)
- [024 — `rehydrationConflict` auto-recovery](024-rehydration-conflict-autorecovery.md)
- [025 — Title prepend for proper-noun query alignment](025-title-prepend-for-query-alignment.md)
- [026 — FTS title field, V0→V1 schema migration, and BM25 title-weight config](026-fts-title-field-and-schema-migration.md)
- [027 — Tiled-batch KMeans assignment to bound peak memory](027-tiled-batch-kmeans-assignment.md)
- [028 — Query-token surface-form length filter](028-query-token-surface-form-filter.md)
- [029 — Orphan chunk detection and recovery](029-orphan-chunk-detection-recovery.md)
- [030 — Mid-operation compaction ledger divergence](030-mid-operation-compaction-ledger-divergence.md)
- [031 — `Embedder.resetState()`: explicit consumer-controlled ANE IOSurface pool flush](031-embedder-reset-state-iosurface-pool-flush.md)
