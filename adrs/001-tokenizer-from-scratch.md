# ADR 001 — Tokenizer: from scratch vs. swift-transformers dependency

**Status**: Accepted  
**Date**: 2026-04-28  
**Issue**: #6 (Phase 1: Tokenizer)

## Decision

Implement the full `tokenizer.json` pipeline from scratch in `SwitchcraftCore` rather than adopting `apple/swift-transformers` as a package dependency.

## Context

`google/xtr-base-en` uses a Unigram (SentencePiece) tokenizer expressed in HuggingFace's `tokenizer.json` format. The tokenization pipeline is: `Precompiled` normalizer → `Metaspace` pre-tokenizer → `Unigram` model → `TemplateProcessing` post-processor → `Metaspace` decoder.

`apple/swift-transformers` (Apache 2.0) contains `UnigramTokenizer.swift` and `PrecompiledNormalizer.swift`, which appear relevant. However:

- `PrecompiledNormalizer` contains a stub that does **not** decode the DoubleArray binary embedded in the `precompiled_charsmap` field. It uses a hardcoded partial NMT codepoint table instead.
- Its `UnigramTokenizer` assumes a BPE-style input and does not implement the full Viterbi DP described in the HuggingFace `tokenizers` Rust crate.
- The library does not support the `Sequence` normalizer type used by xtr-base-en.

A `swift-transformers` adoption would still require implementing all the missing pieces ourselves, while adding an external dependency and a foreign module boundary.

## Alternatives considered

1. **Full adoption of swift-transformers** — ruled out. Cannot pass bit-exact normalization tests; its Unigram model does not match the Rust reference algorithm; its BPE-oriented design does not fit the xtr-base-en pipeline.

2. **Partial adoption for utility types** — ruled out. Zero-external-dependency posture is a project tenet (see CLAUDE.md). A partial dependency that we still must extend is strictly worse than writing our own: we take on the maintenance overhead without gaining the correctness we need.

3. **From scratch in `SwitchcraftCore`** — accepted. All pipeline components (DoubleArray decoder, Viterbi DP, template post-processor) are small enough to implement directly, and we can validate each against the Research stage's known (input → token ID) reference table.

## Consequences

- No new entries in `Package.swift` dependencies.
- All tokenizer source files live under `Sources/SwitchcraftCore/Tokenizer/`.
- Future tokenizer improvements (BPE support, offset API) must be implemented in-house; we cannot pull in upstream fixes from swift-transformers.
