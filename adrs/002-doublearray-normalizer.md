# ADR 002 — Precompiled normalizer: decode DoubleArray binary at init time

**Status**: Accepted  
**Date**: 2026-04-28  
**Issue**: #6 (Phase 1: Tokenizer)

## Decision

Decode the `Precompiled` normalizer's `precompiled_charsmap` binary (a 237 KB DoubleArray trie from `huggingface/spm_precompiled`) at `Tokenizer.init` time rather than hard-coding an NMT codepoint rules table.

## Context

The `normalizer` section of `xtr-base-en.tokenizer.json` contains a `Precompiled` step whose `precompiled_charsmap` field is a base64-encoded binary blob (237 KB decoded). This blob encodes the full `nmt_nfkc` normalization map as a DoubleArray trie produced by SentencePiece.

The binary format (from `huggingface/spm_precompiled`):
```
[u32 LE: trie_size_bytes]            (4 bytes)
[u32[trie_size/4]: DoubleArray nodes]
[UTF-8 bytes: null-terminated replacement strings]
```

Each DoubleArray node is a 32-bit word with bit fields:
- `has_leaf  = (node >> 8) & 1`
- `value     = node & 0x7FFFFFFF`   (valid when `has_leaf` is true on the leaf)
- `label     = node & 0x800000FF`   (byte label for traversal)
- `offset    = (node >> 10) << ((node & (1 << 9)) >> 6)`

Lookup: `common_prefix_search` iterates over input bytes; for each byte it XORs the current position with the byte, reads the unit, confirms the label, then XORs the offset to advance. When `has_leaf` is set it records the replacement-string offset.

## Alternatives considered

1. **Hand-coded NMT codepoint table** — ruled out. The NMT rules delete roughly 20 control codepoints and map ~10 whitespace variants to U+0020, plus standard NFKC recomposition. Any hand-written `switch` statement would need to be kept in sync with the binary manually, would diverge silently on any future charsmap update, and would be wrong for codepoints that require multi-codepoint → multi-codepoint mapping that NFKC alone handles via the trie.

2. **Decode binary at init time** — accepted. The binary already encodes all `nmt_nfkc` mappings; decoding it guarantees identity with the upstream Rust crate for all inputs, including edge cases the hand-coded table might miss.

## Consequences

- `DoubleArrayTrie.swift` implements the binary decoder (~100 lines of bit manipulation).
- The decoded trie is stored as `([UInt32], [UInt8])` — node array and string blob — both allocated once at init.
- Correctness is validated by unit tests that apply the decoded trie to known (grapheme → normalized form) pairs from the Research stage reference table before the full pipeline is assembled.
