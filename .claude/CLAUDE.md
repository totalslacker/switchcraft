# CLAUDE.md

## Project

Switchcraft is a Swift port of [Witchcraft](https://github.com/dropbox/witchcraft) (Dropbox, Apache 2.0), the Rust reimplementation of XTR-Warp. It brings token-level semantic search with sub-linear retrieval to native Apple platforms (macOS, iOS, iPadOS, visionOS) and is intended to be released as an Apache 2.0 Swift package.

The implementation plan lives at `docs/Plan.md`. Read it before making non-trivial design decisions.

### Design tenets

- **Stay close to the Rust original** during the initial port. Easier correctness verification against upstream tests, easier to follow the algorithm.
- **Storage is pluggable.** SQLite is the reference backend, but the search/index/scoring layer must not depend on SQLite specifics. New backends (DuckDB, RocksDB, LMDB, Postgres+pgvector, in-memory) should be addable behind the storage protocol without touching the engine.
- **Apache 2.0 throughout.** No GPL/copyleft dependencies.
- **Async Swift public API, synchronous core.** Public surface is an actor (`SwitchcraftStore`); the internal implementation matches Witchcraft's synchronous shape.

## Fabrik Workflow

This project uses [Fabrik](https://github.com/tenaciousvc/fabrik) for AI-supervised development. Issues on the GitHub Project board are the unit of work. Board columns define workflow stages:

**Backlog → Specify → Research → Plan → Implement → Review → Validate → Done**

Stage configs live in `.fabrik/stages/` (YAML files). Each stage runs Claude Code in an isolated git worktree with stage-specific skills, models, and tool restrictions.

### Key Concepts

- **Issues are work units.** All work flows through GitHub Issues on the project board.
- **Board columns = stages.** Moving an issue to a column triggers the corresponding stage.
- **Worktrees isolate work.** Each issue gets its own `fabrik/issue-N` branch and worktree.
- **Labels are state.** `fabrik:locked:<user>`, `fabrik:editing`, `stage:<name>:complete`, `model:<name>`.
- **Comments steer work.** Comment on an issue to provide feedback; Fabrik processes it with the stage's comment prompt.
- **`FABRIK_STAGE_COMPLETE`** — signal in Claude output that marks a stage as done.
- **PRs include `Closes #N`** — every PR must reference its issue for auto-close on merge.

### Tracking progress in `docs/Plan.md`

`docs/Plan.md` is the source of truth for what's planned, in progress, and done across the whole port — independent of any single issue. When an Implement stage completes a unit of work, **check off the corresponding box(es) in the same commit**:

- Convert `- [ ]` to `- [x]` for items that are now complete.
- If a checked item turns out to be wrong or incomplete, uncheck it and explain why in the commit message.
- If you're adding work that wasn't in the plan, add a new checkbox item under the appropriate section rather than doing it silently.
- Keep the "Progress at a Glance" section consistent with the detailed checklists below it.

Treat the checkbox list like a precise commit log of feature completeness — a future contributor (or future you) should be able to read `docs/Plan.md` and know exactly what's been built without spelunking through git history. The plan complements the issue tracker: issues track *the current chunk of work*; the plan tracks *what's actually built*.

### Branch & PR Rules

- Always work on a Fabrik-assigned branch (`fabrik/issue-N`), never commit directly to main.
- Commit frequently — after each logical unit of work, not just at the end.
- Push regularly so progress is visible on the draft PR.
- Check `git status` first in any stage (may have uncommitted work from a previous session).
- Rebase onto latest main in Review and Validate stages.
- Every PR must reference its issue with `Closes #N` so the issue auto-closes on merge.

## Testing Requirements

### Tests Are Mandatory

**All new functionality MUST have accompanying tests.** This is non-negotiable.

- New features → write tests that verify the feature works
- Bug fixes → write tests that reproduce and verify the fix
- Refactoring → ensure existing tests still pass, add tests for edge cases

### Regression Tests for Modified Files

When modifying an existing file, you MUST write at least one regression test for each existing behavior you could break in that file:

1. **Storage backend files**: Test that existing reads/writes still return correct results after schema or codec changes.
2. **Index / search files**: Test that existing queries still produce the same ranking and scores within tolerance.
3. **Codec files (Q4 quantize/dequantize, GEMM, entropy coding)**: Property-based round-trip and reference-data comparison.
4. **Any file recently modified by other merges**: Treat as high-risk. Add extra regression coverage.

The rule: if you touch a file, you own proving you didn't break what was already there. New-code-only tests are insufficient.

### Cross-Implementation Validation

Switchcraft's primary correctness gate is **producing identical results to upstream Witchcraft** for the same inputs. The 33-fact corpus and NFCorpus benchmark from Witchcraft are ported verbatim and must pass:

- Same documents retrieved, same ranking order, scores within ±0.01 tolerance.
- NFCorpus NDCG@10 must land in 0.31-0.33 (same range as Witchcraft).

See `docs/Plan.md` for the full testing strategy, including pre-computed reference data committed under `Tests/Fixtures/`.

### Running Tests

This is a Swift Package. Tests run via SPM:

```bash
# Run the full test suite
swift test

# Run a specific suite or test
swift test --filter CodecTests
swift test --filter "Matrix Operations"

# Release-mode tests (closer to production performance)
swift test -c release
```

Performance-sensitive tests should be run in release configuration. CoreML inference tests are marked as integration tests and require model assets present locally.

### Before Committing Code — Hard Rule

**Always run the full test suite before every commit. No exceptions.**

This is non-negotiable, even for small changes. "It's just a one-line fix" is exactly when bugs slip through.

- Run `swift test` and wait for it to finish green before staging the commit.
- Performance regression tests must not regress beyond their thresholds (see `docs/Plan.md` performance section).
- If tests fail, **do not commit**. Fix the tests or fix the code. Do not stash, do not commit "WIP and fix later", do not skip with `--no-verify`.
- Documentation-only commits (markdown/comment-only edits with no code or build changes) may skip the suite, but err on the side of running it.

### Before Merging — Hard Rule

**Always run the full test suite before merging to main.** Even if CI ran on the branch, re-run locally on the merge result so a bad merge can't sneak in.

## Storage Layer

The storage layer is a Swift protocol (`SwitchcraftStorage` or similar) that abstracts:

- Document CRUD
- Chunk storage and lookup by content hash
- LSM generation metadata
- Bucket reads/writes (centroid, indices, residual blobs)
- Full-text search hooks (FTS5 by default; any BM25-capable backend can substitute)
- A filter expression language that can be lowered to each backend's native query form

When making changes that touch storage:

- Modify the protocol first, then update each backend implementation.
- Do not leak SQLite-specific types (e.g. `Statement`, raw SQL strings) into the engine layer.
- New backends must pass the same conformance test suite — same inputs, same retrieval results.

## ML Inference

T5 encoder runs via CoreML for the MVP. Hot paths (Q4 dequant + matmul, centroid similarity, residual scoring) are candidates for Metal compute shaders in Phase 2.

- Tokenizer is a Swift port of the HuggingFace BPE tokenizer JSON; tokenization must produce identical token IDs to the upstream Rust implementation.
- CoreML model output must match Candle reference output within FP precision for known inputs (validated by committed reference fixtures).

## References

- `docs/Plan.md` — full implementation plan
- [Witchcraft (Rust upstream)](https://github.com/dropbox/witchcraft) — reference implementation we are porting
- [XTR-Warp paper (SIGIR'25)](https://arxiv.org/abs/2501.17788)
- [XTR paper](https://arxiv.org/abs/2304.01982)
