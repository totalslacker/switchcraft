# Contributing to Switchcraft

Thanks for your interest in Switchcraft. This document covers the basics of
working on the package and the release process.

## Maintenance expectations

Switchcraft is maintained on a best-effort basis. PRs are welcome; review
turnaround is not guaranteed. Bug fixes with regression tests, documentation
improvements, and small focused features are the easiest to land.

## Development prerequisites

- **Swift 6.0 toolchain** (Xcode 16+ on macOS).
- macOS 13+ for development; CI also runs against macOS.
- The CoreML model asset is **not** required to develop or run the
  default test suite — asset-gated tests skip cleanly when
  `SWITCHCRAFT_XTR_MLPACKAGE` is unset.

## Running the test suite

```bash
swift test
swift test -c release          # release-mode regressions
```

**Every PR must keep the test suite green** in both configurations. If
you are touching performance-sensitive code, also confirm the release-mode
suite stays within the thresholds documented in ADR 012.

When modifying an existing file, add at least one regression test for each
existing behavior the change could break — see the rules under "Regression
Tests for Modified Files" in the project's `CLAUDE.md`.

## Style and architecture

For non-trivial changes, follow the existing ADR-driven approach:

- Read [`docs/Plan.md`](docs/Plan.md) for the implementation roadmap.
- Browse [`adrs/`](adrs/) for prior architectural decisions. If your
  change deviates from one, propose a new ADR rather than silently
  diverging.
- Stay close to upstream [Witchcraft](https://github.com/dropbox/witchcraft)
  while the port is incomplete; it makes correctness verification easier.
- New `Sources/` files must begin with `// SPDX-License-Identifier: Apache-2.0`
  as the first line.

## Validating documentation

The public API is documented with DocC `///` comments. To validate them
locally without committing a documentation plugin to `Package.swift`:

```bash
xcodebuild docbuild \
    -scheme Switchcraft \
    -derivedDataPath /tmp/switchcraft-docc \
    -destination 'generic/platform=macOS'
```

The build must succeed without DocC warnings.

## Release process

Switchcraft uses semantic versioning and a manual tag-and-go release flow:

1. Land all changes for the release on `main` via PR.
2. Update `CHANGELOG.md`: rename the in-flight `## [Unreleased]` /
   `## [x.y.z] - Unreleased` heading to a dated `## [x.y.z] - YYYY-MM-DD`
   entry; open a fresh `## [Unreleased]` section above it.
3. Maintainer pushes the tag against the merge commit:
   ```bash
   git tag v<x.y.z>
   git push origin v<x.y.z>
   ```
4. The existing `ci.yml` workflow validates `swift test` and
   `swift test -c release` on the merge commit; there is no
   tag-triggered `release.yml` for v0.1.0 (deferred).
5. Create a GitHub release pointing at the new tag. Use the
   `CHANGELOG.md` entry as the release notes.

## Reporting issues

Use the [GitHub issue tracker](https://github.com/totalslacker/switchcraft/issues).
Include:

- The Swift toolchain version (`swift --version`).
- A minimal reproduction (a small snippet against the in-memory backend
  is ideal).
- Whether the issue is in the always-on suite or the asset-gated suite.

## License

By contributing, you agree that your contributions will be licensed under
the project's [Apache License 2.0](LICENSE).
