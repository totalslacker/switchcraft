# ADR 018 — Separate ObjC clang target for the CoreML exception bridge

**Status**: Accepted
**Date**: 2026-05-04
**Issue**: #78 (T5CoreMLEmbedder crash-safety: ObjC exception bridge)

This ADR records the decision to introduce the Objective-C `@try/@catch`
bridge as a standalone **clang-only target** (`SwitchcraftCoreMLObjC`)
rather than mixing ObjC source files directly into the existing
`SwitchcraftCoreML` Swift target.

---

## (a) Context

`T5CoreMLEmbedder.encode(_:)` calls `MLModel.prediction(from:)`, which can
raise **Objective-C exceptions** from internal failure sites such as
`MLE5BindEmptyMemoryObjectToPort`. Swift's `do-catch` does not intercept ObjC
exceptions — they propagate past Swift's error machinery and terminate the host
process via `std::terminate`. The fix requires an ObjC `@try/@catch` wrapper
that converts any caught `NSException` into a returnable value the Swift caller
can translate into a typed error.

Before this ADR, the Switchcraft repository contained zero Objective-C source
files. Adding ObjC introduces a new build consideration: where do the `.h` and
`.m` files live?

---

## (b) Options considered

### Option A — Mixed Swift + ObjC in `SwitchcraftCoreML`

Place `MLExceptionCatcher.h` and `MLExceptionCatcher.m` directly under
`Sources/SwitchcraftCoreML/` (e.g., in a `MLExceptionCatcher/` subdirectory)
and set `publicHeadersPath: "MLExceptionCatcher"` on the `SwitchcraftCoreML`
target.

**Advantages**: file layout matches the spec exactly; fewer Package.swift
changes; ObjC bridge lives alongside the Swift code that uses it.

**Disadvantages**: SwiftPM 6's mixed-language targets (Swift + ObjC in the
same target) have historically produced subtle module-map generation failures
under strict concurrency mode. When ObjC and Swift files share a target with
Swift 6's concurrency checks enabled, the generated module map can emit
ambiguous or missing header paths, producing cryptic linker errors rather than
actionable diagnostics. The combination of `publicHeadersPath` inside a mixed
target and `swift-tools-version: 6.0` is underspecified in SwiftPM's
documentation and has been a source of build instability in practice.

### Option B — Separate clang-only target `SwitchcraftCoreMLObjC` ← **chosen**

Create a dedicated clang-only target with `publicHeadersPath: "include"` under
`Sources/SwitchcraftCoreMLObjC/`. `SwitchcraftCoreML` lists
`SwitchcraftCoreMLObjC` as a dependency and imports it with
`import SwitchcraftCoreMLObjC`.

**Advantages**:
- Explicit module boundary: SwiftPM generates a clean, single-language module
  map for each target. No ambiguity.
- `publicHeadersPath: "include"` is the canonical pattern for clang targets in
  SwiftPM and is well-tested.
- Strict concurrency checking in `SwitchcraftCoreML` does not apply to the ObjC
  target; the two targets are compiled separately.
- The ObjC bridge is reusable: if `T5MetalEmbedder` or other future code needs
  `@try/@catch` coverage, it can depend on `SwitchcraftCoreMLObjC` without
  modifying `SwitchcraftCoreML`.
- `SwitchcraftCoreMLObjC` is not listed in `products`, so it is invisible to
  downstream consumers of the Switchcraft package.

**Disadvantages**: ObjC files live in a different directory than the Swift
code that uses them, which deviates slightly from the spec's stated file
layout. This is an internal implementation detail with no user-visible impact.

---

## (c) Decision

**Option B is adopted.** The build reliability advantage of a separate clang
target outweighs the minor deviation from the spec's file-path description,
particularly given that the spec explicitly scoped "Package.swift changes to
enable mixed Swift/ObjC" as in-scope.

File layout:

```
Sources/
  SwitchcraftCoreMLObjC/
    include/
      MLExceptionCatcher.h     ← publicHeadersPath
    MLExceptionCatcher.m
  SwitchcraftCoreML/
    MLExceptionCatcher.swift   ← Swift facade; imports SwitchcraftCoreMLObjC
    ...
```

Package.swift adds:

```swift
.target(
    name: "SwitchcraftCoreMLObjC",
    path: "Sources/SwitchcraftCoreMLObjC",
    publicHeadersPath: "include"
),
```

and `"SwitchcraftCoreMLObjC"` in `SwitchcraftCoreML`'s `dependencies`.

---

## (d) Consequences

- Any future ObjC code that needs to interop with `SwitchcraftCoreML` should be
  added to `SwitchcraftCoreMLObjC` (or a new analogous target) rather than
  mixing ObjC into a Swift target.
- If `T5MetalEmbedder` needs `@try/@catch` coverage (a deferred audit per
  issue #78's out-of-scope section), it can depend directly on
  `SwitchcraftCoreMLObjC` without touching `SwitchcraftCoreML`.
- The `catchingNSException` Swift function and `CoreMLNativeError` type are
  `internal` to `SwitchcraftCoreML`. The promotion path to `@_spi` or `public`
  is deferred until a concrete external-consumer need is demonstrated (per
  issue #78 discussion).
