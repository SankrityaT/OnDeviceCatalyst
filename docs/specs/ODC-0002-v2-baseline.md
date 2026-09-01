---
id: ODC-0002
title: Reproduce v2 build and dependency state
type: baseline
status: REVISION
milestone: P0
owner: SankrityaT
dependencies: ODC-0000
founder_approved: pending
last_updated: 2026-09-01
evidence_fresh_until: 2026-09-15
unresolved_questions: none
---

# ODC-0002: Reproduce v2 build and dependency state

## Summary

Capture a privacy-safe, mechanically reproducible baseline of the v2.0.4 tree at
revision `59da80b` before any package, dependency, source, or runtime repair.

This spec is a rewrite. Review pass two rejected the previous version
([review](../reviews/ODC-0002-review-pass-2.md)) on six blocking findings: an
unqualified "simulator build passes" claim that hid a 7,936-byte stub, a
self-contradictory `Package.resolved` deliverable, total omission of the
divergent demo-app source fork, unpinned confounders, unfalsifiable acceptance
criteria, and missing required sections. Each is resolved below and cross
referenced by finding number.

The baseline is an evidence contract, not a repair. ODC-0003, ODC-0004 and
ODC-0005 will cite it, so every recorded result carries the command that
produced it and every claim that cannot be decided by an exit code has been
deleted rather than softened.

Structure follows [`docs/templates/baseline-spec.md`](../templates/baseline-spec.md),
created by this ticket because `type: baseline` previously had no template
(blocking finding 6).

## Goals

- Pin the full measurement environment, including build numbers and cache state,
  so two honest runs cannot disagree (blocking finding 4).
- Record the exact repository revision, package manifest metadata, dependency
  pins, and every binary artifact slice with its byte size and exported-symbol
  count (blocking finding 1).
- Record each build path with a three-valued result that separates "the compiler
  accepted it" from "this configuration can perform inference" (blocking
  finding 1).
- Resolve `Package.resolved` to one decided outcome and evaluate the lockfile
  format change as a consumer-tooling consequence (blocking finding 2).
- Characterize the divergent duplicate runtime under `OnDeviceCatalyst/` and
  quantify its drift against `Sources/OnDeviceCatalyst/` (blocking finding 3).
- Record all eight confirmed v2 defects as characterized findings with
  `path:line` evidence, each mapped to a follow-up ticket.
- Produce a human report and a schema-validated machine manifest.
- Deliver the capture and check as scripts so re-running is one command and
  drift is a diff.

## Non-goals

- Fix any reproduced problem. All eight defects in
  `## Current state and evidence` are characterized here and repaired elsewhere.
- Change `Package.swift`, any file under `Sources/`, `Tests/`,
  `OnDeviceCatalyst/`, or `OnDeviceCatalyst.xcodeproj/`.
- Change the published binary artifact or its checksum.
- Download model weights or run inference.
- Publish device serials, UDIDs, private paths, credentials, or tokens.
- Decide v3 dependencies or package architecture.
- Decide which of the two runtime copies survives. This ticket names both and
  quantifies the gap; the choice is a separate decision ticket (see
  `## Ticket allocation`).

`Package.resolved` is deliberately absent from the non-goals. It is the one
tracked file this ticket rewrites, argued in `## Migration and compatibility
impact` (blocking finding 2).

## Pinned environment and clean state

Resolves blocking finding 4. Nothing in `## Required procedure` may run until
this block is captured verbatim into both deliverables.

### Toolchain identity

Every field is an exact build number, not a marketing version. The values below
are the verified identity at spec time; a run that does not match them records
its own values and flags the delta.

| Field | Command | Value at spec time |
| --- | --- | --- |
| Xcode version and build | `xcodebuild -version` | 26.6 (17F113) |
| Swift compiler | `swift --version` | 6.3.3 (swiftlang-6.3.3.1.3 clang-2100.1.1.101) |
| swift-driver | `swift --version` | 1.148.6 |
| Host target triple | `swift --version` | `arm64-apple-macosx26.0` |
| Active developer dir | `xcode-select -p` | recorded, path-redacted |
| macOS product and build | `sw_vers` | 26.4 (25E246) |
| Chip | `sysctl -n machdep.cpu.brand_string` | Apple M5 Pro |
| Memory | `sysctl -n hw.memsize` | 25769803776 (24 GiB) |
| Host model identifier | `sysctl -n hw.model` | recorded, no serial |
| macOS SDK build | `xcodebuild -showsdks` | macosx26.5 |
| iOS SDK build | `xcodebuild -showsdks` | iphoneos26.5 |
| iOS Simulator SDK build | `xcodebuild -showsdks` | iphonesimulator26.5 |

### Simulator and device identity

The previous spec recorded a "connected test-device class" that no step
consumed. That goal is deleted. In its place:

- No simulator runtime is instantiated. `swift build --triple
  arm64-apple-ios17.0-simulator` requires only the simulator SDK, and the report
  must state that explicitly so a reader does not infer a device run happened.
- The Xcode project step names its destination as
  `-destination 'generic/platform=iOS Simulator'`, which likewise instantiates
  no runtime. Record `xcodebuild -showdestinations` output for the scheme,
  redacted of device identifiers.
- No physical device is attached, built for, installed on, or recorded.

### Cache and clean-state policy

`swift package describe` behaves differently against a warm `.build/` than a
cold one: with the repository's existing `.build/` present it completes without
rewriting `Package.resolved`, and with no `.build/` present resolution rewrites
the lockfile. That single confounder is why the previous procedure's step 5 was
not reproducible. The policy is therefore:

1. Every measurement runs in a scratch tree outside the repository, created with
   `mktemp -d`. No scratch path is created inside the working tree, so no
   `.gitignore` entry is required and the tracked ignore file is unchanged
   (blocking finding 4c). The report records the scratch path as `$SCRATCH`,
   never the literal expansion.
2. The scratch tree is populated by `git archive 59da80b | tar -x -C "$SCRATCH"`
   so it contains tracked content only, at the pinned revision, with no
   untracked or ignored local artifacts.
3. Two SwiftPM cache states are measured, and both are recorded:
   - **cold**: `--cache-path "$SCRATCH/spm-cache" --scratch-path
     "$SCRATCH/.build"` against freshly created directories. This is the
     normative result. A cold run is mandatory.
   - **warm**: the same commands re-run in place, to demonstrate whether any
     recorded result is cache-dependent.
   A result that differs between cold and warm is recorded as
   `cache_dependent: true` and is not reported as a single value.
4. Before and after the procedure, capture into the report:
   `git status --porcelain --ignored` and `git clean -ndx`, so untracked and
   ignored artifacts in the operator's tree are visible to the reader. This is
   what makes finding 4(b) below legible rather than mysterious.
5. `.build/` and `DerivedData/` inside the working tree are never consumed by
   any measurement. The procedure does not delete the operator's copies; it
   ignores them by construction, because deleting them is a side effect and the
   scratch tree makes it unnecessary.

### Note on `.context/`

The 2.3 GB `.context/` tree is ignored only through a machine-local
`.git/info/exclude` entry, not through the tracked `.gitignore`. It is
consequently invisible to a fresh clone's ignore rules. No step of this
procedure reads or writes it, and the `git archive` population in point 2 above
excludes it by construction. Recorded here so the discrepancy is documented
rather than latent; correcting the tracked ignore file is not in scope.

## Current state and evidence

Resolves blocking findings 1, 3, and 6, and folds in majors 8, 10, 11, and 14.

Everything in this section is already verified at `59da80b` by read-only
inspection and by builds in scratch copies. It is stated here, before the
procedure, so that the procedure confirms known facts rather than being trusted
to discover them. Each item names the command that reproduces it.

### E1. The XCFramework has two slices, and the simulator slice is a stub

`swift package describe --type json` resolves the binary target `llama` from the
v2.0.4 release asset whose checksum matches the `xcframeworkChecksum` literal at
`Package.swift:13`. Its `Info.plist` lists exactly two `AvailableLibraries`
entries and no macOS entry:

| Slice | Platform | Variant | Arch | Bytes | `nm -gU` lines | Defined symbols |
| --- | --- | --- | --- | --- | --- | --- |
| `ios-arm64` | ios | none | arm64 | 17,151,376 | 16,870 | 16,420 |
| `ios-arm64-simulator` | ios | simulator | arm64 | **7,936** | **55** | **51** |

The simulator archive contains two objects, `llama_stub.o` and
`llama_sim_stubs.o`. It is 1/2161 the size of the device slice. It is a stub,
not llama.cpp.

Reproduce with, per slice:

```
plutil -p "$XCF/Info.plist"
stat -f%z "$XCF/$SLICE/libllama_combined.a"
nm -gU "$XCF/$SLICE/libllama_combined.a" | wc -l
nm -gU "$XCF/$SLICE/libllama_combined.a" | grep -c '^[0-9a-f]\{8,\} [A-Za-z] '
ar -t "$XCF/$SLICE/libllama_combined.a"
```

Two symbol counts are recorded because they are not the same measurement.
`nm -gU | wc -l` includes per-object header and blank lines and yields
16,870 / 55, the figures quoted in review pass two.
`grep -c '^[0-9a-f]\{8,\} [A-Za-z] '` counts defined symbols only and yields
16,420 / 51. Both are recorded, with the command beside each, so a future reader
cannot mistake one methodology for the other.

### E2. `Package.swift:36-37` contains a false justification for the stub

The manifest comment reads:

```
// Includes an arm64-simulator stub slice so consumers can build for the iOS
// Simulator (all llama usage is guarded #if !targetEnvironment(simulator)).
```

Both halves of the parenthetical are false:

- `grep -rn '#if !targetEnvironment' Sources/` returns **0** matches.
- The only four `#if targetEnvironment(simulator)` occurrences in the package
  are `Core Foundation/DeviceOptimizer.swift:20`,
  `Core Foundation/SafetyManager.swift:76`,
  `Core Foundation/ModelArchitecture.swift:373`, and
  `Core Foundation/SimulatorSupport.swift:27`. None is negated and none guards a
  llama call.
- `Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:12` and
  `Sources/OnDeviceCatalyst/Backend/LlamaCppBackend.swift:9` both
  `import llama` unconditionally, and `LlamaBridge` calls the C API
  unconditionally at `:56`, `:140`, `:158` and elsewhere.

This is recorded as a documented finding in its own right: a tracked build input
carries a comment that a reader would reasonably rely on and that is
contradicted by the source. Mapped to a follow-up ticket.

### E3. The simulator build result is "links, cannot infer"

`swift build --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" --triple
arm64-apple-ios17.0-simulator -c debug` exits 0. That is a true statement about
the compiler and a false statement about the product. The normative recorded
value for this cell is the literal string:

```
links, cannot infer
```

never `pass`, never `green`, never `succeeds` unqualified. The report must carry
the qualification in the same table cell as the result, not in a footnote.

Precision about what "links" means here, since a reviewer will press on it:
SwiftPM building a library target does **not** perform a final link of
`libllama_combined.a`. It compiles Swift, resolves the `llama` module through the
XCFramework's headers and module map, and emits a module and object files. So:

- module resolution and compile-time symbol availability against the simulator
  slice's header set: **covered by this baseline, and they succeed**;
- link-time symbol resolution against either slice's archive: **not covered by
  this baseline**, because no SwiftPM invocation in this procedure links it;
- runtime inference on the simulator slice: **not covered, and known impossible**
  by E1 plus E2, because llama is called unconditionally and the simulator
  archive contains stubs.

Link-time and runtime coverage is deferred to the ticket named in
`## Ticket allocation`.

To ensure the matrix contains at least one cell compiled against the real header
set, the procedure adds a device-triple compile:
`swift build --sdk "$(xcrun --sdk iphoneos --show-sdk-path)" --triple
arm64-apple-ios17.0 -c debug`, verified to exit 0 at `59da80b`.

### E4. macOS build fails, and `swift test` is blocked by the same failure

Root cause, stated up front rather than left to be "captured":
`Package.swift:21` declares `.macOS(.v14)`, but the XCFramework exposes only
`ios-arm64` and `ios-arm64-simulator` (E1). There is no macOS slice, so
`import llama` cannot resolve on any macOS triple.

```
$ swift build -c debug
Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:12:8: error: no such module 'llama'
error: emit-module command failed with exit code 1
exit=1
```

Under whole-module optimization the same root cause is reported once per
compiled file (72 raw occurrences, 1 distinct message). `swift test` fails
identically, at the same `path:line`, because the test target depends on the
library target that never compiles.

Therefore `swift build` and `swift test` on macOS are **one** data point, not
two. The report records `swift test` as `blocked-by-build`, with zero test
signal, and is forbidden from presenting a test result. This closes the previous
step 9, which invited a report implying tests had been exercised.

Taxonomy: this failure is simultaneously a package claim and a binary
limitation. It is recorded in the dedicated bucket
`package-declares-platform-binary-does-not-support` (see
`## Failure behavior`), not forced into either single-sided bucket.

### E5. Eight unhandled files, and the Metal Engine is dead code in package form

The diagnostic at `59da80b`, emitted by both the macOS attempt and the
successful simulator build, is:

```
warning: found 8 file(s) which are unhandled; explicitly declare them as
resources or exclude from the target
```

**Eight**, not seven. The seven `.metal` files under
`Sources/OnDeviceCatalyst/Metal Engine/Shaders/` (`activations.metal`,
`attention.metal`, `embedding.metal`, `matmul.metal`, `matvec.metal`,
`rms_norm.metal`, `rope.metal`) **plus** `Sources/OnDeviceCatalyst/Assets.xcassets`.
An asset catalogue shipping inside a library target is itself a recorded
finding. `swift package describe --type json` confirms the target declares no
`resources` entry at all.

This is not a cosmetic packaging warning. It is a functional defect:
`Sources/OnDeviceCatalyst/Metal Engine/Compute/MetalComputeEngine.swift:89`
calls `device.makeDefaultLibrary()`, which loads a precompiled `default.metallib`
that SwiftPM never produces because the `.metal` sources are unhandled. The
`guard` therefore fails and throws for any consumer selecting
`InstanceSettings.backendType == .metal`. The consequence is that the entire
Metal Engine subtree (`MetalBackend`, `TransformerGraph`, `KVCache`,
`ModelWeights`, the GGUF parser and tokenizer) is **unreachable when the package
is consumed as a package**. The report must state this in its results, with the
`makeDefaultLibrary()` citation, and must not file it under packaging warnings.

The other warning buckets are accurate. ODC-owned warning counts from the
simulator build: deprecated llama API, 9 sites, all in
`API Bridge/LlamaBridge.swift` (`:56`, `:140`, `:158`, `:194`, `:200`, `:205`,
`:210`, `:216` twice); unused code, 6 sites (`ModelProfile.swift:78`,
`ModelWeights.swift:87`, `ConfigurationPresets.swift:110`,
`MLXInstance.swift:220`, `GGUFTokenizer.swift:191`,
`Service Layer/Catalyst.swift:291`); concurrency and lifecycle, 2 sites, both in
`Service Layer/Catalyst.swift` (`:421` unused `sync(flags:execute:)` result,
`:496` no `async` operations within an `await` expression).

### E6. `swift-tools-version: 5.12` is not a shipped Swift release

`Package.swift:1` declares `// swift-tools-version: 5.12` and
`swift package tools-version` reports `5.12.0`. No Swift 5.12 toolchain has ever
shipped; the sequence is 5.9, 5.10, 6.0. SwiftPM accepts the declaration only
because the installed tools version compares greater. The practical effect is
that the package advertises 5.x compatibility while in fact requiring a
toolchain whose numeric tools version exceeds 5.12, that is, Swift 6.x.

The manifest therefore records both `package.tools_version_declared` and
`package.tools_version_reported`, plus the required finding line: the declared
tools version does not correspond to a released Swift release, and the minimum
usable toolchain is the measured one. Mapped to a follow-up ticket. This also
carries the load in `## Migration and compatibility impact` below.

### E7. The Xcode project is a divergent fork of the runtime

Resolves blocking finding 3, the largest single fact about the tree and one the
previous procedure could not have discovered.

`OnDeviceCatalyst.xcodeproj/project.pbxproj` contains **zero**
`XCRemoteSwiftPackageReference` entries. The application target does not consume
the package at all. It compiles its own copy of the runtime from
`OnDeviceCatalyst/`, and that copy has drifted.

File census at `59da80b`:

- `Sources/OnDeviceCatalyst/`: 35 Swift files.
- `OnDeviceCatalyst/`: 25 Swift files.
- Same-named in both: **22**.
- Package-only: 13 (the entire `Metal Engine/{Compute,GGUF,Model}` subtree,
  `Backend/{InferenceBackend,LlamaCppBackend,MLXInstance}.swift`,
  `Service Layer/ModelDownloader.swift`, `Tools/ToolSupport.swift`). Everything
  built since the backend-abstraction refactor is absent from the app, not
  merely stale.
- App-only: 3 (`ContentView.swift`, `OnDeviceCatalystApp.swift`,
  `Qwen3TestView.swift`), which are legitimate app-shell files, not drift.

Of the 22 shared files, **12 differ and 10 are identical**. The normative drift
command, fixed here so the number is not methodology-dependent, is:

```
diff "Sources/OnDeviceCatalyst/$f" "OnDeviceCatalyst/$f" | grep -c '^[<>]'
```

| Shared file | Changed lines |
| --- | --- |
| `Core Engine/LlamaInstance.swift` | **591** |
| `Service Layer/Catalyst.swift` | **218** |
| `API Bridge/LlamaBridge.swift` | 104 |
| `Core Foundation/InstanceSettings.swift` | 47 |
| `Chat System/StreamResponse.swift` | 37 |
| `Core Engine/SamplingEngine.swift` | 35 |
| `Core Engine/PromptFormatting.swift` | 18 |
| `Core Foundation/DeviceOptimizer.swift` | 18 |
| `Core Foundation/ModelProfile.swift` | 16 |
| `Core Foundation/ModelArchitecture.swift` | 11 |
| `Chat System/StopSequenceHandler.swift` | 4 |
| `Core Foundation/PredictionConfig.swift` | 2 |
| 10 remaining shared files | 0 |

Recorded discrepancy, so that a later reader does not treat it as an error:
review pass two reported "13 of 21 shared files differ" with `LlamaInstance.swift`
at 587, and the independent verification reported "12 of 22" with 478. All three
figures describe the same tree under three different counts. `12 of 22` and the
table above are the values produced by the normative command; 478 / 190 / 99 are
the values produced by counting unified-diff body lines
(`diff -u ... | grep -c '^[-+][^-+]'`); the pass-two totals differ by an
off-by-one in the shared-file census and a slightly different line count for the
largest file. The procedure emits the table generated, never hand-maintained, so
this class of disagreement cannot recur.

The divergence is architectural, not cosmetic. The package copy holds
`internal var backend: InferenceBackend?`
(`Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:25-27`); the app copy
still holds raw `cModel` / `cContext` / `cBatch` `OpaquePointer` state and its
own `import llama`
(`OnDeviceCatalyst/Core Engine/LlamaInstance.swift:26-29`). One consequence is
already observable: the app copy does not carry defect D3 below, because its
`publishProgress` uses two separate branches. Defect D1 is reproduced verbatim
in the app copy.

This matters far beyond ODC-0002. ODC-0004 cannot begin until the project has
decided *which* v2 is being characterized, and ODC-0300 cannot begin until it is
decided whether the app fork is deleted or reconciled. The baseline names both
copies and hands the decision to a ticket; it does not make it.

### E8. Compiled tests versus orphaned files under `Tests/`

`Package.swift:58-62` declares the test target with `path:
"Tests/OnDeviceCatalystTests"`. The only compiled test source is
`Tests/OnDeviceCatalystTests/OnDeviceCatalystTests.swift`, 4 XCTest methods
(`testModelProfileCreation`, `testInstanceSettingsValidation`,
`testPredictionConfigPresets`, `testCatalystServiceInitialization`).

Three files sit directly under `Tests/` outside the declared target path and are
compiled by no target: `Tests/EmbeddingTest.swift`,
`Tests/test_embedding.swift`, `Tests/BERTEmbeddingTest.swift`. `swift package
describe --type json` confirms they appear in no target's `sources`.

None of the 4 compiled tests exercises any of the defects below, so existing
coverage of every recorded defect is zero. ODC-0004 builds directly on this
inventory, so the manifest records `compiled_test_sources` and
`orphaned_test_files` as separate arrays.

### E9. The eight confirmed v2 defects

Recorded as characterized findings. **This ticket fixes none of them.** Each is
carried into the report with its `path:line` citation, its reproduction, and its
mapped follow-up ticket. The novel-implementation gate is closed; this spec
characterizes.

| ID | Defect | Evidence |
| --- | --- | --- |
| D1 | `releaseInstance` caches a ready instance and asynchronously shuts it down, with no happens-before between the cache insert and `cleanup()`; the only cache reader does not re-check `isReady` | `Service Layer/Catalyst.swift:495-522` (insert `:507`, `Task { await instance.shutdown() }` `:510-512`), `Service Layer/CacheSettings.swift:111-138` (separate queue), `Service Layer/Catalyst.swift:99-108` (reader), `Core Engine/LlamaInstance.swift:18` (plain class), `:65` (`isReady`), `:232-248` (`cleanup` nils `backend`) |
| D2 | `performGeneration` appends a second, always-`.natural` completion chunk after `generateTokens` has already emitted one for its actual termination reason, so every generation emits two completions and consumers that break on the first lose the metadata and, for four of five paths, get the wrong reason | `Core Engine/LlamaInstance.swift` `generateTokens` from `:443` (emit points `:466`, `:487`, `~:500-506`, `:517`, `:535`), `performGeneration` from `:283` (`:365`, `:384`, `:386`); in-repo consumer `Service Layer/Catalyst.swift:~468` |
| D3 | `publishProgress` gates on `if case .ready = progress, case .failed = progress`, a compound AND over one value, which is unsatisfiable; the continuation is never finished on success, so a consumer awaiting stream termination hangs | `Core Engine/LlamaInstance.swift:580-586`, success path `:123` |
| D4 | `.macOS(.v14)` declared with no macOS slice; `swift build` and `swift test` both fail with `no such module 'llama'` | E4 above; `Package.swift:21`, `API Bridge/LlamaBridge.swift:12` |
| D5 | 8 unhandled files; no `.metallib` is produced; `makeDefaultLibrary()` makes the whole Metal Engine unreachable in package form | E5 above; `Metal Engine/Compute/MetalComputeEngine.swift:89` |
| D6 | `Package.resolved` at HEAD pins `mlx-swift-lm` to `branch: main`, which cannot satisfy the manifest's `exact: "2.29.3"` | E10 below; `Package.swift:30` |
| D7 | `OnDeviceCatalyst/` is a divergent fork of the runtime, 12 of 22 shared files drifted, architecturally divergent | E7 above |
| D8 | `handleInitializationError` calls `cleanup()`, which finishes and nils `loadingContinuation`, **before** `attemptFallbackInitialization` runs; every `publishProgress` on the fallback path is therefore a silent no-op and the consumer never learns whether fallback succeeded or failed | `Core Engine/LlamaInstance.swift:184-196` (`cleanup()` at `:187`, fallback dispatch at `:192`), `cleanup` `:237-248` (`:246-247`), `attemptFallbackInitialization` from `:198`, `publishProgress` `:580-586` |

D8 was not on the original defect list; it was found during independent
verification and is recorded here so the baseline is complete at eight.

### E10. `Package.resolved` disagrees with the manifest, and any resolve rewrites it

`Package.swift:30` requires `.package(url: ".../mlx-swift-lm/", exact: "2.29.3")`.
The committed lockfile at `59da80b` pins that identity to
`{ "branch": "main", "revision": "6bb84aac..." }` in format `"version": 2` with
no `originHash`.

A branch pin cannot satisfy an `exact:` requirement. Consequently every
`swift build`, `swift test`, `swift package describe` and `swift package resolve`
rewrites the file on disk:

| Field | HEAD | Resolver output |
| --- | --- | --- |
| `mlx-swift-lm` | `branch: main` / `6bb84aac...` | `2.29.3` / `5064b8c5...` |
| `mlx-swift` (transitive) | `0.30.6` / `6ba4827f...` | `0.29.1` / `072b684a...` |
| lockfile `version` | `2` | `3` |
| `originHash` | absent | present |

A second `swift package resolve` after the first is stable. The *effective*
dependency graph in `.build/workspace-state.json` at HEAD is already
`mlx-swift 0.29.1` / `mlx-swift-lm 2.29.3`, so correcting the pins is truthful
bookkeeping about a graph that is already in force. The format bump is the only
real change, and it is evaluated in `## Migration and compatibility impact`.

## Design

Resolves blocking finding 6. The previous spec substituted a runbook for a
design and therefore never stated what the artifacts model.

The baseline is a **data product with two renderings of one model**, produced by
one capture script and checked by one check script:

- `docs/baselines/v2.0.4-environment.json` is the model. It is normative,
  machine-consumed, and schema-validated.
- `docs/baselines/v2.0.4.md` is a rendering of that model for humans, plus prose
  that a JSON document cannot carry: root-cause chains, consequences, and the
  mapping from finding to ticket.
- `scripts/capture-baseline.sh` emits both. Re-running the baseline is one
  command and drift is a diff, which is what makes the 14-day evidence rule
  affordable.
- `scripts/check-baseline.py` decides every acceptance criterion in this spec
  that is not already a one-line shell command. Its exit code is the gate.

Normalization rules, which is the word the previous spec left undefined:

- Every version is recorded as the exact string the tool printed, plus a
  separate build-number field where one exists. No reformatting, no truncation.
- Every path in the manifest is repository-relative or a redaction token. No
  absolute path is ever stored.
- Every measured count carries the command that produced it in an adjacent
  `_command` sibling field, because E1 and E7 both demonstrate that a count
  without its methodology is not a fact.
- Every build-matrix result is one of four enumerated strings, never free text.
- Arrays with a natural key are sorted by that key so two runs diff cleanly.

Stability guarantee for consumers: `schema_version` is bumped on any
incompatible change to the manifest. ODC-0003's result manifest and ODC-0303's
compatibility matrix may depend on `schema_version`, `repo.revision`,
`build_matrix[]`, `artifact.slices[]`, `dependencies[]` and `duplicate_sources[]`.
Everything else is advisory.

## Interfaces and data flow

Flow: pinned environment capture, then per-domain commands, then normalization
into the manifest, then rendering into the report, then check-script gate, then
ticket allocation.

### Manifest schema

Resolves major finding 7. The previous validation, "JSON manifest parses with
Python's standard library", is satisfied by `{}`. The schema below is normative
and is embedded in `scripts/check-baseline.py` as
`BASELINE_ENVIRONMENT_SCHEMA`, so there is exactly one copy and no link to a
file that must be kept in sync.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "OnDeviceCatalyst baseline environment manifest",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "captured_at", "repo", "toolchain", "host",
               "sdks", "package", "dependencies", "artifact", "build_matrix",
               "warnings", "findings", "duplicate_sources", "tests"],
  "properties": {
    "schema_version": { "type": "integer", "const": 1 },
    "captured_at": { "type": "string", "format": "date" },
    "repo": {
      "type": "object", "additionalProperties": false,
      "required": ["revision", "describe", "dirty"],
      "properties": {
        "revision": { "type": "string", "pattern": "^[0-9a-f]{40}$" },
        "describe": { "type": "string" },
        "dirty": { "type": "boolean" }
      }
    },
    "toolchain": {
      "type": "object", "additionalProperties": false,
      "required": ["swift", "swift_driver", "host_triple", "xcode_version",
                   "xcode_build", "developer_dir_redacted"],
      "properties": {
        "swift": { "type": "string" },
        "swift_driver": { "type": "string" },
        "host_triple": { "type": "string" },
        "xcode_version": { "type": "string" },
        "xcode_build": { "type": "string" },
        "developer_dir_redacted": { "type": "string" }
      }
    },
    "host": {
      "type": "object", "additionalProperties": false,
      "required": ["model_identifier", "chip", "cores", "memory_bytes",
                   "os_product", "os_version", "os_build"],
      "properties": {
        "model_identifier": { "type": "string" },
        "chip": { "type": "string" },
        "cores": { "type": "integer", "minimum": 1 },
        "memory_bytes": { "type": "integer", "minimum": 1 },
        "os_product": { "type": "string" },
        "os_version": { "type": "string" },
        "os_build": { "type": "string" }
      }
    },
    "sdks": {
      "type": "array", "minItems": 3,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["name", "version", "canonical_name"],
        "properties": {
          "name": { "type": "string" },
          "version": { "type": "string" },
          "canonical_name": { "type": "string" }
        }
      }
    },
    "package": {
      "type": "object", "additionalProperties": false,
      "required": ["name", "tools_version_declared", "tools_version_reported",
                   "platforms", "products", "targets"],
      "properties": {
        "name": { "type": "string" },
        "tools_version_declared": { "type": "string" },
        "tools_version_reported": { "type": "string" },
        "platforms": {
          "type": "array",
          "items": {
            "type": "object", "additionalProperties": false,
            "required": ["name", "version"],
            "properties": {
              "name": { "type": "string" },
              "version": { "type": "string" }
            }
          }
        },
        "products": { "type": "array", "items": { "type": "string" } },
        "targets": {
          "type": "array",
          "items": {
            "type": "object", "additionalProperties": false,
            "required": ["name", "type", "path", "source_count", "resources"],
            "properties": {
              "name": { "type": "string" },
              "type": { "type": "string" },
              "path": { "type": "string" },
              "source_count": { "type": "integer", "minimum": 0 },
              "resources": { "type": "array", "items": { "type": "string" } }
            }
          }
        }
      }
    },
    "dependencies": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["identity", "requirement", "resolved_version",
                     "resolved_revision", "direct"],
        "properties": {
          "identity": { "type": "string" },
          "requirement": { "type": "string" },
          "resolved_version": { "type": ["string", "null"] },
          "resolved_revision": { "type": "string", "pattern": "^[0-9a-f]{40}$" },
          "direct": { "type": "boolean" }
        }
      }
    },
    "lockfile": {
      "type": "object", "additionalProperties": false,
      "required": ["format_version_before", "format_version_after",
                   "origin_hash_added", "second_resolve_stable",
                   "raises_minimum_consumer_toolchain"],
      "properties": {
        "format_version_before": { "type": "integer" },
        "format_version_after": { "type": "integer" },
        "origin_hash_added": { "type": "boolean" },
        "second_resolve_stable": { "type": "boolean" },
        "raises_minimum_consumer_toolchain": { "type": "boolean" }
      }
    },
    "artifact": {
      "type": "object", "additionalProperties": false,
      "required": ["url", "checksum", "format_version", "slices"],
      "properties": {
        "url": { "type": "string" },
        "checksum": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
        "format_version": { "type": "string" },
        "slices": {
          "type": "array", "minItems": 1,
          "items": {
            "type": "object", "additionalProperties": false,
            "required": ["identifier", "platform", "variant", "architectures",
                         "bytes", "nm_line_count", "defined_symbol_count",
                         "objects", "is_stub", "_command"],
            "properties": {
              "identifier": { "type": "string" },
              "platform": { "type": "string" },
              "variant": { "type": ["string", "null"] },
              "architectures": { "type": "array", "items": { "type": "string" } },
              "bytes": { "type": "integer", "minimum": 0 },
              "nm_line_count": { "type": "integer", "minimum": 0 },
              "defined_symbol_count": { "type": "integer", "minimum": 0 },
              "objects": { "type": "array", "items": { "type": "string" } },
              "is_stub": { "type": "boolean" },
              "_command": { "type": "string" }
            }
          }
        }
      }
    },
    "build_matrix": {
      "type": "array", "minItems": 4,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["cell", "system", "triple", "sdk", "command", "exit_code",
                     "result", "cache_state", "cache_dependent",
                     "warning_count", "first_root_failure"],
        "properties": {
          "cell": { "type": "string" },
          "system": { "enum": ["swiftpm", "xcodebuild"] },
          "triple": { "type": ["string", "null"] },
          "sdk": { "type": ["string", "null"] },
          "command": { "type": "string" },
          "exit_code": { "type": "integer" },
          "result": {
            "enum": ["links, cannot infer", "compiles", "fails",
                     "blocked-by-build"]
          },
          "cache_state": { "enum": ["cold", "warm"] },
          "cache_dependent": { "type": "boolean" },
          "warning_count": { "type": "integer", "minimum": 0 },
          "first_root_failure": { "type": ["string", "null"] }
        }
      }
    },
    "warnings": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["group", "file", "line", "text"],
        "properties": {
          "group": {
            "enum": ["cosmetic-packaging", "non-functional-subsystem",
                     "deprecated-llama-api", "unused-code",
                     "concurrency-lifecycle"]
          },
          "file": { "type": "string" },
          "line": { "type": ["integer", "null"] },
          "text": { "type": "string" }
        }
      }
    },
    "findings": {
      "type": "array", "minItems": 8,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["id", "summary", "evidence", "bucket", "actionable",
                     "ticket"],
        "properties": {
          "id": { "type": "string", "pattern": "^D[0-9]+$" },
          "summary": { "type": "string" },
          "evidence": { "type": "array", "minItems": 1,
                        "items": { "type": "string" } },
          "bucket": { "type": "string" },
          "actionable": { "type": "boolean" },
          "ticket": { "type": "string", "pattern": "^ODC-[0-9]{4}$" }
        }
      }
    },
    "duplicate_sources": {
      "type": "object", "additionalProperties": false,
      "required": ["package_root", "app_root", "app_consumes_package",
                   "shared", "package_only", "app_only", "_command"],
      "properties": {
        "package_root": { "type": "string" },
        "app_root": { "type": "string" },
        "app_consumes_package": { "type": "boolean" },
        "shared": {
          "type": "array",
          "items": {
            "type": "object", "additionalProperties": false,
            "required": ["path", "identical", "changed_lines",
                         "package_sha256", "app_sha256"],
            "properties": {
              "path": { "type": "string" },
              "identical": { "type": "boolean" },
              "changed_lines": { "type": "integer", "minimum": 0 },
              "package_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
              "app_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" }
            }
          }
        },
        "package_only": { "type": "array", "items": { "type": "string" } },
        "app_only": { "type": "array", "items": { "type": "string" } },
        "_command": { "type": "string" }
      }
    },
    "tests": {
      "type": "object", "additionalProperties": false,
      "required": ["target_path", "compiled_test_sources", "test_case_names",
                   "orphaned_test_files", "runnable_on_host",
                   "defect_coverage_count"],
      "properties": {
        "target_path": { "type": "string" },
        "compiled_test_sources": { "type": "array", "items": { "type": "string" } },
        "test_case_names": { "type": "array", "items": { "type": "string" } },
        "orphaned_test_files": { "type": "array", "items": { "type": "string" } },
        "runnable_on_host": { "type": "boolean" },
        "defect_coverage_count": { "type": "integer", "minimum": 0 }
      }
    }
  }
}
```

### Report and manifest correspondence

Resolves the "both deliverables exist and agree" gap in blocking finding 5.
"Agree" is now a named field correspondence, enforced by
`scripts/check-baseline.py`. The report must contain, as literal substrings, the
values of exactly these manifest fields:

- `repo.revision`
- `toolchain.xcode_build`, `toolchain.swift`
- every `sdks[].canonical_name`
- `package.tools_version_declared`, `package.tools_version_reported`
- every `dependencies[].resolved_revision`
- `artifact.checksum`
- every `artifact.slices[].bytes` and `artifact.slices[].nm_line_count`
- every `build_matrix[].result` paired with its `build_matrix[].cell`
- `duplicate_sources.shared[].changed_lines` for every non-identical entry
- every `findings[].id` paired with its `findings[].ticket`

Any manifest value in that list absent from the report is a check-script
failure. This is a one-directional containment check, which is the strongest
form that is cheap to compute and impossible to satisfy vacuously.

## Required procedure

Every step names its command and the manifest field it fills. All steps run in
`$SCRATCH` per `## Pinned environment and clean state`, cold cache first, then
warm.

1. Create `$SCRATCH` with `mktemp -d`; populate with
   `git archive 59da80b | tar -x -C "$SCRATCH"`. Capture
   `git status --porcelain --ignored` and `git clean -ndx` from the working tree
   into the report. Fills `repo.*`.
2. Capture `swift --version`, `xcodebuild -version`, `xcodebuild -showsdks`,
   `xcode-select -p`, `sw_vers`. Fills `toolchain.*`, `sdks[]`.
3. Capture `sysctl -n hw.model machdep.cpu.brand_string hw.ncpu hw.memsize`.
   Host class only; no serial, no UDID. Fills `host.*`.
4. Run `swift package dump-package` (manifest only, no resolution) and
   `swift package tools-version`. This replaces the previous step 5, which asked
   for `swift package describe` "before resolution"; `describe` resolves
   implicitly, so a pre-resolution observation was only reachable from an
   unpinned warm `.build/`. Fills `package.tools_version_*`,
   `package.platforms`, `package.products`.
5. Run `swift package resolve` cold. Record the lockfile before and after by
   `sha256`, the `version` field before and after, and whether `originHash`
   appeared. Run `swift package resolve` a second time and record byte
   stability. Fills `lockfile.*`.
6. Run `swift package describe --type json` (post-resolution). Fills
   `package.targets[]`, `dependencies[]`, `tests.*`.
7. Locate the fetched XCFramework under the scratch scratch-path. For each slice
   record `LibraryIdentifier`, `SupportedPlatform`, `SupportedPlatformVariant`,
   `SupportedArchitectures`, byte size, `nm -gU | wc -l`, defined-symbol count,
   and `ar -t` object list. Set `is_stub` where the object list contains a
   `*stub*.o` member. Fills `artifact.*`. Also record the absence of any macOS
   slice as an explicit fact, not an omission.
8. Build cell `ios-simulator`:
   `swift build --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" --triple
   arm64-apple-ios17.0-simulator -c debug`. Record `result` as
   `links, cannot infer` and carry the E3 qualification into the report cell.
9. Build cell `ios-device`:
   `swift build --sdk "$(xcrun --sdk iphoneos --show-sdk-path)" --triple
   arm64-apple-ios17.0 -c debug`. This is the only cell compiled against the
   real header set. Record `result` as `compiles`, with the same statement that
   no link step occurred.
10. Build cell `macos`: `swift build -c debug`. Record `result` as `fails`, with
    `first_root_failure` set to the `path:line` and message from E4. Then record
    cell `macos-test` with `result` `blocked-by-build`, `exit_code` from
    `swift test`, and an explicit statement that zero test signal was produced.
    Do not present this as an independent test result.
11. Build cell `xcodeproj`:
    `xcodebuild -project OnDeviceCatalyst.xcodeproj -scheme OnDeviceCatalyst
    -destination 'generic/platform=iOS Simulator'
    CODE_SIGNING_ALLOWED=NO build`. Record `result` `fails` and classify the
    failure under `environment-dependent-gitignored-artifact` per
    `## Failure behavior`. Record that `project.pbxproj` contains zero
    `XCRemoteSwiftPackageReference` entries.
12. Characterize the source duplication. For each Swift file under both roots,
    record `identical`, `changed_lines` by the normative command in E7, and
    `sha256` of both copies; record `package_only` and `app_only` lists; record
    `app_consumes_package: false`. The report's table is emitted by the script,
    never hand-maintained. Fills `duplicate_sources.*`.
13. Group every ODC-owned warning into one of the five enumerated groups. The
    unhandled-resource diagnostic is split: `Assets.xcassets` into
    `cosmetic-packaging`; the seven `.metal` files into
    `non-functional-subsystem`, with the `MetalComputeEngine.swift:89`
    consequence stated in the report body. Fills `warnings[]`.
14. Record all eight defects D1 through D8 from E9 into `findings[]`, each with
    evidence citations, actionability per the rubric, and a mapped ticket. Fills
    `findings[]`.
15. Re-run steps 4 through 13 warm. For any cell whose result differs between
    cold and warm, set `cache_dependent: true` and record both values.
16. Run `scripts/check-baseline.py`. Its exit code gates completion.

The human report includes summarized output, never full build logs. Build
products and fetched dependencies live only under `$SCRATCH`, which is outside
the repository.

## Failure behavior

Resolves the taxonomy gap in blocking finding 4. The taxonomy is exhaustive over
the failures observed at `59da80b`; a failure that fits no bucket blocks
completion until a bucket is added by spec revision.

| Bucket | Meaning | Blocks completion |
| --- | --- | --- |
| `environmental-network` | Artifact or dependency download failure. Retried once, then classified. | No |
| `environment-dependent-gitignored-artifact` | The command's result depends on an untracked or gitignored local file, so two machines legitimately disagree. | No, but the result is recorded twice: present and absent. |
| `package-declares-platform-binary-does-not-support` | The manifest claims a platform the binary artifact has no slice for. | No |
| `project-configuration` | The Xcode project's own settings, independent of local artifacts. | No |
| `host-limitation` | The host cannot perform the measurement at all. | No |
| `procedure-defect` | The spec's own command is wrong or unreproducible. | Yes |
| `mutation` | A command modified a file under `Sources/`, `Tests/`, `Package.swift`, `OnDeviceCatalyst/`, or `OnDeviceCatalyst.xcodeproj/`. | Yes, and must be reverted |
| `unstable-resolution` | A second `swift package resolve` changes the lockfile. | Yes |

The `environment-dependent-gitignored-artifact` bucket exists because of a
concrete case. The Xcode project build fails with:

```
error: There is no XCFramework found at '.../OnDeviceCatalyst/llama.xcframework'.
(in target 'OnDeviceCatalyst' from project 'OnDeviceCatalyst')
** BUILD FAILED ** (exit 65)
```

`.gitignore:18` and `:20` ignore `llama.xcframework/` and `*.xcframework/`. A
developer who has ever unzipped the release asset into `OnDeviceCatalyst/` gets
a *different* result from a fresh clone, and neither `git status` nor a
"preserve unrelated work" instruction would reveal it. The procedure's
`git archive` scratch population guarantees the absent case is what gets
measured; the `git status --porcelain --ignored` capture in step 1 makes the
operator's own state visible; and the report must state both outcomes so a
reader with the artifact present is not misled.

Unexpected success is recorded as evidence and does not trigger speculative
cleanup.

## Security, privacy, and redaction

Resolves the missing security section (blocking finding 6) and the unverifiable
redaction criterion (blocking finding 5).

Deliberately not captured: device serial numbers, UDIDs, provisioning profiles,
Apple ID or team identifiers, network identifiers, `$HOME`-rooted absolute
paths, environment variables, and the contents of `~/Library/Caches`.

The redaction gate is a denylist regex applied to both deliverables by
`scripts/check-baseline.py`. Any match fails the check:

```
/Users/[^/ "]+
\$HOME
[0-9A-Fa-f]{8}-[0-9A-Fa-f]{16}
[0-9A-Fa-f]{40}-[0-9A-Fa-f]{16}
ghp_[A-Za-z0-9]{20,}
github_pat_[A-Za-z0-9_]{20,}
xox[baprs]-[A-Za-z0-9-]{10,}
-----BEGIN [A-Z ]*PRIVATE KEY-----
```

Two allowed exceptions, both narrow and both enumerated in the check script:
40-hex git revisions and 64-hex checksums are permitted, since they are the
evidence. The UDID-shaped patterns above are deliberately written to not match
either.

The report's fenced command blocks are additionally checked for the substring
`/Users/`, which makes the previous "every report command is copyable" criterion
mechanical.

## Migration and compatibility impact

Resolves blocking finding 2. The previous spec offered "preserve HEAD **or**
commit the resolver's output" while also requiring that a second resolve leave
the lockfile unchanged. Those are jointly unsatisfiable, which is exactly the
unresolved implementation decision program rule 6 forbids.

**Decision: commit the resolver's deterministic manifest-compatible output.**

Reason, recorded here as required:

- "Preserve HEAD" is not a real option. HEAD pins `mlx-swift-lm` to
  `branch: main` (E10) while `Package.swift:30` requires `exact: "2.29.3"`. A
  branch pin cannot satisfy an exact requirement, so every resolving command
  rewrites the file. A spec that both preserves HEAD and asserts resolve
  stability describes an impossible tree.
- The effective graph does not change. `.build/workspace-state.json` at HEAD
  already shows `mlx-swift 0.29.1` / `mlx-swift-lm 2.29.3`. Committing the
  resolver output makes the tracked file agree with the graph that is already in
  force. It is bookkeeping, not a dependency change.
- This is the single tracked-file change in the ticket, and it is argued here
  rather than slipped in as a deliverable.

**Consumer-tooling consequence of the format bump.** Committing the resolver
output moves the lockfile from `"version": 2` to `"version": 3` and adds
`originHash`. Evaluated:

- The floor on consumer tooling is set by `Package.swift:1`, not by the
  lockfile. `swift-tools-version: 5.12` is not a shipped Swift release (E6), and
  SwiftPM accepts it only because the installed tools version compares greater.
  The manifest therefore already requires a toolchain newer than any that
  writes format 2.
- Consequently the format bump **cannot** raise the effective minimum consumer
  toolchain above the level the manifest already demands. The manifest records
  this as `lockfile.raises_minimum_consumer_toolchain: false`, and the procedure
  must verify it rather than assume it: the pinned toolchain's own resolver
  output is format 3, recorded in step 5, which demonstrates that no toolchain
  capable of building this package emits format 2.
- If a future run measures a toolchain that satisfies `Package.swift` and emits
  format 2, `raises_minimum_consumer_toolchain` becomes `true` and that is a
  compatibility change requiring its own decision ticket. Stating the falsifier
  is what makes the claim a claim.

Nothing else migrates. No public API, no runtime source, no binary artifact, and
no Xcode project file is touched.

## Tests and benchmarks

Explicitly out of scope, as a decision rather than a gap.

- **Tests are owned by ODC-0004.** This ticket adds no test. It supplies
  ODC-0004's input: the `tests` object in the manifest, which separates the 4
  compiled XCTest cases from the 3 orphaned files under `Tests/`, records
  `runnable_on_host: false` with its root cause (E4), and records
  `defect_coverage_count: 0` for the eight defects in E9.
- **Benchmarks are owned by ODC-0003.** This ticket runs no benchmark and
  produces no timing claim. Build durations are deliberately excluded from the
  manifest, because a wall-clock number in a baseline invites exactly the kind
  of unpinned comparison the program's benchmark contract exists to prevent.
- The only executable this ticket adds is capture and check tooling, which the
  program plan permits before the research gate as baseline reproduction and
  project infrastructure.

## Validation

Each line is a command and the condition on its exit code or output.

1. `python3 scripts/check-baseline.py` exits 0. This single command decides
   schema conformance, report/manifest correspondence, redaction, and the
   copyable-command rule.
2. `python3 -c "import json,sys;json.load(open(sys.argv[1]))"
   docs/baselines/v2.0.4-environment.json` exits 0.
3. `jq -e '.version == 3' Package.resolved` exits 0 after step 5 of the
   procedure.
4. `swift package resolve && sha256sum Package.resolved > /tmp/a && swift package
   resolve && sha256sum Package.resolved > /tmp/b && diff /tmp/a /tmp/b` exits 0.
5. `git diff --stat 59da80b -- Sources Tests Package.swift OnDeviceCatalyst
   OnDeviceCatalyst.xcodeproj` produces empty output.
6. The `ios-simulator` cell's `result` field equals the literal
   `links, cannot infer`;
   `jq -e '.build_matrix[]|select(.cell=="ios-simulator")|.result ==
   "links, cannot infer"'` exits 0.
7. Every `findings[]` entry has a non-empty `ticket` matching `^ODC-[0-9]{4}$`
   and that ticket exists in `Tickets.md`; enforced by
   `scripts/check-baseline.py`.
8. `python3 scripts/validate-project-state.py` exits 0.

Scope note on item 8, so no reader over-trusts it. That script checks front
matter *key presence*, ticket-to-spec status agreement, founder-approval state
for approved statuses, `unresolved_questions == "none"` for approved statuses,
ADR ID and status validity, dependency existence, local Markdown link
resolution, and the roadmap's current-ticket pointer. It does **not** read
`docs/baselines/`, does not inspect spec section structure, and does not
evaluate `evidence_fresh_until` beyond requiring the key to exist. It is
therefore a project-state gate, not a baseline-content gate.
`scripts/check-baseline.py` is the baseline-content gate. Freshness of
`evidence_fresh_until` is enforced by founder review, not by the validator; this
is stated so its absence is a known limit rather than an assumed feature.

## Acceptance criteria

Every criterion is decided by a command's exit code. Criteria that could not be
decided this way have been deleted, not softened (blocking finding 5).

| # | Criterion | Deciding command |
| --- | --- | --- |
| A1 | Both deliverables exist | `test -f docs/baselines/v2.0.4.md && test -f docs/baselines/v2.0.4-environment.json` |
| A2 | Manifest conforms to the schema in `## Interfaces and data flow` | `python3 scripts/check-baseline.py --schema-only` |
| A3 | Report and manifest agree on every field in the correspondence list | `python3 scripts/check-baseline.py --correspondence` |
| A4 | No secret or personal identifier in either deliverable | `python3 scripts/check-baseline.py --redaction` |
| A5 | No fenced command block contains `/Users/` | `python3 scripts/check-baseline.py --copyable` |
| A6 | Every build path has an enumerated result, and the simulator cell is `links, cannot infer` | `python3 scripts/check-baseline.py --matrix` |
| A7 | Every slice carries bytes, both symbol counts, its object list, and `is_stub` | `python3 scripts/check-baseline.py --slices` |
| A8 | Dependency and artifact revisions are pinned: every `dependencies[].resolved_revision` is 40-hex and `artifact.checksum` is 64-hex | `python3 scripts/check-baseline.py --pins` |
| A9 | All eight defects D1 to D8 are present in `findings[]` with evidence and a ticket that exists in `Tickets.md` | `python3 scripts/check-baseline.py --findings` |
| A10 | The source-duplication record is present and non-vacuous: `app_consumes_package` is false and `shared` has 22 entries | `python3 scripts/check-baseline.py --duplication` |
| A11 | No tracked file outside `Package.resolved` and `docs/` changed since `59da80b` | `git diff --stat 59da80b -- Sources Tests Package.swift OnDeviceCatalyst OnDeviceCatalyst.xcodeproj` is empty |
| A12 | The resolved dependency graph is unchanged by this ticket | `.build/workspace-state.json` dependency identities and revisions before and after are byte-identical |
| A13 | A second resolve is stable | Validation item 4 exits 0 |
| A14 | Project state is consistent | `python3 scripts/validate-project-state.py` exits 0 |

Deleted from the previous version: "runtime behavior and public API are
unchanged". It has no possible evidence. The package does not build on macOS
(E4), `swift test` cannot run (E4), there is no recorded API baseline to compare
against, and no inference is executed anywhere in this ticket. A11 and A12
together carry the defensible half of that claim: no source changed, and the
dependency graph did not move.

## Ticket allocation

Resolves major finding 9. Without an allocation rule, two implementers produce
different ledgers from the same evidence.

**Reserved range.** `ODC-0010` through `ODC-0049` is reserved for tickets
created by this procedure. The range is currently empty and sits below the
`ODC-0100` milestone block, so it does not collide.

**Actionable rubric.** A finding is `actionable: true` and gets a ticket if
either holds:

1. It changes what a consumer of the package can do (a build path fails, a
   subsystem is unreachable, a documented claim is false), or
2. It blocks a named downstream ticket from starting.

Otherwise it is `actionable: false`, is recorded in `findings[]` and in the
report, and gets no ticket. By this rubric: D1 through D8 and E2 are actionable.
An unused local variable at `ModelWeights.swift:87` is not; it is recorded in
`warnings[]` only.

**Default ledger column values** for tickets this procedure creates, chosen so
`scripts/validate-project-state.py` still passes immediately after creation:

| Column | Default |
| --- | --- |
| `Type` | `bug` for D1, D2, D3, D8; `packaging` for D4, D5; `chore` for D6; `decision` for D7 and E2 |
| `Milestone` | `P0` for D4, D5, D7; `P1` otherwise |
| `Status` | `BACKLOG` |
| `Priority` | `P0` for D4, D5, D7; `P2` otherwise |
| `Dependencies` | `ODC-0002` |
| `Spec` | `TBD` |
| `GitHub Issue` | `TBD` |
| `Owner` | `unassigned` |
| `Updated` | the capture date |
| `Next Gate` | `discovery` |

`BACKLOG` is deliberate: it is outside `SPEC_REQUIRED_STATUSES`, so a
newly created ticket does not require a spec link and the validator passes on
the same commit that creates it.

**Named downstream consumers.** D7 (the source fork) blocks ODC-0004, which
cannot characterize an ambiguous target, and blocks ODC-0300. The decision of
which copy survives is the first ticket allocated from the reserved range, at
`P0`. E3's uncovered link-time and runtime resolution against the stub slice is
allocated its own ticket in the same range; it is not folded into D4.

## Alternatives considered

- **Fixing packaging while measuring.** Rejected: it destroys the baseline that
  ODC-0003, ODC-0004 and ODC-0005 depend on, and the novel-implementation gate
  is closed.
- **Recording only successful commands.** Rejected: the deterministic failures
  are the most valuable input to the v3 plan.
- **Preserving the HEAD lockfile.** Rejected as impossible, not merely
  undesirable; see `## Migration and compatibility impact`.
- **Recording the simulator build as a plain pass.** Rejected: it is the single
  most misleading statement the baseline could make, and three downstream
  tickets would cite it. E1 and E3 exist to make the stub visible in the
  evidence rather than inferable from it.
- **Keeping the procedure as a manual runbook.** Rejected: twelve manual steps
  across three build systems, re-run by hand under a 14-day freshness rule, will
  rot silently. The capture and check scripts make re-running one command and
  drift a diff.
- **A separate `docs/schemas/*.json` file for the manifest schema.** Rejected:
  two copies of a schema drift. The schema is inlined here and embedded in the
  check script, which is the one place that enforces it.
- **Deleting the app fork as part of this ticket.** Rejected: it is a design
  decision with consumer impact, not a measurement, and it belongs to a ticket
  that can weigh reconciliation against deletion.

## Review record

- 2026-08-25, pass one, completeness review: commands, redaction, outputs,
  failure classes, lockfile decision, and acceptance evidence were specified.
  No artifact was recorded for this pass.
- 2026-08-25, pass two, adversarial review: prevented baseline repair, personal
  identifier leakage, generated-log commits, and mislabeled environmental
  failures. No artifact was recorded for this pass.
- 2026-09-01, pass two, adversarial spec review, verdict REJECT, returned to
  `REVISION`. Artifact: [`docs/reviews/ODC-0002-review-pass-2.md`](../reviews/ODC-0002-review-pass-2.md).
  Six blocking findings, six major, five minor.
- 2026-09-01, revision: this rewrite. Blocking findings 1 through 6 are resolved
  at `## Current state and evidence` (E1, E2, E3, E7, E9), `## Migration and
  compatibility impact`, `## Pinned environment and clean state`,
  `## Acceptance criteria`, and by the creation of
  [`docs/templates/baseline-spec.md`](../templates/baseline-spec.md). Majors 7,
  8, 9, 10, 11 and minors 12, 13, 14, 15, 16 are resolved at
  `## Interfaces and data flow`, E5, `## Ticket allocation`, E4, E6,
  procedure step 4, `## Pinned environment and clean state`, E8, this record,
  and `## Design` respectively.
- Founder review: pending. Advancing this spec beyond `REVISION` is a separate
  gate.

## Validation evidence

Not implemented. The procedure has not been executed; founder approval is
required before execution. `evidence_fresh_until` is set to 2026-09-15, fourteen
days from the revision date, per the program's evidence rule. The verified facts
cited throughout `## Current state and evidence` were established on 2026-09-01
against revision `59da80b` by read-only inspection and by builds performed in
scratch copies outside the working tree; no file under `Sources/`, `Tests/`,
`OnDeviceCatalyst/`, `Package.swift`, or `Package.resolved` was modified while
establishing them.
