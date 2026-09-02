---
id: ODC-0004
title: V2 characterization suite
type: test
status: DONE
milestone: P0
owner: unassigned
dependencies: ODC-0002
founder_approved: delegated-to-manager-2026-09-01
last_updated: 2026-09-02
evidence_fresh_until: 2026-09-15
unresolved_questions: none
---

# ODC-0004: V2 characterization suite

## Summary

Build a test suite that pins what v2.0.4 does today, including the eight defects
recorded by [ODC-0002](ODC-0002-v2-baseline.md), without changing one line of
runtime behavior. When v3 work later changes any pinned behavior, the suite must
fail loudly, name the defect ticket, and state what the assertion should become.

This is not a bug-fix ticket. A characterization test asserts current behavior
even when that behavior is wrong. Every such assertion carries a mandatory
comment block naming the defect, the ticket that will repair it, and the
assertion that replaces it after repair.

The baseline handed this ticket three hard constraints, and a fourth was
discovered while drafting:

1. The package does not build on macOS at all. `swift build` and `swift test`
   both fail with `no such module 'llama'` because the XCFramework has no macOS
   slice. Host `swift test` is therefore not an execution surface and never
   appears in this spec as one.
2. The iOS simulator slice is a 7,936-byte, 51-defined-symbol stub. It cannot
   perform inference.
3. Three files under `Tests/` sit outside the declared target path and are
   compiled by no target.
4. **New**: the declared test target does not compile on any triple. See N1.
   The four "existing tests" were never built by anything.

Against those constraints, this spec establishes a **measured** execution
surface. On 2026-09-01, against revision `6d72193` in a scratch tree outside the
working tree, the existing test target was built for
`arm64-apple-ios17.0-simulator` with `swift build --build-tests`, linked
successfully against the stub slice, was repackaged as a flat iOS bundle, and
executed on an iOS 26.5 simulator through the iPhoneSimulator platform's `xctest`
agent. Four test cases ran. Three passed and one failed. Purpose-written probes
then reproduced defect D8's consumer-visible symptom and half of D1 on that same
surface, with no physical device, no model weights, and no Xcode project. The
evidence is in `## Current state and evidence`.

The suite is consequently **not** device-only. It is partitioned by what each
assertion actually requires, and only the smallest partition needs hardware and a
model file.

Structure follows [`docs/templates/test-spec.md`](../templates/test-spec.md),
created by this ticket because `type: test` previously had no template. ODC-0002
set that precedent for `type: baseline`.

## Goals

- Pin every behavior that a v3 change could silently alter, with an assertion
  that fails loudly and names the ticket that owns the change.
- Partition the suite by execution requirement, so that no test is written which
  cannot be run somewhere, and no test silently passes because its precondition
  was absent.
- State, for each of the eight baseline defects, whether it is testable as
  executable code, and if not, what mechanical assertion replaces the test.
- Make the distinction between a characterization assertion and a correctness
  assertion visible in the test name, so no reader mistakes a pinned bug for
  intended behavior.
- Repair the one thing that blocks all of the above: the test target must
  compile. That is a test-source change, not a runtime change.
- Deliver a runner and a checker as tracked scripts, so re-running the suite is
  one command and CI can gate on it.
- Leave `Sources/`, `Package.swift`, `OnDeviceCatalyst/`, and
  `OnDeviceCatalyst.xcodeproj/` byte-identical.

## Non-goals

- Repair any of D1, D2, D3, D4, D5, D7, or D8. Each has its own ticket
  (ODC-0010 through ODC-0017). This ticket pins them.
- Decide which of the two runtime copies survives. That is ODC-0016. This suite
  characterizes `Sources/OnDeviceCatalyst/` only, and says so in every artifact.
- Add, remove, or reorder any target, product, dependency, or platform in
  `Package.swift`.
- Add a macOS slice, a Metal resource declaration, or any other packaging repair.
- Produce a timing, throughput, or memory claim. Benchmarks belong to ODC-0003.
- Commit model weights, or any artifact whose license or size makes it
  unsuitable for the repository.
- Delete or revive the three orphaned files under `Tests/`. Their disposition is
  allocated to a ticket in `## Ticket allocation`; this ticket pins their current
  uncompiled state so the decision is visible rather than accidental.
- Achieve a coverage percentage. Coverage of the eight defects is the target;
  line coverage is not a goal and is not measured.

## Current state and evidence

Everything below was verified on 2026-09-01. Facts labelled `B` are inherited
from [`docs/baselines/v2.0.4.md`](../baselines/v2.0.4.md) and its manifest
[`v2.0.4-environment.json`](../baselines/v2.0.4-environment.json). Facts labelled
`N` are new, established by this ticket's discovery, and are not in the baseline.
Every new fact names the command that produced it.

All new measurements ran in a scratch tree created with `mktemp -d` outside the
repository and populated with `git archive HEAD | tar -x -C "$SCRATCH"`. The
scratch tree, its build products, and a temporary simulator device created for
the run were deleted afterwards. `git status --porcelain` was empty before and
after.

### B1. The eight baseline defects and their tickets

| ID | Defect | Repair ticket |
| --- | --- | --- |
| D1 | `releaseInstance` caches a ready instance and asynchronously shuts it down; the only cache reader does not re-check `isReady` | ODC-0010 |
| D2 | `performGeneration` appends a second, always-`.natural` completion after `generateTokens` already emitted one | ODC-0011 |
| D3 | `publishProgress` gates on `if case .ready = progress, case .failed = progress`, a compound AND over one value, which is unsatisfiable | ODC-0012 |
| D4 | `.macOS(.v14)` declared with no macOS slice | ODC-0013 |
| D5 | 8 unhandled files; no `.metallib`; `makeDefaultLibrary()` leaves the Metal Engine unreachable | ODC-0014 |
| D6 | Lockfile pinned `mlx-swift-lm` to a branch against an `exact:` requirement. **Corrected at HEAD** | ODC-0002 |
| D7 | `OnDeviceCatalyst/` is a divergent fork; 12 of 22 shared files drifted; zero package references in the app target | ODC-0016 |
| D8 | `handleInitializationError` calls `cleanup()` before `attemptFallbackInitialization`, so every fallback-path `publishProgress` is a silent no-op | ODC-0015 |

D6 is different in kind from the other seven: it has already been fixed. It
therefore gets a **regression** assertion, not a characterization assertion, and
a different name prefix. See `## Design`.

### B2. Existing coverage of those defects is zero

`Package.swift:58-62` declares the test target at `Tests/OnDeviceCatalystTests`.
The single compiled source declares four cases: `testModelProfileCreation`,
`testInstanceSettingsValidation`, `testPredictionConfigPresets`,
`testCatalystServiceInitialization`. None touches instance caching or release,
streaming generation, load-progress termination, fallback initialization, or the
Metal backend. The baseline records `defect_coverage_count: 0`.

### B3. macOS is not an execution surface

`Package.swift:21` declares `.macOS(.v14)`; the XCFramework exposes only
`ios-arm64` and `ios-arm64-simulator`. `swift build -c debug` fails at
`Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:12:8: error: no such
module 'llama'`, and `swift test` fails identically because the test target
depends on a library target that never compiles. The baseline records this as one
data point, not two, with `swift test` classified `blocked-by-build` and zero
test signal.

Consequence for this ticket, stated plainly: **`swift test` on the host cannot
run and this spec proposes no test that depends on it.** Repairing that is
ODC-0013, not ODC-0004.

### N1. The declared test target does not compile on any triple

New, and it reframes B2. The four cases were never built by anything, so calling
them "existing coverage" overstates them. Reproduction:

```
swift build --build-tests \
  --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
  --triple arm64-apple-ios17.0-simulator -c debug
```

The library target compiled. The test target did not:

```
Tests/OnDeviceCatalystTests/OnDeviceCatalystTests.swift:31:46: error: type
'PredictionConfig' has no member 'quality'
```

`Sources/OnDeviceCatalyst/Core Foundation/PredictionConfig.swift` declares
exactly five presets: `balanced` (`:71`), `creative` (`:76`), `speed` (`:87`),
`deterministic` (`:100`), `mirostat` (`:110`). There is no `quality`. The test
file encodes a belief about the API that the API does not satisfy.

This is a new defect. It is allocated a ticket in `## Ticket allocation` and it
is the one defect this ticket repairs, because every other deliverable is blocked
behind it and the repair is confined to test sources.

### N2. The stub slice defines exactly the symbol set the package references, so the test bundle links

ODC-0002 explicitly left link-time resolution against the stub uncovered
("not covered by this baseline, because no SwiftPM invocation in this procedure
links it"). This ticket closes it, twice.

Statically. The set of `llama_*` symbols left undefined by the compiled library
objects and the set defined by the stub archive are **identical, 51 and 51, with
an empty symmetric difference**:

```
nm -gU "$XCF/ios-arm64-simulator/libllama_combined.a" \
  | awk '$2 ~ /^[A-Za-z]$/ {print $3}' | sort -u > stub_defined.txt
nm -u .build/arm64-apple-ios-simulator/debug/OnDeviceCatalyst.build/*.o \
  | grep -E '^_(llama|ggml)' | sort -u > required.txt
comm -3 stub_defined.txt required.txt   # empty
```

Dynamically. After N1 was worked around in the scratch tree, the same
`swift build --build-tests` invocation reported `Linking
OnDeviceCatalystPackageTests` and `Build complete!`. The stub is not an accident
of packaging; it was cut to exactly this package's call set.

Two link-step warnings are recorded because they are evidence, not noise:

```
clang: warning: using sysroot for 'MacOSX' but targeting 'iPhone'
ld: warning: object file libllama_combined.a llama_sim_stubs.o was built for
newer 'iOS-simulator' version (26.2) than being linked (17.0)
```

### N3. The stub returns null and zero without trapping, so failure paths are deterministic on the simulator

`ar -t` on the simulator slice lists `llama_stub.o` (defines nothing) and
`llama_sim_stubs.o` (defines all 51). Disassembly with `otool -tvV` shows
`_llama_backend_init` is a bare `ret`:

```
_llama_backend_init:
    ret
```

`_llama_load_model_from_file` returns null unconditionally, but not via the
two-instruction sequence an earlier draft of this section quoted. The actual
disassembly is six instructions, a stack frame set up and torn down around the
null return:

```
_llama_load_model_from_file:
    sub sp, sp, #0x10
    str x0, [sp, #0x8]
    str x1, [sp]
    mov x0, #0x0
    add sp, sp, #0x10
    ret
```

The functional conclusion is unchanged by the correction: the return is
unconditional and there is no trap. `_llama_new_context_with_model` likewise
returns null, and `_llama_n_ctx` returns zero. A scan for `brk`, `udf`,
`_abort`, `_exit`, and `trap` across both objects returns **zero** matches.

Consequence, and it is the single most useful fact in this spec: on the iOS
Simulator, `LlamaBridge.loadModel` (`API Bridge/LlamaBridge.swift:55-57`)
deterministically takes its failure branch for any file that passes preflight
validation, and it does so without crashing the test process. The model-load
failure paths, and everything downstream of them, are therefore executable on the
simulator with no model weights.

### N4. The simulator slice's minimum iOS version exceeds the package's declared floor

The linker warning in N2 states the stub object was built for iOS Simulator
26.2, while `Package.swift:20` declares `.iOS(.v17)`. The package advertises an
iOS 17 floor that its own simulator artifact does not honor. Recorded as a
finding and mapped to ODC-0017, which already owns the false simulator-guard
comment at `Package.swift:36-37`.

### N5. The loading stream delivers no terminal event on any failure path

Measured on the simulator surface with purpose-written probes and two synthetic
fixtures, each a 2 MB file whose first four bytes are `GGUF`, so both pass
`ModelProfile.validateModel()` (`Core Foundation/ModelProfile.swift:106-126`) and
`LlamaBridge.validateModelFile` (`API Bridge/LlamaBridge.swift:104-134`).

Fixture A is named `phi-probe.gguf`. `LlamaBridge.createModelLoadingError`
(`:64-100`) branches on the **filename**, so a name containing `phi` yields
`.architectureUnsupported`, whose `isRecoverable` is `true`
(`Core Foundation/CatalystError.swift:89-102`). That is the recoverable class,
which reaches `attemptFallbackInitialization` and therefore reaches D8.

Fixture B is named `generic-probe.gguf` and yields `.modelLoadingFailed`, whose
`isRecoverable` is `false`. That is the non-recoverable class, which reaches
`publishProgress(.failed(...))` directly.

Both produced the identical observable sequence:

```
preparing("Validating model file")
loading("Initializing llama.cpp backend")
loading("Loading model from <path>")
<stream terminated>
```

Three events, then termination, with `isReady == false`. Neither `.ready` nor
`.failed` was ever delivered, on either path.

Three separate defects combine to produce this:

- `cleanup()` (`Core Engine/LlamaInstance.swift:237-248`) finishes and nils
  `loadingContinuation` at `:246-247`, and `handleInitializationError` calls it
  at `:188` **before** dispatching either branch. That is D8.
- Every subsequent `publishProgress` is therefore a no-op against a nil
  continuation, on both the fallback branch and the non-recoverable branch.
- The gate at `:583` would not have finished the stream anyway, because
  `if case .ready = progress, case .failed = progress` is unsatisfiable. That is
  D3.

The consumer-visible result is that **a recoverable failure and a
non-recoverable failure are externally indistinguishable**, and neither is
reported at all. `Service Layer/Catalyst.swift:208` and `:400` both consume the
loading stream with `if case .failed(let message) = progress { throw ... }`. That
branch cannot fire. The loop simply ends and the caller proceeds with a
not-ready instance.

Recorded honestly: the probe observed the event sequences, not which internal
branch was taken. That the two classes are indistinguishable is precisely the
finding.

### N6. The instance cache returns a not-ready instance

Also measured on the simulator surface. `ModelProfile(mlxModelId:name:)`
(`Core Foundation/ModelProfile.swift:56-65`) performs no file access, so a
`LlamaInstance` can be constructed with no model at all. Storing it with
`ModelCache.shared.storeInstance` and reading it back with
`ModelCache.shared.getInstance` (`Service Layer/CacheSettings.swift:93-107`)
returned the instance with `isReady == false`.

This is the reader half of D1, and it is executable with no inference. The write
half, in which `Catalyst.releaseInstance` caches a ready instance at
`Service Layer/Catalyst.swift:507` and then shuts it down at `:510-512`, needs a
genuinely ready instance and is therefore device-and-model work.

One more property fell out. `storeInstance` writes under
`queue.async(flags: .barrier)` (`Service Layer/CacheSettings.swift:117`), so it
is asynchronous. An immediate `getInstance` can legitimately return nil; the
probe had to poll. Any test of this pair must poll with a bounded wait rather
than assume ordering, and the polling bound is a harness parameter, not a
performance claim.

### N7. The xcodebuild route is available but costs more than it returns

Measured, because "run the tests with xcodebuild" is the obvious first proposal
and it has three concrete problems.

First, `xcodebuild -list` executed at the repository root reports the project
`OnDeviceCatalyst` with a single scheme `OnDeviceCatalyst`, not the package. The
tracked `.xcodeproj` shadows the package for every bare `xcodebuild` invocation.

Second, a workspace referencing the package makes the package visible, but its
auto-generated scheme is a library-product scheme with no test action:

```
xcodebuild: error: Scheme OnDeviceCatalyst is not currently configured for the
test action.
```

Adding a tracked shared scheme under `.swiftpm/xcode/xcshareddata/xcschemes/`
does produce a scheme with a test action, and `xcodebuild test` then accepts an
iOS Simulator destination and starts building.

Third, and decisively, that build fails on this host:

```
error: cannot execute tool 'metal' due to missing Metal Toolchain;
use: xcodebuild -downloadComponent MetalToolchain
Testing failed: The Metal Toolchain was not installed and could not compile the
Metal source files.
```

Xcode's build system schedules `CompileMetalFile` for all seven shaders under
`Sources/OnDeviceCatalyst/Metal Engine/Shaders/` plus nine from `mlx-swift`. The
Xcode 26 Metal Toolchain is a separately downloaded component, so any
xcodebuild-driven route adds a multi-gigabyte CI prerequisite. `swift build`
avoids it only because SwiftPM does not compile the `.metal` files at all, which
is defect D5. D5 is what makes the cheap route work.

That observation also **refines** D5 rather than merely confirming it. Under
Xcode the shaders are compiled, into the resource-bundle target
`OnDeviceCatalyst_OnDeviceCatalyst`, whereas
`Metal Engine/Compute/MetalComputeEngine.swift:89` calls
`device.makeDefaultLibrary()`, which reads `Bundle.main`, not `Bundle.module`. So
the shader library would not be found even when it is built. Not observed
directly here, because the toolchain component is absent; recorded as open
question Q2 and handed to ODC-0014.

### N8. Two behavioral details that a naive test would get wrong

`PredictionConfig.balanced.maxTokens` is `-1` and `PredictionConfig.speed
.maxTokens` is `1024` (`Core Foundation/PredictionConfig.swift:71-96`). The
existing `testPredictionConfigPresets` asserts the opposite ordering, and after
N1 was worked around it failed at run time:

```
XCTAssertGreaterThan failed: ("-1") is not greater than ("1024")
```

`ModelArchitecture.detectFromPath("phi-probe.gguf")` returns `.unknown`
(`Core Foundation/ModelArchitecture.swift:71-110`), even though
`LlamaBridge.createModelLoadingError` has a dedicated `phi` branch. Two different
filename classifiers in the same package disagree about the same filename.

### N9. The three orphaned files are not merely uncompiled

`Tests/test_embedding.swift` begins `#!/usr/bin/env swift` and executes
statements at file scope. Top-level code outside `main.swift` does not compile in
a Swift target, so this file cannot be folded into the test target as it stands.

`Tests/BERTEmbeddingTest.swift:15` hard-codes an absolute home-directory path
into a tracked file. `Tests/EmbeddingTest.swift:17` hard-codes
`/path/to/bge-small-en-v1.5.gguf`. Both then `throw XCTSkip(...)` when the file
is missing, which means that if either were compiled it would skip forever and
report green. That is exactly the silent-pass failure mode this spec's skip
protocol exists to prevent, and the repository already contains two working
examples of it.

### N10. Measured execution surface

The following sequence ran the existing test target to completion on an iOS 26.5
simulator with no Xcode project, no scheme, no code signing, and no Metal
Toolchain:

```
swift build --build-tests \
  --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
  --triple arm64-apple-ios17.0-simulator -c debug
# repackage .build/arm64-apple-ios-simulator/debug/OnDeviceCatalystPackageTests
#   .xctest/Contents/MacOS/<binary> into a flat iOS bundle with an Info.plist
xcrun simctl spawn "$UDID" \
  "$(xcode-select -p)/Platforms/iPhoneSimulator.platform/Developer/Library/Xcode/Agents/xctest" \
  -XCTest All <flat-bundle>
```

Result: `Executed 4 tests, with 1 failure (0 unexpected)`. The three passes and
the one failure are reported in N8.

The repackaging step exists because SwiftPM emits a macOS-shaped bundle
(`Contents/MacOS/<binary>`, no `Info.plist`) even when cross-compiling for the
simulator, and the platform `xctest` agent requires a flat iOS bundle. Passing
the SwiftPM bundle directly fails with
`incompatible platform (have 'iOS-simulator', need 'macOS')`.

### Evidence freshness

All facts above were established on 2026-09-01 with the toolchain pinned by
ODC-0002: Xcode 26.6 (17F113), Swift 6.3.3, macOS 26.4 (25E246), SDKs
`iphoneos26.5`, `iphonesimulator26.5`, `macosx26.5`. `evidence_fresh_until` is
2026-09-15 per the program's 14-day rule. An expired spec returns to discovery
before implementation.

## Design

### What a characterization assertion is here

A characterization assertion states what the code does today. It is allowed, and
often required, to assert something wrong. It is not allowed to be silent about
that. Every characterization case carries this four-line block immediately above
it, and the block is machine-checked:

```
/// CHARACTERIZATION D2 (ODC-0011)
/// Today: performGeneration yields a second completion whose reason is always
///        .natural, after generateTokens already yielded the real reason.
/// Should be: exactly one completion chunk, carrying the real reason.
/// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:384-386
```

`scripts/check-characterization.py --naming` fails if any test whose name marks
it as a characterization case lacks all four lines, or names a ticket that is
absent from `Tickets.md`.

When ODC-0011 lands, the case fails. That is the intended outcome. The failure
message is the `Should be:` line, so the person who broke it is told, in the
failure output, what to change the assertion to.

### Requirement classes

Tests are partitioned by what they actually need, not by where they happen to
run. Four classes, and every case in `## Tests` declares exactly one.

| Class | Requires | Rationale |
| --- | --- | --- |
| `R0` | Nothing. No compilation, no simulator, no device. | Packaging and repository facts. Not XCTest at all. |
| `R1` | The module compiles and the test bundle links. No `llama_*` return value is consumed. | Pure Swift value and lifecycle behavior. |
| `R2` | A `llama_*` call whose **failure** result is consumed. Satisfied by the stub (N3) or by real hardware. | Model-load failure paths, fallback, progress termination. |
| `R3` | Real inference: hardware with the device slice, plus a model asset. | Generation, completion reasons, the ready-instance lifecycle. |

### Execution surfaces

| Surface | What it is | Runs | Measured |
| --- | --- | --- | --- |
| `SFC-A` | Any host with Python 3 and a checkout. No build. | `R0` | Yes |
| `SFC-B` | iOS Simulator, arm64, stub slice, via SwiftPM plus `simctl` per N10. | `R0`, `R1`, `R2` | Yes, 4 cases executed |
| `SFC-C` | Physical arm64 iOS device, real slice. | `R0`, `R1`, `R2`, and `R3` when a model asset is present | No. See Q1 and Q3 |
| `SFC-X` | macOS host `swift test`. | Nothing. Blocked by D4. | Yes, fails |

`SFC-X` is listed only so that a reader who reaches for `swift test` finds the
answer here instead of rediscovering B3.

The suite's normative surface is `SFC-B`, because it is the only one that runs
unattended in CI today. `SFC-C` is operator-run. Every `R3` case is therefore
written to be skipped, explicitly and loudly, on `SFC-B`.

**`SFC-C`'s design status, stated plainly rather than left to be inferred from
Q1 alone.** `SFC-C` carries two distinct open questions, not one. Q1 asks
whether a physical device is *available*; even if the answer is yes, this
ticket does not specify the *mechanism* by which a package-only, no-Xcode
-project test bundle gets code-signed, provisioned, installed, and launched on
that device. `SFC-B` required solving several non-obvious problems to reach
"measured" (N7, N10: SwiftPM's macOS-shaped bundle output, a hand-authored
`Info.plist`, the platform `xctest` agent). None of that discovery was
repeated for the device case, and the device case is not a smaller version of
it: it plausibly needs a provisioning profile, an entitlements file, and a
device-deployment tool (`devicectl`, `ios-deploy`, or equivalent), none of
which is named here. `SFC-C` is consequently **not** asserted as a solved,
executable surface in this revision; it is specified only to the level of "two
operator-supplied inputs," and this ticket does not claim more than that. See
`### Runner` below and Q3.

### Why the runner is SwiftPM plus simctl rather than xcodebuild

Decided, not left open, because rule 6 forbids an approved spec containing an
unresolved implementation decision.

`swift build --build-tests` plus a repackaging step plus
`simctl spawn <platform xctest agent>` is the runner. Reasons, each measured in
`## Current state and evidence`:

- It works today, end to end, and produced real test output (N10).
- It needs no Xcode project, no workspace, no shared scheme, and no code
  signing, so it adds no tracked file that can drift against `Package.swift`.
- It does not require the Xcode 26 Metal Toolchain component, which the
  xcodebuild route does (N7), and which would add a large CI prerequisite to a
  job whose purpose is to run four seconds of logic tests.
- The tracked `.xcodeproj` shadows the package for bare `xcodebuild` (N7), so
  the xcodebuild route would require either a workspace or a `-project` that
  points at a file this ticket is forbidden to touch.

Cost, stated because it is real: the repackaging step is hand-rolled and
unsupported. SwiftPM emits a macOS-shaped bundle even for a simulator triple. The
mitigation is that the repackaging is a tracked script with a self-test
(`scripts/test-run-characterization.sh`), that its `Info.plist` is a tracked
build input rather than a heredoc inside a pipeline, and that the runner asserts
the bundle it produced actually executed at least one test before reporting
success. A runner that produces zero executed tests is a failure, not a pass.
This is `harness-defect` in `## Failure behavior`.

### Framework choice: XCTest

Decided: **XCTest**, for the whole suite, with a named revisit trigger.

The deployment targets are `Package.swift:20-21`, iOS 17 and macOS 14. Both
frameworks are usable at those targets under Xcode 26, so availability alone does
not decide it. Four things do.

1. **Negative asynchronous assertions are the core of this suite.** D3 requires
   asserting that a stream does **not** terminate within a bounded interval. D8
   requires asserting that an event is **never** delivered. XCTest expresses the
   first directly with `XCTestExpectation.isInverted` plus `XCTWaiter`, which is
   a supported primitive with defined semantics. Swift Testing covers the second
   well with `confirmation(expectedCount: 0)`, but has no first-class inverted
   wait for the first; it must be hand-rolled as a task race against a sleep. A
   hand-rolled timing primitive at the foundation of the project's trust artifact
   is the wrong trade, and the probes in N5 already had to write exactly that
   race to work around its absence.
2. **The subjects are not `Sendable`. This ground is a design judgment, not a
   measured compiler-enforced blocker, and is stated as such.**
   `LlamaInstance` is a non-final class with mutable state
   (`Core Engine/LlamaInstance.swift:18`), `ModelCache` is a class over a
   concurrent `DispatchQueue` (`Service Layer/CacheSettings.swift:53-61`), and
   `Catalyst` is a singleton over a barrier queue. D1 is *about* that sharing,
   and none of the three types conforms to `Sendable`. That much is fact. But
   `Package.swift:1` declares `// swift-tools-version: 5.12`, `swift package
   tools-version` reports `5.12.0`, and no target sets `swiftLanguageMode` or
   any strict-concurrency `swiftSettings` (verified by inspection: `grep -n
   'swiftLanguageMode\|swiftSettings' Package.swift` returns nothing). SwiftPM's
   default for a manifest below tools-version 6.0 is Swift 5 language mode,
   whose default Sendable enforcement is minimal. In that actual mode, a free
   `@Test` function in Swift Testing capturing a non-`Sendable` `LlamaInstance`
   is not forced by the compiler into `@unchecked Sendable` boxing or explicit
   isolation the way Swift 6 mode's strict checking would force it; this ground
   was not probed (no `@Test` function was attempted against these types, and
   none is planned only to settle this argument), so its severity was never
   measured against the language mode that actually governs it. The honest
   version of the claim is narrower than the original: adopting Swift Testing
   here would still cross a semantic line, handing genuinely shared, mutable,
   non-`Sendable` state to a free function whose calling convention assumes
   isolation is either absent or the caller's problem, and that is a real
   coupling-of-test-intent-to-implementation-detail cost worth avoiding. It is
   not, today, a compiler error this package's language mode would raise. The
   revisit trigger below already covers the case where this changes: when
   ODC-0101 moves the runtime to Swift 6 mode, this ground becomes the
   measured, compiler-enforced one its original phrasing assumed it already
   was. `XCTestCase` methods are ordinary methods on a reference type and need
   none of this discussion regardless of language mode, which is why the
   decision does not depend on this ground alone; reason 1 carries it on its
   own.
3. **The manifest is a hazard.** `Package.swift:1` declares
   `swift-tools-version: 5.12`, a version that never shipped (ODC-0002 E6).
   Introducing a second test framework whose integration is gated on
   tools-version semantics, on top of a fictitious tools version, while ODC-0013
   and ODC-0017 are still open, adds a failure mode this ticket cannot
   characterize. XCTest requires no manifest change at all.
4. **The four existing cases are XCTest and their content is itself
   characterized** (N1, N8). Keeping one framework in one bundle keeps the
   runner's "at least N cases executed" arithmetic a single number rather than a
   sum over two reporting formats.

Revisit trigger, named so this is a decision rather than a habit: re-evaluate
when ODC-0101 lands the Swift 6 concurrency and lifecycle model and the runtime
is Sendable-clean. Recorded in `## Ticket allocation`.

### Naming conventions

A reader must be able to tell, from the name alone and without opening the file,
whether an assertion pins a bug or pins correct behavior.

| Kind | Class name suffix | Method prefix | Meaning |
| --- | --- | --- | --- |
| Characterization of a defect | `CharacterizationTests` | `test_characterizes_` | Asserts current, wrong behavior. Must end in `__ODC_00NN`. |
| Characterization of undocumented but not wrong behavior | `CharacterizationTests` | `test_characterizes_` | Must end in `__no_defect`. |
| Regression, correct today and must stay correct | `RegressionTests` | `test_requires_` | Asserts intended behavior. Never carries a defect suffix. |
| Surface canary | `SurfaceTests` | `test_surface_` | Asserts the environment predicate itself. |

Examples:

```
D3LoadProgressCharacterizationTests
  test_characterizes_publishProgressGate_isUnsatisfiable__ODC_0012
  test_characterizes_loadingStream_terminatesWithNoTerminalEvent__ODC_0012
LockfileRegressionTests
  test_requires_resolvedLockfileSatisfiesExactPin
CharacterizationSurfaceTests
  test_surface_reportsSimulatorStubSlice
```

Three rules make this mechanical rather than aspirational, all enforced by
`scripts/check-characterization.py --naming`:

- A method beginning `test_characterizes_` must end with `__ODC_00NN` or
  `__no_defect`, and any ticket it names must exist in `Tickets.md`.
- A method beginning `test_requires_` must **not** contain `__ODC_`, so a
  characterization case cannot be quietly relabelled as a correctness case.
- Every `test_characterizes_` method must carry the four-line block above it.

### Skip protocol

A test that cannot run must say so. It must never pass by doing nothing, and it
must never `return` early.

- Skips use `XCTSkipUnless` or `XCTSkipIf` and nothing else. An early `return`
  inside a test body is a `check-characterization.py --naming` failure.
- Every skip reason begins with a bracketed code from this closed set:
  `SKIP[requires-device]`, `SKIP[requires-model-asset]`,
  `SKIP[requires-simulator-stub]`, `SKIP[requires-real-llama-slice]`,
  `SKIP[requires-metal-device]`. A reason string outside that set is a checker
  failure.
- Each run declares its surface. The runner knows the expected-executed set for
  that surface: on `SFC-B` every `R0`, `R1`, and `R2` case must execute, and
  every `R3` case must skip. A skip inside the declared set fails the run. An
  execution outside it also fails the run, because it means a precondition
  predicate is wrong.
- Each requirement class has a surface canary that asserts the predicate itself.
  If `SimulatorSupport.isSimulator` were to start returning `false` on `SFC-B`,
  the canary fails immediately rather than letting every `R2` case skip into a
  green run.
- The runner prints a skip ledger and `scripts/check-characterization.py
  --skips` parses it. Zero-executed and all-skipped runs both exit non-zero.

The repository already contains two counter-examples,
`Tests/EmbeddingTest.swift` and `Tests/BERTEmbeddingTest.swift` (N9), which skip
on a hard-coded absolute path and would report green forever. They are the reason
this protocol is written down.

### Pinning what cannot be executed

Some behavior cannot be executed on any available surface, and packaging facts
are not behavior at all. Both are pinned by **source fingerprints** rather than
left unasserted.

A fingerprint is a SHA-256 over a normalized extract of a named region of a
source file. The region is located by an **anchor**, a regular expression
matching a declaration, and extends to the matching close brace at the anchor's
indentation. Normalization strips comments, collapses whitespace runs, and
strips trailing whitespace. Line numbers are recorded for human reference only
and are never part of the hash, because unrelated edits above the anchor would
otherwise break every fingerprint at once.

Fingerprints live in `docs/characterization/v2-fingerprints.json`, one entry per
defect site, each carrying the defect id, the repair ticket, the file, the
anchor, the hash, and a human note. `scripts/check-characterization.py
--fingerprints` recomputes and compares. A mismatch fails with the defect id,
the ticket, and the sentence "this defect site changed; if that was deliberate,
update the characterization case and this fingerprint in the same commit".

Fingerprints do not replace behavioral tests where a behavioral test is
possible. Every fingerprint in `## Tests` is paired with at least one executable
case, except where the table says otherwise.

### Disposition of the orphaned files

Decided here, so it stops being ambient.

The three files under `Tests/` outside the declared target path stay exactly
where they are, untouched, uncompiled, in this ticket. They are not deleted, not
moved, and not added to the target.

Reasons: `Tests/test_embedding.swift` executes at file scope and cannot compile
inside a target (N9); the other two encode absolute paths and would skip forever
(N9); and deleting tracked files that ODC-0002 recorded as evidence is a
decision with its own cost, not a cleanup.

What this ticket does instead is pin the fact. `check-characterization.py
--orphans` asserts that exactly those three files exist, that their SHA-256
values are unchanged, and that `swift package describe --type json` lists them in
no target's `sources`. If someone revives one, the check fails and names the
disposition ticket. The delete-or-rewrite decision is allocated in
`## Ticket allocation`.

### Permitted changes to tracked files

The suite is not allowed to change what it measures. The allowed change set is
closed and mechanically checked.

| Path | Allowed | Note |
| --- | --- | --- |
| `Sources/**` | No | Any diff fails A11. |
| `Package.swift` | No | The test target path already covers the new files. |
| `Package.resolved` | No | ODC-0002 owns it. |
| `OnDeviceCatalyst/**`, `OnDeviceCatalyst.xcodeproj/**` | No | ODC-0016 owns the fork. |
| `Tests/OnDeviceCatalystTests/**` | Yes | New characterization sources, plus the N1 repair. Owned exhaustively by this ticket; see `## Interfaces`, Checker, for the boundary against ODC-0003's `Benchmarks/` directory. |
| `Tests/EmbeddingTest.swift`, `Tests/test_embedding.swift`, `Tests/BERTEmbeddingTest.swift` | No | Pinned, see above. |
| `scripts/**` | Yes, additive | Runner, checker, and their self-tests. |
| `docs/characterization/**` | Yes | Fingerprints and the human record. |
| `.github/workflows/apple-build.yml` | Yes, additive | One job. |

The N1 repair is the only edit to an existing test source. It replaces the
reference to the non-existent `PredictionConfig.quality` with assertions over the
five presets that do exist, and it converts the four original cases into named
characterization or regression cases under the conventions above, preserving what
each one was trying to assert. The original intent of
`testPredictionConfigPresets` is preserved as a characterization case, because
what it believed (`quality` exists, and a quality preset allows more tokens than
a speed preset) is false in two separate ways (N1, N8), and both are worth
pinning.

## Interfaces

### File layout

```
Tests/OnDeviceCatalystTests/
  OnDeviceCatalystTests.swift              existing, repaired per N1
  Support/CharacterizationSurface.swift    surface detection and skip helpers
  Support/SyntheticModelFixture.swift      GGUF-magic fixture creation and teardown
  Support/StreamRecorder.swift             bounded collection of AsyncStream events
  R1/CacheCharacterizationTests.swift
  R1/StreamContractCharacterizationTests.swift
  R1/LoadProgressGateCharacterizationTests.swift
  R1/MetalEngineCharacterizationTests.swift
  R1/ValueTypeCharacterizationTests.swift
  R2/InitializationFailureCharacterizationTests.swift
  R3/GenerationCharacterizationTests.swift
  R3/InstanceLifecycleCharacterizationTests.swift
  SurfaceTests.swift
scripts/
  run-characterization.sh
  test-run-characterization.sh
  check-characterization.py
  test-check-characterization.py
docs/characterization/
  v2-fingerprints.json
  v2.0.4-characterization.md
```

### Support API

```
enum CharacterizationSurface {
    case simulatorStub
    case device
    static var current: CharacterizationSurface { get }
    static var hasRealLlamaSlice: Bool { get }
    static var modelAssetPath: String? { get }
    static var hasMetalDevice: Bool { get }
}

extension XCTestCase {
    func requireDevice(file: StaticString, line: UInt) throws
    func requireModelAsset(file: StaticString, line: UInt) throws -> String
    func requireSimulatorStub(file: StaticString, line: UInt) throws
}

struct SyntheticModelFixture {
    static func make(named: String, bytes: Int) throws -> URL   // "GGUF" magic
    static func removeAll()
}

actor StreamRecorder<Element> {
    func record(_ stream: AsyncStream<Element>, timeout: Duration) async -> Recording
    struct Recording { let events: [Element]; let terminated: Bool }
}
```

`CharacterizationSurface.modelAssetPath` reads the environment variable
`ODC_CHARACTERIZATION_MODEL_PATH`, verifies the file exists and has GGUF magic,
and returns nil otherwise. It never searches the filesystem and never embeds a
default path, which is the mistake in `Tests/BERTEmbeddingTest.swift` (N9).

`StreamRecorder.record` is the primitive behind every stream assertion. It
returns both the events and whether the stream terminated within the bound, so
"terminated with no terminal event" (N5) and "never terminated" (D3 success path)
are the same call with different expectations.

### Runner

```
scripts/run-characterization.sh --surface simulator [--device-udid <udid>]
scripts/run-characterization.sh --surface device --destination-id <id>
```

`--surface simulator` creates a temporary simulator device, boots it, builds with
`swift build --build-tests` for `arm64-apple-ios17.0-simulator`, repackages the
bundle, spawns the platform `xctest` agent, writes the raw log to a path outside
the repository, prints the executed and skipped ledger, deletes the temporary
device, and exits non-zero if any expected-executed case did not execute.

`--surface device` requires a signing identity and a destination id supplied by
the operator, never discovered and never written into any tracked file.

**This interface is declared, not designed, and that gap is named rather than
hidden behind two plausible-looking flags.** The two parameters name the
inputs an eventual device-execution path will need, but this ticket does not
specify how a package-only test bundle, built the same way `SFC-B`'s is
(`swift build --build-tests`, no `.xcodeproj` involvement), gets code-signed
with the supplied identity, packaged with a provisioning profile and
entitlements, installed on the device named by `--destination-id`, and
launched there through some device-deployment tool. None of that was measured
by this ticket's discovery, unlike every other design decision in `## Design`.
Until it is, `--surface device` is a documented but unimplemented entry point:
invoking it is expected to fail, loudly, at whichever step of that unspecified
sequence has not yet been built, and that failure is `harness-defect`, not a
silent no-op. See Q3 for the disposition and the obligation this places on
`ODC-0010`, `ODC-0011`, and `ODC-0012`.

### Checker

```
python3 scripts/check-characterization.py --packaging     # R0 packaging facts
python3 scripts/check-characterization.py --fingerprints  # defect-site hashes
python3 scripts/check-characterization.py --naming        # names and comment blocks
python3 scripts/check-characterization.py --orphans       # the three orphaned files
python3 scripts/check-characterization.py --inventory     # catalog matches this spec
python3 scripts/check-characterization.py --skips <log>   # skip ledger audit
python3 scripts/check-characterization.py                 # all of the above
```

Exit 0 means every selected check passed, 1 means at least one failed, 2 means
the checker could not run. This mirrors `scripts/check-baseline.py`.

`--inventory` is the anti-drift check. Every case id in `## Tests` must exist as
a test method or an `R0` check, and every test method must appear in `## Tests`.
A suite that grows without its spec, or a spec that promises cases nobody wrote,
fails here.

**Ownership boundary, stated explicitly so it cannot be reintroduced.**
`--inventory` walks `Tests/OnDeviceCatalystTests/**` in full, with no directory
exclusion, naming-prefix carve-out, or owner-marker mechanism, because none is
needed: **ODC-0004 owns `Tests/OnDeviceCatalystTests/**` entirely.** This was
a blocking finding in review pass two (a collision with ODC-0003's planned
benchmark harness) and it is resolved at the source, not by scoping logic in
this checker. ODC-0003 places no file under `Tests/OnDeviceCatalystTests/`, or
under any other path under `Tests/`; its benchmark harness lives in its own
top-level directory, `Benchmarks/`, as a separate SwiftPM test target with its
own path, reusing this ticket's build-and-deploy *pattern* (`swift build
--build-tests`, repackage into a flat iOS bundle, run through the platform
`xctest` agent) without sharing this ticket's directory. See
[`docs/specs/ODC-0003-benchmark-contract.md`](ODC-0003-benchmark-contract.md),
"Relationship to ODC-0004." With that boundary drawn, a future test method
appearing under `Tests/OnDeviceCatalystTests/**` that this spec's `## Tests`
catalog does not name is unambiguously drift this checker exists to catch, not
a legitimate sibling-ticket addition this checker needs to tolerate.

## Data flow

```
repository at HEAD
  -> check-characterization.py --packaging/--fingerprints/--orphans   (SFC-A)
  -> swift build --build-tests --triple arm64-apple-ios17.0-simulator
  -> repackage into a flat iOS .xctest bundle
  -> simctl spawn <platform xctest agent> -XCTest All
  -> textual XCTest log: "Test Case '...' passed | failed | skipped"
  -> run-characterization.sh parses the log into an executed/skipped ledger
  -> check-characterization.py --skips audits the ledger against the declared
     surface's expected-executed set
  -> exit code
```

The raw log is written outside the repository. It is a CI artifact, not a
tracked file.

What is tracked, and why:

- `docs/characterization/v2-fingerprints.json`. Tracked, because it is the pin.
- `docs/characterization/v2.0.4-characterization.md`. Tracked, because a human
  needs to read what is pinned, what it should become, and which ticket owns it,
  without running anything.
- Run results are **not** tracked. They depend on the operator's surface and
  would rot within a day, and a tracked file that is wrong is worse than no file.
  This is a deliberate divergence from ODC-0002, which tracked its manifest
  because a baseline is a dated measurement while a suite result is a gate.

## Failure behavior

Exhaustive over the failures observed while drafting. A failure that fits no
bucket blocks completion until a bucket is added by spec revision.

| Bucket | Meaning | Blocks |
| --- | --- | --- |
| `characterized-behavior-changed` | A characterization case failed. The pinned behavior moved. | Yes. Resolve by repairing the regression, or by updating the case and its fingerprint in the same commit as the deliberate change, citing the ticket. |
| `regression` | A `test_requires_` case failed. Correct behavior broke. | Yes |
| `surface-unavailable` | A case's requirement class is not satisfied by the declared surface. | No, if the case is outside the expected-executed set. Yes, if inside it. |
| `harness-defect` | The runner built or repackaged wrongly: zero cases executed, bundle would not load, agent not found. | Yes |
| `environment-toolchain-component` | A required Xcode component is absent, for example the Metal Toolchain (N7). | No for `SFC-B`, which does not need it. Yes for any route that does. |
| `flake` | A case whose result varies across three consecutive runs of the same binary on the same surface. | Yes. No retry loops in CI. Quarantine requires a ticket and an expiry date. |
| `mutation` | A run modified a tracked file. | Yes, and must be reverted. |
| `precondition-predicate-wrong` | A surface canary failed, or a case executed outside its expected surface. | Yes |

Two failure modes are called out because they are the ones that make a suite
worthless rather than merely red:

- **Silent pass.** Prevented by the skip protocol and by `--skips`. An
  all-skipped run exits non-zero.
- **Silent drift.** Prevented by `--inventory` and `--fingerprints`. A defect
  site cannot be edited without a named, deliberate update.

## Security and privacy

- **No model weights are committed.** Ever, in any tier. `R3` reads a path from
  `ODC_CHARACTERIZATION_MODEL_PATH`. Licensing and size are both disqualifying,
  and a repository that carries weights invites a benchmark claim that ODC-0003
  owns.
- **No synthetic fixture is committed.** `SyntheticModelFixture` writes into
  `NSTemporaryDirectory()` at run time and removes what it wrote in `tearDown`.
  The fixtures are a few megabytes of zeroes behind a four-byte magic; there is
  nothing to preserve.
- **No absolute home-directory path may appear in any file this ticket adds.**
  `check-characterization.py --naming` applies the ODC-0002 denylist pattern
  `/Users/[^/ "]+` to every added test source, script, and document. The existing
  violation at `Tests/BERTEmbeddingTest.swift:15` is recorded as a finding and
  handed to the disposition ticket; this ticket does not edit that file, and the
  checker exempts exactly those three pinned paths by name so the exemption is
  visible rather than implicit.
- **No device identifier is recorded.** Simulator UDIDs are created and deleted
  inside the runner and never written to a tracked file. Device destinations are
  supplied by the operator through an environment variable. The same denylist
  patterns ODC-0002 applies to its deliverables apply to
  `docs/characterization/**`.
- **No network access at test time.** Every fixture is generated locally. The
  only network access in the pipeline is SwiftPM dependency resolution, which is
  already part of the existing CI job.
- **The runner must not write inside the repository.** Build products go to a
  scratch path, logs go outside the tree, and a `git status --porcelain` check
  before and after the run is part of the runner's own exit condition.

## Migration and compatibility impact

Nothing migrates. This is the section's whole content, and it is stated
positively so that a reviewer can check it.

- **No public API changes.** No file under `Sources/` is touched, so every
  symbol, signature, and behavior a consumer sees is byte-identical.
- **No package graph changes.** `Package.swift` is not edited. The new test
  sources land under the already-declared target path
  `Tests/OnDeviceCatalystTests`, so no target, product, dependency, or platform
  moves. `swift package dump-package` output must be byte-identical before and
  after, and that is acceptance criterion A3.
- **No lockfile change.** `Package.resolved` is ODC-0002's deliverable and this
  ticket does not resolve into a different graph.
- **One test-source change**, the N1 repair, which changes only whether the test
  target compiles. It cannot affect any consumer, because test targets are not
  part of the product.
- **One CI job added.** Additive to `.github/workflows/apple-build.yml`. The
  existing `ios-simulator` job is unchanged, so a failure in the new job is
  attributable.
- **Forward compatibility is the point of the ticket.** When ODC-0010 through
  ODC-0017 land, the corresponding characterization cases will fail. That is
  designed, not accidental, and each failure message carries the replacement
  assertion. Each repair ticket therefore inherits an obligation: update the
  cases this suite pins, in the same commit as the repair. That obligation is
  recorded in `## Ticket allocation` rather than left to memory.

## Tests

This section is the catalog and it is normative.
`scripts/check-characterization.py --inventory` enforces that the implemented
suite and this table agree in both directions.

### Which defects are testable as code, and which are not

| Defect | Executable as a test? | How it is pinned | Surfaces |
| --- | --- | --- | --- |
| D1 | Partly | Reader half executable at `R1` (measured, N6). Writer half needs a ready instance, so `R3`. Fingerprint on `releaseInstance`. | `SFC-B`, `SFC-C` |
| D2 | Partly | Consumer-side consequence executable at `R1` from synthesized chunk sequences. Producer-side duplication needs generation, so `R3`. Fingerprint on both emit sites. | `SFC-B`, `SFC-C` |
| D3 | Partly | The gate is unsatisfiable by construction, so a mirror predicate at `R1` plus a fingerprint. Failure-path termination executable at `R2` (measured, N5). Success-path non-termination is `R3`. | `SFC-B`, `SFC-C` |
| D4 | **No** | Packaging fact. Asserted by `--packaging`: the manifest declares `.macOS(.v14)`, the artifact `Info.plist` has no macOS entry, and `swift build -c debug` on a macOS host exits non-zero with `no such module 'llama'`. Not an XCTest. | `SFC-A` |
| D5 | Partly | Packaging half is `--packaging`: 8 unhandled files, zero declared resources, no `default.metallib` in the SwiftPM product. Consequence half is executable at `R1`: `MetalComputeEngine()` throws. | `SFC-A`, `SFC-B` |
| D6 | **No, and not a characterization** | Already repaired. Pinned as a regression assertion in `--packaging`: the lockfile satisfies the `exact:` pin and a second resolve is byte-stable. | `SFC-A` |
| D7 | **No** | Repository fact. Asserted by `--packaging`: zero `XCRemoteSwiftPackageReference` entries, and the 22 / 12 / 13 / 3 file census. Not an XCTest. | `SFC-A` |
| D8 | Yes | Executable at `R2` (measured, N5), because the stub's null return makes the failure path deterministic without a model. | `SFC-B`, `SFC-C` |
| N1 | **No, and repaired here** | The test target must compile. Asserted by the runner: a build failure is `harness-defect`. | `SFC-B` |

Four of the nine are not code tests at all. Saying so is the point of the table.

### R0, checker assertions, surface `SFC-A`

| ID | Asserts today | Becomes after repair | Ticket |
| --- | --- | --- | --- |
| `C-D4-1` | `Package.swift` declares `.macOS(.v14)` and the XCFramework `Info.plist` lists exactly two `AvailableLibraries`, neither macOS. | A macOS slice exists, or the platform declaration is removed. | ODC-0013 |
| `C-D4-2` | `swift build -c debug` on a macOS host exits non-zero, and the first root failure is `no such module 'llama'` at `API Bridge/LlamaBridge.swift:12`. | Exits 0. | ODC-0013 |
| `C-D5-1` | The simulator build emits `found 8 file(s) which are unhandled`. | Emits none. | ODC-0014 |
| `C-D5-2` | `swift package describe --type json` reports zero `resources` on the `OnDeviceCatalyst` target. | Reports the seven shaders as declared resources. | ODC-0014 |
| `C-D5-3` | No `default.metallib` exists anywhere in the SwiftPM build product. | One exists and is loadable from the module bundle. | ODC-0014 |
| `C-D7-1` | `OnDeviceCatalyst.xcodeproj/project.pbxproj` contains zero `XCRemoteSwiftPackageReference` entries. | The app consumes the package, or the fork is deleted. | ODC-0016 |
| `C-D7-2` | The census is 22 shared, 12 drifted, 13 package-only, 3 app-only, by ODC-0002's normative drift command. | Zero shared files, because one copy is gone. | ODC-0016 |
| `C-E2-1` | `Package.swift:36-37` claims all llama usage is guarded, and `grep -rn '#if !targetEnvironment' Sources/` returns zero matches. | The comment matches the source. | ODC-0017 |
| `C-N4-1` | The simulator slice's minimum iOS version is 26.2 while `Package.swift:20` declares iOS 17. | They agree. | ODC-0017 |
| `C-N2-1` | The `llama_*` symbol set the package references and the set the stub defines are identical, 51 and 51. | The slice is a real build, so it defines a superset. | ODC-0013 |
| `C-N9-1` | Exactly three files sit under `Tests/` outside the target path, with pinned SHA-256 values, and appear in no target's `sources`. | They are deleted or adopted. | disposition ticket |
| `R-D6-1` | `Package.resolved` satisfies `Package.swift`'s `exact: "2.29.3"` pin, and a second `swift package resolve` leaves the file byte-identical. | Unchanged. This is a regression assertion. | ODC-0002 |
| `F-D1-1` | Fingerprint of `Catalyst.releaseInstance`. | Updated with the repair. | ODC-0010 |
| `F-D2-1` | Fingerprint of `LlamaInstance.performGeneration`'s completion emit site. | Updated with the repair. | ODC-0011 |
| `F-D2-2` | Fingerprint of `LlamaInstance.generateTokens`'s five emit sites. | Updated with the repair. | ODC-0011 |
| `F-D3-1` | Fingerprint of `LlamaInstance.publishProgress`. | Updated with the repair. | ODC-0012 |
| `F-D8-1` | Fingerprint of `LlamaInstance.handleInitializationError`. | Updated with the repair. | ODC-0015 |

### R1, no llama return value consumed, surfaces `SFC-B` and `SFC-C`

| ID | Asserts today | Becomes after repair | Ticket |
| --- | --- | --- | --- |
| `C-D1-1` | `ModelCache.getInstance` returns an instance whose `isReady` is `false`. Measured, N6. | Returns nil, or only ready instances. | ODC-0010 |
| `C-D1-2` | `ModelCache.storeInstance` is asynchronous, so an immediate `getInstance` may return nil and a reader must poll within a bounded wait. | The store is ordered with respect to the read. | ODC-0010 |
| `C-D2-1` | Given the chunk sequence v2 emits, a real reason followed by `.natural`, `StreamingResponse` reports `.natural` and the real reason is lost. | Only one completion exists, so the real reason survives. | ODC-0011 |
| `C-D2-2` | `collectResponse()` and `collectContent()` both break on the first completion, so a consumer that breaks sees the real reason and a consumer that drains sees `.natural`. Two consumers of the same stream disagree. | Both see the same reason. | ODC-0011 |
| `C-D3-1` | A predicate transcribed verbatim from `publishProgress`'s gate is `false` for all four `LoadProgress` cases, so no value can satisfy it. | The predicate is true for `.ready` and `.failed`. | ODC-0012 |
| `C-D5-4` | `MetalComputeEngine()` throws, and the error is `CatalystError.unknown` whose message names either the absent Metal device or the absent shader library. | Construction succeeds on a Metal-capable device. | ODC-0014 |
| `X-CFG-1` | `PredictionConfig.balanced.maxTokens` is `-1` and `.speed.maxTokens` is `1024`; the five presets are exactly `balanced`, `creative`, `speed`, `deterministic`, `mirostat`. Measured, N1 and N8. | Unchanged unless a preset changes deliberately. | none |
| `X-ARCH-1` | `ModelArchitecture.detectFromPath("phi-probe.gguf")` returns `.unknown`, while `LlamaBridge.createModelLoadingError` has a dedicated `phi` branch for the same name. Two classifiers disagree. Measured, N5 and N8. | The two agree. | none, until ODC-0104 |
| `X-PROFILE-1` | `ModelProfile(filePath:)` throws `.modelFileNotFound` for a missing path, `.modelFileCorrupted` for a file under 1 MiB, and `.modelFileCorrupted` for bad magic. | Unchanged. | none |
| `X-STOP-1` | `StopSequenceHandler` and `StreamProcessor` emit a `.stopSequenceFound` completion at the byte position they emit it today, for each architecture's default stop set. | Unchanged. | none |
| `X-PROMPT-1` | `StandardPromptFormatter.formatPrompt` produces the exact strings it produces today, per architecture, as golden values. | Unchanged unless a template changes deliberately. | none |
| `X-SETTINGS-1` | `InstanceSettings.iphone16ProMax` is context 2048, batch 256, gpu layers 25, cpu threads 6, and `validate()` accepts it. Preserves the original `testInstanceSettingsValidation`. | Unchanged. | none |
| `X-CATALYST-1` | `Catalyst.shared` is constructible and its initializer prints and completes without a model. Preserves the original `testCatalystServiceInitialization`. | Unchanged. | none |

### R2, llama failure path consumed, surfaces `SFC-B` and `SFC-C`

| ID | Asserts today | Becomes after repair | Ticket |
| --- | --- | --- | --- |
| `C-D8-1` | With a recoverable-class fixture (`phi-*.gguf`), the loading stream yields exactly `preparing`, `loading`, `loading`, then terminates, delivering neither `.ready` nor `.failed`. Measured, N5. | The fallback's own progress events are delivered, ending in `.ready` or `.failed`. | ODC-0015 |
| `C-D8-2` | With a non-recoverable-class fixture, the sequence is identical, so the two error classes are externally indistinguishable. Measured, N5. | The two are distinguishable, and the non-recoverable path delivers `.failed`. | ODC-0015 |
| `C-D8-3` | After the stream ends, `isReady` is `false` and no further event arrives within the bounded wait. | A terminal event precedes termination. | ODC-0015 |
| `C-D3-2` | The failure-path stream terminates because `cleanup()` finished the continuation, not because the gate at `publishProgress` fired; asserted by the absence of any terminal event before termination. | The gate fires and the terminal event precedes termination. | ODC-0012 |
| `C-D4-3` | On `SFC-B`, `LlamaBridge.loadModel` fails for every file, because the slice is a stub. This is the runtime face of D4 and it is asserted with `requireSimulatorStub`. | On a real slice this case skips, which is why it declares `SKIP[requires-simulator-stub]`. | ODC-0013 |

### R3, real inference, surface `SFC-C` with a model asset

Every case here calls `requireDevice` and `requireModelAsset` and emits
`SKIP[requires-device]` or `SKIP[requires-model-asset]` otherwise. On `SFC-B`
all of them must skip, and a skip ledger showing anything else fails the run.

| ID | Asserts today | Becomes after repair | Ticket |
| --- | --- | --- | --- |
| `C-D2-3` | A single bounded generation yields exactly two chunks whose `isComplete` is `true`. | Exactly one. | ODC-0011 |
| `C-D2-4` | The second completion's reason is `.natural` even when the first reported `.maxTokensReached`, so a draining consumer records the wrong reason. | The single completion carries the true reason. | ODC-0011 |
| `C-D3-3` | After `.ready` is delivered, the loading stream does not terminate within the bounded wait, so a consumer awaiting termination hangs. Asserted with an inverted expectation. | The stream finishes after `.ready`. | ODC-0012 |
| `C-D1-3` | After `releaseInstance` drops the last reference, the instance is inserted into the cache and then shut down, so a subsequent cache read returns an instance that is not ready. | The cached instance stays ready, or is not cached. | ODC-0010 |
| `X-GEN-1` | A bounded generation with the model asset produces non-empty content. Canary: if this fails, every other `R3` result is meaningless. | Unchanged. | none |

### Surface canaries, every surface

| ID | Asserts |
| --- | --- |
| `S-1` | `CharacterizationSurface.current` matches the surface the runner declared. |
| `S-2` | On `SFC-B`, `SimulatorSupport.isSimulator` is `true`; on `SFC-C` it is `false`. |
| `S-3` | On `SFC-B`, `LlamaBridge.loadModel` fails for a valid-magic fixture, confirming the stub is in place. |
| `S-4` | `CharacterizationSurface.modelAssetPath` is nil unless the environment variable names an existing file with GGUF magic. |

## Benchmarks

Out of scope, as a decision rather than a gap.

Benchmarks are owned by **ODC-0003**, the cross-backend benchmark contract. This
ticket produces no timing, throughput, memory, or energy claim, and the checker
enforces it: `check-characterization.py --naming` fails on any assertion that
compares a measured duration to a threshold.

The suite does contain time bounds, and they are not benchmarks. Every bounded
wait (`StreamRecorder` timeouts, the cache polling window in `C-D1-2`, the
inverted expectation in `C-D3-3`) is a **liveness bound**, chosen generously,
declared as a named harness constant, and recorded in
`docs/characterization/v2.0.4-characterization.md`. A bound that fires is a
harness or behavior failure, never a performance result, and no bound value may
be quoted as a measurement of the runtime.

The one thing this ticket owes ODC-0003: the runner's build-and-run pipeline for
`SFC-B` and `SFC-C` is reusable, and ODC-0003 may consume
`scripts/run-characterization.sh` rather than reinventing the surface. That is a
dependency in one direction only, and it does not give ODC-0004 any benchmark
obligation.

## Open questions and gates

Recorded rather than assumed away. All three must be closed before this spec
can move past `SPEC_REVIEW`.

**Q1. Is a physical device surface available to the project?** `SFC-C` was not
measured. The `R3` tier (5 cases) is specified but unproven, and GitHub-hosted
runners have no attached devices, so `R3` cannot run in CI under any design.
Deciding command: `xcrun devicectl list devices`, plus a signing identity check.
Two acceptable outcomes, and the ticket is decision-complete under either:

- A device is available. `R3` is executed by the operator before this ticket
  reaches `DONE`, and its ledger is recorded in the characterization document.
- No device is available. `R3` cases are still written, still skip explicitly
  with `SKIP[requires-device]`, and the ticket reaches `DONE` with `R0`, `R1`,
  and `R2` executed. The characterization document records `R3` as
  `specified, unexecuted`, and the five cases become an obligation on ODC-0010,
  ODC-0011, and ODC-0012, each of which must execute them before its own repair
  is accepted.

The second outcome is not a weakening. It is what honesty about a hardware
constraint looks like, and it is strictly better than pretending `R3` ran.

**Q2. Is the Metal Engine actually unreachable for Xcode-built consumers?**
N7 shows Xcode compiles the seven shaders into the resource-bundle target while
`MetalComputeEngine.swift:89` calls `device.makeDefaultLibrary()`, which reads
`Bundle.main`. If that is right, D5's severity is unchanged but its root cause
has two independent halves, and `C-D5-4`'s expected error message differs between
build systems. Not observed here because the host lacks the Metal Toolchain
component. Deciding command: `xcodebuild -downloadComponent MetalToolchain`, then
build the package through Xcode and inspect the product for `default.metallib`
and the bundle it lands in. Owner: **ODC-0014**. Until it is closed, `C-D5-4`
asserts only that construction throws, and does not assert which of the two
messages is produced.

**Q3. What is the device-execution mechanism itself, distinct from device
availability (Q1)?** Not measured by this ticket. `scripts/run-characterization.sh
--surface device --destination-id <id>` declares the two operator-supplied
inputs but does not specify code signing, provisioning, entitlements, or a
device-deployment tool for a package-only test bundle built the same way
`SFC-B`'s is (`swift build --build-tests`, no `.xcodeproj` involvement).
`SFC-B`'s equivalent problem required real, non-obvious engineering (N7, N10);
the device problem was not given the same discovery budget, and this ticket
does not claim otherwise. There is no deciding command for this question,
because a command that decided it would be the design this question says is
missing; that absence is itself the answer, and is why Q3 cannot be closed the
way Q1 and Q2 were, by running one command. Two acceptable outcomes, mirroring
Q1's structure, and the ticket is decision-complete under either:

- The mechanism is measured before this ticket reaches `DONE`: at minimum,
  build for `arm64-apple-ios17.0`, sign with an available identity, install
  and run one sanity `R0`/`R1` case on a physical device, and record what that
  took in `docs/characterization/v2.0.4-characterization.md`, extending
  `--surface device` to match what was actually measured.
- The mechanism is not measured here. `SFC-C` remains `Measured: No` for the
  device-execution path itself, which is a broader claim than Q1's
  "no device was available for `R3`": even a hypothetical available device
  could not run anything today, because the deployment mechanism does not
  exist yet. Whichever of **ODC-0010**, **ODC-0011**, or **ODC-0012** first
  needs to execute an `R3` case inherits the obligation to design and measure
  it, before that ticket's own repair is accepted. Recorded as an explicit
  obligation in `## Ticket allocation`, not assumed solved.

The second outcome is not a weakening, for the same reason Q1's is not: it is
what honesty about an unbuilt mechanism looks like, and it is strictly better
than a runner interface that looks complete and fails opaquely the first time
someone invokes it against real hardware.

## Validation

Each line is a command and the condition on its exit code or output. Where a
command below needs a base revision to diff against, it is `ebea213`
("ODC-0002: execute v2 baseline, allocate finding tickets, correct
lockfile"), the commit at which ODC-0002 reached `DONE`: it is the last commit
to touch `Package.resolved` (D6's own correction) and it predates every commit
in this ticket's own history (`1e63c34` onward). It is therefore the correct
zero-point for "did this ticket change any runtime, manifest, lockfile, fork,
or project file": diffing from `ebea213` isolates exactly what ODC-0004 itself
has touched, without also flagging ODC-0002's own, already-accepted, D6
repair as if this ticket had made it. (The spec's discovery narrative
elsewhere cites `6d72193`, two commits earlier, as the revision the scratch
-tree measurements in `## Current state and evidence` were taken against;
that citation is unaffected and is not this criterion's base, because
`6d72193` predates the D6 lockfile correction and diffing from it would report
ODC-0002's own accepted change as a violation, which is not what A2 is for.)
That literal hash is substituted directly below rather than left as a
placeholder, the same fix ODC-0002's own pass-two review required for the
identical criterion, resolved there with `59da80b`. Verified now: `git diff
--stat ebea213 -- Sources Package.swift Package.resolved OnDeviceCatalyst
OnDeviceCatalyst.xcodeproj` produces empty output at the time of this
revision.

1. `python3 scripts/check-characterization.py` exits 0. This single command
   decides the `R0` assertions, the fingerprints, the naming and comment-block
   rules, the orphan pins, and catalog-to-spec agreement.
2. `scripts/run-characterization.sh --surface simulator` exits 0, and its ledger
   reports every `R0`, `R1`, and `R2` case executed and every `R3` case skipped.
3. `python3 scripts/check-characterization.py --skips <ledger>` exits 0.
4. `python3 scripts/test-check-characterization.py` exits 0. The checker has its
   own tests, following `scripts/test-project-state-validator.py`.
5. `scripts/test-run-characterization.sh` exits 0. The runner has its own test,
   following `scripts/test-check-dco.sh`.
6. `git diff --stat ebea213 -- Sources Package.swift Package.resolved
   OnDeviceCatalyst OnDeviceCatalyst.xcodeproj` produces empty output.
7. `swift package dump-package` output is byte-identical before and after.
8. `python3 scripts/check-baseline.py` still exits 0, so this ticket did not
   invalidate ODC-0002's deliverables.
9. `python3 scripts/validate-project-state.py` exits 0.

### What the repository validators do and do not check

Stated so no reader over-trusts item 9, and so the gap this ticket must cover is
explicit.

`scripts/validate-project-state.py` **does** check: ticket-row shape and cell
count; ticket ID pattern and uniqueness; status membership in the allowed set;
that a ticket in a spec-requiring status carries a Markdown spec link in the
`spec` column; that the
linked file exists, is the spec whose front matter declares that ID, and whose
front-matter `status` equals the ticket's `status`; that specs carry the nine
required front-matter keys; that an approved-state ticket is not
`founder_approved: pending` and has `unresolved_questions: none`; ADR ID pattern,
uniqueness, and status membership; that every ticket dependency exists as a
ticket, except `ODR-` dependencies which are skipped; that every local Markdown
link in root, `docs/`, and `.github/` Markdown resolves on disk; that no em dash
appears in those files; and that `ROADMAP.md` names a current public ticket that
exists.

It **does not** check: anything under `Sources/`, `Tests/`, `scripts/`, or
`docs/characterization/`; whether any test exists, compiles, or runs; spec
section structure or content; whether `evidence_fresh_until` has passed, only
that the key is present; whether a spec's claims are true; ticket-column values
other than ID and status; or `Package.swift` and `Package.resolved`.

`scripts/check-baseline.py` is the baseline-content gate and covers ODC-0002's
deliverables only. It does not read this suite.

Therefore `scripts/check-characterization.py` is the only gate that decides
anything about this ticket's content, and every acceptance criterion below names
either it or the runner.

One expected condition at the time of writing: the `ODC-0004` row in
`Tickets.md` still carries `Spec | TBD` and `Status | BACKLOG`. Because the row
carries no spec link, `validate_specs` skips the status comparison for this
ticket, so the validator passes as it stands. Linking the row and moving it to
`SPEC_DRAFT` are the founder's edits, not this spec's, and they must happen
together: linking the spec while leaving the row `BACKLOG` would fail the
status-agreement check, since this spec declares `status: SPEC_DRAFT`.

## Acceptance criteria

Every criterion is decided by a command's exit code or by empty output. Criteria
that could not be decided this way were deleted rather than softened.

| # | Criterion | Deciding command |
| --- | --- | --- |
| A1 | The test target compiles for the simulator triple | `swift build --build-tests --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" --triple arm64-apple-ios17.0-simulator -c debug` exits 0 |
| A2 | No runtime source, manifest, lockfile, fork, or project file changed since `ebea213`, the commit at which ODC-0002 reached `DONE` and last touched `Package.resolved` (see `## Validation` for why this commit, not the discovery revision `6d72193`, is the correct base) | `git diff --stat ebea213 -- Sources Package.swift Package.resolved OnDeviceCatalyst OnDeviceCatalyst.xcodeproj` is empty |
| A3 | The package graph is unchanged | `swift package dump-package` output before and after is byte-identical |
| A4 | Every `R0` assertion in `## Tests` passes | `python3 scripts/check-characterization.py --packaging` |
| A5 | Every defect-site fingerprint matches | `python3 scripts/check-characterization.py --fingerprints` |
| A6 | Every characterization case carries a conforming name and the four-line block naming a ticket that exists | `python3 scripts/check-characterization.py --naming` |
| A7 | The three orphaned files are unchanged and still compiled by no target | `python3 scripts/check-characterization.py --orphans` |
| A8 | The implemented suite and this spec's catalog agree in both directions | `python3 scripts/check-characterization.py --inventory` |
| A9 | The suite runs on the simulator surface, and every `R0`, `R1`, `R2` case executed | `scripts/run-characterization.sh --surface simulator` |
| A10 | No expected-executed case was skipped, and the run was not vacuous | `python3 scripts/check-characterization.py --skips <ledger>` |
| A11 | All eight baseline defects plus N1 are represented, each by at least one case or checker assertion naming its ticket | `python3 scripts/check-characterization.py --inventory --require-defects D1,D2,D3,D4,D5,D6,D7,D8,N1` |
| A12 | The checker and the runner have passing self-tests | `python3 scripts/test-check-characterization.py && scripts/test-run-characterization.sh` |
| A13 | No secret, home-directory path, or device identifier in any added file | `python3 scripts/check-characterization.py --naming` applies the ODC-0002 denylist |
| A14 | ODC-0002's deliverables are still valid | `python3 scripts/check-baseline.py` |
| A15 | Project state is consistent | `python3 scripts/validate-project-state.py` |
| A16 | The run leaves the working tree unchanged | `git status --porcelain` is empty after `scripts/run-characterization.sh` |
| A17 | Q1 is closed in the characterization document with one of its two enumerated outcomes | `python3 scripts/check-characterization.py --inventory` requires an `r3_disposition` field of `executed` or `specified-unexecuted` |
| A18 | Q3 is closed in the characterization document with one of its two enumerated outcomes | `python3 scripts/check-characterization.py --inventory` requires a `device_execution_disposition` field of `measured` or `unmeasured-deferred` |

Deliberately absent, with reasons:

- **A coverage threshold.** Line coverage is not the goal and a number would
  invite gaming. A11 requires coverage of the named defects instead.
- **"Runtime behavior is unchanged."** ODC-0002 deleted the same criterion for
  the same reason: there is no possible evidence, because no inference runs. A2
  and A3 carry the defensible half.
- **A device-tier pass requirement.** Q1 may legitimately resolve to
  `specified-unexecuted`. A17 requires the disposition to be recorded, not that
  it be favorable.
- **A device-execution-mechanism design requirement.** Q3 may legitimately
  resolve to `unmeasured-deferred`. A18 requires the disposition to be
  recorded and the obligation assigned, not that the mechanism ship in this
  revision.

## Ticket allocation

ODC-0002 reserved `ODC-0010` through `ODC-0049` for tickets created from
baseline evidence, and allocated `ODC-0010` through `ODC-0017`. This ticket's
own discovery identified three further findings needing tickets from the same
range: N1 (the test target does not compile), the disposition of the three
orphaned files (N9), and the Swift Testing revisit trigger (`## Design`,
Framework choice, reason 2). Those three rows, `ODC-0018` through `ODC-0020`,
were added to `Tickets.md` by commit `1e63c34` ("ODC-0004: characterization
suite spec; allocate ODC-0018/0019/0020"), the same commit that drafted the
first version of this spec.

**They are allocated, not proposed, and this revision states that plainly.**
An earlier version of this section described them as "proposals for the
founder" while `Tickets.md` already carried them as real `BACKLOG` rows; pass
two's review (finding 1) correctly identified that as the document and the
ledger disagreeing about the ledger's own state. That disagreement is
corrected here by description, not by reverting the rows: reverting would
erase real findings from the canonical ledger to make a stale sentence true,
which is a worse fix than making the sentence match reality. What remains
true, and is restated because it is the part that still matters, is that
**this document, the spec file itself, edits no file.** The ledger mutation
happened in `Tickets.md`'s own commit history under this ticket, not in this
prose. The founder's review of this revision is accordingly also, implicitly,
a ruling on whether that ordering (ledger before adversarial review) was
acceptable, separate from whether the three rows' content is acceptable.

The table below transcribes the three rows verbatim from `Tickets.md` as they
stand today, so a reader has the rationale next to the ticket, not as a second
copy that could drift: `Tickets.md` remains the single source of truth, and
`scripts/validate-project-state.py` checks the real file, never this table.

| ID | Type | Title | Milestone | Status | Priority | Dependencies | Next Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ODC-0018 | bug | Declared test target does not compile on any triple (`PredictionConfig.quality`) | P0 | BACKLOG | P0 | ODC-0002 | discovery |
| ODC-0019 | decision | Disposition of three orphaned test files outside the target path | P0 | BACKLOG | P2 | ODC-0004 | discovery |
| ODC-0020 | decision | Revisit XCTest versus Swift Testing after the concurrency model lands | P0 | BACKLOG | P2 | ODC-0101 | discovery |

ODC-0018 is unusual in that ODC-0004 repairs it as a precondition. It still got
a row, because the defect was real, it was invisible to the baseline, and a
finding that is fixed in passing and never recorded is a finding that will
recur.

**Reconciliation with the baseline's own defect-to-ticket mapping.** `## B1`
above maps the eight baseline defects to `ODC-0010` through `ODC-0017`
(D6 to `ODC-0002`, which corrected the lockfile). `docs/baselines/v2.0.4.md`
originally carried a different, placeholder mapping for the same defects
(`ODC-0101`, `ODC-0202`, `ODC-0103`, `ODC-0300`), and pass two's review
(finding 7) correctly flagged the two canonical documents disagreeing about
which ticket owns each defect. That divergence no longer exists: commit
`838bdfa` updated `docs/baselines/v2.0.4.md`'s defect table,
`docs/baselines/v2.0.4-environment.json`'s manifest, and
`scripts/render-baseline.py`'s `FINDING_TICKETS` mapping to the same
`ODC-0010`-through-`ODC-0017`/`ODC-0002` allocation this spec's `## B1` table
already used, and `python3 scripts/check-baseline.py` passes against the
reconciled mapping. The two documents agree as of this revision; no further
follow-up ticket is needed for the reconciliation itself.

**Obligations this ticket places on other tickets.** Recorded here because a
characterization suite that nobody is required to update decays into a wall of
red.

| Ticket | Obligation |
| --- | --- |
| ODC-0010 | Update `C-D1-1`, `C-D1-2`, `C-D1-3`, `F-D1-1` in the same commit as the repair. If this is the first of ODC-0010/0011/0012 to execute an `R3` case, also design and measure the device-execution mechanism (Q3) before that repair is accepted. |
| ODC-0011 | Update `C-D2-1` through `C-D2-4`, `F-D2-1`, `F-D2-2`. Inherits the Q3 obligation above if ODC-0010 has not already discharged it. |
| ODC-0012 | Update `C-D3-1`, `C-D3-2`, `C-D3-3`, `F-D3-1`. Inherits the Q3 obligation above if neither ODC-0010 nor ODC-0011 has already discharged it. |
| ODC-0013 | Update `C-D4-1`, `C-D4-2`, `C-D4-3`, `C-N2-1`. Adding a macOS slice also makes `SFC-X` real, at which point the runner gains a macOS surface. |
| ODC-0014 | Update `C-D5-1` through `C-D5-4`, and close Q2. |
| ODC-0015 | Update `C-D8-1`, `C-D8-2`, `C-D8-3`, `F-D8-1`. |
| ODC-0016 | Update `C-D7-1`, `C-D7-2`. If the fork is deleted, the census assertion becomes "zero shared files". |
| ODC-0017 | Update `C-E2-1`, `C-N4-1`. |
| ODC-0003 | May consume `scripts/run-characterization.sh` for its device surface. No obligation in the other direction. ODC-0003 places no file under `Tests/OnDeviceCatalystTests/**`; its harness lives in its own `Benchmarks/` directory, so no scope overlap exists for `--inventory` to police. |

## Alternatives considered

- **Run the suite with `swift test` on the host.** Impossible, not merely
  undesirable. D4 blocks it at `no such module 'llama'` and no amount of test
  design changes that. Rejected before it was designed.
- **Run the suite with `xcodebuild test`.** Measured and rejected. The tracked
  `.xcodeproj` shadows the package, the auto-generated package scheme has no test
  action, and the route requires the Xcode 26 Metal Toolchain component that the
  SwiftPM route does not (N7). It buys `xcresult` output, which the textual
  XCTest protocol makes unnecessary for a suite this size.
- **Add a host application and a unit-test bundle in a new `.xcodeproj`.**
  Rejected. It adds a `project.pbxproj` that must be kept in agreement with
  `Package.swift` by hand, in a repository that already contains one divergent
  fork (D7) caused by exactly that pattern.
- **Wait for a device before writing anything.** Rejected. N2, N3, N5, and N6
  show that three of the four requirement classes and six of the nine defects are
  reachable on the simulator today. Blocking the suite on hardware would leave the
  v3 work with no safety net for months.
- **Write only fingerprints and skip behavioral tests.** Rejected. A fingerprint
  detects that a site changed; it cannot tell you the behavior was preserved or
  that the change was correct. Fingerprints are the fallback for what cannot
  execute, not the design.
- **Fold the three orphaned files into the test target.** Rejected on evidence:
  one executes at file scope and cannot compile in a target, and the other two
  skip on hard-coded absolute paths and would report green forever (N9).
- **Delete the three orphaned files here.** Rejected. Deleting tracked files that
  ODC-0002 recorded as evidence is a decision with its own cost. Pinned and
  allocated instead.
- **Repair D1, D2, D3, or D8 while writing their tests.** Rejected. It destroys
  the artifact. A characterization test written after the fix pins the fix, not
  the behavior v3 must be measured against, and each defect has its own ticket
  and its own review.
- **Use Swift Testing.** Rejected on four grounds in `## Design`, with a named
  revisit trigger and a proposed ticket rather than a silent preference.
- **Track a run-results manifest, as ODC-0002 does.** Rejected. A baseline is a
  dated measurement and a suite result is a gate. A tracked result file depends on
  the operator's surface, rots within a day, and a tracked file that is wrong is
  worse than no file. The fingerprints and the human record are tracked; the run
  log is a CI artifact.
- **Assert a coverage percentage.** Rejected. The target is the nine named
  defects, which A11 decides exactly. A percentage would be satisfiable by
  writing tests for whatever is easiest.

## Review record

- 2026-09-01, drafted against revision `6d72193`, unblocked by ODC-0002 reaching
  `DONE`. Discovery for this draft established N1 through N10 by read-only
  inspection and by builds and test runs performed in a scratch tree outside the
  working tree. No file under `Sources/`, `Tests/`, `Package.swift`, or
  `Package.resolved` was modified; `git status --porcelain` was empty before and
  after.
- Review pass one, completeness: passed, no artifact recorded for this pass.
- 2026-09-02, review pass two, adversarial, verdict REJECT, returned to
  `REVISION`. Artifact:
  [`docs/reviews/ODC-0004-review-pass-2.md`](../reviews/ODC-0004-review-pass-2.md).
  Five blocking findings, seven major, three minor. The review independently
  reproduced nearly every empirical claim in `## Current state and evidence`,
  several to the exact byte or symbol count.
- 2026-09-02, revision: this pass. Blocking findings resolved as follows.
  Finding 1 (`Tickets.md` already carried `ODC-0018` through `ODC-0020` while
  this spec's prose called them proposals): resolved in `## Ticket allocation`,
  which now states plainly that the rows are allocated, names commit `1e63c34`,
  and keeps the honest, narrower claim that this spec document itself edits no
  file. Finding 2 (undefined `<base>` in A2 and Validation item 6): resolved by
  naming a literal, verified base commit, `ebea213`, in `## Validation` and in
  A2's own text. Finding 3 (the `--inventory` versus ODC-0003 scope collision):
  resolved per the manager's ruling that ODC-0004 owns
  `Tests/OnDeviceCatalystTests/**` in full and ODC-0003 moved its harness to a
  separate `Benchmarks/` directory; the boundary is now stated explicitly in
  `## Interfaces`, Checker. Finding 4 (`SFC-C` asserted but not designed to an
  executable level): resolved by adding Q3, marking the device-execution
  mechanism explicitly unmeasured and distinct from device availability (Q1),
  and assigning the obligation to whichever of ODC-0010/0011/0012 executes
  `R3` first. Finding 5 (the Sendable argument for XCTest cited no evidence):
  resolved by stating the package's actual Swift 5 language mode
  (`swift package tools-version` reports `5.12.0`, no `swiftLanguageMode` or
  strict-concurrency settings) and narrowing reason 2 to a design judgment
  rather than a measured compiler-enforced blocker. Majors and minors: finding
  6 (N3's misquoted disassembly) corrected to the actual six-instruction
  sequence, independently re-verified with `otool -tvV` against the tracked
  XCFramework artifact during this revision. Finding 7 (ticket-mapping
  divergence against `docs/baselines/v2.0.4.md`) was already reconciled by
  commit `838bdfa`, which this revision's `## Ticket allocation` now
  cross-references. Finding 8 (`docs/templates/test-spec.md` guidance depth)
  resolved by bringing the template to `docs/templates/baseline-spec.md`'s
  standard, including the "cannot be decided by a command" line under
  Acceptance criteria. Finding 10 (three off-by-one `path:line` citations)
  corrected after re-verifying each against the tracked source. Findings 9, 11,
  and 12 (minor) are not addressed in this revision.
- Founder review: pending. `founder_approved` is `pending` and
  `unresolved_questions` names Q1, Q2, and Q3, so this spec cannot enter an
  approved status until all three are closed.

## Validation evidence

Not implemented. No deliverable in `## Interfaces` exists yet; founder approval
is required before implementation begins.

The evidence that already exists is the discovery evidence in
`## Current state and evidence`, established 2026-09-01 against revision
`6d72193` with the toolchain pinned by ODC-0002. Specifically, and these are the
claims a reviewer should press on first:

- The test target does not compile (N1), reproduced by one command.
- The test bundle links against the stub, shown both by symbol-set equality and
  by an actual successful link (N2).
- The stub returns null without trapping, shown by disassembly (N3).
- The existing four cases executed on an iOS 26.5 simulator, three passing and
  one failing, through the runner design this spec adopts (N10, N8).
- D8's consumer-visible symptom and D1's reader half were reproduced on that
  same surface with no device and no model weights (N5, N6).

`evidence_fresh_until` is 2026-09-15, fourteen days from the drafting date. If
this spec is approved after that date it returns to discovery first, and the
cheapest way to refresh it is to re-run the four commands in N1, N2, N3, and N10.
