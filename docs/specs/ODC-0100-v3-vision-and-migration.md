---
id: ODC-0100
title: V3 vision and migration
type: architecture
status: REVISION
milestone: P1
owner: SankrityaT
dependencies: ODC-0005, docs/requirements/memory-and-admission.md
founder_approved: pending
last_updated: 2026-09-06
evidence_fresh_until: 2026-09-20
unresolved_questions: see Sequencing
---

# ODC-0100: V3 vision and migration

## Summary and user problem

This spec defines the v3 vision, module boundaries, compatibility policy,
requirement ownership, and migration path for OnDeviceCatalyst, per
`docs/decisions/ODC-ADR-0004-apple-api-conformance-over-competition.md`
(governing) and `docs/requirements/memory-and-admission.md` (binding, R1
through R6). It decomposes into the P1 architecture tickets already listed in
`Tickets.md` (ODC-0101 through ODC-0104) and names what must be true before
each can start implementation. It changes no code: `Package.swift`, `Sources/`
and `Tests/` are downstream of the tickets this spec unblocks. Its own ledger row
was advanced and linked by the manager in commit `61aae27`, which is the normal
workflow: specs propose ledger changes and the manager applies them.

The user problem, stated once: an application developer choosing on-device
inference today cannot get a straight answer to "will this model load on this
device, and if it does, what will it cost me in memory and how do I get out of
it cleanly." `docs/baselines/v2.0.4.md` shows the v2 answer is currently wrong
in specific, characterized ways (see `## Migration from v2`). V3 exists to make
that answer honest and mechanically checked rather than declared.

## Vision and non-goals

**Vision, in one paragraph a contributor can repeat:** OnDeviceCatalyst v3 is a
Swift package whose value is the execution-policy layer sitting between an
application's request and whichever on-device backend actually serves it:
device-aware backend selection, memory budgeting and eviction, lifecycle and
backgrounding behavior, cancellation that reaches real work, and performance
reporting that is honest about what device produced it. This is ADR-0004's
decision point 4, made concrete. It is not a wrapper whose purpose is to hide
which backend ran; it is a runtime that decides, admits, measures, and reports
truthfully across whichever backends are compiled in.

**What v3 is not**, per ADR-0004 decision point 2 (`docs/decisions/ODC-ADR-0004-apple-api-conformance-over-competition.md`,
"Decision" item 2):

- V3 is not a unified multi-backend API positioned as the reason to adopt the
  package. `docs/specs/ODC-0005-apple-platform-design-brief.md` (`##
  Architecture and data flow`) is explicit that this exact position is about
  to be occupied by Apple's own `LanguageModel`/`LanguageModelExecutor`
  protocol and by `mlx-swift-lm`'s `MLXFoundationModels` bridge, and that
  contest is not one this project can win on a schedule measured in weeks.
  Catalyst's execution-policy layer is the differentiator instead.
- V3 is not a bespoke abstraction that competes with Apple's provider
  protocol. Per ADR-0004 decision point 3, Apple's protocol is adopted later
  as an optional, additive adapter over an existing Catalyst backend once it
  reaches a stable SDK, never as a core dependency.
- V3 is not built around any preview-only 27.0 API as a requirement source.
  ADR-0004 decision point 5: preview APIs inform planning, they do not define
  acceptance criteria, until they ship in a stable SDK.
- V3 is not a rewrite of the pre-26 compatibility floor. ADR-0004 decision
  point 1 keeps the core at iOS 17 / macOS 14 because that installed base is
  unaddressed by any Apple capability surveyed in ODC-0005 and the gap widens
  with each Apple release, not closes.
- V3 does not specify an allocator, a caching policy, or an eviction
  algorithm as a fixed implementation; `docs/requirements/memory-and-admission.md`
  `## Non-goals` states this directly, and this spec inherits it unchanged.

## Package boundaries

`docs/ARCHITECTURE.md` `## V3 constraints already approved` already commits
this project to: the public core stays independent of any concrete backend;
llama.cpp, MLX, and Apple system-model integration are optional products;
heavy backend dependencies do not resolve for core-only consumers; and backend
C or C++ types never cross the public core API. This section names the module
boundaries that satisfy those constraints and assigns each to a ticket already
present in `Tickets.md`. Concrete package/target names, and the manifest
change itself, are ODC-0103's decision, not this spec's; this section
constrains what ODC-0103 must produce.

**Modules and what each owns:**

- **Core contracts** (owned by ODC-0102): request, response, event, and
  capability types. No backend type, C type, or C++ type appears in this
  module's public surface. This is the only module every consumer depends on.
- **Execution-policy layer** (owned by ODC-0101, with the admission and memory
  surface from this spec's `## Requirement ownership` living here): device-aware
  backend selection, memory budgeting and eviction, lifecycle and
  backgrounding, cancellation propagation, and performance reporting. Depends
  only on core contracts. Depends on zero backend modules; backends register
  into it through the backend-conformance protocol core contracts define, so
  the dependency edge points from backend to core, never core to backend.
- **Model identity and asset lifecycle** (owned by ODC-0104, ODC-0205):
  model identity, verified local files, and ordinary downloads. Depends only
  on core contracts. `docs/ARCHITECTURE.md` already commits that "local model
  files never require the downloader service," so this module's download path
  is additive, not load-bearing for the local-file path.
- **Backend conformance protocol**: a protocol lives in core contracts; each
  backend module below conforms to it from the outside. This is what makes
  the dependency edge point inward.
- **llama.cpp backend** (ODC-0200), **MLX backend** (ODC-0201), **Metal
  backend** (hardware-gated within itself per `docs/specs/ODC-0005-apple-platform-design-brief.md`
  `## Architecture and data flow`, repaired by ODC-0014 before it is a v3
  product at all), and **Apple system-model backend** (ODC-0207, 26.0+ only):
  each is an independently resolvable optional product, gated behind its own
  SwiftPM trait (per "Dependency-graph optionality" below), disabled by
  default. Each depends on core contracts and the execution-policy layer's
  backend-conformance protocol, and on nothing else in this list. None of
  these four modules depends on any other of the four.
- **Apple custom-provider adapter** (post-GA, not started before iOS 27
  general availability per ADR-0004 point 5, tracked under ODC-0023): an
  additive adapter over one existing backend module, reachable through Apple's
  session API. It depends on the backend it adapts and on Apple's provider
  protocol; no backend module or the execution-policy layer depends on it.
- **Background Assets integration** (ODC-0206): depends only on the model
  identity and asset lifecycle module. No backend module and no
  execution-policy module depends on it, per ODC-0005's `## Architecture and
  data flow` ("Background Assets' place: entirely outside the execution-policy
  layer and the backend list").
- **Benchmark contract** (ODC-0003): depends on core contracts and the
  execution-policy layer's reporting surface (to read peak-memory and basis
  fields for R6). No backend module or the execution-policy layer depends on
  the benchmark module.

**Dependency graph, in text:**

```
Core contracts (ODC-0102)
   ^  ^  ^  ^  ^  ^
   |  |  |  |  |  |
   |  |  |  |  |  +-- Model identity & asset lifecycle (ODC-0104, ODC-0205)
   |  |  |  |  |          ^
   |  |  |  |  |          +-- Background Assets integration (ODC-0206)
   |  |  |  |  |
   |  |  |  |  +-- Benchmark contract (ODC-0003)
   |  |  |  |
   |  |  |  +-- Execution-policy layer (ODC-0101)
   |  |  |          ^
   |  |  |          | (backend-conformance protocol, edge points inward)
   |  |  |          |
   |  +--+----------+-- llama.cpp backend (ODC-0200) [trait LlamaCPPBackend, off by default]
   |     |
   |     +-------------- MLX backend (ODC-0201) [trait MLXBackend, off by default, physical-device only]
   |
   +---------------------- Metal backend [optional product, gated on ODC-0014 repair]
   +---------------------- Apple system-model backend (ODC-0207) [optional product, 26.0+]
                                ^
                                +-- Apple custom-provider adapter [optional, post-GA, ODC-0023]
```

**Dependency-graph optionality is not automatic, and this document commits to
the mechanism that delivers it.** `swift package show-dependencies` reports
the package graph resolved from a manifest's `dependencies:` array; that
resolution happens once per manifest, independent of which product a
downstream consumer selects. Selecting a library product changes what `swift
build` compiles and links; it does not change what SwiftPM resolves and
reports. Today's manifest confirms this is not hypothetical: `Package.swift:1`
declares `// swift-tools-version: 5.12`, and `Package.swift:28-31` declares
`mlx-swift-lm` as an unconditional dependency of the single existing target.
A consumer who selected only a hypothetical core-only product from that same
manifest, unchanged, would still see `mlx-swift-lm` (and, once added,
llama.cpp) in `swift package show-dependencies`, because both are resolved for
the whole manifest, not per selected product. Stating optionality as an
already-secured fact of "optional products" alone, as an earlier draft of this
section did, is false under SwiftPM's real resolution semantics, and this
document does not repeat that claim.

This document commits ODC-0103 to closing the gap with **SwiftPM Package
Traits (SE-0450, "Package traits", status Implemented, Swift 6.1)**, not to a
multi-package split and not to weakening the claim to a build/link-only check.

**Precision required here, because the mechanism is narrower than "traits make
dependencies optional" suggests.** SwiftPM has two distinct steps, and traits
affect only the second:

1. **Version pinning**, which produces `Package.resolved`. SE-0450's own Future
   Directions section states the implementation considers traits only *after*
   this step. Pruning at the fetch and pin level is explicitly not yet
   implemented; SwiftPM carries it behind an off-by-default
   `--experimental-prune-unused-dependencies` flag. **A trait-disabled
   dependency is still fetched and still appears in `Package.resolved`.**
2. **Module-graph construction**, which is what `swift package show-dependencies`
   reads. This step *is* trait-filtered: a package reachable only through
   trait-guarded target-dependency edges that are disabled never enters the
   graph.

The gate below depends on step 2 and is therefore sound. Any claim that a
trait-disabled dependency is not fetched, or is absent from `Package.resolved`,
would depend on step 1 and would be false today. Do not make it. The reasoning for that choice is in `##
Alternatives considered`; the consequence for ODC-0103 is:

- ODC-0103 raises the manifest's declared tools version from `5.12` to `6.1`.
  **The consumer cost is concrete and must be stated in release notes:** Swift
  6.1 shipped 2025-03-31 in Xcode 16.3, so a consumer on Xcode 16.2 or earlier
  cannot resolve the manifest at all. This is a toolchain floor, entirely
  separate from the iOS 17 / macOS 14 deployment floor, which is unchanged.
  Note also that `5.12` is not a shipped Swift release; SwiftPM's own
  `ToolsVersion` constants jump from `.v5_10` to `.v6_0`, so the current value
  is itself invalid and the migration is a correction, not only an upgrade
  (the first tools version with trait support), and declares a named trait per
  heavy backend, for example `LlamaCPPBackend` and `MLXBackend`, with the
  llama.cpp binary target's dependency edge and the `mlx-swift-lm` package
  dependency each conditioned on its trait. No trait is enabled by default, so
  a consumer who declares no trait requirement gets neither dependency
  resolved.
- **The tools-version floor this imposes is a toolchain requirement, not a
  deployment-target requirement, and the two must not be conflated.** A
  manifest's `// swift-tools-version:` line states the minimum SwiftPM/Xcode
  toolchain a consumer must use to resolve and build the manifest at all,
  regardless of target OS. The manifest's separate `platforms:` array (`##
  Compatibility policy`) states the minimum OS version the compiled binary can
  run on. Raising tools-version to `6.1` means every consumer, including one
  who selects only core contracts, must resolve this package with a Swift 6.1
  or later toolchain; it does not raise, lower, or otherwise touch the iOS
  17 / macOS 14 deployment floor, which remains declared in `platforms:`
  exactly as `## Compatibility policy` states. A consumer building with a
  Swift 6.1 toolchain can still ship a binary whose minimum OS is iOS 17;
  those are independent axes. This document does not name the specific first
  Xcode release shipping Swift tools version 6.1, since verifying that number
  requires a toolchain check outside this document's read-only, no-build
  drafting constraints; ODC-0103's own spec must state it before ODC-0103 is
  approved.
- **The revised, checkable gate:** ODC-0103's manifest is APPROVED only if,
  for a trait selection that enables no backend trait (what a consumer gets by
  declaring no trait preference), `swift package show-dependencies` lists
  neither the llama.cpp binary target's backing reference nor `mlx-swift-lm`,
  and, separately, enabling only `LlamaCPPBackend` or only `MLXBackend` causes
  exactly that one dependency to appear. This gate is conditioned on trait
  selection, which SwiftPM filters at the module-graph level, rather than on
  product selection alone, which it does not filter at all. It is testable once
  ODC-0103's manifest declares the traits, using the real flags
  `--disable-default-traits`, `--traits <comma-list>` and `--enable-all-traits`.
  These are accepted by `show-dependencies` through the shared global option
  group even though they do not appear in that subcommand's own `--help`, so
  ODC-0103 must verify the flags behave as expected on the toolchain in use
  rather than assuming them from the help text.

  **This gate does not and must not assert anything about `Package.resolved`.**
  Trait-disabled dependencies are still fetched and pinned. See the two-step
  distinction above.

## Compatibility policy

ADR-0004 decision point 1 holds the core package's deployment target at iOS 17
/ macOS 14. ADR-0004's own `## Consequences` states the commitment this
creates without softening it: "the iOS 17-25 compatibility promise becomes a
load-bearing product commitment and must be tested, not merely declared."

**This is a deployment-target floor, not a tools-version floor, and `##
Package boundaries`'s Package Traits adoption changes only the latter.** The
`platforms:` array in `Package.swift` states the deployment floor named above;
the `// swift-tools-version:` line states the SwiftPM/Xcode toolchain a
consumer must resolve the manifest with. ODC-0103 raising tools-version to
`6.1` to gain trait support changes which toolchain can open this package at
all; it does not raise the iOS 17 / macOS 14 deployment floor stated here, and
this document treats the two as independent commitments throughout.

**What the declared floor commits the project to:**

- The core module and the execution-policy layer must compile and link on
  every declared platform (iOS/iPadOS 17-26, macOS 14-26, per
  `docs/ARCHITECTURE.md` `## V3 constraints already approved`) with no backend
  compiled in, since backends are optional products.
- No core-module type or API may require an SDK feature above the declared
  floor. `docs/specs/ODC-0005-apple-platform-design-brief.md` `##
  Compatibility and migration` already establishes the model this inherits:
  every Apple capability surveyed there is strictly additive, gated at 26.0,
  26.4, or 27.0-preview, and never lowers or raises the core floor.
- Optional backends with a higher floor than iOS 17 / macOS 14 (the Apple
  system-model backend at 26.0+, and, later, the custom-provider adapter at
  27.0+) declare their own higher platform requirement on their own SwiftPM
  target only. A consumer who does not select that product target never
  observes the higher floor, consistent with the additive framing ODC-0005
  already established.

**What must be tested to make the floor real rather than declared:** the
declared-versus-compiling gap is not hypothetical. `docs/baselines/v2.0.4.md`
`## Build matrix` measured the v2.0.4 macOS cell as `fails`, exit code 1, with
root cause "`Package.swift:21` declares `.macOS(.v14)`, but the XCFramework
exposes only `ios-arm64` and `ios-arm64-simulator`... `import llama` cannot
resolve on any macOS triple" (mapped to D4/ODC-0013). The same report records
`macos-test` as `blocked-by-build`, meaning "zero test signal" was produced
for the declared macOS floor. This is the standing warning this policy exists
to answer: a platform line in `Package.swift` is a claim, not evidence.

Concretely, ODC-0302 (Apple-platform lifecycle matrix) and ODC-0303 (device and
model compatibility matrix) must each produce, per platform in the declared
matrix:

- A cold-cache `compiles` or equivalent result for every declared triple, with
  no cell recorded as `fails` or `blocked-by-build`, mirroring the four
  enumerated result values `docs/baselines/v2.0.4.md` `## Build matrix`
  already uses.
- At least one executed test per platform, so that a `blocked-by-build`
  outcome (zero test signal, as macOS was in v2.0.4) cannot recur silently.
- R5's lifecycle tests (memory-pressure response, backgrounding,
  foregrounding, cancellation mid-load, repeated load/unload) run on a real
  device surface before any lifecycle claim is documented, per
  `docs/requirements/memory-and-admission.md` R5. This is currently blocked on
  ODC-0021, named explicitly in R5's text and in `## Sequencing` below.

**How optional backends with higher floors are handled:** each optional
backend module declares its own platform floor on its own target
(`docs/ARCHITECTURE.md`: "V3 may wrap that system model as an optional
Catalyst backend while preserving its own older-OS API"). The core's iOS 17 /
macOS 14 floor is never raised to accommodate a backend's higher requirement;
instead the backend module is simply unavailable to a consumer building below
its floor, and ODC-0103's package manifest must make that unavailability a
compile-time fact (the product does not resolve) rather than a runtime
availability check a consumer could miss.

## Requirement ownership

Per `docs/requirements/memory-and-admission.md`, R1 through R6 are binding.
Each row below names the module from `## Package boundaries` that owns the
requirement and the interface that satisfies it. A requirement with no owning
module is not architected; none of the six is left without one here.

| Req | Owning module | Interface satisfying it |
| --- | --- | --- |
| R1 | Execution-policy layer (ODC-0101), surfaced through Core contracts (ODC-0102) | A memory-figure value type in Core contracts that carries a mandatory `basis` field (platform interface and field consulted) alongside every measured number; no constructor exists that omits it. Enforced mechanically by ODC-0003's manifest schema, per R1's own "How it is checked" row. |
| R2 | Execution-policy layer (ODC-0101) reporting surface, cross-checked by the Benchmark contract (ODC-0003) | A derived, non-authored comparability flag computed from two figures' `basis` fields, never accepted as an input. `docs/requirements/memory-and-admission.md` R2 names this precedent directly: "See ODC-0003, which applies the same rule to the benchmark contract and derives the comparability flag rather than accepting it as authored." Where bases differ, the interface returns both figures side by side, never a single reduced number. |
| R3 | Execution-policy layer (ODC-0101) admission decision logic, consulting an artifact owned by Model identity and asset lifecycle (ODC-0104, ODC-0205) | An admission API in Core contracts that decides load-or-refuse from a per-device-class, per-model compatibility artifact, not from a single runtime number. A platform availability signal (for example `SystemLanguageModel.availability`, owned by the Apple system-model backend, ODC-0207) may be one input to this API; the interface must not expose that signal alone as a complete memory-position account, per R3's second paragraph. |
| R4 | Execution-policy layer (ODC-0101) | A typed admission-failure error in Core contracts, raised before any backend allocation call, naming the model, the device class, the basis used, and the measured figure that caused refusal. Silent degradation, partial load, and load-then-crash are all excluded by construction: the admission API in R3's interface returns this error type or a success value, with no third silent outcome. |
| R5 | Execution-policy layer (ODC-0101) implementation; validated on the device surface owned by ODC-0021 and executed under ODC-0302 | A test suite, not a claim: per-lifecycle-transition tests (memory-pressure response, backgrounding, foregrounding, cancellation mid-load, repeated load/unload) that must run on a real device surface before the corresponding behavior is documented. `docs/requirements/memory-and-admission.md` states R5 "is currently blocked on ODC-0021, which owns establishing that device surface"; this spec carries that block forward unchanged rather than declaring R5 satisfied by design alone. |
| R6 | Execution-policy layer (ODC-0101) reporting surface; enforced by the Benchmark contract (ODC-0003) | A performance-report value type in Core contracts that cannot be constructed without an accompanying peak-memory figure on a declared basis, from the same run as the throughput or latency figure. `docs/requirements/memory-and-admission.md` names ODC-0003's acceptance criteria as the enforcement point for R6 directly. |

R5 is the one requirement this spec cannot mark architecturally complete: its
owning module's implementation can be designed now, but the requirement itself
is defined as untested until ODC-0021 exists, and this spec does not weaken
that definition to make the row look done.

**Internal seams inside the execution-policy layer:** `## Package boundaries`
names the layer's five responsibilities (device-aware backend selection,
memory budgeting and eviction, lifecycle and backgrounding, cancellation
propagation, performance reporting) and the table above routes all six
requirements through the single ticket ODC-0101. That is correct ownership but
not yet an architecture ODC-0101 can be scoped against; this subsection draws
the seam. The layer is not one undifferentiated type. It decomposes into four
named sub-components, each a distinct actor or protocol conformance inside the
ODC-0101 target, and each owning a disjoint subset of R1-R6:

- **Backend Selector** (`BackendSelecting`): device-aware backend selection.
  Given a device class and a set of compiled-in, trait-enabled backends, it
  narrows the candidate set before admission is consulted. It owns no
  requirement row directly; it is the Admission Controller's first input.
- **Admission Controller** (`AdmissionDeciding`): owns R3 and R4. Consults the
  Backend Selector's candidate set and the compatibility artifact
  (ODC-0104/ODC-0205) to decide load-or-refuse, and raises the typed
  admission-failure error before any backend allocation call. This is the
  data-shape-heavy half of the layer the review asked to be separated out.
- **Memory Reporter** (`MemoryReporting`): owns R1, R2, and R6. Attaches a
  mandatory `basis` field to every measured figure, derives the comparability
  flag from two figures' bases rather than accepting it as authored, and
  refuses to construct a performance-report value without an accompanying
  peak-memory figure. This is the other half of the data-shape-heavy concern,
  separate from the Admission Controller because it has no load-or-refuse
  decision to make, only figures to attach and compare.
- **Lifecycle Controller** (`LifecycleManaging`): owns R5 and the memory
  budgeting/eviction and cancellation-propagation responsibilities named in
  `## Package boundaries`. It owns actor-scoped instance state (the D1/D8
  replacement for v2's unsafe cache), backgrounding and foregrounding
  transitions, cancellation that reaches backend work, and repeated
  load/unload cycles. This is the behavior-heavy concern the review asked to
  be separated from the data-shape-heavy pair above; it is validated on the
  device surface ODC-0021 owns, per R5's row.

The Backend Selector and Admission Controller both consult data the Memory
Reporter produces (a memory-figure value type with a declared basis), so the
dependency edge runs Memory Reporter to Admission Controller to Backend
Selector, never the reverse; the Lifecycle Controller depends on the Admission
Controller's decision (nothing loads without admission) but nothing depends on
the Lifecycle Controller. Whether these four become four Swift types, four
actors, or four protocol conformances on fewer types is ODC-0101's own
implementation decision; this document's commitment is the seam and the
requirement-to-component mapping, not the concrete type count.

## Migration from v2

`docs/baselines/v2.0.4.md` `## Characterized findings` records eight defects,
D1 through D8, none fixed by that ticket. Each is mapped to a follow-up
ticket; this section states what v3's package boundaries mean for a v2
consumer given that each defect exists today.

**The eight defects, and what they mean for migration:**

- **D1** (ODC-0010, mapped to ODC-0101): `releaseInstance` caches a ready
  instance and asynchronously shuts it down with no happens-before against the
  cache read (`Catalyst.swift:495-522`, `LlamaInstance.swift` not an actor).
  This is exactly the "eviction done wrong today" example `docs/specs/ODC-0005-apple-platform-design-brief.md`
  `## Lifecycle, concurrency, and cancellation` cites. V3's execution-policy
  layer replaces the cache/shutdown mechanism entirely with actor-owned
  lifecycle state; a v2 consumer relying on `Catalyst`'s instance cache
  directly must migrate to the new lifecycle API once ODC-0101 defines it.
- **D2** (ODC-0011, mapped to ODC-0202): `performGeneration` emits two
  terminal completions per generation. A v2 consumer that breaks on the first
  completion (as the in-repo consumer at `Catalyst.swift:468` does) is
  depending on undefined stream behavior. V3's stream contract (ODC-0202) has
  exactly one documented terminal event, per `docs/ARCHITECTURE.md`'s already
  approved constraint; a v2 consumer must stop assuming a second completion
  event is safe to ignore, because it will not be sent.
- **D3** (ODC-0012, mapped to ODC-0202): `publishProgress`'s success gate is
  unsatisfiable, so the loading stream's continuation is only ever finished by
  `cleanup()`, never by success. This is the concrete failure mode
  `docs/ARCHITECTURE.md`'s "cancellation reaches underlying work" constraint
  exists to prevent. A v2 consumer awaiting a terminal `.ready` progress event
  on the success path is awaiting an event v2 never sends; v3's stream
  contract must send it.
- **D4** (ODC-0013, mapped to ODC-0103): `.macOS(.v14)` is declared with no
  macOS slice in the llama XCFramework, so `swift build` and `swift test` both
  fail with "no such module 'llama'" on macOS. This is the standing warning
  named in this spec's `## Compatibility policy`: a v2 consumer targeting
  macOS was never actually able to build against the llama backend, regardless
  of what `Package.swift` declared. V3's optional-product packaging (`##
  Package boundaries`) makes this failure mode structural rather than
  incidental: a backend's own target declares only the platforms it actually
  supports.
- **D5** (ODC-0014, mapped to ODC-0103): eight unhandled files (seven `.metal`
  shaders under `Metal Engine/Shaders/` plus `Assets.xcassets`) are never
  declared as target resources, so `makeDefaultLibrary()` returns nil and the
  entire Metal Engine subtree is unreachable when the package is consumed as a
  package. A v2 consumer selecting `InstanceSettings.backendType == .metal`
  today gets a throw, not a working Metal backend, no matter what the API
  surface promises. V3's Metal backend module (`## Package boundaries`)
  carries this forward as a hard precondition: ODC-0014 must package the
  shaders as target resources before Metal is a v3 product at all, not merely
  document the limitation.
- **D6** (ODC-0002, lockfile-manifest disagreement): `Package.resolved` at the
  pinned revision pins `mlx-swift-lm` to `branch: main`, which cannot satisfy
  the manifest's `exact: "2.29.3"` requirement, so every resolve rewrites the
  lockfile. This is a v2 packaging-hygiene defect with no v3 API surface
  consequence beyond ODC-0103 committing a resolver-stable `Package.resolved`
  as part of its own deliverable; it is not itself a migration item for a v2
  consumer's code.
- **D7** (ODC-0016, mapped to ODC-0300): `OnDeviceCatalyst/` at the repository
  root is a divergent fork of the runtime; `OnDeviceCatalyst.xcodeproj/project.pbxproj`
  contains zero `XCRemoteSwiftPackageReference` entries, and 12 of 22
  same-named shared files have drifted from the package's sources. **This
  demo app is dropped, not migrated.** It is not a v3 consumer in any sense
  the package boundaries in this spec define, because it never depended on
  the package to begin with. ODC-0300 (maintained package-consuming sample
  app) replaces it as a net-new artifact that depends on the package the way
  every other consumer must.
- **D8** (ODC-0015, mapped to ODC-0101): `handleInitializationError` calls
  `cleanup()`, which finishes and nils `loadingContinuation`, before
  `attemptFallbackInitialization` runs, so every `publishProgress` call on the
  fallback path is a silent no-op. A v2 consumer relying on fallback-path
  progress events is relying on events that are never delivered. V3's
  execution-policy layer owns lifecycle sequencing (`## Requirement
  ownership`, R4) precisely so that a fallback path cannot silently swallow
  its own progress reporting.

**The public API that does not do what it says:** beyond the eight defects,
`docs/baselines/v2.0.4.md` `## The manifest comment at Package.swift:36-37 is
false` records that the manifest's own comment claims an `arm64-simulator`
stub slice guarded by `#if !targetEnvironment(simulator)`, and `grep -rn
'#if !targetEnvironment' Sources/` returns zero matches; `LlamaBridge.swift`
and `LlamaCppBackend.swift` both `import llama` and call the C API
unconditionally. A v2 consumer reading that comment and concluding the
simulator target is safe to run llama on is relying on a false claim in a
tracked build input. V3 does not carry this comment forward in any form; the
optional-product boundary in `## Package boundaries` makes simulator
reachability a property `swift package show-dependencies` and the build
matrix in `## Compatibility policy` can check mechanically, not a claim in a
source comment.

**The v2 public API surface beyond the eight characterized defects:** D1
through D8 do not cover all of v2's public surface, and a v2 consumer using
any of the following needs a named destination, not silence:

- **Tool calling** (`CatalystTool`, `CatalystToolCall`, `ToolCallParser`,
  `ToolPromptFormatter`, `Sources/OnDeviceCatalyst/Tools/ToolSupport.swift`):
  destination is **ODC-0203** (structured generation and tools), already
  listed in `Tickets.md` depending on ODC-0202. A v2 consumer parsing tool
  calls from raw model output migrates to ODC-0203's structured tool-call
  contract once it lands; this spec does not restate ODC-0203's own interface,
  only its ownership.
- **Session/state persistence** (`StatePersistence`,
  `Sources/OnDeviceCatalyst/Core Engine/StatePersistence.swift`) and **content
  safety** (`SafetyManager`,
  `Sources/OnDeviceCatalyst/Core Foundation/SafetyManager.swift`): neither has
  an owning ticket in `Tickets.md` today. This spec proposes one new ledger
  row for the manager to add: `ODC-0208 | runtime | Session state persistence
  and safety guards | P2 | BACKLOG | P2 | ODC-0101, ODC-0102 | TBD | TBD |
  unassigned | 2026-09-06 | dependency approval`, depending on ODC-0101 (the
  Lifecycle Controller owns the actor-scoped state a persisted snapshot must
  be taken from and restored into) and ODC-0102 (the value types a snapshot
  serializes). `SafetyManager`'s memory-percentage heuristic
  (`isMemoryUsageSafe()`) is additionally superseded in function, not just
  relocated: its role is subsumed by R1/R2's basis-tracked memory reporting
  and R3/R4's admission decision, both owned by ODC-0101/ODC-0102 already, so
  ODC-0208's safety-guard scope is what remains after that subsumption
  (request-shape guards, not memory-admission logic).
- **Model download identity** (`CatalystModel`, `ModelDownloader`,
  `Sources/OnDeviceCatalyst/Service Layer/ModelDownloader.swift`): destination
  is **Model identity and asset lifecycle (ODC-0104, ODC-0205)**, already
  named as the owning module in `## Package boundaries` above but not
  previously connected to this concrete v2 type pair. A v2 consumer calling
  `ModelDownloader.shared.ensure(_:)` migrates to ODC-0205's asset-lifecycle
  API; `CatalystModel`'s enum-of-known-models shape does not carry forward
  as-is, consistent with `## Migration from v2`'s general rule that no v2 type
  signature survives unchanged.
- **Embeddings**: `ODC-0204` (single and batch embeddings) already owns this
  surface per `Tickets.md`, depending on ODC-0200/ODC-0201; the two embedding
  presets in `CatalystModel` (`nomicEmbedV1_5`, `gteQwen2_1_5B`) migrate to
  whatever model-identity mechanism ODC-0204 and ODC-0104/ODC-0205 define
  together, not to `CatalystModel` itself.

**What carries forward, concretely:** the four-layer shape in
`docs/ARCHITECTURE.md`'s current diagram (application, facade, backend
instances, supporting services) survives as a concept: a facade-like Core
contracts layer, backend instances rebuilt as independently optional modules,
and supporting services (chat/prompt formatting, tool-call parsing, settings)
redistributed into Core contracts or the execution-policy layer per `##
Package boundaries`. No v2 type signature carries forward as-is: `Catalyst`'s
direct instance cache, `LlamaInstance` as a plain class, and the
`InstanceSettings.backendType` enum-driven dispatch are all superseded by the
actor-owned lifecycle state and backend-conformance protocol this spec and
ODC-0101/ODC-0102 define.

**What a v2 consumer must concretely change:** stop depending on
`OnDeviceCatalyst/` (the demo-app fork, dropped per D7); stop assuming a
second terminal completion is safe to ignore (D2) or that a success-path
`.ready` progress event ever arrives (D3); stop selecting `.metal` as a
backend until ODC-0014 lands (D5); stop building against `.macOS(.v14)` with
the llama backend selected until ODC-0103's per-backend platform floors land
(D4); replace any direct use of `Catalyst`'s instance cache with the
lifecycle API ODC-0101 defines (D1, D8); replace direct `ToolSupport.swift`
parsing with ODC-0203's structured contract once it lands; replace
`StatePersistence`/`SafetyManager` usage with ODC-0208's successor once
allocated; and replace `ModelDownloader`/`CatalystModel` usage with
ODC-0104/ODC-0205's asset-lifecycle API. None of these changes is optional
for a consumer that wants the behavior the v2 API surface currently claims but
does not provide.

## Deliberately not built

ADR-0004 names one item directly: **a bespoke, Catalyst-owned unified
multi-backend API positioned as the reason to adopt the package** (decision
point 2). `docs/specs/ODC-0005-apple-platform-design-brief.md` `##
Architecture and data flow` restates this as the one thing the brief
"affirmatively rules out": investing further in unified-backend positioning,
on the `LanguageModel`/`LanguageModelExecutor` evidence that this position is
about to be occupied by Apple itself on a schedule measured in weeks. This
spec carries that decision forward unchanged.

The baseline and the Apple brief imply several more, each with its own
reason:

- **A fixed allocator, caching policy, or eviction algorithm specified at the
  architecture level.** `docs/requirements/memory-and-admission.md` `##
  Non-goals` states this directly: the requirements document "does not
  specify an allocator, a caching policy, or an eviction strategy." Building
  one now would fix a policy decision ahead of the measured evidence R3
  requires it to be based on.
- **A single boolean or single platform-provided number as the sole admission
  signal.** R3's own text prohibits this: "A platform availability signal may
  inform the decision. It may not be the sole input." The v2 baseline's D1
  (`docs/baselines/v2.0.4.md`) is a concrete instance of a lifecycle decision
  made without adequate evidence, and this spec does not repeat that shape at
  the admission layer.
- **Adoption of Apple's custom-provider protocol before it reaches a stable
  SDK.** ADR-0004 decision point 5 is explicit that preview APIs may inform
  planning but must not define acceptance criteria; the adapter module in
  `## Package boundaries` is scoped as not-started work, tracked under
  ODC-0023, until iOS 27 general availability.
- **A cross-backend performance ranking or leaderboard that reduces figures
  with different bases to one number.** R2 prohibits presenting a cross-backend
  comparison unless both figures share a basis, and requires side-by-side
  reporting otherwise. Building a ranking feature ahead of that mechanical
  check would recreate the exact ambiguity R1 and R2 exist to close.
- **Making Background Assets a core dependency.** `docs/specs/ODC-0005-apple-platform-design-brief.md`
  `## Security, privacy, and licensing` records that Background Assets'
  Apple-hosted mode ties asset publication to an App Store Connect
  relationship; `## Architecture and data flow` places it "entirely outside
  the execution-policy layer and the backend list," and `##
  Package boundaries` above keeps it an optional, ODC-0206-owned integration
  for exactly this reason.
- **Carrying forward or repairing the `OnDeviceCatalyst/` demo-app fork
  in place.** D7 (`docs/baselines/v2.0.4.md`) documents it as a divergent
  fork with zero package references and 12 of 22 drifted shared files. V3
  does not attempt to reconcile that fork with the package; ODC-0300 builds a
  net-new sample app that depends on the package instead, per `## Migration
  from v2` above.
- **A restated or load-bearing claim of the false `Package.swift:36-37`
  simulator comment.** `docs/baselines/v2.0.4.md`'s finding on that comment is
  carried forward only as a defect to remove, never as a design input; v3
  does not build any simulator-safety claim that is not backed by the
  mechanical build-matrix check in `## Compatibility policy`.

## Sequencing

`Tickets.md` already lists ODC-0101 through ODC-0104 with `Next Gate:
dependency approval`, naming this spec (ODC-0100) as the blocking dependency.
This section states the order those tickets must proceed in and what else
must be true first, without asserting this spec's own review outcome.

**Decomposition, in dependency order:**

1. **ODC-0101 (Swift 6 concurrency and lifecycle)**, depends on ODC-0100.
   Implements the execution-policy layer's actor-owned lifecycle state that
   replaces D1 and D8's cache/shutdown and fallback-progress defects.
2. **ODC-0102 (public inference contracts)**, depends on ODC-0100 and
   ODC-0101. Implements the Core contracts module: request/response/event/
   capability types, the backend-conformance protocol, and the memory-figure,
   admission-error, and performance-report interfaces named in `##
   Requirement ownership`.
3. **ODC-0103 (modular package graph)**, depends on ODC-0100 and ODC-0102.
   Implements the manifest-level package boundaries in `## Package
   boundaries`: per-backend optional products, the D4/D6 packaging fixes, and
   the `swift package show-dependencies` check this spec names as ODC-0103's
   approval gate.
4. **ODC-0104 (model identity and ownership)**, depends on ODC-0102. Owns the
   compatibility artifact R3's admission interface consults.
5. **ODC-0200 (llama.cpp backend)** and **ODC-0201 (MLX backend)**, each
   depend on ODC-0103 and ODC-0104. Each closes its own D4/D5-class packaging
   defect as a precondition of being a v3 product, per `## Migration from v2`.
6. **ODC-0202 (sessions, streams, cancellation, and context)**, depends on
   ODC-0200 and ODC-0201. Closes D2 and D3's stream-protocol defects with the
   single-terminal-event contract `docs/ARCHITECTURE.md` already approved.
7. **ODC-0014 (Metal packaging repair)** must land, independently of the
   chain above, before the Metal backend module in `## Package boundaries`
   can be treated as a v3 product; it is not gated by ODC-0100 through
   ODC-0104 and can proceed in parallel.
8. **ODC-0207 (Apple system-model backend)** and, later, the custom-provider
   adapter under **ODC-0023** (blocked on iOS 27 general availability, per
   ADR-0004 point 5) follow once ODC-0102's backend-conformance protocol
   exists; neither is on the critical path for ODC-0101 through ODC-0202.
9. **ODC-0203 (structured generation and tools)** and **ODC-0204 (single and
   batch embeddings)**, depending on ODC-0202 and on ODC-0200/ODC-0201
   respectively per `Tickets.md`, own the tool-calling and embeddings surface
   named in `## Migration from v2`'s v2-surface-beyond-D1-D8 subsection. Both
   follow ODC-0202, consistent with `Tickets.md`'s own dependency column.
10. **ODC-0208 (session state persistence and safety guards)**, proposed in
    `## Migration from v2` and not yet a `Tickets.md` row, would depend on
    ODC-0101 and ODC-0102 and follow both, for the same reason ODC-0203 and
    ODC-0204 follow ODC-0202: it consumes interfaces those tickets define
    rather than defining new ones of its own.

**What ODC-0021 blocks, named explicitly:** `docs/requirements/memory-and-admission.md`
states R5 "is currently blocked on ODC-0021, which owns establishing that
device surface," and `Tickets.md` records the same block on ODC-0003
("implementation, blocked on ODC-0021 execution surface") and on ODC-0004
("5 R3 cases inert pending ODC-0021" -- ODC-0004's own device-execution tier,
also labeled `R3` in that document's vocabulary, distinct from this document's
requirement `R3` above; the two share a label by coincidence of both documents
independently numbering their own tiers, not by shared meaning). This spec
carries that block forward without exception: the interfaces this spec assigns
to requirement R3, R4, and R5 in `## Requirement ownership` can be designed
and implemented by ODC-0101/ODC-0102 without ODC-0021, but no claim that
requirement R3's admission decisions or R5's lifecycle behavior actually hold
on real hardware may be documented until ODC-0021 exists and ODC-0302/ODC-0303
execute against it. Any ticket whose acceptance criteria require device
evidence, not just an implemented interface, is blocked on ODC-0021 regardless
of its position in the dependency chain above.

**The contingency if ODC-0021 never delivers a device surface:** this is not
hypothetical risk-listing; it is a named consequence. If ODC-0021 cannot
establish a real-device execution surface at all (for example if signing,
provisioning, or physical hardware access proves permanently unavailable to
this project), then requirement R5 and the device-evidence half of
requirement R3 remain permanently unsatisfied, not merely delayed, and that
fact must be disclosed in any release built on this architecture rather than
silently omitted. Concretely: the Lifecycle Controller and Admission
Controller (see `## Requirement ownership`'s internal-seams subsection) can
still be implemented, unit-tested on the simulator surface `docs/specs/ODC-0004-v2-characterization-suite.md`
already uses, and shipped, because their interfaces do not require ODC-0021 to
exist. What cannot happen is any documentation, release note, or compatibility
claim stating that lifecycle behavior or admission decisions have been
verified on real hardware; that claim converts from "not yet true" to
"knowingly false" the moment it is written down after ODC-0021 is confirmed
unobtainable. This spec's architecture does not change if ODC-0021 fails: the
interfaces still satisfy R1 through R6 as designed. Only the R5 and R3
device-evidence claims permanently downgrade from "blocked, pending" to
"unsatisfiable, disclosed," and ODC-0304 (release and support policy) is the
ticket that must carry that disclosure into any release built before or
without a resolution to ODC-0021.

**What must be true before implementation starts, generally:** a ticket in
the ODC-0101 through ODC-0104 set may leave `BACKLOG` for implementation only
once this spec has been reviewed and reaches an approved state through the
program's own workflow (`Tickets.md` `## Status definitions`); this spec does
not itself declare that outcome.

## Acceptance criteria

Every criterion is decided by a command's exit code or output, per the
program's evidence rule. No criterion asserts a specific literal value for
this document's own `status` field, since the review workflow is what changes
that field. No criterion diffs a path owned by a sibling ticket; the diff
criterion below is scoped to exactly the paths this spec's own constraints
name as off-limits to it.

| # | Criterion | Deciding command |
| --- | --- | --- |
| A1 | The document exists at the required path | `test -f docs/specs/ODC-0100-v3-vision-and-migration.md` |
| A2 | Front matter declares a status in the workflow vocabulary, and `founder_approved` is `pending` unless the status is `APPROVED` or `DONE` | `python3 -c "import re,sys;t=open('docs/specs/ODC-0100-v3-vision-and-migration.md').read();st=re.search(r'^status: (\S+)',t,re.M).group(1);fa=re.search(r'^founder_approved: (\S+)',t,re.M).group(1);ok=st in {'BACKLOG','DISCOVERY','SPEC_DRAFT','SPEC_REVIEW','REVISION','APPROVED','IMPLEMENTING','VALIDATING','DONE','BLOCKED','DEFERRED','REJECTED'} and (fa=='pending' or st in {'APPROVED','DONE'});sys.exit(0 if ok else 1)"` |
| A3 | No em dash character appears anywhere in the document | `! grep -qP '\xe2\x80\x94' docs/specs/ODC-0100-v3-vision-and-migration.md` |
| A4 | Every required section from this ticket's scope is present | `grep -qE '^## Vision and non-goals' docs/specs/ODC-0100-v3-vision-and-migration.md && grep -qE '^## Package boundaries' docs/specs/ODC-0100-v3-vision-and-migration.md && grep -qE '^## Compatibility policy' docs/specs/ODC-0100-v3-vision-and-migration.md && grep -qE '^## Requirement ownership' docs/specs/ODC-0100-v3-vision-and-migration.md && grep -qE '^## Migration from v2' docs/specs/ODC-0100-v3-vision-and-migration.md && grep -qE '^## Deliberately not built' docs/specs/ODC-0100-v3-vision-and-migration.md && grep -qE '^## Sequencing' docs/specs/ODC-0100-v3-vision-and-migration.md` |
| A5 | Each of R1 through R6 has an owning-module row in the requirement ownership table | `for r in R1 R2 R3 R4 R5 R6; do grep -qE "^\| $r \|" docs/specs/ODC-0100-v3-vision-and-migration.md || echo "missing $r"; done` produces no output |
| A6 | Each of the eight characterized v2 defects (D1 through D8) is addressed in the migration section | `for d in D1 D2 D3 D4 D5 D6 D7 D8; do grep -q "\*\*$d\*\*" docs/specs/ODC-0100-v3-vision-and-migration.md || echo "missing $d"; done` produces no output |
| A7 | ADR-0004's named deliberately-not-built item is stated explicitly, not merely implied | `grep -qi 'unified multi-backend API' docs/specs/ODC-0100-v3-vision-and-migration.md` |
| A8 | Package boundaries name every module and cite the constraint each satisfies | `grep -qE '^## Package boundaries' docs/specs/ODC-0100-v3-vision-and-migration.md && grep -qi 'show-dependencies' docs/specs/ODC-0100-v3-vision-and-migration.md` |
| A9 | This ticket modifies no file outside its own spec, scoped only to the paths this ticket's own constraints name as off-limits (sibling-owned paths under `docs/specs/` for other tickets are deliberately excluded from this check, since another ticket's own spec work is not this ticket's concern) | `git diff --stat -- Sources Tests Package.swift Package.resolved Tickets.md ROADMAP.md docs/requirements .github` produces empty output |
| A10 | Project state remains internally consistent | `python3 scripts/validate-project-state.py` exits 0 |
| A11 | Every follow-up ticket this spec names by ID as an existing `Tickets.md` row (ODC-0021, ODC-0023, ODC-0101, ODC-0102, ODC-0103, ODC-0104, ODC-0014, ODC-0200, ODC-0201, ODC-0202, ODC-0203, ODC-0204, ODC-0207, ODC-0300, ODC-0302, ODC-0303, ODC-0304) exists in `Tickets.md`. ODC-0208 is deliberately excluded: this spec proposes it as a new row rather than asserting it already exists. | `for t in ODC-0021 ODC-0023 ODC-0101 ODC-0102 ODC-0103 ODC-0104 ODC-0014 ODC-0200 ODC-0201 ODC-0202 ODC-0203 ODC-0204 ODC-0207 ODC-0300 ODC-0302 ODC-0303 ODC-0304; do grep -q "$t" Tickets.md || echo "missing $t"; done` produces no output |

## Alternatives considered

- **Keep a unified multi-backend abstraction as the v3 centerpiece.** Rejected
  per ADR-0004 decision point 2: Apple's own `LanguageModel`/
  `LanguageModelExecutor` protocol and `mlx-swift-lm`'s `MLXFoundationModels`
  bridge are about to occupy that exact position, on a schedule measured in
  weeks from ADR-0004's 2026-09-01 evidence date.
- **Raise the core deployment floor to iOS 26 to reach Apple's on-device APIs
  directly.** Rejected per ADR-0004 decision point 1: the pre-26 installed
  base is unaddressed by every Apple capability ODC-0005 surveys, and that gap
  widens, not closes, with each Apple release. The floor stays at iOS 17 /
  macOS 14 and Apple capabilities are reached through optional higher-floor
  backend modules instead (`## Package boundaries`).
- **Ship one monolithic package rather than per-backend optional products.**
  Rejected: it would violate `docs/ARCHITECTURE.md`'s already-approved
  constraint that heavy backend dependencies do not resolve for core-only
  consumers, and would make D4's macOS/llama packaging failure
  (`docs/baselines/v2.0.4.md`) a structural certainty rather than a fixable
  per-target platform floor.
- **Treat a platform availability boolean (for example
  `SystemLanguageModel.availability`) as the complete admission decision.**
  Rejected directly by R3's text (`docs/requirements/memory-and-admission.md`):
  a platform signal may inform admission, it may not be the sole input.
- **Repair the `OnDeviceCatalyst/` demo-app fork in place rather than
  replacing it.** Rejected: D7 records 12 of 22 shared files already drifted
  and zero package references in the Xcode project; reconciling a fork this
  divergent costs more than building ODC-0300's net-new sample app against
  the package's actual public surface.
- **Defer this spec until iOS 27 general availability, to avoid a second
  rewrite.** Rejected: ODC-0101 through ODC-0104 are already blocked on this
  spec in `Tickets.md`, and ADR-0004 decision point 5 already isolates the
  27.0-preview risk into the optional, not-yet-started adapter module and the
  separately scheduled ODC-0023 recheck, so the core architecture here does
  not depend on iOS 27's final shape.
- **Split heavy backends into separate packages or repositories, each with its
  own manifest, instead of adopting Package Traits inside one manifest.**
  Rejected for cost, not correctness: a genuine multi-package split delivers
  real dependency-graph optionality too, but at the cost of multi-repo release
  coordination (versioning llama.cpp and MLX backends against a core contracts
  package released on its own schedule) that this project's single-maintainer
  scale does not need yet, and it would contradict the "OnDeviceCatalyst v3 is
  a Swift package" framing in `## Vision and non-goals`. Package Traits solve
  the identical resolution-level problem inside the one-package shape this
  document's vision already commits to.
- **Weaken the optionality claim to what a single-package, multiple-product
  manifest can deliver today without adopting Traits (zero heavy-backend code
  in the linked binary for a core-only consumer, checked at build/link time
  instead of resolution time), and drop the `show-dependencies` gate
  entirely.** Rejected: `docs/ARCHITECTURE.md`'s already-approved constraint
  is that "heavy backend dependencies do not resolve for core-only consumers"
  (`docs/ARCHITECTURE.md:41`), stated in terms of resolution, not linking.
  Dropping to a build/link-only check would silently narrow an
  already-approved constraint rather than satisfy it, and would leave
  `Package.resolved` listing llama.cpp and mlx-swift-lm for every consumer
  regardless of need, which is the exact ambiguity a developer evaluating
  "will this dependency show up in my build" would still hit. Traits keep the
  stronger, already-approved, resolution-level claim true instead of retreating
  from it.

## Review record

This document has been reviewed once: `docs/reviews/ODC-0100-review-pass-2.md`
(adversarial, dated 2026-09-06), verdict REJECT, returned to `REVISION`. This
revision responds to that review's two blocking findings and four major
findings: the SwiftPM optionality claim (`## Package boundaries`, "Dependency-
graph optionality"), the execution-policy layer's internal seams (`##
Requirement ownership`, "Internal seams inside the execution-policy layer"),
the `R3` naming collision (`## Sequencing`, "What ODC-0021 blocks"), the
migration gaps for tool calling, persistence, safety, and model-download
identity (`## Migration from v2`, "The v2 public API surface beyond the eight
characterized defects"), and the ODC-0021 contingency (`## Sequencing`, "The
contingency if ODC-0021 never delivers a device surface"). The `Tickets.md`
ledger-claim finding from that same review was corrected separately, described
in `## Summary and user problem`. `status: REVISION` and `founder_approved:
pending` above reflect that this revision has not yet re-entered
`SPEC_REVIEW`. This section is updated again by the next review pass, not by
this draft; no reviewer, decision, or approval beyond the record above is
made by the author of this revision.

## Validation evidence

All eleven acceptance criteria (A1 through A11) were run against this
document at draft time and passed:

- A1 through A8, A11: each deciding command exited 0 / produced no output,
  confirming the file exists, front matter is well-formed, no em dash is
  present, every required section header is present, every R1-R6 row and
  every D1-D8 marker is present, the named unified-multi-backend-API phrase
  is present, the package-boundaries section names `show-dependencies`, and
  every named follow-up ticket ID exists in `Tickets.md`.
- A9: `git diff --stat -- Sources Tests Package.swift Package.resolved
  Tickets.md ROADMAP.md docs/requirements .github` produced empty output at
  draft time, confirming this ticket touched none of the paths its own
  constraints place off-limits.
- A10: `python3 scripts/validate-project-state.py` printed
  `project state valid: 40 tickets, 8 specs, 4 ADRs` and exited 0.

One expectation from the task that authorized this draft did not hold: a
missing-spec-link error for `ODC-0100` was anticipated because `Tickets.md`'s
`ODC-0100` row still carries `TBD` in its `Spec` column. The validator did not
raise one, because at drafting time `Tickets.md` recorded `ODC-0100`'s status
as `DISCOVERY`, and `scripts/validate-project-state.py` then only required a
linked spec for statuses in `{SPEC_DRAFT, SPEC_REVIEW, REVISION, APPROVED,
IMPLEMENTING, VALIDATING, DONE}`.

**That gap has since been closed.** The validator now requires any spec present
on disk to be linked from its ticket row regardless of status, and additionally
forbids a spec from asserting that it does not or cannot change `Tickets.md`,
because four pass-two reviews found that exact claim false. Both checks carry
negative fixtures in `scripts/test-project-state-validator.py`.

**Re-run for this revision (2026-09-06, responding to `docs/reviews/ODC-0100-review-pass-2.md`):**
all eleven criteria were re-run against the revised document and passed:

- A1 through A8: unchanged in mechanism, still exit 0 / produce no output
  after the "Dependency-graph optionality," "Internal seams inside the
  execution-policy layer," and "The v2 public API surface beyond the eight
  characterized defects" subsections were added; the required section headers
  and R1-R6/D1-D8 markers are unaffected by those additions.
- A9: `git diff --stat -- Sources Tests Package.swift Package.resolved
  Tickets.md ROADMAP.md docs/requirements .github` still produces empty
  output; every edit in this revision is confined to this spec file.
- A10: `python3 scripts/validate-project-state.py` still prints
  `project state valid: 40 tickets, 8 specs, 4 ADRs` and exits 0.
- A11: the criterion's own ticket list was extended to include ODC-0203,
  ODC-0204, and ODC-0304, all three newly named in this revision's migration
  and sequencing additions; `ODC-0208` (the persistence/safety ticket this
  revision proposes) is deliberately excluded from A11's list because it does
  not yet exist as a `Tickets.md` row, and this document does not claim
  otherwise. The re-run command produced no output.
