---
id: ODC-0100
title: V3 vision and migration
type: architecture
status: SPEC_DRAFT
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
each can start implementation. It does not itself change `Package.swift`,
`Sources/`, `Tests/`, or `Tickets.md`; those are downstream of the tickets this
spec unblocks.

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
  each is an independently resolvable optional product. Each depends on core
  contracts and the execution-policy layer's backend-conformance protocol, and
  on nothing else in this list. None of these four modules depends on any
  other of the four.
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
   |  +--+----------+-- llama.cpp backend (ODC-0200) [optional product]
   |     |
   |     +-------------- MLX backend (ODC-0201) [optional product, physical-device only]
   |
   +---------------------- Metal backend [optional product, gated on ODC-0014 repair]
   +---------------------- Apple system-model backend (ODC-0207) [optional product, 26.0+]
                                ^
                                +-- Apple custom-provider adapter [optional, post-GA, ODC-0023]
```

**Compile-time consequence, stated concretely:** an application that depends
only on core contracts, the execution-policy layer, and the Apple
system-model backend product must resolve and compile with zero references to
the llama.cpp XCFramework or to `mlx-swift`/`mlx-swift-lm`. This is the SwiftPM
target-level test of `docs/ARCHITECTURE.md`'s "heavy backend dependencies do
not resolve for core-only consumers," and ODC-0103's manifest is APPROVED only
if `swift package show-dependencies` for that consumer configuration lists
neither dependency.

## Compatibility policy

ADR-0004 decision point 1 holds the core package's deployment target at iOS 17
/ macOS 14. ADR-0004's own `## Consequences` states the commitment this
creates without softening it: "the iOS 17-25 compatibility promise becomes a
load-bearing product commitment and must be tested, not merely declared."

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
(D4); and replace any direct use of `Catalyst`'s instance cache with the
lifecycle API ODC-0101 defines (D1, D8). None of these changes is optional
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

**What ODC-0021 blocks, named explicitly:** `docs/requirements/memory-and-admission.md`
states R5 "is currently blocked on ODC-0021, which owns establishing that
device surface," and `Tickets.md` records the same block on ODC-0003
("implementation, blocked on ODC-0021 execution surface") and on ODC-0004
("5 R3 cases inert pending ODC-0021"). This spec carries that block forward
without exception: the interfaces this spec assigns to R3, R4, and R5 in `##
Requirement ownership` can be designed and implemented by ODC-0101/ODC-0102
without ODC-0021, but no claim that R3's admission decisions or R5's lifecycle
behavior actually hold on real hardware may be documented until ODC-0021
exists and ODC-0302/ODC-0303 execute against it. Any ticket whose acceptance
criteria require device evidence, not just an implemented interface, is
blocked on ODC-0021 regardless of its position in the dependency chain above.

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
| A11 | Every follow-up ticket this spec names by ID (ODC-0021, ODC-0023, ODC-0101, ODC-0102, ODC-0103, ODC-0104, ODC-0014, ODC-0200, ODC-0201, ODC-0202, ODC-0207, ODC-0300, ODC-0302, ODC-0303) exists in `Tickets.md` | `for t in ODC-0021 ODC-0023 ODC-0101 ODC-0102 ODC-0103 ODC-0104 ODC-0014 ODC-0200 ODC-0201 ODC-0202 ODC-0207 ODC-0300 ODC-0302 ODC-0303; do grep -q "$t" Tickets.md || echo "missing $t"; done` produces no output |

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

## Review record

Not yet reviewed. `status: SPEC_DRAFT` and `founder_approved: pending` above
reflect that this document has not entered `SPEC_REVIEW`. This section is
updated by the review process, not by this draft; no reviewer, decision, or
approval is recorded here by the author.

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
raise one, because `Tickets.md` records `ODC-0100`'s status as `DISCOVERY`,
and `scripts/validate-project-state.py` only requires a linked spec for
tickets whose status is in `{SPEC_DRAFT, SPEC_REVIEW, REVISION, APPROVED,
IMPLEMENTING, VALIDATING, DONE}`; `DISCOVERY` is not in that set. This is
reported as observed rather than adjusted to match the original expectation,
since editing `Tickets.md`'s status or spec-link field for `ODC-0100` is
outside this spec's authority and outside the constraints it was drafted
under.
