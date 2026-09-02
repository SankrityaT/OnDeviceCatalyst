---
id: ODC-0005
title: Apple platform capability brief for v3 architecture
type: design
status: REVISION
milestone: P0
owner: unassigned
dependencies: ODC-0002
founder_approved: pending
last_updated: 2026-09-02
evidence_fresh_until: 2026-09-16
unresolved_questions: Q1 iOS 27 general availability date and the final shape of LanguageModel/LanguageModelExecutor at GA are not yet known, Q2 Metal 4's chip-level hardware floor is not confirmed by primary Apple documentation fetched in this pass, Q3 Foundation Models' numeric context-window limit is not published by Apple, Q4 Background Assets automatic eviction and integrity-metadata behavior are not confirmed by primary Apple documentation fetched in this pass
---

# ODC-0005: Apple platform capability brief for v3 architecture

## Title correction

The ledger title for this ticket, "WWDC25 Apple-native design brief," is stale
and this spec does not use it. It was written when the program's fixed
assumptions treated iOS 26 as the newest shipping system and explicitly
excluded iOS 27 from product requirements. `docs/decisions/ODC-ADR-0004-apple-api-conformance-over-competition.md`
supersedes that assumption: it records that Apple's own custom-provider
protocol is already visible in beta, that the pre-26 installed base remains
permanently unaddressed by Apple, and that the program's differentiator moves
to the execution-policy layer. A brief anchored to a single named WWDC year
would go stale on the same schedule the ADR just corrected for. This document
is the corrected brief; ADR-0004 is its authority and this spec implements it
rather than contradicts it.

Proposed corrected title, for the ledger owner to apply: **"Apple platform
capability brief for v3 architecture."** It names what the document is
(a capability brief that feeds v3 module boundaries) instead of the event that
produced the first draft of the underlying facts, so it does not need
correcting again at the next WWDC. The ticket's own 14-day evidence-freshness
rule, not its title, is what keeps the content current.

## Summary and user problem

A contributor starting v3 work needs one place that states, with dated primary
citations, what Apple's shipping and preview platform APIs actually offer, what
they deliberately withhold, and what that split implies for OnDeviceCatalyst's
module boundaries. Without it, contributors either re-derive availability facts
ad hoc (and get them wrong, as the stale title itself demonstrates) or treat
Apple's roadmap as a reason to abandon the project, which ADR-0004 already
rejected as the wrong conclusion.

This brief answers two questions for a contributor, in order: first, given what
`SystemLanguageModel` and `LanguageModelSession` ship today, what problem does a
third-party on-device runtime still solve; second, given ADR-0004's decision to
hold the deployment target at iOS 17 / macOS 14 and move the differentiator to
execution policy, which Apple capabilities become optional backends, which
become adapters, and which are deliberately not pursued.

Every availability claim below was independently fetched from Apple's own DocC
JSON data endpoints (`developer.apple.com/tutorials/data/documentation/...`) on
2026-09-02, the date of this draft, not summarized from a secondary source. A
claim that could not be confirmed that way is marked `UNVERIFIED` and is never
used to define an acceptance criterion anywhere in this program, per ADR-0004
and the program's own evidence rule.

## Goals

- State what `SystemLanguageModel` and `LanguageModelSession` expose today
  (iOS/iPadOS/macOS/visionOS 26.0, shipping, non-beta) and what they
  deliberately do not expose, so the absences, not the presences, motivate a
  third-party runtime.
- State the custom-provider protocol's exact preview status, per ADR-0004's
  constraint: it may inform planning, it may not define acceptance criteria
  until it ships in a stable SDK. Design the adapter boundary so adopting it
  later is additive.
- State what the current Swift package's MLX integration supports and why MLX
  must stay an optional product, never a core dependency, consistent with
  `docs/ARCHITECTURE.md`'s already-approved v3 constraints.
- State Metal's availability precisely, without inferring hardware-universal
  capability from the newest silicon.
- State Background Assets' delivery model precisely enough for ODC-0206 to
  scope from, while stating that the core package must work with plain local
  files and ordinary downloads regardless of Background Assets' presence.
- State the deployment-target decision's cost and its evidence obligation, so
  "iOS 17 compatible" stays a tested claim rather than a declared one.
- Translate all of the above into concrete v3 module-boundary consequences:
  optional backend, optional adapter, or explicitly not pursued.
- Ground every claim in the measured v2.0.4 baseline where the baseline is the
  relevant evidence (the package does not build on macOS, the simulator slice
  is a stub, the Metal shaders are unpackaged for SwiftPM consumers).

## Non-goals

- This is not an implementation spec. No file under `Sources/`, `Tests/`,
  `Package.swift`, or `Package.resolved` is touched by this ticket, and no
  concrete v3 package name or public type is decided here; that remains
  blocked on the ODC-0100 series per `docs/ARCHITECTURE.md`.
- This is not a competitive analysis. It states what Apple's public APIs do and
  do not offer, for architectural planning, not a gap analysis against any
  third party or against confidential research.
- This does not adopt any 27.0-introduced API as a product requirement. Every
  27.0 fact here is scoped explicitly as preview and non-normative, per
  ADR-0004 point 5.
- This does not decide ODC-0206 (Background Assets integration) or ODC-0207
  (the optional Apple system-model backend). It supplies the factual and
  architectural grounding those tickets depend on; it does not write their
  interfaces.
- This does not re-litigate ADR-0004's deployment-target decision. It explains
  the decision's cost and evidence obligation; changing the decision requires
  a new ADR.
- This does not repair any v2 defect. D1 through D8 and their tickets
  (ODC-0010 through ODC-0017) are out of scope here, as they were for ODC-0002
  and ODC-0004.

## Current behavior and evidence

### 1. Foundation Models: what ships, what it deliberately withholds

`FoundationModels` is a shipping, non-beta module on iOS, iPadOS, macOS,
visionOS, and Mac Catalyst starting at version 26.0. It is not available on
watchOS or tvOS at 26.0.
[developer.apple.com/documentation/foundationmodels](https://developer.apple.com/documentation/foundationmodels)
DocC data, accessed 2026-09-02.

`SystemLanguageModel` is the on-device model handle, shipping (not beta) at
26.0 on the same platform set. Its `availability` property exposes an
`Availability` enum with `.available` and `.unavailable(UnavailableReason)`
cases; the confirmed reasons in Apple's own usage sample are
`.deviceNotEligible` and `.modelNotReady`, plus an open case for other reasons
the framework may add. Apple's own documentation states the model is versioned
independently of the OS: as of this evidence date it names three model
generations aligned to OS releases, 26.0 through 26.3, 26.4, and 27.0, so
availability and model identity are two separate, both-checkable facts, not
one.
[developer.apple.com/tutorials/data/documentation/foundationmodels/systemlanguagemodel.json](https://developer.apple.com/tutorials/data/documentation/foundationmodels/systemlanguagemodel.json),
accessed 2026-09-02.

`LanguageModelSession` is shipping (not beta) at 26.0 on iOS, iPadOS, macOS,
visionOS, and Mac Catalyst; watchOS support was added later and is `beta:
true, introducedAt: 27.0`. A session records interactions in a `Transcript`,
supports both single and streaming responses (`streamResponse`, described by
Apple as partial-snapshot streaming rather than token-delta streaming), and
raises `contextSizeExceeded` when a session's context is full.
[developer.apple.com/tutorials/data/documentation/foundationmodels/languagemodelsession.json](https://developer.apple.com/tutorials/data/documentation/foundationmodels/languagemodelsession.json),
accessed 2026-09-02.

Guided generation is shipping (not beta) at 26.0. `GenerationSchema` "guide[s]
the output of a language model to deterministically ensure the output is in
the desired format," constructible from a static property list, a string
enumeration, or, for runtime-built schemas, `DynamicGenerationSchema`. The
`@Generable` and `@Guide` macros attach schema information to Swift types
directly.
[developer.apple.com/tutorials/data/documentation/foundationmodels/generationschema.json](https://developer.apple.com/tutorials/data/documentation/foundationmodels/generationschema.json),
accessed 2026-09-02.

Tool calling is shipping (not beta) at 26.0 on iOS, iPadOS, macOS, visionOS,
and Mac Catalyst (`beta: true, introducedAt: 27.0` on watchOS only). The `Tool`
protocol lets a model "gather information at runtime or perform side effects,"
with the framework injecting the tool's name and description into the prompt
and letting the model decide when and how often to call it; tool definitions
consume context-window budget because they are prompt content, not an
out-of-band channel.
[developer.apple.com/tutorials/data/documentation/foundationmodels/tool.json](https://developer.apple.com/tutorials/data/documentation/foundationmodels/tool.json),
accessed 2026-09-02.

`SystemLanguageModel.tokenCount(for:)` is shipping (not beta), but at
**26.4**, a point release after the 26.0 baseline, on the same five
platforms. A contributor who assumes every Foundation Models capability lands
at 26.0 will be wrong about this one specifically.
[developer.apple.com/tutorials/data/documentation/foundationmodels/systemlanguagemodel/tokencount(for:).json](https://developer.apple.com/tutorials/data/documentation/foundationmodels/systemlanguagemodel/tokencount(for:).json),
accessed 2026-09-02.

Guardrails and refusal behavior are shipping (not beta) at 26.0.
`SystemLanguageModel.Guardrails` documents a `default` guardrail set that
"blocks unsafe content" in both prompts and responses, and a
`permissiveContentTransformations` option for cases needing broader content
handling. A triggered guardrail raises `LanguageModelError.guardrailViolation(_:)`,
a typed, catchable refusal rather than a silent empty response.
[developer.apple.com/tutorials/data/documentation/foundationmodels/systemlanguagemodel/guardrails.json](https://developer.apple.com/tutorials/data/documentation/foundationmodels/systemlanguagemodel/guardrails.json),
accessed 2026-09-02.

`UNVERIFIED`, marked as such rather than asserted: the framework's numeric
context-window size. Apple's own shipping documentation describes only the
`contextSizeExceeded` error condition, not a published token-count constant.
Community developer-forum and blog figures (roughly 4K tokens at the 26.0
model, with later reports of a larger figure after a model refresh) exist but
were not located in Apple's own primary documentation during this pass and
must not be used to size any Catalyst buffer or claim.

**What `SystemLanguageModel` deliberately does not expose**, which is the
architecturally load-bearing half of this section: no raw logits and no
sampling-internals access beyond the documented `GenerationOptions` surface
(temperature and related generation preferences); no way to inspect or modify
the model's KV cache; no way to load arbitrary third-party model weights into
`SystemLanguageModel` itself. These are absences confirmed by the shape of the
shipping API surface fetched above, which exposes a session and a schema
system but no weight-loading, cache-control, or logit-access entry point
anywhere in the `FoundationModels` module topic list. A third-party on-device
runtime that owns its own weights, its own sampling internals, and its own
cache management is not competing with a capability Apple ships; it occupies
the exact space Apple's shipping API declines to occupy. This is the
architectural reason a Catalyst-owned runtime remains useful even where
`SystemLanguageModel` is available and eligible.

### 2. The custom-provider protocol: preview, not a requirement source

`LanguageModel` and `LanguageModelExecutor` are the protocol pair that would
let a third-party model serve `LanguageModelSession`. Both are confirmed
`beta: true, introducedAt: 27.0` on every documented platform, iOS, iPadOS,
Mac Catalyst, macOS, visionOS, and watchOS, fetched directly from Apple's own
DocC JSON data on 2026-09-02, the same day as every other citation in this
document.
[developer.apple.com/tutorials/data/documentation/foundationmodels/languagemodel.json](https://developer.apple.com/tutorials/data/documentation/foundationmodels/languagemodel.json)
and
[developer.apple.com/tutorials/data/documentation/foundationmodels/languagemodelexecutor.json](https://developer.apple.com/tutorials/data/documentation/foundationmodels/languagemodelexecutor.json),
accessed 2026-09-02.

Their documented shape: a `LanguageModel` describes a model's capabilities and
executor configuration; a paired `LanguageModelExecutor` "translates framework
types into platform-specific types and streams results back" through a
generation channel, implementing `respond(to:model:streamingInto:)`,
converting `GenerationOptions` and `ContextOptions` into whatever the
underlying model expects, approximating unsupported sampling modes where
necessary (Apple's own doc example shows approximating greedy sampling with a
temperature of zero when a backend has no native greedy mode), and optionally
implementing `prewarm(model:transcript:)`. A session constructed with a custom
model (`LanguageModelSession(model: MyCustomServerLanguageModel())`) is used
through the same `respond(to:)` call site as a session backed by the system
model, which is the specific design property that makes an adapter additive:
the call site does not change based on which model backs it.

**Per ADR-0004, this section is planning input only.** iOS 27 has not reached
general availability as of this evidence date, and the protocol pages are
marked beta on every platform in Apple's own metadata. No acceptance criterion
anywhere in this program may cite `LanguageModel` or `LanguageModelExecutor`
until they ship in a stable SDK. `Q1` in this document's front matter records
that the exact GA date and the protocol's final shape at GA are both still
open.

**Adapter-boundary design consequence**, stated now so the constraint is
useful rather than merely restrictive: the Catalyst backend abstraction
(`InferenceBackend` in the current v2 source, its v3 successor pending
ODC-0102/ODC-0103) should already look, from its call sites' perspective, like
a `respond`-shaped async streaming call over an opaque model handle, with
generation preferences passed as a value type rather than threaded through
backend-specific parameters. If that shape holds, conforming a future stable
`LanguageModel`/`LanguageModelExecutor` pair to Catalyst's own backend
protocol, or the reverse, wrapping a Catalyst backend to satisfy Apple's
protocol, is an additive conformance written against an already-stable
internal shape. It is not a rewrite of the execution-policy layer, the
session/stream contract, or the public request and response types, because
none of those depend on which protocol a given backend happens to also
satisfy. The concrete conformance is out of scope for this ticket and belongs
to ODC-0207 once ODC-0102 exists and once, separately, the protocol pair
itself is no longer beta.

### 3. MLX: what the current package supports, and why it stays optional

The current v2.0.4 package integrates MLX through `mlx-swift-lm`, pinned at
`Package.swift:30` to `exact: "2.29.3"`, with `mlx-swift` resolving
transitively to `0.29.1`.
[`docs/baselines/v2.0.4.md`](../baselines/v2.0.4.md), "Dependency pins"
section, captured 2026-09-01 against revision `59da80b`. `Sources/OnDeviceCatalyst/Backend/MLXInstance.swift`
is the current integration point, per `docs/ARCHITECTURE.md`'s source-of-truth
diagram.

MLX's own framing, from its published design principles, is an array
framework for machine learning built around Apple silicon's unified-memory
architecture, with lazy evaluation and compiled functions as core, documented
design choices, so tensors are not copied between a CPU-addressed and a
GPU-addressed heap the way they are on non-unified-memory hardware. This
project-level framing is well established from MLX's own materials, but this
brief did not re-fetch MLX's own documentation text directly in this pass, so
it is recorded as design context rather than a numbered availability fact with
a DocC citation.

`UNVERIFIED` (community-sourced, not confirmed against an MLX-project primary
document in this pass): MLX requires Metal and does not run correctly in the
iOS Simulator, because the Simulator exposes a reduced GPU-family emulation
that lacks features MLX's Metal kernels depend on. This claim is directionally
consistent with the measured v2.0.4 baseline's own finding that the package's
*llama.cpp* path ships a non-functional simulator stub for an unrelated
reason (E1 in `docs/baselines/v2.0.4.md`), so "MLX needs a physical device" and
"the current package's simulator slice cannot infer" are two independently
plausible but separately evidenced claims about the same simulator
limitation, and must not be merged into one citation.

**Why MLX must stay an optional product, never a core dependency**, restating
and grounding the already-approved constraint in `docs/ARCHITECTURE.md`
("Llama.cpp, MLX, and Apple system-model integration are optional products" /
"Heavy backend dependencies do not resolve for core-only consumers"): MLX has
a hard physical-device requirement the core package's own deployment-target
promise (iOS 17 through 26, macOS 14 through 26, per ADR-0004 point 1) does
not share, and a unified-memory design assumption that does not hold on every
architecture the core package's deployment floor was chosen to still reach.
A consumer building for the simulator, for CI, or for a device class MLX does
not target must be able to depend on the core package without MLX resolving
at all. This is a package-graph requirement (a separate product target, not a
compile-time flag inside one target), and it is already recorded as approved
in `docs/ARCHITECTURE.md`; this brief supplies the platform evidence that
justifies it rather than re-deciding it.

### 4. Metal: availability, and precise hardware gating

`MTLTensor`, the Metal 4 tensor resource, is shipping (not beta) at 26.0 on
iOS, iPadOS, Mac Catalyst, macOS, tvOS, and visionOS.
[developer.apple.com/tutorials/data/documentation/metal/mtltensor.json](https://developer.apple.com/tutorials/data/documentation/metal/mtltensor.json),
accessed 2026-09-02. Apple's own symbol documentation, fetched directly, does
**not** state a specific chip or GPU-family floor for `MTLTensor` itself; it
documents the type's shape (`dimensions`, `dataType`, `strides`, `buffer`,
`usage`) without a hardware-requirements paragraph.

`UNVERIFIED` (not confirmed against primary Apple documentation fetched in
this pass, `Q2`): a commonly cited hardware floor for Metal 4 overall places
compatibility at M1 and later, A14 Bionic and later, and states A13 Bionic and
Intel Macs are excluded. This program has not independently re-verified that
specific chip list against Apple's own developer-session transcript text in
this pass, and it must not be treated as confirmed until it is.

Separately, and confirmed by an Apple-published PDF rather than a DocC page,
the Metal Performance Primitives guide states plainly that "Metal 4 introduces
the tensor resource and the Metal Performance Primitives (MPP) framework for
authoring machine learning kernels that leverage GPU neural accelerators in
the Apple M5 chip," and the guide's own tuning constants are stated as
M5-specific starting points.
[developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf),
dated 2026-03-16, "Version 1." This program has not independently re-fetched
and re-read that PDF's full text in this pass; it is cited here on its own
published URL and date rather than left uncited, and its M5-specific framing
is recorded as exactly that: a statement about GPU-integrated Neural
Accelerator hardware present on the newest chip generation, not a statement
about `MTLTensor` or Metal 4 generally. `UNVERIFIED` (community-sourced only
in this pass) is a separate claim that the same accelerator hardware also
exists on the A19 / A19 Pro generation; it is not treated as confirmed here.

**The precision this brief insists on, because the task that produced it
named this failure mode explicitly**: `MTLTensor` and the Metal 4 ML command
path are available, as software, across the documented 26.0 platform floor.
Whether a given device's GPU has dedicated Neural Accelerator silicon behind
that software path is a separate, narrower, chip-generation-scoped question
that only Apple's M5-specific MPP guide answers directly, and it answers it
for exactly one chip family. A v3 Metal backend must therefore treat
"`MTLTensor` compiles and runs" and "this device has accelerated tensor
hardware behind it" as two different facts, checked separately, with the
second one defaulting to unaccelerated (ordinary compute-kernel) behavior
rather than assuming acceleration from API availability alone. This is a
direct instance of the device-aware behavior ADR-0004 assigns to the
execution-policy layer, not to a single Metal capability check.

**Grounded in the measured baseline, because this is the concrete failure a
v3 Metal path must not repeat**: `docs/baselines/v2.0.4.md` records that the
seven `.metal` shader sources under `Sources/OnDeviceCatalyst/Metal Engine/Shaders/`
are unhandled by SwiftPM (`swift package describe --type json` reports zero
declared resources on the library target), so no `default.metallib` is ever
produced for a SwiftPM consumer, and `MetalComputeEngine.swift:89`'s call to
`device.makeDefaultLibrary()` throws for every such consumer. `docs/specs/ODC-0004-v2-characterization-suite.md`
(finding N7) additionally recorded that `xcodebuild` *does* compile the
shaders, into a resource-bundle target, but `makeDefaultLibrary()` reads
`Bundle.main`, not `Bundle.module`, so the shader library is not found there
either. The Metal Engine subtree is consequently unreachable both as a
package consumer and, for an unrelated reason, under the one build system that
does compile the shaders. Any v3 Metal backend inherits this as its starting
defect inventory, not as a clean slate; ODC-0014 owns the repair, and this
brief's job is only to state that Metal availability and Metal *reachability
from this codebase* are currently two different, both-false-in-different-ways
facts.

### 5. Background Assets: delivery, not a core dependency

The Background Assets framework's base APIs (`BADownloadManager`,
`BAURLDownload`, `BADownload`, essential-asset background downloads) are
shipping (not beta) starting iOS/iPadOS/Mac Catalyst 16.0, macOS 13.0, tvOS
18.4, and visionOS 2.4 (or 1.0+ for compatible iPad/iPhone apps running under
visionOS compatibility).
[developer.apple.com/tutorials/data/documentation/backgroundassets.json](https://developer.apple.com/tutorials/data/documentation/backgroundassets.json),
accessed 2026-09-02. The newer system-managed asset-pack layer,
`AssetPackManager`, `AssetPack`, and `AssetPackManifest`, ships starting at
26.0 across iOS, iPadOS, macOS, tvOS, visionOS, and Mac Catalyst.
[developer.apple.com/tutorials/data/documentation/backgroundassets/assetpackmanager.json](https://developer.apple.com/tutorials/data/documentation/backgroundassets/assetpackmanager.json),
accessed 2026-09-02.

Both hosted and self-hosted modes exist as documented, separate paths: Apple
can host and version asset packs through App Store Connect for
`AssetPackManager`-managed delivery, or an app can host its own manifest
through the older, still-present `BADownloadManager` / `BAURLDownload` /
`BADownloaderExtension` surface, independent of App Store Connect. Download
policy is expressed through three named strategies documented against the
managed layer: essential downloads block first launch
(`BAEssentialMaxInstallSize`), prefetch downloads happen after install without
blocking launch (`BAMaxInstallSize`), and on-demand downloads are triggered
explicitly by app code at runtime. Eviction has one confirmed API,
`AssetPackManager.remove(assetPackWithID:)`, an explicit, developer-triggered
removal call, fetched and confirmed on the same pass as the platform table
above.

`UNVERIFIED` (`Q4`, not confirmed against primary Apple documentation fetched
in this pass): whether the system performs *automatic* eviction of managed
asset packs under storage pressure, beyond the explicit `remove(assetPackWithID:)`
call. Also `UNVERIFIED`: a named, documented integrity or checksum field on
the asset-pack manifest beyond the general "the system automatically manages
downloads, updates, compression" language in Apple's own framework summary;
no specific hash or signature field name was located in the sections fetched
in this pass.

App Store coupling is real for the Apple-hosted managed path (uploading and
versioning through App Store Connect is how that specific mode works) but is
not a property of the framework as a whole; the self-hosted, unmanaged path
requires no App Store Connect relationship. This distinction matters directly
for OnDeviceCatalyst: **the core package must still work with plain local
files and ordinary downloads**, per `docs/ARCHITECTURE.md`'s already-recorded
constraint ("Local model files never require the downloader service" /
"Model delivery must support local files and ordinary application downloads"),
so Background Assets, in either mode, is scoped as an optional delivery
integration (ODC-0206) layered above model identity and asset lifecycle
(ODC-0104, ODC-0205), never as a required path to load a model at all.

### 6. The deployment-target decision: cost, benefit, and evidence obligation

ADR-0004 holds the core package's deployment target at iOS 17 / macOS 14,
unchanged from the current `Package.swift:20-21` declaration
(`.iOS(.v17)`, `.macOS(.v14)`), because "Apple has shipped nothing for iOS 17
through 25" and that gap widens, not narrows, with each Apple release. Stated
plainly, restating ADR-0004's own reasoning rather than re-deriving it: every
Foundation Models capability in section 1 above requires 26.0 at minimum
(several require 26.4 or the still-preview 27.0), so the iOS 17-25 installed
base receives nothing from Apple's system-model story regardless of what ships
next, and a runtime that only supports 26+ abandons exactly the population
Apple's own announcements do not reach.

**What the decision costs.** The core package cannot assume any Apple
on-device model API, any Metal 4 tensor path, or the newer Background Assets
managed layer, none of which reach below their stated floors (26.0, or 26.4,
or 27.0-preview). Every one of those becomes conditionally available code, not
baseline code, which is more surface for the execution-policy layer described
in `## Architecture and data flow` below to gate correctly, and it is more
surface that must be tested on an old-OS configuration a maintainer may not
have convenient hardware for. It also means the core package's own object-C
and Swift language-feature floor is set by iOS 17 / macOS 14 tooling
constraints, not by whatever the newest Swift version enables, which is a real
ergonomics cost against v3's separately-approved move to Swift 6 actor
ownership (`docs/ARCHITECTURE.md`, "Lifecycle state moves to explicit Swift 6
actor ownership"); that tension is inherited by ODC-0101, not resolved here.

**What the decision buys.** It is, in ADR-0004's own words, "the one durable
compatibility position available": no Apple announcement removes it, because
Apple's roadmap only adds capability at 26 and above, never extends coverage
downward. It keeps OnDeviceCatalyst usable by applications that cannot raise
their deployment target for business reasons unrelated to AI features at all,
which is a real and Apple-independent population.

**What must be tested to make the promise real, not merely declared.** The
measured v2.0.4 baseline is the concrete warning here: `Package.swift:21`
already declares `.macOS(.v14)`, and `docs/baselines/v2.0.4.md` records that
`swift build -c debug` on macOS fails outright with `no such module 'llama'`,
because the shipped XCFramework has no macOS slice at all (baseline finding
E4/D4). A platform floor stated in a manifest and a platform floor that
actually compiles and runs are not the same fact, and v2.0.4 is the
proof: the gap between them is a currently-failing build, not a hypothetical
risk. Carrying the iOS 17 / macOS 14 promise into v3 without repeating this
requires, at minimum: a compiling build at the declared floor on every
declared platform (the macOS cell in particular, given D4's history); the
absence of any unconditional import of a capability gated above the floor
(the same class of defect ADR-0004's own baseline evidence already found once,
in `LlamaBridge.swift`'s unconditional `import llama` against a stub-only
simulator slice); and a device or simulator run, not only a compile, on the
oldest declared OS the toolchain can still target, because `docs/baselines/v2.0.4.md`
also demonstrates that "the compiler accepted it" and "the product works" are
different claims (the `links, cannot infer` result for the current simulator
slice is exactly that distinction, made concrete). ODC-0302 (Apple-platform
lifecycle matrix) and ODC-0303 (device and model compatibility matrix) are the
tickets that own turning this into a recurring, evidenced check; this brief
states the obligation so those tickets inherit a named requirement rather than
an implicit hope.

## Proposed interfaces

No public Swift type is proposed or decided by this ticket; concrete package
names, public API shapes, and backend-selection types remain blocked on the
ODC-0100 series per `docs/ARCHITECTURE.md`. What follows is the shape
constraint this brief's evidence implies, stated at the level of a boundary
sketch so ODC-0102/ODC-0103 have a starting shape to accept, modify, or
reject, not a shipped interface.

- An **optional Apple-system backend product** (ODC-0207) that conforms to
  Catalyst's own backend abstraction and internally calls
  `LanguageModelSession` where `SystemLanguageModel.availability == .available`.
  It ships only on platforms where `FoundationModels` exists (26.0+), and its
  absence must not affect compilation or behavior of the core product on any
  platform, including iOS 17-25 and any macOS/visionOS version below 26.
- An **optional custom-provider adapter**, additive and not started until
  `LanguageModel`/`LanguageModelExecutor` ship outside beta, that lets an
  existing Catalyst backend also satisfy Apple's protocol pair, so a consumer
  on 27+ can route Catalyst-owned model weights through `LanguageModelSession`
  if they choose to. Per ADR-0004, this adapter is planning-shaped only in
  this document; no interface is fixed here, and no acceptance criterion in
  this program depends on it existing.
- **Device-aware selection, memory budgeting, lifecycle, cancellation, and
  performance reporting live in one execution-policy layer**, above every
  concrete backend (llama.cpp, MLX, Metal, the optional Apple-system backend),
  never duplicated per backend. This is the direct implementation of ADR-0004
  point 4 ("Move the differentiator to the execution-policy layer") and is
  expanded in `## Architecture and data flow` below.

## Architecture and data flow

ADR-0004's consequence is explicit: "the public v3 architecture ticket must be
rewritten around execution policy rather than around backend abstraction as an
end in itself." This section states what that means at the level of module
boundaries, for ODC-0100/ODC-0102/ODC-0103 to take as an input.

```text
Public request/response/event/capability contracts (ODC-0102)
        |
        v
Execution-policy layer                     <- the differentiator, per ADR-0004
    +-- device-aware backend selection      (what can this device even run)
    +-- memory budgeting and eviction       (what fits, what gets evicted)
    +-- lifecycle and backgrounding         (suspend, resume, thermal state)
    +-- cancellation                        (reaches underlying work, not just the stream)
    +-- honest performance reporting        (measured, device-scoped, never universal)
        |
        v
Backend conformances, each optional and independently resolvable
    +-- llama.cpp backend        (ODC-0200)   required stable backend
    +-- MLX backend              (ODC-0201)   required stable backend, physical-device only
    +-- Metal backend            (ODC-0014 repair, then a v3 product)  hardware-gated within itself
    +-- Apple system-model backend (ODC-0207) optional, 26.0+ only, conformance target not competitor
    +-- Apple custom-provider adapter          optional, additive, not started pre-GA (ADR-0004 pt. 5)
```

Consequences, stated concretely rather than left implicit:

- **Optional backends**: the Apple system-model backend (ODC-0207) and, later,
  the Apple custom-provider adapter. Both are reachable only through the
  execution-policy layer's selection logic and both are absent by default from
  any consumer that does not opt in, consistent with `docs/ARCHITECTURE.md`'s
  "Heavy backend dependencies do not resolve for core-only consumers."
- **Adapters, not backends**: the custom-provider conformance is specifically
  an adapter over an *existing* Catalyst backend, not a new source of model
  capability. It changes how a backend is reached from outside the package
  (through Apple's session API) without changing what the backend does. This
  is why ADR-0004 can call it "additive rather than a rewrite": the
  execution-policy layer and the backend's own logic are unaffected by whether
  the adapter is compiled in.
- **Deliberately not pursued**: a bespoke, Catalyst-owned unified multi-backend
  API positioned as *the* reason to adopt the package. ADR-0004 point 2 is
  explicit that this contest is lost on a schedule measured in weeks once
  Apple's own protocol reaches GA, and section 2 above shows Apple's own MLX
  team is separately building the exact same "any model behind one session
  API" bridge for its own models. Investing further in that positioning is
  the one option this brief affirmatively rules out, per the ADR it
  implements.
- **Metal's place**: a hardware-gated backend, not a universally available
  one, whose GPU-accelerator tier (section 4) is itself a device-aware
  selection input to the execution-policy layer rather than a compile-time
  assumption. Repairing its packaging (ODC-0014) is a precondition for it
  being a v3 backend at all, independent of everything else in this brief.
- **Background Assets' place**: entirely outside the execution-policy layer
  and the backend list above. It is a delivery mechanism for model and shader
  assets that the model-identity and asset-lifecycle layer (ODC-0104,
  ODC-0205) may optionally use; no backend depends on it, and the core package
  never requires it to load a model from a local file or an ordinary URL
  download.

## Lifecycle, concurrency, and cancellation

This section names what the execution-policy layer in `## Architecture and
data flow` must own, translating ADR-0004 point 4 into obligations rather than
leaving it as a phrase.

- **Device-aware selection** must be able to answer, per device and per
  session request, which backends are even eligible: `SystemLanguageModel.availability`
  for the optional Apple backend (section 1); a physical-device check for MLX
  (section 3); a Metal-capability and, separately, an accelerator-tier check
  for the Metal backend (section 4); and, for llama.cpp, whatever
  architecture and quantization support the loaded model actually requires. No
  single boolean stands in for all four.
- **Memory budgeting and eviction** is a policy concern this program does not
  get from any Apple API surveyed here: `SystemLanguageModel` manages its own
  memory internally and exposes no budget control to the caller, and MLX's
  unified-memory design (section 3) makes memory pressure a real,
  currently-uncontrolled risk for a process also running other unified-memory
  work. The v2.0.4 baseline's own defect D1 (`docs/baselines/v2.0.4.md`,
  "confirmed defects") is a concrete instance of eviction done wrong today,
  a cached instance is asynchronously shut down with no happens-before
  against the cache read that might still be using it, so the execution-policy
  layer's ownership of this concern is not aspirational; it is replacing a
  known-broken mechanism.
- **Lifecycle and backgrounding** (suspend, resume, thermal transitions) has
  no Apple-provided answer specific to third-party on-device inference in any
  API surveyed in this brief; `SystemLanguageModel` handles its own lifecycle
  internally and exposes none of it to a caller running a different backend.
  This is squarely the kind of capability ADR-0004 point 4 names as "no API
  surface Apple has announced supplies any of this."
- **Cancellation** must reach underlying backend work, not only stop a
  `AsyncStream` from being read further. `docs/ARCHITECTURE.md`'s already
  approved v3 constraint ("cancellation reaches underlying work") is the
  correct target; the v2.0.4 baseline's defect D3 (an unsatisfiable
  `publishProgress` gate that means a stream's terminal event is never
  delivered on some paths) is the concrete failure mode this constraint
  exists to prevent, and it is currently present in the codebase this brief
  is grounded in.
- **Honest performance reporting** must be device-scoped and never presented
  as a universal claim, per the roadmap's own operating gate 6 ("No universal
  performance claim may be inferred from a model-specific or device-specific
  result"). This applies identically whether the measured backend is
  llama.cpp, MLX, Metal, or the optional Apple system-model backend; Apple's
  own model-generation versioning (section 1, three model generations named
  as of this evidence date) is itself a reason a performance claim about
  `SystemLanguageModel` has a shorter shelf life than a claim about a backend
  whose weights Catalyst controls directly.

## Failure modes

- **Guardrail refusal** (`LanguageModelError.guardrailViolation(_:)`, section
  1) is a typed, catchable error on the Apple-system backend path only; it has
  no equivalent concept on a llama.cpp or MLX backend loaded with
  unrestricted weights. A consumer that switches backends through the
  execution-policy layer must not assume refusal behavior is portable across
  that switch, and the public error contract (ODC-0102) must model this as a
  backend-reportable capability, not a universal case.
- **Model unavailability** (`SystemLanguageModel.Availability.unavailable(.deviceNotEligible)`
  or `.modelNotReady`, section 1) is a normal, expected state on real hardware,
  not an exceptional one; the execution-policy layer's device-aware selection
  must treat it as a routine input to backend choice, not as an error path
  bolted on afterward.
- **Simulator non-inference**: the measured v2.0.4 baseline's simulator slice
  links but cannot infer (`docs/baselines/v2.0.4.md`, build-matrix result
  `links, cannot infer`), and MLX's own physical-device requirement (section
  3, `UNVERIFIED` but directionally consistent) means the simulator is not a
  reliable inference surface for either of the package's two heavy backends
  today. Any v3 CI or characterization surface (per `docs/specs/ODC-0004-v2-characterization-suite.md`'s
  own partitioning by requirement class) must keep treating the simulator as
  a compile-and-failure-path surface, not an inference surface, until a
  backend demonstrates otherwise with evidence.
- **Metal reachability**: as stated in section 4, the current package's Metal
  Engine is unreachable both as a SwiftPM consumer (no `default.metallib` is
  produced) and, for the separate `Bundle.main`-versus-`Bundle.module` reason
  found by ODC-0004, under `xcodebuild`. A v3 Metal backend must not assume
  either build system currently delivers a working shader library; ODC-0014's
  repair is a precondition, not a detail.
- **Preview-API churn**: `LanguageModel`/`LanguageModelExecutor` are beta as
  of this evidence date (section 2), and this brief's own `evidence_fresh_until`
  is 14 days out. A design decision that treats today's preview shape as fixed
  risks being wrong before iOS 27 even reaches general availability; the
  adapter-boundary design in `## Proposed interfaces` is deliberately shaped
  to tolerate that churn rather than assume today's beta signatures are final.

## Compatibility and migration

Nothing in this brief changes `Package.swift`, any file under `Sources/` or
`Tests/`, or any currently declared platform floor. The deployment-target
decision this brief grounds (section 6) is ADR-0004's, already accepted, and
this document does not reopen it.

The one compatibility fact this brief adds beyond what ADR-0004 already
states: every Apple capability surveyed in section 1 through section 5 is
strictly additive to the existing iOS 17 / macOS 14 floor, gated at 26.0,
26.4, or 27.0-preview, never lowering the floor and never requiring the core
package to raise it. A consumer who never opts into an optional Apple-system
product target continues to build against the exact floor `Package.swift`
already declares. This is the concrete form of `docs/ARCHITECTURE.md`'s
"Specific package names, public types, migration behavior, and backend
selection remain blocked until their ODC-0100 series specs are approved":
this brief supplies the platform facts those specs will migrate against, it
does not perform the migration.

## Security, privacy, and licensing

- **No raw weights or logits cross the Apple-system backend boundary.**
  Section 1 establishes that `SystemLanguageModel` exposes no weight-loading
  or logit-access surface; a Catalyst execution-policy layer that routes a
  request to that backend is, by construction, unable to leak model internals
  it never had access to, which is a privacy property of Apple's own API
  shape rather than something Catalyst must separately enforce.
- **Guardrail behavior is backend-specific, not a Catalyst-wide security
  property**, restating the failure-mode note above from a privacy and safety
  angle: a consumer relying on the Apple-system backend's default guardrails
  for content safety must not assume the same protection exists when the
  execution-policy layer selects a llama.cpp or MLX backend loaded with
  different weights. The public capability contract (ODC-0102) must surface
  this as a queryable backend capability.
- **MLX and its transitive dependencies are MIT-licensed** at the versions
  pinned in `docs/baselines/v2.0.4.md`'s dependency table (`mlx-swift-lm`
  2.29.3, `mlx-swift` 0.29.1, and their own transitive dependencies), which is
  compatible with this project's Apache-2.0 license per
  `docs/decisions/ODC-ADR-0002-apache-dco.md`. This brief does not re-audit
  every transitive license; it records that the existing pin is the one
  measured and does not introduce a new dependency.
- **Background Assets' Apple-hosted mode is an App Store Connect–coupled
  workflow** (section 5): using it ties asset publication to an App Store
  Connect relationship for that specific delivery path. The self-hosted mode
  has no such coupling. ODC-0206 must document which mode a given consumer is
  opting into, because the privacy and operational posture differs by mode.
- No model weights, no device identifiers, and no personal data are
  introduced, collected, or referenced anywhere in this brief, consistent with
  the redaction discipline already established in
  `docs/baselines/v2.0.4.md` and `docs/specs/ODC-0004-v2-characterization-suite.md`.

## Tests and device validation

This brief proposes no test code and adds nothing under `Tests/`. It states
what later tickets must validate so the claims here do not become declared
rather than demonstrated, mirroring the honesty standard ODC-0002 and ODC-0004
already set for this program.

- **ODC-0302** (Apple-platform lifecycle matrix) inherits the obligation from
  `## Current behavior and evidence`, section 6: a compiling, running build at
  the declared iOS 17 / macOS 14 floor, on every declared platform, including
  a macOS cell that must not repeat the current baseline's `no such module
  'llama'` failure.
- **ODC-0303** (device and model compatibility matrix) inherits the
  device-aware selection surface from `## Lifecycle, concurrency, and
  cancellation`: `SystemLanguageModel.availability` outcomes, Metal
  accelerator-tier detection, and MLX's physical-device requirement all need
  a recorded, per-device evidence row, not an assumed universal state.
- **ODC-0207** (optional Apple system-model backend) inherits the guardrail
  and availability-state failure modes from `## Failure modes` as required
  test cases once that ticket's spec exists, not as a suggestion.
- **ODC-0014** (Metal packaging repair) inherits the two independent
  reachability failures from section 4 (`no default.metallib` for a SwiftPM
  consumer; `Bundle.main` versus `Bundle.module` for an `xcodebuild` consumer)
  as its defect inventory, already characterized by ODC-0004's finding N7 and
  restated here for traceability.
- This document's own `UNVERIFIED` markers (`Q2` Metal's chip-level floor,
  `Q3` the Foundation Models context-window size, `Q4` Background Assets
  eviction and integrity metadata) are open questions for whichever ticket
  next depends on the underlying fact to resolve with its own primary-source
  citation before treating the fact as settled.

## Acceptance criteria

Every criterion is decided by a command's exit code or output, per the
program's evidence rule.

| # | Criterion | Deciding command |
| --- | --- | --- |
| A1 | The document exists at the required path | `test -f docs/specs/ODC-0005-apple-platform-design-brief.md` |
| A2 | Front matter declares `status: SPEC_DRAFT` and `founder_approved: pending` | `grep -q '^status: SPEC_DRAFT$' docs/specs/ODC-0005-apple-platform-design-brief.md && grep -q '^founder_approved: pending$' docs/specs/ODC-0005-apple-platform-design-brief.md` |
| A3 | No em dash character appears anywhere in the document | `! grep -qP '\xe2\x80\x94' docs/specs/ODC-0005-apple-platform-design-brief.md` |
| A4 | Every required capability area is covered by a named section | `grep -qE '^### 1\. Foundation Models' docs/specs/ODC-0005-apple-platform-design-brief.md && grep -qE '^### 2\. The custom-provider protocol' docs/specs/ODC-0005-apple-platform-design-brief.md && grep -qE '^### 3\. MLX' docs/specs/ODC-0005-apple-platform-design-brief.md && grep -qE '^### 4\. Metal' docs/specs/ODC-0005-apple-platform-design-brief.md && grep -qE '^### 5\. Background Assets' docs/specs/ODC-0005-apple-platform-design-brief.md && grep -qE '^### 6\. The deployment-target decision' docs/specs/ODC-0005-apple-platform-design-brief.md` |
| A5 | ADR-0004's preview-API constraint is stated explicitly, not merely implied | `grep -qi 'may not define acceptance criteria' docs/specs/ODC-0005-apple-platform-design-brief.md` |
| A6 | Every UNVERIFIED claim is labeled as such rather than asserted | `grep -c 'UNVERIFIED' docs/specs/ODC-0005-apple-platform-design-brief.md` returns 4 or more |
| A7 | Every DocC JSON citation used for a SHIP/PREVIEW claim carries an access date of this draft | `grep -c 'accessed 2026-09-02' docs/specs/ODC-0005-apple-platform-design-brief.md` returns 10 or more |
| A8 | No file outside this spec is modified by this ticket | `git diff --stat -- . ':!docs/specs/ODC-0005-apple-platform-design-brief.md'` produces empty output for tracked files, excluding untracked new files this ticket did not create |
| A9 | Project state remains internally consistent | `python3 scripts/validate-project-state.py` exits 0 |
| A10 | Every named follow-up ticket (ODC-0014, ODC-0102, ODC-0103, ODC-0104, ODC-0206, ODC-0207, ODC-0302, ODC-0303) referenced in this brief exists in `Tickets.md` | `for t in ODC-0014 ODC-0102 ODC-0103 ODC-0104 ODC-0206 ODC-0207 ODC-0302 ODC-0303; do grep -q "$t" Tickets.md || echo "missing $t"; done` produces no output |

## Alternatives considered

- **Keep the "WWDC25" title and scope the brief to only what shipped by that
  event.** Rejected: it is the exact failure mode ADR-0004 exists to correct,
  a fixed-date framing that goes stale the moment Apple ships again, and it
  would have excluded the 27.0-preview facts ADR-0004 requires this brief to
  account for as planning input.
- **Treat the 27.0 custom-provider protocol as a normative requirement now,
  reasoning that GA is close.** Rejected, directly by ADR-0004 point 5: preview
  APIs may inform planning; they may not define acceptance criteria until they
  ship in a stable SDK. "Close" is not "shipped," and this program's own
  14-day evidence rule exists precisely because "close" claims decay.
- **Fold this brief's factual survey into ODC-0100 (v3 vision) directly rather
  than keeping it a separate ticket.** Rejected: ODC-0100 is blocked on the
  private research thesis gate (`docs/decisions/ODC-ADR-0004-apple-api-conformance-over-competition.md`'s
  own status note; `ROADMAP.md`'s R0 gate), while this brief's platform facts
  are usable immediately and are needed by P0-adjacent tickets (ODC-0014,
  ODC-0206) that are not themselves blocked on that gate. Keeping it separate
  lets the platform facts ship without waiting on the thesis gate.
  `Tickets.md` already records ODC-0005 as an independent, unblocked P0
  ticket; this brief preserves that scheduling property.
- **Cite the private landscape review directly, since it already contains
  dated Apple platform facts.** Rejected by `docs/decisions/ODC-ADR-0003-public-private-research-boundary.md`:
  this is a public specification, and every load-bearing fact here is instead
  independently re-verified against Apple's own DocC JSON endpoints and PDFs,
  with this document's own access dates, so the public brief stands on public
  evidence and carries no dependency on a document this repository's public
  contributors cannot read.
- **Treat Metal 4 as universally available because it exists on the newest
  hardware.** Rejected, per this brief's own explicit task and section 4's
  precision requirement: `MTLTensor`'s software floor (26.0, confirmed) and
  GPU-integrated Neural Accelerator hardware (M5-specific, per Apple's own MPP
  guide) are kept as two separate claims throughout, specifically so a v3
  Metal backend does not repeat the "compiles, therefore works everywhere"
  mistake the measured v2.0.4 baseline already demonstrates in a different
  subsystem (the simulator slice that links but cannot infer).

## Review record

- 2026-09-02, initial draft. Written against ADR-0004 (accepted 2026-09-01)
  and the measured v2.0.4 baseline (`docs/baselines/v2.0.4.md`, captured
  2026-09-01 against revision `59da80b`). Every Apple platform-availability
  claim was independently fetched from Apple's own DocC JSON data or a
  directly cited Apple PDF on 2026-09-02, the draft date, rather than sourced
  secondhand. No prior review pass exists for this ticket.
- Founder review: pending. This spec is `SPEC_DRAFT` and has not entered
  `SPEC_REVIEW`. The ticket title correction proposed in `## Title correction`
  is a recommendation for the ledger owner to apply; this document does not
  and cannot self-apply it, per program rule 2 and this ticket's own
  constraint against editing `Tickets.md`.

## Validation evidence

Not implemented. This is a specification; no code, test, or script exists yet
to validate beyond the acceptance criteria above, which check the document
itself and the project's ledger consistency. `evidence_fresh_until` is set to
2026-09-16, fourteen days from this draft's `last_updated` date, per the
program's evidence-freshness rule (`ROADMAP.md`, operating gate 4). Every
DocC JSON and PDF citation in `## Current behavior and evidence` was fetched
live on 2026-09-02 in the course of writing this draft; no citation in this
document was carried over from an earlier, unverified pass.
