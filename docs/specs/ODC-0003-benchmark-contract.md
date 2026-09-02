---
id: ODC-0003
title: Cross-backend benchmark contract
type: benchmark
status: APPROVED
milestone: P0
owner: unassigned
dependencies: ODC-0002
founder_approved: delegated-to-manager-2026-09-01
last_updated: 2026-09-02
evidence_fresh_until: 2026-09-16
unresolved_questions: none
---

# ODC-0003: Cross-backend benchmark contract

## Question and allowed claim

What this ticket answers: **how do we measure and report an inference
performance number for this package such that two honest operators, on two
different days, on the same declared hardware, model, and configuration, do
not disagree, and such that the number cannot be misread as a claim it does
not support.**

This is a contract, not a result. It fixes definitions, gates, statistics, a
machine-readable schema, and a citability rule before any number exists. It
does not run a benchmark and does not publish one. `## Validation evidence`
states why, plainly.

**The allowed claim, and only this claim:** a reported number describes one
backend, one model, one quantization, one context length, one offload
configuration, on one named device and OS build, measured under this
contract's gates and statistics. Nothing else may be inferred from it. This is
not a stylistic preference; it is `ROADMAP.md`'s operating gate 6, restated as
an enforceable rule: no universal performance claim may be inferred from a
model-specific or device-specific result. A sentence of the form "OnDeviceCatalyst
is faster on iOS" or "MLX beats llama.cpp on this package" is not a claim this
contract can license, regardless of what any individual result shows, and
`## Raw artifact schema` makes that mechanical rather than aspirational.

**Relationship to ODC-0004, stated so scope is not duplicated, and so the
directory boundary cannot be reintroduced by a future reader.** ODC-0004 owns
tests: it characterizes v2's current, including wrong, behavior, and its own
checker forbids any assertion that compares a measured duration to a
threshold. This ticket owns every timing, throughput, and memory claim the
project makes. ODC-0004's "Permitted changes to tracked files" table grants
itself `Tests/OnDeviceCatalystTests/**` in full, with no carve-out, and its
`--inventory` checker walks that same path expecting every test method in it
to be named in ODC-0004's own `## Tests`. This ticket therefore places no file
anywhere under `Tests/`, ever. The benchmark harness lives in a separate,
top-level directory this ticket owns outright, `Benchmarks/` (see
`## Reproduction procedure` step 4 for the exact layout), as its own SwiftPM
test target, distinct from `OnDeviceCatalystTests`. This gives the two specs
zero path overlap: ODC-0004's inventory checker never walks `Benchmarks/` and
needs no exclusion list to say so, and this contract never touches
`Tests/OnDeviceCatalystTests/`. `Benchmarks/` reuses the same build-and-deploy
*pattern* ODC-0004 documents (`swift build --build-tests`, repackage into a
flat iOS bundle, run through the platform `xctest` agent via `simctl` or
`devicectl`) rather than inventing a second build system, but it is a
separate SwiftPM target with its own path, not a subdirectory of ODC-0004's
target. See `## Reproduction procedure`.

**Boundary statement, because this is a public document under
[`ODC-ADR-0003`](../decisions/ODC-ADR-0003-public-private-research-boundary.md).**
Drafting this contract included a read-only review of private research
methodology, solely to avoid designing a public benchmark around a
measurement approach already known, privately, to be unreliable. No private
hypothesis, finding, measured value, or reasoning chain is reproduced here.
Where this contract requires a memory basis to be declared and never silently
mixed across runtimes, that requirement is stated as ordinary measurement
discipline, not as a discovery, and the reason is given in general terms only:
different runtimes allocate memory through different mechanisms, so one API
call is not guaranteed to see everything a given runtime holds resident.

## Prior art and freshness date

There is no existing v2 benchmark artifact in this repository. What exists:

- [`docs/baselines/v2.0.4.md`](../baselines/v2.0.4.md) and
  [`v2.0.4-environment.json`](../baselines/v2.0.4-environment.json) (ODC-0002,
  `DONE`), which established the build matrix this contract must respect and
  is quoted verbatim in `## Backends and pinned revisions` below.
- [`docs/specs/ODC-0004-v2-characterization-suite.md`](ODC-0004-v2-characterization-suite.md)
  (`REVISION`), which built and ran a test bundle against the iOS Simulator
  with no Xcode project, and states in its own text that this ticket "may
  consume `scripts/run-characterization.sh`" for that build-and-deploy
  pattern, with "no obligation in the other direction."

Neither artifact contains a timing, throughput, or memory number. ODC-0002
excluded build durations by name, on the grounds that an unpinned wall-clock
figure in a baseline invites exactly the kind of comparison this contract
exists to prevent.

**Why `dependencies:` above names only ODC-0002, stated explicitly rather than
left to be inferred.** This contract's reproduction procedure is easiest to
execute by reusing the build-and-deploy pattern ODC-0004 already proved out
(`swift build --build-tests`, repackage into a flat iOS bundle, run through the
platform `xctest` agent via `simctl` or `devicectl`), and `## Reproduction
procedure` step 4 says so. But that pattern is a documented, replicable
sequence of ordinary SwiftPM and Apple platform commands, not a proprietary
capability only ODC-0004 can supply: any equivalent build/deploy mechanism
that produces the same artifact (a flat iOS `.xctest` bundle for this
contract's own `Benchmarks/` target) satisfies this contract equally.
ODC-0004's script is reused only because it already exists and is proven, not
because this contract cannot function without it, and this contract adds no
file under `Tests/OnDeviceCatalystTests/` and requires no change to
ODC-0004's deliverables to be executed. That is why ODC-0004 is not listed as
a hard dependency: a scheduler reading `Tickets.md`'s dependency column alone
would otherwise conclude this contract cannot start until ODC-0004 reaches
`DONE`, which is stronger than what is actually true.

Freshness follows `ROADMAP.md` operating gate 4: external hardware, model,
runtime, and SDK evidence expires after 14 days. `evidence_fresh_until` above
is 14 days from `last_updated`. Every pinned identity this contract cites
(toolchain build numbers, dependency revisions, artifact checksum) must be
re-verified, not assumed inherited from ODC-0002, at the start of any session
that executes this contract; `## Reproduction procedure` step 1 makes that
literal.

## Models and representations

This contract does not assume weights, formats, or download mechanisms it has
not verified. What is required, regardless of format:

- Every model entry in the manifest's `models[]` array declares: a stable
  `id`, the backend it is paired with, architecture name, parameter count,
  quantization scheme, context length, file format, and a `sha256` of the
  exact file bytes used.
- The llama.cpp backend loads a model through `LlamaBridge`
  (`Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift`), which calls the
  llama.cpp C API unconditionally (ODC-0002 E2) and, per the baseline, has
  never successfully loaded a real model in this repository's own
  measurements. This contract does not assume a specific file format beyond
  what llama.cpp itself requires; it only requires the checksum and the
  format label be recorded.
- The MLX backend loads through `Backend/MLXInstance.swift` against
  `mlx-swift-lm`. This contract does not assert what file format or download
  path that dependency uses internally; verifying it is implementation work
  for whoever builds the harness, not a methodology decision this contract
  can make in advance. The manifest's `format` field is a free string for
  exactly this reason, and the checker validates only that it is non-empty
  and that the checksum matches the file the harness actually opened.
- **No model weights are committed to this repository, ever, in any tier.**
  This mirrors ODC-0004's stance exactly, for the same two reasons: licensing
  and size. A model is supplied to the harness through a local, untracked
  manifest file named by the environment variable
  `ODC_BENCHMARK_MODEL_MANIFEST`, whose entries are validated against the
  `models[]` declarations before any run (see `## Correctness gate`, gate 1).
- **Cross-backend model equivalence is declared, not inferred.** Two models on
  two backends are only comparable under this contract if their `architecture`,
  `parameter_count`, `quantization`, and `context_length` fields are recorded
  as intentionally matched by whoever configured the session. The checker
  cannot verify that a Q4 GGUF checkpoint and an MLX-quantized checkpoint of
  the same nominal model are truly equivalent; it can only verify that the
  operator declared them as a matched pair and that the declaration is
  present. A comparison across an undeclared pairing is `non-citable` per
  `## Raw artifact schema`.

## Backends and pinned revisions

Two backends exist in the v2 source tree today: llama.cpp, consumed through
the `llama` binary target, and MLX, consumed through `mlx-swift-lm`. The
in-house Metal Engine subtree (`MetalComputeEngine`, `TransformerGraph`,
`KVCache`, the GGUF parser) is explicitly **out of scope**: ODC-0002 finding D5
established that it is unreachable when the package is consumed as a package,
because the seven `.metal` shaders are unhandled resources and
`makeDefaultLibrary()` (`Metal Engine/Compute/MetalComputeEngine.swift:89`)
throws. Benchmarking dead code is not a benchmark; D5's repair ticket owns
making it reachable, and only after that can it be added to this contract by
revision.

Pins, as a manifest must record them, at the revision this contract was
drafted against (`7b5d847`, which carries ODC-0002's corrected lockfile):

| Field | Source | Value at drafting time |
| --- | --- | --- |
| `llama` XCFramework checksum | `Package.swift:13` | `741c3b584228c290c06bfbced9db161c3e7cb920c85d4d8df4ec54e8188a4e39` |
| `llama` device slice | ODC-0002 E1 | `ios-arm64`, 17,151,376 bytes, 16,420 defined symbols, not a stub |
| `llama` simulator slice | ODC-0002 E1 | `ios-arm64-simulator`, 7,936 bytes, 51 defined symbols, **is a stub** |
| `mlx-swift-lm` | `Package.swift:30`, `Package.resolved` | `exact 2.29.3`, resolved `5064b8c5d8ed3b0bbb71385c4124f0fc102e74a2` |
| `mlx-swift` (transitive) | `Package.resolved` | `0.29.1`, resolved `072b684acaae80b6a463abab3a103732f33774bf` |

A session's manifest re-measures every one of these fields with the same
commands ODC-0002 used (`swift package describe --type json`, artifact
`Info.plist` inspection) rather than copying this table, because the table can
go stale the moment a dependency moves and a stale pin silently invalidates
every result captured under it. This is `## Confounders and fairness
controls` item 1, restated here because a backend identity that does not
match its pin is not a backend, it is an unknown.

**Where this contract can and cannot run today, stated plainly, because
softening it would misrepresent a specification as a capability.** ODC-0002's
build matrix is the ground truth:

| Cell | Result | What it means for this contract |
| --- | --- | --- |
| `macos` | fails, `no such module 'llama'` | Not a surface. The package does not build on macOS at all (ODC-0002 E4). No macOS benchmark can exist until ODC-0013 lands a macOS slice. |
| `ios-simulator` | **links, cannot infer** | Not a surface for llama.cpp. ODC-0004 N2 and N3 show the simulator slice defines exactly the referenced symbol set and returns null or zero without trapping, deterministically, for every call. A "benchmark" run against it would time a function that does no work; `## Correctness gate` gate 3 refuses that number by construction, it does not merely discourage it. |
| `ios-device` | compiles against the real header set; never linked or run by any ODC ticket to date | The only candidate surface. Whether it can actually execute a benchmark harness depends on `## Open questions` Q1: no physical device has been measured by this project as of this draft. |
| `xcodeproj` | fails, gitignored artifact absent; zero package references even when present | Not a surface. Building through the tracked `.xcodeproj` would benchmark the divergent fork (D7), not the package, and ODC-0004 N7 separately shows it requires a Metal Toolchain component the SwiftPM route does not. |

The MLX backend was not exercised by any cell in the baseline matrix, in
either direction: the baseline's build matrix measured whether the package's
single library target compiles, not which backend a caller selects at
runtime, so it says nothing about whether `MLXInstance` initializes
successfully anywhere. That is `## Open questions` Q2, and this contract does
not assume an answer.

**Net position:** as of this draft, **no execution surface exists on which
this project has ever run inference with either backend.** This contract can
be authored, reviewed, and approved without one; it cannot be executed without
one. That is why `## Validation evidence` reports nothing.

## Hardware, OS, and toolchains

Recorded per session, never assumed inherited from a prior baseline or a prior
benchmark session, because a device can receive an OS update, a toolchain
update, or a thermal-policy change between sessions and any of those
invalidates a comparison.

| Field | Command |
| --- | --- |
| Xcode version and build | `xcodebuild -version` |
| Swift compiler | `swift --version` |
| Host and target triple | `swift --version`, `xcrun --sdk <sdk> --show-sdk-path` |
| iOS / iOS Simulator / macOS SDK builds | `xcodebuild -showsdks` |
| Device model identifier and chip | `sysctl -n hw.model machdep.cpu.brand_string` (simulator host) or `MobileGestalt`-equivalent device class only, no serial or UDID, matching ODC-0002's redaction policy |
| OS product, version, build | `sw_vers`, or the device's reported OS version |
| Total memory | `sysctl -n hw.memsize` or the device's reported physical memory |
| Thermal state at session start | `ProcessInfo.thermalState` (or the platform-equivalent call) sampled before the first run |
| Power source | charging or on battery, sampled per run |

No device serial, UDID, provisioning identifier, or personal device name is
ever captured, matching ODC-0002's `## Security, privacy, and redaction`
denylist verbatim. A device-class benchmark manifest is not a device-tracking
manifest.

## Workloads and metrics

A workload is a fixed prompt plus a fixed target generation length, shared
verbatim across every backend and every model paired against it. Workloads are
tracked text fixtures under `docs/benchmarks/fixtures/*.txt`, each named by a
`workload_id` and a `sha256` of its exact bytes, so "the same prompt" is a
checksum comparison, not an assertion. Every workload declares `max_tokens` as
an exact target, not a ceiling, so that token-count parity across backends is
a property of the workload definition, not a post-hoc truncation the harness
performs after the fact.

Six metrics, each defined so that two operators measuring the same run cannot
disagree about what they measured.

1. **Time to first token (TTFT).** Wall-clock duration, on a monotonic clock,
   from the instant the generation call is issued (immediately after prompt
   tokenization completes) to the instant the first generated token is
   delivered to the caller. TTFT necessarily includes prefill compute time; a
   report that describes it as excluding prefill is wrong by definition and
   the checker rejects any manifest field named to suggest otherwise.
   llama.cpp and MLX are not guaranteed to expose the same hook for "prompt
   tokenization completes," so the manifest's `ttft_start_hook` field records,
   per run, the specific function or call site the clock was actually started
   from (for example, the return of the tokenizer call, or the call site
   immediately preceding the generation request). This is the same discipline
   `prefill_duration_boundary` already applies to the prefill/decode boundary
   below, applied here so two honest implementers cannot silently instrument
   different instants and still both claim to satisfy this definition.
2. **Prefill throughput.** Prompt tokens per second, computed as
   `prompt_token_count / prefill_duration_seconds`. Where the backend exposes
   a distinct boundary between "prompt consumed" and "first decode step
   begins," that boundary is `prefill_duration`. Where it does not, `TTFT` is
   used as `prefill_duration` and the manifest's
   `prefill_duration_boundary` field records which case applies, per backend,
   every session. This field is required, not optional, because silently
   treating the two cases the same would make prefill throughput numbers
   across backends non-comparable without saying so.
3. **Decode throughput.** Steady-state tokens per second during decode,
   computed from per-token timestamps as
   `(generated_token_count - 1) / (timestamp_last - timestamp_first)`, which
   excludes the first-token latency already captured by TTFT. A backend that
   cannot report per-token timestamps cannot report decode throughput under
   this contract; an aggregate-only estimate is not accepted as a substitute,
   because it silently absorbs TTFT into the rate and biases it.
4. **Peak memory**, on a declared basis. See the dedicated requirement
   immediately below. Never a bare number.
5. **Model load time**, first-load-in-process and repeat-load-in-process
   distinguished. This metric does not claim, and this contract has no
   mechanism that could claim, anything about the OS disk or page cache
   state. Two independent reasons, stated plainly rather than left implicit:
   neither macOS nor iOS gives a sandboxed process a supported way to force
   the OS file cache cold the way ODC-0002 could force a SwiftPM cache cold
   with `--cache-path`, and, separately, `## Reproduction procedure` step 2
   requires a `sha256` verification pass over every model file before any
   load happens, which itself reads the file end to end and would populate
   any OS-level file cache that existed to populate, even on a platform that
   did expose one. No reordering of this contract's own procedure changes
   that: the checksum must be verified before a model is trusted enough to
   load, and verifying a checksum means reading the bytes. **First-load**:
   the first successful load of this exact model file, by checksum, since the
   harness process (and, on iOS, the app session) started; no prior load of
   that file has occurred in the current process lifetime. **Repeat-load**:
   any subsequent load of the same file in the same process lifetime. The
   manifest's `load_recurrence` field (schema enum
   `first-load-in-process` / `repeat-load-in-process`) is defined
   operationally, per the rule above, and every run records which case it
   measured rather than which case the operator intended. A reader who needs
   a genuinely cache-cold measurement will not find one in this contract;
   none is offered.
6. **Completion reason.** The stop reason the runtime itself reports for how
   generation ended, captured verbatim as a string from a declared, per-backend
   enumeration. That enumeration is not a prose list an operator is trusted to
   honor; it is a manifest field, `backends[].completion_reason_enum`, a
   required, non-empty array of strings (for example, natural stop, maximum
   tokens reached, stop sequence matched, cancelled, error) declared once per
   backend, per session, and both `runs[].metrics.completion_reason` and
   `runs[].correctness_gates.stop_reason` are checked against the matching
   backend's array by the checker, never accepted as self-certified by the
   harness. See `### Fields the checker computes, never accepts as authored`.
   The harness records exactly what the runtime returned; it never infers a
   reason from token counts or timing, and
   a value outside the declared enumeration is a harness defect, not a new
   category to paper over silently.

### The memory-basis requirement

This is methodology, not a finding, and it is required because two different
runtimes in this package allocate through different underlying mechanisms
(mapped model files versus GPU or driver-owned buffers, among others), so a
single memory-accounting API call is not guaranteed to observe everything a
given runtime holds resident. A benchmark that reports one number per backend
without saying how it was obtained invites a comparison neither number can
support.

- **Every peak-memory measurement declares its basis**: the exact OS API or
  tool and the exact field name it read, recorded as a string in the same
  manifest entry as the number, never in a footnote and never assumed
  identical across backends.
- **Peak memory is reported per backend, with its basis labeled in the same
  field.** A table that lists two numbers under one unlabeled "memory" column
  is not a citable table under this contract; see `## Raw artifact schema`.
- **Cross-backend memory comparison is permitted only when both numbers share
  a stated, matching basis, and the manifest says so explicitly.** Where the
  bases differ, or where matching validity has not been checked, the
  comparison entry is recorded with `comparable: false` and a
  `basis_mismatch` reason, not silently omitted and not silently merged. This
  is not a rule an operator is trusted to honor by hand: `comparable` is
  never accepted as written by the harness. The checker recomputes it from
  `bases[0] == bases[1]` and rejects any manifest that authors a mismatched
  `bases` pair alongside `comparable: true`; see `### Fields the checker
  computes, never accepts as authored`.
- **Every peak-memory entry additionally records**: the sampling cadence (peak
  is sampling-rate dependent, and an unrecorded cadence makes the number
  irreproducible), the measurement window (model-load only, load plus
  generation, or generation only), and whether the value is a single sample or
  a maximum over samples.
- **Where an independent cross-check instrument is available on the
  measurement platform** (as `vmmap`/`footprint`-class tooling is on macOS,
  outside any sandboxed app process), a session should record it alongside the
  primary basis so the primary reading's validity is falsifiable rather than
  assumed. Where no such cross-check exists on the platform being measured (as
  is the case for third-party code on iOS today), the manifest records
  `cross_check_available: false` explicitly rather than omitting the field,
  so a reader cannot mistake absence of the field for an unchecked assumption.

## Correctness gate

**No performance, throughput, load-time, or memory number from a run is
reported, aggregated, or written into a citable field until that run passes
all three gates below.** A run that fails any gate produces no number: its
metrics fields are `null`, its `citable` flag is `false`, and it is recorded
in `gate_failures[]` with the specific gate and reason, never coerced into a
zero, a placeholder, or an average with the passing runs.

1. **The model actually loaded.** The load call returned success, and the
   loaded model's runtime-reported identity (at minimum, whichever of
   architecture name, vocabulary size, parameter count, or context length the
   backend exposes) matches the corresponding `models[]` declaration. A file
   that opens without error but is the wrong file is not a passed gate.
2. **The intended backend and offload configuration were really used.** The
   harness queries the runtime for what it actually initialized on, not what
   was requested: active backend identity, GPU offload layer count or
   equivalent, and thread count where applicable. The observed configuration
   must equal the requested configuration field for field
   (`correctness_gates.backend_match`, `.offload_match`, computed by the
   checker per `### Fields the checker computes, never accepts as authored`,
   not self-declared). A silent fallback, for example a GPU request that
   actually ran on CPU, is a gate failure, not a result mislabeled by its
   requested configuration. Both the requested and observed values are
   recorded, so a fallback is diagnosable, not merely flagged.
3. **Tokens were genuinely generated.** The observed generated-token count
   matches the workload's `max_tokens` target, or generation ended earlier
   through one of the completion reasons declared valid for that backend in
   `## Workloads and metrics` item 6, checked against
   `backends[].completion_reason_enum` by the checker
   (`correctness_gates.stop_reason_valid`, same subsection). The produced
   output is non-empty and passes a basic sanity check appropriate to the
   backend's output type (for example, decodes to valid text, or every token
   id is within the declared vocabulary range). Zero tokens, a caught
   exception silently converted into an empty result, or a stop reason
   outside the declared enumeration all fail this gate.

Gate evaluation is per run, not a pre-session checklist and not a
post-session filter applied to convenient results. `## Abort criteria` states
what happens when a gate fails repeatedly rather than once.

## Warmup, repetitions, and statistical method

**Warmup.** Each arm (one backend, one model, one workload, one configuration)
executes one discarded warmup generation against the same workload
immediately after model load and before timed repetitions begin. The warmup
run is still subject to every correctness gate; a warmup that fails a gate
aborts the arm exactly as a timed run would, because a warmup that cannot pass
the gate says the arm cannot produce a number at all. Warmup exists to remove
first-call measurement artifacts from the decode and prefill statistics; it
does not apply to the model-load-time metric, which is a first-class result
and is captured on every load, first-load and repeat-load alike, never
discarded.

**Minimum repetitions, and what the floor is actually for.** Ten timed
repetitions per arm, per metric, is the floor. This is an engineering default
chosen for this contract, not inherited from any external study, and this
document does not claim the floor of ten alone achieves any specific
precision; it is deliberately the smaller of two numbers working together.
The precision target this contract actually commits to is downstream of the
CV escalation rule immediately below: an arm's aggregate for a metric is
accepted once its coefficient of variation is at or under `0.15`, using as
few as ten repetitions and as many as thirty, or is honestly labeled
`high_variance: true` if `0.15` is not reached even at the cap. Ten is the
minimum sample size below which computing any interval at all, bootstrap or
otherwise, is not attempted, not a claim that ten repetitions alone are
sufficient. If, after ten repetitions, the coefficient of variation
(`stdev / mean`) for a metric exceeds `0.15`, **and `stdev` is at or above
that metric's absolute floor** (see immediately below; this qualifier did not
exist before this revision and is load-bearing), the harness runs additional
repetitions up to a cap of thirty before accepting the arm's statistics for
that metric; if the coefficient of variation still exceeds the threshold at
the cap, the arm's aggregate for that metric is recorded with
`high_variance: true` rather than silently reported as stable. These
constants (`10`, `0.15`, `30`, and the absolute floors below) are named
harness parameters in the manifest, not folklore; changing them is a spec
revision, not a runtime flag decided silently.

**The CV escalation rule has an absolute-scale floor, because CV alone is not
enough.** Coefficient of variation is scale-free by construction: a
sub-millisecond mean with a fraction of a millisecond of ordinary timer
jitter can produce a CV above `0.15` from noise no engineer would call
meaningful, most plausibly for `model_load_ms` on a repeat-load or `ttft_ms`
on fast hardware. Escalating to thirty repetitions and permanently recording
`high_variance: true` for that reason would misrepresent noise as
instability. The escalation rule above therefore does not fire unless
`stdev` for that metric is also at or above a named absolute floor,
`harness_parameters.cv_absolute_floor` in `## Raw artifact schema`, one fixed
constant per metric. Below its floor, a metric's aggregate is accepted at the
ten-repetition minimum without escalation and is never flagged
`high_variance` on CV grounds alone; the floor does not silence a genuinely
unstable metric, because a metric unstable enough to matter will also have a
`stdev` above the floor.

**Required dispersion reporting.** Every aggregate reports median, mean,
sample standard deviation, minimum, maximum, and a 95 percent confidence
interval computed by the bias-corrected (BC) percentile bootstrap (10,000
resamples) around the median. This is a deliberate choice against the plainer
percentile bootstrap, not a default: `## Warmup, repetitions, and statistical
method`'s own next paragraph states decode and prefill timings are typically
right-skewed by rare, long-tail stalls, and the plain percentile method is
the bootstrap variant most sensitive to bias under exactly that kind of
skew. The bias-corrected variant corrects the interval's median bias without
requiring the additional jackknife-estimated acceleration term a full BCa
bootstrap would add, which is not worth its extra implementation cost at the
sample sizes (10 to 30) this contract actually collects. **Central tendency
alone is insufficient and the schema enforces it**:
`## Raw artifact schema` makes `stdev`, `min`, `max`, and both confidence
bounds required siblings of every reported median, so an aggregate object
missing them fails schema validation rather than merely looking incomplete
to a careful reader.

**Why the median, not the mean, is primary.** Decode and prefill timings are
typically right-skewed by rare, long-tail stalls (thermal throttling, a GC or
compaction pause, a background OS task), and a small number of such stalls
distorts a mean far more than a median. The mean is still reported, in full,
alongside it, so a reader who needs it is never blocked; it is simply not the
number used to answer "how fast is this."

## Confounders and fairness controls

Every control below is a fact a session's manifest must record; none is
optional and none may be inferred after the fact from an absent field.

1. **Pinned runtime revisions**, re-measured every session per `## Backends
   and pinned revisions`, never copied from a prior session's manifest.
2. **Model checksums**, verified against `models[]` before any run of that
   model begins; a mismatch aborts that model's arms entirely, per
   `## Abort criteria`, and is never treated as "close enough."
3. **Identical prompts and token counts**, guaranteed by construction: every
   arm compared against a given workload uses the exact same tracked prompt
   fixture, verified by its `sha256`, and the exact same `max_tokens` target.
4. **Interleaved execution order.** Arms are never run one backend fully, then
   the next. The session's arm sequence is randomized or round-robin across
   backends and models per repetition block, recorded verbatim as
   `execution_order[]`, and the checker rejects any manifest in which more
   than one consecutive run shares the same arm, so a thermal or cache
   advantage accumulated by running one backend first cannot silently bleed
   into its numbers.
5. **Thermal state recorded**, at the start and end of every run, not only at
   session start. A run beginning in a non-nominal thermal state is retained
   in the raw record (never deleted) and flagged `thermal-non-nominal` in
   `non_citable_reasons`; see `## Abort criteria` for when repeated
   throttling stops the session outright rather than merely flagging runs.
6. **First-load versus repeat-load, per process**, per the operational
   definition in `## Workloads and metrics` item 5, recorded per run, never
   assumed from session position (the first run in a session is not
   automatically a first-load-in-process measurement of every model; a prior
   session's process may still hold a given file's state on some platforms,
   and the harness must check, not assume). Restated because it matters here
   too: this control never claims anything about OS disk-cache state, only
   about position within this harness process's own load history.
7. **Device and OS identity**, recorded per session per `## Hardware, OS, and
   toolchains`, because a session that spans an OS update mid-run is
   measuring two different systems and must say so.
8. **Power source and thermal policy**, recorded per run; a session that mixes
   battery and charging runs within one arm's repetitions is recorded as such,
   not silently averaged together.
9. **Memory-basis parity across compared backends**, restated here as a
   fairness control rather than only a metric-definition rule: a comparison
   table is never assembled from two backends' peak-memory numbers unless
   `## Workloads and metrics`'s basis-parity condition is met and recorded.

## Raw artifact schema

Two deliverables, mirroring ODC-0002's convention exactly: a normative,
schema-validated JSON manifest, and a human-rendered Markdown report that
must contain, as literal substrings, every field in a correspondence list the
checker enforces. Proposed paths, siblings of the baseline's own naming:
`docs/benchmarks/v2.0.4-benchmark-results.json` and
`docs/benchmarks/v2.0.4-benchmark.md`. Neither is created by this ticket; both
are created by whichever ticket first executes this contract, gated on the
open questions in `## Open questions` closing.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "OnDeviceCatalyst cross-backend benchmark result manifest",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "captured_at", "session_id", "repo",
               "toolchain", "host", "sdks", "backends", "models", "workloads",
               "harness_parameters", "execution_order", "runs", "aggregates",
               "comparisons", "gate_failures", "excluded"],
  "properties": {
    "schema_version": { "type": "integer", "const": 1 },
    "captured_at": { "type": "string", "format": "date" },
    "session_id": { "type": "string", "minLength": 1 },
    "repo": {
      "type": "object", "additionalProperties": false,
      "required": ["revision", "dirty"],
      "properties": {
        "revision": { "type": "string", "pattern": "^[0-9a-f]{40}$" },
        "dirty": { "type": "boolean" }
      }
    },
    "toolchain": {
      "type": "object", "additionalProperties": false,
      "required": ["swift", "xcode_version", "xcode_build", "host_triple"],
      "properties": {
        "swift": { "type": "string" },
        "xcode_version": { "type": "string" },
        "xcode_build": { "type": "string" },
        "host_triple": { "type": "string" }
      }
    },
    "host": {
      "type": "object", "additionalProperties": false,
      "required": ["device_class", "os_product", "os_version", "os_build",
                   "memory_bytes", "thermal_policy_note"],
      "properties": {
        "device_class": { "type": "string" },
        "os_product": { "type": "string" },
        "os_version": { "type": "string" },
        "os_build": { "type": "string" },
        "memory_bytes": { "type": "integer", "minimum": 1 },
        "thermal_policy_note": { "type": "string" }
      }
    },
    "sdks": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["name", "canonical_name"],
        "properties": {
          "name": { "type": "string" },
          "canonical_name": { "type": "string" }
        }
      }
    },
    "backends": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["id", "name", "resolved_revision", "artifact_checksum",
                     "surface", "is_stub", "completion_reason_enum"],
        "properties": {
          "id": { "type": "string" },
          "name": { "enum": ["llama-cpp", "mlx"] },
          "resolved_revision": { "type": ["string", "null"], "pattern": "^[0-9a-f]{40}$" },
          "artifact_checksum": { "type": ["string", "null"], "pattern": "^[0-9a-f]{64}$" },
          "surface": { "enum": ["ios-device", "ios-simulator", "macos"] },
          "is_stub": { "type": "boolean" },
          "completion_reason_enum": {
            "type": "array", "minItems": 1, "uniqueItems": true,
            "items": { "type": "string", "minLength": 1 }
          }
        }
      }
    },
    "models": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["id", "backend_id", "architecture", "parameter_count",
                     "quantization", "context_length", "format", "sha256",
                     "equivalence_group"],
        "properties": {
          "id": { "type": "string" },
          "backend_id": { "type": "string" },
          "architecture": { "type": "string" },
          "parameter_count": { "type": "integer", "minimum": 1 },
          "quantization": { "type": "string" },
          "context_length": { "type": "integer", "minimum": 1 },
          "format": { "type": "string", "minLength": 1 },
          "sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
          "equivalence_group": { "type": "string" }
        }
      }
    },
    "workloads": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["id", "prompt_path", "prompt_sha256", "prompt_token_count",
                     "max_tokens"],
        "properties": {
          "id": { "type": "string" },
          "prompt_path": { "type": "string" },
          "prompt_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
          "prompt_token_count": { "type": "integer", "minimum": 1 },
          "max_tokens": { "type": "integer", "minimum": 1 }
        }
      }
    },
    "harness_parameters": {
      "type": "object", "additionalProperties": false,
      "required": ["min_repetitions", "max_repetitions",
                   "cv_escalation_threshold", "warmup_repetitions",
                   "cv_absolute_floor"],
      "properties": {
        "min_repetitions": { "type": "integer", "const": 10 },
        "max_repetitions": { "type": "integer", "const": 30 },
        "cv_escalation_threshold": { "type": "number", "const": 0.15 },
        "warmup_repetitions": { "type": "integer", "const": 1 },
        "cv_absolute_floor": {
          "type": "object", "additionalProperties": false,
          "required": ["ttft_ms", "prefill_tokens_per_s",
                       "decode_tokens_per_s", "peak_memory_bytes",
                       "model_load_ms"],
          "properties": {
            "ttft_ms": { "type": "number", "const": 5 },
            "prefill_tokens_per_s": { "type": "number", "const": 5 },
            "decode_tokens_per_s": { "type": "number", "const": 2 },
            "peak_memory_bytes": { "type": "number", "const": 1048576 },
            "model_load_ms": { "type": "number", "const": 20 }
          }
        }
      }
    },
    "execution_order": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["sequence_index", "arm_id"],
        "properties": {
          "sequence_index": { "type": "integer", "minimum": 0 },
          "arm_id": { "type": "string" }
        }
      }
    },
    "runs": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["run_id", "sequence_index", "arm_id", "backend_id",
                     "model_id", "workload_id", "load_recurrence",
                     "thermal_state_start", "thermal_state_end",
                     "power_source", "correctness_gates", "metrics",
                     "citable", "non_citable_reasons", "raw_log_path"],
        "properties": {
          "run_id": { "type": "string" },
          "sequence_index": { "type": "integer", "minimum": 0 },
          "arm_id": { "type": "string" },
          "backend_id": { "type": "string" },
          "model_id": { "type": "string" },
          "workload_id": { "type": "string" },
          "load_recurrence": { "enum": ["first-load-in-process", "repeat-load-in-process"] },
          "thermal_state_start": { "type": "string" },
          "thermal_state_end": { "type": "string" },
          "power_source": { "enum": ["charging", "battery"] },
          "correctness_gates": {
            "type": "object", "additionalProperties": false,
            "required": ["model_loaded", "model_identity_verified",
                         "requested_backend", "observed_backend",
                         "backend_match", "requested_offload",
                         "observed_offload", "offload_match",
                         "observed_token_count", "stop_reason",
                         "stop_reason_valid", "output_sane", "gate_passed"],
            "properties": {
              "model_loaded": { "type": "boolean" },
              "model_identity_verified": { "type": "boolean" },
              "requested_backend": { "type": "string" },
              "observed_backend": { "type": "string" },
              "backend_match": { "type": "boolean" },
              "requested_offload": { "type": "string" },
              "observed_offload": { "type": "string" },
              "offload_match": { "type": "boolean" },
              "observed_token_count": { "type": "integer", "minimum": 0 },
              "stop_reason": { "type": "string" },
              "stop_reason_valid": { "type": "boolean" },
              "output_sane": { "type": "boolean" },
              "gate_passed": { "type": "boolean" }
            }
          },
          "metrics": {
            "type": ["object", "null"], "additionalProperties": false,
            "required": ["ttft_ms", "ttft_start_hook", "prefill_tokens_per_s",
                         "prefill_duration_boundary", "decode_tokens_per_s",
                         "peak_memory_bytes", "peak_memory_basis",
                         "peak_memory_sample_cadence_ms",
                         "peak_memory_window", "cross_check_available",
                         "model_load_ms", "completion_reason"],
            "properties": {
              "ttft_ms": { "type": "number", "minimum": 0 },
              "ttft_start_hook": { "type": "string", "minLength": 1 },
              "prefill_tokens_per_s": { "type": "number", "minimum": 0 },
              "prefill_duration_boundary": {
                "enum": ["distinct", "coincides-with-first-token"]
              },
              "decode_tokens_per_s": { "type": "number", "minimum": 0 },
              "peak_memory_bytes": { "type": "integer", "minimum": 0 },
              "peak_memory_basis": { "type": "string", "minLength": 1 },
              "peak_memory_sample_cadence_ms": { "type": "number", "minimum": 0 },
              "peak_memory_window": {
                "enum": ["load-only", "load-plus-generation", "generation-only"]
              },
              "cross_check_available": { "type": "boolean" },
              "model_load_ms": { "type": "number", "minimum": 0 },
              "completion_reason": { "type": "string" }
            }
          },
          "citable": { "type": "boolean" },
          "non_citable_reasons": {
            "type": "array",
            "items": {
              "enum": ["missing-manifest-field", "raw-log-not-persisted",
                       "unpinned-revision", "single-run-not-replicated",
                       "gate-failed", "basis-mismatch",
                       "thermal-non-nominal", "interleaving-violated",
                       "high-variance-unresolved"]
            }
          },
          "raw_log_path": { "type": "string", "minLength": 1 }
        }
      }
    },
    "aggregates": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["arm_id", "backend_id", "model_id", "workload_id",
                     "metric", "unit", "n", "median", "mean", "stdev", "min",
                     "max", "ci95_low", "ci95_high", "high_variance",
                     "basis"],
        "properties": {
          "arm_id": { "type": "string" },
          "backend_id": { "type": "string" },
          "model_id": { "type": "string" },
          "workload_id": { "type": "string" },
          "metric": { "type": "string" },
          "unit": { "type": "string" },
          "n": { "type": "integer", "minimum": 10 },
          "median": { "type": "number" },
          "mean": { "type": "number" },
          "stdev": { "type": "number", "minimum": 0 },
          "min": { "type": "number" },
          "max": { "type": "number" },
          "ci95_low": { "type": "number" },
          "ci95_high": { "type": "number" },
          "high_variance": { "type": "boolean" },
          "basis": { "type": ["string", "null"] }
        }
      }
    },
    "comparisons": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["metric", "workload_id", "backend_ids", "bases",
                     "comparable", "reason"],
        "properties": {
          "metric": { "type": "string" },
          "workload_id": { "type": "string" },
          "backend_ids": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": { "type": "string" }
          },
          "bases": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": { "type": ["string", "null"] }
          },
          "comparable": { "type": "boolean" },
          "reason": { "type": ["string", "null"] }
        }
      }
    },
    "gate_failures": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["run_id", "gate", "reason", "backend_id", "model_id",
                     "workload_id"],
        "properties": {
          "run_id": { "type": "string" },
          "gate": { "enum": ["model_loaded", "backend_offload", "tokens_generated"] },
          "reason": { "type": "string" },
          "backend_id": { "type": "string" },
          "model_id": { "type": "string" },
          "workload_id": { "type": "string" }
        }
      }
    },
    "excluded": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["scope", "id", "reason"],
        "properties": {
          "scope": { "enum": ["run", "arm", "session"] },
          "id": { "type": "string" },
          "reason": { "type": "string" }
        }
      }
    }
  }
}
```

### Fields the checker computes, never accepts as authored

A small set of boolean and derived fields exist in the schema above so a
harness can record what it observed, but the schema alone cannot stop a
harness (or an operator hand-editing a manifest) from writing an
inconsistent value into them. For exactly these fields,
`scripts/check-benchmark.py` recomputes the value from the fields it derives
from and rejects the manifest (non-zero exit) if the authored value disagrees
with the recomputed one. None of the fields below is ever taken on faith from
the harness's own output; the harness's writing of them at capture time is a
convenience for a human reading the raw manifest before the checker runs, not
the fact that decides citability.

- **`comparisons[].comparable`**, derived as `bases[0] == bases[1]`, exact
  string match, both non-null. A `comparisons[]` entry whose two `bases`
  differ, or where either is `null`, must have `comparable: false`; the
  checker rejects any manifest where such an entry instead has
  `comparable: true`. This is the mechanical enforcement `## Workloads and
  metrics`'s memory-basis requirement describes in prose: a manifest cannot
  declare mismatched bases and mark the comparison usable at the same time,
  because the checker, not the harness, has the final word on the boolean.
- **`correctness_gates.stop_reason_valid`**, derived as membership: `true`
  exactly when `stop_reason` appears in the matching `backends[].id`'s
  `completion_reason_enum`. A `stop_reason` outside that backend's declared
  enumeration is exactly the "harness defect, not a new category to paper
  over silently" `## Workloads and metrics` item 6 describes, and the checker
  is what turns that sentence into a failing exit code rather than an
  unenforced expectation: `metrics.completion_reason` is checked the same
  way, against the same enumeration.
- **`correctness_gates.backend_match`**, derived as
  `observed_backend == requested_backend`, and **`.offload_match`**, derived
  as `observed_offload == requested_offload`, both exact string match, per
  `## Correctness gate` gate 2.
- **`correctness_gates.gate_passed`**, derived as the conjunction of
  `model_loaded`, `model_identity_verified`, `backend_match`,
  `offload_match`, `stop_reason_valid`, and `output_sane`.

`## Reproduction procedure` step 9 lists running these derivations as a named
checker responsibility, not an implicit side effect a reader has to infer.

### What makes a result non-citable

A run or an aggregate is `non-citable` (or, at the run level, `citable: false`)
when any of the following holds. This list is the enumeration behind
`non_citable_reasons`, and the checker computes it; it is never
self-declared by the harness at capture time.

- **Missing manifest fields.** Any required field absent or failing its
  schema constraint. An empty object satisfies nothing under this schema, by
  construction, because every array in `## Raw artifact schema` above carries
  `minItems` or is required outright.
- **Unpersisted logs.** `raw_log_path` does not resolve to a durable artifact
  outside the repository, or the per-token log it names is absent. A summary
  statistic with no retained raw record behind it is not reproducible and is
  not citable, regardless of how clean the summary looks.
- **Unpinned revisions.** Any of `backends[].resolved_revision`,
  `backends[].artifact_checksum`, or `models[].sha256` is null, or does not
  match a value independently re-derived from the pinned source at capture
  time. ODC-0002 finding D6, a branch pin unable to satisfy an exact
  requirement, is the public, concrete precedent for exactly this failure
  mode; this rule exists so its benchmark-side equivalent cannot recur
  silently.
- **A single unreplicated run for a noisy quantity.** Any aggregate with
  `n < 10` for `ttft_ms`, `prefill_tokens_per_s`, `decode_tokens_per_s`,
  `peak_memory_bytes`, or `model_load_ms`. `completion_reason` and the
  correctness-gate booleans are not statistical quantities and may be
  reported from a single run; every timing or memory number may not.
- **Basis mismatch.** A `comparisons[]` entry with `comparable: false`,
  produced whenever two backends' peak-memory bases differ or have not been
  checked for parity.
- **Thermal or power non-conformance left unresolved.** A run flagged
  `thermal-non-nominal` that is included in an aggregate without the
  aggregate itself being flagged `high_variance` or otherwise annotated is a
  checker failure, not a warning.
- **Interleaving violated.** Two or more consecutive entries in
  `execution_order[]` share the same `arm_id`.

## Abort criteria

Unlike a gate failure, which discards one run, an abort criterion stops
further data collection for an arm or a session, because continuing would
either be unsafe or would generate more data under a condition already known
to invalidate it.

| Criterion | Scope aborted | Reason |
| --- | --- | --- |
| Any correctness gate fails on two consecutive runs of the same arm | Arm | The arm cannot produce the work this contract measures; escalating repetitions would only accumulate more non-citable runs. |
| A model checksum does not match its `models[]` declaration | That model's arms, session-wide | Every subsequent run would measure an unverified file. |
| Thermal state is non-nominal at the start of three consecutive runs and does not return to nominal after a bounded cooldown wait | Session | Continuing would confound every remaining measurement with an uncontrolled thermal state; the session resumes only after a fresh nominal start. |
| The harness's own build or deploy step fails | Session, before any run | A harness defect produces no partial data; nothing is recorded as a result. |
| Two consecutive entries in `execution_order[]` share the same `arm_id` | Session | Interleaving is a precondition of every fairness claim this contract makes, not a nicety; a scheduler defect here is a procedure defect, not a soft warning. |
| Power source changes mid-arm between charging and battery without being a declared, intentional condition of that arm | That arm's remaining repetitions | Power state affects thermal and performance state; an undeclared mid-arm change is a confound, not noise to average out. |

An abort is recorded in `excluded[]` with its scope and reason. It is never
silently dropped, and a session that aborts is still reported: `## Honest
reporting rules` (folded into `## Acceptance criteria` below) requires that an
aborted arm appear in the report as aborted, not omitted as if it had never
been attempted.

## Reproduction procedure

1. Re-verify every pin named in `## Backends and pinned revisions` against the
   current tree: `swift package describe --type json`, the XCFramework
   `Info.plist`, and `Package.resolved`. A session that skips this step is
   measuring an assumed identity, not a verified one.
2. Verify the model manifest named by `ODC_BENCHMARK_MODEL_MANIFEST` exists,
   and that every entry's `sha256` matches the file on disk, before any model
   is loaded. A mismatch aborts that model per `## Abort criteria`. This step
   reads every model file end to end; that is unavoidable for a checksum and
   is exactly why `## Workloads and metrics` item 5 defines its load-time
   metric the way it does, rather than as a claim about disk-cache state this
   step would already have falsified.
3. Capture `## Hardware, OS, and toolchains` and confirm thermal state is
   nominal before the first run.
4. Build and deploy the harness. This ticket's benchmark cases live entirely
   under a new top-level directory, `Benchmarks/`, as their own SwiftPM test
   target (for example `OnDeviceCatalystBenchmarks`), added to `Package.swift`
   by whichever ticket first executes this contract; not this spec, and not
   `Tests/OnDeviceCatalystTests/`, which stays exclusively ODC-0004's per
   `## Question and allowed claim`. Build and deploy follow the same pattern
   ODC-0004 documents for its own target: `swift build --build-tests` for the
   target triple, repackaged into a flat iOS bundle, executed through the
   platform's `xctest` agent via `simctl` (simulator) or `devicectl` (device),
   pointed at `OnDeviceCatalystBenchmarks` instead of `OnDeviceCatalystTests`.
   A second `.testTarget` entry in the same, single `Package.swift` is not
   the "second build system" ODC-0004 rejects `xcodebuild` to avoid; it
   introduces no new tracked project file that must be kept in agreement with
   `Package.swift` by hand (there is only ever one `Package.swift`, and it
   already lists both targets) and no dependency on the Metal Toolchain
   component. This is a design decision, not an open question, so it is
   stated as one.
5. Run the harness's own self-tests before trusting any of its output; see
   `## Harness self-tests`. A self-test failure aborts the session before any
   arm runs, per `## Abort criteria`'s "harness's own build or deploy step
   fails" row, because a checker or runner that cannot pass its own tests has
   no claim on any number it would go on to produce.
6. Generate the session's `execution_order[]` by randomized interleaving
   across every arm, then execute one warmup repetition per arm (discarded,
   still gated), then timed repetitions per `## Warmup, repetitions, and
   statistical method` until each metric reaches its required `n` or its
   variance-escalation cap.
7. Evaluate `## Correctness gate` live, per run. A failing run is recorded in
   `gate_failures[]` with `metrics: null`; it is never retried silently in
   place of the recorded failure.
8. Compute `aggregates[]` and `comparisons[]` once every arm's repetitions are
   complete or aborted.
9. Render the human report from the manifest, following ODC-0002's
   correspondence convention: the report must contain, as literal substrings,
   every `backends[].resolved_revision`, `models[].sha256`,
   `aggregates[].median` paired with its metric and arm, and every
   `comparisons[].comparable` paired with its reason.
10. Run the checker (`scripts/check-benchmark.py`, to be built by whichever
    ticket first executes this contract) against both deliverables. Its exit
    code is the gate. The checker's responsibilities include, and are not
    limited to, every derivation named in `### Fields the checker computes,
    never accepts as authored`: recomputing `comparisons[].comparable` from
    `bases` and rejecting any manifest where the authored value disagrees;
    validating `metrics.completion_reason` and `correctness_gates.stop_reason`
    against the matching `backends[].completion_reason_enum`; and recomputing
    `correctness_gates.backend_match`, `.offload_match`, and `.gate_passed`.

No step of this procedure downloads model weights into the repository, writes
build products inside the working tree, or leaves the tree dirty; a session
that leaves `git status --porcelain` non-empty has not conformed to this
contract regardless of what its numbers say.

## Harness self-tests

`## Validation evidence` is explicit that no execution surface has ever run
inference with either backend under this project. That is a statement about
model inference, not about this contract's own logic, and ODC-0004 has
already shown the iOS Simulator stub runs code deterministically today,
returning null or zero without trapping, for every call. That is a cheap,
already-available way to verify this contract's harness and checker are
correct before any scarce device time is spent debugging the harness itself
rather than measuring anything, and this contract requires it rather than
leaving it optional, mirroring ODC-0004's own convention for
`scripts/test-run-characterization.sh` and
`scripts/test-check-characterization.py`.

Two self-test deliverables, both additive under `scripts/**`, both built by
whichever ticket first executes this contract, alongside
`scripts/check-benchmark.py`:

- **`scripts/test-check-benchmark.py`.** Unit tests for the checker, runnable
  with no device, no simulator, and no model asset: schema validation against
  hand-constructed fixture manifests (a valid manifest, an empty object per
  A3, a manifest with `comparisons[].comparable: true` and mismatched `bases`
  per `### Fields the checker computes, never accepts as authored`, a
  `stop_reason` outside its backend's `completion_reason_enum`, a `stdev`
  below `cv_absolute_floor` paired with a CV over `0.15`), asserting the
  checker accepts the valid fixture and rejects each broken one with the
  specific reason named, not merely a non-zero exit.
- **A stub-based smoke test of `## Correctness gate`.** Using the iOS
  Simulator llama.cpp stub ODC-0004 N2/N3 already characterizes as
  deterministic, build and run a single minimal benchmark case against it and
  assert `## Correctness gate` gate 1 correctly fails
  (`model_loaded: false` or `model_identity_verified: false`, because the
  stub performs no real load) rather than passing or hanging. This is a
  CI-runnable regression test of the harness's own gate logic, independent of
  `## Open questions` Q1 and Q2 closing: it proves the gate rejects a known
  non-answer before either question is ever resolved, and a future change
  that silently weakens gate 1 breaks this test immediately rather than
  surfacing only when a real model is finally available.

## Acceptance criteria

This ticket is a contract, not an executed benchmark, so its own acceptance
criteria decide whether the contract is well-formed and enforceable, not
whether a number exists. Every criterion names a deciding command.

| # | Criterion | Deciding command |
| --- | --- | --- |
| A1 | The spec file exists and carries the required front matter | `test -f docs/specs/ODC-0003-benchmark-contract.md` |
| A2 | The manifest schema in `## Raw artifact schema` is itself valid JSON Schema | `python3 -c "import json,jsonschema; s=json.load(open('/dev/stdin')); jsonschema.Draft202012Validator.check_schema(s)"` fed the extracted schema block |
| A3 | An empty object `{}` does not satisfy the schema | `python3 -c "import json,jsonschema; jsonschema.validate({}, json.load(open('schema.json')))"` raises `ValidationError` |
| A4 | Every metric in `## Workloads and metrics` has a corresponding required field in `runs[].metrics` | manual cross-check enforced by `scripts/check-benchmark.py --schema-coverage` once the checker exists |
| A5 | Every fairness control in `## Confounders and fairness controls` has a corresponding manifest field or checker rule | `scripts/check-benchmark.py --fairness-coverage` |
| A6 | No em dash appears in this document | `python3 -c "import sys; sys.exit(1 if chr(0x2014) in open('docs/specs/ODC-0003-benchmark-contract.md').read() else 0)"` exits `0` |
| A7 | No confidential research term appears in this document outside this row's own description of the check | fenced command block immediately below, "A7 deciding command" |
| A8 | This ticket changes no runtime or test source, and touches only its own row in `Tickets.md` | fenced command block immediately below, "A8 deciding command" |
| A9 | Project state is consistent | `python3 scripts/validate-project-state.py` |
| A10 | The harness commits to self-testing itself, per `## Harness self-tests` | not yet satisfiable: `python3 scripts/test-check-benchmark.py` once the checker and its self-tests are built, mirroring A4/A5's status |

Deliberately absent, with reasons: **any criterion asserting a benchmark ran,
passed a gate, or produced a number.** None did. `## Validation evidence`
states this rather than a criterion softening it away.

**A7 deciding command.** Presented as a fenced block rather than embedded in
a table cell, because the command itself contains a shell pipe, and a
markdown-table cell cannot hold a literal `|` without escaping it, which the
previous revision of this row did with `\|`, a backslash-escaped literal
character, not a working pipe. A reader copying that escaped form got a
`grep` invocation fed nonexistent filenames, not the self-exclusion the row
described. This form has no table cell to escape around:

```bash
grep -Ein "jetsam|phys_footprint|resident_size|ceiling escape|victim.rank|os_proc_available_memory" docs/specs/ODC-0003-benchmark-contract.md | grep -v '| A7 |'
```

Outputs nothing when this row's own description of the check (naming the
denylist terms so the check itself is auditable) is the only place any of
those terms appear.

**A8 deciding command.** The original row anchored this check to a fixed
historical revision, `7b5d847`. That is the wrong kind of anchor for "did
this repair round touch anything it shouldn't": other tickets' rows in
`Tickets.md` change legitimately and often, for reasons that have nothing to
do with this ticket, and a diff against a fixed past commit accumulates every
one of those unrelated changes forever, so the check would eventually fail
for a reason this ticket did not cause, the same class of self-contradiction
this finding was raised to fix. This revision anchors to `HEAD` instead,
which names whatever this working tree's last commit is at the moment the
criterion is checked, immediately before this ticket's own changes are
committed. That is exactly the question this criterion asks: are there
uncommitted changes to a disallowed path staged or unstaged right now. The
check has two parts: no diff at all under runtime or test paths, and,
separately, a `Tickets.md` diff that touches only the `ODC-0003` row, because
the ordinary act of advancing this ticket necessarily changes that row and a
path-level check alone cannot tell "this ticket's own row changed" from
"some other ticket's row changed in the same window":

```bash
git diff --stat HEAD -- Sources Tests Package.swift Package.resolved OnDeviceCatalyst OnDeviceCatalyst.xcodeproj
git diff HEAD -- Tickets.md | grep -E '^[+-]' | grep -v '^+++' | grep -v '^---' | grep -v 'ODC-0003'
```

Both must produce no output. Run this once, right before committing this
ticket's own changes; it is not meant to be replayed later against a moving
history the way a fixed-revision diff would be.

## Review record

- 2026-09-02, drafted against revision `7b5d847`, after ODC-0002 reached
  `DONE`. Drafting included a read-only review of private research
  methodology restricted to `.context/research/specs/ODR-0006-preregistration.md`
  sections 2.3 and 5, and `.context/research/results/2026-09-01-o2-measurement-audit.md`,
  solely to avoid designing a memory metric around a measurement approach
  already known privately to be unreliable. No file under `.context/research/`
  was modified. No hypothesis, finding, measured value, or private reasoning
  chain from either document is reproduced in this spec. What the private
  review did inform, precisely, so this claim is not only about what was
  copied but about what was structural: `## Workloads and metrics`'s
  memory-basis requirement (always declare a basis, report bases side by
  side, never silently merge, mark non-comparable explicitly) takes its
  *shape* from the private methodology's own discipline for cross-runtime
  measurement, even though its stated reason here is given independently, in
  general terms, and no private finding, number, or hypothesis name appears.
  `## Question and allowed claim` states the content boundary this draft was
  held to; this sentence states the structural one.
- 2026-09-02, pass two, adversarial spec review, verdict REJECT, returned to
  `REVISION`. Artifact:
  [`docs/reviews/ODC-0003-review-pass-2.md`](../reviews/ODC-0003-review-pass-2.md).
  Six blocking findings, five major, one minor.
- 2026-09-02, revision: this rewrite. Blocking findings 1 through 6 are
  resolved at `## Workloads and metrics` item 5 (cold-load renamed to
  first-load-in-process, disk-cache claim withdrawn), item 6 and `### Fields
  the checker computes, never accepts as authored` (`completion_reason_enum`
  and its checker-side enforcement), the same subsection (`comparisons[].
  comparable` derivation), `## Acceptance criteria`'s A7 and A8 fenced
  deciding commands, and `## Question and allowed claim` plus `##
  Reproduction procedure` step 4 (the `Benchmarks/` directory boundary with
  ODC-0004). Majors are resolved at `## Prior art and freshness date` (the
  soft ODC-0004 dependency, stated explicitly), `## Warmup, repetitions, and
  statistical method` (the precision target, the bias-corrected bootstrap,
  and the CV absolute floor), `## Workloads and metrics` item 1
  (`ttft_start_hook`), and `## Harness self-tests` (new section). The minor
  finding is resolved by this record's second bullet above.
- Founder review: pending. `founder_approved` is `pending` and this spec
  carries open questions, so it cannot enter an approved status until both
  close.

## Validation evidence

Not implemented, and not merely unimplemented in the ordinary "code not
written yet" sense: **no execution surface capable of running inference with
either backend has ever been measured by this project**, per
`## Backends and pinned revisions`. `ROADMAP.md` operating gate 5, correctness
is established before performance is measured, is not satisfied for either
backend on any surface today, and this contract is written so that fact is
structurally impossible to paper over: `## Correctness gate` refuses a number
from a run that has not done the work, and no run has been attempted.

Closing this gap depends on two things outside this spec's control, both
already named as `Q1` and `Q2` in the front matter and repeated here so they
are not lost to a header:

- **Q1.** Physical iOS device availability, the same constraint ODC-0004
  registers as its own Q1. Deciding command: `xcrun devicectl list devices`,
  plus a signing identity check. Until a device is available, `ios-device` is
  a compiling cell, not an execution surface, and this contract has no legal
  surface to run on at all, since `ios-simulator` cannot infer with llama.cpp
  and no surface has ever been confirmed for MLX.
- **Q2.** Whether `MLXInstance` initializes and performs inference on any
  measured surface. Deciding command: build and run a minimal harness that
  calls `MLXInstance`'s initialization path with a real model asset, on
  whichever device Q1 resolves, and record the result under
  `## Correctness gate` gate 1 and gate 2. Until this closes, the `mlx`
  backend entry in a session's `backends[]` is unverified, not merely
  unmeasured.

This spec can be approved without either question closing, per program rule:
an approved spec states its open questions rather than assuming them away.
Execution cannot begin until both are closed, and `## Reproduction procedure`
step 1 through 3 are exactly the commands that close them.
