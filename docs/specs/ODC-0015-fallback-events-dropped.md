---
id: ODC-0015
title: Fallback-path progress events are silent no-ops
type: bug
status: SPEC_DRAFT
milestone: P0
owner: unassigned
dependencies: ODC-0002, ODC-0004, ODC-0012
founder_approved: pending
last_updated: 2026-09-06
evidence_fresh_until: 2026-09-20
unresolved_questions: Q1
---

# ODC-0015: Fallback-path progress events are silent no-ops

## Reproduction

At revision `e9d16a4` (working tree clean), executable at `R2` on the
simulator surface with no model weights and no device, because the stub's
deterministic null return (`docs/specs/ODC-0004-v2-characterization-suite.md`
N3) makes the failure path run every time:

```
bash scripts/run-characterization.sh --surface simulator
```

Four already-landed, already-passing cases in
`Tests/OnDeviceCatalystTests/R2/InitializationFailureCharacterizationTests.swift`
pin this defect today:

- `test_characterizes_recoverableFailure_loadingStream_deliversNoTerminalEvent__ODC_0015`
  (`:64-76`): with a `phi-probe.gguf` fixture (filename-classified
  `.architectureUnsupported`, `isRecoverable == true`), the loading stream
  yields `preparing`, `loading`, `loading`, then terminates with neither
  `.ready` nor `.failed` ever delivered.
- `test_characterizes_nonRecoverableFailure_loadingStream_deliversNoTerminalEvent__ODC_0015`
  (`:88-100`): identical result with a `generic-probe.gguf` fixture
  (`.modelLoadingFailed`, `isRecoverable == false`).
- `test_characterizes_recoverableAndNonRecoverableFailures_areExternallyIndistinguishable__ODC_0015`
  (`:112-125`): the two fixtures' event-kind sequences are asserted equal.
- `test_characterizes_afterStreamEnds_instanceIsNotReadyWithNoFurtherEvent__ODC_0015`
  (`:143-155`): after the stream ends, `instance.isReady` is `false` with no
  diagnostic ever surfaced.

Read-only confirmation performed for this spec without a build:

```
sed -n '184,249p' "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift"
```

## Expected and actual behavior

Expected: a consumer of the loading stream can tell, from the stream alone,
(a) whether the model ultimately became ready, (b) if not, why, and (c)
whether a recoverable class of failure triggered a fallback attempt versus a
non-recoverable failure that did not. All three are properties `Service
Layer/Catalyst.swift` already assumes: its consumers (`:150-153`,
`:205-208`, `:397-400`) each read the stream expecting `if case .failed(let
message) = progress { throw ... }` to eventually fire on failure.

Actual: none of the above is observable. Every failure, recoverable or not,
produces the identical three-event sequence (`preparing`, `loading`,
`loading`) and then the stream silently ends. `.ready` and `.failed` are
never delivered on any measured failure path. A recoverable failure that
successfully falls back and a non-recoverable failure that never attempts to
are, from outside the instance, indistinguishable, and neither is
distinguishable from an outright hang, except that this one actually
terminates. `Catalyst`'s `if case .failed(...) { throw ... }` branches can
never fire, so a caller proceeds with a `LlamaInstance` whose `isReady` is
`false` and receives no error.

## Root cause evidence

- `Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:184-196`:

  ```swift
  private func handleInitializationError(_ error: CatalystError) async {
      print("Catalyst: Initialization failed - \(error.localizedDescription)")

      // Attempt cleanup
      cleanup()

      // Try fallback initialization if error is recoverable
      if error.isRecoverable {
          await attemptFallbackInitialization(originalError: error)
      } else {
          publishProgress(.failed(error.localizedDescription))
      }
  }
  ```

  `cleanup()` runs unconditionally at `:188`, **before** the branch on
  `error.isRecoverable` at `:191`. `cleanup()` (`:237-248`) unconditionally
  executes `loadingContinuation?.finish(); loadingContinuation = nil`
  (`:246-247`). By the time either branch runs, `loadingContinuation` is
  already `nil`. `publishProgress` (`:580-586`) reads
  `loadingContinuation?.yield(progress)`; against a `nil` continuation this
  is a safe, silent no-op. So:
  - The non-recoverable branch's `publishProgress(.failed(...))` at `:194`
    never delivers anything.
  - Every `publishProgress` call inside `attemptFallbackInitialization`
    (`:198-229`, including its own terminal `publishProgress(.ready(...))`
    at `:224` and `publishProgress(.failed(...))` at `:227`) never delivers
    anything either, because the continuation was already finished and
    nilled two calls before `attemptFallbackInitialization` even began.

- This is a pure ordering defect: `cleanup()` at `:188` runs strictly before
  `attemptFallbackInitialization` at `:192` on every call, regardless of
  `error.isRecoverable`, and `cleanup()`'s side effect on
  `loadingContinuation` is what silences everything downstream.

- Confirmed, measured, on the simulator stub (spec N5;
  `Tests/OnDeviceCatalystTests/R2/InitializationFailureCharacterizationTests.swift`):
  both the recoverable-class and non-recoverable-class fixtures produce the
  identical three-event-then-silent-termination sequence, and the
  after-effects match `isReady == false` with the stream already ended.

## In-scope repair

The defect is an ordering bug, but the minimal-looking fix, moving
`cleanup()` to run unconditionally *after* the branch instead of before it,
is wrong for a different reason than D3's naive comma-swap, and it is
recorded here precisely because it is the fix an implementer is likely to
reach for first:

```swift
// WRONG: would tear down a successful fallback
private func handleInitializationError(_ error: CatalystError) async {
    if error.isRecoverable {
        await attemptFallbackInitialization(originalError: error)
    } else {
        publishProgress(.failed(error.localizedDescription))
    }
    cleanup()   // <-- runs even after attemptFallbackInitialization succeeded
}
```

`attemptFallbackInitialization` (`:198-229`) assigns `self.backend =
newBackend` and `samplingEngine = SamplingEngine(...)` (`:214-215`) *before*
its own terminal `publishProgress(.ready(...))` at `:224`. If
`handleInitializationError` unconditionally calls the full resource-tearing
`cleanup()` after the branch returns, a **successful** fallback would have
its brand-new, working backend immediately shut down and nilled
(`cleanup()`'s `backend?.shutdown(); backend = nil` at `:238-239`) right
after being declared ready. That trades D8 (silence on failure) for a new
and worse defect: a reported success that is immediately, invisibly
undone. This is exactly the class of mistake this ticket's evidence review
exists to catch before it ships, in the same spirit as D3's "not merely a
comma."

The repair must therefore distinguish the two things `cleanup()` currently
conflates: releasing backend/sampling-engine resources, and finishing the
loading continuation. Only the second was ever supposed to happen at
`:188`; the first belongs to whichever branch actually ends in failure.

1. Remove the unconditional `cleanup()` call at `:188`, ahead of the branch.
2. On the non-recoverable branch (`error.isRecoverable == false`): publish
   `.failed(error.localizedDescription)` first, then release resources
   (equivalent to today's `cleanup()`, including finishing
   `loadingContinuation` as a safety net). Order here matters for the
   reason ODC-0012 exists: with this ticket's dependency on ODC-0012's gate
   fix (`progress.isComplete`, not the unsatisfiable AND), publishing
   `.failed(...)` while `loadingContinuation` is still open lets the gate
   itself finish the stream with the real terminal value observed first.
   A subsequent, redundant `cleanup()` call finding the continuation
   already nil is a harmless no-op.
3. On the recoverable branch (`error.isRecoverable == true`): before calling
   `attemptFallbackInitialization`, release only `backend` and
   `samplingEngine` (shut down and nil them) if the primary attempt had
   already assigned them before the recoverable error was thrown.
   `isRecoverable` is `true` for, among others, `.contextCreationFailed`,
   `.memoryInsufficient`, and `.samplingFailed` (`Core Foundation/CatalystError.swift:89-97`);
   on the primary path `self.backend` and `samplingEngine` are assigned at
   `:110` and `:114`, strictly before `warmup()` runs at `:118`, so a
   recoverable error surfaced by warmup (plausible for `.memoryInsufficient`
   or `.samplingFailed`, even though the simulator stub never exercises
   warmup at all, per spec N3) would otherwise leave a live, orphaned
   backend behind the moment `attemptFallbackInitialization` overwrites
   `self.backend` at `:214` with its own new instance. This resource-only
   release must not touch `loadingContinuation`, or it reintroduces this
   ticket's own defect one branch earlier. With that done,
   `attemptFallbackInitialization` runs with `loadingContinuation` left open
   and a clean backend slot, and it already correctly manages both of its
   own outcomes from there:
   - success: assigns the new backend/sampling engine, then
     `publishProgress(.ready(...))`, which (post-ODC-0012) now finishes the
     continuation with the real success value. No cleanup call should run
     after this branch; the fallback's own resources must survive.
   - failure (its `catch` at `:226-227`): publishes
     `.failed("Both primary and fallback initialization failed: ...")`,
     which (post-ODC-0012) finishes the continuation with the real failure
     value. Resource release for whatever partial state the failed fallback
     attempt left behind (see Q1 below) belongs inside this catch branch,
     not in `handleInitializationError`, since only
     `attemptFallbackInitialization` knows whether it got far enough to
     assign a new backend before failing.
4. `handleInitializationError` itself no longer calls `cleanup()`
   unconditionally; each branch is responsible for triggering resource
   release exactly when its own outcome is failure, and never when its own
   outcome is success.

This repair depends on ODC-0012 landing first, as stated in `## Review
record`: steps 2 and 3 above only produce a correctly-terminating stream
because the gate inside `publishProgress` is assumed fixed
(`progress.isComplete`). Landing this repair against today's still-broken
gate would remove the one thing (`cleanup()`'s blunt, early finish) that
currently makes the stream terminate at all on the failure paths, turning
today's silent-drop into a hang, which is the exact regression ODC-0012's
own review record warns against creating.

## Non-goals

- Does not repair the gate itself (`if case .ready = progress, case .failed
  = progress`). That is ODC-0012 (D3), and this ticket depends on it.
- Does not change `error.isRecoverable`'s classification logic
  (`Core Foundation/CatalystError.swift:89-102`) or
  `LlamaBridge.createModelLoadingError`'s filename-based branching
  (`API Bridge/LlamaBridge.swift:65-100`). Whether "contains `phi`" is a
  sound way to classify architecture-unsupported errors is a separate,
  unaddressed question this ticket does not open.
- Does not change `attemptFallbackInitialization`'s fallback settings
  (context length, batch size, GPU layers) or its retry strategy.
- Does not add a new `LoadProgress` case to distinguish "fell back and
  succeeded" from "succeeded on the first attempt," or "fell back and
  failed" from "failed with no fallback attempted," beyond what the
  existing message strings already convey (`"Model ready with fallback
  settings..."` vs. `"Model ready for inference"`; `"Both primary and
  fallback initialization failed: ..."` vs. the primary error's own
  message). A consumer can already distinguish these cases by string
  content once they are delivered at all; adding a structured field is a
  reasonable follow-on but is out of scope for restoring delivery.
- Does not repair D2 (ODC-0011) or, beyond the stated dependency, D3
  (ODC-0012) itself.

## Regression risks

- **The fallback-success-gets-torn-down hazard, named explicitly above, is
  the primary risk this spec exists to prevent**, not merely to document.
  Any implementation of this ticket must demonstrate, in its test
  evidence, that a successful fallback leaves `isReady == true` and the
  fallback's backend alive; today's suite has no case that reaches fallback
  success (the simulator stub fails deterministically at `loadModel`, so
  fallback always fails too, on the only surface measured so far), so this
  specific claim is currently `R3`-only and unmeasured. See Q1.
- **Medium: resource leak on a failed fallback.** If
  `attemptFallbackInitialization`'s `try newBackend.loadModel(...)` (or
  `createContext`, or `warmup`) throws after `self.backend = newBackend`
  was already assigned (possible only if `warmup()` is the failing call,
  since `self.backend` is assigned before `warmup()` runs, at `:214` versus
  `:220`), the `catch` branch at `:226-227` today only calls
  `publishProgress(.failed(...))` and does not tear down that partially
  initialized `newBackend`. This repair does not newly introduce that gap,
  it already exists in `attemptFallbackInitialization` today, but moving
  resource-release responsibility into each branch (per `## In-scope
  repair`, step 3) makes it the natural place to close it, and an
  implementation that does not is leaving a known gap open rather than
  closing one it happened to touch. Recorded, not silently left for a
  reader to notice later.
- **Medium: a primary backend leaking into a fallback attempt, newly
  introduced if step 3's release-before-dispatch is omitted.** Unlike the
  two risks above, which are pre-existing gaps this repair merely inherits,
  this one is a risk this repair's own removal of the unconditional
  `cleanup()` at `:188` would create if implemented carelessly. Today,
  `cleanup()` running before every dispatch means a primary attempt that
  got as far as assigning `self.backend`/`samplingEngine` (`:110`, `:114`)
  before failing with a recoverable error always has that backend shut down
  before `attemptFallbackInitialization` overwrites `self.backend` at
  `:214`. An implementation of this ticket that removes `cleanup()` from
  `:188` without adding the resource-only release step 3 requires would
  leak that primary backend silently. `## In-scope repair` step 3 exists
  specifically to close this, and any implementation must be checked
  against it.
- **Low, for the non-recoverable branch.** That branch never assigns a new
  backend before failing (the error originates from
  `performInitialization`'s own `try newBackend.loadModel(...)` /
  `createContext` / `warmup` calls, all before `self.backend` is set at
  `:110`, except when the failure is itself a post-assignment warmup
  failure on the *primary* attempt, which is the same class of gap as
  above, pre-existing, not introduced here).

## Tests

Per ODC-0004's ticket-allocation table, this ticket updates `C-D8-1`,
`C-D8-2`, `C-D8-3`, and fingerprint `F-D8-1`, in the same commit as the
repair. Because this ticket depends on ODC-0012, `C-D3-2` (assigned to
ODC-0012's obligations, but only truly discharged once this ticket also
lands, per ODC-0012's own `## Tests` table) is included here as well for
completeness, not as a duplicate obligation.

| Test | Class / surface | Today | Must assert after this repair |
| --- | --- | --- | --- |
| `test_characterizes_recoverableFailure_loadingStream_deliversNoTerminalEvent__ODC_0015` | `D8InitializationFailureCharacterizationTests`, `R2` | Asserts exactly 3 events, no `.ready`/`.failed`, on the `phi-probe.gguf` fixture. | The recoverable path must attempt fallback and, on the stub (which fails fallback's `loadModel` too, deterministically), deliver a fourth event (`.loading("Retrying with fallback settings")`) followed by a terminal `.failed(...)` event. Assertion becomes: the sequence ends in `.failed`, and a terminal event is observed. Rename to `test_requires_`. |
| `test_characterizes_nonRecoverableFailure_loadingStream_deliversNoTerminalEvent__ODC_0015` | `D8InitializationFailureCharacterizationTests`, `R2` | Asserts exactly 3 events, no `.ready`/`.failed`, on the `generic-probe.gguf` fixture. | Assertion becomes: the sequence ends in `.failed(error.localizedDescription)` directly, with no fallback-retry event in between. Rename to `test_requires_`. |
| `test_characterizes_recoverableAndNonRecoverableFailures_areExternallyIndistinguishable__ODC_0015` | `D8InitializationFailureCharacterizationTests`, `R2` | Asserts the two fixtures' event-kind sequences are equal. | Assertion inverts: the two sequences must now differ (the recoverable path has one extra `loading` event for the retry, both end in `.failed` on the stub, but at different lengths and via different call paths). Rename to `test_requires_recoverableAndNonRecoverableFailures_areDistinguishable`. |
| `test_characterizes_afterStreamEnds_instanceIsNotReadyWithNoFurtherEvent__ODC_0015` | `D8InitializationFailureCharacterizationTests`, `R2` | Asserts `isReady == false` and no further event, silently. | Assertion becomes: a terminal event (`.failed`, on the stub) precedes stream termination, and `isReady` reflects that outcome (`false`, since the stub's fallback also fails). The "no further event" half stays true; the "no terminal event ever" half is what flips. |
| `test_characterizes_failurePathStream_terminatesWithoutTheGateFiring__ODC_0012` | `D3FailurePathTerminationCharacterizationTests`, `R2` (owned by ODC-0012, discharged jointly) | Asserts termination with no `.ready`/`.failed` observed. | Assertion inverts to: a `.failed` value is observed before termination, and termination is attributable to the (now-fixed) gate rather than to `cleanup()`. This is the test that makes the ODC-0012/ODC-0015 dependency concrete and checkable. |

`F-D8-1` (`docs/characterization/v2-fingerprints.json`, anchored on
`handleInitializationError`) must be regenerated from the post-repair source
and its `sha256` updated in the same commit.

A new `R3` regression test should be added (no existing `C-D8-*` ID covers
it) asserting that a **successful** fallback leaves `isReady == true` and
the fallback's backend alive after the loading stream terminates with
`.ready`; this can only run once a real device and a model asset are
available (Q1, ODC-0021).

## Acceptance criteria

| ID | Criterion | Deciding command |
| --- | --- | --- |
| A1 | `handleInitializationError` no longer calls `cleanup()` unconditionally ahead of the recoverable/non-recoverable dispatch, per the updated fingerprint. | `python3 scripts/check-characterization.py --fingerprints` exits 0 |
| A2 | On the simulator stub, both the recoverable-class and non-recoverable-class fixtures now deliver a terminal `.failed` event before the loading stream terminates, and the two event sequences differ in shape. | `bash scripts/run-characterization.sh --surface simulator` exits 0 |
| A3 | The suite's own inventory of defects and tests stays internally consistent. | `python3 scripts/check-characterization.py --naming --inventory --require-defects D1,D2,D3,D4,D5,D6,D7,D8,N1` exits 0 |
| A4 | No file outside `Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift`, `Tests/OnDeviceCatalystTests/**`, and `docs/characterization/v2-fingerprints.json` is touched by the repair commit. | `git diff --stat <repair-commit>~1 <repair-commit> -- . ':!Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift' ':!Tests/OnDeviceCatalystTests' ':!docs/characterization/v2-fingerprints.json'` outputs nothing |
| A5 | Project state stays consistent. | `python3 scripts/validate-project-state.py` exits 0 |

## Review record

Not yet reviewed. Drafted alongside ODC-0011 and ODC-0012 because all three
sit in `LlamaInstance`'s stream lifecycle.

**Ordering: this ticket must land after ODC-0012, never before it and never
in the same uncoordinated change without respecting that order.** Restated
from ODC-0012's own review record because the dependency is load-bearing
for this ticket's correctness, not just a sequencing preference: this
ticket's repair removes the one mechanism (`cleanup()`'s premature,
blunt `loadingContinuation?.finish()`) that currently makes the failure
paths terminate at all. If ODC-0012's gate fix (`progress.isComplete`) is
not already in place, removing that mechanism replaces a silent, terminated
stream (today's D8 symptom) with a stream that never terminates (D3's
symptom, now hitting the failure paths it does not hit today). That is a
strictly worse outcome under
[`docs/requirements/memory-and-admission.md`](../requirements/memory-and-admission.md)
R4, which requires admission failure to be explicit, early, **and
actionable**; a hang is none of the three, while today's silent drop is at
least early (it happens promptly, it is just unreported).

ODC-0011 has no ordering dependency with this ticket; it touches a disjoint
pair of methods (`performGeneration`, `generateTokens`) in the same file and
may land in any position.

**What ODC-0012 landing changes about what this ticket must emit,
concretely.** Before ODC-0012: any `publishProgress` call is either
silenced by an already-nil continuation (today) or, hypothetically, would
hang against the unsatisfiable gate (if this ticket alone reordered
`cleanup()`). After ODC-0012: every `publishProgress(.ready(...))` or
`publishProgress(.failed(...))` call this ticket's repair allows to reach a
still-open continuation becomes the actual, sole terminator of the stream.
This ticket's design in `## In-scope repair` is written against that
post-ODC-0012 behavior; it would need to be re-derived, not merely
re-ordered, if ODC-0012 changed its own approach (for example if it chose
to terminate via a different mechanism than fixing `publishProgress`'s
gate in place).

**Concurrency.** Same governing facts as the other two specs in this group:
`LlamaInstance` is a plain, non-`Sendable` class under Swift 5 language mode
with minimal `Sendable` enforcement. `handleInitializationError`,
`attemptFallbackInitialization`, and `cleanup()` are all reachable only from
the single `Task` created inside `initialize()` (`:70-78`), invoked
serially: `performInitialization` awaits `handleInitializationError`, which
awaits `attemptFallbackInitialization`. This repair redistributes which
branch calls which subset of `cleanup()`'s existing effects; it does not
introduce a second concurrent writer of `backend`, `samplingEngine`, or
`loadingContinuation`, and does not change how many tasks touch `self`.
Safe to land under the current concurrency model; does not need to wait for
ODC-0101. The pre-existing hazard of a concurrent `shutdown()` (a different
entry point onto `cleanup()`, callable from any task, per `:232-235`) racing
a still-running `performInitialization()` is real, unguarded by any actor
isolation today, and is not introduced or worsened by this ticket; it is
D1-adjacent territory and belongs to ODC-0101's general concurrency model,
not to this narrow ordering fix.

**Proposed ledger row for `Tickets.md`.** The `ODC-0015` row's `Spec` column
becomes `[spec](docs/specs/ODC-0015-fallback-events-dropped.md)`; its
`Status` column becomes `SPEC_DRAFT` to match this file's frontmatter; its
`Dependencies` column changes from `ODC-0002` to `ODC-0002, ODC-0004,
ODC-0012`, adding ODC-0012 to record the load-bearing ordering established
above and ODC-0004 because this ticket's tests live inside that suite; its
`Next Gate` column becomes `review`.

Q1 (unresolved): whether `attemptFallbackInitialization` must additionally
tear down a partially assigned `newBackend`/`samplingEngine` when `warmup()`
throws after `:214-215` but before `publishProgress(.ready(...))` at
`:224`, so that `isReady` cannot remain `true`-by-accident after a reported
fallback failure. Flagged under `## Regression risks` as pre-existing and
not measured on any surface reached so far (the stub fails at `loadModel`,
never reaches `warmup`); needs `R3` (real hardware, a model that loads but
fails warmup) to characterize before it can be decided as in- or
out-of-scope. Left open rather than silently absorbed into this ticket's
repair.

## Validation evidence

None yet; `status: SPEC_DRAFT`. The reproduction and root-cause evidence
above were gathered by read-only inspection (`sed`, `grep`) against revision
`e9d16a4` and by reading the already-landed, already-passing characterization
tests named in `## Tests`; no build was run to produce this spec.
