---
id: ODC-0012
title: Loading stream never terminates, impossible gate
type: bug
status: APPROVED
milestone: P0
owner: unassigned
dependencies: ODC-0002, ODC-0004
founder_approved: delegated-to-manager-2026-09-01
last_updated: 2026-09-06
evidence_fresh_until: 2026-09-20
unresolved_questions: none
---

# ODC-0012: Loading stream never terminates, impossible gate

## Reproduction

At revision `e9d16a4` (working tree clean):

The gate itself is executable at `R0`/`R1` with no build beyond the test
target, and its failure-path consequence is executable at `R2` on the
simulator stub, per
[`docs/specs/ODC-0004-v2-characterization-suite.md`](ODC-0004-v2-characterization-suite.md):

```
bash scripts/run-characterization.sh --surface simulator
```

Three already-landed, already-passing cases pin this defect today:

- `Tests/OnDeviceCatalystTests/R1/LoadProgressGateCharacterizationTests.swift:44`
  (`test_characterizes_publishProgressGate_isUnsatisfiableForEveryCase__ODC_0012`)
  transcribes the gate's predicate verbatim and shows it is `false` for all
  four `LoadProgress` cases.
- `Tests/OnDeviceCatalystTests/R2/InitializationFailureCharacterizationTests.swift:176`
  (`test_characterizes_failurePathStream_terminatesWithoutTheGateFiring__ODC_0012`)
  shows the failure-path stream still ends, but not because the gate fired.
- `Tests/OnDeviceCatalystTests/R3/InstanceLifecycleCharacterizationTests.swift:24`
  (`test_characterizes_afterReady_loadingStream_doesNotTerminate__ODC_0012`)
  states the success-path claim but always skips on the simulator surface
  today (`SKIP[requires-device]`), because reaching `.ready` needs a model
  that actually loads, which the simulator stub can never do (spec N3: the
  stub's `_llama_load_model_from_file` returns null unconditionally).

Read-only confirmation performed for this spec without a build:

```
sed -n '580,588p' "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift"
sed -n '95,123p'  "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift"
```

## Expected and actual behavior

Expected: `LlamaInstance.initialize()`'s `AsyncStream<LoadProgress>`
terminates exactly once, immediately after the terminal progress value
(`.ready` or `.failed`) is delivered, on every code path: success, primary
failure with no fallback, and fallback attempt (success or failure).

Actual: the stream's only intended terminator, the gate inside
`publishProgress`, can never fire, because it requires a single
`LoadProgress` value to match two mutually exclusive enum cases at once.
The **success path currently does not terminate the stream at all** under
any circumstance. The failure path happens to terminate today, but only as
an accidental side effect of a second, separate defect (D8, ODC-0015), not
because this gate does anything. See `## Root cause evidence` for both
claims with citations.

## Root cause evidence

- `Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:580-586`:

  ```swift
  private func publishProgress(_ progress: LoadProgress) {
      loadingContinuation?.yield(progress)

      if case .ready = progress, case .failed = progress {
          loadingContinuation?.finish()
          loadingContinuation = nil
      }
  }
  ```

  A comma between two `case` pattern-match conditions inside one `if` is a
  logical AND (Swift's [conditional-list
  syntax](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/statements/#Conditional-Clauses)),
  not an OR. `progress` is a single `LoadProgress` value; it cannot
  simultaneously match the `.ready` pattern and the `.failed` pattern, which
  are distinct cases of the same enum. The compound condition is `false` for
  every value of every case: `.preparing`, `.loading`, `.ready`, and
  `.failed` alike. This is dead code that can never execute its body, in a
  private method that ships in every build; nothing above it, including
  `LoadProgress` itself (`:592-613`, note the existing `isComplete` computed
  property at `:598-604` that already expresses "either `.ready` or
  `.failed`" correctly), diagnoses this.

- **What the gate was meant to express.** `LoadProgress` (`:592-613`)
  already defines exactly this predicate as `isComplete`:

  ```swift
  public var isComplete: Bool {
      switch self {
      case .ready, .failed:
          return true
      case .preparing, .loading:
          return false
      }
  }
  ```

  The evident intent of the `publishProgress` gate is "finish the
  continuation once a terminal state has been reached," i.e. exactly
  `progress.isComplete`. The `if case ..., case ...` form is the kind of
  mistake that results from trying to write "either of these two patterns"
  using the comma-separated conditional-clause syntax, which performs AND,
  not the `,` inside a single `case` pattern (which can express OR for
  associated-value-free cases, e.g. `case .ready, .failed:` in a `switch`,
  as `isComplete` already does above) or explicit `||`.

- **The success path does not terminate today.** `performInitialization`
  (`:80-132`) calls `publishProgress(.ready("Model ready for inference"))`
  at `:125` and then returns, with no further statement. Nothing else in the
  success path calls `cleanup()` or otherwise touches `loadingContinuation`.
  The only two places that ever call `loadingContinuation?.finish()` are the
  (dead) gate at `:583-584` and `cleanup()` at `:246-247`. `cleanup()` is
  called from `handleInitializationError` (`:188`, only on a failure path),
  `shutdown()` (`:232-235`, only on explicit external shutdown), and
  `deinit` (`:58-60`, only on deallocation). None of these runs on a
  successful `performInitialization()`. Consequently, after `.ready` is
  delivered, the `AsyncStream`'s continuation is simply held open,
  referenced by `self.loadingContinuation`, indefinitely: **a successful
  model load never ends its own loading stream.** A consumer awaiting
  stream termination (for example `for await progress in stream { }`
  without an explicit `break` on `.isComplete`) hangs forever. This is
  independently confirmed by ODC-0004's own characterization
  (`C-D3-3` / `test_characterizes_afterReady_loadingStream_doesNotTerminate__ODC_0012`,
  currently unexecuted on `SFC-B` because the simulator stub can never reach
  `.ready`, see `## Reproduction`) and by the baseline
  (`docs/baselines/v2.0.4.md` D3: "the continuation is never finished on
  success").

- **The failure path terminates today, but not via this gate.** In
  `handleInitializationError` (`:184-196`), `cleanup()` runs at `:188`,
  which unconditionally finishes and nils `loadingContinuation`
  (`:246-247`), **before** either the fallback branch or the direct
  `publishProgress(.failed(...))` branch (`:194`) executes. So on the
  non-recoverable failure path, the stream is already finished by the time
  `publishProgress(.failed(...))` runs; that call is a no-op against a nil
  continuation, and the gate this ticket exists to fix never gets a chance
  to evaluate meaningfully either way. `test_characterizes_failurePathStream_terminatesWithoutTheGateFiring__ODC_0012`
  (`Tests/OnDeviceCatalystTests/R2/InitializationFailureCharacterizationTests.swift:176-193`)
  measures exactly this: the stream terminates, but no `.ready` or `.failed`
  value ever precedes that termination, so the termination is attributable
  to `cleanup()`, not to the gate. This is D8 (ODC-0015), and it is the
  reason D3's failure-path breakage has not, until ODC-0004's suite, been
  independently visible: D8 currently masks it.

## In-scope repair

Not a comma fix in isolation; a predicate-intent fix. Replace the
unsatisfiable compound condition with the predicate `LoadProgress` already
defines for exactly this purpose:

```swift
private func publishProgress(_ progress: LoadProgress) {
    loadingContinuation?.yield(progress)

    if progress.isComplete {
        loadingContinuation?.finish()
        loadingContinuation = nil
    }
}
```

This one-line change is sufficient at the call site, but its consequence is
not one-line: for the first time, `publishProgress(.ready(...))` on the
success path (`performInitialization:125`) actually finishes the
continuation, and `publishProgress(.failed(...))` calls that are not
pre-empted by a still-open continuation actually finish it too. That is the
correct termination this ticket delivers: the stream ends immediately after
its own terminal value, on every path that reaches a terminal `publishProgress`
call at all.

What this repair does **not** by itself fix: the failure path
(`handleInitializationError`) still calls `cleanup()` before dispatching to
either branch (`:188`), so `publishProgress(.failed(...))` on the
non-recoverable branch, and every `publishProgress` call inside
`attemptFallbackInitialization` on the recoverable branch, still run against
an already-nil `loadingContinuation`. Fixing the gate alone does not resolve
D8; it only makes the gate meaningful for the one caller that is not
pre-empted, namely the success path. D8's repair is ODC-0015, and the two
tickets must land in the order stated in `## Review record`.

## Non-goals

- Does not reorder or otherwise touch `handleInitializationError` or
  `cleanup()`. That is ODC-0015's scope (D8).
- Does not change `LoadProgress`'s cases, `isComplete`, or `message`.
- Does not change `generate()` / `performGeneration` / `generateTokens`
  (the generation stream). That is ODC-0011's scope (D2); it is a
  structurally separate `AsyncThrowingStream`.
- Does not add a new terminal state, a timeout, or a cancellation path to
  `initialize()`. The fix makes the existing two terminal states
  (`.ready`, `.failed`) actually terminate the stream; it does not add
  behavior beyond that.
- Does not resolve Q1/Q3 (device-execution mechanism), needed to un-skip
  `test_characterizes_afterReady_loadingStream_doesNotTerminate__ODC_0012`
  in CI. That is ODC-0021's scope, inherited here per ODC-0004's
  ticket-allocation table if neither ODC-0010 nor ODC-0011 has already
  discharged it.

## Regression risks

- **Low, for the predicate change itself.** `isComplete` is public,
  already used elsewhere in the codebase (for example
  `Service Layer/Catalyst.swift:152`, `:207`, `:399`, all `if
  progress.isComplete`), and already exercised by this same suite's other
  cases. Reusing it rather than hand-rolling a new condition removes a
  chance to introduce a second, differently wrong predicate.
- **Medium, for any code that depended on the loading stream never ending
  on success.** A grep across the package
  (`grep -rn 'for await progress in' Sources/`) finds every present consumer
  already guards with `if progress.isComplete { break }` or equivalent
  (`Service Layer/Catalyst.swift:150-153`, `:205-208`, `:397-400`,
  `:655-658`), so none of them was relying on the stream running forever;
  each already expects to see a terminal value and stop. This repair makes
  that expectation true rather than accidentally true. No consumer in this
  repository needs a corresponding change.
- **The interaction risk is with ODC-0015, not with this ticket's own
  callers.** See `## Review record`.

## Tests

Per ODC-0004's ticket-allocation table, this ticket updates `C-D3-1`,
`C-D3-2`, `C-D3-3`, and fingerprint `F-D3-1`, in the same commit as the
repair.

| Test | Class / surface | Today | Must assert after this repair |
| --- | --- | --- | --- |
| `test_characterizes_publishProgressGate_isUnsatisfiableForEveryCase__ODC_0012` | `D3LoadProgressGateCharacterizationTests`, `R1` | Transcribes the old predicate (`if case .ready = progress, case .failed = progress`) and asserts it is `false` for all four cases. | The mirrored predicate (`publishProgressGateMirror`) must be updated to `progress.isComplete`'s logic, and the assertion inverted: `true` for `.ready` and `.failed`, `false` for `.preparing` and `.loading`. Rename to a `test_requires_` regression case. |
| `test_characterizes_failurePathStream_terminatesWithoutTheGateFiring__ODC_0012` | `D3FailurePathTerminationCharacterizationTests`, `R2` | Asserts the failure-path stream terminates with no `.ready`/`.failed` value ever observed. | **Does not flip from this ticket alone.** As `## Root cause evidence` establishes, the non-recoverable failure path is currently pre-empted by D8's premature `cleanup()` before this gate ever gets a chance to run meaningfully; fixing only the gate leaves `cleanup()` still finishing the continuation first. This assertion flips only once ODC-0015 also lands (see `## Review record`). Recorded here, not just there, because the obligation table assigns this test ID to this ticket, and the honest state is that this ticket alone cannot discharge it. |
| `test_characterizes_afterReady_loadingStream_doesNotTerminate__ODC_0012` | `D3ReadyStreamLifecycleCharacterizationTests`, `R3` | Always `XCTSkip("SKIP[requires-device] ...")` on `SFC-B`. | Assertion body must change from an inverted expectation ("does not terminate within the bound") to "the stream finishes after `.ready`, within the bound." This is the one `C-D3-*` case this ticket's fix, alone, actually makes true; it does not flip in CI today only because it is not executed in CI today (`R3`), not because the fix does not apply to it. |

`F-D3-1` (`docs/characterization/v2-fingerprints.json`, anchored on
`publishProgress`) must be regenerated from the post-repair source and its
`sha256` updated in the same commit, or
`python3 scripts/check-characterization.py --fingerprints` fails by design.

## Acceptance criteria

| ID | Criterion | Deciding command |
| --- | --- | --- |
| A1 | `publishProgress`'s gate is satisfiable for `.ready` and `.failed` and unsatisfiable for `.preparing` and `.loading`, per the updated fingerprint and the updated `C-D3-1` case. | `python3 scripts/check-characterization.py --fingerprints` exits 0 |
| A2 | The success path terminates the loading stream: `test_characterizes_afterReady_loadingStream_doesNotTerminate__ODC_0012`'s updated assertion holds whenever it executes (skip is acceptable pending ODC-0021; a hang or an assertion failure is not). | `bash scripts/run-characterization.sh --surface simulator` exits 0 |
| A3 | The suite's own inventory of defects and tests stays internally consistent. | `python3 scripts/check-characterization.py --naming --inventory --require-defects D1,D2,D3,D4,D5,D6,D7,D8,N1` exits 0 |
| A4 | No file outside `Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift`, `Tests/OnDeviceCatalystTests/**`, and `docs/characterization/v2-fingerprints.json` is touched by the repair commit. | `git diff --stat <repair-commit>~1 <repair-commit> -- . ':!Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift' ':!Tests/OnDeviceCatalystTests' ':!docs/characterization/v2-fingerprints.json'` outputs nothing |
| A5 | Project state stays consistent. | `python3 scripts/validate-project-state.py` exits 0 |

## Review record

Not yet reviewed. Drafted alongside ODC-0011 and ODC-0015 because all three
sit in `LlamaInstance`'s stream lifecycle.

**Ordering: this ticket must land before ODC-0015, not after and not
simultaneously without sequencing.** The two are tightly coupled through the
same private method pair (`publishProgress`, `handleInitializationError`),
and the coupling is directional:

- If ODC-0012 lands first, alone: the success path is fixed (a real
  improvement, `C-D3-3`'s eventual concern). The failure path is unaffected,
  because D8's premature `cleanup()` (`:188`) still finishes the
  continuation before either failure branch's `publishProgress` call runs;
  those calls remain no-ops, exactly as characterized today
  (`C-D3-2`, `C-D8-1`, `C-D8-2`, `C-D8-3` all continue to pass unchanged).
  Landing this ticket alone introduces no new defect and regresses nothing.
- If ODC-0015 landed first, alone, without this ticket's gate fix: traced
  against ODC-0015's own repair design (`docs/specs/ODC-0015-fallback-events-dropped.md`
  `## In-scope repair`), the two branches behave differently, not
  identically. The **non-recoverable branch** publishes `.failed(...)` and
  then explicitly runs a safety-net resource-release call "equivalent to
  today's `cleanup()`, including finishing `loadingContinuation` as a
  safety net" - that finish is unconditional, independent of this gate's
  state, so that branch would still terminate the stream even with this
  ticket's fix absent; it would not hang. The **recoverable/fallback
  branch**, on either of its two outcomes (fallback succeeds and calls
  `publishProgress(.ready(...))` with no cleanup call following it, so "the
  fallback's own resources must survive"; or fallback fails and calls
  `publishProgress(.failed(...))` with resource release routed into that
  branch's own catch, not paired with a continuation-finishing safety net)
  has no unconditional finish following its terminal `publishProgress`
  call. Against this ticket's still-unsatisfiable gate, that terminal value
  would be yielded but the continuation would never finish: those two
  outcomes would **hang** rather than silently drop their terminal event. A
  caller with a bounded wait would never observe termination at all on
  those two outcomes, which is a worse outcome under
  [`docs/requirements/memory-and-admission.md`](../requirements/memory-and-admission.md)'s
  R4 ("admission failure is explicit, early, and actionable") than today's
  silent-drop-but-still-terminates behavior. ODC-0015's spec must not be
  implemented ahead of this one for that reason, even though its
  non-recoverable branch alone would not be affected.

  This is independently confirmed by the actual harness every `R2`/`R3`
  case in this suite uses to decide "terminated":
  `Tests/OnDeviceCatalystTests/Support/StreamRecorder.swift:28-50`'s
  `record(_:timeout:)` drains via a bare `for await element in stream {
  events.append(element) }` with no early break on element content -
  termination is decided purely by the stream itself finishing, or a
  watchdog `Task.sleep` cancelling the collector after the liveness bound.
  This means the test harness itself would hang for the full bound (20s in
  `InitializationFailureCharacterizationTests.swift:26`) on exactly the two
  outcomes identified above, and would not hang on the non-recoverable
  branch, which is drained to closure by its own safety-net `cleanup()`
  regardless of this gate.

Landing order: **ODC-0012, then ODC-0015.** ODC-0011 is independent of both
and may land in any position relative to them; see its own spec's review
record.

**D1 interaction (informational, not addressed by this ticket).** D1
(cache-then-shutdown race, ODC-0010) lives entirely in
`Service Layer/Catalyst.swift`'s `releaseInstance` (`:495-522`: the cache
insert at `:507` runs inside `if !forceShutdown && instance.isReady`, with
an async `Task { await instance.shutdown() }` at `:510-512` that has no
happens-before relationship to that cache insert). This ticket does not
touch `releaseInstance`, `shutdown()`, or `cleanup()`'s core teardown body,
so D1's race is neither fixed nor made more likely here. There is a small,
second-order effect worth recording so a later reader does not assume D1 is
addressed by this ticket: today, D3 (this ticket's own defect) and D8
(ODC-0015) together mean a load failure often delivers no `.failed` event,
so `Catalyst.generate()`'s existing `if case .failed(let message) =
progress { ... throw }` branch (`:399-403`) cannot fire, and a caller has
no trigger to call `releaseInstance(forceShutdown: true)` (`:210`). Once
this ticket and ODC-0015 both land, that branch starts firing correctly on
failure, routing more calls through `releaseInstance(forceShutdown: true)`,
which (per `:504`) skips the `if !forceShutdown && instance.isReady` branch
entirely - the exact branch that creates D1's race. So landing this ticket
together with ODC-0015 very slightly reduces exposure to D1; it does not
fix it, and D1 remains open, tracked under ODC-0010 / ODC-0101.

**Concurrency.** Same governing facts as ODC-0011's review record:
`LlamaInstance` is a plain, non-`Sendable` class under Swift 5 language mode
with minimal `Sendable` enforcement (`Package.swift` declares no
`swiftLanguageMode` or strict-concurrency `swiftSettings`; confirmed by
inspection). `publishProgress` is a private method invoked only from within
the single `Task` created by `initialize()` (`:70-78`) and from
`attemptFallbackInitialization`, itself only reachable from that same `Task`
via `handleInitializationError`. This repair changes only the boolean
condition evaluated inside that already-serial call chain; it introduces no
new shared mutable state and no new cross-task access to
`loadingContinuation` beyond what already exists. Safe to land under the
current concurrency model; does not need to wait for ODC-0101. The
pre-existing hazard that a second, concurrent call to `initialize()` (or a
`shutdown()` racing a still-running `performInitialization()`) could mutate
`loadingContinuation` and `backend` without any compiler-enforced isolation
is real, but it is not introduced or worsened by this one-line predicate
change; it is the same class of hazard ODC-0002 and ODC-0004 recorded under
D1, and its general resolution is ODC-0101's, not this ticket's.

**Proposed ledger row for `Tickets.md`.** The `ODC-0012` row's `Spec` column
becomes `[spec](docs/specs/ODC-0012-loading-stream-never-terminates.md)`;
its `Status` column becomes `SPEC_DRAFT` to match this file's frontmatter;
its `Dependencies` column becomes `ODC-0002, ODC-0004`; its `Next Gate`
column becomes `review`.

## Validation evidence

None yet; `status: SPEC_DRAFT`. The reproduction and root-cause evidence
above were gathered by read-only inspection (`sed`, `grep`) against revision
`e9d16a4` and by reading the already-landed, already-passing (or
already-correctly-skipped) characterization tests named in `## Tests`; no
build was run to produce this spec.

### Correction: the C-D3-1 rename is not implementable as worded

This spec asked for C-D3-1 to be renamed to a `test_requires_` case once the
repair landed. **That instruction cannot be carried out**, and the implementer
correctly refused to force it by editing files outside their scope.

Two independent rules in `scripts/check-characterization.py` block it:

1. Its naming rule forbids a `test_requires_` name from containing `__ODC_`.
2. It hardcodes the literal name
   `test_characterizes_publishProgressGate_isUnsatisfiableForEveryCase__ODC_0012`
   as C-D3-1's inventory entry.

So the rename fails `--naming` and `--inventory` simultaneously, with no in-scope
fix available to an implementer restricted to source and tests.

**Resolution adopted:** the rename is withdrawn. The test keeps its name and
class, and only its internal predicate, assertions and comment block were flipped
to the post-repair expectation. That satisfies the behavioural intent of the flip
table, which is what matters; the name is a label, not a contract.

If the naming taxonomy should distinguish repaired characterizations from
unrepaired ones, that is a change to `scripts/check-characterization.py` and to
ODC-0004's naming convention, and belongs in its own ticket rather than being
smuggled into a one-line bug fix.
