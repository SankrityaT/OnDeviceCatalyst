---
id: ODC-0011
title: Generation emits duplicate terminal completions
type: bug
status: SPEC_DRAFT
milestone: P0
owner: unassigned
dependencies: ODC-0002, ODC-0004
founder_approved: pending
last_updated: 2026-09-06
evidence_fresh_until: 2026-09-20
unresolved_questions: Q1
---

# ODC-0011: Generation emits duplicate terminal completions

## Reproduction

At revision `e9d16a4` (working tree clean):

Executable today, no device or model required, per
[`docs/specs/ODC-0004-v2-characterization-suite.md`](ODC-0004-v2-characterization-suite.md)'s
`R1` partition. Run the consumer-side contract cases:

```
bash scripts/run-characterization.sh --surface simulator
```

The three cases in
`Tests/OnDeviceCatalystTests/R1/StreamContractCharacterizationTests.swift`
already reproduce the defect's consumer-visible consequence without building
or executing this ticket's fix; they pin it. The producer-side reproduction
(`generateTokens` and `performGeneration` both yielding a completion for the
same generation) needs real inference and is recorded, unexecuted, as `R3` in
`Tests/OnDeviceCatalystTests/R3/GenerationCharacterizationTests.swift`
(`test_characterizes_boundedGeneration_yieldsTwoCompletionChunks__ODC_0011`,
`test_characterizes_secondCompletion_reportsNaturalEvenAfterMaxTokensReached__ODC_0011`),
both of which always skip on the simulator surface today
(`SKIP[requires-device]`) because no device-execution mechanism exists yet
(spec ODC-0004 Q1/Q3, absorbed by ODC-0021).

Read-only confirmation of the source shape, performed for this spec without a
build:

```
sed -n '365,392p' "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift"
sed -n '443,540p' "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift"
```

## Expected and actual behavior

Expected: one bounded generation produces exactly one terminal `StreamChunk`
whose `isComplete` is `true`, carrying the real reason generation stopped and
the full `ResponseMetadata` (token counts, duration, tokens/sec).

Actual: every generation call yields **two** terminal chunks.

1. `generateTokens` (`Core Engine/LlamaInstance.swift:443-537`) yields a
   completion at one of five sites, each carrying the true
   `CompletionReason` and no metadata:
   - `:466` `.userCancelled`, on interrupt
   - `:487` `.natural`, on end-of-generation token
   - inside `processor.processToken(tokenText)`'s returned chunks (loop at
     `:499-509`), which can include `.stopSequenceFound`
   - `:517` `.contextWindowFull`
   - `:535` `.maxTokensReached`, when the loop exhausts `maxTokens`

   In every one of these five cases, `generateTokens` **returns normally**
   (an `Int` token count) rather than throwing, so control returns to
   `performGeneration`.

2. `performGeneration` (`:283-393`) then unconditionally executes its own
   completion at `:372-386`: it computes a full `ResponseMetadata` from
   `tokensGenerated` and the elapsed duration, builds a completion chunk
   hardcoding `reason: .natural` (`:384`), yields it, and calls
   `continuation.finish()` (`:386`).

The consequence is not symmetric across consumer styles, and both are wrong
in different ways:

- A consumer that **breaks** on the first `isComplete` chunk
  (`StreamResponse.swift`'s `collectResponse()` at `:275-286` and
  `collectContent()` at `:289-300`, and the in-repo call site at
  `Service Layer/Catalyst.swift:481-483`) sees the true reason from
  `generateTokens`, but **never** sees the `ResponseMetadata` this method
  computes, because that metadata exists only on the second chunk it never
  reaches.
- A consumer that **drains** the stream to its natural end (for example
  manually calling `StreamingResponse.addChunk` in a loop with no `break`,
  which overwrites `completionReason`/`metadata` on every `isComplete` chunk
  it sees) ends up recording `reason: .natural` regardless of why generation
  actually stopped, together with the real metadata, which happens to be
  attached to the wrong reason.

Two consumers of the identical stream disagree about why generation stopped.
Metadata (`tokensGenerated`, `tokensPerSecond`, `generationTimeMs`,
`promptTokens`, `totalTokens`) and the true stop reason are never both
observable together by any consumer today.

## Root cause evidence

- `Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:365-386`:

  ```swift
  tokensGenerated = try await generateTokens(
      maxTokens: maxNewTokens,
      config: config,
      backend: backend,
      continuation: continuation
  )
  ...
  let completionChunk = StreamChunk.completion(reason: .natural, metadata: metadata)
  continuation.yield(completionChunk)
  continuation.finish()
  ```

  `generateTokens`'s return type is `Int` (`tokensGenerated`, declared at
  `:443-448`). The reason it actually stopped for is discarded the moment
  `generateTokens` returns; `performGeneration` has no way to know it, and
  fabricates `.natural` unconditionally.

- `Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:443-537`, five
  emit sites listed above, each `continuation.yield(chunk)` followed by
  `return generatedCount` (or, for the stop-sequence branch, `return
  generatedCount` inside the `for chunk in chunks` loop at `:499-509`).

- `Sources/OnDeviceCatalyst/Chat System/StreamResponse.swift:90-99`
  (`StreamingResponse.addChunk`) has no guard against being called more than
  once with `chunk.isComplete == true`; each call overwrites
  `completionReason` and `metadata` with the latest chunk's values.

- `Sources/OnDeviceCatalyst/Chat System/StreamResponse.swift:275-300`
  (`collectResponse()`, `collectContent()`) both `break` after the first
  `isComplete` chunk, which is why they observe the true reason but not the
  metadata.

- Consumer confirmation in-repo: `Sources/OnDeviceCatalyst/Service
  Layer/Catalyst.swift:481-483` breaks on the first completion, so today it
  silently discards the real `ResponseMetadata` on every generation call it
  makes.

- Characterization evidence, already landed and passing against today's
  behavior: `Tests/OnDeviceCatalystTests/R1/StreamContractCharacterizationTests.swift:45-99`
  (`test_characterizes_streamingResponse_drainedCollector_reportsNaturalOverRealReason__ODC_0011`,
  `test_characterizes_collectResponse_breakingCollector_disagreesWithDrainedCollector__ODC_0011`,
  `test_characterizes_collectContent_breaksOnFirstCompletion__ODC_0011`).

## In-scope repair

Exactly one terminal event per generation, carrying the real reason and the
full metadata. Naively deleting the `:372-386` block is insufficient and is
called out here because it is the obvious wrong fix: the `ResponseMetadata`
computed there (`tokensGenerated`, `tokensPerSecond`, `generationTimeMs`,
`promptTokens`, `totalTokens`) is not duplicated anywhere else. Deleting the
block that carries it, without relocating the computation, would leave every
one of the five real completion reasons with no metadata at all, trading
"wrong reason with metadata" for "right reason with no metadata." Neither is
the target state.

The repair changes where the reason and the metadata are computed and
attached, so both end up on the single chunk that survives:

1. Change `generateTokens`'s signature so it returns the `CompletionReason`
   it actually stopped for, alongside the token count it already returns
   (for example a `(count: Int, reason: CompletionReason)` tuple, or an
   equivalent struct). This is a private method; the signature change has no
   public-API surface.
2. At each of the five sites (`:466`, `:487`, the stop-sequence branch,
   `:517`, `:535`), stop calling `continuation.yield(chunk)` directly.
   Instead, construct the reason value and `return` it (with the token
   count) to the caller. `generateTokens` no longer yields any chunk itself.
3. In `performGeneration` (`:365-386`), consume the returned reason, build
   `ResponseMetadata` exactly as today (unchanged fields and computation),
   and yield exactly one `StreamChunk.completion(reason:, metadata:)` using
   the *real* reason rather than a hardcoded `.natural`, then
   `continuation.finish()`.
4. The `userCancelled` early-return path (`:463-468`) needs the same
   treatment: today it yields its own chunk and returns directly from
   `generateTokens` without going through `performGeneration`'s completion
   step at all (`return generatedCount` inside the `while` loop, ahead of
   the `while` loop's normal exit). After the repair this path must also
   flow back through the single emission site in `performGeneration`, so
   that userCancelled generations also get metadata and there remains
   exactly one emission site in the whole call graph.

Non-goals make explicit what does not change: `StopSequenceHandler` /
`StreamProcessor`'s content-chunk emission (non-terminal chunks) is
untouched; only the terminal `isComplete == true` chunk's plumbing changes.

## Non-goals

- No change to `ResponseMetadata`'s fields, computation, or the
  `tokensPerSecond` formula.
- No change to `StreamingResponse.addChunk`'s overwrite-on-every-isComplete
  behavior, `collectResponse()`, or `collectContent()`. Once only one
  completion chunk exists, breaking and draining consumers necessarily
  agree; the consumer-side code does not need to change to get that
  agreement.
- No change to the `error` branch (`:389-392`), which already yields exactly
  one completion (`CompletionReason.error(...)`) and already finishes the
  continuation with a thrown error. That branch is not part of D2.
- No change to `StreamChunk`, `ChunkMetadata`, or `CompletionReason`'s
  public shape.
- Does not repair D1 (ODC-0010), D3 (ODC-0012), or D8 (ODC-0015). Those are
  the loading stream (`initialize()` / `LoadProgress`), a structurally
  separate `AsyncStream` from the generation stream this ticket touches
  (`generate()` / `AsyncThrowingStream<StreamChunk, Error>`).
- Does not resolve Q1/Q3 (device-execution mechanism for `R3` cases). That
  is ODC-0021's scope, inherited here per ODC-0004's ticket-allocation
  obligation on whichever of ODC-0010/0011/0012 executes an `R3` case
  first.

## Regression risks

- **Low, for the reason/metadata pairing itself.** The fix consolidates two
  emission sites into one and threads a value through an existing return
  path; it does not introduce new control flow, new state, or a new type
  visible outside the file.
- **Medium, for the `userCancelled` path specifically.** That path currently
  returns directly from inside `generateTokens`'s `while` loop, bypassing
  `performGeneration`'s post-loop code (prompt-token accounting, duration
  computation) entirely. Routing it back through the single emission site
  means it now also computes `ResponseMetadata`, which it did not do before;
  the `tokensGenerated` count for a cancelled generation is whatever
  `generatedCount` had reached, which is correct, but this is a genuinely
  new code path being exercised and deserves its own test, not just
  inference from the other four sites.
- **Low, for external callers.** No consumer in this repository or asserted
  in the characterization suite depends on receiving two completions; both
  present consumer styles are already characterized as getting a degraded
  result, not a design they rely on.

## Tests

Per ODC-0004's ticket-allocation table, this ticket updates `C-D2-1` through
`C-D2-4` and fingerprints `F-D2-1`, `F-D2-2` in the same commit as the repair.
`docs/characterization/v2-fingerprints.json` pins `F-D2-1` on
`performGeneration`'s body and `F-D2-2` on `generateTokens`'s body; both
change under this repair, so both fingerprint entries must be updated to the
new source hash in that same commit, or
`python3 scripts/check-characterization.py --fingerprints` fails by design
(`docs/characterization/v2-fingerprints.json`'s check reports "this defect
site changed; if that was deliberate, update the characterization case and
this fingerprint in the same commit").

Tests that flip, and what each must assert afterward:

| Test | Class / surface | Today | Must assert after this repair |
| --- | --- | --- | --- |
| `test_characterizes_streamingResponse_drainedCollector_reportsNaturalOverRealReason__ODC_0011` | `D2StreamContractCharacterizationTests`, `R1` | Asserts a drained collector reports `.natural` and not the real reason, from a synthetic two-completion stream. | The synthetic fixture (`makeD2ChunkStream`) must change to emit exactly one completion, and the assertion becomes: a drained collector reports the real reason. Rename to a `test_requires_` regression case; the defect it pinned no longer exists. |
| `test_characterizes_collectResponse_breakingCollector_disagreesWithDrainedCollector__ODC_0011` | `D2StreamContractCharacterizationTests`, `R1` | Asserts a breaking collector and a draining collector disagree. | With one completion chunk, both collectors necessarily agree; the disagreement assertion must be replaced with an agreement assertion. Rename to `test_requires_`. |
| `test_characterizes_collectContent_breaksOnFirstCompletion__ODC_0011` | `D2StreamContractCharacterizationTests`, `R1` | Asserts `collectContent()` stops at the first completion. | **Does not flip.** This asserts `collectContent()`'s own break-on-first-completion behavior, which this ticket does not change; the spec's own "Should be" line already says "unchanged in observable shape." Left as a characterization of the (correct) consumer contract, or reclassified `__no_defect` since it no longer pins a wrong belief once only one completion exists to break on. |
| `test_characterizes_boundedGeneration_yieldsTwoCompletionChunks__ODC_0011` | `D2GenerationCharacterizationTests`, `R3` | Always `XCTSkip("SKIP[requires-device] ...")` on `SFC-B`; never executes today. | Assertion body must change from "exactly two" to "exactly one" `isComplete` chunk, so that whichever of ODC-0010/0011/0012 first resolves Q3 exercises the corrected assertion. Does not flip from passing to failing in CI today, because it does not run in CI today; flips only once ODC-0021 supplies a device surface. |
| `test_characterizes_secondCompletion_reportsNaturalEvenAfterMaxTokensReached__ODC_0011` | `D2GenerationCharacterizationTests`, `R3` | Always skip, as above. | Assertion becomes: the single completion carries the true reason (`.maxTokensReached` in that scenario), not `.natural`. Same skip caveat as above. |

`F-D2-1` and `F-D2-2` (`docs/characterization/v2-fingerprints.json`) must be
regenerated from the post-repair source and the entries' `sha256` fields
updated in the same commit.

A new regression test should be added for the `userCancelled` path
specifically (flagged under Regression risks above), asserting it now
carries `ResponseMetadata`; this is additive and has no `C-D2-*` ID today
because the current suite does not cover it.

## Acceptance criteria

| ID | Criterion | Deciding command |
| --- | --- | --- |
| A1 | `performGeneration` and `generateTokens` no longer contain more than one call to `continuation.yield` whose `StreamChunk` has `isComplete: true` reachable from a single `generateTokens` invocation; enforced by the updated fingerprint matching the new source. | `python3 scripts/check-characterization.py --fingerprints` exits 0 |
| A2 | The full characterization suite still builds and runs on the simulator surface after the repair, with the `C-D2-*` cases reflecting the table in `## Tests`. | `bash scripts/run-characterization.sh --surface simulator` exits 0 |
| A3 | The suite's own inventory of defects and tests stays internally consistent (naming, four-line comment blocks, catalog agreement). | `python3 scripts/check-characterization.py --naming --inventory --require-defects D1,D2,D3,D4,D5,D6,D7,D8,N1` exits 0 |
| A4 | No file outside `Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift`, `Tests/OnDeviceCatalystTests/**`, and `docs/characterization/v2-fingerprints.json` is touched by the repair commit. | `git diff --stat <repair-commit>~1 <repair-commit> -- . ':!Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift' ':!Tests/OnDeviceCatalystTests' ':!docs/characterization/v2-fingerprints.json'` outputs nothing |
| A5 | Project state stays consistent. | `python3 scripts/validate-project-state.py` exits 0 |

## Review record

Not yet reviewed. Drafted alongside ODC-0012 and ODC-0015 because all three
sit in `LlamaInstance`'s stream lifecycle, but this ticket's mechanism
(generation stream, `performGeneration`/`generateTokens`) is independent of
the other two (loading stream, `initialize()`/`publishProgress`): fixing
ODC-0011 has no ordering dependency on ODC-0012 or ODC-0015 and can land
before, after, or between them. The only shared-file consideration is
avoiding overlapping diffs in `LlamaInstance.swift` if landed concurrently
with the other two; the functions each ticket touches (`performGeneration`
and `generateTokens` here, versus `publishProgress` and
`handleInitializationError` for the loading-stream tickets) do not overlap.

**Concurrency.** `LlamaInstance` is a plain, non-`Sendable` class
(`Core Engine/LlamaInstance.swift:18`), and the package manifest declares no
`swiftLanguageMode` or strict-concurrency `swiftSettings`, so it runs under
Swift 5's minimal `Sendable` enforcement (confirmed by inspection, matching
ODC-0004's `## Design`, "Framework choice: XCTest," reason 2). This repair's
mutations are confined to local variables and parameters within
`performGeneration` and `generateTokens`, both already executing serially
inside the single `Task` created by `generate()` (`:283-291`); it introduces
no new shared mutable state, no new escaping closure, and no new consumer of
`self` across a task boundary that was not already present. `AsyncThrowingStream.Continuation`'s
`yield`/`finish` are documented safe to call from any context, so
concentrating both into one call site is not a concurrency-widening change.
This repair is safe to land under the current Swift 5 / minimal-enforcement
model and does not need to wait for ODC-0101. It should be revisited only in
the sense that ODC-0101's future actor-isolation design should account for
`generateTokens` returning a value type rather than yielding directly, which
is already this repair's shape and imposes no extra migration cost.

Q1 (unresolved): whether the `userCancelled` early-return path should carry
a zero-duration or partial-duration `ResponseMetadata`, since it is
cancelled mid-generation rather than completed; this repair computes
whatever `performGeneration`'s existing duration math produces for that
case (elapsed time up to cancellation) but no product decision has been
recorded on whether that is the desired number. Left open for founder input
rather than decided unilaterally in a bug-fix spec.

**Proposed ledger row for `Tickets.md`.** The `ODC-0011` row's `Spec` column
becomes `[spec](docs/specs/ODC-0011-duplicate-terminal-completions.md)`; its
`Status` column becomes `SPEC_DRAFT` to match this file's frontmatter, which
the validator requires once a spec is linked; its `Dependencies` column
becomes `ODC-0002, ODC-0004` (unchanged in substance: this ticket's fix has
no ordering dependency on ODC-0012 or ODC-0015, but its tests live inside
the ODC-0004 suite); its `Next Gate` column becomes `review`.

## Validation evidence

None yet; `status: SPEC_DRAFT`. The reproduction and root-cause evidence
above were gathered by read-only inspection (`sed`, `grep`) against revision
`e9d16a4` and by reading the already-landed, already-passing characterization
tests named in `## Tests`; no build was run to produce this spec, per this
ticket's own writing constraints.
