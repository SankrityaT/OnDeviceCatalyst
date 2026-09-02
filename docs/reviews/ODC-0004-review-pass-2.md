---
review_of: ODC-0004
spec: docs/specs/ODC-0004-v2-characterization-suite.md
pass: 2 (adversarial)
date: 2026-09-02
reviewer: adversarial spec review
repo_revision_reviewed: c2a89757b44fad306557a830303ffc0001a463b9
verdict: REJECT
---

# ODC-0004 review pass two (adversarial)

## Verdict

**REJECT**, return to `REVISION`.

The empirical core of this spec is unusually strong. Nearly every load-bearing
claim in `## Current state and evidence` was independently reproduced, several
to the exact byte, symbol count, or failure message. That puts the discovery
work well above the standard ODC-0002 shipped at its first adversarial pass.
But the spec fails on the same axis ODC-0002 failed on: process discipline and
decision-completeness. Five problems justify REJECT rather than "accept with
revisions":

1. The ticket ledger this spec claims not to touch has already been touched.
   `Tickets.md` carries `ODC-0018`, `ODC-0019`, and `ODC-0020` at `BACKLOG`,
   added by the same commit that drafted this spec, while the spec's own text
   (`## Ticket allocation`) says "This spec does not edit `Tickets.md`" and
   calls the rows "proposals for the founder." The artifact and its own prose
   disagree about its state.
2. Acceptance criterion A2 and Validation item 6 both depend on `<base>`, a
   token that is never defined anywhere in the spec. As written, neither is
   decidable by a command, which is exactly the defect class program rule 6
   forbids and the ODC-0002 review required be fixed for the same criterion.
3. `check-characterization.py --inventory`, one of this ticket's own required
   gates (A8), claims exhaustive ownership of every test method under
   `Tests/OnDeviceCatalystTests/`, the same target ODC-0003 explicitly plans to
   add benchmark harness methods to. Nothing in this spec's permitted-changes
   table or `--inventory` description carves out that exception. The two
   sibling specs will fight over the same directory the day both land.
4. `SFC-C`, the physical-device surface, is asserted to exist as a runnable
   mode (`scripts/run-characterization.sh --surface device`) but is specified
   only as two CLI flags. Nothing describes how a package-only, no-Xcode-project
   test bundle gets code-signed, installed, and launched on real hardware, the
   device equivalent of the non-trivial repackaging work N10 had to solve for
   the simulator. `SFC-C` is not measured, which the spec discloses honestly,
   but it is also not designed to the point of being executable later without
   further discovery, contrary to program rule 6.
5. One of the four grounds for choosing XCTest over Swift Testing (the
   non-Sendable argument) is the only one of the four with no cited evidence,
   no probe, no `N`-number, in a spec whose entire ethos is "measured, not
   asserted." Independently checked: `swift package tools-version` reports
   `5.12.0` and `Package.swift` sets no `swiftLanguageMode` or strict-concurrency
   `swiftSettings`, so the package's default is Swift 5 language mode with
   minimal Sendable checking, a fact that bears directly on how forcefully the
   claimed "must wrap in `@unchecked Sendable` boxes" problem would actually
   bite, and the spec never states it.

Everything below was verified against the working tree at HEAD
(`c2a89757b44fad306557a830303ffc0001a463b9`) with read-only commands, plus
builds and simulator test runs performed entirely inside `mktemp -d` scratch
trees outside the working tree. No file under `Sources/`, `Tests/`,
`Package.swift`, or `Package.resolved` was modified. `git status --porcelain`
was empty before and after every scratch operation, and is empty now except for
one pre-existing, unrelated untracked file (`docs/reviews/ODC-0003-review-pass-2.md`)
that this review did not create or touch.

### Environment used for verification

| Item | Value |
| --- | --- |
| Xcode | 26.6 (17F113) |
| Swift | 6.3.3 (swiftlang-6.3.3.1.3), swift-driver 1.148.6 |
| iOS Simulator SDK | 26.5 (`iphonesimulator26.5`) |
| Simulator runtime used | iOS 26.5 (23F77), device type iPhone 17 Pro |
| Host target | `arm64-apple-macosx26.0` |

This matches the toolchain the spec pins almost exactly (spec: Xcode 26.6/17F113,
Swift 6.3.3, macOS 26.4 vs. this session's macOS 26.4/25E246 report from
`sw_vers` was not independently re-checked, but Xcode build number and Swift
compiler version, the two figures that matter for build reproducibility, match
exactly).

---

## Empirical claims independently verified

All of these were reproduced from a clean `git archive HEAD | tar -x` scratch
tree, not from the working tree, and the scratch trees were deleted afterward.

- **N1, test target does not compile.** `swift build --build-tests --sdk
  "$(xcrun --sdk iphonesimulator --show-sdk-path)" --triple
  arm64-apple-ios17.0-simulator -c debug` fails with exactly
  `Tests/OnDeviceCatalystTests/OnDeviceCatalystTests.swift:31:46: error: type
  'PredictionConfig' has no member 'quality'`. Reproduced verbatim, including
  column number. `PredictionConfig.swift` was independently confirmed to declare
  exactly five presets (`balanced`, `creative`, `speed`, `deterministic`,
  `mirostat`), none named `quality`.
- **N2, symbol-set equality and successful link.** After a minimal, scratch-only
  workaround for N1 (substituting `.creative` for the nonexistent `.quality`,
  purely to unblock the build; this file was never touched in the real repo),
  the build produced `stub_defined.txt` and `required.txt` each with **exactly
  51 lines**, and `comm -3` between them was **empty**. The link step reported
  `Linking OnDeviceCatalystPackageTests` and `Build complete!`, with the exact
  two warnings the spec quotes: `clang: warning: using sysroot for 'MacOSX' but
  targeting 'iPhone'` and `ld: warning: object file ... was built for newer
  'iOS-simulator' version (26.2) than being linked (17.0)`. The stub archive is
  confirmed **7,936 bytes**.
- **N3, disassembly, mostly confirmed, one claim inaccurate (see finding 6).**
  `_llama_backend_init` disassembles to a bare `ret`. `_llama_new_context_with_model`
  and `_llama_n_ctx` both return zero unconditionally. A scan for
  `brk|udf|_abort|_exit|trap` across both stub objects returned zero matches.
  `_llama_load_model_from_file` does return null unconditionally, confirming the
  functional claim, but the literal instruction sequence quoted in the spec is
  wrong; see finding 6.
- **N4, simulator minimum OS exceeds declared floor.** The same linker warning
  above (`26.2` vs. the package's declared `iOS(.v17)`) is exact, independently
  reproduced evidence for this claim.
- **N5 and D8, exactly 3 events then termination, neither `.ready` nor
  `.failed`.** Constructed both synthetic fixtures described in the spec
  (`phi-probe.gguf`, `generic-probe.gguf`, 2 MiB, `GGUF` magic), ran
  `LlamaInstance.initialize()` against each through the real (non-stubbed) Swift
  source, on the simulator surface, and collected the `AsyncStream<LoadProgress>`
  to completion. Both fixtures produced **exactly three events**,
  `preparing("Validating model file")`, `loading("Initializing llama.cpp
  backend")`, `loading("Loading model from <path>")`, then the stream ended.
  Neither `.ready` nor `.failed` was ever delivered on either path, and console
  output confirmed the `phi` fixture actually entered
  `attemptFallbackInitialization` ("Catalyst: Attempting fallback
  initialization") while the generic fixture did not, exactly matching the
  recoverable/non-recoverable branch split N5 describes, and both still
  produced the identical externally observable 3-event sequence.
- **N6, `ModelCache.getInstance` returns a not-ready instance.** Constructed a
  `LlamaInstance` via the file-access-free `ModelProfile(mlxModelId:name:)`
  initializer, stored it with `ModelCache.shared.storeInstance`, polled
  `ModelCache.shared.getInstance` with a bounded wait (confirming the spec's own
  note that the store is asynchronous and an immediate read can return nil).
  The returned instance was non-nil with `isReady == false`, reproduced exactly.
- **N8, existing test failure.** Running the (workaround-patched) four original
  test cases through the exact N10 procedure (SwiftPM build, repackage into a
  flat iOS bundle, `simctl spawn` the platform `xctest` agent) produced
  `Executed 4 tests, with 1 failure (0 unexpected)`, with the failure being
  `XCTAssertGreaterThan failed: ("-1") is not greater than ("1024")` at
  `OnDeviceCatalystTests.swift:37`, an exact match to the spec's quoted output.
- **N10, the runner procedure itself works end to end.** The full sequence
  (build for the simulator triple with `--build-tests`, repackage the
  macOS-shaped SwiftPM bundle into a flat iOS `.xctest` bundle with a
  hand-written `Info.plist`, boot a temporary simulator, `simctl spawn` the
  platform `xctest` agent) executed successfully with no Xcode project, no
  scheme, and no code signing, exactly as claimed.
- **D4, macOS build failure.** `swift build -c debug` on the host, from a
  separate clean scratch tree, fails with the exact quoted diagnostic:
  `Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:12:8: error: no such
  module 'llama'`.
- **B3/N1 consistency with ODC-0002.** `git diff --stat 6d72193 HEAD -- Sources
  Tests Package.swift Package.resolved OnDeviceCatalyst OnDeviceCatalyst.xcodeproj`
  shows only `Package.resolved` changed (ODC-0002's lockfile correction, 6
  lines), confirming the evidence in `## Current state and evidence` was not
  invalidated by intervening commits.

## Empirical claims that did not reproduce as literally stated

- **N3's `_llama_load_model_from_file` disassembly.** The spec states (line
  216) that disassembly "shows ... `_llama_load_model_from_file` is `mov x0,
  #0x0; ret`." The actual `otool -tvV` output is six instructions:
  ```
  _llama_load_model_from_file:
      sub sp, sp, #0x10
      str x0, [sp, #0x8]
      str x1, [sp]
      mov x0, #0x0
      add sp, sp, #0x10
      ret
  ```
  The functional conclusion (unconditional null return, no trap) is correct and
  independently confirmed. The literal quoted instruction sequence is not what
  the binary contains. See finding 6.

No other load-bearing empirical claim in `## Current state and evidence` failed
to reproduce. `SFC-C` (device) claims were not attempted because the spec
itself marks them unmeasured (`Measured: No`, Q1 open), which is honest, not a
defect in itself; see finding 4 for why the surface is nonetheless
underspecified as a design.

---

## Findings

### 1. BLOCKING, the ticket ledger this spec disclaims editing has already been edited

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:1138-1156`
(`## Ticket allocation`), specifically line 1143 ("This spec does not edit
`Tickets.md`. The rows below are proposals for the founder..."). Compare
`Tickets.md:23-25`.

**Problem.** `git log --oneline -- Tickets.md` shows commit `1e63c34`
("ODC-0004: characterization suite spec; allocate ODC-0018/0019/0020") added
exactly the three rows this spec's text describes as unadopted proposals:

```
| ODC-0018 | bug | Declared test target does not compile on any triple (PredictionConfig.quality) | P0 | BACKLOG | P0 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0019 | decision | Disposition of three orphaned test files outside the target path | P0 | BACKLOG | P2 | ODC-0004 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0020 | decision | Revisit XCTest versus Swift Testing after the concurrency model lands | P0 | BACKLOG | P2 | ODC-0101 | TBD | TBD | unassigned | 2026-09-01 | discovery |
```

That commit is on this branch, ahead of `HEAD`. The spec being reviewed still
says, in prose, "This spec does not edit `Tickets.md`," and frames the three
rows as something "the founder" has yet to act on. The document and the
repository state it describes disagree. This is not cosmetic: the program's
rule 5 ("the founder makes the final accept, revise, or reject decision") and
rule 6 ("approved specs contain no unresolved implementation decisions")
presuppose that ledger mutations follow approval, not precede review. A ticket
ledger that already reflects a spec's proposals before pass one, pass two, or
founder review has happened is exactly the failure mode a two-pass review
process exists to prevent: it launders the outcome of the review before the
review runs. It is also worth noting `Tickets.md`'s ODC-0017 row already carries
editorial commentary lifted from this spec's own findings ("link-time half
ANSWERED by ODC-0004, stub defines all 51 referenced symbols and links"),
reinforcing that this ticket's conclusions were written into the canonical
ledger before an adversarial pass evaluated them.

**Fix.** Revert the `Tickets.md` changes that added `ODC-0018`, `ODC-0019`,
`ODC-0020` (and the ODC-0017 commentary sourced from this spec) until this spec
reaches `APPROVED`, or rewrite `## Ticket allocation`'s prose to honestly state
that the rows already exist and explain, as a deliberate process exception, why
they were added before review. Silence is not an option; the current text is
false.

---

### 2. BLOCKING, acceptance criterion A2 and Validation item 6 depend on an undefined `<base>`

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:1055-1056`
(Validation item 6), `:1110` (acceptance criterion A2).

**Problem.** Both read:

```
git diff --stat <base> -- Sources Package.swift Package.resolved OnDeviceCatalyst OnDeviceCatalyst.xcodeproj
```

`<base>` is never defined anywhere in this document as a concrete git
reference. `grep -n '<base>'` against the spec returns exactly these two lines
and nothing else; there is no companion sentence saying what `<base>` resolves
to (the ticket's starting revision? The commit before `IMPLEMENTING` began? The
merge-base with `main`?). As written, neither Validation item 6 nor acceptance
criterion A2 is decidable by a command; a human must supply a missing parameter
before either can run, which is precisely the defect ODC-0002's own pass-two
review flagged (finding 5) for the identical criterion, and ODC-0002 fixed it
by naming a literal commit hash (`59da80b`). This spec had that precedent
directly available and did not apply it to its own analogous criterion.

**Fix.** Replace `<base>` with a concrete, resolvable reference, for example
"the commit at which this ticket's `IMPLEMENTING` status began, recorded in the
ticket's `last_updated` history" or simply the specific hash this spec was
drafted against. Whatever the choice, state it in prose next to both
occurrences so the command is copy-pasteable without substitution.

---

### 3. BLOCKING, `--inventory`'s exhaustive test-method ownership collides with ODC-0003's planned benchmark tests

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:774-777`
(`--inventory` description), `:653-663` (permitted-changes table, row
`Tests/OnDeviceCatalystTests/**`). Compare
`docs/specs/ODC-0003-benchmark-contract.md:791-798`.

**Problem.** ODC-0004 states:

> `--inventory` is the anti-drift check. Every case id in `## Tests` must exist
> as a test method or an `R0` check, and every test method must appear in
> `## Tests`. A suite that grows without its spec, or a spec that promises
> cases nobody wrote, fails here.

That is an unqualified claim over the whole test target: "every test method
must appear in `## Tests`." ODC-0003, a sibling spec this project's own
program explicitly asks reviewers to check for scope collisions against,
states its intended benchmark harness will build and deploy

> under the same test target ODC-0004 uses, in their own subdirectory
> (proposed: `Tests/OnDeviceCatalystTests/Benchmarks/`)

ODC-0004's permitted-changes table marks `Tests/OnDeviceCatalystTests/**` as
"Yes, new characterization sources, plus the N1 repair" with no carve-out for
another ticket's additions, and `--inventory` as described has no directory
scoping, naming-prefix scoping, or any other mechanism that would let it ignore
methods ODC-0003 adds. The instant ODC-0003 lands `Benchmarks/` files with test
methods, `check-characterization.py --inventory`, which is acceptance criterion
A8 for this ticket, will start failing against test methods it never catalogs
and does not own. Two specs that each name the other and claim to coordinate
scope still produce a script that cannot tell the difference between "the suite
grew without its spec" and "a sibling ticket added its own, separately
specified, tests to a shared directory."

**Fix.** Either scope `--inventory`'s ownership explicitly to
`Tests/OnDeviceCatalystTests/**` excluding `Benchmarks/**` (or whatever
directory ODC-0003 settles on), or require every non-characterization test
method under the shared target to declare which spec owns it (a lightweight
marker comment or a second manifest) so `--inventory` can partition by owner
instead of assuming total ownership. State the resolution in both specs, not
just one.

---

### 4. MAJOR, `SFC-C` is asserted as an executable mode but is not specified to the level the simulator surface required

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:454-464`
(execution surfaces table), `:743-757` (Runner), `:1011-1024` (Q1).

**Problem.** The spec's own discovery work (N7, N10) shows that getting a bare
SwiftPM test target to run on the iOS Simulator without Xcode required solving
several non-obvious problems: SwiftPM emits a macOS-shaped bundle even when
cross-compiling for the simulator, the platform `xctest` agent needs a flat iOS
bundle and a hand-authored `Info.plist`, and the tracked `.xcodeproj` shadows
the package for bare `xcodebuild`. All of that is measured, solved, and
reproducible (independently confirmed above).

The device surface is asserted to exist as a runner mode with two flags:

```
scripts/run-characterization.sh --surface device --destination-id <id>
```

and one sentence: "`--surface device` requires a signing identity and a
destination id supplied by the operator, never discovered and never written
into any tracked file." Nothing describes how a package-only test bundle,
built the same way `SFC-B` is (via `swift build --build-tests`, no `.xcodeproj`
involvement), gets code-signed, installed onto a physical device, and launched
there. Physical-device XCTest execution outside Xcode is not a smaller version
of the simulator problem; it typically requires a provisioning profile, an
entitlements file, and a device-deployment tool (`devicectl`, `ios-deploy`, or
equivalent), none of which is named, and none of which was measured, unlike
every other design decision in this spec. Q1's "no device available" outcome
papers over this by letting the ticket reach `DONE` without executing `R3`, but
that only defers the gap. The obligation table (`## Ticket allocation`)
requires ODC-0010, ODC-0011, and ODC-0012 to "execute [the five `R3` cases]
before its own repair is accepted," which means some future ticket inherits a
device-execution procedure that does not yet exist, with no discovery budget
allocated to build it, because this ticket's own discovery (correctly) spent
its effort on the simulator surface instead.

**Fix.** Either measure the device surface at least once, even a minimal "build
for `arm64-apple-ios17.0`, sign with a personal team, install and run one
sanity case on a physical device" pass, and record what that took, or
explicitly state in `## Open questions and gates` that the device-execution
mechanism itself (not just device availability) is a second open question,
owned by whichever of ODC-0010/0011/0012 executes it first, so the obligation
table does not silently assume a solved problem.

---

### 5. MAJOR, the non-Sendable argument for choosing XCTest is the one ground with no cited evidence, and an easily checked fact undercuts its stated severity

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:516-525`
(Framework choice, reason 2).

**Problem.** Compare the four grounds for choosing XCTest over Swift Testing:

- Reason 1 (inverted expectations) cites a concrete mechanism
  (`XCTestExpectation.isInverted`) and self-corroborates: "the probes in N5
  already had to write exactly that race to work around its absence." That
  claim is real; `XCTestExpectation.isInverted` and `confirmation(expectedCount:
  0)` are both real, accurately described APIs.
- Reason 3 cites `ODC-0002 E6` (the fictitious `swift-tools-version: 5.12`).
- Reason 4 cites `N1` and `N8` directly.
- Reason 2 (the Sendable argument) cites nothing. It asserts that
  characterizing `LlamaInstance`, `ModelCache`, and `Catalyst` under Swift
  Testing "would require wrapping them in `@unchecked Sendable` boxes or adding
  isolation, which changes the thing being measured," with no probe, no
  attempted Swift Testing case, no compiler diagnostic quoted, in a document
  whose stated discipline is "everything measured."

Independently checked: `swift package tools-version` reports `5.12.0`, and
`Package.swift` sets no `swiftLanguageMode` and no strict-concurrency
`swiftSettings` on any target. SwiftPM's default language mode for a manifest
declaring a tools-version below 6.0 is Swift 5 mode, with materially weaker
default Sendable enforcement than Swift 6 mode. This does not make reason 2
false, `LlamaInstance` genuinely is a non-Sendable class over mutable state,
and that fact is real regardless of language mode, but it means the severity
of the claimed problem ("must wrap in `@unchecked Sendable` boxes") was
asserted without checking the one fact (effective language mode) that would
tell a reader how forcefully Swift's compiler actually enforces it here. This
is exactly the gap the other three reasons do not have.

This finding does not overturn the XCTest decision. Reason 1 alone, backed by
real evidence, is sufficient grounds for it. The problem is evidentiary
discipline: one of four "measured" reasons is not measured, in a spec that
otherwise holds itself and this exact repository to a standard of citing the
command that produced every claim.

**Fix.** Either add a probe (attempt one Swift Testing `@Test` function
capturing a `LlamaInstance` and quote the actual diagnostic or its absence), or
soften reason 2's language to state plainly that this ground is a design
judgment about coupling test intent to implementation detail, not a measured
compiler-enforced blocker, and note the actual (Swift 5) language mode so a
future reader is not misled about severity.

---

### 6. MAJOR, N3's `_llama_load_model_from_file` disassembly quote does not match the binary

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:211-218` (N3).

**Problem.** Quoted above in "Empirical claims that did not reproduce as
literally stated." The spec states disassembly "shows `_llama_backend_init` is
a bare `ret`, `_llama_load_model_from_file` is `mov x0, #0x0; ret`," implying
both are quoted at the same level of literalness. The `_llama_backend_init`
quote is accurate (verified: a single `ret` instruction). The
`_llama_load_model_from_file` quote is not: the actual function is six
instructions including stack frame setup and teardown around the `mov x0,
#0x0`. The functional conclusion this section draws (deterministic null return,
no trap, so failure paths are reachable without a model) is correct and
independently reproduced. But this document's central methodological claim is
that its evidence is exact and quoted, not paraphrased, precisely so a
fingerprint mismatch or a characterization failure can be trusted at face
value. A misquoted disassembly line, however harmless its conclusion, is a
crack in that standard, in the one document whose entire value proposition is
that it does not do this.

**Fix.** Replace the quote with the actual six-instruction sequence, or
paraphrase honestly ("returns null via a short prologue/epilogue around `mov
x0, #0x0`") rather than presenting a two-instruction quote as literal
disassembly output.

---

### 7. MAJOR, this spec's ticket mapping for the eight baseline defects silently diverges from the baseline document's own mapping

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:113-124` (B1
table). Compare `docs/baselines/v2.0.4.md:388-393` (summary table) and
`:396-464` (per-defect detail), specifically line 473.

**Problem.** `docs/baselines/v2.0.4.md`, the document this spec calls its
inherited source of truth and cites throughout, maps each defect to a
v3-roadmap architecture ticket: D1 to `ODC-0101`, D2 and D3 to `ODC-0202`, D4,
D5, and D6 to `ODC-0103`, D7 to `ODC-0300`, D8 to `ODC-0101`. The baseline
document itself explains why (line 473): those are placeholder mappings to
"the existing ticket that most nearly owns [each finding]," because the
procedure that produced the baseline did not create the reserved `ODC-0010`
through `ODC-0049` range, and it says the real allocation "remains
outstanding."

ODC-0004's own B1 table performs that allocation, mapping the same eight
defects to `ODC-0010` through `ODC-0017`, a completely disjoint set of ticket
IDs from the baseline's table. That allocation is reasonable and is this
ticket's to make. What is missing is any acknowledgment that it supersedes the
baseline document's own table, or any note in `docs/baselines/v2.0.4.md` (or
here) reconciling the two. A reader who opens the baseline first and follows D1
to `ODC-0101` will not discover `ODC-0010` exists, or that it is the ticket
that actually owns the repair, without independently cross-referencing
`Tickets.md`. This is precisely the class of confounder-across-documents an
adversarial pass is supposed to catch, two canonical artifacts disagreeing
about which ticket owns the same named defect, with nothing in either
document flagging the disagreement.

**Fix.** Add one sentence to `## Current state and evidence` or `## Ticket
allocation` stating that this spec's `ODC-0010` through `ODC-0017` mapping
supersedes the placeholder mapping in `docs/baselines/v2.0.4.md`'s defect
table, and file a small follow-up to update that table (or add a forwarding
note to it) so the two documents agree.

---

### 8. MAJOR, `docs/templates/test-spec.md` does not carry forward the rigor ODC-0002's own review just demanded of its sibling template

**Location:** `docs/templates/test-spec.md` (all sections after `## Design`
and `## Tests`). Compare `docs/templates/baseline-spec.md`.

**Problem.** `docs/templates/baseline-spec.md` was added (or substantially
written) as a direct, explicit response to ODC-0002's pass-two review finding
6 ("`type: baseline` has no template... Add `docs/templates/baseline-spec.md`,
then bring ODC-0002 to it"). That template carries explanatory guidance prose
under nearly every section: what `## Pinned environment and clean state` must
contain and why ("a baseline spec is invalid without this section"), what `##
Current state and evidence` is for, what `## Failure behavior`'s taxonomy must
satisfy ("exhaustive over the observed failures"), and, critically, this
sentence under `## Acceptance criteria`: "A criterion that cannot be decided by
a command does not belong here."

`docs/templates/test-spec.md`, created by this same ticket, carries guidance
prose under exactly two sections (`## Design`, `## Tests`). Every other
section, including `## Acceptance criteria`, `## Failure behavior`, `##
Security and privacy`, `## Migration and compatibility impact`, and `## Ticket
allocation`, is a bare header with no steering text at all. This is not a
hypothetical gap: finding 2 above (the undefined `<base>` in acceptance
criterion A2) is exactly the mistake `baseline-spec.md`'s "cannot be decided by
a command does not belong here" line exists to prevent, and the template
drafted alongside this very spec does not carry that lesson forward, so this
spec fell into the gap the missing guidance would have caught.

**Fix.** Bring `docs/templates/test-spec.md` to the same standard as
`docs/templates/baseline-spec.md`: add guidance prose to every section, in
particular the "cannot be decided by a command does not belong here" line
under `## Acceptance criteria`, and a note under `## Ticket allocation` about
stating the reserved ID range and default column values, matching what
`baseline-spec.md` already does for its own type.

---

### 9. MINOR, a characterization failure and a real regression are indistinguishable at the CI status level, only at the log level

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:808-822`
(Failure behavior table), `:64-73` (Goals).

**Problem.** The naming convention and the four-line comment block make the
distinction between "pinned bug" and "intended behavior" visible to a reader
who opens a failing test's source or its full log output (Goals line 71-73
states this explicitly as a design objective, and the mechanism is sound).
But the Failure behavior table's two relevant buckets, `characterized-behavior-changed`
and `regression`, both simply "block" with no differentiated exit code, job
name, or annotation. At the level a triager actually looks first, a red CI
check or a failed GitHub Actions job summary, an expected characterization
failure caused by a landed repair and an actual regression look identical:
both are "the suite failed." The mitigation (open the log, read the four-line
block) works, but only after someone has already reacted to an undifferentiated
red signal as if it might be either.

**Fix.** Consider giving the two buckets a distinguishing surface signal, for
example a separate CI step or job per bucket, or a machine-readable annotation
in the ledger output, so a scanning glance at a CI summary can tell "an owned
repair landed, update the pinned case" from "something broke" without opening
full logs.

---

### 10. MINOR, three `path:line` citations are off by one against the actual source

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:375-377` (N9),
repeated at `:845` (Security), and `:271` (N5 narrative).

**Problem.** Verified against the actual tracked files:

- Spec says `Tests/BERTEmbeddingTest.swift:14` hard-codes an absolute
  home-directory path. The actual line is **15**.
- Spec says `Tests/EmbeddingTest.swift:16` hard-codes `/path/to/bge-small-en-v1.5.gguf`.
  The actual line is **17**.
- Spec says "the gate at `:582`" refers to the unsatisfiable `publishProgress`
  condition. The actual condition (`if case .ready = progress, case .failed =
  progress`) is at **line 583**; line 582 is a blank line.

None of these change the substance of any claim; the described code exists and
behaves exactly as stated. But this spec explicitly builds its trust model on
exact, reproducible citation (the fingerprinting design deliberately excludes
line numbers from the hash "because unrelated edits above the anchor would
otherwise break every fingerprint," precisely so the prose citations remain the
reliable index), so citation accuracy matters here more than in an ordinary
document.

**Fix.** Correct the three line numbers in the same revision that fixes finding
2.

---

### 11. MINOR, the fingerprint extraction algorithm is described informally, with no stated implementation, and is plausibly fragile

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:603-626`
(Pinning what cannot be executed).

**Problem.** The anchor-to-fingerprint extraction is described as "extends to
the matching close brace at the anchor's indentation," with normalization that
"strips comments, collapses whitespace runs, and strips trailing whitespace."
No implementation approach is named (a real Swift parser such as swift-syntax,
versus a naive brace-counting text scan). Several of the fingerprinted
functions (for example anything in `LlamaBridge.swift`, which is dense with
`print` statements using string interpolation) could plausibly contain a `}`
inside a string literal at a column that confuses a naive counter. The spec
elsewhere is candid about the cost of an unreliable check ("a benign reformat
... trains contributors to ignore it" is exactly the risk this section itself
raises about fingerprints in general), but does not say which implementation
approach avoids it.

**Fix.** State the extraction implementation explicitly (a real parser is
strongly preferable to text scanning for this exact reason), or, if a text scan
is chosen for simplicity, note the known failure mode and require the fixture
self-test (`test-check-characterization.py`) to include at least one anchor
whose body contains a string literal with a brace, so the risk is covered by a
test rather than left implicit.

---

### 12. MINOR, the "expected-executed set" that the skip audit compares against is never given a stated source of truth

**Location:** `docs/specs/ODC-0004-v2-characterization-suite.md:586-590` (Skip
protocol), `:677-702` (File layout).

**Problem.** The skip protocol states "the runner knows the expected-executed
set for that surface: on `SFC-B` every `R0`, `R1`, and `R2` case must execute,
and every `R3` case must skip," and that a mismatch fails the run. This is the
single most load-bearing mechanism behind the answer to "can a misconfigured
CI run green with zero tests executed," and the answer the spec gives (no,
because of this comparison) is only as strong as this mechanism's actual
implementation. The File layout groups sources into `R1/`, `R2/`, `R3/`
subdirectories, which is a plausible source of truth (directory name implies
requirement class), but nothing in `## Design` or `## Interfaces` states that
directory placement is the normative signal the checker reads, as opposed to
parsing the `## Tests` markdown tables (fragile against spec/code drift) or a
separate manifest file (not mentioned). Without a stated mechanism, this is
correctly described as a policy in prose, but its "mechanically enforced"
status, which findings elsewhere in this review and the ticket's own acceptance
criteria (A9, A10) depend on, is asserted rather than specified.

**Fix.** State explicitly, in `## Interfaces` or `## Design`, which artifact is
authoritative for a test method's requirement class (directory name, a
protocol conformance, an attribute, or a generated manifest), and how
`check-characterization.py --skips` reads it.

---

## Scope discipline assessment

Clean on the program's "no runtime code before the gate" rule: `Sources/`,
`Package.swift`, `Package.resolved`, `OnDeviceCatalyst/`, and
`OnDeviceCatalyst.xcodeproj/` are all correctly declared off-limits and, per
the `git diff --stat` check above, are in fact untouched relative to
`6d72193`. The one edit this ticket makes to a tracked source
(`Tests/OnDeviceCatalystTests/OnDeviceCatalystTests.swift`, the N1 repair) is
correctly scoped to test sources only and does not touch runtime behavior.

The one leak is not a runtime leak, it is a process leak: finding 1
(`Tickets.md` already mutated) is exactly the kind of scope violation the
program's staged-review model exists to prevent, even though it involves no
source code at all.

No defect repair is smuggled into this ticket. `D1` through `D8` are correctly
characterized rather than fixed; the one exception (N1/`PredictionConfig.quality`)
is explicitly framed as a test-source-only precondition repair, not a runtime
change, and that framing is accurate.

## Answering the review brief's specific questions

- **Characterization vs. correctness confusion (attack 2).** The naming
  convention and mandatory four-line block are sound and mechanically checked
  by `--naming`. The remaining gap is CI-surface-level, not naming-level: see
  finding 9.
- **Skip discipline (attack 3).** The design (`XCTSkipUnless`/`XCTSkipIf` only,
  closed skip-code set, per-surface expected-executed sets, canaries, ledger
  audit that fails on all-skipped or zero-executed) is well thought through on
  paper and correctly closes the specific hole the repository already
  demonstrates twice (`Tests/EmbeddingTest.swift`, `Tests/BERTEmbeddingTest.swift`).
  It is not yet mechanically enforced because none of it is implemented
  (`## Validation evidence` says so explicitly), and the one piece of the
  design load-bearing enough to make the "no zero-executed green run" claim
  actually true, the expected-executed set's source of truth, is unspecified;
  see finding 12.
- **Surface partitioning (attack 4).** `SFC-A`, `SFC-B`, and `SFC-X` are
  concretely and correctly specified, and `SFC-B` was independently
  reproduced end to end. `SFC-C` is not specified to an executable level; see
  finding 4.
- **Framework choice (attack 5).** The `isInverted` argument is sound and
  verified. The non-Sendable argument is the weak link; see finding 5.
- **Source fingerprints (attack 6).** Line-number independence is correctly
  designed (hash excludes line numbers by construction). The extraction
  mechanism itself is underspecified; see finding 11.
- **Acceptance criteria (attack 7).** All but one are decidable by command as
  written. A2 is not; see finding 2.
- **Scope collision (attack 8).** Real collision found against ODC-0003; see
  finding 3. No collision found against the ODC-0010 through ODC-0020 defect
  tickets themselves beyond the ticket-mapping inconsistency in finding 7; this
  spec does not attempt to fix any of D1 through D8.
- **Template review (attack 9).** Real rigor gap against its own sibling
  template; see finding 8.
- **Maintenance cost (attack 10).** Defensible, stated plainly. 1,252 lines is
  large for a test spec, but the spec is the safety net for a codebase the
  project has already committed to replacing (ODC-0100 onward), and the
  precedent (ODC-0002) shows what an under-specified version of this exact
  document costs later: a 657-line adversarial review and a full revision
  cycle. The more expensive, less visible cost is not this document's length,
  it is the recurring "update the cases this suite pins, in the same commit as
  the repair" obligation this ticket places on eight future tickets (`##
  Ticket allocation`, obligations table), none of which yet exist as specs and
  none of which currently reference this obligation from their own side. That
  tax is real, unquantified, and worth naming explicitly in a future revision,
  but it is not, by itself, grounds to shrink this spec.

---

## Minimum set of changes required to reach APPROVED

1. **Resolve the `Tickets.md` state.** Either revert the premature `ODC-0018`,
   `ODC-0019`, `ODC-0020` rows and the ODC-0017 commentary until this spec is
   approved, or rewrite `## Ticket allocation` to honestly describe the current
   state instead of claiming rows that already exist are merely proposed
   (finding 1).
2. **Define `<base>`** concretely wherever it appears (Validation item 6,
   acceptance criterion A2), the same fix ODC-0002's own review required for
   the identical criterion (finding 2).
3. **Scope `--inventory`'s test-method ownership** to exclude ODC-0003's
   planned `Benchmarks/` subdirectory, or add an owner-marker mechanism, and
   record the resolution in both specs (finding 3).
4. **Either measure the device surface at least minimally, or name the
   device-execution mechanism itself as a second open question** distinct from
   device availability, so the obligation table does not assume a solved
   problem (finding 4).
5. **Either add a probe for the Sendable argument, or soften its language** and
   state the package's actual (Swift 5, per `swift package tools-version
   5.12.0`) default language mode so severity is not overstated (finding 5).
6. **Correct the `_llama_load_model_from_file` disassembly quote** to match the
   actual six-instruction sequence (finding 6).
7. **Reconcile the ticket-mapping divergence** between this spec's B1 table
   and `docs/baselines/v2.0.4.md`'s defect table, with one sentence stating
   which supersedes which (finding 7).
8. **Bring `docs/templates/test-spec.md` to the same guidance standard as
   `docs/templates/baseline-spec.md`**, in particular the "cannot be decided by
   a command does not belong here" line under Acceptance criteria (finding 8).
9. **Correct the three off-by-one `path:line` citations** (finding 10).
10. **State the expected-executed set's source of truth** explicitly in `##
    Interfaces` or `## Design` (finding 12).

Items 1 through 5 are the blocking set. Items 6 through 10 are required for a
spec that contains zero unresolved implementation decisions and zero
unverifiable citations, which program rule 6 and this project's own stated
evidentiary standard make a precondition for `APPROVED`.
