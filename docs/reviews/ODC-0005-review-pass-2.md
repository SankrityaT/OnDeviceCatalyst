---
review_of: ODC-0005
spec: docs/specs/ODC-0005-apple-platform-design-brief.md
pass: 2 (adversarial)
date: 2026-09-02
reviewer: adversarial spec review
repo_revision_reviewed: c65b3eb4e90a587790e323dab13e0aee04cee9ef
verdict: REJECT
---

# ODC-0005 review pass two (adversarial)

## Verdict

**REJECT**, return to `REVISION`.

The external half of this document, the twelve Apple DocC citations and the
one PDF, is the strongest evidentiary work seen in this program so far. Every
load-bearing Apple availability claim checked in this pass, including the one
the task flagged as consequential enough to have changed a hardware-purchase
recommendation, was independently re-verified against Apple's live
documentation and found correct, in several cases to the exact JSON field.
That is real and should be preserved through revision.

But the internal half, this document's citations of its own program's prior
artifacts (`docs/baselines/v2.0.4.md`, `docs/specs/ODC-0004-v2-characterization-suite.md`,
its own `Tickets.md` state, and its own cited "program rule"), does not meet
the same bar the external half sets. Two of the problems below are the same
class of defect that produced REJECT on ODC-0003 and ODC-0004 earlier in this
same review cycle: a document's own prose asserting a state of the world (an
untouched ledger, an independently-shown fact) that the working tree
contradicts. A third and fourth are citations to findings that do not exist as
cited, in a document whose entire stated premise is that no claim escapes a
dated, checkable citation.

## Independently verified Apple claims

Fetched live in this pass (2026-09-02), against the same endpoints the spec
cites, with no reliance on the spec's own transcription:

| Claim | Spec location | Verified |
| --- | --- | --- |
| `FoundationModels` shipping, non-beta, `introducedAt: 26.0` on iOS/iPadOS/macOS/visionOS/Mac Catalyst; watchOS only at `27.0`, `beta: true` | lines 115-119 | Matches exactly |
| `SystemLanguageModel.tokenCount(for:)` `beta: false`, `introducedAt: "26.4"` on the same five platforms | lines 162-167 | Matches exactly, byte for byte against the fetched JSON |
| `LanguageModel` `beta: true`, `introducedAt: "27.0"` on iOS/iPadOS/macOS/visionOS/Mac Catalyst/watchOS | lines 203-212 | Matches exactly |
| `LanguageModelExecutor` `beta: true`, `introducedAt: "27.0"`, same platform set | lines 203-212 | Matches exactly |
| `MTLTensor` `beta: false`, `introducedAt: "26.0"` on iOS/iPadOS/macOS/visionOS/Mac Catalyst/tvOS; no chip or GPU-family floor stated anywhere in the symbol JSON | lines 301-307 | Matches exactly |
| Metal Performance Primitives PDF: "Metal 4 introduces the tensor resource and the Metal Performance Primitives (MPP) framework for authoring machine learning kernels that leverage GPU neural accelerators in the Apple M5 chip," dated 2026-03-16, "Version 1" | lines 316-323 | Verbatim match, confirmed via `pdftotext` on the downloaded PDF; every M5-specific tuning constant in the guide (simdgroup tile size, `SM == SN == 32`, `BK == 128`) is scoped to "the M5 chip" exactly as the spec states, and the PDF contains **no** mention of A19, A18, A17, or any iPhone chip anywhere in its text, which independently supports the spec's decision to mark the A19 claim `UNVERIFIED` rather than assume parity |
| `LanguageModelSession` shipping at `26.0` on the same five platforms, `beta: true` / `27.0` on watchOS only; `ResponseStream` described as snapshots of partially generated content (partial-snapshot streaming); `contextSizeExceeded` documented | lines 134-141 | Matches |
| `BackgroundAssets` base APIs: iOS/iPadOS/Mac Catalyst `16.0`, macOS `13.0`, tvOS `18.4`, visionOS `2.4` (or `1.0`+ for compatible iPad/iPhone apps under visionOS compatibility) | lines 366-371 | Matches exactly, including the visionOS compatibility-app carve-out |

This is seven of the twelve DocC citations, all confirmed. The specific claim
the task called out as consequential, that `MTLTensor` is software-everywhere
at 26.0 while the GPU Neural Accelerator hardware backing it is M5-specific,
holds exactly as written and is the one claim in this document I would call
fully load-bearing-safe.

## Findings

### 1. [BLOCKING] The document's own "not self-applied" claim is false as of this revision

`## Review record` (lines 790-794) states: "The ticket title correction
proposed in `## Title correction` is a recommendation for the ledger owner to
apply; this document does not and cannot self-apply it, per program rule 2 and
this ticket's own constraint against editing `Tickets.md`."

That is contradicted by the working tree. `git show c2a8975 -- Tickets.md`
shows the same commit that added this spec file (`c2a8975`, "ODC-0005: Apple
platform capability brief; rename from the WWDC25 framing") also rewrote
`Tickets.md`'s ODC-0005 row title from "WWDC25 Apple-native design brief" to
"Apple platform capability brief for v3 architecture", the exact string
proposed in `## Title correction` (line 32). `Tickets.md:14` currently reads
that corrected title. The correction was not left pending for a ledger owner;
it was applied in the same commit as the document that claims it was not.

This is the identical defect class `docs/reviews/ODC-0004-review-pass-2.md`
finding 1 required fixed for a sibling spec on the same review day ("The
ticket ledger this spec claims not to touch has already been touched... The
artifact and its own prose disagree about its state"). `ODC-0005`'s commit
(`c2a8975`, 02:12:00) predates that finding (`838bdfa`, 02:26:44), so the
author had not yet seen the lesson when writing it, but the defect is present
in the current HEAD and this pass must catch it regardless of authorship
order.

**Fix**: Rewrite `## Title correction` and `## Review record` to state the
true, current situation: the title has already been corrected in `Tickets.md`
(state the commit), not that it remains an unapplied recommendation. If the
title change was itself premature (applied before founder review of this
`SPEC_DRAFT`), say so explicitly as a process note, the way ODC-0004's
revision reconciled its own broken ticket mapping. Also strengthen acceptance
criterion A8 (line 741), which only diffs the current working tree and cannot
detect a change already folded into the ticket's own commit history; it
currently cannot fail even when this exact defect is present.

### 2. [BLOCKING] A load-bearing claim is attributed to a section that does not contain it, and is never independently verified

`## Architecture and data flow`, "Deliberately not pursued" (lines 541-548):
"ADR-0004 point 2 is explicit that this contest is lost on a schedule measured
in weeks once Apple's own protocol reaches GA, and **section 2 above shows**
Apple's own MLX team is separately building the exact same 'any model behind
one session API' bridge for its own models."

Section 2 (`## Current behavior and evidence`, "2. The custom-provider
protocol: preview, not a requirement source", lines 201-251) discusses only
`LanguageModel` and `LanguageModelExecutor`. It never mentions MLX,
`mlx-swift-lm`, or `MLXFoundationModels` anywhere. The fact being cited
("Apple's own MLX team is separately building" the bridge) exists only in
`docs/decisions/ODC-ADR-0004-apple-api-conformance-over-competition.md`'s
Context section ("`mlx-swift-lm` already ships `MLXFoundationModels`,
documented as a bridge from MLX models into `FoundationModels.LanguageModel`
and requiring the 27.0 SDK"), which this brief never re-cites, never
re-fetches, and never dates in this pass.

This breaks the document's own stated evidence rule twice at once: the claim
is misattributed to a section that does not support it, and it is presented
as settled fact rather than marked `UNVERIFIED`, in a document whose opening
paragraph (lines 55-60) says exactly this situation, an unconfirmed claim used
to support an architectural conclusion, is the one thing that must never
happen silently.

**Fix**: Either independently re-verify the `MLXFoundationModels` claim (check
`mlx-swift-lm`'s current source or release notes with a fresh access date) and
cite it properly, or mark it `UNVERIFIED` and correct "section 2 above shows"
to "ADR-0004 records." The underlying architectural conclusion (do not build a
bespoke unified-backend API) does not depend on this specific fact, ADR-0004
point 2 already mandates it directly, so the fix is citation hygiene, not a
reversal of the decision.

### 3. [MAJOR] Two citation labels do not exist in the document they cite

Line 279 (section 3, MLX): "...the package's *llama.cpp* path ships a
non-functional simulator stub for an unrelated reason (**E1** in
`docs/baselines/v2.0.4.md`)."

Line 452 (section 6): "...the shipped XCFramework has no macOS slice at all
(baseline finding **E4/D4**)."

`docs/baselines/v2.0.4.md`'s only enumerated defect labels are `D1` through
`D8` (`## Characterized findings`, lines 381-394; confirmed by
`grep -n "^| D" docs/baselines/v2.0.4.md`, which returns exactly `D1`-`D8` and
nothing prefixed `E`). No `E1` or `E4` label appears anywhere in that file.
The simulator "links, cannot infer" result the first citation is pointing at
is real (`docs/baselines/v2.0.4.md` lines 176-197, `## Build matrix`), and the
macOS failure the second citation points at is real and correctly also
labeled `D4` in the same sentence, but the `E`-prefixed labels themselves are
fabricated and do not resolve to anything in the cited source.

**Fix**: Drop `E1` and `E4`; cite the build-matrix row directly (e.g., "the
`ios-simulator` `links, cannot infer` result, `docs/baselines/v2.0.4.md` lines
176-177") and cite the macOS failure as `D4` only, without the invented `E4`
alias.

### 4. [MAJOR] A finding this brief treats as settled is an open, unobserved question in its own cited source

Section 4 (lines 352-356): "`docs/specs/ODC-0004-v2-characterization-suite.md`
(finding N7) additionally recorded that `xcodebuild` *does* compile the
shaders, into a resource-bundle target, but `makeDefaultLibrary()` reads
`Bundle.main`, not `Bundle.module`, so the shader library is not found there
either."

`docs/specs/ODC-0004-v2-characterization-suite.md` itself does not state this
as recorded fact. N7's own text (lines 365-372) says: "That observation also
**refines** D5 rather than merely confirming it... `makeDefaultLibrary()`,
which reads `Bundle.main`, not `Bundle.module`. So the shader library would
not be found even when it is built. **Not observed directly here**, because
the toolchain component is absent; recorded as open question Q2 and handed to
ODC-0014." ODC-0004's own open-questions section (lines 1119-1129) repeats the
hedge explicitly: "**If that is right**, D5's severity is unchanged but its
root cause has two independent halves... Not observed here because the host
lacks the Metal Toolchain component... Until it is closed, `C-D5-4` asserts
only that construction throws, and does not assert which of the two messages
is produced."

ODC-0005 drops every hedge from its source and states the `Bundle.main` /
`Bundle.module` mismatch as a recorded fact, with no `UNVERIFIED` marker and
no note that ODC-0004 itself calls this an unobserved, open question (Q2).
This is exactly the failure mode the review brief's attack 2 asks about: an
unmarked claim the argument treats as settled when its own source does not.

**Fix**: Add a caveat matching ODC-0004's own hedge, e.g., "not directly
observed by ODC-0004 (its own open question Q2), pending the Metal Toolchain
component," and consider whether this claim needs its own `UNVERIFIED` mark
or an explicit note in the front matter's unresolved-questions list.

### 5. [MINOR] Misquoted section heading

Line 579-580: "The v2.0.4 baseline's own defect D1 (`docs/baselines/v2.0.4.md`,
'confirmed defects')..." `docs/baselines/v2.0.4.md` has no section titled
"confirmed defects"; the section containing D1 is `## Characterized findings`
(line 381), whose own text says "**This ticket fixes none of these**", the
opposite emphasis of "confirmed defects" as a heading label.

**Fix**: Quote the actual heading, `## Characterized findings`, or drop the
quotation marks and just cite the section by its real name.

### 6. [MINOR] Unlocatable citation

Line 794 cites "program rule 2 and this ticket's own constraint against
editing `Tickets.md`" as the basis for not self-applying the title correction.
Neither is locatable: `ROADMAP.md`'s seven numbered Operating gates contain no
rule about ledger edits (gate 2 is "Founder approval is required before
implementation"); `.context/plans/ondevicecatalyst-disruption-program.md`'s
nine numbered process rules (rule 2 is "Draft specs define goals, non-goals,
design, interfaces, data flow, failure behavior, security, migration, tests,
benchmarks, and acceptance criteria") also contain no such constraint; and
this ticket's own `## Non-goals` section never states a constraint against
editing `Tickets.md`. This citation cannot be checked against anything.

**Fix**: Either name the actual rule this is based on, or replace the citation
with the real justification (founder-approval-before-implementation, rule 5 of
the disruption-program plan: "The founder makes the final accept, revise, or
reject decision").

### 7. [MODERATE] Staleness obligation is generic, not concretely enumerated

The document states its 14-day freshness window (`evidence_fresh_until:
2026-09-16`) and, in `## Failure modes` (lines 638-643), notes preview-API
churn generically. But ADR-0004's own consequence list states plainly: "A
second landscape refresh is required immediately after iOS 27 general
availability, because this decision is built on a beta API's shipping
behavior" (`docs/decisions/ODC-ADR-0004-apple-api-conformance-over-competition.md`,
line 67). ODC-0005 never restates this as its own obligation, never names
which of its twelve citations are the ones most likely to move at iOS 27 GA
(everything currently `26.4`-gated or `27.0`-beta-gated, i.e., `tokenCount`,
`LanguageModel`/`LanguageModelExecutor`, and the watchOS rows), versus the
ones that are stable regardless (the `16.0`/`13.0` Background Assets floors).
A reader who returns to this document after iOS 27 ships has to re-derive
which sections need rechecking rather than being told.

**Fix**: Add one sentence to `## Failure modes` or `## Tests and device
validation` naming the specific re-verification obligation ADR-0004 already
places on the program (a landscape refresh at iOS 27 GA) and which sections
and citations in this document it applies to first.

### 8. [OBSERVATION, non-blocking] Self-contradictory phrasing in section 1

Lines 126-130: "Apple's own documentation states the model is versioned
independently of the OS: as of this evidence date it names three model
generations aligned to OS releases..." "Independently of the OS" and "aligned
to OS releases" read as contradictory on a literal pass. The live Apple text
(fetched in this pass) actually says: "Apple periodically updates
`SystemLanguageModel` in routine OS updates... there are 3 model versions that
align with" three OS ranges, i.e., the model version does not bump on every
OS point release, not that it is unrelated to OS versioning. The underlying
fact the spec cites (three generations: `26.0`-`26.3`, `26.4`, `27.0`) is
correct; only the framing sentence is confusing.

**Fix**: Reword to something like "the model version does not change on every
OS point release" rather than "independently of the OS."

### 9. [WATCH, non-blocking] Adapter-boundary language is close to the ADR-0004 line but currently on the correct side

`## Current behavior and evidence`, section 2, "Adapter-boundary design
consequence" (lines 236-251) recommends that the v3 backend abstraction
"should already look... like a `respond`-shaped async streaming call over an
opaque model handle" because that is the shape of the beta 27.0 protocol.
ADR-0004 point 5 permits preview APIs to "inform planning" but forbids them
defining "acceptance criteria." This passage is explicitly hedged ("if that
shape holds"), explicitly non-binding ("no interface is fixed here"), and no
acceptance criterion in `## Acceptance criteria` depends on it. It stays on
the permitted side of the line as written. Flagged only so a future revision
does not let this passage harden into an implicit requirement on ODC-0102 /
ODC-0103 without a fresh accept/reject decision at that time.

## Attack-by-attack summary

- **Apple claims (attack 1)**: Seven of twelve DocC citations plus the PDF
  independently spot-checked; all confirmed exactly, including the
  consequential `MTLTensor`/M5-Neural-Accelerator split. No defect found on
  this axis.
- **UNVERIFIED marks (attack 2)**: The ten markers present are honestly
  placed for the facts they cover (Foundation Models context window, MLX
  simulator behavior, Metal's chip floor, A19 accelerator parity, Background
  Assets eviction and integrity metadata); genuinely non-load-bearing as
  claimed. The real problem is the inverse failure mode: load-bearing claims
  that should carry a marker but do not (findings 2 and 4).
- **ADR-0004 conformance (attack 3)**: All four decision points (iOS 17 floor,
  no bespoke unified API, optional adapter only post-GA, differentiator to
  execution policy) are correctly implemented in `## Architecture and data
  flow` and `## Proposed interfaces`. No contradiction found.
- **Preview-API discipline (attack 4)**: No acceptance criterion depends on a
  27.0 API. See finding 9 for the one passage worth watching going forward.
- **Actionability (attack 5)**: The architecture diagram and its "stated
  concretely" bullets (lines 527-559) give real module-boundary decisions, not
  just description. No defect found beyond the citation issues above.
- **Acceptance criteria (attack 6)**: All ten criteria were run against the
  working tree. All ten pass exactly as written, including `A9`
  (`python3 scripts/validate-project-state.py`, exit 0) and `A7`/`A6`'s count
  thresholds. Unlike ODC-0003's review pass, no criterion failed to run as
  printed and none is structurally unpassable.
- **iOS 17 promise (attack 7)**: Section 6 names three concrete testing
  obligations (compiling build at the floor, no unconditional above-floor
  import, a device or simulator run on the oldest OS) and assigns them to
  ODC-0302/ODC-0303, not merely a risk note. No defect found.
- **Boundary compliance (attack 8)**: Checked the public brief against
  `.context/research/decisions/ODR-ADR-0003-thesis-selection.md`'s actual
  private thesis terms (`jetsam`, `phys_footprint`, `victim-ranking`,
  `vmmap`, `fc-thrashing`, `mmap`, `resident_size`); none appear anywhere in
  `docs/specs/ODC-0005-apple-platform-design-brief.md`. The brief's emphasis
  on memory budgeting and eviction as an execution-policy concern is drawn
  from already-public `ADR-0004` point 4 language, not from the private
  thesis. No leakage found, and the private thesis is not inferable from what
  this document emphasizes.
- **Staleness (attack 9)**: States its own expiry via `evidence_fresh_until`
  but does not concretely enumerate the re-check obligation; see finding 7.
- **Maintenance cost (attack 10)**: Defensible. The document is 805 lines for
  a capability survey feeding multiple downstream tickets (ODC-0014, ODC-0102,
  ODC-0103, ODC-0104, ODC-0206, ODC-0207, ODC-0302, ODC-0303), and its own
  14-day evidence rule already forces a maintenance cadence. The cost worth
  naming explicitly is the citation-precision debt found above: a document
  that stakes its authority on "no claim without a dated citation" degrades
  faster than a looser document if its internal citations (to its own
  program's baseline and characterization artifacts) are allowed to drift
  from that standard while its external Apple citations stay rigorous.

## Minimum set of changes required to reach APPROVED

1. **Correct `## Title correction` and `## Review record`** to state that the
   ledger title has already been changed (in commit `c2a8975`), not that it
   remains an unapplied recommendation, and strengthen A8 so it can actually
   detect this class of defect (finding 1).
2. **Fix the "section 2 above shows" MLX-bridge claim**: either independently
   verify and cite it with a fresh access date, or mark it `UNVERIFIED` and
   correct the false cross-reference (finding 2).
3. **Remove the fabricated `E1` and `E4` citation labels** and cite the
   underlying baseline rows directly (finding 3).
4. **Add ODC-0004's own hedge to the `Bundle.main`/`Bundle.module` claim** in
   section 4, noting it is that document's own unobserved, open question
   (finding 4).
5. **Fix the "confirmed defects" misquote** to the baseline's actual heading
   (finding 5).
6. **Replace the unlocatable "program rule 2" citation** with a real,
   checkable rule (finding 6).
7. **Name the concrete iOS 27 GA re-verification obligation** and which
   sections it hits first (finding 7).

Items 1 through 4 are the blocking set: two are the same ledger/prose
disagreement class this program's own precedent already treats as REJECT, and
two are citations that do not resolve to what they claim to cite in a document
whose sole differentiator from a blog post is that its citations resolve.
Items 5 through 7 are required for a document with zero unverifiable internal
citations, the same bar its Apple-facing half already clears.
