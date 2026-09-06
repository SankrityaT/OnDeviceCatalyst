---
review_of: ODC-0100
spec: docs/specs/ODC-0100-v3-vision-and-migration.md
pass: 2 (adversarial)
date: 2026-09-06
reviewer: adversarial spec review
repo_revision_reviewed: 61aae27411e9b78496f1970d67a9ae965b3ec854
verdict: REJECT
---

# ODC-0100 review pass two (adversarial)

## Verdict

**REJECT**, return to `REVISION`.

This is the most consequential document in the program, and parts of it earn
that weight: the requirement-ownership table is complete, the eight
characterized v2 defects are mapped to specific tickets with accurate ticket
IDs cross-checked against `docs/baselines/v2.0.4.md` `## Characterized
findings`, the dependency-order sequencing matches `Tickets.md`'s own
dependency columns exactly, and all eleven acceptance criteria were
independently re-run and pass exactly as the document claims. But two defects
are blocking, and both are the kind this program's own precedent (four prior
`REJECT`s in this cycle) says must not survive: the document's own prose makes
a claim about its effect on `Tickets.md` that the commit introducing the
document itself contradicts, and the document's central, mechanically-checked
architectural claim (dependency optionality provable by `swift package
show-dependencies`) is not something the package structure this document
describes can actually produce under SwiftPM's real resolution semantics.

## Findings

### 1. [BLOCKING] The document's own "does not change Tickets.md" claim is false in the same commit that states it

`## Summary and user problem` (lines 25-27): "It does not itself change
`Package.swift`, `Sources/`, `Tests/`, or `Tickets.md`; those are downstream
of the tickets this spec unblocks."

`git show 61aae27 -- Tickets.md` shows the same commit that adds this 563-line
spec file also rewrites `Tickets.md`'s own `ODC-0100` row: `status` from
`DISCOVERY` to `SPEC_DRAFT`, the `Spec` column from `TBD` to
`[spec](docs/specs/ODC-0100-v3-vision-and-migration.md)`, and `Next Gate` from
`draft spec against docs/requirements/memory-and-admission.md` to
`adversarial review pass`. `Tickets.md:29` currently carries that edit. This
is not a downstream ticket's row (ODC-0101 through ODC-0104 are untouched by
this commit); it is `ODC-0100`'s own row, changed by the commit that
introduces the very document claiming `Tickets.md` was not changed.

This is exactly the defect class `docs/reviews/ODC-0005-review-pass-2.md`
finding 1 required fixed for a sibling spec on the same review day: "The
artifact and its own prose disagree about its state." It recurs here in a
sharper form, because in `ODC-0005` the contradicting edit was in a prior
commit a reader would have to look up; here it is the *same* commit, and the
commit message itself narrates the edit in plain language ("The row is now
advanced and linked") two paragraphs after the spec file it describes makes
the opposite claim in its own prose.

The document's own `## Validation evidence` section compounds this rather
than curing it. Lines 553-563 state, in the present tense, that "editing
`Tickets.md`'s status or spec-link field for `ODC-0100` is outside this
spec's authority and outside the constraints it was drafted under" as the
reason the anticipated validator error did not fire. That sentence is written
as though the edit did not happen or would be improper if it did. It already
happened, in this exact commit, and the author's own commit message says so.
The document should say plainly that it performed this edit to its own ledger
row, name the commit, and either justify why that is acceptable (advancing a
ticket's own status/spec-link when the spec that satisfies the requirement is
written in the same change is a defensible workflow step) or correct the
process. What it must not do is state, twice, in two different sections, that
this file was not touched or that touching it would be out of scope, when the
same commit touched it.

**Why A9 cannot catch this**: `## Acceptance criteria` A9 is `git diff --stat
-- Sources Tests Package.swift Package.resolved Tickets.md ROADMAP.md
docs/requirements .github` and the document's own commentary calls it "scoped
only to the paths this ticket's own constraints name as off-limits to it."
Verified independently: `git diff --stat -- Sources Tests Package.swift
Package.resolved Tickets.md ROADMAP.md docs/requirements .github` at the
current `HEAD` produces empty output, so A9 passes exactly as claimed. But
this is the same structural gap the `ODC-0005` review already named: a bare
`git diff` against the working tree cannot detect a change already folded
into the introducing commit's own history. A9 is correctly *scoped* (it does
not diff `docs/specs/` and so does not repeat the sibling-diffing defect that
sank an earlier `ODC-0003`/`ODC-0004` pass), but it is not *capable* of
detecting the specific defect present in finding 1, and the document asserts
A9's pass as evidence the constraint held, which overstates what the command
actually checked.

**Fix**: Rewrite the `Tickets.md` sentence in `## Summary and user problem` to
state plainly that this ticket's own ledger row (status and spec link) was
updated in the same commit as this draft, name the commit, and state the
scope limit correctly: the constraint is about not editing *other* tickets'
rows or *content* fields (title, dependencies, priority) for `ODC-0100`
itself, not about the status/spec-link mechanics every drafted spec must
eventually trigger. Correct `## Validation evidence`'s closing paragraph to
stop describing the edit as hypothetical or out-of-authority when it already
occurred in this commit. Either strengthen A9 to diff against the parent
commit of this ticket's own first commit (so a self-referential ledger edit in
the introducing commit is visible) or drop the implication that A9 verifies
this property at all.

### 2. [BLOCKING] The SwiftPM dependency-optionality claim is not achievable by the package shape this document describes

`## Package boundaries`, "Compile-time consequence, stated concretely" (lines
154-161): "an application that depends only on core contracts, the
execution-policy layer, and the Apple system-model backend product must
resolve and compile with zero references to the llama.cpp XCFramework or to
`mlx-swift`/`mlx-swift-lm`... ODC-0103's manifest is APPROVED only if `swift
package show-dependencies` for that consumer configuration lists neither
dependency."

This is stated as an already-decided, mechanically-checkable architectural
fact and made an approval gate for a downstream ticket (ODC-0103). It does not
hold for the package shape this document itself describes, and the gap is not
a detail, it is a property of how SwiftPM resolves dependency graphs.

`swift package show-dependencies` reports the resolved package graph declared
in a manifest's `dependencies:` array. That resolution happens at the
*package* level, not the *product* or *target* level: SwiftPM fetches and
resolves every `.package(url:...)` entry a manifest declares, regardless of
whether the specific product a downstream consumer selects actually needs it.
Selecting only certain library products changes what gets *compiled and
linked* (`swift build` only builds targets reachable from the selected
product), but it does not change what gets *resolved and reported*. This is
not a matter of interpretation; it is the exact problem SwiftPM's Package
Traits feature (SE-0452, tools-version 6.1+) was introduced to solve, and its
own proposal motivation states plainly that before traits, "all dependencies
declared in a package's manifest are resolved... regardless of whether they
are used" by the product a consumer actually depends on.

Everything in `## Package boundaries` (lines 85-152) describes llama.cpp, MLX,
Metal, and the Apple system-model backend as "optional products" that "each
depends on core contracts and the execution-policy layer's backend-conformance
protocol," drawn as one dependency tree rooted at "Core contracts (ODC-0102)."
Nothing in this document proposes splitting the backends into genuinely
separate Swift packages (separate manifests/repositories) rather than
separate products of one manifest, and nothing proposes adopting Package
Traits. The current manifest confirms this is not a hypothetical concern:
`Package.swift:1` declares `// swift-tools-version: 5.12`, which predates
Traits (6.1+) by a full major version series, and `Package.swift:28-31`
already declares `mlx-swift-lm` as a manifest-level `.package(url:...)`
dependency of the single target that exists today. If ODC-0103 keeps llama.cpp
and mlx-swift-lm as manifest-level dependencies of the same package that also
ships the Apple system-model product (which is what "optional products" of
one package graph implies throughout this section), then a consumer who
selects only the Apple system-model product will still show llama.cpp and
mlx-swift-lm in `swift package show-dependencies`, because those dependencies
are resolved for the whole manifest, not per selected product. The claim as
written would fail its own stated verification command.

This is not a minor gap. `docs/ARCHITECTURE.md:41`'s already-approved
constraint ("Heavy backend dependencies do not resolve for core-only
consumers") is exactly the property this section exists to make concrete, and
this document's job as the architecture spec was to name the *mechanism* by
which SwiftPM can actually deliver that property (a genuine multi-package
split, or a tools-version bump to adopt Traits with the migration cost that
implies for the iOS 17/macOS 14 floor this same document commits to in `##
Compatibility policy`). It does neither. It states the property as though
"optional product" packaging already secures it, hands ODC-0103 an approval
gate phrased as a fact rather than a design question, and leaves ODC-0103 to
discover on its own that the stated gate cannot be met without an
architectural decision (multi-package split vs. Traits vs. relaxing the
claim) this document was supposed to make.

**Fix**: Either (a) commit explicitly to a multi-package split, one
repository/manifest per heavy backend, and state that `## Package boundaries`
describes products *across* packages, not products *of* one package, or (b)
commit explicitly to adopting SwiftPM Package Traits and state the
tools-version floor that requires (with its consequence for the iOS
17/macOS 14 compatibility promise this document itself makes load-bearing), or
(c) weaken the claim to what a single-package, multiple-product manifest can
actually deliver (zero llama.cpp/mlx-swift-lm code in the *linked binary* for
a core-only consumer, which `swift build` plus a binary-symbol check can
verify) and drop the `show-dependencies`-based gate, which cannot pass under
that shape. Whichever is chosen, name it here; do not leave it as an
implementation detail for ODC-0103 to discover the stated gate is
unsatisfiable.

## Major findings (non-blocking individually, required before APPROVED)

### 3. The execution-policy layer is asserted five responsibilities and zero internal seams; every one of R1-R6 routes through the same single ticket

`## Package boundaries` (lines 90-96) gives the execution-policy layer one
paragraph: "device-aware backend selection, memory budgeting and eviction,
lifecycle and backgrounding, cancellation propagation, and performance
reporting." `## Requirement ownership` (lines 231-238) then assigns every one
of R1 through R6, all six binding requirements from
`docs/requirements/memory-and-admission.md`, to this same module (ODC-0101) as
primary owner, with only R1/R2/R6 additionally naming a cross-check by
ODC-0102/ODC-0003 and R3 additionally naming ODC-0104 as the artifact owner.
No row names a second owning module for the *decision logic* itself; ODC-0101
is the constant across all six rows.

Individually, the five responsibilities are not merely asserted: each traces
to specific evidence (D1's cache/shutdown race for eviction, D3's unsatisfiable
progress gate for cancellation, `docs/specs/ODC-0005-apple-platform-design-brief.md`'s
per-backend eligibility checks for selection). This is not the version of the
"bucket" defect where a name stands in for nothing. But this document, whose
job is architecture rather than survey, never subdivides the layer into
separable interfaces or types the way `## Requirement ownership`'s "Interface
satisfying it" column does for the six requirements' *data shapes* (a
memory-figure value type, a comparability flag, an admission API, a typed
error, a test suite, a performance-report value type). Nothing here says
whether backend selection and memory budgeting are the same actor, whether
cancellation propagation is a protocol requirement every backend conformance
must implement or a capability the layer imposes unilaterally, or whether
"the execution-policy layer" denotes one Swift type or several. A single
ticket (ODC-0101) inheriting literal ownership of every one of six independent
binding product requirements, with no internal boundary drawn between them
inside this document, is the concrete shape of the warning this review was
asked to test for: not proof the layer is empty, but proof it has not yet been
decomposed into anything ODC-0101 could be scoped against.

**Fix**: Add a subsection naming at least the seam between the admission/
memory-reporting concern (R1-R4, R6, data-shape-heavy) and the
lifecycle/cancellation concern (R5, behavior-heavy), even if both still live
in one SwiftPM target, so ODC-0101 has an internal contract to implement
against rather than five bullet points and six requirement rows all pointing
at the same ticket ID.

### 4. `R3` collides with an unrelated, already-`DONE` sibling meaning of `R3`, and this document repeats the collision without disambiguating

`docs/specs/ODC-0004-v2-characterization-suite.md:474` defines its own `R3` as
a test-reachability tier: "Real inference: hardware with the device slice,
plus a model asset." That document reached `DONE` (`Tickets.md:13`) before
`docs/requirements/memory-and-admission.md` was adopted
(`git log --oneline -- Tickets.md` shows `3441791` "ODC-0004 and ODC-0018
DONE" preceding `444abe9` "requirements: adopt memory and admission
requirements"). `docs/requirements/memory-and-admission.md:38` later defines
an unrelated `R3`: "Model admission is decided from measured evidence." Both
are now live, binding vocabulary in the same program, and both are named
`R3`.

This document inherits the collision without flagging it and, in one place,
uses both meanings three lines apart. `## Sequencing`, "What ODC-0021 blocks"
(lines 451-462) discusses "R3's admission decisions" (the requirement) and, in
the same paragraph, quotes `Tickets.md` verbatim: "and on ODC-0004 ('5 R3
cases inert pending ODC-0021')" (line 454), where that quoted `R3` is
ODC-0004's device-tier `R3`, not the admission requirement. A reader who does
not already know both documents' internal vocabularies would reasonably read
this paragraph as saying the same `R3` twice.

This is not this document's error to have originated (the root collision is
`docs/requirements/memory-and-admission.md` reusing `R1`-`R6` against an
already-established `R0`-`R3` tier vocabulary from a `DONE` sibling spec), but
`ODC-0100` is the architecture document whose job is requirement ownership,
and it is the document that puts both meanings in the same paragraph without
comment.

**Fix**: In the Sequencing paragraph, disambiguate the quoted `Tickets.md`
string, e.g., "ODC-0004's own device-execution tier, also labeled `R3` in that
document's vocabulary, distinct from this document's requirement `R3`." This
does not require renaming either vocabulary retroactively, only naming the
collision where both appear together.

### 5. Migration section omits the v2 public API surface `Tickets.md` itself assigns to named follow-up tickets this document never mentions

`## Migration from v2` (lines 245-355) is thorough for the eight
characterized defects (D1-D8), each correctly mapped to the ticket ID
`docs/baselines/v2.0.4.md` `## Characterized findings` assigns it. But the v2
package's public surface is larger than the eight defects, and this document,
whose stated job is "what a v2 consumer must concretely change," never
mentions three substantial public API areas:

- Tool calling: `CatalystTool`, `CatalystToolCall`, `ToolCallParser`,
  `ToolPromptFormatter` (`Sources/OnDeviceCatalyst/Tools/ToolSupport.swift:13,
  47, 67, 148`).
- Session/state persistence: `StatePersistence`
  (`Sources/OnDeviceCatalyst/Core Engine/StatePersistence.swift:12`).
- Content safety: `SafetyManager`
  (`Sources/OnDeviceCatalyst/Core Foundation/SafetyManager.swift:13`).
- Model download identity: the `CatalystModel` enum and `ModelDownloader`
  actor (`Sources/OnDeviceCatalyst/Service Layer/ModelDownloader.swift:13,
  137`).

`Tickets.md:37-38` already assigns tool calling and embeddings to named
tickets, `ODC-0203` ("Structured generation and tools") and `ODC-0204`
("Single and batch embeddings"), both dependents of `ODC-0202`. Neither ticket
ID appears anywhere in `docs/specs/ODC-0100-v3-vision-and-migration.md`
(confirmed by `grep -n "ODC-0203\|ODC-0204"`, zero matches). This document's
`## Sequencing` decomposition (lines 419-449) enumerates `ODC-0101` through
`ODC-0104`, `ODC-0200`-`ODC-0202`, `ODC-0014`, `ODC-0207`, `ODC-0023`, but
skips `ODC-0203`/`ODC-0204` entirely, and `## Migration from v2`'s closing
"What a v2 consumer must concretely change" (lines 346-355) is silent on
tool-calling, persistence, safety, and download-identity consumers, who get no
guidance at all from the one document whose job is to give it.

Line 338's passing mention of "tool-call parsing" as a "supporting service"
redistributed "into Core contracts or the execution-policy layer" is the only
place this surface is acknowledged, and it names neither a destination
ticket nor a migration action.

**Fix**: Either extend `## Migration from v2` with a row or paragraph per
omitted API area naming its destination ticket (`ODC-0203` for tool calling,
presumably `ODC-0204` for any embeddings surface, and a named or newly
allocated ticket for `StatePersistence`/`SafetyManager`/`ModelDownloader` if
none currently owns them), or state explicitly that this document's migration
section is scoped only to the eight characterized defects and that the
broader API-surface migration is out of scope, owned by a named later
document. The current text does neither; it reads as complete ("What a v2
consumer must concretely change") while leaving out API surface `Tickets.md`
itself already tracks elsewhere.

### 6. R5 has no answer for the case where ODC-0021 cannot deliver a device surface at all

`## Requirement ownership` (line 240-243) and `## Sequencing` (lines 451-462)
both state, correctly and honestly, that R5 cannot be marked satisfied until
`ODC-0021` exists. Neither section, nor `docs/requirements/memory-and-admission.md`
itself, addresses what happens if `ODC-0021` (a `BACKLOG`, unassigned,
`P0` ticket per `Tickets.md:26`, itself blocked on nothing but still
undelivered) proves unable to establish a real-device execution surface, for
example if signing, provisioning, or physical hardware access turns out to be
unavailable to this project. As written, R5 is not "architecturally assigned
but not yet satisfiable," it is assigned with an implicit assumption that
`ODC-0021` will eventually succeed, and nothing in this document or its
sibling requirements document names a contingency, a substitute evidence
standard, or a decision point for what R5 becomes if that assumption is wrong.
This matters because this document is the one place a reader would look for
what the architecture does if the device-evidence premise underneath R3 and
R5 fails to materialize, and it is silent on that question.

**Fix**: Add one sentence naming the contingency, even if the contingency is
"if ODC-0021 cannot establish a device surface, R5 and the device-evidence
half of R3 remain permanently unsatisfied and this fact must be disclosed in
any release built on this architecture," so the risk is named rather than
assumed away by omission.

## Answering the task's specific questions

### The validator gap (attack 7): real, and worth fixing

The gap is real. `scripts/validate-project-state.py:31-39` (`SPEC_REQUIRED_STATUSES`)
only requires a linked spec for tickets whose status is `SPEC_DRAFT` or later;
`DISCOVERY` and `BACKLOG` are exempt. Separately, `validate_specs()`
(`scripts/validate-project-state.py:170-172`) only checks that every spec
*file found on disk* has a corresponding row in `Tickets.md` at all, not that
the row's status or spec-link field reflects the file's existence. A fully
written, `SPEC_DRAFT`-quality document can therefore sit in `docs/specs/`
indefinitely, its own frontmatter possibly still saying an earlier status,
while `Tickets.md`'s row for the same ticket ID says `DISCOVERY` with `Spec:
TBD`, and `python3 scripts/validate-project-state.py` will exit 0 the entire
time. This is a real gap in the one tool that guards project-state
consistency for the whole program, not a cosmetic one: it means the ledger
that every other acceptance criterion in this program treats as ground truth
("`grep -q "$t" Tickets.md`", "ticket status matches spec status") can
silently diverge from the filesystem for as long as a ticket's row is left in
`DISCOVERY` or `BACKLOG`, with no mechanical signal.

**Recommended fix** (for a follow-up ticket, not this spec): in
`validate_specs()`, for every spec file discovered under `docs/specs/`, if its
`id` frontmatter matches a `Tickets.md` row, require that row to carry a
matching spec link regardless of the row's current status, not only when the
status is in `SPEC_REQUIRED_STATUSES`. The existing `unlisted` check (line
170-172) already proves the file-to-row direction is enforced; it is the
row-to-file *link* that is left unchecked below `SPEC_DRAFT`.

### Execution-policy-layer question: coherent as a concept, undecomposed as an architecture

See finding 3. The layer is not a name standing in for nothing: each of its
five responsibilities traces to specific, independently verifiable evidence
(a v2 defect or an ODC-0005 platform fact), and R1-R6's "Interface satisfying
it" column gives each requirement a distinct data shape. But the document
never draws an internal seam inside the layer itself, and having all six
binding requirements' decision logic point at one ticket ID with no internal
boundary is the concrete, checkable version of the risk this review was asked
to test for. This should be closed before `ODC-0101` starts, not discovered
during it.

### SwiftPM optionality claim: does not hold as stated

See finding 2. The claim is falsifiable by SwiftPM's own documented package-
versus-product resolution semantics and by the current manifest's
tools-version (`5.12`, pre-Traits). This is a blocking defect, not a wording
issue: the acceptance gate it sets for ODC-0103 cannot be met by the package
shape this document describes without an architectural decision (multi-package
split or a Traits adoption with its own compatibility-floor consequence) that
this document does not make.

## Attack-by-attack summary

- **Differentiator architecture (attack 1)**: real evidence behind each
  responsibility, no internal decomposition inside the layer itself. Finding
  3.
- **Optionality claim (attack 2)**: does not hold under SwiftPM's actual
  resolution model given the described package shape and the current
  tools-version. Finding 2, blocking.
- **R5 unsatisfiable (attack 3)**: honestly scoped as blocked, no contingency
  named for the case ODC-0021 fails to deliver. Finding 6.
- **Migration concreteness (attack 4)**: the eight characterized defects are
  handled well; tool calling, persistence, safety, and download identity are
  omitted entirely, including the ticket IDs (`ODC-0203`, `ODC-0204`) that
  already own two of them. Finding 5.
- **Eight not-built items (attack 5)**: six are directly required by cited
  binding text (R2, R3, ADR-0004 points) and hold up. Two are weaker: dropping
  the `OnDeviceCatalyst/` fork is justified by relative cost ("costs more than
  building ODC-0300") rather than by principle, which is a defensible but
  different kind of argument than the others make, and worth stating as such
  rather than folded into the same "rejected per evidence" framing; and
  "not carrying forward the false simulator comment" is not a design
  alternative anyone would seriously propose, so listing it as a peer item
  inflates the count of substantive non-goals with something that is really
  just "do not repeat a known-false claim."
- **Acceptance criteria (attack 6)**: all eleven re-run independently and pass
  exactly as claimed. A9 is correctly scoped against sibling `docs/specs/`
  paths (does not repeat the ODC-0003/ODC-0004 sibling-diffing defect) and A2
  does not hardcode a specific status (does not repeat that defect either).
  But A9 cannot detect finding 1's specific defect, a ledger edit folded into
  the same commit as the spec, and the document's own prose overstates what
  A9's pass demonstrates.
- **Validator gap (attack 7)**: real and worth fixing at the tool level; see
  above.
- **Boundary compliance (attack 8)**: checked against
  `.context/research/decisions/ODR-ADR-0003-thesis-selection.md`'s actual
  private terms (`jetsam`, `victim-ranking`, `ceiling escape`,
  `phys_footprint`, `resident_size`, `os_proc_available_memory`); none appear
  anywhere in `docs/specs/ODC-0100-v3-vision-and-migration.md`. The document's
  emphasis on memory budgeting, eviction, and admission is traceable entirely
  to already-public `ADR-0004` point 4 and `docs/requirements/memory-and-admission.md`,
  not to the private thesis, and the private thesis's specific mechanism
  (jetsam victim-ranking distortion) is not inferable from what this document
  chooses to emphasize; every memory-safety concern named here is the generic
  concern any on-device inference runtime would state. No leakage found.
- **Sibling consistency (attack 9)**: dependency ordering matches `Tickets.md`
  exactly across every named ticket; requirement-to-ticket mapping does not
  contradict `docs/requirements/memory-and-admission.md`'s own validation
  table. The one real defect is the `R3` naming collision with `ODC-0004`;
  see finding 4.
- **Actionability (attack 10)**: `ODC-0101` cannot be scoped without finding
  3's internal decomposition, and `ODC-0103` cannot be scoped without finding
  2's package-shape decision. `ODC-0104` is left without any definition of the
  compatibility artifact's shape (format, storage, versioning) that R3's
  admission API is stated to consult; this document names the artifact's
  owner but not its contract, which `ODC-0104` will have to invent on its own.

## Minimum set of changes required to reach APPROVED

1. **Correct the `Tickets.md` claim** in `## Summary and user problem` and
   `## Validation evidence` to state plainly that this ticket's own ledger row
   was advanced in the same commit as this draft, name the commit, and narrow
   the scope claim to what it actually means (not editing other tickets'
   rows or content fields) (finding 1, blocking).
2. **Resolve the SwiftPM optionality claim** by committing to a concrete
   mechanism, a multi-package split or a Traits adoption with its
   tools-version and compatibility-floor consequences stated, or by weakening
   the claim to a build/link check the described single-package shape can
   actually pass (finding 2, blocking).
3. **Add an internal seam inside the execution-policy layer** distinguishing
   at least the admission/reporting concern from the lifecycle/cancellation
   concern, so `ODC-0101` has more than a five-item bullet list and six
   requirement rows pointing at one ticket ID (finding 3).
4. **Disambiguate the `R3` collision** with `ODC-0004`'s device-tier `R3`
   where both appear in the same paragraph (finding 4).
5. **Extend `## Migration from v2`** to name a destination for tool calling,
   persistence, safety, and model-download-identity consumers, or state
   explicitly that this section is scoped only to the eight characterized
   defects (finding 5).
6. **Name the ODC-0021 contingency** for R5 and R3's device-evidence half
   (finding 6).

Items 1 and 2 are the blocking set: one is the same ledger/prose disagreement
class this program's own precedent has already required fixed twice this
cycle (`ODC-0004`, `ODC-0005`), and the other is an architectural claim this
document makes an approval gate for a downstream ticket that the package shape
described here cannot actually satisfy. Items 3 through 6 are required to
bring the document to the standard its own siblings were held to: no
unresolved implementation decision left for a downstream ticket to discover
on its own, and no requirement or naming gap silently inherited from a
sibling without comment.
