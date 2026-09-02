---
review_of: ODC-0003
spec: docs/specs/ODC-0003-benchmark-contract.md
pass: 2 (adversarial)
date: 2026-09-02
reviewer: adversarial spec review
repo_revision_reviewed: c15347f
verdict: REJECT
---

# ODC-0003 review pass two (adversarial)

## Verdict

**REJECT** - return to `REVISION`.

The document is careful and, on the whole, honest about what it cannot do: it
concedes plainly that no execution surface has ever run inference with either
backend, and it resists softening that into a result. That honesty is real and
should be preserved through revision. But a benchmark contract's entire job is
to be mechanically enforceable, and this pass found six independent places
where the mechanism promised in prose does not match what the schema, the
acceptance commands, or the sibling spec actually do. Two of those are
self-contradictions the contract inflicts on itself (the cold-load metric, and
acceptance criterion A8), one is a schema gap that makes a stated rule
unenforceable (`completion_reason`/`stop_reason`), one is a mechanical check
that does not do what its own row claims when run as printed (A7), one is an
unresolved boundary with the sibling spec over a shared directory, and one is
an undeclared hard dependency. None of these require inventing anything; each
is verified below against the working tree, the schema block in the document
itself, or the sibling spec's own text.

Everything below was checked with read-only commands against the working tree
at `c15347f`. No file under `Sources/`, `Tests/`, or `docs/specs/` was
modified. A narrowly scoped, read-only comparison against
`.context/research/specs/ODR-0006-preregistration.md` sections 2.3 and 4, and
`.context/research/results/2026-09-01-o2-measurement-audit.md`'s section
headers, was performed solely to check the boundary claim in finding 12; no
private content is reproduced anywhere below.

---

## Findings

### 1. BLOCKING - The contract's own required procedure guarantees its "cold" load-time metric can never measure a cold cache

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:245-254` (metric
definition), `:786-788` (Reproduction procedure step 2), `:791-801` (step 4-5,
first load).

**Problem.** Item 5 defines cold/warm operationally, by process lifetime, and
says plainly that neither macOS nor iOS lets a sandboxed process force the OS
file cache cold. That concession is honest. But `## Reproduction procedure`
step 2 requires, before any model is loaded: "Verify the model manifest named
by `ODC_BENCHMARK_MODEL_MANIFEST` exists, and that every entry's `sha256`
matches the file on disk, before any model is loaded." Computing a `sha256`
over a model file requires reading every byte of it, which populates the OS
page cache for that file. Step 4-5 then perform the harness's first ("cold" by
the contract's own definition) load of that same file, immediately after. The
contract's own mandated sequence therefore guarantees that by the time a
"cold" load happens, the file has already been read end to end by step 2's
checksum pass moments earlier. This is not the platform limitation the spec
already concedes; it is a self-inflicted ordering choice inside a procedure
this document fully controls, and it means `cache_state: cold` under this
contract measures something closer to "first load in this process, with the
OS cache already warmed by our own checksum step" than anything resembling a
cold read. Nothing in `## Workloads and metrics` item 5 or `## Confounders and
fairness controls` item 6 acknowledges this interaction.

**Fix.** Either reorder so checksum verification does not read the file
through the same path the loader will use (verify via a separate handle or a
previously cached digest recorded at manifest-authoring time), or drop the
implication that `cache_state: cold` says anything about disk-cache state at
all and rename the field to something that only claims what it can prove
(for example `first_load_in_process`), with an explicit sentence that no
version of this contract's procedure can produce a genuinely cache-cold
measurement.

---

### 2. BLOCKING - `completion_reason` and `stop_reason` promise a checked enumeration the schema does not have

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:255-261` (metric
definition), `:322-330` (Correctness gate 3), `:604-628` (metrics schema,
`completion_reason` at `:627`), `:580-603` (correctness_gates schema,
`stop_reason` at `:598`).

**Problem.** Item 6 requires completion reason be "captured verbatim as a
string from a declared, per-backend enumeration" and states "a value outside
the declared enumeration is a harness defect, not a new category to paper over
silently." Gate 3 repeats this: "a stop reason outside the declared
enumeration all fail this gate." But the JSON Schema in `## Raw artifact
schema` defines both fields as unconstrained strings:

```
"completion_reason": { "type": "string" }        # :627
"stop_reason": { "type": "string" }               # :598
```

Neither field carries an `enum`, and no other object in the schema (not
`backends[]`, not a new top-level array) has a place to declare what the
per-backend enumeration actually is. The document clearly knows how to write
an enum constraint when it wants one: `surface`, `cache_state`, `power_source`,
`prefill_duration_boundary`, and `peak_memory_window` are all schema-level
`enum`s a few lines away. The omission on exactly the two fields the prose
calls out as gate-checked is inconsistent, not a JSON Schema limitation, and it
means "a value outside the declared enumeration is a harness defect" cannot be
enforced by the schema as written; it would depend entirely on unwritten logic
inside a checker script that does not exist yet.

**Fix.** Add a `backends[].completion_reason_enum` (and, if `stop_reason` is
meant to be the same set, reuse it) as a required, non-empty array of strings
in the schema, and constrain `metrics.completion_reason` /
`correctness_gates.stop_reason` to be checked against it by the (still
unbuilt) checker, with the check named explicitly in `## Raw artifact schema`
rather than left implicit.

---

### 3. BLOCKING - The memory-basis rule has no mechanical link between `bases` and `comparable`

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:280-284` (prose
rule), `:673-688` (`comparisons[]` schema), `:748-750` ("Basis mismatch"
non-citability rule).

**Problem.** The prose is unambiguous: "Cross-backend memory comparison is
permitted only when both numbers share a stated, matching basis... Where the
bases differ... the comparison entry is recorded with `comparable: false`."
The `comparisons[]` schema (`:673-688`) requires `bases` (an array of
nullable strings) and `comparable` (a plain boolean) as independent sibling
fields with no schema-level constraint relating them - no `if`/`then`, no
enum, nothing that would reject a comparison entry whose `bases` array holds
two different strings while `comparable` is `true`. `## Raw artifact schema`
elsewhere states that non-citability determinations are "never self-declared
by the harness at capture time" and are computed by the checker, but that
claim is made for `non_citable_reasons`, not for `comparisons[].comparable`,
and no section of this document assigns the checker responsibility for
deriving `comparable` from `bases` rather than accepting whatever the harness
wrote. As written, nothing - not the schema, not a stated checker
responsibility - stops an operator from recording two different basis strings
and `comparable: true` in the same entry.

**Fix.** Either add the derivation explicitly to `## Raw artifact schema`
("`comparisons[].comparable` is computed by the checker as
`bases[0] == bases[1]`, never accepted as written by the harness") and add
that computation to the `scripts/check-benchmark.py` responsibilities listed
in `## Reproduction procedure` step 9, or restructure the field so `comparable`
is not independently settable at all (for example, drop it and let the
checker synthesize it purely from `bases` at render time).

---

### 4. BLOCKING - A7's own deciding command does not do what its row claims when run as printed

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:840`.

**Problem.** The task asked specifically that this be run, not assumed.
Copied verbatim, including the markdown-escaped pipe:

```
grep -Ein "jetsam|phys_footprint|resident_size|ceiling escape|victim.rank|os_proc_available_memory" docs/specs/ODC-0003-benchmark-contract.md \| grep -v '| A7 |'
```

Run exactly as printed (verified with both a `ugrep`-backed `grep` and
`/usr/bin/grep`), the `\|` is not a shell pipe; it is a backslash-escaped
literal character, so the shell passes `|`, `grep`, `-v`, and `| A7 |` to the
*first* `grep` invocation as extra filename arguments. Those files do not
exist, and depending on the grep implementation this either warns and still
searches the one real file, or - because trailing `-v` gets permuted into the
first invocation's option set by default argument parsing - inverts the match
entirely and dumps nearly the whole document as "output," which is very much
not "outputs nothing." Verified both ways in this environment; the literal
command does not perform the self-exclusion the row describes.

The semantically intended version (drop the backslash so `|` is a real shell
pipe) does work correctly today: it finds exactly one match, the A7 row
itself, which the `grep -v '| A7 |'` filter correctly removes, leaving no
output. That mechanism is sound. The problem is narrower but still real: the
row's own literal text, copy-pasted as an engineer following the "deciding
command" column would do, is not the command that performs the check.

**Fix.** Present the command as a real fenced code block (already used
elsewhere in this document, for example A2, A3, A6) rather than embedding a
markdown-table-escaped pipe inline in a cell, so what a reader copies is what
actually runs. Verify the corrected form still isolates exactly the A7 row
after the fix (it does, per the check above).

---

### 5. BLOCKING - A8's own deciding command is not empty against the current repository, and fails by the workflow's own necessity

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:841`.

**Problem.** A8 requires
`git diff --stat 7b5d847 -- Sources Tests Package.swift Package.resolved OnDeviceCatalyst OnDeviceCatalyst.xcodeproj Tickets.md ROADMAP.md`
to be empty. Run against the actual working tree:

```
$ git diff --stat 7b5d847 -- Sources Tests Package.swift Package.resolved OnDeviceCatalyst OnDeviceCatalyst.xcodeproj Tickets.md ROADMAP.md
 Tickets.md | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)
```

`Tickets.md` changed because this very ticket's row moved from `BACKLOG` to
`SPEC_DRAFT` and gained a spec link, which is exactly the ledger update the
program's workflow (`.context/plans/ondevicecatalyst-disruption-program.md`
rule 2) requires when a spec is drafted. A8's own included path list names
`Tickets.md` as a file that must not change, but the ordinary, required act of
drafting this spec necessarily changes it. The criterion as written can never
pass under the workflow that produces the state it is meant to check, which is
the same class of self-contradiction ODC-0002's review pass two flagged in its
finding 2 for `Package.resolved`. (Separately, the same diff interval also
touched ODC-0005's row in `Tickets.md`, unrelated to this ticket - a reminder
that a file-level diff cannot distinguish "this ticket updated its own row"
from "this ticket touched something else's row," which A8 would need to do to
mean what it says.)

**Fix.** Scope A8 to what it actually intends: no diff under `Sources`,
`Tests`, `Package.swift`, `Package.resolved`, `OnDeviceCatalyst`,
`OnDeviceCatalyst.xcodeproj` (drop `Tickets.md` and `ROADMAP.md` from the path
list, since those are expected to change), and add a separate, row-scoped
check if the intent is "no other ticket's row changed" (for example, a diff of
`Tickets.md` restricted to lines matching `^| ODC-0003 |`).

---

### 6. BLOCKING - Unresolved scope collision with ODC-0004 over `Tests/OnDeviceCatalystTests/`

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:791-801` (proposes
`Tests/OnDeviceCatalystTests/Benchmarks/`); `docs/specs/ODC-0004-v2-characterization-suite.md:648-666`
("Permitted changes to tracked files") and `:759-778` (`--inventory` checker
description).

**Problem.** ODC-0003 states it will add benchmark cases "under the same test
target ODC-0004 uses, in their own subdirectory (proposed:
`Tests/OnDeviceCatalystTests/Benchmarks/`)," explicitly to reuse ODC-0004's
build/deploy mechanism rather than invent a second one. ODC-0004's own
"Permitted changes to tracked files" table grants itself write authority over
the *entire* `Tests/OnDeviceCatalystTests/**` path ("New characterization
sources, plus the N1 repair") with no carve-out, mention, or acknowledgment of
a benchmarks subdirectory belonging to a different ticket. More concretely,
ODC-0004's `check-characterization.py --inventory` is specified to enforce
that "every case id in `## Tests` must exist as a test method... and every
test method must appear in `## Tests`" of ODC-0004's own spec - a check that,
run against a tree that also contains ODC-0003's benchmark test methods under
the same target, would either need to be told to ignore a subdirectory neither
spec currently names as excluded, or would fail on files ODC-0003 added for
reasons entirely outside ODC-0004's control. Neither document resolves this:
ODC-0003 assumes the directory is available to it: ODC-0004 assumes the whole
target is its own. This is exactly the class of gap program rule 6 forbids in
an approved spec - an unresolved implementation decision, here about who owns
part of a shared directory tree and whose checker accounts for the other.

**Fix.** Add a sentence to ODC-0004's "Permitted changes to tracked files"
table carving out `Tests/OnDeviceCatalystTests/Benchmarks/**` as owned by
ODC-0003, and state in ODC-0004's `--inventory` description that it excludes
that subdirectory by name. Cross-reference the same agreement from ODC-0003's
side so a reader of either spec sees the same boundary stated once, not
assumed twice.

---

### 7. MAJOR - The hard dependency on ODC-0004 is not declared in front matter

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:8` (`dependencies:
ODC-0002`), contradicted in substance by `:71-74` and `:791-801`.

**Problem.** The front matter names only `ODC-0002` as a dependency. But
`## Prior art and freshness date` says ODC-0004 "explicitly hands this ticket
the reusable half of that pipeline," and `## Reproduction procedure` step 4
makes the entire execution path depend on "the mechanism ODC-0004
establishes." ODC-0004 is itself `SPEC_DRAFT`, not `APPROVED` or `DONE`, and by
its own text its test target does not even compile yet until its N1 repair
lands. A spec whose only realistic reproduction path runs through a sibling
spec's not-yet-existing deliverable, without that sibling listed as a
dependency, misrepresents its own readiness to a reader or a scheduler working
from `Tickets.md`'s dependency column alone.

**Fix.** Add `ODC-0004` to `dependencies:` in the front matter, or explicitly
state in `## Prior art and freshness date` why the dependency is soft enough
not to require it (for example, "any equivalent build/deploy mechanism
satisfies this contract; ODC-0004's is reused only because it already exists,
not because it is required").

---

### 8. MAJOR - The statistics are asserted, and the specific choices sit oddly against the document's own skew claim

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:348-361` (10 minimum,
0.15 CV escalation, 30 cap), `:363-377` (percentile bootstrap, median
rationale).

**Problem.** The document is candid that these constants are "an engineering
default chosen for this contract, not inherited from any external study,"
which is honest, but honesty about the absence of justification is not the
same as justification, and the specific choices work against each other.
`## Warmup, repetitions, and statistical method` itself argues decode and
prefill timings are "typically right-skewed by rare, long-tail stalls," which
is exactly the condition under which (a) ten repetitions is a thin sample for
computing a percentile-based interval at all, and (b) the plain percentile
bootstrap is the bootstrap variant most susceptible to bias under skew,
compared to a bias-corrected variant. No sentence in this section explains why
the plain percentile method was chosen over a bias-corrected one given the
document's own skew claim, and no sentence gives a target precision or power
that ten repetitions is meant to achieve, the way a defensible minimum-`n`
choice normally would.

**Fix.** Either state a concrete precision target the floor of ten is meant to
satisfy for a representative metric (analogous to how `## Confounders and
fairness controls` requires every other parameter to be a stated, not
assumed, fact), or adopt a bias-corrected bootstrap variant and say why it was
chosen over the plain percentile method for a distribution the document itself
calls skewed.

---

### 9. MAJOR - The CV escalation threshold has no absolute-scale floor

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:353-358`.

**Problem.** `coefficient of variation (stdev / mean)` is scale-free by
construction, which makes it unstable for any metric whose typical value is
small in absolute terms even when the underlying noise is practically
irrelevant. `model_load_ms` for a warm load, or `ttft_ms` on fast hardware, are
both candidates: a sub-millisecond mean with a fraction of a millisecond of
ordinary timer jitter can produce a CV well above `0.15` from noise that no
engineer would call meaningful variance. As written, such an arm escalates to
the 30-repetition cap and is then permanently recorded `high_variance: true`,
every session, for a reason that has nothing to do with the runtime's actual
stability. The document requires the constant be treated as fixed policy
("changing them is a spec revision, not a runtime flag"), which makes this
failure mode permanent rather than something an operator can quietly work
around.

**Fix.** Add an absolute-scale floor alongside the relative one (for example,
"the CV escalation rule does not apply when `stdev` is below N milliseconds/
bytes", with `N` a named, per-metric constant subject to the same
spec-revision discipline as the other three), so a metric that is genuinely
small and stable is not indistinguishable, under this contract, from one that
is large and noisy.

---

### 10. MAJOR - TTFT's start instant is comparatively under-instrumented

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:221-226`.

**Problem.** The document is unusually careful elsewhere about making sure two
implementers cannot silently instrument different things: `peak_memory_basis`
is a required manifest field precisely so the basis is auditable rather than
assumed, and `prefill_duration_boundary` exists precisely because llama.cpp
and MLX may not expose the same prefill/decode boundary. TTFT gets no
equivalent field. It is defined as starting "immediately after prompt
tokenization completes," but llama.cpp and MLX do not necessarily expose the
same hook for "tokenization completes," and nothing in the manifest schema
requires recording which specific API call or code path was used to mark that
instant. Two competent, honest implementers instrumenting two different
backends could reasonably choose different points that both satisfy the prose
definition and still disagree by a nontrivial margin on fast hardware.

**Fix.** Add a required field analogous to `prefill_duration_boundary` - for
example `ttft_start_hook`, a free string naming the function or call site the
clock was started from, per backend, per session - so the choice is recorded
rather than assumed identical.

---

### 11. MAJOR - No self-test requirement for the harness/checker this contract commits future work to build

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:817-819`
(`scripts/check-benchmark.py`, "to be built by whichever ticket first executes
this contract"). Compare `docs/specs/ODC-0004-v2-characterization-suite.md:743-758`
(`scripts/test-run-characterization.sh`, `scripts/test-check-characterization.py`).

**Problem.** ODC-0004, this contract's own sibling and the source of its reuse
mechanism, requires self-tests for its runner and checker as a named
deliverable. ODC-0003 names `scripts/check-benchmark.py` as a future
deliverable with no equivalent requirement. Given the entire contract is
currently unexercised - `## Validation evidence` is explicit that no execution
surface has ever run inference with either backend - and given ODC-0004 has
already demonstrated that the iOS Simulator stub *can* run code deterministically
today (returning null/zero without trapping), this contract misses a cheap,
already-available opportunity to verify its own correctness-gate logic (for
example, that gate 1 correctly rejects the stub's null model-load result)
before any scarce device time is spent debugging the harness itself rather than
measuring anything.

**Fix.** Add a self-test requirement for the future checker and runner,
mirroring ODC-0004's convention, and require at least one smoke test that
exercises `## Correctness gate` against the known-deterministic simulator stub
failure mode as a cheap, CI-runnable regression test independent of Q1/Q2
closing.

---

### 12. MINOR - Boundary compliance holds on content; tighten the review record's precision on methodology structure

**Location:** `docs/specs/ODC-0003-benchmark-contract.md:51-61` (boundary
statement), `:263-297` (memory-basis requirement), `:848-858` (review record).

**Problem.** `A6` (no em dash) and `A7` (confidential-term denylist, mechanism
verified sound in finding 4 once the escaping is fixed) both pass against the
current text, and a targeted, read-only comparison against the two private
documents named in the review record found no reproduced hypothesis, finding,
measured value, or private reasoning chain - the specific terms in A7's
denylist, the private numeric results, and the private hypothesis names do not
appear here. That is a genuinely clean result and should be recorded as such.

Worth tightening rather than blocking on: the memory-basis requirement's
overall shape - always declare a basis, report per-backend bases side by side,
never silently merge, mark non-comparable explicitly rather than omitting the
comparison - is the same shape a rigorous methodology document would use for
any cross-runtime measurement problem, and the boundary statement already
anticipates and defends exactly this resemblance in general terms. Given how
close the shape is, the review record's claim that "no private hypothesis,
finding, measured value, or reasoning chain is reproduced" would be strengthened
by one more sentence stating plainly that the *structure* of this section, not
only its stated reason, was informed by the private review, so the claim is
precise about what was and was not carried over.

**Fix.** Add one sentence to `## Review record` making that distinction
explicit. No content change is required.

---

## Scope discipline assessment

The spec is clean on the program's "no new runtime code before the research
gate" rule: `## Reproduction procedure`'s closing line and `A8` both aim at
"no file under `Sources/`, `Tests/`, `Package.swift`... changes," and, modulo
finding 5's scoping bug, that is the correct instinct. No behavior change is
smuggled in, and the deliberately absent acceptance criteria section
(`:844-846`, "any criterion asserting a benchmark ran, passed a gate, or
produced a number") is exactly the kind of honesty this review wants to see
more of, not less.

The one real scope leak is finding 6: the directory ownership question with
ODC-0004 is not a hypothetical future problem, it is a decision this spec
already tries to make unilaterally ("proposed:
`Tests/OnDeviceCatalystTests/Benchmarks/`") without the sibling spec's
agreement, which is the same category of gap ODC-0002's review pass two
flagged when it found the Xcode-app source fork unmentioned by a spec whose
procedure depended on it.

---

## Minimum set of changes required to reach APPROVED

1. **Resolve the cold-load ordering problem** (finding 1): either reorder
   checksum verification off the load path, or rename the field so it claims
   only what it can prove.
2. **Add the missing enumeration to the schema** (finding 2): a
   `backends[].completion_reason_enum` (or equivalent) that `completion_reason`
   and `stop_reason` are actually checked against.
3. **Add the missing basis/comparable derivation** (finding 3): state that the
   checker computes `comparable` from `bases`, never accepts it as written by
   the harness, and add that responsibility to the checker's listed duties.
4. **Fix A7's presentation** (finding 4): a real fenced code block, not an
   inline markdown-table-escaped pipe, so what a reader copies is what runs.
5. **Fix A8's path list** (finding 5): drop `Tickets.md` and `ROADMAP.md` from
   the "must be empty" diff, or replace with a row-scoped check.
6. **Resolve the `Tests/OnDeviceCatalystTests/` ownership question with
   ODC-0004** (finding 6): a carve-out in ODC-0004's permitted-changes table
   and inventory-checker description, cross-referenced from both specs.
7. **Add `ODC-0004` to `dependencies:`** in the front matter, or explicitly
   justify why the dependency is soft (finding 7).
8. **Justify or revise the statistics** (findings 8-9): state a precision
   target for the ten-repetition floor or adopt a bias-corrected bootstrap
   variant with a stated reason, and add an absolute-scale floor to the CV
   escalation rule.
9. **Add a `ttft_start_hook`-equivalent field** (finding 10), matching the
   rigor already applied to `peak_memory_basis` and `prefill_duration_boundary`.
10. **Add a self-test requirement for the future checker/runner and a stub-based
    smoke test of the correctness gate** (finding 11).
11. **Add one sentence to `## Review record`** making explicit that the
    memory-basis section's structure, not only its reason, was informed by the
    private review (finding 12).

Items 1-6 are the blocking set. Items 7-11 are required for a spec that
contains zero unresolved implementation decisions, which program rule 6 makes
a precondition for `APPROVED`.
