---
id: ODC-NNNN
title: Replace with title
type: test
status: SPEC_DRAFT
milestone: P0
owner: unassigned
dependencies: none
founder_approved: pending
last_updated: YYYY-MM-DD
evidence_fresh_until: YYYY-MM-DD
unresolved_questions: list-or-none
---

# ODC-NNNN: Title

## Summary

One paragraph. What behavior is being pinned, at which revision, and what a
later change must do when a pinned case fails. Name the ticket or spec this
one inherits its subject from, if any.

## Goals

## Non-goals

## Current state and evidence

The facts already known at spec time, each with a `path:line` citation and the
command that reproduces it. Distinguish facts inherited from an earlier
baseline or spec (label them, for example `B`) from facts this ticket's own
discovery established (label them, for example `N`). A test spec that
characterizes a claim it never independently verified is not characterizing
anything; every new fact needs the command that produced it, not just a
description of it.

## Design

State the test framework and justify it against the deployment targets. A
justification with no cited evidence is not a justification; every ground for
the choice needs a probe, a diagnostic, or a fact that was actually checked,
not merely asserted plausible. State the execution surfaces, and for each one
state what can and cannot run there, and whether it was actually measured or
only asserted. A surface that is asserted runnable but not specified to an
executable level (signing, provisioning, deployment, or whatever the surface
actually requires) must say so explicitly and name the ticket that owns
closing the gap, rather than being left to look solved. State the naming
convention that distinguishes a characterization assertion (pins current,
possibly wrong, behavior) from a correctness assertion (asserts intended
behavior and must never regress), and how it is mechanically checked rather
than left to reviewer memory. State the skip protocol, including the closed
set of skip reasons and how an absent precondition produces a recorded skip
rather than a silent pass, because a suite that can go green with zero cases
executed is worse than no suite.

## Interfaces

The exact shape of every support type, runner flag, and checker mode this
ticket adds, precise enough that a second implementer produces the same
command surface without guessing. An interface that exists only as prose, with
no named flags, exit codes, or file layout, is not yet decided.

## Data flow

The direction of flow from source under test, to execution surface, to raw
output, to checker verdict, to exit code. State what is tracked in the
repository and what is not, and why: a result that depends on the operator's
environment and would rot within days is usually a reason not to track it.

## Failure behavior

Every failure class the suite or its runner can encounter, the bucket it is
recorded in, and whether it blocks completion. The taxonomy must be exhaustive
over the failures observed while drafting, including failures caused by
untracked or ignored local state. Call out, specifically, whether a pinned
defect failing (expected, because a repair landed) and a genuine regression
failing (unexpected) are distinguishable at the level a reader actually looks
first, not only after opening the full log.

## Security and privacy

The denylist, the scanner command, and what this ticket deliberately never
commits, for example model weights, device identifiers, or absolute local
paths. State it for what this ticket adds; do not describe the codebase's
security posture in general.

## Migration and compatibility impact

Any tracked file this ticket changes, and the consumer-facing consequence.
A characterization or regression suite usually earns the right to say "no
public API changes" plainly, because it observes behavior rather than
changing it, but say so explicitly rather than leaving the section bare.
"None" is an acceptable answer only when stated explicitly.

## Tests

The catalog, and it is normative: the implemented suite and this table must
agree in both directions, mechanically checked. Every case names its
identifier, its execution surface, its requirement class if the spec defines
one, the defect or behavior it pins, what it asserts today, and what the
assertion becomes once the defect is repaired or the behavior changes
deliberately.

## Benchmarks

State plainly whether this ticket produces any timing, throughput, or memory
claim. If benchmarks are out of scope, name the ticket that owns them and
state the mechanism, if any, that prevents a benchmark claim from being
smuggled into a test assertion.

## Open questions and gates

Every open question that must close before this spec can leave `SPEC_REVIEW`,
each with a deciding command (or an honest statement that no deciding command
exists yet) and the enumerated set of acceptable outcomes. An outcome that
resolves to "not measured" is acceptable only when it is a deliberate
decision with a named owner for what happens next, not an omission.

## Validation

Each line is a command and the condition on its exit code or output decides
it. If any command below needs a base revision, a log path, or any other
symbol, resolve it here to a literal, concrete value; do not leave a
placeholder like `<base>` for a human to fill in later. State, if the project
has one, what the general-purpose validator already checks and what only this
ticket's own checker enforces, so a reviewer does not over-trust a green
general validation run.

## Acceptance criteria

Each line is a command plus the exit condition that decides it. **A criterion
that cannot be decided by a command does not belong here.** A criterion that
depends on an undefined symbol, an unstated path, or an unresolved
placeholder is not decidable and must not ship that way.

## Ticket allocation

The reserved ID range this ticket allocates from, and the default ledger
column values for any ticket it creates. If a ticket was already added to
`Tickets.md` before this spec reached an approved status, say so explicitly,
name the commit, and explain why, rather than describing an already-real
ledger row as a proposal; the ledger and the spec's own prose must never
disagree about the ledger's state. State any obligation this suite places on
other tickets: a case that pins a defect another ticket repairs, which that
ticket must update in the same commit as its repair, so the suite does not
decay into a wall of red nobody is obligated to fix.

## Alternatives considered

## Review record

Pass, date, reviewer, and the path to the review artifact.

## Validation evidence
