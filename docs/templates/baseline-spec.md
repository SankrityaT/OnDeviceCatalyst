---
id: ODC-NNNN
title: Replace with title
type: baseline
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

One paragraph. What state is being measured, at which revision, and which later
tickets consume the result.

## Goals

## Non-goals

## Pinned environment and clean state

Everything a second operator must match before any measurement command runs.
A baseline spec is invalid without this section, because an unpinned confounder
turns a disagreement between two honest runs into an unfalsifiable dispute.
At minimum: exact toolchain build numbers, exact SDK build numbers, host and
device identity classes, dependency cache policy, and the mandatory clean-state
commands whose output is captured into the report.

## Current state and evidence

The facts already known at spec time, each with a `path:line` citation and the
command that reproduces it. This is the section that prevents the procedure from
"discovering" only what the author expected. Anything a reviewer has already
verified belongs here, not in the procedure.

## Design

What the deliverables model and why. A baseline is a data product: state its
schema, its normalization rules, its stability guarantees, and what downstream
consumers are allowed to depend on.

## Interfaces and data flow

The exact shape of every emitted artifact, the exact commands that produce it,
and the direction of flow from command output to normalized field to consumer
ticket.

## Required procedure

Numbered, ordered, each step naming its command and the artifact field it fills.

## Failure behavior

Every failure class the procedure can encounter, the bucket it is recorded in,
and whether it blocks completion. The taxonomy must be exhaustive over the
observed failures, including failures caused by untracked or ignored local
state.

## Security, privacy, and redaction

The denylist, the scanner command, and what is deliberately not captured.

## Migration and compatibility impact

Any tracked file this ticket changes, and the consumer-facing consequence.
"None" is an acceptable answer only when stated explicitly.

## Tests and benchmarks

Either the tests and benchmarks this ticket adds, or an explicit statement that
they are out of scope with the owning ticket named.

## Validation

Each line is a command plus the exit condition that decides it.

## Acceptance criteria

Each line is a command plus the exit condition that decides it. A criterion that
cannot be decided by a command does not belong here.

## Ticket allocation

The reserved ID range, the rubric that separates actionable from recorded, and
the default ledger column values for tickets this procedure creates.

## Alternatives considered

## Review record

Pass, date, reviewer, and the path to the review artifact.

## Validation evidence
