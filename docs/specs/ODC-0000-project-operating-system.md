---
id: ODC-0000
title: Project operating system
type: process
status: DONE
milestone: P0
owner: SankrityaT
dependencies: none
founder_approved: 2026-08-25
last_updated: 2026-08-25
evidence_fresh_until: not-applicable
unresolved_questions: none
---

# ODC-0000: Project operating system

## Summary

Create the tracked roadmap, ticket, specification, ADR, validation, and
compaction-continuity system that governs every later OnDeviceCatalyst change.

## Goals

- Make repository state sufficient to resume work after context compaction.
- Separate durable direction, execution state, specifications, and decisions.
- Prevent implementation before design and acceptance criteria are approved.
- Preserve a sanitized boundary between public product work and private research.
- Keep the process light enough for a small project and legible to contributors.

## Non-goals

- Change runtime source or behavior.
- Select a v3 engine architecture.
- Select a model family or research optimization.
- Publish confidential research context.

## Design

The public repository uses:

- `ROADMAP.md` for durable product direction and current state.
- `Tickets.md` for canonical execution status.
- `docs/specs` for one reviewed design per active ticket.
- `docs/decisions` for lasting accepted decisions.
- `docs/templates` for consistent proposal and evidence structure.
- `AGENTS.md` for mandatory start and completion protocols.
- `scripts/validate-project-state.py` for machine-checkable consistency.

GitHub Issues mirror public discussion but are not the source of execution
status. The founder accepts, returns, or rejects specs.

## Status flow

`BACKLOG → DISCOVERY → SPEC_DRAFT → SPEC_REVIEW → REVISION → APPROVED → IMPLEMENTING → VALIDATING → DONE`

Additional states are `BLOCKED`, `DEFERRED`, and `REJECTED`.

## Failure behavior

- A material unanswered decision prevents approval.
- A material design change during implementation returns the ticket to revision.
- Validation failure stays within the ticket only when the repair is in scope.
- Inconsistent ledgers, specs, or links fail the project-state validator.

## Acceptance criteria

- Roadmap identifies the active ticket, next gate, blockers, and fixed scope.
- Ticket ledger contains every approved roadmap ticket.
- Every non-backlog active ticket links to an existing spec.
- Agent instructions enforce the approval and completion gates.
- Spec and ADR templates exist.
- Project-state validation detects missing specs, status mismatches, invalid IDs,
  unresolved approved specs, and broken local Markdown links.
- No runtime source is changed.

## Review record

- 2026-08-25, completeness review: process artifacts, statuses, compaction rules,
  and public-private boundary are fully specified.
- 2026-08-25, adversarial review: avoided duplicate GitHub Issue authority,
  prohibited implementation with open decisions, and added machine validation.
- 2026-08-25, founder approval: approved through the disruption-program plan.

## Validation evidence

- `python3 scripts/validate-project-state.py` passed with 26 tickets, 3 specs,
  and 3 ADRs.
- `python3 scripts/test-project-state-validator.py` passed a valid fixture and
  rejected status mismatch, broken-link, unresolved-approved-spec, and
  unknown-dependency fixtures.
- All local Markdown links resolved during project-state validation.
- Every GitHub workflow and issue-form YAML file parsed successfully.
- `git diff --check` passed after the bootstrap replacement.
- The active ticket and next gate can be recovered from `ROADMAP.md`,
  `Tickets.md`, and this spec without chat history.
