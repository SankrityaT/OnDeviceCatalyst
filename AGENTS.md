# Repository Agent Instructions

These instructions apply to every task in this repository.

## Start-of-work protocol

Before planning or changing files:

1. Read `ROADMAP.md`.
2. Read `Tickets.md`.
3. Read the active ticket's spec and every ADR it references.
4. Inspect `git status` and preserve unrelated user work.
5. Confirm that external evidence used by the spec is no older than 14 days.

Tracked repository state is authoritative after context compaction. Do not infer
approval or progress from remembered chat text when it conflicts with files.

## Implementation gate

- Do not implement a ticket unless its ledger and spec statuses are `APPROVED`
  or `IMPLEMENTING`.
- Project infrastructure, baseline reproduction, correctness tests, and public
  benchmark tooling are the only code work allowed before the private research
  thesis gate opens.
- If implementation requires a material design decision absent from the spec,
  stop, set the ticket to `REVISION`, and update the spec.
- Do not add dependencies, change platform minimums, or change public API without
  an approved spec and ADR.

## Completion protocol

Before marking a ticket done:

1. Compare the diff against every acceptance criterion.
2. Run the required tests and builds.
3. Record exact evidence in the spec.
4. Run `python3 scripts/validate-project-state.py`.
5. Update `Tickets.md` and the Current state section of `ROADMAP.md`.
6. Update documentation and decisions affected by the change.

Failures inside the approved scope are fixed and retested. Work outside the
approved scope becomes a new ticket.

## Public and private boundary

Never add confidential research hypotheses, kernels, raw results, negative
results, experiment notes, or manuscript content to this public repository.
Only sanitized research status, public benchmark methods, and founder-approved
released artifacts belong here.

## Contribution requirements

Public commits require DCO sign-off. Never commit model weights, credentials,
private model URLs, downloaded binary artifacts, or personal device identifiers.
