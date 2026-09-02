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

## Ledger and prose must be changed together

**Recurring defect, observed three times on 2026-09-01.** Adversarial reviews of
ODC-0004 and ODC-0005 both returned blocking findings of the same shape: a spec's
prose said it does not, or cannot, change `Tickets.md`, while the very commit that
added the spec had already changed it.

The cause is a division of ownership. Specs propose ledger changes; the manager
owns `Tickets.md` and applies them. When the manager applies a proposal without
updating the prose that describes it as unapplied, the two records disagree, and
the disagreement is committed before any reviewer sees it.

**Rule.** Applying a ledger change that a spec proposed is not complete until the
proposing spec's text is corrected in the same edit. Concretely, after editing
`Tickets.md` on a spec's behalf:

1. Grep that spec for language asserting it does not, cannot, or will not modify
   the ledger, and for any section presenting the change as a proposal.
2. Correct that language to state the change is applied, naming the commit.
3. If the spec is being edited concurrently by another agent, hand the correction
   to that agent explicitly rather than editing the file underneath it.

**Related rule.** Do not run `git add -A` while an agent is editing files. Commit
`953ac15` is titled for ODC-0003 but contains 267 lines of an in-flight ODC-0004
revision because of exactly that. Use targeted `git add <path>` instead. The
history is a record, and a commit message that misdescribes its contents is a
defect even when every line in it is legitimate.
