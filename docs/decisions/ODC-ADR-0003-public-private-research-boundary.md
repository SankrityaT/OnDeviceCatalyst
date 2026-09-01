---
id: ODC-ADR-0003
title: Separate public product work from private novel research
status: accepted
date: 2026-08-25
ticket: ODC-0000
---

# ODC-ADR-0003: Separate public product work from private novel research

## Context

The public package needs open contribution while the founder wants novel engine
research, raw results, and a possible paper to remain confidential until the
result is reliable.

## Decision

Keep product code, benchmark methodology, baselines, and released artifacts
public. Keep novel hypotheses, experimental kernels, raw results, negative
results, provenance, and manuscripts in a separate private repository until a
founder-approved validation and release audit is complete.

## Consequences

- Public contributors have a clear, honest product scope.
- Confidential research cannot be stored in public tickets or specs.
- Cross-track status must be sanitized and synchronized deliberately.
- Released research requires a provenance and licensing audit before Apache-2.0
  publication.
