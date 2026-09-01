---
id: ODC-ADR-0001
title: Use an in-repository RFC-lite workflow
status: accepted
date: 2026-08-25
ticket: ODC-0000
---

# ODC-ADR-0001: Use an in-repository RFC-lite workflow

## Context

The project needs durable context across contributors, agents, and compaction,
but a small maintainer group cannot sustain the ceremony of a language-level RFC
process for every change.

## Decision

Use a tracked roadmap, canonical ticket ledger, per-ticket specifications, and
ADRs for lasting decisions. GitHub Issues mirror public discussion but do not
own status. Every active ticket requires a spec and founder approval before
implementation.

## Consequences

- Intent and evidence survive chat and issue churn.
- Contributors can understand why work is blocked or accepted.
- Small tickets still carry documentation overhead, mitigated by short templates.
- Ticket and spec consistency requires machine validation.
