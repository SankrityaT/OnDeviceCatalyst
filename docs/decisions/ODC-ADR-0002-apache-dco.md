---
id: ODC-ADR-0002
title: License public work under Apache-2.0 with DCO
status: accepted
date: 2026-08-25
ticket: ODC-0001
---

# ODC-ADR-0002: License public work under Apache-2.0 with DCO

## Context

Inference runtimes and kernels may involve patent-bearing contributions. The
project must remain freely usable by individuals and companies while keeping
contribution provenance clear and participation accessible.

## Decision

License public OnDeviceCatalyst work under Apache-2.0. Require matching DCO
sign-offs on public commits. Do not require a CLA or copyright assignment.

## Consequences

- Users receive permissive rights and an explicit patent grant.
- Contributors retain copyright in their work.
- Pull requests with unsigned commits fail repository-owned validation.
- Relicensing would require respecting all contributor rights.
