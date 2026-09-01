---
id: ODC-0001
title: Open-source foundation
type: process
status: DONE
milestone: P0
owner: SankrityaT
dependencies: ODC-0000
founder_approved: 2026-08-25
last_updated: 2026-08-25
evidence_fresh_until: not-applicable
unresolved_questions: none
---

# ODC-0001: Open-source foundation

## Summary

Establish a genuinely free public project with explicit licensing, contribution
provenance, governance, security reporting, structured community workflows, and
repository-owned enforcement.

## Goals

- License the public work under Apache-2.0 with an explicit patent grant.
- Require DCO sign-off without a copyright-assignment CLA.
- Give contributors clear scope, build, review, privacy, and evidence rules.
- Provide issue and pull-request templates that reference tickets and specs.
- Add public CI for project-state validation, DCO, and the current iOS build.

## Non-goals

- Change runtime behavior.
- Install a third-party DCO application.
- Enable or mutate GitHub repository settings outside tracked files.
- Offer a formal support or vulnerability-response SLA before v3.

## Design

- `LICENSE` contains the unmodified Apache License 2.0 text.
- `NOTICE` identifies OnDeviceCatalyst and keeps third-party terms separate.
- `CONTRIBUTING.md` requires `git commit -s` and spec-driven pull requests.
- Repository-owned `scripts/check-dco.sh` validates pull-request commits.
- `GOVERNANCE.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` define public
  expectations without exposing confidential channels or research.
- GitHub forms collect reproducible environment, model, backend, and safety
  information without demanding private data.

## Acceptance criteria

- GitHub can identify an Apache-2.0 license after the files reach the default
  branch.
- DCO checker passes matching sign-offs and rejects missing or mismatched ones.
- Pull-request template requires an approved spec and evidence.
- Bug form requests revision, device, OS, backend, model, reproduction, and
  redacted logs.
- Feature form begins with the developer problem and maintenance impact.
- CI has read-only repository permission and no stored credentials after use.
- No runtime source is changed.

## Review record

- 2026-08-25, completeness review: licensing, contribution provenance,
  governance, security, community templates, and CI are included.
- 2026-08-25, adversarial review: chose repository-owned DCO enforcement to
  avoid requiring an external GitHub application or CLA.
- 2026-08-25, founder approval: Apache-2.0 and DCO approved in the program plan.

## Validation evidence

- `scripts/test-check-dco.sh` passed signed, unsigned, and mismatched-signoff
  fixtures.
- All GitHub workflow and issue-form YAML files parsed successfully.
- Repository-owned CI uses read-only contents permission and disables checkout
  credential persistence.
- The exact Xcode 26 build command completed for
  `arm64-apple-ios17.0-simulator` using the installed iOS 26.5 SDK.
- Existing v2 warnings were preserved for ODC-0002 rather than repaired here.
