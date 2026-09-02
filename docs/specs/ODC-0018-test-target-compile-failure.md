---
id: ODC-0018
title: Declared test target does not compile on any triple
type: bug
status: DONE
milestone: P0
owner: SankrityaT
dependencies: ODC-0002
founder_approved: delegated-to-manager-2026-09-01
last_updated: 2026-09-01
evidence_fresh_until: 2026-09-15
unresolved_questions: none
---

# ODC-0018: Declared test target does not compile on any triple

## Reproduction

Build the declared test target for any supported triple, for example:

```
swift build --build-tests --sdk "$(xcrun --sdk iphonesimulator --show-sdk-path)" \
  --triple arm64-apple-ios17.0-simulator
```

Before the repair this failed at
`Tests/OnDeviceCatalystTests/OnDeviceCatalystTests.swift:31:46`.

## Expected and actual behavior

Expected: the declared test target compiles, so its cases can run.

Actual: it referenced `PredictionConfig.quality`, which does not exist. The five
real presets are `balanced`, `creative`, `speed`, `deterministic` and `mirostat`.

The consequence is larger than a compile error. The ODC-0002 baseline recorded
four existing test cases as the project's entire test coverage. Because the
target never compiled, **those cases were never built or run by anything, so real
coverage was zero rather than four.** The baseline's count was accurate about the
file and misleading about the state.

## Root cause evidence

`Tests/OnDeviceCatalystTests/OnDeviceCatalystTests.swift:31:46` referenced a
preset that no version of `PredictionConfig` defines. Discovered during ODC-0004
discovery by building the target rather than reading it, which is why ODC-0002's
source-level pass did not catch it.

## In-scope repair

Remove the reference to the nonexistent preset. The four original cases were
reclassified rather than deleted:

- Two became `test_requires_*` regression cases.
- One became `test_requires_catalystShared_isASingletonAndIsConstructible`.
- `testPredictionConfigPresets` became a characterization case pinning what it
  got wrong twice: no `.quality` preset exists, and `speed.maxTokens` is
  numerically **greater** than `balanced.maxTokens`, the reverse of what the
  original assumed.

The identifier now survives only in comments explaining why it was wrong.

## Non-goals

No runtime source changes. No change to `PredictionConfig` itself. Whether the
preset values are sensible is a product question owned by later tickets, not a
bug fixed here.

## Regression risks

Low. The change is confined to a test file that did not compile, so nothing could
have depended on its behaviour.

## Tests

Covered by the ODC-0004 characterization suite, which cannot build without this
repair. See `docs/specs/ODC-0004-v2-characterization-suite.md`.

## Acceptance criteria

| ID | Criterion | Deciding command |
| --- | --- | --- |
| A1 | The nonexistent preset is referenced nowhere outside explanatory comments | `grep -rn 'PredictionConfig\.quality' Tests Sources \| grep -v '^\S*:[0-9]*: *//'` outputs nothing |
| A2 | The test target compiles for the simulator triple | `bash scripts/run-characterization.sh --surface simulator` exits 0 |
| A3 | No runtime source changed | `git diff --stat ebea213 -- Sources Package.swift Package.resolved` outputs nothing |
| A4 | Project state is consistent | `python3 scripts/validate-project-state.py` exits 0 |

## Review record

Found by ODC-0004 discovery on 2026-09-01 and recorded as a new defect beyond the
eight in the ODC-0002 baseline. Repaired as a prerequisite of ODC-0004's
implementation, because nothing in that suite could build otherwise.

This spec was written after the repair, when the project-state validator
correctly refused to accept a DONE ticket with no spec link. That ordering is a
process deviation and is recorded rather than concealed: the fix shipped first
and its specification followed.

Approval recorded as delegated to the acting manager, not as founder review.

## Validation evidence

All four acceptance criteria pass as of 2026-09-01. The characterization suite
built and ran 29 of 34 cases with zero failures on the iOS 26.5 simulator, which
is only possible because this repair landed.
