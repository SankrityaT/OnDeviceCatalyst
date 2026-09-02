# OnDeviceCatalyst Tickets

This is the canonical public execution ledger. GitHub Issues are discussion and
assignment mirrors. A backlog ticket may have no spec yet, but every active
ticket must link to one.

| ID | Type | Title | Milestone | Status | Priority | Dependencies | Spec | GitHub Issue | Owner | Updated | Next Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ODC-0000 | process | Project operating system | P0 | DONE | P0 | none | [spec](docs/specs/ODC-0000-project-operating-system.md) | TBD | SankrityaT | 2026-08-25 | complete |
| ODC-0001 | process | Open-source foundation | P0 | DONE | P0 | ODC-0000 | [spec](docs/specs/ODC-0001-open-source-foundation.md) | TBD | SankrityaT | 2026-08-25 | complete |
| ODC-0002 | baseline | Reproduce v2 build and dependency state | P0 | DONE | P0 | ODC-0000 | [spec](docs/specs/ODC-0002-v2-baseline.md) | TBD | SankrityaT | 2026-09-01 | complete; unblocks ODC-0003/0004/0005 |
| ODC-0003 | benchmark | Cross-backend benchmark contract | P0 | APPROVED | P0 | ODC-0002 | [spec](docs/specs/ODC-0003-benchmark-contract.md) | TBD | SankrityaT | 2026-09-01 | implementation, blocked on ODC-0021 execution surface |
| ODC-0004 | test | V2 characterization suite | P0 | APPROVED | P0 | ODC-0002 | [spec](docs/specs/ODC-0004-v2-characterization-suite.md) | TBD | SankrityaT | 2026-09-01 | implementation; R3 tests blocked on ODC-0021 |
| ODC-0005 | design | Apple platform capability brief for v3 architecture | P0 | SPEC_DRAFT | P0 | ODC-0002 | [spec](docs/specs/ODC-0005-apple-platform-design-brief.md) | TBD | SankrityaT | 2026-09-01 | adversarial review pass |
| ODC-0010 | bug | Cached instance is shut down after caching (D1) | P0 | BACKLOG | P1 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0011 | bug | Generation emits duplicate terminal completions (D2) | P0 | BACKLOG | P1 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0012 | bug | Loading stream never terminates, impossible gate (D3) | P0 | BACKLOG | P1 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0013 | packaging | No macOS xcframework slice despite declared platform (D4) | P0 | BACKLOG | P1 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0014 | packaging | Metal shaders unpackaged, Metal Engine dead in package form (D5) | P0 | BACKLOG | P1 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery; refined by ODC-0004 Q2, xcodebuild does compile the shaders but makeDefaultLibrary reads Bundle.main not Bundle.module |
| ODC-0015 | bug | Fallback-path progress events are silent no-ops (D8) | P0 | BACKLOG | P1 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0016 | decision | Resolve the demo-app source fork (D7) | P0 | BACKLOG | P0 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0017 | decision | False simulator guard comment in Package.swift (E2) | P0 | BACKLOG | P0 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery; link-time half ANSWERED by ODC-0004, stub defines all 51 referenced symbols and links |
| ODC-0018 | bug | Declared test target does not compile on any triple (PredictionConfig.quality) | P0 | BACKLOG | P0 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0019 | decision | Disposition of three orphaned test files outside the target path | P0 | BACKLOG | P2 | ODC-0004 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0020 | decision | Revisit XCTest versus Swift Testing after the concurrency model lands | P0 | BACKLOG | P2 | ODC-0101 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0021 | infra | Establish a physical-device execution surface, including signing, provisioning and deployment | P0 | BACKLOG | P0 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery; absorbs ODC-0003 Q1 and ODC-0004 Q1/Q3 |
| ODC-0022 | bug | Determine whether the MLX backend initializes on any measured surface | P0 | BACKLOG | P1 | ODC-0002 | TBD | TBD | unassigned | 2026-09-01 | discovery |
| ODC-0100 | architecture | V3 vision and migration | P1 | BLOCKED | P0 | ODR-0005, ODC-0005 | TBD | TBD | unassigned | 2026-08-25 | research thesis |
| ODC-0101 | architecture | Swift 6 concurrency and lifecycle | P1 | BACKLOG | P0 | ODC-0100 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0102 | api | Public inference contracts | P1 | BACKLOG | P0 | ODC-0100, ODC-0101 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0103 | packaging | Modular package graph | P1 | BACKLOG | P0 | ODC-0100, ODC-0102 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0104 | api | Model identity and ownership | P1 | BACKLOG | P1 | ODC-0102 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0200 | backend | Maintained llama.cpp backend | P2 | BACKLOG | P0 | ODC-0103, ODC-0104 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0201 | backend | Current MLX backend | P2 | BACKLOG | P0 | ODC-0103, ODC-0104 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0202 | runtime | Sessions, streams, cancellation, and context | P2 | BACKLOG | P0 | ODC-0200, ODC-0201 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0203 | runtime | Structured generation and tools | P2 | BACKLOG | P1 | ODC-0202 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0204 | runtime | Single and batch embeddings | P2 | BACKLOG | P1 | ODC-0200, ODC-0201 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0205 | assets | Verified model asset lifecycle | P2 | BACKLOG | P1 | ODC-0104 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0206 | assets | Apple Background Assets integration | P2 | BACKLOG | P2 | ODC-0205, ODC-0005 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0207 | backend | IOS 26 Apple system-model backend | P2 | BACKLOG | P1 | ODC-0102, ODC-0005 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0300 | example | Maintained package-consuming sample app | P3 | BACKLOG | P1 | ODC-0202, ODC-0203, ODC-0204 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0301 | docs | DocC and v2-to-v3 migration | P3 | BACKLOG | P1 | ODC-0200, ODC-0201 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0302 | validation | Apple-platform lifecycle matrix | P3 | BACKLOG | P0 | ODC-0200, ODC-0201, ODC-0202 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0303 | validation | Device and model compatibility matrix | P3 | BACKLOG | P1 | ODC-0003, ODC-0302 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0304 | process | Release and support policy | P3 | BACKLOG | P1 | ODC-0301, ODC-0302 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0399 | release | V3 release review | P3 | BACKLOG | P0 | ODC-0300, ODC-0301, ODC-0302, ODC-0303, ODC-0304 | TBD | TBD | unassigned | 2026-08-25 | dependency approval |
| ODC-0800 | backend | Integrate released research backend | R1 | BLOCKED | P2 | ODR-0202, ODC-0399 | TBD | TBD | unassigned | 2026-08-25 | public research release |

## Status definitions

- `BACKLOG`: identified, but discovery has not started.
- `DISCOVERY`: gathering current facts, prior art, and repository evidence.
- `SPEC_DRAFT`: a spec exists but has not entered formal review.
- `SPEC_REVIEW`: review is active and founder approval is pending.
- `REVISION`: review found changes required before approval.
- `APPROVED`: decision-complete and authorized for implementation.
- `IMPLEMENTING`: code or documentation is being changed to satisfy the spec.
- `VALIDATING`: implementation is complete and acceptance evidence is running.
- `DONE`: accepted implementation and evidence are complete.
- `BLOCKED`: a named dependency prevents progress.
- `DEFERRED`: intentionally postponed.
- `REJECTED`: considered and declined.
