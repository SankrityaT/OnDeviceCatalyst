# OnDeviceCatalyst Roadmap

Last updated: 2026-08-25

## Current state

- Current milestone: P0, Project operating system.
- Current public ticket: ODC-0002, Reproduce the v2 baseline.
- Latest completed ticket: ODC-0001, Open-source foundation.
- Latest founder-approved program plan: 2026-08-25 disruption program.
- Latest public landscape check: not completed.
- Research dependency: private ODR-0001 is in discovery.
- Runtime implementation gate: closed until private ODR-0005 selects a
  disruption thesis and publishes sanitized product requirements.
- Next gate: founder review of `docs/specs/ODC-0002-v2-baseline.md`.

## Mission

OnDeviceCatalyst will be a free, Apple-first inference package that is reliable
enough for real applications and rigorous enough to support reproducible systems
research.

The product and research tracks have different disclosure boundaries:

- The public repository owns product APIs, maintained backends, correctness
  contracts, benchmark methodology, compatibility evidence, and released
  research artifacts.
- A private research repository owns confidential hypotheses, experimental
  kernels, raw and negative results, provenance, and manuscripts until a result
  passes its release gates.

## Fixed product boundary

- Latest shipping baseline: Xcode 26, iOS and iPadOS 26, and macOS 26.
- Deployment compatibility target: iOS and iPadOS 17-26, macOS 14-26.
- First v3 workloads: LLM generation, streaming, structured and tool output,
  single embeddings, and batch embeddings.
- Required stable v3 backends: llama.cpp and MLX.
- Optional stable-system integration: Apple `SystemLanguageModel` on eligible
  iOS, iPadOS, and macOS 26 devices.
- Preview-only platform APIs may be studied privately but cannot define product
  acceptance criteria.
- The public project is Apache-2.0 and uses DCO sign-off without a CLA.

## Operating gates

1. Every active ticket has a tracked specification.
2. Founder approval is required before implementation.
3. Material implementation changes return the spec to revision.
4. External model, runtime, hardware, performance, and SDK evidence expires
   after 14 days.
5. Correctness is established before performance is measured.
6. No universal performance claim may be inferred from a model-specific or
   device-specific result.
7. Done requires validation evidence and an updated project snapshot.

## P0: Project operating system

Goal: make the existing repository understandable, reproducible, and safe for
spec-driven contribution without changing runtime behavior.

Tickets:

- ODC-0000, planning, review, ADR, and compaction-continuity system.
- ODC-0001, Apache-2.0, DCO, governance, security, and community workflow.
- ODC-0002, reproducible v2 build and dependency baseline.
- ODC-0003, public cross-backend benchmark contract and result manifest.
- ODC-0004, v2 characterization tests without behavior changes.
- ODC-0005, WWDC25 Apple-native design brief.

Exit criteria:

- A fresh contributor can recover current state from tracked files.
- Public CI validates the project system and builds the current iOS package.
- The exact v2 failures, warnings, dependencies, and supported build paths are
  recorded without marketing overstatement.
- Public benchmark tooling can record pinned, reproducible baseline runs.
- The Apple-native design brief covers Foundation Models, MLX, Metal 4,
  Background Assets, availability, safety, and OS model updates.

Before the private disruption thesis is approved, code changes are restricted
to project infrastructure, correctness tests, benchmark tooling, and baseline
reproduction.

## R0: Private disruption thesis

Public status projection only. Confidential detail remains private.

Goals:

- Maintain a current model, architecture, runtime, hardware, and research radar.
- Reproduce relevant baselines on available reference hardware.
- Profile actual bottlenecks instead of assuming the existing Metal engine is
  the correct starting architecture.
- Select and preregister a novel, falsifiable research thesis.

Exit criteria:

- Landscape evidence is no older than 14 days.
- Relevant prior art and runnable competitors are documented.
- Baselines include failures and negative results.
- The chosen thesis has correctness, performance, abort, and claim boundaries.
- Sanitized requirements are available to the public v3 architecture ticket.

## P1: v3 architecture

Blocked by the R0 thesis gate.

Tickets:

- ODC-0100, v3 vision, compatibility, package boundaries, and migration.
- ODC-0101, Swift 6 concurrency and lifecycle ownership.
- ODC-0102, public request, response, event, capability, and error contracts.
- ODC-0103, modular core, llama.cpp, MLX, Apple system-model, and umbrella
  products.
- ODC-0104, model identity, source, format, discovery, selection, and ownership.

Exit criteria:

- The public API compiles back to the declared deployment targets.
- Heavy backends are optional and do not burden core-only consumers.
- Lifecycle and stream terminal behavior are explicit and testable.
- V2 migration and compatibility decisions are recorded before implementation.

## P2: Product-grade inference

Tickets:

- ODC-0200, maintained llama.cpp backend.
- ODC-0201, current MLX backend.
- ODC-0202, sessions, canonical templates, streaming, cancellation, and context.
- ODC-0203, structured generation and tool calling.
- ODC-0204, normalized single and batch embeddings.
- ODC-0205, verified model asset lifecycle.
- ODC-0206, optional Apple Background Assets delivery.
- ODC-0207, optional iOS 26 Apple system-model backend.

Exit criteria:

- Llama.cpp and MLX satisfy one documented product behavior contract.
- Stream cancellation and exactly-one-terminal-event rules pass stress tests.
- Model lifecycle survives repeated load, generation, cancellation, and unload.
- Apple system-model integration handles every documented availability state.

## P3: Product readiness

Tickets:

- ODC-0300, maintained sample app consuming packages rather than copied source.
- ODC-0301, DocC and v2-to-v3 migration guide.
- ODC-0302, lifecycle, backgrounding, memory-pressure, and device validation.
- ODC-0303, reproducible device and model compatibility matrix.
- ODC-0304, release, support, security, and deprecation policy.
- ODC-0399, v3 release review.

Exit criteria:

- Supported public products build independently in CI.
- Automated tests require no model download.
- Optional model tests skip with explicit reasons.
- Physical-device evidence includes exact hardware, OS, model revision, backend,
  and configuration.
- Documentation and sample code match the shipped API.

V3 does not wait for the private research engine.

## R1: Validated research release

Detailed experimental tickets are intentionally not public before the thesis is
validated.

Public integration sequence:

- ODR-0200, independent reproduction artifact.
- ODR-0201, provenance, license, attribution, and confidential-data audit.
- ODR-0202, approved Apache-2.0 research release.
- ODC-0800, optional product integration of the released backend.
- ODR-0300, manuscript and public artifact package.

Required evidence includes correctness, pinned baselines, equivalent workloads,
raw repetitions, negative results, memory, prefill, decode, time to first token,
thermal behavior, energy, independent repetition, and a 14-day prior-art refresh.

## Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Ecosystem changes faster than implementation | Obsolete architecture or claim | Fourteen-day freshness gate and dated source register |
| Research contaminates public product scope | Unmaintainable framework | Separate private repository and explicit graduation gate |
| Backend dependencies inflate every consumer | Slow builds and large binaries | Optional package products and lightweight core |
| Existing v2 behavior is changed before capture | Lost regression reference | ODC-0002 and ODC-0004 before runtime fixes |
| Performance work outruns correctness | Invalid results | Correctness gate before benchmark acceptance |
| Public contributors expose confidential research | Loss of novelty | Sanitized cross-track status only |
| Product claims exceed tested evidence | Loss of trust | Exact device, model, context, and metric in every claim |

## Success evidence

Technical success is measured through reproducible correctness, performance,
memory, energy, and lifecycle data. Product success is measured through
independent integrations, external contributors, releases, issue resolution,
and community reproduction of published results. Research success is measured
through validated novelty, reusable artifacts, citations, and publication-ready
methodology, not through private benchmark screenshots.

## Updating this roadmap

Update the Current state section whenever an active ticket changes gate. Change
fixed scope or milestone exit criteria only through an approved spec and ADR.
Keep confidential hypotheses and raw results out of this file.
