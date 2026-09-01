---
id: ODC-ADR-0004
title: Conform to Apple's model-provider API rather than compete with it
status: accepted
date: 2026-09-01
ticket: ODC-0002
---

# ODC-ADR-0004: Conform to Apple's model-provider API rather than compete with it

## Context

The program's fixed assumptions were written when iOS 26 was the newest shipping
system, and they explicitly excluded iOS 27 and any preview-only custom-provider
API from product requirements.

A dated landscape review on 2026-09-01 established two facts that expire that
assumption:

1. `LanguageModel` and `LanguageModelExecutor` - the protocol allowing a
   third-party model to serve a `LanguageModelSession` - are confirmed
   `beta: true, introducedAt: 27.0` on every platform, verified against Apple's
   DocC JSON rather than summarized documentation. iOS 27 general availability is
   expected in approximately two weeks.
2. `mlx-swift-lm` already ships `MLXFoundationModels`, documented as a bridge from
   MLX models into `FoundationModels.LanguageModel` and requiring the 27.0 SDK.

Taken together, Apple plus Apple's own MLX team are about to occupy the
"many backends behind one Swift API" position. That position was the most
commonly assumed reason for this project to exist.

Separately, every capability Apple announced requires iOS 26 at minimum and most
requires 27. Apple has shipped nothing for iOS 17 through 25.

## Decision

1. **Keep the core package's deployment target at iOS 17 / macOS 14.** The
   pre-26 installed base is entirely unaddressed by Apple, and that gap widens
   with each release rather than closing. It is the one durable compatibility
   position available.
2. **Stop treating a unified multi-backend API as the differentiator.** Do not
   invest in a bespoke abstraction whose purpose is to be the thing Apple's
   protocol will be. That contest is lost on a schedule measured in weeks.
3. **Adopt Apple's provider protocol as an optional adapter once it reaches a
   stable SDK, rather than competing with it.** On systems that have it,
   Catalyst backends should be reachable through Apple's session API. On systems
   that do not, Catalyst keeps its own iOS 17-compatible surface. The adapter is
   additive and optional; it must not become a dependency of the core package.
4. **Move the differentiator to the execution-policy layer.** The scarce, unowned
   capability is not "which API do you call" but how a runtime behaves over a
   real session on real hardware: device-aware backend selection, memory
   budgeting and eviction, sustained-load behavior, thermal and backgrounding
   transitions, cancellation, and honest performance reporting. No API surface
   Apple has announced supplies any of this, and it is already in the public
   roadmap under P3.
5. **Do not adopt any 27.0 API as a product requirement until it ships in a
   stable SDK.** Preview APIs may inform planning; they may not define acceptance
   criteria. This preserves the original rule while allowing the plan to react
   when 27 ships.

## Consequences

- The public v3 architecture ticket must be rewritten around execution policy
  rather than around backend abstraction as an end in itself.
- The optional Apple-system-model backend becomes a conformance target instead of
  a competitor, which reduces long-term maintenance rather than increasing it.
- A second landscape refresh is required immediately after iOS 27 general
  availability, because this decision is built on a beta API's shipping behavior.
- The iOS 17–25 compatibility promise becomes a load-bearing product commitment
  and must be tested, not merely declared.
- The program's fixed assumption that "iOS 26 is the latest shipping baseline"
  is superseded by this ADR as of iOS 27 general availability.

## Status note

Adopted by the acting program manager under delegated authority, on dated
evidence recorded in the private landscape review of 2026-09-01. It should be
revisited at the first landscape refresh after iOS 27 ships, when the beta
protocol's final shape is known.
