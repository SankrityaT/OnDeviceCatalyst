# Product requirements: memory reporting and model admission

Status: adopted 2026-09-01
Owner: SankrityaT
Feeds: ODC-0100, ODC-0102, ODC-0302

## Why this document exists

These are product requirements, stated without their derivation. They are
adopted because on-device inference is bounded by memory long before it is
bounded by speed, and because a runtime that reports its own consumption
inaccurately cannot be operated safely by an application developer.

Nothing here depends on unpublished work. Each requirement is testable, and each
is stated so that a contributor can implement it without further context.

## R1: Memory figures carry a declared basis

Any memory number the library reports, logs, or exposes through its API must
carry the basis it was measured on: the exact platform interface and field
consulted. A number without a basis is not comparable to any other number and
must not be presented as though it were.

Rationale limited to what is publishable: different runtimes and different
allocation paths account for memory differently, so a bare figure is ambiguous.

## R2: Cross-backend memory figures are only compared on a matching basis

The library must not present a memory comparison between two backends unless
both figures were obtained on the same basis, and it must record that the bases
matched. Where no common basis exists, figures are reported side by side with
their bases, never reduced to a single number.

This must be enforced mechanically rather than by convention. See ODC-0003,
which applies the same rule to the benchmark contract and derives the
comparability flag rather than accepting it as authored.

## R3: Model admission is decided from measured evidence

Whether a given model can be loaded on a given device must be decided from
measured evidence for that device class and that model, recorded in a
compatibility artifact, rather than inferred from a single platform-provided
number at runtime.

A platform availability signal may inform the decision. It may not be the sole
input, and the library must not present it to callers as though it were a
complete account of the process's memory position.

## R4: Admission failure is explicit, early, and actionable

When a model cannot be admitted, the library must fail before allocation, with
an error naming the model, the device class, the basis used, and the measured
figure that caused refusal. Silent degradation, partial load, and
load-then-crash are all prohibited outcomes.

## R5: Lifecycle survival is a tested property, not a claim

Memory-pressure response, backgrounding, foregrounding, cancellation mid-load,
and repeated load and unload cycles must each have a test that runs on a real
device surface before the behaviour may be described in documentation. A
compatibility promise that has never been executed is not a promise.

This requirement is currently blocked by ODC-0021, which owns establishing that
device surface.

## R6: Peak memory is reported with every performance figure

Any published throughput or latency figure must be accompanied by peak memory on
a declared basis, from the same run. Performance obtained by exceeding what a
device can sustain is not a result, and reporting it without its memory cost is
misleading.

## Non-goals

This document does not specify an allocator, a caching policy, or an eviction
strategy. It does not commit the project to any particular measurement tool. It
makes no claim about how other runtimes behave.

## Validation

| ID | Requirement | How it is checked |
| --- | --- | --- |
| R1 | Declared basis on every figure | API review plus ODC-0003 manifest schema |
| R2 | Matching-basis comparison only | Mechanical check in the benchmark checker |
| R3 | Evidence-based admission | Compatibility artifact exists and is consulted |
| R4 | Explicit early failure | Characterization and correctness tests |
| R5 | Tested lifecycle | Device-surface test results, blocked on ODC-0021 |
| R6 | Peak memory alongside performance | ODC-0003 acceptance criteria |
