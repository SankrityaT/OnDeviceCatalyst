# Contributing to OnDeviceCatalyst

Thank you for helping build a trustworthy, high-performance on-device inference
package for Apple platforms.

## Start with the project state

Before proposing or implementing work, read:

1. [ROADMAP.md](ROADMAP.md).
2. [Tickets.md](Tickets.md).
3. [AGENTS.md](AGENTS.md).
4. The active ticket specification and referenced ADRs.

Large design work starts as a feature issue and moves through discovery, spec
drafting, two review passes, and founder approval. Implementation pull requests
must reference an approved specification.

Good early contributions include documentation corrections, reproducible v2
build reports, characterization tests, benchmark adapters, and device evidence.
New runtime architecture and optimization code remains gated until the private
research thesis has been selected and sanitized requirements are public.

## Source of truth

`Sources/OnDeviceCatalyst` is the maintained package target. The top-level
`OnDeviceCatalyst` directory is a legacy demonstration app with older source
copies. Do not fix the package only in the legacy tree.

## Developer Certificate of Origin

Every public commit must certify the Developer Certificate of Origin by adding a
sign-off matching the commit author:

```text
Signed-off-by: Your Name <your-email@example.com>
```

Git can add it automatically:

```sh
git commit -s
```

The repository-owned DCO check rejects unsigned pull-request commits. A DCO
sign-off states that you have the right to submit the contribution under the
project's Apache-2.0 license. It is not a copyright assignment.

## Current development setup

Use Xcode 26. Resolve dependencies and build the current package for an iOS
simulator:

```sh
swift package resolve
CATALYST_SDK_PATH="$(xcrun --sdk iphonesimulator --show-sdk-path)"
swift build \
  --triple arm64-apple-ios17.0-simulator \
  --sdk "$CATALYST_SDK_PATH"
```

The current v2.0.4 llama artifact has no macOS slice. ODC-0002 owns the complete
baseline, so a macOS `swift test` failure is not automatically a contributor
setup error.

## Pull requests

A pull request must:

- Reference its ticket, GitHub Issue, and approved spec.
- Explain the user-visible or research-tooling problem.
- Stay inside the approved scope.
- Add tests where behavior can be exercised without model weights.
- Record physical-device evidence when automation is insufficient.
- Avoid unrelated formatting and generated-file churn.
- Update the spec evidence, `Tickets.md`, and `ROADMAP.md` when it changes gate.
- Disclose material AI-assisted code generation and any non-original sources.

Never commit model weights, credentials, private URLs, access tokens, downloaded
binary artifacts, device serials, UDIDs, or confidential research material.

Model-dependent reports must include the model repository and revision,
filename, checksum when available, quantization, backend, settings, device, OS,
toolchain, repetitions, and raw result location.

## Review and completion

Review is evidence-driven, not a vote. The founder makes the final accept,
revise, or reject decision while the maintainer community is small.

Before marking a ticket done:

```sh
python3 scripts/validate-project-state.py
```

Run every build, test, and benchmark required by the approved spec, record the
evidence, and rerun the affected suite after fixes.

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report
security-sensitive problems through [SECURITY.md](SECURITY.md), never a public
issue.
