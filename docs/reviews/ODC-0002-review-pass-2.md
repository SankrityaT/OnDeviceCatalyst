---
review_of: ODC-0002
spec: docs/specs/ODC-0002-v2-baseline.md
pass: 2 (adversarial)
date: 2026-09-01
reviewer: adversarial spec review
repo_revision_reviewed: 59da80b
verdict: REJECT
---

# ODC-0002 review pass two (adversarial)

## Verdict

**REJECT** - return to `REVISION`.

The spec is well-written process prose, but as an *evidence contract* it is not
yet sound. Three independent classes of problem justify rejection rather than
"accept with revisions":

1. The one build path the spec designates as the success case
   (`arm64-apple-ios17.0-simulator`) links a **7,936-byte stub** rather than
   llama.cpp, and `Package.swift` contains a false claim used to justify that
   stub. A baseline that records "iOS simulator: green" without that annotation
   is actively misleading and would be cited by ODC-0003, ODC-0004 and ODC-0005.
2. The spec's own `Deliverables` and `Validation` sections are mutually
   unsatisfiable on the `Package.resolved` question, which is exactly the
   "unresolved implementation decision" that program rule 6 forbids in an
   APPROVED spec.
3. The largest single fact about the current v2 tree - that
   `OnDeviceCatalyst/` (the Xcode app target) is a **divergent fork** of
   `Sources/OnDeviceCatalyst/`, with 13 of 21 shared files differing and
   `LlamaInstance.swift` differing by 587 lines - is not mentioned anywhere in
   the spec, and the spec's procedure would not discover it.

Everything below was verified against the working tree at HEAD (`59da80b`) with
read-only commands. No file under `Sources/` was modified; all builds ran in
`/tmp/odc-fresh` and `/tmp/odc-review` scratch copies.

### Environment used for verification

| Item | Value |
| --- | --- |
| Xcode | 26.6 (17F113) |
| Swift | 6.3.3 (swiftlang-6.3.3.1.3), swift-driver 1.148.6 |
| Host target | `arm64-apple-macosx26.0` |
| iOS SDK | 26.5 (`iphoneos26.5`) |
| iOS Simulator SDK | 26.5 (`iphonesimulator26.5`) |
| macOS SDK | 26.5 (`macosx26.5`) |

---

## Findings

### 1. BLOCKING - The designated "successful build path" links a stub, and `Package.swift` justifies it with a false claim

**Location:** `docs/specs/ODC-0002-v2-baseline.md:30` (goal), `:62` (procedure
step 8), `:94` (acceptance: "Current successful and failed build paths are
explicit"). Root cause at `Package.swift:33-42`.

**Problem.** `Package.swift:36-37` states:

> `// Includes an arm64-simulator stub slice so consumers can build for the iOS`
> `// Simulator (all llama usage is guarded #if !targetEnvironment(simulator)).`

Both halves of that parenthetical are wrong, and the spec inherits the error.

Verified:

- The XCFramework has exactly two slices and no macOS slice:
  `ios-arm64` (`libllama_combined.a`, 17,151,376 bytes, 16,870 exported
  symbols) and `ios-arm64-simulator` (`libllama_combined.a`, **7,936 bytes, 55
  exported symbols**, objects `llama_stub.o` / `llama_sim_stubs.o`).
- `grep -rn '#if !targetEnvironment' Sources/` returns **0 matches**. There are
  only four `#if targetEnvironment(simulator)` occurrences in the entire
  package (`Core Foundation/SimulatorSupport.swift:27`,
  `SafetyManager.swift:76`, `ModelArchitecture.swift:373`,
  `DeviceOptimizer.swift:20`), none of them negated and none of them guarding a
  llama call.
- `Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:12` and
  `Sources/OnDeviceCatalyst/Backend/LlamaCppBackend.swift:9` both
  `import llama` unconditionally, and `LlamaBridge` calls the C API
  unconditionally (e.g. `LlamaBridge.swift:56`, `:140`, `:158`).

Consequence: `swift build --sdk "$(xcrun --sdk iphonesimulator
--show-sdk-path)" --triple arm64-apple-ios17.0-simulator` **succeeds** (verified,
`Build complete! (31.89s)`), but the resulting configuration cannot perform
inference - it resolves llama symbols to stubs at link time and calls them at
run time. The spec would record this as the baseline's success case with no
caveat.

**Suggested fix.** Amend goal `:30` and step 8 to require that the simulator
result is recorded as **`builds = yes, functional = no (stub-linked)`**, and add
to the required procedure:

- record each XCFramework slice's `LibraryIdentifier`, `SupportedPlatform`,
  `SupportedPlatformVariant`, byte size, and exported-symbol count
  (`nm -gU`), so the stub is visible in the evidence rather than inferred;
- add a device-triple compile
  (`--sdk "$(xcrun --sdk iphoneos --show-sdk-path)" --triple arm64-apple-ios17.0`;
  verified to succeed at HEAD) so the matrix contains at least one cell built
  against the real header set;
- state explicitly that neither SwiftPM invocation *links* the static library,
  so link-time symbol resolution against either slice is **not** covered by this
  baseline, and name the ticket that will cover it;
- record `Package.swift:36-37` as a documented-but-false comment and map it to a
  follow-up ticket.

---

### 2. BLOCKING - The `Package.resolved` deliverable is self-contradictory and its consumer impact is unevaluated

**Location:** `docs/specs/ODC-0002-v2-baseline.md:49-51` (deliverable),
`:59-60` (procedure step 6), `:85` (validation), `:97` (acceptance),
`:104-105` (alternatives).

**Problem A - the two offered options are not both valid.** Deliverable 3 says
"preserve HEAD **or** commit the resolver's deterministic manifest-compatible
output". Validation `:85` says "A second `swift package resolve` leaves the
selected lockfile unchanged." These cannot both hold for the "preserve HEAD"
branch: HEAD's lockfile is manifest-incompatible, so any resolve rewrites it.
Verified in a clean scratch copy of `Package.swift` + `Package.resolved` +
`Sources` + `Tests`:

```
$ swift package resolve      # first run
8f655df…  ->  3a4f3634…      # Package.resolved rewritten
$ swift package resolve      # second run
3a4f3634…                    # stable thereafter
```

The diff HEAD → resolver output:

| Field | HEAD | Resolver |
| --- | --- | --- |
| `mlx-swift` | `0.30.6` / `6ba4827f…` | `0.29.1` / `072b684a…` |
| `mlx-swift-lm` | `branch: main` / `6bb84aac…` | `2.29.3` / `5064b8c5…` |
| lockfile `version` | `2` | `3` |
| `originHash` | absent | `49eba206…` |

The `mlx-swift-lm` pin at HEAD is a **branch** pin, which cannot satisfy
`Package.swift:30`'s `exact: "2.29.3"`. "Preserve HEAD" is therefore not a real
option; the spec presents a decision that has only one answer while formally
leaving it open. That is a program-rule-6 violation (`ODC-0000` acceptance:
"unresolved approved specs" are a validator failure class, and rule 6 of the
program plan requires zero unresolved implementation decisions).

**Problem B - the format bump is an unevaluated consumer-facing change.**
Committing the resolver output moves the tracked lockfile from format `version:
2` to `version: 3` and introduces `originHash`. The spec's Non-goals `:38-42`
protect `Package.swift`, runtime source, tests, the Xcode project and the binary
artifact, but deliberately leave `Package.resolved` writable - yet acceptance
`:97` asserts "Runtime behavior and public API are unchanged" without
distinguishing "the built graph is unchanged" from "the tracked file is
unchanged". (Note: the *effective* graph is already `mlx-swift 0.29.1` /
`mlx-swift-lm 2.29.3` - `.build/workspace-state.json` at HEAD confirms it - so
the pin correction is truthful bookkeeping. The format bump is the real change.)

**Suggested fix.**

- Replace the "preserve HEAD or commit resolver output" fork with a single
  decided outcome, and record *in the spec* the evidence that HEAD's branch pin
  cannot satisfy the `exact:` requirement.
- Add a required sub-step: record which SwiftPM / Xcode versions can read
  `Package.resolved` format 3, and state whether that raises the project's
  minimum consumer tooling. If it does, that is a compatibility change and needs
  its own decision line, not a lockfile refresh.
- Split acceptance `:97` into two mechanically distinct claims: "no file under
  `Sources/`, `Tests/`, `Package.swift`, or `OnDeviceCatalyst.xcodeproj/`
  differs from `59da80b`" (checkable with `git diff --stat`) and "the resolved
  dependency graph is byte-identical before and after" (checkable by diffing
  `.build/workspace-state.json`).

---

### 3. BLOCKING - The baseline omits the single largest fact about the v2 tree: a divergent duplicate of the runtime

**Location:** `docs/specs/ODC-0002-v2-baseline.md:32` (goal), `:64` (step 10),
`:45-48` (deliverables). Nothing in the spec covers this.

**Problem.** `OnDeviceCatalyst.xcodeproj/project.pbxproj` contains **zero**
`XCRemoteSwiftPackageReference` entries. The app target does not consume the
package at all - it compiles its own copy of the runtime from
`OnDeviceCatalyst/`, and that copy has drifted. Verified per-file diff of the 21
shared Swift files:

| File | Changed lines (`diff` `<`/`>`) |
| --- | --- |
| `Core Engine/LlamaInstance.swift` | **587** |
| `Service Layer/Catalyst.swift` | **218** |
| `API Bridge/LlamaBridge.swift` | 104 |
| `Core Foundation/InstanceSettings.swift` | 47 |
| `Chat System/StreamResponse.swift` | 37 |
| `Core Engine/SamplingEngine.swift` | 35 |
| `Core Foundation/DeviceOptimizer.swift`, `Core Engine/PromptFormatting.swift` | 18 each |
| `Core Foundation/ModelProfile.swift` | 16 |
| `Core Foundation/ModelArchitecture.swift` | 11 |
| `Chat System/StopSequenceHandler.swift` | 4 |
| `Core Foundation/PredictionConfig.swift` | 2 |
| 8 remaining files | identical |

The divergence is architectural, not cosmetic: the app copy of
`LlamaInstance.swift` still holds raw `cModel` / `cContext` / `cBatch`
`OpaquePointer` state, while the package copy uses the
`InferenceBackend` abstraction (`Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:25-27`
vs `OnDeviceCatalyst/Core Engine/LlamaInstance.swift:26-29`).

This matters far beyond ODC-0002. ODC-0004 ("V2 characterization tests without
changing existing runtime behavior") cannot start until the project has decided
*which* v2 is being characterized. A baseline that does not name the two copies
hands ODC-0004 an ambiguous target.

**Suggested fix.** Add a goal and deliverable section "Source duplication and
target ownership":

- record that `OnDeviceCatalyst.xcodeproj` has no SPM package reference and
  builds `OnDeviceCatalyst/` directly;
- emit the per-file identical/differs/app-only table above into
  `docs/baselines/v2.0.4.md` (generated, not hand-maintained);
- record the divergence in the JSON manifest as a stable
  `duplicate_sources` array with per-file `sha256` for both copies, so ODC-0004
  can key off it;
- map it to a follow-up ticket (ODC-0300 is the natural home but is P3 and
  currently `BACKLOG`; a P0 decision ticket is likely needed first).

---

### 4. BLOCKING - Confounders are not pinned; two honest runs of this procedure will disagree

**Location:** `docs/specs/ODC-0002-v2-baseline.md:26-27` (goal), `:54-57`
(steps 1-4), `:69-70`, `:83-88` (validation).

**Problem.** The spec pins "Xcode 26" as a *major* version and nothing else.
Two concrete, demonstrated sources of disagreement:

**(a) Warm vs. cold `.build` changes whether step 5 mutates the lockfile.**
In a scratch copy that included the repo's existing `.build/`,
`swift package describe --type json` completed (`exit=0`) and left
`Package.resolved` byte-identical (`8f655df4…` before and after). In a scratch
copy with no `.build/`, `swift package resolve` rewrote it. Step 5 ("Run `swift
package describe` before and after resolution") therefore produces different
evidence depending on undocumented machine state - and "before resolution" is
not reachable at all from a truly clean checkout, because `describe` resolves
implicitly.

**(b) The Xcode-project failure is caused by an untracked, gitignored file.**
The verified failure is:

```
error: There is no XCFramework found at
'…/richmond/OnDeviceCatalyst/llama.xcframework'.
(in target 'OnDeviceCatalyst' from project 'OnDeviceCatalyst')
** BUILD FAILED **   (exit 65)
```

`.gitignore:19-22` ignores `llama.xcframework/` and `*.xcframework/`. A
developer who has ever unzipped the release asset into `OnDeviceCatalyst/` gets
a *different* result from a fresh clone, and neither `git status` nor the spec's
step 1 ("preserve unrelated work") would reveal it. The spec's failure taxonomy
`:74-79` has no bucket for "artifact absent locally but not a network failure",
so this failure is currently unclassifiable under the spec's own rules.

**(c) The scratch/ignored-directory claim is false on a fresh clone.**
`:69-70` states "Build products and fetched dependencies remain in ignored
directories." The 2.3 GB `.context/` tree (which already holds three prior SPM
scratch trees: `bootstrap-build/`, `open-source-build/`, `spm-sim/`) is ignored
only via a **machine-local** `.git/info/exclude` entry
(`git check-ignore -v .context` →
`/Users/…/ondevicecatalyst/.git/info/exclude:7:.context/`). The tracked
`.gitignore` has no `.context/` entry.

**Suggested fix.** Add a "Pinned environment and clean state" section that
requires, before any command runs:

- exact `xcodebuild -version` build number (26.6 / 17F113 today), exact
  `swift --version` string, exact SDK build versions from `xcodebuild -showsdks`,
  and `xcode-select -p`;
- a machine-state precondition block: `git status --porcelain --ignored` and
  `git clean -ndx` output captured *into the report*, so untracked artifacts are
  visible; explicit removal of `.build/` and `DerivedData/`; and a declared
  SwiftPM cache policy (reuse `~/Library/Caches/org.swift.swiftpm` vs.
  `--cache-path` to a fresh directory - the two give different network exposure
  and different failure modes for moved tags);
- simulator/device identity: the spec's step 10 says only "a generic iOS
  simulator" and step 4 records a "connected device model" that no procedure
  step ever uses. Either name a simulator runtime version or state that no
  simulator runtime is instantiated (true today - `build` needs none) and drop
  the device goal;
- an explicit scratch directory that is ignored by the **tracked** `.gitignore`.

---

### 5. BLOCKING - Acceptance criteria are not mechanically checkable

**Location:** `docs/specs/ODC-0002-v2-baseline.md:92-97` and `:83-88`.

**Problem.** Four criteria a reasonable engineer could declare "done" with no
evidence:

- `:92` "Both deliverables exist and agree." *Agree* is undefined. There is no
  named field correspondence between the Markdown report and the JSON manifest.
- `:93` "No secrets or stable personal device identifiers appear in either
  deliverable." No denylist, no scanner, no command.
- `:84` "Every report command is copyable and uses no personal absolute path."
  Not machine-checkable as written.
- `:97` "Runtime behavior and public API are unchanged." There is no API
  baseline to compare against, `swift test` cannot run at all (see finding 10),
  and the package does not build on the host platform - so this criterion has no
  possible evidence.

`:88` offers `python3 scripts/validate-project-state.py` as the mechanical
backstop, but that script does not check any of this. Verified at
`scripts/validate-project-state.py:148-168`: it checks only front-matter *key
presence*, ticket/spec status agreement, founder-approval state, and (for
approved states) `unresolved_questions == "none"`. It never inspects spec
section structure, never reads `docs/baselines/`, and never evaluates
`evidence_fresh_until` - it merely requires the key to exist. It currently
reports `project state valid: 26 tickets, 3 specs, 3 ADRs` for a spec with all
of the defects in this review.

**Suggested fix.** Replace each criterion with a command:

- "agree" → name the exact JSON keys that must equal values quoted in the report
  (e.g. `toolchain.xcode_build`, `sdks.iphonesimulator`, `resolved[].revision`,
  `artifact.slices[].symbol_count`) and add a check script that asserts it;
- redaction → a concrete denylist regex run over both deliverables (`$HOME`,
  `/Users/[^/ ]+`, `[0-9A-F]{8}-[0-9A-F]{16}` UDID shape, `ghp_`, `github_pat_`)
  with the command recorded;
- "copyable" → assert no `/Users/` substring in any fenced command block;
- "unchanged" → `git diff --stat 59da80b -- Sources Tests Package.swift
  OnDeviceCatalyst.xcodeproj` is empty. Drop the unprovable "runtime behavior"
  half or scope it to "no source under `Sources/` was modified".

---

### 6. MAJOR - Spec is missing sections both templates require, and `type: baseline` has no template

**Location:** front matter `:5` (`type: baseline`); compare
`docs/templates/process-spec.md` and `docs/templates/product-spec.md`.

**Problem.** `docs/templates/` contains `product-spec.md`, `benchmark-spec.md`,
`bug-spec.md`, `process-spec.md`, `release-spec.md` - there is **no**
`baseline-spec.md`, so `type: baseline` has no defined shape. Measured against
the nearest template (`process-spec.md`), ODC-0002 omits:

- **`## Current state and evidence`** - required by both templates, and the
  single most important section for a *baseline* ticket. Its absence is why
  findings 1, 3 and 10 are absent from the spec.
- **`## Design`** - required by `process-spec.md`. ODC-0002 substitutes
  `## Required procedure`, which is a runbook, not a design; there is no
  statement of what the manifest models or why.
- Any security/privacy section, despite the spec being unusually
  redaction-heavy (`:27`, `:41`, `:57`, `:93`).

Program rule 2 requires draft specs to define "goals, non-goals, design,
interfaces, data flow, failure behavior, security, migration, tests, benchmarks,
and acceptance criteria". ODC-0002 covers goals, non-goals, failure behavior and
acceptance criteria. It does not cover design, interfaces, data flow, security,
migration, tests, or benchmarks.

**Suggested fix.** Add `docs/templates/baseline-spec.md`, then bring ODC-0002 to
it. At minimum add `## Current state and evidence`, `## Manifest schema`,
`## Redaction and privacy`, and an explicit
`## Tests and benchmarks - not applicable, owned by ODC-0004 / ODC-0003`
so the omission is a decision rather than a gap.

---

### 7. MAJOR - The JSON manifest has no schema; its only validation is trivially satisfiable

**Location:** `:47-48` (deliverable), `:83` (validation).

**Problem.** The manifest is described as "normalized toolchain, SDK, platform,
dependency, and artifact metadata" and validated by "JSON manifest parses with
Python's standard library." A file containing `{}` satisfies that criterion. The
word "normalized" is doing load-bearing work with no definition - normalized
how? Which key names? Which version-string format? This is an unresolved
implementation decision under program rule 6, and it is the interface that
ODC-0003's result manifest and ODC-0303's compatibility matrix will have to
consume.

**Suggested fix.** Inline the full key schema in the spec (or add
`docs/schemas/baseline-environment.schema.json` and reference it), and change
validation `:83` to "manifest validates against
`docs/schemas/baseline-environment.schema.json`". Include at minimum:
`schema_version`, `repo.revision`, `toolchain.{swift,xcode_version,xcode_build}`,
`sdks[]`, `host.{model_identifier,chip,cores,memory_gb}`,
`package.{tools_version,platforms,products,targets}`,
`dependencies[].{identity,requirement,resolved_version,resolved_revision}`,
`artifact.{url,checksum,slices[].{identifier,platform,variant,architectures,bytes,symbol_count}}`,
`build_matrix[].{triple,sdk,result,warning_count}`,
`warnings[].{group,file,line,text}`.

---

### 8. MAJOR - Warning inventory is undercounted and one "warning" is actually a functional defect

**Location:** `:33` (goal), `:65-66` (step 11).

**Problem.** The spec anticipates unhandled *package resources*. The actual
diagnostic at HEAD is:

```
warning: 'odc-fresh': found 8 file(s) which are unhandled;
explicitly declare them as resources or exclude from the target
```

 -  **8**, not 7: the seven `.metal` files under
`Sources/OnDeviceCatalyst/Metal Engine/Shaders/` plus
`Sources/OnDeviceCatalyst/Assets.xcassets`. (An asset catalogue shipping inside
a library target is itself a finding worth recording.)

More importantly, step 11 files this under "package resources" as if it were
cosmetic. It is not.
`Sources/OnDeviceCatalyst/Metal Engine/Compute/MetalComputeEngine.swift:89`
calls `device.makeDefaultLibrary()`. With the `.metal` files unhandled, SwiftPM
compiles no `.metallib` into the package bundle, so the entire Metal Engine
(`MetalBackend.swift`, `TransformerGraph.swift`, `KVCache.swift`,
`ModelWeights.swift`, the GGUF parser/tokenizer) is unreachable when the package
is consumed as a package. That is a whole subsystem that is dead in the shipping
configuration, and the baseline should say so in the results, not bury it in a
warning bucket.

The spec's other three buckets are accurate and do have content - verified
counts from the iOS-simulator build, ODC-owned warnings only:

- deprecated llama API: 9 sites, all in `API Bridge/LlamaBridge.swift`
  (`:56`, `:140`, `:158`, `:194`, `:200`, `:205`, `:210`, `:216` ×2);
- unused code: 6 sites (`ModelProfile.swift:78`, `ModelWeights.swift:87`,
  `ConfigurationPresets.swift:110`, `MLXInstance.swift:220`,
  `GGUFTokenizer.swift:191`, `Service Layer/Catalyst.swift:291`);
- concurrency/lifecycle: 2 sites, both in `Service Layer/Catalyst.swift`
  (`:421` "result of call to `sync(flags:execute:)` is unused"; `:496` "no
  `async` operations occur within `await` expression").

**Suggested fix.** Correct the count to 8 and name `Assets.xcassets`
separately. Split step 11's "package resources" bucket into "cosmetic packaging
warnings" and "warnings that indicate a non-functional subsystem", and require
the report to state the `makeDefaultLibrary()` consequence with a mapped ticket.

---

### 9. MAJOR - "Every actionable failure maps to a ticket" has no allocation rule

**Location:** `:96` (acceptance), `:45-46` (deliverable "links to follow-up
tickets").

**Problem.** The spec requires ticket creation but specifies no ID range, no
severity rubric distinguishing "actionable" from "recorded", no milestone, no
owner, and no priority policy. `Tickets.md` currently has a dense P0 block
(ODC-0000…0005) and then jumps to ODC-0100; where do baseline-defect tickets
land? A reviewer cannot tell whether "actionable" means "the macOS build fails"
(clearly yes) or "`tensorPtr` is unused at `ModelWeights.swift:87`" (probably
no). Two implementers would produce different ledgers from the same evidence.
Program rule 6 again.

**Suggested fix.** Fix the ID range in the spec (e.g. ODC-0010…ODC-0049 reserved
for baseline defects), define "actionable" as a two-line rubric, and state the
default `Milestone`/`Priority`/`Owner`/`Next Gate` values for tickets created by
this procedure so `scripts/validate-project-state.py` still passes afterwards.

---

### 10. MAJOR - Step 9 conflates two outcomes; `swift test` cannot run at all, and the macOS root cause should be asserted up front

**Location:** `:31` (goal), `:63` (step 9), `:87` (validation).

**Problem.** Step 9 says "Run macOS `swift build` and `swift test` and capture
their first root failure" - as if these are two data points. They are not.
Verified:

```
$ swift build            # macOS, arm64-apple-macosx26.0
…/Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:12:8:
error: no such module 'llama'
exit=1
```

`swift test` fails identically and produces zero test signal, because the
library target never compiles. The root cause is deterministic and already
knowable: `Package.swift:21` declares `.macOS(.v14)`, but the XCFramework
`Info.plist` exposes only `ios-arm64` and `ios-arm64-simulator` - there is no
macOS slice, so `import llama` cannot resolve on any macOS triple. The macOS
platform declaration is unbacked.

Because the spec treats this as something to be "captured", the report can be
written without ever stating the causal chain, and validation `:87` ("Root
causes distinguish package, binary, project, and host limitations") gives no
guidance for which bucket this belongs to - it is simultaneously a *package*
claim (`.macOS(.v14)`) and a *binary* limitation (no slice).

**Suggested fix.** State the expected finding and its root cause in the spec's
`Current state and evidence` section, and require the report to record it as
"package manifest declares a platform the binary artifact does not support".
Change step 9 to "run macOS `swift build`; record that `swift test` is blocked
by the same failure and yields no test evidence" so the report cannot imply that
tests were exercised. Map to a follow-up ticket.

---

### 11. MAJOR - `swift-tools-version: 5.12` is not a real Swift release and the baseline never records the true minimum toolchain

**Location:** `Package.swift:1`; spec `:29` ("Record package products, targets,
dependency pins, and binary artifact slices") does not cover manifest metadata.

**Problem.** `Package.swift:1` declares `// swift-tools-version: 5.12`, and
`swift package tools-version` reports `5.12.0`. No Swift 5.12 toolchain has ever
shipped (5.9 → 5.10 → 6.0). SwiftPM accepts it here only because the installed
tools version (6.3) compares greater. The practical effect is that the package
silently requires a toolchain whose numeric tools version exceeds 5.12 - i.e.
Swift 6.x - while appearing to advertise 5.x compatibility. A baseline whose
whole purpose is the build matrix should record the *actual* minimum consumer
toolchain, not the declared one.

**Suggested fix.** Add `package.tools_version` (declared and as reported by
`swift package tools-version`) to the manifest schema, and add a required
finding line: "declared tools version does not correspond to a released Swift
release; minimum usable toolchain is <measured>". Map to a follow-up ticket.

---

### 12. MINOR - Step 5 is not executable as written

**Location:** `:58` ("Run `swift package describe` before and after
resolution").

`swift package describe` performs resolution implicitly, so there is no
"before resolution" state to observe from a clean checkout; the only way to get
one is a pre-populated `.build/`, which finding 4(a) shows is an unpinned
confounder. Reword to "run `swift package dump-package` (manifest only, no
resolution) and `swift package describe --type json` (post-resolution)", which
gives the intended before/after without depending on machine state.

---

### 13. MINOR - The device-class goal is orphaned

**Location:** `:27` (goal, "connected test-device class"), `:57` (step 4).

No procedure step builds for, installs on, or runs anything on a device, and
Non-goals `:40` forbids running inference. Recording a connected device model
that is never used adds a redaction risk (finding 5) for zero evidentiary value.
Either drop goal `:27`/step 4, or add the device build from finding 1 and say
what the device identity is evidence *of*.

---

### 14. MINOR - Orphaned test sources are not inventoried

**Location:** `Package.swift:58-62`; spec `:29`.

`Tests/EmbeddingTest.swift`, `Tests/test_embedding.swift` and
`Tests/BERTEmbeddingTest.swift` sit outside the declared test-target path
(`path: "Tests/OnDeviceCatalystTests"`) and are never compiled by any target.
The only compiled test file is
`Tests/OnDeviceCatalystTests/OnDeviceCatalystTests.swift` (44 lines). Since
ODC-0004 will build directly on this inventory, the baseline should record
"compiled test sources" vs. "orphaned files under `Tests/`".

---

### 15. MINOR - The review record is ambiguous about provenance

**Location:** `:109-113`.

The record already lists both a completeness review and an "adversarial review"
dated 2026-08-25 - i.e. before this pass-two review existed - and neither names
a reviewer or references an artifact. Under the program's two-pass model the
record should name the pass, the reviewer, and the review document path, or a
future reader cannot tell whether pass two actually happened.

---

### 16. MINOR - Maintenance cost is asserted but never bounded

**Location:** `:67` (step 12, "Run all commands again that the report labels
reproducible"), `:86` (validation, "repeatable from a clean scratch path").

**Problem.** The spec commits the project to an indefinitely re-runnable manual
procedure with no automation and no expiry. Twelve manual steps across three
build systems, re-run by hand, will rot: the first person to change
`Package.swift` invalidates the report with no signal. The program's 14-day
evidence rule (`evidence_fresh_until: 2026-09-08` in this spec's front matter)
makes this concrete - but `scripts/validate-project-state.py` never evaluates
that field (verified: `:148-168` requires the key's *presence* only), so nothing
detects expiry.

Is it worth it? Yes - the underlying evidence gates three tickets. But it should
be a script, not a runbook.

**Suggested fix.** Require `scripts/capture-baseline.sh` (or `.py`) that emits
both deliverables, so re-running is one command and drift is a diff; and either
extend `validate-project-state.py` to fail on an expired
`evidence_fresh_until` for non-DONE tickets, or state in the spec that freshness
is enforced by founder review only.

---

## Scope discipline assessment

The spec is *mostly* clean on the program's "no new runtime code before the
research gate" rule. Non-goals `:38-42` are correctly drawn, and the
`Alternatives considered` rejection of "fixing packaging while measuring"
(`:101`) is the right instinct.

The one leak is deliverable 3 (finding 2): committing regenerated
`Package.resolved` output is a change to a tracked build-input file, made under
a ticket whose acceptance criterion asserts nothing changed. It is defensible  - 
the effective graph is unchanged - but it must be argued in the spec rather than
slipped in as a deliverable, and the lockfile-format bump must be evaluated as a
compatibility change.

No other behavior change is smuggled in.

---

## Minimum set of changes required to reach APPROVED

1. **Add `## Current state and evidence`** to the spec, containing the verified
   facts that the procedure would otherwise miss: the stub simulator slice
   (finding 1), the divergent `OnDeviceCatalyst/` source fork (finding 3), the
   macOS root cause (finding 10), the 8-file unhandled-resource warning and its
   `makeDefaultLibrary()` consequence (finding 8), and the `5.12` tools-version
   anomaly (finding 11).
2. **Resolve the `Package.resolved` decision to one outcome** and record why
   "preserve HEAD" is not viable; add a required sub-step evaluating the
   lockfile format 2 → 3 / `originHash` consumer impact (finding 2).
3. **Annotate the simulator build result as stub-linked**, add slice
   size/symbol-count capture, add an `arm64-apple-ios17.0` device-triple compile,
   and state that neither SwiftPM path links the static library (finding 1).
4. **Add a "Pinned environment and clean state" section**: exact Xcode build
   number, Swift version, SDK build versions, `xcode-select -p`, a declared
   SwiftPM cache policy, mandatory `.build`/`DerivedData` removal, and captured
   `git status --porcelain --ignored` + `git clean -ndx` output (finding 4).
5. **Add a `.gitignore` entry for the scratch directory** used by this procedure,
   or name a scratch path already covered by the tracked `.gitignore`
   (finding 4c).
6. **Replace the four unverifiable acceptance criteria with commands**
   (finding 5), including a concrete redaction denylist and a
   `git diff --stat 59da80b -- …` no-change assertion.
7. **Inline the environment-manifest schema** (or add a JSON Schema file) and
   change validation `:83` to schema validation (finding 7).
8. **Fix the ticket-allocation gap**: reserved ID range, "actionable" rubric,
   and default ledger column values (finding 9).
9. **Add `docs/templates/baseline-spec.md`** and add the explicit
   "tests/benchmarks are out of scope, owned by ODC-0004/ODC-0003" statement
   (finding 6).
10. **Correct step 9** so `swift test` is recorded as blocked-by-build rather
    than as an independent result (finding 10), and **correct step 5** to
    `dump-package` + `describe` (finding 12).
11. **Refresh `last_updated` / `evidence_fresh_until`** on revision - the current
    window closes 2026-09-08, seven days from this review, and the procedure has
    not been executed.

Items 1-6 are the blocking set. Items 7-11 are required for a spec that contains
zero unresolved implementation decisions, which program rule 6 makes a
precondition for `APPROVED`.
