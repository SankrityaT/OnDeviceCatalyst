# ODC-0100 v3 vision and migration: verification of the Package Traits claim

Scope: verify, from primary sources only, the claim in
`docs/specs/ODC-0100-v3-vision-and-migration.md` `## Package boundaries`, `##
Compatibility policy`, and `## Alternatives considered` that SwiftPM Package
Traits let `ODC-0103` gate `llama.cpp` and `mlx-swift-lm` such that `swift
package show-dependencies` with no traits enabled lists neither dependency,
and enabling a backend's trait makes exactly that dependency appear. All
access dates below are 2026-09-02 (as instructed), reflecting when the cited
pages were fetched for this review.

## Verdict: PARTIAL

The underlying mechanism is real and the acceptance gate as literally written
(`show-dependencies`, no traits vs. one trait) is achievable and independently
supported by SwiftPM's own source code. But the spec cites the wrong proposal
number, and the spec's prose about "resolution" glosses over a real
distinction SwiftPM itself draws between dependency *version resolution*
(pinning, `Package.resolved`) and *module graph construction* (what `show-
dependencies` actually reports). ODC-0103 must fix the citation and tighten
the wording before this is safe to ship as a checkable gate. Details below,
per question.

---

### 1. Does SE-0452 exist, and is it the traits proposal?

**No. The spec cites the wrong proposal number.** SE-0452 is "Integer Generic
Parameters" (adds fixed-size collection support to Swift, unrelated to
SwiftPM), reviewed through November 19, 2024, implemented in Swift 6.2. The
actual Package Traits proposal is **SE-0450, "Package traits."**

- SE-0450 status line, fetched verbatim from the proposal source: `* Status:
  **Implemented (Swift 6.1)**`. Authors: Franz Busch, Max Desiatov. Review
  manager: Mishal Shah. Acceptance thread: "Accepted with modifications."
  Source: https://raw.githubusercontent.com/swiftlang/swift-evolution/main/proposals/0450-swiftpm-package-traits.md
  (accessed 2026-09-02).
- Confirms it is a different proposal: https://github.com/swiftlang/swift-evolution/blob/main/proposals/0452-integer-generic-parameters.md
  (accessed 2026-09-02), title "Integer Generic Parameters."

**Action for ODC-0103 / this spec:** every occurrence of "SE-0452" in `##
Package boundaries` must be corrected to **SE-0450**. This is a citation
defect, not evidence the mechanism itself is fictional - the mechanism
("SwiftPM Package Traits," named traits, disabled-by-default, tools-version
6.1) matches SE-0450 exactly. But a spec that gets its own load-bearing
citation wrong is not yet trustworthy, and this is exactly the kind of error
this project has now shipped multiple times.

### 2. What tools-version does SE-0450 require? Is 6.1 correct?

**Confirmed correct.** SE-0450's status line states "Implemented (Swift
6.1)," and Swift 6.1 was released 2025-03-31, shipped in Xcode 16.3. Source:
https://www.swift.org/blog/swift-6.1-released/ (accessed 2026-09-02): "Swift
6.1 is included in Xcode 16.3, now available from the App Store," and
describes package traits as a new Swift 6.1 feature. The spec's "6.1 is the
first tools version with trait support" is correct.

### 3. THE CRITICAL QUESTION - do traits gate resolution or only compilation?

**One sentence answer:** Traits do not affect SwiftPM's version-pinning
("dependency resolution" in SE-0450's own vocabulary, the step that produces
`Package.resolved`), but they do prune the subsequent module-graph
construction that tools like `show-dependencies` read from, so the specific
command the spec's gate names will in practice omit a trait-disabled
dependency even though the proposal text technically denies that traits
affect "resolution."

Primary-source detail:

SE-0450 itself, in `## Future directions` -> `### Consider traits during
dependency resolution`, says, quoted verbatim from
https://raw.githubusercontent.com/swiftlang/swift-evolution/main/proposals/0450-swiftpm-package-traits.md
(accessed 2026-09-02):

> "The implementation to this proposal only considers traits **after** the
> dependency resolution when constructing the module graph. This is inline
> with how platform specific dependencies are currently handled. In the
> future, both platform specific dependencies and traits can be taken into
> consideration during dependency resolution to avoid fetching an optional
> dependency that is not enabled by a trait. Changing this **doesn't**
> require a Swift evolution proposal since it is just an implementation
> detail of how dependency resolution currently works."

Read literally, this is exactly the failure mode the spec is trying to avoid:
by SE-0450's own definition, "dependency resolution" (version pinning,
`Package.resolved`, which packages get fetched) does **not** consider
traits today; not fetching a disabled trait's dependency is listed as a
**future direction**, not current behavior. This is corroborated by a hidden,
explicitly experimental SwiftPM CLI flag found in the current source
(`Sources/CoreCommands/Options.swift`, accessed 2026-09-02 via
https://raw.githubusercontent.com/swiftlang/swift-package-manager/main/Sources/CoreCommands/Options.swift):

> `--experimental-prune-unused-dependencies`: "Enables the ability to prune
> unused dependencies of the package to avoid redundant loads during
> resolution." Comment above it: "Hidden from the generated help text
> because this feature is currently only being considered for traits."

That flag is off by default, which confirms pruning during the fetch/pin step
is not yet standard.

However, SwiftPM draws a second, separate line inside what it colloquially
calls "the resolved dependency graph": **module graph construction**, which
happens after version pinning and is where `swift package show-dependencies`
actually reads from (its own command abstract, verbatim from
`Sources/Commands/PackageCommands/ShowDependencies.swift`, accessed
2026-09-02: `abstract: "Print the resolved dependency graph."`, and its `run`
body calls `swiftCommandState.loadPackageGraph()` and dumps the resulting
`ModulesGraph`). Tracing that code path in the current `main` branch of
`swiftlang/swift-package-manager` (accessed 2026-09-02):

- `Sources/PackageModel/Manifest/Manifest.swift`, `dependenciesRequired(for
  productFilter:, _ enabledTraits:)`, filters `self.dependencies` by
  `isTargetDependencyEnabled(... enabledTraits:)` (or, on the
  `#if ENABLE_TARGET_BASED_DEPENDENCY_RESOLUTION` path, by
  `isPackageDependencyUsed($0, enabledTraits:)`). A package dependency whose
  only referencing target-dependency edges are trait-guarded and whose trait
  is not in `enabledTraits` is dropped from the returned list.
- `Sources/PackageGraph/ModulesGraph+Loading.swift`'s graph-loading traversal
  (`nodeSuccessorProvider`) walks `node.item.requiredDependencies +
  node.item.traitGuardedDependencies` to decide which package nodes
  (`allNodes`) enter the graph at all. Because `requiredDependencies` is
  exactly the trait-filtered list above, a fully trait-gated, disabled
  dependency's package node is never visited and never enters the
  `ModulesGraph` that `show-dependencies` dumps.

This is corroborated by SwiftPM's own bug tracker describing the intended
behavior in the same terms: issue
https://github.com/swiftlang/swift-package-manager/issues/8398 ("[Traits]
Used package dependency omitted when traits guard some target dependencies,"
accessed 2026-09-02) states the bug as "SwiftPM preemptively omits a package
dependency it believes to be trait-guarded when there are existing target
dependencies that aren't guarded" - the complaint is about over-omission in
an edge case, which presupposes that omission of disabled-trait dependencies
from the graph is the designed, normal behavior being over-applied.

**Net effect for this spec:** the acceptance gate as literally written,
naming `show-dependencies`, is technically achievable, because that specific
tool's output tracks the trait-aware module graph, not the trait-blind
version-pinning step. But the spec's own prose in `## Package boundaries`
("`swift package show-dependencies` reports the package graph resolved from
a manifest's `dependencies:` array; that resolution happens once per
manifest, independent of which product a downstream consumer selects")
describes the pre-SE-0450 behavior accurately for *product* selection, but
does not yet state, and should state, that trait selection is the mechanism
that changes this specific tool's output, distinct from and narrower than
"resolution" in SE-0450's own vocabulary. A reader who takes "resolution" at
SE-0450's own word (version pinning) would wrongly conclude the gate is
unachievable; a reader who tests `show-dependencies` empirically will find it
does track trait state. The spec should say explicitly: "traits do not
change what SwiftPM fetches or pins in `Package.resolved` by default (that
remains a future direction per SE-0450), but they do change what enters the
module graph that `show-dependencies` reports, which is the concrete
behavior this gate depends on" - and should not describe this as
"resolution" without that caveat, given that word is the exact one the prior,
rejected draft was faulted for using loosely.

### 4. Is `show-dependencies` trait-aware? What are the real flags?

**Yes, mechanically** (see question 3), and **yes, its CLI surface accepts
trait flags**, though not documented in its own `--help` text. Traced in
`swiftlang/swift-package-manager` `main` branch, accessed 2026-09-02:

- `Sources/Commands/PackageCommands/ShowDependencies.swift` declares
  `@OptionGroup(visibility: .hidden) var globalOptions: GlobalOptions`.
- `Sources/CoreCommands/Options.swift`, `GlobalOptions`, includes `@OptionGroup(title:
  "Trait Options") public var traits: TraitOptions`.
- `TraitOptions` (same file) defines the real flags:
  - `--traits <comma-separated-list>` - "Enable the specified traits of the
    package... When enabling specific traits, you must also explicitly
    enable the default traits by passing `defaults` to this option." (i.e.
    `--traits defaults,MLXBackend` to keep defaults plus add one; `--traits
    MLXBackend` alone if there are no default traits to preserve.)
  - `--enable-all-traits` - "Enable all traits of the package."
  - `--disable-default-traits` - "Disable all default traits of the
    package."

Because `ShowDependencies` includes `GlobalOptions` (hidden but functional),
`swift package show-dependencies --traits MLXBackend`, `swift package
show-dependencies --disable-default-traits`, and `swift package
show-dependencies --enable-all-traits` are all real, working invocations
today, not proposed or future syntax.

One caveat worth recording: as recently as 2025-08-15, a SwiftPM community
member filed https://github.com/swiftlang/swift-package-manager/issues/9033
("expand the information provided by `swift package show-dependencies` to
include traits") stating "Right now it's hard to determine what features are
exposed (and enabled) by dependencies you're using _other_ than inspecting
the relevant Package.swift" - i.e. the tool's *display* of trait state was
incomplete. The fix, https://github.com/swiftlang/swift-package-manager/pull/9034,
"adds traits to dependency output in show-dependencies --json and --text,"
merged 2025-08-27. This PR is about **annotating** which traits are enabled
per dependency in the output, not about the underlying inclusion/exclusion
logic verified in question 3 (which is older, part of the original SE-0450
implementation). Practically: ODC-0103's acceptance-gate authors should pin
a SwiftPM/Xcode toolchain version and confirm empirically (once builds are
possible again) that their toolchain's `show-dependencies` behaves as
described, since this area of SwiftPM has had multiple bug reports
(`#8398`, `#8626`, `#9286`, `#10448`) about trait-guarded dependencies
leaking into places (linking, `dump-symbol-graph`, plugin `getSymbolGraph`,
resolution failures on some registries) where they should have been
excluded. UNVERIFIED: whether the exact SwiftPM point release ODC-0103 will
target has any of these specific bugs unfixed; that requires a build-time
check this review was barred from performing.

### 5. What toolchain floor does tools-version 6.1 impose on consumers?

Confirmed from https://www.swift.org/blog/swift-6.1-released/ (accessed
2026-09-02): Swift 6.1 released 2025-03-31, "included in Xcode 16.3." A
manifest declaring `// swift-tools-version: 6.1` therefore requires every
consumer, including one who wants only core contracts, to resolve the
manifest with **Xcode 16.3 / Swift 6.1 or later** - Xcode 16.2 and earlier,
and any Swift toolchain below 6.1, cannot open the manifest at all. This
matches what the spec already states in `## Package boundaries` (it declines
to name the Xcode number itself, saying that requires a toolchain check;
16.3 is that number).

### 6. Tools-version vs. deployment-target: is the spec's distinction correct?

**Yes, precisely correct**, and this is standard, documented SwiftPM
behavior, not something specific to traits. From
https://docs.swift.org/package-manager/PackageDescription/PackageDescription.html
(accessed 2026-09-02):

> "The Swift tools version declares the version of the `PackageDescription`
> library, the minimum version of the Swift tools and Swift language
> compatibility version to process the manifest, and the minimum version of
> the Swift tools that are needed to use the Swift package."

versus, for `platforms:`/`SupportedPlatform`:

> "By default, the Swift Package Manager assigns a predefined minimum
> deployment version for each supported platform unless you configure
> supported platforms using the `platforms` API."

These are independent axes: `swift-tools-version` gates which SwiftPM/Xcode
toolchain can *read and resolve* the manifest; `platforms:` gates the
*minimum OS the compiled binary can run on*. Raising the former to 6.1 does
not touch the latter. The spec's `## Compatibility policy` and `## Package
boundaries` state this correctly and do not conflate the two.

### 7. `swift-tools-version: 5.12` is not a shipped release - what does that imply?

Corroborated: SwiftPM's own `ToolsVersion` model defines discrete constants
through `.v5_10`, then jumps directly to `.v6_0` - there is no `5.11` or
`5.12` constant in the shipped tool. (Source: SwiftPM's `ToolsVersion.swift`
and PackageDescription documentation, cross-checked 2026-09-02; this
corroborates, rather than newly establishes, the finding the prompt says an
earlier review already made.) Concretely, this means today's
`Package.swift:1` (`// swift-tools-version: 5.12`) is not a value any real
Swift toolchain shipped; it does not correspond to the 5.9/5.10/6.0/6.1
lineage. For the migration path this implies ODC-0103 is not "bumping" a
real, currently-valid tools-version to 6.1 - it is replacing a value that
was never valid in the first place. That strengthens, not weakens, the case
for fixing it now: there is no working intermediate state to preserve, and
no consumer today can be resolving this manifest correctly at 5.12, so the
jump to 6.1 has no real transitional compatibility cost beyond what raising
tools-version always costs (see question 5).

---

## Summary of required fixes to the spec

1. Replace every "SE-0452" reference with **SE-0450** in `## Package
   boundaries` (and anywhere else it appears). This is a factual citation
   error, confirmed against the proposal text and against SE-0452's actual,
   unrelated content (Integer Generic Parameters).
2. Tighten the "resolution" language in the "Dependency-graph optionality"
   subsection to distinguish, in SwiftPM's own terms, *dependency resolution*
   (version pinning / fetching / `Package.resolved`, which traits do **not**
   affect per SE-0450's explicit "Future directions" text) from *module graph
   construction* (which traits **do** affect, and which is what `swift
   package show-dependencies` actually reports, per the current SwiftPM
   source). The acceptance gate itself (testable via `show-dependencies` with
   `--disable-default-traits` / `--traits <name>` / `--enable-all-traits`) is
   sound and should be kept, but the surrounding prose should not claim
   traits gate "resolution" in the sense SE-0450 uses that word, or a future
   adversarial review will find the same kind of gap the prior draft was
   rejected for, just one level deeper.
3. Note, for completeness, that a disabled-trait dependency's manifest may
   still be fetched over the network during version pinning by default
   (pruning that is currently only available behind the experimental,
   non-default `--experimental-prune-unused-dependencies` flag) - this is a
   real, minor cost (network/disk during `swift package resolve`) distinct
   from the "does it appear in `show-dependencies` / get linked" question the
   spec's gate is actually about, and should not be conflated with it either
   way.

None of this requires abandoning Package Traits for a multi-package split or
weakening the claim to build/link-only optionality - the second and third
`## Alternatives considered` options remain correctly rejected. The trait
mechanism, once cited correctly and described precisely, supports the
resolution-level (module-graph-level) claim the spec wants.
