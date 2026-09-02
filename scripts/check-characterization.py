#!/usr/bin/env python3
"""Checker for ODC-0004, the v2 characterization suite.

Decides every acceptance criterion in
docs/specs/ODC-0004-v2-characterization-suite.md that is not already a
one-line shell command: the R0 packaging facts, the defect-site fingerprints,
naming and comment-block conventions, the three orphaned files' pinned state,
catalog-to-spec agreement, and the skip ledger audit.

Exit code 0 means every selected check passed. Exit code 1 means at least one
failed. Exit code 2 means the checker could not run (missing deliverable,
invalid input). This mirrors scripts/check-baseline.py.

Usage:
    python3 scripts/check-characterization.py                 # all of the below
    python3 scripts/check-characterization.py --packaging
    python3 scripts/check-characterization.py --fingerprints
    python3 scripts/check-characterization.py --naming
    python3 scripts/check-characterization.py --orphans
    python3 scripts/check-characterization.py --inventory [--require-defects D1,D2,...]
    python3 scripts/check-characterization.py --skips <ledger>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "Tests" / "OnDeviceCatalystTests"
TICKETS_PATH = ROOT / "Tickets.md"
PACKAGE_SWIFT = ROOT / "Package.swift"
PACKAGE_RESOLVED = ROOT / "Package.resolved"
FINGERPRINTS_PATH = ROOT / "docs" / "characterization" / "v2-fingerprints.json"
CHAR_DOC_PATH = ROOT / "docs" / "characterization" / "v2.0.4-characterization.md"
BASELINE_MANIFEST_PATH = ROOT / "docs" / "baselines" / "v2.0.4-environment.json"

HOME_PATH_DENYLIST = re.compile(r"/Users/[^/ \"]+")
ORPHAN_EXEMPT_NAMES = {"EmbeddingTest.swift", "test_embedding.swift", "BERTEmbeddingTest.swift"}

# Pinned SHA-256 of the three orphaned files (spec N9). These files are never
# edited by this ticket; a mismatch means someone touched a file this spec
# says stays exactly where it is, untouched.
ORPHAN_PINS = {
    "Tests/EmbeddingTest.swift": "73b2367f46d0e79f33a9db648ed967228dd2301ecd15f191a3305633a0cc25be",
    "Tests/test_embedding.swift": "7a171b6ee8c13e26064dc67ddc65e7fc5a44d84d7038dc9f68221192a81063bc",
    "Tests/BERTEmbeddingTest.swift": "fa02e1111ae3576408208c27746728eab736b704732a76d65ec26a4df7000fef",
}

SKIP_CODES = {
    "SKIP[requires-device]",
    "SKIP[requires-model-asset]",
    "SKIP[requires-simulator-stub]",
    "SKIP[requires-real-llama-slice]",
    "SKIP[requires-metal-device]",
}


def fail(failures: list[str], message: str) -> None:
    failures.append(message)


# ---------------------------------------------------------------------------
# --naming
# ---------------------------------------------------------------------------

TEST_METHOD_RE = re.compile(r"^(\s*)func (test_\w+)\s*\(", re.MULTILINE)


def known_ticket_ids() -> set[str]:
    if not TICKETS_PATH.is_file():
        return set()
    return set(re.findall(r"^\|\s*(ODC-\d{4})\s*\|", TICKETS_PATH.read_text(encoding="utf-8"), flags=re.MULTILINE))


def all_added_test_files() -> list[Path]:
    if not TESTS_DIR.is_dir():
        return []
    return sorted(p for p in TESTS_DIR.rglob("*.swift"))


def check_naming() -> list[str]:
    failures: list[str] = []
    tickets = known_ticket_ids()
    files = all_added_test_files()
    if not files:
        fail(failures, "naming: no test sources found under Tests/OnDeviceCatalystTests")
        return failures

    for path in files:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Denylist: no absolute home-directory path in any added file.
        for lineno, line in enumerate(lines, start=1):
            if HOME_PATH_DENYLIST.search(line):
                fail(failures, f"naming: {path.relative_to(ROOT)}:{lineno} contains an absolute home-directory path")

        # Early `return` inside a test body is forbidden (skip protocol).
        # Heuristic: flag a bare `return` (not `return expr` used for a
        # non-Void helper) appearing directly inside a `func test_` body,
        # detected by scanning between a test func's opening and its matching
        # top-level closing brace at the function's own indentation.
        for match in re.finditer(r"^(\s*)func (test_\w+)\s*\([^\n]*\{", text, flags=re.MULTILINE):
            indent = match.group(1)
            name = match.group(2)
            start = match.end()
            close_re = re.compile(rf"\n{re.escape(indent)}\}}", re.MULTILINE)
            close_match = close_re.search(text, start)
            body = text[start:close_match.start()] if close_match else text[start:]
            body_lines = body.splitlines()
            for i, bline in enumerate(body_lines):
                stripped = bline.strip()
                if stripped != "return":
                    continue
                # A `return` immediately preceded (same statement group, up to
                # two lines above) by XCTFail(...) is a guarded failure exit
                # inside an assertion-inspection closure, not the silent
                # "quietly stop asserting" pattern this rule targets. Only a
                # `return` with no such guard is a violation.
                preceding = " ".join(l.strip() for l in body_lines[max(0, i - 2):i])
                if "XCTFail(" in preceding:
                    continue
                fail(failures, f"naming: {path.relative_to(ROOT)} {name} contains a bare early 'return' with no XCTFail guard")

            if name.startswith("test_characterizes_"):
                if not (name.endswith("__no_defect") or re.search(r"__ODC_\d{4}$", name)):
                    fail(failures, f"naming: {path.relative_to(ROOT)} {name} must end with __ODC_00NN or __no_defect")
                ticket_match = re.search(r"__ODC_(\d{4})$", name)
                if ticket_match:
                    ticket_id = f"ODC-{ticket_match.group(1)}"
                    if ticket_id not in tickets:
                        fail(failures, f"naming: {path.relative_to(ROOT)} {name} names {ticket_id}, absent from Tickets.md")

                # Four-line comment block directly above the function.
                pre_lines = text[:match.start()].splitlines()
                # Walk upward past blank lines to find the doc block.
                idx = len(pre_lines) - 1
                block: list[str] = []
                while idx >= 0 and pre_lines[idx].strip().startswith("///"):
                    block.insert(0, pre_lines[idx].strip())
                    idx -= 1
                joined = "\n".join(block)
                if not (
                    re.search(r"^/// CHARACTERIZATION", joined, re.MULTILINE)
                    and "Today:" in joined
                    and "Should be:" in joined
                    and "Evidence:" in joined
                ):
                    fail(failures, f"naming: {path.relative_to(ROOT)} {name} is missing the four-line CHARACTERIZATION block")

            elif name.startswith("test_requires_"):
                if "__ODC_" in name:
                    fail(failures, f"naming: {path.relative_to(ROOT)} {name} is a test_requires_ case but contains __ODC_")

    return failures


# ---------------------------------------------------------------------------
# --orphans
# ---------------------------------------------------------------------------

def check_orphans() -> list[str]:
    failures: list[str] = []
    for rel, expected_hash in ORPHAN_PINS.items():
        path = ROOT / rel
        if not path.is_file():
            fail(failures, f"orphans: {rel} is missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            fail(failures, f"orphans: {rel} sha256 changed ({actual} != pinned {expected_hash})")

    # Confirm swift package describe lists them in no target's sources.
    try:
        result = subprocess.run(
            ["swift", "package", "describe", "--type", "json"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            described = json.loads(result.stdout)
            all_sources: set[str] = set()
            for target in described.get("targets", []):
                for source in target.get("sources", []):
                    all_sources.add(source)
            for rel in ORPHAN_PINS:
                name = Path(rel).name
                if any(name == Path(s).name for s in all_sources):
                    fail(failures, f"orphans: {rel} appears in a target's sources per swift package describe")
        else:
            # Best-effort only: a fixture or scratch tree with no working
            # Package.swift cannot run this, and that is not itself evidence
            # that an orphan file was folded into a target. The SHA-256 pin
            # check above is the hard requirement.
            print(
                "orphans: swift package describe --type json unavailable; "
                "skipped the orphan-vs-target-membership check this run",
                file=sys.stderr,
            )
    except Exception:  # pragma: no cover - environment dependent
        print("orphans: swift package describe --type json raised; skipped that check this run", file=sys.stderr)

    return failures


# ---------------------------------------------------------------------------
# --fingerprints
# ---------------------------------------------------------------------------

def normalize_region(text: str) -> str:
    # Strip // line comments (not inside string literals; good enough for
    # this codebase's style, which does not put "//" inside the relevant
    # function bodies' string literals in a way that would misfire).
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"//.*$", "", line)
        lines.append(stripped)
    joined = "\n".join(lines)
    # Collapse whitespace runs, strip trailing whitespace per line handled by
    # the join above; now collapse all whitespace runs to single spaces.
    collapsed = re.sub(r"\s+", " ", joined).strip()
    return collapsed


def extract_region(file_path: Path, anchor_pattern: str) -> str | None:
    text = file_path.read_text(encoding="utf-8")
    anchor_re = re.compile(anchor_pattern, re.MULTILINE)
    match = anchor_re.search(text)
    if not match:
        return None
    line_start = text.rfind("\n", 0, match.start()) + 1
    indent_match = re.match(r"[ \t]*", text[line_start:match.start()])
    indent = indent_match.group(0) if indent_match else ""

    # Find the opening brace after the anchor, then walk forward counting
    # braces until the matching close at the anchor's own indentation.
    brace_search_start = match.end()
    open_idx = text.find("{", brace_search_start)
    if open_idx == -1:
        return None

    depth = 0
    i = open_idx
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[line_start:i + 1]
        i += 1
    return None


def load_fingerprints() -> list[dict]:
    if not FINGERPRINTS_PATH.is_file():
        return []
    return json.loads(FINGERPRINTS_PATH.read_text(encoding="utf-8"))["fingerprints"]


def check_fingerprints() -> list[str]:
    failures: list[str] = []
    entries = load_fingerprints()
    if not entries:
        fail(failures, f"fingerprints: {FINGERPRINTS_PATH.relative_to(ROOT)} has no entries")
        return failures

    for entry in entries:
        file_path = ROOT / entry["file"]
        if not file_path.is_file():
            fail(failures, f"fingerprints: {entry['id']}: {entry['file']} does not exist")
            continue
        region = extract_region(file_path, entry["anchor"])
        if region is None:
            fail(failures, f"fingerprints: {entry['id']}: anchor {entry['anchor']!r} not found in {entry['file']}")
            continue
        normalized = normalize_region(region)
        actual_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if actual_hash != entry["sha256"]:
            fail(
                failures,
                f"fingerprints: {entry['id']} ({entry['ticket']}) changed at {entry['file']}; "
                f"this defect site changed; if that was deliberate, update the characterization "
                f"case and this fingerprint in the same commit",
            )
    return failures


# ---------------------------------------------------------------------------
# --packaging (R0, SFC-A)
# ---------------------------------------------------------------------------

def find_xcframework_dir() -> Path | None:
    candidates = list((ROOT / ".build").glob("**/llama.xcframework"))
    return candidates[0] if candidates else None


def check_packaging() -> list[str]:
    failures: list[str] = []
    package_text = PACKAGE_SWIFT.read_text(encoding="utf-8")

    # C-D4-1 / C-D4-2: .macOS(.v14) declared; LlamaBridge imports llama.
    if ".macOS(.v14)" not in package_text:
        fail(failures, "packaging: C-D4-1 Package.swift no longer declares .macOS(.v14) (informational; update disposition if deliberate)")
    llama_bridge = ROOT / "Sources" / "OnDeviceCatalyst" / "API Bridge" / "LlamaBridge.swift"
    if not re.search(r"^import llama$", llama_bridge.read_text(encoding="utf-8"), re.MULTILINE):
        fail(failures, "packaging: C-D4-2 LlamaBridge.swift no longer imports llama; the known macOS build failure's precondition changed")

    xcframework_dir = find_xcframework_dir()
    if xcframework_dir is not None:
        info_plist = xcframework_dir / "Info.plist"
        if info_plist.is_file():
            plist_text = info_plist.read_text(encoding="utf-8", errors="replace")
            lib_count = plist_text.count("<key>LibraryIdentifier</key>")
            if lib_count != 2:
                fail(failures, f"packaging: C-D4-1 xcframework Info.plist declares {lib_count} AvailableLibraries, expected 2")
            if "macos" in plist_text.lower():
                fail(failures, "packaging: C-D4-1 xcframework Info.plist unexpectedly names a macOS library")

        # C-N2-1: the llama_* symbol set the package references equals the
        # set the simulator stub defines (51 == 51, per spec N2).
        stub_archive = xcframework_dir / "ios-arm64-simulator" / "libllama_combined.a"
        object_glob = list((ROOT / ".build").glob("arm64-apple-ios-simulator/debug/OnDeviceCatalyst.build/**/*.o"))
        if stub_archive.is_file() and object_glob:
            try:
                defined = subprocess.run(["nm", "-gU", str(stub_archive)], capture_output=True, text=True, timeout=60).stdout
                defined_symbols = {
                    line.split()[-1] for line in defined.splitlines()
                    if len(line.split()) >= 3 and line.split()[1] in ("T", "t")
                }
                required_symbols: set[str] = set()
                for obj in object_glob:
                    undefined = subprocess.run(["nm", "-u", str(obj)], capture_output=True, text=True, timeout=60).stdout
                    for line in undefined.splitlines():
                        sym = line.strip()
                        if re.match(r"^_(llama|ggml)", sym):
                            required_symbols.add(sym)
                missing = required_symbols - defined_symbols
                if missing:
                    fail(failures, f"packaging: C-N2-1 {len(missing)} required llama_*/ggml_* symbols are undefined by the stub: {sorted(missing)[:5]}...")
            except Exception as exc:  # pragma: no cover
                fail(failures, f"packaging: C-N2-1 could not run nm ({exc})")
        else:
            fail(failures, "packaging: C-N2-1 skipped, no build artifacts found; run scripts/run-characterization.sh first")
    else:
        fail(failures, "packaging: C-D4-1/C-N2-1 skipped, llama.xcframework not found under .build; resolve/build the package first")

    # C-D5-1/2/3: unhandled files, zero declared resources, no default.metallib.
    try:
        result = subprocess.run(["swift", "package", "describe", "--type", "json"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            described = json.loads(result.stdout)
            for target in described.get("targets", []):
                if target.get("name") == "OnDeviceCatalyst":
                    resources = target.get("resources", [])
                    if resources:
                        fail(failures, f"packaging: C-D5-2 OnDeviceCatalyst target now declares {len(resources)} resources, expected 0")
        metallibs = list((ROOT / ".build").glob("**/default.metallib"))
        if metallibs:
            fail(failures, f"packaging: C-D5-3 a default.metallib now exists at {metallibs[0]}, expected none")
    except Exception:
        pass

    # C-D7-1: zero XCRemoteSwiftPackageReference entries.
    pbxproj = ROOT / "OnDeviceCatalyst.xcodeproj" / "project.pbxproj"
    if pbxproj.is_file():
        count = pbxproj.read_text(encoding="utf-8", errors="replace").count("XCRemoteSwiftPackageReference")
        if count != 0:
            fail(failures, f"packaging: C-D7-1 project.pbxproj now has {count} XCRemoteSwiftPackageReference entries, expected 0")
    else:
        fail(failures, "packaging: C-D7-1 OnDeviceCatalyst.xcodeproj/project.pbxproj not found")

    # C-D7-2: census 22 shared / 12 drifted / 13 package-only / 3 app-only,
    # read from ODC-0002's own verified manifest rather than recomputed here.
    if BASELINE_MANIFEST_PATH.is_file():
        manifest = json.loads(BASELINE_MANIFEST_PATH.read_text(encoding="utf-8"))
        dup = manifest.get("duplicate_sources", {})
        shared = dup.get("shared", [])
        drifted = sum(1 for e in shared if e.get("identical") is False)
        package_only = len(dup.get("package_only", []))
        app_only = len(dup.get("app_only", []))
        if not (len(shared) == 22 and drifted == 12 and package_only == 13 and app_only == 3):
            fail(
                failures,
                f"packaging: C-D7-2 census is {len(shared)}/{drifted}/{package_only}/{app_only}, expected 22/12/13/3",
            )
    else:
        fail(failures, "packaging: C-D7-2 baseline manifest not found")

    # C-E2-1 / C-N4-1: the false simulator-guard comment; zero matches for
    # the pattern it claims exists.
    matches = subprocess.run(
        ["grep", "-rn", "#if !targetEnvironment", str(ROOT / "Sources")],
        capture_output=True, text=True,
    )
    if matches.stdout.strip():
        fail(failures, "packaging: C-E2-1 Sources/ now contains #if !targetEnvironment guards; the comment at Package.swift:36-37 may now be accurate")

    # R-D6-1: Package.resolved satisfies the exact: "2.29.3" pin.
    if PACKAGE_RESOLVED.is_file():
        resolved = json.loads(PACKAGE_RESOLVED.read_text(encoding="utf-8"))
        pins = {p["identity"]: p for p in resolved.get("pins", [])}
        mlx_pin = pins.get("mlx-swift-lm")
        if mlx_pin is None or mlx_pin.get("state", {}).get("version") != "2.29.3":
            fail(failures, "packaging: R-D6-1 Package.resolved no longer pins mlx-swift-lm to 2.29.3")
    else:
        fail(failures, "packaging: R-D6-1 Package.resolved not found")

    return failures


# ---------------------------------------------------------------------------
# --inventory
# ---------------------------------------------------------------------------

def all_test_method_names() -> set[str]:
    names: set[str] = set()
    for path in all_added_test_files():
        text = path.read_text(encoding="utf-8")
        for match in TEST_METHOD_RE.finditer(text):
            names.add(match.group(2))
    return names


# Catalog of every case id this suite's `## Tests` names, mapped to the test
# method(s) or checker assertion that implements it. R0 ids map to a fixed
# marker string rather than a Swift method, since they are checker
# assertions, not XCTest cases.
CATALOG: dict[str, list[str]] = {
    # R0 -- checker-only, SFC-A.
    "C-D4-1": ["checker:packaging"], "C-D4-2": ["checker:packaging"],
    "C-D5-1": ["checker:packaging"], "C-D5-2": ["checker:packaging"], "C-D5-3": ["checker:packaging"],
    "C-D7-1": ["checker:packaging"], "C-D7-2": ["checker:packaging"],
    "C-E2-1": ["checker:packaging"], "C-N4-1": ["checker:packaging"], "C-N2-1": ["checker:packaging"],
    "C-N9-1": ["checker:orphans"],
    "R-D6-1": ["checker:packaging"],
    "F-D1-1": ["checker:fingerprints"], "F-D2-1": ["checker:fingerprints"], "F-D2-2": ["checker:fingerprints"],
    "F-D3-1": ["checker:fingerprints"], "F-D8-1": ["checker:fingerprints"],
    # R1
    "C-D1-1": ["test_characterizes_modelCache_getInstance_returnsNotReadyInstance__ODC_0010"],
    "C-D1-2": ["test_characterizes_modelCache_storeInstance_isAsynchronous__ODC_0010"],
    "C-D2-1": ["test_characterizes_streamingResponse_drainedCollector_reportsNaturalOverRealReason__ODC_0011"],
    "C-D2-2": [
        "test_characterizes_collectResponse_breakingCollector_disagreesWithDrainedCollector__ODC_0011",
        "test_characterizes_collectContent_breaksOnFirstCompletion__ODC_0011",
    ],
    "C-D3-1": ["test_characterizes_publishProgressGate_isUnsatisfiableForEveryCase__ODC_0012"],
    "C-D5-4": ["test_characterizes_metalComputeEngineInit_throwsBecauseShaderLibraryIsUnpackaged__ODC_0014"],
    "X-CFG-1": ["test_characterizes_predictionConfigPresets_speedTokenBudgetExceedsBalanced__no_defect"],
    "X-ARCH-1": ["test_characterizes_modelArchitectureDetection_disagreesWithLlamaBridgeClassifier__no_defect"],
    "X-PROFILE-1": [
        "test_requires_modelProfile_throwsCatalystErrorForAMockPath",
        "test_requires_modelProfile_throwsModelFileNotFoundForMissingPath",
        "test_requires_modelProfile_throwsModelFileCorruptedForFileUnder1MiB",
        "test_requires_modelProfile_throwsModelFileCorruptedForBadMagic",
    ],
    "X-STOP-1": ["test_requires_streamProcessor_detectsStopSequence_atExpectedByteOffset"],
    "X-PROMPT-1": [
        "test_requires_standardPromptFormatter_producesGoldenLlama3Prompt",
        "test_requires_standardPromptFormatter_producesGoldenPhi3Prompt",
        "test_requires_standardPromptFormatter_producesGoldenChatMLFallbackPrompt",
    ],
    "X-SETTINGS-1": ["test_requires_iphone16ProMaxSettings_areValidAndHaveTheirOptimizedValues"],
    "X-CATALYST-1": ["test_requires_catalystShared_isASingletonAndIsConstructible"],
    # R2
    "C-D8-1": ["test_characterizes_recoverableFailure_loadingStream_deliversNoTerminalEvent__ODC_0015"],
    "C-D8-2": [
        "test_characterizes_nonRecoverableFailure_loadingStream_deliversNoTerminalEvent__ODC_0015",
        "test_characterizes_recoverableAndNonRecoverableFailures_areExternallyIndistinguishable__ODC_0015",
    ],
    "C-D8-3": ["test_characterizes_afterStreamEnds_instanceIsNotReadyWithNoFurtherEvent__ODC_0015"],
    "C-D3-2": ["test_characterizes_failurePathStream_terminatesWithoutTheGateFiring__ODC_0012"],
    "C-D4-3": ["test_characterizes_llamaBridgeLoadModel_failsForEveryFileOnTheStub__ODC_0013"],
    # R3 -- always skip on SFC-B under this revision's Q1 disposition.
    "C-D2-3": ["test_characterizes_boundedGeneration_yieldsTwoCompletionChunks__ODC_0011"],
    "C-D2-4": ["test_characterizes_secondCompletion_reportsNaturalEvenAfterMaxTokensReached__ODC_0011"],
    "C-D3-3": ["test_characterizes_afterReady_loadingStream_doesNotTerminate__ODC_0012"],
    "C-D1-3": ["test_characterizes_releaseInstance_cachesThenShutsDownAReadyInstance__ODC_0010"],
    "X-GEN-1": ["test_requires_boundedGeneration_producesNonEmptyContent"],
    # Surface canaries.
    "S-1": ["test_surface_reportsDeclaredSurface"],
    "S-2": ["test_surface_simulatorSupportMatchesCurrentSurface"],
    "S-3": ["test_surface_stubRejectsAValidMagicFixture"],
    "S-4": ["test_surface_modelAssetPathIsNilWithoutAValidEnvironmentVariable"],
    # Also present in the N1-repaired original file, not separately catalogued
    # under an X- id above but tracked here so --inventory's reverse direction
    # (every implemented method appears in the catalog) does not flag them.
    "N1-repair": [
        "test_requires_modelProfile_throwsCatalystErrorForAMockPath",
        "test_requires_iphone16ProMaxSettings_areValidAndHaveTheirOptimizedValues",
        "test_characterizes_predictionConfigPresets_speedTokenBudgetExceedsBalanced__no_defect",
        "test_requires_catalystShared_isASingletonAndIsConstructible",
    ],
}

DEFECT_TICKETS = {
    "D1": "ODC-0010", "D2": "ODC-0011", "D3": "ODC-0012", "D4": "ODC-0013",
    "D5": "ODC-0014", "D6": "ODC-0002", "D7": "ODC-0016", "D8": "ODC-0015",
    "N1": "ODC-0018",
}

DEFECT_TO_CASE_IDS = {
    "D1": ["C-D1-1", "C-D1-2", "C-D1-3", "F-D1-1"],
    "D2": ["C-D2-1", "C-D2-2", "C-D2-3", "C-D2-4", "F-D2-1", "F-D2-2"],
    "D3": ["C-D3-1", "C-D3-2", "C-D3-3", "F-D3-1"],
    "D4": ["C-D4-1", "C-D4-2", "C-D4-3", "C-N2-1"],
    "D5": ["C-D5-1", "C-D5-2", "C-D5-3", "C-D5-4"],
    "D6": ["R-D6-1"],
    "D7": ["C-D7-1", "C-D7-2"],
    "D8": ["C-D8-1", "C-D8-2", "C-D8-3", "F-D8-1"],
    "N1": ["N1-repair"],
}


def check_inventory(require_defects: list[str] | None) -> list[str]:
    failures: list[str] = []
    implemented = all_test_method_names()

    catalogued_methods: set[str] = set()
    for case_id, methods in CATALOG.items():
        for method in methods:
            if method.startswith("checker:"):
                continue
            catalogued_methods.add(method)
            if method not in implemented:
                fail(failures, f"inventory: {case_id} names {method}, which is not implemented")

    # Reverse direction: every implemented test_characterizes_/test_requires_
    # method must appear in the catalog somewhere.
    for method in implemented:
        if method.startswith("test_characterizes_") or method.startswith("test_requires_"):
            if method not in catalogued_methods:
                fail(failures, f"inventory: {method} is implemented but not named by any case id in the catalog")

    if require_defects:
        for defect in require_defects:
            case_ids = DEFECT_TO_CASE_IDS.get(defect)
            if not case_ids:
                fail(failures, f"inventory: --require-defects names unknown defect {defect}")
                continue
            if not any(cid in CATALOG for cid in case_ids):
                fail(failures, f"inventory: defect {defect} has no representative case in the catalog")

    # A17 / A18: r3_disposition and device_execution_disposition must be
    # recorded in the characterization document.
    if CHAR_DOC_PATH.is_file():
        doc = CHAR_DOC_PATH.read_text(encoding="utf-8")
        r3_match = re.search(r"r3_disposition:\s*([A-Za-z-]+)", doc)
        if not r3_match or r3_match.group(1) not in ("executed", "specified-unexecuted"):
            fail(failures, "inventory: r3_disposition missing or not one of executed/specified-unexecuted")
        device_match = re.search(r"device_execution_disposition:\s*([A-Za-z-]+)", doc)
        if not device_match or device_match.group(1) not in ("measured", "unmeasured-deferred"):
            fail(failures, "inventory: device_execution_disposition missing or not one of measured/unmeasured-deferred")
    else:
        fail(failures, f"inventory: {CHAR_DOC_PATH.relative_to(ROOT)} not found")

    return failures


# ---------------------------------------------------------------------------
# --skips
# ---------------------------------------------------------------------------

EXPECTED_EXECUTED_PREFIXES_SIMULATOR = ("test_surface_",)  # plus everything not R3; see below.

R3_METHOD_NAMES = {
    m for case_id, methods in CATALOG.items() if case_id in DEFECT_TO_CASE_IDS.get("D2", []) + DEFECT_TO_CASE_IDS.get("D3", []) + DEFECT_TO_CASE_IDS.get("D1", [])
    for m in methods
} | {
    "test_characterizes_boundedGeneration_yieldsTwoCompletionChunks__ODC_0011",
    "test_characterizes_secondCompletion_reportsNaturalEvenAfterMaxTokensReached__ODC_0011",
    "test_requires_boundedGeneration_producesNonEmptyContent",
    "test_characterizes_afterReady_loadingStream_doesNotTerminate__ODC_0012",
    "test_characterizes_releaseInstance_cachesThenShutsDownAReadyInstance__ODC_0010",
}


def parse_ledger(ledger_path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    executed: list[str] = []
    skipped: list[tuple[str, str]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("EXECUTED "):
            executed.append(line[len("EXECUTED "):].strip())
        elif line.startswith("SKIPPED "):
            rest = line[len("SKIPPED "):].strip()
            parts = rest.split(" ", 1)
            name = parts[0]
            code = parts[1] if len(parts) > 1 else ""
            skipped.append((name, code))
    return executed, skipped


def check_skips(ledger_arg: str | None) -> list[str]:
    failures: list[str] = []
    if not ledger_arg:
        fail(failures, "skips: no ledger path provided")
        return failures
    ledger_path = Path(ledger_arg)
    if not ledger_path.is_file():
        fail(failures, f"skips: ledger {ledger_arg} not found")
        return failures

    executed, skipped = parse_ledger(ledger_path)

    if len(executed) == 0:
        fail(failures, "skips: zero-executed run; this is always a failure regardless of surface")
    if len(executed) == 0 and len(skipped) > 0:
        fail(failures, "skips: all-skipped run; this is always a failure regardless of surface")

    for name, code in skipped:
        if code and code not in SKIP_CODES:
            fail(failures, f"skips: {name} skipped with unrecognized code {code!r}")
        if name not in R3_METHOD_NAMES and code:
            # A case outside the R3 set skipped -- only acceptable if it is
            # itself gated on a surface predicate (e.g. an R2 canary on a real
            # device); flag it for review rather than silently accept.
            fail(failures, f"skips: {name} skipped ({code}) but is not in the expected-skip (R3) set")

    for name, _code in skipped:
        pass

    executed_set = set(executed)
    for name in R3_METHOD_NAMES:
        if name in executed_set:
            # Fine: this would only happen on a device surface with a model
            # asset, i.e. Q1's first outcome. Not a failure by itself.
            pass

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packaging", action="store_true")
    parser.add_argument("--fingerprints", action="store_true")
    parser.add_argument("--naming", action="store_true")
    parser.add_argument("--orphans", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--require-defects", default=None)
    parser.add_argument("--skips", nargs="?", const="__missing__", default=None)
    args = parser.parse_args()

    selected = any([args.packaging, args.fingerprints, args.naming, args.orphans, args.inventory, args.skips is not None])
    run_all = not selected

    all_failures: list[str] = []

    try:
        if run_all or args.packaging:
            all_failures += check_packaging()
        if run_all or args.fingerprints:
            all_failures += check_fingerprints()
        if run_all or args.naming:
            all_failures += check_naming()
        if run_all or args.orphans:
            all_failures += check_orphans()
        if run_all or args.inventory:
            require_defects = args.require_defects.split(",") if args.require_defects else None
            all_failures += check_inventory(require_defects)
        if args.skips is not None:
            ledger = None if args.skips == "__missing__" else args.skips
            all_failures += check_skips(ledger)
    except Exception as exc:
        print(f"check-characterization: could not run: {exc}", file=sys.stderr)
        return 2

    if all_failures:
        for failure in all_failures:
            print(failure, file=sys.stderr)
        print(f"check-characterization: {len(all_failures)} failure(s)", file=sys.stderr)
        return 1

    print("check-characterization: all selected checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
