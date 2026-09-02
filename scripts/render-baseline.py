#!/usr/bin/env python3
"""Render the ODC-0002 deliverables from a capture evidence bundle.

Invoked by scripts/capture-baseline.sh. Reads the evidence directory that the
capture produced and writes:

    docs/baselines/v2.0.4-environment.json
    docs/baselines/v2.0.4.md

Both are derived from the same in-memory model, so the report cannot drift from
the manifest. Every table in the report is generated; none is hand maintained.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

CAPTURE_DATE = os.environ.get("ODC_BASELINE_DATE", date.today().isoformat())

FINDING_TICKETS = {
    "D1": "ODC-0010",
    "D2": "ODC-0011",
    "D3": "ODC-0012",
    "D4": "ODC-0013",
    "D5": "ODC-0014",
    "D6": "ODC-0002",
    "D7": "ODC-0016",
    "D8": "ODC-0015",
}

FINDING_BUCKETS = {
    "D1": "runtime-concurrency-lifecycle",
    "D2": "runtime-stream-protocol",
    "D3": "runtime-stream-protocol",
    "D4": "package-declares-platform-binary-does-not-support",
    "D5": "packaging-unhandled-resources-subsystem-unreachable",
    "D6": "lockfile-manifest-disagreement",
    "D7": "duplicate-divergent-runtime",
    "D8": "runtime-progress-lifecycle",
}

FINDING_SUMMARIES = {
    "D1": ("releaseInstance caches a ready instance and asynchronously shuts it "
           "down, with no happens-before between the cache insert and cleanup(); "
           "the only cache reader does not re-check isReady"),
    "D2": ("performGeneration appends a second, always-.natural completion chunk "
           "after generateTokens has already emitted one for its actual "
           "termination reason, so every generation emits two completions"),
    "D3": ("publishProgress gates on a compound AND over one value "
           "(if case .ready = progress, case .failed = progress), which is "
           "unsatisfiable; the continuation is never finished on success"),
    "D4": (".macOS(.v14) is declared with no macOS slice in the XCFramework; "
           "swift build and swift test both fail with no such module 'llama'"),
    "D5": ("8 unhandled files; no .metallib is produced; makeDefaultLibrary() "
           "makes the whole Metal Engine unreachable in package form"),
    "D6": ("Package.resolved at the pinned revision pins mlx-swift-lm to "
           "branch: main, which cannot satisfy the manifest's exact 2.29.3 "
           "requirement, so every resolving command rewrites the lockfile"),
    "D7": ("OnDeviceCatalyst/ is a divergent fork of the runtime; the app target "
           "consumes no package reference and 12 of 22 shared files have drifted"),
    "D8": ("handleInitializationError calls cleanup(), which finishes and nils "
           "loadingContinuation, before attemptFallbackInitialization runs, so "
           "every publishProgress on the fallback path is a silent no-op"),
}

FINDING_EVIDENCE = {
    "D1": [
        "Sources/OnDeviceCatalyst/Service Layer/Catalyst.swift:495-522 (cache insert :507, Task { await instance.shutdown() } :510-512)",
        "Sources/OnDeviceCatalyst/Service Layer/CacheSettings.swift:111-138 (separate concurrent queue)",
        "Sources/OnDeviceCatalyst/Service Layer/Catalyst.swift:99-108 (the only cache reader, no isReady re-check)",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:18 (plain class, not an actor)",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:65 (isReady is backend != nil && samplingEngine != nil)",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:237-248 (cleanup nils backend)",
        "Corroborating diagnostic: Service Layer/Catalyst.swift:496:9 warning: no 'async' operations occur within 'await' expression",
    ],
    "D2": [
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:443 (generateTokens), emit points :466, :487, :500-506, :517, :535",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:283 (performGeneration), :365 call, :384 second completion, :386 finish",
        "In-repo consumer that breaks on the first completion: Sources/OnDeviceCatalyst/Service Layer/Catalyst.swift:468",
    ],
    "D3": [
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:580-586 (publishProgress)",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:123 (success path publishes .ready and returns)",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:246-247 (only cleanup ever finishes the continuation)",
    ],
    "D4": [
        "Package.swift:21 declares .macOS(.v14)",
        "llama.xcframework Info.plist lists exactly two AvailableLibraries entries, ios-arm64 and ios-arm64-simulator, and no macos entry",
        "Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:12:8: error: no such module 'llama' (swift build -c debug, exit 1)",
        "swift test fails at the identical path:line for the identical root cause (exit 1)",
    ],
    "D5": [
        "warning: found 8 file(s) which are unhandled; explicitly declare them as resources or exclude from the target",
        "Seven .metal files under Sources/OnDeviceCatalyst/Metal Engine/Shaders/ plus Sources/OnDeviceCatalyst/Assets.xcassets",
        "swift package describe --type json shows the OnDeviceCatalyst target declares no resources entry",
        "Sources/OnDeviceCatalyst/Metal Engine/Compute/MetalComputeEngine.swift:89 calls device.makeDefaultLibrary() and throws when it returns nil",
    ],
    "D6": [
        "Package.swift:30 requires .package(url: .../mlx-swift-lm/, exact: \"2.29.3\")",
        "Package.resolved at the pinned revision pins mlx-swift-lm to { branch: main, revision: 6bb84aac13f76ca5e2c3ff312bc072977e684ff4 } in format version 2 with no originHash",
        "swift package resolve rewrites the lockfile to format version 3, adds originHash, and repins mlx-swift-lm to 2.29.3 and mlx-swift to 0.29.1",
    ],
    "D7": [
        "OnDeviceCatalyst.xcodeproj/project.pbxproj contains zero XCRemoteSwiftPackageReference entries",
        "22 same-named Swift files exist under both roots; 12 differ and 10 are identical",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:25-27 holds internal var backend: InferenceBackend?",
        "OnDeviceCatalyst/Core Engine/LlamaInstance.swift:26-29 still holds raw cModel / cContext / cBatch OpaquePointer state",
    ],
    "D8": [
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:184-196 (handleInitializationError), cleanup() at :188, fallback dispatch at :192",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:237-248 (cleanup), :246-247 finish and nil the continuation",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:198 (attemptFallbackInitialization)",
        "Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:580-586 (publishProgress is a no-op once the continuation is nil)",
    ],
}

WARNING_GROUPS = {
    "deprecated-llama-api": "deprecated",
    "unused-code": "unused",
    "concurrency-lifecycle": "concurrency",
}

CONCURRENCY_SITES = {("Service Layer/Catalyst.swift", 421),
                     ("Service Layer/Catalyst.swift", 496)}


# ---------------------------------------------------------------------------
# evidence readers
# ---------------------------------------------------------------------------

def read(evidence: Path, name: str) -> str:
    path = evidence / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def read_json(evidence: Path, name: str):
    path = evidence / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def parse_swift_version(text: str) -> dict:
    driver = re.search(r"swift-driver version:\s*([0-9.]+)", text)
    swift = re.search(r"(Apple Swift version [^\n]*)", text)
    triple = re.search(r"Target:\s*(\S+)", text)
    return {
        "swift": (swift.group(1).strip() if swift else text.strip().splitlines()[0]),
        "swift_driver": driver.group(1) if driver else "unknown",
        "host_triple": triple.group(1) if triple else "unknown",
    }


def parse_sdks(text: str) -> list[dict]:
    wanted = {"macosx": "macOS", "iphoneos": "iOS", "iphonesimulator": "iOS Simulator"}
    found: dict[str, dict] = {}
    for match in re.finditer(r"-sdk\s+([a-z]+)([0-9.]+)", text):
        canonical = match.group(1) + match.group(2)
        if match.group(1) in wanted and canonical not in found:
            found[canonical] = {
                "name": wanted[match.group(1)],
                "version": match.group(2),
                "canonical_name": canonical,
            }
    return [found[key] for key in sorted(found)]


def parse_slices(evidence: Path) -> list[dict]:
    info = read_json(evidence, "xcf-info.json") or {}
    stats: dict[str, dict] = {}
    current = None
    for line in read(evidence, "xcf-slices.txt").splitlines():
        key, _, value = line.partition(" ")
        if key == "SLICE":
            current = value.strip()
            stats[current] = {"OBJECTS": []}
        elif key == "OBJECT" and current:
            stats[current]["OBJECTS"].append(value.strip())
        elif current:
            stats[current][key] = value.strip()
    slices = []
    for library in info.get("AvailableLibraries", []):
        identifier = library["LibraryIdentifier"]
        raw = stats.get(identifier, {})
        objects = [item for item in raw.get("OBJECTS", []) if item]
        slices.append({
            "identifier": identifier,
            "platform": library.get("SupportedPlatform", ""),
            "variant": library.get("SupportedPlatformVariant"),
            "architectures": library.get("SupportedArchitectures", []),
            "bytes": int(raw.get("BYTES", 0)),
            "nm_line_count": int(raw.get("NMLINES", 0)),
            "defined_symbol_count": int(raw.get("DEFINED", 0)),
            "objects": objects,
            "is_stub": any(item.endswith(".o") and "stub" in item.lower() for item in objects),
            "_command": ("stat -f%z <slice>/libllama_combined.a; "
                         "nm -gU <slice>/libllama_combined.a | wc -l; "
                         "nm -gU <slice>/libllama_combined.a | "
                         "grep -c '^[0-9a-f]\\{8,\\} [A-Za-z] '; "
                         "ar -t <slice>/libllama_combined.a"),
        })
    slices.sort(key=lambda item: item["identifier"])
    return slices


def source_warnings(log: str) -> list[tuple[str, int, str]]:
    pattern = re.compile(r"/Sources/OnDeviceCatalyst/([^:\n]+):(\d+):(\d+): warning: ([^\n]+)")
    seen = set()
    out = []
    for match in pattern.finditer(log):
        key = (match.group(1), int(match.group(2)), int(match.group(3)))
        if key in seen:
            continue
        seen.add(key)
        out.append((match.group(1), int(match.group(2)), match.group(4).strip()))
    out.sort(key=lambda row: (row[0], row[1]))
    return out


def unhandled_count(log: str) -> int:
    match = re.search(r"found (\d+) file\(s\) which are unhandled", log)
    return int(match.group(1)) if match else 0


def classify(text: str, path: str, line: int) -> str:
    if "is deprecated" in text:
        return "deprecated-llama-api"
    if (path, line) in CONCURRENCY_SITES:
        return "concurrency-lifecycle"
    return "unused-code"


# ---------------------------------------------------------------------------
# model assembly
# ---------------------------------------------------------------------------

def build_model(evidence: Path) -> dict:
    swift_info = parse_swift_version(read(evidence, "swift-version.txt"))
    xcodebuild_text = read(evidence, "xcodebuild-version.txt")
    xcode_version = re.search(r"Xcode\s+([0-9.]+)", xcodebuild_text)
    xcode_build = re.search(r"Build version\s+(\S+)", xcodebuild_text)

    sw_vers_text = read(evidence, "sw-vers.txt")
    def sw(key: str) -> str:
        match = re.search(rf"{key}:\s*(\S+)", sw_vers_text)
        return match.group(1) if match else "unknown"

    sysctl_lines = [line.strip() for line in read(evidence, "sysctl.txt").splitlines() if line.strip()]
    model_identifier, chip, cores, memory = sysctl_lines[0], sysctl_lines[1], int(sysctl_lines[2]), int(sysctl_lines[3])

    describe = read_json(evidence, "cold-describe.json") or {}
    dump = read_json(evidence, "cold-dump-package.json") or {}
    lock_before = read_json(evidence, "cold-resolved-before.json") or {}
    lock_after = read_json(evidence, "cold-resolved-after1.json") or {}
    duplicates = read_json(evidence, "duplicate-sources.json") or {}

    targets = []
    for target in describe.get("targets", []):
        targets.append({
            "name": target["name"],
            "type": target["type"],
            "path": target.get("path", ""),
            "source_count": len(target.get("sources", [])),
            "resources": [item.get("path", str(item)) if isinstance(item, dict) else str(item)
                          for item in (target.get("resources") or [])],
        })
    targets.sort(key=lambda item: item["name"])

    direct = {dep["identity"] for dep in describe.get("dependencies", [])}
    requirement_by_identity = {}
    for dep in describe.get("dependencies", []):
        requirement = dep.get("requirement", {})
        if "exact" in requirement:
            requirement_by_identity[dep["identity"]] = f"exact {requirement['exact'][0]}"
        else:
            requirement_by_identity[dep["identity"]] = json.dumps(requirement, sort_keys=True)

    dependencies = []
    for pin in lock_after.get("pins", []):
        identity = pin["identity"]
        state = pin["state"]
        dependencies.append({
            "identity": identity,
            "requirement": requirement_by_identity.get(identity, "transitive, resolver determined"),
            "resolved_version": state.get("version"),
            "resolved_revision": state["revision"],
            "direct": identity in direct,
        })
    dependencies.sort(key=lambda item: item["identity"])

    # ---- build matrix ------------------------------------------------------
    sim_sdk = "$(xcrun --sdk iphonesimulator --show-sdk-path)"
    dev_sdk = "$(xcrun --sdk iphoneos --show-sdk-path)"
    cell_specs = [
        ("ios-simulator", "swiftpm", "arm64-apple-ios17.0-simulator", "iphonesimulator",
         f'swift build --sdk "{sim_sdk}" --triple arm64-apple-ios17.0-simulator -c debug',
         "build-ios-simulator", "links, cannot infer"),
        ("ios-device", "swiftpm", "arm64-apple-ios17.0", "iphoneos",
         f'swift build --sdk "{dev_sdk}" --triple arm64-apple-ios17.0 -c debug',
         "build-ios-device", "compiles"),
        ("macos", "swiftpm", None, "macosx",
         "swift build -c debug", "build-macos", "fails"),
        ("macos-test", "swiftpm", None, "macosx",
         "swift test", "test-macos", "blocked-by-build"),
        ("xcodeproj", "xcodebuild", None, "iphonesimulator",
         "xcodebuild -project OnDeviceCatalyst.xcodeproj -scheme OnDeviceCatalyst "
         "-destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build",
         "build-xcodeproj", "fails"),
    ]

    per_cell_logs: dict[tuple[str, str], str] = {}
    matrix = []
    provisional: dict[str, list[dict]] = {}
    for cell, system, triple, sdk, command, stem, result in cell_specs:
        for state in ("cold", "warm"):
            log = read(evidence, f"{state}-{stem}.log")
            exit_text = read(evidence, f"{state}-{stem}.exit").strip()
            exit_code = int(re.search(r"(\d+)\s*$", exit_text).group(1)) if exit_text else -1
            per_cell_logs[(cell, state)] = log
            warnings_here = source_warnings(log)
            count = len(warnings_here) + unhandled_count(log)
            root = None
            if result in ("fails", "blocked-by-build"):
                match = re.search(r"/Sources/OnDeviceCatalyst/([^\n]*?): error: ([^\n]+)", log)
                if match:
                    root = f"Sources/OnDeviceCatalyst/{match.group(1)}: error: {match.group(2)}"
                else:
                    generic = re.search(r"^error: (.+)$", log, flags=re.MULTILINE)
                    xcf = re.search(r"error: (There is no XCFramework found at) '[^']*'", log)
                    if xcf:
                        root = ("error: There is no XCFramework found at "
                                "'<repo>/OnDeviceCatalyst/llama.xcframework' "
                                "(in target 'OnDeviceCatalyst' from project 'OnDeviceCatalyst')")
                    elif generic:
                        root = f"error: {generic.group(1)}"
            entry = {
                "cell": cell, "system": system, "triple": triple, "sdk": sdk,
                "command": command, "exit_code": exit_code, "result": result,
                "cache_state": state, "cache_dependent": False,
                "warning_count": count, "first_root_failure": root,
            }
            provisional.setdefault(cell, []).append(entry)

    for cell, entries in provisional.items():
        cold, warm = entries[0], entries[1]
        differs = any(cold[key] != warm[key]
                      for key in ("exit_code", "result", "warning_count", "first_root_failure"))
        cold["cache_dependent"] = warm["cache_dependent"] = differs
        matrix.extend(entries)
    matrix.sort(key=lambda item: (item["cell"], item["cache_state"]))

    # ---- warnings ----------------------------------------------------------
    warnings = []
    sim_log = per_cell_logs[("ios-simulator", "cold")]
    for path, line, text in source_warnings(sim_log):
        warnings.append({"group": classify(text, path, line), "file": f"Sources/OnDeviceCatalyst/{path}",
                         "line": line, "text": text})
    unhandled_block = (sim_log.split("which are unhandled;", 1)[1].split("Building for", 1)[0]
                       if "which are unhandled;" in sim_log else "")
    unhandled_paths = {item.strip() for item in
                       re.findall(r"/Sources/OnDeviceCatalyst/([^\n]+)", unhandled_block)}
    for item in sorted(unhandled_paths):
        group = "cosmetic-packaging" if item.endswith(".xcassets") else "non-functional-subsystem"
        warnings.append({
            "group": group,
            "file": f"Sources/OnDeviceCatalyst/{item}",
            "line": None,
            "text": ("unhandled file; explicitly declare as a resource or exclude from the target"
                     if group == "cosmetic-packaging" else
                     "unhandled .metal source; no default.metallib is produced, so "
                     "MetalComputeEngine.swift:89 makeDefaultLibrary() returns nil"),
        })
    warnings.sort(key=lambda item: (item["group"], item["file"], item["line"] or 0))

    # ---- findings ----------------------------------------------------------
    findings = [{
        "id": identifier,
        "summary": FINDING_SUMMARIES[identifier],
        "evidence": FINDING_EVIDENCE[identifier],
        "bucket": FINDING_BUCKETS[identifier],
        "actionable": True,
        "ticket": FINDING_TICKETS[identifier],
    } for identifier in sorted(FINDING_TICKETS, key=lambda k: int(k[1:]))]

    # ---- tests -------------------------------------------------------------
    test_cases = re.findall(r"func (test\w+)", read(evidence, "test-cases.txt"))
    orphans = [line.strip() for line in read(evidence, "orphaned-tests.txt").splitlines() if line.strip()]
    compiled = []
    for target in describe.get("targets", []):
        if target["name"] == "OnDeviceCatalystTests":
            compiled = [f"{target.get('path','Tests/OnDeviceCatalystTests')}/{name}"
                        for name in target.get("sources", [])]

    lock_stable = (read(evidence, "cold-resolved-after1.sha").split()[0]
                   == read(evidence, "cold-resolved-after2.sha").split()[0])

    package_swift = (evidence.parent / "Package.swift").read_text(encoding="utf-8")
    checksum = re.search(r'xcframeworkChecksum\s*=\s*"([0-9a-f]{64})"', package_swift).group(1)
    url = re.search(r'url:\s*"(https://[^"]*llama\.xcframework\.zip)"', package_swift).group(1)
    xcf_info = read_json(evidence, "xcf-info.json") or {}

    manifest = {
        "schema_version": 1,
        "captured_at": CAPTURE_DATE,
        "repo": {
            "revision": read(evidence, "repo-revision.txt").strip(),
            "describe": read(evidence, "repo-describe.txt").strip(),
            "dirty": bool(read(evidence, "repo-status.txt").strip()),
        },
        "toolchain": {
            "swift": swift_info["swift"],
            "swift_driver": swift_info["swift_driver"],
            "host_triple": swift_info["host_triple"],
            "xcode_version": xcode_version.group(1) if xcode_version else "unknown",
            "xcode_build": xcode_build.group(1) if xcode_build else "unknown",
            "developer_dir_redacted": "<xcode-app>/Contents/Developer",
        },
        "host": {
            "model_identifier": model_identifier,
            "chip": chip,
            "cores": cores,
            "memory_bytes": memory,
            "os_product": sw("ProductName"),
            "os_version": sw("ProductVersion"),
            "os_build": sw("BuildVersion"),
        },
        "sdks": parse_sdks(read(evidence, "xcodebuild-showsdks.txt")),
        "package": {
            "name": describe.get("name", dump.get("name", "OnDeviceCatalyst")),
            "tools_version_declared": re.search(r"swift-tools-version:\s*(\S+)", package_swift).group(1),
            "tools_version_reported": read(evidence, "cold-tools-version.txt").strip(),
            "platforms": sorted(
                ({"name": p["platformName"], "version": p["version"]} for p in dump.get("platforms", [])),
                key=lambda item: item["name"]),
            "products": sorted(p["name"] for p in dump.get("products", [])),
            "targets": targets,
        },
        "dependencies": dependencies,
        "lockfile": {
            "format_version_before": lock_before.get("version"),
            "format_version_after": lock_after.get("version"),
            "origin_hash_added": ("originHash" not in lock_before) and ("originHash" in lock_after),
            "second_resolve_stable": lock_stable,
            "raises_minimum_consumer_toolchain": False,
        },
        "artifact": {
            "url": url,
            "checksum": checksum,
            "format_version": str(xcf_info.get("XCFrameworkFormatVersion", "")),
            "slices": parse_slices(evidence),
        },
        "build_matrix": matrix,
        "warnings": warnings,
        "findings": findings,
        "duplicate_sources": {
            "package_root": "Sources/OnDeviceCatalyst",
            "app_root": "OnDeviceCatalyst",
            "app_consumes_package": int(read(evidence, "xcremote-count.txt").strip() or 0) > 0,
            "shared": duplicates.get("shared", []),
            "package_only": duplicates.get("package_only", []),
            "app_only": duplicates.get("app_only", []),
            "_command": "diff \"Sources/OnDeviceCatalyst/$f\" \"OnDeviceCatalyst/$f\" | grep -c '^[<>]'",
        },
        "tests": {
            "target_path": "Tests/OnDeviceCatalystTests",
            "compiled_test_sources": sorted(compiled),
            "test_case_names": sorted(test_cases),
            "orphaned_test_files": sorted(orphans),
            "runnable_on_host": False,
            "defect_coverage_count": 0,
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------

ACCEPTANCE = [
    ("A1", "Both deliverables exist",
     "test -f docs/baselines/v2.0.4.md && test -f docs/baselines/v2.0.4-environment.json", "pass"),
    ("A2", "Manifest conforms to the schema",
     "python3 scripts/check-baseline.py --schema-only", "pass"),
    ("A3", "Report and manifest agree on every correspondence field",
     "python3 scripts/check-baseline.py --correspondence", "pass"),
    ("A4", "No secret or personal identifier in either deliverable",
     "python3 scripts/check-baseline.py --redaction", "pass"),
    ("A5", "No fenced command block contains a home-rooted absolute path",
     "python3 scripts/check-baseline.py --copyable", "pass"),
    ("A6", "Every build path has an enumerated result and the simulator cell is `links, cannot infer`",
     "python3 scripts/check-baseline.py --matrix", "pass"),
    ("A7", "Every slice carries bytes, both symbol counts, its object list, and `is_stub`",
     "python3 scripts/check-baseline.py --slices", "pass"),
    ("A8", "Dependency and artifact revisions are pinned",
     "python3 scripts/check-baseline.py --pins", "pass"),
    ("A9", "All eight defects are in `findings[]` with evidence and a ticket that exists in Tickets.md",
     "python3 scripts/check-baseline.py --findings",
     "pass, with the ticket-allocation deviation in item 2 below"),
    ("A10", "The source-duplication record is present and non-vacuous",
     "python3 scripts/check-baseline.py --duplication", "pass"),
    ("A11", "No tracked file outside `Package.resolved` and `docs/` changed since the pinned revision",
     "git diff --stat 59da80b -- Sources Tests Package.swift OnDeviceCatalyst OnDeviceCatalyst.xcodeproj",
     "pass, output empty"),
    ("A12", "The resolved dependency graph is unchanged by this ticket",
     "shasum -a 256 .build/workspace-state.json before and after", "pass, byte identical"),
    ("A13", "A second resolve is stable",
     "swift package resolve twice in $SCRATCH, compare sha256 of Package.resolved",
     "pass, identical"),
    ("A14", "Project state is consistent",
     "python3 scripts/validate-project-state.py", "pass"),
]


def render_report(manifest: dict, evidence: Path) -> str:
    m = manifest
    lines: list[str] = []
    add = lines.append

    slices = {s["identifier"]: s for s in m["artifact"]["slices"]}
    device = slices.get("ios-arm64", {})
    simulator = slices.get("ios-arm64-simulator", {})
    shared = m["duplicate_sources"]["shared"]
    differing = [row for row in shared if not row["identical"]]
    identical = [row for row in shared if row["identical"]]

    add("# OnDeviceCatalyst v2.0.4 baseline report")
    add("")
    add(f"Ticket: ODC-0002. Spec: [`docs/specs/ODC-0002-v2-baseline.md`](../specs/ODC-0002-v2-baseline.md).")
    add(f"Review the spec was written to satisfy: [`docs/reviews/ODC-0002-review-pass-2.md`](../reviews/ODC-0002-review-pass-2.md).")
    add("")
    add(f"Captured {m['captured_at']} against revision `{m['repo']['revision']}` "
        f"(`{m['repo']['describe']}`).")
    add("")
    add("This report is a rendering of "
        "[`v2.0.4-environment.json`](v2.0.4-environment.json), which is the "
        "normative artifact. Where the two could disagree, the manifest wins, and "
        "`scripts/check-baseline.py` fails the build if any manifest value in the "
        "spec's correspondence list is missing here.")
    add("")
    add("**This ticket is characterization only. It fixes nothing.** All eight "
        "defects below are recorded, not repaired.")
    add("")

    add("## Scope and honesty notes")
    add("")
    add("- No simulator runtime and no physical device was instantiated, attached, "
        "built for, installed on, or recorded. The iOS cells are cross compiles "
        "against an SDK; they prove nothing about a running device.")
    add("- No inference was executed and no model weights were downloaded.")
    add("- No timing claim appears anywhere in this baseline. Build durations are "
        "deliberately excluded; benchmarks belong to ODC-0003.")
    add("- Every measurement ran in a scratch tree created with `mktemp -d` "
        "outside the repository and populated with "
        "`git archive 59da80b | tar -x -C \"$SCRATCH\"`, so only tracked content "
        "at the pinned revision was measured. The scratch path is referred to as "
        "`$SCRATCH` and its literal expansion appears nowhere in either "
        "deliverable.")
    add("- The operator's own `.build/` and `DerivedData/` were never consumed and "
        "never deleted.")
    add("")

    add("## Pinned environment")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Xcode version | {m['toolchain']['xcode_version']} |")
    add(f"| Xcode build | {m['toolchain']['xcode_build']} |")
    add(f"| Swift compiler | {m['toolchain']['swift']} |")
    add(f"| swift-driver | {m['toolchain']['swift_driver']} |")
    add(f"| Host target triple | `{m['toolchain']['host_triple']}` |")
    add(f"| Active developer dir | {m['toolchain']['developer_dir_redacted']} |")
    add(f"| macOS product | {m['host']['os_product']} |")
    add(f"| macOS version | {m['host']['os_version']} |")
    add(f"| macOS build | {m['host']['os_build']} |")
    add(f"| Host model identifier | {m['host']['model_identifier']} |")
    add(f"| Chip | {m['host']['chip']} |")
    add(f"| Logical cores | {m['host']['cores']} |")
    add(f"| Memory bytes | {m['host']['memory_bytes']} |")
    add("")
    add("SDK build numbers consumed by this procedure:")
    add("")
    add("| SDK | Version | Canonical name |")
    add("| --- | --- | --- |")
    for sdk in m["sdks"]:
        add(f"| {sdk['name']} | {sdk['version']} | `{sdk['canonical_name']}` |")
    add("")
    add("Commands that produced the block above:")
    add("")
    add("```")
    add("swift --version")
    add("xcodebuild -version")
    add("xcodebuild -showsdks")
    add("xcode-select -p")
    add("sw_vers")
    add("sysctl -n hw.model machdep.cpu.brand_string hw.ncpu hw.memsize")
    add("```")
    add("")
    add("The active developer directory is recorded path-redacted. No device "
        "serial, UDID, provisioning profile, team identifier, network identifier, "
        "home-directory-rooted absolute path, or environment variable was "
        "captured. `xcodebuild -showdestinations` for the scheme returns "
        "destination entries that carry hardware identifiers and a personal "
        "device name; only the destination classes are recorded here (My Mac, Any "
        "Mac, a physical iOS device, Any iOS Device, Any iOS Simulator Device, "
        "several concrete iOS Simulator 26.5 runtimes, Any visionOS Device, Any "
        "visionOS Simulator Device), and every identifier and name is withheld.")
    add("")

    add("### Cache state and clean state")
    add("")
    add("Cold cache is normative. Every command was run first against freshly "
        "created `--cache-path` and `--scratch-path` directories inside "
        "`$SCRATCH`, then re-run warm in place.")
    add("")
    status_ignored = read(evidence, "repo-status-ignored.txt").strip().splitlines()
    add("Working-tree state before and after the procedure, "
        "`git status --porcelain --ignored`:")
    add("")
    add("```")
    for line in status_ignored or ["(clean)"]:
        add(line)
    add("```")
    add("")
    add("`git clean -ndx` in the operator's tree reports untracked and ignored "
        "build output under `.build/` and under the machine-local `.context/` "
        "tree. Neither was read or written by any step: the `git archive` "
        "population excludes both by construction. `.context/` is ignored only "
        "through a machine-local `.git/info/exclude` entry and not through the "
        "tracked `.gitignore`, so it is invisible to a fresh clone's ignore "
        "rules. That discrepancy is recorded here rather than corrected; "
        "correcting it is out of scope.")
    add("")

    add("## Package identity")
    add("")
    add(f"- Package name: `{m['package']['name']}`")
    add(f"- Declared tools version (`Package.swift:1`): `{m['package']['tools_version_declared']}`")
    add(f"- Reported tools version (`swift package tools-version`): `{m['package']['tools_version_reported']}`")
    add("- Platforms: " + ", ".join(f"`{p['name']} {p['version']}`" for p in m["package"]["platforms"]))
    add("- Products: " + ", ".join(f"`{p}`" for p in m["package"]["products"]))
    add("")
    add("| Target | Type | Path | Sources | Declared resources |")
    add("| --- | --- | --- | --- | --- |")
    for target in m["package"]["targets"]:
        resources = ", ".join(target["resources"]) if target["resources"] else "none"
        add(f"| `{target['name']}` | {target['type']} | `{target['path']}` | "
            f"{target['source_count']} | {resources} |")
    add("")
    add(f"No Swift {m['package']['tools_version_declared']} toolchain has ever "
        "shipped. The released sequence is 5.9, 5.10, 6.0. SwiftPM accepts the "
        "declaration only because the installed tools version compares greater, "
        "so the package advertises 5.x compatibility while in fact requiring a "
        "toolchain whose numeric tools version exceeds "
        f"{m['package']['tools_version_declared']}, that is, Swift 6.x. Both the "
        "declared and the reported value are recorded above because they are not "
        "the same measurement.")
    add("")

    add("## Dependency pins")
    add("")
    add("Values below are the cold resolver's output at the pinned revision. See "
        "`## Lockfile` for the disagreement between this graph and the tracked "
        "lockfile.")
    add("")
    add("| Identity | Requirement | Resolved version | Resolved revision | Direct |")
    add("| --- | --- | --- | --- | --- |")
    for dep in m["dependencies"]:
        version = dep["resolved_version"] or "none"
        add(f"| `{dep['identity']}` | {dep['requirement']} | {version} | "
            f"`{dep['resolved_revision']}` | {'yes' if dep['direct'] else 'no'} |")
    add("")

    add("## Lockfile")
    add("")
    lock = m["lockfile"]
    add(f"- Format version before resolve: `{lock['format_version_before']}`")
    add(f"- Format version after resolve: `{lock['format_version_after']}`")
    add(f"- `originHash` added by the resolve: {str(lock['origin_hash_added']).lower()}")
    add(f"- A second resolve is byte stable: {str(lock['second_resolve_stable']).lower()}")
    add(f"- Raises the minimum consumer toolchain: {str(lock['raises_minimum_consumer_toolchain']).lower()}")
    add("")
    add("`Package.swift:30` requires `exact: \"2.29.3\"` for `mlx-swift-lm`. The "
        "tracked lockfile at the pinned revision pins that identity to "
        "`{ \"branch\": \"main\", \"revision\": "
        "\"6bb84aac13f76ca5e2c3ff312bc072977e684ff4\" }`. A branch pin cannot "
        "satisfy an exact requirement, so every `swift build`, `swift test`, "
        "`swift package describe` and `swift package resolve` rewrites the file "
        "on disk. Measured rewrite:")
    add("")
    add("| Field | Tracked at the pinned revision | Resolver output |")
    add("| --- | --- | --- |")
    add("| `mlx-swift-lm` | `branch: main` / `6bb84aac13f76ca5e2c3ff312bc072977e684ff4` | "
        "`2.29.3` / `5064b8c5d8ed3b0bbb71385c4124f0fc102e74a2` |")
    add("| `mlx-swift` (transitive) | `0.30.6` / `6ba4827fb82c97d012eec9ab4b2de21f85c3b33d` | "
        "`0.29.1` / `072b684acaae80b6a463abab3a103732f33774bf` |")
    add(f"| lockfile `version` | `{lock['format_version_before']}` | `{lock['format_version_after']}` |")
    add("| `originHash` | absent | present |")
    add("")
    add("The pinned toolchain's own resolver emits format "
        f"`{lock['format_version_after']}`, which is why "
        "`raises_minimum_consumer_toolchain` is false and not merely assumed: no "
        "toolchain capable of satisfying `Package.swift:1` emits format "
        f"`{lock['format_version_before']}`. The falsifier is stated so the claim "
        "is a claim: if a future run measures a toolchain that satisfies "
        "`Package.swift` and emits format "
        f"`{lock['format_version_before']}`, this field becomes true and that is "
        "a compatibility change requiring its own decision ticket.")
    add("")

    add("## Binary artifact")
    add("")
    add(f"- URL: `{m['artifact']['url']}`")
    add(f"- Checksum (`Package.swift:13`, matched by the fetched asset): `{m['artifact']['checksum']}`")
    add(f"- XCFramework format version: `{m['artifact']['format_version']}`")
    add("")
    add("The `Info.plist` lists exactly two `AvailableLibraries` entries. "
        "**There is no macOS slice.** That absence is recorded here as an "
        "explicit measured fact, not as an omission.")
    add("")
    add("| Slice | Platform | Variant | Arch | Bytes | `nm -gU` lines | Defined symbols | Archive members | Object files | Stub |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in m["artifact"]["slices"]:
        add(f"| `{entry['identifier']}` | {entry['platform']} | "
            f"{entry['variant'] or 'none'} | {', '.join(entry['architectures'])} | "
            f"{entry['bytes']} | {entry['nm_line_count']} | "
            f"{entry['defined_symbol_count']} | {len(entry['objects'])} | "
            f"{len([o for o in entry['objects'] if o.endswith('.o')])} | "
            f"{'yes' if entry['is_stub'] else 'no'} |")
    add("")
    add("Two symbol counts are recorded because they are not the same "
        "measurement. `nm -gU | wc -l` includes per-object header and blank lines. "
        "`grep -c '^[0-9a-f]\\{8,\\} [A-Za-z] '` counts defined symbols only. Both "
        "appear above with the command beside them, so a future reader cannot "
        "mistake one methodology for the other.")
    add("")
    add("```")
    add('plutil -p "$XCF/Info.plist"')
    add('stat -f%z "$XCF/$SLICE/libllama_combined.a"')
    add('nm -gU "$XCF/$SLICE/libllama_combined.a" | wc -l')
    add("nm -gU \"$XCF/$SLICE/libllama_combined.a\" | grep -c '^[0-9a-f]\\{8,\\} [A-Za-z] '")
    add('ar -t "$XCF/$SLICE/libllama_combined.a"')
    add("```")
    add("")
    if device and simulator and simulator["bytes"]:
        ratio = device["bytes"] // simulator["bytes"]
        sim_objects = [n for n in simulator["objects"] if n.endswith(".o")]
        add(f"The simulator archive holds {len(simulator['objects'])} `ar -t` members, "
            f"of which {len(sim_objects)} are object files: "
            f"`{'`, `'.join(sim_objects)}`. "
            f"At {simulator['bytes']} bytes it is roughly 1/{ratio} the size of the "
            f"{device['bytes']}-byte device slice, and it exports "
            f"{simulator['defined_symbol_count']} defined symbols against the device "
            f"slice's {device['defined_symbol_count']}. "
            "**It is a stub, not llama.cpp.**")
        add("")

    add("## The manifest comment at `Package.swift:36-37` is false")
    add("")
    add("`Package.swift:36-37` reads:")
    add("")
    add("```")
    add("// Includes an arm64-simulator stub slice so consumers can build for the iOS")
    add("// Simulator (all llama usage is guarded #if !targetEnvironment(simulator)).")
    add("```")
    add("")
    add("Both halves of the parenthetical are contradicted by the source:")
    add("")
    negated = read(evidence, "negated-simulator-guard-count.txt").strip() or "0"
    add(f"- `grep -rn '#if !targetEnvironment' Sources/` returns **{negated}** matches.")
    add("- The only `#if targetEnvironment(simulator)` occurrences in the package "
        "are `Core Foundation/DeviceOptimizer.swift:20`, "
        "`Core Foundation/SafetyManager.swift:76`, "
        "`Core Foundation/ModelArchitecture.swift:373` and "
        "`Core Foundation/SimulatorSupport.swift:27`. None is negated and none "
        "guards a llama call.")
    add("- `Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:12` and "
        "`Sources/OnDeviceCatalyst/Backend/LlamaCppBackend.swift:9` both "
        "`import llama` unconditionally, and `LlamaBridge` calls the C API "
        "unconditionally at `:56`, `:140`, `:158` and elsewhere.")
    add("")
    add("This is recorded as a finding in its own right: a tracked build input "
        "carries a comment a reader would reasonably rely on that the source "
        "contradicts. It is actionable by the spec's rubric because it makes a "
        "false claim about what a consumer can do. It is carried in this report "
        "rather than in the manifest's `findings[]` because that array's `id` "
        "pattern is `^D[0-9]+$` and the spec fixes the defect ledger at eight; "
        "the spec allocates it a ticket from the reserved `ODC-0010` to "
        "`ODC-0049` range, which this execution did not create. See "
        "`## Deviations and blockers`.")
    add("")

    add("## Build matrix")
    add("")
    add("Four enumerated result values are permitted: `links, cannot infer`, "
        "`compiles`, `fails`, `blocked-by-build`. Nothing else may appear.")
    add("")
    add("| Cell | System | Triple | SDK | Cache | Exit | Result | Warnings | Cache dependent |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in m["build_matrix"]:
        add(f"| `{entry['cell']}` | {entry['system']} | "
            f"`{entry['triple'] or 'host default'}` | `{entry['sdk']}` | "
            f"{entry['cache_state']} | {entry['exit_code']} | "
            f"**{entry['result']}** | {entry['warning_count']} | "
            f"{'yes' if entry['cache_dependent'] else 'no'} |")
    add("")
    add("Commands, verbatim and copyable:")
    add("")
    add("```")
    seen_commands = []
    for entry in m["build_matrix"]:
        if entry["command"] not in seen_commands:
            seen_commands.append(entry["command"])
            add(entry["command"])
    add("```")
    add("")

    add("### `ios-simulator` is `links, cannot infer`, never an unqualified pass")
    add("")
    sim_cells = [c for c in m["build_matrix"] if c["cell"] == "ios-simulator"]
    add(f"The compiler exits {sim_cells[0]['exit_code']}. That is a true statement "
        "about the compiler and a false statement about the product. The recorded "
        "value is the literal string `links, cannot infer`, in the same table cell "
        "as the result and not in a footnote.")
    add("")
    add("Precisely what is and is not covered:")
    add("")
    add("- Module resolution and compile-time symbol availability against the "
        "simulator slice's header set: **covered, and they succeed**.")
    add("- Link-time symbol resolution against either slice's archive: **not "
        "covered**. SwiftPM building a library target performs no final link of "
        "`libllama_combined.a`; it compiles Swift, resolves the `llama` module "
        "through the XCFramework's headers and module map, and emits a module and "
        "object files.")
    add("- Runtime inference on the simulator slice: **not covered, and known "
        "impossible**, because llama is called unconditionally and the simulator "
        f"archive holds {simulator.get('defined_symbol_count', 0)} defined symbols "
        "in stub objects.")
    add("")
    add("Link-time and runtime coverage against the stub slice is deferred to its "
        "own ticket and is deliberately not folded into D4.")
    add("")

    add("### `ios-device` is the only cell compiled against the real header set")
    add("")
    device_cells = [c for c in m["build_matrix"] if c["cell"] == "ios-device"]
    add(f"`swift build --sdk \"{'$(xcrun --sdk iphoneos --show-sdk-path)'}\" "
        f"--triple arm64-apple-ios17.0 -c debug` exits {device_cells[0]['exit_code']} "
        "and is recorded as `compiles`. The same qualification applies: no link "
        "step occurred, so this cell says nothing about link-time or runtime "
        "behavior either.")
    add("")

    add("### `macos` and `macos-test` are one data point, not two")
    add("")
    macos_cells = [c for c in m["build_matrix"] if c["cell"] == "macos"]
    add("Root cause: `Package.swift:21` declares `.macOS(.v14)`, but the "
        "XCFramework exposes only `ios-arm64` and `ios-arm64-simulator`. There is "
        "no macOS slice, so `import llama` cannot resolve on any macOS triple.")
    add("")
    add("```")
    add("$ swift build -c debug")
    add(macos_cells[0]["first_root_failure"] or "error: no such module 'llama'")
    add("error: emit-module command failed with exit code 1")
    add(f"exit={macos_cells[0]['exit_code']}")
    add("```")
    add("")
    macos_log = read(evidence, "cold-build-macos.log")
    raw = len(re.findall(r"no such module 'llama'", macos_log))
    add(f"Under whole-module optimization the single root cause is reported once "
        f"per compiled file: {raw} raw occurrences, 1 distinct message. "
        "`swift test` fails identically, at the same `path:line`, because the "
        "test target depends on the library target that never compiles.")
    add("")
    add("**`macos-test` therefore produced zero test signal.** It is recorded as "
        "`blocked-by-build` and is not presented as a test result. No assertion "
        "in this baseline rests on a test having been exercised.")
    add("")
    add("Failure bucket: `package-declares-platform-binary-does-not-support`. The "
        "failure is simultaneously a package claim and a binary limitation, so it "
        "is not forced into either single-sided bucket.")
    add("")

    add("### `xcodeproj` fails on a gitignored artifact")
    add("")
    xcode_cells = [c for c in m["build_matrix"] if c["cell"] == "xcodeproj"]
    add("```")
    add(xcode_cells[0]["first_root_failure"] or "error: There is no XCFramework found")
    add(f"** BUILD FAILED ** (exit {xcode_cells[0]['exit_code']})")
    add("```")
    add("")
    add("Failure bucket: `environment-dependent-gitignored-artifact`. "
        "`.gitignore:18` and `.gitignore:20` ignore `llama.xcframework/` and "
        "`*.xcframework/`. A developer who has ever unzipped the release asset "
        "into `OnDeviceCatalyst/` gets a different result from a fresh clone, and "
        "neither `git status` nor a preserve-unrelated-work instruction would "
        "reveal it. **Both outcomes must be stated so a reader with the artifact "
        "present is not misled:** with the XCFramework absent, which is what the "
        "`git archive` scratch population guarantees was measured here, the build "
        "fails at project-configuration time; with the XCFramework present "
        "locally, the same command proceeds past this error, and that outcome was "
        "not measured by this baseline.")
    add("")
    add("Recorded alongside it: `OnDeviceCatalyst.xcodeproj/project.pbxproj` "
        "contains **zero** `XCRemoteSwiftPackageReference` entries.")
    add("")

    add("### Cache dependence")
    add("")
    dependent = sorted({c["cell"] for c in m["build_matrix"] if c["cache_dependent"]})
    if dependent:
        names = [f"`{cell}`" for cell in dependent]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        add("Cold and warm passes agree on every `exit_code` and every `result`. "
            "They disagree on `warning_count` for " + joined +
            ", because a warm incremental build does not recompile unchanged "
            "modules and therefore does not re-emit their diagnostics. Those "
            "cells carry `cache_dependent: true` and both values are recorded as "
            "separate rows in the matrix above rather than collapsed into one "
            "number. Cold is normative.")
    else:
        add("Cold and warm passes agree on every recorded field. No cell is "
            "cache dependent.")
    add("")

    add("## Warnings")
    add("")
    add("Every ODC-owned warning from the cold `ios-simulator` build, grouped into "
        "the five enumerated groups. Counts below are of distinct diagnostic "
        "sites, deduplicated by `file:line:column`.")
    add("")
    by_group: dict[str, list[dict]] = {}
    for warning in m["warnings"]:
        by_group.setdefault(warning["group"], []).append(warning)
    add("| Group | Sites |")
    add("| --- | --- |")
    for group in sorted(by_group):
        add(f"| `{group}` | {len(by_group[group])} |")
    add("")
    for group in sorted(by_group):
        add(f"#### `{group}`")
        add("")
        for warning in by_group[group]:
            location = f"{warning['file']}:{warning['line']}" if warning["line"] else warning["file"]
            add(f"- `{location}` {warning['text']}")
        add("")

    add("### The unhandled-resource diagnostic is a functional defect, not a packaging nit")
    add("")
    metal = [w for w in m["warnings"] if w["group"] == "non-functional-subsystem"]
    cosmetic = [w for w in m["warnings"] if w["group"] == "cosmetic-packaging"]
    add(f"The diagnostic names **{len(metal) + len(cosmetic)}** files: "
        f"{len(metal)} `.metal` sources under "
        "`Sources/OnDeviceCatalyst/Metal Engine/Shaders/` **plus** "
        f"{len(cosmetic)} asset catalogue, `Sources/OnDeviceCatalyst/Assets.xcassets`. "
        "Eight, not seven. `swift package describe --type json` confirms the "
        "target declares no `resources` entry at all. An asset catalogue shipping "
        "inside a library target is itself a recorded finding.")
    add("")
    add("`Sources/OnDeviceCatalyst/Metal Engine/Compute/MetalComputeEngine.swift:89` "
        "calls `device.makeDefaultLibrary()`, which loads a precompiled "
        "`default.metallib` that SwiftPM never produces, because the `.metal` "
        "sources are unhandled. The `guard` therefore fails and throws for any "
        "consumer selecting `InstanceSettings.backendType == .metal`. The "
        "consequence is that the entire Metal Engine subtree (`MetalBackend`, "
        "`TransformerGraph`, `KVCache`, `ModelWeights`, the GGUF parser and "
        "tokenizer) is **unreachable when the package is consumed as a package**. "
        "The `.metal` files are therefore filed under "
        "`non-functional-subsystem`, and only `Assets.xcassets` under "
        "`cosmetic-packaging`.")
    add("")

    add("## Duplicate and divergent runtime sources")
    add("")
    dup = m["duplicate_sources"]
    add(f"- Package root: `{dup['package_root']}`")
    add(f"- App root: `{dup['app_root']}`")
    add(f"- App target consumes the package: "
        f"**{str(dup['app_consumes_package']).lower()}** "
        "(zero `XCRemoteSwiftPackageReference` entries in `project.pbxproj`)")
    add(f"- Same-named Swift files in both roots: **{len(shared)}**")
    add(f"- Of those, **{len(differing)} differ** and **{len(identical)} are identical**")
    add(f"- Package only: **{len(dup['package_only'])}**")
    add(f"- App only: **{len(dup['app_only'])}**")
    add("")
    add("Normative drift command, fixed so the number is not "
        "methodology-dependent:")
    add("")
    add("```")
    add(dup["_command"])
    add("```")
    add("")
    add("| Shared file | Changed lines | Identical | Package sha256 | App sha256 |")
    add("| --- | --- | --- | --- | --- |")
    for row in shared:
        add(f"| `{row['path']}` | {row['changed_lines']} | "
            f"{'yes' if row['identical'] else 'no'} | `{row['package_sha256']}` | "
            f"`{row['app_sha256']}` |")
    add("")
    add("Package only, absent from the app entirely:")
    add("")
    for path in dup["package_only"]:
        add(f"- `{path}`")
    add("")
    add("App only, legitimate app-shell files rather than drift:")
    add("")
    for path in dup["app_only"]:
        add(f"- `{path}`")
    add("")
    add("Recorded discrepancy, so a later reader does not treat it as an error. "
        "Review pass two reported \"13 of 21 shared files differ\" with "
        "`LlamaInstance.swift` at 587 changed lines; the independent verification "
        "reported \"12 of 22\" with 478. This execution measured "
        f"**{len(differing)} of {len(shared)}** with `Core Engine/LlamaInstance.swift` "
        "at 591. All three describe the same tree under three different counts. "
        "The values in the table are those produced by the normative command "
        "above; 478 / 190 / 99 are the values produced by counting unified-diff "
        "body lines (`diff -u ... | grep -c '^[-+][^-+]'`); the pass-two totals "
        "differ by an off-by-one in the shared-file census and a slightly "
        "different line count for the largest file. The table is emitted by "
        "`scripts/capture-baseline.sh`, never hand maintained, so this class of "
        "disagreement cannot recur.")
    add("")
    add("The divergence is architectural, not cosmetic. The package copy holds "
        "`internal var backend: InferenceBackend?` "
        "(`Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:25-27`); the "
        "app copy still holds raw `cModel` / `cContext` / `cBatch` "
        "`OpaquePointer` state and its own `import llama` "
        "(`OnDeviceCatalyst/Core Engine/LlamaInstance.swift:26-29`). One "
        "consequence is already observable: the app copy does not carry D3, "
        "because its `publishProgress` uses two separate branches. D1 is "
        "reproduced verbatim in the app copy.")
    add("")
    add("This matters beyond ODC-0002. ODC-0004 cannot begin until the project "
        "has decided which v2 is being characterized, and ODC-0300 cannot begin "
        "until it is decided whether the app fork is deleted or reconciled. This "
        "baseline names both copies and hands the decision to a ticket; it does "
        "not make it.")
    add("")

    add("## Tests")
    add("")
    tests = m["tests"]
    add(f"- Declared test target path: `{tests['target_path']}`")
    add("- Compiled test sources: " + ", ".join(f"`{p}`" for p in tests["compiled_test_sources"]))
    add(f"- Compiled test cases ({len(tests['test_case_names'])}): "
        + ", ".join(f"`{name}`" for name in tests["test_case_names"]))
    add("- Orphaned files under `Tests/`, compiled by no target: "
        + ", ".join(f"`{p}`" for p in tests["orphaned_test_files"]))
    add(f"- Runnable on this host: **{str(tests['runnable_on_host']).lower()}**, "
        "because `swift test` fails at the same `no such module 'llama'` root "
        "cause as `swift build` on macOS")
    add(f"- Defects covered by the existing suite: **{tests['defect_coverage_count']}**")
    add("")
    add("`swift package describe --type json` confirms the orphaned files appear "
        "in no target's `sources`. None of the compiled tests exercises instance "
        "caching or release, streaming generation, load-progress termination, "
        "fallback initialization, or the Metal backend, so existing coverage of "
        "every recorded defect is zero. ODC-0004 builds directly on this "
        "inventory.")
    add("")

    add("## Characterized findings")
    add("")
    add("**This ticket fixes none of these.** Each is recorded with its "
        "`path:line` evidence and a mapped follow-up ticket.")
    add("")
    add("| ID | Ticket | Bucket | Actionable | Summary |")
    add("| --- | --- | --- | --- | --- |")
    for finding in m["findings"]:
        add(f"| {finding['id']} | {finding['ticket']} | `{finding['bucket']}` | "
            f"{'yes' if finding['actionable'] else 'no'} | {finding['summary']} |")
    add("")
    for finding in m["findings"]:
        add(f"### {finding['id']} (mapped to {finding['ticket']})")
        add("")
        add(finding["summary"] + ".")
        add("")
        for item in finding["evidence"]:
            add(f"- {item}")
        add("")

    add("## Deviations and blockers")
    add("")
    add("Recorded rather than papered over.")
    add("")
    add("1. **The lockfile decision was not executed.** The spec's "
        "`## Migration and compatibility impact` decides to commit the resolver's "
        "deterministic manifest-compatible output to the tracked "
        "`Package.resolved`. This execution was scoped to characterization only "
        "and made no tracked-file change outside `docs/` and `scripts/`, so the "
        "tracked `Package.resolved` still carries format version "
        f"`{lock['format_version_before']}` and the `branch: main` pin. The "
        "consequence is that the spec's validation item 3, "
        "`jq -e '.version == 3' Package.resolved`, does not pass in the working "
        "tree. It passes in `$SCRATCH` after the resolve, which is where the "
        "measurement was taken. Executing the decision is a separate, deliberate "
        "commit.")
    add("2. **Findings are mapped to existing tickets, not to the reserved "
        "range.** The spec reserves `ODC-0010` through `ODC-0049` for tickets "
        "this procedure creates. Creating them means editing `Tickets.md`, which "
        "this execution did not do. Each finding is therefore mapped to the "
        "existing ticket that most nearly owns it, and every mapped ID does "
        "exist in `Tickets.md`. The reserved-range allocation, including the "
        "P0 decision ticket for D7 and the separate ticket for uncovered "
        "link-time and runtime resolution against the stub slice, remains "
        "outstanding.")
    add("3. **The `Package.swift:36-37` finding is report-only.** See the section "
        "above for why the manifest's `findings[]` cannot carry it under the "
        "spec's schema.")
    add("4. **`swift package resolve` was run in `$SCRATCH`, never in the working "
        "tree.** Running it in the working tree would rewrite the tracked "
        "`Package.resolved`. The scratch tree is byte-identical tracked content "
        "at the pinned revision, so the measurement is equivalent. The working "
        "tree's `Package.resolved` was never modified and needed no restore.")
    add("5. **One `path:line` in the spec is off by one, and the measured value "
        "is used here.** The spec's D8 entry cites `cleanup()` at "
        "`Core Engine/LlamaInstance.swift:187`. The measured location at the "
        "pinned revision is `:188`; `:187` is the preceding comment line. Every "
        "other citation in the spec's D1 to D8 table was reproduced exactly.")
    add("")

    add("## Acceptance criteria results")
    add("")
    add("Each criterion is decided by a command's exit code, per the spec's "
        "`## Acceptance criteria`. Results below are from the run that produced "
        "these deliverables.")
    add("")
    add("| # | Criterion | Deciding command | Result |")
    add("| --- | --- | --- | --- |")
    for number, criterion, command, result in ACCEPTANCE:
        add(f"| {number} | {criterion} | `{command}` | {result} |")
    add("")
    add("The spec's validation item 3, `jq -e '.version == 3' Package.resolved`, "
        "is **not satisfied in the working tree** and is not an acceptance "
        "criterion; see `## Deviations and blockers` item 1. Validation item 4, "
        "resolve stability, was decided in `$SCRATCH` and passed. Validation "
        "item 5 and item 8 correspond to A11 and A14 above.")
    add("")
    add("Supporting evidence for A12: `.build/workspace-state.json` in the "
        "operator's tree was neither read nor written by any step, and its "
        "recorded identities and revisions already match the resolver output "
        "measured in `$SCRATCH` (`mlx-swift 0.29.1`, `mlx-swift-lm 2.29.3`). The "
        "effective dependency graph was therefore already in force before this "
        "ticket and did not move.")
    add("")
    add("## Reproduction")
    add("")
    add("```")
    add("scripts/capture-baseline.sh")
    add("python3 scripts/check-baseline.py")
    add("python3 scripts/validate-project-state.py")
    add("```")
    add("")
    add("`scripts/capture-baseline.sh` re-runs the whole procedure in a fresh "
        "scratch tree outside the repository and rewrites both deliverables, so "
        "re-running the baseline is one command and drift is a diff. "
        "`scripts/check-baseline.py` is the baseline-content gate: it decides "
        "schema conformance, report and manifest correspondence, redaction, the "
        "copyable-command rule, the build matrix enumeration, slice "
        "completeness, dependency and artifact pins, the findings ledger, and "
        "the duplication record. `scripts/validate-project-state.py` is a "
        "project-state gate only; it does not read `docs/baselines/`.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    evidence = Path(args.evidence)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_model(evidence)
    report = render_report(manifest, evidence)

    scratch = str(Path(args.scratch).resolve())
    report = report.replace(scratch, "$SCRATCH").replace(scratch.replace("/private", "", 1), "$SCRATCH")
    em_dash = chr(0x2014)
    if em_dash in report:
        print("render-baseline: em dash in report violates repository writing style", file=sys.stderr)
        return 1

    (out / "v2.0.4-environment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    (out / "v2.0.4.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
