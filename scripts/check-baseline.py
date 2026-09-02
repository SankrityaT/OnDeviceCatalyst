#!/usr/bin/env python3
"""Baseline-content gate for ODC-0002.

Decides every acceptance criterion in docs/specs/ODC-0002-v2-baseline.md that is
not already a one-line shell command: manifest schema conformance,
report/manifest correspondence, redaction, the copyable-command rule, the build
matrix enumeration, artifact slice completeness, dependency and artifact pins,
the findings ledger, and the source-duplication record.

Exit code 0 means every selected check passed. Exit code 1 means at least one
failed. Exit code 2 means the script could not run (missing deliverable, invalid
JSON).

Usage:
    python3 scripts/check-baseline.py                 # every check
    python3 scripts/check-baseline.py --schema-only
    python3 scripts/check-baseline.py --correspondence
    python3 scripts/check-baseline.py --redaction
    python3 scripts/check-baseline.py --copyable
    python3 scripts/check-baseline.py --matrix
    python3 scripts/check-baseline.py --slices
    python3 scripts/check-baseline.py --pins
    python3 scripts/check-baseline.py --findings
    python3 scripts/check-baseline.py --duplication
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "docs" / "baselines" / "v2.0.4-environment.json"
REPORT_PATH = ROOT / "docs" / "baselines" / "v2.0.4.md"
TICKETS_PATH = ROOT / "Tickets.md"

REQUIRED_FINDING_IDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
REQUIRED_MATRIX_CELLS = ["ios-simulator", "ios-device", "macos", "macos-test"]
SIMULATOR_RESULT = "links, cannot infer"
EXPECTED_SHARED_COUNT = 22


# ---------------------------------------------------------------------------
# Normative manifest schema. One copy only, per the spec's Design section.
# ---------------------------------------------------------------------------

BASELINE_ENVIRONMENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OnDeviceCatalyst baseline environment manifest",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "captured_at", "repo", "toolchain", "host", "sdks",
        "package", "dependencies", "artifact", "build_matrix", "warnings",
        "findings", "duplicate_sources", "tests",
    ],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "captured_at": {"type": "string", "format": "date"},
        "repo": {
            "type": "object", "additionalProperties": False,
            "required": ["revision", "describe", "dirty"],
            "properties": {
                "revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "describe": {"type": "string"},
                "dirty": {"type": "boolean"},
            },
        },
        "toolchain": {
            "type": "object", "additionalProperties": False,
            "required": ["swift", "swift_driver", "host_triple", "xcode_version",
                         "xcode_build", "developer_dir_redacted"],
            "properties": {
                "swift": {"type": "string"},
                "swift_driver": {"type": "string"},
                "host_triple": {"type": "string"},
                "xcode_version": {"type": "string"},
                "xcode_build": {"type": "string"},
                "developer_dir_redacted": {"type": "string"},
            },
        },
        "host": {
            "type": "object", "additionalProperties": False,
            "required": ["model_identifier", "chip", "cores", "memory_bytes",
                         "os_product", "os_version", "os_build"],
            "properties": {
                "model_identifier": {"type": "string"},
                "chip": {"type": "string"},
                "cores": {"type": "integer", "minimum": 1},
                "memory_bytes": {"type": "integer", "minimum": 1},
                "os_product": {"type": "string"},
                "os_version": {"type": "string"},
                "os_build": {"type": "string"},
            },
        },
        "sdks": {
            "type": "array", "minItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "version", "canonical_name"],
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "canonical_name": {"type": "string"},
                },
            },
        },
        "package": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "tools_version_declared", "tools_version_reported",
                         "platforms", "products", "targets"],
            "properties": {
                "name": {"type": "string"},
                "tools_version_declared": {"type": "string"},
                "tools_version_reported": {"type": "string"},
                "platforms": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["name", "version"],
                        "properties": {
                            "name": {"type": "string"},
                            "version": {"type": "string"},
                        },
                    },
                },
                "products": {"type": "array", "items": {"type": "string"}},
                "targets": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["name", "type", "path", "source_count", "resources"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "path": {"type": "string"},
                            "source_count": {"type": "integer", "minimum": 0},
                            "resources": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        },
        "dependencies": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["identity", "requirement", "resolved_version",
                             "resolved_revision", "direct"],
                "properties": {
                    "identity": {"type": "string"},
                    "requirement": {"type": "string"},
                    "resolved_version": {"type": ["string", "null"]},
                    "resolved_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                    "direct": {"type": "boolean"},
                },
            },
        },
        "lockfile": {
            "type": "object", "additionalProperties": False,
            "required": ["format_version_before", "format_version_after",
                         "origin_hash_added", "second_resolve_stable",
                         "raises_minimum_consumer_toolchain"],
            "properties": {
                "format_version_before": {"type": "integer"},
                "format_version_after": {"type": "integer"},
                "origin_hash_added": {"type": "boolean"},
                "second_resolve_stable": {"type": "boolean"},
                "raises_minimum_consumer_toolchain": {"type": "boolean"},
            },
        },
        "artifact": {
            "type": "object", "additionalProperties": False,
            "required": ["url", "checksum", "format_version", "slices"],
            "properties": {
                "url": {"type": "string"},
                "checksum": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "format_version": {"type": "string"},
                "slices": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["identifier", "platform", "variant",
                                     "architectures", "bytes", "nm_line_count",
                                     "defined_symbol_count", "objects", "is_stub",
                                     "_command"],
                        "properties": {
                            "identifier": {"type": "string"},
                            "platform": {"type": "string"},
                            "variant": {"type": ["string", "null"]},
                            "architectures": {"type": "array", "items": {"type": "string"}},
                            "bytes": {"type": "integer", "minimum": 0},
                            "nm_line_count": {"type": "integer", "minimum": 0},
                            "defined_symbol_count": {"type": "integer", "minimum": 0},
                            "objects": {"type": "array", "items": {"type": "string"}},
                            "is_stub": {"type": "boolean"},
                            "_command": {"type": "string"},
                        },
                    },
                },
            },
        },
        "build_matrix": {
            "type": "array", "minItems": 4,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["cell", "system", "triple", "sdk", "command", "exit_code",
                             "result", "cache_state", "cache_dependent",
                             "warning_count", "first_root_failure"],
                "properties": {
                    "cell": {"type": "string"},
                    "system": {"enum": ["swiftpm", "xcodebuild"]},
                    "triple": {"type": ["string", "null"]},
                    "sdk": {"type": ["string", "null"]},
                    "command": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "result": {
                        "enum": ["links, cannot infer", "compiles", "fails",
                                 "blocked-by-build"]
                    },
                    "cache_state": {"enum": ["cold", "warm"]},
                    "cache_dependent": {"type": "boolean"},
                    "warning_count": {"type": "integer", "minimum": 0},
                    "first_root_failure": {"type": ["string", "null"]},
                },
            },
        },
        "warnings": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["group", "file", "line", "text"],
                "properties": {
                    "group": {
                        "enum": ["cosmetic-packaging", "non-functional-subsystem",
                                 "deprecated-llama-api", "unused-code",
                                 "concurrency-lifecycle"]
                    },
                    "file": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                    "text": {"type": "string"},
                },
            },
        },
        "findings": {
            "type": "array", "minItems": 8,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "summary", "evidence", "bucket", "actionable",
                             "ticket"],
                "properties": {
                    "id": {"type": "string", "pattern": "^D[0-9]+$"},
                    "summary": {"type": "string"},
                    "evidence": {"type": "array", "minItems": 1,
                                 "items": {"type": "string"}},
                    "bucket": {"type": "string"},
                    "actionable": {"type": "boolean"},
                    "ticket": {"type": "string", "pattern": "^ODC-[0-9]{4}$"},
                },
            },
        },
        "duplicate_sources": {
            "type": "object", "additionalProperties": False,
            "required": ["package_root", "app_root", "app_consumes_package",
                         "shared", "package_only", "app_only", "_command"],
            "properties": {
                "package_root": {"type": "string"},
                "app_root": {"type": "string"},
                "app_consumes_package": {"type": "boolean"},
                "shared": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["path", "identical", "changed_lines",
                                     "package_sha256", "app_sha256"],
                        "properties": {
                            "path": {"type": "string"},
                            "identical": {"type": "boolean"},
                            "changed_lines": {"type": "integer", "minimum": 0},
                            "package_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                            "app_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                    },
                },
                "package_only": {"type": "array", "items": {"type": "string"}},
                "app_only": {"type": "array", "items": {"type": "string"}},
                "_command": {"type": "string"},
            },
        },
        "tests": {
            "type": "object", "additionalProperties": False,
            "required": ["target_path", "compiled_test_sources", "test_case_names",
                         "orphaned_test_files", "runnable_on_host",
                         "defect_coverage_count"],
            "properties": {
                "target_path": {"type": "string"},
                "compiled_test_sources": {"type": "array", "items": {"type": "string"}},
                "test_case_names": {"type": "array", "items": {"type": "string"}},
                "orphaned_test_files": {"type": "array", "items": {"type": "string"}},
                "runnable_on_host": {"type": "boolean"},
                "defect_coverage_count": {"type": "integer", "minimum": 0},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Minimal JSON Schema validator covering the keywords the schema above uses.
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _type_ok(value, expected) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        python_type = _TYPE_MAP[name]
        if name == "integer":
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return True
            continue
        if name == "number" and isinstance(value, bool):
            continue
        if name == "boolean":
            if isinstance(value, bool):
                return True
            continue
        if isinstance(value, python_type):
            return True
    return False


def validate_schema(value, schema, path: str, errors: list[str]) -> None:
    if "type" in schema and not _type_ok(value, schema["type"]):
        errors.append(f"{path}: expected type {schema['type']}, got {type(value).__name__}")
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "pattern" in schema and isinstance(value, str):
        if not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if schema.get("format") == "date" and isinstance(value, str):
        if not _DATE_RE.match(value):
            errors.append(f"{path}: {value!r} is not an ISO date")
    if "minimum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema["minimum"]:
            errors.append(f"{path}: {value} is below minimum {schema['minimum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} items, minimum {schema['minItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_schema(item, item_schema, f"{path}[{index}]", errors)
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected key {key!r}")
        for key, sub_schema in properties.items():
            if key in value:
                validate_schema(value[key], sub_schema, f"{path}.{key}", errors)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

DENYLIST = [
    r"/Users/[^/ \"]+",
    r"\$HOME",
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{16}",
    r"[0-9A-Fa-f]{40}-[0-9A-Fa-f]{16}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
]

# The two narrow exceptions the spec enumerates: bare 40-hex git revisions and
# bare 64-hex checksums are evidence and are permitted. The UDID-shaped patterns
# above are written so they cannot match either, because both exception forms
# carry no hyphen.
EXCEPTION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def check_redaction(report: str, manifest_text: str) -> list[str]:
    failures = []
    for label, text in (("report", report), ("manifest", manifest_text)):
        for pattern in DENYLIST:
            for match in re.finditer(pattern, text):
                if EXCEPTION_RE.match(match.group(0)):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"redaction: {label}:{line} matches {pattern!r} -> {match.group(0)!r}"
                )
    return failures


def fenced_blocks(report: str) -> list[tuple[int, str]]:
    blocks = []
    inside = False
    start = 0
    buffer: list[str] = []
    for number, line in enumerate(report.splitlines(), start=1):
        if line.startswith("```"):
            if inside:
                blocks.append((start, "\n".join(buffer)))
                buffer = []
                inside = False
            else:
                inside = True
                start = number
            continue
        if inside:
            buffer.append(line)
    return blocks


def check_copyable(report: str) -> list[str]:
    failures = []
    for start, block in fenced_blocks(report):
        if "/Users/" in block:
            failures.append(f"copyable: fenced block at line {start} contains an absolute home path")
    return failures


# ---------------------------------------------------------------------------
# Correspondence
# ---------------------------------------------------------------------------

def check_correspondence(manifest: dict, report: str) -> list[str]:
    failures = []
    report_lines = report.splitlines()

    def require(value, label):
        if str(value) not in report:
            failures.append(f"correspondence: report is missing {label} = {value!r}")

    def require_pair(first, second, label):
        for line in report_lines:
            if str(first) in line and str(second) in line:
                return
        failures.append(
            f"correspondence: report has no single line pairing {label}: "
            f"{first!r} with {second!r}"
        )

    require(manifest["repo"]["revision"], "repo.revision")
    require(manifest["toolchain"]["xcode_build"], "toolchain.xcode_build")
    require(manifest["toolchain"]["swift"], "toolchain.swift")
    for sdk in manifest["sdks"]:
        require(sdk["canonical_name"], "sdks[].canonical_name")
    require(manifest["package"]["tools_version_declared"], "package.tools_version_declared")
    require(manifest["package"]["tools_version_reported"], "package.tools_version_reported")
    for dependency in manifest["dependencies"]:
        require(dependency["resolved_revision"], f"dependencies[{dependency['identity']}].resolved_revision")
    require(manifest["artifact"]["checksum"], "artifact.checksum")
    for slice_ in manifest["artifact"]["slices"]:
        require(slice_["bytes"], f"artifact.slices[{slice_['identifier']}].bytes")
        require(slice_["nm_line_count"], f"artifact.slices[{slice_['identifier']}].nm_line_count")
    for cell in manifest["build_matrix"]:
        require_pair(cell["cell"], cell["result"], "build_matrix cell and result")
    for entry in manifest["duplicate_sources"]["shared"]:
        if not entry["identical"]:
            require_pair(entry["path"], entry["changed_lines"], "duplicate_sources shared row")
    for finding in manifest["findings"]:
        require_pair(finding["id"], finding["ticket"], "finding and ticket")
    return failures


# ---------------------------------------------------------------------------
# Domain checks
# ---------------------------------------------------------------------------

def check_matrix(manifest: dict) -> list[str]:
    failures = []
    allowed = set(BASELINE_ENVIRONMENT_SCHEMA["properties"]["build_matrix"]["items"]
                  ["properties"]["result"]["enum"])
    cells = {}
    for entry in manifest["build_matrix"]:
        if entry["result"] not in allowed:
            failures.append(f"matrix: {entry['cell']} has non-enumerated result {entry['result']!r}")
        cells.setdefault(entry["cell"], []).append(entry)
    for required in REQUIRED_MATRIX_CELLS:
        if required not in cells:
            failures.append(f"matrix: required cell {required!r} is absent")
    for entry in cells.get("ios-simulator", []):
        if entry["result"] != SIMULATOR_RESULT:
            failures.append(
                f"matrix: ios-simulator ({entry['cache_state']}) result is "
                f"{entry['result']!r}, must be {SIMULATOR_RESULT!r}"
            )
    return failures


def check_slices(manifest: dict) -> list[str]:
    failures = []
    slices = manifest["artifact"]["slices"]
    if not slices:
        failures.append("slices: artifact.slices is empty")
    for entry in slices:
        identifier = entry.get("identifier", "<unnamed>")
        for key in ("bytes", "nm_line_count", "defined_symbol_count"):
            if not isinstance(entry.get(key), int) or isinstance(entry.get(key), bool):
                failures.append(f"slices: {identifier} lacks integer {key}")
        if not entry.get("objects"):
            failures.append(f"slices: {identifier} has an empty object list")
        if not isinstance(entry.get("is_stub"), bool):
            failures.append(f"slices: {identifier} lacks a boolean is_stub")
        if not entry.get("_command"):
            failures.append(f"slices: {identifier} lacks the _command methodology field")
    return failures


def check_pins(manifest: dict) -> list[str]:
    failures = []
    for dependency in manifest["dependencies"]:
        revision = dependency.get("resolved_revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            failures.append(
                f"pins: dependency {dependency.get('identity')} revision {revision!r} is not 40-hex"
            )
    checksum = manifest["artifact"].get("checksum", "")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        failures.append(f"pins: artifact.checksum {checksum!r} is not 64-hex")
    return failures


def known_ticket_ids() -> set[str]:
    if not TICKETS_PATH.is_file():
        return set()
    return set(re.findall(r"^\|\s*(ODC-\d{4})\s*\|", TICKETS_PATH.read_text(encoding="utf-8"),
                          flags=re.MULTILINE))


def check_findings(manifest: dict) -> list[str]:
    failures = []
    tickets = known_ticket_ids()
    present = {}
    for finding in manifest["findings"]:
        present[finding.get("id")] = finding
    for required in REQUIRED_FINDING_IDS:
        if required not in present:
            failures.append(f"findings: {required} is absent")
    for identifier, finding in present.items():
        if not finding.get("evidence"):
            failures.append(f"findings: {identifier} has no evidence")
        ticket = finding.get("ticket", "")
        if not re.fullmatch(r"ODC-\d{4}", ticket):
            failures.append(f"findings: {identifier} ticket {ticket!r} is malformed")
        elif ticket not in tickets:
            failures.append(f"findings: {identifier} ticket {ticket} is not in Tickets.md")
    return failures


def check_duplication(manifest: dict) -> list[str]:
    failures = []
    duplicates = manifest["duplicate_sources"]
    if duplicates.get("app_consumes_package") is not False:
        failures.append("duplication: app_consumes_package must be false")
    shared = duplicates.get("shared", [])
    if len(shared) != EXPECTED_SHARED_COUNT:
        failures.append(
            f"duplication: shared has {len(shared)} entries, expected {EXPECTED_SHARED_COUNT}"
        )
    for entry in shared:
        identical = entry.get("identical")
        changed = entry.get("changed_lines")
        if identical is True and changed != 0:
            failures.append(f"duplication: {entry.get('path')} identical but changed_lines={changed}")
        if identical is False and changed == 0:
            failures.append(f"duplication: {entry.get('path')} not identical but changed_lines=0")
        if entry.get("package_sha256") == entry.get("app_sha256") and identical is not True:
            failures.append(f"duplication: {entry.get('path')} sha256 pair contradicts identical")
    return failures


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("schema-only", "correspondence", "redaction", "copyable", "matrix",
                 "slices", "pins", "findings", "duplication"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()

    selected = {
        "schema": args.schema_only,
        "correspondence": args.correspondence,
        "redaction": args.redaction,
        "copyable": args.copyable,
        "matrix": args.matrix,
        "slices": args.slices,
        "pins": args.pins,
        "findings": args.findings,
        "duplication": args.duplication,
    }
    if not any(selected.values()):
        selected = {key: True for key in selected}

    if not MANIFEST_PATH.is_file():
        print(f"check-baseline: missing {MANIFEST_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 2
    if not REPORT_PATH.is_file():
        print(f"check-baseline: missing {REPORT_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 2

    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        print(f"check-baseline: manifest is not valid JSON: {error}", file=sys.stderr)
        return 2

    failures: list[str] = []
    ran: list[str] = []

    if selected["schema"]:
        ran.append("schema")
        schema_errors: list[str] = []
        validate_schema(manifest, BASELINE_ENVIRONMENT_SCHEMA, "$", schema_errors)
        failures.extend(f"schema: {item}" for item in schema_errors)

    # Every remaining check reads fields the schema guarantees. Skip them when
    # the schema already failed, so the output names the root cause once.
    schema_clean = not any(item.startswith("schema:") for item in failures)

    if selected["redaction"]:
        ran.append("redaction")
        failures.extend(check_redaction(report_text, manifest_text))
    if selected["copyable"]:
        ran.append("copyable")
        failures.extend(check_copyable(report_text))
    if schema_clean:
        if selected["correspondence"]:
            ran.append("correspondence")
            failures.extend(check_correspondence(manifest, report_text))
        if selected["matrix"]:
            ran.append("matrix")
            failures.extend(check_matrix(manifest))
        if selected["slices"]:
            ran.append("slices")
            failures.extend(check_slices(manifest))
        if selected["pins"]:
            ran.append("pins")
            failures.extend(check_pins(manifest))
        if selected["findings"]:
            ran.append("findings")
            failures.extend(check_findings(manifest))
        if selected["duplication"]:
            ran.append("duplication")
            failures.extend(check_duplication(manifest))

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        print(f"check-baseline FAILED: {len(failures)} problem(s)", file=sys.stderr)
        return 1

    print(f"check-baseline OK: {', '.join(ran)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
