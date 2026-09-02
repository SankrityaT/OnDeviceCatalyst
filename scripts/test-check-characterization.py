#!/usr/bin/env python3
"""Self-test for scripts/check-characterization.py, following
scripts/test-project-state-validator.py's positive/negative fixture pattern.

Builds small, self-contained fixture trees (a copy of the checker plus
synthetic Tests/, Sources/, Tickets.md, and docs/characterization/ content)
rather than exercising the checker against this repository, so that a
negative case (a missing four-line block, a mismatched fingerprint, a
mutated orphan file, a malformed skip ledger) can be constructed
deliberately and cheaply, with no dependency on a Swift build.

--packaging and --inventory are exercised against the real repository
directly (see the ledger this ticket's implementation report cites), not
through synthetic fixtures here, because faithfully mocking Package.swift,
the .xcodeproj, and the ODC-0002 baseline manifest would be substantially
more fixture machinery than the logic it protects; that is a deliberate,
disclosed scope limit of this self-test, not an oversight.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "scripts" / "check-characterization.py"


def run(fixture: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(fixture / "scripts" / "check-characterization.py"), *args],
        cwd=fixture, capture_output=True, text=True,
    )


def base_fixture(tmp: Path) -> Path:
    fixture = tmp / "fixture"
    (fixture / "scripts").mkdir(parents=True)
    shutil.copy2(CHECKER, fixture / "scripts" / "check-characterization.py")
    (fixture / "Tests" / "OnDeviceCatalystTests").mkdir(parents=True)
    (fixture / "docs" / "characterization").mkdir(parents=True)
    (fixture / "Sources" / "OnDeviceCatalyst").mkdir(parents=True)
    (fixture / "Tickets.md").write_text(
        "| ID | Type |\n| --- | --- |\n| ODC-0099 | bug |\n", encoding="utf-8"
    )
    return fixture


def fail(message: str) -> None:
    print(f"test-check-characterization: FAIL: {message}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# --naming
# ---------------------------------------------------------------------------

GOOD_TEST_FILE = '''
import XCTest

final class FixtureCharacterizationTests: XCTestCase {

    /// CHARACTERIZATION D9 (ODC-0099)
    /// Today: the fixture does X.
    /// Should be: the fixture should do Y.
    /// Evidence: Sources/OnDeviceCatalyst/Fixture.swift:1
    func test_characterizes_fixtureDoesX__ODC_0099() {
        XCTAssertTrue(true)
    }

    func test_requires_fixtureIsAlwaysTrue() {
        XCTAssertTrue(true)
    }
}
'''


def test_naming_positive() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        (fixture / "Tests" / "OnDeviceCatalystTests" / "Fixture.swift").write_text(GOOD_TEST_FILE, encoding="utf-8")
        result = run(fixture, "--naming")
        if result.returncode != 0:
            fail(f"naming positive fixture should pass: {result.stdout}{result.stderr}")


def test_naming_missing_block() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        bad = GOOD_TEST_FILE.replace(
            "    /// CHARACTERIZATION D9 (ODC-0099)\n"
            "    /// Today: the fixture does X.\n"
            "    /// Should be: the fixture should do Y.\n"
            "    /// Evidence: Sources/OnDeviceCatalyst/Fixture.swift:1\n",
            "",
        )
        (fixture / "Tests" / "OnDeviceCatalystTests" / "Fixture.swift").write_text(bad, encoding="utf-8")
        result = run(fixture, "--naming")
        if result.returncode == 0:
            fail("missing four-line block should fail --naming")


def test_naming_unknown_ticket() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        bad = GOOD_TEST_FILE.replace("ODC_0099", "ODC_9999").replace("ODC-0099", "ODC-9999")
        (fixture / "Tests" / "OnDeviceCatalystTests" / "Fixture.swift").write_text(bad, encoding="utf-8")
        result = run(fixture, "--naming")
        if result.returncode == 0:
            fail("a ticket absent from Tickets.md should fail --naming")


def test_naming_requires_with_odc_suffix() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        bad = GOOD_TEST_FILE.replace(
            "func test_requires_fixtureIsAlwaysTrue()",
            "func test_requires_fixtureIsAlwaysTrue__ODC_0099()",
        )
        (fixture / "Tests" / "OnDeviceCatalystTests" / "Fixture.swift").write_text(bad, encoding="utf-8")
        result = run(fixture, "--naming")
        if result.returncode == 0:
            fail("a test_requires_ method naming a ticket should fail --naming")


def test_naming_home_path_denylist() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        bad = GOOD_TEST_FILE.replace(
            "XCTAssertTrue(true)\n    }\n\n    func test_requires",
            'let p = "/Users/someone/secret.gguf"\n        XCTAssertTrue(true)\n    }\n\n    func test_requires',
        )
        (fixture / "Tests" / "OnDeviceCatalystTests" / "Fixture.swift").write_text(bad, encoding="utf-8")
        result = run(fixture, "--naming")
        if result.returncode == 0:
            fail("an absolute home-directory path should fail --naming")


def test_naming_bad_suffix() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        bad = GOOD_TEST_FILE.replace("test_characterizes_fixtureDoesX__ODC_0099", "test_characterizes_fixtureDoesX")
        (fixture / "Tests" / "OnDeviceCatalystTests" / "Fixture.swift").write_text(bad, encoding="utf-8")
        result = run(fixture, "--naming")
        if result.returncode == 0:
            fail("a test_characterizes_ method with no __ODC_00NN/__no_defect suffix should fail --naming")


# ---------------------------------------------------------------------------
# --fingerprints
# ---------------------------------------------------------------------------

def write_fingerprint_fixture(fixture: Path, body: str) -> None:
    src = fixture / "Sources" / "OnDeviceCatalyst" / "Fixture.swift"
    src.write_text(f"struct Fixture {{\n    func target() {{\n{body}\n    }}\n}}\n", encoding="utf-8")


def compute_fixture_hash(fixture: Path, body: str) -> str:
    sys.path.insert(0, str(fixture / "scripts"))
    import importlib
    if "check_characterization" in sys.modules:
        del sys.modules["check_characterization"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_characterization_fixture", fixture / "scripts" / "check-characterization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    region = module.extract_region(fixture / "Sources" / "OnDeviceCatalyst" / "Fixture.swift", r"func target\(")
    normalized = module.normalize_region(region)
    import hashlib
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_fingerprints_positive_and_negative() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        write_fingerprint_fixture(fixture, "        let x = 1 // a comment\n        print(x)")
        digest = compute_fixture_hash(fixture, "")
        import json
        fp_path = fixture / "docs" / "characterization" / "v2-fingerprints.json"
        fp_path.write_text(json.dumps({
            "fingerprints": [{
                "id": "F-FIXTURE-1", "ticket": "ODC-0099",
                "file": "Sources/OnDeviceCatalyst/Fixture.swift",
                "anchor": r"func target\(",
                "note": "fixture",
                "sha256": digest,
            }]
        }), encoding="utf-8")

        result = run(fixture, "--fingerprints")
        if result.returncode != 0:
            fail(f"fingerprints positive fixture should pass: {result.stdout}{result.stderr}")

        # Now mutate the source region and expect a mismatch.
        write_fingerprint_fixture(fixture, "        let x = 2\n        print(x)")
        result = run(fixture, "--fingerprints")
        if result.returncode == 0:
            fail("a changed defect site should fail --fingerprints")
        if "this defect site changed" not in (result.stdout + result.stderr):
            fail("fingerprint mismatch message should explain the mutation")


# ---------------------------------------------------------------------------
# --orphans
# ---------------------------------------------------------------------------

def test_orphans_pin_mismatch() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        # The real checker hardcodes the three real repo paths and hashes;
        # exercise it against the real repo files copied into the fixture at
        # their expected relative paths so the pins actually match, then
        # mutate one and expect a mismatch.
        (fixture / "Tests").mkdir(exist_ok=True)
        for name in ("EmbeddingTest.swift", "test_embedding.swift", "BERTEmbeddingTest.swift"):
            shutil.copy2(ROOT / "Tests" / name, fixture / "Tests" / name)

        result = run(fixture, "--orphans")
        if result.returncode != 0:
            fail(f"orphans positive fixture should pass: {result.stdout}{result.stderr}")

        (fixture / "Tests" / "EmbeddingTest.swift").write_text("// mutated\n", encoding="utf-8")
        result = run(fixture, "--orphans")
        if result.returncode == 0:
            fail("a mutated orphan file should fail --orphans")


# ---------------------------------------------------------------------------
# --skips
# ---------------------------------------------------------------------------

def test_skips_zero_executed_and_all_skipped_fail() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        ledger = fixture / "empty.ledger"
        ledger.write_text("", encoding="utf-8")
        result = run(fixture, "--skips", str(ledger))
        if result.returncode == 0:
            fail("an empty ledger (zero executed) should fail --skips")

        ledger.write_text("SKIPPED test_characterizes_x__ODC_0011 SKIP[requires-device]\n", encoding="utf-8")
        result = run(fixture, "--skips", str(ledger))
        if result.returncode == 0:
            fail("an all-skipped ledger should fail --skips")


def test_skips_unrecognized_code_fails() -> None:
    with tempfile.TemporaryDirectory() as raw:
        fixture = base_fixture(Path(raw))
        ledger = fixture / "bad-code.ledger"
        ledger.write_text(
            "EXECUTED test_requires_something\n"
            "SKIPPED test_characterizes_x__ODC_0011 SKIP[not-a-real-code]\n",
            encoding="utf-8",
        )
        result = run(fixture, "--skips", str(ledger))
        if result.returncode == 0:
            fail("an unrecognized SKIP code should fail --skips")


def main() -> int:
    tests = [
        test_naming_positive,
        test_naming_missing_block,
        test_naming_unknown_ticket,
        test_naming_requires_with_odc_suffix,
        test_naming_home_path_denylist,
        test_naming_bad_suffix,
        test_fingerprints_positive_and_negative,
        test_orphans_pin_mismatch,
        test_skips_zero_executed_and_all_skipped_fail,
        test_skips_unrecognized_code_fails,
    ]
    for test in tests:
        test()
        print(f"test-check-characterization: {test.__name__} passed")
    print("test-check-characterization: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
