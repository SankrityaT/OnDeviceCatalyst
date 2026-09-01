#!/usr/bin/env python3
"""Exercise positive and negative project-state validation fixtures."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent.parent


def make_fixture(destination: Path) -> None:
    (destination / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "validate-project-state.py", destination / "scripts")
    shutil.copy2(ROOT / "Tickets.md", destination)
    shutil.copy2(ROOT / "ROADMAP.md", destination)
    shutil.copy2(ROOT / "LICENSE", destination)
    shutil.copy2(ROOT / "NOTICE", destination)
    for markdown in ROOT.glob("*.md"):
        if markdown.name not in {"Tickets.md", "ROADMAP.md"}:
            shutil.copy2(markdown, destination)
    shutil.copytree(ROOT / "docs", destination / "docs")
    shutil.copytree(ROOT / ".github", destination / ".github")


def run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(root / "scripts" / "validate-project-state.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def expect_failure(name: str, mutate: Callable[[Path], None]) -> None:
    with tempfile.TemporaryDirectory(prefix="odc-project-state-") as raw:
        fixture = Path(raw)
        make_fixture(fixture)
        mutate(fixture)
        result = run_validator(fixture)
        if result.returncode == 0:
            raise AssertionError(f"negative fixture passed: {name}")


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise AssertionError(f"fixture text not found in {path}: {old}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="odc-project-state-") as raw:
        fixture = Path(raw)
        make_fixture(fixture)
        result = run_validator(fixture)
        if result.returncode != 0:
            raise AssertionError(f"valid fixture failed: {result.stderr}")

    expect_failure(
        "ticket and spec status mismatch",
        lambda root: replace(
            root / "docs" / "specs" / "ODC-0002-v2-baseline.md",
            "status: SPEC_REVIEW",
            "status: SPEC_DRAFT",
        ),
    )
    expect_failure(
        "broken local Markdown link",
        lambda root: (root / "README.md").write_text(
            (root / "README.md").read_text(encoding="utf-8")
            + "\n[missing](docs/does-not-exist.md)\n",
            encoding="utf-8",
        ),
    )
    expect_failure(
        "approved spec with unresolved questions",
        lambda root: replace(
            root / "docs" / "specs" / "ODC-0000-project-operating-system.md",
            "unresolved_questions: none",
            "unresolved_questions: open decision",
        ),
    )
    expect_failure(
        "unknown public dependency",
        lambda root: replace(
            root / "Tickets.md",
            "| ODC-0003 | benchmark | Cross-backend benchmark contract | P0 | BACKLOG | P0 | ODC-0002 |",
            "| ODC-0003 | benchmark | Cross-backend benchmark contract | P0 | BACKLOG | P0 | ODC-9999 |",
        ),
    )

    print("project-state validator tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
