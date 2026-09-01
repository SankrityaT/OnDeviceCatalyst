#!/usr/bin/env python3
"""Validate OnDeviceCatalyst's tracked roadmap, tickets, specs, and ADRs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TICKETS_PATH = ROOT / "Tickets.md"
ROADMAP_PATH = ROOT / "ROADMAP.md"
SPEC_DIR = ROOT / "docs" / "specs"
ADR_DIR = ROOT / "docs" / "decisions"

ALLOWED_STATUSES = {
    "BACKLOG",
    "DISCOVERY",
    "SPEC_DRAFT",
    "SPEC_REVIEW",
    "REVISION",
    "APPROVED",
    "IMPLEMENTING",
    "VALIDATING",
    "DONE",
    "BLOCKED",
    "DEFERRED",
    "REJECTED",
}
SPEC_REQUIRED_STATUSES = {
    "SPEC_DRAFT",
    "SPEC_REVIEW",
    "REVISION",
    "APPROVED",
    "IMPLEMENTING",
    "VALIDATING",
    "DONE",
}
APPROVED_STATUSES = {"APPROVED", "IMPLEMENTING", "VALIDATING", "DONE"}
TICKET_PATTERN = re.compile(r"^ODC-\d{4}$")
ADR_PATTERN = re.compile(r"^ODC-ADR-\d{4}$")
MARKDOWN_LINK_PATTERN = re.compile(r"\]\(([^)]+)\)")
SPEC_LINK_PATTERN = re.compile(r"\[spec\]\(([^)]+)\)")


class ValidationError(Exception):
    pass


def parse_frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValidationError(f"{path.relative_to(ROOT)}: missing frontmatter")

    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return metadata
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValidationError(
                f"{path.relative_to(ROOT)}: invalid frontmatter line: {line}"
            )
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    raise ValidationError(f"{path.relative_to(ROOT)}: unterminated frontmatter")


def parse_ticket_rows() -> dict[str, dict[str, str]]:
    columns = [
        "id",
        "type",
        "title",
        "milestone",
        "status",
        "priority",
        "dependencies",
        "spec",
        "issue",
        "owner",
        "updated",
        "next_gate",
    ]
    tickets: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(
        TICKETS_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not re.match(r"^\|\s*ODC-\d{4}\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(columns):
            raise ValidationError(
                f"Tickets.md:{line_number}: expected {len(columns)} cells, "
                f"found {len(cells)}"
            )
        ticket = dict(zip(columns, cells))
        ticket_id = ticket["id"]
        if ticket_id in tickets:
            raise ValidationError(f"Tickets.md:{line_number}: duplicate {ticket_id}")
        if not TICKET_PATTERN.fullmatch(ticket_id):
            raise ValidationError(f"Tickets.md:{line_number}: invalid ID {ticket_id}")
        if ticket["status"] not in ALLOWED_STATUSES:
            raise ValidationError(
                f"Tickets.md:{line_number}: invalid status {ticket['status']}"
            )
        tickets[ticket_id] = ticket
    if not tickets:
        raise ValidationError("Tickets.md: no ticket rows found")
    return tickets


def validate_specs(tickets: dict[str, dict[str, str]]) -> None:
    found_specs: dict[str, tuple[Path, dict[str, str]]] = {}
    for path in sorted(SPEC_DIR.glob("ODC-*.md")):
        metadata = parse_frontmatter(path)
        ticket_id = metadata.get("id", "")
        if not TICKET_PATTERN.fullmatch(ticket_id):
            raise ValidationError(
                f"{path.relative_to(ROOT)}: invalid or missing ticket ID"
            )
        if ticket_id in found_specs:
            raise ValidationError(f"duplicate spec ID {ticket_id}")
        found_specs[ticket_id] = (path, metadata)

    for ticket_id, ticket in tickets.items():
        match = SPEC_LINK_PATTERN.fullmatch(ticket["spec"])
        if ticket["status"] in SPEC_REQUIRED_STATUSES and not match:
            raise ValidationError(
                f"Tickets.md: {ticket_id} status {ticket['status']} requires a spec link"
            )
        if not match:
            continue
        spec_path = ROOT / match.group(1)
        if not spec_path.is_file():
            raise ValidationError(f"Tickets.md: {ticket_id} missing {match.group(1)}")
        if ticket_id not in found_specs:
            raise ValidationError(f"Tickets.md: {ticket_id} spec ID not discovered")
        actual_path, metadata = found_specs[ticket_id]
        if actual_path.resolve() != spec_path.resolve():
            raise ValidationError(f"Tickets.md: {ticket_id} links the wrong spec")
        if metadata.get("status") != ticket["status"]:
            raise ValidationError(
                f"{ticket_id}: ticket status {ticket['status']} does not match "
                f"spec status {metadata.get('status', '<missing>')}"
            )
        required = {
            "title",
            "type",
            "milestone",
            "owner",
            "dependencies",
            "founder_approved",
            "last_updated",
            "evidence_fresh_until",
            "unresolved_questions",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise ValidationError(
                f"{actual_path.relative_to(ROOT)}: missing metadata {', '.join(missing)}"
            )
        if ticket["status"] in APPROVED_STATUSES:
            if metadata["founder_approved"] == "pending":
                raise ValidationError(f"{ticket_id}: approved state lacks founder approval")
            if metadata["unresolved_questions"] != "none":
                raise ValidationError(f"{ticket_id}: approved state has open questions")

    unlisted = sorted(set(found_specs) - set(tickets))
    if unlisted:
        raise ValidationError(f"spec IDs missing from Tickets.md: {', '.join(unlisted)}")


def validate_adrs() -> None:
    found: set[str] = set()
    for path in sorted(ADR_DIR.glob("ODC-ADR-*.md")):
        metadata = parse_frontmatter(path)
        adr_id = metadata.get("id", "")
        if not ADR_PATTERN.fullmatch(adr_id):
            raise ValidationError(f"{path.relative_to(ROOT)}: invalid ADR ID {adr_id}")
        if adr_id in found:
            raise ValidationError(f"duplicate ADR ID {adr_id}")
        found.add(adr_id)
        if metadata.get("status") not in {"proposed", "accepted", "superseded"}:
            raise ValidationError(f"{adr_id}: invalid ADR status")


def validate_dependencies(tickets: dict[str, dict[str, str]]) -> None:
    for ticket_id, ticket in tickets.items():
        if ticket["dependencies"] == "none":
            continue
        for dependency in [item.strip() for item in ticket["dependencies"].split(",")]:
            if dependency.startswith("ODR-"):
                continue
            if dependency not in tickets:
                raise ValidationError(f"{ticket_id}: unknown dependency {dependency}")


def validate_markdown_links() -> None:
    paths = [
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob(".github/**/*.md"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "—" in text:
            raise ValidationError(
                f"{path.relative_to(ROOT)}: em dash violates repository writing style"
            )
        for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_without_anchor = target.split("#", 1)[0]
            if not target_without_anchor:
                continue
            resolved = (path.parent / target_without_anchor).resolve()
            if not resolved.exists():
                raise ValidationError(
                    f"{path.relative_to(ROOT)}: broken local link {target}"
                )


def validate_roadmap(tickets: dict[str, dict[str, str]]) -> None:
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    match = re.search(r"Current public ticket:\s*(ODC-\d{4})", roadmap)
    if not match:
        raise ValidationError("ROADMAP.md: missing Current public ticket")
    if match.group(1) not in tickets:
        raise ValidationError(
            f"ROADMAP.md: current ticket {match.group(1)} is not in Tickets.md"
        )


def main() -> int:
    try:
        tickets = parse_ticket_rows()
        validate_specs(tickets)
        validate_adrs()
        validate_dependencies(tickets)
        validate_markdown_links()
        validate_roadmap(tickets)
    except ValidationError as error:
        print(f"project-state validation failed: {error}", file=sys.stderr)
        return 1

    spec_count = len(list(SPEC_DIR.glob("ODC-*.md")))
    adr_count = len(list(ADR_DIR.glob("ODC-ADR-*.md")))
    print(
        f"project state valid: {len(tickets)} tickets, "
        f"{spec_count} specs, {adr_count} ADRs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
