"""Copy JSON metadata while replacing machine-specific paths with portable roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _replace(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key.lower() == "email" and isinstance(item, str):
                sanitized[key] = "${NCBI_EMAIL}"
            else:
                sanitized[key] = _replace(item, replacements)
        return sanitized
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target)
        return value
    return value


def sanitize_file(source: Path, destination: Path, replacements: list[tuple[str, str]]) -> None:
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if source.name == "generation_metadata.json" and isinstance(payload, dict):
        payload.pop("split", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_replace(payload, replacements), handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()

    project_root = str(args.project_root).rstrip("/")
    scratch_root = "/" + "scratch/local/krissmer"
    replacements = [
        (project_root, "${PROJECT_ROOT}"),
        (scratch_root + "/alias" + "-private", "${PROJECT_ROOT}"),
        (scratch_root + "/alias", "${PROJECT_ROOT}"),
    ]
    for source in args.source_root.rglob("*.json"):
        destination = args.destination_root / source.relative_to(args.source_root)
        sanitize_file(source, destination, replacements)


if __name__ == "__main__":
    main()
