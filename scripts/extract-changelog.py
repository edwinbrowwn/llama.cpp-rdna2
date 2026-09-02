#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def find_section(markdown: str, tag: str) -> str | None:
    bare_tag = tag[1:] if tag.startswith("v") else tag
    headers = {
        tag,
        bare_tag,
        f"[{tag}]",
        f"[{bare_tag}]",
    }

    lines = markdown.splitlines()
    start = None
    header_re = re.compile(r"^##\s+(.+?)\s*$")
    for index, line in enumerate(lines):
        match = header_re.match(line)
        if not match:
            continue
        title = match.group(1).strip()
        first_token = title.split(maxsplit=1)[0]
        if title in headers or first_token in headers:
            start = index + 1
            break

    if start is None:
        return None

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break

    section = "\n".join(lines[start:end]).strip()
    return section or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--changelog", required=True)
    parser.add_argument("--fallback-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    changelog_path = Path(args.changelog)
    output_path = Path(args.output)

    section = None
    if changelog_path.exists():
        section = find_section(changelog_path.read_text(encoding="utf-8"), args.tag)

    if section is None:
        section = f"[CHANGELOG.md]({args.fallback_url})"

    output_path.write_text(section.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
