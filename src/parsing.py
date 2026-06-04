"""Parser for the Writer agent's structured Markdown output.

The Writer is instructed to emit:

    ### RATIONALE
    <one short paragraph>

    ### ENTRY
    run.py

    ### FILE: run.py
    ```python
    ...
    ```

    ### FILE: helpers.py
    ```python
    ...
    ```

This module turns that text into a (rationale, entry_point, code_files)
triple. It is intentionally permissive about whitespace and tolerates
optional language tags on the fences (```python, ```, ```py).

Parsing failures raise `ParseError`. The Writer is expected to be re-prompted
when this happens, not silently fixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Match `### RATIONALE`, `### ENTRY`, `### FILE: <name>` headers (case-insensitive,
# allow extra whitespace). The header sits on its own line.
HEADER_RE = re.compile(
    r"^[ \t]*###[ \t]+(RATIONALE|ENTRY|FILE)\b[ \t]*(?::[ \t]*(?P<name>.+?))?[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)

# A fenced code block: ```optional-lang\n ... \n```
FENCE_RE = re.compile(
    r"```[ \t]*([\w+\-.]+)?[ \t]*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


class ParseError(ValueError):
    """Raised when the Writer's output cannot be parsed into the expected shape."""


@dataclass
class ParsedWriterOutput:
    rationale: str
    entry_point: str
    code_files: dict[str, str]


def parse_writer_output(text: str) -> ParsedWriterOutput:
    """Parse a Writer LLM response into structured fields.

    Raises:
        ParseError: if no `### ENTRY` block, no `### FILE:` blocks, the
            declared entry point is missing from the file list, or any
            `### FILE:` section lacks a fenced code block.
    """
    sections = _split_sections(text)
    if not sections:
        raise ParseError("Writer output has no recognised `### ` section headers.")

    rationale = ""
    entry_point: str | None = None
    code_files: dict[str, str] = {}

    for tag, name, body in sections:
        tag_upper = tag.upper()
        if tag_upper == "RATIONALE":
            rationale = body.strip()
        elif tag_upper == "ENTRY":
            entry_point = _extract_entry_name(body)
        elif tag_upper == "FILE":
            if not name:
                raise ParseError("`### FILE:` section is missing its filename.")
            filename = name.strip()
            _check_relative_path(filename)
            code_files[filename] = _extract_fenced_block(body, filename)

    if entry_point is None:
        raise ParseError("Writer output has no `### ENTRY` block declaring the entry point.")
    if not code_files:
        raise ParseError("Writer output has no `### FILE:` blocks.")
    if entry_point not in code_files:
        raise ParseError(
            f"Declared ENTRY '{entry_point}' is not present among FILE blocks "
            f"(have: {sorted(code_files)})."
        )

    return ParsedWriterOutput(
        rationale=rationale,
        entry_point=entry_point,
        code_files=code_files,
    )


def _split_sections(text: str) -> list[tuple[str, str | None, str]]:
    """Yield (tag, name_or_None, body_text) tuples for each top-level header."""
    matches = list(HEADER_RE.finditer(text))
    out: list[tuple[str, str | None, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end]
        out.append((m.group(1), m.group("name"), body))
    return out


def _extract_entry_name(body: str) -> str:
    # The ENTRY block's body should be a single filename, possibly inside a
    # fenced block. Be tolerant.
    fenced = FENCE_RE.search(body)
    raw = fenced.group("body") if fenced else body
    # Strip backticks and whitespace.
    cleaned = raw.strip().strip("`").strip()
    if "\n" in cleaned:
        # Sometimes the LLM puts trailing comments; take the first non-empty line.
        for line in cleaned.splitlines():
            if line.strip():
                cleaned = line.strip()
                break
    if not cleaned:
        raise ParseError("`### ENTRY` block is empty.")
    _check_relative_path(cleaned)
    return cleaned


def _extract_fenced_block(body: str, filename: str) -> str:
    m = FENCE_RE.search(body)
    if m is None:
        raise ParseError(
            f"`### FILE: {filename}` section has no fenced code block."
        )
    return m.group("body")


def _check_relative_path(path: str) -> None:
    if path.startswith("/"):
        raise ParseError(f"Absolute paths not allowed: {path!r}.")
    if ".." in path.split("/"):
        raise ParseError(f"`..` not allowed in file paths: {path!r}.")
    if "\\" in path:
        raise ParseError(f"Backslashes not allowed in file paths (use /): {path!r}.")
