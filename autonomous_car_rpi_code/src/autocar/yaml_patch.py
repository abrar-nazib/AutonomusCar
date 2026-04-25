"""Minimal YAML value-patching helper.

PyYAML's `safe_dump` would throw away comments and reorder keys. The autocar
config has a lot of inline comments we'd like to keep. This helper rewrites
only the scalar values for known keys, line by line, so the file stays
otherwise untouched.

Supported keys are the ones in `tuning.KNOBS` — their dotted paths identify
a `<section>.<leaf>` or `<section>.pid.<leaf>` location in the yaml. We
track section indentation by blank-line / unindent boundaries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


_LEAF_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s*(?P<value>.*?)\s*(?P<comment>#.*)?$")


def patch_yaml_file(path: Path, updates: Dict[str, object]) -> None:
    """updates = {"vision.sheet_threshold": 95, "control.pid.kp": 1.4, ...}"""
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    # Build a lookup keyed by the FULL dotted path so we can match nested
    # sections (control.pid.kp etc.).
    remaining = dict(updates)

    # Stack of (indent, key) frames so a line's full path is known.
    stack: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.rstrip("\n\r")
        if not stripped.strip() or stripped.strip().startswith("#"):
            continue
        m = _LEAF_RE.match(stripped)
        if not m:
            continue
        indent = len(m.group("indent").expandtabs(2))
        key = m.group("key")
        value = m.group("value")
        # Pop frames whose indent is >= this line's indent.
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full_path = ".".join([k for _, k in stack] + [key])

        if value == "" or value is None:
            # Mapping header: push onto stack.
            stack.append((indent, key))
            continue

        if full_path in remaining:
            new_val = remaining.pop(full_path)
            new_val_str = _format_scalar(new_val)
            comment = m.group("comment") or ""
            rebuilt = f"{m.group('indent')}{key}: {new_val_str}"
            if comment:
                rebuilt = f"{rebuilt}  {comment}"
            lines[i] = rebuilt + ("\n" if line.endswith("\n") else "")
            continue

    if remaining:
        # Any keys not found are appended under a "# auto-added by yaml_patch"
        # footer so they're still saved rather than lost.
        footer = ["\n# --- auto-added by yaml_patch ---\n"]
        for k, v in remaining.items():
            footer.append(f"# {k}: {_format_scalar(v)}\n")
        lines.extend(footer)

    path.write_text("".join(lines))


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Drop trailing zeros but keep at least one decimal place.
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if any(c in value for c in " :#") or value == "":
            return f'"{value}"'
        return value
    return repr(value)
