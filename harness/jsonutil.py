"""Lenient JSON parsing for model outputs (fences, trailing commas, unquoted keys)."""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def _strip_fences(s: str) -> str:
    return s.replace("```json", "").replace("```", "").strip()


def _balanced_slice(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _sanitize(s: str) -> str:
    s = _strip_fences(s)
    bal = _balanced_slice(s)
    if bal:
        s = bal
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"(?<=\d)\.(?=[,\]\}\s])", ".0", s)
    s = re.sub(r"(?<=[:\s\[])\.(\d)", r"0.\1", s)
    s = re.sub(r"([{,]\s*)([A-Za-z_][\w\-]*)(\s*:)", r'\1"\2"\3', s)
    return s


def parse_json_lenient(text: Optional[str]) -> Optional[Any]:
    """Return the parsed object or None.  Never raises."""
    if not text:
        return None
    for candidate in (_strip_fences(text), _balanced_slice(text) or "", _sanitize(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None
