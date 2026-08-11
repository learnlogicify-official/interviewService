"""Safe-ish local Python runner for interview coding tests."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


FORBIDDEN = (
    "import os",
    "import subprocess",
    "import socket",
    "import pathlib",
    "__import__",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "system(",
)


def _normalize(text: str) -> str:
    text = text.strip()
    try:
        return repr(ast.literal_eval(text))
    except Exception:
        return " ".join(text.split())


def run_python(code: str, stdin_data: str, timeout: float = 2.0) -> tuple[bool, str, str]:
    lowered = code.lower()
    for bad in FORBIDDEN:
        if bad in lowered:
            return False, "", f"Forbidden pattern: {bad}"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "main.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                input=stdin_data,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "", "Time limit exceeded"
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "Runtime error").strip()
            return False, "", err[-500:]
        return True, (proc.stdout or "").strip(), ""


def evaluate_code(code: str, tests: list[dict[str, str]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    passed = 0
    for i, t in enumerate(tests):
        ok, out, err = run_python(code, t.get("stdin", ""))
        expected = t.get("expected", "")
        match = ok and _normalize(out) == _normalize(expected)
        if match:
            passed += 1
        details.append(
            {
                "index": i,
                "passed": match,
                "stdin": t.get("stdin", ""),
                "expected": expected,
                "stdout": out,
                "error": err,
            }
        )
    total = len(tests)
    return {
        "ok": passed == total and total > 0,
        "passed": passed,
        "total": total,
        "details": details,
        "message": f"{passed}/{total} tests passed",
    }
