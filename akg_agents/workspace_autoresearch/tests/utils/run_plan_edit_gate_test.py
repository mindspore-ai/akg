#!/usr/bin/env python3
"""Focused checks for plan skill binding and effective edit gating."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import shutil
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "engine"))

from create_plan import _validate_items  # noqa: E402
from quick_check import effective_edit_issue  # noqa: E402


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True,
                   capture_output=True, text=True)


def _repo_with_kernel(content: str) -> Path:
    td = Path(tempfile.mkdtemp(prefix="ar_edit_gate_"))
    _run(["git", "init"], td)
    _run(["git", "config", "user.name", "test"], td)
    _run(["git", "config", "user.email", "test@example.invalid"], td)
    (td / "kernel.py").write_text(content, encoding="utf-8")
    _run(["git", "add", "kernel.py"], td)
    _run(["git", "commit", "-m", "seed"], td)
    return td


def _issue_after(new_content: str | None) -> dict | None:
    td = _repo_with_kernel("x = 1\n")
    try:
        if new_content is not None:
            (td / "kernel.py").write_text(new_content, encoding="utf-8")
        cfg = SimpleNamespace(editable_files=["kernel.py"])
        return effective_edit_issue(str(td), cfg)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_effective_edit_gate() -> None:
    clean = _issue_after(None)
    assert clean and "Zero-edit" in clean["report"]

    comments = _issue_after("x = 1\n# cite direct-invoke/SKILL.md\n")
    assert comments and "Comment-only edit" in comments["report"]

    code = _issue_after("x = 2\n")
    assert code is None


def test_plan_requires_skill_field() -> None:
    base = {
        "desc": "Fuse the vector pass into one kernel",
        "rationale": "This removes redundant global-memory traffic.",
    }
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            _validate_items([dict(base), dict(base), dict(base)],
                            require_skill=True)
        except SystemExit:
            pass
        else:
            raise AssertionError("missing <skill> was accepted")

    cited = {
        "desc": "Fuse the vector pass into one kernel",
        "rationale": (
            "This removes redundant global-memory traffic by fusing "
            "the producer and consumer vector passes."
        ),
        "skill": "ascendc-direct-invoke/SKILL.md",
    }
    _validate_items([dict(cited), dict(cited), dict(cited)],
                    require_skill=True)


def main() -> int:
    test_effective_edit_gate()
    test_plan_requires_skill_field()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
