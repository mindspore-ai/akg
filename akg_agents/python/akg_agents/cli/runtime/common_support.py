from __future__ import annotations

import difflib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @classmethod
    def from_cwd(cls) -> "WorkspacePaths":
        return cls(Path(os.getcwd()).resolve())

    def normalize(self, path_value: str | Path) -> Path:
        path = Path(str(path_value)).expanduser()
        if path.is_absolute():
            return path
        return self.root / path

    def ensure_within(self, path: Path) -> tuple[bool, str]:
        resolved = self._safe_resolve(path)
        try:
            resolved.relative_to(self.root)
        except Exception:
            return False, self._outside_error(resolved)
        return True, ""

    def display(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except Exception:
            return str(path)

    @staticmethod
    def _safe_resolve(path: Path) -> Path:
        try:
            return path.resolve()
        except Exception:
            return path

    def _outside_error(self, resolved: Path) -> str:
        return (
            f"[ERROR] sandbox: path outside workspace: {resolved} (workspace={self.root})"
        )


def read_file_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def format_numbered_lines(lines: list[str], start_line_no: int) -> str:
    formatted: list[str] = []
    line_no = start_line_no
    for line in lines:
        trimmed = line.rstrip("\n")
        if len(trimmed) > 2000:
            trimmed = trimmed[:2000]
        formatted.append(f"{line_no:6}\t{trimmed}")
        line_no += 1
    return "\n".join(formatted)


def diff_text(path: Path, before: str, after: str, paths: WorkspacePaths) -> str:
    from_file = f"a/{paths.display(path)}"
    to_file = f"b/{paths.display(path)}"
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=from_file,
            tofile=to_file,
            lineterm="",
        )
    )
    return "\n".join(diff_lines).strip()


class DeltaFormatter:
    _cmd_cache: str | None = None

    def format(self, diff_text: str) -> str:
        text = (diff_text or "").strip()
        if not text or text.startswith("["):
            return diff_text
        cmd = self._get_cmd()
        if not cmd:
            return diff_text
        return self._run_delta(cmd, diff_text)

    def _get_cmd(self) -> str:
        if self._cmd_cache is not None:
            return self._cmd_cache
        self._cmd_cache = self._find_cmd() or ""
        return self._cmd_cache

    @staticmethod
    def _find_cmd() -> str:
        for name in ("delta", "git-delta"):
            if shutil.which(name):
                return name
        return ""

    @staticmethod
    def _run_delta(cmd: str, diff_text: str) -> str:
        try:
            result = subprocess.run(
                [cmd, "--paging=never", "--color-only"],
                input=diff_text,
                text=True,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            return diff_text
        if result.returncode == 0 and (result.stdout or "").strip():
            return result.stdout.rstrip()
        return diff_text
