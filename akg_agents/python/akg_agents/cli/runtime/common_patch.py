from __future__ import annotations

import re
import subprocess
from pathlib import Path

from akg_agents.cli.runtime.common_support import (
    WorkspacePaths,
    diff_text,
    read_text,
    write_text,
)


def parse_hunk_header(line: str) -> tuple[int, int]:
    match = re.match(r"@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@", line)
    if not match:
        raise ValueError(f"Invalid hunk header: {line}")
    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    return old_start, old_count


class HunkApplier:
    def __init__(self, original: list[str], hunk_lines: list[str]):
        self.original = original
        self.hunk_lines = hunk_lines
        self.new_lines: list[str] = []
        self.src_idx = 0
        self.hunk_idx = 0

    def apply(self) -> list[str]:
        while self.hunk_idx < len(self.hunk_lines):
            if self._current().startswith("@@"):
                self._apply_block()
            else:
                self.hunk_idx += 1
        self.new_lines.extend(self.original[self.src_idx :])
        return self.new_lines

    def _apply_block(self) -> None:
        header = self._current()
        old_start, _old_count = parse_hunk_header(header)
        self._copy_until(max(old_start - 1, 0))
        self.hunk_idx += 1
        while self._has_more() and not self._current().startswith("@@"):
            self._apply_hunk_line(self._current())
            self.hunk_idx += 1

    def _apply_hunk_line(self, hline: str) -> None:
        if hline.startswith(" "):
            self._expect_line(hline[1:], "Context mismatch when applying patch")
            self._copy_line()
            return
        if hline.startswith("-"):
            self._expect_line(hline[1:], "Delete mismatch when applying patch")
            self.src_idx += 1
            return
        if hline.startswith("+"):
            self.new_lines.append(hline[1:] + "\n")
            return
        raise ValueError(f"Unexpected hunk line: {hline}")

    def _copy_until(self, target_idx: int) -> None:
        while self.src_idx < target_idx and self.src_idx < len(self.original):
            self.new_lines.append(self.original[self.src_idx])
            self.src_idx += 1

    def _copy_line(self) -> None:
        self.new_lines.append(self.original[self.src_idx])
        self.src_idx += 1

    def _expect_line(self, expected: str, error: str) -> None:
        if self.src_idx >= len(self.original):
            raise ValueError(error)
        if self.original[self.src_idx].rstrip("\n") != expected:
            raise ValueError(error)

    def _current(self) -> str:
        return self.hunk_lines[self.hunk_idx]

    def _has_more(self) -> bool:
        return self.hunk_idx < len(self.hunk_lines)


def apply_hunks_to_lines(original_lines: list[str], hunk_lines: list[str]) -> list[str]:
    return HunkApplier(original_lines, hunk_lines).apply()


def _collect_block_lines(lines: list[str], idx: int) -> tuple[list[str], int]:
    block: list[str] = []
    while idx < len(lines) and not lines[idx].startswith("*** "):
        block.append(lines[idx])
        idx += 1
    return block, idx


def _parse_codex_header(line: str) -> tuple[str, str] | None:
    if line.startswith("*** Update File: "):
        return ("update", line[len("*** Update File: ") :].strip())
    if line.startswith("*** Add File: "):
        return ("add", line[len("*** Add File: ") :].strip())
    if line.startswith("*** Delete File: "):
        return ("delete", line[len("*** Delete File: ") :].strip())
    if line.startswith("*** End Patch"):
        return ("end", "")
    return None


def _iter_codex_blocks(lines: list[str]):
    idx = 1
    while idx < len(lines):
        header = _parse_codex_header(lines[idx])
        if not header:
            idx += 1
            continue
        kind, file_path = header
        if kind == "end":
            break
        idx += 1
        block_lines, idx = _collect_block_lines(lines, idx)
        yield kind, file_path, block_lines


def _validate_codex_header(lines: list[str]) -> str | None:
    if not lines or lines[0].strip() != "*** Begin Patch":
        return "[ERROR] apply_patch: invalid patch header"
    return None


def _apply_update_block(paths: WorkspacePaths, file_path: str, hunk_lines: list[str]) -> str | None:
    path = paths.normalize(file_path)
    if not path.exists():
        return f"[ERROR] apply_patch: file not found: {path}"
    ok, err = paths.ensure_within(path)
    if not ok:
        return err
    original = read_text(path).splitlines(keepends=True)
    try:
        updated = apply_hunks_to_lines(original, hunk_lines)
    except Exception as exc:
        msg = str(exc)
        if "Invalid hunk header" in msg:
            msg += ". Expected format: @@ -old_start,old_count +new_start,new_count @@"
        return f"[ERROR] apply_patch: {msg}"
    write_text(path, "".join(updated))
    return None


def _apply_add_block(paths: WorkspacePaths, file_path: str, block_lines: list[str]) -> str | None:
    path = paths.normalize(file_path)
    ok, err = paths.ensure_within(path)
    if not ok:
        return err
    content = [line[1:] if line.startswith("+") else line for line in block_lines]
    suffix = "\n" if content else ""
    write_text(path, "\n".join(content) + suffix)
    return None


def _apply_delete_block(paths: WorkspacePaths, file_path: str) -> str | None:
    path = paths.normalize(file_path)
    ok, err = paths.ensure_within(path)
    if not ok:
        return err
    if path.exists():
        path.unlink()
    return None


def apply_codex_patch(patch_text: str, paths: WorkspacePaths) -> str:
    lines = patch_text.splitlines()
    header_error = _validate_codex_header(lines)
    if header_error:
        return header_error
    for kind, file_path, block in _iter_codex_blocks(lines):
        if kind == "update":
            error = _apply_update_block(paths, file_path, block)
        elif kind == "add":
            error = _apply_add_block(paths, file_path, block)
        else:
            error = _apply_delete_block(paths, file_path)
        if error:
            return error
    return "[SUCCESS] apply_patch: patch applied"


def _normalize_patch_path(paths: WorkspacePaths, file_path: str) -> Path:
    return paths.normalize(file_path)


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _collect_codex_paths(patch_text: str, paths: WorkspacePaths) -> list[Path]:
    items: list[Path] = []
    for line in patch_text.splitlines():
        header = _parse_codex_header(line)
        if not header or header[0] == "end":
            continue
        file_path = header[1]
        if file_path:
            items.append(_normalize_patch_path(paths, file_path))
    return _unique_paths(items)


def _collect_unified_paths(patch_text: str, paths: WorkspacePaths) -> list[Path]:
    items: list[Path] = []
    for line in patch_text.splitlines():
        if not line.startswith("--- ") and not line.startswith("+++ "):
            continue
        file_path = line[4:].split("\t")[0].strip()
        if file_path in {"", "/dev/null"}:
            continue
        if file_path.startswith("a/") or file_path.startswith("b/"):
            file_path = file_path[2:]
        if file_path:
            items.append(_normalize_patch_path(paths, file_path))
    return _unique_paths(items)


def collect_before_contents(paths: list[Path]) -> dict[Path, str]:
    return {path: read_text(path) if path.exists() else "" for path in paths}


def build_patch_diff(paths: list[Path], before: dict[Path, str], ws: WorkspacePaths) -> str:
    diffs: list[str] = []
    for path in paths:
        after = read_text(path) if path.exists() else ""
        diff = diff_text(path, before.get(path, ""), after, ws)
        if diff:
            diffs.append(diff)
    return "\n\n".join(diffs).strip()


def _ensure_paths_within(paths: WorkspacePaths, targets: list[Path]) -> str | None:
    for path in targets:
        ok, err = paths.ensure_within(path)
        if not ok:
            return err
    return None


def _run_system_patch(patch_text: str) -> tuple[int, str, str]:
    result = subprocess.run(
        ["patch", "-p0", "-u", "-"],
        input=patch_text,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def _patch_error(stderr: str) -> str:
    hint = ""
    if "Only garbage was found" in stderr or "malformed patch" in stderr:
        hint = "\nHint: use standard unified diff hunks: @@ -old_start,old_count +new_start,new_count @@"
    return f"[ERROR] apply_patch: patch failed\n{stderr}{hint}".rstrip()


def apply_patch(patch_text: str, paths: WorkspacePaths) -> str:
    if not patch_text:
        return "[ERROR] apply_patch: patch content is required"
    text = str(patch_text)
    if "*** Begin Patch" in text:
        return _apply_codex_wrapper(text, paths)
    return _apply_unified_wrapper(text, paths)


def _apply_codex_wrapper(patch_text: str, paths: WorkspacePaths) -> str:
    targets = _collect_codex_paths(patch_text, paths)
    if not targets:
        return "[ERROR] apply_patch: no target files detected"
    err = _ensure_paths_within(paths, targets)
    if err:
        return err
    before = collect_before_contents(targets)
    result = apply_codex_patch(patch_text, paths)
    if result.startswith("[ERROR]"):
        return result
    diff = build_patch_diff(targets, before, paths)
    return diff or "[INFO] apply_patch: no changes"


def _apply_unified_wrapper(patch_text: str, paths: WorkspacePaths) -> str:
    targets = _collect_unified_paths(patch_text, paths)
    if not targets:
        return "[ERROR] apply_patch: no target files detected"
    err = _ensure_paths_within(paths, targets)
    if err:
        return err
    before = collect_before_contents(targets)
    try:
        code, _stdout, stderr = _run_system_patch(patch_text)
    except FileNotFoundError:
        return "[ERROR] apply_patch: system patch command not found"
    except Exception as exc:
        return f"[ERROR] apply_patch: {exc}"
    if code != 0:
        return _patch_error(stderr.strip())
    diff = build_patch_diff(targets, before, paths)
    return diff or "[INFO] apply_patch: no changes"
