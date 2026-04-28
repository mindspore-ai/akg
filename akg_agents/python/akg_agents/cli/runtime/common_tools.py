from __future__ import annotations

import fnmatch
import glob
import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool

from akg_agents.cli.runtime.common_bash import (
    analyze_bash_command,
    classify_path_arg,
    unwrap_command_words,
)
from akg_agents.cli.runtime.common_constants import GENERIC_ARGS_SCHEMA
from akg_agents.cli.runtime.common_patch import apply_patch
from akg_agents.cli.runtime.common_support import (
    WorkspacePaths,
    diff_text,
    format_numbered_lines,
    read_file_lines,
    read_text,
    write_text,
)


@dataclass
class CommonToolState:
    session_id: str
    todos: list[dict] = field(default_factory=list)
    tool_exec: dict[str, callable] = field(default_factory=dict)
    mode: str = "build"
    plan_path: Optional[str] = None


def _tool_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tool"


def _extract_path_arg(kwargs: dict) -> str | None:
    return kwargs.get("filePath") or kwargs.get("file_path") or kwargs.get("path")


def _parse_offset_limit(kwargs: dict) -> tuple[int, Optional[int]]:
    offset = _coerce_int(kwargs.get("offset"), 0)
    limit_raw = kwargs.get("limit")
    limit = _coerce_int(limit_raw, None) if limit_raw is not None else None
    return offset, limit


def _coerce_int(value: Any, default: Optional[int]) -> Optional[int]:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _format_lines_slice(lines: list[str], offset: int, limit: Optional[int]) -> str:
    start = max(offset, 0)
    end = start + limit if limit is not None and limit >= 0 else None
    chunk = lines[start:end]
    if not chunk:
        return "[EMPTY]"
    return format_numbered_lines(chunk, start + 1)


def _get_text_kwargs(kwargs: dict) -> tuple[str | None, Any]:
    return _extract_path_arg(kwargs), kwargs.get("content")


def _parse_edit_kwargs(kwargs: dict) -> tuple[str | None, str | None, str | None, bool]:
    file_path = _extract_path_arg(kwargs)
    old = kwargs.get("oldString") or kwargs.get("old")
    new = kwargs.get("newString") or kwargs.get("new")
    replace_all = bool(kwargs.get("replaceAll") or kwargs.get("replace_all"))
    return file_path, old, new, replace_all


def _apply_single_edit(content: str, old: str, new: str, replace_all: bool) -> tuple[str, str | None]:
    count = content.count(old)
    if count == 0:
        return content, "[ERROR] edit: oldString not found"
    if count > 1 and not replace_all:
        return content, "[ERROR] edit: oldString found multiple times; set replaceAll=true"
    updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
    return updated, None


def _normalize_edit(edit: dict) -> tuple[str | None, str | None, bool]:
    old = edit.get("oldString")
    new = edit.get("newString")
    replace_all = bool(edit.get("replaceAll"))
    return old, new, replace_all


def _apply_multiple_edits(content: str, edits: list[dict]) -> tuple[str, str | None]:
    updated = content
    for edit in edits:
        old, new, replace_all = _normalize_edit(edit)
        if old is None or new is None:
            return updated, "[ERROR] multiedit: each edit requires oldString and newString"
        updated, error = _apply_single_edit(updated, old, new, replace_all)
        if error:
            return updated, error.replace("edit:", "multiedit:")
    return updated, None


def _parse_todos_payload(kwargs: dict) -> Any:
    todos = kwargs.get("todos") or kwargs.get("items") or kwargs.get("tasks")
    if todos is None and "input" in kwargs:
        todos = kwargs.get("input")
        if isinstance(todos, dict):
            todos = todos.get("todos") or todos.get("tasks") or todos.get("items")
    return todos


def _parse_todos_input(todos: Any) -> tuple[Any, str | None]:
    if isinstance(todos, str):
        try:
            return json.loads(todos), None
        except Exception:
            return None, "[ERROR] todowrite: invalid todos JSON"
    if isinstance(todos, dict):
        items = todos.get("todos") or todos.get("tasks") or todos.get("items") or []
        return items, None
    return todos, None


def _normalize_todo_item(item: Any) -> dict | None:
    if isinstance(item, str):
        return _build_todo_dict(item, "pending", "medium", uuid.uuid4().hex)
    if isinstance(item, dict):
        return _normalize_todo_dict(item)
    return None


def _normalize_todo_dict(item: dict) -> dict | None:
    content = str(
        item.get("content")
        or item.get("name")
        or item.get("description")
        or item.get("text")
        or item.get("title")
        or ""
    )
    if not content:
        return None
    status = str(item.get("status") or "pending")
    priority = str(item.get("priority") or "medium")
    item_id = str(item.get("id") or uuid.uuid4().hex)
    return _build_todo_dict(content, status, priority, item_id)


def _build_todo_dict(content: str, status: str, priority: str, item_id: str) -> dict:
    return {
        "content": content,
        "status": status,
        "priority": priority,
        "id": item_id,
    }


def _parse_yes_no_reply(reply: str) -> Optional[bool]:
    text = (reply or "").strip().lower()
    if not text:
        return None
    negative = {"no", "n", "cancel", "reject", "stop", "否", "不", "取消", "拒绝"}
    positive = {"yes", "y", "ok", "okay", "confirm", "sure", "好", "好的", "是", "确认", "可以"}
    if any(tok in text for tok in negative):
        return False
    if any(tok in text for tok in positive):
        return True
    return None


def _tool_requires_approval(tool_name: str) -> bool:
    return tool_name in {"write", "edit", "multiedit", "apply_patch"}


def _wrap_tool_with_approval(tool_name: str, func: callable) -> callable:
    from functools import wraps
    from akg_agents.core.tools.basic_tools import request_tool_approval

    @wraps(func)
    def _wrapped(**kwargs):
        if tool_name == "bash":
            return func(**kwargs)
        if _tool_requires_approval(tool_name):
            if not request_tool_approval(tool_name, kwargs):
                return f"[CANCELLED] {tool_name}: execution denied by user"
        return func(**kwargs)

    return _wrapped


class BashTool:
    def __init__(self, paths: WorkspacePaths):
        self.paths = paths

    def run(self, **kwargs) -> str:
        command, cwd, cwd_path, timeout = self._parse_args(kwargs)
        if not command:
            return "[ERROR] bash: command is required"
        analysis = analyze_bash_command(command, cwd_path)
        approved = self._request_approval(command, cwd_path, timeout, analysis)
        if not approved:
            return "[CANCELLED] bash: execution denied by user"
        return self._execute(command, cwd, timeout)

    def _parse_args(self, kwargs: dict) -> tuple[str, str | None, Path, float]:
        command = kwargs.get("command") or kwargs.get("cmd") or ""
        workdir = kwargs.get("workdir") or kwargs.get("cwd")
        cwd, cwd_path = self._resolve_cwd(workdir)
        timeout = self._coerce_timeout(kwargs.get("timeout"))
        return command, cwd, cwd_path, timeout

    def _resolve_cwd(self, workdir: Any) -> tuple[str | None, Path]:
        if not workdir:
            cwd_path = self.paths.root
            return None, cwd_path
        cwd_path = Path(str(workdir)).expanduser()
        if not cwd_path.is_absolute():
            cwd_path = self.paths.root / cwd_path
        return str(cwd_path), cwd_path.resolve()

    def _coerce_timeout(self, timeout: Any) -> float:
        try:
            timeout_val = float(timeout) if timeout is not None else 120.0
            return timeout_val / 1000.0 if timeout_val > 1000 else timeout_val
        except Exception:
            return 120.0

    def _request_approval(
        self,
        command: str,
        cwd_path: Path,
        timeout: float,
        analysis: dict[str, Any] | None,
    ) -> bool:
        from akg_agents.core.tools.basic_tools import request_tool_approval
        approval_args = self._approval_args(command, cwd_path, timeout)
        if analysis is None:
            approval_args["note"] = "tree-sitter-bash parser unavailable; external path detection disabled"
            return request_tool_approval("bash:unparsed", approval_args)
        external, unknown = self._collect_path_issues(analysis, cwd_path)
        if external or unknown:
            return self._request_external_approval(approval_args, external, unknown)
        return request_tool_approval("bash", approval_args)

    def _approval_args(self, command: str, cwd_path: Path, timeout: float) -> dict[str, Any]:
        return {
            "cwd": str(cwd_path),
            "workspace": str(self.paths.root),
            "command": command,
            "timeout": timeout,
        }

    def _request_external_approval(
        self,
        approval_args: dict[str, Any],
        external: list[str],
        unknown: list[str],
    ) -> bool:
        from akg_agents.core.tools.basic_tools import request_tool_approval

        approval_args.update({
            "external_paths": self._dedupe(external),
            "unknown_paths": self._dedupe(unknown),
        })
        return request_tool_approval("bash:external_paths", approval_args)

    def _collect_path_issues(
        self,
        analysis: dict[str, Any],
        cwd_path: Path,
    ) -> tuple[list[str], list[str]]:
        external: list[str] = []
        unknown: list[str] = []
        for words in analysis.get("commands", []):
            external, unknown = self._classify_command(words, cwd_path, external, unknown)
        for target in analysis.get("redirects", []):
            self._classify_arg(target, cwd_path, external, unknown)
        return external, unknown

    def _classify_command(
        self,
        words: list[str],
        cwd_path: Path,
        external: list[str],
        unknown: list[str],
    ) -> tuple[list[str], list[str]]:
        words = unwrap_command_words(words)
        if not words:
            return external, unknown
        cmd = words[0]
        if cmd not in self._path_commands():
            return external, unknown
        for arg in words[1:]:
            if arg.startswith("-") or (cmd == "chmod" and arg.startswith("+")):
                continue
            self._classify_arg(arg, cwd_path, external, unknown)
        return external, unknown

    def _classify_arg(
        self,
        arg: str,
        cwd_path: Path,
        external: list[str],
        unknown: list[str],
    ) -> None:
        result = classify_path_arg(arg, cwd_path, self.paths)
        if not result:
            return
        kind, value = result
        if kind == "external":
            external.append(value)
        elif kind == "unknown":
            unknown.append(value)

    def _path_commands(self) -> set[str]:
        return {
            "cd",
            "rm",
            "cp",
            "mv",
            "mkdir",
            "rmdir",
            "touch",
            "chmod",
            "chown",
            "chgrp",
            "ln",
            "install",
        }

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))

    @staticmethod
    def _execute(command: str, cwd: str | None, timeout: float) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[ERROR] bash: command timed out after {timeout}s"
        except Exception as exc:
            return f"[ERROR] bash: {type(exc).__name__}: {exc}"
        output = "\n".join([p for p in [result.stdout, result.stderr] if p])
        output_text = output.strip() or "[EMPTY]"
        return f"[exit_code={result.returncode}]\n{output_text}"


class ToolRunner:
    def __init__(self, state: CommonToolState):
        self.state = state
        self.paths = WorkspacePaths.from_cwd()
        self.bash_tool = BashTool(self.paths)

    def _refresh_paths(self) -> WorkspacePaths:
        self.paths = WorkspacePaths.from_cwd()
        self.bash_tool.paths = self.paths
        return self.paths

    def read(self, **kwargs) -> str:
        paths = self._refresh_paths()
        file_path = _extract_path_arg(kwargs)
        if not file_path:
            return "[ERROR] read: filePath is required"
        path = paths.normalize(file_path)
        error = self._ensure_file(path, "read")
        if error:
            return error
        offset, limit = _parse_offset_limit(kwargs)
        return _format_lines_slice(read_file_lines(path), offset, limit)

    def write(self, **kwargs) -> str:
        paths = self._refresh_paths()
        file_path, content = _get_text_kwargs(kwargs)
        if not file_path:
            return "[ERROR] write: filePath is required"
        if content is None:
            return "[ERROR] write: content is required"
        path = paths.normalize(file_path)
        error = self._ensure_within(path)
        if error:
            return error
        before = read_text(path) if path.exists() else ""
        text = str(content)
        write_text(path, text)
        diff = diff_text(path, before, text, paths)
        return diff or f"[INFO] write: no changes for {path}"

    def edit(self, **kwargs) -> str:
        paths = self._refresh_paths()
        file_path, old, new, replace_all = _parse_edit_kwargs(kwargs)
        if not file_path:
            return "[ERROR] edit: filePath is required"
        if old is None or new is None:
            return "[ERROR] edit: oldString and newString are required"
        path = paths.normalize(file_path)
        error = self._ensure_file(path, "edit")
        if error:
            return error
        before = read_text(path)
        updated, err = _apply_single_edit(before, old, new, replace_all)
        if err:
            return err
        write_text(path, updated)
        diff = diff_text(path, before, updated, paths)
        return diff or f"[INFO] edit: no changes for {path}"

    def multiedit(self, **kwargs) -> str:
        paths = self._refresh_paths()
        file_path = _extract_path_arg(kwargs)
        edits = kwargs.get("edits")
        if not file_path:
            return "[ERROR] multiedit: filePath is required"
        if not isinstance(edits, list) or not edits:
            return "[ERROR] multiedit: edits must be a non-empty list"
        path = paths.normalize(file_path)
        error = self._ensure_file(path, "multiedit")
        if error:
            return error
        before = read_text(path)
        updated, err = _apply_multiple_edits(before, edits)
        if err:
            return err
        write_text(path, updated)
        diff = diff_text(path, before, updated, paths)
        return diff or f"[INFO] multiedit: no changes for {path}"

    def apply_patch(self, **kwargs) -> str:
        self._refresh_paths()
        patch_text = kwargs.get("patch") or kwargs.get("input") or kwargs.get("content")
        return apply_patch(str(patch_text or ""), self.paths)

    def ls(self, **kwargs) -> str:
        paths = self._refresh_paths()
        path = kwargs.get("path") or kwargs.get("dir") or kwargs.get("directory")
        ignore = kwargs.get("ignore") or []
        ignore_list = [ignore] if isinstance(ignore, str) else list(ignore)
        base = Path(str(path)).expanduser() if path else paths.root
        if not base.is_absolute():
            base = paths.root / base
        if not base.exists():
            return f"[ERROR] ls: path not found: {base}"
        if not base.is_dir():
            return f"[ERROR] ls: not a directory: {base}"
        items = [self._format_ls_entry(entry) for entry in base.iterdir()]
        items = sorted([item for item in items if item], key=str.lower)
        filtered = [item for item in items if not self._is_ignored(item, ignore_list)]
        return "\n".join(filtered) if filtered else "[EMPTY]"

    def glob(self, **kwargs) -> str:
        paths = self._refresh_paths()
        pattern = kwargs.get("pattern") or kwargs.get("glob")
        base = kwargs.get("path") or kwargs.get("dir") or kwargs.get("directory")
        if not pattern:
            return "[ERROR] glob: pattern is required"
        base_path = Path(str(base)).expanduser() if base else paths.root
        if not base_path.is_absolute():
            base_path = paths.root / base_path
        matches = [m for m in glob.glob(str(base_path / pattern), recursive=True) if Path(m).exists()]
        matches.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
        return "\n".join(matches) if matches else "[EMPTY]"

    def grep(self, **kwargs) -> str:
        paths = self._refresh_paths()
        pattern = kwargs.get("pattern")
        if not pattern:
            return "[ERROR] grep: pattern is required"
        include_patterns = _coerce_include_patterns(kwargs.get("include"))
        base = kwargs.get("path") or kwargs.get("dir") or kwargs.get("directory") or os.getcwd()
        base_path = Path(str(base)).expanduser()
        if not base_path.is_absolute():
            base_path = paths.root / base_path
        regex = self._compile_regex(pattern)
        if isinstance(regex, str):
            return regex
        files = _collect_grep_files(base_path, include_patterns)
        results = _search_files(regex, files)
        return "\n".join(results) if results else "[EMPTY]"

    def bash(self, **kwargs) -> str:
        self._refresh_paths()
        return self.bash_tool.run(**kwargs)

    def question(self, **kwargs) -> str:
        self._refresh_paths()
        from akg_agents.core.tools.basic_tools import ask_user

        message = _build_question_message(kwargs)
        if hasattr(ask_user, "invoke"):
            return ask_user.invoke({"message": message})
        if hasattr(ask_user, "run"):
            return ask_user.run(message=message)
        return ask_user(message)

    def todowrite(self, **kwargs) -> str:
        self._refresh_paths()
        todos = _parse_todos_payload(kwargs)
        items, error = _parse_todos_input(todos)
        if error:
            return error
        if not isinstance(items, list):
            return "[ERROR] todowrite: todos must be a list"
        normalized = [item for item in (_normalize_todo_item(i) for i in items) if item]
        self.state.todos = normalized
        return json.dumps(self.state.todos, ensure_ascii=False, indent=2)

    def todoread(self, **kwargs) -> str:
        self._refresh_paths()
        return json.dumps(self.state.todos, ensure_ascii=False, indent=2)

    def plan_enter(self, **kwargs) -> str:
        self._refresh_paths()
        plan_path = self._plan_path()
        if self.state.mode == "plan":
            return f"[NOOP] plan-enter: already in plan mode (plan file: {plan_path})"
        reply = self._plan_prompt("enter")
        decision = _parse_yes_no_reply(reply)
        if decision is not True:
            return "[CANCELLED] plan-enter: user declined"
        self.state.mode = "plan"
        return self._plan_confirm("enter")

    def plan_exit(self, **kwargs) -> str:
        self._refresh_paths()
        plan_path = self._plan_path()
        if self.state.mode == "build":
            return f"[NOOP] plan-exit: already in build mode (plan file: {plan_path})"
        reply = self._plan_prompt("exit")
        decision = _parse_yes_no_reply(reply)
        if decision is not True:
            return "[CANCELLED] plan-exit: user declined"
        self.state.mode = "build"
        self.state.plan_path = plan_path
        return self._plan_confirm("exit")

    def webfetch(self, **kwargs) -> str:
        self._refresh_paths()
        url = kwargs.get("url") or kwargs.get("link")
        if not url:
            return "[ERROR] webfetch: url is required"
        try:
            import httpx

            resp = httpx.get(url, timeout=20.0)
            text = (resp.text or "")
            if len(text) > 8000:
                text = text[:8000] + "\n...[truncated]..."
            return text
        except Exception as exc:
            return f"[ERROR] webfetch: {exc}"

    def batch(self, **kwargs) -> str:
        self._refresh_paths()
        calls = _parse_batch_calls(kwargs)
        if isinstance(calls, str):
            return calls
        results = [self._run_batch_call(call) for call in calls]
        return json.dumps(results, ensure_ascii=False, indent=2)

    def _run_batch_call(self, call: dict) -> dict:
        tool_name = call.get("tool") if isinstance(call, dict) else None
        params = call.get("parameters") if isinstance(call, dict) else None
        if not tool_name:
            return {"error": "missing tool name"}
        if tool_name == "batch":
            return {"tool": tool_name, "error": "nested batch not allowed"}
        func = self.state.tool_exec.get(tool_name)
        if func is None:
            return {"tool": tool_name, "error": "tool not found"}
        try:
            return {"tool": tool_name, "output": func(**(params or {}))}
        except Exception as exc:
            return {"tool": tool_name, "error": str(exc)}

    def _plan_prompt(self, mode: str) -> str:
        from akg_agents.core.tools.basic_tools import ask_user

        message = self._plan_prompt_message(mode)
        if hasattr(ask_user, "invoke"):
            response = ask_user.invoke({"message": message})
        elif hasattr(ask_user, "run"):
            response = ask_user.run(message=message)
        else:
            response = ask_user(message)
        reply = str(response or "")
        return reply[len("用户回复:") :].strip() if reply.startswith("用户回复:") else reply

    def _plan_prompt_message(self, mode: str) -> str:
        plan_path = self._plan_path()
        if mode == "enter":
            return (
                f"Would you like to switch to plan mode and create a plan at {plan_path}?\n"
                "Reply 'Yes' to switch, or 'No' to stay in build mode."
            )
        return (
            f"Plan at {plan_path} is complete. Switch to build mode and start implementing?\n"
            "Reply 'Yes' to switch, or 'No' to continue planning."
        )

    def _plan_path(self) -> str:
        if not self.state.plan_path:
            self.state.plan_path = str((self.paths.root / "plan.md").resolve())
        return self.state.plan_path

    def _plan_confirm(self, mode: str) -> str:
        plan_path = self._plan_path()
        if mode == "enter":
            return (
                f"User confirmed switch to plan mode. Plan file: {plan_path}. "
                "Begin planning and call plan-exit when ready."
            )
        return (
            f"User approved switching to build mode. Plan file: {plan_path}. "
            "You can now edit files and execute the plan."
        )

    def _ensure_within(self, path: Path) -> str | None:
        ok, err = self.paths.ensure_within(path)
        return err if not ok else None

    def _ensure_file(self, path: Path, action: str) -> str | None:
        if not path.exists():
            return f"[ERROR] {action}: file not found: {path}"
        if not path.is_file():
            return f"[ERROR] {action}: not a file: {path}"
        return self._ensure_within(path)

    @staticmethod
    def _format_ls_entry(entry: Path) -> str:
        return entry.name + ("/" if entry.is_dir() else "")

    @staticmethod
    def _is_ignored(item: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            if fnmatch.fnmatch(item.rstrip("/"), pattern):
                return True
        return False

    @staticmethod
    def _compile_regex(pattern: str):
        import re

        try:
            return re.compile(pattern)
        except re.error as exc:
            return f"[ERROR] grep: invalid regex: {exc}"


class ToolDocs:
    def __init__(self) -> None:
        self.tool_dir = _tool_dir()

    def load(self) -> list[tuple[str, str]]:
        if not self.tool_dir.exists():
            return []
        docs: list[tuple[str, str]] = []
        for path in sorted(self.tool_dir.glob("*.txt")):
            content = self._read_doc(path)
            if content:
                docs.append((path.stem, content))
        return docs

    @staticmethod
    def _read_doc(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""


def _augment_tool_desc(name: str, desc: str) -> str:
    name = name.strip().lower()
    if name in {"read", "write", "ls", "glob", "grep"}:
        desc += (
            "\n\nNote: In akg_cli common, relative paths are allowed and resolved from the current working directory."
            " Use ./file to write or read in the current directory."
        )
    if name in {"write", "edit", "multiedit", "apply_patch"}:
        desc += "\n\nNote: Writes are restricted to the current workspace (cwd)."
    if name == "bash":
        desc += "\n\nNote: In akg_cli common, workdir defaults to the current working directory."
    return desc


def _bind_state(func: callable, state: CommonToolState) -> callable:
    def _wrapped(**kwargs):
        return func(state, **kwargs)

    return _wrapped


def _build_question_message(kwargs: dict) -> str:
    question = kwargs.get("question") or kwargs.get("prompt") or kwargs.get("message")
    options = kwargs.get("options") or []
    custom = kwargs.get("custom", True)
    question = str(question or "Please provide input:").strip()
    lines = [question]
    if isinstance(options, list) and options:
        lines.append("Options:")
        for idx, opt in enumerate(options, start=1):
            label = opt.get("label") if isinstance(opt, dict) else str(opt)
            desc = opt.get("description") or "" if isinstance(opt, dict) else ""
            lines.append(f"{idx}. {label}{' - ' + desc if desc else ''}")
    if custom:
        lines.append("You can also type a custom answer.")
    return "\n".join(lines)


def _coerce_include_patterns(include: Any) -> list[str]:
    if isinstance(include, (list, tuple)):
        return [str(item) for item in include if str(item)]
    return [str(include)] if include else []


def _matches_include(path: Path, base_path: Path, patterns: list[str]) -> bool:
    if not patterns:
        return True
    try:
        rel = path.relative_to(base_path).as_posix()
    except Exception:
        rel = path.name
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern):
            return True
        if "/" not in pattern and "\\" not in pattern:
            if fnmatch.fnmatch(path.name, pattern):
                return True
    return False


def _collect_grep_files(base_path: Path, patterns: list[str]) -> list[Path]:
    if base_path.is_file():
        return [base_path]
    files: list[Path] = []
    for root, _, filenames in os.walk(base_path):
        for fname in filenames:
            path = Path(root) / fname
            if _matches_include(path, base_path, patterns):
                files.append(path)
    return files


def _search_files(regex, files: list[Path]) -> list[str]:
    results: list[str] = []
    for path in files:
        try:
            lines = read_file_lines(path)
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            if regex.search(line):
                results.append(f"{path}:{idx}:{line}")
    return results


def _parse_batch_calls(kwargs: dict) -> list[dict] | str:
    calls = kwargs.get("calls") or kwargs.get("batch") or kwargs.get("items") or kwargs.get("payload")
    if calls is None and isinstance(kwargs.get("payload"), dict):
        calls = kwargs.get("payload")
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except Exception:
            return "[ERROR] batch: invalid JSON payload"
    if isinstance(calls, dict):
        calls = calls.get("payload") or calls.get("items") or calls.get("calls") or calls.get("batch")
    if not isinstance(calls, list):
        return f"[ERROR] batch: payload must be a list (got {type(calls).__name__})"
    return calls


def build_common_tools(state: CommonToolState | None = None) -> list:
    state = state or CommonToolState("common")
    runner = ToolRunner(state)
    adapter_map = _build_adapter_map(runner)
    tools = _load_structured_tools(adapter_map, state)
    _register_tool_exec(tools, adapter_map, state)
    return tools


def _build_adapter_map(runner: ToolRunner) -> dict[str, callable]:
    return {
        "read": runner.read,
        "write": runner.write,
        "ls": runner.ls,
        "glob": runner.glob,
        "grep": runner.grep,
        "bash": runner.bash,
        "edit": runner.edit,
        "multiedit": runner.multiedit,
        "apply_patch": runner.apply_patch,
        "question": runner.question,
        "todoread": runner.todoread,
        "todowrite": runner.todowrite,
        "plan-enter": runner.plan_enter,
        "plan-exit": runner.plan_exit,
        "webfetch": runner.webfetch,
        "batch": runner.batch,
    }


def _load_structured_tools(adapter_map: dict[str, callable], state: CommonToolState) -> list:
    tools: list = []
    existing = set()
    for name, desc in ToolDocs().load():
        if name in existing:
            continue
        tool = _make_structured_tool(name, desc, adapter_map, state)
        tools.append(tool)
        existing.add(name)
    return tools


def _make_structured_tool(
    name: str,
    desc: str,
    adapter_map: dict[str, callable],
    state: CommonToolState,
):
    desc = _augment_tool_desc(name, desc)
    func = _wrap_tool_with_approval(name, adapter_map.get(name) or _make_stub_tool_func(name))
    tool = StructuredTool.from_function(
        name=name,
        description=desc,
        func=func,
        args_schema=GENERIC_ARGS_SCHEMA,
    )
    state.tool_exec[name] = func
    return tool


def _register_tool_exec(tools: list, adapter_map: dict[str, callable], state: CommonToolState) -> None:
    for tool in tools:
        tname = getattr(tool, "name", "")
        if tname and tname not in state.tool_exec:
            state.tool_exec[tname] = adapter_map.get(tname, tool.func)


def _make_stub_tool_func(name: str):
    def _stub(**_kwargs):
        return f"[ERROR] Tool '{name}' is not implemented in akg_cli common."

    return _stub
