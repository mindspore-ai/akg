"""
Tool definitions and execution for the programmatic agent.

Follows the dispatch map pattern:
  TOOLS     — Anthropic-native tool schemas (sent to LLM)
  TOOL_HANDLERS — {name: handler} dispatch map (executed by loop)

Domain-specific tools (eval, quick_check) remain internal —
triggered automatically by the loop after edits, not by the LLM.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..framework.config import TaskConfig
    from ..framework.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Structured tool result — replaces the "OK"/"ERROR: ..." string protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolResult:
    """Structured result from a tool execution.

    Attributes:
        ok:      True if the operation succeeded.
        message: Human-readable text sent to the LLM as tool_result content.
        kind:    Tool category — "read", "patch", "write", "check", "eval".
    """
    ok: bool
    message: str
    kind: str = ""


# ---------------------------------------------------------------------------
# Anthropic-native tool schemas — sent directly to client.messages.create()
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read file contents. "
            "mode='full' (default): entire file with line/char count header. "
            "mode='range': specific line range (requires target='start-end'). "
            "Files already shown in conversation do NOT need re-reading."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to the task directory.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["full", "range"],
                    "description": (
                        "'full': entire file with metadata (default). "
                        "'range': specific line range, requires target."
                    ),
                    "default": "full",
                },
                "target": {
                    "type": "string",
                    "description": "Line range for mode='range', e.g. '50-100'. Ignored for 'full'.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "patch_file",
        "description": (
            "Make a TARGETED edit by replacing an exact substring. "
            "PREFER this over write_file for incremental changes. "
            "Fails if old_str is not found or appears more than once. "
            "You may call this multiple times in one turn for coupled changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from task_dir. Must be in editable_files.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact substring to replace (must appear once).",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string.",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description for the experiment log.",
                },
                "plan_item_id": {
                    "type": "string",
                    "description": "ID of the active plan item this edit belongs to (e.g. 'p1').",
                },
            },
            "required": ["path", "old_str", "new_str", "description", "plan_item_id"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write complete content to an editable file. Use ONLY when a full "
            "rewrite is truly necessary. For incremental changes, PREFER patch_file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from task_dir. Must be in editable_files.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete new file content.",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description for the experiment log.",
                },
                "plan_item_id": {
                    "type": "string",
                    "description": "ID of the active plan item this edit belongs to (e.g. 'p1').",
                },
            },
            "required": ["path", "content", "description", "plan_item_id"],
        },
    },
    {
        "name": "update_plan",
        "description": (
            "Submit a new optimization plan. Each '- [ ]' item becomes a plan item "
            "executed in order. The first item is activated immediately. "
            "Can only be called when no item is currently active (before first plan, "
            "or after all items are settled). Items are auto-settled by eval results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": (
                        "Complete optimization plan (markdown). "
                        "Use '- [ ]' for each actionable item. "
                        "Items are executed sequentially; system assigns IDs (p1, p2, ...)."
                    ),
                },
            },
            "required": ["plan"],
        },
    },
    {
        "name": "compact",
        "description": (
            "Manually trigger context compression. Call when you notice the "
            "conversation is getting long or history is being truncated. "
            "Summarizes history into a compact summary. Does NOT trigger eval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "finish",
        "description": (
            "Signal that optimization is complete. Call when you have exhausted "
            "improvement ideas or the framework will stop you at max_rounds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished.",
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handler implementations
# ---------------------------------------------------------------------------

def execute_read_file(path: str, task_dir: str,
                      mode: str = "full", target: str | None = None) -> ToolResult:
    """Read a file's contents. Sandboxed to repo root.

    Modes:
        full  — entire file, no truncation, with line/char count header.
        range — specific line range (1-based), requires *target* like '50-100'.
    """
    if not os.path.isabs(path):
        resolved = os.path.join(task_dir, path)
    else:
        resolved = path
    resolved = os.path.normpath(os.path.abspath(resolved))

    # Sandbox: must be within git repo root
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, stderr=subprocess.DEVNULL, text=True
        ).strip()
        repo_root = os.path.normpath(os.path.abspath(repo_root))
    except Exception:
        # Walk up from task_dir looking for .git; fall back to task_dir itself
        _d = os.path.normpath(os.path.abspath(task_dir))
        repo_root = _d  # conservative default
        while True:
            if os.path.isdir(os.path.join(_d, ".git")):
                repo_root = _d
                break
            _parent = os.path.dirname(_d)
            if _parent == _d:
                break
            _d = _parent

    try:
        common = os.path.commonpath([resolved, repo_root])
    except ValueError:
        common = ""
    if os.path.normpath(common) != os.path.normpath(repo_root):
        return ToolResult(ok=False, message=f"ERROR: Path '{path}' escapes project root.", kind="read")

    if not os.path.exists(resolved):
        return ToolResult(ok=False, message=f"ERROR: File not found: {path}", kind="read")
    if not os.path.isfile(resolved):
        return ToolResult(ok=False, message=f"ERROR: Not a file: {resolved}", kind="read")

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return ToolResult(ok=False, message=f"ERROR reading {resolved}: {e}", kind="read")

    source_lines = content.splitlines()
    total_lines = len(source_lines)
    total_chars = len(content)
    rel_path = os.path.relpath(resolved, task_dir)

    # --- range mode: return specific line range ---
    if mode == "range":
        if not target:
            return ToolResult(ok=False,
                message="ERROR: mode='range' requires target (e.g. '50-100')", kind="read")
        try:
            parts = target.split("-")
            start = int(parts[0])
            end = int(parts[1]) if len(parts) > 1 else start
        except (ValueError, IndexError):
            return ToolResult(ok=False,
                message=f"ERROR: invalid range '{target}', expected 'start-end'", kind="read")
        start = max(1, start)
        end = min(end, total_lines)
        if start > total_lines:
            return ToolResult(ok=False,
                message=f"ERROR: start line {start} exceeds file length ({total_lines} lines)",
                kind="read")
        extracted = "\n".join(source_lines[start - 1 : end])
        header = f"[{rel_path}: lines {start}-{end} of {total_lines}, {total_chars} chars total]"
        return ToolResult(ok=True, message=f"{header}\n{extracted}", kind="read")

    # --- full mode (default): entire file, no truncation, with metadata header ---
    header = f"[{rel_path}: {total_lines} lines, {total_chars} chars]"
    return ToolResult(ok=True, message=f"{header}\n{content}", kind="read")


def _diff_lines(old: str, new: str) -> list[str]:
    """Return lines that are in *new* but not in *old* (added/changed)."""
    old_lines = set(old.splitlines())
    return [l for l in new.splitlines() if l not in old_lines]


def _check_edit_guardrails(new_content: str, config: "TaskConfig",
                           old_content: str | None = None) -> str | None:
    """Check edit content against guardrails.

    config.forbidden_patterns is a dict with two keys:
      - "content": list of regex patterns matched against new file content
      - "diff": list of regex patterns matched against changed lines;
                reject if ALL changed lines match any pattern

    Returns error message or None.
    """
    if len(new_content) > config.max_patch_size:
        return (
            f"Edit too large ({len(new_content)} chars). "
            f"Maximum: {config.max_patch_size} chars."
        )
    fp = config.forbidden_patterns
    # Backward compat: list → treat as content patterns
    if isinstance(fp, list):
        fp = {"content": fp}
    for pattern in fp.get("content", []):
        if re.search(pattern, new_content):
            return f"Edit matches forbidden content pattern '{pattern}'."
    if old_content is not None:
        changed = _diff_lines(old_content, new_content)
        for pattern in fp.get("diff", []):
            if all(re.match(pattern, l) for l in changed):
                return f"Edit rejected: all changed lines match diff pattern '{pattern}'."
    return None


def _validate_editable_path(
    path: str, task_dir: str, config: "TaskConfig", kind: str,
) -> tuple[str | None, ToolResult | None]:
    """Resolve path and check editable_files whitelist.

    Returns (abs_path, None) on success, or (None, error_result) on failure.
    """
    if os.path.isabs(path):
        target_abs = os.path.normpath(path)
    else:
        target_abs = os.path.normpath(os.path.join(task_dir, path))

    allowed_abs = {
        os.path.normpath(os.path.join(task_dir, f))
        for f in config.editable_files
    }
    if target_abs not in allowed_abs:
        label = "Write" if kind == "write" else "Patch"
        return None, ToolResult(
            ok=False, kind=kind,
            message=(
                f"ERROR: {label} rejected. '{path}' not in editable_files.\n"
                f"Allowed: {list(config.editable_files)}"
            ),
        )
    return target_abs, None


def execute_write_file(path: str, content: str, task_dir: str, config: "TaskConfig") -> ToolResult:
    """Write content to a file. Enforces editable_files whitelist."""
    target_abs, err = _validate_editable_path(path, task_dir, config, "write")
    if err:
        return err

    old_content = None
    if os.path.exists(target_abs):
        try:
            with open(target_abs, "r", encoding="utf-8") as f:
                old_content = f.read()
        except Exception:
            pass

    guardrail_error = _check_edit_guardrails(content, config, old_content=old_content)
    if guardrail_error:
        return ToolResult(ok=False, message=f"ERROR: {guardrail_error}", kind="write")

    try:
        os.makedirs(os.path.dirname(target_abs), exist_ok=True)
        with open(target_abs, "w", encoding="utf-8") as f:
            f.write(content)
        rel = os.path.relpath(target_abs, task_dir)
        return ToolResult(ok=True, message=f"OK: wrote {len(content)} chars to {rel}", kind="write")
    except Exception as e:
        return ToolResult(ok=False, message=f"ERROR writing {path}: {e}", kind="write")


def execute_patch_file(path: str, old_str: str, new_str: str, task_dir: str, config: "TaskConfig") -> ToolResult:
    """Targeted replacement. Exactly one occurrence of old_str. Enforces editable_files."""
    target_abs, err = _validate_editable_path(path, task_dir, config, "patch")
    if err:
        return err

    try:
        with open(target_abs, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return ToolResult(ok=False, message=f"ERROR reading {path}: {e}", kind="patch")

    count = content.count(old_str)
    if count == 0:
        return ToolResult(ok=False, message=f"ERROR: old_str not found in {path}. Match exactly (including whitespace).", kind="patch")
    if count > 1:
        return ToolResult(ok=False, message=f"ERROR: old_str appears {count} times in {path}. Make it more specific.", kind="patch")

    new_content = content.replace(old_str, new_str, 1)

    guardrail_error = _check_edit_guardrails(new_content, config, old_content=content)
    if guardrail_error:
        return ToolResult(ok=False, message=f"ERROR: {guardrail_error}", kind="patch")

    try:
        with open(target_abs, "w", encoding="utf-8") as f:
            f.write(new_content)
        rel = os.path.relpath(target_abs, task_dir)
        return ToolResult(ok=True, message=f"OK: patched {rel} ({len(old_str)} → {len(new_str)} chars)", kind="patch")
    except Exception as e:
        return ToolResult(ok=False, message=f"ERROR writing {path}: {e}", kind="patch")


def execute_quick_check(task_dir: str, config: "TaskConfig", device_id: int = None) -> ToolResult:
    """
    Pre-flight check: syntax + import + optional smoke test.
    Returns ToolResult with ok=True on success.
    """
    errors = []
    for f in config.editable_files:
        fpath = os.path.join(task_dir, f)
        if not fpath.endswith(".py") or not os.path.exists(fpath):
            continue

        # Syntax check
        try:
            import py_compile
            py_compile.compile(fpath, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"SyntaxError in {f}: {e}")
            continue

        # DSL compliance check
        if config.dsl:
            try:
                from akg_agents.op.utils.code_checker import CodeChecker
                checker = CodeChecker(backend=config.backend or "",
                                      dsl=config.dsl)
                with open(fpath, "r", encoding="utf-8") as fh:
                    code = fh.read()
                dsl_errors = checker._check_dsl_compliance(code)
                for err in dsl_errors:
                    errors.append(f"DSL compliance in {f}: {err['detail']}")
            except ImportError:
                pass  # standalone mode — no CodeChecker available

        # Import check (skipped when import_timeout <= 0, e.g. AKG mode)
        if config.import_timeout > 0:
            from ..framework.device import get_device_env
            import_env = get_device_env(device_id)
            try:
                result = subprocess.run(
                    [sys.executable, "-c",
                     f"import importlib.util; "
                     f"spec = importlib.util.spec_from_file_location('_check', r'{fpath}'); "
                     f"mod = importlib.util.module_from_spec(spec); "
                     f"spec.loader.exec_module(mod)"],
                    capture_output=True, text=True, timeout=config.import_timeout,
                    cwd=task_dir, env=import_env,
                )
                if result.returncode != 0:
                    stderr = result.stderr.strip()
                    last_lines = stderr.split("\n")[-3:]
                    errors.append(f"ImportError in {f}: {chr(10).join(last_lines)}")
            except subprocess.TimeoutExpired:
                errors.append(f"Import check timed out for {f} (>{config.import_timeout}s)")
            except Exception as e:
                errors.append(f"Check failed for {f}: {e}")

    if errors:
        return ToolResult(ok=False, message="quick_check failed:\n" + "\n".join(errors), kind="check")

    # Syntax-only mode: skip import check + smoke test when import_timeout <= 0
    if config.import_timeout <= 0:
        return ToolResult(ok=True, message="quick_check passed (syntax only)", kind="check")

    # Smoke test
    smoke_cmd = None
    if config.smoke_test_script:
        smoke_path = os.path.join(task_dir, config.smoke_test_script)
        if os.path.exists(smoke_path):
            smoke_cmd = [sys.executable, smoke_path]
    elif config.dsl and config.framework and config.backend:
        # 使用 adapter 生成的评测脚本 --smoke
        try:
            from ..framework.eval_generator import generate_eval_script_file
        except ImportError:
            generate_eval_script_file = None
        if generate_eval_script_file is not None:
            eval_path = generate_eval_script_file(
                dsl=config.dsl, framework=config.framework, backend=config.backend,
                output_dir=os.path.join(task_dir, ".eval_cache"),
            )
            smoke_cmd = [sys.executable, eval_path, "--task-dir", task_dir, "--smoke"]

    if smoke_cmd is not None:
        from ..framework.device import get_device_env
        smoke_env = get_device_env(device_id)
        try:
            result = subprocess.run(
                smoke_cmd,
                capture_output=True, text=True,
                timeout=config.smoke_test_timeout,
                cwd=task_dir, env=smoke_env,
            )
            if result.returncode != 0:
                limit = config.agent.smoke_output_limit
                stderr_tail = result.stderr[-limit:] if result.stderr else ""
                stdout_tail = result.stdout[-limit:] if result.stdout else ""
                errors.append(
                    f"Smoke test failed (exit {result.returncode}):\n"
                    f"{stderr_tail}\n{stdout_tail}".strip()
                )
            else:
                for line in reversed(result.stdout.strip().split("\n")):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            if not data.get("correctness", False):
                                errors.append(f"Smoke test correctness=false: {line}")
                            break
                        except json.JSONDecodeError:
                            continue
        except subprocess.TimeoutExpired:
            errors.append(f"Smoke test timed out after {config.smoke_test_timeout}s")
        except Exception as e:
            errors.append(f"Smoke test error: {e}")

    if errors:
        return ToolResult(ok=False, message="quick_check failed:\n" + "\n".join(errors), kind="check")
    return ToolResult(ok=True, message="OK", kind="check")


async def execute_run_eval(description: str, runner: "ExperimentRunner",
                           raw_output_tail: int = 2_048) -> str:
    """
    Run evaluation via runner.run_one_round().
    INTERNAL ONLY — not exposed to the LLM.
    Returns JSON string with round record summary.
    """
    record = await runner.run_one_round(description)
    r = record.result

    # Three-way status:
    #   KEEP    — accepted (correct + improved + constraints met)
    #   FAIL    — code is broken (correctness / constraint / infrastructure error)
    #   DISCARD — code is correct but not an improvement
    if record.accepted:
        status = "KEEP"
    elif not r.correctness or record.constraint_violations:
        status = "FAIL"
    else:
        status = "DISCARD"

    # Unified fail_reason for FAIL status (diagnostic detail for the agent)
    fail_reason = None
    if status == "FAIL":
        if r.error:
            fail_reason = r.error
        elif not r.correctness:
            fail_reason = "correctness mismatch"
        elif record.constraint_violations:
            fail_reason = "constraint: " + "; ".join(record.constraint_violations)

    result = {
        "round": record.round_num,
        "description": record.description,
        "accepted": record.accepted,
        "status": status,
        "correctness": r.correctness,
        "metrics": r.metrics,
        "commit": record.commit_hash,
        "fail_reason": fail_reason,
        "constraint_violations": record.constraint_violations,
        "duration_sec": round(record.duration_sec, 1),
    }
    raw = record.result.raw_output or ""
    if raw:
        result["raw_output_tail"] = raw[-raw_output_tail:]
    return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Diagnostic subagent — fresh context, read-only, returns concise diagnosis
# ---------------------------------------------------------------------------

SUBAGENT_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file within the project.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "finish_diagnosis",
        "description": (
            "Call this when you have gathered enough information to produce "
            "your diagnosis. Pass your full analysis as the 'report' argument. "
            "This immediately ends the diagnostic session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "report": {
                    "type": "string",
                    "description": (
                        "Your complete diagnosis in the format: "
                        "**Root cause**: ... / **Fix**: ... / **Avoid**: ..."
                    ),
                },
            },
            "required": ["report"],
        },
    },
]


_DIAGNOSE_SYSTEM = "You are a diagnostic agent. Analyze errors and suggest fixes."


async def run_diagnostic_subagent(
    llm,  # ConversationAdapter — reuses retry, provider abstraction, thinking
    error_context: str,
    current_code: dict,
    task_description: str,
    task_dir: str,
    knowledge_prompt: str = None,
    max_iterations: int = 15,
    code_truncate: int = 8_000,
    result_truncate: int = 10_000,
) -> str:
    """
    Spawn a subagent with fresh messages[] to diagnose a persistent failure.

    Uses ConversationAdapter for LLM calls — inherits retry, thinking,
    reasoning_effort, and provider abstraction from the main agent.
    The subagent has read-only access (read_file only) and returns a concise
    root-cause analysis with specific fix suggestions.
    """
    code_section = ""
    for fname, content in current_code.items():
        truncated = content[:code_truncate] if len(content) > code_truncate else content
        code_section += f"\n### {fname}\n```\n{truncated}\n```\n"

    prompt = (
        f"You are a diagnostic agent. The optimization agent is stuck with repeated failures.\n\n"
        f"## Task\n{task_description}\n\n"
        f"## Current Code\n{code_section}\n"
        f"## Problem\n{error_context}\n\n"
        f"## Instructions\n"
        f"1. Use read_file to examine relevant files (eval script, reference, config) **only if needed**.\n"
        f"2. Identify the ROOT CAUSE of the repeated failures.\n"
        f"3. When you have enough information, call **finish_diagnosis** with your analysis.\n"
        f"   Format your report as:\n"
        f"   **Root cause**: (1-2 sentences)\n"
        f"   **Fix**: (specific code-level suggestion)\n"
        f"   **Avoid**: (what approach to NOT try)\n"
        f"\nIMPORTANT: Do NOT read files unnecessarily. If the problem and current code "
        f"already give you enough context, call finish_diagnosis immediately. "
        f"Be specific and actionable. The optimization agent will use your diagnosis directly."
    )

    sub_msgs = [{"role": "user", "content": prompt}]

    for iteration in range(max_iterations):
        print(f"  [diagnose] iteration {iteration + 1}/{max_iterations}", flush=True)

        diagnose_system = _DIAGNOSE_SYSTEM
        if knowledge_prompt:
            diagnose_system = knowledge_prompt + "\n\n" + _DIAGNOSE_SYSTEM
        response = await llm.call(diagnose_system, sub_msgs, tools=SUBAGENT_TOOLS)
        llm.append_assistant(sub_msgs, response)

        text = llm.get_response_text(response)
        if text:
            print(f"  [diagnose] {text[:300]}", flush=True)

        if llm.get_stop_reason(response) != "tool_use":
            return text or "(no diagnosis)"

        tool_calls = llm.extract_tool_calls(response)
        # Build tool results
        tool_results = []
        for tc in tool_calls:
            if tc["tool_name"] == "finish_diagnosis":
                report = tc["arguments"].get("report", "")
                print(f"  [diagnose] finish_diagnosis called (iter {iteration + 1})", flush=True)
                return report or "(empty diagnosis)"
            if tc["tool_name"] == "read_file":
                print(f"  [diagnose] read_file: {tc['arguments'].get('path', '')}", flush=True)
                result = execute_read_file(tc["arguments"].get("path", ""), task_dir)
                content = result.message[:result_truncate]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["tool_use_id"],
                    "content": content,
                })

        if tool_results:
            sub_msgs.append({"role": "user", "content": tool_results})

    return "(diagnostic subagent reached max iterations)"


# ---------------------------------------------------------------------------
# Dispatch map — TurnExecutor calls TOOL_HANDLERS[name](**input)
#
# read_file, patch_file, write_file need task_dir/config injected.
# update_plan, compact, finish are handled directly by TurnExecutor.
# ---------------------------------------------------------------------------

def build_tool_handlers(task_dir: str, config: "TaskConfig") -> dict:
    """
    Build the TOOL_HANDLERS dispatch map bound to a specific task.

    Returns {name: callable(**input) -> ToolResult}.
    update_plan / compact / finish are NOT included —
    TurnExecutor handles them as state mutations before dispatching to this map.
    """
    return {
        "read_file":  lambda **kw: execute_read_file(
            kw["path"], task_dir,
            mode=kw.get("mode", "full"),
            target=kw.get("target"),
        ),
        "patch_file": lambda **kw: execute_patch_file(
            path=kw["path"], old_str=kw["old_str"], new_str=kw["new_str"],
            task_dir=task_dir, config=config,
        ),
        "write_file": lambda **kw: execute_write_file(
            path=kw["path"], content=kw["content"],
            task_dir=task_dir, config=config,
        ),
    }
