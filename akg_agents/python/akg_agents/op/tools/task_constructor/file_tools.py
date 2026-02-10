# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
任务构造器文件操作工具

核心功能：文件读写、workspace 管理、目录浏览、函数提取、搜索。
遵循 akg_agents v2 工具规范。
"""

import re
import ast
import logging
from pathlib import Path
from typing import Dict, Any, List

from akg_agents.op.tools.task_constructor.tool_registry import TaskToolRegistry
from akg_agents.op.tools.task_constructor.path_utils import resolve_path

logger = logging.getLogger(__name__)

# 文件读取配置
READ_FILE_MAX_LINES = 300


# ==================== workspace 辅助 ====================

def _save_to_workspace(workspace_dir: Path, filename: str, content: str) -> Path:
    """保存内容到 workspace 目录"""
    path = workspace_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ==================== 工具函数 ====================

def read_file(
    file_path: str,
    offset: int = None,
    limit: int = None,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """读取文件内容（带行号）。超长文件自动截断。"""
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None
    path = resolve_path(file_path, workspace_dir=ws, output_dir=od)

    if not path.exists():
        return {"status": "error", "output": "",
                "error_information": f"文件不存在: {path}"}
    if not path.is_file():
        if path.is_dir():
            return {"status": "error", "output": "",
                    "error_information": f"这是目录: {path}\n请使用 scan_dir 查看。"}
        return {"status": "error", "output": "", "error_information": f"不是文件: {path}"}

    try:
        all_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        total = len(all_lines)

        if offset is not None:
            start = max(int(offset) - 1, 0)
            end = start + int(limit) if limit else min(start + READ_FILE_MAX_LINES, total)
            lines = all_lines[start:end]
            line_offset = start + 1
        else:
            if total > READ_FILE_MAX_LINES and limit is None:
                lines = all_lines[:READ_FILE_MAX_LINES]
                line_offset = 1
            else:
                lines = all_lines
                line_offset = 1

        numbered = [f"{line_offset + i:>5}| {line.rstrip()}" for i, line in enumerate(lines)]
        content = "\n".join(numbered)

        meta = f"[文件: {path.name}, 总行数: {total}]"
        if offset is not None:
            end_line = line_offset + len(lines) - 1
            meta += f" [显示: 第{line_offset}-{end_line}行]"
        elif total > READ_FILE_MAX_LINES and limit is None:
            meta += (f" [已截断: 仅显示前{READ_FILE_MAX_LINES}行]"
                     f" 提示: 用 assemble_task 的 source_files 参数直接引用此文件即可")

        return {"status": "success", "output": f"{meta}\n{content}", "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def write_file(
    file_path: str,
    content: str,
    overwrite: bool = True,
    output_dir: str = "",
) -> Dict[str, Any]:
    """写入文件。相对路径写到 output 目录。"""
    path = Path(file_path).expanduser()
    if not path.is_absolute() and output_dir:
        path = Path(output_dir) / path
    path = path.resolve()

    if path.exists() and not overwrite:
        return {"status": "error", "output": "",
                "error_information": f"文件已存在: {path}"}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"status": "success", "output": f"已写入: {path}", "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def append_to_file(
    file_path: str,
    content: str,
    output_dir: str = "",
) -> Dict[str, Any]:
    """追加内容到已有文件末尾"""
    path = Path(file_path).expanduser()
    if not path.is_absolute() and output_dir:
        path = Path(output_dir) / path
    path = path.resolve()

    if not path.exists():
        return {"status": "error", "output": "",
                "error_information": f"文件不存在: {path}\n请先用 write_file 创建文件。"}

    try:
        existing = path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"
        new_content = existing + content
        path.write_text(new_content, encoding="utf-8")

        total_lines = new_content.count("\n") + 1
        appended_lines = content.count("\n") + 1
        return {"status": "success",
                "output": f"已追加 {appended_lines} 行 -> {path.name} (现共 {total_lines} 行)",
                "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def scan_dir(
    path: str = ".",
    max_depth: int = 3,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """一步浏览目录: 列出文件 + Python 文件摘要"""
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None
    dir_path = resolve_path(path, workspace_dir=ws, output_dir=od)
    if not dir_path.exists():
        return {"status": "error", "output": "", "error_information": f"路径不存在: {dir_path}"}
    if dir_path.is_file():
        return {"status": "error", "output": "",
                "error_information": f"这是文件: {dir_path}\n请使用 read_file 读取。"}

    try:
        results = [f"[DIR] {dir_path}"]
        py_files = []
        other_files = []
        for item in sorted(dir_path.rglob("*")):
            rel = item.relative_to(dir_path)
            if any(part.startswith('.') for part in rel.parts):
                continue
            if "__pycache__" in str(rel):
                continue
            if len(rel.parts) > max_depth:
                continue
            if item.is_file():
                (py_files if item.suffix == ".py" else other_files).append(item)

        if other_files:
            results.append(f"\n Other files ({len(other_files)}):")
            for f in other_files[:30]:
                results.append(f"  {f.relative_to(dir_path)}")

        results.append(f"\n Python files ({len(py_files)}):")
        for f in py_files:
            rel = f.relative_to(dir_path)
            try:
                content = f.read_text(encoding="utf-8")
                lines = content.count("\n") + 1
                defs = []
                for m in re.finditer(r'^(class |def |async def )(\w+)', content, re.MULTILINE):
                    defs.append(f"{m.group(1).strip()} {m.group(2)}")
                defs_str = ", ".join(defs[:15])
                if len(defs) > 15:
                    defs_str += f" ... (+{len(defs)-15})"
                results.append(f"  {rel} ({lines} lines) -> {defs_str}")
            except (UnicodeDecodeError, PermissionError):
                results.append(f"  {rel} (unreadable)")

        return {"status": "success", "output": "\n".join(results), "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def copy_to_workspace(
    file_path: str,
    workspace_name: str = "",
    workspace_dir: str = "",
) -> Dict[str, Any]:
    """复制整个源文件到 workspace 工作区"""
    ws = Path(workspace_dir) if workspace_dir else None
    path = resolve_path(file_path, workspace_dir=ws)
    if not path.exists() or not path.is_file():
        return {"status": "error", "output": "",
                "error_information": f"文件不存在或不是文件: {path}"}

    if not ws:
        return {"status": "error", "output": "",
                "error_information": "workspace_dir 未配置"}

    try:
        content = path.read_text(encoding="utf-8")
        total_lines = content.count("\n") + 1

        if not workspace_name:
            workspace_name = path.name
        ws_path = _save_to_workspace(ws, workspace_name, content)

        defs = []
        for m in re.finditer(r'^(class |def |async def )(\w+)', content, re.MULTILINE):
            defs.append(f"  {m.group(1).strip()} {m.group(2)} (line {content[:m.start()].count(chr(10))+1})")

        summary = (
            f"[已复制到工作区] {ws_path}\n"
            f"源文件: {path}\n"
            f"总行数: {total_lines}\n"
            f"包含 {len(defs)} 个函数/类:\n" + "\n".join(defs)
        )
        return {"status": "success", "output": summary, "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def read_function(
    file_path: str,
    function_name: str,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """精确提取函数/类定义（AST解析）。自动保存到 workspace。"""
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None
    path = resolve_path(file_path, workspace_dir=ws, output_dir=od)

    if not path.exists() or not path.is_file():
        return {"status": "error", "output": "",
                "error_information": f"文件不存在或不是文件: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        total = len(lines)
        func_code = None
        start_line = 0
        end_line = 0

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == function_name:
                        start = node.lineno - 1
                        end = node.end_lineno
                        if node.decorator_list:
                            start = min(d.lineno - 1 for d in node.decorator_list)
                        func_code = "\n".join(lines[start:end])
                        start_line = start + 1
                        end_line = end
                        break
        except SyntaxError:
            pass

        if func_code is None:
            pattern = re.compile(
                rf'^(class |def |async def ){re.escape(function_name)}\s*[\(:]',
                re.MULTILINE,
            )
            match = pattern.search(content)
            if match:
                start_pos = content[:match.start()].count("\n")
                def_line = lines[start_pos]
                indent = len(def_line) - len(def_line.lstrip())
                end_pos = start_pos + 1
                while end_pos < total:
                    line = lines[end_pos]
                    if line.strip() == "":
                        end_pos += 1
                        continue
                    if len(line) - len(line.lstrip()) <= indent and line.strip():
                        break
                    end_pos += 1
                func_code = "\n".join(lines[start_pos:end_pos])
                start_line = start_pos + 1
                end_line = end_pos

        if func_code is None:
            return {"status": "error", "output": "",
                    "error_information": f"函数 '{function_name}' 在文件中未找到"}

        ws_filename = f"{path.stem}__{function_name}.py"
        ws_info = ""
        if ws:
            ws_path = _save_to_workspace(ws, ws_filename, func_code)
            ws_info = f"\n[已保存到工作区: workspace/{ws_filename}]"

        func_lines = end_line - start_line + 1
        preview_lines = func_code.splitlines()[:30]
        numbered_preview = [f"{start_line+i:>5}| {l}" for i, l in enumerate(preview_lines)]
        if len(func_code.splitlines()) > 30:
            numbered_preview.append(f"  ... (共{func_lines}行，完整代码已保存到工作区)")

        meta = (f"[函数: {function_name}, 文件: {path.name}, "
                f"行{start_line}-{end_line}, 文件总行数: {total}]"
                f"{ws_info}")

        return {"status": "success",
                "output": f"{meta}\n" + "\n".join(numbered_preview),
                "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def grep_search(
    pattern: str,
    path: str = ".",
    glob: str = "*.py",
    max_results: int = 50,
    context_lines: int = 0,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """在文件/目录中搜索正则"""
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None
    search_path = resolve_path(path, workspace_dir=ws, output_dir=od)
    results = []
    try:
        files = [search_path] if search_path.is_file() else sorted(search_path.rglob(glob))
        regex = re.compile(pattern, re.IGNORECASE)
        for f in files:
            try:
                file_lines = f.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(file_lines):
                    if regex.search(line):
                        full_path = str(f.resolve()).replace('\\', '/')
                        results.append(f"{full_path}:{i+1}: {line.rstrip()}")
                        if context_lines > 0:
                            start = max(0, i - context_lines)
                            end = min(len(file_lines), i + context_lines + 1)
                            for j in range(start, end):
                                if j != i:
                                    results.append(f"  {j+1}| {file_lines[j].rstrip()}")
                        if len(results) >= max_results * (1 + context_lines * 2):
                            break
            except (UnicodeDecodeError, PermissionError):
                continue
            if len(results) >= max_results * (1 + context_lines * 2):
                break
        output = "\n".join(results) if results else "no matches"
        return {"status": "success", "output": output, "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def list_workspace(
    workspace_dir: str = "",
) -> Dict[str, Any]:
    """列出工作区中已保存的所有文件"""
    if not workspace_dir:
        return {"status": "error", "output": "",
                "error_information": "workspace_dir 未配置"}
    ws = Path(workspace_dir)
    try:
        files = sorted(ws.rglob("*"))
        if not files:
            return {"status": "success", "output": "工作区为空", "error_information": ""}

        results = [f"[工作区] {ws}"]
        for f in files:
            if f.is_file():
                rel = f.relative_to(ws)
                try:
                    file_lines = f.read_text(encoding="utf-8").count("\n") + 1
                    results.append(f"  {rel} ({file_lines} lines)")
                except Exception:
                    results.append(f"  {rel}")
        return {"status": "success", "output": "\n".join(results), "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def save_to_workspace(
    filename: str,
    content: str,
    workspace_dir: str = "",
) -> Dict[str, Any]:
    """手动保存内容到工作区文件"""
    if not workspace_dir:
        return {"status": "error", "output": "",
                "error_information": "workspace_dir 未配置"}
    try:
        ws_path = _save_to_workspace(Path(workspace_dir), filename, content)
        file_lines = content.count("\n") + 1
        return {"status": "success",
                "output": f"[已保存到工作区] {ws_path} ({file_lines}行)",
                "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def multi_file_search(
    keywords: List[str],
    path: str = ".",
    glob: str = "*.py",
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """搜索多个关键词，返回跨文件匹配"""
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None
    search_path = resolve_path(path, workspace_dir=ws, output_dir=od)
    results = {}
    try:
        files = sorted(search_path.rglob(glob)) if search_path.is_dir() else [search_path]
        for kw in keywords:
            regex = re.compile(re.escape(kw))
            matches = []
            for f in files:
                try:
                    content = f.read_text(encoding="utf-8")
                    for i, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            rel = str(f.relative_to(search_path)) if search_path.is_dir() else f.name
                            matches.append(f"{rel}:{i}: {line.rstrip()}")
                except (UnicodeDecodeError, PermissionError):
                    continue
            results[kw] = matches[:20]
        out = []
        for kw, ms in results.items():
            out.append(f"=== {kw} ({len(ms)} matches) ===")
            out.extend(ms)
        return {"status": "success", "output": "\n".join(out), "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


# ==================== 工具注册 ====================

def _register_all():
    """注册所有文件操作工具"""

    TaskToolRegistry.register(
        "read_file", "读取文件（带行号）。超过300行截断。支持 workspace/xxx.py 短路径。",
        {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "文件路径"},
            "offset": {"type": "integer", "description": "起始行号(1-based)"},
            "limit": {"type": "integer", "description": "读取行数"},
        }, "required": ["file_path"]},
        read_file,
    )

    TaskToolRegistry.register(
        "write_file", "创建/覆盖写入文件。",
        {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "目标路径"},
            "content": {"type": "string", "description": "要写入的内容"},
            "overwrite": {"type": "boolean", "description": "覆盖已有文件"},
        }, "required": ["file_path", "content"]},
        write_file,
    )

    TaskToolRegistry.register(
        "append_to_file", "【分段生成】追加内容到已有文件末尾。",
        {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "目标文件路径"},
            "content": {"type": "string", "description": "要追加的内容"},
        }, "required": ["file_path", "content"]},
        append_to_file,
    )

    TaskToolRegistry.register(
        "scan_dir", "一步浏览目录: 文件列表 + Python文件摘要。",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "目录路径"},
            "max_depth": {"type": "integer", "description": "最大深度"},
        }, "required": ["path"]},
        scan_dir,
    )

    TaskToolRegistry.register(
        "copy_to_workspace", "【推荐】复制源文件到工作区。",
        {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "要复制的源文件路径"},
            "workspace_name": {"type": "string", "description": "工作区文件名"},
        }, "required": ["file_path"]},
        copy_to_workspace,
    )

    TaskToolRegistry.register(
        "read_function", "精确提取函数/类定义（AST解析）。自动保存到工作区。",
        {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "文件路径"},
            "function_name": {"type": "string", "description": "函数名或类名"},
        }, "required": ["file_path", "function_name"]},
        read_function,
    )

    TaskToolRegistry.register(
        "grep_search", "在文件/目录中搜索正则。",
        {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "正则表达式"},
            "path": {"type": "string", "description": "搜索路径"},
            "glob": {"type": "string", "description": "文件过滤 glob"},
            "max_results": {"type": "integer", "description": "最大结果数"},
            "context_lines": {"type": "integer", "description": "上下文行数"},
        }, "required": ["pattern"]},
        grep_search,
    )

    TaskToolRegistry.register(
        "save_to_workspace", "手动保存内容到工作区文件。",
        {"type": "object", "properties": {
            "filename": {"type": "string", "description": "工作区文件名"},
            "content": {"type": "string", "description": "要保存的内容"},
        }, "required": ["filename", "content"]},
        save_to_workspace,
    )

    TaskToolRegistry.register(
        "list_workspace", "查看工作区中已保存的所有文件。",
        {"type": "object", "properties": {}, "required": []},
        list_workspace,
    )

    TaskToolRegistry.register(
        "multi_file_search", "搜索多个关键词，返回跨文件匹配。",
        {"type": "object", "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "关键词列表"},
            "path": {"type": "string", "description": "搜索根目录"},
            "glob": {"type": "string", "description": "文件过滤"},
        }, "required": ["keywords"]},
        multi_file_search,
    )


# 模块加载时自动注册
_register_all()
