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
TaskConstructor 专用文件操作工具

这些工具的语义绑定在 TaskConstructor 的 workspace 概念上，
不适合作为通用工具（通用文件操作在 core_v2/tools/basic_tools.py 中）。
"""

import re
import ast
import logging
from pathlib import Path
from typing import Dict, Any, List

from akg_agents.core_v2.tools.tool_registry import ToolRegistry
from akg_agents.op.tools.task_constructor.path_utils import resolve_path

logger = logging.getLogger(__name__)


# ==================== workspace 辅助 ====================

def _save_to_workspace(workspace_dir: Path, filename: str, content: str) -> Path:
    path = workspace_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ==================== TC 专用工具函数 ====================


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
    """注册 TaskConstructor 专用工具到统一 ToolRegistry"""

    ToolRegistry.register(
        name="append_to_file",
        description="【分段生成】追加内容到已有文件末尾。文件必须已存在。",
        parameters={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "目标文件路径"},
            "content": {"type": "string", "description": "要追加的内容"},
        }, "required": ["file_path", "content"]},
        func=append_to_file,
        category="basic",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="copy_to_workspace",
        description="【推荐】复制整个源文件到工作区，便于后续操作。",
        parameters={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "要复制的源文件路径"},
            "workspace_name": {"type": "string", "description": "工作区中的文件名（可选）"},
        }, "required": ["file_path"]},
        func=copy_to_workspace,
        category="basic",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="read_function",
        description=(
            "精确提取函数/类定义（AST 解析）。自动保存到工作区。\n"
            "适用于: 查看目标函数签名、阅读外部依赖函数的实现。\n"
            "注意: trace_dependencies 返回的同文件内依赖函数不需要逐个 read_function，"
            "直接将 functions 列表传给 assemble_task 即可。"
        ),
        parameters={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Python 文件路径"},
            "function_name": {"type": "string", "description": "要提取的函数名或类名"},
        }, "required": ["file_path", "function_name"]},
        func=read_function,
        category="code_analysis",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="save_to_workspace",
        description="手动保存内容到工作区文件。",
        parameters={"type": "object", "properties": {
            "filename": {"type": "string", "description": "工作区中的文件名"},
            "content": {"type": "string", "description": "要保存的内容"},
        }, "required": ["filename", "content"]},
        func=save_to_workspace,
        category="basic",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="list_workspace",
        description="查看工作区中已保存的所有文件及行数。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=list_workspace,
        category="basic",
        scopes=["task_constructor"],
    )

    ToolRegistry.register(
        name="multi_file_search",
        description="同时搜索多个关键词，返回跨文件的匹配结果。",
        parameters={"type": "object", "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "关键词列表"},
            "path": {"type": "string", "description": "搜索根目录"},
            "glob": {"type": "string", "description": "文件过滤 glob 模式"},
        }, "required": ["keywords"]},
        func=multi_file_search,
        category="basic",
        scopes=["task_constructor"],
    )


_register_all()
