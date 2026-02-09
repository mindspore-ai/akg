"""
文件操作工具 - 带 workspace 持久化机制

核心改进:
  - workspace 机制: 提取的代码自动保存到 output/workspace/，不丢失
  - copy_to_workspace: 复制整个源文件到工作区（对多依赖场景最高效）
  - read_function: 提取函数并自动保存到 workspace
  - save_to_workspace: 手动保存任意内容到工作区
  - list_workspace: 查看工作区已保存的所有文件
"""
import os
import re
import ast
from pathlib import Path
from typing import Dict, Any, List

from .registry import ToolRegistry
from ..config import OUTPUT_DIR, READ_FILE_MAX_LINES, WORKSPACE_DIR


# ===================== workspace 辅助 =====================

def _save_to_workspace(filename: str, content: str) -> Path:
    """保存内容到 workspace 目录，返回路径"""
    path = WORKSPACE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _workspace_relative(path: Path) -> str:
    """返回相对于 OUTPUT_DIR 的路径字符串"""
    try:
        return str(path.relative_to(OUTPUT_DIR))
    except ValueError:
        return str(path)


# ===================== read_file =====================

def _resolve_path(file_path: str) -> Path:
    """解析路径，支持 workspace/ 和 output/ 短路径"""
    p = Path(file_path).expanduser()
    if not p.is_absolute():
        # workspace/ 前缀
        if file_path.startswith("workspace/") or file_path.startswith("workspace\\"):
            ws_path = WORKSPACE_DIR / file_path.split("/", 1)[-1].split("\\", 1)[-1]
            if ws_path.exists():
                return ws_path.resolve()
        # output/ 前缀
        if file_path.startswith("output/") or file_path.startswith("output\\"):
            out_path = OUTPUT_DIR / file_path.split("/", 1)[-1].split("\\", 1)[-1]
            if out_path.exists():
                return out_path.resolve()
        # 直接在 workspace 下找
        ws_path = WORKSPACE_DIR / file_path
        if ws_path.exists():
            return ws_path.resolve()
        # 在 output 下找
        out_path = OUTPUT_DIR / file_path
        if out_path.exists():
            return out_path.resolve()
    return p.resolve()


def read_file(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    读取文件内容。返回行号 + 总行数信息。
    超长文件自动截断，提示使用 offset/limit 分块读取。
    支持 workspace/xxx.py 短路径。
    """
    file_path = args.get("file_path", "")
    offset = args.get("offset")
    limit = args.get("limit")

    path = _resolve_path(file_path)
    if not path.exists():
        return {"status": "error", "output": "", "error": f"文件不存在: {path}"}
    if not path.is_file():
        if path.is_dir():
            return {"status": "error", "output": "",
                    "error": f"这是一个目录而非文件: {path}\n请使用 scan_dir 查看目录内容。"}
        return {"status": "error", "output": "", "error": f"不是文件: {path}"}

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

        numbered = []
        for i, line in enumerate(lines):
            numbered.append(f"{line_offset + i:>5}| {line.rstrip()}")
        content = "\n".join(numbered)

        meta = f"[文件: {path.name}, 总行数: {total}]"
        if offset is not None:
            end_line = line_offset + len(lines) - 1
            meta += f" [显示: 第{line_offset}-{end_line}行]"
        elif total > READ_FILE_MAX_LINES and limit is None:
            meta += (f" [已截断: 仅显示前{READ_FILE_MAX_LINES}行]"
                     f" 提示: 不需要阅读全部内容！用 assemble_task 的 source_files 参数直接引用此文件即可")

        return {"status": "success", "output": f"{meta}\n{content}", "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== write_file =====================

def write_file(args: Dict[str, Any]) -> Dict[str, Any]:
    """写入文件。默认写到 output 目录下。"""
    file_path = args["file_path"]
    content = args["content"]
    overwrite = args.get("overwrite", True)

    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = OUTPUT_DIR / path
    path = path.resolve()

    if path.exists() and not overwrite:
        return {"status": "error", "output": "", "error": f"文件已存在: {path}"}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"status": "success", "output": f"已写入: {path}", "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== append_to_file =====================

def append_to_file(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    追加内容到已有文件末尾。用于分段生成长文件。
    与 write_file 配合使用: 先 write_file 创建文件写入第一段，再多次 append_to_file 追加后续段。
    每次调用追加 100-200 行代码是最佳实践。
    """
    file_path = args["file_path"]
    content = args["content"]

    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = OUTPUT_DIR / path
    path = path.resolve()

    if not path.exists():
        return {"status": "error", "output": "",
                "error": f"文件不存在: {path}\n请先用 write_file 创建文件。"}

    try:
        existing = path.read_text(encoding="utf-8")
        # 确保拼接处有换行
        if existing and not existing.endswith("\n"):
            existing += "\n"
        new_content = existing + content
        path.write_text(new_content, encoding="utf-8")

        total_lines = new_content.count("\n") + 1
        appended_lines = content.count("\n") + 1
        return {
            "status": "success",
            "output": f"已追加 {appended_lines} 行 -> {path.name} (现共 {total_lines} 行)",
            "error": "",
        }
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== list_dir =====================

def list_dir(args: Dict[str, Any]) -> Dict[str, Any]:
    dir_path = args.get("path", ".")
    recursive = args.get("recursive", False)
    pattern = args.get("pattern", "*")
    path = Path(dir_path).expanduser().resolve()
    if not path.exists():
        return {"status": "error", "output": "", "error": f"目录不存在: {path}"}
    if path.is_file():
        return {"status": "error", "output": "",
                "error": f"这是文件而非目录: {path}\n请使用 read_file 读取。"}
    try:
        if recursive:
            entries = sorted(str(p.relative_to(path)) for p in path.rglob(pattern)
                             if not any(part.startswith('.') for part in p.relative_to(path).parts))
        else:
            entries = sorted(str(p.relative_to(path)) for p in path.glob(pattern))
        return {"status": "success", "output": "\n".join(entries[:500]), "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== scan_dir =====================

def scan_dir(args: Dict[str, Any]) -> Dict[str, Any]:
    """一步浏览目录: 列出文件 + Python 文件摘要"""
    dir_path = args.get("path", ".")
    max_depth = args.get("max_depth", 3)
    path = Path(dir_path).expanduser().resolve()
    if not path.exists():
        return {"status": "error", "output": "", "error": f"路径不存在: {path}"}
    if path.is_file():
        return {"status": "error", "output": "",
                "error": f"这是文件而非目录: {path}\n请使用 read_file 读取。"}
    try:
        results = [f"[DIR] {path}"]
        py_files = []
        other_files = []
        for item in sorted(path.rglob("*")):
            rel = item.relative_to(path)
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
                results.append(f"  {f.relative_to(path)}")

        results.append(f"\n Python files ({len(py_files)}):")
        for f in py_files:
            rel = f.relative_to(path)
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

        return {"status": "success", "output": "\n".join(results), "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== copy_to_workspace =====================

def copy_to_workspace(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    复制整个源文件到 workspace 工作区。
    适用于：文件有很多互相依赖的函数，逐个提取太慢，不如一次性复制。
    复制后可用 read_file workspace/xxx.py 读取，或 assemble_task 引用。
    """
    file_path = args["file_path"]
    workspace_name = args.get("workspace_name", "")

    path = _resolve_path(file_path)
    if not path.exists() or not path.is_file():
        return {"status": "error", "output": "", "error": f"文件不存在或不是文件: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
        total_lines = content.count("\n") + 1

        # 生成 workspace 文件名
        if not workspace_name:
            workspace_name = path.name
        ws_path = _save_to_workspace(workspace_name, content)

        # 提取文件中的函数/类列表
        defs = []
        for m in re.finditer(r'^(class |def |async def )(\w+)', content, re.MULTILINE):
            defs.append(f"  {m.group(1).strip()} {m.group(2)} (line {content[:m.start()].count(chr(10))+1})")

        summary = (
            f"[已复制到工作区] {ws_path}\n"
            f"源文件: {path}\n"
            f"总行数: {total_lines}\n"
            f"包含 {len(defs)} 个函数/类:\n" + "\n".join(defs)
        )
        return {"status": "success", "output": summary, "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== save_to_workspace =====================

def save_to_workspace(args: Dict[str, Any]) -> Dict[str, Any]:
    """手动保存内容到工作区文件"""
    filename = args["filename"]
    content = args["content"]

    try:
        ws_path = _save_to_workspace(filename, content)
        lines = content.count("\n") + 1
        return {"status": "success",
                "output": f"[已保存到工作区] {ws_path} ({lines}行)",
                "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== list_workspace =====================

def list_workspace(args: Dict[str, Any]) -> Dict[str, Any]:
    """列出工作区中已保存的所有文件"""
    try:
        files = sorted(WORKSPACE_DIR.rglob("*"))
        if not files:
            return {"status": "success", "output": "工作区为空", "error": ""}

        results = [f"[工作区] {WORKSPACE_DIR}"]
        for f in files:
            if f.is_file():
                rel = f.relative_to(WORKSPACE_DIR)
                try:
                    lines = f.read_text(encoding="utf-8").count("\n") + 1
                    results.append(f"  {rel} ({lines} lines)")
                except Exception:
                    results.append(f"  {rel}")
        return {"status": "success", "output": "\n".join(results), "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== grep_search =====================

def grep_search(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = args["pattern"]
    search_path = args.get("path", ".")
    file_glob = args.get("glob", "*.py")
    max_results = args.get("max_results", 50)
    context_lines = args.get("context_lines", 0)

    path = _resolve_path(search_path)
    results = []
    try:
        files = [path] if path.is_file() else sorted(path.rglob(file_glob))
        regex = re.compile(pattern, re.IGNORECASE)
        for f in files:
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines):
                    if regex.search(line):
                        rel = f.relative_to(path) if path.is_dir() else f.name
                        results.append(f"{rel}:{i+1}: {line.rstrip()}")
                        if context_lines > 0:
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            for j in range(start, end):
                                if j != i:
                                    results.append(f"  {j+1}| {lines[j].rstrip()}")
                        if len(results) >= max_results * (1 + context_lines * 2):
                            break
            except (UnicodeDecodeError, PermissionError):
                continue
            if len(results) >= max_results * (1 + context_lines * 2):
                break
        output = "\n".join(results) if results else "no matches"
        return {"status": "success", "output": output, "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== read_function =====================

def read_function(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    精确提取函数/类定义（AST解析）。
    自动保存到 workspace，返回摘要 + workspace 路径。
    LLM 不需要在消息中保留完整代码，需要时从 workspace 读取。
    """
    file_path = args["file_path"]
    func_name = args["function_name"]

    path = _resolve_path(file_path)
    if not path.exists() or not path.is_file():
        return {"status": "error", "output": "", "error": f"文件不存在或不是文件: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        total = len(lines)
        func_code = None
        start_line = 0
        end_line = 0

        # 方法1: AST 解析
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == func_name:
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

        # 方法2: 正则 fallback
        if func_code is None:
            pattern = re.compile(rf'^(class |def |async def ){re.escape(func_name)}\s*[\(:]', re.MULTILINE)
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
                    "error": f"函数 '{func_name}' 在文件中未找到"}

        # 自动保存到 workspace
        ws_filename = f"{path.stem}__{func_name}.py"
        ws_path = _save_to_workspace(ws_filename, func_code)
        func_lines = end_line - start_line + 1

        # 返回摘要 + 前30行预览（不截断保存的内容，只截断发给 LLM 的）
        preview_lines = func_code.splitlines()[:30]
        numbered_preview = [f"{start_line+i:>5}| {l}" for i, l in enumerate(preview_lines)]
        if len(func_code.splitlines()) > 30:
            numbered_preview.append(f"  ... (共{func_lines}行，完整代码已保存到工作区)")

        meta = (f"[函数: {func_name}, 文件: {path.name}, "
                f"行{start_line}-{end_line}, 文件总行数: {total}]\n"
                f"[已保存到工作区: {_workspace_relative(ws_path)}]")

        return {"status": "success",
                "output": f"{meta}\n" + "\n".join(numbered_preview),
                "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ===================== multi_file_search =====================

def multi_file_search(args: Dict[str, Any]) -> Dict[str, Any]:
    keywords = args["keywords"]
    search_path = args.get("path", ".")
    file_glob = args.get("glob", "*.py")
    path = Path(search_path).expanduser().resolve()
    results = {}
    try:
        files = sorted(path.rglob(file_glob)) if path.is_dir() else [path]
        for kw in keywords:
            regex = re.compile(re.escape(kw))
            matches = []
            for f in files:
                try:
                    content = f.read_text(encoding="utf-8")
                    for i, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            rel = str(f.relative_to(path)) if path.is_dir() else f.name
                            matches.append(f"{rel}:{i}: {line.rstrip()}")
                except (UnicodeDecodeError, PermissionError):
                    continue
            results[kw] = matches[:20]
        out = []
        for kw, ms in results.items():
            out.append(f"=== {kw} ({len(ms)} matches) ===")
            out.extend(ms)
        return {"status": "success", "output": "\n".join(out), "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


# ========== 注册所有工具 ==========

ToolRegistry.register(
    "read_file",
    "读取文件（带行号）。超过300行截断。支持 workspace/xxx.py 短路径。长文件不需要全读，用 assemble_task 引用即可。",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "文件路径（支持workspace/下的文件）"},
            "offset": {"type": "integer", "description": "起始行号(1-based)"},
            "limit": {"type": "integer", "description": "读取行数"},
        },
        "required": ["file_path"],
    },
    read_file,
)

ToolRegistry.register(
    "append_to_file",
    "【分段生成】追加内容到已有文件末尾。用于分段构建长文件：先 write_file 创建文件写入第一段（import+前几个函数），再多次 append_to_file 追加后续函数。每次 100-200 行最佳。",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "目标文件路径（需已存在，用write_file创建）"},
            "content": {"type": "string", "description": "要追加的代码内容"},
        },
        "required": ["file_path", "content"],
    },
    append_to_file,
)

ToolRegistry.register(
    "write_file",
    "创建/覆盖写入文件。相对路径写到 output/ 目录。可配合 append_to_file 分段生成长文件。",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "目标路径（相对路径 → output/）"},
            "content": {"type": "string", "description": "要写入的内容"},
            "overwrite": {"type": "boolean", "description": "覆盖已有文件，默认true"},
        },
        "required": ["file_path", "content"],
    },
    write_file,
)

ToolRegistry.register(
    "list_dir",
    "列出目录下的文件。",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录路径"},
            "recursive": {"type": "boolean", "description": "是否递归"},
            "pattern": {"type": "string", "description": "glob模式"},
        },
        "required": ["path"],
    },
    list_dir,
)

ToolRegistry.register(
    "scan_dir",
    "一步浏览目录: 列出所有文件 + Python文件的函数/类摘要。比 list_dir 更高效。",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录路径"},
            "max_depth": {"type": "integer", "description": "最大深度，默认3"},
        },
        "required": ["path"],
    },
    scan_dir,
)

ToolRegistry.register(
    "copy_to_workspace",
    "【推荐】复制源文件到工作区。复制后可用 assemble_task 的 source_files 引用此文件（工具从磁盘读取拼接，不经过JSON）。",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "要复制的源文件路径"},
            "workspace_name": {"type": "string", "description": "工作区中的文件名（可选，默认用原文件名）"},
        },
        "required": ["file_path"],
    },
    copy_to_workspace,
)

ToolRegistry.register(
    "save_to_workspace",
    "手动保存内容到工作区文件。用于保存中间结果、拼凑的代码等。",
    {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "工作区中的文件名"},
            "content": {"type": "string", "description": "要保存的内容"},
        },
        "required": ["filename", "content"],
    },
    save_to_workspace,
)

ToolRegistry.register(
    "list_workspace",
    "查看工作区中已保存的所有文件。",
    {
        "type": "object",
        "properties": {},
    },
    list_workspace,
)

ToolRegistry.register(
    "grep_search",
    "在文件/目录中搜索正则。可设 context_lines 显示上下文。",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则表达式"},
            "path": {"type": "string", "description": "搜索路径"},
            "glob": {"type": "string", "description": "文件过滤 glob"},
            "max_results": {"type": "integer", "description": "最大结果数"},
            "context_lines": {"type": "integer", "description": "上下文行数"},
        },
        "required": ["pattern"],
    },
    grep_search,
)

ToolRegistry.register(
    "read_function",
    "精确提取函数/类定义（AST解析）。自动保存到工作区，返回摘要+预览。完整代码在工作区文件中。",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "文件路径"},
            "function_name": {"type": "string", "description": "函数名或类名"},
        },
        "required": ["file_path", "function_name"],
    },
    read_function,
)

ToolRegistry.register(
    "multi_file_search",
    "搜索多个关键词，返回跨文件匹配。",
    {
        "type": "object",
        "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "关键词列表"},
            "path": {"type": "string", "description": "搜索根目录"},
            "glob": {"type": "string", "description": "文件过滤"},
        },
        "required": ["keywords"],
    },
    multi_file_search,
)
