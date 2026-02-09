"""
代码操作工具 - apply_patch, run_code, validate_task, assemble_task
"""
import ast
import subprocess
import sys
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from .registry import ToolRegistry
from ..config import CODE_EXEC_TIMEOUT, PYTHON_EXECUTABLE, OUTPUT_DIR, WORKSPACE_DIR


def _resolve_path(file_path: str) -> Path:
    """解析文件路径，支持 workspace/ 短路径"""
    p = Path(file_path).expanduser()
    if not p.is_absolute():
        if file_path.startswith("workspace/") or file_path.startswith("workspace\\"):
            ws_path = WORKSPACE_DIR / file_path[len("workspace/"):]
            if ws_path.exists():
                return ws_path.resolve()
        if file_path.startswith("output/") or file_path.startswith("output\\"):
            out_path = OUTPUT_DIR / file_path[len("output/"):]
            if out_path.exists():
                return out_path.resolve()
        ws_path = WORKSPACE_DIR / file_path
        if ws_path.exists():
            return ws_path.resolve()
        out_path = OUTPUT_DIR / file_path
        if out_path.exists():
            return out_path.resolve()
    return p.resolve()


# ==================== 模块检测 ====================

def _is_installable_module(module_name: str) -> bool:
    """
    判断一个模块名是否是标准库/已安装包（而非本地文件）。
    
    使用 importlib 检测，比白名单更通用：
    - 标准库 (math, os, typing, ...) → True
    - 已安装的包 (torch, numpy, ...) → True
    - 本地文件模块 (mmdit_v2_win_block_func, ...) → False
    """
    import importlib.util
    
    # 空名或 __ 开头的跳过
    if not module_name or module_name.startswith('__'):
        return False
    
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False
        # 如果 origin 是 None（namespace package）或者来自 site-packages/stdlib → 可安装
        origin = getattr(spec, 'origin', None)
        if origin is None:
            # namespace package 或 built-in
            return True
        origin_str = str(origin)
        # 如果路径包含 site-packages 或 lib/python → 已安装包
        if 'site-packages' in origin_str or 'lib/python' in origin_str.replace('\\', '/'):
            return True
        # frozen 模块 (如 _frozen_importlib)
        if origin_str.startswith('<') or 'frozen' in origin_str:
            return True
        # 标准库路径（Python 安装目录下）
        import sysconfig
        stdlib_path = sysconfig.get_paths().get('stdlib', '')
        if stdlib_path and origin_str.startswith(stdlib_path):
            return True
        # 额外：检查是否能在 sys.stdlib_module_names 中找到（Python 3.10+）
        stdlib_names = getattr(sys, 'stdlib_module_names', set())
        if module_name in stdlib_names:
            return True
        # 对于 Windows 上的 DLL（如 _socket.pyd）
        if origin_str.endswith(('.pyd', '.so')):
            return True
        return False
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


# ==================== AST 函数提取 ====================

def _extract_file_header(content: str, tree: ast.Module, clean_local_imports: bool = False) -> str:
    """
    提取文件头部：import语句、模块注释、顶层常量（def/class之前的所有内容）
    
    Args:
        content: 文件内容
        tree: AST
        clean_local_imports: 是否清除非标准库的本地import（如 from xxx import ...），
                            用于选择性提取时避免残留的外部模块引用
    """
    lines = content.splitlines()
    header_end = len(lines)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            header_end = node.lineno - 1
            if node.decorator_list:
                header_end = min(header_end, min(d.lineno - 1 for d in node.decorator_list))
            break

    header_lines = lines[:header_end]
    
    if clean_local_imports:
        cleaned = []
        for line in header_lines:
            stripped = line.strip()
            # 检查 from xxx import ... 或 import xxx
            if stripped.startswith('from ') or stripped.startswith('import '):
                # 提取模块名
                if stripped.startswith('from '):
                    parts = stripped.split()
                    module = parts[1] if len(parts) > 1 else ''
                else:
                    parts = stripped.split()
                    module = parts[1].split(',')[0].strip() if len(parts) > 1 else ''
                
                root_module = module.split('.')[0]
                if root_module and not _is_installable_module(root_module):
                    # 跳过本地模块的 import（如 from mmdit_v2_win_block_func import ...）
                    continue
                # 跳过 sys.path 操作
                if 'sys.path' in stripped:
                    continue
            elif stripped.startswith('sys.path'):
                continue
            cleaned.append(line)
        header_lines = cleaned
    
    header = "\n".join(header_lines).rstrip()
    return header


def _extract_functions(
    content: str, tree: ast.Module, names: List[str],
    strip_decorators: bool = True,
) -> Tuple[str, List[str]]:
    """
    从 AST 中提取指定名称的函数/类。

    Args:
        strip_decorators: 是否移除装饰器（如 @register_decomposition 等），
                         避免在独立文件中执行时引发副作用
    Returns:
        (extracted_code, not_found_names)
    """
    lines = content.splitlines()
    extracted = []
    found = set()

    # 按源文件中的出现顺序提取（保持依赖顺序）
    nodes_map = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names and node.name not in nodes_map:
                nodes_map[node.name] = node

    for name in names:
        if name in nodes_map:
            node = nodes_map[name]
            end = node.end_lineno
            if strip_decorators or not node.decorator_list:
                # 不包含装饰器，直接从 def/class 行开始
                start = node.lineno - 1
            else:
                # 包含装饰器
                start = min(node.lineno - 1, min(d.lineno - 1 for d in node.decorator_list))
            extracted.append("\n".join(lines[start:end]))
            found.add(name)

    not_found = [n for n in names if n not in found]
    return "\n\n\n".join(extracted), not_found


def _extract_functions_from_file(file_path: Path, function_names: List[str]) -> Tuple[str, str, List[str]]:
    """
    从文件中提取文件头 + 指定函数。

    Returns:
        (header, functions_code, not_found_names)
    """
    content = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "", "", function_names

    # 选择性提取时清除非标准库的本地import，避免残留的外部模块引用
    header = _extract_file_header(content, tree, clean_local_imports=True)
    funcs_code, not_found = _extract_functions(content, tree, function_names)
    return header, funcs_code, not_found


# ==================== 工具实现 ====================

def apply_patch(args: Dict[str, Any]) -> Dict[str, Any]:
    """通过 old_string/new_string 修改文件"""
    file_path = args["file_path"]
    old_string = args["old_string"]
    new_string = args["new_string"]

    path = _resolve_path(file_path)

    if old_string == "":
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_string, encoding="utf-8")
            return {"status": "success", "output": f"文件已创建/写入: {path}", "error": ""}
        except Exception as e:
            return {"status": "error", "output": "", "error": str(e)}

    if not path.exists():
        return {"status": "error", "output": "", "error": f"文件不存在: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
        if old_string not in content:
            lines = content.split("\n")
            old_lines = old_string.split("\n")
            found = False
            for i in range(len(lines) - len(old_lines) + 1):
                block = lines[i:i + len(old_lines)]
                if all(a.strip() == b.strip() for a, b in zip(block, old_lines)):
                    original_block = "\n".join(block)
                    content = content.replace(original_block, new_string, 1)
                    found = True
                    break
            if not found:
                return {"status": "error", "output": "", "error": "old_string 在文件中未找到"}
        else:
            count = content.count(old_string)
            if count > 1:
                return {"status": "error", "output": "",
                        "error": f"old_string 有 {count} 处匹配"}
            content = content.replace(old_string, new_string, 1)

        path.write_text(content, encoding="utf-8")
        return {"status": "success", "output": f"已修改: {path}", "error": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


def run_code(args: Dict[str, Any]) -> Dict[str, Any]:
    """运行 Python 代码"""
    code = args.get("code", "")
    file_path = args.get("file_path", "")
    timeout = args.get("timeout", CODE_EXEC_TIMEOUT)

    try:
        if file_path:
            path = _resolve_path(file_path)
            if not path.exists():
                return {"status": "error", "output": "", "error": f"文件不存在: {path}"}
            cmd = [PYTHON_EXECUTABLE, str(path)]
        elif code:
            tmp = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8")
            tmp.write(code)
            tmp.close()
            cmd = [PYTHON_EXECUTABLE, tmp.name]
        else:
            return {"status": "error", "output": "", "error": "需要 code 或 file_path"}

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        combined = ""
        if stdout:
            combined += f"[stdout]\n{stdout}\n"
        if stderr:
            combined += f"[stderr]\n{stderr}\n"

        if result.returncode == 0:
            return {"status": "success", "output": combined or "success, no output", "error": ""}
        else:
            return {"status": "error", "output": combined, "error": f"exit code {result.returncode}"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "", "error": f"timeout ({timeout}s)"}
    except Exception as e:
        return {"status": "error", "output": "", "error": str(e)}


def assemble_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    【核心工具】拼装 KernelBench 任务文件。

    支持三种模式：
    1. 完整嵌入: source_files=["workspace/file.py"] → 嵌入整个文件
    2. 选择性提取: source_files=[{"path": "workspace/file.py", "functions": ["func1", "func2"]}]
       → 只提取指定函数（自动包含文件头 import，自动清除非标准库的本地 import）
    3. 排除模式: source_files=[{"path": "workspace/file.py", "exclude_functions": ["unused"]}]
       → 嵌入整个文件但移除指定函数

    LLM 只需提供 Model 类和 get_inputs 代码（<100行），源码由工具从磁盘读取。
    """
    source_files = args.get("source_files", [])
    imports_code = args.get("imports_code", "")
    model_code = args.get("model_code", "")
    helper_code = args.get("helper_code", "")
    get_inputs_code = args.get("get_inputs_code", "")
    get_init_inputs_code = args.get("get_init_inputs_code", "")
    output_file = args.get("output_file", "task_output.py")

    parts = []
    headers_seen = set()
    warnings = []

    # 1. 处理源文件
    for sf in source_files:
        # 支持三种格式:
        #   字符串: 完整嵌入
        #   dict + functions: 选择性提取
        #   dict + exclude_functions: 嵌入除排除函数外的所有内容
        if isinstance(sf, str):
            # 完整嵌入
            sf_path = _resolve_path(sf)
            if not sf_path.exists():
                return {"status": "error", "output": "",
                        "error": f"源文件不存在: {sf} (resolved: {sf_path})"}
            try:
                content = sf_path.read_text(encoding="utf-8")
                parts.append(f"# ===== Source: {sf_path.name} =====\n{content}")
            except Exception as e:
                return {"status": "error", "output": "", "error": f"读取 {sf} 失败: {e}"}

        elif isinstance(sf, dict):
            sf_path_str = sf.get("path", "")
            func_names = sf.get("functions", [])
            exclude_names = sf.get("exclude_functions", [])
            include_header = sf.get("include_header", True)

            if not sf_path_str:
                return {"status": "error", "output": "", "error": "source_files dict 需要 'path' 字段"}

            sf_path = _resolve_path(sf_path_str)
            if not sf_path.exists():
                return {"status": "error", "output": "",
                        "error": f"源文件不存在: {sf_path_str} (resolved: {sf_path})"}

            if exclude_names and not func_names:
                # 排除模式：嵌入整个文件，但移除指定的函数/类
                try:
                    content = sf_path.read_text(encoding="utf-8")
                    tree = ast.parse(content)
                except Exception as e:
                    return {"status": "error", "output": "", "error": f"解析 {sf_path_str} 失败: {e}"}

                lines = content.splitlines()
                exclude_set = set(exclude_names)
                # 收集需要排除的行区间
                remove_ranges = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node.name in exclude_set:
                            start = node.lineno - 1
                            if node.decorator_list:
                                start = min(start, min(d.lineno - 1 for d in node.decorator_list))
                            end = node.end_lineno
                            remove_ranges.append((start, end))

                # 按行移除
                if remove_ranges:
                    remove_ranges.sort()
                    keep_lines = []
                    skip_until = 0
                    for i, line in enumerate(lines):
                        if i < skip_until:
                            continue
                        removed = False
                        for rstart, rend in remove_ranges:
                            if rstart <= i < rend:
                                skip_until = rend
                                removed = True
                                break
                        if not removed:
                            keep_lines.append(line)
                    content = "\n".join(keep_lines)
                    excluded_count = len(remove_ranges)
                    warnings.append(f"已从 {sf_path.name} 排除 {excluded_count} 个函数: {exclude_names}")

                parts.append(f"# ===== Source: {sf_path.name} (excluded: {', '.join(exclude_names)}) =====\n{content}")

            elif func_names:
                # 选择性提取模式
                try:
                    header, funcs_code, not_found = _extract_functions_from_file(sf_path, func_names)
                except Exception as e:
                    return {"status": "error", "output": "", "error": f"提取函数失败: {e}"}

                if not_found:
                    warnings.append(f"未找到函数: {not_found} (in {sf_path.name})")

                # 只在第一次加入 header（避免重复 import）
                source_parts = []
                if include_header and header and sf_path.name not in headers_seen:
                    source_parts.append(header)
                    headers_seen.add(sf_path.name)
                if funcs_code:
                    source_parts.append(funcs_code)

                if source_parts:
                    label = ", ".join(func_names[:5])
                    if len(func_names) > 5:
                        label += f" ... ({len(func_names)} total)"
                    parts.append(f"# ===== From {sf_path.name}: {label} =====\n" + "\n\n".join(source_parts))
            else:
                return {"status": "error", "output": "",
                        "error": f"{sf_path_str}: dict 格式需要 'functions' 或 'exclude_functions' 字段"}
        else:
            return {"status": "error", "output": "", "error": f"source_files 元素格式错误: {type(sf)}"}

    # 2. 额外 import（去重：如果源文件已包含则跳过）
    if imports_code.strip():
        # 检查源文件中是否已有这些 import
        source_content = "\n".join(parts)
        dedup_lines = []
        for line in imports_code.strip().splitlines():
            stripped = line.strip()
            if stripped and (stripped.startswith("import ") or stripped.startswith("from ")):
                if stripped in source_content:
                    continue  # 跳过已存在的 import
            dedup_lines.append(line)
        dedup_imports = "\n".join(dedup_lines).strip()
        if dedup_imports:
            parts.append(f"\n# ===== Additional imports =====\n{dedup_imports}")

    # 3. 辅助代码（去重 import 行）
    if helper_code.strip():
        source_content = "\n".join(parts)
        dedup_lines = []
        for line in helper_code.strip().splitlines():
            stripped = line.strip()
            if stripped and (stripped.startswith("import ") or stripped.startswith("from ")):
                if stripped in source_content:
                    continue  # 跳过已存在的 import
            dedup_lines.append(line)
        dedup_helper = "\n".join(dedup_lines).strip()
        if dedup_helper:
            parts.append(f"\n# ===== Helper =====\n{dedup_helper}")

    # 4. Model 类
    if model_code.strip():
        parts.append(f"\n# ===== Model =====\n{model_code}")

    # 5. get_inputs
    if get_inputs_code.strip():
        parts.append(f"\n# ===== Test inputs =====\n{get_inputs_code}")

    # 6. get_init_inputs
    if get_init_inputs_code.strip():
        parts.append(f"\n{get_init_inputs_code}")

    full_code = "\n\n".join(parts)

    # 保存
    out_path = Path(output_file).expanduser()
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR / output_file
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_code, encoding="utf-8")

    total_lines = full_code.count("\n") + 1
    output_msg = f"[任务文件已生成] {out_path}\n总行数: {total_lines}"
    if warnings:
        output_msg += "\n警告: " + "; ".join(warnings)

    return {"status": "success", "output": output_msg, "error": ""}


def validate_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """预运行验证 KernelBench 格式的任务代码"""
    from ..task.test_constructor import TestConstructor

    task_code = args.get("task_code", "")
    task_file = args.get("task_file", "")

    if task_file and not task_code:
        path = _resolve_path(task_file)
        if path.exists():
            task_code = path.read_text(encoding="utf-8")
        else:
            return {"status": "error", "output": "", "error": f"任务文件不存在: {path}"}

    if not task_code.strip():
        return {"status": "error", "output": "", "error": "task_code 不能为空"}

    return TestConstructor.run_validation(task_code, timeout=args.get("timeout", 60))


def test_with_reference(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    对比测试：将生成的 Model 与 reference 函数对比，支持多组输入。

    reference_code 必须定义 reference_forward(inputs, init_inputs) 函数。
    multi_inputs_code 可选，定义 get_multi_test_inputs() 返回 [{name, inputs}] 列表。
    """
    task_file = args.get("task_file", "")
    task_code = args.get("task_code", "")
    reference_code = args.get("reference_code", "")
    multi_inputs_code = args.get("multi_inputs_code", "")

    if task_file and not task_code:
        path = _resolve_path(task_file)
        if path.exists():
            task_code = path.read_text(encoding="utf-8")
        else:
            return {"status": "error", "output": "", "error": f"任务文件不存在: {path}"}

    if not task_code.strip():
        return {"status": "error", "output": "", "error": "task_code 不能为空"}
    if not reference_code.strip():
        return {"status": "error", "output": "", "error": "reference_code 不能为空"}

    from ..task.test_constructor import TestConstructor  # noqa: F811
    return TestConstructor.run_reference_test(
        task_code, reference_code, multi_inputs_code,
        timeout=args.get("timeout", 120),
    )


# ========== 注册 ==========

ToolRegistry.register(
    "apply_patch",
    "修改文件: 查找 old_string 替换为 new_string。支持 workspace/ 路径。\n"
    "【骨架+填充模式】可先用 write_file 写骨架（def func(): pass），再用 apply_patch 替换 pass 为真实代码。",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "文件路径"},
            "old_string": {"type": "string", "description": "要替换的文本（空=创建文件）"},
            "new_string": {"type": "string", "description": "替换后文本"},
        },
        "required": ["file_path", "old_string", "new_string"],
    },
    apply_patch,
)

ToolRegistry.register(
    "run_code",
    "运行 Python 代码。传入 code 或 file_path。支持 workspace/ 路径。",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python 代码"},
            "file_path": {"type": "string", "description": "Python 文件路径"},
            "timeout": {"type": "integer", "description": "超时秒数"},
        },
        "required": [],
    },
    run_code,
)

ToolRegistry.register(
    "assemble_task",
    "【选择性拼装】从 workspace 源文件中提取指定函数 + 你的 Model/get_inputs → 自包含任务文件。\n"
    "三种用法:\n"
    '  完整嵌入: source_files=["workspace/file.py"]\n'
    '  选择函数: source_files=[{"path": "workspace/file.py", "functions": ["func1", "func2"]}]\n'
    '  排除函数: source_files=[{"path": "workspace/file.py", "exclude_functions": ["unused1"]}]\n'
    "工具用 AST 解析精确提取/排除函数，自动包含 import，自动清除非标准库的本地 import。\n"
    "你只写 Model + get_inputs（<100行）。",
    {
        "type": "object",
        "properties": {
            "source_files": {
                "type": "array",
                "description": "源文件列表。字符串=完整嵌入，dict={path, functions}=选择性提取，dict={path, exclude_functions}=排除模式"
            },
            "imports_code": {
                "type": "string",
                "description": "额外import（可选）"
            },
            "model_code": {
                "type": "string",
                "description": "Model 类代码"
            },
            "helper_code": {
                "type": "string",
                "description": "辅助代码（常量、配置函数等）"
            },
            "get_inputs_code": {
                "type": "string",
                "description": "get_inputs() 函数"
            },
            "get_init_inputs_code": {
                "type": "string",
                "description": "get_init_inputs() 函数"
            },
            "output_file": {
                "type": "string",
                "description": "输出文件名（默认task_output.py）"
            },
        },
        "required": ["source_files", "model_code", "get_inputs_code", "get_init_inputs_code"],
    },
    assemble_task,
)

ToolRegistry.register(
    "validate_task",
    "预运行验证任务代码。传 task_file（文件路径）或 task_code（代码字符串）。\n"
    "检查: 实例化→forward→NaN/Inf→一致性→输出统计",
    {
        "type": "object",
        "properties": {
            "task_code": {"type": "string", "description": "任务代码（与 task_file 二选一）"},
            "task_file": {"type": "string", "description": "任务文件路径"},
            "timeout": {"type": "integer", "description": "超时秒数"},
        },
        "required": [],
    },
    validate_task,
)

ToolRegistry.register(
    "test_with_reference",
    "【正确性验证】将 Model 输出与 reference 函数对比，支持多组输入。\n"
    "用法：提供 reference_code（定义 reference_forward(inputs, init_inputs)）和可选的 multi_inputs_code。\n"
    "示例 reference_code:\n"
    '  def reference_forward(inputs, init_inputs):\\n'
    '      return torch._chunk_cat(list(inputs), init_inputs[0], init_inputs[1])\\n'
    "示例 multi_inputs_code:\n"
    '  def get_multi_test_inputs():\\n'
    '      return [{"name": "case1", "inputs": [torch.randn(4,5), torch.randn(6,5)]}, ...]',
    {
        "type": "object",
        "properties": {
            "task_file": {"type": "string", "description": "任务文件路径"},
            "task_code": {"type": "string", "description": "任务代码（与 task_file 二选一）"},
            "reference_code": {
                "type": "string",
                "description": "Reference 代码，必须定义 reference_forward(inputs, init_inputs) 函数"
            },
            "multi_inputs_code": {
                "type": "string",
                "description": "可选：定义 get_multi_test_inputs() 返回 [{name, inputs}] 列表"
            },
            "timeout": {"type": "integer", "description": "超时秒数（默认120）"},
        },
        "required": ["reference_code"],
    },
    test_with_reference,
)
