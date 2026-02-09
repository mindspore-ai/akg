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


def _resolve_attribute_chain(node) -> Optional[str]:
    """
    解析 AST 属性链为点分字符串。
    例: torch._ops.ops.aten → "torch._ops.ops.aten"
    返回 None 如果不是简单的属性链。
    """
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()
        return ".".join(parts)
    return None


def _parse_import_aliases(tree: ast.Module) -> Dict[str, str]:
    """
    从文件 import 语句和顶层赋值中解析模块/名字别名映射。

    Returns:
        {alias_name: source_module_path}
        例: {"utils": "torch._prims_common", "aten": "torch._ops.ops.aten",
             "torch": "torch", "Tensor": "torch.Tensor"}
    """
    aliases: Dict[str, str] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                aliases[name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                source = f"{module}.{alias.name}" if module else alias.name
                aliases[name] = source
        elif isinstance(node, ast.Assign):
            # 顶层赋值如: aten = torch._ops.ops.aten
            if (len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                source = _resolve_attribute_chain(node.value)
                if source:
                    aliases[node.targets[0].id] = source
    return aliases


def _has_private_segments(module_path: str) -> bool:
    """
    检查模块路径是否包含私有段（根包之后的以 _ 开头的部分）。
    内部/私有模块中的函数通常需要内联到独立文件中。

    例: "torch._prims_common" → True  (_prims_common 以 _ 开头)
        "torch.nn.functional" → False
        "torch" → False
    """
    segments = module_path.split(".")
    for seg in segments[1:]:  # 跳过根包名
        if seg.startswith("_"):
            return True
    return False


def _collect_local_names(func_node: ast.AST) -> set:
    """
    收集函数直接作用域内的局部变量名（参数、赋值、循环变量等）。
    不进入嵌套函数/类，避免内层作用域污染。
    """
    locals_set = set()

    # 函数参数
    if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in (func_node.args.args + func_node.args.posonlyargs
                    + func_node.args.kwonlyargs):
            locals_set.add(arg.arg)
        if func_node.args.vararg:
            locals_set.add(func_node.args.vararg.arg)
        if func_node.args.kwarg:
            locals_set.add(func_node.args.kwarg.arg)

    def _walk_shallow(node):
        """遍历 AST 但不进入嵌套函数/类定义"""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield child
            yield from _walk_shallow(child)

    def _add_target_names(target):
        """从赋值目标中提取变量名"""
        if isinstance(target, ast.Name):
            locals_set.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    locals_set.add(elt.id)

    for child in _walk_shallow(func_node):
        if isinstance(child, ast.Assign):
            for t in child.targets:
                _add_target_names(t)
        elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(child.target, ast.Name):
                locals_set.add(child.target.id)
        elif isinstance(child, ast.For):
            _add_target_names(child.target)
        elif isinstance(child, ast.comprehension):
            _add_target_names(child.target)
        elif isinstance(child, ast.With):
            for item in child.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    locals_set.add(item.optional_vars.id)

    return locals_set


def _trace_function_deps(
    content: str, tree: ast.Module, entry_names: List[str]
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    AST 依赖追踪：给定入口函数名，递归找出同文件内所有被调用的函数/类。
    通过分析文件 import 自动检测需要内联的外部模块调用（无硬编码列表）。

    外部调用判定规则（完全基于 import 分析）：
    - 属性调用 X.method(): 若 X 是 import 别名且来源模块含私有段 → 报告
    - 直接调用 func(): 若 func 是从内部模块 import 的名字 → 报告
    - X 是函数局部变量 → 跳过（如 tensor.size()）
    - X 是公共 API 模块别名（如 torch, numpy, F）→ 跳过

    Returns:
        (有序的完整依赖列表, 外部调用列表)
        - 依赖列表按源文件出现顺序排列
        - 外部调用列表: [(调用表达式, 来源模块路径), ...]
    """
    # 1. 收集文件内所有顶层函数/类名
    toplevel_names = set()
    toplevel_nodes = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            toplevel_names.add(node.name)
            toplevel_nodes[node.name] = node

    # 2. 解析文件 import 得到别名映射
    import_aliases = _parse_import_aliases(tree)

    # 3. 提取每个函数/类体内调用的函数名（只扫描 body，跳过装饰器）
    def _get_called_names(node: ast.AST) -> Tuple[set, set]:
        """返回 (同文件调用集合, 外部调用集合)"""
        internal = set()
        external = set()  # {(call_str, source_module)}
        local_names = _collect_local_names(node)

        # 只遍历 body，跳过装饰器（装饰器在提取时会被自动移除）
        body = getattr(node, 'body', [])
        for stmt in body:
            for child in ast.walk(stmt):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func

                # 直接调用: func_name(...)
                if isinstance(func, ast.Name):
                    if func.id in toplevel_names:
                        internal.add(func.id)
                    elif func.id in import_aliases:
                        source = import_aliases[func.id]
                        if _has_private_segments(source):
                            external.add((func.id, source))

                # 属性调用: obj.method(...)
                elif isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        obj_name = func.value.id
                        method_name = func.attr
                        if obj_name in toplevel_names:
                            internal.add(obj_name)
                        elif obj_name == "self":
                            pass
                        elif obj_name in local_names:
                            pass  # 局部变量的方法调用（如 tensor.size()）
                        elif obj_name in import_aliases:
                            source = import_aliases[obj_name]
                            if _has_private_segments(source):
                                external.add((f"{obj_name}.{method_name}", source))

        return internal, external

    # 4. BFS 追踪依赖
    visited = set()
    all_external = set()
    queue = list(entry_names)
    for name in queue:
        if name in visited:
            continue
        visited.add(name)
        if name in toplevel_nodes:
            internal, external = _get_called_names(toplevel_nodes[name])
            all_external.update(external)
            for callee in internal:
                if callee not in visited:
                    queue.append(callee)

    # 5. 按源文件出现顺序排列
    ordered = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in visited:
                ordered.append(node.name)

    return ordered, sorted(all_external, key=lambda x: x[0])


def _clean_unused_imports(header: str, funcs_code: str) -> str:
    """
    精简 header 中未被 funcs_code 使用的 import。

    使用 AST 级别处理，通过 word-boundary 正则检查引用。
    无硬编码白名单，完全基于实际引用分析。

    保留策略：
    - import 引入的名字在 funcs_code 中被引用 → 保留
    - 非 import 行（注释、常量、空行）→ 总是保留
    """
    try:
        header_tree = ast.parse(header)
    except SyntaxError:
        return header  # 解析失败不做清理

    header_lines = header.splitlines()
    remove_ranges = []

    def _name_used(name: str) -> bool:
        """检查名字是否作为独立标识符出现在 funcs_code 中"""
        return bool(re.search(r'\b' + re.escape(name) + r'\b', funcs_code))

    for node in ast.iter_child_nodes(header_tree):
        if isinstance(node, ast.Import):
            # import X, import X.Y → 检查引入的名字是否被使用
            any_used = False
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                if _name_used(name):
                    any_used = True
                    break
            if not any_used:
                remove_ranges.append((node.lineno - 1, node.end_lineno))

        elif isinstance(node, ast.ImportFrom):
            # from X import a, b → 检查 a, b 是否被使用
            any_used = False
            for alias in node.names:
                name = alias.asname or alias.name
                if _name_used(name):
                    any_used = True
                    break
            if not any_used:
                remove_ranges.append((node.lineno - 1, node.end_lineno))

    if not remove_ranges:
        return header

    # 按行移除
    remove_ranges.sort()
    keep_lines = []
    skip_until = 0
    for i, line in enumerate(header_lines):
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

    # 去除连续空行
    result = []
    prev_empty = False
    for line in keep_lines:
        is_empty = line.strip() == ""
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty

    return "\n".join(result).rstrip()


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

    # 精简 header 中未被 funcs_code 使用的 import
    if funcs_code:
        header = _clean_unused_imports(header, funcs_code)

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

    # 0. 额外 import 放在最前面（确保源文件中使用的类型如 Optional 已定义）
    if imports_code.strip():
        parts.append(f"# ===== Imports =====\n{imports_code.strip()}")

    # 0.5 辅助代码（内联的外部函数等 helper）放在源文件之前
    if helper_code.strip():
        parts.append(f"# ===== Helper =====\n{helper_code.strip()}")

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

    # 2. (imports_code 和 helper_code 已在步骤 0/0.5 放置到源文件之前)

    # 3. Model 类
    if model_code.strip():
        parts.append(f"\n# ===== Model =====\n{model_code}")

    # 4. get_inputs
    if get_inputs_code.strip():
        parts.append(f"\n# ===== Test inputs =====\n{get_inputs_code}")

    # 5. get_init_inputs
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
        timeout=args.get("timeout", 120),  # noqa: E501
    )


def trace_dependencies(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    AST 依赖追踪：给定文件路径 + 入口函数名，自动找出所有被调用的同文件函数。
    用于 assemble_task 之前，确保不遗漏依赖。
    """
    file_path = args.get("file_path", "")
    entry_functions = args.get("entry_functions", [])

    if not file_path:
        return {"status": "error", "output": "", "error": "需要 file_path"}
    if not entry_functions:
        return {"status": "error", "output": "", "error": "需要 entry_functions 列表"}

    path = _resolve_path(file_path)
    if not path.exists():
        return {"status": "error", "output": "", "error": f"文件不存在: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except Exception as e:
        return {"status": "error", "output": "", "error": f"解析失败: {e}"}

    full_deps, external_calls = _trace_function_deps(content, tree, entry_functions)
    added = [n for n in full_deps if n not in entry_functions]

    output = f"入口函数: {entry_functions}\n"
    output += f"完整依赖链 ({len(full_deps)} 个函数):\n"
    for name in full_deps:
        marker = " <- 入口" if name in entry_functions else " <- 自动发现"
        output += f"  - {name}{marker}\n"

    if added:
        output += f"\n自动发现的额外依赖: {added}\n"
        output += "建议在 assemble_task 的 functions 中包含这些函数。\n"
    else:
        output += "\n无额外同文件依赖。\n"

    if external_calls:
        output += f"\n[!] 需要内联的外部调用 ({len(external_calls)} 个, 来自内部模块):\n"
        for call_str, source_module in external_calls:
            output += f"  - {call_str}  (来源: {source_module})\n"
        output += (
            "\n处理方式:\n"
            "  1. 用 read_function 在来源模块中查看原始函数签名和实现\n"
            "  2. 在 helper_code 中内联等价实现，参数签名必须与原始完全一致\n"
            "  3. 对于模块别名（如 aten = torch._ops.ops.aten），在 helper_code 中定义"
        )

    return {"status": "success", "output": output, "error": ""}


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
    "reference_forward 应调用原始 torch 函数作为 ground truth。\n"
    "multi_inputs_code 中的 get_multi_test_inputs() 返回多组测试用例，每组可指定 per-case init_inputs。",
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

ToolRegistry.register(
    "trace_dependencies",
    "【依赖追踪】给定入口函数名，自动找出同文件内所有被调用的函数。\n"
    "通过分析文件 import 自动识别需要内联的外部模块调用，附带来源模块路径。\n"
    "返回完整的依赖链、建议的 functions 列表、以及需要处理的外部依赖。",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "workspace 中的文件路径"},
            "entry_functions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "入口函数名列表"
            },
        },
        "required": ["file_path", "entry_functions"],
    },
    trace_dependencies,
)
