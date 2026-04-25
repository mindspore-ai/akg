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

"""AST 解析工具：文件头提取、函数提取、依赖追踪"""

import ast
import sys
import re
from typing import Dict, List, Tuple, Optional


def is_installable_module(module_name: str) -> bool:
    """判断模块名是否是标准库/已安装包"""
    import importlib.util

    if not module_name or module_name.startswith('__'):
        return False

    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False
        origin = getattr(spec, 'origin', None)
        if origin is None:
            return True
        origin_str = str(origin)
        if 'site-packages' in origin_str or 'lib/python' in origin_str.replace('\\', '/'):
            return True
        if origin_str.startswith('<') or 'frozen' in origin_str:
            return True
        import sysconfig
        stdlib_path = sysconfig.get_paths().get('stdlib', '')
        if stdlib_path and origin_str.startswith(stdlib_path):
            return True
        stdlib_names = getattr(sys, 'stdlib_module_names', set())
        if module_name in stdlib_names:
            return True
        if origin_str.endswith(('.pyd', '.so')):
            return True
        return False
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def extract_file_header(content: str, tree: ast.Module, clean_local_imports: bool = False) -> str:
    """提取文件头部：import语句、模块注释、顶层常量"""
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
            if stripped.startswith('from ') or stripped.startswith('import '):
                if stripped.startswith('from '):
                    parts = stripped.split()
                    module = parts[1] if len(parts) > 1 else ''
                else:
                    parts = stripped.split()
                    module = parts[1].split(',')[0].strip() if len(parts) > 1 else ''
                root_module = module.split('.')[0]
                if root_module and not is_installable_module(root_module):
                    continue
                if 'sys.path' in stripped:
                    continue
            elif stripped.startswith('sys.path'):
                continue
            cleaned.append(line)
        header_lines = cleaned

    header = "\n".join(header_lines).rstrip()
    return header


def extract_functions(
    content: str, tree: ast.Module, names: List[str],
    strip_decorators: bool = True,
) -> Tuple[str, List[str]]:
    """从 AST 中提取指定名称的函数/类"""
    lines = content.splitlines()
    extracted = []
    found = set()

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
                start = node.lineno - 1
            else:
                start = min(node.lineno - 1, min(d.lineno - 1 for d in node.decorator_list))
            extracted.append("\n".join(lines[start:end]))
            found.add(name)

    not_found = [n for n in names if n not in found]
    return "\n\n\n".join(extracted), not_found


def resolve_attribute_chain(node) -> Optional[str]:
    """解析 AST 属性链为点分字符串"""
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


def parse_import_aliases(tree: ast.Module) -> Dict[str, str]:
    """从文件 import 语句和顶层赋值中解析模块/名字别名映射"""
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
            if (len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                source = resolve_attribute_chain(node.value)
                if source:
                    aliases[node.targets[0].id] = source
    return aliases


def has_private_segments(module_path: str) -> bool:
    """检查模块路径是否包含私有段（如 torch._prims_common）"""
    segments = module_path.split(".")
    for seg in segments[1:]:
        if seg.startswith("_"):
            return True
    return False


def collect_local_names(func_node: ast.AST) -> set:
    """收集函数直接作用域内的局部变量名"""
    locals_set = set()

    if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in (func_node.args.args + func_node.args.posonlyargs
                    + func_node.args.kwonlyargs):
            locals_set.add(arg.arg)
        if func_node.args.vararg:
            locals_set.add(func_node.args.vararg.arg)
        if func_node.args.kwarg:
            locals_set.add(func_node.args.kwarg.arg)

    def _walk_shallow(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield child
            yield from _walk_shallow(child)

    def _add_target_names(target):
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


def trace_function_deps(
    content: str, tree: ast.Module, entry_names: List[str]
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    AST 依赖追踪：给定入口函数名，递归找出同文件内所有被调用的函数/类。
    通过分析文件 import 自动检测需要内联的外部模块调用。
    """
    toplevel_names = set()
    toplevel_nodes = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            toplevel_names.add(node.name)
            toplevel_nodes[node.name] = node

    import_aliases = parse_import_aliases(tree)

    def _get_called_names(node: ast.AST) -> Tuple[set, set]:
        internal = set()
        external = set()
        local_names = collect_local_names(node)

        body = getattr(node, 'body', [])
        for stmt in body:
            for child in ast.walk(stmt):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func

                if isinstance(func, ast.Name):
                    if func.id in toplevel_names:
                        internal.add(func.id)
                    elif func.id in import_aliases:
                        source = import_aliases[func.id]
                        if has_private_segments(source):
                            external.add((func.id, source))

                elif isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        obj_name = func.value.id
                        method_name = func.attr
                        if obj_name in toplevel_names:
                            internal.add(obj_name)
                        elif obj_name == "self":
                            pass
                        elif obj_name in local_names:
                            pass
                        elif obj_name in import_aliases:
                            source = import_aliases[obj_name]
                            if has_private_segments(source):
                                external.add((f"{obj_name}.{method_name}", source))

        return internal, external

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

    ordered = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in visited:
                ordered.append(node.name)

    return ordered, sorted(all_external, key=lambda x: x[0])


def clean_unused_imports(header: str, funcs_code: str) -> str:
    """精简 header 中未被 funcs_code 使用的 import"""
    try:
        header_tree = ast.parse(header)
    except SyntaxError:
        return header

    header_lines = header.splitlines()
    remove_ranges = []

    def _name_used(name: str) -> bool:
        return bool(re.search(r'\b' + re.escape(name) + r'\b', funcs_code))

    for node in ast.iter_child_nodes(header_tree):
        if isinstance(node, ast.Import):
            any_used = False
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                if _name_used(name):
                    any_used = True
                    break
            if not any_used:
                remove_ranges.append((node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.ImportFrom):
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

    result = []
    prev_empty = False
    for line in keep_lines:
        is_empty = line.strip() == ""
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty

    return "\n".join(result).rstrip()
