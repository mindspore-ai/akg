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
参数解析器 - 解析工具参数中的动态表达式

支持的表达式:
- read_json_file('/path/to/file.json')             → 读取 JSON 文件，返回 dict
- read_json_file('/path/to/file.json')['key']       → 读取 JSON 文件中的指定字段
- read_json_file('/path/to/file.json')['k1']['k2']  → 嵌套字段访问

使用方式:
    from akg_agents.core_v2.tools.arg_resolver import read_json_file, resolve_arguments
"""

import json
import re
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ==================== 核心工具函数 ====================

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    读取 JSON 文件并返回解析后的字典
    
    可直接 import 后使用:
        from akg_agents.core_v2.tools.arg_resolver import read_json_file
        data = read_json_file('/path/to/result.json')
        code = data['code']
    
    Args:
        file_path: JSON 文件路径（绝对路径或相对路径）
    
    Returns:
        解析后的字典（或列表等 JSON 值）
    
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 解析失败
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {file_path}")
    
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


# ==================== 表达式解析 ====================

# 匹配 read_json_file('path')['key1']['key2'] 格式的表达式
# 支持单引号和双引号
_EXPR_PATTERN = re.compile(
    r"""read_json_file\(\s*['"](.+?)['"]\s*\)((?:\s*\[\s*['"].*?['"]\s*\])*)"""
)


def _extract_keys(bracket_expr: str) -> list:
    """
    从方括号表达式中提取 key 列表
    
    例: "['code']['sub_key']" → ['code', 'sub_key']
    
    Args:
        bracket_expr: 方括号表达式字符串
    
    Returns:
        key 列表
    """
    return re.findall(r"""\[\s*['"](.*?)['"]\s*\]""", bracket_expr)


def _resolve_single_expr(file_path: str, bracket_part: str) -> Any:
    """
    解析单个 read_json_file 表达式
    
    Args:
        file_path: JSON 文件路径
        bracket_part: 方括号访问部分（可为空）
    
    Returns:
        解析后的值
    """
    data = read_json_file(file_path)
    
    # 提取并应用 key 访问链
    keys = _extract_keys(bracket_part)
    result = data
    for key in keys:
        if isinstance(result, dict):
            if key not in result:
                raise KeyError(f"键 '{key}' 不存在于 JSON 数据中。可用键: {list(result.keys())}")
            result = result[key]
        elif isinstance(result, list):
            # 支持数字索引
            try:
                result = result[int(key)]
            except (ValueError, IndexError) as e:
                raise KeyError(f"无法用 '{key}' 索引列表: {e}")
        else:
            raise KeyError(f"无法在 {type(result).__name__} 类型上访问键 '{key}'")
    
    return result


def resolve_value(value: str) -> Any:
    """
    解析单个参数值中的表达式
    
    如果值包含 read_json_file(...) 表达式，则执行并返回结果。
    如果值是纯字符串（不包含表达式），则原样返回。
    
    规则:
    - 如果整个值就是一个表达式，返回原始类型（可以是 dict, list, str 等）
    - 如果表达式嵌入在更大的字符串中，替换为字符串表示
    
    Args:
        value: 参数值
    
    Returns:
        解析后的值
    """
    if not isinstance(value, str):
        return value
    
    match = _EXPR_PATTERN.search(value)
    if not match:
        return value
    
    # 如果整个值就是一个表达式（去掉首尾空格后完全匹配），返回原始类型
    if match.group(0).strip() == value.strip():
        file_path = match.group(1)
        bracket_part = match.group(2)
        return _resolve_single_expr(file_path, bracket_part)
    
    # 否则，替换字符串中所有的表达式为其字符串表示
    def _replacer(m):
        fp = m.group(1)
        bp = m.group(2)
        resolved = _resolve_single_expr(fp, bp)
        return str(resolved)
    
    return _EXPR_PATTERN.sub(_replacer, value)


def resolve_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析参数字典中所有包含动态表达式的值
    
    遍历 arguments 中的每个值:
    - 如果是字符串且包含 read_json_file(...) 表达式，执行解析
    - 否则原样保留
    
    失败时保留原值并打印警告（不会抛出异常）。
    
    Args:
        arguments: 工具参数字典
    
    Returns:
        解析后的参数字典
    
    示例:
        >>> args = {
        ...     "task_desc": "read_json_file('/path/result.json')['code']",
        ...     "dsl": "triton",
        ...     "backend": "cuda"
        ... }
        >>> resolved = resolve_arguments(args)
        >>> # resolved["task_desc"] 会是 result.json 中 'code' 字段的值
        >>> # resolved["dsl"] 仍然是 "triton"
    """
    resolved = {}
    for key, value in arguments.items():
        try:
            resolved[key] = resolve_value(value)
            if resolved[key] != value:
                logger.info(f"[ArgResolver] 参数 '{key}' 表达式已解析")
        except Exception as e:
            logger.warning(f"[ArgResolver] 解析参数 '{key}' 失败: {e}, 保持原值")
            resolved[key] = value
    return resolved

