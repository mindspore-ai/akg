# Copyright 2025 Huawei Technologies Co., Ltd
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

import os
import sys
import yaml
import logging
import tempfile
import json
import re
import yaml
import hashlib
from dataclasses import dataclass
from pydantic import create_model as create_pydantic_model
from langchain.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)


@dataclass
class PyAikgStatus:
    success: bool = False
    error_log: str = ""
    info_log: str = ""
    phase: str = ""

    def copy(self):
        new_status = PyAikgStatus()
        new_status.success = self.success
        new_status.error_log = self.error_log
        new_status.info_log = self.info_log
        new_status.phase = self.phase
        return new_status

    def __str__(self):
        status_str = f"Phase: {self.phase}\n"
        status_str += f"Success: {self.success}\n"
        if self.info_log:
            status_str += f"Info: {self.info_log}\n"
        if self.error_log:
            status_str += f"Error: {self.error_log}"
        return status_str


def get_prompt_path():
    module = sys.modules['ai_kernel_generator']
    module_path = os.path.abspath(str(module.__file__))
    root_dir = os.path.dirname(os.path.dirname(module_path))
    prompt_dir = os.path.join(root_dir, "ai_kernel_generator/resources/prompts/")
    return prompt_dir


def load_yaml(yaml_path: str):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # 加载配置
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_log_dir(prefix_name: str = "test_log"):
    """
    在当前工作目录下创建唯一的临时目录用于测试日志存储。
    Args:
        prefix_name (str): 目录前缀名
    Returns:
        str: 创建的临时目录绝对路径
    """
    tmp_dir = os.path.abspath("tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    log_dir = tempfile.mkdtemp(prefix=f"{prefix_name}_", dir=tmp_dir)
    return log_dir


class ParserFactory:
    """
    解析器工厂类，提供常用的解析器创建和获取功能
    """

    # 类级别的预定义解析器缓存
    _code_parser = None
    _check_parser = None
    _api_parser = None
    _feature_parser = None
    @classmethod
    def get_code_parser(cls):
        """获取代码解析器"""
        if cls._code_parser is None:
            cls._code_parser = cls.create_output_parser(
                "CodeBlock",
                {
                    'code': (str, ...),
                    'description': (str, ...)
                }
            )
        return cls._code_parser

    @classmethod
    def get_check_parser(cls):
        """获取分析解析器"""
        if cls._check_parser is None:
            cls._check_parser = cls.create_output_parser(
                "CheckBlock",
                {
                    'result': (int, ...),
                    'reason': (str, ...),
                    'suggestions': (str, ...)
                }
            )
        return cls._check_parser

    @classmethod
    def get_api_parser(cls):
        """获取API解析器"""
        if cls._api_parser is None:
            cls._api_parser = cls.create_output_parser(
                "ApiBlock",
                {
                    'compute': (list[str], ...),
                    'composite': (list[str], ...),
                    'move': (list[str], ...),
                    'slicedata': (list[str], ...)
                }
            )
        return cls._api_parser

    @classmethod
    def get_feature_parser(cls):
        """获取特征匹配的的解析器"""
        if cls._feature_parser is None:
            cls._feature_parser = cls.create_output_parser(
                "FeatureBlock",
                {
                    'op_type': (str, ...),
                    'op_name_primary': (str, ...),
                    'op_name_secondary': (list[str], ...),
                    'op_shape': (list[str], ...),
                    'description': (str, ...)
                }
            )
        return cls._feature_parser
    @staticmethod
    def create_output_parser(parser_name, fields):
        """创建输出解析器

        Args:
            parser_name: 模型类名
            fields: 字段定义字典，格式为 {字段名: (类型, 默认值或...)}

        Returns:
            配置好的PydanticOutputParser实例
        """
        model_class = create_pydantic_model(parser_name, **fields)
        return PydanticOutputParser(pydantic_object=model_class)

    @staticmethod
    def robust_parse(content: str, parser: PydanticOutputParser):
        """稳健的解析方法，支持多种解析策略

        Args:
            content: 待解析的内容
            parser: PydanticOutputParser实例

        Returns:
            解析后的对象
        """

        # 策略1: 直接解析（适用于标准格式）
        try:
            return parser.parse(content)
        except Exception:
            logger.debug("直接解析失败，尝试提取JSON块")

        # 策略2: 多位置JSON提取
        try:
            extracted_json = ParserFactory._extract_json_comprehensive(content)
            if extracted_json:
                return parser.parse(extracted_json)
        except Exception:
            logger.debug("JSON提取解析失败")

        # 所有策略都失败
        logger.warning("无法从内容中提取有效的JSON格式")
        return ""

    @staticmethod
    def _extract_json_comprehensive(text: str) -> str:
        """全面的JSON提取，支持检测各个位置的JSON"""

        # 方法1: 优先查找末尾的完整JSON块（最常见）
        json_candidate = ParserFactory._extract_final_json(text)
        if json_candidate:
            return json_candidate

        # 方法2: 查找```json代码块
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        for block in reversed(json_blocks):  # 从后往前尝试
            try:
                json.loads(block.strip())
                return block.strip()
            except json.JSONDecodeError:
                continue

        # 方法3: 查找所有花括号包围的内容
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        # 优先尝试较后面的匹配（通常更完整）
        for match in reversed(matches):
            try:
                parsed = json.loads(match)
                # 验证是否包含期望的字段
                if isinstance(parsed, dict) and ('code' in parsed or 'description' in parsed):
                    return match
            except json.JSONDecodeError:
                continue

        # 方法4: 查找通用代码块中的JSON
        code_blocks = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
        for block in reversed(code_blocks):
            block = block.strip()
            if block.startswith('{') and block.endswith('}'):
                try:
                    json.loads(block)
                    return block
                except json.JSONDecodeError:
                    continue

        return None

    @staticmethod
    def _extract_final_json(text: str) -> str:
        """从文本末尾提取完整的JSON块"""
        # 查找最后一个{...}块
        last_brace_start = text.rfind('{')
        if last_brace_start == -1:
            return None

        # 从最后一个{开始，向后匹配完整的JSON
        brace_count = 0
        json_end = -1

        for i in range(last_brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end > 0:
            json_candidate = text[last_brace_start:json_end]
            try:
                # 验证是否为有效JSON
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass

        return None


def remove_copyright_from_text(text: str) -> str:
    """
    清除文本中的copyright信息

    Args:
        text: 原始文本内容

    Returns:
        清除copyright后的文本
    """
    if not text or not text.strip():
        return text

    # 定义常见的copyright匹配模式
    copyright_patterns = [
        # 匹配标准的 # Copyright ... License 格式
        r'(?m)^#\s*Copyright.*?^#\s*limitations\s+under\s+the\s+License\.\s*$',

        # 匹配 // Copyright ... License 格式（C/C++/Java风格）
        r'(?m)^//\s*Copyright.*?^//\s*limitations\s+under\s+the\s+License\.\s*$',

        # 匹配 /* Copyright ... License */ 格式（C风格多行注释）
        r'(?s)/\*.*?Copyright.*?limitations\s+under\s+the\s+License\..*?\*/',

        # 匹配其他常见许可证格式
        r'(?m)^#\s*Copyright.*?^#\s*(?:distributed\s+under\s+|subject\s+to\s+|under\s+the\s+terms\s+of\s+).*?(?:License|MIT|BSD|GPL)\.?\s*$',

        # 匹配简单的 Copyright 单行格式
        r'(?m)^#\s*Copyright\s+\d{4}.*?(?:Inc\.|Ltd\.|Corp\.|Co\.).*?$',

        # 匹配 -*- coding: -*- 和 copyright 组合
        r'(?m)^#.*?-\*-.*?coding.*?-\*-.*?(?:\n#.*?)*?Copyright.*?$',
    ]

    cleaned_text = text

    # 依次应用所有匹配模式
    for pattern in copyright_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    # 清理多余的空行（保留最多2个连续空行）
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)

    # 清理开头的空行
    cleaned_text = cleaned_text.lstrip('\n\r\t ')

    # 如果清理后文本为空或只有空白字符，返回原文本（避免过度清理）
    if not cleaned_text.strip():
        return text

    return cleaned_text

def get_md5_hash(**kwargs) -> str:
    """
    生成参数的MD5哈希值

    Args:
        **kwargs: 任意数量的关键字参数

    Returns:
        str: 16进制格式的MD5哈希字符串

    Example:
        >>> get_md5_hash(a=1, b="test")
        'a7262b12b8a1a379e4e71c879e0d5b2d'
    """
    # 过滤空值并排序
    filtered_params = {k: v for k, v in kwargs.items() if v is not None}
    if not filtered_params:
        raise ValueError("至少需要提供一个有效参数")

    # 标准化参数序列
    sorted_params = sorted(filtered_params.items(), key=lambda x: x[0])
    param_str = '&'.join(f"{k}={v}" for k, v in sorted_params)

    # 生成MD5
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()