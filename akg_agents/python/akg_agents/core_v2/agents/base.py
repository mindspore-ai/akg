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
AgentBase - Agent 基类

提供 Agent 通用功能，保持与现有代码兼容：
- run_llm(): 调用 LLM
- load_template(): 加载 Jinja2 模板
- load_doc(): 加载资源文档
- split_think(): 分离 </think> 内容
- _stream_enabled(): 检查流式输出
"""

import os
import logging
from abc import ABC
from typing import Dict, Any, Tuple, Optional

# Jinja2 模板支持
from jinja2 import Environment, BaseLoader

from akg_agents.core_v2.config import get_settings
from akg_agents.core_v2.config.settings import get_akg_env_var
from akg_agents.core_v2.llm.factory import create_llm_client

logger = logging.getLogger(__name__)


class LLMAPIError(Exception):
    """LLM API 调用失败的自定义异常"""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error
        self.user_message = message
    
    def __str__(self):
        """返回友好的错误消息"""
        return self.user_message


class Jinja2TemplateWrapper:
    """
    原生 Jinja2 模板包装器，兼容 LangChain PromptTemplate 接口
    
    LangChain 1.0 的 PromptTemplate 使用 SandboxedEnvironment，
    限制了对 loop.index 等属性的访问。此包装器使用原生 Jinja2，
    支持完整的 Jinja2 功能。
    """
    
    def __init__(self, template_str: str):
        """
        初始化 Jinja2 模板包装器
        
        Args:
            template_str: 模板字符串
        """
        self._template_str = template_str
        self._env = Environment(loader=BaseLoader())
        self._template = self._env.from_string(template_str)
    
    def format(self, **kwargs) -> str:
        """
        渲染模板（兼容 PromptTemplate.format 接口）
        
        Args:
            **kwargs: 模板变量
            
        Returns:
            渲染后的字符串
        """
        return self._template.render(**kwargs)


class AgentBase(ABC):
    """
    AIKG Agent 基类
    
    提供 Agent 通用功能：
    - run_llm(): 调用 LLM（保持原签名兼容）
    - load_template(): 加载 Jinja2 模板
    - load_doc(): 加载资源文档
    - _stream_enabled(): 检查流式输出
    - split_think(): 分离 </think> 内容
    - load_tool_config(): 加载 agent 工具配置为字典格式
    
    子类需要定义以下类属性以支持 load_tool_config():
    - TOOL_NAME: 工具名称（如 "call_op_task_builder"）
    - DESCRIPTION: 功能描述
    - PARAMETERS_SCHEMA: 参数 schema（JSON Schema 格式的字典）
    """
    
    # 子类需要覆盖的元数据属性
    TOOL_NAME: Optional[str] = None
    DESCRIPTION: Optional[str] = None
    PARAMETERS_SCHEMA: Optional[Dict[str, Any]] = None
    
    def __init__(self, context: dict = None, config: dict = None):
        """
        初始化 Agent 基类
        
        Args:
            context: Agent 上下文（包含 agent_name, session_id, op_name 等）
            config: 配置信息（包含 agent_model_config, docs_dir 等）
        """
        self.context = context or {}
        self.config = config or {}
        
        # 获取项目根目录
        try:
            from akg_agents import get_project_root
            self.root_dir = get_project_root()
        except ImportError:
            self.root_dir = os.getcwd()
    
    # ========================= LLM 调用 =========================
    
    async def run_llm(
        self,
        prompt,
        input: Dict[str, Any],
        model_level: str
    ) -> Tuple[str, str, str]:
        """
        调用 LLM，返回 (content, formatted_prompt, reasoning_content)
        
        保持与原 AgentBase.run_llm() 签名兼容。
        
        Args:
            prompt: Jinja2 模板（Jinja2TemplateWrapper 或 PromptTemplate）
            input: 模板变量字典
            model_level: settings.json 中 models 的级别/名称
                - 预定义级别："complex" / "standard" / "fast"
                - 如果配置中没有该级别，使用 default_model
        
        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        
        Raises:
            LLMAPIError: LLM API 调用失败
            ValueError: 启用流式输出但未提供 session_id
        """
        try:
            # 格式化 prompt
            formatted_prompt = prompt.format(**input)
            self._check_input_dict(input)
            
            # 提取上下文信息
            agent_name = str(self.context.get("agent_name") or "unknown")
            session_id = str(self.context.get("session_id") or "").strip()
            
            # 检查流式输出和 session_id
            stream = self._stream_enabled()
            if stream and not session_id:
                raise ValueError(f"Agent {agent_name} 启用流式输出但未提供 session_id")
            
            # 创建 LLMClient
            client = create_llm_client(model_level=model_level, session_id=session_id)
            
            # 调用 LLM
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": formatted_prompt}
            ]
            result = await client.generate(
                messages,
                stream=stream,
                agent_name=agent_name,
            )
            
            content = result.get("content", "")
            reasoning_content = result.get("reasoning_content", "")
            
            # 分离 </think> 内容
            if "</think>" in content:
                content, extracted = self.split_think(content)
                if extracted:
                    reasoning_content = extracted
            
            # 如果 content 为空，使用 reasoning_content
            if not content:
                content = reasoning_content
            
            return content, formatted_prompt, reasoning_content
            
        except ValueError as e:
            # ValueError 直接抛出（如 session_id 检查失败）
            raise
        except Exception as e:
            # 捕获所有其他异常，转换为 LLMAPIError
            agent_name = str(self.context.get("agent_name") or "unknown")
            error_type = type(e).__name__
            error_message = str(e)
            
            logger.error(f"[AgentBase] LLM API 调用失败: {error_type}: {error_message}")
            
            # 构建用户友好的错误消息
            user_message = (
                f"❌ 模型 API 调用失败\n\n"
                f"错误类型: {error_type}\n"
                f"错误信息: {error_message}\n\n"
                f"请检查模型配置文件：\n"
                f"1. `.akg/settings.json` (项目级配置)\n"
                f"2. `~/.akg/settings.json` (用户级配置)\n"
                f"3. 或使用环境变量覆盖 (AKG_AGENTS_* 或 AIKG_*)\n\n"
                f"详细配置说明请参考: docs/API.md\n\n"
                f"使用的模型级别: {model_level}\n"
                f"Agent: {agent_name}"
            )
            
            raise LLMAPIError(user_message, original_error=e)
    
    # ========================= 流式输出控制 =========================
    
    @staticmethod
    def _stream_enabled() -> bool:
        """
        检查是否启用流式输出
        
        优先级（从高到低）：
        1. ContextVar 覆盖（stream_output_override）
        2. 环境变量 AKG_AGENTS_STREAM_OUTPUT 或 AIKG_STREAM_OUTPUT
        3. settings.json 中的 stream_output
        4. 默认值 False
        
        Returns:
            bool: 是否启用流式输出
        """
        # 1. ContextVar 覆盖（最高优先级）
        try:
            from akg_agents.utils.stream_output import get_stream_output_override
            override = get_stream_output_override()
            if override is not None:
                return bool(override)
        except ImportError:
            pass
        except Exception:
            pass
        
        # 2. 环境变量（支持 AKG_AGENTS_* 和 AIKG_*）
        env_value = get_akg_env_var("STREAM_OUTPUT")
        if env_value is not None:
            return env_value.lower() == "on"
        
        # 3. settings.json
        try:
            settings = get_settings()
            if settings.stream_output is not None:
                return settings.stream_output
        except Exception:
            pass
        
        # 4. 默认关闭
        return False
    
    # ========================= 模板和文档加载 =========================
    
    def load_template(self, template_path: str, template_format: str = "jinja2"):
        """
        从指定路径加载 Jinja2 模板
        
        Args:
            template_path: 模板文件的相对路径（相对于 prompts 目录）
            template_format: 模板格式，默认为 "jinja2"
        
        Returns:
            Jinja2TemplateWrapper 或 PromptTemplate
        
        Raises:
            ValueError: 模板加载失败
        """
        try:
            from akg_agents.utils.common_utils import get_prompt_path
            prompt_dir = get_prompt_path()
            template_full_path = os.path.join(prompt_dir, template_path)
            template_str = self.read_file(template_full_path)
            
            if template_format == "jinja2":
                return Jinja2TemplateWrapper(template_str)
            else:
                # 其他格式使用 LangChain 的 PromptTemplate
                try:
                    from langchain_core.prompts import PromptTemplate
                except ImportError:
                    from langchain.prompts import PromptTemplate
                return PromptTemplate(template=template_str, template_format=template_format)
        except Exception as e:
            raise ValueError(f"Failed to load template {template_path}: {e}")
    
    def load_doc(self, doc_path: str) -> str:
        """
        从 resources/docs 目录加载资源文档
        
        支持配置化文档目录
        
        Args:
            doc_path: 文档文件的相对路径
        
        Returns:
            str: 文档内容
        """
        try:
            resolved_path = self._resolve_configurable_doc_path(doc_path)
            full_path = os.path.join(self.root_dir, resolved_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"Resource doc not found: {full_path}")
                return ""
            
            logger.info(f"Loading resource doc: {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to load resource doc {doc_path}: {e}")
            return ""
    
    def _resolve_configurable_doc_path(self, doc_path: str) -> str:
        """
        解析配置化的文档路径
        
        Args:
            doc_path: 原始文档路径
        
        Returns:
            str: 解析后的文档路径
        """
        if not self.config:
            raise ValueError("No config provided. Cannot resolve document path.")
        
        agent_type = self.__class__.__name__.lower()
        docs_dir_config = self.config.get('docs_dir', {})
        
        if agent_type in docs_dir_config:
            docs_dir = docs_dir_config[agent_type]
            return os.path.join(docs_dir, doc_path)
        
        raise ValueError(f"No doc directory configured for agent type '{agent_type}'.")
    
    # ========================= 工具方法 =========================
    
    @staticmethod
    def read_file(file_path: str, encoding: str = "utf-8") -> str:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            encoding: 文件编码，默认为 utf-8
        
        Returns:
            str: 文件内容
        """
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")
        except Exception as e:
            raise Exception(f"文件读取失败: {file_path}, 错误: {str(e)}")
    
    @staticmethod
    def split_think(content: str) -> Tuple[str, str]:
        """
        按首次出现的 '</think>' 对文本进行拆分
        
        Args:
            content: 待处理的原始文本
        
        Returns:
            tuple: (new_content, reasoning_content)
        """
        if content is None:
            return "", ""
        
        marker = "</think>"
        pos = content.find(marker)
        if pos == -1:
            return content, ""
        
        reasoning_content = content[:pos]
        new_content = content[pos + len(marker):].lstrip("\r\n ")
        return new_content, reasoning_content
    
    @staticmethod
    def count_tokens(text: str, model_name: str = "", context: dict = None) -> int:
        """
        计算文本的 token 数量
        
        Args:
            text: 要统计的文本
            model_name: 模型名称（用于日志）
            context: 上下文信息
        
        Returns:
            int: token 数量
        """
        if not text:
            return 0
        
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            token_count = len(tokens)
            
            agent_name = (context or {}).get("agent_name", "")
            logger.debug(f"LLM Start: [agent] {agent_name} -- [model] {model_name} -- [tokens] {token_count}")
            
            return token_count
        except ImportError:
            # tiktoken 未安装，简单估算
            return len(text) // 4
        except Exception:
            return len(text) // 4
    
    def _check_input_dict(self, input: Dict[str, Any]) -> None:
        """
        检查并记录 input 字典中每个 key 的值是否为空
        
        Args:
            input: 要检查的输入字典
        """
        logger.debug("=" * 60)
        logger.debug(f"检查 input 字典内容 (Agent: {self.context.get('agent_name', '')})")
        logger.debug("=" * 60)
        
        if not input:
            logger.debug("input 字典为空!")
            return
        
        for key, value in input.items():
            is_empty = False
            if value is None:
                is_empty = True
            elif isinstance(value, str) and value.strip() == "":
                is_empty = True
            elif isinstance(value, (list, dict)) and len(value) == 0:
                is_empty = True
            
            status = "空" if is_empty else "有值"
            value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            logger.debug(f"{status} {key}: {value_preview}")
        
        logger.debug("=" * 60)
    
    # ========================= 工具配置加载 =========================
    
    @classmethod
    def load_tool_config(cls) -> Dict[str, Any]:
        """
        加载 Agent 的工具配置为字典格式
        
        子类必须定义以下类属性：
        - TOOL_NAME: 工具名称
        - DESCRIPTION: 功能描述
        - PARAMETERS_SCHEMA: 参数 schema（JSON Schema 格式）
        
        agent_name 会自动从类名获取
        
        Returns:
            dict: 工具配置字典
        
        Raises:
            ValueError: 如果缺少必需的类属性
        """
        # 验证必需字段
        cls._validate_tool_metadata()
        
        # 构建工具配置字典
        tool_name = cls.TOOL_NAME or "unknown"
        config = {
            tool_name: {
                "type": "call_agent",
                "agent_name": cls.__name__,
                "function": {
                    "name": cls.TOOL_NAME,
                    "description": cls._format_description(cls.DESCRIPTION),
                    "parameters": cls.PARAMETERS_SCHEMA or {}
                }
            }
        }
        
        return config

    @classmethod
    def _validate_tool_metadata(cls) -> None:
        """
        验证子类是否提供了必需的元数据
        
        Raises:
            ValueError: 如果缺少必需的类属性
        """
        missing_fields = []
        
        if not cls.TOOL_NAME:
            missing_fields.append("TOOL_NAME")
        if not cls.DESCRIPTION:
            missing_fields.append("DESCRIPTION")
        if cls.PARAMETERS_SCHEMA is None:
            missing_fields.append("PARAMETERS_SCHEMA")
        
        if missing_fields:
            class_name = cls.__name__
            fields_str = ", ".join(missing_fields)
            raise ValueError(
                f"{class_name} 必须定义以下类属性才能使用 load_tool_config(): {fields_str}\n"
                f"示例：\n"
                f"class {class_name}(AgentBase):\n"
                f"    TOOL_NAME = 'call_your_agent'\n"
                f"    DESCRIPTION = 'Agent description'\n"
                f"    PARAMETERS_SCHEMA = {{\n"
                f"        'type': 'object',\n"
                f"        'properties': {{}},\n"
                f"        'required': []\n"
                f"    }}"
            )
    
    @staticmethod
    def _format_description(description: str) -> str:
        """
        格式化描述文本，处理多行字符串
        
        Args:
            description: 原始描述文本
        
        Returns:
            str: 格式化后的描述
        """
        if not description:
            return ""
        
        # 去除首尾空白，但保留内部换行
        lines = description.strip().split('\n')
        formatted_lines = [line.rstrip() for line in lines]
        return '\n'.join(formatted_lines)
