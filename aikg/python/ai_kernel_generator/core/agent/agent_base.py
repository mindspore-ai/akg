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
import logging
from abc import ABC
from typing import Dict, Any

from langchain.prompts import PromptTemplate

from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.llm.model_loader import create_model
from ai_kernel_generator.utils.common_utils import get_prompt_path
from ai_kernel_generator.utils.collector import get_collector

logger = logging.getLogger(__name__)
aikg_stream_output = os.getenv("AIKG_STREAM_OUTPUT", "off").lower() == "on"


class AgentBase(ABC):
    """AIKG代理基类，提供基础功能和接口"""

    def __init__(self, context: dict = {}, config: dict = None):
        self.context = context
        self.root_dir = get_project_root()
        self.config = config

    @staticmethod
    def count_tokens(text: str, model_name: str, context: dict = {}) -> None:
        """使用tiktoken准确计算字符串的token数量并打印日志

        Args:
            text: 要统计的文本
            model_name: 模型名称，用于日志打印
            context: 代理详情信息

        """
        agent_name = context.get("agent_name", "Unknown")
        if not text:
            logger.debug(f"LLM Start:  [status] %s -- [model] %s -- [token_count] %s", agent_name, model_name, 0)
            return

        try:
            import tiktoken
        except ImportError:
            return

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            token_count = len(tokens)
            logger.debug(f"LLM Start:  [status] %s -- [model] %s -- [token_count] %s",
                         agent_name, model_name, token_count)

        except Exception as e:
            return

    @staticmethod
    def read_file(file_path: str, encoding: str = "utf-8") -> str:
        """读取文件内容

        Args:
            file_path: 文件路径
            encoding: 文件编码，默认为utf-8

        Returns:
            str: 文件内容
        """
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")
        except UnicodeDecodeError:
            raise UnicodeDecodeError(f"文件编码错误: {file_path}, 编码: {encoding}")
        except Exception as e:
            raise Exception(f"文件读取失败: {file_path}, 错误: {str(e)}")

    def load_template(self, template_path: str, template_format: str = "jinja2") -> PromptTemplate:
        """
        从指定路径加载Jinja2模板

        Args:
            template_path (str): 模板文件的相对路径，相对于prompts目录

        Returns:
            PromptTemplate: 加载的模板对象

        Raises:
            FileNotFoundError: 模板文件不存在时抛出异常
        """
        try:
            prompt_dir = get_prompt_path()
            template_full_path = os.path.join(prompt_dir, template_path)
            template_str = self.read_file(template_full_path)
            prompt_template = PromptTemplate(
                template=template_str,
                template_format=template_format
            )
            return prompt_template
        except Exception as e:
            raise ValueError(f"Failed to load template {template_path}: {e}")

    def load_doc(self, doc_path: str) -> str:
        """
        从resources/docs目录加载资源文档
        支持配置化文档目录

        Args:
            doc_path (str): 文档文件的相对路径

        Returns:
            str: 文档内容

        Raises:
            FileNotFoundError: 文档文件不存在时抛出异常
        """
        try:
            # 解析配置化的文档路径
            resolved_path = self._resolve_configurable_doc_path(doc_path)

            # 拼接完整路径
            full_path = os.path.join(self.root_dir, resolved_path)

            if not os.path.exists(full_path):
                logger.warning(f"Resource doc not found: {full_path}")
                return ""

            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to load resource doc {doc_path}: {e}")
            return ""

    def _resolve_configurable_doc_path(self, doc_path: str) -> str:
        """
        解析配置化的文档路径

        Args:
            doc_path (str): 原始文档路径（文件名或相对路径）

        Returns:
            str: 解析后的文档路径
        """
        try:
            # 检查是否有传入的config
            if not self.config:
                raise ValueError("No config provided. Cannot resolve document path.")

            # 获取agent类型
            agent_type = self._get_agent_type()

            # 尝试从docs_dir配置中获取对应agent的文档目录
            docs_dir_config = self.config.get('docs_dir', {})
            if agent_type in docs_dir_config:
                docs_dir = docs_dir_config[agent_type]
                return os.path.join(docs_dir, doc_path)
            # 如果都没有配置，抛出明确的错误信息
            raise ValueError(f"No doc directory configured for agent type '{agent_type}'. "
                             f"Please add '{agent_type}' to docs_dir in config.")

        except Exception as e:
            raise ValueError(f"Failed to resolve configurable doc path: {e}")

    def _get_agent_type(self) -> str:
        """
        从agent名称中提取agent类型

        Returns:
            str: agent类型 (designer, coder, conductor等)
        """
        class_name = self.__class__.__name__.lower()

        # 直接从类名判断
        if "designer" in class_name:
            return "designer"
        elif "coder" in class_name:
            return "coder"
        elif "conductor" in class_name:
            return "conductor"
        else:
            return class_name.lower()

    def _check_input_dict(self, input: Dict[str, Any]) -> None:
        """
        检查并记录input字典中每个key的值是否为空

        Args:
            input: 要检查的输入字典
        """
        logger.debug("=" * 60)
        logger.debug(f"检查 input 字典内容 (Agent: {self.context.get('agent_name', 'Unknown')})")
        logger.debug("=" * 60)

        if not input:
            logger.debug("❌ input 字典为空!")
            return

        for key, value in input.items():
            # 检查值是否为空
            is_empty = False
            if value is None:
                is_empty = True
            elif isinstance(value, str) and value.strip() == "":
                is_empty = True
            elif isinstance(value, (list, dict)) and len(value) == 0:
                is_empty = True

            # 记录状态
            status = "❌ 空" if is_empty else "✅ 有值"
            value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            logger.debug(f"{status} {key}: {value_preview}")

        logger.debug("=" * 60)

    async def run_llm(self, prompt: PromptTemplate, input: Dict[str, Any], model_name: str) -> tuple[str, str, str]:
        """运行LLM

        Args:
            prompt: 提示模板
            input: 输入
            model_name: 模型名称

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        formatted_prompt = prompt.format(**input)
        self._check_input_dict(input)
        # self.count_tokens(formatted_prompt, model_name, self.context) # 暂不开启token统计
        # 创建模型
        model = create_model(model_name)

        try:
            # 如果是VLLM模型（openai.AsyncOpenAI客户端）
            if model_name.startswith("vllm_"):
                # 将formatted_prompt转换为OpenAI格式的消息
                messages = [
                    {"role": "system", "content": ""},  # 空的system prompt
                    {"role": "user", "content": formatted_prompt}
                ]

                # 直接调用OpenAI API
                response = await model.chat.completions.create(
                    model=model.model_name,
                    messages=messages,
                    temperature=model.temperature,
                    top_p=model.top_p,
                    stream=False
                )

                content = response.choices[0].message.content
                reasoning_content = response.choices[0].message.reasoning_content

                response_metadata = f"completion_tokens: {response.usage.completion_tokens}, " + \
                    f"prompt_tokens: {response.usage.prompt_tokens}, total_tokens: {response.usage.total_tokens}"
                logger.info(f"response_metadata: {response_metadata}")

            else:
                # 其他模型使用原来的chain方式
                chain = prompt | model

                if not aikg_stream_output:
                    raw_result = await chain.ainvoke(input)
                    content = raw_result.content
                    reasoning_content = raw_result.additional_kwargs.get("reasoning_content", "")
                else:
                    content = ""
                    reasoning_content = ""
                    async for raw_result in chain.astream(input):
                        if raw_result.content != "":
                            print(raw_result.content, end='', flush=True)
                            content += raw_result.content
                        elif "reasoning_content" in raw_result.additional_kwargs:
                            print(raw_result.additional_kwargs.get("reasoning_content"), end='', flush=True)
                            reasoning_content += raw_result.additional_kwargs.get("reasoning_content")
                    print()

                response_metadata = f"response_metadata: {raw_result.response_metadata}\n" + \
                    f"usage_metadata: {raw_result.usage_metadata}"
                logger.info(response_metadata)

            logger.debug(f"LLM End:    [status] %s -- [model] %s",
                         self.context.get('agent_name', 'Unknown'), model_name)

            if os.getenv("AIKG_DATA_COLLECT", "off").lower() == "on":
                # 使用collector收集数据
                try:
                    collector = await get_collector()
                    collected_data = {
                        "hash": self.context.get('hash', 'Unknown'),
                        "agent_name": self.context.get('agent_name', 'Unknown'),
                        "model_name": model_name,
                        "content": content,
                        "formatted_prompt": formatted_prompt,
                        "reasoning_content": reasoning_content,
                        "response_metadata": response_metadata,
                    }
                    await collector.collect(collected_data)
                except Exception as e:
                    logger.warning(f"Failed to collect data: {e}")

            return content, formatted_prompt, reasoning_content
        except Exception as e:
            logger.error(f"LLM Failed: [status] %s -- [model] %s -- [error] %s",
                         self.context.get('agent_name', 'Unknown'), model_name, e)
            logger.error(f"Exception in run_llm: {type(e).__name__}: {e}")
            raise
