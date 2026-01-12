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
import time
from abc import ABC
from typing import Dict, Any, Optional

try:
    from openai import AsyncOpenAI as OpenAIAsyncClient
except ImportError:
    OpenAIAsyncClient = None

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # Fallback for older langchain versions
    from langchain.prompts import PromptTemplate

# 使用原生 Jinja2
# LangChain 1.0 的 PromptTemplate 使用 SandboxedEnvironment，限制了属性访问
from jinja2 import Environment, BaseLoader

from textual import log as textual_log

from ai_kernel_generator import get_project_root
from ai_kernel_generator.cli.messages import LLMEndMessage, LLMStartMessage, LLMStreamMessage
from ai_kernel_generator.cli.server.message_sender import send_message
from ai_kernel_generator.utils.task_label import resolve_task_label


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
    
    def __or__(self, other):
        """
        支持 prompt | model 链式调用（LangChain 风格）
        
        Args:
            other: 链中的下一个组件（通常是 LLM）
            
        Returns:
            RunnableSequence 或类似对象
        """
        # 创建一个可运行的序列
        from langchain_core.runnables import RunnableLambda
        
        def render_template(inputs: dict) -> str:
            return self.format(**inputs)
        
        runnable = RunnableLambda(render_template)
        return runnable | other


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
        agent_name = context.get("agent_name", "")
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

    @staticmethod
    def split_think(content: str) -> tuple[str, str]:
        """按首次出现的 '</think>' 对文本进行拆分。

        Args:
            content: 待处理的原始文本

        Returns:
            tuple[str, str]: (new_content, reasoning_content)。若未包含该标记，返回 (content, "")。
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

    def load_template(self, template_path: str, template_format: str = "jinja2") -> PromptTemplate:
        """
        从指定路径加载Jinja2模板

        Args:
            template_path (str): 模板文件的相对路径，相对于prompts目录
            template_format (str): 模板格式，默认为 "jinja2"

        Returns:
            PromptTemplate: 加载的模板对象（对于 jinja2 格式返回原生 Jinja2Template 包装器）

        Raises:
            FileNotFoundError: 模板文件不存在时抛出异常
        """
        try:
            prompt_dir = get_prompt_path()
            template_full_path = os.path.join(prompt_dir, template_path)
            template_str = self.read_file(template_full_path)
            
            if template_format == "jinja2":
                # 使用原生 Jinja2（支持完整功能，包括 loop.index）
                # LangChain 1.0 的 PromptTemplate 使用 SandboxedEnvironment，限制了属性访问
                return Jinja2TemplateWrapper(template_str)
            else:
                # 其他格式使用 LangChain 的 PromptTemplate
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
        logger.debug(f"检查 input 字典内容 (Agent: {self.context.get('agent_name', '')})")
        logger.debug("=" * 60)

        if not input:
            logger.debug("input 字典为空!")
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
            status = "空" if is_empty else "有值"
            value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            logger.debug(f"{status} {key}: {value_preview}")

        logger.debug("=" * 60)

    @staticmethod
    def _stream_enabled() -> bool:
        """检查是否启用流式输出"""
        return os.getenv("AIKG_STREAM_OUTPUT", "off").lower() == "on"

    @staticmethod
    def _extract_task_id(ctx: Dict[str, Any]) -> str:
        """从上下文中提取 task_id"""
        task_id = ctx.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            return task_id.strip()
        return ""

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """安全转换为整数"""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _safe_send(self, session_id: str, message) -> None:
        """安全发送消息，失败不影响主流程"""
        if not session_id:
            return
        try:
            send_message(session_id, message)
        except Exception as e:
            textual_log.warning("[Agent] send_message failed; ignored", exc_info=e)

    async def run_llm(self, prompt: PromptTemplate, input: Dict[str, Any], model_name: str) -> tuple[str, str, str]:
        """运行LLM（整合了消息发送和 token 统计功能）

        Args:
            prompt: 提示模板
            input: 输入
            model_name: 模型名称

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        start_time = time.time()

        # 格式化 prompt
        formatted_prompt = prompt.format(**input)
        self._check_input_dict(input)

        # 提取上下文信息
        context = self.context if isinstance(self.context, dict) else {}
        agent_name = str(context.get("agent_name") or "unknown")
        session_id = str(context.get("session_id") or "").strip()
        task_id = self._extract_task_id(context)
        task_label = str(context.get("task_label") or "").strip()
        if not task_label and isinstance(self.config, dict):
            task_label = str(self.config.get("task_label") or "").strip()
        if not task_label:
            task_label = resolve_task_label(
                op_name=str(context.get("op_name") or ""),
                parallel_index=1,
            )

        # 检查流式输出和 session_id
        stream = self._stream_enabled()
        if stream and not session_id:
            raise ValueError(f"[Agent] agent={agent_name} 的 context 中必须包含 session_id（AIKG_STREAM_OUTPUT=on）")

        # 创建模型
        model = create_model(model_name)
        effective_model_name = getattr(model, "model_name", model_name)

        # 发送开始消息
        self._safe_send(
            session_id,
            LLMStartMessage(
                agent=agent_name,
                model=effective_model_name,
                task_id=task_id,
                task_label=task_label,
            )
        )

        content = ""
        reasoning_content = ""
        usage_metadata = None

        try:
            # 检查是否是 OpenAI AsyncClient
            is_openai_async = False
            try:
                from openai import AsyncOpenAI as OpenAIAsyncClient
                is_openai_async = isinstance(model, OpenAIAsyncClient)
            except ImportError:
                is_openai_async = False

            # VLLM 或 OpenAI AsyncClient 模型
            if effective_model_name.startswith("vllm_") or is_openai_async:
                messages = [{"role": "system", "content": ""}, {"role": "user", "content": formatted_prompt}]
                create_kwargs = {
                    "model": getattr(model, "model_name", model_name),
                    "messages": messages,
                    "temperature": getattr(model, "temperature", 0.0),
                    "top_p": getattr(model, "top_p", 1.0),
                    "stream": bool(stream),
                }
                extra_body = getattr(model, "extra_body", None)
                if extra_body:
                    create_kwargs["extra_body"] = extra_body

                if not stream:
                    response = await model.chat.completions.create(**create_kwargs)
                    content = response.choices[0].message.content
                    reasoning_content = getattr(response.choices[0].message, "reasoning_content", "") or ""
                else:
                    stream_iter = await model.chat.completions.create(**create_kwargs)
                    async for chunk in stream_iter:
                        delta = chunk.choices[0].delta
                        delta_text = getattr(delta, "content", None)
                        if isinstance(delta_text, str) and delta_text:
                            print(delta_text, end="", flush=True)
                            content += delta_text
                            self._safe_send(
                                session_id,
                                LLMStreamMessage(
                                    agent=agent_name,
                                    chunk=delta_text,
                                    task_id=task_id,
                                    is_reasoning=False,
                                    task_label=task_label,
                                ),
                            )
                            continue
                        delta_reasoning = getattr(delta, "reasoning_content", None)
                        if isinstance(delta_reasoning, str) and delta_reasoning:
                            print(delta_reasoning, end="", flush=True)
                            reasoning_content += delta_reasoning
                            self._safe_send(
                                session_id,
                                LLMStreamMessage(
                                    agent=agent_name,
                                    chunk=delta_reasoning,
                                    task_id=task_id,
                                    is_reasoning=True,
                                    task_label=task_label,
                                ),
                            )
                    print()
            else:
                # 其他模型使用 LangChain chain
                chain = prompt | model
                if not stream:
                    raw_result = await chain.ainvoke(input)
                    content = raw_result.content
                    reasoning_content = raw_result.additional_kwargs.get("reasoning_content", "") or ""
                    usage_metadata = getattr(raw_result, "usage_metadata", None)
                else:
                    async for raw_result in chain.astream(input):
                        chunk_text = getattr(raw_result, "content", "")
                        if chunk_text:
                            print(chunk_text, end="", flush=True)
                            content += chunk_text
                            self._safe_send(
                                session_id,
                                LLMStreamMessage(
                                    agent=agent_name,
                                    chunk=chunk_text,
                                    task_id=task_id,
                                    is_reasoning=False,
                                    task_label=task_label,
                                ),
                            )
                            continue
                        additional_kwargs = getattr(raw_result, "additional_kwargs", {}) or {}
                        if isinstance(additional_kwargs, dict) and "reasoning_content" in additional_kwargs:
                            chunk_reasoning = additional_kwargs.get("reasoning_content") or ""
                            if chunk_reasoning:
                                print(chunk_reasoning, end="", flush=True)
                                reasoning_content += chunk_reasoning
                                self._safe_send(
                                    session_id,
                                    LLMStreamMessage(
                                        agent=agent_name,
                                        chunk=chunk_reasoning,
                                        task_id=task_id,
                                        is_reasoning=True,
                                        task_label=task_label,
                                    ),
                                )
                    print()
                    usage_metadata = getattr(raw_result, "usage_metadata", None)

            # 后处理：从 content 中剥离可能包含的 reasoning 片段
            if hasattr(self, "split_think"):
                content, extracted = self.split_think(content)
                if extracted:
                    reasoning_content = extracted
            if not content:
                content = reasoning_content

            # 提取 token 统计信息
            prompt_tokens = None
            output_tokens = None
            reasoning_tokens = None
            total_tokens = None

            usage_md = usage_metadata if isinstance(usage_metadata, dict) else None
            if usage_md:
                prompt_tokens = self._safe_int(usage_md.get("input_tokens"))
                completion_tokens = self._safe_int(usage_md.get("output_tokens"))
                total_tokens = self._safe_int(usage_md.get("total_tokens"))
                out_details = usage_md.get("output_token_details") or {}
                if isinstance(out_details, dict):
                    reasoning_tokens = self._safe_int(out_details.get("reasoning"))
                    if reasoning_tokens is None:
                        reasoning_tokens = self._safe_int(out_details.get("reasoning_tokens"))

                if completion_tokens is None:
                    output_tokens = None
                elif reasoning_tokens is None:
                    output_tokens = completion_tokens
                else:
                    output_tokens = max(completion_tokens - reasoning_tokens, 0)

            # 发送结束消息
            self._safe_send(
                session_id,
                LLMEndMessage(
                    agent=agent_name,
                    model=effective_model_name,
                    response=content,
                    duration=time.time() - start_time,
                    task_id=task_id,
                    task_label=task_label,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    total_tokens=total_tokens,
                )
            )

            # 数据收集
            if os.getenv("AIKG_DATA_COLLECT", "off").lower() == "on":
                try:
                    collector = await get_collector()
                    collected_data = {
                        "hash": context.get("hash", ""),
                        "agent_name": context.get("agent_name", ""),
                        "op_name": context.get("op_name", ""),
                        "dsl": context.get("dsl", ""),
                        "backend": context.get("backend", ""),
                        "arch": context.get("arch", ""),
                        "framework": context.get("framework", ""),
                        "workflow_name": context.get("workflow_name", ""),
                        "task_desc": context.get("task_desc", ""),
                        "model_name": effective_model_name,
                        "content": content,
                        "formatted_prompt": formatted_prompt,
                        "reasoning_content": reasoning_content,
                        "response_metadata": "",
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "total_tokens": total_tokens,
                    }
                    await collector.collect(collected_data)
                except Exception as e:
                    textual_log.warning("[Agent] data collect failed; ignored", exc_info=e)

            return content, formatted_prompt, reasoning_content
        except Exception:
            # 发送失败消息
            self._safe_send(
                session_id,
                LLMEndMessage(
                    agent=agent_name,
                    model=effective_model_name,
                    response=content or "",
                    duration=time.time() - start_time,
                    task_id=task_id,
                    task_label=task_label,
                )
            )
            raise
