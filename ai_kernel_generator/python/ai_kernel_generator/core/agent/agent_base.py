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
from pathlib import Path

from langchain.prompts import PromptTemplate

from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.llm.model_loader import create_model
from ai_kernel_generator.utils.common_utils import get_prompt_path

logger = logging.getLogger(__name__)
stream_output_mode = os.getenv("STREAM_OUTPUT_MODE", "off").lower() == "on"


class AgentBase(ABC):
    """AIKG代理基类，提供基础功能和接口"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.root_dir = get_project_root()

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

    def load_template(self, file_path: str, template_format: str = "jinja2") -> PromptTemplate:
        """加载单个提示模板

        Args:
            file_path: 模板文件路径，基于项目根目录/resources/prompts/generation/
            template_format: 模板格式，默认为jinja2

        Returns:
            PromptTemplate: 加载的提示模板
        """
        template_path = Path(get_prompt_path()) / "generation" / file_path
        template_str = self.read_file(template_path)
        
        prompt_template = PromptTemplate(
            template=template_str,
            template_format=template_format
        )
        return prompt_template

    def load_doc(self, file_path: str) -> str:
        """加载提示文档

        Args:
            file_path: 提示文档路径，基于项目根目录/resources/docs/

        Returns:
            str: 加载的提示文档内容
        """
        doc_path = Path(self.root_dir) / "resources" / "docs" / file_path
        return self.read_file(doc_path)

    async def run_llm(self, prompt: PromptTemplate, input: Dict[str, Any], model_name: str) -> tuple[str, str, str]:
        """运行LLM

        Args:
            prompt: 提示模板
            input: 输入
            model_name: 模型名称

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        logger.debug(f"LLM Start:  [status] %s -- [model] %s", self.agent_name, model_name)

        formatted_prompt = prompt.format(**input)

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
                
            else:
                # 其他模型使用原来的chain方式
                chain = prompt | model
                
                if not stream_output_mode:
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

            logger.debug(f"LLM End:    [status] %s -- [model] %s", self.agent_name, model_name)

            return content, formatted_prompt, reasoning_content
        except Exception as e:
            logger.error(f"LLM Failed: [status] %s -- [model] %s -- [error] %s", self.agent_name, model_name, e)
            return "", "", ""
