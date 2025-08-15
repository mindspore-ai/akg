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

import logging
from typing import Tuple, List
from pathlib import Path

from ai_kernel_generator.utils.common_utils import ParserFactory, remove_copyright_from_text
from ai_kernel_generator.utils.parser_registry import create_step_parser
from ai_kernel_generator.utils.hardware_utils import get_hardware_doc
from ai_kernel_generator.utils.swft_docs_loader import get_swft_docs_content
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)


def get_inspirations(inspirations: List[dict]) -> str:
    """
    将inspirations列表转换为字符串

    Args:
        inspirations: 包含字典的列表，每个字典格式为:
                     {'strategy_mode':xxx, 'impl_code':str, 'profile':float}

    Returns:
        str: 拼接后的字符串，包含所有impl_code和profile信息
    """
    if not inspirations:
        return ""

    result_parts = []

    for i, inspiration in enumerate(inspirations):
        if not isinstance(inspiration, dict):
            logger.warning(f"跳过非字典类型的inspiration: {type(inspiration)}")
            continue

        impl_code = inspiration.get('impl_code', '')
        profile = inspiration.get('profile', float('inf'))

        if impl_code:  # 只有当impl_code不为空时才添加
            inspiration_text = f"## Inspiration {i+1} 代码执行耗时(s): {profile}\n"
            inspiration_text += f"代码：\n```\n{impl_code}\n```\n"
            result_parts.append(inspiration_text)

    return "\n".join(result_parts)


class Coder(AgentBase):
    def __init__(self, op_name: str, task_desc: str, dsl: str, framework: str, backend: str, arch: str = "", workflow_config_path: str = None, config: dict = None):
        self.op_name = op_name
        self.task_desc = remove_copyright_from_text(task_desc)
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.workflow_config_path = workflow_config_path
        self.config = config
        self.codegen_step_count = 0
        self.api_step_count = 0

        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Coder")

        context = {
            "agent_name": "coder",
            "dsl": dsl,
            "op_name": op_name,
            "framework": framework,
            "backend": backend,
            "arch": arch,
            "task_desc": task_desc,
        }
        super().__init__(context=context, config=config)

        # 直接使用从workflow.yaml获取的coder解析器
        self.code_parser = create_step_parser("coder", self.workflow_config_path)
        if not self.code_parser:
            raise ValueError(
                "Failed to create coder parser from workflow config. Please check your workflow.yaml configuration.")
        self.format_instructions = self.code_parser.get_format_instructions()

        if "triton" in self.dsl:
            self.func_name = f"{self.op_name}_triton_{self.framework}"
        else:
            self.func_name = f"{self.op_name}_{self.dsl}_{self.framework}"

        # 初始化coder生成模板
        self.coder_prompt = self.load_template("coder/codegen.j2")
        self.api_docs_prompt = self.load_template("utils/api_gen_template.j2")
        self.dsl_examples_prompt = self.load_template("utils/dsl_examples_template.j2")

        # 准备基础文档数据
        self.base_doc = {
            "op_name": self.op_name,
            "task_desc": self.task_desc,
            "framework": self.framework,
            "dsl": self.dsl,
            "func_name": self.func_name,
            "format_instructions": self.format_instructions,

            "api_docs": self.load_doc("api/api.md"),
            "dsl_basic_docs": self.load_doc("basic_docs.md"),
            "dsl_examples": self._load_dsl_examples(),
            "expert_suggestion": self.load_doc("suggestion_docs.md"),

            # 可选参数
            "hardware_docs": get_hardware_doc(self.backend, self.arch),
            "arch_name": self.arch,
            "database_examples": "",
        }

    def _load_dsl_examples(self) -> str:
        """
        根据framework加载对应的DSL示例代码

        Returns:
            str: 示例代码内容，如果找不到对应示例则返回空字符串
        """
        if not self.framework:
            logger.warning("framework为空，无法加载示例代码")
            return ""

        # 使用配置化的文档路径
        try:
            # 从config中获取coder的docs_dir
            if not self.config:
                raise ValueError("No config provided. Cannot resolve document path.")

            docs_dir_config = self.config.get('docs_dir', {})
            if 'coder' not in docs_dir_config:
                raise ValueError("No doc directory configured for coder agent.")

            coder_docs_dir = docs_dir_config['coder']
            base_dir = Path(get_project_root()) / coder_docs_dir / "examples"

        except Exception as e:
            logger.warning(f"Failed to resolve configurable doc path: {e}, using fallback path")
            # 降级到硬编码路径
            base_dir = Path(get_project_root()) / "resources" / "docs" / "triton_docs" / "examples"

        if not base_dir.exists():
            logger.warning(f"Triton示例目录不存在: {base_dir}, 返回空字符串")
            return ""

        all_code = []
        # 支持多种文件格式：py, md, txt等
        supported_extensions = ['*.py', '*.md', '*.txt']

        for extension in supported_extensions:
            # 使用glob模式匹配framework开头的文件
            for file_path in base_dir.glob(f"{self.framework}_{extension}"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            # 根据文件类型添加不同的注释标识
                            if file_path.suffix == '.py':
                                all_code.append(f"# Python File: {file_path.name}\n{content}\n")
                            elif file_path.suffix == '.md':
                                all_code.append(f"# Markdown File: {file_path.name}\n{content}\n")
                            elif file_path.suffix == '.txt':
                                all_code.append(f"# Text File: {file_path.name}\n{content}\n")
                            else:
                                all_code.append(f"# File: {file_path.name}\n{content}\n")
                except Exception as e:
                    logger.warning(f"读取示例文件 {file_path} 时发生错误: {str(e)}")
                    continue

        if not all_code:
            logger.warning(f"未找到{self.framework}相关的示例文件")
            return ""

        return "\n".join(all_code)

    def load_doc(self, doc_path: str) -> str:
        """
        重写load_doc方法，特殊处理swft后端
        逻辑：
        1. 常规后端：使用标准的文档读取方式
        2. swft后端：
           - 如果指定的文档路径存在，就读取该文档
           - 如果指定的文档路径不存在，则默认使用swft的本地文档

        Args:
            doc_path (str): 文档文件的相对路径

        Returns:
            str: 文档内容
        """
        # 检查是否是swft后端
        if self.dsl.lower() == "swft":
            logger.info(f"检测到swft后端，尝试读取文档: {doc_path}")

            try:
                # 首先尝试使用父类的标准方法读取指定文档
                standard_content = super().load_doc(doc_path)
                if standard_content:
                    logger.info(f"成功读取到swft的指定文档: {doc_path}")
                    return standard_content
                else:
                    logger.info(f"swft指定文档为空，使用默认的swft本地文档")
                    return get_swft_docs_content()

            except Exception as e:
                logger.warning(f"读取swft指定文档失败: {str(e)}，使用默认的swft本地文档")
                return get_swft_docs_content()
        else:
            # 对于其他后端，使用父类的标准方法
            return super().load_doc(doc_path)

    async def run(self, task_info: dict) -> Tuple[str, str, str]:
        """执行代码生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态

        Returns:
            Tuple[str, str, str]: 生成的代码、提示信息和推理过程
        """
        try:
            # 从task_info中获取代码信息
            sketch = task_info.get('designer_code', '')

            # 从task_info中获取conductor的建议
            conductor_suggestion = task_info.get('conductor_suggestion', '')

            # 获取api文档
            if len(self.base_doc["api_docs"]) > 5000:  # 如果api文档过长，使用llm进行content压缩
                api_parser = ParserFactory.get_api_parser()
                format_api_instructions = api_parser.get_format_instructions()
                api_input_data = {
                    **self.base_doc,
                    "sketch": sketch,  # AUL代码作为sketch
                    "llm_suggestions": conductor_suggestion,  # Conductor建议
                    "error_log": task_info.get('verifier_error', ''),
                    "format_instructions": format_api_instructions
                }

                self.api_step_count += 1
                to_update_api_details = {
                    "agent_name": "api",
                    "hash": task_info.get("task_id", "Api"),
                    "task_id": task_info.get("task_id", ""),
                    "step": self.api_step_count,
                }
                self.context.update(to_update_api_details)

                api_docs_json, _, _ = await self.run_llm(self.api_docs_prompt, api_input_data, self.model_config.get("api_generator", "default"))
                parsed_content = api_parser.parse(api_docs_json)
                api_docs_suitable = "\n\n".join(
                    f"API name: {name}\nAPI description:{desc}\nAPI implement：\n{impl}"
                    for name, desc, impl in zip(
                        parsed_content.api_name,
                        parsed_content.api_desc,
                        parsed_content.api_example
                    )
                )
            else:
                api_docs_suitable = self.base_doc["api_docs"]

            # 获取dsl示例代码
            # TODO
            # if len(self.base_doc["dsl_examples"]) > 5000:  # 如果dsl示例代码过长，使用llm进行content压缩
            #     dsl_examples_suitable, _, _ = await self.run_llm(self.dsl_examples_prompt, self.base_doc, self.model_config.get("example_compressor", "default"))
            # else:
            dsl_examples_suitable = self.base_doc["dsl_examples"]

            # 基于base_doc构建输入，只更新变化的部分
            input_data = {
                **self.base_doc,
                "sketch": sketch,  # AUL代码作为sketch
                "llm_suggestions": conductor_suggestion,  # Conductor建议
                "error_log": task_info.get('verifier_error', ''),
                "inspirations": get_inspirations(task_info.get('inspirations', [])),
                "api_docs_suitable": api_docs_suitable,
                "dsl_examples_suitable": dsl_examples_suitable,
            }

            # 执行LLM生成前更新context，确保正确性
            self.codegen_step_count += 1
            to_update_codegen_details = {
                "agent_name": "coder",
                "hash": task_info.get("task_id", "Coder"),
                "task_id": task_info.get("task_id", ""),
                "step": self.codegen_step_count,
            }
            self.context.update(to_update_codegen_details)

            # 执行LLM生成
            return await self.run_llm(self.coder_prompt, input_data, self.model_config["coder"])
        except Exception as e:
            logger.error(f"Exception in coder.run: {type(e).__name__}: {e}")
            raise
