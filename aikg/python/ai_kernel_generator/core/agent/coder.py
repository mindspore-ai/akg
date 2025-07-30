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
from typing import Tuple

from ai_kernel_generator.utils.common_utils import remove_copyright_from_text
from ai_kernel_generator.utils.parser_registry import create_step_parser
from ai_kernel_generator.utils.hardware_utils import get_hardware_doc
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)


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

        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Coder")

        agent_name = f"Coder -- [dsl] {self.dsl} -- [op_name] {self.op_name} -- [framework] {self.framework}"
        super().__init__(agent_name=agent_name, config=config)

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

        # 准备基础文档数据
        self.base_doc = {
            "op_name": self.op_name,
            "task_desc": self.task_desc,
            "framework": self.framework,
            "dsl": self.dsl,
            "func_name": self.func_name,
            "format_instructions": self.format_instructions,

            # API和文档
            "api_docs": self.load_doc("api/api.md"),
            "dsl_basic_docs": self.load_doc("basic_docs.md"),
            "dsl_sample_code": "",
            "expert_suggestion": self.load_doc("suggestion_docs.md"),

            # 可选参数
            "hardware_docs": get_hardware_doc(self.backend, self.arch),
            "arch_name": self.arch,
            "database_examples": "",
            "evolve_attempts": "",
        }

    async def run(self, task_info: dict) -> Tuple[str, str, str]:
        """执行代码生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态

        Returns:
            Tuple[str, str, str]: 生成的代码、提示信息和推理过程
        """
        # 从task_info中获取代码信息
        sketch = task_info.get('designer_code', '')

        # 从task_info中获取conductor的建议
        conductor_suggestion = task_info.get('conductor_suggestion', '')

        # 基于base_doc构建输入，只更新变化的部分
        input_data = {
            **self.base_doc,
            "sketch": sketch,  # AUL代码作为sketch
            "llm_suggestions": conductor_suggestion,  # Conductor建议
            "error_log": task_info.get('verifier_error', ''),
        }

        # 执行LLM生成
        return await self.run_llm(self.coder_prompt, input_data, self.model_config["coder"])
