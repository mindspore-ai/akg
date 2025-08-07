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

from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import remove_copyright_from_text
from ai_kernel_generator.utils.markdown_utils import extract_function_details
from ai_kernel_generator.utils.hardware_utils import get_hardware_doc
from ai_kernel_generator.utils.parser_registry import create_step_parser
from ai_kernel_generator.database.database import Database

logger = logging.getLogger(__name__)


class Designer(AgentBase):
    def __init__(
        self,
        op_name: str,
        task_desc: str,
        dsl: str = "",
        backend: str = "",
        arch: str = "",
        workflow_config_path: str = None,
        config: dict = None,
    ):
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.arch = arch
        self.backend = backend
        self.workflow_config_path = workflow_config_path
        self.config = config
        self.have_meta_prompt = True

        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Designer")

        agent_name = f"Designer -- [dsl] {self.dsl} -- [op_name] {self.op_name}"
        super().__init__(agent_name=agent_name, config=config)

        # 直接使用从workflow.yaml获取的designer解析器
        self.code_parser = create_step_parser("designer", self.workflow_config_path)
        if not self.code_parser:
            raise ValueError(
                "Failed to create designer parser from workflow config. Please check your workflow.yaml configuration."
            )
        self.format_instructions = self.code_parser.get_format_instructions()

        # 初始化designer生成模板
        self.designer_prompt = self.load_template("designer/gen_sketch.j2")

        # TODO: 加入数据库中的相似代码
        self.base_doc = {
            "dsl": self.dsl,
            "dsl_basic_docs": self.load_doc("basic_docs.md"),
            "arch_name": self.arch,
            "backend": self.backend,
            "op_name": self.op_name,
            "task_desc": remove_copyright_from_text(self.task_desc),
            "hardware_docs": get_hardware_doc(self.backend, self.arch),
            "format_instructions": self.format_instructions,
            "similar_code": "",
            "sketch_example": ""
        }

        # 为SWFT实现类型添加支持的API
        if self.dsl == "swft":
            try:
                supported_compute_api_str = extract_function_details()
                self.base_doc["supported_compute_api"] = supported_compute_api_str
            except Exception as e:
                logger.warning(f"获取SWFT支持的API失败: {e}")


    async def run(self, task_info: dict, meta_prompts: list | None) -> Tuple[str, str, str]:
        """执行AUL设计代码生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        # 从task_info中获取conductor的建议
        conductor_suggestion = task_info.get("conductor_suggestion", "")

        # 基于aul_base_doc构建输入，只更新变化的部分
        input_data = {
            **self.base_doc,
            "llm_suggestions": conductor_suggestion,  # Conductor建议
            "meta_prompts": str(meta_prompts) if meta_prompts else "",  # 初始启发
        }

        # 执行LLM生成
        return await self.run_llm(
            self.designer_prompt, input_data, self.model_config["designer"]
        )
