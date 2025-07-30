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

logger = logging.getLogger(__name__)


class Designer(AgentBase):
    def __init__(self, op_name: str, task_desc: str, dsl: str = "", backend: str = "", arch: str = "", workflow_config_path: str = None, config: dict = None):
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.arch = arch
        self.backend = backend
        self.workflow_config_path = workflow_config_path
        self.config = config

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
                "Failed to create designer parser from workflow config. Please check your workflow.yaml configuration.")
        self.format_instructions = self.code_parser.get_format_instructions()

        # 初始化designer生成模板
        self.designer_prompt = self.load_template("designer/aul_gen_template.j2")

        # 准备基础文档数据
        self.base_doc = {
            "op_name": self.op_name,
            "task_desc": remove_copyright_from_text(self.task_desc),
            "aul_spec": self.get_aul_base_doc(),
            "supported_compute_api": "",
            "aul_tiling": self.load_doc("aul_tiling.md"),
            "hardware_docs": get_hardware_doc(self.backend, self.arch),
            "format_instructions": self.format_instructions,
        }

        # 为SWFT实现类型添加支持的API
        if self.dsl == "swft":
            try:
                supported_compute_api_str = extract_function_details()
                self.base_doc["supported_compute_api"] = supported_compute_api_str
            except Exception as e:
                logger.warning(f"获取SWFT支持的API失败: {e}")

    def get_aul_base_doc(self) -> str:
        """加载AUL规范文档"""
        # 按顺序定义要加载的AUL文档文件
        if self.dsl == "swft":
            aul_doc_files = [
                "aul_base.md",
                "aul_rules.md",
                "aul_npu.md",
                "aul_npu_special_op.md",
                "aul_npu_templetes.md",
                "aul_suggestions.md"
            ]
        elif "triton" in self.dsl:
            # triton
            aul_doc_files = [
                "aul_base.md",
                "aul_rules.md",
                "aul_npu.md",
                "aul_npu_templetes.md",
                "aul_suggestions.md"
            ]

        # 使用配置化的文档目录
        combined_spec = ""
        for doc_file in aul_doc_files:
            try:
                # 使用load_doc方法，自动根据agent类型选择正确的目录
                content = self.load_doc(doc_file)
                if content:  # 只有当内容不为空时才添加
                    combined_spec += content + "\n\n"
            except Exception as e:
                logger.warning(f"加载AUL文档失败 {doc_file}: {e}")
                continue

        return combined_spec

    async def run(self, task_info: dict) -> Tuple[str, str, str]:
        """执行AUL设计代码生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        # 从task_info中获取conductor的建议
        conductor_suggestion = task_info.get('conductor_suggestion', '')

        # 基于base_doc构建输入，只更新变化的部分
        input_data = {
            **self.base_doc,
            "llm_suggestions": conductor_suggestion,  # Conductor建议
        }

        # 执行LLM生成
        return await self.run_llm(self.designer_prompt, input_data, self.model_config["designer"])
