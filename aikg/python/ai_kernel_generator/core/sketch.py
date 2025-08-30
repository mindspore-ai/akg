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

from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import remove_copyright_from_text, ParserFactory
from ai_kernel_generator.utils.hardware_utils import get_hardware_doc

logger = logging.getLogger(__name__)


class Sketch(AgentBase):
    """
    Sketch Agent - 将生成的triton/swft代码转换为通用算子草图

    功能：
    1. 从coder生成的具体代码中提取算法结构
    2. 识别并行策略、切分方式、核数配置
    3. 转换为符合sketch_rule.md标准的通用草图
    4. 保留核心算法逻辑，移除DSL特定语法
    """

    def __init__(
        self,
        op_name: str,
        task_desc: str,
        dsl: str = "",
        backend: str = "",
        arch: str = "",
        config: dict = None,
    ):
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.arch = arch
        self.backend = backend
        self.llm_step_count = 0

        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Sketch")

        context = {
            "agent_name": "sketch",
            "dsl": self.dsl,
            "op_name": self.op_name,
            "backend": self.backend,
            "arch": self.arch,
            "task_desc": self.task_desc,
        }
        super().__init__(context=context, config=config)

        # 使用common_utils中的sketch解析器
        self.code_parser = ParserFactory.get_sketch_parser()
        self.format_instructions = self.code_parser.get_format_instructions()

        # 初始化sketch转换模板
        self.sketch_prompt = self.load_template("sketch/code_to_sketch.j2")

        self.base_doc = {
            "dsl": self.dsl,
            "arch_name": self.arch,
            "backend": self.backend,
            "op_name": self.op_name,
            "task_desc": remove_copyright_from_text(self.task_desc),
            "hardware_docs": get_hardware_doc(self.backend, self.arch),
            "format_instructions": self.format_instructions
        }

    async def run(self, task_info: dict) -> str:
        """执行代码到Sketch转换生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态

        Returns:
            str: 解析后的算子草图内容
        """
        # 从task_info中获取coder生成的代码
        coder_code = task_info.get("coder_code", "")
        if not coder_code:
            logger.error("No coder_code found in task_info for sketch conversion")
            raise ValueError("No coder_code found in task_info for sketch conversion")

        # 记录转换信息
        logger.info(f"Converting {self.dsl} code to sketch for operator: {self.op_name}")
        logger.debug(f"Code length: {len(coder_code)} characters")

        # 基于base_doc构建输入
        input_data = {
            **self.base_doc,
            "coder_code": coder_code,
        }

        # 执行LLM生成前更新context，确保正确性
        self.llm_step_count += 1
        to_update_context = {
            "agent_name": "sketch",
            "framework": task_info.get("framework", ""),
            "hash": task_info.get("task_id", "Sketch"),
            "task_id": "",
            "workflow_name": task_info.get("workflow_name", ""),
            "step": self.llm_step_count,
        }
        self.context.update(to_update_context)

        # 获取模型配置，优先使用sketch配置，否则使用默认配置
        model_config = self.model_config.get("sketch") or self.model_config.get("default")
        if not model_config:
            logger.warning("No model config found for sketch, using first available config")
            model_config = next(iter(self.model_config.values())) if self.model_config else None

        if not model_config:
            raise ValueError("No valid model configuration found for sketch generation")

        # 执行LLM生成
        try:
            # 获取大模型输出的完整信息
            content, _, _ = await self.run_llm(
                self.sketch_prompt, input_data, model_config
            )

            # 使用解析器解析content
            try:
                parsed_result = ParserFactory.robust_parse(content, self.code_parser)
                sketch_content = parsed_result.sketch

                return sketch_content

            except Exception as parse_error:
                logger.error(f"Failed to parse sketch content: {parse_error}")
                raise

        except Exception as e:
            logger.error(f"Failed to generate sketch for {self.op_name}: {e}")
            raise
