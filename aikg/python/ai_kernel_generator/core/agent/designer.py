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

from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import remove_copyright_from_text
from ai_kernel_generator.utils.markdown_utils import extract_function_details
from ai_kernel_generator.utils.hardware_utils import get_hardware_doc
from ai_kernel_generator.utils.parser_registry import create_step_parser
from ai_kernel_generator.database.database import Database

logger = logging.getLogger(__name__)


def get_inspirations(inspirations: List[dict]) -> str:
    """
    将inspirations列表转换为字符串

    Args:
        inspirations: 包含字典的列表，每个字典格式为:
                     {'strategy_mode':xxx, 'impl_code':str, 'profile':float, 'is_parent':bool}

    Returns:
        str: 拼接后的字符串，包含所有impl_code和profile信息
    """
    if not inspirations:
        return ""

    result_parts = []
    has_parent = False

    for i, inspiration in enumerate(inspirations):
        if not isinstance(inspiration, dict):
            logger.warning(f"跳过非字典类型的inspiration: {type(inspiration)}")
            continue

        sketch = inspiration.get('sketch', '')
        impl_code = inspiration.get('impl_code', '')
        profile = inspiration.get('profile', float('inf'))
        is_parent = inspiration.get('is_parent', False)
        
        # 检测是否有父代
        if is_parent:
            has_parent = True

        if sketch or impl_code:  # 只有当sketch或impl_code不为空时才添加
            # 处理profile信息，支持三元组格式
            if isinstance(profile, (list, tuple)) and len(profile) >= 3:
                gen_time, base_time, speedup = profile[0], profile[1], profile[2]
                profile_text = f"根据此方案草图生成的代码计算耗时: {gen_time:.4f}us, 基准代码耗时: {base_time:.4f}us, 加速比: {speedup:.2f}x"
            elif isinstance(profile, (list, tuple)) and len(profile) >= 1:
                profile_text = f"代码执行耗时: {profile[0]:.4f}us"
            else:
                profile_text = f"代码执行耗时: {profile:.4f}us" if profile != float('inf') else "代码执行耗时: N/A"

            # 如果是父代，添加标记
            parent_mark = " 【父代方案】" if is_parent else ""
            inspiration_text = f"## Inspiration {i+1}{parent_mark} {profile_text}\n"
            if sketch:
                inspiration_text += f"算法草图 ：\n```\n{sketch}\n```\n"
            if impl_code:
                inspiration_text += f"代码：\n```\n{impl_code}\n```\n"
            result_parts.append(inspiration_text)

    # 如果有父代，在开头添加进化优化策略说明
    if has_parent and result_parts:
        strategy_note = (
            "**进化优化策略**：\n"
            "- 标记为【父代方案】的是本次进化的基础，请以它为主要参考进行改进和优化\n"
            "- 其他 Inspiration 可作为补充参考，用于交叉变异和借鉴优化思路\n"
            "- 请在父代方案的基础上，结合其他方案的优点，生成优化后的草图\n\n"
        )
        result_parts.insert(0, strategy_note)

    return "\n".join(result_parts)


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
        self.llm_step_count = 0

        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Designer")

        context = {
            "agent_name": "designer",
            "dsl": self.dsl,
            "op_name": self.op_name,
            "backend": self.backend,
            "arch": self.arch,
            "task_desc": self.task_desc,
        }
        super().__init__(context=context, config=config)

        # 直接使用从workflow.yaml获取的designer解析器
        self.code_parser = create_step_parser("designer", self.workflow_config_path)
        if not self.code_parser:
            raise ValueError(
                "Failed to create designer parser from workflow config. Please check your workflow.yaml configuration."
            )
        self.format_instructions = self.code_parser.get_format_instructions()

        # 初始化designer生成模板
        self.designer_prompt = self.load_template("designer/gen_sketch.j2")

        self.base_doc = {
            "dsl": self.dsl,
            "dsl_basic_docs": "",
            "arch_name": self.arch,
            "backend": self.backend,
            "op_name": self.op_name,
            "task_desc": remove_copyright_from_text(self.task_desc),
            "hardware_docs": get_hardware_doc(self.backend, self.arch),
            "format_instructions": self.format_instructions,
            "sketch_guide": self.load_doc("sketch_rule.md")
        }

        # 为SWFT实现类型添加支持的API
        if self.dsl == "swft":
            try:
                supported_compute_api_str = extract_function_details()
                self.base_doc["supported_compute_api"] = supported_compute_api_str
            except Exception as e:
                logger.warning(f"获取SWFT支持的API失败: {e}")

    async def run(self, task_info: dict) -> Tuple[str, str, str]:
        """执行AUL设计代码生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态
            meta_prompts: 格式化后的meta prompts字符串

        Returns:
            tuple: (生成内容, 格式化提示词, 推理内容)
        """
        # 从task_info中获取conductor的建议
        conductor_suggestion = task_info.get("conductor_suggestion", "")

        # 基于aul_base_doc构建输入，只更新变化的部分
        input_data = {
            **self.base_doc,
            "llm_suggestions": conductor_suggestion,  # Conductor建议
            "inspirations": get_inspirations(task_info.get('inspirations', [])),
            "meta_prompts": task_info.get("meta_prompts", ""),
        }

        # 执行LLM生成前更新context，确保正确性
        self.llm_step_count += 1
        to_update_context = {
            "agent_name": "designer",
            "framework": task_info.get("framework", ""),
            "hash": task_info.get("task_id", "Designer"),
            "task_id": "",
            "step": self.llm_step_count,
            "workflow_name": task_info.get("workflow_name", ""),
        }
        self.context.update(to_update_context)

        # 执行LLM生成
        return await self.run_llm(
            self.designer_prompt, input_data, self.model_config["designer"]
        )
