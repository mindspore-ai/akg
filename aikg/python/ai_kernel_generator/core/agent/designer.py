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
            {
                'strategy_mode': str,
                'impl_code': str,
                'sketch': str,
                'profile': {
                    'gen_time': float,
                    'base_time': float,
                    'speedup': float,
                    'autotune_summary': str (可选，仅triton+ascend)
                },
                'is_parent': bool
            }

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
        profile = inspiration.get('profile', {})
        is_parent = inspiration.get('is_parent', False)
        
        # 检测是否有父代
        if is_parent:
            has_parent = True

        if sketch or impl_code:  # 只有当sketch或impl_code不为空时才添加
            # 处理profile信息（dict格式）
            gen_time = profile.get('gen_time', float('inf'))
            base_time = profile.get('base_time', 0.0)
            speedup = profile.get('speedup', 0.0)
            autotune_summary = profile.get('autotune_summary', '')
            
            if gen_time != float('inf'):
                profile_text = f"根据此方案草图生成的代码计算耗时: {gen_time:.4f}us, 基准代码耗时: {base_time:.4f}us, 加速比: {speedup:.2f}x"
                # 如果有autotune信息，添加到profile_text
                if autotune_summary:
                    profile_text += f"\n\nAutotune配置详情:\n{autotune_summary}"
            else:
                profile_text = "代码执行耗时: N/A"

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
        workflow_config_path: str = None,  # 已废弃，保留用于向后兼容
        parser_config_path: str = None,     # 新的 parser 配置路径
        config: dict = None,
    ):
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.arch = arch
        self.backend = backend
        self.workflow_config_path = workflow_config_path  # 保留用于向后兼容
        self.parser_config_path = parser_config_path  # 新的配置路径
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

        # 使用新的 parser loader（不依赖 workflow.yaml）
        from ai_kernel_generator.utils.parser_loader import create_agent_parser
        self.code_parser = create_agent_parser("designer", self.parser_config_path)
        if not self.code_parser:
            raise ValueError(
                "Failed to create designer parser. Please check your parser_config.yaml configuration."
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
            "sketch_guide": self.load_doc("SKETCH_DESIGN_v2.md")
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
        
        # 通过task_id判断是否是evolve场景以及轮次
        # evolve场景的task_id格式: "{round_idx}_{island_idx}_{pid}" 或 "{round_idx}_{pid}"
        # 非evolve场景的task_id格式: 任意字符串（如"0", "test_task"等）
        task_id = task_info.get("task_id", "")
        evolve_first_round = False  # 默认不显示available_tiling
        
        if "_" in task_id:
            # 可能是evolve场景，尝试解析第一个数字作为round_idx
            try:
                round_idx = int(task_id.split("_")[0])
                # 如果round_idx > 1，说明是第二轮及以后，显示available_tiling
                if round_idx > 1:
                    evolve_first_round = True
            except (ValueError, IndexError):
                # 不是evolve场景的task_id格式，保持默认值False
                pass
        
        # ============ Hint模式检测 ============
        enable_hint_mode = self.config.get("enable_hint_mode", False)
        has_hint = False
        
        if enable_hint_mode:
            # 检测task_desc中是否有@hint
            has_hint = 'hint' in self.base_doc["task_desc"].lower()
            
            if has_hint:
                task_id = task_info.get('task_id', '0')
                logger.info(f"[Task {task_id}] 检测到hint，启用Hint模式")

        # 基于aul_base_doc构建输入，只更新变化的部分
        input_data = {
            **self.base_doc,
            "llm_suggestions": conductor_suggestion,  # Conductor建议
            "inspirations": get_inspirations(task_info.get('inspirations', [])),
            "meta_prompts": task_info.get("meta_prompts", ""),
            "handwrite_suggestions": task_info.get("handwrite_suggestions", []),
            "evolve_first_round": evolve_first_round,  # 控制是否显示available_tiling
            "enable_llm_range_inference": self.config.get("enable_llm_range_inference", False),  # LLM推理模式
            "enable_hint_mode": enable_hint_mode,  # Hint模式
            "has_hint": has_hint,  # 是否检测到hint
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
        # run_llm返回: (生成内容, 格式化提示词, 推理内容)
        llm_result, formatted_prompt, llm_reasoning = await self.run_llm(
            self.designer_prompt, input_data, self.model_config["designer"]
        )
        
        # ============ 处理Hint模式的输出 ============
        if enable_hint_mode and has_hint:
            import json
            try:
                # 解析JSON格式的生成内容
                result_dict = json.loads(llm_result)
                sketch = result_dict.get("sketch", "")
                reasoning = result_dict.get("reasoning", llm_reasoning)
                
                # 如果有space_config，保存到task_info（用于MultiCaseGenerator采样）
                result_for_return = {"code": sketch}
                if "space_config" in result_dict:
                    space_config_code = result_dict["space_config"]
                    task_info["space_config_code"] = space_config_code
                    result_for_return["space_config_code"] = space_config_code  # 也返回给 LangGraph
                    task_id = task_info.get('task_id', '0')
                    logger.info(f"[Task {task_id}] Designer生成了参数空间配置")
                
                # 转换为标准格式（支持 parser_config.yaml 定义：code + 可选的 space_config_code）
                # 将{"sketch": "...", "space_config": "...", "reasoning": "..."} 转换为 {"code": "...", "space_config_code": "..."}
                standard_result = json.dumps(result_for_return, ensure_ascii=False)
                
                # 返回: (标准格式的JSON字符串, 格式化提示词, 推理内容)
                return standard_result, formatted_prompt, reasoning
            except json.JSONDecodeError as e:
                # 如果解析失败，按原有流程返回
                logger.warning(f"[{self.op_name}] Hint模式下JSON解析失败: {e}，使用原始输出")
                return llm_result, formatted_prompt, llm_reasoning
        
        # 非Hint模式，直接返回run_llm的结果
        return llm_result, formatted_prompt, llm_reasoning
