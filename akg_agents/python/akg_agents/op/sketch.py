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
import time

from akg_agents.cli.messages import DisplayMessage
from akg_agents.cli.runtime.message_sender import send_message
from akg_agents.core_v2.agents import AgentBase, register_agent
from akg_agents.core_v2.workflow_logger import WorkflowLogger
from akg_agents.utils.common_utils import remove_copyright_from_text, ParserFactory
from akg_agents.utils.hardware_utils import get_hardware_doc
from akg_agents.utils.task_label import resolve_task_label

logger = logging.getLogger(__name__)


@register_agent
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
        if config and "session_id" in config:
            context["session_id"] = config["session_id"]
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
            "format_instructions": self.format_instructions,
            "sketch_guide": self.load_doc("SKETCH_DESIGN_v2.md")
        }

    async def run(self, task_info: dict) -> str:
        """执行代码到Sketch转换生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态

        Returns:
            str: 解析后的算子草图内容
        """
        start_time = time.time()
        session_id = str(task_info.get("session_id") or "").strip()
        task_id = str(task_info.get("task_id") or "")
        task_label = str(task_info.get("task_label") or "").strip()
        if not task_label:
            raise ValueError("[Sketch] task_info must include task_label")
        if session_id:
            send_message(
                session_id,
                DisplayMessage(
                    text="▶ sketch",
                ),
            )

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
            "task_id": task_id,
            "task_label": task_label,
            "workflow_name": task_info.get("workflow_name", ""),
            "step": self.llm_step_count,
        }
        self.context.update(to_update_context)

        # 获取模型级别，优先使用sketch配置，否则使用默认配置，最终兜底 "standard"
        model_level = self.model_config.get("sketch") or self.model_config.get("default") or "standard"

        # 执行LLM生成
        try:
            # 获取大模型输出的完整信息（content, prompt, reasoning）
            content, formatted_prompt, reasoning_content = await self.run_llm(
                self.sketch_prompt, input_data, model_level
            )

            # 使用解析器解析content
            parsed_result = ParserFactory.robust_parse(content, self.code_parser)
            if isinstance(parsed_result, str):
                logger.warning(f"Sketch parse returned raw string for {self.op_name}, using as-is")
                sketch_content = parsed_result
            else:
                sketch_content = parsed_result.sketch
            
            # 记录到 Trace（如果有 log_dir 和 task_id）
            elapsed = time.time() - start_time
            log_dir = self.config.get("log_dir", "") if self.config else ""
            if log_dir and task_id:
                try:
                    trace = WorkflowLogger(log_dir=log_dir, category=self.op_name, task_id=task_id)
                    trace.insert_agent_record(
                        agent_name="sketch",
                        result=content,  # 保存原始 LLM 输出
                        prompt=formatted_prompt,
                        reasoning=reasoning_content,
                        session_id=session_id,
                        elapsed_s=elapsed,
                    )
                except Exception as e:
                    # trace 记录失败不应影响主流程
                    logger.warning(f"Failed to insert agent record for sketch: {e}")
            
            return sketch_content

        except Exception as e:
            logger.warning(f"Failed to generate sketch for {self.op_name}: {e}")
            return ""
