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

"""
TestCaseGenerator Agent: 生成多测试用例的 task_desc
"""

import logging
import json
from typing import Tuple
from pathlib import Path

from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)


class TestCaseGenerator(AgentBase):
    """
    测试用例生成 Agent
    
    基于已验证通过的 kernel 代码，分析其中的 assert 约束和设计范围，
    生成包含多个测试 case 的新 task_desc
    """
    
    def __init__(
        self,
        op_name: str,
        task_desc: str,
        framework: str = "torch",
        dsl: str = "",
        config: dict = None,
    ):
        self.op_name = op_name
        self.task_desc = task_desc
        self.framework = framework
        self.dsl = dsl
        self.config = config
        self.original_task_desc = task_desc
        self.llm_step_count = 0
        
        # 从 config 中获取 model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for TestCaseGenerator")
        
        # 创建 context
        context = {
            "op_name": op_name,
            "framework": framework,
            "dsl": dsl,
            "task_desc": task_desc,
        }
        super().__init__(context=context, config=config)
        
        # 加载 prompt 模板
        self.test_gen_prompt = self.load_template("test_case_generator/gen_multi_case.j2")
        
        # 设置输出格式（简单的 JSON 格式，不需要复杂的 parser）
        self.format_instructions = """
{
    "new_task_desc": "完整的新 task_desc 代码，包含 get_inputs_dyn_list 函数",
    "reasoning": "生成测试用例的思考过程和策略说明"
}
"""
        
        logger.debug(f"TestCaseGenerator initialized for {op_name}")
    
    async def run(self, task_info: dict) -> Tuple[str, str, str]:
        """
        生成包含多 case 的新 task_desc
        
        Args:
            task_info: 任务信息，包含 coder_code, designer_code 等
            
        Returns:
            Tuple[str, str, str]: (新 task_desc, prompt, reasoning)
        """
        try:
            # 提取必要信息
            sketch = task_info.get('designer_code', '')
            coder_code = task_info.get('coder_code', '')
            
            # 构建 prompt 输入（包含之前的错误信息）
            input_data = {
                "op_name": self.op_name,
                "framework": self.framework,
                "original_task_desc": self.original_task_desc,
                "sketch": sketch,
                "coder_code": coder_code,
                "previous_error": task_info.get("previous_error", ""),  # 传递之前的多 case 错误
                "format_instructions": self.format_instructions
            }
            
            # 更新 context
            self.llm_step_count += 1
            to_update_context = {
                "agent_name": "test_case_generator",
                "hash": f"test_gen_{self.op_name}",
                "task_id": task_info.get("task_id", ""),
                "step": self.llm_step_count,
            }
            self.context.update(to_update_context)
            
            # 执行 LLM 生成
            logger.info(f"开始生成多测试用例的 task_desc for {self.op_name}")
            llm_output, prompt, reasoning = await self.run_llm(
                self.test_gen_prompt,
                input_data,
                self.model_config.get("test_case_generator") or self.model_config.get("default")
            )
            
            # 解析 JSON 输出，提取 new_task_desc
            try:
                result = json.loads(llm_output)
                new_task_desc = result.get("new_task_desc", "")
                llm_reasoning = result.get("reasoning", "")
                
                if not new_task_desc:
                    raise ValueError("LLM 返回的 JSON 中没有 new_task_desc 字段")
                
                logger.info(f"成功解析 LLM 输出，提取到 new_task_desc ({len(new_task_desc)} 字符)")
                
                # 返回解析后的 new_task_desc（Python 代码），而不是 JSON
                return new_task_desc, prompt, llm_reasoning
                
            except json.JSONDecodeError as e:
                logger.error(f"无法解析 LLM 输出为 JSON: {e}")
                logger.error(f"LLM 原始输出:\n{llm_output[:500]}...")
                raise ValueError(f"LLM 输出不是有效的 JSON 格式: {e}")
            
        except Exception as e:
            logger.error(f"TestCaseGenerator 执行失败: {e}")
            raise

