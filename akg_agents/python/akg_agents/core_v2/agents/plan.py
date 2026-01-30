# Copyright 2026 Huawei Technologies Co., Ltd
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
Planning Agent - 独立的规划 Agent

支持两阶段规划：
1. 全局规划 (global_todo_list) - 高层次任务概览
2. 详细规划 (detailed_todo_list) - 具体工具调用和文件路径
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from akg_agents.core_v2.agents.base import AgentBase, Jinja2TemplateWrapper

logger = logging.getLogger(__name__)


class PlanningAgent(AgentBase):
    """规划 Agent
    
    负责生成任务执行计划，支持两种模式：
    1. 全局规划：生成高层次的任务概览
    2. 详细规划：生成具体的工具调用和文件路径
    
    Example:
        agent = PlanningAgent(task_id="task_001")
        
        # 生成全局 todolist
        global_todo = await agent.generate_global_todolist(
            user_input="生成一个 ReLU 算子",
            available_tools=[...]
        )
        
        # 生成详细 todolist
        detailed_todo = await agent.generate_detailed_todolist(
            user_input="生成一个 ReLU 算子",
            global_todo_list=global_todo,
            available_tools=[...]
        )
    """
    
    def __init__(self, task_id: str, model_level: str = "standard"):
        """初始化 PlanningAgent
        
        Args:
            task_id: 任务 ID
            model_level: 模型级别 ("complex" / "standard" / "fast")
        """
        super().__init__(context={"task_id": task_id, "agent_name": "PlanningAgent"}, config={})
        self.agent_name = "PlanningAgent"
        self.model_level = model_level
        self._global_prompt_template = self._load_global_prompt_template()
        self._detailed_prompt_template = self._load_detailed_prompt_template()
    
    def _get_prompts_dir(self) -> Path:
        """获取 prompts 目录路径（位于 op/resources/prompts/planning/）"""
        from akg_agents import get_project_root
        return Path(get_project_root()) / "op" / "resources" / "prompts" / "planning"
    
    def _load_global_prompt_template(self) -> Jinja2TemplateWrapper:
        """加载全局规划 prompt 模板"""
        prompt_file = self._get_prompts_dir() / "global_planning.j2"
        with open(prompt_file, "r", encoding="utf-8") as f:
            return Jinja2TemplateWrapper(f.read())
    
    def _load_detailed_prompt_template(self) -> Jinja2TemplateWrapper:
        """加载详细规划 prompt 模板"""
        prompt_file = self._get_prompts_dir() / "detailed_planning.j2"
        with open(prompt_file, "r", encoding="utf-8") as f:
            return Jinja2TemplateWrapper(f.read())
    
    def _format_tools(self, tools: List[Dict]) -> str:
        """格式化工具列表
        
        Args:
            tools: 工具定义列表，格式为 OpenAI function calling 格式
            
        Returns:
            格式化后的工具描述字符串
        """
        if not tools:
            return "（无可用工具）"
        result = []
        for i, tool in enumerate(tools, 1):
            func = tool.get("function", {})
            name = func.get('name', 'unknown')
            description = func.get('description', '')
            parameters = func.get('parameters', {})
            
            # 提取参数信息
            props = parameters.get('properties', {})
            param_list = []
            for param_name, param_info in props.items():
                param_desc = param_info.get('description', '')
                param_list.append(f"    - {param_name}: {param_desc}")
            
            tool_str = f"{i}. {name}: {description}"
            if param_list:
                tool_str += "\n   参数:\n" + "\n".join(param_list)
            result.append(tool_str)
        return "\n".join(result)
    
    def _format_action_history(self, action_history: List[Dict]) -> str:
        """格式化动作历史
        
        Args:
            action_history: 动作历史列表
            
        Returns:
            格式化后的动作历史字符串
        """
        if not action_history:
            return "（尚无执行历史）"
        result = []
        for i, action in enumerate(action_history, 1):
            tool_name = action.get("tool_name", "unknown")
            status = action.get("result", {}).get("status", "unknown")
            output_path = action.get("result", {}).get("output_path", "")
            
            action_str = f"{i}. {tool_name} → {status}"
            if output_path:
                action_str += f" (输出: {output_path})"
            result.append(action_str)
        return "\n".join(result)
    
    # ==================== 全局规划 ====================
    
    async def generate_global_todolist(
        self,
        user_input: str,
        available_tools: List[Dict],
        has_task_desc: bool = False
    ) -> str:
        """生成全局 todolist
        
        首次调用时生成高层次的任务概览。
        
        Args:
            user_input: 用户输入的需求描述
            available_tools: 可用工具列表
            has_task_desc: 用户是否已提供 task_desc 代码
            
        Returns:
            全局 todolist 字符串，格式如：
            <global_todo_list>
            1. 生成对应的算子任务
            2. 基于算子任务，做基本代码生成
            3. 进一步做代码优化
            </global_todo_list>
        """
        try:
            result, _, _ = await self.run_llm(
                prompt=self._global_prompt_template,
                input={
                    "user_input": user_input,
                    "available_tools": self._format_tools(available_tools),
                    "has_task_desc": has_task_desc
                },
                model_level=self.model_level
            )
            logger.info(f"[PlanningAgent] 全局规划生成完成")
            return result
        
        except Exception as e:
            logger.error(f"[PlanningAgent] 全局规划生成失败: {e}")
            return self._get_default_global_todolist(has_task_desc)
    
    def _get_default_global_todolist(self, has_task_desc: bool = False) -> str:
        """获取默认的全局 todolist
        
        Args:
            has_task_desc: 用户是否已提供 task_desc
        """
        if has_task_desc:
            return """<global_todo_list>
1. 实现 kernel 代码
2. 验证 kernel 精度并输出结果
</global_todo_list>"""
        else:
            return """<global_todo_list>
1. 生成算子的 task_desc 定义
2. 实现 kernel 代码
3. 验证 kernel 精度并输出结果
</global_todo_list>"""
      
    # ==================== 详细规划 ====================
    
    async def generate_detailed_todolist(
        self,
        user_input: str,
        global_todo_list: str,
        available_tools: List[Dict],
        action_history: List[Dict] = None
    ) -> str:
        """生成详细 todolist
        
        根据全局 todolist 生成具体的工具调用计划。
        
        Args:
            user_input: 用户输入的需求描述
            global_todo_list: 全局 todolist
            available_tools: 可用工具列表
            action_history: 已完成的动作历史
            
        Returns:
            详细 todolist 字符串，格式如：
            <todo_list>
            1. 调用 call_designer args={input:"init/user_input.txt", output:"designer/output.txt"}
            2. 调用 call_coder args={input:"designer/output.txt", output:"coder/kernel.py"}
            ...
            </todo_list>
        """
        try:
            result, _, _ = await self.run_llm(
                prompt=self._detailed_prompt_template,
                input={
                    "user_input": user_input,
                    "global_todo_list": global_todo_list,
                    "available_tools": self._format_tools(available_tools),
                    "action_history": self._format_action_history(action_history or [])
                },
                model_level=self.model_level
            )
            logger.info(f"[PlanningAgent] 详细规划生成完成")
            return result
        
        except Exception as e:
            logger.error(f"[PlanningAgent] 详细规划生成失败: {e}")
            return self._get_default_detailed_todolist()
    
    def _get_default_detailed_todolist(self) -> str:
        """获取默认的详细 todolist（算子生成默认流程）"""
        return """<todo_list>
1. 调用 call_op_task_builder args={input:"init/user_input.txt", output:"op_task_builder/task_desc.py", op_name:"unknown"}
2. 调用 call_coder_only args={input:"op_task_builder/task_desc.py", output:"coder/kernel_code.py", op_name:"unknown"}
3. 调用 call_kernel_verifier args={input:"coder/kernel_code.py", task_desc:"op_task_builder/task_desc.py", output:"verifier/verify_result.json", op_name:"unknown"}
4. 完成任务并输出结果
</todo_list>"""
    
