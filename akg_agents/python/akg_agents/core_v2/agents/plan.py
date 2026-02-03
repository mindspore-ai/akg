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
Plan Agent - 任务规划 Agent

负责：
- 分析用户需求，判断信息是否完整
- 根据可用工具和历史动作生成高层次执行计划
- 返回结构化的规划结果供 MainAgent 使用
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper

logger = logging.getLogger(__name__)


@register_agent
class PlanAgent(AgentBase):
    """
    任务规划 Agent
    
    作为 MainAgent 调用的工具，负责生成高层次任务规划。
    不直接与用户对话，信息不完整时返回消息给 MainAgent 处理。
    """
    
    # Agent 工具配置元数据
    TOOL_NAME = "call_plan"
    DESCRIPTION = "分析用户需求并生成高层次任务规划。如果信息不完整，一次性返回所有需要补充的信息。"
    
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "user_input": {
                "type": "string",
                "description": "用户的需求描述"
            },
            "available_tools": {
                "type": "array",
                "description": "可用工具列表（OpenAI function calling 格式）",
                "items": {"type": "object"},
                "default": []
            },
            "history_compress": {
                "type": "array",
                "description": "压缩后的历史记录列表",
                "items": {"type": "object"},
                "default": []
            },
            "task_id": {
                "type": "string",
                "description": "任务 ID（可选）",
                "default": ""
            },
            "model_level": {
                "type": "string",
                "description": "模型级别（'standard', 'fast', 'complex'）",
                "default": "standard"
            }
        },
        "required": ["user_input"]
    }
    
    def __init__(self):
        """初始化 PlanAgent"""
        context = {
            "agent_name": "plan",
            "task_label": "planning",
        }
        super().__init__(context=context)
        
        # 加载 prompt 模板（从 core_v2/agents/prompts/ 目录）
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> Jinja2TemplateWrapper:
        """加载 prompt 模板"""
        prompt_file = Path(__file__).parent / "prompts" / "plan.j2"
        with open(prompt_file, "r", encoding="utf-8") as f:
            return Jinja2TemplateWrapper(f.read())
    
    def _format_tools(self, tools: List[Dict]) -> str:
        """格式化工具列表
        
        Args:
            tools: 工具定义列表
            
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
    
    def _format_history_compress(self, history_compress: List[Dict]) -> str:
        """格式化压缩后的历史记录
        
        Args:
            history_compress: 压缩后的历史记录列表
            
        Returns:
            格式化后的历史字符串
        """
        if not history_compress:
            return "（尚无执行历史）"
        
        result = []
        for i, item in enumerate(history_compress, 1):
            # 支持多种格式的历史记录
            if isinstance(item, dict):
                tool_name = item.get("tool_name", item.get("action", "unknown"))
                status = item.get("status", item.get("result", {}).get("status", "unknown"))
                summary = item.get("summary", item.get("output", ""))
                
                action_str = f"{i}. {tool_name} → {status}"
                if summary:
                    summary_preview = summary[:100] + "..." if len(str(summary)) > 100 else summary
                    action_str += f"\n   摘要: {summary_preview}"
            else:
                action_str = f"{i}. {str(item)[:100]}"
            
            result.append(action_str)
        return "\n".join(result)
    
    def _extract_nested_json(self, text: str) -> Optional[str]:
        """提取多层嵌套的 JSON 字符串
        
        使用括号匹配算法，支持任意深度的嵌套。
        优先提取最外层、最完整的 JSON 对象。
        
        Args:
            text: 包含 JSON 的文本
            
        Returns:
            提取的 JSON 字符串，失败返回 None
        """
        # 方法1: 尝试从 ```json 代码块提取
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_block_match:
            candidate = json_block_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        
        # 方法2: 使用括号匹配找到完整的 JSON 对象
        # 找所有 { 的位置，尝试从每个位置提取完整 JSON
        candidates = []
        
        for i, char in enumerate(text):
            if char == '{':
                # 尝试从这个位置提取完整的 JSON
                depth = 0
                in_string = False
                escape_next = False
                end_pos = -1
                
                for j in range(i, len(text)):
                    c = text[j]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if c == '\\' and in_string:
                        escape_next = True
                        continue
                    
                    if c == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if in_string:
                        continue
                    
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end_pos = j + 1
                            break
                
                if end_pos > 0:
                    candidate = text[i:end_pos]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            # 记录候选项：(起始位置, JSON字符串, 解析结果)
                            candidates.append((i, candidate, parsed))
                    except json.JSONDecodeError:
                        pass
        
        if not candidates:
            return None
        
        # 优先返回包含 tool_name 和 result 的完整结构
        for _, json_str, parsed in candidates:
            if "tool_name" in parsed and "result" in parsed:
                return json_str
        
        # 否则返回最长的有效 JSON
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        return candidates[0][1]
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 响应
        
        Args:
            response: LLM 原始响应
            
        Returns:
            结构化的规划结果
        """
        # 使用专用的嵌套 JSON 提取器
        json_str = self._extract_nested_json(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                # 验证返回格式是否完整
                if self._validate_plan_result(parsed):
                    return parsed
                else:
                    logger.warning(f"[PlanAgent] JSON 格式不完整: {json_str[:200]}")
            except json.JSONDecodeError as e:
                logger.warning(f"[PlanAgent] JSON 解析失败: {e}")
              
        # 默认返回失败
        return self._build_fail_response("无法理解用户需求，请提供更详细的描述")
    
    def _validate_plan_result(self, result: Dict[str, Any]) -> bool:
        """验证规划结果格式是否正确
        
        Args:
            result: 解析后的 JSON 对象
            
        Returns:
            格式是否有效
        """
        # 必须有 result 字段
        if "result" not in result:
            return False
        
        res = result["result"]
        # result 必须有 status 和 desc
        if not isinstance(res, dict) or "status" not in res or "desc" not in res:
            return False
        
        # 如果是成功状态，必须有 arguments.steps
        if res.get("status") == "success":
            args = result.get("arguments", {})
            if not isinstance(args, dict) or "steps" not in args:
                return False
            steps = args["steps"]
            if not isinstance(steps, list) or len(steps) == 0:
                return False
        
        return True
    
    def _build_fail_response(self, message: str) -> Dict[str, Any]:
        """构建失败响应"""
        return {
            "tool_name": "plan",
            "arguments": {},
            "result": {
                "desc": message,
                "status": "fail"
            }
        }
    
    async def run(
        self,
        user_input: str,
        available_tools: Optional[List[Dict]] = None,
        history_compress: Optional[List[Dict]] = None,
        task_id: str = "",
        model_level: str = "standard"
    ) -> Tuple[Dict[str, Any], str, str]:
        """执行任务规划
        
        Args:
            user_input: 用户需求描述
            available_tools: 可用工具列表
            history_compress: 压缩后的历史记录
            task_id: 任务 ID
            model_level: 模型级别
            
        Returns:
            Tuple[Dict, str, str]: (规划结果, 完整 prompt, 推理过程)
        """
        available_tools = available_tools or []
        history_compress = history_compress or []
        
        try:
            # 更新上下文
            self.context.update({
                "task_id": task_id,
                "user_input": user_input[:100] + "..." if len(user_input) > 100 else user_input
            })
            
            # 格式化输入
            formatted_tools = self._format_tools(available_tools)
            formatted_history = self._format_history_compress(history_compress)
            
            # 调用 LLM
            response, full_prompt, reasoning = await self.run_llm(
                prompt=self.prompt_template,
                input={
                    "user_input": user_input,
                    "available_tools": formatted_tools,
                    "history_compress": formatted_history
                },
                model_level=model_level
            )
            
            # 解析响应
            plan_result = self._parse_llm_response(response)
            
            status = plan_result.get("result", {}).get("status", "unknown")
            logger.info(f"[PlanAgent] 规划完成: status={status}")
            
            return plan_result, full_prompt, reasoning
            
        except Exception as e:
            logger.error(f"[PlanAgent] 规划失败: {e}")
            return self._build_fail_response(f"规划过程出错: {str(e)}"), "", ""
