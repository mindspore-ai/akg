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
import asyncio
import logging
from typing import Dict, Any, List

from contextlib import nullcontext
from langchain.tools import tool

from akg_agents.core.sub_agent_registry import SubAgentBase, SubAgentRegistry
from akg_agents.core.tools.basic_tools import request_tool_approval
from akg_agents.core.tools.tool_schemas import SubAgentInput, OpTaskBuilderInput
from akg_agents.utils.stream_output import stream_output_override

logger = logging.getLogger(__name__)


def _build_tool_description(info: Dict[str, Any]) -> str:
    """构建 Tool 描述"""
    name = info.get("name", "")
    desc = info.get("description", "")
    use_cases = info.get("use_cases", [])
    
    lines = [f"调用 {name}。{desc}"]
    if use_cases:
        lines.append("适用场景: " + "; ".join(use_cases[:3]))
    return "\n".join(lines)


def create_sub_agent_tools(
    registry: SubAgentRegistry,
    config: dict,
    framework: str = "torch",
    backend: str = "cuda",
    arch: str = "a100",
    dsl: str = "triton"
) -> List:
    """从 SubAgentRegistry 创建所有 SubAgent 的 Tool"""
    tools = []
    
    for agent_name in registry.list_agents().keys():
        try:
            sub_agent = registry.get_agent(
                agent_name=agent_name,
                config=config,
                framework=framework,
                backend=backend,
                arch=arch,
                dsl=dsl
            )
            
            if sub_agent is None:
                logger.warning(f"Failed to create SubAgent: {agent_name}")
                continue
            
            tool_func = _create_tool(sub_agent)
            tools.append(tool_func)
            logger.info(f"Created tool: call_{agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to create tool for {agent_name}: {e}")
    
    logger.info(f"Created {len(tools)} SubAgent tools")
    return tools


def _create_tool(sub_agent: SubAgentBase):
    info = sub_agent.get_detailed_info()
    agent_name = sub_agent.get_name()
    tool_name = f"call_{agent_name}"
    tool_desc = _build_tool_description(info)
    
    if agent_name == "op_task_builder":
        return _create_op_task_builder_tool(sub_agent, tool_name, tool_desc)
    
    # 其他 SubAgent 使用通用的 schema
    @tool(tool_name, args_schema=SubAgentInput, description=tool_desc)
    async def sub_agent_tool(
        task_code: str,
        op_name: str,
        task_id: str = "default_task",
        task_label: str = "",
        task_type: str = "precision_only",
        generated_code: str = "",
        device_id: int = 0,
        user_requirements: str = ""
    ) -> Dict[str, Any]:
        """调用 SubAgent
        
        Args:
            user_requirements: 用户的额外需求说明，如'高性能优化'、'使用向量化'等。
                              会传递给 Coder 作为优化指导。
        """
        approval_args = {
            "op_name": op_name,
            "task_id": task_id,
            "task_label": task_label,
            "task_type": task_type,
            "device_id": device_id,
            "user_requirements": user_requirements,
            "task_code_len": len(task_code or ""),
            "generated_code_len": len(generated_code or ""),
        }
        if not request_tool_approval(tool_name, approval_args):
            return {"success": False, "error": "[CANCELLED] tool execution denied by user"}

        logger.info(f"Calling {sub_agent.get_name()}, op={op_name}")
        if user_requirements:
            logger.info(f"  user_requirements: {user_requirements[:100]}...")
        
        try:
            # 关键：避免通过 os.environ 临时开关导致的并发冲突。
            # （在 ReAct 主流式输出场景下，可避免子 agent 的 LLMStreamMessage 与主 stream 互相穿插。）
            cm = (
                nullcontext()
                if sub_agent.get_name() in ("codeonly",)
                else stream_output_override(False)
            )
            with cm:
                success, result = await sub_agent.execute(
                    task_code=task_code,
                    op_name=op_name,
                    task_id=task_id,
                    task_label=task_label,
                    task_type=task_type,
                    generated_code=generated_code,
                    device_id=device_id,
                    user_requirements=user_requirements,
                )
            
            result["success"] = success
            return result
        except asyncio.CancelledError:
            # 用户取消操作，记录日志并向上传播
            logger.info(f"SubAgent {sub_agent.get_name()} was cancelled by user")
            raise  # 重新抛出让上层正确处理取消
        except Exception as e:
            logger.error(f"SubAgent {sub_agent.get_name()} failed: {e}")
            return {"success": False, "error": str(e)}
    
    return sub_agent_tool


def _create_op_task_builder_tool(sub_agent: SubAgentBase, tool_name: str, tool_desc: str):
    """为 OpTaskBuilder 创建专门的 Tool"""
    
    @tool(tool_name, args_schema=OpTaskBuilderInput, description=tool_desc)
    async def op_task_builder_tool(
        user_request: str,
        user_feedback: str = None,
        task_code: str = "",
        op_name: str = "",
        task_id: str = "default_task"
    ) -> Dict[str, Any]:
        """调用 OpTaskBuilder 生成 task_desc 代码
        
        将用户的自然语言需求转换为 KernelBench 格式的 task 代码。
        生成的 task 需要传给其他子 Agent 生成 Triton 代码。
        """
        approval_args = {
            "user_request": user_request,
            "user_feedback": user_feedback,
            "task_code_len": len(task_code or ""),
            "op_name": op_name,
            "task_id": task_id,
        }
        if not request_tool_approval(tool_name, approval_args):
            return {"success": False, "error": "[CANCELLED] tool execution denied by user"}

        logger.info(f"Calling op_task_builder, user_request: {user_request[:50]}...")
        
        try:
            success, result = await sub_agent.execute(
                task_code=task_code,
                op_name=op_name,
                task_id=task_id,
                user_request=user_request,
                user_feedback=user_feedback,
            )
            
            result["success"] = success
            return result
        except asyncio.CancelledError:
            logger.info("OpTaskBuilder was cancelled by user")
            raise
        except Exception as e:
            logger.error(f"OpTaskBuilder failed: {e}")
            return {"success": False, "error": str(e)}
    
    return op_task_builder_tool
