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
input
planlist = planagent -> llm -> ask_user -> planlist
planlist -》llm -》tool_call -> result (tool， planlist) -> planlist(status全部成功) -> finish
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from akg_agents.core_v2.agents.base import AgentBase
from akg_agents.core_v2.agents.registry import register_agent
from akg_agents.core_v2.tools.tool_executor import ToolExecutor
from akg_agents.core_v2.filesystem import (
    TraceSystem,
    ActionRecord,
    NodeState,
)
from akg_agents.core_v2.filesystem.models import (
    AgentInfo,
    TaskInfo,
    ExecutionInfo,
)
from akg_agents.core_v2.llm.factory import create_llm_client

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class KernelAgent(AgentBase):

    MAX_ITERATIONS = 50  # 最大迭代次数（防止无限循环）
    DEFAULT_MAX_RETRIES = 3  # 默认最大重试次数
    
    def __init__(
        self,
        task_id: str,
        model_level: str = None,
        config: Dict = None,
        framework: str = "torch",
        backend: str = "cuda",
        arch: str = "a100",
        dsl: str = "triton",
        base_dir: Optional[str] = None
    ):
        super().__init__(
            context={"task_id": task_id, "agent_name": "KernelAgent"},
            config=config
        )
        
        self.task_id = task_id
        self.model_level = model_level
        self.config = config or {}
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.dsl = dsl

        self.base_dir = base_dir or str(Path.home() / ".aikg")
        self.trace = TraceSystem(task_id=task_id, base_dir=self.base_dir)

        self.history: List[ActionRecord] = []
        self.plan_list: List[Dict] = []
        self._initialized = False
        self._original_user_input: Optional[str] = None

        self.llm_client = create_llm_client(model_level=self.model_level or "standard")

        self.available_tools = self._load_available_tools()
        self.agent_registry = self._load_agent_registry()

        self.tool_executor = ToolExecutor(
            agent_registry=self.agent_registry,
            agent_context={
                "task_id": self.task_id,
                "dsl": self.dsl,
                "framework": self.framework,
                "backend": self.backend,
                "arch": self.arch,
                "config": self.config or {},
                "model_level": self.model_level or "standard"
            },
            history=self.history
        )

        prompt_file = Path(__file__).parent.parent / "kernel_agent" / "prompts" / "kernel_agent_system.j2"
        with open(prompt_file, "r", encoding="utf-8") as f:
            from akg_agents.core_v2.agents import Jinja2TemplateWrapper
            self.system_prompt_template = Jinja2TemplateWrapper(f.read())
    
    def _load_available_tools(self) -> List[Dict]:
        """加载可用工具列表"""
        import yaml
        
        # 延迟导入避免循环依赖
        from akg_agents import get_project_root
        tools_file = Path(get_project_root()) / "core_v2" / "config" / "domain_tools.yaml"
        with open(tools_file, "r", encoding="utf-8") as f:
            tools_config = yaml.safe_load(f)
        
        available_tools = []
        for tool_name, tool_def in tools_config.get("tools", {}).items():
            func = tool_def.get("function", {})
            if func:
                available_tools.append({
                    "type": "function",
                    "function": func
                })
        return available_tools
    
    def _load_agent_registry(self) -> Dict[str, Any]:
        """动态加载所有注册的 Agent"""
        from akg_agents.core_v2.agents.registry import AgentRegistry
        
        agent_registry = {}
        
        # 加载 op scope agents
        op_agent_names = AgentRegistry.list_agents(scope="op")
        logger.info(f"[KernelAgent] 发现 {len(op_agent_names)} 个 op scope agents")
        
        for agent_name in op_agent_names:
            try:
                agent_class = AgentRegistry.get_agent_class(agent_name)
                if not hasattr(agent_class, 'TOOL_NAME') or not agent_class.TOOL_NAME:
                    continue
                
                for tool_name, tool_def in agent_class.load_tool_config().items():
                    agent_registry[tool_name] = {
                        "agent_class": agent_class,
                        "config": tool_def
                    }
                    self.available_tools.append({
                        "type": "function",
                        "function": tool_def.get("function", {})
                    })
                    logger.info(f"[KernelAgent] 注册工具: {tool_name}")
            except Exception as e:
                logger.warning(f"[KernelAgent] 加载 Agent '{agent_name}' 失败: {e}")
        
        # 加载 plan agent
        try:
            plan_class = AgentRegistry.get_agent_class("plan")
            if hasattr(plan_class, 'TOOL_NAME') and plan_class.TOOL_NAME:
                for tool_name, tool_def in plan_class.load_tool_config().items():
                    agent_registry[tool_name] = {
                        "agent_class": plan_class,
                        "config": tool_def
                    }
                    self.available_tools.append({
                        "type": "function",
                        "function": tool_def.get("function", {})
                    })
                    logger.info(f"[KernelAgent] 注册工具: {tool_name}")
        except Exception as e:
            logger.warning(f"[KernelAgent] 加载 PlanAgent 失败: {e}")
        
        return agent_registry
    
    def _initialize_task(self, user_input: str):
        """初始化任务"""
        self.trace.initialize(force=True)
        
        root_state = NodeState(
            node_id="root",
            turn=0,
            status="init",
            agent_info=AgentInfo(
                agent_name="KernelAgent",
                agent_id="root"
            ).to_dict(),
            task_info=TaskInfo(
                task_id=self.task_id,
                task_input=user_input,
                op_name="",
                dsl=self.dsl,
                backend=self.backend,
                arch=self.arch
            ).to_dict(),
            execution_info=ExecutionInfo(
                tool_call_counter=0,
                first_thinking_done=False,
                current_turn=0
            ).to_dict()
        )
        
        self.trace.fs.save_node_state("root", root_state)
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        流程：
        1. input → planlist = planagent -> llm -> ask_user -> planlist
        2. planlist → llm → tool_call -> result -> planlist(LLM更新状态) → finish
        """
        # 初始化
        if not self._initialized:
            self._initialize_task(user_input)
            self._original_user_input = user_input
            self.tool_executor.agent_context["user_input"] = user_input
            self._initialized = True
        else:
            self._handle_user_response(user_input)
        
        # ReAct 循环
        iteration = 0
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            logger.info(f"[ReAct {iteration}] ========")
            
            # 1. LLM 决定下一步：返回 tool_call 和更新后的 plan_list
            llm_response = await self._get_next_action()
            
            if not llm_response:
                return self._build_error_response("LLM 调用失败")
            
            # 更新 plan_list（LLM 负责更新状态）
            if "plan_list" in llm_response:
                self.plan_list = llm_response["plan_list"]
            
            tool_name = llm_response.get("tool_name")
            arguments = llm_response.get("arguments", {})
            reason = llm_response.get("reason", "")
            
            if tool_name == "finish":
                logger.info(f"[ReAct] 任务完成")
                return self._build_success_response()
            
            logger.info(f"[Reasoning] {tool_name} - {reason}")
            
            # 2. 执行 tool
            if tool_name == "ask_user":
                return self._handle_ask_user(arguments)
            
            result = await self._execute_tool(tool_name, arguments)
            
            # 如果是 plan tool，提取 plan_list
            if tool_name == "plan":
                self._update_plan_from_result(result)
            
            logger.info(f"[Observation] history: {len(self.history)} 条")
        
        return self._build_error_response("达到最大迭代次数")
    
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """执行工具并记录"""
        start_time = time.time()
        result = await self.tool_executor.execute(tool_name, arguments)
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"[Acting] {tool_name}: {result.get('status')} ({duration_ms}ms)")
        
        self.history.append(ActionRecord(
            action_id=f"action_{len(self.history) + 1}",
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=duration_ms
        ))
        
        return result
    
    async def _get_next_action(self) -> Optional[Dict]:
        """调用 LLM，获取下一步要执行的工具和更新后的 plan_list"""
        # 格式化 prompt（传递当前状态给 LLM）
        prompt = self.system_prompt_template.format(
            available_tools=json.dumps([t["function"] for t in self.available_tools], indent=2, ensure_ascii=False),
            user_input=self._original_user_input or "",
            plan_list=json.dumps(self.plan_list, indent=2, ensure_ascii=False) if self.plan_list else "",
            action_history=json.dumps([self._format_action_record(r) for r in self.history], indent=2, ensure_ascii=False) if self.history else ""
        )
        
        from akg_agents.core_v2.agents import Jinja2TemplateWrapper
        template = Jinja2TemplateWrapper("{{ prompt }}")
        
        try:
            response, _, _ = await self.run_llm(
                template,
                {"prompt": prompt},
                self.model_level or "standard"
            )
            llm_response = json.loads(response)
            logger.info(f"[LLM] {llm_response.get('tool_name')} - {llm_response.get('reason', '')[:50]}...")
            
            return llm_response
        
        except json.JSONDecodeError as e:
            logger.error(f"[LLM 解析失败] {e}")
            logger.debug(f"响应内容: {response[:200]}")
            return {
                "tool_name": "ask_user",
                "arguments": {"message": "系统错误，LLM 响应格式错误"},
                "reason": "LLM 响应解析失败"
            }
        except Exception as e:
            logger.error(f"[LLM 调用失败] {e}")
            return None
    
    def _handle_user_response(self, user_input: str):
        """记录用户响应到 history"""
        for record in reversed(self.history):
            if record.tool_name == "ask_user" and record.result.get("status") == "waiting":
                record.result["status"] = "responded"
                record.result["user_response"] = user_input
                logger.info(f"[用户响应] {user_input[:50]}...")
                break
    
    def _handle_ask_user(self, arguments: Dict) -> Dict[str, Any]:
        """暂停执行，等待用户响应"""
        message = arguments.get("message", "请提供更多信息")
        
        self.history.append(ActionRecord(
            action_id=f"action_{len(self.history) + 1}",
            tool_name="ask_user",
            arguments=arguments,
            result={"status": "waiting", "message": message},
            duration_ms=0
        ))
        
        logger.info(f"[等待用户] {message[:100]}...")
        
        return {
            "status": "waiting_for_user",
            "message": message,
            "output": f"等待用户响应: {message}",
            "plan_list": self.plan_list,
            "history": [self._format_action_record(r) for r in self.history]
        }
    
    def _update_plan_from_result(self, result: Dict):
        """从 plan 工具的结果中提取 plan_list"""
        try:
            output = result.get("output", {})
            if isinstance(output, dict) and "steps" in output:
                self.plan_list = output["steps"]
                logger.info(f"[Plan 更新] {len(self.plan_list)} 个步骤")
            elif isinstance(output, str) and output.strip():  # 检查非空字符串
                plan_data = json.loads(output)
                if "steps" in plan_data:
                    self.plan_list = plan_data["steps"]
                    logger.info(f"[Plan 更新] {len(self.plan_list)} 个步骤")
            else:
                logger.warning(f"[Plan 更新] output 为空或格式不正确")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[Plan 更新失败] {e}")
    
    
    def _build_success_response(self) -> Dict[str, Any]:
        """构建成功响应"""
        return {
            "status": "success",
            "output": "所有任务已完成",
            "error_information": "",
            "plan_list": self.plan_list,
            "history": [self._format_action_record(r) for r in self.history],
            "total_actions": len(self.history)
        }
    
    def _build_error_response(self, error_msg: str) -> Dict[str, Any]:
        """构建错误响应"""
        return {
            "status": "error",
            "output": "",
            "error_information": error_msg,
            "plan_list": self.plan_list,
            "history": [self._format_action_record(r) for r in self.history],
            "total_actions": len(self.history)
        }
    
    def _format_action_record(self, record: ActionRecord) -> Dict:
        """格式化 ActionRecord 为字典"""
        return {
            "action_id": record.action_id,
            "tool_name": record.tool_name,
            "arguments": record.arguments,
            "result": record.result,
            "duration_ms": record.duration_ms
        }
