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
ReActAgent - ReAct 模式 Agent 基类

提供通用的 ReAct（Reasoning + Acting）循环实现：
- ReAct 主循环（节点优先流程：先创建节点获取 cur_path，再执行工具）
- 工具执行 + 参数表达式解析（read_json_file 等）
- 结果持久化（result.json + 特殊文件）
- LLM 调用和响应解析
- Trace 系统集成
- 用户交互处理

子类需要实现以下抽象方法：
- _load_prompt_template(): 加载 prompt 模板
- _build_prompt_context(): 构建 prompt 上下文变量
- _get_agent_context(): 获取 agent 上下文（传递给 ToolExecutor）
- _load_available_tools(): 加载可用工具列表
- _get_agent_name(): 获取 agent 名称
- _get_task_info_extra(): 获取任务信息的额外字段（可选）
"""

import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from akg_agents.core_v2.agents.base import AgentBase, Jinja2TemplateWrapper
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


class ReActAgent(AgentBase, ABC):
    """
    ReAct 模式 Agent 基类
    
    实现通用的 ReAct 循环，子类只需实现业务特定的方法。
    
    核心流程（节点优先）：
    1. 每次迭代前，创建新节点 → 获取 cur_path
    2. 将 cur_path 注入 system prompt，调用 LLM 获取下一步动作
    3. 将 cur_path 注入 arguments，执行工具
    4. 将结果写入 cur_path/result.json + 特殊文件
    5. 更新 trace 节点
    """
    
    def __init__(
        self,
        task_id: str,
        model_level: str = None,
        config: Dict = None,
        base_dir: Optional[str] = None
    ):
        """
        初始化 ReActAgent
        
        Args:
            task_id: 任务 ID
            model_level: 模型级别（如 "standard", "fast", "complex"）
            config: 配置信息
            base_dir: 基础目录（用于 trace 持久化）
        """
        super().__init__(
            context={"agent_name": self._get_agent_name()},
            config=config
        )
        
        self.task_id = task_id
        self.model_level = model_level
        self.base_dir = base_dir or str(Path.home() / ".akg")
        self.trace = TraceSystem(task_id=task_id, base_dir=self.base_dir)
        
        # 提前初始化 trace（创建基本结构，支持恢复）
        self.trace.initialize(force=False)
        self.current_node_id = self.trace.get_current_node()
        
        self.plan_list: List[Dict] = []
        self._initialized = False
        self._original_user_input: Optional[str] = None
        self._last_prompt: str = ""  # 最近一次 LLM 调用的完整 prompt（用于保存到节点目录）
        
        # 加载可用工具
        self.available_tools = self._load_available_tools()
        
        # 加载 agent registry
        self.agent_registry = self._load_agent_registry()
        
        # 加载 workflow registry（子类可覆盖）
        self.workflow_registry = self._load_workflow_registry()
        
        # 初始化工具执行器
        self.tool_executor = ToolExecutor(
            agent_registry=self.agent_registry,
            workflow_registry=self.workflow_registry,
            agent_context=self._get_agent_context(),
            history=[]
        )
        
        # 加载 prompt 模板
        self.system_prompt_template = self._load_prompt_template()
    
    # ==================== 抽象方法（子类必须实现） ====================
    
    @abstractmethod
    def _get_agent_name(self) -> str:
        """
        获取 Agent 名称
        
        Returns:
            Agent 名称字符串
        """
        pass
    
    @abstractmethod
    def _load_prompt_template(self) -> Jinja2TemplateWrapper:
        """
        加载 prompt 模板
        
        子类需要返回特定的 Jinja2 模板。
        
        Returns:
            Jinja2TemplateWrapper 实例
        """
        pass
    
    @abstractmethod
    def _build_prompt_context(self) -> Dict[str, Any]:
        """
        构建 prompt 上下文变量
        
        子类需要返回用于填充 prompt 模板的变量字典。
        
        Returns:
            包含模板变量的字典
        """
        pass
    
    @abstractmethod
    def _get_agent_context(self) -> Dict[str, Any]:
        """
        获取 agent 上下文
        
        子类需要返回传递给 ToolExecutor 的上下文信息。
        
        Returns:
            上下文字典
        """
        pass
    
    @abstractmethod
    def _load_available_tools(self) -> List[Dict]:
        """
        加载可用工具列表
        
        子类需要返回工具定义列表。
        
        Returns:
            工具定义列表
        """
        pass
    
    # ==================== 可覆盖方法（子类可选实现） ====================
    
    def _load_agent_registry(self) -> Dict[str, Any]:
        """
        加载 Agent 注册表
        
        子类可覆盖此方法以加载特定的 agents。
        默认实现加载所有注册的 agents。
        
        Returns:
            Agent 注册表字典
        """
        from akg_agents.core_v2.agents.registry import AgentRegistry
        
        # 尝试导入 plan agent
        try:
            from akg_agents.core_v2.agents import plan  # noqa: F401
        except Exception as e:
            logger.warning(f"[ReActAgent] 导入 plan 失败: {e}")
        
        agent_registry = {}
        all_agent_names = AgentRegistry.list_agents()
        logger.info(f"[ReActAgent] 发现 {len(all_agent_names)} 个已注册 agents")
        
        for agent_name in all_agent_names:
            try:
                agent_class = AgentRegistry.get_agent_class(agent_name)
                if not hasattr(agent_class, 'TOOL_NAME') or not agent_class.TOOL_NAME:
                    logger.debug(f"[ReActAgent] Agent '{agent_name}' 没有 TOOL_NAME，跳过")
                    continue
                
                # 加载 agent 的工具配置
                for tool_name, tool_def in agent_class.load_tool_config().items():
                    agent_registry[tool_name] = {
                        "agent_class": agent_class,
                        "config": tool_def
                    }
                    self.available_tools.append({
                        "type": "function",
                        "function": tool_def.get("function", {})
                    })
                    logger.info(f"[ReActAgent] 注册工具: {tool_name} (来自 {agent_name})")
            except Exception as e:
                logger.warning(f"[ReActAgent] 加载 Agent '{agent_name}' 失败: {e}", exc_info=True)
        
        return agent_registry
    
    def _load_workflow_registry(self) -> Dict[str, Any]:
        """
        加载 Workflow 注册表
        
        子类可覆盖此方法以加载特定的 workflows。
        默认返回空字典。
        
        Returns:
            Workflow 注册表字典
        """
        return {}
    
    def _get_task_info_extra(self) -> Dict[str, Any]:
        """
        获取任务信息的额外字段
        
        子类可覆盖此方法以添加特定的任务信息。
        
        Returns:
            额外字段字典
        """
        return {}
    
    def _on_plan_updated(self, result: Dict):
        """
        Plan 更新回调
        
        当 plan 工具执行成功后调用。子类可覆盖此方法以处理 plan 结果。
        
        Args:
            result: plan 工具的执行结果
        """
        pass
    
    # ==================== 核心 ReAct 循环 ====================
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        执行 ReAct 循环（节点优先流程）
        
        每次迭代:
        1. 创建新节点 → 获取 cur_path
        2. 将 cur_path 注入 prompt → 调用 LLM
        3. 解析 LLM 响应 → 获取 tool_name + arguments
        4. 注入 cur_path 到 arguments → 执行工具
        5. 保存 result.json → 更新节点
        
        Args:
            user_input: 用户输入
        
        Returns:
            执行结果字典
        """
        if not self._initialized:
            self._initialize_task(user_input)
            self._original_user_input = user_input
            self.tool_executor.agent_context["user_input"] = user_input
            self._initialized = True
        else:
            # 判断是对 ask_user 的响应，还是新任务
            current_node = self.trace.get_node(self.current_node_id)
            is_ask_user_response = (current_node.action and 
                                   current_node.action.get("type") == "ask_user")
            
            if is_ask_user_response:
                # 用户在回答问题
                self._handle_user_response(user_input)
            else:
                # 新任务：重置状态
                logger.info(f"[新任务] 重置状态，开始新任务")
                self._original_user_input = user_input
                self.tool_executor.agent_context["user_input"] = user_input
                self.plan_list = []
        
        iteration = 0
        while True:
            iteration += 1
            
            # ====== STEP 1: 获取当前节点的 cur_path（用于 prompt） ======
            cur_path = str(self.trace.fs.get_node_dir(self.current_node_id))
            
            # ====== STEP 2: LLM 调用（cur_path 注入 prompt） ======
            llm_response = await self._get_next_action(cur_path=cur_path)
            
            if not llm_response:
                return self._build_error_response("LLM 调用失败")
            
            tool_name = llm_response.get("tool_name")
            arguments = llm_response.get("arguments", {})
            reason = llm_response.get("reason", "")
            
            if tool_name == "finish":
                logger.info(f"[ReAct] 任务完成")
                return self._build_success_response()
            
            logger.info(f"[Reasoning] {tool_name} - {reason}")

            if tool_name == "ask_user":
                return self._handle_ask_user(arguments)
            
            # ====== STEP 3-5: 创建节点 → 执行工具 → 保存结果 ======
            result = await self._execute_tool(tool_name, arguments)

            if tool_name == "plan":
                if result.get("status") == "fail":
                    error_msg = result.get("error_information", "规划失败，请提供更多信息")
                    logger.info(f"[Plan 失败] {error_msg}")
                    return self._handle_ask_user({"message": error_msg})
                self._update_plan_from_result(result)
                self._on_plan_updated(result)

            full_history = self.trace.get_full_action_history(self.current_node_id)
            logger.info(f"[Observation] history: {len(full_history)} 条, 当前节点: {self.current_node_id}")
    
    # ==================== 工具执行（节点优先流程） ====================
    
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """
        执行工具（节点优先流程）
        
        流程:
        1. 创建新节点（状态=pending）→ 获取 cur_path
        2. 注入 cur_path 到 arguments
        3. 调用 tool_executor 执行（内部解析 read_json_file 表达式）
        4. 保存 result.json 到 cur_path
        5. 更新 trace 节点的 result
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数（可能包含 read_json_file 表达式）
        
        Returns:
            执行结果
        """
        # ====== 1. 创建节点 BEFORE 执行 ======
        action = {
            "type": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat()
        }
        # 使用 placeholder result 创建节点
        new_node_id = self.trace.add_node(
            action=action,
            result={"status": "pending"},
            metrics={}
        )
        self.current_node_id = new_node_id
        
        # ====== 2. 获取 cur_path 并注入 ======
        cur_path = str(self.trace.fs.get_node_dir(new_node_id))
        arguments["cur_path"] = cur_path
        
        logger.info(f"[Execute] 节点 {new_node_id}, cur_path: {cur_path}")
        
        # ====== 2.5 保存 LLM prompt 到节点目录（用于调试和审计） ======
        self._save_node_prompt(cur_path)
        
        # ====== 3. 同步硬件配置 ======
        # 如果 LLM 在 arguments 中指定了硬件参数，更新 agent_context
        # 以确保后续工具调用使用最新配置
        self._sync_hardware_config(arguments)
        
        # ====== 4. 执行工具 ======
        start_time = time.time()
        result = await self.tool_executor.execute(tool_name, arguments)
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"[Acting] {tool_name}: {result.get('status')} ({duration_ms}ms)")
        
        # ====== 5. 保存 result.json 到 cur_path ======
        self._save_node_result(cur_path, result)
        
        # ====== 6. 更新 trace 节点 ======
        self.trace.update_node_result(
            new_node_id, result, {"duration_ms": duration_ms}
        )
        
        # 记录到 tool executor history
        action_record = ActionRecord(
            action_id=new_node_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result
        )
        self.tool_executor.history.append(action_record)
        
        return result
    
    def _save_node_result(self, cur_path: str, result: Dict[str, Any]):
        """
        保存工具结果到节点目录
        
        保存内容:
        - result.json: 完整结果字典
        - code/code.py: 如果结果包含 generated_code 或 code
        - output.txt: 如果 output 较长（可能是代码等）
        
        Args:
            cur_path: 节点目录路径
            result: 工具执行结果
        """
        node_dir = Path(cur_path)
        
        # 保存 result.json
        result_file = node_dir / "result.json"
        try:
            result_file.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8"
            )
            logger.info(f"[Result] 已保存: {result_file}")
        except Exception as e:
            logger.warning(f"[Result] 保存 result.json 失败: {e}")
        
        # 保存代码文件
        code_content = result.get("generated_code") or result.get("code") or ""
        if code_content and isinstance(code_content, str) and len(code_content) > 10:
            code_dir = node_dir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            try:
                (code_dir / "code.py").write_text(code_content, encoding="utf-8")
                logger.info(f"[Result] 代码已保存: {code_dir / 'code.py'}")
            except Exception as e:
                logger.warning(f"[Result] 保存代码文件失败: {e}")
        
        # 保存较长的 output
        output = result.get("output", "")
        if isinstance(output, str) and len(output) > 200:
            try:
                (node_dir / "output.txt").write_text(output, encoding="utf-8")
            except Exception as e:
                logger.warning(f"[Result] 保存 output.txt 失败: {e}")
    
    def _save_node_prompt(self, cur_path: str):
        """
        保存 LLM 的完整 prompt 到节点目录
        
        便于调试和审计每一步 LLM 的输入。
        
        Args:
            cur_path: 节点目录路径
        """
        if not self._last_prompt:
            return
        
        try:
            prompt_file = Path(cur_path) / "prompt.txt"
            prompt_file.write_text(self._last_prompt, encoding="utf-8")
            logger.debug(f"[Prompt] 已保存: {prompt_file}")
        except Exception as e:
            logger.warning(f"[Prompt] 保存 prompt.txt 失败: {e}")
    
    # ==================== LLM 调用 ====================
    
    def _get_context_window(self) -> int:
        """获取当前模型的上下文窗口大小（从 settings 读取）"""
        try:
            from akg_agents.core_v2.config import get_settings
            settings = get_settings()
            return settings.context_window
        except Exception:
            return 128000  # 默认 128k
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        估算文本的 token 数量
        
        使用 tiktoken 精确计算，降级为字符数 / 4 估算。
        """
        if not text:
            return 0
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            return len(text) // 4
        except Exception:
            return len(text) // 4
    
    def _build_action_node_map(self) -> Dict[str, str]:
        """
        构建 action_id → node_id 的映射
        
        因为 action_id ("action_001") 和 node_id ("node_001") 是不同的序列，
        需要遍历路径上的节点来建立映射关系。
        
        Returns:
            {action_id: node_id} 映射字典
        """
        mapping = {}
        try:
            path = self.trace.get_path_to_node(self.current_node_id)
            for node_id in path:
                history = self.trace.fs.load_action_history_fact(node_id)
                for action in history.actions:
                    mapping[action.action_id] = node_id
        except Exception as e:
            logger.warning(f"[PathRegistry] 构建 action→node 映射失败: {e}")
        return mapping
    
    def _build_path_registry(self) -> str:
        """
        构建路径注册表 —— 列出所有历史节点的 result_path 和 result_keys
        
        直接遍历路径上的节点（而非依赖 action_id），确保即使历史被压缩，
        路径注册表始终完整保留，LLM 能正确构建 read_json_file 表达式。
        
        Returns:
            格式化的路径注册表字符串
        """
        try:
            path = self.trace.get_path_to_node(self.current_node_id)
        except Exception as e:
            logger.warning(f"[PathRegistry] 获取路径失败: {e}")
            return ""
        
        lines = []
        for node_id in path:
            if node_id == "root":
                continue
            
            # 加载该节点的动作历史
            try:
                history = self.trace.fs.load_action_history_fact(node_id)
            except Exception:
                continue
            
            for record in history.actions:
                if record.tool_name in ("history_summary", "user_request"):
                    continue
                
                result_path = str(
                    self.trace.fs.get_node_dir(node_id) / "result.json"
                )
                result_keys = list(record.result.keys()) if isinstance(record.result, dict) else []
                status = record.result.get("status", "unknown") if isinstance(record.result, dict) else "unknown"
                
                keys_str = ", ".join(result_keys[:8])
                if len(result_keys) > 8:
                    keys_str += ", ..."
                
                lines.append(
                    f"- **{node_id}** (`{record.tool_name}`, {status}): "
                    f"`{result_path}` → [{keys_str}]"
                )
        
        if not lines:
            return ""
        
        return "\n".join(lines)
    
    async def _get_next_action(self, cur_path: str = "") -> Optional[Dict]:
        """
        调用 LLM 获取下一个动作（智能压缩 + 路径注册表）
        
        策略:
        1. 先用完整历史构建 prompt，估算 token 数
        2. 如果 < 80% 上下文窗口，直接使用完整历史
        3. 如果 >= 80%，使用压缩历史
        4. 始终附加路径注册表（path_registry），确保压缩后路径不丢失
        5. 保存完整 prompt 到 self._last_prompt 以便持久化
        
        Args:
            cur_path: 当前节点路径（注入到 prompt 中）
        
        Returns:
            LLM 响应字典，包含 tool_name, arguments, reason
        """
        context_window = self._get_context_window()
        compress_threshold = int(context_window * 0.8)  # 80% 阈值
        
        # 获取完整历史
        full_history = self.trace.get_full_action_history(self.current_node_id)
        
        # 构建 action_id → node_id 映射（让 _format_action_record 能正确解析路径）
        action_node_map = self._build_action_node_map()
        
        # 构建路径注册表（始终完整，不受压缩影响）
        path_registry = self._build_path_registry()
        
        # 格式化完整历史
        full_action_history = json.dumps(
            [self._format_action_record(r, action_node_map) for r in full_history],
            indent=2,
            ensure_ascii=False
        ) if full_history else ""
        
        # 构建 prompt 上下文（先用完整历史估算）
        prompt_context = self._build_prompt_context()
        prompt_context.update({
            "available_tools": json.dumps(
                [t["function"] for t in self.available_tools],
                indent=2,
                ensure_ascii=False
            ),
            "user_input": self._original_user_input or "",
            "plan_list": json.dumps(self.plan_list, indent=2, ensure_ascii=False) if self.plan_list else "",
            "action_history": full_action_history,
            "path_registry": path_registry,
            "cur_path": cur_path,
            "task_base_path": str(self.trace.fs.task_dir),
        })
        
        prompt = self.system_prompt_template.format(**prompt_context)
        estimated_tokens = self._estimate_tokens(prompt)
        
        logger.info(
            f"[Prompt] 完整历史: {len(full_history)} 条, "
            f"预估 tokens: {estimated_tokens}, "
            f"阈值: {compress_threshold} (80% of {context_window})"
        )
        
        # 判断是否需要压缩
        if estimated_tokens >= compress_threshold and len(full_history) > 5:
            logger.info(f"[Prompt] 超过 80% 上下文窗口，启动历史压缩")
            try:
                llm_client = create_llm_client(model_level=self.model_level or "standard")
                compressed_history = await self.trace.get_compressed_history_for_llm(
                    llm_client=llm_client,
                    node_id=self.current_node_id,
                    max_tokens=2000
                )
                
                # 用压缩历史重建 prompt
                compressed_action_history = json.dumps(
                    [self._format_action_record(r, action_node_map) for r in compressed_history],
                    indent=2,
                    ensure_ascii=False
                ) if compressed_history else ""
                
                prompt_context["action_history"] = compressed_action_history
                prompt = self.system_prompt_template.format(**prompt_context)
                
                new_tokens = self._estimate_tokens(prompt)
                logger.info(
                    f"[Prompt] 压缩后: {len(compressed_history)} 条, "
                    f"tokens: {estimated_tokens} → {new_tokens}"
                )
            except Exception as e:
                logger.warning(f"[压缩历史失败] {e}，使用完整历史")
        else:
            logger.info(f"[Prompt] 未超过阈值，使用完整历史（无压缩）")
        
        # 保存完整 prompt（用于后续持久化到节点目录）
        self._last_prompt = prompt
        
        template = Jinja2TemplateWrapper("{{ prompt }}")
        
        try:
            response, _, _ = await self.run_llm(
                template,
                {"prompt": prompt},
                self.model_level or "standard"
            )

            # 使用 PlanAgent 的嵌套 JSON 解析器
            from akg_agents.core_v2.agents.plan import PlanAgent
            json_str = PlanAgent._extract_nested_json(response)
            
            # 检查是否成功提取 JSON（包括空白字符串）
            if not json_str or not json_str.strip():
                logger.error(f"[LLM 解析失败] 无法提取 JSON")
                logger.error(f"LLM 原始响应: {response[:500]}")
                return {
                    "tool_name": "ask_user",
                    "arguments": {"message": "系统错误，LLM 响应格式错误，请重试"},
                    "reason": "LLM 响应解析失败"
                }
            
            llm_response = json.loads(json_str)
            logger.info(f"[LLM] {llm_response.get('tool_name')} - {llm_response.get('reason', '')[:50]}...")
            
            return llm_response
        
        except json.JSONDecodeError as e:
            logger.error(f"[LLM 解析失败] {e}")
            logger.error(f"提取的 JSON 字符串: {json_str[:200] if json_str else 'None'}")
            logger.error(f"LLM 原始响应: {response[:500]}")
            return {
                "tool_name": "ask_user",
                "arguments": {"message": "系统错误，LLM 响应格式错误，请重试"},
                "reason": "LLM 响应解析失败"
            }
        except Exception as e:
            logger.error(f"[LLM 调用失败] {e}")
            return None
    
    # ==================== 用户交互 ====================
    
    def _handle_user_response(self, user_input: str):
        """
        处理用户响应
        
        记录用户响应到最后一个 ask_user 节点。
        
        Args:
            user_input: 用户输入
        """
        node = self.trace.get_node(self.current_node_id)
        
        # 检查当前节点是否是 ask_user
        if node.action and node.action.get("type") == "ask_user":
            # 更新 trace tree 中的节点
            node.result["status"] = "responded"
            node.result["user_response"] = user_input
            self.trace._save_trace()
            
            # 更新文件系统中的动作历史
            history_fact = self.trace.fs.load_action_history_fact(self.current_node_id)
            if history_fact.actions:
                history_fact.actions[0].result["status"] = "responded"
                history_fact.actions[0].result["user_response"] = user_input
                self.trace.fs.save_action_history_fact(self.current_node_id, history_fact)
            
            logger.info(f"[用户响应] {user_input[:50]}...")
        else:
            logger.warning(f"[用户响应] 当前节点不是 ask_user: {self.current_node_id}")
    
    def _handle_ask_user(self, arguments: Dict) -> Dict[str, Any]:
        """
        处理 ask_user 请求
        
        暂停执行，等待用户响应。
        
        Args:
            arguments: ask_user 参数
        
        Returns:
            等待用户响应的结果
        """
        message = arguments.get("message", "请提供更多信息")
        
        # 添加 ask_user 节点到 trace
        action = {
            "type": "ask_user",
            "arguments": arguments,
            "timestamp": datetime.now().isoformat()
        }
        result = {"status": "waiting", "message": message}
        
        new_node_id = self.trace.add_node(
            action=action,
            result=result,
            metrics={"duration_ms": 0}
        )
        self.current_node_id = new_node_id
        
        logger.info(f"[等待用户] {message[:100]}...")
        
        # 获取完整历史用于返回
        full_history = self.trace.get_full_action_history(self.current_node_id)
        anm = self._build_action_node_map()
        
        return {
            "status": "waiting_for_user",
            "message": message,
            "output": f"等待用户响应: {message}",
            "plan_list": self.plan_list,
            "history": [self._format_action_record(r, anm) for r in full_history]
        }
    
    # ==================== 任务初始化 ====================
    
    def _initialize_task(self, user_input: str):
        """
        初始化任务
        
        创建根节点或恢复现有任务。
        
        Args:
            user_input: 用户输入
        """
        if self.current_node_id == "root" and not self.trace.fs.node_exists("root"):
            # 获取额外的任务信息
            extra_info = self._get_task_info_extra()
            
            root_state = NodeState(
                node_id="root",
                turn=0,
                status="init",
                agent_info=AgentInfo(
                    agent_name=self._get_agent_name(),
                    agent_id="root"
                ).to_dict(),
                task_info=TaskInfo(
                    task_id=self.task_id,
                    task_input=user_input,
                    op_name=extra_info.get("op_name", ""),
                    dsl=extra_info.get("dsl", ""),
                    backend=extra_info.get("backend", ""),
                    arch=extra_info.get("arch", "")
                ).to_dict(),
                execution_info=ExecutionInfo(
                    tool_call_counter=0,
                    first_thinking_done=False,
                    current_turn=0
                ).to_dict()
            )
            self.trace.fs.save_node_state("root", root_state)
            logger.info(f"[{self._get_agent_name()}] 创建新任务: {self.task_id}")
        else:
            logger.info(f"[{self._get_agent_name()}] 恢复任务: {self.task_id}, 当前节点: {self.current_node_id}")
    
    # ==================== Plan 处理 ====================
    
    def _update_plan_from_result(self, result: Dict):
        """
        从 plan 工具结果中提取 plan_list
        
        处理两种格式:
        - dict: {"steps": [...]}
        - str: '{"steps": [...]}'
        
        Args:
            result: plan 工具的执行结果
        """
        try:
            output = result.get("output", {})
            
            # 统一转换为 dict
            plan_data = output if isinstance(output, dict) else json.loads(output) if output else {}
            
            # 提取 steps
            if "steps" in plan_data:
                self.plan_list = plan_data["steps"]
                logger.info(f"[Plan 更新] {len(self.plan_list)} 个步骤")
            else:
                logger.warning(f"[Plan 更新] output 中没有 steps 字段")
                
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[Plan 更新失败] {e}")
    
    # ==================== 响应构建 ====================
    
    def _build_response(self, status: str, output: str = "", error_msg: str = "") -> Dict[str, Any]:
        """
        构建统一的响应格式
        
        Args:
            status: 状态 (success/error)
            output: 输出信息（成功时使用）
            error_msg: 错误信息（失败时使用）
        
        Returns:
            响应字典
        """
        full_history = self.trace.get_full_action_history(self.current_node_id)
        anm = self._build_action_node_map()
        return {
            "status": status,
            "output": output,
            "error_information": error_msg,
            "plan_list": self.plan_list,
            "history": [self._format_action_record(r, anm) for r in full_history],
            "total_actions": len(full_history),
            "current_node": self.current_node_id
        }
    
    def _build_success_response(self) -> Dict[str, Any]:
        """构建成功响应"""
        return self._build_response("success", output="所有任务已完成")
    
    def _build_error_response(self, error_msg: str) -> Dict[str, Any]:
        """构建错误响应"""
        return self._build_response("error", error_msg=error_msg)
    
    # ==================== 配置同步 ====================
    
    # 需要同步到 agent_context 的硬件参数 key
    _HARDWARE_CONFIG_KEYS = {"framework", "backend", "arch", "dsl"}
    
    def _sync_hardware_config(self, arguments: Dict[str, Any]):
        """
        将 LLM arguments 中的硬件参数同步到 agent_context
        
        当 LLM 在调用工具时指定了 backend/dsl/arch/framework，
        说明这是用户期望的配置。同步到 agent_context 确保后续调用使用最新值。
        
        子类可覆盖 _on_hardware_config_updated() 来同步到自身属性。
        
        Args:
            arguments: LLM 生成的工具参数
        """
        updated = {}
        for key in self._HARDWARE_CONFIG_KEYS:
            value = arguments.get(key)
            if value and isinstance(value, str):
                old_value = self.tool_executor.agent_context.get(key)
                if value != old_value:
                    self.tool_executor.agent_context[key] = value
                    updated[key] = value
        
        if updated:
            logger.info(f"[ConfigSync] 硬件配置已更新: {updated}")
            self._on_hardware_config_updated(updated)
    
    def _on_hardware_config_updated(self, updated: Dict[str, str]):
        """
        硬件配置更新回调（子类可覆盖）
        
        当 agent_context 中的硬件参数被更新时调用。
        子类应在此方法中同步到自身属性和 prompt 上下文。
        
        Args:
            updated: 已更新的参数字典 {"backend": "ascend", ...}
        """
        pass
    
    # ==================== 工具方法 ====================
    
    def _format_action_record(self, record: ActionRecord, action_node_map: Dict[str, str] = None) -> Dict:
        """
        格式化 ActionRecord 为字典
        
        包含 result_path 和 result_keys 用于 LLM 数据引用:
        - result_path: 结果文件路径（供 read_json_file 使用）
        - result_keys: 结果中的可用 key 列表
        
        Args:
            record: ActionRecord 实例
            action_node_map: action_id → node_id 映射（用于解析文件路径）
        
        Returns:
            格式化后的字典
        """
        # 特殊处理 history_summary
        if record.tool_name == "history_summary":
            return {
                "action_id": record.action_id,
                "tool_name": "history_summary",
                "summary": record.result.get("summary", ""),
                "original_actions": record.arguments.get("original_actions", 0),
                "compressed": True
            }
        
        formatted = {
            "action_id": record.action_id,
            "tool_name": record.tool_name,
            "arguments": record.arguments,
            "result": record.result,
            "duration_ms": record.duration_ms
        }
        
        # 解析 node_id：优先从映射获取，其次检查 action_id 本身
        node_id = None
        if action_node_map and record.action_id in action_node_map:
            node_id = action_node_map[record.action_id]
        elif record.action_id and record.action_id.startswith("node_"):
            # 兼容 tool_executor.history 中 action_id == node_id 的情况
            node_id = record.action_id
        
        # 添加 result_path（供 LLM 构建 read_json_file 表达式）
        if node_id:
            try:
                result_path = str(
                    self.trace.fs.get_node_dir(node_id) / "result.json"
                )
                formatted["result_path"] = result_path
                formatted["node_id"] = node_id
                
                # 添加 result 中的可用 key 列表
                if isinstance(record.result, dict):
                    formatted["result_keys"] = list(record.result.keys())
            except Exception:
                pass
        
        return formatted
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """
        获取 trace 摘要信息（用于调试）
        
        Returns:
            trace 摘要字典
        """
        try:
            full_history = self.trace.get_full_action_history(self.current_node_id)
            path = self.trace.get_path_to_node(self.current_node_id)
            leaf_nodes = self.trace.get_all_leaf_nodes()
            
            return {
                "task_id": self.task_id,
                "current_node": self.current_node_id,
                "path_length": len(path),
                "total_actions": len(full_history),
                "path": path,
                "leaf_nodes": leaf_nodes
            }
        except Exception as e:
            # 如果 trace 未初始化或出错，返回基本信息
            logger.warning(f"[Trace Summary] 无法获取完整摘要: {e}")
            return {
                "task_id": self.task_id,
                "current_node": self.current_node_id,
                "path_length": 0,
                "total_actions": 0,
                "path": [],
                "leaf_nodes": []
            }
