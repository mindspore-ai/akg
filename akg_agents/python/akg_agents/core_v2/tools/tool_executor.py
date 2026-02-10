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

import inspect
import io
import logging
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

from akg_agents.core_v2.tools.arg_resolver import resolve_arguments

logger = logging.getLogger(__name__)


class _ErrorCapture:
    """工具执行期间只捕获关键报错信息（WARNING+ 级别日志 + stderr）
    
    设计原则:
    - 只捕获 WARNING/ERROR/CRITICAL 级别的日志，不捕获 DEBUG/INFO
    - 只捕获 stderr（不捕获 stdout，stdout 通常是正常输出，太多噪音）
    - 截断到 max_chars 防止污染上下文
    - 完整日志保存到 cur_path 供事后调试，result 中只注入精简摘要
    """
    
    MAX_CHARS = 800  # 注入到 result 的最大字符数
    
    def __init__(self):
        self._stderr_buf = io.StringIO()
        self._error_records: List[logging.LogRecord] = []
        self._log_handler: Optional[logging.Handler] = None
        self._old_stderr = None
    
    def __enter__(self):
        # 只捕获 stderr（tee 模式，不影响原始输出）
        self._old_stderr = sys.stderr
        sys.stderr = _TeeStream(self._old_stderr, self._stderr_buf)
        
        # 只捕获 WARNING+ 级别的日志
        self._log_handler = _ListHandler(self._error_records, logging.WARNING)
        logging.getLogger().addHandler(self._log_handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._old_stderr
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
        return False
    
    @property
    def stderr(self) -> str:
        return self._stderr_buf.getvalue()
    
    @property
    def error_summary(self) -> str:
        """只包含 WARNING+ 的精简日志摘要"""
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        lines = []
        for record in self._error_records:
            lines.append(formatter.format(record))
        return "\n".join(lines)
    
    @property
    def has_errors(self) -> bool:
        return bool(self._error_records) or bool(self.stderr.strip())
    
    def get_concise_errors(self) -> str:
        """获取精简的错误信息（用于注入 result，有截断）"""
        parts = []
        stderr = self.stderr.strip()
        errors = self.error_summary.strip()
        if errors:
            parts.append(errors)
        if stderr:
            parts.append(f"[stderr] {stderr}")
        
        combined = "\n".join(parts)
        if len(combined) > self.MAX_CHARS:
            combined = combined[:self.MAX_CHARS] + f"\n... (truncated, total {len(combined)} chars)"
        return combined
    
    def save_full_log(self, cur_path: str):
        """保存完整捕获内容到 cur_path/captured_errors.log（供事后调试）"""
        if not cur_path or not self.has_errors:
            return
        try:
            log_file = Path(cur_path) / "captured_errors.log"
            with open(log_file, "w", encoding="utf-8") as f:
                stderr = self.stderr.strip()
                errors = self.error_summary.strip()
                if errors:
                    f.write("=== WARNING/ERROR Logs ===\n")
                    f.write(errors + "\n\n")
                if stderr:
                    f.write("=== Stderr ===\n")
                    f.write(stderr + "\n")
        except Exception:
            pass


class _TeeStream:
    """同时写入两个流的 tee 包装器"""
    
    def __init__(self, original, capture_buf: io.StringIO):
        self._original = original
        self._capture = capture_buf
    
    def write(self, text):
        self._original.write(text)
        self._capture.write(text)
        return len(text)
    
    def flush(self):
        self._original.flush()
    
    def __getattr__(self, name):
        return getattr(self._original, name)


class _ListHandler(logging.Handler):
    """将 LogRecord 收集到列表中的 Handler"""
    
    def __init__(self, record_list: list, level: int = logging.WARNING):
        super().__init__(level)
        self._records = record_list
    
    def emit(self, record: logging.LogRecord):
        self._records.append(record)


class ToolExecutor:
    def __init__(self, agent_registry: Dict[str, Any] = None,
                 workflow_registry: Dict[str, Any] = None,
                 agent_context: Dict[str, Any] = None,
                 history: List = None):
        self.agent_registry = agent_registry or {}
        self.workflow_registry = workflow_registry or {}
        self.agent_context = agent_context or {}
        self.history = history or []
        self.tool_types = self._load_tool_types()
    
    def _get_hardware_params(self, arguments: Dict[str, Any]) -> Dict[str, str]:
        """统一获取硬件参数（从 arguments 或 agent_context）
        
        优先级: arguments > agent_context > 默认值
        """
        return {
            "framework": arguments.get("framework", self.agent_context.get("framework", "torch")),
            "backend": arguments.get("backend", self.agent_context.get("backend", "cuda")),
            "arch": arguments.get("arch", self.agent_context.get("arch", "a100")),
            "dsl": arguments.get("dsl", self.agent_context.get("dsl", "triton"))
        }
    
    def _build_history_compress(self, max_items: int = 10) -> List[Dict]:
        """构建压缩的历史记录"""
        if not self.history:
            return []
        return [
            {"tool_name": r.tool_name, "arguments": r.arguments, "result": r.result}
            for r in self.history[-max_items:]
        ]
    
    def _load_tool_types(self) -> Dict[str, str]:
        """加载工具类型映射（从统一的 tools.yaml）"""
        try:
            from akg_agents import get_project_root
            tools_file = Path(get_project_root()) / "core_v2" / "config" / "tools.yaml"
            with open(tools_file, "r", encoding="utf-8") as f:
                tools_config = yaml.safe_load(f)
            
            tool_types = {}
            for tool_name, tool_def in tools_config.get("tools", {}).items():
                tool_types[tool_name] = tool_def.get("type", "basic_tool")
            return tool_types
        except Exception as e:
            logger.warning(f"[ToolExecutor] 加载工具类型失败: {e}")
            return {}
    
    # ==================== 主入口 ====================
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工具（统一入口）
        
        流程:
        1. 解析参数中的动态表达式 (read_json_file 等)
        2. 根据工具类型分派执行
        3. 只捕获关键报错（WARNING+ 日志 + stderr），避免污染上下文
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数（可能包含 read_json_file 表达式）
        
        Returns:
            标准结果字典 {"status", "output", "error_information", ...}
        """
        # 解析参数中的动态表达式
        try:
            resolved_args = resolve_arguments(arguments)
        except Exception as e:
            logger.warning(f"[ToolExecutor] 参数表达式解析失败: {e}，使用原始参数")
            resolved_args = arguments
        
        # 带错误捕获执行（只捕获 WARNING+ 和 stderr）
        with _ErrorCapture() as capture:
            # 优先检查是否是注册的 Agent
            if tool_name in self.agent_registry:
                result = await self._execute_agent(tool_name, resolved_args)
            # 检查是否是注册的 Workflow
            elif tool_name in self.workflow_registry:
                result = await self._execute_workflow(tool_name, resolved_args)
            else:
                # 检查工具类型
                tool_type = self.tool_types.get(tool_name, "basic_tool")
                if tool_type == "domain_tool":
                    result = await self._execute_domain_tool(tool_name, resolved_args)
                else:
                    result = await self._execute_basic_tool(tool_name, resolved_args)
        
        # 保存完整错误日志到 cur_path（供事后调试，不注入 result）
        cur_path = resolved_args.get("cur_path", "")
        capture.save_full_log(cur_path)
        
        # 只在工具失败时，注入精简的错误信息到 result
        if isinstance(result, dict) and result.get("status") in ("error", "fail"):
            if capture.has_errors:
                concise = capture.get_concise_errors()
                # 如果工具自身没有提供 error_information，用捕获的错误补充
                if not result.get("error_information"):
                    result["error_information"] = concise
                else:
                    # 追加到已有的错误信息后面（但有长度限制）
                    existing = result["error_information"]
                    if len(existing) < 500:
                        result["error_information"] = f"{existing}\n---\n{concise}"
        
        return result
    
    # ==================== Agent 执行（通用） ====================
    
    async def _execute_agent(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        通用 Agent 执行
        
        通过 introspection 自动匹配 agent.run() 的参数签名，
        不再需要为每个 agent 写 if/else 分支。
        
        流程:
        1. 合并默认参数（硬件参数、task_id、history_compress 等）
        2. 创建 agent 实例
        3. 通过 introspection 调用 agent.run()
        4. 归一化返回结果
        """
        try:
            agent_info = self.agent_registry[tool_name]
            agent_class = agent_info["agent_class"]
            
            # 合并默认参数（arguments 中已有的优先）
            hw_params = self._get_hardware_params(arguments)
            for k, v in hw_params.items():
                arguments.setdefault(k, v)
            
            arguments.setdefault('task_id', self.agent_context.get('task_id', ''))
            arguments.setdefault('history_compress', self._build_history_compress())
            arguments.setdefault('user_input', self.agent_context.get('user_input', ''))
            arguments.setdefault('model_level', self.agent_context.get('model_level', 'standard'))
            
            # 创建 agent 实例
            agent = self._create_agent_instance(agent_class)
            
            # 验证 agent 有 run 方法
            if not hasattr(agent, 'run') or not callable(getattr(agent, 'run', None)):
                return {
                    "status": "error", "output": "",
                    "error_information": f"Agent {agent_class.__name__} 没有可调用的 run() 方法"
                }
            
            # 调用 agent.run()（自动匹配参数签名）
            logger.info(f"[ToolExecutor] 执行 Agent: {tool_name} ({agent_class.__name__})")
            result = await self._call_agent_run(agent, arguments)
            
            # 归一化返回结果
            return self._normalize_agent_result(result)
        
        except Exception as e:
            logger.error(f"[ToolExecutor] Agent 执行失败: {tool_name}, {e}", exc_info=True)
            return {"status": "error", "output": "", "error_information": f"Agent 执行失败: {str(e)}"}
    
    def _create_agent_instance(self, agent_class):
        """
        通用 Agent 实例创建
        
        尝试多种实例化模式:
        1. 无参构造 agent_class()
        2. parser_config_path=None
        3. config=dict
        """
        # 尝试无参构造
        try:
            return agent_class()
        except TypeError:
            pass
        
        # 尝试 parser_config_path=None（KernelGen, KernelDesigner 等）
        try:
            return agent_class(parser_config_path=None)
        except TypeError:
            pass
        
        # 尝试 config 参数
        try:
            return agent_class(config=self.agent_context.get("config", {}))
        except TypeError as e:
            raise RuntimeError(
                f"无法创建 Agent 实例 ({agent_class.__name__}): {e}。"
                f"请确认 agent 支持以下构造方式之一: "
                f"无参 / parser_config_path=None / config=dict"
            )
    
    async def _call_agent_run(self, agent, arguments: Dict[str, Any]):
        """
        通过 introspection 调用 agent.run()，自动匹配参数签名
        
        支持三种模式:
        1. run(state): 单个 state/dict 参数 → 将 arguments 整体作为 state 传入
        2. run(**kwargs): 接受任意关键字参数 → 传入所有 arguments
        3. run(param1, param2, ...): 具名参数 → 过滤出匹配的参数
        
        注意: 对于模式 1，如果 LLM 生成了嵌套的 state 结构（如 {"state": {"user_input": ...}}），
        会自动解包并合并到顶层，确保 agent.run() 收到扁平化的参数。
        """
        sig = inspect.signature(agent.run)
        params = {name: p for name, p in sig.parameters.items() if name != 'self'}
        
        # 模式 1: 单个 state/dict 参数（如 OpTaskBuilder.run(state)）
        param_names = list(params.keys())
        if len(param_names) == 1:
            first_param = params[param_names[0]]
            # 如果是位置参数且名称暗示是 dict/state
            if first_param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            ):
                param_name = param_names[0]
                logger.debug(f"[ToolExecutor] 使用 state 模式调用: {param_name}")
                
                # 兼容处理: 如果 LLM 生成了嵌套结构 {"state": {...}, ...}
                # 将 arguments["state"] 的内容解包到顶层，再合并其他顶层参数
                state_arg = arguments
                if param_name in arguments and isinstance(arguments[param_name], dict):
                    nested = arguments[param_name]
                    # 将嵌套内容提升到顶层，其他顶层参数作为补充（不覆盖嵌套中的值）
                    state_arg = dict(nested)
                    for k, v in arguments.items():
                        if k != param_name:
                            state_arg.setdefault(k, v)
                    logger.debug(f"[ToolExecutor] 解包嵌套 '{param_name}' 参数，keys: {list(state_arg.keys())}")
                
                if inspect.iscoroutinefunction(agent.run):
                    return await agent.run(state_arg)
                return agent.run(state_arg)
        
        # 模式 2: 接受 **kwargs
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        
        if has_var_keyword:
            filtered_args = arguments
        else:
            # 模式 3: 过滤到方法接受的参数
            accepted = set(params.keys())
            filtered_args = {k: v for k, v in arguments.items() if k in accepted}
            
            # 记录被过滤掉的参数（调试用）
            dropped = set(arguments.keys()) - accepted
            if dropped:
                logger.debug(f"[ToolExecutor] 过滤掉未接受的参数: {dropped}")
        
        if inspect.iscoroutinefunction(agent.run):
            return await agent.run(**filtered_args)
        return agent.run(**filtered_args)
    
    def _normalize_agent_result(self, result) -> Dict[str, Any]:
        """
        归一化不同 Agent 的返回格式为标准字典
        
        支持:
        - dict with "status" → 直接返回（已经是标准格式）
        - dict with "result" → Plan Agent 格式: {"result": {"status": ...}, "arguments": {...}}
        - tuple of 3 → (output, full_prompt, reasoning) 格式（KernelGen 等）
        - 其他 → 转为字符串
        """
        # 已经是标准 dict 格式
        if isinstance(result, dict):
            if "status" in result:
                return result
            
            # Plan Agent 格式: {"result": {"status": ...}, "arguments": {...}}
            if "result" in result:
                plan_result = result["result"]
                if isinstance(plan_result, dict):
                    if plan_result.get("status") == "success":
                        return {
                            "status": "success",
                            "output": result.get("arguments", {}),
                            "error_information": ""
                        }
                    else:
                        return {
                            "status": "fail",
                            "output": "",
                            "error_information": plan_result.get("desc", "规划失败")
                        }
                return {"status": "error", "output": "", "error_information": "返回格式错误"}
            
            # 普通 dict（没有 status 也没有 result）
            return {"status": "success", "output": result, "error_information": ""}
        
        # Tuple 格式: (output, full_prompt, reasoning)
        if isinstance(result, tuple):
            if len(result) == 3:
                first, full_prompt, reasoning = result
                # 第一个元素是 dict → 递归归一化
                if isinstance(first, dict):
                    normalized = self._normalize_agent_result(first)
                    normalized["full_prompt"] = full_prompt
                    normalized["reasoning"] = reasoning
                    return normalized
                # 第一个元素是 string → 生成代码格式
                return {
                    "status": "success",
                    "output": str(first),
                    "error_information": "",
                    "generated_code": str(first),
                    "full_prompt": full_prompt,
                    "reasoning": reasoning
                }
            if len(result) == 2:
                first, second = result
                return {
                    "status": "success",
                    "output": str(first),
                    "error_information": "",
                    "extra": second
                }
        
        # 兜底: 转字符串
        return {"status": "success", "output": str(result), "error_information": ""}
    
    # ==================== Workflow 执行 ====================
    
    async def _execute_workflow(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 Workflow
        
        Args:
            tool_name: 工具名称（如 "use_coder_only_workflow"）
            arguments: 工具参数（已解析表达式）
        
        Returns:
            执行结果字典
        """
        try:
            workflow_info = self.workflow_registry[tool_name]
            workflow_class = workflow_info["workflow_class"]
            workflow_name = workflow_info["workflow_name"]
            
            logger.info(f"[ToolExecutor] 开始执行 workflow: {workflow_name}")
            
            # 获取 workflow 资源（通过回调）
            get_resources = self.agent_context.get("get_workflow_resources")
            if not get_resources:
                raise ValueError("无法获取 workflow 资源，请检查 KernelAgent 配置")
            
            workflow_resources = get_resources()
            
            # 检查必需资源
            if not workflow_resources.get("agents"):
                raise ValueError("Workflow 需要 agents 资源，但未提供")
            
            logger.info(f"[ToolExecutor] Workflow 资源准备完成，agents: {list(workflow_resources['agents'].keys())}")
            
            # 让 workflow 类确保所需资源就位（如注册 worker 等）
            if hasattr(workflow_class, 'ensure_resources'):
                await workflow_class.ensure_resources(workflow_resources, arguments)
            
            # 让 workflow 类预处理配置（如 cur_path 重定向 log_dir）
            if hasattr(workflow_class, 'prepare_config'):
                workflow_class.prepare_config(workflow_resources, arguments)
            
            # 创建 workflow 实例
            workflow = workflow_class(**workflow_resources)
            
            # 编译 workflow
            app = workflow.compile()
            logger.info(f"[ToolExecutor] Workflow {workflow_name} 编译完成")
            
            # 构建初始状态（包含 cur_path）
            initial_state = self._build_workflow_state(arguments)
            logger.info(f"[ToolExecutor] 开始执行 workflow，初始状态: op_name={initial_state.get('op_name')}, dsl={initial_state.get('dsl')}")
            
            # 执行 workflow
            final_state = await app.ainvoke(initial_state)
            
            # 格式化结果：使用 workflow 自身的 format_result
            result = workflow.format_result(final_state)
            
            logger.info(f"[ToolExecutor] Workflow {workflow_name} 执行完成: {result.get('status')}")
            
            return result
        
        except Exception as e:
            logger.error(f"[ToolExecutor] Workflow 执行失败: {tool_name}, {e}", exc_info=True)
            return {
                "status": "fail",
                "error_information": f"Workflow 执行失败: {str(e)}",
                "output": ""
            }
    
    def _build_workflow_state(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建 workflow 初始状态
        
        Args:
            arguments: 工具调用参数（已解析）
        
        Returns:
            workflow 初始状态字典
        """
        # 从 arguments 或 agent_context 获取参数
        return {
            "op_name": arguments.get("op_name", ""),
            "task_desc": arguments.get("task_desc", ""),
            "dsl": arguments.get("dsl", self.agent_context.get("dsl", "")),
            "framework": arguments.get("framework", self.agent_context.get("framework", "")),
            "backend": arguments.get("backend", self.agent_context.get("backend", "")),
            "arch": arguments.get("arch", self.agent_context.get("arch", "")),
            "task_id": arguments.get("task_id", self.agent_context.get("task_id", "")),
            "user_requirements": arguments.get("user_requirements", ""),
            "cur_path": arguments.get("cur_path", ""),
            "result": {},
            "should_continue": True,
            "current_step": "",
            "iterations": 0,
            "max_iterations": arguments.get("max_iterations", 10),
        }
    
    # ==================== Domain Tool / Basic Tool 执行 ====================
    
    async def _execute_domain_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行 domain_tool（如 verify_kernel, profile_kernel）"""
        try:
            from akg_agents.core_v2.tools import domain_tools
            
            tool_func = getattr(domain_tools, tool_name, None)
            if not tool_func:
                return {
                    "status": "error",
                    "output": "",
                    "error_information": f"未知领域工具: {tool_name}"
                }
            
            # 合并硬件默认参数（LLM 可能未显式传递 backend/arch 等）
            hw_params = self._get_hardware_params(arguments)
            for k, v in hw_params.items():
                arguments.setdefault(k, v)
            
            # 通过 introspection 过滤参数
            filtered_args = self._filter_func_args(tool_func, arguments)
            
            logger.info(f"[ToolExecutor] 执行 domain_tool: {tool_name}, hw_params: {hw_params}")
            
            # 执行工具函数（支持异步和同步）
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**filtered_args)
            else:
                result = tool_func(**filtered_args)
            
            if isinstance(result, dict) and "status" in result:
                return result
            
            return {"status": "success", "output": str(result), "error_information": ""}
        
        except Exception as e:
            logger.error(f"[ToolExecutor] Domain 工具执行失败: {tool_name}, {e}", exc_info=True)
            return {"status": "error", "output": "", "error_information": str(e)}
    
    async def _execute_basic_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行 basic_tool"""
        try:
            from akg_agents.core_v2.tools import basic_tools
            
            tool_func = getattr(basic_tools, tool_name, None)
            if not tool_func:
                return {"status": "error", "output": "", "error_information": f"未知工具: {tool_name}"}
            
            # 通过 introspection 过滤参数
            filtered_args = self._filter_func_args(tool_func, arguments)
            
            result = tool_func(**filtered_args)
            
            if isinstance(result, dict) and "status" in result:
                return result
            
            return {"status": "success", "output": str(result), "error_information": ""}
        
        except Exception as e:
            logger.error(f"[ToolExecutor] 工具执行失败: {tool_name}, {e}")
            return {"status": "error", "output": "", "error_information": str(e)}
    
    # ==================== 工具方法 ====================
    
    @staticmethod
    def _filter_func_args(func, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据函数签名过滤参数
        
        避免传入函数不接受的参数（如 cur_path 传给 read_file）
        
        Args:
            func: 目标函数
            arguments: 原始参数字典
        
        Returns:
            过滤后的参数字典
        """
        sig = inspect.signature(func)
        params = sig.parameters
        
        # 如果函数接受 **kwargs，传入所有参数
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if has_var_keyword:
            return arguments
        
        # 否则只传入函数接受的参数
        accepted = {name for name in params.keys() if name != 'self'}
        return {k: v for k, v in arguments.items() if k in accepted}
