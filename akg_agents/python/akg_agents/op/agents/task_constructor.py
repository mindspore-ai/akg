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
TaskConstructor: 从代码仓中提取算子实现，构建 KernelBench 格式的标准化任务文件。

作为 v2 sub-agent 使用，内部运行完整的 ReAct 循环并调用专用工具集。
注册方式:
  - @register_agent(scopes=["op"]) → 自动作为 call_task_constructor 工具可用
  - 由 KernelAgent._load_agent_registry() 导入触发注册

返回值设计:
  - output: 给外部 main agent 的紧凑摘要（不含完整代码，不含内部 history）
  - task_code: 完整代码（扩展字段）
  - task_code_path: 代码文件路径（扩展字段，供后续 agent 使用）
"""

import json
import re
import time
import shutil
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from akg_agents.core_v2.agents import AgentBase, register_agent

logger = logging.getLogger(__name__)

# 导入统一工具注册表 + 触发工具注册（模块加载时自动注册所有工具）
from akg_agents.core_v2.tools.tool_registry import ToolRegistry
import akg_agents.core_v2.tools.basic_tools                    # noqa: F401 - 触发注册
from akg_agents.op.tools.task_constructor import code_tools   # noqa: F401 - 触发注册
from akg_agents.op.tools.task_constructor import file_tools    # noqa: F401 - 触发注册


# ==================== ReAct 循环辅助 ====================

@dataclass
class StepRecord:
    """单步执行记录"""
    step: int
    thought: str
    action: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    raw_response: str = ""
    timestamp: float = field(default_factory=time.time)


class RobustJsonParser:
    """
    多策略 JSON 解析器。
    LLM 经常输出格式不完美的 JSON，需要多种 fallback 策略。
    """

    @staticmethod
    def parse(text: str) -> Optional[Dict]:
        text = text.strip()

        # 策略 1: 直接解析
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "action" in data:
                return RobustJsonParser._normalize(data)
        except json.JSONDecodeError:
            pass

        # 策略 2: 提取 ```json 代码块
        json_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        for block in json_blocks:
            try:
                data = json.loads(block.strip())
                if isinstance(data, dict) and "action" in data:
                    return RobustJsonParser._normalize(data)
            except json.JSONDecodeError:
                continue

        # 策略 3: 平衡大括号提取
        balanced = RobustJsonParser._extract_balanced_braces(text)
        if balanced:
            try:
                data = json.loads(balanced)
                if isinstance(data, dict) and "action" in data:
                    return RobustJsonParser._normalize(data)
            except json.JSONDecodeError:
                pass

        # 策略 4: 修复常见错误
        fixed = RobustJsonParser._fix_common_errors(text)
        if fixed:
            try:
                data = json.loads(fixed)
                if isinstance(data, dict) and "action" in data:
                    return RobustJsonParser._normalize(data)
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _normalize(data: Dict) -> Dict:
        return {
            "thought": data.get("thought", ""),
            "action": data.get("action", ""),
            "arguments": data.get("arguments", data.get("args", {})),
        }

    @staticmethod
    def _extract_balanced_braces(text: str) -> Optional[str]:
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    @staticmethod
    def _fix_common_errors(text: str) -> Optional[str]:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end + 1]
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        candidate = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', candidate)
        return candidate

    @staticmethod
    def is_truncated(text: str) -> bool:
        return (
            len(text) > 5000
            and '"content"' in text[:500]
            and text.count('{') > text.count('}')
        )


class SessionLogger:
    """会话日志管理"""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "session.log"
        self.messages_file = self.log_dir / "messages.jsonl"
        self._last_messages = None
        self._write_log(f"=== Session {datetime.now().isoformat()} ===\n")

    def log_step(self, step: int, thought: str, action: str,
                 args: Dict, result: Dict, raw_response: str = ""):
        entry = {
            "step": step, "timestamp": time.time(),
            "thought": thought, "action": action,
            "arguments": self._safe_truncate(args),
            "result_status": result.get("status", ""),
            "result_error": result.get("error_information", ""),
        }
        self._write_jsonl(entry)

    def log_llm_call(self, step: int, msg_count: int, response: str,
                     messages: List[Dict[str, str]] = None):
        self._write_log(
            f"\n--- LLM Call (step={step}, msgs={msg_count}) ---\n"
            f"Response ({len(response)} chars):\n{response[:2000]}\n"
        )
        # 缓存最新的 messages，仅在 log_final 时写入 prompt_final.json
        if messages:
            self._last_messages = messages

    def log_system_prompt(self, prompt: str):
        (self.log_dir / "system_prompt.txt").write_text(prompt, encoding="utf-8")

    def log_initial_message(self, msg: str):
        (self.log_dir / "initial_message.txt").write_text(msg, encoding="utf-8")

    def log_final(self, result: Dict, messages: List[Dict[str, str]] = None):
        result_copy = {k: v for k, v in result.items() if k != "messages"}
        (self.log_dir / "result.json").write_text(
            json.dumps(result_copy, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.get("task_code"):
            (self.log_dir / "task_output.py").write_text(result["task_code"], encoding="utf-8")
        # 保存最终一步的完整 messages（供排查 prompt 问题）
        final_msgs = messages or getattr(self, '_last_messages', None)
        if final_msgs:
            try:
                (self.log_dir / "prompt_final.json").write_text(
                    json.dumps(final_msgs, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            except Exception:
                pass

    def _write_log(self, text: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)

    def _write_jsonl(self, entry: Dict):
        with open(self.messages_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_truncate(d: Dict, max_len=500) -> Dict:
        result = {}
        for k, v in d.items():
            if isinstance(v, str) and len(v) > max_len:
                result[k] = v[:max_len] + f"...(+{len(v) - max_len} chars)"
            else:
                result[k] = v
        return result


# ==================== Agent 定义 ====================

@register_agent(scopes=["op"])
class TaskConstructor(AgentBase):
    """
    从代码仓中提取算子实现并构建 KernelBench 格式标准化任务文件的 Agent。

    内部运行完整的 ReAct 循环，使用专用工具集（AST 分析、任务装配、格式验证等）。
    支持从文件、目录、代码片段输入构建任务。
    """

    TOOL_NAME = "call_task_constructor"
    DESCRIPTION = """
从代码仓中提取算子实现，构建为 KernelBench 格式的单文件自包含标准化任务。

功能：
- 从 PyTorch/Triton 代码仓中定位目标算子
- AST 依赖追踪自动发现所有依赖函数
- 智能选择构建策略（排除式/选择性/完整嵌入）
- 自动清理 import、移除装饰器
- 格式验证（结构 + 运行时 + NaN/Inf 检查）
- 参考对比测试（多组输入验证正确性）

适用场景：
- 用户提供代码仓路径，需要提取某个算子构建 task_code
- 用户指定 torch 内部函数（如 torch._chunk_cat），需要提取其分解实现
- 用户提供代码片段，需要包装成标准化任务格式

输出：KernelBench 格式的 task_code（单文件自包含 Python）
"""

    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "user_input": {
                "type": "string",
                "description": "用户需求描述"
            },
            "source_path": {
                "type": "string",
                "description": "可选：源代码路径（文件或目录）。如果用户提供了代码仓路径则传入。",
                "default": ""
            },
        },
        "required": ["user_input"],
    }

    # ReAct 循环配置
    MAX_STEPS = 50
    MAX_RETRIES_PER_STEP = 3
    MAX_MESSAGES = 50

    def __init__(self):
        context = {
            "agent_name": "task_constructor",
            "task_label": "main",
        }
        super().__init__(context=context)

        # 加载 system prompt 模板
        self.system_prompt_template = self.load_template("task_constructor/system_prompt.j2")

        # 运行时状态（每次 run() 重置）
        self.messages: List[Dict[str, str]] = []
        self.history: List[StepRecord] = []
        self._llm_client = None

        # 目录（每次 run() 初始化）
        self.output_dir: Optional[Path] = None
        self.workspace_dir: Optional[Path] = None
        self.session_log: Optional[SessionLogger] = None

    def _get_llm_client(self, model_level: str = "standard"):
        """获取 LLM 客户端"""
        if self._llm_client is None:
            from akg_agents.core_v2.llm.factory import create_llm_client
            self._llm_client = create_llm_client(model_level=model_level)
        return self._llm_client

    def _init_workspace(self, cur_path: str = ""):
        """初始化工作目录"""
        if cur_path:
            base = Path(cur_path)
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = Path.home() / ".akg" / "task_constructor" / run_id

        self.output_dir = base.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = self.output_dir / "workspace"
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        log_dir = self.output_dir / "logs"
        self.session_log = SessionLogger(log_dir)

    def _build_system_prompt(self) -> str:
        """构建 system prompt（注入工具描述）"""
        tools_desc = ToolRegistry.get_tools_for_prompt(
            scope="task_constructor", format="markdown"
        )
        # 添加 finish 伪工具（特殊处理，不在 ToolRegistry 中注册）
        tools_desc += (
            "\n\n### finish\n"
            "任务完成时调用。\n参数:\n"
            "    - task_code: 任务文件路径或代码 (必填)\n"
            "    - summary: 任务摘要 (可选)\n"
            "    - error: 错误信息（失败时） (可选)"
        )
        return self.system_prompt_template.format(tools_description=tools_desc)

    def _build_initial_message(self, user_input: str, source_path: str = "") -> str:
        """构建初始用户消息"""
        parts = ["请帮我将以下内容构建为 KernelBench 格式的标准化算子任务。\n"]

        parts.append(f"**描述**: {user_input}\n")

        if source_path:
            # 标准化路径：统一使用正斜杠，确保 LLM 输出的路径跨平台兼容
            normalized_path = source_path.replace('\\', '/')
            p = Path(source_path)
            if p.is_file():
                parts.append(f"**输入类型**: 文件 (`{normalized_path}`)")
                parts.append("请先用 `copy_to_workspace` 复制此文件，然后 `read_function` 提取目标函数。")
            elif p.is_dir():
                parts.append(f"**输入类型**: 代码仓 (`{normalized_path}`)")
                parts.append("请直接用 `grep_search` 在代码仓中搜索目标函数，不要先 scan_dir。")
            else:
                parts.append(f"**路径**: `{normalized_path}`")
            parts.append(f"\n**重要**: 使用 grep_search 返回的精确路径来调用 copy_to_workspace，不要自行猜测或修改路径。")

        parts.append(
            f"\n**工作区路径**: workspace/（所有 workspace/ 前缀的路径会自动解析到正确的工作区目录）\n"
            f"\n**高效流程**（目标: 20步内完成）:\n"
            f"1. `grep_search` 在源码中搜索目标函数定义\n"
            f"2. `copy_to_workspace` 使用 grep 返回的精确路径复制源文件\n"
            f"3. `read_function` 提取目标函数\n"
            f"4. `trace_dependencies` 自动追踪全部依赖 [关键]\n"
            f"5. 处理外部依赖（`read_function` 从来源提取 → 内联到 helper_code）\n"
            f"6. `assemble_task` 构建任务文件\n"
            f"7. `optimize_task` 清理代码\n"
            f"8. `test_with_reference` 验证+对比测试（先 execute_script 验证原始函数环境可用）\n"
            f"9. `finish`\n"
            f"\n**注意**:\n"
            f"- 不要用 grep_search 搜 workspace 文件内容（直接 read_function 提取更高效）\n"
            f"- 路径会自动解析: workspace/xxx → 工作区, 相对路径 → 输出目录或工作区\n"
            f"\n请开始工作。"
        )

        return "\n".join(parts)

    # ==================== ReAct 循环 ====================

    async def _call_llm_async(self, messages: List[Dict[str, str]], model_level: str) -> str:
        """异步调用 LLM"""
        client = self._get_llm_client(model_level)
        result = await client.generate(messages)
        return result.get("content", "")

    async def _get_next_action(self, step: int, model_level: str) -> Tuple[Optional[Dict], str]:
        """调用 LLM 获取下一步动作"""
        for attempt in range(self.MAX_RETRIES_PER_STEP + 1):
            try:
                response = await self._call_llm_async(self.messages, model_level)
                if self.session_log:
                    self.session_log.log_llm_call(
                        step, len(self.messages), response, messages=self.messages
                    )

                if not response or not response.strip():
                    logger.warning(f"LLM 返回空响应 (attempt {attempt + 1})")
                    continue

                action = RobustJsonParser.parse(response)
                if action:
                    return action, response

                is_truncated = RobustJsonParser.is_truncated(response)
                logger.warning(f"解析失败 (attempt {attempt + 1})"
                               f"{'，JSON被截断' if is_truncated else ''}")

                if is_truncated:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "你的 JSON 太长被截断了！每次代码不能超过 150 行！\n"
                            "用 assemble_task 引用源文件，不要把代码写在 JSON 里。"
                        )
                    })
                else:
                    short = response[:1000] + ("..." if len(response) > 1000 else "")
                    self.messages.append({"role": "assistant", "content": short})
                    self.messages.append({
                        "role": "user",
                        "content": (
                            'JSON 解析失败。请严格输出纯 JSON：\n'
                            '{"thought": "思考", "action": "工具名", "arguments": {参数}}'
                        )
                    })

            except Exception as e:
                logger.error(f"LLM 调用异常 (attempt {attempt + 1}): {e}")
                if attempt < self.MAX_RETRIES_PER_STEP:
                    await asyncio.sleep(2)

        return None, ""

    WRITE_TOOLS = {"write_file", "edit_file"}

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行内部工具（路径预解析 + 注入 workspace/output 路径）"""
        exec_args = dict(arguments)
        exec_args = self._resolve_tool_paths(tool_name, exec_args)
        # 始终注入 workspace/output 路径；ToolRegistry._filter_func_args
        # 会自动过滤不接受这些参数的函数（如 basic_tools）
        exec_args.setdefault("workspace_dir", str(self.workspace_dir))
        exec_args.setdefault("output_dir", str(self.output_dir))
        return ToolRegistry.execute(tool_name, exec_args)

    def _resolve_tool_paths(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """将 workspace/xxx、相对路径解析为绝对路径（区分读写操作）"""
        from akg_agents.op.tools.task_constructor.path_utils import resolve_path as _tc_resolve
        PATH_KEYS = ("file_path", "path", "script_path")
        for key in PATH_KEYS:
            val = arguments.get(key)
            if not val or not isinstance(val, str):
                continue
            p = Path(val)
            if p.is_absolute():
                continue
            fwd = val.replace('\\', '/')
            if fwd.startswith("workspace/"):
                arguments[key] = str((self.workspace_dir / fwd[len("workspace/"):]).resolve())
                continue
            if tool_name in self.WRITE_TOOLS:
                arguments[key] = str((self.output_dir / p).resolve())
                continue
            for base in [self.workspace_dir, self.output_dir]:
                candidate = base / p
                if candidate.exists():
                    arguments[key] = str(candidate.resolve())
                    break
            else:
                resolved = _tc_resolve(val, workspace_dir=self.workspace_dir, output_dir=self.output_dir)
                arguments[key] = str(resolved)
        return arguments

    def _manage_history(self):
        """管理消息历史长度"""
        if len(self.messages) <= self.MAX_MESSAGES:
            return

        system = self.messages[0]
        initial_user = self.messages[1]
        removed = self.messages[2:-self.MAX_MESSAGES + 4]

        summary_parts = []
        workspace_refs = set()

        for msg in removed:
            content = msg.get("content", "")
            for line in content.splitlines():
                if "workspace/" in line and ("已保存" in line or "已复制" in line):
                    workspace_refs.add(line.strip())
            if msg["role"] == "assistant":
                try:
                    d = json.loads(content)
                    summary_parts.append(f"[{d.get('action', '')}] {d.get('thought', '')[:80]}")
                except (json.JSONDecodeError, AttributeError):
                    pass

        compressed = "[操作历史摘要]\n" + "\n".join(summary_parts[-15:])
        if workspace_refs:
            compressed += "\n\n[工作区文件]:\n" + "\n".join(sorted(workspace_refs))

        recent = self.messages[-self.MAX_MESSAGES + 4:]
        self.messages = [system, initial_user, {"role": "user", "content": compressed}] + recent

    def _resolve_task_code(self, task_code: str) -> str:
        """如果 task_code 是文件路径引用，读取文件内容"""
        if not task_code or not task_code.strip():
            return task_code
        text = task_code.strip()
        if text.endswith(".py") and len(text) < 500 and "\n" not in text:
            candidates = [
                self.workspace_dir / text,
                self.workspace_dir / Path(text).name,
                self.output_dir / text,
                Path(text),
            ]
            for p in candidates:
                p = p.expanduser().resolve()
                if p.exists() and p.is_file():
                    return p.read_text(encoding="utf-8")
        return task_code

    @staticmethod
    def _truncate(text: str, max_len: int = 4000) -> str:
        if len(text) <= max_len:
            return text
        half = max_len // 2
        return text[:half] + f"\n\n... (截断，原始 {len(text)} 字符) ...\n\n" + text[-half:]

    # ==================== 主入口 ====================

    async def run(
        self,
        user_input: str = "",
        source_path: str = "",
        op_name: str = "",
        model_level: str = "standard",
        cur_path: str = "",
        non_interactive: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行完整的标准化任务构建流程。

        内部运行 ReAct 循环，自动完成代码提取、依赖追踪、任务装配和验证。

        Args:
            user_input: 用户需求描述
            source_path: 源代码路径（文件或目录）
            op_name: 算子名称
            model_level: LLM 模型级别
            cur_path: 输出目录（可选，由 ToolExecutor 注入）
            non_interactive: 非交互模式（作为子 agent 被调用时设为 True，
                            ask_user 将自动回复让 LLM 自主决策，不阻塞调用链）

        Returns:
            标准结果字典 {status, output, error_information, task_code, summary}
        """
        if not user_input:
            return {
                "status": "error",
                "output": "",
                "error_information": "user_input 不能为空",
            }

        # 自动检测: 如果是被 ToolExecutor 调用的子 agent，自动启用非交互模式
        if not non_interactive and ('history_compress' in kwargs or 'task_id' in kwargs):
            non_interactive = True
            logger.info("[TaskConstructor] 检测到 ToolExecutor 调用，启用非交互模式")

        # 0. 初始化
        self._init_workspace(cur_path)
        self.messages = []
        self.history = []
        self._llm_client = None

        logger.info(f"[TaskConstructor] 开始任务: {user_input[:100]}...")
        logger.info(f"[TaskConstructor] 工作目录: {self.output_dir}")

        # 1. 构建 system prompt
        system_prompt = self._build_system_prompt()
        self.messages.append({"role": "system", "content": system_prompt})
        if self.session_log:
            self.session_log.log_system_prompt(system_prompt)

        # 2. 构建初始用户消息
        initial_msg = self._build_initial_message(user_input, source_path)
        self.messages.append({"role": "user", "content": initial_msg})
        if self.session_log:
            self.session_log.log_initial_message(initial_msg)

        # 3. ReAct 循环
        for step in range(1, self.MAX_STEPS + 1):
            logger.info(f"[TaskConstructor] Step {step}/{self.MAX_STEPS}")

            # 3.1 调用 LLM
            action, raw_response = await self._get_next_action(step, model_level)
            if action is None:
                result = self._build_result("error",
                                            error="LLM 多次返回无效格式，终止")
                if self.session_log:
                    self.session_log.log_final(result, messages=self.messages)
                return result

            thought = action.get("thought", "")
            act_name = action.get("action", "")
            act_args = action.get("arguments", {})

            logger.info(f"  思考: {thought[:100]}...")
            logger.info(f"  动作: {act_name}")

            # 3.2 检查 finish
            if act_name == "finish":
                task_code = act_args.get("task_code", "")
                error = act_args.get("error", "")
                summary = act_args.get("summary", "")
                # op_name 可能由 LLM 在 finish 中给出，或者由用户传入
                final_op_name = act_args.get("op_name", op_name)

                if error:
                    result = self._build_result("error", error=error, op_name=final_op_name)
                    if self.session_log:
                        self.session_log.log_final(result, messages=self.messages)
                    return result

                task_code = self._resolve_task_code(task_code)

                result = self._build_result(
                    "success", task_code=task_code,
                    summary=summary, op_name=final_op_name,
                )
                if self.session_log:
                    self.session_log.log_final(result, messages=self.messages)
                return result

            # 3.3 检查 ask_user → 根据 non_interactive 模式决定行为
            if act_name == "ask_user":
                message = act_args.get("message", "需要更多信息")
                logger.info(f"  [ask_user] {message}")

                # 追加 assistant 消息
                self.messages.append({
                    "role": "assistant",
                    "content": json.dumps(action, ensure_ascii=False),
                })

                if non_interactive:
                    # 非交互模式（被 KernelAgent 等外部 agent 调用时）
                    # 自动回复，不阻塞调用链
                    user_reply = "请根据已有信息自行做出合理决策，不要再询问。"
                    logger.info(f"  [ask_user] 非交互模式，自动回复: {user_reply}")
                else:
                    # 交互模式（独立运行/测试时）
                    try:
                        print(f"\n{'='*60}")
                        print(f"Agent 询问: {message}")
                        print(f"{'='*60}")
                        user_reply = input("请回复: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        user_reply = "请自行做出合理决策，不要再询问。"

                self.messages.append({
                    "role": "user",
                    "content": f"用户回复: {user_reply}",
                })

                if self.session_log:
                    self.session_log.log_step(
                        step, thought, "ask_user", act_args,
                        {"status": "success", "output": user_reply, "error_information": ""},
                    )
                self._manage_history()
                continue  # 继续 ReAct 循环

            # 3.4 执行工具
            tool_names = ToolRegistry.list_names(scope="task_constructor")
            if act_name not in tool_names:
                tool_result = {
                    "status": "error", "output": "",
                    "error_information": f"未知工具: {act_name}，可用: {tool_names}",
                }
            else:
                tool_result = self._execute_tool(act_name, act_args)

            # 3.5 记录
            record = StepRecord(
                step=step, thought=thought, action=act_name,
                arguments=act_args, result=tool_result, raw_response=raw_response,
            )
            self.history.append(record)
            if self.session_log:
                self.session_log.log_step(step, thought, act_name, act_args, tool_result)

            # 3.6 追加到消息
            self.messages.append({
                "role": "assistant",
                "content": json.dumps(action, ensure_ascii=False),
            })

            raw_output = tool_result.get("output", "")
            tool_result_msg = f"工具 [{act_name}] 执行结果:\n状态: {tool_result['status']}\n"

            if raw_output:
                ws_refs = [l for l in raw_output.splitlines()
                           if "workspace" in l.lower() or "已保存" in l or "已复制" in l]
                truncated_output = self._truncate(raw_output)
                tool_result_msg += f"输出:\n{truncated_output}\n"
                if ws_refs:
                    ws_info = "\n".join(ws_refs)
                    if ws_info not in truncated_output:
                        tool_result_msg += f"\n[工作区引用]:\n{ws_info}\n"

            error_info = tool_result.get("error_information", "")
            if error_info:
                tool_result_msg += f"错误: {error_info}\n"

            self.messages.append({"role": "user", "content": tool_result_msg})

            # 3.7 消息管理
            self._manage_history()

        # 超过最大步数
        result = self._build_result("error", error=f"超过最大步数 {self.MAX_STEPS}", op_name=op_name)
        if self.session_log:
            self.session_log.log_final(result, messages=self.messages)
        return result

    def _build_result(
        self,
        status: str,
        task_code: str = "",
        summary: str = "",
        error: str = "",
        op_name: str = "",
    ) -> Dict[str, Any]:
        """
        构建标准返回结果（兼容 v2 ToolExecutor 归一化）。

        返回设计原则:
        - output: 给外部 main agent 的紧凑摘要（不含完整代码）
        - task_code: 完整代码内容（扩展字段）
        - task_code_path: 代码文件保存路径（扩展字段，供后续 agent 使用）
        - 不暴露内部 ReAct history 细节
        """
        task_code_path = ""

        # 成功时: 保存 task_code 到文件，返回路径
        if status == "success" and task_code:
            filename = f"{op_name}_task.py" if op_name else "task.py"
            save_path = self.output_dir / filename
            try:
                save_path.write_text(task_code, encoding="utf-8")
                task_code_path = str(save_path)
                logger.info(f"[TaskConstructor] 任务代码已保存: {task_code_path}")
            except Exception as e:
                logger.warning(f"[TaskConstructor] 保存任务代码失败: {e}")

        # 构建紧凑摘要给外部 agent
        if status == "success":
            code_lines = task_code.count("\n") + 1 if task_code else 0
            output_summary = (
                f"标准化任务构建成功。\n"
                f"算子: {op_name or '(未命名)'}\n"
                f"代码: {code_lines} 行\n"
                f"文件: {task_code_path}\n"
                f"步骤: {len(self.history)} 步\n"
            )
            if summary:
                output_summary += f"摘要: {summary}\n"
        else:
            output_summary = f"标准化任务构建失败: {error}"

        return {
            # 标准 v2 字段
            "status": "success" if status == "success" else "fail",
            "output": output_summary,
            "error_information": error if status != "success" else "",
            # 扩展字段 - 供后续 agent（如 KernelGen）使用
            "task_code": task_code,
            "task_code_path": task_code_path,
            "op_name": op_name,
            "summary": summary,
            "steps": len(self.history),
            "log_dir": str(self.session_log.log_dir) if self.session_log else "",
        }
