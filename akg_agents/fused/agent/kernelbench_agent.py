"""
KernelBench 任务构建 Agent（融合版）

融合设计:
┌───────────────────────────────────────────────┐
│            KernelBenchAgent                   │
│                                               │
│  ┌─────────────┐  ┌────────────────────────┐  │
│  │ akg_agents  │  │ demo 增强              │  │
│  │ v2 基础架构 │  │ · 4 策略 JSON 解析     │  │
│  │ · 异步      │  │ · workspace 引用保护   │  │
│  │ · 配置系统  │  │ · 重试 + 引导         │  │
│  │ · Skill加载 │  │ · 截断检测            │  │
│  └──────┬──────┘  └──────────┬─────────────┘  │
│         │                    │                │
│         └────────┬───────────┘                │
│                  │                            │
│        ┌────────┴────────┐                    │
│        │ KernelBench     │                    │
│        │ Domain Tools    │                    │
│        │ (via adapter)   │                    │
│        └─────────────────┘                    │
└───────────────────────────────────────────────┘

核心来源:
- ReAct 循环骨架: demo/agent/react_loop.py (鲁棒性)
- 工具系统: demo/tools/ (领域工具) + akg_agents basic_tools (基础工具)
- Skill 系统: akg_agents core_v2/skill/ (知识管理)
- 配置系统: akg_agents core_v2/config/ (多级配置)
"""

import json
import re
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ============ 从 demo 迁移的鲁棒性组件 ============

class RobustJsonParser:
    """
    多策略 JSON 解析器（从 demo/react_loop.py 迁移）。

    这是 demo 的核心竞争力之一。LLM 经常输出格式不完美的 JSON，
    需要多种 fallback 策略来保证解析成功率。

    策略优先级:
    1. 直接 json.loads()
    2. 提取 ```json 代码块
    3. 提取最外层 {...} 大括号
    4. 修复常见 JSON 错误（尾逗号/单引号/截断）
    """

    @staticmethod
    def parse(text: str) -> Optional[Dict]:
        """尝试从 LLM 输出中解析 JSON。返回 None 表示全部失败。"""
        if not text or not text.strip():
            return None

        text = text.strip()

        # 策略 1: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略 2: 提取 ```json 代码块
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 策略 3: 提取最外层 {...}
        brace_match = RobustJsonParser._extract_balanced_braces(text)
        if brace_match:
            try:
                return json.loads(brace_match)
            except json.JSONDecodeError:
                pass

        # 策略 4: 修复常见错误
        candidates = [text]
        if brace_match:
            candidates.append(brace_match)

        for candidate in candidates:
            fixed = RobustJsonParser._fix_common_errors(candidate)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _extract_balanced_braces(text: str) -> Optional[str]:
        """提取第一个平衡的 {...} 块"""
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
        """修复常见 JSON 格式错误"""
        if not text.strip().startswith('{'):
            return None
        fixed = text
        # 尾逗号
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        # 单引号 → 双引号（简单替换，不处理嵌套）
        if "'" in fixed and '"' not in fixed:
            fixed = fixed.replace("'", '"')
        return fixed

    @staticmethod
    def is_truncated(text: str) -> bool:
        """检测 JSON 是否被截断"""
        if len(text) > 5000:
            opens = text.count('{') + text.count('[')
            closes = text.count('}') + text.count(']')
            if opens > closes + 2:
                return True
        return False


class WorkspaceAwareHistoryManager:
    """
    工作区感知的历史管理器（从 demo/react_loop.py 迁移）。

    核心能力: 在压缩消息历史时保留 workspace 文件引用。
    这对代码生成任务至关重要——Agent 需要记住已提取代码的位置。
    """

    WORKSPACE_PATTERN = re.compile(r'workspace/\S+\.py')
    MAX_MESSAGES = 50

    @staticmethod
    def compress_history(
        messages: List[Dict[str, str]],
        max_messages: int = 50,
    ) -> List[Dict[str, str]]:
        """
        压缩消息历史，保留关键信息。

        策略:
        1. 始终保留 system prompt 和初始 user message
        2. 压缩中间消息为摘要（保留 workspace 引用）
        3. 保留最近 N 条消息
        """
        if len(messages) <= max_messages:
            return messages

        # 分段: [system, initial_user, ..., recent_N]
        preserved_head = messages[:2]  # system + initial user
        recent = messages[-max_messages + 3:]  # 保留近期
        middle = messages[2:-max_messages + 3]

        if not middle:
            return messages

        # 从中间消息中提取 workspace 引用
        ws_refs = set()
        operations = []
        for msg in middle:
            content = msg.get("content", "")
            ws_refs.update(WorkspaceAwareHistoryManager.WORKSPACE_PATTERN.findall(content))
            # 提取工具操作摘要
            if "工具" in content and "执行结果" in content:
                tool_name = re.search(r'\[(.+?)\]', content)
                if tool_name:
                    operations.append(tool_name.group(1))

        # 构建压缩摘要
        summary_parts = [f"(之前 {len(middle)} 条消息已压缩)"]
        if ws_refs:
            summary_parts.append(f"已提取的 workspace 文件: {', '.join(sorted(ws_refs))}")
        if operations:
            summary_parts.append(f"已执行的操作: {', '.join(operations)}")

        summary_msg = {
            "role": "user",
            "content": "\n".join(summary_parts)
        }

        return preserved_head + [summary_msg] + recent


@dataclass
class StepRecord:
    """单步执行记录"""
    step: int
    thought: str = ""
    action: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class KernelBenchAgent:
    """
    KernelBench 任务构建 Agent（融合版）。

    融合了:
    - demo 的鲁棒性: 多策略 JSON 解析、workspace 历史保护、重试引导
    - akg_agents 的架构: 异步支持、Skill 加载、统一配置

    使用方式:

    ```python
    # 方式 1: 独立使用（类似 demo/main.py）
    agent = KernelBenchAgent()
    result = agent.run("path/to/repo", "提取 softmax 算子")

    # 方式 2: 作为 akg_agents v2 子 Agent
    # 在 sub_agent_registry.py 中注册
    ```
    """

    def __init__(
        self,
        model_level: str = None,
        max_steps: int = 50,
        max_retries: int = 3,
        verbose: bool = True,
        workspace_dir: str = None,
    ):
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose
        self.history: List[StepRecord] = []
        self.messages: List[Dict[str, str]] = []

        # 初始化工具集
        from fused.tools.adapter import create_kernelbench_tools
        self.toolkit = create_kernelbench_tools(workspace_dir=workspace_dir)

        # 初始化 LLM 客户端（复用 akg_agents 配置）
        self._model_level = model_level
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            from demo.config import get_llm_client
            self._llm_client = get_llm_client(self._model_level)
        return self._llm_client

    def run(self, user_input: str, description: str = "") -> Dict[str, Any]:
        """
        同步执行入口。

        Args:
            user_input: 用户输入（代码/文件路径/目录路径）
            description: 任务描述

        Returns:
            {status, task_code, summary, error, steps, history}
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已有事件循环（如 jupyter），创建新线程
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run,
                        self.arun(user_input, description)
                    ).result()
            else:
                return asyncio.run(self.arun(user_input, description))
        except RuntimeError:
            return asyncio.run(self.arun(user_input, description))

    async def arun(self, user_input: str, description: str = "") -> Dict[str, Any]:
        """
        异步执行入口（融合 demo 的循环骨架 + 异步支持）。
        """
        # 1. 初始化
        self.history = []
        self.messages = []

        # 2. 构建 prompt
        system_prompt = self._build_system_prompt()
        initial_message = self._build_initial_message(user_input, description)

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_message},
        ]

        # 3. ReAct 循环
        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  Step {step}/{self.max_steps}")
                print(f"{'='*60}")

            # 3a. 获取 LLM 动作（含重试）
            action_data = await self._get_next_action(step)
            if action_data is None:
                return self._make_result("error", error="LLM 多次调用失败")

            thought = action_data.get("thought", "")
            action = action_data.get("action", "")
            arguments = action_data.get("arguments", {})

            if self.verbose:
                print(f"  Thought: {thought[:100]}...")
                print(f"  Action: {action}")

            # 3b. 检查 finish
            if action == "finish":
                return self._make_result(
                    "success",
                    task_code=arguments.get("task_code", ""),
                    summary=arguments.get("summary", ""),
                )

            # 3c. 执行工具
            result = self.toolkit.execute(action, arguments)

            # 3d. 记录
            record = StepRecord(
                step=step, thought=thought, action=action,
                arguments=arguments, result=result,
            )
            self.history.append(record)

            # 3e. 将结果加入对话
            status = result.get("status", "error")
            output = result.get("output", "")
            error = result.get("error", "")

            # 截断过长输出
            if len(output) > 6000:
                half = 3000
                output = output[:half] + f"\n\n... (截断，共{len(output)}字符) ...\n\n" + output[-half:]

            result_text = f"工具 [{action}] 执行结果:\n状态: {status}\n"
            if output:
                result_text += f"输出:\n{output}\n"
            if error:
                result_text += f"错误:\n{error}\n"

            self.messages.append({"role": "user", "content": result_text})

            # 3f. 管理历史
            self.messages = WorkspaceAwareHistoryManager.compress_history(
                self.messages,
                max_messages=WorkspaceAwareHistoryManager.MAX_MESSAGES,
            )

        return self._make_result("error", error=f"达到最大步数 {self.max_steps}")

    async def _get_next_action(self, step: int) -> Optional[Dict]:
        """
        调用 LLM 获取下一步动作（融合 demo 的多策略解析 + 重试引导）。
        """
        for retry in range(self.max_retries):
            try:
                response = await self._call_llm()
                if not response:
                    continue

                # 多策略 JSON 解析
                parsed = RobustJsonParser.parse(response)
                if parsed:
                    # 记录原始响应
                    if self.history or not self.messages:
                        pass
                    return parsed

                # 解析失败 → 引导
                if RobustJsonParser.is_truncated(response):
                    guidance = (
                        "你的上一次回复被截断了（JSON 不完整）。"
                        "请减少 arguments 中代码的长度，或使用工具（如 write_file）代替内联代码。"
                        "请重新输出完整的 JSON。"
                    )
                else:
                    guidance = (
                        "你的回复不是有效的 JSON。请严格按格式输出：\n"
                        '{"thought": "思考", "action": "工具名", "arguments": {...}}'
                    )
                self.messages.append({"role": "user", "content": guidance})

            except Exception as e:
                logger.error(f"Step {step} retry {retry}: LLM 调用失败: {e}")
                if retry < self.max_retries - 1:
                    await asyncio.sleep(2)

        return None

    async def _call_llm(self) -> Optional[str]:
        """调用 LLM（支持 akg_agents 的异步客户端）"""
        try:
            client = self.llm_client
            # 尝试异步调用
            if hasattr(client, 'achat'):
                response = await client.achat(self.messages)
            elif hasattr(client, 'chat'):
                # 同步客户端包装为异步
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, client.chat, self.messages
                )
            else:
                # 兼容 OpenAI 风格客户端
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.chat.completions.create(
                        messages=self.messages,
                        model=getattr(client, 'model_name', 'default'),
                    )
                )
                return response.choices[0].message.content

            if isinstance(response, str):
                return response
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"LLM 调用异常: {e}")
            raise

    def _build_system_prompt(self) -> str:
        """构建系统提示词（集成 Skill 内容 + 工具描述）"""
        # 加载 Skill 内容
        skill_content = self._load_skill()

        # 工具描述
        tools_desc = self.toolkit.get_tools_for_prompt()

        return f"""\
你是一个算子任务构建专家。目标: 根据用户输入，提取代码并构建为标准 KernelBench 单文件自包含任务。

{skill_content}

## 可用工具

{tools_desc}

## 输出格式

每步响应必须是纯 JSON（不要有其他文字）：

{{"thought": "思考", "action": "工具名", "arguments": {{...}}}}

完成时:
{{"thought": "总结", "action": "finish", "arguments": {{"task_code": "task_output.py", "summary": "..."}}}}
"""

    def _load_skill(self) -> str:
        """加载 KernelBench Task Builder Skill"""
        skill_path = Path(__file__).parent.parent / "skill" / "kernelbench-task-builder" / "SKILL.md"
        if skill_path.exists():
            content = skill_path.read_text(encoding="utf-8")
            # 跳过 YAML frontmatter
            if content.startswith("---"):
                end = content.find("---", 3)
                if end != -1:
                    content = content[end + 3:].strip()
            return content
        # fallback: 使用内联的最小工作流
        return "## 工作流: 搜索代码 → 追踪依赖 → 装配任务 → 验证"

    def _build_initial_message(self, user_input: str, description: str) -> str:
        """构建初始用户消息"""
        parts = [f"用户输入: {user_input}"]
        if description:
            parts.append(f"描述: {description}")
        parts.append("\n请开始工作。")
        return "\n".join(parts)

    def _make_result(self, status: str, task_code: str = "",
                     summary: str = "", error: str = "") -> Dict[str, Any]:
        return {
            "status": status,
            "task_code": task_code,
            "summary": summary,
            "error": error,
            "steps": len(self.history),
            "history": [
                {
                    "step": r.step,
                    "action": r.action,
                    "thought": r.thought,
                    "result_status": r.result.get("status", ""),
                }
                for r in self.history
            ],
        }


# 导入 Path
from pathlib import Path
