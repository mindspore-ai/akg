"""
ReAct Agent 主循环

修复:
  - JSON 解析: 多策略解析 + 长代码场景特殊处理
  - 日志系统: 每次运行保存完整 prompt/响应日志
  - 输出路径: 写入固定 output 目录
  - 消息管理: 防止 history 过长导致 token 溢出
"""
import asyncio
import json
import re
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..config import (
    MAX_REACT_STEPS, MAX_RETRIES_PER_STEP, OUTPUT_DIR, LOG_DIR, WORKSPACE_DIR,
    get_llm_client, get_run_id,
)
from ..task.input_parser import InputParser, ParsedInput
from ..task.task_builder import TaskBuilder
from ..tools.registry import ToolRegistry

# 确保所有工具模块被导入（触发注册）
from ..tools import file_tools   # noqa: F401
from ..tools import code_tools   # noqa: F401
from ..tools import user_tools   # noqa: F401

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _run_async(coro):
    """在同步上下文中运行 async 函数"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


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


class SessionLogger:
    """会话日志管理 - 保存完整的 prompt/响应到文件"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.log_dir = LOG_DIR / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "session.log"
        self.messages_file = self.log_dir / "messages.jsonl"
        self._write_log(f"=== Session {run_id} ===\n")

    def log_step(self, step: int, thought: str, action: str,
                 args: Dict, result: Dict, raw_response: str = ""):
        """记录一步"""
        entry = {
            "step": step,
            "timestamp": time.time(),
            "thought": thought,
            "action": action,
            "arguments": self._safe_truncate_dict(args),
            "result_status": result.get("status", ""),
            "result_output_len": len(result.get("output", "")),
            "result_error": result.get("error", ""),
        }
        self._write_jsonl(entry)
        self._write_log(
            f"\n{'='*60} Step {step} {'='*60}\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Args: {json.dumps(self._safe_truncate_dict(args), ensure_ascii=False)[:500]}\n"
            f"Result: {result.get('status', '')} | output_len={len(result.get('output', ''))}\n"
        )

    def log_llm_call(self, step: int, messages_count: int, response: str):
        """记录 LLM 调用"""
        self._write_log(
            f"\n--- LLM Call (step={step}, messages={messages_count}) ---\n"
            f"Response ({len(response)} chars):\n{response[:2000]}\n"
            f"{'...(truncated)' if len(response) > 2000 else ''}\n"
        )

    def log_system_prompt(self, prompt: str):
        """保存 system prompt"""
        prompt_file = self.log_dir / "system_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

    def log_initial_message(self, msg: str):
        """保存初始用户消息"""
        init_file = self.log_dir / "initial_message.txt"
        init_file.write_text(msg, encoding="utf-8")

    def log_final(self, result: Dict):
        """保存最终结果"""
        final_file = self.log_dir / "result.json"
        # 去掉 messages（太大），单独保存
        result_copy = {k: v for k, v in result.items() if k != "messages"}
        final_file.write_text(
            json.dumps(result_copy, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.get("task_code"):
            code_file = self.log_dir / "task_output.py"
            code_file.write_text(result["task_code"], encoding="utf-8")

    def _write_log(self, text: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)

    def _write_jsonl(self, entry: Dict):
        with open(self.messages_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_truncate_dict(d: Dict, max_str_len=500) -> Dict:
        """截断 dict 中的长字符串，用于日志"""
        result = {}
        for k, v in d.items():
            if isinstance(v, str) and len(v) > max_str_len:
                result[k] = v[:max_str_len] + f"...(+{len(v)-max_str_len} chars)"
            else:
                result[k] = v
        return result


class ReactAgent:
    """
    ReAct Agent - 算子任务构建

    维护完整的 LLM 输入/输出历史，支持逐步任务完成。
    """

    def __init__(self, model_level: str = None, verbose: bool = True):
        self.model_level = model_level
        self.verbose = verbose
        self.history: List[StepRecord] = []
        self.messages: List[Dict[str, str]] = []
        self._client = None
        self.run_id = get_run_id()
        self.session_log = SessionLogger(self.run_id)

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client(self.model_level)
        return self._client

    # ======================== 主入口 ========================

    def run(self, user_input: str, description: str = "") -> Dict[str, Any]:
        """
        执行完整的 ReAct 循环

        Returns:
            {"status": ..., "task_code": ..., "summary": ..., "history": ..., "log_dir": ...}
        """
        # 0. 清理上次运行的 workspace
        self._clean_workspace()

        # 1. 解析输入
        parsed = InputParser.parse(user_input, description)
        self._log(f"输入类型: {parsed.input_type}")
        if parsed.file_path:
            self._log(f"文件路径: {parsed.file_path} ({parsed.total_lines}行)")
        if parsed.dir_path:
            self._log(f"目录路径: {parsed.dir_path} ({len(parsed.files)} 个文件, 共{parsed.total_lines}行)")

        # 2. 构建 system prompt
        tools_desc = self._build_tools_description()
        system = SYSTEM_PROMPT.format(tools_description=tools_desc)
        self.messages = [{"role": "system", "content": system}]
        self.session_log.log_system_prompt(system)

        # 3. 初始用户消息
        initial_msg = self._build_initial_message(parsed)
        self.messages.append({"role": "user", "content": initial_msg})
        self.session_log.log_initial_message(initial_msg)

        # 4. ReAct 循环
        for step in range(1, MAX_REACT_STEPS + 1):
            self._log(f"\n{'='*50} Step {step}/{MAX_REACT_STEPS} {'='*50}")

            # 4.1 调用 LLM
            action, raw_response = self._get_next_action(step)
            if action is None:
                result = self._make_result("error", error="LLM 多次返回无效格式，终止")
                self.session_log.log_final(result)
                return result

            thought = action.get("thought", "")
            act_name = action.get("action", "")
            act_args = action.get("arguments", {})

            self._log(f"思考: {thought}")
            self._log(f"动作: {act_name}")

            # 4.2 检查 finish
            if act_name == "finish":
                task_code = act_args.get("task_code", "")
                error = act_args.get("error", "")
                summary = act_args.get("summary", "")

                if error:
                    result = self._make_result("error", error=error)
                    self.session_log.log_final(result)
                    return result

                # task_code 可能是文件路径引用，尝试读取
                task_code = self._resolve_task_code(task_code)

                if task_code:
                    validation = TaskBuilder.validate_task_code(task_code)
                    if not validation["valid"]:
                        self._log(f"格式检查问题: {validation['issues']}")

                result = self._make_result("success", task_code=task_code, summary=summary)
                self.session_log.log_final(result)
                return result

            # 4.3 执行工具
            if act_name not in ToolRegistry.list_ids():
                tool_result = {
                    "status": "error", "output": "",
                    "error": f"未知工具: {act_name}，可用工具: {ToolRegistry.list_ids()}"
                }
            else:
                tool_result = ToolRegistry.execute(act_name, act_args)

            self._log(f"结果: {tool_result['status']}")
            if tool_result.get("output"):
                output_preview = tool_result["output"][:500]
                self._log(f"输出: {output_preview}{'...' if len(tool_result['output']) > 500 else ''}")
            if tool_result.get("error"):
                self._log(f"错误: {tool_result['error']}")

            # 4.4 记录
            record = StepRecord(
                step=step, thought=thought, action=act_name,
                arguments=act_args, result=tool_result, raw_response=raw_response,
            )
            self.history.append(record)
            self.session_log.log_step(step, thought, act_name, act_args, tool_result, raw_response)

            # 4.5 追加到 LLM 消息
            self.messages.append({
                "role": "assistant",
                "content": json.dumps(action, ensure_ascii=False),
            })

            raw_output = tool_result.get("output", "")
            tool_result_msg = f"工具 [{act_name}] 执行结果:\n状态: {tool_result['status']}\n"

            # 截断输出但保留 workspace 引用
            if raw_output:
                # 提取 workspace 相关的关键信息（路径引用不能丢）
                ws_refs = [line for line in raw_output.splitlines()
                           if "workspace" in line.lower() or "已保存" in line or "已复制" in line]
                truncated_output = self._truncate(raw_output, max_len=4000)
                tool_result_msg += f"输出:\n{truncated_output}\n"
                # 确保 workspace 引用不被截断
                if ws_refs:
                    ws_info = "\n".join(ws_refs)
                    if ws_info not in truncated_output:
                        tool_result_msg += f"\n[工作区引用]:\n{ws_info}\n"

            if tool_result.get("error"):
                tool_result_msg += f"错误: {tool_result['error']}\n"

            self.messages.append({"role": "user", "content": tool_result_msg})

            # 4.6 消息历史管理: 防止 token 溢出
            self._manage_history()

        result = self._make_result("error", error=f"超过最大步数 {MAX_REACT_STEPS}")
        self.session_log.log_final(result)
        return result

    # ======================== LLM 调用 ========================

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        async def _async_call():
            result = await self.client.generate(messages)
            return result.get("content", "")
        return _run_async(_async_call())

    def _get_next_action(self, step: int) -> tuple:
        """
        调用 LLM 获取下一步动作。
        返回 (action_dict, raw_response) 或 (None, "")
        """
        for attempt in range(MAX_RETRIES_PER_STEP + 1):
            try:
                response = self._call_llm(self.messages)
                self.session_log.log_llm_call(step, len(self.messages), response)

                if not response or not response.strip():
                    self._log(f"LLM 返回空响应 (attempt {attempt+1})")
                    continue

                action = self._parse_action(response)
                if action:
                    return action, response

                # 检测是否是因为 JSON 太长被截断
                is_truncated = (
                    len(response) > 5000
                    and '"content"' in response[:500]
                    and response.count('{') > response.count('}')
                )

                self._log(f"解析失败 (attempt {attempt+1})"
                          f"{'，JSON被截断' if is_truncated else ''}"
                          f"，原始响应:\n{response[:300]}")

                # 不把超长的截断响应追加进消息（浪费 token）
                if is_truncated:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "你的 JSON 太长被截断了！每次代码不能超过 150 行！\n\n"
                            "【解决方案 - 选择一种】\n\n"
                            "方案1: assemble_task 选择性拼装（源函数不需修改时）:\n"
                            '{"thought": "...", "action": "assemble_task", "arguments": {\n'
                            '  "source_files": [{"path": "workspace/src.py", "functions": ["f1","f2"]}],\n'
                            '  "model_code": "class Model(nn.Module):\\n    ...",\n'
                            '  "get_inputs_code": "def get_inputs():\\n    ...",\n'
                            '  "get_init_inputs_code": "def get_init_inputs():\\n    return []"}}\n\n'
                            "方案2: 分段生成（需修改源函数时）:\n"
                            '  先 write_file 写前150行，再多次 append_to_file 追加后续段。\n\n'
                            "方案3: 骨架+填充:\n"
                            '  write_file 写骨架（def func(): pass），再 apply_patch 逐个替换 pass。'
                        )
                    })
                else:
                    # 短响应：正常追加并给格式提示
                    short_response = response[:1000] + ("..." if len(response) > 1000 else "")
                    self.messages.append({"role": "assistant", "content": short_response})
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "JSON 解析失败。请严格输出纯 JSON（不要有其他文字）：\n\n"
                            '{"thought": "思考", "action": "工具名", "arguments": {参数}}\n\n'
                            "每次 arguments 中的代码不要超过 150 行。长文件用 write_file + append_to_file 分段生成。"
                        )
                    })

            except Exception as e:
                self._log(f"LLM 调用异常 (attempt {attempt+1}): {e}")
                if attempt < MAX_RETRIES_PER_STEP:
                    time.sleep(2)

        return None, ""

    def _parse_action(self, response: str) -> Optional[Dict]:
        """
        从 LLM 响应中解析 JSON 动作。
        多种策略，从严格到宽松。
        """
        text = response.strip()

        # 策略 1: 直接 JSON 解析
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "action" in data:
                return self._normalize_action(data)
        except json.JSONDecodeError:
            pass

        # 策略 2: 提取 ```json ... ``` 代码块
        json_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        for block in json_blocks:
            try:
                data = json.loads(block.strip())
                if isinstance(data, dict) and "action" in data:
                    return self._normalize_action(data)
            except json.JSONDecodeError:
                continue

        # 策略 3: 找第一个平衡的 {...} 块
        brace_content = self._extract_balanced_json(text)
        if brace_content:
            try:
                data = json.loads(brace_content)
                if isinstance(data, dict) and "action" in data:
                    return self._normalize_action(data)
            except json.JSONDecodeError:
                pass

        # 策略 4: 修复常见 JSON 错误（尾随逗号、单引号等）
        cleaned = self._fix_json(text)
        if cleaned:
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and "action" in data:
                    return self._normalize_action(data)
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _extract_balanced_json(text: str) -> Optional[str]:
        """提取第一个平衡的 {} 块"""
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
                    return text[start:i+1]
        return None

    @staticmethod
    def _fix_json(text: str) -> Optional[str]:
        """尝试修复常见 JSON 错误"""
        # 找到第一个 { 和最后一个 }
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end+1]
        # 移除尾随逗号
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        # 单引号 → 双引号（简单情况）
        # 注意: 只在看起来像 key 的位置替换
        candidate = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', candidate)
        return candidate

    @staticmethod
    def _normalize_action(data: Dict) -> Dict:
        """标准化 action 格式"""
        return {
            "thought": data.get("thought", ""),
            "action": data.get("action", ""),
            "arguments": data.get("arguments", data.get("args", {})),
        }

    # ======================== 消息管理 ========================

    def _manage_history(self):
        """
        管理消息历史长度，防止 token 溢出。
        保留 system prompt + 最近 N 轮 + workspace 引用摘要。
        """
        MAX_MESSAGES = 50  # 允许更多消息（workspace 引用使消息更紧凑）
        if len(self.messages) <= MAX_MESSAGES:
            return

        system = self.messages[0]
        initial_user = self.messages[1]

        # 从被压缩的消息中提取 workspace 引用和操作摘要
        removed = self.messages[2:-MAX_MESSAGES+4]
        summary_parts = []
        workspace_refs = set()

        for msg in removed:
            content = msg.get("content", "")
            # 提取 workspace 引用
            for line in content.splitlines():
                if "workspace/" in line and ("已保存" in line or "已复制" in line or "[" in line):
                    workspace_refs.add(line.strip())
            # 操作摘要
            if msg["role"] == "assistant":
                try:
                    d = json.loads(content)
                    summary_parts.append(f"[{d.get('action','')}] {d.get('thought','')[:80]}")
                except (json.JSONDecodeError, AttributeError):
                    pass

        compressed = "[操作历史摘要]\n" + "\n".join(summary_parts[-15:])
        if workspace_refs:
            compressed += "\n\n[工作区中已保存的文件]:\n" + "\n".join(sorted(workspace_refs))

        compressed_msg = {"role": "user", "content": compressed}
        recent = self.messages[-MAX_MESSAGES+4:]
        self.messages = [system, initial_user, compressed_msg] + recent

    # ======================== 辅助方法 ========================

    def _build_tools_description(self) -> str:
        tools = ToolRegistry.list_for_prompt()
        lines = []
        for t in tools:
            params = t["parameters"].get("properties", {})
            param_strs = []
            required = t["parameters"].get("required", [])
            for name, info in params.items():
                req = " (必填)" if name in required else " (可选)"
                param_strs.append(f"    - {name}: {info.get('description', info.get('type', ''))}{req}")
            lines.append(f"### {t['name']}\n{t['description']}\n参数:\n" + "\n".join(param_strs))
        return "\n\n".join(lines)

    @staticmethod
    def _resolve_task_code(task_code: str) -> str:
        """
        如果 task_code 是文件路径引用，读取文件内容。
        支持: workspace/xxx.py, output/xxx.py, 绝对路径
        """
        if not task_code or not task_code.strip():
            return task_code

        text = task_code.strip()
        # 如果看起来像文件路径（短字符串，以 .py 结尾）
        if text.endswith(".py") and len(text) < 500 and "\n" not in text:
            candidates = [
                WORKSPACE_DIR / text,
                WORKSPACE_DIR / Path(text).name,
                OUTPUT_DIR / text,
                Path(text),
            ]
            for p in candidates:
                p = p.expanduser().resolve()
                if p.exists() and p.is_file():
                    return p.read_text(encoding="utf-8")
        return task_code

    @staticmethod
    def _clean_workspace():
        """清理上次运行的 workspace 文件"""
        import shutil
        if WORKSPACE_DIR.exists():
            shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    def _build_initial_message(self, parsed: ParsedInput) -> str:
        parts = []
        parts.append("请帮我将以下内容构建为 KernelBench 标准格式的算子任务。\n")

        if parsed.description:
            parts.append(f"**描述**: {parsed.description}\n")

        if parsed.input_type == "code":
            code_preview = parsed.code[:3000]
            if len(parsed.code) > 3000:
                code_preview += f"\n... (代码共{parsed.total_lines}行，已截断)"
            parts.append(f"**输入类型**: 直接代码\n**代码内容**:\n```python\n{code_preview}\n```")

        elif parsed.input_type == "file":
            if parsed.code:
                code_preview = parsed.code[:3000]
                truncated = len(parsed.code) > 3000
                parts.append(
                    f"**输入类型**: 文件 ({parsed.file_path}, {parsed.total_lines}行)\n"
                    f"**文件内容**:\n```python\n{code_preview}\n```"
                )
                if truncated:
                    parts.append(f"\n(文件较长，已截断。可使用 read_file + offset/limit 查看完整内容)")
            else:
                parts.append(f"**输入类型**: 文件路径 ({parsed.file_path})，请先用 read_file 读取此文件。")

        elif parsed.input_type == "directory":
            parts.append(f"**输入类型**: 代码仓 ({parsed.dir_path}, 共{parsed.total_lines}行)")
            parts.append(f"包含 {len(parsed.files)} 个 Python 文件:")
            for name, line_count in list(parsed.files.items())[:30]:
                parts.append(f"  - {name} ({line_count}行)")
            if len(parsed.files) > 30:
                parts.append(f"  ... 共 {len(parsed.files)} 个文件")
            parts.append("\n请先用 scan_dir 浏览结构，然后用 copy_to_workspace 复制关键文件到工作区。")

        parts.append(f"\n**参考格式**:\n```python\n{TaskBuilder.format_reference()}\n```")
        parts.append(
            "\n**工作区路径**: workspace/\n"
            "\n**推荐流程**:\n"
            "1. scan_dir 浏览目录\n"
            "2. copy_to_workspace 复制源文件和 benchmark/test 文件\n"
            "3. **依赖分析**（关键！）: read_function 提取目标函数，列出它调用的所有函数，递归分析完整依赖链\n"
            "4. read_function 提取 benchmark 的 create_config/create_inputs（了解输入形状）\n"
            "5. 检查 NPU/Triton 代码是否有 fallback（如 except ImportError: return torch_impl(...)）\n"
            "6. 选择策略构建任务文件:\n"
            "   - 依赖大部分函数但有少量不需要的 → assemble_task 排除模式: "
            '{\"path\": \"workspace/src.py\", \"exclude_functions\": [\"unused1\"]}\n'
            "   - 依赖部分函数 → assemble_task 选择性提取: "
            '{\"path\": \"workspace/src.py\", \"functions\": [\"f1\",\"f2\"]}\n'
            "   - 需修改源函数 → write_file + append_to_file（从workspace复制原始代码，不要重写！）\n"
            "   - 从其他文件提取辅助函数 → helper_code 参数内联，不要用 source_files 提取（避免引入外部 import）\n"
            "7. validate_task task_file=\"task_output.py\"\n"
            "8. finish task_code=\"task_output.py\"\n"
            "\n【禁止重写复杂函数！】原始代码的索引计算很精确，自己写的'简化版'几乎必定有bug。\n"
            "【每次代码不超过150行】长文件用分段生成。\n"
            "【从其他文件提取函数】建议用 helper_code 内联而非 source_files，避免引入外部模块的 import。\n"
            "\n请开始工作。"
        )

        return "\n".join(parts)

    def _make_result(self, status: str, task_code: str = "",
                     summary: str = "", error: str = "") -> Dict[str, Any]:
        return {
            "status": status,
            "task_code": task_code,
            "summary": summary,
            "error": error,
            "steps": len(self.history),
            "log_dir": str(self.session_log.log_dir),
            "history": [
                {
                    "step": r.step,
                    "action": r.action,
                    "thought": r.thought,
                    "result_status": r.result.get("status", ""),
                }
                for r in self.history
            ],
            "messages": self.messages,
        }

    @staticmethod
    def _truncate(text: str, max_len: int = 6000) -> str:
        if len(text) <= max_len:
            return text
        half = max_len // 2
        return text[:half] + f"\n\n... (截断，原始长度 {len(text)} 字符) ...\n\n" + text[-half:]

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
        logger.info(msg)
        self.session_log._write_log(msg + "\n")
