#!/usr/bin/env python3
"""KernelAgent 测试

交互模式（默认）:
  python examples/run_kernel_agent.py

自动模式（无需人工确认，跑完即退出）:
  python examples/run_kernel_agent.py --auto --requirement "实现一个向量加法算子"
  python examples/run_kernel_agent.py --auto -r "实现一个ReLU算子" --max-rounds 30
"""

import asyncio
import os
import sys
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


# ==================== 结构化日志系统 ====================

class SessionLogger:
    """
    结构化会话日志
    
    目录结构（位于 conversation 目录下）:
        ~/.akg/conversations/{task_id}/logs/
        ├── session.log                          ← 人类可读时间线
        ├── events.jsonl                         ← 紧凑元数据索引（大文本用文件引用）
        ├── prompts/                             ← LLM 完整 prompt
        │   ├── Round01_LLMCall001.txt
        │   └── Round02_LLMCall001.txt
        ├── responses/                           ← LLM 完整响应
        │   ├── Round01_LLMCall001.txt
        │   └── Round02_LLMCall001.txt
        └── tool_calls/                          ← 工具调用详情（pretty-printed JSON）
            ├── Round01_ToolCall001.json
            └── Round01_ToolCall002.json
    
    命名规则:
        Round{NN}       - 第 NN 轮用户交互
        LLMCall{NNN}    - 该轮内第 NNN 次 LLM 调用
        ToolCall{NNN}   - 该轮内第 NNN 次工具调用
    """
    
    def __init__(self, task_id: str = "", base_dir: str = ""):
        self._ts = datetime.now()
        self._task_id = task_id
        self._base_dir = Path(base_dir) if base_dir else Path.home() / ".akg"
        
        # 计数器
        self._llm_call_counter = {}  # round -> count
        self._tool_call_counter = {}  # round -> count
        
        # 目录和文件句柄（延迟到 bind_task 时创建）
        self._session_dir = None
        self._session_log = None
        self._events_jsonl = None
        
        # 如果初始化时就有 task_id，直接创建目录
        if task_id:
            self._init_dirs(task_id)
    
    def _init_dirs(self, task_id: str):
        """创建日志目录结构: {base_dir}/conversations/{task_id}/logs/"""
        self._session_dir = self._base_dir / "conversations" / task_id / "logs"
        for sub in ("prompts", "responses", "tool_calls"):
            (self._session_dir / sub).mkdir(parents=True, exist_ok=True)
        self._session_log = self._session_dir / "session.log"
        self._events_jsonl = self._session_dir / "events.jsonl"
    
    def bind_task(self, task_id: str, base_dir: str = ""):
        """
        绑定 task_id，创建日志目录
        
        日志目录: {base_dir}/conversations/{task_id}/logs/
        """
        if self._task_id == task_id and self._session_dir is not None:
            return  # 已经绑定
        
        self._task_id = task_id
        if base_dir:
            self._base_dir = Path(base_dir)
        
        self._init_dirs(task_id)
    
    @property
    def session_dir(self) -> Path:
        return self._session_dir
    
    # ---------- 内部方法 ----------
    
    def _next_llm_id(self, round_num: int) -> str:
        """生成 LLM 调用 ID，如 Round01_LLMCall001"""
        cnt = self._llm_call_counter.get(round_num, 0) + 1
        self._llm_call_counter[round_num] = cnt
        return f"Round{round_num:02d}_LLMCall{cnt:03d}"
    
    def _next_tool_id(self, round_num: int) -> str:
        """生成工具调用 ID，如 Round01_ToolCall001"""
        cnt = self._tool_call_counter.get(round_num, 0) + 1
        self._tool_call_counter[round_num] = cnt
        return f"Round{round_num:02d}_ToolCall{cnt:03d}"
    
    def _append_log(self, line: str):
        """追加一行到 session.log"""
        if not self._session_log:
            return
        with open(self._session_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    
    def _append_event(self, event: dict):
        """追加一条紧凑事件到 events.jsonl"""
        if not self._events_jsonl:
            return
        with open(self._events_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    def _save_text(self, subdir: str, file_id: str, ext: str, content: str) -> Path:
        """保存大文本到独立文件，返回路径"""
        if not self._session_dir:
            return Path("/dev/null")
        filepath = self._session_dir / subdir / f"{file_id}{ext}"
        filepath.write_text(content, encoding="utf-8")
        return filepath
    
    def _ts_str(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
    
    # ---------- 公开接口 ----------
    
    def log_init(self, task_id: str, total_tools: int,
                 basic_tools: list, agent_tools: list):
        """记录初始化"""
        header = (
            f"{'=' * 60}\n"
            f"  KernelAgent Session\n"
            f"{'=' * 60}\n"
            f"时间:   {self._ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"任务ID: {task_id}\n"
            f"工具:   {total_tools} (Basic: {len(basic_tools)}, Agent: {len(agent_tools)})\n"
            f"        Basic: {', '.join(basic_tools)}\n"
            f"        Agent: {', '.join(agent_tools)}\n"
            f"{'=' * 60}\n"
        )
        self._append_log(header)
        self._append_event({
            "ts": self._ts_str(), "type": "init",
            "task_id": task_id,
            "tools": {"total": total_tools, "basic": basic_tools, "agent": agent_tools}
        })
    
    def log_round_start(self, round_num: int, user_input: str, current_node: str):
        """记录轮次开始 + 用户输入"""
        self._append_log(
            f"\n{'─' * 40} Round {round_num} {'─' * 40}\n"
            f"[{self._ts_str()}] 用户: {user_input}\n"
            f"           (node: {current_node})"
        )
        self._append_event({
            "ts": self._ts_str(), "round": round_num, "type": "user_input",
            "input": user_input, "node": current_node
        })
    
    def log_llm_request(self, round_num: int, model_level: str,
                        prompt_text: str, current_node: str,
                        context_details: dict = None):
        """记录 LLM 请求（prompt 保存到独立文件）"""
        call_id = self._next_llm_id(round_num)
        prompt_file = self._save_text("prompts", call_id, ".txt", prompt_text)
        
        # 可读日志
        ctx_line = ""
        if context_details:
            ratio = context_details.get("compression_ratio", "?/?")
            compressed = "压缩" if context_details.get("is_compressed") else "完整"
            ctx_line = f"\n           History: {ratio} ({compressed})"
        
        self._append_log(
            f"[{self._ts_str()}] -> LLM 请求 (model={model_level}, "
            f"{len(prompt_text)} chars, node={current_node})"
            f"{ctx_line}\n"
            f"           prompt: {prompt_file.name}"
        )
        
        # 紧凑事件（不含 prompt 全文）
        event = {
            "ts": self._ts_str(), "round": round_num, "type": "llm_request",
            "call_id": call_id, "model": model_level,
            "prompt_len": len(prompt_text), "prompt_file": str(prompt_file.name),
            "node": current_node,
        }
        if context_details:
            event["context"] = context_details
        self._append_event(event)
        
        return call_id
    
    def log_llm_response(self, round_num: int, call_id: str, response: str):
        """记录 LLM 响应
        
        如果响应内容是 JSON（可能包裹在 ```json ... ``` 中），
        则提取并格式化为 .json 文件；否则保存为 .txt。
        """
        # 尝试提取 JSON 内容并格式化保存
        tool_hint = ""
        parsed_json = None
        
        try:
            from akg_agents.core_v2.agents.plan import PlanAgent
            json_str = PlanAgent._extract_nested_json(response)
            if json_str:
                parsed_json = json.loads(json_str)
                tn = parsed_json.get("tool_name", "")
                if tn:
                    tool_hint = f"\n           决策: {tn}"
        except Exception:
            pass
        
        if parsed_json is not None:
            # 内容是 JSON → 保存为格式化的 .json
            resp_file = self._save_text(
                "responses", call_id, ".json",
                json.dumps(parsed_json, indent=2, ensure_ascii=False)
            )
        else:
            # 非 JSON → 保存为 .txt
            resp_file = self._save_text("responses", call_id, ".txt", response)
        
        self._append_log(
            f"[{self._ts_str()}] <- LLM 响应 ({len(response)} chars)"
            f"{tool_hint}\n"
            f"           response: {resp_file.name}"
        )
        self._append_event({
            "ts": self._ts_str(), "round": round_num, "type": "llm_response",
            "call_id": call_id, "response_len": len(response),
            "response_file": str(resp_file.name),
        })
    
    def log_tool_call(self, round_num: int, tool_name: str,
                      arguments: dict, result: dict, duration_ms: float = 0):
        """记录工具调用（详情保存到独立 JSON 文件）"""
        call_id = self._next_tool_id(round_num)
        
        # 保存详细的 pretty-printed JSON
        detail = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "duration_ms": duration_ms,
        }
        detail_file = self._save_text(
            "tool_calls", call_id, ".json",
            json.dumps(detail, indent=2, ensure_ascii=False)
        )
        
        status = result.get("status", "?") if isinstance(result, dict) else "?"
        duration_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        
        self._append_log(
            f"[{self._ts_str()}] 工具: {tool_name} -> {status}{duration_str}\n"
            f"           detail: {detail_file.name}"
        )
        self._append_event({
            "ts": self._ts_str(), "round": round_num, "type": "tool_call",
            "call_id": call_id, "tool": tool_name,
            "status": status, "duration_ms": duration_ms,
            "detail_file": str(detail_file.name),
        })
    
    def log_round_result(self, round_num: int, status: str, output: str = "",
                         error: str = "", current_node: str = ""):
        """记录轮次结果"""
        status_icon = {"success": "[OK]", "error": "[ERR]",
                       "waiting_for_user": "[WAIT]"}.get(status, f"[{status}]")
        
        output_preview = (output[:120] + "...") if len(output) > 120 else output
        output_preview = output_preview.replace("\n", " ")
        
        lines = [f"[{self._ts_str()}] 结果: {status_icon}"]
        if output_preview:
            lines.append(f"           输出: {output_preview}")
        if error:
            lines.append(f"           错误: {error[:200]}")
        
        self._append_log("\n".join(lines))
        self._append_event({
            "ts": self._ts_str(), "round": round_num, "type": "result",
            "status": status, "output_len": len(output),
            "error": error[:300] if error else "", "node": current_node,
        })
    
    def log_session_end(self, total_rounds: int, summary: dict = None):
        """记录会话结束"""
        lines = [
            f"\n{'=' * 60}",
            f"  Session End",
            f"{'=' * 60}",
            f"总轮次: {total_rounds}",
        ]
        if summary:
            lines.append(f"当前节点: {summary.get('current_node', '?')}")
            lines.append(f"总动作数: {summary.get('total_actions', 0)}")
        lines.append(f"日志目录: {self._session_dir}")
        lines.append("=" * 60)
        
        self._append_log("\n".join(lines))
    
    def print_summary(self):
        """打印日志摘要"""
        if not self._session_dir:
            print("\n[日志摘要] 未绑定 task_id，无日志")
            return
        
        print(f"\n[日志摘要]")
        print(f"   目录: {self._session_dir}")
        
        # 统计文件数量
        prompt_count = len(list((self._session_dir / "prompts").glob("*.txt")))
        resp_count = len(list((self._session_dir / "responses").glob("*.json")))
        tool_count = len(list((self._session_dir / "tool_calls").glob("*.json")))
        
        print(f"   Prompts:   {prompt_count} 个文件")
        print(f"   Responses: {resp_count} 个文件")
        print(f"   ToolCalls: {tool_count} 个文件")
        print(f"   可读日志:  {self._session_log}")
        print(f"   事件索引:  {self._events_jsonl}")


# ==================== 全局日志实例 ====================
# task_id 在 agent 创建后通过 bind_task() 绑定
session_logger = SessionLogger()

_current_round = 0


# ==================== 诊断 ====================

import akg_agents

def diagnose_imports():
    """诊断可能的导入问题"""
    issues = []
    available = []
    
    try:
        from akg_agents.op.agents.kernel_agent import KernelAgent
        available.append("KernelAgent")
    except ImportError as e:
        issues.append(f"KernelAgent 导入失败: {e}")
        return issues, []
    
    try:
        from akg_agents.core_v2.agents.plan import PlanAgent
        available.append("PlanAgent")
    except ImportError as e:
        issues.append(f"PlanAgent 导入失败: {e}")
    
    op_agents = {
        "kernel_gen": "KernelGen",
        "kernel_designer": "KernelDesigner",
        "op_task_builder": "OpTaskBuilder"
    }
    
    for module_name, class_name in op_agents.items():
        try:
            __import__(f"akg_agents.op.agents.{module_name}")
            available.append(class_name)
        except ImportError as e:
            issues.append(f"{class_name} 导入失败: {e}")
    
    return issues, available


from akg_agents.op.agents.kernel_agent import KernelAgent


# ==================== Agent Logging Wrapper ====================

def wrap_agent_with_logging(agent):
    """包装 agent.run_llm 和 agent.run_llm_with_tools 以记录 LLM 交互到结构化日志"""

    def _get_context_details():
        try:
            if hasattr(agent, 'trace') and hasattr(agent, 'current_node_id'):
                full_history = agent.trace.get_full_action_history(agent.current_node_id)
                return {
                    "full_history_count": len(full_history),
                    "current_plan_steps": len(getattr(agent, 'plan_list', [])),
                    "original_user_input": getattr(agent, '_original_user_input', 'N/A')[:100],
                    "compression_ratio": f"{len(full_history)}/{len(full_history)}",
                    "is_compressed": False,
                }
        except Exception as e:
            return {"error": str(e)}
        return None

    # ---- hook run_llm（子 agent 仍然使用） ----
    original_run_llm = agent.run_llm

    async def logged_run_llm(template, input_data, model_level="standard"):
        global _current_round
        prompt_text = template.format(**input_data) if hasattr(template, 'format') else str(template)

        call_id = session_logger.log_llm_request(
            _current_round, model_level, prompt_text,
            getattr(agent, 'current_node_id', 'unknown'),
            _get_context_details()
        )
        print(f"   [LLM Request] {len(prompt_text)} chars, node={getattr(agent, 'current_node_id', '?')}")

        result = await original_run_llm(template, input_data, model_level)

        if isinstance(result, tuple) and result:
            response = result[0] if len(result) > 0 else ""
            session_logger.log_llm_response(_current_round, call_id, response)
            print(f"   [LLM Response] {len(response) if isinstance(response, str) else 0} chars")

        return result

    agent.run_llm = logged_run_llm

    # ---- hook run_llm_with_tools（ReAct agent 的 function calling 路径） ----
    original_run_llm_with_tools = agent.run_llm_with_tools

    async def logged_run_llm_with_tools(messages, tools, model_level="standard"):
        global _current_round
        import json as _json

        prompt_text = "\n\n".join(
            f"[{m.get('role', '?')}]\n{m.get('content', '')}" for m in messages
        )
        tools_summary = f"\n\n[tools] {len(tools)} 个工具定义"
        full_text = prompt_text + tools_summary

        call_id = session_logger.log_llm_request(
            _current_round, model_level, full_text,
            getattr(agent, 'current_node_id', 'unknown'),
            _get_context_details()
        )
        print(f"   [LLM Request(tools)] {len(full_text)} chars, {len(tools)} tools, node={getattr(agent, 'current_node_id', '?')}")

        result = await original_run_llm_with_tools(messages, tools, model_level)

        response_text = result.get("content", "")
        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            response_text += f"\n\n[tool_calls] {_json.dumps(tool_calls, ensure_ascii=False, indent=2)}"
        session_logger.log_llm_response(_current_round, call_id, response_text)
        print(f"   [LLM Response(tools)] content={len(result.get('content', ''))} chars, tool_calls={len(tool_calls)}")

        return result

    agent.run_llm_with_tools = logged_run_llm_with_tools
    return agent


# ==================== 自动模式配置 ====================

# 自动模式下，当 Agent 流程中需要用户确认时，使用的默认回复
AUTO_REPLY_MESSAGE = "同意你的方案，请直接继续执行，不需要再确认。初始需求完成后直接结束任务，不需要进一步优化。"

# 自动模式下的最大轮次（防止无限循环）
AUTO_MAX_ROUNDS = 50


# ==================== 主测试函数 ====================

async def test_kernel_agent(auto_mode: bool = False, requirement: str = "",
                            max_rounds: int = AUTO_MAX_ROUNDS,
                            auto_reply: str = AUTO_REPLY_MESSAGE):
    """KernelAgent 测试
    
    Args:
        auto_mode: 是否自动模式（无需人工确认，跑完即退出）
        requirement: 自动模式下的初始需求（交互模式下可为空）
        max_rounds: 自动模式下的最大轮次
        auto_reply: 自动模式下对 Agent 流程中确认的默认回复
    """
    
    mode_label = "自动模式" if auto_mode else "交互模式"
    
    print("\n" + "=" * 80)
    print(f"KernelAgent 测试 ({mode_label})")
    print("=" * 80 + "\n")
    
    if auto_mode:
        print(f"[自动模式] 自动回复: \"{auto_reply}\"")
        print(f"[自动模式] 最大轮次: {max_rounds}")
        if requirement:
            print(f"[自动模式] 初始需求: {requirement[:100]}")
        print()
    
    # 诊断导入
    print("[诊断] 检查模块导入...")
    import_issues, available_agents = diagnose_imports()
    
    if available_agents:
        print(f"  可用: {', '.join(available_agents)}")
    if import_issues:
        for issue in import_issues:
            print(f"  注意: {issue}")
    else:
        print("  所有模块正常\n")
    
    try:
        import time
        task_id = f"test_{int(time.time())}"
        
        # 将日志绑定到 conversation 目录
        session_logger.bind_task(task_id)
        
        print(f"[初始化] 创建 KernelAgent...")
        print(f"  任务ID:   {task_id}")
        print(f"  日志目录: {session_logger.session_dir}\n")
        
        agent = KernelAgent(task_id=task_id, model_level="complex")
        agent = wrap_agent_with_logging(agent)
        
        # 工具信息
        all_tool_names = [t.get("function", {}).get("name", "Unknown") for t in agent.available_tools]
        agent_tool_names = list(agent.agent_registry.keys())
        workflow_tool_names = list(getattr(agent, 'workflow_registry', {}).keys())
        basic_tool_names = [t for t in all_tool_names
                           if t not in agent_tool_names and t not in workflow_tool_names]
        
        print(f"[已注册工具] 共 {len(all_tool_names)} 个")
        print(f"  Basic:    {basic_tool_names}")
        print(f"  Agent:    {agent_tool_names}")
        if workflow_tool_names:
            print(f"  Workflow: {workflow_tool_names}")
        
        if not agent_tool_names:
            print(f"\n  警告：没有加载任何 Agent Tools！")
        
        # 记录初始化
        session_logger.log_init(task_id, len(all_tool_names), basic_tool_names, agent_tool_names)
        print()
        
        round_num = 0
        first_input = True  # 标记是否为第一次输入
        
        while True:
            round_num += 1
            global _current_round
            _current_round = round_num
            
            # 自动模式下检查最大轮次
            if auto_mode and round_num > max_rounds:
                print(f"\n[自动模式] 已达到最大轮次 {max_rounds}，自动退出")
                break
            
            print("\n" + "=" * 80)
            print(f"Round {round_num}")
            print("=" * 80)
            
            # 获取用户输入
            if auto_mode:
                if first_input:
                    # 自动模式第一轮：使用命令行传入的 requirement
                    if requirement:
                        user_input = requirement
                    else:
                        # 没有通过命令行传 requirement，交互式获取一次
                        try:
                            user_input = input("\n请输入需求: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n\n用户中断")
                            break
                    first_input = False
                else:
                    # 自动模式后续轮：自动回复
                    user_input = auto_reply
                    print(f"\n[自动回复] {user_input}")
            else:
                # 交互模式：每次都等待用户输入
                try:
                    user_input = input("\n请输入需求（quit 退出）: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n用户中断")
                    break
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
            
            if not user_input:
                round_num -= 1
                continue
            
            print(f"\n[处理中] {user_input[:100]}\n")
            
            # 记录用户输入
            trace_before = agent.get_trace_summary() if hasattr(agent, 'get_trace_summary') else {}
            session_logger.log_round_start(
                round_num, user_input,
                getattr(agent, 'current_node_id', 'unknown')
            )
            
            # 执行
            result = await agent.run(user_input)
            
            # 记录新增的工具调用
            trace_after = agent.get_trace_summary() if hasattr(agent, 'get_trace_summary') else {}
            try:
                full_history = agent.trace.get_full_action_history(agent.current_node_id)
                new_actions = full_history[trace_before.get('total_actions', 0):]
                for action in new_actions:
                    session_logger.log_tool_call(
                        round_num,
                        action.tool_name,
                        action.arguments,
                        action.result,
                        action.duration_ms or 0
                    )
            except Exception:
                pass
            
            # 记录轮次结果
            session_logger.log_round_result(
                round_num,
                result.get('status', 'unknown'),
                result.get('output', ''),
                result.get('error_information', ''),
                result.get('current_node', 'unknown')
            )
            
            # 控制台输出
            print("\n" + "=" * 80)
            print(f"[执行结果]")
            print("-" * 80)
            print(f"  状态: {result.get('status')}")
            
            output_text = result.get('output', '')
            if output_text:
                preview = output_text[:500]
                print(f"  输出: {preview}")
                if len(output_text) > 500:
                    print(f"       (截断，共 {len(output_text)} 字符)")
            
            if result.get('error_information'):
                print(f"  错误: {result.get('error_information')}")
            
            # 显示计划
            if result.get('plan_list'):
                print(f"\n[执行计划] {len(result['plan_list'])} 步:")
                for step in result['plan_list']:
                    icon = {"pending": " ", "success": "+", "failed": "x"}.get(
                        step.get("status"), "?")
                    desc = step.get('desc') or step.get('description') or step.get('tool', 'Unknown')
                    print(f"  [{icon}] Step {step.get('step_id', '?')}: {desc[:80]}")
            
            # 显示历史（最近 5 个）
            if result.get('history'):
                print(f"\n[执行历史] 共 {len(result['history'])} 个动作:")
                recent = result['history'][-5:]
                start_idx = len(result['history']) - len(recent) + 1
                for i, r in enumerate(recent, start_idx):
                    tool = r.get('tool_name', 'Unknown')
                    if r.get('compressed'):
                        print(f"  [{i}] {tool} [压缩: {r.get('original_actions', 0)} -> 1]")
                    else:
                        status = r.get('result', {}).get('status', 'N/A')
                        print(f"  [{i}] {tool} -> {status}")
            
            print("=" * 80)
            
            # 需要用户响应
            if result.get('status') == 'waiting_for_user':
                task_completed = result.get('task_completed', False)
                
                print(f"\n[Agent 询问] (task_completed={task_completed})")
                print("-" * 80)
                print(result.get('message'))
                print("-" * 80)
                
                if auto_mode and task_completed:
                    # LLM 标记任务已完成 → 直接退出循环
                    print(f"\n[自动模式] LLM 标记 task_completed=true，任务已完成，自动退出")
                    break
                elif auto_mode:
                    # 流程中的确认 → 自动回复
                    print(f"[自动模式] 流程中确认，将自动回复: \"{auto_reply}\"")
                
                continue
            
            # 完成或出错
            if result.get('status') in ['success', 'error']:
                status_tag = "OK" if result.get('status') == 'success' else "ERROR"
                print(f"\n[{status_tag}] 任务 {result.get('status').upper()}!")
                if result.get('status') == 'success':
                    print(f"  总共执行了 {result.get('total_actions', 0)} 个动作")
                
                if auto_mode:
                    # 自动模式下 finish/error 也直接退出
                    print(f"\n[自动模式] 任务已{result.get('status')}，自动退出")
                    break
                else:
                    continue
        
        # 会话结束
        summary = agent.get_trace_summary() if hasattr(agent, 'get_trace_summary') else {}
        session_logger.log_session_end(round_num, summary)
        
        print("\n" + "=" * 80)
        print(f"Session End")
        print("-" * 80)
        print(f"  总轮次: {round_num}")
        
        if summary:
            print(f"  当前节点: {summary.get('current_node', '?')}")
            print(f"  总动作数: {summary.get('total_actions', 0)}")
        
        # 打印日志摘要
        session_logger.print_summary()
        print(f"\n  查看可读日志: cat {session_logger.session_dir / 'session.log'}")
        print(f"  查看工具: python examples/view_test_logs.py {session_logger.session_dir}")
        print("=" * 80)
        return True
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="KernelAgent 测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式（默认）
  python examples/run_kernel_agent.py

  # 自动模式 + 命令行指定需求
  python examples/run_kernel_agent.py --auto --requirement "实现一个向量加法算子"

  # 自动模式 + 自定义回复 + 限制轮次
  python examples/run_kernel_agent.py --auto -r "实现矩阵乘法算子" --max-rounds 30

  # 自动模式，不指定需求（会交互式输入一次需求，之后全自动）
  python examples/run_kernel_agent.py --auto
        """
    )
    
    parser.add_argument(
        "--auto", action="store_true",
        help="启用自动模式：Agent 需要确认时自动回复，task_completed=true 时自动退出"
    )
    parser.add_argument(
        "-r", "--requirement",
        type=str, default="",
        help="初始需求（自动模式下可选，若不指定则交互式输入一次）"
    )
    parser.add_argument(
        "--max-rounds",
        type=int, default=AUTO_MAX_ROUNDS,
        help=f"自动模式下的最大轮次（默认: {AUTO_MAX_ROUNDS}）"
    )
    parser.add_argument(
        "--auto-reply",
        type=str, default=AUTO_REPLY_MESSAGE,
        help=f"自动模式下对 Agent 流程中确认的默认回复"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = asyncio.run(test_kernel_agent(
        auto_mode=args.auto,
        requirement=args.requirement,
        max_rounds=args.max_rounds,
        auto_reply=args.auto_reply,
    ))
    sys.exit(0 if success else 1)
