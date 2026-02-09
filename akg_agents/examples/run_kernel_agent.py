#!/usr/bin/env python3
"""KernelAgent 测试 - 运行: python examples/run_kernel_agent.py"""

import asyncio
import os
import sys
import logging
import json
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
    
    目录结构:
        ~/.akg/logs/sessions/YYYY-MM-DD/HH-MM-SS/
        ├── session.log           ← 人类可读时间线
        ├── events.jsonl          ← 紧凑元数据索引（大文本用文件引用）
        ├── prompts/              ← LLM 完整 prompt
        │   ├── R01_001.txt
        │   └── R02_001.txt
        ├── responses/            ← LLM 完整响应
        │   ├── R01_001.txt
        │   └── R02_001.txt
        └── tool_calls/           ← 工具调用详情（pretty-printed JSON）
            ├── R01_001.json
            └── R01_002.json
    """
    
    def __init__(self):
        self._ts = datetime.now()
        self._log_dir = Path.home() / ".akg" / "logs"
        self._session_dir = (
            self._log_dir / "sessions"
            / self._ts.strftime("%Y-%m-%d")
            / self._ts.strftime("%H-%M-%S")
        )
        
        # 创建子目录
        for sub in ("prompts", "responses", "tool_calls"):
            (self._session_dir / sub).mkdir(parents=True, exist_ok=True)
        
        # 文件句柄
        self._session_log = self._session_dir / "session.log"
        self._events_jsonl = self._session_dir / "events.jsonl"
        
        # 计数器
        self._llm_call_counter = {}  # round -> count
        self._tool_call_counter = {}  # round -> count
        
        # 写入 latest 指针
        try:
            (self._log_dir / "latest_session.txt").write_text(
                str(self._session_dir), encoding="utf-8"
            )
        except Exception:
            pass
    
    @property
    def session_dir(self) -> Path:
        return self._session_dir
    
    # ---------- 内部方法 ----------
    
    def _next_llm_id(self, round_num: int) -> str:
        cnt = self._llm_call_counter.get(round_num, 0) + 1
        self._llm_call_counter[round_num] = cnt
        return f"R{round_num:02d}_{cnt:03d}"
    
    def _next_tool_id(self, round_num: int) -> str:
        cnt = self._tool_call_counter.get(round_num, 0) + 1
        self._tool_call_counter[round_num] = cnt
        return f"R{round_num:02d}_{cnt:03d}"
    
    def _append_log(self, line: str):
        """追加一行到 session.log"""
        with open(self._session_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    
    def _append_event(self, event: dict):
        """追加一条紧凑事件到 events.jsonl"""
        with open(self._events_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    def _save_text(self, subdir: str, file_id: str, ext: str, content: str) -> Path:
        """保存大文本到独立文件，返回路径"""
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
        """记录 LLM 响应（响应保存到独立文件）"""
        resp_file = self._save_text("responses", call_id, ".txt", response)
        
        # 提取摘要：尝试解析 JSON 获取 tool_name
        tool_hint = ""
        try:
            # 尝试从响应中提取 tool_name
            from akg_agents.core_v2.agents.plan import PlanAgent
            json_str = PlanAgent._extract_nested_json(response)
            if json_str:
                parsed = json.loads(json_str)
                tn = parsed.get("tool_name", "")
                if tn:
                    tool_hint = f"\n           决策: {tn}"
        except Exception:
            pass
        
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
        print(f"\n[日志摘要]")
        print(f"   目录: {self._session_dir}")
        
        # 统计文件数量
        prompt_count = len(list((self._session_dir / "prompts").glob("*.txt")))
        resp_count = len(list((self._session_dir / "responses").glob("*.txt")))
        tool_count = len(list((self._session_dir / "tool_calls").glob("*.json")))
        
        print(f"   Prompts:   {prompt_count} 个文件")
        print(f"   Responses: {resp_count} 个文件")
        print(f"   ToolCalls: {tool_count} 个文件")
        print(f"   可读日志:  {self._session_log}")
        print(f"   事件索引:  {self._events_jsonl}")


# ==================== 全局日志实例 ====================
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
    """包装 agent.run_llm 以记录 LLM 交互到结构化日志"""
    original_run_llm = agent.run_llm
    
    async def logged_run_llm(template, input_data, model_level="standard"):
        global _current_round
        
        # 生成 prompt
        prompt_text = template.format(**input_data) if hasattr(template, 'format') else str(template)
        
        # 获取上下文详情
        context_details = None
        try:
            if hasattr(agent, 'trace') and hasattr(agent, 'current_node_id'):
                full_history = agent.trace.get_full_action_history(agent.current_node_id)
                context_details = {
                    "full_history_count": len(full_history),
                    "current_plan_steps": len(getattr(agent, 'plan_list', [])),
                    "original_user_input": getattr(agent, '_original_user_input', 'N/A')[:100],
                    "compression_ratio": f"{len(full_history)}/{len(full_history)}",
                    "is_compressed": False,
                }
        except Exception as e:
            context_details = {"error": str(e)}
        
        # 记录请求
        call_id = session_logger.log_llm_request(
            _current_round, model_level, prompt_text,
            getattr(agent, 'current_node_id', 'unknown'),
            context_details
        )
        
        # 控制台摘要
        print(f"   [LLM Request] {len(prompt_text)} chars, node={getattr(agent, 'current_node_id', '?')}")
        
        # 调用 LLM
        result = await original_run_llm(template, input_data, model_level)
        
        # 记录响应
        if isinstance(result, tuple) and result:
            response = result[0] if len(result) > 0 else ""
            session_logger.log_llm_response(_current_round, call_id, response)
            print(f"   [LLM Response] {len(response) if isinstance(response, str) else 0} chars")
        
        return result
    
    agent.run_llm = logged_run_llm
    return agent


# ==================== 主测试函数 ====================

async def test_kernel_agent():
    """KernelAgent 交互式测试"""
    
    print("\n" + "=" * 80)
    print("KernelAgent 测试")
    print("=" * 80 + "\n")
    
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
    
    # 检查 API Key
    from akg_agents.utils.environment_check import _check_llm_api
    if not _check_llm_api():
        raise ValueError("LLM API Key 配置或连接有问题，请检查。")
    
    try:
        import time
        task_id = f"test_{int(time.time())}"
        
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
        
        while True:
            round_num += 1
            global _current_round
            _current_round = round_num
            
            print("\n" + "=" * 80)
            print(f"Round {round_num}")
            print("=" * 80)
            
            # 获取用户输入
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
                print(f"\n[Agent 询问]")
                print("-" * 80)
                print(result.get('message'))
                print("-" * 80)
                continue
            
            # 完成或出错
            if result.get('status') in ['success', 'error']:
                status_tag = "OK" if result.get('status') == 'success' else "ERROR"
                print(f"\n[{status_tag}] 任务 {result.get('status').upper()}!")
                if result.get('status') == 'success':
                    print(f"  总共执行了 {result.get('total_actions', 0)} 个动作")
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


if __name__ == "__main__":
    success = asyncio.run(test_kernel_agent())
    sys.exit(0 if success else 1)
