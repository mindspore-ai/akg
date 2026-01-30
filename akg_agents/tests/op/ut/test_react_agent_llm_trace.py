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

"""
ReActAgent 多轮对话测试

# 直接运行
python tests/ut/test_react_agent_llm_trace.py

# 查看日志文件（完整 prompt 上下文）
# 日志目录: ~/akg_agents_llm_trace_logs/

"""

import os
import sys
import uuid
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from akg_agents.core.agent.react_agent import MainOpAgent

# 日志目录: ~/akg_agents_llm_trace_logs/
DEFAULT_LOG_DIR = Path.home() / "akg_agents_llm_trace_logs"


def check_llm_env() -> bool:
    """检查 LLM 环境变量"""
    return bool(os.getenv("AKG_AGENTS_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))


def get_test_config(use_full_config: bool = False) -> dict:
    """
    获取测试配置
    
    Args:
        use_full_config: 是否使用完整配置（需要 Worker）
    """
    import tempfile
    
    if use_full_config:
        try:
            from akg_agents.utils.common_utils import load_yaml
            from akg_agents import get_project_root
            config_path = os.path.join(get_project_root(), "op", "config", "default_triton_cuda_config.yaml")
            if os.path.exists(config_path):
                config = load_yaml(config_path)
                print(f"✅ 加载完整配置: {config_path}")
                return config
        except Exception as e:
            print(f"⚠️ 加载完整配置失败: {e}，使用最小配置")
    
    return {
        "agent_model_config": {"default": "default"},
        "log_dir": tempfile.gettempdir(),
        "docs_dir": {
            "designer": "resources/docs/sketch_docs",
            "coder": "resources/docs/triton_cuda_docs",
            "sketch": "resources/docs/sketch_docs"
        },
        "verify_timeout": 300,
        "profile_settings": {"run_times": 50, "warmup_times": 5}
    }

class LLMTraceCallback(BaseCallbackHandler):
    """记录 LLM 请求/响应到日志文件"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.request_count = 0
        self.tool_calls: List[Dict[str, Any]] = []
        self.current_round_tools = 0  # 本轮 tool 调用数
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        
        self._init_log_file()
    
    def _init_log_file(self):
        """初始化日志文件"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"# LLM Trace Log\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write(f"# ============================================================\n")
            f.write(f"#\n")
            f.write(f"# 本日志记录 ReAct Agent 的多轮对话过程。\n")
            f.write(f"#\n")
            f.write(f"# 每个 [LLM REQUEST #N] 段落包含该轮次发送给 LLM 的完整 messages：\n")
            f.write(f"#   - Message 0 (system): System Prompt，包含 Skills 元数据\n")
            f.write(f"#   - Message 1+ (user/assistant/tool): 对话历史\n")
            f.write(f"#\n")
            f.write(f"# 要查看 Skill 是否被加载，搜索 '[TOOL CALL] read_file'，\n")
            f.write(f"# 查看其 file_path 是否包含 'skills/' 或 'SKILL.md'。\n")
            f.write(f"#\n")
            f.write(f"# ============================================================\n\n")
        print(f"📝 日志文件: {self.log_file}")
    
    def _log(self, content: str):
        """写入日志文件"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
    
    @staticmethod
    def _get_role(msg: BaseMessage) -> str:
        """获取消息角色"""
        type_name = type(msg).__name__
        if type_name.endswith("SystemMessage"):
            return "system"
        if type_name.endswith("HumanMessage"):
            return "user"
        if type_name.endswith("AIMessage"):
            return "assistant"
        if type_name.endswith("ToolMessage"):
            return "tool"
        return type_name
    
    def on_chat_model_start(
        self, 
        serialized: Dict[str, Any], 
        messages: Sequence[Sequence[BaseMessage]], 
        **kwargs
    ) -> None:
        """记录 LLM 请求"""
        self.request_count += 1
        self.current_round_tools = 0  # 每轮 LLM 请求重置
        timestamp = datetime.now().isoformat()
        model_name = serialized.get("kwargs", {}).get("model_name", "unknown")
        batch = messages[0] if messages else []
        
        self._log("\n" + "=" * 100)
        self._log(f"[LLM REQUEST #{self.request_count}]")
        self._log(f"Time: {timestamp}")
        self._log(f"Model: {model_name}")
        self._log(f"Messages count: {len(batch)}")
        self._log("=" * 100)
        
        for i, msg in enumerate(batch):
            role = self._get_role(msg)
            content = str(getattr(msg, 'content', ''))
            tool_calls = getattr(msg, 'tool_calls', None)
            
            # 标记重要信息
            if role == "system":
                self._log(f"\n[Message {i}] role={role} (← System Prompt，包含 Skills 元数据)")
                # 检查 Skills 是否在 prompt 中
                if "## Skills" in content:
                    self._log("   ✓ Skills 元数据已注入")
            elif role == "tool":
                tool_name = getattr(msg, 'name', 'unknown')
                if 'skills/' in content or '---\nname:' in content:
                    self._log(f"\n[Message {i}] role={role}, tool={tool_name} (← Skill 完整内容已加载！)")
                else:
                    self._log(f"\n[Message {i}] role={role}, tool={tool_name}")
            else:
                self._log(f"\n[Message {i}] role={role}, len={len(content)}")
            
            self._log("-" * 80)
            self._log(content)
            
            if tool_calls:
                self._log(f"\nTool calls: {json.dumps(tool_calls, ensure_ascii=False, indent=2)}")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """记录 LLM 响应"""
        timestamp = datetime.now().isoformat()
        
        self._log("\n" + "-" * 100)
        self._log(f"[LLM RESPONSE] Time: {timestamp}")
        self._log("-" * 100)
        
        # 尝试从 llm_output 获取 token 统计（DeepSeek 格式）
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                input_tokens = token_usage.get("prompt_tokens", 0) or 0
                output_tokens = token_usage.get("completion_tokens", 0) or 0
                reasoning_tokens = token_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_reasoning_tokens += reasoning_tokens
                self._log(f"\n📊 Token: input={input_tokens}, output={output_tokens}, reasoning={reasoning_tokens}")
        
        for gen in response.generations:
            for g in gen:
                text = ""
                reasoning = ""
                tool_calls = None
                
                if hasattr(g, 'message'):
                    msg = g.message
                    text = getattr(msg, 'content', str(g))
                    tool_calls = getattr(msg, 'tool_calls', None)
                    
                    if hasattr(msg, 'additional_kwargs'):
                        kwargs_dict = msg.additional_kwargs or {}
                        reasoning = kwargs_dict.get('reasoning_content', '') or ""
                elif hasattr(g, 'text') and g.text:
                    text = g.text
                else:
                    text = str(g)
                
                # 1. Reasoning
                if reasoning:
                    self._log(f"\n🧠 Reasoning ({len(reasoning)} chars):")
                    self._log(reasoning)
                
                # 2. Content
                self._log(f"\n💬 Response ({len(text)} chars):")
                self._log(text if text else "(empty)")
                
                # 3. Tool Calls - 重点检查 user_requirements
                if tool_calls:
                    self._log(f"\n🔧 Tool calls ({len(tool_calls)} 个):")
                    for i, tc in enumerate(tool_calls):
                        tc_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                        tc_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                        tc_id = tc.get('id', '') if isinstance(tc, dict) else getattr(tc, 'id', '')
                        
                        is_sub_agent = tc_name.startswith("call_") and tc_name != "call_op_task_builder"
                        
                        self._log(f"\n  [{i}] {tc_name}")
                        self._log(f"      id: {tc_id}")
                        self._log(f"      args: {json.dumps(tc_args, ensure_ascii=False, indent=8)}")
                        
                        # 检查子Agent的 user_requirements
                        if is_sub_agent:
                            user_req = tc_args.get("user_requirements", "")
                            if user_req:
                                self._log(f"      ⭐ user_requirements: {user_req}")
                            else:
                                self._log(f"      ⚠️ user_requirements: (空)")
                else:
                    self._log(f"\n🔧 Tool calls: None")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """记录 Tool 调用开始"""
        tool_name = serialized.get("name", "unknown")
        timestamp = datetime.now().isoformat()
        
        self.tool_calls.append({"name": tool_name, "input": input_str})
        self.current_round_tools += 1
        
        is_skill_load = tool_name == "read_file" and ("skills/" in input_str or "SKILL.md" in input_str)
        is_sub_agent = tool_name.startswith("call_")
        
        self._log("\n" + "-" * 60)
        if is_skill_load:
            self._log(f"[TOOL CALL #{len(self.tool_calls)}] {tool_name} (← 加载 Skill！)")
        elif is_sub_agent:
            self._log(f"[TOOL CALL #{len(self.tool_calls)}] {tool_name} (← 子Agent调用！)")
        else:
            self._log(f"[TOOL CALL #{len(self.tool_calls)}] {tool_name}")
        self._log(f"Time: {timestamp}")
        self._log("-" * 60)
        self._log(f"Input: {input_str}")
        
        # 检查子Agent调用中的 user_requirements
        if is_sub_agent and "user_requirements" in input_str:
            try:
                args = json.loads(input_str) if input_str.startswith("{") else {}
                user_req = args.get("user_requirements", "")
                if user_req:
                    self._log(f"\n⭐ user_requirements 已传递: {user_req}")
                    print(f"   ⭐ user_requirements: {user_req}")
                else:
                    self._log(f"\n⚠️ user_requirements 为空")
            except:
                pass
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """记录 Tool 调用结束"""
        output_str = str(output) if output else ""
        self._log(f"Output ({len(output_str)} chars):")
        self._log(output_str)
    
    def write_summary(self):
        """写入摘要"""
        self._log("\n" + "=" * 100)
        self._log("📊 SUMMARY")
        self._log("=" * 100)
        self._log(f"  LLM 请求次数: {self.request_count}")
        self._log(f"  Tool 调用次数: {len(self.tool_calls)}")
        self._log(f"  Total input tokens: {self.total_input_tokens}")
        self._log(f"  Total output tokens: {self.total_output_tokens}")
        self._log(f"  Total reasoning tokens: {self.total_reasoning_tokens}")
        
        # 统计 Skill 加载
        skill_loads = [tc for tc in self.tool_calls if tc['name'] == 'read_file' and 'skills/' in tc['input']]
        if skill_loads:
            self._log(f"\n  📚 Skill 加载 ({len(skill_loads)} 次):")
            for tc in skill_loads:
                self._log(f"     - {tc['input']}")
        
        self._log("=" * 100)


def _build_thinking_extra_body(thinking_mode: str = None) -> dict:
    """构建 DeepSeek thinking 模式的 extra_body"""
    if thinking_mode is None:
        return None
    
    normalized = str(thinking_mode).strip().lower()
    
    if normalized in {"enabled", "disabled"}:
        return {"thinking": {"type": normalized}}
    
    if normalized in {"true", "false"}:
        return {"chat_template_kwargs": {"thinking": normalized == "true"}}
    
    return None


def create_deepseek_llm(callback: LLMTraceCallback):
    """创建 ChatDeepSeek（thinking mode 启用时使用 ThinkingAwareChatDeepSeek）"""
    import httpx
    from langchain_deepseek import ChatDeepSeek
    
    env_api_key = os.getenv("AKG_AGENTS_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    env_model_name = os.getenv("AKG_AGENTS_MODEL_NAME", "deepseek-chat")
    env_enable_think = os.getenv("AKG_AGENTS_MODEL_ENABLE_THINK")
    
    if not env_api_key:
        raise RuntimeError("需要配置 AKG_AGENTS_API_KEY 或 DEEPSEEK_API_KEY")
    
    masked_key = env_api_key[:8] + "*" * max(0, len(env_api_key) - 12) + env_api_key[-4:] if len(env_api_key) > 12 else "***"
    
    extra_body = _build_thinking_extra_body(env_enable_think)
    
    # 判断是否启用 thinking mode
    thinking_enabled = (
        extra_body
        and extra_body.get("thinking", {}).get("type") == "enabled"
    )
    
    print("\n" + "=" * 60)
    if thinking_enabled:
        print("🔧 创建 ThinkingAwareChatDeepSeek（支持 thinking mode）")
    else:
        print("🔧 创建 ChatDeepSeek")
    print(f"   model: {env_model_name}")
    print(f"   api_key: {masked_key}")
    if env_enable_think:
        print(f"   thinking_mode: {env_enable_think}")
    print("=" * 60)
    
    timeout = httpx.Timeout(60, read=60 * 20)
    
    model_kwargs = {
        "model": env_model_name,
        "api_key": env_api_key,
        "temperature": 0.2,
        "max_tokens": 8192,
        "callbacks": [callback],
    }
    
    if extra_body:
        model_kwargs["extra_body"] = extra_body
    
    if thinking_enabled:
        # 使用支持 thinking mode 的自定义 ChatModel
        from akg_agents.core.llm.thinking_chat_model import ThinkingAwareChatDeepSeek
        llm = ThinkingAwareChatDeepSeek(timeout=timeout, **model_kwargs)
        print(f"✅ ThinkingAwareChatDeepSeek 创建成功\n")
    else:
        # 使用标准 ChatDeepSeek
        http_client = httpx.Client(verify=False, timeout=timeout)
        async_http_client = httpx.AsyncClient(verify=False, timeout=timeout)
        model_kwargs["http_client"] = http_client
        model_kwargs["http_async_client"] = async_http_client
        llm = ChatDeepSeek(**model_kwargs)
        print(f"✅ ChatDeepSeek 创建成功\n")
    
    return llm

async def register_worker(backend: str = "cuda", arch: str = "a100", device_ids: list = None) -> bool:
    """
    注册 Worker（用于执行 SubAgent 工具）
    
    Args:
        backend: 后端类型（cuda/ascend）
        arch: 硬件架构（a100/910b）
        device_ids: 设备 ID 列表
        
    Returns:
        是否注册成功
    """
    if device_ids is None:
        device_ids = [0]
    
    print(f"\n⚙️  注册 Worker: backend={backend}, arch={arch}, devices={device_ids}")
    
    try:
        from akg_agents.core.worker.manager import register_local_worker
        await register_local_worker(
            device_ids=device_ids,
            backend=backend,
            arch=arch
        )
        print("✅ Worker 注册成功")
        return True
    except Exception as e:
        print(f"⚠️  Worker 注册失败: {e}")
        print("   注意: 没有 Worker 时，call_codeonly 等 SubAgent 工具将无法执行")
        print("   如果只是测试 LLM 交互流程，可以继续")
        return False

async def run_multi_turn_conversation(
    register_worker_flag: bool = True,
    backend: str = "cuda",
    arch: str = "a100",
    device_ids: list = None
):
    """
    交互式多轮对话
    
    Args:
        register_worker_flag: 是否注册 Worker
        backend: 后端类型
        arch: 硬件架构
        device_ids: 设备 ID 列表
    """
    from langgraph.types import Command
    
    if device_ids is None:
        device_ids = [0]
    
    print("\n" + "=" * 80)
    print("🚀 ReActAgent 多轮对话测试")
    print("=" * 80)
    
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = DEFAULT_LOG_DIR / f"conversation_{timestamp}.log"
    callback = LLMTraceCallback(log_file)
    
    # 注册 Worker
    worker_registered = False
    if register_worker_flag:
        worker_registered = await register_worker(backend, arch, device_ids)
    else:
        print("\n⚠️  跳过 Worker 注册（--no-worker）")
        print("   注意: call_codeonly 等 SubAgent 工具将无法执行")
    
    # 创建 LLM
    print("\n⚙️  创建 Agent...")
    try:
        llm = create_deepseek_llm(callback)
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请安装: pip install langchain-deepseek")
        return
    
    config = get_test_config(use_full_config=worker_registered)
    
    # 创建 Agent
    agent = MainOpAgent(
        config=config,
        model=llm,
        framework="torch",
        backend=backend,
        arch=arch,
        dsl="triton"
    )
    
    print(f"✅ Agent 创建成功")
    print(f"   Tools ({len(agent.tools)}): {[t.name for t in agent.tools[:5]]}...")
    
    # 显示可用 Skills 和 System Prompt 中的 Skills 部分
    if agent.skill_loader:
        skills = agent.skill_loader.skills
        print(f"\n📚 可用 Skills ({len(skills)} 个):")
        for skill in skills:
            print(f"   • {skill['name']}: {skill['description'][:40]}...")
            print(f"     路径: {skill['path']}")
        
        # 验证 Skills 是否注入到 System Prompt
        system_prompt = agent.get_system_prompt()
        if "## Skills" in system_prompt:
            print(f"\n✅ Skills 元数据已注入 System Prompt")
            # 显示 Skills 部分
            skills_start = system_prompt.find("## Skills")
            skills_section = system_prompt[skills_start:skills_start+500]
            print(f"   预览:\n{skills_section[:300]}...")
        else:
            print(f"\n⚠️ Skills 未注入 System Prompt")
    
    print("\n" + "-" * 80)
    print("💬 开始对话（输入 'quit' 退出）")
    print("💡 提示: 输入 '帮我生成一个 ReLU 算子' 可能触发 kernel-workflow Skill")
    print(f"📝 日志文件: {log_file}")
    print("-" * 80)
    
    # 对话配置
    thread_id = str(uuid.uuid4())
    stream_config = {"configurable": {"thread_id": thread_id}}
    
    pending_interrupt = False
    
    while True:
        try:
            if pending_interrupt:
                user_input = input("\n👤 请输入回复: ").strip()
            else:
                user_input = input("\n👤 请输入: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\n🔄 处理中...")
            
            # 构建输入
            if pending_interrupt:
                inputs = Command(resume=user_input)
                pending_interrupt = False
            else:
                inputs = {"messages": [{"role": "user", "content": user_input}]}
            
            # 执行 ReAct 循环
            current_action = None
            loop_count = 0
            skills_loaded = []
            
            async for event in agent.agent.astream(inputs, stream_config, stream_mode="updates"):
                if isinstance(event, dict):
                    # 处理 model 节点输出
                    if "model" in event or "agent" in event:
                        loop_count += 1
                        node_output = event.get("model") or event.get("agent")
                        messages = node_output.get("messages", [])
                        
                        print(f"\n{'─' * 20} Loop {loop_count} {'─' * 20}")
                        
                        for msg in messages:
                            content = str(getattr(msg, 'content', ''))
                            tool_calls = getattr(msg, 'tool_calls', [])
                            
                            # 显示 reasoning_content
                            if hasattr(msg, 'additional_kwargs'):
                                kwargs_dict = msg.additional_kwargs or {}
                                reasoning = kwargs_dict.get('reasoning_content', '')
                                if reasoning:
                                    display = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                                    print(f"\n🧠 Think: {display}")
                            
                            # 显示 content
                            if content:
                                display = content[:200] + "..." if len(content) > 200 else content
                                print(f"\n🤖 AI: {display}")
                            
                            # 显示 Tool 调用
                            for tc in tool_calls or []:
                                tc_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                                tc_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                
                                if tc_name == "read_file":
                                    file_path = tc_args.get('file_path', '')
                                    if 'skills/' in file_path:
                                        skill_name = file_path.split('skills/')[-1].split('/')[0]
                                        skills_loaded.append(skill_name)
                                        print(f"   📚 加载 Skill: {skill_name}")
                                    else:
                                        print(f"   🔧 {tc_name}")
                                elif tc_name == "ask_user":
                                    message = tc_args.get('message', '')[:100]
                                    print(f"   💬 询问用户: {message}...")
                                    current_action = "ask_user"
                                elif tc_name == "finish":
                                    answer = tc_args.get('final_answer', '')[:200]
                                    print(f"\n✅ 完成: {answer}...")
                                elif tc_name.startswith("call_"):
                                    # 子Agent调用，显示 user_requirements
                                    user_req = tc_args.get('user_requirements', '')
                                    op_name = tc_args.get('op_name', '')
                                    if user_req:
                                        print(f"   🚀 {tc_name}(op={op_name})")
                                        print(f"      ⭐ user_requirements: {user_req}")
                                    else:
                                        print(f"   🚀 {tc_name}(op={op_name})")
                                else:
                                    print(f"   🔧 {tc_name}")
                    
                    # 处理 tools 节点输出
                    if "tools" in event:
                        messages = event["tools"].get("messages", [])
                        for msg in messages:
                            tool_name = getattr(msg, 'name', 'unknown')
                            content = str(getattr(msg, 'content', ''))
                            
                            if tool_name == "read_file" and ('skills/' in content or '---\nname:' in content):
                                print(f"   📥 Skill 内容已加载")
                            else:
                                print(f"   📥 {tool_name} 完成")
            
            # 检查是否需要等待用户输入
            if current_action == "ask_user":
                pending_interrupt = True
                print("\n   ⏸️  等待您的回复...")
            
            # 本轮统计
            print(f"\n📊 累计: {callback.request_count} 次 LLM, {len(callback.tool_calls)} 次 Tool")
            print(f"   Token: in={callback.total_input_tokens}, out={callback.total_output_tokens}, think={callback.total_reasoning_tokens}")
            if skills_loaded:
                print(f"   📚 Skills: {', '.join(skills_loaded)}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ 检测到 Ctrl+C")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 写入摘要
    callback.write_summary()
    
    print("\n" + "=" * 80)
    print("📊 对话结束")
    print(f"   LLM 请求: {callback.request_count} 次")
    print(f"   Tool 调用: {len(callback.tool_calls)} 次")
    print(f"   Input tokens: {callback.total_input_tokens}")
    print(f"   Output tokens: {callback.total_output_tokens}")
    print(f"   Reasoning tokens: {callback.total_reasoning_tokens}")
    print(f"\n📝 完整日志: {log_file}")
    print("=" * 80)

def print_help():
    """打印帮助信息"""
    print("""
ReActAgent 多轮对话测试

日志目录:
    ~/akg_agents_llm_trace_logs/

    """)


if __name__ == "__main__":
    # 解析命令行参数
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        sys.exit(0)
    
    if not check_llm_env():
        print("❌ 需要配置 API Key:")
        print("   export AKG_AGENTS_API_KEY=your-deepseek-api-key")
        print("   # 或")
        print("   export DEEPSEEK_API_KEY=your-deepseek-api-key")
        sys.exit(1)
    
    # 解析参数
    register_worker_flag = "--no-worker" not in sys.argv
    backend = "cuda"
    arch = "a100"
    
    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
        if arg == "--arch" and i + 1 < len(sys.argv):
            arch = sys.argv[i + 1]
    
    asyncio.run(run_multi_turn_conversation(
        register_worker_flag=register_worker_flag,
        backend=backend,
        arch=arch
    ))
