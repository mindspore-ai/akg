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
ReActAgent 多轮对话测试 - C++ CPU (x86_64)

运行方式:
    python tests/ut/test_react_agent_cpp_cpu.py

日志目录: ~/aikg_llm_trace_logs/
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

from ai_kernel_generator.core.agent.react_agent import MainOpAgent

DEFAULT_LOG_DIR = Path.home() / "aikg_llm_trace_logs"


def check_llm_env() -> bool:
    """检查 LLM 环境变量"""
    return bool(os.getenv("AIKG_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))


def get_cpp_cpu_config() -> dict:
    """获取 C++ CPU 配置"""
    try:
        from ai_kernel_generator.config.config_validator import load_config
        from ai_kernel_generator import get_project_root
        
        config_path = os.path.join(
            get_project_root(), 
            "config", 
            "vllm_cpp_coderonly_config.yaml"
        )
        if os.path.exists(config_path):
            config = load_config(config_path=config_path)
            print(f"✅ 加载配置: {config_path}")
            return config
    except Exception as e:
        print(f"⚠️ 加载配置失败: {e}")
    
    # 最小配置
    import tempfile
    return {
        "agent_model_config": {"default": "default"},
        "log_dir": tempfile.gettempdir(),
        "docs_dir": {
            "designer": "resources/docs/sketch_docs",
            "coder": "resources/docs/cpp_docs",
            "sketch": "resources/docs/sketch_docs"
        },
        "verify_timeout": 300,
    }


class LLMTraceCallback(BaseCallbackHandler):
    """记录 LLM 请求/响应"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.request_count = 0
        self.tool_calls: List[Dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"# LLM Trace - C++ CPU (x86_64)\n")
            f.write(f"# {datetime.now().isoformat()}\n\n")
        print(f"📝 日志: {log_file}")
    
    def _log(self, content: str):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
    
    def on_chat_model_start(self, serialized: Dict, messages: Sequence[Sequence[BaseMessage]], **kwargs):
        self.request_count += 1
        batch = messages[0] if messages else []
        self._log(f"\n{'='*80}\n[REQUEST #{self.request_count}] messages={len(batch)}\n{'='*80}")
        
        for i, msg in enumerate(batch):
            role = type(msg).__name__.replace("Message", "").lower()
            content = str(getattr(msg, 'content', ''))[:500]
            self._log(f"\n[{i}] {role}: {content}...")
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        self._log(f"\n{'-'*80}\n[RESPONSE]\n{'-'*80}")
        
        # Token 统计
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
        
        for gen in response.generations:
            for g in gen:
                if hasattr(g, 'message'):
                    msg = g.message
                    text = getattr(msg, 'content', '')
                    tool_calls = getattr(msg, 'tool_calls', None)
                    
                    self._log(f"\n💬 Content: {text[:300]}...")
                    
                    if tool_calls:
                        self._log(f"\n🔧 Tools ({len(tool_calls)}):")
                        for tc in tool_calls:
                            name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                            args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                            self._log(f"  - {name}: {json.dumps(args, ensure_ascii=False)[:200]}")
                            
                            # 检查 user_requirements
                            if name.startswith("call_") and name != "call_op_task_builder":
                                user_req = args.get("user_requirements", "")
                                if user_req:
                                    self._log(f"    ⭐ user_requirements: {user_req}")
    
    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs):
        name = serialized.get("name", "unknown")
        self.tool_calls.append({"name": name, "input": input_str})
        self._log(f"\n[TOOL] {name}")
        
        # 检查子Agent的 user_requirements
        if name.startswith("call_") and "user_requirements" in input_str:
            try:
                args = json.loads(input_str) if input_str.startswith("{") else {}
                user_req = args.get("user_requirements", "")
                if user_req:
                    self._log(f"  ⭐ user_requirements: {user_req}")
                    print(f"   ⭐ user_requirements: {user_req}")
            except:
                pass
    
    def on_tool_end(self, output: str, **kwargs):
        self._log(f"  Output: {str(output)[:200]}...")


def create_llm(callback: LLMTraceCallback):
    """创建 LLM"""
    import httpx
    
    env_api_key = os.getenv("AIKG_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    env_model = os.getenv("AIKG_MODEL_NAME", "deepseek-chat")
    env_think = os.getenv("AIKG_MODEL_ENABLE_THINK")
    
    if not env_api_key:
        raise RuntimeError("需要 AIKG_API_KEY 或 DEEPSEEK_API_KEY")
    
    print(f"\n🔧 创建 LLM: {env_model}")
    
    timeout = httpx.Timeout(60, read=60 * 20)
    
    model_kwargs = {
        "model": env_model,
        "api_key": env_api_key,
        "temperature": 0.2,
        "max_tokens": 8192,
        "callbacks": [callback],
    }
    
    # 检查是否启用 thinking mode
    thinking_enabled = env_think and env_think.lower() in {"enabled", "true"}
    
    if thinking_enabled:
        model_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        from ai_kernel_generator.core.llm.thinking_chat_model import ThinkingAwareChatDeepSeek
        return ThinkingAwareChatDeepSeek(timeout=timeout, **model_kwargs)
    else:
        from langchain_deepseek import ChatDeepSeek
        http_client = httpx.Client(verify=False, timeout=timeout)
        async_http_client = httpx.AsyncClient(verify=False, timeout=timeout)
        model_kwargs["http_client"] = http_client
        model_kwargs["http_async_client"] = async_http_client
        return ChatDeepSeek(**model_kwargs)


async def register_worker(arch: str = "x86_64") -> bool:
    """注册 Worker"""
    print(f"\n⚙️ 注册 Worker: backend=cpu, arch={arch}")
    try:
        from ai_kernel_generator.core.worker.manager import register_local_worker
        await register_local_worker(device_ids=[0], backend="cpu", arch=arch)
        print("✅ Worker 注册成功")
        return True
    except Exception as e:
        print(f"⚠️ Worker 注册失败: {e}")
        return False


async def main():
    """主函数"""
    from langgraph.types import Command
    
    # 配置
    backend = "cpu"
    arch = "x86_64"
    dsl = "cpp"
    framework = "torch"
    
    print("\n" + "=" * 60)
    print(f"🚀 ReActAgent - C++ CPU ({arch})")
    print("=" * 60)
    
    # 日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = DEFAULT_LOG_DIR / f"cpp_cpu_{timestamp}.log"
    callback = LLMTraceCallback(log_file)
    
    # 注册 Worker
    worker_ok = await register_worker(arch)
    
    # 创建 LLM
    llm = create_llm(callback)
    
    # 配置
    config = get_cpp_cpu_config()
    
    # 创建 Agent
    agent = MainOpAgent(
        config=config,
        model=llm,
        framework=framework,
        backend=backend,
        arch=arch,
        dsl=dsl
    )
    
    print(f"✅ Agent 创建成功")
    print(f"   DSL: {dsl}, Backend: {backend}, Arch: {arch}")
    print(f"   Tools: {[t.name for t in agent.tools[:5]]}...")
    
    print("\n" + "-" * 60)
    print("💬 开始对话 (输入 'quit' 退出)")
    print("💡 示例: '生成一个 ReLU 算子' 或 '生成 relu，使用 AVX2 向量化'")
    print("-" * 60)
    
    thread_id = str(uuid.uuid4())
    stream_config = {"configurable": {"thread_id": thread_id}}
    pending_interrupt = False
    
    while True:
        try:
            prompt = "\n👤 回复: " if pending_interrupt else "\n👤 输入: "
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\n🔄 处理中...")
            
            if pending_interrupt:
                inputs = Command(resume=user_input)
                pending_interrupt = False
            else:
                inputs = {"messages": [{"role": "user", "content": user_input}]}
            
            loop_count = 0
            current_action = None
            
            async for event in agent.agent.astream(inputs, stream_config, stream_mode="updates"):
                if not isinstance(event, dict):
                    continue
                
                if "model" in event or "agent" in event:
                    loop_count += 1
                    node_output = event.get("model") or event.get("agent")
                    messages = node_output.get("messages", [])
                    
                    print(f"\n{'─' * 15} Loop {loop_count} {'─' * 15}")
                    
                    for msg in messages:
                        content = str(getattr(msg, 'content', ''))
                        tool_calls = getattr(msg, 'tool_calls', [])
                        
                        # reasoning
                        if hasattr(msg, 'additional_kwargs'):
                            reasoning = (msg.additional_kwargs or {}).get('reasoning_content', '')
                            if reasoning:
                                print(f"\n🧠 Think: {reasoning[:150]}...")
                        
                        if content:
                            print(f"\n🤖 AI: {content[:150]}...")
                        
                        for tc in tool_calls or []:
                            tc_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                            tc_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                            
                            if tc_name == "ask_user":
                                print(f"   💬 询问: {tc_args.get('message', '')[:80]}...")
                                current_action = "ask_user"
                            elif tc_name == "finish":
                                print(f"\n✅ 完成: {tc_args.get('final_answer', '')[:150]}...")
                            elif tc_name.startswith("call_"):
                                op_name = tc_args.get('op_name', '')
                                user_req = tc_args.get('user_requirements', '')
                                print(f"   🚀 {tc_name}(op={op_name})")
                                if user_req:
                                    print(f"      ⭐ user_requirements: {user_req}")
                            elif tc_name == "read_file":
                                path = tc_args.get('file_path', '')
                                if 'skills/' in path:
                                    print(f"   📚 加载 Skill: {path.split('skills/')[-1].split('/')[0]}")
                                else:
                                    print(f"   📄 读取: {path}")
                            else:
                                print(f"   🔧 {tc_name}")
                
                if "tools" in event:
                    for msg in event["tools"].get("messages", []):
                        name = getattr(msg, 'name', 'unknown')
                        print(f"   📥 {name} 完成")
            
            if current_action == "ask_user":
                pending_interrupt = True
                print("\n   ⏸️ 等待回复...")
            
            print(f"\n📊 累计: {callback.request_count} LLM, {len(callback.tool_calls)} Tool")
            print(f"   Token: in={callback.total_input_tokens}, out={callback.total_output_tokens}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Ctrl+C")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📊 结束")
    print(f"   LLM: {callback.request_count}, Tool: {len(callback.tool_calls)}")
    print(f"   Token: in={callback.total_input_tokens}, out={callback.total_output_tokens}")
    print(f"📝 日志: {log_file}")
    print("=" * 60)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    if not check_llm_env():
        print("❌ 需要: export AIKG_API_KEY=your-key")
        sys.exit(1)
    
    asyncio.run(main())

