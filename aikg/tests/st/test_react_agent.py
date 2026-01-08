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

测试 ReAct 架构的 Agent。
1. 直接跑demo：
python tests/st/test_react_agent.py


环境变量配置（需要 LLM API）：
    export AIKG_API_KEY=your-api-key
    export AIKG_MODEL_NAME=deepseek-chat           # 可选，默认 deepseek-chat
    export AIKG_BASE_URL=https://api.deepseek.com  # 可选
    export AIKG_MODEL_ENABLE_THINK=enabled         # 可选，启用思考模式
"""

import os
import uuid
import pytest
import logging

logger = logging.getLogger(__name__)


def get_test_config(use_full_config: bool = False) -> dict:
    import tempfile
    
    if use_full_config:
        try:
            from ai_kernel_generator.utils.common_utils import load_yaml
            from ai_kernel_generator import get_project_root
            config_path = os.path.join(get_project_root(), "config", "default_triton_cuda_config.yaml")
            if os.path.exists(config_path):
                config = load_yaml(config_path)
                logger.info(f"Loaded full config from: {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load full config: {e}, using minimal config")
    
    return {
        "agent_model_config": {
            "default": "default"
        },
        "log_dir": os.path.join(tempfile.gettempdir(), "aikg_test_logs"),
        "docs_dir": {
            "designer": "resources/docs/sketch_docs",
            "coder": "resources/docs/triton_cuda_docs",
            "sketch": "resources/docs/sketch_docs"
        },
        "verify_timeout": 300,
        "profile_settings": {
            "run_times": 50,
            "warmup_times": 5
        }
    }


def check_llm_env() -> bool:
    """检查 LLM 环境变量是否配置（只需要 API Key）"""
    env_api_key = os.getenv("AIKG_API_KEY")
    return bool(env_api_key)


def _build_thinking_extra_body(thinking_mode: str = None) -> dict:
    """
    构建 DeepSeek thinking 模式的 extra_body
    
    """
    if thinking_mode is None:
        return None
    
    normalized = str(thinking_mode).strip().lower()
    
    if normalized in {"enabled", "disabled"}:
        return {"thinking": {"type": normalized}}
    
    if normalized in {"true", "false"}:
        return {"chat_template_kwargs": {"thinking": normalized == "true"}}
    
    return None


def create_llm_chat_openai():
    """
    创建 LangChain ChatOpenAI，但是这个接口不会有reasoning_content
        AIKG_MODEL_ENABLE_THINK: 启用思考模式（可选，enabled/disabled）
    """
    import httpx
    from langchain_openai import ChatOpenAI
    
    env_model_name = os.getenv("AIKG_MODEL_NAME", "deepseek-chat")
    env_api_key = os.getenv("AIKG_API_KEY")
    env_base_url = os.getenv("AIKG_BASE_URL", "https://api.deepseek.com")
    env_enable_think = os.getenv("AIKG_MODEL_ENABLE_THINK")
    
    if not env_api_key:
        raise RuntimeError(
            "需要配置 API Key:\n"
            "  export AIKG_API_KEY=your-api-key\n"
            "  export AIKG_MODEL_NAME=deepseek-chat  # 可选\n"
            "  export AIKG_BASE_URL=https://api.deepseek.com  # 可选\n"
            "  export AIKG_MODEL_ENABLE_THINK=enabled  # 可选，启用思考模式"
        )
    
    masked_key = env_api_key[:8] + "*" * (len(env_api_key) - 12) + env_api_key[-4:] if len(env_api_key) > 12 else "***"
    print("=" * 60)
    print("创建 LangChain ChatOpenAI (DeepSeek 兼容)")
    print(f"  model: {env_model_name}")
    print(f"  base_url: {env_base_url}")
    print(f"  api_key: {masked_key}")
    if env_enable_think:
        print(f"  thinking_mode: {env_enable_think}")
    print("=" * 60)
    
    timeout = httpx.Timeout(60, read=60 * 20)
    http_client = httpx.Client(verify=False, timeout=timeout)
    async_http_client = httpx.AsyncClient(verify=False, timeout=timeout)
    
    extra_body = _build_thinking_extra_body(env_enable_think)
    
    model_kwargs = {
        "model": env_model_name,
        "api_key": env_api_key,
        "base_url": env_base_url,
        "temperature": 0.2,
        "max_tokens": 8192,
        "http_client": http_client,
        "http_async_client": async_http_client,
        "max_retries": 3,
    }
    
    # 启用了 thinking 模式的话加进去吧
    if extra_body:
        model_kwargs["extra_body"] = extra_body
        print(f"  extra_body: {extra_body}")
    
    model = ChatOpenAI(**model_kwargs)
    
    print(f"✅ ChatOpenAI 创建成功: {env_model_name}")
    return model


def create_llm_chat_deepseek():
    """
    创建 LangChain ChatDeepSeek
    """
    import httpx
    from langchain_deepseek import ChatDeepSeek
    
    env_model_name = os.getenv("AIKG_MODEL_NAME", "deepseek-chat")
    env_api_key = os.getenv("AIKG_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    env_enable_think = os.getenv("AIKG_MODEL_ENABLE_THINK")
    
    if not env_api_key:
        raise RuntimeError(
            "需要配置 API Key:\n"
            "  export AIKG_API_KEY=your-api-key\n"
            "  # 或\n"
            "  export DEEPSEEK_API_KEY=your-api-key\n"
            "  export AIKG_MODEL_NAME=deepseek-reasoner  # 推荐使用 reasoner 模型\n"
            "  export AIKG_MODEL_ENABLE_THINK=enabled  # 可选，启用思考模式"
        )
    
    masked_key = env_api_key[:8] + "*" * (len(env_api_key) - 12) + env_api_key[-4:] if len(env_api_key) > 12 else "***"
    print("=" * 60)
    print("创建 LangChain ChatDeepSeek (原生)")
    print(f"  model: {env_model_name}")
    print(f"  api_key: {masked_key}")
    if env_enable_think:
        print(f"  thinking_mode: {env_enable_think}")
    print("=" * 60)
    
    timeout = httpx.Timeout(60, read=60 * 20)
    http_client = httpx.Client(verify=False, timeout=timeout)
    async_http_client = httpx.AsyncClient(verify=False, timeout=timeout)
    
    extra_body = _build_thinking_extra_body(env_enable_think)
    
    model_kwargs = {
        "model": env_model_name,
        "api_key": env_api_key,
        "temperature": 0.2,
        "max_tokens": 8192,
        "http_client": http_client,
        "http_async_client": async_http_client,
    }
    
    if extra_body:
        model_kwargs["extra_body"] = extra_body
        print(f"  extra_body: {extra_body}")
    
    model = ChatDeepSeek(**model_kwargs)
    
    print(f"✅ ChatDeepSeek 创建成功: {env_model_name}")
    return model


def create_llm():
    """
    创建 LangChain ChatModel
    """

    return create_llm_chat_openai()


async def demo_react_agent(
    backend: str = "cuda",
    arch: str = "a100",
    device_ids: list = None,
    register_worker: bool = True
):
    """
    ReAct 流程
    
    同一 thread_id 内的多轮对话会保留上下文，退出对话后，记忆会丢失
    """
    from ai_kernel_generator.core.agent.react_agent import MainOpAgent
    
    TASK_BUILD_TOOLS = {"call_op_task_builder"}  
    SUB_AGENT_TOOLS = {"call_codeonly", "call_evolve", "call_adaptive_search", "call_kernel_verifier"}
    INTERACT_TOOLS = {"ask_user", "finish"}
    SKILL_KEYWORDS = ["skills/", "SKILL.md"]
    
    if device_ids is None:
        device_ids = [0]
    
    thread_id = str(uuid.uuid4())
    stream_config = {"configurable": {"thread_id": thread_id}}
    
    print("\n" + "=" * 80)
    print("🚀 ReActAgent Demo 演示")
    print(f"📝 Session ID: {thread_id[:8]}... (用于多轮对话记忆)")
    print("=" * 80)
    
    worker_registered = False
    if register_worker:
        print(f"\n⚙️  注册 Worker: backend={backend}, arch={arch}, devices={device_ids}")
        try:
            from ai_kernel_generator.core.worker.manager import register_local_worker
            await register_local_worker(
                device_ids=device_ids,
                backend=backend,
                arch=arch
            )
            print("✅ Worker 注册成功")
            worker_registered = True
        except Exception as e:
            print(f"⚠️  Worker 注册失败: {e}")
            print("   注意: 没有 Worker 时，call_codeonly 等 SubAgent 工具将无法执行")
            print("   如果只是测试 LLM 交互流程，可以继续")
    else:
        print("\n⚠️  跳过 Worker 注册（register_worker=False）")
        print("   注意: call_codeonly 等 SubAgent 工具将无法执行")
    
    model = create_llm()
    
    config = get_test_config(use_full_config=worker_registered)
    
    print("\n📦 创建 ReActAgent...")
    # 这里是生成triton，如果是别的dsl的话，自己修改吧
    agent = MainOpAgent(
        config=config,
        model=model,
        framework="torch",
        backend=backend,
        arch=arch,
        dsl="triton"
    )
    
    print(f"\n🔧 可用tools ({len(agent.tools)} 个):")
    for tool in agent.tools:
        name = getattr(tool, 'name', str(tool))
        if name in TASK_BUILD_TOOLS:
            print(f"   • [TaskBuild] {name}")
        elif name in SUB_AGENT_TOOLS:
            print(f"   • [SubAgent] {name}")
        elif name in INTERACT_TOOLS:
            print(f"   • [Interact] {name}")
        else:
            print(f"   • [Other] {name}")
    
    if agent.skill_loader:
        skills = agent.skill_loader.skills
        print(f"\n📚 可用 Skills ({len(skills)} 个):")
        for skill in skills:
            print(f"   • {skill['name']}: {skill['description'][:50]}...")
    print("   • 按 Ctrl+C 退出")
    print()
    
    while True:
        try:
            user_input = input("👤 请输入您的需求: ").strip()
        except KeyboardInterrupt:
            print("\n\n⚠️ 检测到 Ctrl+C，退出程序")
            break
        
        if not user_input:
            print("💡 请输入您的需求")
            continue
        print(f"\n🔄 开始 ReAct 流程...")
        print("-" * 80)
        
        loop_count = 0  
        tools_called = []
        skills_used = []
        task_finished = False
        current_action = None  
        
        try:
            inputs = {"messages": [{"role": "user", "content": user_input}]}
            
            async for event in agent.agent.astream(inputs, stream_config, stream_mode="updates"):
                # print(f"event: {event}")
                # aimessage
                if isinstance(event, dict):
                    # model 节点 = Think + Action（循环开始）
                    if "model" in event or "agent" in event:
                        loop_count += 1
                        
                        print(f"\n{'─' * 25} Loop {loop_count} {'─' * 25}")
                        
                        node_output = event.get("model") or event.get("agent")
                        messages = node_output.get("messages", [])
                        # print(f"messages: {messages}")
                        for msg in messages:
                            content = getattr(msg, 'content', '')
                            
                            # 获取 reasoning_content（DeepSeek 模型的think）
                            reasoning_content = ""
                            if hasattr(msg, 'additional_kwargs'):
                                kwargs = msg.additional_kwargs or {}
                                reasoning_content = kwargs.get('reasoning_content', '') or ""
                            
                            # 其他的可能显示 content
                            think_content = reasoning_content or content
                            if think_content:
                                print(f"\n🧠 [Think]")
                                display = think_content[:500] + ('...' if len(think_content) > 500 else '')
                                for line in display.split('\n'):
                                    print(f"   {line}")
                            
                            # Action: 检查工具调用
                            tool_calls = getattr(msg, 'tool_calls', [])
                            if tool_calls:
                                for tc in tool_calls:
                                    tc_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                                    tc_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                    
                                    tools_called.append(tc_name)
                                    
                                    print(f"\n⚡ [Action] {tc_name}")
                                    
                                    if tc_name == "call_op_task_builder":
                                        print(f"   📝 类型: TaskBuild（生成 Torch task）")
                                        if 'user_request' in tc_args:
                                            req = tc_args['user_request']
                                            print(f"   📥 user_request: {req[:80]}{'...' if len(req) > 80 else ''}")
                                    
                                    elif tc_name in SUB_AGENT_TOOLS:
                                        print(f"   🚀 类型: SubAgent（生成 Triton）")
                                        if 'op_name' in tc_args:
                                            print(f"   📥 op_name: {tc_args['op_name']}")
                                        if 'task_type' in tc_args:
                                            print(f"   📥 task_type: {tc_args['task_type']}")
                                    
                                    elif tc_name == "ask_user":
                                        message = tc_args.get('message', '')
                                        print(f"   📤 消息预览: {message[:100]}{'...' if len(message) > 100 else ''}")
                                    
                                    elif tc_name == "finish":
                                        print(f"   ✅ 类型: 任务完成")
                                        answer = tc_args.get('final_answer', '')
                                        print(f"   📤 结果预览: {answer[:150]}{'...' if len(answer) > 150 else ''}")
                                        task_finished = True
                                    
                                    elif tc_name == "read_file":
                                        file_path = tc_args.get('file_path', '')
                                        is_skill = any(kw in file_path for kw in SKILL_KEYWORDS)
                                        if is_skill:
                                            skill_name = "unknown"
                                            if "skills/" in file_path:
                                                parts = file_path.split("skills/")
                                                if len(parts) > 1:
                                                    skill_name = parts[1].split("/")[0]
                                            skills_used.append(skill_name)
                                            print(f"   📚 类型: 加载 Skill ({skill_name})")
                                        else:
                                            print(f"   📄 类型: 读取文件")
                                        print(f"   📥 file_path: {file_path}")
                    # toolmessage
                    if "tools" in event:
                        tools_output = event["tools"]
                        messages = tools_output.get("messages", [])
                        
                        print(f"\n👁️ [Observation] 工具执行结果")
                        for msg in messages:
                            tool_name = getattr(msg, 'name', 'unknown')
                            content = getattr(msg, 'content', '')
                            
                            if tool_name == "call_op_task_builder":
                                print(f"   📤 {tool_name}: 已生成 Torch task 代码")
                            
                            elif tool_name in SUB_AGENT_TOOLS:
                                print(f"   📤 {tool_name}: 已生成 Triton 代码")
                            
                            elif tool_name == "ask_user":
                                # ask_user 现在是阻塞式的，返回用户的实际回复
                                preview = content[:100] + ('...' if len(content) > 100 else '')
                                print(f"   📤 {tool_name}: {preview}")
                            
                            elif tool_name == "finish":
                                print(f"   📤 {tool_name}: 任务完成")
                            
                            elif tool_name == "read_file":
                                if any(kw in content for kw in ["---", "name:", "description:"]):
                                    print(f"   📤 {tool_name}: Skill 内容已加载")
                                else:
                                    preview = content[:100] + ('...' if len(content) > 100 else '')
                                    print(f"   📤 {tool_name}: {preview}")
            
            # 打印本轮总结
            print("\n" + "=" * 80)
            print("📊 本轮 ReAct 循环总结:")
            
            if tools_called:
                print(f"\n   🔧 调用的工具 ({len(tools_called)} 次):")
                for i, tool in enumerate(tools_called, 1):
                    if tool in TASK_BUILD_TOOLS:
                        print(f"      {i}. [TaskBuild] {tool}")
                    elif tool in SUB_AGENT_TOOLS:
                        print(f"      {i}. [SubAgent] {tool}")
                    elif tool in INTERACT_TOOLS:
                        print(f"      {i}. [Interact] {tool}")
                    else:
                        print(f"      {i}. [Other] {tool}")
            
            if skills_used:
                print(f"\n   📚 使用的 Skills: {', '.join(skills_used)}")
            
            print(f"\n   📈 执行轮次: {loop_count}")
            
            if task_finished:
                print("\n   ✅ 状态: 任务完成")
            else:
                print("\n   🔄 状态: 循环结束")
            
            print("=" * 80 + "\n")
        
        except Exception as e:
            print(f"\n❌ 执行异常: {e}")
            import traceback
            traceback.print_exc()
            print("   继续等待下一轮输入...")
    
    print("\n" + "=" * 80)
    print("🎉 感谢使用 ReActAgent Demo!")
    print("=" * 80)


def test_skill_loader():
    """测试 SkillLoader 基本功能"""
    from ai_kernel_generator.core.skills import SkillLoader
    
    loader = SkillLoader()
    
    skills = loader.skills
    print(f"\n📚 发现 {len(skills)} 个 Skills:")
    for skill in skills:
        print(f"   - {skill['name']}: {skill['description'][:50]}...")
        print(f"     路径: {skill.get('path', 'N/A')}")
    
    skill_names = loader.get_skill_names()
    print(f"\n📝 Skill 名称列表: {skill_names}")
    assert isinstance(skill_names, list)
    
    if skill_names:
        skill_name = skill_names[0]
        skill = loader.find_skill(skill_name)
        print(f"\n🔍 find_skill('{skill_name}'): {skill}")
        assert skill is not None
        assert skill["name"] == skill_name
        assert "path" in skill
    
    print("\n✅ SkillLoader 测试通过!")


def test_read_skill_via_read_file():
    """测试通过 read_file 工具读取 SKILL.md"""
    from ai_kernel_generator.core.tools.basic_tools import read_file
    from ai_kernel_generator.core.skills import SkillLoader
    
    loader = SkillLoader()
    skills = loader.skills
    print(f"\n📚 可用 Skills:")
    
    for skill in skills:
        print(f"   - {skill['name']}: {skill['description'][:50]}...")
        print(f"     路径: {skill.get('path', 'N/A')}")
    
    if skills:
        skill_path = skills[0].get("path")
        if skill_path:
            skill_content = read_file(skill_path)
            print(f"\n📖 read_file('{skill_path}') 返回 ({len(skill_content)} 字符):\n{skill_content[:300]}...")
            assert not skill_content.startswith("[ERROR]")
    
    print("\n✅ read_file 读取 SKILL.md 测试通过!")


def test_list_all_tools():
    """列出所有注册的工具"""
    from ai_kernel_generator.core.agent.react_agent import ReActAgent
    
    class MockModel:
        pass
    
    config = get_test_config()
    agent = ReActAgent(
        config=config,
        model=MockModel()
    )
    
    print("\n" + "=" * 60)
    print("📦 ReActAgent 工具清单")
    print("=" * 60)
    
    TASK_BUILD_TOOLS = {"call_op_task_builder"}
    SUB_AGENT_TOOLS = {"call_codeonly", "call_evolve", "call_adaptive_search", "call_kernel_verifier"}
    BASIC_TOOLS = {"ask_user", "finish", "read_file"}
    
    tool_categories = {
        "TaskBuild": [],
        "SubAgent": [],
        "Basic": [],
        "Other": []
    }
    
    for tool in agent.tools:
        name = tool.name
        desc = getattr(tool, 'description', '')[:80]
        
        if name in TASK_BUILD_TOOLS:
            tool_categories["TaskBuild"].append((name, desc))
        elif name in SUB_AGENT_TOOLS:
            tool_categories["SubAgent"].append((name, desc))
        elif name in BASIC_TOOLS:
            tool_categories["Basic"].append((name, desc))
        else:
            tool_categories["Other"].append((name, desc))
    
    for category, tools in tool_categories.items():
        if tools:
            print(f"\n📁 {category} ({len(tools)} 个):")
            for name, desc in tools:
                print(f"   • {name}")
                if desc:
                    print(f"     {desc[:60]}...")
    
    if agent.skill_loader:
        skills = agent.skill_loader.skills
        print(f"\n📚 Skills ({len(skills)} 个):")
        for skill in skills:
            print(f"   • {skill['name']}: {skill['description'][:50]}...")
    
    print("\n" + "=" * 60)
    print(f"总计: {len(agent.tools)} 个工具, {len(agent.skill_loader.skills) if agent.skill_loader else 0} 个 Skills")
    print("=" * 60)


def extract_reasoning_content(response) -> str:
    """
    从 LangChain AIMessage 中提取 reasoning_content
    
    DeepSeek Reasoner 的 reasoning_content 可能在以下位置：
    1. response.reasoning_content (某些版本直接暴露)
    2. response.additional_kwargs['reasoning_content'] (LangChain 标准方式)
    3. response.response_metadata 中的某些字段
    
    Args:
        response: LangChain AIMessage 对象
        
    Returns:
        reasoning_content 字符串，如果没有则返回空字符串
    """
    # 方式 1: 直接属性
    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        return response.reasoning_content
    
    # 方式 2: additional_kwargs (LangChain 标准)
    if hasattr(response, 'additional_kwargs'):
        kwargs = response.additional_kwargs or {}
        if 'reasoning_content' in kwargs:
            return kwargs['reasoning_content'] or ""
    
    # 方式 3: 检查 response_metadata
    if hasattr(response, 'response_metadata'):
        meta = response.response_metadata or {}
        # 某些实现可能把 reasoning 放在 metadata 中
        if 'reasoning_content' in meta:
            return meta['reasoning_content'] or ""
    
    return ""


@pytest.mark.skipif(not check_llm_env(), reason="需要配置 LLM 环境变量")
def test_create_llm():
    """
    测试 create_llm 创建模型，并详细解析响应字段
    
    重点关注 DeepSeek reasoning_content 的获取方式
    """
    print("\n" + "=" * 70)
    print("📦 测试 create_llm - DeepSeek 响应字段解析")
    print("=" * 70)
    
    llm = create_llm()
    
    print(f"\n✅ 模型类型: {type(llm).__name__}")
    
    print("\n🔄 发送测试请求: '简单介绍一下自己'")
    response = llm.invoke("简单介绍一下自己，用一句话")
    
    # ============================================================
    # 1. 主要内容
    # ============================================================
    print("\n" + "=" * 70)
    print("💬 Content (模型输出):")
    print("=" * 70)
    print(response.content)
    
    # ============================================================
    # 2. Reasoning Content (思考过程) - 关键部分
    # ============================================================
    print("\n" + "=" * 70)
    print("🧠 Reasoning Content (思考过程):")
    print("=" * 70)
    
    reasoning = extract_reasoning_content(response)
    if reasoning:
        print(reasoning)
    else:
        print("[未找到 reasoning_content]")
        print("")
        print("可能原因：")
        print("  1. 未启用 thinking 模式: export AIKG_MODEL_ENABLE_THINK=enabled")
        print("  2. 模型不支持 reasoning (如 deepseek-chat vs deepseek-reasoner)")
        print("  3. LangChain 版本未正确解析 DeepSeek 的 reasoning_content 字段")
        print("")
        print("解决方案：")
        print("  - 使用原生 openai SDK (参考 agent_base.py 路径 A)")
        print("  - 或升级 langchain-openai >= 0.3.0")
    
    # ============================================================
    # 3. Additional Kwargs (额外参数)
    # ============================================================
    print("\n" + "=" * 70)
    print("📎 Additional Kwargs:")
    print("=" * 70)
    if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
        for k, v in response.additional_kwargs.items():
            v_str = str(v)
            if len(v_str) > 500:
                v_str = v_str[:500] + "..."
            print(f"   • {k}: {v_str}")
    else:
        print("   [空]")
    
    # ============================================================
    # 4. Token 统计
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 Token 统计:")
    print("=" * 70)
    
    # 从 usage_metadata 获取
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        print(f"   • input_tokens: {getattr(usage, 'input_tokens', None)}")
        print(f"   • output_tokens: {getattr(usage, 'output_tokens', None)}")
        print(f"   • total_tokens: {getattr(usage, 'total_tokens', None)}")
        
        # 检查 output_token_details 中的 reasoning
        if hasattr(usage, 'output_token_details'):
            details = usage.output_token_details
            if details:
                reasoning_tokens = getattr(details, 'reasoning', None)
                if reasoning_tokens is None and isinstance(details, dict):
                    reasoning_tokens = details.get('reasoning')
                print(f"   • reasoning_tokens: {reasoning_tokens}")
    
    # 从 response_metadata 获取更详细的信息
    if hasattr(response, 'response_metadata') and response.response_metadata:
        meta = response.response_metadata
        token_usage = meta.get('token_usage', {})
        if token_usage:
            details = token_usage.get('completion_tokens_details', {})
            if details:
                print(f"\n   completion_tokens_details:")
                print(f"      • reasoning_tokens: {details.get('reasoning_tokens')}")
                print(f"      • accepted_prediction_tokens: {details.get('accepted_prediction_tokens')}")
    
    # ============================================================
    # 5. Response Metadata (响应元数据)
    # ============================================================
    print("\n" + "=" * 70)
    print("📋 Response Metadata:")
    print("=" * 70)
    if hasattr(response, 'response_metadata') and response.response_metadata:
        meta = response.response_metadata
        print(f"   • model_name: {meta.get('model_name')}")
        print(f"   • finish_reason: {meta.get('finish_reason')}")
        print(f"   • id: {meta.get('id')}")
    
    # ============================================================
    # 6. 所有属性列表（调试用）
    # ============================================================
    print("\n" + "=" * 70)
    print("🔍 响应对象所有非方法属性:")
    print("=" * 70)
    for attr in sorted(dir(response)):
        if not attr.startswith('_'):
            try:
                value = getattr(response, attr)
                if not callable(value):
                    type_name = type(value).__name__
                    print(f"   • {attr} ({type_name})")
            except Exception:
                pass
    
    print("\n" + "=" * 70)
    assert response.content, "模型应该返回非空响应"
    print("✅ test_create_llm 测试通过!")


@pytest.mark.skipif(not check_llm_env(), reason="需要配置 LLM 环境变量")
def test_stream_chat_openai():
    """
    测试 ChatOpenAI 流式输出
    
    使用 ChatOpenAI 连接 DeepSeek API，测试流式获取 content 和 reasoning_content。
    打印每个 chunk 的详细内容。
    
    AIMessageChunk 结构（参考 LangChain 文档）:
    ```
    AIMessageChunk {
      "id": "chatcmpl-xxx",
      "content": "The",
      "additional_kwargs": {
        "reasoning_content": "..."
      },
      "response_metadata": {...}
    }
    ```
    
    参考: https://reference.langchain.com/javascript/classes/_langchain_openai.ChatOpenAI.html
    """
    print("\n" + "=" * 70)
    print("🌊 测试 ChatOpenAI 流式输出 (打印每个 chunk)")
    print("=" * 70)
    
    llm = create_llm_chat_openai()
    
    # 收集流式输出
    full_content = ""
    full_reasoning = ""
    chunk_count = 0
    
    print("\n📤 发送请求: '1+1等于多少？请简短回答'")
    print("\n" + "-" * 70)
    print("🔄 Chunk 详情:")
    print("-" * 70)
    
    for chunk in llm.stream("1+1等于多少？请简短回答"):
        chunk_count += 1
        
        # 获取 content
        content = chunk.content or ""
        
        # 获取 reasoning_content (在 additional_kwargs 中)
        reasoning = ""
        if hasattr(chunk, 'additional_kwargs'):
            kwargs = chunk.additional_kwargs or {}
            reasoning = kwargs.get('reasoning_content', '') or ""
        
        # 打印每个 chunk 的详情
        print(f"\n[Chunk {chunk_count}]")
        print(f"  content: {repr(content)}")
        if reasoning:
            # reasoning 可能很长，只显示前 100 字符
            reasoning_preview = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
            print(f"  reasoning_content: {repr(reasoning_preview)}")
        
        # 累积内容
        if content:
            full_content += content
        if reasoning:
            full_reasoning += reasoning
    
    print("\n" + "-" * 70)
    
    # 打印统计
    print(f"📊 统计:")
    print(f"   • 总 chunk 数: {chunk_count}")
    print(f"   • content 长度: {len(full_content)} 字符")
    print(f"   • reasoning 长度: {len(full_reasoning)} 字符")
    
    # 显示完整 content
    print("\n" + "-" * 70)
    print("💬 Content (完整):")
    print("-" * 70)
    print(full_content)
    
    # 显示完整 reasoning（如果有）
    print("\n" + "-" * 70)
    print("🧠 Reasoning Content (完整):")
    print("-" * 70)
    if full_reasoning:
        print(full_reasoning[:2000] + "..." if len(full_reasoning) > 2000 else full_reasoning)
    else:
        print("[无 reasoning_content]")
        print("提示: 使用 deepseek-reasoner 模型并启用 thinking 模式:")
        print("  export AIKG_MODEL_NAME=deepseek-reasoner")
        print("  export AIKG_MODEL_ENABLE_THINK=enabled")
    
    print("\n" + "=" * 70)
    assert full_content, "流式输出应该有内容"
    print("✅ test_stream_chat_openai 测试通过!")


@pytest.mark.skipif(not check_llm_env(), reason="需要配置 LLM 环境变量")
def test_stream_chat_deepseek():
    """
    测试 ChatDeepSeek 流式输出
    
    使用原生 ChatDeepSeek 客户端，测试流式获取 content 和 reasoning_content。
    打印每个 chunk 的详细内容。
    
    AIMessageChunk 结构（参考 LangChain DeepSeek 文档）:
    ```
    AIMessageChunk {
      "content": "The",
      "additional_kwargs": {
        "reasoning_content": "..."  # DeepSeek 原生支持
      },
      "response_metadata": {
        "finishReason": null
      }
    }
    ```
    
    参考: https://reference.langchain.com/javascript/classes/_langchain_deepseek.ChatDeepSeek.html
    """
    print("\n" + "=" * 70)
    print("🌊 测试 ChatDeepSeek 流式输出 (打印每个 chunk)")
    print("=" * 70)
    
    try:
        llm = create_llm_chat_deepseek()
    except ImportError:
        pytest.skip("langchain-deepseek 未安装: pip install langchain-deepseek")
        return
    
    # 收集流式输出
    full_content = ""
    full_reasoning = ""
    chunk_count = 0
    
    print("\n📤 发送请求: '1+1等于多少？请简短回答'")
    print("\n" + "-" * 70)
    print("🔄 Chunk 详情:")
    print("-" * 70)
    
    for chunk in llm.stream("1+1等于多少？请简短回答"):
        chunk_count += 1
        
        # 获取 content
        content = chunk.content or ""
        
        # 获取 reasoning_content
        reasoning = ""
        if hasattr(chunk, 'additional_kwargs'):
            kwargs = chunk.additional_kwargs or {}
            reasoning = kwargs.get('reasoning_content', '') or ""
        
        # 打印每个 chunk 的详情
        print(f"\n[Chunk {chunk_count}]")
        print(f"  content: {repr(content)}")
        if reasoning:
            # reasoning 可能很长，只显示前 100 字符
            reasoning_preview = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
            print(f"  reasoning_content: {repr(reasoning_preview)}")
        
        # 累积内容
        if content:
            full_content += content
        if reasoning:
            full_reasoning += reasoning
    
    print("\n" + "-" * 70)
    
    # 打印统计
    print(f"📊 统计:")
    print(f"   • 总 chunk 数: {chunk_count}")
    print(f"   • content 长度: {len(full_content)} 字符")
    print(f"   • reasoning 长度: {len(full_reasoning)} 字符")
    
    # 显示完整 content
    print("\n" + "-" * 70)
    print("💬 Content (完整):")
    print("-" * 70)
    print(full_content)
    
    # 显示完整 reasoning（如果有）
    print("\n" + "-" * 70)
    print("🧠 Reasoning Content (完整):")
    print("-" * 70)
    if full_reasoning:
        print(full_reasoning[:2000] + "..." if len(full_reasoning) > 2000 else full_reasoning)
    else:
        print("[无 reasoning_content]")
        print("提示: 使用 deepseek-reasoner 模型并启用 thinking 模式:")
        print("  export AIKG_MODEL_NAME=deepseek-reasoner")
        print("  export AIKG_MODEL_ENABLE_THINK=enabled")
    
    print("\n" + "=" * 70)
    assert full_content, "流式输出应该有内容"
    print("✅ test_stream_chat_deepseek 测试通过!")

if __name__ == "__main__":
    import sys
    import asyncio
    
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("""
ReActAgent Demo - ReAct 循环演示

用法:
    python tests/st/test_react_agent.py                    # 运行演示（自动注册 Worker）
    python tests/st/test_react_agent.py --no-worker        # 跳过 Worker 注册（仅测试 LLM 交互）
    python tests/st/test_react_agent.py --backend ascend   # 指定后端
    python tests/st/test_react_agent.py --help             # 显示帮助

工作流程:
    1. 用户输入 "生成 ReLU 算子"
    2. Agent 调用 call_op_task_builder（生成 Torch task）
    3. Agent 调用 ask_user 等待确认
    4. 用户输入 "确认"
    5. Agent 调用 call_codeonly（生成 Triton，需要 Worker）
    6. Agent 调用 finish

环境变量:
    export AIKG_API_KEY=your-api-key
    export AIKG_MODEL_NAME=deepseek-chat           # 可选
    export AIKG_BASE_URL=https://api.deepseek.com  # 可选
    export AIKG_MODEL_ENABLE_THINK=enabled         # 可选，启用思考模式

注意:
    - 完整流程需要 GPU 环境（Worker 注册）
    - 没有 GPU 时可使用 --no-worker，但 call_codeonly 等工具会失败

pytest:
    pytest tests/st/test_react_agent.py -v -s
        """)
        sys.exit(0)
    
    if not check_llm_env():
        print("❌ 需要配置 API Key:")
        print("   export AIKG_API_KEY=your-api-key")
        print("   export AIKG_MODEL_NAME=deepseek-chat           # 可选")
        print("   export AIKG_BASE_URL=https://api.deepseek.com  # 可选")
        print("   export AIKG_MODEL_ENABLE_THINK=enabled         # 可选")
        sys.exit(1)
    
    # 解析命令行参数
    register_worker = "--no-worker" not in sys.argv
    backend = "cuda"
    arch = "a100"
    
    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
        if arg == "--arch" and i + 1 < len(sys.argv):
            arch = sys.argv[i + 1]
    
    asyncio.run(demo_react_agent(
        backend=backend,
        arch=arch,
        register_worker=register_worker
    ))
