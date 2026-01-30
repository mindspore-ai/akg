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
AIKG Core V2 - 重构后的核心框架

特点：
- 配置统一到 ~/.akg/settings.json
- 统一使用 OpenAI 兼容接口（不再需要 ChatDeepSeek / ChatOllama）
- 精简的 AgentBase（~200 行）
- 支持流式输出和 UI 消息发送
- 可选的 Agent 注册机制

使用示例：
    # 1. 创建 LLMClient
    from akg_agents.core_v2.llm import create_llm_client
    client = create_llm_client(model_level="standard", session_id="xxx")
    result = await client.generate([{"role": "user", "content": "Hello"}])
    
    # 2. 使用 AgentBase
    from akg_agents.core_v2.agents import AgentBase, register_agent
    
    @register_agent
    class MyAgent(AgentBase):
        async def run(self, task_info):
            return await self.run_llm(self.template, input_data, "standard")
    
    # 3. 配置管理
    from akg_agents.core_v2.config import get_settings
    settings = get_settings()
    print(settings.models)
"""

__version__ = "2.0.0"

# 配置
from .config import (
    ModelConfig,
    EmbeddingConfig,
    AKGSettings,
    get_settings,
    get_settings_path,
    save_settings_file,
    create_default_settings_file,
)

# LLM
from .llm import (
    LLMProvider,
    LLMClient,
    create_llm_client,
    OpenAICompatibleEmbeddings,
    create_embedding_model,
)

# Agents
from .agents import (
    AgentBase,
    Jinja2TemplateWrapper,
    AgentRegistry,
    register_agent,
)

__all__ = [
    # 版本
    "__version__",
    # 配置
    "ModelConfig",
    "EmbeddingConfig",
    "AKGSettings",
    "get_settings",
    "get_settings_path",
    "save_settings_file",
    "create_default_settings_file",
    # LLM
    "LLMProvider",
    "LLMClient",
    "create_llm_client",
    # Embedding
    "OpenAICompatibleEmbeddings",
    "create_embedding_model",
    # Agents
    "AgentBase",
    "Jinja2TemplateWrapper",
    "AgentRegistry",
    "register_agent",
]
