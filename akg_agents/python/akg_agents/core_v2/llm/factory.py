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
LLM 工厂函数 - 根据配置创建 LLMClient 和 Embedding 模型

支持多模型配置：complex / standard / fast / 自定义 key（如 coder, designer）
"""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings

from akg_agents.core_v2.config import get_settings
from .client import LLMClient
from .cache import attach_cache_to_client
from .providers.openai_provider import LLMProvider
from .providers.embedding_provider import OpenAICompatibleEmbeddings

logger = logging.getLogger(__name__)


def create_llm_client(
    model_level: Optional[str] = None,
    session_id: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    cache_config_path: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    根据配置创建 LLMClient

    优先级：
    1. 函数参数（直接指定）
    2. 配置文件/环境变量中的模型配置
    3. 默认值

    Args:
        model_level: settings.json 中 models 的级别/名称
                     - 预定义级别："complex" / "standard" / "fast"
                     - 如果为 None，使用 default_model
        session_id: UI 会话 ID（流式输出时用于发送消息到 UI）
        model_name: 模型名称（直接指定，覆盖配置）
        base_url: API 地址（直接指定，覆盖配置）
        api_key: API 密钥（直接指定，覆盖配置）
        temperature: 温度参数（直接指定，覆盖配置）
        max_tokens: 最大 token 数（直接指定，覆盖配置）
        cache_config_path: 缓存配置文件路径（可选）
        **kwargs: 其他配置

    Returns:
        LLMClient 实例

    使用示例：
        # 方式 1：使用配置中的模型级别
        complex_client = create_llm_client(model_level="complex")
        standard_client = create_llm_client(model_level="standard")
        fast_client = create_llm_client(model_level="fast")

        # 方式 2：使用自定义配置（如 coder, designer）
        coder_client = create_llm_client(model_level="coder", session_id="xxx")

        # 方式 3：直接指定参数（不依赖配置）
        client = create_llm_client(
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-xxx",
            temperature=0.7
        )

        # 方式 4：使用默认配置
        client = create_llm_client()  # 使用 default_model
    """
    # 加载配置
    settings = get_settings()

    # 确定要使用的 model_level
    if model_level is None:
        model_level = settings.default_model

    # 从配置中获取对应的模型配置
    model_config = settings.models.get(model_level)

    if model_config is None:
        # 如果调用方显式传了连接参数，允许无配置使用
        if base_url and api_key and model_name:
            logger.info(f"Model level '{model_level}' not in config, using provided parameters")
            final_base_url = base_url
            final_api_key = api_key
            final_model_name = model_name
            final_temperature = temperature if temperature is not None else 0.2
            final_max_tokens = max_tokens if max_tokens is not None else 8192
            final_top_p = top_p if top_p is not None else 0.9
            final_frequency_penalty = frequency_penalty
            final_presence_penalty = presence_penalty
            final_timeout = kwargs.pop("timeout", 300)
            final_extra_body = kwargs.pop("extra_body", {})
            # 向后兼容：thinking_enabled=True → 默认 extra_body
            if not final_extra_body and kwargs.pop("thinking_enabled", False):
                final_extra_body = {"thinking": {"type": "enabled"}}
        else:
            available = list(settings.models.keys()) if settings.models else []
            raise ValueError(
                f"模型级别 '{model_level}' 未配置，无法创建 LLM 客户端。\n"
                f"  可用级别: {available}\n"
                f"  请通过环境变量或配置文件设置该级别的模型信息:\n"
                f"    环境变量: export AKG_AGENTS_{model_level.upper()}_*=...\n"
                f"    配置文件: ~/.akg/settings.json 或 .akg/settings.json 或 .akg/settings.local.json"
            )
    else:
        # 使用配置中的模型，但参数可以覆盖
        final_base_url = base_url or model_config.base_url
        final_api_key = api_key or model_config.api_key
        final_model_name = model_name or model_config.model_name
        final_temperature = temperature if temperature is not None else model_config.temperature
        final_max_tokens = max_tokens if max_tokens is not None else model_config.max_tokens
        final_top_p = top_p if top_p is not None else model_config.top_p
        final_frequency_penalty = (
            frequency_penalty if frequency_penalty is not None else model_config.frequency_penalty
        )
        final_presence_penalty = presence_penalty if presence_penalty is not None else model_config.presence_penalty
        final_timeout = kwargs.pop("timeout", model_config.timeout)
        final_extra_body = kwargs.pop("extra_body", model_config.extra_body)
        # 向后兼容：thinking_enabled=True → 默认 extra_body
        if not final_extra_body and kwargs.pop("thinking_enabled", False):
            final_extra_body = {"thinking": {"type": "enabled"}}

    logger.info(
        f"Creating LLMClient: level={model_level}, model={final_model_name}, "
        f"base_url={final_base_url}, extra_body={bool(final_extra_body)}"
    )

    # 创建 Provider
    provider = LLMProvider(
        model_name=final_model_name,
        api_key=final_api_key,
        base_url=final_base_url,
        timeout=final_timeout,
        extra_body=final_extra_body,
        **kwargs
    )

    # 创建 Client
    client = LLMClient(
        provider=provider,
        session_id=session_id,
        temperature=final_temperature,
        max_tokens=final_max_tokens,
        top_p=final_top_p,
        frequency_penalty=final_frequency_penalty,
        presence_penalty=final_presence_penalty,
        **kwargs
    )

    try:
        client = attach_cache_to_client(
            client,
            cache_config_path=cache_config_path,
        )
    except Exception as exc:
        logger.warning(f"Failed to attach LLM cache: {exc}")

    return client


def create_embedding_model(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Embeddings:
    """
    根据配置创建 Embedding 模型

    优先级：
    1. 函数参数（直接指定）
    2. 环境变量 AKG_AGENTS_EMBEDDING_*
    3. settings.json 中的 embedding 配置

    Args:
        base_url: API 地址（直接指定，覆盖配置）
        api_key: API 密钥（直接指定，覆盖配置）
        model_name: 模型名称（直接指定，覆盖配置）
        timeout: 超时时间（秒）

    Returns:
        Embeddings: LangChain 兼容的 Embedding 模型实例

    Raises:
        ValueError: 配置不完整时抛出

    使用示例：
        # 方式 1：使用配置（环境变量或 settings.json）
        embedding = create_embedding_model()

        # 方式 2：直接指定参数
        embedding = create_embedding_model(
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-xxx",
            model_name="BAAI/bge-large-zh-v1.5"
        )

    环境变量：
        AKG_AGENTS_EMBEDDING_BASE_URL: API 地址
        AKG_AGENTS_EMBEDDING_API_KEY: API 密钥
        AKG_AGENTS_EMBEDDING_MODEL_NAME: 模型名称
        AKG_AGENTS_EMBEDDING_TIMEOUT: 超时时间（秒）
    """
    # 加载配置
    settings = get_settings()

    # 确定最终配置（参数 > 配置）
    final_base_url = base_url or settings.embedding.base_url
    final_api_key = api_key or settings.embedding.api_key
    final_model_name = model_name or settings.embedding.model_name
    final_timeout = timeout if timeout is not None else settings.embedding.timeout

    # 检查配置是否完整
    if not final_base_url or not final_api_key or not final_model_name:
        missing = []
        if not final_base_url:
            missing.append("base_url (AKG_AGENTS_EMBEDDING_BASE_URL)")
        if not final_api_key:
            missing.append("api_key (AKG_AGENTS_EMBEDDING_API_KEY)")
        if not final_model_name:
            missing.append("model_name (AKG_AGENTS_EMBEDDING_MODEL_NAME)")

        raise ValueError(
            f"Embedding 配置不完整，缺少: {', '.join(missing)}。\n"
            f"请设置环境变量或在 settings.json 中配置 embedding 字段。"
        )

    # 构建 embedding API 端点
    embedding_url = f"{final_base_url.rstrip('/')}/embeddings"

    logger.info(f"Creating embedding model: url={embedding_url}, model={final_model_name}")

    return OpenAICompatibleEmbeddings(
        api_url=embedding_url,
        model_name=final_model_name,
        api_key=final_api_key,
        verify_ssl=False,
        timeout=final_timeout,
    )
