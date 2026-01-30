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

import os
import yaml
import logging
import requests
from pathlib import Path
from typing import Optional, Union, List, Any

import httpx
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings
import openai


# 配置文件路径
CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"

# 设置日志
logger = logging.getLogger(__name__)

# 环境变量
OLLAMA_API_BASE_ENV = "AKG_AGENTS_OLLAMA_API_BASE"
VLLM_API_BASE_ENV = "AKG_AGENTS_VLLM_API_BASE"
DISABLE_REPEAT_PENALTY_ENV = "AKG_AGENTS_DISABLE_REPEAT_PENALTY"


def create_model(name: Optional[str] = None, config_path: Optional[str] = None) -> Union[ChatDeepSeek, ChatOllama]:
    """
    根据预设名称创建模型

    Args:
        name: 预设配置名称，如果为None则使用默认配置
        config_path: 配置文件路径，如果为None则使用默认路径

    Returns:
        ChatDeepSeek | ChatOllama: 创建的模型实例
    """
    # 定义 thinking_mode 处理函数（提前定义，供环境变量模式使用）
    def _build_thinking_extra_body(thinking_mode: Optional[str],
                                   extra_body: Optional[dict]) -> Optional[dict]:
        """根据配置构造 thinking 相关的 extra_body"""
        if thinking_mode is None:
            return extra_body

        # 统一用字符串匹配，兼容 True/False 以及 enabled/disabled
        if isinstance(thinking_mode, bool):
            normalized = "true" if thinking_mode else "false"
        else:
            normalized = str(thinking_mode).strip().lower()

        extra_body = dict(extra_body or {})
        if normalized in {"enabled", "disabled"}:
            extra_body["thinking"] = {"type": normalized}
        elif normalized in {"true", "false"}:
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
            chat_template_kwargs["thinking"] = (normalized == "true")
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        else:
            logger.warning("不支持的 thinking_mode '%s'，请使用 enabled/disabled 或 True/False", thinking_mode)
            return extra_body if extra_body else None

        return extra_body

    def _strip_repeat_penalty(params: dict) -> dict:
        """可选地移除重复惩罚参数，默认保持与 llm_config.yaml 兼容。"""
        raw = os.getenv(DISABLE_REPEAT_PENALTY_ENV)
        if raw is None or raw == "":
            return params

        normalized = raw.strip().lower()
        if normalized not in {"1", "true", "yes", "on"}:
            return params

        overridden = dict(params)
        overridden.pop("frequency_penalty", None)
        overridden.pop("presence_penalty", None)
        return overridden

    # 【最高优先级】检查环境变量覆盖
    env_base_url = os.getenv("AKG_AGENTS_BASE_URL")
    env_model_name = os.getenv("AKG_AGENTS_MODEL_NAME")
    env_api_key = os.getenv("AKG_AGENTS_API_KEY")
    env_enable_think = os.getenv("AKG_AGENTS_MODEL_ENABLE_THINK")
    
    if env_base_url and env_model_name and env_api_key:
        # 使用环境变量创建模型
        logger.info("=" * 60)
        logger.info("使用环境变量覆盖模式创建模型")
        logger.info(f"  AKG_AGENTS_BASE_URL: {env_base_url}")
        logger.info(f"  AKG_AGENTS_MODEL_NAME: {env_model_name}")
        # 只显示前8位和后4位，保护API密钥安全
        masked_key = env_api_key[:8] + "*" * (len(env_api_key) - 12) + \
            env_api_key[-4:] if len(env_api_key) > 12 else "***"
        logger.info(f"  AKG_AGENTS_API_KEY: {masked_key}")
        if env_enable_think:
            logger.info(f"  AKG_AGENTS_MODEL_ENABLE_THINK: {env_enable_think}")
        logger.info("=" * 60)
        
        # 设置默认参数
        defaults = {
            "temperature": 0.2,
            "max_tokens": 8192,
            "top_p": 0.9,
        }
        defaults = _strip_repeat_penalty(defaults)
        default_temperature = defaults["temperature"]
        default_max_tokens = defaults["max_tokens"]
        default_top_p = defaults["top_p"]
        
        logger.info(f"使用默认参数: temperature={default_temperature}, max_tokens={default_max_tokens}, top_p={default_top_p}")
        
        # 处理 thinking_mode 配置
        extra_body = _build_thinking_extra_body(env_enable_think, None)
        if extra_body:
            logger.info(f"启用 thinking 模式: {extra_body}")
        
        # 设置20分钟的timeout
        timeout = httpx.Timeout(60, read=60 * 20)
        
        # 使用OpenAI API创建客户端
        model = openai.AsyncOpenAI(
            base_url=env_base_url,
            api_key=env_api_key,
            http_client=httpx.AsyncClient(verify=False, timeout=timeout)
        )
        
        # 将配置参数保存到模型对象上，供后续使用
        model.model_name = env_model_name
        model.temperature = default_temperature
        model.max_tokens = default_max_tokens
        model.top_p = default_top_p
        model.other_params = {}
        model.extra_body = extra_body
        
        logger.info("环境变量覆盖模式：模型创建完成")
        return model
    
    # 使用默认路径或指定路径
    config_path = config_path or CONFIG_PATH

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 如果未指定预设名称，使用默认预设
    name = name or config.get("default_preset")

    # 检查预设是否存在
    if name not in config:
        available_presets = [k for k in config.keys() if k != "default_preset"]
        raise ValueError(f"预设 '{name}' 未找到。可用预设: {', '.join(available_presets)}")

    # 获取预设配置
    preset_config = config[name]

    # 在调试级别打印配置
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"使用预设 '{name}' 的配置:")
        for key, value in preset_config.items():
            if key == "api_key_env":
                logger.debug(f"  {key}: {value} (环境变量)")
            else:
                logger.debug(f"  {key}: {value}")

    # 判断是否为Ollama模型
    if name.startswith("ollama_"):
        # 提取模型参数
        model_params = {k: v for k, v in preset_config.items()}
        model_params = _strip_repeat_penalty(model_params)

        # 检查是否有环境变量覆盖api_base
        if OLLAMA_API_BASE_ENV in os.environ:
            model_params["api_base"] = os.environ[OLLAMA_API_BASE_ENV]
            logger.debug(f"使用环境变量 {OLLAMA_API_BASE_ENV} 覆盖 api_base: {model_params['api_base']}")
        else:
            model_params["api_base"] = "http://localhost:11434"
            logger.debug(f"未设置环境变量 {OLLAMA_API_BASE_ENV}，使用默认 api_base: {model_params['api_base']}")

        # 记录连接信息
        logger.info(
            f"创建Ollama模型 '{name}': api_base={model_params['api_base']}, model={model_params.get('model', 'N/A')}")
        # 显示环境变量信息
        if OLLAMA_API_BASE_ENV in os.environ:
            logger.info(f"  环境变量 {OLLAMA_API_BASE_ENV}: {os.environ[OLLAMA_API_BASE_ENV]}")
        else:
            logger.info(f"  环境变量 {OLLAMA_API_BASE_ENV}: 未设置 (使用默认值)")

        # 创建Ollama模型实例
        model = ChatOllama(
            base_url=model_params.pop("api_base"),
            model=model_params.pop("model"),
            **model_params
        )
    elif name.startswith("vllm_"):
        # 判断是否为VLLM模型
        # 提取模型参数
        model_params = {k: v for k, v in preset_config.items()}
        model_params = _strip_repeat_penalty(model_params)

        # 检查是否有环境变量覆盖api_base
        if VLLM_API_BASE_ENV in os.environ:
            model_params["api_base"] = os.environ[VLLM_API_BASE_ENV]
            logger.debug(f"使用环境变量 {VLLM_API_BASE_ENV} 覆盖 api_base: {model_params['api_base']}")
        else:
            model_params["api_base"] = "http://localhost:8001/v1"
            logger.debug(f"未设置环境变量 {VLLM_API_BASE_ENV}，使用默认 api_base: {model_params['api_base']}")

        # 记录连接信息
        logger.info(
            f"创建VLLM模型 '{name}': api_base={model_params['api_base']}, model={model_params.get('model', 'N/A')}")
        # 显示环境变量信息
        if VLLM_API_BASE_ENV in os.environ:
            logger.info(f"  环境变量 {VLLM_API_BASE_ENV}: {os.environ[VLLM_API_BASE_ENV]}")
        else:
            logger.info(f"  环境变量 {VLLM_API_BASE_ENV}: 未设置 (使用默认值)")

        # 设置20分钟的timeout
        timeout = httpx.Timeout(60, read=60 * 20)
        # 直接返回openai.AsyncOpenAI客户端
        model = openai.AsyncOpenAI(
            base_url=model_params.pop("api_base"),
            api_key="dummy",
            http_client=httpx.AsyncClient(verify=False, timeout=timeout)
        )

        # 将配置参数保存到模型对象上，供后续使用
        model.model_name = model_params.pop("model")
        model.temperature = model_params.get("temperature", 0.1)
        model.max_tokens = model_params.get("max_tokens", 8192)
        model.top_p = model_params.get("top_p", 0.95)
        thinking_mode = model_params.pop("thinking_mode", None)
        extra_body = model_params.pop("extra_body", None)
        extra_body = _build_thinking_extra_body(thinking_mode, extra_body)
        model.other_params = model_params
        model.extra_body = extra_body

    else:
        # 获取API密钥
        api_key_env = preset_config.get("api_key_env")
        if not api_key_env:
            raise ValueError(f"预设 '{name}' 未配置 api_key_env")

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API密钥未找到。请设置环境变量 {api_key_env}")

        # 提取模型参数 - 只排除api_key_env
        model_params = {k: v for k, v in preset_config.items()
                        if k != "api_key_env"}
        model_params = _strip_repeat_penalty(model_params)

        # 统一处理 thinking 配置，允许在 YAML 中通过 thinking_mode 字段控制
        thinking_mode = model_params.pop("thinking_mode", None)
        extra_body = model_params.pop("extra_body", None)
        extra_body = _build_thinking_extra_body(thinking_mode, extra_body)
        if extra_body:
            model_params["extra_body"] = extra_body

        # 记录连接信息
        logger.info(
            f"创建Langchain模型 '{name}': api_base={model_params.get('api_base', 'N/A')}, model={model_params.get('model', 'N/A')}")
        # 显示环境变量信息
        if api_key_env in os.environ:
            api_key_value = os.environ[api_key_env]
            # 只显示前8位和后4位，保护API密钥安全
            masked_key = api_key_value[:8] + "*" * (len(api_key_value) - 12) + \
                api_key_value[-4:] if len(api_key_value) > 12 else "***"
            logger.info(f"  环境变量 {api_key_env}: {masked_key}")
        else:
            logger.info(f"  环境变量 {api_key_env}: 未设置")

        timeout = httpx.Timeout(60, read=60 * 10)
        # 创建DeepSeek模型实例
        model = ChatDeepSeek(
            api_key=api_key,
            http_client=httpx.Client(verify=False, timeout=timeout),
            http_async_client=httpx.AsyncClient(verify=False, timeout=timeout),
            **model_params
        )

    return model


def create_langchain_chat_model(
    name: Optional[str] = None, config_path: Optional[str] = None
) -> Any:
    """
    ReAct 专用：创建 LangChain ChatModel（BaseChatModel）。

    注意：
    - 不改动原 create_model() 的返回类型与行为（旧链路可能依赖 openai.AsyncOpenAI）。
    - 本函数确保返回的是 LangChain ChatModel，以满足 langchain.agents.create_agent 的要求。
    """

    def _strip_repeat_penalty(params: dict) -> dict:
        raw = os.getenv(DISABLE_REPEAT_PENALTY_ENV)
        if raw is None or raw == "":
            return params
        normalized = str(raw).strip().lower()
        if normalized not in {"1", "true", "yes", "on"}:
            return params
        overridden = dict(params)
        overridden.pop("frequency_penalty", None)
        overridden.pop("presence_penalty", None)
        return overridden

    def _build_thinking_extra_body(
        thinking_mode: Optional[str], extra_body: Optional[dict]
    ) -> Optional[dict]:
        if thinking_mode is None:
            return extra_body
        normalized = str(thinking_mode).strip().lower()
        extra_body = dict(extra_body or {})
        if normalized in {"enabled", "disabled"}:
            extra_body["thinking"] = {"type": normalized}
            return extra_body
        if normalized in {"true", "false"}:
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
            chat_template_kwargs["thinking"] = normalized == "true"
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            return extra_body
        logger.warning(
            "不支持的 thinking_mode '%s'，请使用 enabled/disabled 或 true/false",
            thinking_mode,
        )
        return extra_body if extra_body else None

    def _create_chat_openai(
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        extra_body: Optional[dict],
    ):
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise RuntimeError(
                "ReAct 模式需要 LangChain ChatModel。\n"
                "当前配置走的是 openai-compatible endpoint（如 vllm / 自定义 base_url），"
                "需要安装 `langchain-openai` 才能创建 ChatOpenAI。\n"
                "可选方案：\n"
                "  - pip install -U langchain-openai\n"
                "  - 或改用 deepseek/ollama preset\n"
            ) from e

        timeout = httpx.Timeout(60, read=60 * 20)
        
        # 检查是否需要启用 thinking mode
        thinking_enabled = (
            extra_body 
            and extra_body.get("thinking", {}).get("type") == "enabled"
        )

        if thinking_enabled:
            logger.warning(
                "如需使用 thinking mode，请使用 ChatDeepSeek preset。"
            )
            extra_body["thinking"] = {"type": "disabled"}
        
        # 使用标准 httpx 客户端
        http_client = httpx.Client(verify=False, timeout=timeout)
        http_async_client = httpx.AsyncClient(verify=False, timeout=timeout)
        
        extra_body = extra_body or {}
        if "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "disabled"}

        kwargs: dict = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "http_client": http_client,
            "http_async_client": http_async_client,
            "max_retries": 3,
            "extra_body": extra_body,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        return ChatOpenAI(**kwargs)

    # 【最高优先级】环境变量覆盖（openai-compatible）
    env_base_url = os.getenv("AKG_AGENTS_BASE_URL")
    env_model_name = os.getenv("AKG_AGENTS_MODEL_NAME")
    env_api_key = os.getenv("AKG_AGENTS_API_KEY")
    env_enable_think = os.getenv("AKG_AGENTS_MODEL_ENABLE_THINK")
    if env_base_url and env_model_name and env_api_key:
        defaults = _strip_repeat_penalty(
            {"temperature": 0.2, "max_tokens": 8192, "top_p": 0.9}
        )
        extra_body = _build_thinking_extra_body(env_enable_think, None)
        return _create_chat_openai(
            model=env_model_name,
            base_url=env_base_url,
            api_key=env_api_key,
            temperature=float(defaults["temperature"]),
            max_tokens=int(defaults["max_tokens"]),
            top_p=float(defaults["top_p"]),
            extra_body=extra_body,
        )

    # 使用默认路径或指定路径
    config_path = config_path or CONFIG_PATH
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    name = name or config.get("default_preset")
    if name not in config:
        available_presets = [k for k in config.keys() if k != "default_preset"]
        raise ValueError(f"预设 '{name}' 未找到。可用预设: {', '.join(available_presets)}")

    preset_config = config[name] or {}

    # Ollama：天然是 LangChain ChatModel
    if str(name).startswith("ollama_"):
        model_params = {k: v for k, v in preset_config.items()}
        model_params = _strip_repeat_penalty(model_params)
        api_base = (
            os.getenv(OLLAMA_API_BASE_ENV)
            or model_params.pop("api_base", None)
            or "http://localhost:11434"
        )
        return ChatOllama(
            base_url=api_base,
            model=model_params.pop("model"),
            **model_params,
        )

    # vLLM/openai-compatible：需要 ChatOpenAI
    if str(name).startswith("vllm_"):
        model_params = {k: v for k, v in preset_config.items()}
        model_params = _strip_repeat_penalty(model_params)
        api_base = (
            os.getenv(VLLM_API_BASE_ENV)
            or model_params.pop("api_base", None)
            or "http://localhost:8001/v1"
        )
        model_name = model_params.pop("model")
        temperature = float(model_params.pop("temperature", 0.1))
        max_tokens = int(model_params.pop("max_tokens", 8192))
        top_p = float(model_params.pop("top_p", 0.95))
        thinking_mode = model_params.pop("thinking_mode", None)
        extra_body = model_params.pop("extra_body", None)
        extra_body = _build_thinking_extra_body(thinking_mode, extra_body)
        # vllm 通常无需真实 key；沿用旧逻辑 dummy
        return _create_chat_openai(
            model=model_name,
            base_url=api_base,
            api_key="dummy",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_body=extra_body,
        )

    # 其他 preset：沿用 ChatDeepSeek（LangChain ChatModel）
    api_key_env = preset_config.get("api_key_env")
    if not api_key_env:
        raise ValueError(f"预设 '{name}' 未配置 api_key_env")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"API密钥未找到。请设置环境变量 {api_key_env}")

    model_params = {k: v for k, v in preset_config.items() if k != "api_key_env"}
    model_params = _strip_repeat_penalty(model_params)

    # 处理 thinking_mode 配置
    thinking_mode = model_params.pop("thinking_mode", None)
    extra_body = model_params.pop("extra_body", None)
    extra_body = _build_thinking_extra_body(thinking_mode, extra_body)
    
    # 检查是否需要启用 thinking mode
    thinking_enabled = (
        extra_body 
        and extra_body.get("thinking", {}).get("type") == "enabled"
    )
    
    timeout = httpx.Timeout(60, read=60 * 10)
    
    if thinking_enabled:
        # 使用支持 thinking mode 的自定义 ChatModel
        from akg_agents.core.llm.thinking_chat_model import ThinkingAwareChatDeepSeek
        logger.info("[ReAct] 使用 ThinkingAwareChatDeepSeek，支持 DeepSeek thinking mode")
        return ThinkingAwareChatDeepSeek(
            api_key=api_key,
            timeout=timeout,
            extra_body=extra_body,
            **model_params,
        )
    else:
        # 使用标准 ChatDeepSeek，显式禁用 thinking mode
        extra_body = extra_body or {}
        if "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "disabled"}
        logger.info("[ReAct] 使用标准 ChatDeepSeek，thinking mode 已禁用")
        return ChatDeepSeek(
            api_key=api_key,
            http_client=httpx.Client(verify=False, timeout=timeout),
            http_async_client=httpx.AsyncClient(verify=False, timeout=timeout),
            extra_body=extra_body,
            **model_params,
        )

class OpenAICompatibleEmbeddings(Embeddings):
    """
    调用 OpenAI 兼容格式的 Embedding API。
    支持本地部署（如 vllm）和远程 API（如硅流平台）。
    """
    def __init__(
        self,
        api_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        verify_ssl: bool = True
    ):
        """
        初始化 Embedding 客户端

        Args:
            api_url: Embedding API 的完整 URL（如 http://localhost:8001/v1/embeddings）
            model_name: 模型名称
            api_key: API 密钥（可选，远程 API 需要）
            verify_ssl: 是否验证 SSL 证书
        """
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.verify_ssl = verify_ssl

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成 Embeddings"""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """为查询生成 Embeddings"""
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """调用 API 生成 Embeddings"""
        payload = {
            "model": self.model_name,
            "input": texts
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                verify=self.verify_ssl,
                timeout=60
            )
            response.raise_for_status()

            # 解析返回的 JSON
            result = response.json()

            # 检查返回格式是否正确
            if 'data' not in result:
                raise ValueError(f"API返回格式错误，缺少'data'字段: {result}")

            embeddings = []
            for item in result['data']:
                if 'embedding' not in item:
                    raise ValueError(f"API返回格式错误，缺少'embedding'字段: {item}")
                embeddings.append(item['embedding'])

            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API请求失败: {e}")
            raise RuntimeError(f"Embedding API请求失败: {e}") from e


def create_embedding_model(name: Optional[str] = None, config_path: Optional[str] = None) -> Embeddings:
    """
    根据预设名称创建embedding模型

    Args:
        name: 预设配置名称，如果为None则使用默认配置
        config_path: 配置文件路径，如果为None则使用默认路径

    Returns:
        Embeddings: 创建的embedding模型实例，兼容LangChain
    """
    # 使用默认路径或指定路径
    config_path = config_path or CONFIG_PATH
    
    # 加载配置文件（用于检查预设名和获取默认值）
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # 【最高优先级】检查环境变量覆盖
    env_base_url = os.getenv("AKG_AGENTS_EMBEDDING_BASE_URL")
    env_model_name = os.getenv("AKG_AGENTS_EMBEDDING_MODEL_NAME")
    env_api_key = os.getenv("AKG_AGENTS_EMBEDDING_API_KEY")

    if env_base_url and env_model_name and env_api_key:
        # 检查 env_model_name 是否是预设名（存在于配置文件中）
        # 如果是预设名，从配置文件读取实际模型名
        actual_model_name = env_model_name
        if env_model_name in config:
            preset_config = config[env_model_name]
            actual_model_name = preset_config.get("model", env_model_name)
            logger.info(f"检测到预设名 '{env_model_name}'，使用实际模型名: {actual_model_name}")
        
        # 使用环境变量创建embedding模型
        logger.info("=" * 60)
        logger.info("使用环境变量覆盖模式创建Embedding模型")
        logger.info(f"  AKG_AGENTS_EMBEDDING_BASE_URL: {env_base_url}")
        logger.info(f"  AKG_AGENTS_EMBEDDING_MODEL_NAME: {env_model_name}")
        if actual_model_name != env_model_name:
            logger.info(f"  实际模型名: {actual_model_name}")
        # 只显示前8位和后4位，保护API密钥安全
        masked_key = env_api_key[:8] + "*" * (len(env_api_key) - 12) + \
            env_api_key[-4:] if len(env_api_key) > 12 else "***"
        logger.info(f"  AKG_AGENTS_EMBEDDING_API_KEY: {masked_key}")
        logger.info("=" * 60)

        # 构建 embedding API 端点
        embedding_url = f"{env_base_url.rstrip('/')}/embeddings"
        embedding_model = OpenAICompatibleEmbeddings(
            api_url=embedding_url,
            model_name=actual_model_name,
            api_key=env_api_key,
            verify_ssl=False
        )

        logger.info("环境变量覆盖模式：Embedding模型创建完成")
        return embedding_model

    # 检查配置文件是否存在（如果前面没有加载成功）
    if not config:
        raise FileNotFoundError(f"配置文件未找到或为空: {config_path}")

    # 如果未指定预设名称，使用默认embedding预设
    name = name or config.get("default_embedding_preset", "sflow_qwen3_embedding_8b")

    # 检查预设是否存在
    if name not in config:
        available_presets = [k for k in config.keys() if k.startswith("sflow_") and "embedding" in k]
        raise ValueError(f"embedding预设 '{name}' 未找到。可用预设: {', '.join(available_presets)}")

    # 获取预设配置
    preset_config = config[name]

    if name.startswith("vllm_"):
        # 检查是否有环境变量覆盖api_base
        if VLLM_API_BASE_ENV in os.environ:
            api_base = os.environ[VLLM_API_BASE_ENV]
            logger.debug(f"使用环境变量 {VLLM_API_BASE_ENV} 覆盖 api_base: {api_base}")
        else:
            api_base = "http://localhost:8001/v1"
            logger.debug(f"未设置环境变量 {VLLM_API_BASE_ENV}，使用默认 api_base: {api_base}")

        # 构建完整的embedding API端点
        embedding_url = f"{api_base.rstrip('/')}/embeddings"
        embedding_model = OpenAICompatibleEmbeddings(
            api_url=embedding_url,
            model_name=preset_config.get("model")
        )

    else:
        # 获取API密钥
        api_key_env = preset_config.get("api_key_env")
        if not api_key_env:
            raise ValueError(f"预设 '{name}' 未配置 api_key_env")

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API密钥未找到。请设置环境变量 {api_key_env}")

        # 使用 OpenAICompatibleEmbeddings，绕过 tiktoken 和 SSL 验证问题
        api_base = preset_config.get("api_base", "").rstrip("/")
        embedding_url = f"{api_base}/embeddings"
        embedding_model = OpenAICompatibleEmbeddings(
            api_url=embedding_url,
            model_name=preset_config.get("model"),
            api_key=api_key,
            verify_ssl=False
        )

    return embedding_model
