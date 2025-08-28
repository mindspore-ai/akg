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
from typing import Optional, Union, List

import httpx
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings
import openai


# 配置文件路径
CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"

# 设置日志
logger = logging.getLogger(__name__)

# 环境变量
OLLAMA_API_BASE_ENV = "AIKG_OLLAMA_API_BASE"
VLLM_API_BASE_ENV = "AIKG_VLLM_API_BASE"


def create_model(name: Optional[str] = None, config_path: Optional[str] = None) -> Union[ChatDeepSeek, ChatOllama]:
    """
    根据预设名称创建模型

    Args:
        name: 预设配置名称，如果为None则使用默认配置
        config_path: 配置文件路径，如果为None则使用默认路径

    Returns:
        ChatDeepSeek | ChatOllama: 创建的模型实例
    """
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
        model.other_params = model_params

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


class LocalAPIEmbeddings(Embeddings):
    """使用本地部署的 Embedding API"""
    
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self.model_name = model_name
    
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
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                # 如果部署在内网需要代理，添加以下参数
                proxies={"http": "", "https": ""},  # 清除代理
                timeout=30
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
            # 处理异常情况
            print(f"API请求失败: {e}")
            # 返回零向量，维度为1024（根据Qwen3-Embedding-8B的实际维度）
            return [[0.0] * 1024] * len(texts)


def create_embedding_model(name: Optional[str] = None, config_path: Optional[str] = None):
    """
    根据预设名称创建embedding模型
    
    Args:
        name: 预设配置名称，如果为None则使用默认配置
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        OpenAIEmbeddings: 创建的embedding模型实例，兼容LangChain
    """
    # 使用默认路径或指定路径
    config_path = config_path or CONFIG_PATH
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
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
        embedding_model = LocalAPIEmbeddings(
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

        # 使用LangChain的OpenAIEmbeddings，配置硅流平台
        embedding_model = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=preset_config.get("api_base"),
            model=preset_config.get("model"),
            **({"dimensions": preset_config.get("dimensions")} if preset_config.get("dimensions") else {})
        )
    
    return embedding_model