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
from pathlib import Path
from typing import Optional, Union

import httpx
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
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

        # 直接返回openai.AsyncOpenAI客户端
        model = openai.AsyncOpenAI(
            base_url=model_params.pop("api_base"),
            api_key="dummy"
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

        timeout = httpx.Timeout(60, read=60 * 10)
        # 创建DeepSeek模型实例
        model = ChatDeepSeek(
            api_key=api_key,
            http_client=httpx.Client(verify=False, timeout=timeout),
            http_async_client=httpx.AsyncClient(verify=False, timeout=timeout),
            **model_params
        )

    return model
