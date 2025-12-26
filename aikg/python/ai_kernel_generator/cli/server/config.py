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
配置管理模块

负责加载和管理工作流服务的配置。
"""

import os
import logging
from typing import Dict, List, Tuple

from ai_kernel_generator.config.config_validator import load_config
from textual import log

logger = logging.getLogger(__name__)


def get_server_config() -> Tuple[str, str, List[int]]:
    """
    从环境变量获取服务器配置

    Returns:
        (backend, arch, devices) 元组
    """
    backend = os.environ.get("WORKFLOW_BACKEND", "cuda")
    arch = os.environ.get("WORKFLOW_ARCH", "a100")
    devices_str = os.environ.get("WORKFLOW_DEVICES", "0")

    try:
        devices = [int(d.strip()) for d in devices_str.split(",")]
    except ValueError:
        logger.warning(f"Invalid WORKFLOW_DEVICES: {devices_str}, using [0]")
        devices = [0]

    return backend, arch, devices


def setup_stream_mode(use_stream: bool):
    """
    设置流式输出模式环境变量

    Args:
        use_stream: 是否启用 LLM 流式输出
    """
    if use_stream:
        os.environ["AIKG_STREAM_OUTPUT"] = "on"
        logger.info("Stream output enabled")
    else:
        os.environ["AIKG_STREAM_OUTPUT"] = "off"
        logger.info("Stream output disabled")
