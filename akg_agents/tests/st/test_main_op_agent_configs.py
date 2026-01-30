#!/usr/bin/env python3
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
MainOpAgent 不同硬件配置的 pytest 测试

"""

import logging
import os
import sys
import pytest

# 添加 examples 目录到 sys.path，以便导入 run_main_op_agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples'))

# 直接导入 examples/run_main_op_agent.py 中的 interactive_demo 函数
from run_main_op_agent import interactive_demo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_main_op_agent_cuda():
    """
    测试 CUDA 环境配置（交互式多轮对话）
    
    配置:
        - Backend: cuda
        - Arch: a100
        - Device IDs: [0]
        - DSL: triton
        - Framework: torch
    
    运行方式:
        pytest tests/st/test_main_op_agent_configs.py::test_main_op_agent_cuda -v -s
    """
    logger.info("=" * 80)
    logger.info("开始测试: CUDA 环境配置")
    logger.info("=" * 80)
    
    # 直接调用 examples/run_main_op_agent.py 中的 interactive_demo 函数
    await interactive_demo(
        backend="cuda",
        arch="a100",
        device_ids=[0],
        dsl="triton",
        framework="torch"
    )


@pytest.mark.asyncio
async def test_main_op_agent_ascend():
    """
    测试 Ascend 环境配置（交互式多轮对话）
    
    配置:
        - Backend: ascend
        - Arch: ascend910b4
        - Device IDs: [0]
        - DSL: ascendc
        - Framework: torch
    
    运行方式:
        pytest tests/st/test_main_op_agent_configs.py::test_main_op_agent_ascend -v -s
    """
    logger.info("=" * 80)
    logger.info("开始测试: Ascend 环境配置")
    logger.info("=" * 80)
    
    # 直接调用 examples/run_main_op_agent.py 中的 interactive_demo 函数
    await interactive_demo(
        backend="ascend",
        arch="ascend910b4",
        device_ids=[0],
        dsl="ascendc",
        framework="torch"
    )
