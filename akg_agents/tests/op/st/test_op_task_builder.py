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

"""OpTaskBuilder ST 测试。

当前文件只保留一个最简单的 ReLU 场景用例。
"""

import os

import pytest

from akg_agents.core.worker.manager import register_local_worker
from akg_agents.op.config.config_validator import load_config
from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderStatus
from akg_agents.op.workflows.op_task_builder_workflow import run_op_task_builder

from ..utils import get_device_id

os.environ["AKG_AGENTS_STREAM_OUTPUT"] = "on"

FRAMEWORK = "torch"
BACKEND = "ascend"
ARCH = "ascend910b4"
DSL = "triton_ascend"
DEVICE_ID = get_device_id()


async def ensure_worker_registered():
    """确保本地 Ascend worker 已注册。"""
    from akg_agents.core.worker.manager import get_worker_manager

    worker_manager = get_worker_manager()
    if not await worker_manager.has_worker(backend=BACKEND, arch=ARCH):
        await register_local_worker([DEVICE_ID], backend=BACKEND, arch=ARCH)


def get_test_config() -> dict:
    """优先加载默认配置，失败时退化到最小可用配置。"""
    try:
        config = load_config(config_path="./python/akg_agents/op/config/triton_ascend_coderonly_config.yaml")
        agent_model_config = config.get("agent_model_config", {})
        if "op_task_builder" not in agent_model_config:
            agent_model_config["op_task_builder"] = agent_model_config.get("coder", "default")
            config["agent_model_config"] = agent_model_config
        return config
    except Exception as exc:
        print(f"Warning: failed to load config file: {exc}, using minimal config")
        return {
            "agent_model_config": {
                "op_task_builder": "default",
                "coder": "default",
            },
            "log_dir": "/tmp/akg_agents_test",
            "op_task_builder_max_iterations": 5,
        }


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.asyncio
async def test_simple_relu_request():
    """明确的 ReLU 需求应返回可继续处理的结果。"""
    await ensure_worker_registered()
    config = get_test_config()

    result = await run_op_task_builder(
        user_input="我需要一个ReLU激活函数的算子，输入是16x16384的张量",
        config=config,
        framework=FRAMEWORK,
        backend=BACKEND,
        arch=ARCH,
        dsl=DSL,
    )

    assert result.get("status") in [
        OpTaskBuilderStatus.READY,
        OpTaskBuilderStatus.NEED_CLARIFICATION,
    ]

    if result.get("status") == OpTaskBuilderStatus.READY:
        generated_task_desc = result.get("generated_task_desc", "")
        assert result.get("op_name")
        assert generated_task_desc
        assert "class Model" in generated_task_desc
