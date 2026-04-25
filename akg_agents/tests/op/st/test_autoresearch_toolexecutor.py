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

"""Autoresearch 端到端系统测试 — ToolExecutor 生产路径 (ReLU)

算子: relu(x), shape (11, 37, 8191)
选型理由同 test_autoresearch.py.

覆盖 KernelAgent → OpToolExecutor → AutoresearchWorkflow 的完整 callable workflow 链路:
  KernelAgent.__init__() → tool_executor.execute("call_autoresearch_workflow", args)
    → _execute_workflow() → prepare_config() → build_initial_state() → app.ainvoke()
    → format_result()

额外验证: prepare_config deep-copy 隔离.
自动检测设备: 有 NPU 用 triton_ascend, 有 GPU 用 triton_cuda, 都没有则跳过.
"""

import copy
import tempfile
import torch
import pytest
from akg_agents.op.agents.kernel_agent import KernelAgent
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.task_label import resolve_task_label
from ..utils import get_device_id


def _detect_backend():
    """Detect available accelerator. Returns (dsl, backend, arch, device_str) or None."""
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "triton_ascend", "ascend", "ascend910b4", "npu"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "triton_cuda", "cuda", "a100", "cuda"
    return None


_backend_info = _detect_backend()

pytestmark = pytest.mark.skipif(
    _backend_info is None,
    reason="No NPU or GPU available — skipping autoresearch ToolExecutor ST",
)


def _make_task_desc(device_str: str) -> str:
    return f'''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

def get_inputs():
    return [torch.randn(11, 37, 8191, device='{device_str}')]

def get_init_inputs():
    return []
'''


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.asyncio
async def test_autoresearch_toolexecutor():
    """端到端测试: ToolExecutor 生产路径 — autoresearch 优化 ReLU (自动检测后端)"""
    dsl, backend, arch, device_str = _backend_info

    op_name = "akg_agents_relu_te"
    device_id = get_device_id()
    await register_local_worker([device_id], backend=backend, arch=arch)

    config = load_config(dsl=dsl, backend=backend)
    config["task_label"] = resolve_task_label(op_name=op_name, parallel_index=1)
    config["max_step"] = 10
    config["gen_retries"] = 10

    # ---- 1. Instantiate KernelAgent ----
    agent = KernelAgent(
        task_id="autoresearch_te_test_001",
        model_level="standard",
        config=config,
        framework="torch",
        backend=backend,
        arch=arch,
        dsl=dsl,
    )

    # ---- 2. Verify workflow is registered ----
    assert "call_autoresearch_workflow" in agent.workflow_registry, (
        "autoresearch workflow not found in KernelAgent workflow_registry"
    )

    # ---- 3. Snapshot cached config BEFORE workflow execution ----
    pre_run_config = copy.deepcopy(agent._get_workflow_resources()["config"])

    # ---- 4. Execute via ToolExecutor (production path) ----
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = await agent.tool_executor.execute(
            tool_name="call_autoresearch_workflow",
            arguments={
                "op_name": op_name,
                "task_desc": _make_task_desc(device_str),
                "dsl": dsl,
                "framework": "torch",
                "backend": backend,
                "arch": arch,
                "cur_path": tmp_dir,
                "max_rounds": 10,
            },
        )

    # ---- 5. Verify format_result() output structure ----
    assert isinstance(result, dict)
    assert "status" in result, f"result missing 'status': {result}"
    assert result["status"] == "success", (
        f"ToolExecutor autoresearch failed: {result.get('error_information', '')}"
    )
    assert result.get("code"), "result missing generated code"
    assert result.get("profile"), "result missing profile data"

    # ---- 6. Verify config isolation (prepare_config deep-copy) ----
    post_run_config = agent._get_workflow_resources()["config"]
    for key in ("use_reference_data", "log_dir"):
        pre_val = pre_run_config.get(key)
        post_val = post_run_config.get(key)
        assert pre_val == post_val, (
            f"config['{key}'] leaked: was {pre_val!r}, now {post_val!r}"
        )
    pre_amc = pre_run_config.get("agent_model_config", {})
    post_amc = post_run_config.get("agent_model_config", {})
    assert pre_amc == post_amc, (
        f"agent_model_config leaked: diff {set(pre_amc.items()) ^ set(post_amc.items())}"
    )
