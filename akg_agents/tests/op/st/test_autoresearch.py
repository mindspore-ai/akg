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

"""Autoresearch 端到端系统测试 — ReLU

算子: torch.relu(x)
Shape: (11, 37, 8191)

选型理由:
  - ReLU 是最简单的 elementwise 算子, 任何模型都能写对, 确保 CI 稳定
  - (11, 37, 8191) 全是质数, 8191 = 2^13 - 1, 任何 BLOCK_SIZE 都需要 mask 处理
  - 8191 足够大, 容易爆 UB (unified buffer) 如果 BLOCK_SIZE 选得太大
  - 总元素 11×37×8191 ≈ 3.3M, 不会 OOM 但足以暴露性能问题

自动检测设备: 有 NPU 用 triton_ascend, 有 GPU 用 triton_cuda, 都没有则跳过.
性能门禁: speedup > 1.0x (kernel 必须不比 ref 慢)
"""

import torch
import pytest
from akg_agents.op.langgraph_op.task import LangGraphTask
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

# Skip entire module if no accelerator available
pytestmark = pytest.mark.skipif(
    _backend_info is None,
    reason="No NPU or GPU available — skipping autoresearch ST",
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
async def test_autoresearch_relu():
    """端到端测试: autoresearch 优化 ReLU (自动检测后端)

    覆盖完整链路: preflight → seed 三层验证 → AgentLoop → eval → result
    """
    dsl, backend, arch, device_str = _backend_info

    op_name = "akg_agents_relu"
    device_id = get_device_id()
    await register_local_worker([device_id], backend=backend, arch=arch)

    config = load_config(dsl=dsl, backend=backend)
    config["task_label"] = resolve_task_label(op_name=op_name, parallel_index=1)
    config["max_step"] = 10
    config["gen_retries"] = 10

    task = LangGraphTask(
        op_name=op_name,
        task_desc=_make_task_desc(device_str),
        task_id="autoresearch_test_001",
        backend=backend,
        arch=arch,
        dsl=dsl,
        config=config,
        framework="torch",
        workflow="autoresearch",
    )

    result_op_name, success, final_state = await task.run()

    assert result_op_name == op_name
    assert isinstance(final_state, dict)
    assert success, f"autoresearch workflow failed: {final_state.get('verifier_error', '')}"
    assert final_state.get("verifier_result") is True

    coder_code = final_state.get("coder_code", "")
    assert len(coder_code) > 0

    profile_res = final_state.get("profile_res", {})
    assert "gen_time" in profile_res
    gen_time = profile_res["gen_time"]
    base_time = profile_res.get("base_time")

    if base_time and base_time > 0 and gen_time and gen_time > 0:
        speedup = base_time / gen_time
        assert speedup > 1.0, (
            f"kernel slower than ref: {gen_time:.1f}us vs ref {base_time:.1f}us "
            f"(speedup={speedup:.2f}x, required >1.0x)"
        )
