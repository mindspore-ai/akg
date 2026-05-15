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

"""ST test for SOL-ExecBench + KernelGen-Only workflow on Ascend NPU."""

import json
import os
import tempfile

import pytest

from akg_agents.core.worker.manager import register_local_worker, register_remote_worker
from akg_agents.op.config.config_validator import load_config
from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
from akg_agents.utils.environment_check import check_env_for_task
from ..utils import get_device_id

DEFINITION_JSON = json.dumps({
    "name": "000_mock_relu",
    "description": "A simple ReLU operation for testing SOL-ExecBench integration.",
    "axes": {
        "batch_size": {"type": "var", "description": "Batch size"},
        "seq_len": {"type": "var", "description": "Sequence length"},
        "hidden_size": {"type": "const", "value": 128, "description": "Hidden dimension size"},
    },
    "custom_inputs_entrypoint": None,
    "inputs": {
        "x": {"shape": ["batch_size", "seq_len", "hidden_size"], "dtype": "float32", "description": "Input tensor"}
    },
    "outputs": {
        "out": {"shape": ["batch_size", "seq_len", "hidden_size"], "dtype": "float32", "description": "Output tensor"}
    },
    "reference": (
        "import torch\n\n@torch.no_grad()\ndef run(x: torch.Tensor)"
        " -> torch.Tensor:\n    return torch.nn.functional.relu(x)\n"
    ),
}, indent=4, ensure_ascii=False)

REFERENCE_PY = """\
import torch

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)
"""

WORKLOAD_JSONL = (
    '{"uuid": "1", "axes": {"batch_size": 2, "seq_len": 64}, '
    '"inputs": {"x": {"type": "random"}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 1e-05}}\n'
    '{"uuid": "2", "axes": {"batch_size": 4, "seq_len": 128}, '
    '"inputs": {"x": {"type": "random"}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 1e-05}}\n'
)

TASK_DESC = (
    "请实现一个 Triton Ascend 算子。\n\n"
    f"## definition.json\n```json\n{DEFINITION_JSON}\n```\n\n"
    f"## reference.py\n```python\n{REFERENCE_PY}```\n\n"
    "## workload 示例（共 2 组，以下为第 1 组）\n"
    '```json\n{"uuid": "1", "axes": {"batch_size": 2, "seq_len": 64}, '
    '"inputs": {"x": {"type": "random"}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 1e-05}}\n```\n\n'
    "注意：请使用 Triton 编写 kernel，并将其封装在 ModelNew 类的 forward 方法中。"
)


@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_sol_kernelgen_triton_ascend():
    """SOL-ExecBench 端到端：KernelGen-Only workflow + Triton Ascend NPU"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    worker_mode = os.getenv("AKG_ST_WORKER_MODE", "local").strip().lower()
    remote_url = os.getenv("AKG_ST_WORKER_URL", "http://127.0.0.1:19001").strip()
    device_id = get_device_id()

    config = load_config(
        config_path="./python/akg_agents/op/config/triton_ascend_kernelgen_config.yaml"
    )

    with tempfile.TemporaryDirectory(prefix="sol_st_") as sol_dir:
        for name, content in [
            ("definition.json", DEFINITION_JSON),
            ("reference.py", REFERENCE_PY),
            ("workload.jsonl", WORKLOAD_JSONL),
        ]:
            with open(os.path.join(sol_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        config["sol_problem_dir"] = sol_dir
        config["verify_timeout"] = 300

        check_env_for_task(framework, backend, dsl, config, is_remote=(worker_mode == "remote"))

        if worker_mode == "remote":
            await register_remote_worker(backend=backend, arch=arch, worker_url=remote_url)
        else:
            await register_local_worker([device_id], backend=backend, arch=arch)

        task = AIKGTask(
            op_name="_relu",
            task_desc=TASK_DESC,
            task_id="st_sol_kernelgen_ascend_001",
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            framework=framework,
            workflow="kernelgen_only_workflow",
            bench_type="sol",
        )

        _, success, final_state = await task.run()

    assert success, (
        f"SOL KernelGen Ascend 端到端执行失败: "
        f"{final_state.get('verifier_error') or final_state.get('error_message')}"
    )
