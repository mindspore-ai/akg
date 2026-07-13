# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""ST test for CANN-Bench + KernelGen-Only workflow on Ascend NPU."""

import os
import tempfile
import yaml

import pytest

from akg_agents.core.worker.manager import register_local_worker, register_remote_worker
from akg_agents.op.config.config_validator import load_config
from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
from akg_agents.op.cann_correctness import get_cann_task_desc_for_prompt
from akg_agents.utils.environment_check import check_env_for_task
from ..utils import get_device_id

PROTO_YAML = yaml.dump({
    "operator": {
        "name": "Relu",
        "category": "Elementwise",
        "difficulty": "L1",
        "formula": "y = max(0, x)",
        "description": "ReLU activation for testing CANN-Bench integration",
        "attrs": [],
        "inputs": [
            {"name": "x", "description": "Input tensor", "dtype": ["float32", "float16"]},
        ],
        "outputs": [
            {"name": "y", "description": "Output tensor", "dtype": ["float32", "float16"]},
        ],
        "schema": "relu(Tensor x) -> Tensor y",
    }
}, default_flow_style=False, allow_unicode=True)

GOLDEN_PY = """\
import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)
"""

CASES_YAML = yaml.dump({
    "cases": [
        {
            "operator": "Relu",
            "case_id": 1,
            "input_shape": [[64, 128]],
            "dtype": ["float32"],
            "value_range": [-1, 1],
            "baseline_perf_us": 5.0,
            "t_hw_us": 1.0,
        },
        {
            "operator": "Relu",
            "case_id": 2,
            "input_shape": [[128, 256]],
            "dtype": ["float16"],
            "value_range": [-2, 2],
            "baseline_perf_us": 3.0,
            "t_hw_us": 0.5,
        },
    ]
}, default_flow_style=False, allow_unicode=True)

DESC_MD = "ReLU activation for CANN-Bench integration testing."


@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_cann_kernelgen_triton_ascend():
    """CANN-Bench 端到端：KernelGen-Only workflow + Triton Ascend NPU"""
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

    with tempfile.TemporaryDirectory(prefix="cann_st_") as cann_dir:
        for name, content in [
            ("proto.yaml", PROTO_YAML),
            ("golden.py", GOLDEN_PY),
            ("cases.yaml", CASES_YAML),
            ("desc.md", DESC_MD),
        ]:
            with open(os.path.join(cann_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        config["cann_problem_dir"] = cann_dir
        config["verify_timeout"] = 300

        check_env_for_task(framework, backend, dsl, config, is_remote=(worker_mode == "remote"))

        if worker_mode == "remote":
            await register_remote_worker(backend=backend, arch=arch, worker_url=remote_url)
        else:
            await register_local_worker([device_id], backend=backend, arch=arch)

        task_desc = get_cann_task_desc_for_prompt(cann_dir)

        task = AIKGTask(
            op_name="relu",
            task_desc=task_desc,
            task_id="st_cann_kernelgen_ascend_001",
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            framework=framework,
            workflow="kernelgen_only_workflow",
            bench_type="cann",
        )

        _, success, final_state = await task.run()

    assert success, (
        f"CANN KernelGen Ascend 端到端执行失败: "
        f"{final_state.get('verifier_error') or final_state.get('error_message')}"
    )
