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

"""
ST test for CANN-Bench verify and profile functionality on Ascend NPU.
"""

import os
import tempfile
import yaml
import pytest

from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker, get_worker_manager
from akg_agents.utils.common_utils import create_log_dir
from ..utils import get_device_id


PROTO_YAML = yaml.dump({
    "operator": {
        "name": "Relu",
        "category": "Elementwise",
        "difficulty": "L1",
        "formula": "y = max(0, x)",
        "description": "ReLU activation for testing CANN-Bench verify functionality",
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

device_id = get_device_id()


@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.asyncio
async def test_cann_verify_ascend():
    """Test CANN-Bench verify functionality"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    config = load_config(dsl, backend=backend)

    # Read existing relu implementation (matches proto.yaml definition)
    kernel_path = f"./tests/op/resources/relu_op/relu_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    # Create mock CANN-Bench problem directory
    with tempfile.TemporaryDirectory(prefix="cann_verify_test_") as cann_dir:
        for name, content in [
            ("proto.yaml", PROTO_YAML),
            ("golden.py", GOLDEN_PY),
            ("cases.yaml", CASES_YAML),
        ]:
            with open(os.path.join(cann_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        config["cann_problem_dir"] = cann_dir

        # Register worker
        await register_local_worker([device_id], backend=backend, arch=arch)
        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            raise RuntimeError(f"No available worker for backend={backend}, arch={arch}")

        verifier = KernelVerifier(
            op_name="relu",
            framework_code="",
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            impl_func_name="ModelNew",
            config=config,
            worker=worker,
            bench_type="cann",
        )

        task_info = {"coder_code": kernel_code}
        result, error_log = await verifier.run(task_info, device_id=device_id)

        assert result, f"CANN verify failed: {error_log}"


@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.profiling
@pytest.mark.asyncio
async def test_cann_profile_ascend():
    """Test CANN-Bench profile functionality"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    config = load_config(dsl, backend=backend)

    # Read existing relu implementation
    kernel_path = f"./tests/op/resources/relu_op/relu_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    # Create mock CANN-Bench problem directory
    with tempfile.TemporaryDirectory(prefix="cann_profile_test_") as cann_dir:
        for name, content in [
            ("proto.yaml", PROTO_YAML),
            ("golden.py", GOLDEN_PY),
            ("cases.yaml", CASES_YAML),
        ]:
            with open(os.path.join(cann_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        config["cann_problem_dir"] = cann_dir

        # Register worker
        await register_local_worker([device_id], backend=backend, arch=arch)
        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            raise RuntimeError(f"No available worker for backend={backend}, arch={arch}")

        verifier = KernelVerifier(
            op_name="relu",
            framework_code="",
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            impl_func_name="ModelNew",
            config=config,
            worker=worker,
            bench_type="cann",
        )

        # First verify the implementation
        task_info = {"coder_code": kernel_code}
        result, error_log = await verifier.run(task_info, device_id=device_id)
        assert result, f"CANN verify failed: {error_log}"

        # Then run profile
        profile_settings = {
            "warmup_times": 5,
            "run_times": 20,
        }
        profile_result = await verifier.run_profile(
            task_info,
            current_step=0,
            device_id=device_id,
            profile_settings=profile_settings,
        )

        assert profile_result is not None, "Profile result should not be None"
        assert "base_time" in profile_result, "Should contain base_time"
        assert "gen_time" in profile_result, "Should contain gen_time"
        assert "speedup" in profile_result, "Should contain speedup"

        print(f"CANN profile result: base={profile_result['base_time']:.2f}us, gen={profile_result['gen_time']:.2f}us, speedup={profile_result['speedup']:.2f}x")