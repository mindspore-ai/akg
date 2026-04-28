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

"""
ST test for SOL-ExecBench verify and profile functionality on Ascend NPU.
"""

import json
import os
import tempfile
import pytest

from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker, get_worker_manager
from akg_agents.utils.common_utils import create_log_dir
from ..utils import get_device_id


# Mock SOL data in FlashInfer-Bench format (hf_id="" is the key edge case)
DEFINITION_JSON = json.dumps({
    "name": "mock_relu",
    "hf_id": "",  # Empty string - triggers Pydantic NonEmptyString validation error if not preprocessed
    "description": "ReLU operation for testing SOL-ExecBench verify functionality.",
    "axes": {
        "batch_size": {"type": "var"},
        "hidden_size": {"type": "const", "value": 128},
    },
    "custom_inputs_entrypoint": None,
    "inputs": {
        "x": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}
    },
    "outputs": {
        "y": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}
    },
    "reference": "import torch\n\n@torch.no_grad()\ndef run(x: torch.Tensor) -> torch.Tensor:\n    return torch.nn.functional.relu(x)\n",
}, indent=4, ensure_ascii=False)

REFERENCE_PY = """\
import torch

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)
"""

WORKLOAD_JSONL = (
    '{"uuid": "1", "axes": {"batch_size": 2}, '
    '"inputs": {"x": {"type": "random"}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 1e-05}}\n'
    '{"uuid": "2", "axes": {"batch_size": 4}, '
    '"inputs": {"x": {"type": "random"}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 1e-05}}\n'
)

device_id = get_device_id()


@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.asyncio
async def test_sol_verify_ascend():
    """Test SOL verify functionality"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    config = load_config(dsl, backend=backend)

    # Read existing relu implementation
    kernel_path = f"./tests/op/resources/relu_op/relu_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    # Create mock SOL problem directory
    with tempfile.TemporaryDirectory(prefix="sol_verify_test_") as sol_dir:
        # Write SOL data files
        for name, content in [
            ("definition.json", DEFINITION_JSON),
            ("reference.py", REFERENCE_PY),
            ("workload.jsonl", WORKLOAD_JSONL),
        ]:
            with open(os.path.join(sol_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        config["sol_problem_dir"] = sol_dir

        # Register worker
        await register_local_worker([device_id], backend=backend, arch=arch)
        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            raise RuntimeError(f"No available worker for backend={backend}, arch={arch}")

        verifier = KernelVerifier(
            op_name="relu",
            framework_code="",  # SOL mode doesn't use framework_code
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            impl_func_name="ModelNew",
            config=config,
            worker=worker,
            bench_type="sol",
        )

        task_info = {"coder_code": kernel_code}
        result, error_log = await verifier.run(task_info, device_id=device_id)

        assert result, f"SOL verify failed (hf_id preprocessing issue): {error_log}"


@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.profiling
@pytest.mark.asyncio
async def test_sol_profile_ascend():
    """Test SOL profile functionality"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    config = load_config(dsl, backend=backend)

    # Read existing relu implementation
    kernel_path = f"./tests/op/resources/relu_op/relu_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    # Create mock SOL problem directory
    with tempfile.TemporaryDirectory(prefix="sol_profile_test_") as sol_dir:
        # Write SOL data files
        for name, content in [
            ("definition.json", DEFINITION_JSON),
            ("reference.py", REFERENCE_PY),
            ("workload.jsonl", WORKLOAD_JSONL),
        ]:
            with open(os.path.join(sol_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        config["sol_problem_dir"] = sol_dir

        # Register worker
        await register_local_worker([device_id], backend=backend, arch=arch)
        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            raise RuntimeError(f"No available worker for backend={backend}, arch={arch}")

        verifier = KernelVerifier(
            op_name="relu",
            framework_code="",  # SOL mode doesn't use framework_code
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            impl_func_name="ModelNew",
            config=config,
            worker=worker,
            bench_type="sol",
        )

        # First verify the implementation
        task_info = {"coder_code": kernel_code}
        result, error_log = await verifier.run(task_info, device_id=device_id)
        assert result, f"SOL verify failed: {error_log}"

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

        print(f"SOL profile result: base={profile_result['base_time']:.2f}us, gen={profile_result['gen_time']:.2f}us, speedup={profile_result['speedup']:.2f}x")