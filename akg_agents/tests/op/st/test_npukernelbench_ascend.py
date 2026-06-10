import os
import pytest

from akg_agents.core.worker.manager import (
    register_local_worker, register_remote_worker, get_worker_manager,
)
from akg_agents.op.config.config_validator import load_config
from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.utils.environment_check import check_env_for_task
from ..utils import get_device_id


# ============================================================
# Fixed inputs — copied verbatim from upstream NPUKernelBench/level1/3_Add.py
# (kept identical for the byte-equality guarantee enforced by the
# task_desc loader contract).
# ============================================================

TASK_DESC = """\
import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    \"\"\"
    Simple model that performs element-wise addition with broadcasting support.
    \"\"\"
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        \"\"\"
        Applies element-wise addition to the input tensors with broadcasting support.

        Args:
            x (torch.Tensor): First input tensor of any shape.
            y (torch.Tensor): Second input tensor, broadcastable with x.
            alpha (float, optional): The multiplier for y.

        Returns:
            torch.Tensor: Output tensor x + alpha * y, shape follows broadcasting rules.
        \"\"\"
        return torch.add(x, y, alpha=alpha)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_Add.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        y_info = inputs[1]
        alpha_info = inputs[2]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        y = torch.randn(y_info["shape"], dtype=dtype)
        alpha = alpha_info["value"]
        input_groups.append([x, y, alpha])
    return input_groups


def get_init_inputs():
    return []
"""

# Only the first two cases from upstream 3_Add.json (both float32 1D);
# enough to keep the dynamic-shape pipeline exercised while keeping
# this ST test fast and deterministic.
INPUT_GROUPS_JSONL = (
    '{"inputs": [{"name": "x", "type": "tensor", "required": true, '
    '"dtype": "float32", "shape": [128]}, '
    '{"name": "y", "type": "tensor", "required": true, '
    '"dtype": "float32", "shape": [128]}, '
    '{"name": "alpha", "type": "attr", "required": false, '
    '"dtype": "float", "value": 1.0}]}\n'
    '{"inputs": [{"name": "x", "type": "tensor", "required": true, '
    '"dtype": "float32", "shape": [256]}, '
    '{"name": "y", "type": "tensor", "required": true, '
    '"dtype": "float32", "shape": [256]}, '
    '{"name": "alpha", "type": "attr", "required": false, '
    '"dtype": "float", "value": 2.0}]}\n'
)

ADD_TRITON_ASCEND_MODELNEW = """\
import torch
import triton
import triton.language as tl


@triton.jit
def add_alpha_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + alpha * y
    tl.store(output_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # Broadcast first so the kernel only walks contiguous, identically
        # shaped tensors — matches torch.add(x, y, alpha=alpha) semantics.
        x_b, y_b = torch.broadcast_tensors(x, y)
        x_c = x_b.contiguous()
        y_c = y_b.contiguous()

        n_elements = x_c.numel()
        output = torch.empty_like(x_c)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        add_alpha_kernel[grid](
            x_c, y_c, output, n_elements, float(alpha),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
"""

OP_NAME = "npukb_3_add"
SIDECAR_JSON_NAME = "3_Add.json"
device_id = get_device_id()


def _build_npukb_config(loader_config):
    """Inject NPUKernelBench metadata into ``config`` — same channel SOL uses.

    This is the heart of the unified extensibility contract: any
    benchmark-specific knowledge that ``KernelVerifier`` needs at the
    very end of the call chain rides on the ``config`` dict, so that
    every intermediate layer (workflow / controller / task) stays
    benchmark-agnostic.
    """
    loader_config["framework_aux_files"] = {SIDECAR_JSON_NAME: INPUT_GROUPS_JSONL}
    # framework_factory_names left default — auto-detected from ref source.
    return loader_config


# ============================================================
# Test 1: Verifier-only ST (mirrors test_sol_verifier_ascend)
# ============================================================

@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.asyncio
async def test_npukernelbench_add_verify_ascend():
    """Run KernelVerifier with a fixed Triton ModelNew on NPUKB 3_Add."""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    config = load_config(dsl, backend=backend)
    _build_npukb_config(config)

    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(
            f"No available worker for backend={backend}, arch={arch}"
        )

    verifier = KernelVerifier(
        op_name=OP_NAME,
        framework_code=TASK_DESC,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name="ModelNew",
        config=config,
        worker=worker,
    )

    # Sanity-check the unified config contract before invoking verify.
    # framework_factory_names is now empty by default — KernelVerifier
    # auto-detects the multi-shape factory from the ref source.
    assert verifier.framework_aux_files == {SIDECAR_JSON_NAME: INPUT_GROUPS_JSONL}
    assert verifier._resolve_dyn_factory() == "get_input_groups"
    assert verifier._detect_dynamic_shape() is True

    task_info = {"coder_code": ADD_TRITON_ASCEND_MODELNEW}
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"NPUKernelBench 3_Add verify failed: {error_log}"


# ============================================================
# Test 2: Full kernelgen workflow ST (mirrors test_sol_kernelgen_ascend)
# ============================================================

@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_npukernelbench_add_kernelgen_ascend():
    """End-to-end kernelgen_only_workflow on NPUKernelBench 3_Add."""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    worker_mode = os.getenv("AKG_ST_WORKER_MODE", "local").strip().lower()
    remote_url = os.getenv("AKG_ST_WORKER_URL", "http://127.0.0.1:19001").strip()

    config = load_config(
        config_path="./python/akg_agents/op/config/triton_ascend_kernelgen_config.yaml"
    )
    _build_npukb_config(config)
    config["verify_timeout"] = 300

    check_env_for_task(framework, backend, dsl, config, is_remote=(worker_mode == "remote"))

    if worker_mode == "remote":
        await register_remote_worker(backend=backend, arch=arch, worker_url=remote_url)
    else:
        await register_local_worker([device_id], backend=backend, arch=arch)

    task = AIKGTask(
        op_name=OP_NAME,
        task_desc=TASK_DESC,
        task_id="st_npukb_kernelgen_ascend_3_add",
        backend=backend,
        arch=arch,
        dsl=dsl,
        config=config,
        framework=framework,
        workflow="kernelgen_only_workflow",
    )

    _, success, final_state = await task.run()

    assert success, (
        f"NPUKernelBench 3_Add kernelgen failed: "
        f"{final_state.get('verifier_error') or final_state.get('error_message')}"
    )
