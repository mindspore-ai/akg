import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.core.worker.local_worker import LocalWorker
from akg_agents.core.async_pool.device_pool import DevicePool

@pytest.mark.asyncio
async def test_verifier_fail_fast_timeout():
    """
    Test that the verifier triggers fail-fast when multiple autotune configs timeout.
    """
    # Create a mock target code with 3 autotune configs
    target_code = """
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['x_size']
)
@triton.jit
def mock_kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    pass
"""

    task_info = {
        "coder_code": target_code
    }

    config = {
        "verify_timeout": 1,  # 1 second timeout
        "log_dir": "./test_logs"
    }

    verifier = KernelVerifier(
        op_name="mock_hang_op",
        framework_code="def get_init_inputs(): return []\ndef get_inputs(): return []\nclass Model: pass",
        task_id="test_timeout_001",
        framework="torch",
        dsl="triton_cuda",
        backend="cuda",
        arch="a100",
        config=config
    )

    # Set up a local worker with a dummy device pool
    device_pool = DevicePool([0])
    verifier.worker = LocalWorker(device_pool=device_pool, backend="cuda")

    # Mock the worker's verify method to simulate a timeout
    async def mock_verify(*args, **kwargs):
        return False, "Verification timed out after 1 seconds.", {}

    with patch.object(verifier.worker, 'verify', side_effect=mock_verify):
        # Run the verifier
        success, log = await verifier.run(task_info, current_step=0, device_id=0)

    # Check that it failed
    assert not success

    # Check that fail-fast was triggered in the log
    assert "连续 2 个 config 验证超时" in log or "Fail-Fast" in log

    # Clean up
    import shutil
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")
