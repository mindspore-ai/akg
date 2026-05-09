import os
import pytest
import textwrap
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.core.worker.manager import register_local_worker, get_worker_manager
from akg_agents.op.config.config_validator import load_config
from ..utils import create_log_dir, get_device_id

os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'
device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.tilelang
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_tilelang_ascend_ascend910b4_torch(op_name):
    """测试 KernelVerifier - TileLang Ascend ReLU"""
    framework = "torch"
    dsl = "tilelang_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)

    op_task_file = f"./tests/op/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    kernel_path = f"./tests/op/resources/{op_name}_op/{op_name}_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')

    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}")

    impl_func_name = "ModelNew"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {"coder_code": kernel_code}
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"
