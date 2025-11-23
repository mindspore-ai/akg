import pytest
import asyncio
import tarfile
import io
import os
import textwrap
import uuid
from ai_kernel_generator.core.worker.local_worker import LocalWorker
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.config.config_validator import load_config
from ..utils import get_device_id

device_id = get_device_id()

@pytest.mark.asyncio
async def test_local_worker_verify_success():
    """Test LocalWorker verification success flow."""
    # 1. Create a dummy verification script
    op_name = "dummy_op"
    script_content = """
import os
import sys

print("Starting verification...")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"ASCEND_VISIBLE_DEVICES={os.environ.get('ASCEND_VISIBLE_DEVICES')}")
print("Verification successful!")
sys.exit(0)
"""
    
    # 2. Create a TAR package in memory
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
        # Create file info
        info = tarfile.TarInfo(name=f"verify_{op_name}.py")
        script_bytes = script_content.encode('utf-8')
        info.size = len(script_bytes)
        tar_file.addfile(tarinfo=info, fileobj=io.BytesIO(script_bytes))
    
    package_data = tar_buffer.getvalue()
    
    # 3. Initialize Worker
    device_pool = DevicePool([0, 1])
    worker = LocalWorker(device_pool, backend="cuda")
    
    # 4. Run verification
    task_id = "test_task_001"
    success, log = await worker.verify(package_data, task_id, op_name, timeout=10)
    
    # 5. Assertions
    assert success is True
    assert "Verification successful!" in log
    assert "CUDA_VISIBLE_DEVICES" in log

@pytest.mark.asyncio
async def test_local_worker_verify_failure():
    """Test LocalWorker verification failure flow."""
    op_name = "dummy_fail_op"
    script_content = """
import sys
print("Verification failed intentionally.")
sys.exit(1)
"""
    
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
        info = tarfile.TarInfo(name=f"verify_{op_name}.py")
        script_bytes = script_content.encode('utf-8')
        info.size = len(script_bytes)
        tar_file.addfile(tarinfo=info, fileobj=io.BytesIO(script_bytes))
    package_data = tar_buffer.getvalue()
    
    device_pool = DevicePool([0])
    worker = LocalWorker(device_pool, backend="cuda")
    
    success, log = await worker.verify(package_data, "test_task_002", op_name, timeout=10)
    
    assert success is False
    assert "Verification failed intentionally" in log

@pytest.mark.asyncio
async def test_local_worker_timeout():
    """Test LocalWorker timeout."""
    op_name = "dummy_timeout_op"
    script_content = """
import time
import sys
print("Sleeping...")
time.sleep(5)
sys.exit(0)
"""
    
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
        info = tarfile.TarInfo(name=f"verify_{op_name}.py")
        script_bytes = script_content.encode('utf-8')
        info.size = len(script_bytes)
        tar_file.addfile(tarinfo=info, fileobj=io.BytesIO(script_bytes))
    package_data = tar_buffer.getvalue()
    
    device_pool = DevicePool([0])
    worker = LocalWorker(device_pool, backend="cuda")
    
    # Set short timeout
    success, log = await worker.verify(package_data, "test_task_003", op_name, timeout=1)
    
    assert success is False
    assert "timed out" in log


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.asyncio
async def test_local_worker_real_triton_cuda_relu():
    """Test LocalWorker with real Torch CUDA Triton ReLU verification."""
    # 1. Setup
    op_name = "relu"
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    
    # 2. Load config and code
    config = load_config(dsl, backend=backend)
    
    # Read framework code
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())
    
    # Read kernel code
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()
    
    # 3. Create KernelVerifier to generate verification project
    log_dir = os.path.expanduser("~/.aikg/test_logs/worker_test")
    os.makedirs(log_dir, exist_ok=True)
    config['log_dir'] = log_dir
    
    # Create a LocalWorker (test_worker.py 需要直接使用 DevicePool 测试 Worker)
    device_pool = DevicePool([device_id])
    worker = LocalWorker(device_pool, backend=backend)
    
    # Use unique task_id to avoid directory conflicts
    task_id = str(uuid.uuid4())[:8]
    
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id=task_id,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name="ModelNew",
        config=config,
        worker=worker
    )
    
    # 4. Run verification through worker
    task_info = {"coder_code": kernel_code}
    success, error_log = await verifier.run(task_info, current_step=0, device_id=0)
    
    # 5. Assertions
    assert success, f"验证失败: {error_log}"
    print(f"✓ LocalWorker 成功验证 {op_name} ({framework}/{dsl}/{backend})")


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.asyncio
async def test_local_worker_real_triton_cuda_relu_profile():
    """Test LocalWorker with real Torch CUDA Triton ReLU profiling."""
    # 1. Setup
    op_name = "relu"
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    
    # 2. Load config and code
    config = load_config(dsl, backend=backend)
    
    # Read framework code
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())
    
    # Read kernel code
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()
    
    # 3. Create KernelVerifier with LocalWorker
    log_dir = os.path.expanduser("~/.aikg/test_logs/worker_profile_test")
    os.makedirs(log_dir, exist_ok=True)
    config['log_dir'] = log_dir
    
    # test_worker.py 需要直接使用 DevicePool 测试 Worker
    device_pool = DevicePool([device_id])
    worker = LocalWorker(device_pool, backend=backend)
    
    # Use unique task_id to avoid directory conflicts
    task_id = str(uuid.uuid4())[:8]
    
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id=task_id,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name="ModelNew",
        config=config,
        worker=worker
    )
    
    # 4. First verify (to generate verification project)
    task_info = {"coder_code": kernel_code}
    success, error_log = await verifier.run(task_info, current_step=0, device_id=0)
    assert success, f"验证失败: {error_log}"
    
    # 5. Run profiling through worker
    profile_settings = {
        "run_times": 10,
        "warmup_times": 2
    }
    result = await verifier.run_profile(task_info, current_step=0, device_id="0", profile_settings=profile_settings)
    
    # 6. Assertions
    assert result is not None
    assert 'gen_time' in result
    assert 'base_time' in result
    assert 'speedup' in result
    assert result['gen_time'] < float('inf'), "生成代码时间应该是有效值"
    assert result['base_time'] > 0, "基准代码时间应该大于0"
    
    print(f"✓ LocalWorker 成功 profile {op_name}")
    print(f"  Base time: {result['base_time']:.2f} us")
    print(f"  Gen time: {result['gen_time']:.2f} us")
    print(f"  Speedup: {result['speedup']:.2f}x")


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.asyncio
async def test_remote_worker_via_local_service():
    """Test RemoteWorker by simulating the Worker Service with LocalWorker.
    
    This test simulates the remote worker scenario:
    1. Worker Service (FastAPI) wraps LocalWorker
    2. RemoteWorker (HTTP client) calls the service
    3. Service delegates to LocalWorker
    
    Note: This test directly uses LocalWorker to simulate what RemoteWorker would do.
    In production, RemoteWorker would make HTTP calls to the Worker Service.
    """
    # 1. Setup test data
    op_name = "relu"
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    
    config = load_config(dsl, backend=backend)
    
    # Read framework code
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())
    
    # Read kernel code
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()
    
    # 3. Create KernelVerifier with LocalWorker (simulating RemoteWorker behavior)
    log_dir = os.path.expanduser("~/.aikg/test_logs/remote_worker_test")
    os.makedirs(log_dir, exist_ok=True)
    config['log_dir'] = log_dir
    
    # Use LocalWorker directly to simulate what RemoteWorker would do via HTTP
    # In a real scenario, RemoteWorker would make HTTP calls to the Worker Service
    # test_worker.py 需要直接使用 DevicePool 测试 Worker
    device_pool = DevicePool([device_id])
    worker = LocalWorker(device_pool, backend=backend)
    
    # Use unique task_id to avoid directory conflicts
    task_id = str(uuid.uuid4())[:8]
    
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id=task_id,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name="ModelNew",
        config=config,
        worker=worker
    )
    
    # 4. Run verification
    task_info = {"coder_code": kernel_code}
    success, error_log = await verifier.run(task_info, current_step=0, device_id=0)
    
    # 5. Assertions
    assert success, f"RemoteWorker 验证失败: {error_log}"
    print(f"✓ RemoteWorker (via LocalWorker) 成功验证 {op_name}")
    print(f"  Note: This test uses LocalWorker to simulate RemoteWorker behavior")
    print(f"  In production, RemoteWorker would call Worker Service via HTTP")
