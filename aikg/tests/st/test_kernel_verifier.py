# Copyright 2025 Huawei Technologies Co., Ltd
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

import textwrap
import pytest
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.utils.common_utils import create_log_dir
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.worker.manager import register_local_worker, get_worker_manager
from ..utils import get_device_id

device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.mindspore
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_ascend910b4_mindspore(op_name):
    framework = "mindspore"
    dsl = "triton_ascend"  # 根据测试场景，这里使用ascend
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 统一使用 ModelNew
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
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.mindspore
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("op_name", ["linear"])
@pytest.mark.asyncio
async def test_kernel_verifier_linear_ascend910b4_mindspore(op_name):
    """测试linear算子（mindspore + triton_ascend），验证weight随机种子对齐"""
    framework = "mindspore"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 统一使用 ModelNew
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
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("op_name", ["relu","linear"])
@pytest.mark.asyncio
async def test_kernel_verifier_ascend910b4_torch(op_name):
    framework = "torch"
    dsl = "triton_ascend"  # 根据测试场景，这里使用ascend
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 统一使用 ModelNew
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
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

# ascend310p3


@pytest.mark.level0
@pytest.mark.mindspore
@pytest.mark.swft
@pytest.mark.ascend
@pytest.mark.ascend310p3
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_ascend310p3_mindspore(op_name):
    framework = "mindspore"
    dsl = "swft"
    backend = "ascend"
    arch = "ascend310p3"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = f"{op_name}_{dsl}_{framework}"
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
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.swft
@pytest.mark.ascend
@pytest.mark.ascend310p3
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_ascend310p3_torch(op_name):
    framework = "torch"
    dsl = "swft"
    backend = "ascend"
    arch = "ascend310p3"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = f"{op_name}_{dsl}_{framework}"
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
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.numpy
@pytest.mark.swft
@pytest.mark.ascend
@pytest.mark.ascend310p3
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_ascend310p3_numpy(op_name):
    framework = "numpy"
    dsl = "swft"
    backend = "ascend"
    arch = "ascend310p3"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = f"{op_name}_{dsl}_{framework}"
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
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

# a100


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_a100(op_name):
    framework = "torch"
    dsl = "triton_cuda"  # cuda backend 使用 triton_cuda
    backend = "cuda"
    arch = "a100"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 使用ModelNew而不是函数名
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
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

# v100


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.v100
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_v100(op_name):
    framework = "torch"
    dsl = "triton_cuda"  # cuda backend 使用 triton_cuda
    backend = "cuda"
    arch = "v100"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = f"{op_name}_{dsl}_{framework}"
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
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


# profiling功能测试
@pytest.mark.level0
@pytest.mark.mindspore
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.profiling
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_profiling_ascend910b4_mindspore(op_name):
    framework = "mindspore"
    dsl = "triton_ascend"  # 根据测试场景，这里使用ascend
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_profiling_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 统一使用 ModelNew
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = await verifier.run_profile(
        task_info, current_step=0, device_id=device_id, profile_settings=profile_settings)
    gen_time = result['gen_time']
    base_time = result['base_time']
    speedup = result['speedup']

    print(f"orig performance is {base_time:.2f} us")
    print(f"aikg performance is {gen_time:.2f} us")
    print(f"speedup is {speedup:.2f}x")


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.profiling
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_profiling_ascend910b4_torch(op_name):
    framework = "torch"
    dsl = "triton_ascend"  # 根据测试场景，这里使用ascend
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_profiling_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 统一使用 ModelNew
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = await verifier.run_profile(
        task_info, current_step=0, device_id=device_id, profile_settings=profile_settings)
    gen_time = result['gen_time']
    base_time = result['base_time']
    speedup = result['speedup']

    print(f"orig performance is {base_time:.2f} us")
    print(f"aikg performance is {gen_time:.2f} us")
    print(f"speedup is {speedup:.2f}x")

# profiling功能测试（GPU/CUDA）


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.profiling
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_profiling_a100(op_name):
    framework = "torch"
    dsl = "triton_cuda"  # cuda backend 使用 triton_cuda
    backend = "cuda"
    arch = "a100"
    config = load_config(dsl, backend=backend)  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_profiling_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 统一使用 ModelNew
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = await verifier.run_profile(
        task_info, current_step=0, device_id=device_id, profile_settings=profile_settings)
    gen_time = result['gen_time']
    base_time = result['base_time']
    speedup = result['speedup']
    print(f"orig performance is {base_time:.2f} us")
    print(f"aikg performance is {gen_time:.2f} us")
    print(f"speedup is {speedup:.2f}x")


# profiling功能测试（linear算子）
@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.profiling
@pytest.mark.parametrize("op_name", ["linear"])
@pytest.mark.asyncio
async def test_kernel_verifier_profiling_linear_ascend910b4_torch(op_name):
    """Linear profiling test for ascend910b4_torch"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)

    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="linear_profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = await verifier.run_profile(
        task_info, current_step=0, device_id=device_id, profile_settings=profile_settings)
    gen_time = result['gen_time']
    base_time = result['base_time']
    speedup = result['speedup']

    print(f"Linear Profiling Results:")
    print(f"Operation: {op_name}")
    print(f"orig performance is {base_time:.2f} us")
    print(f"aikg performance is {gen_time:.2f} us")
    print(f"speedup is {speedup:.2f}x")


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.profiling
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_profiling_cpp(op_name):
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "x86_64"
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_cpp_coderonly_config.yaml")  # unused
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_profiling_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 使用ModelNew而不是函数名
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # # 进行性能分析
    # profile_settings = {
    #     "run_times": 50,
    #     "warmup_times": 5
    # }
    # gen_time, base_time, speedup = await verifier.run_profile(
    #     current_step=0, device_id=device_id, profile_settings=profile_settings)
    # print(f"orig performance is {base_time:.2f} us")
    # print(f"aikg performance is {gen_time:.2f} us")
    # print(f"speedup is {speedup:.2f}x")


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.parametrize("op_name", ["linear"])
@pytest.mark.asyncio
async def test_kernel_verifier_linear_cpp(op_name):
    """测试linear算子，验证weight随机种子对齐"""
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "x86_64"
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_cpp_coderonly_config.yaml")
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 使用ModelNew而不是函数名
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="linear_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.parametrize("op_name", ["linear"])
@pytest.mark.asyncio
async def test_kernel_verifier_linear_triton_cuda(op_name):
    """测试linear算子（triton_cuda），验证weight随机种子对齐"""
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    config = load_config(dsl, backend=backend)
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    
    # 新写法：注册 LocalWorker 并从 WorkerManager 获取
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}. Please register a worker first.")
    
    impl_func_name = "ModelNew"  # 使用ModelNew而不是函数名
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="linear_triton_cuda_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


# task_desc 校验功能测试
@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_check_task_desc_static_valid(op_name):
    """测试静态检查：有效的 task_desc 应该通过"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)
    
    # 读取有效的 task_desc
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        valid_task_desc = f.read()
    
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=valid_task_desc,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config
    )
    
    valid, error = verifier.check_task_desc_static(valid_task_desc)
    assert valid, f"静态检查应该通过，但失败了: {error}"


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.asyncio
async def test_check_task_desc_static_missing_model():
    """测试静态检查：缺少 Model 类应该失败"""
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)
    
    # 缺少 Model 类的代码
    invalid_task_desc = """
import torch

def get_inputs():
    return [torch.randn(16, 16384)]

def get_init_inputs():
    return []
"""
    
    verifier = KernelVerifier(
        op_name="test",
        framework_code="",
        framework="torch",
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config
    )
    
    valid, error = verifier.check_task_desc_static(invalid_task_desc)
    assert not valid, "静态检查应该失败（缺少 Model 类）"
    assert "class Model" in error


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.asyncio
async def test_check_task_desc_static_missing_get_inputs():
    """测试静态检查：缺少 get_inputs 函数应该失败"""
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)
    
    # 缺少 get_inputs 函数的代码
    invalid_task_desc = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return torch.relu(x)

def get_init_inputs():
    return []
"""
    
    verifier = KernelVerifier(
        op_name="test",
        framework_code="",
        framework="torch",
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config
    )
    
    valid, error = verifier.check_task_desc_static(invalid_task_desc)
    assert not valid, "静态检查应该失败（缺少 get_inputs）"
    assert "get_inputs" in error


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_check_task_desc_runtime_valid(op_name):
    """测试运行时检查：有效的 task_desc 应该通过"""
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    config = load_config(dsl, backend=backend)
    
    # 读取有效的 task_desc
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        valid_task_desc = f.read()
    
    # 注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}")
    
    try:
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=valid_task_desc,
            task_id="runtime_check_test",
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            config=config,
            worker=worker
        )
        
        valid, error = await verifier.check_task_desc_runtime(valid_task_desc, timeout=60)
        assert valid, f"运行时检查应该通过，但失败了: {error}"
    finally:
        await get_worker_manager().release(worker)
