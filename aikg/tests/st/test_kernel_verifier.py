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
from ..utils import get_device_id

device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.mindspore
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("op_name", ["relu"])
def test_kernel_verifier_ascend910b4_mindspore(op_name):
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
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("op_name", ["relu"])
def test_kernel_verifier_ascend910b4_torch(op_name):
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
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

# ascend310p3


@pytest.mark.level0
@pytest.mark.mindspore
@pytest.mark.swft
@pytest.mark.ascend
@pytest.mark.ascend310p3
@pytest.mark.parametrize("op_name", ["relu"])
def test_kernel_verifier_ascend310p3_mindspore(op_name):
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
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.swft
@pytest.mark.ascend
@pytest.mark.ascend310p3
@pytest.mark.parametrize("op_name", ["relu"])
def test_kernel_verifier_ascend310p3_torch(op_name):
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
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.numpy
@pytest.mark.swft
@pytest.mark.ascend
@pytest.mark.ascend310p3
@pytest.mark.parametrize("op_name", ["relu"])
def test_kernel_verifier_ascend310p3_numpy(op_name):
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
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

# a100


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.parametrize("op_name", ["relu"])
def test_kernel_verifier_a100(op_name):
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
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

# v100


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.v100
@pytest.mark.parametrize("op_name", ["relu"])
def test_kernel_verifier_v100(op_name):
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
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"


# profiling功能测试
@pytest.mark.level0
@pytest.mark.mindspore
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.profiling
@pytest.mark.parametrize("op_name", ["relu"])
def test_kernel_verifier_profiling_ascend910b4_mindspore(op_name):
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
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_profiling_test')
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = verifier.run_profile(
        current_step=0, device_id=device_id, profile_settings=profile_settings)
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
def test_kernel_verifier_profiling_ascend910b4_torch(op_name):
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
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_profiling_test')
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = verifier.run_profile(
        current_step=0, device_id=device_id, profile_settings=profile_settings)
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
def test_kernel_verifier_profiling_a100(op_name):
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
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = verifier.run_profile(
        current_step=0, device_id=device_id, profile_settings=profile_settings)
    gen_time = result['gen_time']
    base_time = result['base_time']
    speedup = result['speedup']
    print(f"orig performance is {base_time:.2f} us")
    print(f"aikg performance is {gen_time:.2f} us")
    print(f"speedup is {speedup:.2f}x")


# AIKGBench dynamic shape profiling功能测试
@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.profiling
@pytest.mark.parametrize("op_name", ["add_dyn"])
def test_kernel_verifier_profiling_dynamic_ascend910b4_torch(op_name):
    """Dynamic shape profiling test for ascend910b4_torch"""
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
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="dynamic_profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = verifier.run_profile(
        current_step=0, device_id=device_id, profile_settings=profile_settings)
    gen_time = result['gen_time']
    base_time = result['base_time']
    speedup = result['speedup']

    print(f"Dynamic Shape Profiling Results:")
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
def test_kernel_verifier_profiling_cpp(op_name):
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
    impl_func_name = f"{op_name}_{dsl}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        task_id="profiling_test_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
    )
    task_info = {}
    task_info["coder_code"] = kernel_code

    # 先进行验证，确保验证通过
    result, error_log = verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    result = verifier.run_profile(
        current_step=0, device_id=device_id, profile_settings=profile_settings)
    gen_time = result['gen_time']
    base_time = result['base_time']
    speedup = result['speedup']
    print(f"orig performance is {base_time:.2f} us")
    print(f"aikg performance is {gen_time:.2f} us")
    print(f"speedup is {speedup:.2f}x")
