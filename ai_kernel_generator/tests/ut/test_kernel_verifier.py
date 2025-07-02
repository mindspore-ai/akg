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
from ai_kernel_generator.core.utils import ParsedCode


@pytest.mark.level0
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.parametrize("framework,impl_type,backend", [
    ("mindspore", "triton", "ascend"),
    ("torch", "triton", "ascend"),
])
def test_kernel_verifier_ascend910b4(op_name, framework, impl_type, backend):
    arch = "ascend910b4"
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{impl_type}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{impl_type}_test')
    impl_func_name = f"{op_name}_{impl_type}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        log_dir=log_dir,
        framework=framework,
        impl_type=impl_type,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name
    )
    parsed_code = ParsedCode()
    if impl_type == "swft":
        parsed_code.swft_code = kernel_code
    elif impl_type == "triton":
        parsed_code.triton_code = kernel_code
    else:
        raise ValueError(f"Invalid implementation type: {impl_type}")
    result, error_log = verifier.run(parsed_code)
    assert result, f"验证失败: {error_log}"

# ascend310p3


@pytest.mark.level0
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.parametrize("framework,impl_type,backend", [
    ("mindspore", "swft", "ascend"),
    ("torch", "swft", "ascend"),
    ("numpy", "swft", "ascend"),
])
def test_kernel_verifier_ascend310p3(op_name, framework, impl_type, backend):
    arch = "ascend310p3"
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{impl_type}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{impl_type}_test')
    impl_func_name = f"{op_name}_{impl_type}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        log_dir=log_dir,
        framework=framework,
        impl_type=impl_type,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name
    )
    parsed_code = ParsedCode()
    if impl_type == "swft":
        parsed_code.swft_code = kernel_code
    elif impl_type == "triton":
        parsed_code.triton_code = kernel_code
    else:
        raise ValueError(f"Invalid implementation type: {impl_type}")
    result, error_log = verifier.run(parsed_code)
    assert result, f"验证失败: {error_log}"

# a100


@pytest.mark.level0
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.parametrize("framework,impl_type,backend", [
    ("torch", "triton", "cuda"),
])
def test_kernel_verifier_a100(op_name, framework, impl_type, backend):
    arch = "a100"
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{impl_type}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{impl_type}_test')
    impl_func_name = f"{op_name}_{impl_type}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        log_dir=log_dir,
        framework=framework,
        impl_type=impl_type,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name
    )
    parsed_code = ParsedCode()
    if impl_type == "swft":
        parsed_code.swft_code = kernel_code
    elif impl_type == "triton":
        parsed_code.triton_code = kernel_code
    else:
        raise ValueError(f"Invalid implementation type: {impl_type}")
    result, error_log = verifier.run(parsed_code)
    assert result, f"验证失败: {error_log}"

# v100


@pytest.mark.level0
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.parametrize("framework,impl_type,backend", [
    ("torch", "triton", "cuda"),
])
def test_kernel_verifier_v100(op_name, framework, impl_type, backend):
    arch = "v100"
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{impl_type}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{impl_type}_test')
    impl_func_name = f"{op_name}_{impl_type}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        log_dir=log_dir,
        framework=framework,
        impl_type=impl_type,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name
    )
    parsed_code = ParsedCode()
    if impl_type == "swft":
        parsed_code.swft_code = kernel_code
    elif impl_type == "triton":
        parsed_code.triton_code = kernel_code
    else:
        raise ValueError(f"Invalid implementation type: {impl_type}")
    result, error_log = verifier.run(parsed_code)
    assert result, f"验证失败: {error_log}"


# profiling功能测试
@pytest.mark.level0
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.parametrize("framework,impl_type,backend", [
    ("mindspore", "triton", "ascend"),
    ("torch", "triton", "ascend"),
])
def test_kernel_verifier_profiling_ascend910b4(op_name, framework, impl_type, backend):
    arch = "ascend910b4"
    # 读取框架实现代码
    op_task_file = f"./tests/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取实现代码
    kernel_path = f"./tests/resources/{op_name}_op/{op_name}_{impl_type}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{impl_type}_profiling_test')
    impl_func_name = f"{op_name}_{impl_type}_{framework}"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        log_dir=log_dir,
        task_id="profiling_test_001",
        framework=framework,
        impl_type=impl_type,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name
    )
    parsed_code = ParsedCode()
    if impl_type == "swft":
        parsed_code.swft_code = kernel_code
    elif impl_type == "triton":
        parsed_code.triton_code = kernel_code
    else:
        raise ValueError(f"Invalid implementation type: {impl_type}")

    # 先进行验证，确保验证通过
    result, error_log = verifier.run(parsed_code)
    assert result, f"验证失败: {error_log}"

    # 进行性能分析
    profile_settings = {
        "run_times": 50,
        "warmup_times": 5
    }
    speedup = verifier.run_profile(current_step=0, device_id=0, profile_settings=profile_settings)

    print(f"Profiling测试通过，加速比: {speedup:.2f}x")
