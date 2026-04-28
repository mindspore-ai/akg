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

import os
import textwrap

import torch
import pytest

from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker, get_worker_manager
from ..utils import get_device_id

device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.asyncio
async def test_two_step_reference_verify_cpu():
    """CPU 两步验证：先生成参考数据（含 inputs），再用参考数据验证另一个实现"""
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "x86_64"
    op_name = "relu"

    ref_cache_dir = os.path.expanduser("~/.akg/.tmp/reference_data")
    os.makedirs(ref_cache_dir, exist_ok=True)

    # ========== Step 1: 生成参考数据（含 inputs） ==========
    config_step1 = load_config(
        config_path="./python/akg_agents/op/config/cpp_coderonly_config.yaml"
    )

    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)

    op_task_file = f"./tests/op/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        framework_code = textwrap.dedent(f.read())

    verifier_step1 = KernelVerifier(
        op_name=op_name,
        framework_code=framework_code,
        task_id="ref_gen_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config_step1,
        worker=worker,
    )

    success, log, ref_bytes = await verifier_step1.generate_reference_data(
        framework_code, save_inputs=True, timeout=60
    )
    assert success, f"Step 1 参考数据生成失败: {log}"
    assert len(ref_bytes) > 0

    # 持久化到 ~/.akg/.tmp/reference_data/relu_reference.pt
    ref_pt_path = os.path.join(ref_cache_dir, f"{op_name}_reference.pt")
    with open(ref_pt_path, "wb") as f:
        f.write(ref_bytes)

    # 验证 .pt 内容完整性
    ref_data = torch.load(ref_pt_path, map_location="cpu", weights_only=False)
    assert ref_data.get("save_inputs") is True
    assert "inputs" in ref_data
    assert "outputs" in ref_data
    assert "init_inputs" in ref_data

    # ========== Step 2: 从磁盘加载参考数据 -> 验证 relu_cpp ==========
    with open(ref_pt_path, "rb") as f:
        loaded_ref_bytes = f.read()

    config_step2 = load_config(
        config_path="./python/akg_agents/op/config/cpp_coderonly_config.yaml"
    )
    config_step2["use_reference_data"] = True
    config_step2["use_reference_inputs"] = True
    config_step2["reference_data"] = loaded_ref_bytes

    kernel_path = f"./tests/op/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    verifier_step2 = KernelVerifier(
        op_name=op_name,
        framework_code=framework_code,
        task_id="ref_verify_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config_step2,
        worker=worker,
    )

    task_info = {"coder_code": kernel_code}
    result, error_log = await verifier_step2.run(task_info, device_id=device_id)
    assert result, f"Step 2 参考数据验证失败: {error_log}"


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.asyncio
async def test_two_step_reference_verify_multi_io_cpu():
    """多输入多输出 + in-place 两步验证：验证 inputs clone 和多输出对比的正确性"""
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "x86_64"
    op_name = "fused_multi_io"

    ref_cache_dir = os.path.expanduser("~/.akg/.tmp/reference_data")
    os.makedirs(ref_cache_dir, exist_ok=True)

    # ========== Step 1: 生成参考数据（含 inputs） ==========
    config_step1 = load_config(
        config_path="./python/akg_agents/op/config/cpp_coderonly_config.yaml"
    )

    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)

    op_task_file = f"./tests/op/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        framework_code = textwrap.dedent(f.read())

    verifier_step1 = KernelVerifier(
        op_name=op_name,
        framework_code=framework_code,
        task_id="ref_gen_multi_io_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config_step1,
        worker=worker,
    )

    success, log, ref_bytes = await verifier_step1.generate_reference_data(
        framework_code, save_inputs=True, timeout=60
    )
    assert success, f"Step 1 参考数据生成失败: {log}"
    assert len(ref_bytes) > 0

    # 持久化
    ref_pt_path = os.path.join(ref_cache_dir, f"{op_name}_reference.pt")
    with open(ref_pt_path, "wb") as f:
        f.write(ref_bytes)

    # 验证 .pt 内容完整性
    ref_data = torch.load(ref_pt_path, map_location="cpu", weights_only=False)
    assert ref_data.get("save_inputs") is True
    assert "inputs" in ref_data
    assert "outputs" in ref_data
    assert "init_inputs" in ref_data

    # 多输入：x, y, bias => 3 个 inputs
    assert len(ref_data["inputs"]) == 3, f"期望 3 个 inputs，实际 {len(ref_data['inputs'])}"
    # 多输出：out_norm, out_gated => 2 个 outputs
    assert len(ref_data["outputs"]) == 2, f"期望 2 个 outputs，实际 {len(ref_data['outputs'])}"
    # init_inputs: [hidden_size]
    assert len(ref_data["init_inputs"]) == 1

    # 关键：验证保存的 inputs 是 forward 前的原始值（未被 in-place 污染）
    # Model.forward 中有 x.add_(bias)，如果没有 clone，saved inputs[0] 会变成 x+bias
    # 验证方法：用保存的 inputs 重新跑一次 Model，输出应该和 ref outputs 一致
    saved_x, saved_y, saved_bias = ref_data["inputs"]
    torch.manual_seed(0)
    import types
    spec_code = framework_code
    mod = types.ModuleType("_ref_model_check")
    exec(compile(spec_code, "<string>", "exec"), mod.__dict__)
    check_model = mod.Model(*ref_data["init_inputs"])
    check_model.eval()
    with torch.no_grad():
        check_out = check_model(saved_x.clone(), saved_y.clone(), saved_bias.clone())
    for i, (ref_o, chk_o) in enumerate(zip(ref_data["outputs"], check_out)):
        assert torch.allclose(ref_o, chk_o, atol=1e-6), \
            f"Output[{i}] inputs 被 in-place 污染：用保存的 inputs 重跑结果不一致"

    # ========== Step 2: 从磁盘加载参考数据 -> 验证 ModelNew ==========
    with open(ref_pt_path, "rb") as f:
        loaded_ref_bytes = f.read()

    config_step2 = load_config(
        config_path="./python/akg_agents/op/config/cpp_coderonly_config.yaml"
    )
    config_step2["use_reference_data"] = True
    config_step2["use_reference_inputs"] = True
    config_step2["reference_data"] = loaded_ref_bytes

    kernel_path = f"./tests/op/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    verifier_step2 = KernelVerifier(
        op_name=op_name,
        framework_code=framework_code,
        task_id="ref_verify_multi_io_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config_step2,
        worker=worker,
    )

    task_info = {"coder_code": kernel_code}
    result, error_log = await verifier_step2.run(task_info, device_id=device_id)
    assert result, f"Step 2 多输入多输出参考数据验证失败: {error_log}"


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_two_step_reference_verify_with_llm_cpu():
    """LLM 端到端：生成参考数据 -> LLM 生成代码 -> 参考数据验证"""
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "x86_64"
    op_name = "relu"

    ref_cache_dir = os.path.expanduser("~/.akg/.tmp/reference_data")
    os.makedirs(ref_cache_dir, exist_ok=True)

    # ========== Step 1: 生成参考数据 ==========
    config_gen = load_config(
        config_path="./python/akg_agents/op/config/cpp_coderonly_config.yaml"
    )

    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)

    op_task_file = f"./tests/op/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        framework_code = textwrap.dedent(f.read())

    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=framework_code,
        task_id="ref_gen_llm_001",
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config=config_gen,
        worker=worker,
    )

    success, log, ref_bytes = await verifier.generate_reference_data(
        framework_code, save_inputs=True, timeout=60
    )
    assert success, f"参考数据生成失败: {log}"

    ref_pt_path = os.path.join(ref_cache_dir, f"{op_name}_llm_reference.pt")
    with open(ref_pt_path, "wb") as f:
        f.write(ref_bytes)

    # ========== Step 2: LLM 生成 + 参考数据验证 ==========
    with open(ref_pt_path, "rb") as f:
        loaded_ref_bytes = f.read()

    config_task = load_config(
        config_path="./python/akg_agents/op/config/cpp_coderonly_config.yaml"
    )
    config_task["use_reference_data"] = True
    config_task["use_reference_inputs"] = True
    config_task["reference_data"] = loaded_ref_bytes
    config_task["max_step"] = 10

    from akg_agents.op.langgraph_op.task import LangGraphTask

    task = LangGraphTask(
        op_name=op_name,
        task_desc=framework_code,
        task_id="ref_llm_001",
        backend=backend,
        arch=arch,
        dsl=dsl,
        config=config_task,
        framework=framework,
        workflow="coder_only_workflow",
    )

    result_op_name, task_success, task_info = await task.run()
    assert task_success, f"LLM 端到端验证失败: {task_info}"
