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

import pytest
import torch
import importlib.util
import sys
from pathlib import Path
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.utils.common_utils import create_log_dir
from ai_kernel_generator.config.config_validator import load_config
from ..utils import get_device_id

device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.asyncio
@pytest.mark.parametrize("op_name", [
    "assign_extend_cache_locs",
    "assign_req_to_token_pool",
    "compute_position",
    "fused_qkvzba_split_reshape_cat",
    "get_mla_kv_buffer",
    "merge_state_triton",
    "moe_align_block_size_triton",
    "moe_sum_reduce_triton",
    "set_mla_kv_buffer",
    "set_mla_kv_scale_buffer",
    "write_req_to_token_pool",
    "triton_tanh",
    "get_last_loc",
    "merge_state_kernel",
    "prefill_attention_fwd_kernel",
    "extend_attention_fwd_kernel",
    "decode_attention_fwd_kernel_stage1",
    "decode_attention_fwd_kernel_stage2",
    "decode_grouped_attention_fwd_kernel_stage1",
    "_fwd_grouped_kernel_stage1_rope",
    "add_tree_reduce_u64",
    "chunked_sgmv_lora_expand",
    "chunked_sgmv_lora_shrink",
    "fmix32",
    "hash_tiles32_kernel_blocked",
    "rotl32",
    "sgemm_lora_a",
    "sgemm_lora_b",
    "qkv_lora_b",
    "gate_up_lora_b",
])
async def test_sglang_verifier_a100(op_name):
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    config = load_config(dsl, backend=backend)

    op_task_file = f"./benchmark/aikgbench/sglang/{op_name}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        framework_code = f.read()

    kernel_code = framework_code.replace("ModelSGLang", "ModelNew")

    log_dir = create_log_dir(f'{op_name}_sglang_{framework}_{backend}_{arch}_{dsl}_test')

    impl_func_name = "ModelNew"

    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=framework_code,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config
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
@pytest.mark.parametrize("op_name", [
    "align_evict_mask_to_page_size",
    "alloc_decode",
    "alloc_extend",
    "assign_draft_cache_locs_page_size_1",
    "assign_draft_cache_locs",
    "copy_all_layer_kv_cache_tiled",
    "create_chunked_prefix_cache_kv_indices",
    "create_extend_after_decode_spec_info",
    "fill_accepted_out_cache_loc",
    "fill_new_verified_id",
    "filter_finished_cache_loc_kernel",
    "generate_draft_decode_kv_indices",
    "get_target_cache_loc",
])
def test_sglang_class_method_no_reference_a100(op_name):
    """
    无标杆验证：针对 class_method kernels，只检查输出是否包含 nan 或 inf
    """
    # 动态导入 kernel 模块
    op_task_file = f"./benchmark/aikgbench/sglang/class_method/{op_name}.py"
    spec = importlib.util.spec_from_file_location(f"sglang_class_method_{op_name}", op_task_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"sglang_class_method_{op_name}"] = module
    spec.loader.exec_module(module)

    # 获取必要的函数和类
    Model = module.Model
    get_inputs = module.get_inputs
    get_init_inputs = module.get_init_inputs

    # 设置设备
    device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cuda:0")

    # 获取初始化参数和输入
    init_params = get_init_inputs()
    inputs = get_inputs()

    # 将输入移到 GPU
    inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]

    # 创建模型并移到 GPU
    model = Model(*init_params)
    if hasattr(model, 'to'):
        model = model.to(device)

    # 运行 forward
    output = model(*inputs)

    # 确保输出是列表形式
    if not isinstance(output, (list, tuple)):
        output = [output]

    # 检查每个输出是否包含 nan 或 inf
    for i, out in enumerate(output):
        if isinstance(out, torch.Tensor):
            # 检查 NaN
            nan_count = torch.isnan(out).sum().item()
            if nan_count > 0:
                raise AssertionError(
                    f"{op_name}: 输出 {i} 包含 {nan_count} 个 NaN 值 "
                    f"(shape: {out.shape}, dtype: {out.dtype})"
                )

            # 检查 Inf
            inf_count = torch.isinf(out).sum().item()
            if inf_count > 0:
                raise AssertionError(
                    f"{op_name}: 输出 {i} 包含 {inf_count} 个 Inf 值 "
                    f"(shape: {out.shape}, dtype: {out.dtype})"
                )

    print(f"{op_name}: 验证通过，输出无 NaN/Inf")


