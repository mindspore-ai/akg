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

"""MindSpore framework adapter."""

import os
from typing import Any, Optional
import mindspore as ms
from mindspore.common import np_dtype
import numpy as np

from .base import FrameworkAdapter


class FrameworkAdapterMindSpore(FrameworkAdapter):
    """Adapter for MindSpore framework."""
    
    def get_import_statements(self) -> str:
        """Return MindSpore import statements."""
        return "import mindspore as ms\nfrom mindspore.common import np_dtype\n"
    
    def get_framework_import(self, op_name: str, is_dynamic_shape: bool) -> str:
        """Return framework model and input function imports."""
        if is_dynamic_shape:
            return f"from {op_name}_mindspore import Model as FrameworkModel, get_init_inputs, get_inputs_dyn_list\n"
        else:
            return f"from {op_name}_mindspore import Model as FrameworkModel, get_init_inputs, get_inputs\n"
    
    def setup_device(self, backend: str, arch: str, device_id: int) -> Any:
        """Setup MindSpore device."""
        os.environ['DEVICE_ID'] = str(device_id)
        if backend == "ascend":
            device = "Ascend"
            supported_ascend_archs = ["ascend910b1", "ascend910b2", "ascend910b2c", 
                                      "ascend910b3", "ascend910b4", "ascend310p3"]
            if arch not in supported_ascend_archs:
                raise ValueError(f"不支持的ascend架构: {arch}，仅支持ascend910b1/b2/b2c/b3/b4和ascend310p3")
            return device
        elif backend == "cpu":
            return "CPU"
        else:
            raise ValueError(f"MindSpore不支持的后端: {backend}")
    
    def process_input(self, x: Any, device: Any) -> Any:
        """Process input (MindSpore doesn't need device movement)."""
        return x
    
    def convert_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert MindSpore tensor to numpy."""
        if isinstance(tensor, ms.Tensor):
            return tensor.flatten().asnumpy()
        return tensor.flatten() if hasattr(tensor, 'flatten') else tensor
    
    def get_limit(self, dtype: Any) -> float:
        """Get precision limit for dtype."""
        if dtype == ms.float16:
            return 0.004
        elif dtype == ms.bfloat16:
            return 0.03
        elif dtype == ms.int8:
            return 0.01
        else:
            return 0.004
    
    def save_tensor(self, tensor: Any, bin_path: str) -> None:
        """Save MindSpore tensor to binary file."""
        tensor_np = tensor.asnumpy()
        uint8_view = tensor_np.view(np.uint8)
        with open(bin_path, 'wb') as f:
            f.write(uint8_view.tobytes())
    
    def load_tensor(self, bin_path: str, reference_tensor: Any) -> Any:
        """Load MindSpore tensor from binary file."""
        with open(bin_path, 'rb') as f:
            data = f.read()
            uint8_array = np.frombuffer(data, dtype=np.uint8)
            numpy_dtype = self.get_dtype_mapping().get(reference_tensor.dtype)
            if numpy_dtype is None:
                raise ValueError(f"不支持的数据类型: {reference_tensor.dtype}")
            numpy_tensor = uint8_array.view(numpy_dtype).reshape(reference_tensor.shape)
            return ms.Tensor(numpy_tensor, dtype=reference_tensor.dtype)
    
    def set_seed(self, backend: Optional[str] = None) -> None:
        """Set random seed."""
        ms.set_seed(0)
    
    def move_model_to_device(self, model: Any, device: Any) -> Any:
        """Move model to device (MindSpore doesn't need explicit move)."""
        return model
    
    def get_tensor_type(self) -> type:
        """Get MindSpore tensor type."""
        return ms.Tensor
    
    def get_tensor_type_name(self) -> str:
        """Get MindSpore tensor type name as string (full path)."""
        return "ms.Tensor"
    
    def get_dtype_mapping(self) -> dict:
        """Get MindSpore to NumPy dtype mapping."""
        return {
            ms.float32: np.float32,
            ms.float16: np.float16,
            ms.bfloat16: np_dtype.bfloat16,
            ms.int8: np.int8,
            ms.int16: np.int16,
            ms.int32: np.int32,
            ms.int64: np.int64,
            ms.uint8: np.uint8,
            ms.uint16: np.uint16,
            ms.uint32: np.uint32,
            ms.uint64: np.uint64,
            ms.bool_: np.bool_,
        }
    
    def _get_save_tensor_code(self, tensor_type: str) -> str:
        """Get save_tensor function code for MindSpore."""
        return """def save_tensor(tensor: TensorType, bin_path: str):
    \"\"\"将MindSpore张量保存为二进制文件\"\"\"
    tensor_np = tensor.asnumpy()
    uint8_view = tensor_np.view(np.uint8)
    with open(bin_path, 'wb') as f:
        f.write(uint8_view.tobytes())

"""
    
    def _get_load_tensor_code(self, tensor_type: str) -> str:
        """Get load_tensor function code for MindSpore."""
        return """def load_tensor(bin_path: str, expect_tensor: TensorType) -> TensorType:
    \"\"\"从二进制文件加载MindSpore张量\"\"\"
    with open(bin_path, 'rb') as f:
        data = f.read()
        uint8_array = np.frombuffer(data, dtype=np.uint8)
        numpy_dtype = MS_TO_NP_DTYPE_MAP.get(expect_tensor.dtype)
        if numpy_dtype is None:
            raise ValueError(f"不支持的数据类型: {expect_tensor.dtype}")
        numpy_tensor = uint8_array.view(numpy_dtype).reshape(expect_tensor.shape)
        return ms.Tensor(numpy_tensor, dtype=expect_tensor.dtype)

"""
    
    def _get_gen_binary_data_code(self, tensor_type: str, op_name: str) -> str:
        """Get gen_binary_data function code."""
        return f"""def gen_binary_data(inputs, outputs, data_dir):
    \"\"\"生成二进制数据文件
    
    Args:
        inputs: 输入张量列表
        outputs: 输出张量列表或单个张量
        data_dir: 数据保存目录
    \"\"\"
    import os
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建输入输出目录
    input_dir = os.path.join(data_dir, "{op_name}", "input")
    output_dir = os.path.join(data_dir, "{op_name}", "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存输入数据
    for i, input_tensor in enumerate(inputs):
        if isinstance(input_tensor, TensorType):
            bin_path = os.path.join(input_dir, f"input{{i}}.bin")
            save_tensor(input_tensor, bin_path)
    
    # 处理输出数据
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]  # 将单个张量转换为列表
    
    # 保存golden输出
    for i, output_tensor in enumerate(outputs):
        if isinstance(output_tensor, TensorType):
            golden_path = os.path.join(output_dir, f"output{{i}}_golden.bin")
            save_tensor(output_tensor, golden_path)

"""
    
    def _get_load_binary_data_code(self, tensor_type: str, op_name: str) -> str:
        """Get load_binary_data function code."""
        return f"""def load_binary_data(data_dir, reference_outputs):
    \"\"\"加载二进制数据文件并转换为张量
    
    Args:
        data_dir: 数据目录
        reference_outputs: 参考输出张量列表或单个张量，用于确定数据类型和形状
    
    Returns:
        加载的张量列表
    \"\"\"
    import os
    if not isinstance(reference_outputs, (list, tuple)):
        reference_outputs = [reference_outputs]
    
    output_dir = os.path.join(data_dir, "{op_name}", "output")
    loaded_outputs = []
    i = 0
    while True:
        output_path = os.path.join(output_dir, f"output{{i}}_actual.bin")
        if not os.path.exists(output_path):
            break
        if i >= len(reference_outputs):
            raise RuntimeError(f"输出文件数量({{i+1}})超过参考输出数量({{len(reference_outputs)}})")
        loaded_outputs.append(load_tensor(output_path, reference_outputs[i]))
        i += 1
    
    if not loaded_outputs:
        raise RuntimeError("未找到任何输出文件, 一般是因为输入数据类型和原任务的输入数据类型不匹配")
    
    return loaded_outputs

"""
    
    def get_device_setup_code(self, backend: str, arch: str, device_id: int) -> str:
        """Get device setup code for MindSpore."""
        code = f"""    os.environ['DEVICE_ID'] = str({device_id})
"""
        if backend == "ascend":
            supported_ascend_archs = ["ascend910b1", "ascend910b2", "ascend910b2c", 
                                      "ascend910b3", "ascend910b4", "ascend310p3"]
            if arch not in supported_ascend_archs:
                raise ValueError(f"不支持的ascend架构: {arch}，仅支持ascend910b1/b2/b2c/b3/b4和ascend310p3")
            code += """    device = "Ascend"
"""
        elif backend == "cpu":
            code += """    device = "CPU"
"""
        return code
    
    def get_process_input_code(self, backend: str, dsl: str) -> str:
        """Get process_input function code for MindSpore."""
        return """    def process_input(x):
        \"\"\"处理输入数据\"\"\"
        return x
"""
    
    def get_set_seed_code(self, backend: str) -> str:
        """Get set seed code for MindSpore.
        
        Note: Returns code without indentation, template will handle indentation.
        """
        return """ms.set_seed(0)
"""
    
    def get_compare_code(self) -> str:
        """Get compare function code using pure MindSpore operations."""
        return '''def get_limit(data_type):
    """Get precision limit for data type"""
    if data_type == ms.float16:
        return 0.004
    elif data_type == ms.bfloat16:
        return 0.03
    elif data_type == ms.int8:
        return 0.01
    else:
        return 0.004

def compare(fw_out, impl_out, limit, data_type):
    """Compare framework output and implementation output using MindSpore"""
    import mindspore.ops as ops
    
    # Flatten tensors
    fw_flat = fw_out.flatten()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, ms.Tensor):
        pass
    else:
        impl_flat = ms.Tensor(impl_flat, dtype=fw_flat.dtype)
    
    size = fw_flat.size
    
    # 1. 检查形状一致性
    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}")
    
    # 转换为numpy进行NaN和Inf检查（MindSpore的isnan/isinf支持有限）
    fw_np = fw_flat.asnumpy()
    impl_np = impl_flat.asnumpy()
    
    # 2. 检查NaN值
    fw_nan_count = np.sum(np.isnan(fw_np))
    impl_nan_count = np.sum(np.isnan(impl_np))
    
    if fw_nan_count > 0 or impl_nan_count > 0:
        raise AssertionError(f"验证失败，检测到NaN值: Framework={fw_nan_count}/{size}, Implementation={impl_nan_count}/{size}")
    
    # 3. 检查Inf值 - 只有当两边Inf位置和符号都匹配时才允许
    fw_inf_mask = np.isinf(fw_np)
    impl_inf_mask = np.isinf(impl_np)
    
    # 检查Inf位置是否匹配
    if not np.array_equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = np.sum(fw_inf_mask)
        impl_inf_count = np.sum(impl_inf_mask)
        raise AssertionError(f"验证失败，Inf位置不匹配: Framework={fw_inf_count}/{size}, Implementation={impl_inf_count}/{size}")
    
    # 检查Inf符号是否匹配
    if np.sum(fw_inf_mask) > 0:
        inf_sign_match = np.array_equal(
            np.sign(fw_np[fw_inf_mask]), 
            np.sign(impl_np[impl_inf_mask])
        )
        if not inf_sign_match:
            raise AssertionError(f"验证失败，Inf符号不匹配")
    
    # 4. 对有限值进行精度比较
    finite_mask = np.isfinite(fw_np) & np.isfinite(impl_np)
    finite_count = np.sum(finite_mask)
    
    if finite_count == 0:
        print(f"警告: 所有值都是Inf，跳过精度检查")
        return
    
    # 提取有限值
    fw_finite = fw_np[finite_mask]
    impl_finite = impl_np[finite_mask]
    
    # 检查是否为布尔类型
    if fw_finite.dtype == bool or impl_finite.dtype == bool:
        if not np.array_equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return
    
    # 统一数据类型
    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.astype(fw_finite.dtype)
    
    # 计算相对误差
    abs_diff = np.abs(fw_finite - impl_finite)
    abs_ref = np.abs(fw_finite)
    eps = 1e-8
    relative_error = np.where(abs_ref > eps, abs_diff / abs_ref, abs_diff)
    
    # 统计错误
    err_cnt = np.sum(relative_error > limit).astype(np.int32)
    limit_cnt = int(finite_count * limit)
    
    if err_cnt > limit_cnt:
        max_error = np.max(relative_error)
        mean_error = np.mean(relative_error)
        
        # 找出不一致的位置
        mismatch_mask = relative_error > limit
        mismatch_indices = np.where(mismatch_mask)[0]
        # 最多打印10个不一致的位置
        num_to_show = min(10, len(mismatch_indices))

        error_msg = f"验证失败，输出不一致(误差数/最大容忍误差数): err_cnt={err_cnt} / {limit_cnt}, dtype={data_type}, limit={limit}\\n"
        error_msg += f"最大相对误差: {max_error:.6e}, 平均相对误差: {mean_error:.6e}\\n"
        error_msg += f"前 {num_to_show} 个不一致的值:\\n"
        for i in range(num_to_show):
            idx = mismatch_indices[i]
            error_msg += f"  位置[{idx}]: framework={fw_np[idx]:.6e}, "
            error_msg += f"impl={impl_np[idx]:.6e}, "
            error_msg += f"相对误差={relative_error[idx]:.6e}\\n"
        
        raise AssertionError(error_msg)

'''
    
    def get_compare_outputs_code(self) -> str:
        """Get code for comparing framework output and impl output."""
        return '''            data_type = framework_output[i].dtype
            limit = get_limit(data_type)
            compare(fw_out, impl_out, limit, data_type)
'''

