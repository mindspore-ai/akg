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

"""NumPy framework adapter."""

from typing import Any, Optional
import numpy as np

from .base import FrameworkAdapter


class FrameworkAdapterNumpy(FrameworkAdapter):
    """Adapter for NumPy framework."""
    
    def get_import_statements(self) -> str:
        """Return NumPy import statements."""
        return "import numpy as np\n"
    
    def get_framework_import(self, op_name: str, is_dynamic_shape: bool) -> str:
        """Return framework model and input function imports."""
        if is_dynamic_shape:
            return f"from {op_name}_numpy import Model as FrameworkModel, get_init_inputs, get_inputs_dyn_list\n"
        else:
            return f"from {op_name}_numpy import Model as FrameworkModel, get_init_inputs, get_inputs\n"
    
    def setup_device(self, backend: str, arch: str, device_id: int) -> Any:
        """Setup device (NumPy doesn't need device)."""
        return None
    
    def process_input(self, x: Any, device: Any) -> Any:
        """Process input (NumPy doesn't need device movement)."""
        return x
    
    def convert_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert to numpy (already numpy)."""
        return tensor.flatten() if hasattr(tensor, 'flatten') else tensor
    
    def get_limit(self, dtype: Any) -> float:
        """Get precision limit for dtype."""
        if dtype == np.float16:
            return 0.004
        elif dtype == np.int8:
            return 0.01
        else:
            return 0.004
    
    def save_tensor(self, tensor: Any, bin_path: str) -> None:
        """Save NumPy array to binary file."""
        uint8_view = tensor.view(np.uint8)
        uint8_view.tofile(bin_path)
    
    def load_tensor(self, bin_path: str, reference_tensor: Any) -> Any:
        """Load NumPy array from binary file."""
        uint8_array = np.fromfile(bin_path, dtype=np.uint8)
        arr = uint8_array.view(reference_tensor.dtype).reshape(reference_tensor.shape)
        return arr.astype(reference_tensor.dtype)
    
    def set_seed(self, backend: Optional[str] = None) -> None:
        """Set random seed."""
        np.random.seed(0)
    
    def move_model_to_device(self, model: Any, device: Any) -> Any:
        """Move model to device (NumPy doesn't need device)."""
        return model
    
    def get_tensor_type(self) -> type:
        """Get NumPy array type."""
        return np.ndarray
    
    def get_tensor_type_name(self) -> str:
        """Get NumPy array type name as string (full path)."""
        return "np.ndarray"
    
    def _get_save_tensor_code(self, tensor_type: str) -> str:
        """Get save_tensor function code for NumPy."""
        return """def save_tensor(tensor: TensorType, bin_path: str):
    \"\"\"将numpy数组保存为二进制文件\"\"\"
    uint8_view = tensor.view(np.uint8)
    uint8_view.tofile(bin_path)

"""
    
    def _get_load_tensor_code(self, tensor_type: str) -> str:
        """Get load_tensor function code for NumPy."""
        return """def load_tensor(bin_path: str, expect_tensor: TensorType) -> TensorType:
    \"\"\"从二进制文件加载numpy数组\"\"\"
    uint8_array = np.fromfile(bin_path, dtype=np.uint8)
    arr = uint8_array.view(expect_tensor.dtype).reshape(expect_tensor.shape)
    return arr.astype(expect_tensor.dtype)

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
        """Get device setup code for NumPy (no-op)."""
        return ""
    
    def get_process_input_code(self, backend: str, dsl: str) -> str:
        """Get process_input function code for NumPy."""
        return """    def process_input(x):
        \"\"\"处理输入数据\"\"\"
        return x
"""
    
    def get_set_seed_code(self, backend: str) -> str:
        """Get set seed code for NumPy.
        
        Note: Returns code without indentation, template will handle indentation.
        """
        return """np.random.seed(0)
"""
    
    def get_compare_code(self) -> str:
        """Get compare function code using pure NumPy operations."""
        return '''def get_limit(data_type):
    """Get precision limit for data type"""
    if data_type == np.float16:
        return 0.004
    elif data_type == np.int8:
        return 0.01
    else:
        return 0.004

def compare(fw_out, impl_out, limit, data_type):
    """Compare framework output and implementation output using pure NumPy"""
    # Flatten arrays
    fw_flat = fw_out.flatten()
    impl_flat = impl_out.flatten()
    
    size = len(fw_flat)
    
    # 1. 检查形状一致性
    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}")
    
    # 2. 检查NaN值
    fw_nan_count = np.sum(np.isnan(fw_flat))
    impl_nan_count = np.sum(np.isnan(impl_flat))
    
    if fw_nan_count > 0 or impl_nan_count > 0:
        raise AssertionError(f"验证失败，检测到NaN值: Framework={fw_nan_count}/{size}, Implementation={impl_nan_count}/{size}")
    
    # 3. 检查Inf值 - 只有当两边Inf位置和符号都匹配时才允许
    fw_inf_mask = np.isinf(fw_flat)
    impl_inf_mask = np.isinf(impl_flat)
    
    # 检查Inf位置是否匹配
    if not np.array_equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = np.sum(fw_inf_mask)
        impl_inf_count = np.sum(impl_inf_mask)
        raise AssertionError(f"验证失败，Inf位置不匹配: Framework={fw_inf_count}/{size}, Implementation={impl_inf_count}/{size}")
    
    # 检查Inf符号是否匹配
    if np.sum(fw_inf_mask) > 0:
        inf_sign_match = np.array_equal(
            np.sign(fw_flat[fw_inf_mask]), 
            np.sign(impl_flat[impl_inf_mask])
        )
        if not inf_sign_match:
            raise AssertionError(f"验证失败，Inf符号不匹配")
    
    # 4. 对有限值进行精度比较
    finite_mask = np.isfinite(fw_flat) & np.isfinite(impl_flat)
    finite_count = np.sum(finite_mask)
    
    if finite_count == 0:
        print(f"警告: 所有值都是Inf，跳过精度检查")
        return
    
    # 提取有限值
    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]
    
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

        error_msg = f"验证失败，输出不一致: err_cnt={err_cnt} / {limit_cnt}, dtype={data_type}, limit={limit}\\n"
        error_msg += f"最大相对误差: {max_error:.6e}, 平均相对误差: {mean_error:.6e}\\n"
        error_msg += f"前 {num_to_show} 个不一致的值:\\n"
        for i in range(num_to_show):
            idx = mismatch_indices[i]
            error_msg += f"  位置[{idx}]: framework={fw_flat[idx]:.6e}, "
            error_msg += f"impl={impl_flat[idx]:.6e}, "
            error_msg += f"相对误差={relative_error[idx]:.6e}\\n"
        
        raise AssertionError(error_msg)

'''
    
    def get_compare_outputs_code(self) -> str:
        """Get code for comparing framework output and impl output."""
        return '''            data_type = framework_output[i].dtype
            limit = get_limit(data_type)
            compare(fw_out, impl_out, limit, data_type)
'''

