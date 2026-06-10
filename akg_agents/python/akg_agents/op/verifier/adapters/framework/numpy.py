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
    
    def get_framework_import(
        self,
        op_name: str,
        is_dynamic_shape: bool,
        inputs_factory_name: Optional[str] = None,
        module_name: Optional[str] = None,
    ) -> str:
        local = "get_inputs_dyn_list" if is_dynamic_shape else "get_inputs"
        factory = inputs_factory_name or local
        module = module_name or f"{op_name}_numpy"
        return (f"from {module} import Model as FrameworkModel, "
                f"get_init_inputs, {factory} as {local}\n")
    
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
        """Get precision rtol for dtype (backward compatibility)."""
        if dtype == np.float32:
            return 1.22e-4
        elif dtype == np.float16:
            return 9.77e-4
        else:
            return 1.22e-4
    
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
        """Get compare function code using layered tolerance (hard-coded, no config)."""
        return '''def _get_tolerance(data_type):
    """Hard-coded tolerance table aligned with CANN / NPUKernelBench.

    Returns (rtol, atol, outlier_rtol, outlier_atol, outlier_ratio).
    - strict_tol  = atol       + rtol       * |ref|
    - relaxed_tol = outlier_atol + outlier_rtol * |ref|
    - outlier_atol = 10 * atol  (aligned with PyTorch MatMul FP32 atol=1e-4)
    - outlier_rtol = 10 * rtol  (aligned with CANN MARE threshold = 10 * MERE threshold)
    """
    if data_type == np.float32:
        return (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)
    elif data_type == np.float16:
        return (9.77e-4, 1e-3, 9.77e-3, 1e-2, 0.005)
    else:
        return (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)

def compare(fw_out, impl_out, data_type):
    """Compare framework output and implementation output using layered tolerance."""
    fw_flat = fw_out.flatten()
    impl_flat = impl_out.flatten()

    size = len(fw_flat)

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}")

    fw_nan_mask = np.isnan(fw_flat)
    impl_nan_mask = np.isnan(impl_flat)
    if not np.array_equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = np.sum(fw_nan_mask)
        impl_nan_count = np.sum(impl_nan_mask)
        raise AssertionError(f"验证失败，NaN位置不匹配: Framework={fw_nan_count}/{size}, Implementation={impl_nan_count}/{size}")
    if np.sum(fw_nan_mask) > 0:
        nan_count = np.sum(fw_nan_mask)
        print(f"检测到NaN值: {nan_count}/{size} (位置一致，继续验证)")

    fw_inf_mask = np.isinf(fw_flat)
    impl_inf_mask = np.isinf(impl_flat)
    if not np.array_equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = np.sum(fw_inf_mask)
        impl_inf_count = np.sum(impl_inf_mask)
        raise AssertionError(f"验证失败，Inf位置不匹配: Framework={fw_inf_count}/{size}, Implementation={impl_inf_count}/{size}")
    if np.sum(fw_inf_mask) > 0:
        inf_sign_match = np.array_equal(
            np.sign(fw_flat[fw_inf_mask]),
            np.sign(impl_flat[impl_inf_mask])
        )
        if not inf_sign_match:
            raise AssertionError(f"验证失败，Inf符号不匹配")

    finite_mask = np.isfinite(fw_flat) & np.isfinite(impl_flat)
    finite_count = np.sum(finite_mask)
    if finite_count == 0:
        print(f"警告: 所有值都是Inf，跳过精度检查")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == bool or impl_finite.dtype == bool:
        if not np.array_equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.astype(fw_finite.dtype)

    rtol, atol, outlier_rtol, outlier_atol, outlier_ratio = _get_tolerance(data_type)

    abs_diff = np.abs(fw_finite - impl_finite)
    abs_ref = np.abs(fw_finite)
    strict_tol = atol + rtol * abs_ref
    relaxed_tol = outlier_atol + outlier_rtol * abs_ref

    strict_pass = abs_diff <= strict_tol
    relaxed_pass = abs_diff <= relaxed_tol

    hard_fail = int(np.sum(~relaxed_pass))
    outlier = int(np.sum((~strict_pass) & relaxed_pass))
    total = fw_finite.size
    cap = int(total * outlier_ratio)

    mere = float(np.mean(abs_diff / (abs_ref + atol)))
    mare = float(np.max(abs_diff / (abs_ref + atol)))
    print(f"[precision] dtype={data_type} total={total} strict={int(np.sum(strict_pass))} outlier={outlier}/{cap} hard={hard_fail} mere={mere:.6e} mare={mare:.6e}")

    if hard_fail > 0:
        hf_mask = ~relaxed_pass
        hf_indices = np.where(hf_mask)[0]
        num_to_show = min(5, len(hf_indices))
        error_msg = f"验证失败，存在 {hard_fail} 个元素超过放宽阈值(hard_fail)\\n"
        error_msg += f"rtol={rtol:.6e} atol={atol:.6e} outlier_rtol={outlier_rtol:.6e} outlier_atol={outlier_atol:.6e} outlier_ratio={outlier_ratio}\\n"
        error_msg += f"mere={mere:.6e} mare={mare:.6e}\\n"
        for i in range(num_to_show):
            idx = hf_indices[i]
            error_msg += f"  位置[{idx}]: ref={fw_finite[idx]:.6e} impl={impl_finite[idx]:.6e} abs_diff={abs_diff[idx]:.6e} relaxed_tol={relaxed_tol[idx]:.6e}\\n"
        raise AssertionError(error_msg)

    if outlier > cap:
        ol_mask = (~strict_pass) & relaxed_pass
        ol_indices = np.where(ol_mask)[0]
        num_to_show = min(5, len(ol_indices))
        error_msg = f"验证失败，超限元素比例超过允许值: outlier={outlier} / cap={cap}\\n"
        error_msg += f"rtol={rtol:.6e} atol={atol:.6e} outlier_rtol={outlier_rtol:.6e} outlier_ratio={outlier_ratio}\\n"
        error_msg += f"mere={mere:.6e} mare={mare:.6e}\\n"
        for i in range(num_to_show):
            idx = ol_indices[i]
            error_msg += f"  位置[{idx}]: ref={fw_finite[idx]:.6e} impl={impl_finite[idx]:.6e} abs_diff={abs_diff[idx]:.6e} strict_tol={strict_tol[idx]:.6e} relaxed_tol={relaxed_tol[idx]:.6e}\\n"
        raise AssertionError(error_msg)

'''

    def get_compare_outputs_code(self) -> str:
        """Get code for comparing framework output and impl output."""
        return '''            data_type = framework_output[i].dtype
            compare(fw_out, impl_out, data_type)
'''
