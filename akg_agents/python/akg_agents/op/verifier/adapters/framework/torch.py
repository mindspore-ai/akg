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

"""PyTorch framework adapter."""

import os
from typing import Any, Optional
import torch
import numpy as np

from .base import FrameworkAdapter


class FrameworkAdapterTorch(FrameworkAdapter):
    """Adapter for PyTorch framework."""
    
    def get_import_statements(self) -> str:
        """Return PyTorch import statements."""
        return "import torch\n"
    
    def get_framework_import(
        self,
        op_name: str,
        is_dynamic_shape: bool,
        inputs_factory_name: Optional[str] = None,
    ) -> str:
        akg_inputs_name = "get_inputs_dyn_list" if is_dynamic_shape else "get_inputs"

        if inputs_factory_name is None or inputs_factory_name == akg_inputs_name:
            if is_dynamic_shape:
                return f"from {op_name}_torch import Model as FrameworkModel, get_init_inputs, get_inputs_dyn_list\n"
            return f"from {op_name}_torch import Model as FrameworkModel, get_init_inputs, get_inputs\n"

        return (
            f"from {op_name}_torch import "
            f"Model as FrameworkModel, "
            f"get_init_inputs, "
            f"{inputs_factory_name} as {akg_inputs_name}\n"
        )
    
    def setup_device(self, backend: str, arch: str, device_id: int) -> Any:
        """Setup PyTorch device."""
        if backend == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
            device = torch.device("cuda")
            torch.cuda.set_device(0)
            return device
        elif backend == "ascend":
            if "ascend910" in arch:
                os.environ['DEVICE_ID'] = str(device_id)
                device = torch.device("npu")
                torch.npu.set_device(device_id)
                return device
            elif "ascend310" in arch:
                os.environ['DEVICE_ID'] = str(device_id)
                return torch.device("cpu")
            else:
                raise ValueError(f"不支持的ascend架构: {arch}")
        elif backend == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"不支持的后端: {backend}")
    
    def process_input(self, x: Any, device: Any) -> Any:
        """Process input and move to device."""
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        elif isinstance(x, (list, tuple)):
            return type(x)(self.process_input(item, device) for item in x)
        elif isinstance(x, (int, float, bool, type(None))):
            return x
        else:
            try:
                return x.to(device)
            except (AttributeError, TypeError):
                return x
    
    def convert_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert PyTorch tensor to numpy."""
        if isinstance(tensor, torch.Tensor):
            return tensor.flatten().detach().cpu().numpy()
        return tensor.flatten() if hasattr(tensor, 'flatten') else tensor
    
    def get_limit(self, dtype: Any) -> float:
        """Get precision rtol for dtype (backward compatibility)."""
        if dtype == torch.float32:
            return 1.22e-4
        elif dtype == torch.float16:
            return 9.77e-4
        elif dtype == torch.bfloat16:
            return 7.81e-3
        else:
            return 1.22e-4
    
    def save_tensor(self, tensor: Any, bin_path: str) -> None:
        """Save PyTorch tensor to binary file."""
        tensor_contiguous = tensor.contiguous().cpu()
        uint8_view = tensor_contiguous.view(torch.uint8)
        with open(bin_path, 'wb') as f:
            f.write(uint8_view.numpy().tobytes())
    
    def load_tensor(self, bin_path: str, reference_tensor: Any) -> Any:
        """Load PyTorch tensor from binary file."""
        with open(bin_path, 'rb') as f:
            data = f.read()
            uint8_tensor = torch.frombuffer(data, dtype=torch.uint8)
            return uint8_tensor.view(reference_tensor.dtype).reshape(reference_tensor.shape)
    
    def set_seed(self, backend: Optional[str] = None) -> None:
        """Set random seed."""
        torch.manual_seed(0)
        if backend == "ascend":
            torch.npu.manual_seed(0)
    
    def move_model_to_device(self, model: Any, device: Any) -> Any:
        """Move model to device."""
        return model.to(device)
    
    def get_tensor_type(self) -> type:
        """Get PyTorch tensor type."""
        return torch.Tensor
    
    def get_tensor_type_name(self) -> str:
        """Get PyTorch tensor type name as string (full path)."""
        return "torch.Tensor"
    
    def _get_save_tensor_code(self, tensor_type: str) -> str:
        """Get save_tensor function code for PyTorch."""
        return """def save_tensor(tensor: TensorType, bin_path: str):
    \"\"\"将PyTorch张量保存为二进制文件\"\"\"
    tensor_contiguous = tensor.contiguous().cpu()
    uint8_view = tensor_contiguous.view(torch.uint8)
    with open(bin_path, 'wb') as f:
        f.write(uint8_view.numpy().tobytes())

"""
    
    def _get_load_tensor_code(self, tensor_type: str) -> str:
        """Get load_tensor function code for PyTorch."""
        return """def load_tensor(bin_path: str, expect_tensor: TensorType) -> TensorType:
    \"\"\"从二进制文件加载PyTorch张量\"\"\"
    with open(bin_path, 'rb') as f:
        data = f.read()
        uint8_tensor = torch.frombuffer(data, dtype=torch.uint8)
        return uint8_tensor.view(expect_tensor.dtype).reshape(expect_tensor.shape)

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
        """Get device setup code for PyTorch."""
        if backend == "cuda":
            return f"""    os.environ['CUDA_VISIBLE_DEVICES'] = str({device_id})
    device = torch.device("cuda")
    torch.cuda.set_device(0)  # 使用第一个CUDA设备
"""
        elif backend == "ascend":
            if "ascend910" in arch or "ascend950" in arch:
                return f"""    import torch_npu
    os.environ['DEVICE_ID'] = str({device_id})
    device = torch.device("npu")
    torch.npu.set_device({device_id})
"""
            elif "ascend310" in arch:
                return f"""    os.environ['DEVICE_ID'] = str({device_id})
    device = torch.device("cpu")
"""
        elif backend == "cpu":
            return """    device = torch.device("cpu")
"""
        return ""
    
    def get_process_input_code(self, backend: str, dsl: str) -> str:
        """Get process_input function code for PyTorch."""
        if dsl == "ascendc":
            return """    def process_input(x):
        \"\"\"处理输入数据，将数据移动到正确的设备\"\"\"
        if isinstance(x, torch.Tensor):
            return x.npu()
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).npu()
        elif isinstance(x, (list, tuple)):
            return type(x)(process_input(item) for item in x)
        elif isinstance(x, (int, float, bool, type(None))):
            return x
        else:
            try:
                return x.npu()
            except (AttributeError, TypeError):
                return x
"""
        else:
            return """    def process_input(x):
        \"\"\"处理输入数据，将数据移动到正确的设备\"\"\"
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        elif isinstance(x, (list, tuple)):
            return type(x)(process_input(item) for item in x)
        elif isinstance(x, (int, float, bool, type(None))):
            return x
        else:
            try:
                return x.to(device)
            except (AttributeError, TypeError):
                return x
"""
    
    def get_set_seed_code(self, backend: str) -> str:
        """Get set seed code for PyTorch.
        
        Note: Returns code without indentation, template will handle indentation.
        """
        if backend == "ascend":
            return """torch.manual_seed(0)
torch.npu.manual_seed(0)
"""
        else:
            return """torch.manual_seed(0)
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
    if data_type == torch.float32:
        return (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)
    elif data_type == torch.float16:
        return (9.77e-4, 1e-3, 9.77e-3, 1e-2, 0.005)
    elif data_type == torch.bfloat16:
        return (7.81e-3, 1e-2, 7.81e-2, 1e-1, 0.01)
    else:
        return (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)

def _merge_consecutive(values):
    """Merge sorted integer list into consecutive ranges.

    [0,1,2,5,6,7,10] -> [(0,3), (5,8), (10,11)]
    """
    if not values:
        return []
    ranges = []
    start = values[0]
    end = values[0]
    for v in values[1:]:
        if v == end + 1:
            end = v
        else:
            ranges.append((start, end + 1))
            start = v
            end = v
    ranges.append((start, end + 1))
    return ranges

def _format_dim(values, dim_size):
    """Format per-dimension error distribution with auto-merge."""
    ranges = _merge_consecutive(values)
    total_error = len(values)
    coverage = total_error / dim_size * 100

    if len(ranges) == 1:
        lo, hi = ranges[0]
        if hi - lo == dim_size:
            return f"[:]" + f"  ({total_error}/{dim_size} = {coverage:.1f}%)"
        return f"[{lo}:{hi}]" + f"  ({total_error}/{dim_size} = {coverage:.1f}%)"

    if len(ranges) <= 5:
        parts = [f"[{lo}:{hi}]" for lo, hi in ranges]
        return ", ".join(parts) + f"  ({total_error}/{dim_size} = {coverage:.1f}%)"

    first3 = [f"[{lo}:{hi}]" for lo, hi in ranges[:3]]
    return ", ".join(first3) + f", ... ({len(ranges)} ranges, {total_error}/{dim_size} = {coverage:.1f}%)"

def _format_error_locations(error_mask, shape):
    """Format per-dimension error distribution without materializing all coords."""
    if len(shape) == 0:
        return "Error location: scalar output"

    lines = ["Error location per dimension ([start:end]=error index range, count/size=coverage):"]
    non_singleton_dims = []
    full_coverage_dims = []
    singleton_dims = []

    for d, dim_size in enumerate(shape):
        if dim_size == 1:
            singleton_dims.append(d)
            continue

        reduce_dims = tuple(i for i in range(len(shape)) if i != d)
        dim_mask = error_mask.any(dim=reduce_dims) if reduce_dims else error_mask
        unique_vals = torch.where(dim_mask)[0].tolist()
        non_singleton_dims.append(d)
        if len(unique_vals) == dim_size:
            full_coverage_dims.append(d)
        lines.append(f"  dim{d}: {_format_dim(unique_vals, dim_size)}")

    if not non_singleton_dims:
        lines.append("  note: 所有输出维度都是单例维，请主要参考下面的样例值。")
    elif len(non_singleton_dims) == 1:
        lines.append("  note: 只有一个非单例输出维度，逐维分布相对样例索引的额外信息较少。")
    elif len(full_coverage_dims) == len(non_singleton_dims):
        lines.append("  note: 错误覆盖所有非单例维度，优先检查全局公式、累加、dtype、store 或 buffer 覆盖，而不是只修局部边界 mask。")

    if singleton_dims:
        lines.append(f"  note: 单例维度 {singleton_dims} 已省略，因为它们提供的定位信息较少。")

    return "\\n".join(lines)

def _coord_from_flat(flat_idx, shape):
    """Convert a flattened index to an ND coordinate tuple."""
    idx = int(flat_idx)
    coord = []
    for dim_size in reversed(shape):
        coord.append(idx % dim_size)
        idx //= dim_size
    return tuple(reversed(coord))

def _format_coord(coord):
    if len(coord) == 0:
        return "[scalar]"
    if len(coord) == 1:
        return f"[{coord[0]}]"
    return str(list(coord))

def compare(fw_out, impl_out, data_type):
    """Compare framework output and implementation output using layered tolerance."""
    fw = fw_out.detach().cpu()
    impl = impl_out.detach().cpu() if isinstance(impl_out, torch.Tensor) else torch.tensor(impl_out, dtype=fw.dtype)

    size = fw.numel()

    if fw.shape != impl.shape:
        raise AssertionError(f"验证失败，输出形状不一致: framework={fw.shape}, impl={impl.shape}")

    fw_nan_mask = torch.isnan(fw)
    impl_nan_mask = torch.isnan(impl)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AssertionError(f"验证失败，NaN位置不匹配: Framework={fw_nan_count}/{size}, Implementation={impl_nan_count}/{size}")
    if fw_nan_mask.any():
        nan_count = fw_nan_mask.sum().item()
        print(f"检测到NaN值: {nan_count}/{size} (位置一致，继续验证)")

    fw_inf_mask = torch.isinf(fw)
    impl_inf_mask = torch.isinf(impl)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(f"验证失败，Inf位置不匹配: Framework={fw_inf_count}/{size}, Implementation={impl_inf_count}/{size}")
    if fw_inf_mask.any():
        if not torch.equal(torch.sign(fw[fw_inf_mask]), torch.sign(impl[fw_inf_mask])):
            raise AssertionError(f"验证失败，Inf符号不匹配")

    finite_mask = torch.isfinite(fw) & torch.isfinite(impl)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        print(f"警告: 所有值都是Inf，跳过精度检查")
        return

    fw_finite = fw[finite_mask]
    impl_finite = impl[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    rtol, atol, outlier_rtol, outlier_atol, outlier_ratio = _get_tolerance(data_type)

    abs_diff = torch.abs(fw_finite.float() - impl_finite.float())
    abs_ref = torch.abs(fw_finite.float())
    strict_tol = atol + rtol * abs_ref
    relaxed_tol = outlier_atol + outlier_rtol * abs_ref

    strict_pass = abs_diff <= strict_tol
    relaxed_pass = abs_diff <= relaxed_tol

    hard_fail = int((~relaxed_pass).sum().item())
    outlier = int(((~strict_pass) & relaxed_pass).sum().item())
    total = fw_finite.numel()
    cap = int(total * outlier_ratio)

    mere = float((abs_diff / (abs_ref + atol)).mean().item())
    mare = float((abs_diff / (abs_ref + atol)).max().item())
    print(f"[precision] dtype={data_type} total={total} strict={int(strict_pass.sum().item())} outlier={outlier}/{cap} hard={hard_fail} mere={mere:.6e} mare={mare:.6e}")

    if hard_fail > 0:
        hard_fail_mask = torch.zeros(fw.shape, dtype=torch.bool)
        hard_fail_mask[finite_mask] = ~relaxed_pass
        sample_flat_indices = torch.where(hard_fail_mask.flatten())[0][:5]
        error_msg = f"验证失败，存在 {hard_fail} 个元素超过放宽阈值(hard_fail)\\n"
        error_msg += f"rtol={rtol:.6e} atol={atol:.6e} outlier_rtol={outlier_rtol:.6e} outlier_atol={outlier_atol:.6e} outlier_ratio={outlier_ratio}\\n"
        error_msg += f"mere={mere:.6e} mare={mare:.6e}\\n"
        error_msg += _format_error_locations(hard_fail_mask, fw.shape) + "\\n"
        for flat_idx in sample_flat_indices.tolist():
            coord = _coord_from_flat(flat_idx, fw.shape)
            ref_value = fw[coord].float()
            impl_value = impl[coord].float()
            sample_abs_diff = torch.abs(ref_value - impl_value)
            sample_relaxed_tol = outlier_atol + outlier_rtol * torch.abs(ref_value)
            error_msg += f"  位置{_format_coord(coord)}: ref={fw[coord]:.6e} impl={impl[coord]:.6e} abs_diff={sample_abs_diff:.6e} relaxed_tol={sample_relaxed_tol:.6e}\\n"
        raise AssertionError(error_msg)

    if outlier > cap:
        outlier_mask = torch.zeros(fw.shape, dtype=torch.bool)
        outlier_mask[finite_mask] = (~strict_pass) & relaxed_pass
        sample_flat_indices = torch.where(outlier_mask.flatten())[0][:5]
        error_msg = f"验证失败，超限元素比例超过允许值: outlier={outlier} / cap={cap}\\n"
        error_msg += f"rtol={rtol:.6e} atol={atol:.6e} outlier_rtol={outlier_rtol:.6e} outlier_atol={outlier_atol:.6e} outlier_ratio={outlier_ratio}\\n"
        error_msg += f"mere={mere:.6e} mare={mare:.6e}\\n"
        error_msg += _format_error_locations(outlier_mask, fw.shape) + "\\n"
        for flat_idx in sample_flat_indices.tolist():
            coord = _coord_from_flat(flat_idx, fw.shape)
            ref_value = fw[coord].float()
            impl_value = impl[coord].float()
            sample_abs_diff = torch.abs(ref_value - impl_value)
            sample_strict_tol = atol + rtol * torch.abs(ref_value)
            sample_relaxed_tol = outlier_atol + outlier_rtol * torch.abs(ref_value)
            error_msg += f"  位置{_format_coord(coord)}: ref={fw[coord]:.6e} impl={impl[coord]:.6e} abs_diff={sample_abs_diff:.6e} strict_tol={sample_strict_tol:.6e} relaxed_tol={sample_relaxed_tol:.6e}\\n"
        raise AssertionError(error_msg)

'''

    def get_compare_outputs_code(self) -> str:
        """Get code for comparing framework output and impl output."""
        return '''            data_type = framework_output[i].dtype
            compare(fw_out, impl_out, data_type)
'''
