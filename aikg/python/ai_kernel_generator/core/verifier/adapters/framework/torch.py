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
    
    def get_framework_import(self, op_name: str, is_dynamic_shape: bool) -> str:
        """Return framework model and input function imports."""
        if is_dynamic_shape:
            return f"from {op_name}_torch import Model as FrameworkModel, get_init_inputs, get_inputs_dyn_list\n"
        else:
            return f"from {op_name}_torch import Model as FrameworkModel, get_init_inputs, get_inputs\n"
    
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
        """Get precision limit for dtype."""
        if dtype == torch.float16:
            return 0.004
        elif dtype == torch.bfloat16:
            return 0.03
        elif dtype == torch.int8:
            return 0.01
        else:
            return 0.02
    
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
            if "ascend910" in arch:
                return f"""    os.environ['DEVICE_ID'] = str({device_id})
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

