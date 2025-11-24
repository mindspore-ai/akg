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

"""Unit tests for Framework Adapters."""

import pytest
from ai_kernel_generator.core.verifier.adapters.factory import get_framework_adapter


class TestFrameworkAdapterTorch:
    """Test PyTorch Framework Adapter."""
    
    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_framework_adapter("torch")
        imports = adapter.get_import_statements()
        assert "import torch" in imports
        assert imports.endswith("\n")
    
    def test_get_framework_import_static(self):
        """Test framework model import for static shape."""
        adapter = get_framework_adapter("torch")
        imports = adapter.get_framework_import("test_op", False)
        assert "from test_op_torch import Model as FrameworkModel" in imports
        assert "get_inputs" in imports
        assert "get_inputs_dyn_list" not in imports
    
    def test_get_framework_import_dynamic(self):
        """Test framework model import for dynamic shape."""
        adapter = get_framework_adapter("torch")
        imports = adapter.get_framework_import("test_op", True)
        assert "from test_op_torch import Model as FrameworkModel" in imports
        assert "get_inputs_dyn_list" in imports
    
    def test_get_device_setup_cuda(self):
        """Test device setup for CUDA."""
        adapter = get_framework_adapter("torch")
        code = adapter.get_device_setup_code("cuda", "a100", 0)
        assert "CUDA_VISIBLE_DEVICES" in code
        assert "torch.device(\"cuda\")" in code
    
    def test_get_device_setup_ascend(self):
        """Test device setup for Ascend."""
        adapter = get_framework_adapter("torch")
        code = adapter.get_device_setup_code("ascend", "ascend910b4", 0)
        assert "DEVICE_ID" in code
        assert "torch.device(\"npu\")" in code
    
    def test_get_process_input_code(self):
        """Test process_input code generation."""
        adapter = get_framework_adapter("torch")
        code = adapter.get_process_input_code("cuda", "triton_cuda")
        assert "def process_input" in code
        assert "x.to(device)" in code
    
    def test_get_process_input_code_ascendc(self):
        """Test process_input code for AscendC."""
        adapter = get_framework_adapter("torch")
        code = adapter.get_process_input_code("ascend", "ascendc")
        assert "def process_input" in code
        assert "x.npu()" in code
    
    def test_get_set_seed_code(self):
        """Test set seed code generation."""
        adapter = get_framework_adapter("torch")
        code = adapter.get_set_seed_code("cuda")
        assert "torch.manual_seed(0)" in code
        assert "torch.npu.manual_seed" not in code
        
        code_ascend = adapter.get_set_seed_code("ascend")
        assert "torch.manual_seed(0)" in code_ascend
        assert "torch.npu.manual_seed(0)" in code_ascend
    
    def test_get_limit(self):
        """Test precision limit calculation."""
        adapter = get_framework_adapter("torch")
        import torch
        assert adapter.get_limit(torch.float16) == 0.004
        assert adapter.get_limit(torch.bfloat16) == 0.03
        assert adapter.get_limit(torch.int8) == 0.01
        assert adapter.get_limit(torch.float32) == 0.02
    
    def test_get_tensor_type(self):
        """Test tensor type."""
        adapter = get_framework_adapter("torch")
        import torch
        assert adapter.get_tensor_type() == torch.Tensor
    
    def test_get_binary_io_functions(self):
        """Test binary I/O functions generation."""
        adapter = get_framework_adapter("torch")
        code = adapter.get_binary_io_functions("test_op")
        assert "def save_tensor" in code
        assert "def load_tensor" in code
        assert "def gen_binary_data" in code
        assert "def load_binary_data" in code
        assert "test_op" in code


class TestFrameworkAdapterMindSpore:
    """Test MindSpore Framework Adapter."""
    
    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_framework_adapter("mindspore")
        imports = adapter.get_import_statements()
        assert "import mindspore as ms" in imports
        assert "from mindspore.common import np_dtype" in imports
    
    def test_get_framework_import(self):
        """Test framework model import."""
        adapter = get_framework_adapter("mindspore")
        imports = adapter.get_framework_import("test_op", False)
        assert "from test_op_mindspore import Model as FrameworkModel" in imports
    
    def test_get_device_setup_ascend(self):
        """Test device setup for Ascend."""
        adapter = get_framework_adapter("mindspore")
        code = adapter.get_device_setup_code("ascend", "ascend910b4", 0)
        assert "DEVICE_ID" in code
        assert "device = \"Ascend\"" in code
    
    def test_get_device_setup_cpu(self):
        """Test device setup for CPU."""
        adapter = get_framework_adapter("mindspore")
        code = adapter.get_device_setup_code("cpu", "x86_64", 0)
        assert "DEVICE_ID" in code
        assert "device = \"CPU\"" in code
    
    def test_get_process_input_code(self):
        """Test process_input code generation."""
        adapter = get_framework_adapter("mindspore")
        code = adapter.get_process_input_code("ascend", "triton_ascend")
        assert "def process_input" in code
        assert "return x" in code
    
    def test_get_set_seed_code(self):
        """Test set seed code generation."""
        adapter = get_framework_adapter("mindspore")
        code = adapter.get_set_seed_code("ascend")
        assert "ms.set_seed(0)" in code
    
    def test_get_dtype_mapping(self):
        """Test dtype mapping."""
        adapter = get_framework_adapter("mindspore")
        mapping = adapter.get_dtype_mapping()
        assert mapping is not None
        import mindspore as ms
        import numpy as np
        assert ms.float32 in mapping
        assert mapping[ms.float32] == np.float32


class TestFrameworkAdapterNumpy:
    """Test NumPy Framework Adapter."""
    
    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_framework_adapter("numpy")
        imports = adapter.get_import_statements()
        assert "import numpy as np" in imports
    
    def test_get_framework_import(self):
        """Test framework model import."""
        adapter = get_framework_adapter("numpy")
        imports = adapter.get_framework_import("test_op", False)
        assert "from test_op_numpy import Model as FrameworkModel" in imports
    
    def test_get_device_setup(self):
        """Test device setup (should be empty for NumPy)."""
        adapter = get_framework_adapter("numpy")
        code = adapter.get_device_setup_code("cpu", "x86_64", 0)
        assert code == ""
    
    def test_get_process_input_code(self):
        """Test process_input code generation."""
        adapter = get_framework_adapter("numpy")
        code = adapter.get_process_input_code("cpu", "swft")
        assert "def process_input" in code
        assert "return x" in code
    
    def test_get_set_seed_code(self):
        """Test set seed code generation."""
        adapter = get_framework_adapter("numpy")
        code = adapter.get_set_seed_code("cpu")
        assert "np.random.seed(0)" in code
    
    def test_get_limit(self):
        """Test precision limit calculation."""
        adapter = get_framework_adapter("numpy")
        import numpy as np
        assert adapter.get_limit(np.float16) == 0.004
        assert adapter.get_limit(np.int8) == 0.01
        assert adapter.get_limit(np.float32) == 0.004


class TestFrameworkAdapterFactory:
    """Test Framework Adapter Factory."""
    
    def test_get_framework_adapter_torch(self):
        """Test getting PyTorch adapter."""
        adapter = get_framework_adapter("torch")
        assert adapter is not None
        assert adapter.__class__.__name__ == "FrameworkAdapterTorch"
    
    def test_get_framework_adapter_mindspore(self):
        """Test getting MindSpore adapter."""
        adapter = get_framework_adapter("mindspore")
        assert adapter is not None
        assert adapter.__class__.__name__ == "FrameworkAdapterMindSpore"
    
    def test_get_framework_adapter_numpy(self):
        """Test getting NumPy adapter."""
        adapter = get_framework_adapter("numpy")
        assert adapter is not None
        assert adapter.__class__.__name__ == "FrameworkAdapterNumpy"
    
    def test_get_framework_adapter_invalid(self):
        """Test getting invalid framework adapter."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            get_framework_adapter("invalid")

