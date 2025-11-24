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

"""Integration tests for KernelVerifier using Adapters."""

import os
import tempfile
import shutil
import pytest
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier


class TestKernelVerifierWithAdapters:
    """Test KernelVerifier integration with adapters."""
    
    def test_gen_verify_project_torch_triton_cuda(self):
        """Test generating verify project for torch + triton_cuda."""
        op_name = "test_op"
        framework = "torch"
        dsl = "triton_cuda"
        backend = "cuda"
        arch = "a100"
        
        framework_code = """
def get_init_inputs():
    return []

class Model:
    def __init__(self, *args):
        pass
    def __call__(self, *args):
        import torch
        return torch.tensor([1.0, 2.0, 3.0])

def get_inputs():
    import torch
    return [torch.tensor([1.0, 2.0, 3.0])]
"""
        
        impl_code = """
def test_op_triton_cuda_torch(x):
    import torch
    return x * 2
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"log_dir": tmpdir}
            verifier = KernelVerifier(
                op_name=op_name,
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                framework_code=framework_code,
                impl_func_name="test_op_triton_cuda_torch",
                config=config
            )
            
            verify_dir = os.path.join(tmpdir, "verify")
            os.makedirs(verify_dir, exist_ok=True)
            
            # Generate verify project
            verifier.gen_verify_project(impl_code, verify_dir, device_id=0)
            
            # Check files are created
            assert os.path.exists(os.path.join(verify_dir, f"{op_name}_{framework}.py"))
            assert os.path.exists(os.path.join(verify_dir, f"{op_name}_{dsl}.py"))
            assert os.path.exists(os.path.join(verify_dir, f"verify_{op_name}.py"))
            
            # Check verify script uses adapter-generated code
            verify_script = os.path.join(verify_dir, f"verify_{op_name}.py")
            with open(verify_script, "r", encoding="utf-8") as f:
                content = f.read()
                # Check framework imports
                assert "import torch" in content
                assert f"from {op_name}_torch import Model as FrameworkModel" in content
                # Check DSL imports
                assert "import triton" in content or "from" in content
                assert f"from {op_name}_triton_cuda import" in content
                # Check device setup
                assert "CUDA_VISIBLE_DEVICES" in content or "device = torch.device" in content
                # Check process_input
                assert "def process_input" in content
    
    def test_gen_verify_project_mindspore_triton_ascend(self):
        """Test generating verify project for mindspore + triton_ascend."""
        op_name = "test_op"
        framework = "mindspore"
        dsl = "triton_ascend"
        backend = "ascend"
        arch = "ascend910b4"
        
        framework_code = """
def get_init_inputs():
    return []

class Model:
    def __init__(self, *args):
        pass
    def __call__(self, *args):
        import mindspore as ms
        return ms.Tensor([1.0, 2.0, 3.0])

def get_inputs():
    import mindspore as ms
    return [ms.Tensor([1.0, 2.0, 3.0])]
"""
        
        impl_code = """
def test_op_triton_ascend_mindspore(x):
    import mindspore as ms
    return x * 2
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"log_dir": tmpdir}
            verifier = KernelVerifier(
                op_name=op_name,
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                framework_code=framework_code,
                impl_func_name="test_op_triton_ascend_mindspore",
                config=config
            )
            
            verify_dir = os.path.join(tmpdir, "verify")
            os.makedirs(verify_dir, exist_ok=True)
            
            # Generate verify project
            verifier.gen_verify_project(impl_code, verify_dir, device_id=0)
            
            # Check files are created
            assert os.path.exists(os.path.join(verify_dir, f"{op_name}_{framework}.py"))
            assert os.path.exists(os.path.join(verify_dir, f"{op_name}_{dsl}.py"))
            assert os.path.exists(os.path.join(verify_dir, f"verify_{op_name}.py"))
            
            # Check verify script
            verify_script = os.path.join(verify_dir, f"verify_{op_name}.py")
            with open(verify_script, "r", encoding="utf-8") as f:
                content = f.read()
                # Check framework imports
                assert "import mindspore as ms" in content
                assert f"from {op_name}_mindspore import Model as FrameworkModel" in content
                # Check device setup
                assert "DEVICE_ID" in content
                assert "device = \"Ascend\"" in content
    
    def test_gen_verify_project_swft_binary_io(self):
        """Test generating verify project for SWFT (needs binary I/O)."""
        op_name = "test_op"
        framework = "torch"
        dsl = "swft"
        backend = "ascend"
        arch = "ascend310p3"
        
        framework_code = """
def get_init_inputs():
    return []

class Model:
    def __init__(self, *args):
        pass
    def __call__(self, *args):
        import torch
        return torch.tensor([1.0, 2.0, 3.0])

def get_inputs():
    import torch
    return [torch.tensor([1.0, 2.0, 3.0])]
"""
        
        impl_code = """
def test_op_swft_torch(device_id):
    pass
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"log_dir": tmpdir}
            verifier = KernelVerifier(
                op_name=op_name,
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                framework_code=framework_code,
                impl_func_name="test_op_swft_torch",
                config=config
            )
            
            verify_dir = os.path.join(tmpdir, "verify")
            os.makedirs(verify_dir, exist_ok=True)
            
            # Generate verify project
            verifier.gen_verify_project(impl_code, verify_dir, device_id=0)
            
            # Check verify script includes binary I/O functions
            verify_script = os.path.join(verify_dir, f"verify_{op_name}.py")
            with open(verify_script, "r", encoding="utf-8") as f:
                content = f.read()
                # Check binary I/O functions are included
                assert "def save_tensor" in content
                assert "def load_tensor" in content
                assert "def gen_binary_data" in content
                assert "def load_binary_data" in content
                # Check SWFT call uses binary I/O
                assert "gen_binary_data" in content
                assert "load_binary_data" in content
    
    def test_dynamic_shape_detection(self):
        """Test dynamic shape detection."""
        op_name = "test_op"
        framework = "torch"
        dsl = "triton_cuda"
        backend = "cuda"
        arch = "a100"
        
        framework_code = """
def get_init_inputs():
    return []

class Model:
    def __init__(self, *args):
        pass
    def __call__(self, *args):
        import torch
        return torch.tensor([1.0, 2.0, 3.0])

def get_inputs_dyn_list():
    import torch
    return [[torch.tensor([1.0])], [torch.tensor([2.0])]]
"""
        
        impl_code = """
def test_op_triton_cuda_torch(x):
    import torch
    return x * 2
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"log_dir": tmpdir}
            verifier = KernelVerifier(
                op_name=op_name,
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                framework_code=framework_code,
                impl_func_name="test_op_triton_cuda_torch",
                config=config
            )
            
            # Check dynamic shape detection
            assert verifier._detect_dynamic_shape() is True
            
            verify_dir = os.path.join(tmpdir, "verify")
            os.makedirs(verify_dir, exist_ok=True)
            
            # Generate verify project
            verifier.gen_verify_project(impl_code, verify_dir, device_id=0)
            
            # Check verify script uses dynamic shape
            verify_script = os.path.join(verify_dir, f"verify_{op_name}.py")
            with open(verify_script, "r", encoding="utf-8") as f:
                content = f.read()
                assert "get_inputs_dyn_list" in content

