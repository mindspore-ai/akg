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

"""Unit tests for profile template generation using Adapters."""

import os
import tempfile
import pytest
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier


class TestProfileTemplateWithAdapters:
    """Test profile template generation with adapters."""
    
    def test_gen_profile_base_torch_triton_cuda(self):
        """Test generating base profile script for torch + triton_cuda."""
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
            
            # Generate profile project
            verifier.gen_profile_project(verify_dir, device_id=0, warmup_times=5, run_times=50)
            
            # Check files are created
            base_profile = os.path.join(verify_dir, f"profile_{op_name}_base.py")
            gen_profile = os.path.join(verify_dir, f"profile_{op_name}_generation.py")
            assert os.path.exists(base_profile)
            assert os.path.exists(gen_profile)
            
            # Check base profile script
            with open(base_profile, "r", encoding="utf-8") as f:
                content = f.read()
                assert "import torch" in content
                assert f"from {op_name}_torch import Model as FrameworkModel" in content
                assert "def run_base_implementations" in content
                assert "triton.testing.do_bench" in content or "framework_model" in content
            
            # Check generation profile script
            with open(gen_profile, "r", encoding="utf-8") as f:
                content = f.read()
                assert "import torch" in content
                assert f"from {op_name}_triton_cuda import" in content
                assert "def run_generation_implementations" in content
    
    def test_gen_profile_swft_binary_io(self):
        """Test generating profile script for SWFT (needs binary I/O)."""
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
            
            # Generate profile project
            verifier.gen_profile_project(verify_dir, device_id=0, warmup_times=5, run_times=50)
            
            # Check generation profile includes binary I/O
            gen_profile = os.path.join(verify_dir, f"profile_{op_name}_generation.py")
            with open(gen_profile, "r", encoding="utf-8") as f:
                content = f.read()
                assert "def save_tensor" in content
                assert "def gen_binary_data" in content
                assert "gen_binary_data" in content
    
    def test_gen_profile_dynamic_shape(self):
        """Test generating profile script with dynamic shape."""
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
            
            verify_dir = os.path.join(tmpdir, "verify")
            os.makedirs(verify_dir, exist_ok=True)
            
            # Generate profile project
            verifier.gen_profile_project(verify_dir, device_id=0, warmup_times=5, run_times=50)
            
            # Check profile scripts use dynamic shape
            base_profile = os.path.join(verify_dir, f"profile_{op_name}_base.py")
            gen_profile = os.path.join(verify_dir, f"profile_{op_name}_generation.py")
            
            for profile_file in [base_profile, gen_profile]:
                with open(profile_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    assert "get_inputs_dyn_list" in content
                    assert "case_count" in content or "case_times" in content

