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

"""Unit tests for DSL Adapters."""

import pytest
from ai_kernel_generator.core.verifier.adapters.factory import get_dsl_adapter, get_framework_adapter


class TestDSLAdapterTritonCuda:
    """Test Triton CUDA DSL Adapter."""
    
    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_dsl_adapter("triton_cuda")
        imports = adapter.get_import_statements("torch")
        assert "import triton" in imports
        assert "import triton.language as tl" in imports
    
    def test_get_impl_import(self):
        """Test implementation import."""
        adapter = get_dsl_adapter("triton_cuda")
        imports = adapter.get_impl_import("test_op", "test_func")
        assert "from test_op_triton_cuda import test_func" in imports
    
    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("triton_cuda")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "impl_output = test_func(*inputs)" in code
    
    def test_needs_binary_io(self):
        """Test binary I/O requirement."""
        adapter = get_dsl_adapter("triton_cuda")
        assert adapter.needs_binary_io() is False
    
    def test_needs_compilation(self):
        """Test compilation requirement."""
        adapter = get_dsl_adapter("triton_cuda")
        assert adapter.needs_compilation() is False
    
    def test_benchmark_impl(self):
        """Test benchmark code generation."""
        adapter = get_dsl_adapter("triton_cuda")
        code = adapter.benchmark_impl("test_func", "inputs", 10, 100, "cuda", "test_op")
        assert "triton.testing.do_bench" in code
        assert "warmup=10" in code
        assert "rep=100" in code


class TestDSLAdapterTritonAscend:
    """Test Triton Ascend DSL Adapter."""
    
    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_dsl_adapter("triton_ascend")
        imports = adapter.get_import_statements("torch")
        assert "import triton" in imports
        assert "apply_triton_patches" in imports
    
    def test_get_impl_import(self):
        """Test implementation import."""
        adapter = get_dsl_adapter("triton_ascend")
        imports = adapter.get_impl_import("test_op", "test_func")
        assert "from test_op_triton_ascend import test_func" in imports
    
    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("triton_ascend")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "impl_output = test_func(*inputs)" in code
    
    def test_benchmark_impl_ascend(self):
        """Test benchmark code generation for Ascend."""
        adapter = get_dsl_adapter("triton_ascend")
        code = adapter.benchmark_impl("test_func", "inputs", 10, 100, "ascend", "test_op")
        assert "profiler_npu" in code
        assert "get_collected_config_timings" in code
        assert "autotune_info_case_" in code
    
    def test_get_special_setup_code(self):
        """Test special setup code."""
        adapter = get_dsl_adapter("triton_ascend")
        code = adapter.get_special_setup_code()
        assert "apply_triton_patches" in code


class TestDSLAdapterSwft:
    """Test SWFT DSL Adapter."""
    
    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_dsl_adapter("swft")
        imports = adapter.get_import_statements("torch")
        assert "from swft.core import *" in imports
        assert "from swft.api import *" in imports
    
    def test_get_impl_import(self):
        """Test implementation import."""
        adapter = get_dsl_adapter("swft")
        imports = adapter.get_impl_import("test_op", "test_func")
        assert "from test_op_swft import test_func" in imports
    
    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("swft")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "gen_binary_data" in code
        assert "load_binary_data" in code
        assert "test_func(device_id=int(0))" in code
    
    def test_needs_binary_io(self):
        """Test binary I/O requirement."""
        adapter = get_dsl_adapter("swft")
        assert adapter.needs_binary_io() is True
    
    def test_benchmark_impl(self):
        """Test benchmark code generation."""
        adapter = get_dsl_adapter("swft")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.benchmark_impl("test_func", "inputs", 10, 100, "ascend", "test_op",
                                      framework_model="framework_model", framework_adapter=framework_adapter, device_id=0)
        assert "gen_binary_data" in code
        assert "test_func(device_id=int(0))" in code


class TestDSLAdapterAscendC:
    """Test AscendC DSL Adapter."""
    
    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_dsl_adapter("ascendc")
        imports = adapter.get_import_statements("torch")
        assert "import torch_npu" in imports
        assert "import subprocess" in imports
    
    def test_get_impl_import(self):
        """Test implementation import (should be empty for AscendC)."""
        adapter = get_dsl_adapter("ascendc")
        imports = adapter.get_impl_import("test_op", "test_func")
        assert imports == ""
    
    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("ascendc")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "subprocess.run" in code
        assert "run.sh" in code
        assert "{impl_func_name}" in code  # AscendC uses template variable
        assert "run_{impl_func_name}" in code
    
    def test_needs_compilation(self):
        """Test compilation requirement."""
        adapter = get_dsl_adapter("ascendc")
        assert adapter.needs_compilation() is True


class TestDSLAdapterCpp:
    """Test C++ DSL Adapter."""
    
    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_dsl_adapter("cpp")
        imports = adapter.get_import_statements("torch")
        assert "import torch" in imports
    
    def test_get_impl_import(self):
        """Test implementation import."""
        adapter = get_dsl_adapter("cpp")
        imports = adapter.get_impl_import("test_op", "test_func")
        assert "from test_op_cpp import test_func" in imports
    
    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("cpp")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "impl_output = test_func(*inputs)" in code
    
    def test_benchmark_impl(self):
        """Test benchmark code generation."""
        adapter = get_dsl_adapter("cpp")
        code = adapter.benchmark_impl("test_func", "inputs", 10, 100, "cpu", "test_op")
        assert "time.perf_counter" in code
        assert "warmup" in code.lower()


class TestDSLAdapterFactory:
    """Test DSL Adapter Factory."""
    
    def test_get_dsl_adapter_triton_cuda(self):
        """Test getting Triton CUDA adapter."""
        adapter = get_dsl_adapter("triton_cuda")
        assert adapter is not None
        assert adapter.__class__.__name__ == "DSLAdapterTritonCuda"
    
    def test_get_dsl_adapter_triton_ascend(self):
        """Test getting Triton Ascend adapter."""
        adapter = get_dsl_adapter("triton_ascend")
        assert adapter is not None
        assert adapter.__class__.__name__ == "DSLAdapterTritonAscend"
    
    def test_get_dsl_adapter_swft(self):
        """Test getting SWFT adapter."""
        adapter = get_dsl_adapter("swft")
        assert adapter is not None
        assert adapter.__class__.__name__ == "DSLAdapterSwft"
    
    def test_get_dsl_adapter_ascendc(self):
        """Test getting AscendC adapter."""
        adapter = get_dsl_adapter("ascendc")
        assert adapter is not None
        assert adapter.__class__.__name__ == "DSLAdapterAscendC"
    
    def test_get_dsl_adapter_invalid(self):
        """Test getting invalid DSL adapter."""
        with pytest.raises(ValueError, match="Unsupported DSL"):
            get_dsl_adapter("invalid")

