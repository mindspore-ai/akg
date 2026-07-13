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
from akg_agents.op.utils.arch_normalize import (
    ascend_direct_invoke_npu_arch,
    ascend_soc_version,
    load_ascend_soc_catalog,
)
from akg_agents.op.verifier.adapters.factory import get_dsl_adapter, get_framework_adapter


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
        # 现在统一使用 ModelNew 类格式，模块名带 _impl 后缀
        assert "from test_op_triton_cuda_impl import ModelNew" in imports
    
    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("triton_cuda")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        # 现在使用 impl_model 调用（ModelNew 实例）
        assert "impl_output = impl_model(*inputs)" in code
    
    def test_needs_binary_io(self):
        """Test binary I/O requirement."""
        adapter = get_dsl_adapter("triton_cuda")
        assert adapter.needs_binary_io is False

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
        # 现在统一使用 ModelNew 类格式，模块名带 _impl 后缀
        assert "from test_op_triton_ascend_impl import ModelNew" in imports
    
    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("triton_ascend")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        # 现在使用 impl_model 调用（ModelNew 实例）
        assert "impl_output = impl_model(*inputs)" in code
    
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


class TestDSLAdapterAscendC:
    """Test AscendC direct-invoke DSL Adapter."""

    def _materialize_project(self, tmp_path, cmake_text):
        adapter = get_dsl_adapter("ascendc")
        project = tmp_path / "src_ascendc_op"
        project.mkdir()
        (project / "CMakeLists.txt").write_text(cmake_text, encoding="utf-8")
        cfg = {
            "arch": "ascend910b4",
            "ascendc_op_src": str(project),
            "verify_timeout": 1,
        }
        adapter.prepare_config(cfg, {"task_dir": str(tmp_path)})
        verify_dir = tmp_path / "verify"
        verify_dir.mkdir()
        adapter.materialize_impl(
            "import torch\n\nclass ModelNew(torch.nn.Module):\n    pass\n",
            str(verify_dir),
            "add_custom",
            "torch",
            "ascendc",
            task_info={"task_dir": str(tmp_path)},
            config=cfg,
        )
        return verify_dir / "ascendc_op" / "CMakeLists.txt"

    def test_direct_invoke_layout(self):
        adapter = get_dsl_adapter("ascendc")
        assert adapter.kernel_arg_is_directory is True
        assert adapter.kernel_project_dir_name == "ascendc_op"
        assert adapter.impl_func_name_template == "ModelNew"
        assert adapter.profile_via_python_script is True

    def test_list_kernel_project_files_keeps_edit_surface_core(self, tmp_path):
        adapter = get_dsl_adapter("ascendc")
        project = tmp_path / "ascendc_op"
        files = [
            "CMakeLists.txt",
            "CMakePresets.json",
            "op_kernel/add_custom_kernel.asc",
            "op_kernel/add_custom_tiling.h",
            "op_extension/add_custom_torch.cpp",
            "op_host/add_custom.asc",
            "src/launcher.cpp",
            "include/epilogue.h",
            "common/host_utils.h",
            "cmake/options.cmake",
            "scripts/gen_data.py",
            "scripts/test_torch.py",
            "run.sh",
            "README.md",
            "docs/DESIGN.md",
            "third_party/tensor_api/include/tensor.h",
            "build/generated.cpp",
        ]
        for rel in files:
            path = project / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// fixture\n", encoding="utf-8")

        editable = set(adapter.list_kernel_project_files(str(project)))

        assert {
            "ascendc_op/CMakeLists.txt",
            "ascendc_op/CMakePresets.json",
            "ascendc_op/op_kernel/add_custom_kernel.asc",
            "ascendc_op/op_kernel/add_custom_tiling.h",
            "ascendc_op/op_extension/add_custom_torch.cpp",
            "ascendc_op/op_host/add_custom.asc",
            "ascendc_op/src/launcher.cpp",
            "ascendc_op/include/epilogue.h",
            "ascendc_op/common/host_utils.h",
            "ascendc_op/cmake/options.cmake",
        } <= editable
        assert "ascendc_op/scripts/gen_data.py" not in editable
        assert "ascendc_op/scripts/test_torch.py" not in editable
        assert "ascendc_op/run.sh" not in editable
        assert "ascendc_op/README.md" not in editable
        assert "ascendc_op/docs/DESIGN.md" not in editable
        assert "ascendc_op/third_party/tensor_api/include/tensor.h" not in editable
        assert "ascendc_op/build/generated.cpp" not in editable

    def test_materialize_patches_literal_cmake_npu_arch(self, tmp_path):
        cmake_path = self._materialize_project(
            tmp_path,
            "add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-old>)\n",
        )

        assert "--npu-arch=dav-2201" in cmake_path.read_text(encoding="utf-8")

    def test_materialize_accepts_cmake_npu_arch_variable_channel(self, tmp_path):
        cmake_path = self._materialize_project(
            tmp_path,
            "set(_akg_arch ${NPU_ARCH})\n",
        )

        assert "${NPU_ARCH}" in cmake_path.read_text(encoding="utf-8")

    def test_materialize_rejects_uncontrollable_cmake_npu_arch(self, tmp_path):
        with pytest.raises(ValueError, match="no controllable NPU arch channel"):
            self._materialize_project(
                tmp_path,
                "cmake_minimum_required(VERSION 3.16)\nproject(no_arch_channel)\n",
            )

    def test_special_setup_checks_unused_cmake_arch_variables(self, tmp_path):
        adapter = get_dsl_adapter("ascendc")
        cfg = {"arch": "ascend910b4", "task_dir": str(tmp_path)}
        adapter.prepare_config(cfg, {"task_dir": str(tmp_path)})

        code = adapter.get_special_setup_code()

        assert "Manually-specified variables were not used by the project" in code
        assert "_unused_arch_vars" in code
        assert "ignored -DNPU_ARCH/-DASCENDC_NPU_ARCH" in code

    def test_get_import_statements(self):
        """Test import statements generation."""
        adapter = get_dsl_adapter("ascendc")
        imports = adapter.get_import_statements("torch")
        assert "import torch" in imports
        assert "import torch_npu" in imports

    def test_get_impl_import(self):
        """Test implementation import."""
        adapter = get_dsl_adapter("ascendc")
        imports = adapter.get_impl_import("test_op", "test_func")
        assert imports == "from kernel import ModelNew\n"

    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("ascendc")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "guarded_call as _akg_guarded_call" in code
        assert "lambda: impl_model(*inputs)" in code

    def test_ascend_soc_version_resolves_via_catalog(self):
        # CANN's platform_config .ini stems are the source of truth for both
        # spelling and case; the lookup is case-insensitive and returns the
        # exact entry — no per-family casing rules.
        catalog = frozenset({
            "Ascend910B4", "Ascend910B2C", "Ascend310P3", "Ascend910_9392",
            "Ascend950PR_957b", "Ascend950PR_9589",
            "Ascend950DT_950x", "Ascend950DT_95A1",
        })
        assert ascend_soc_version("ascend910b4", catalog) == "Ascend910B4"
        assert ascend_soc_version("ascend910b2c", catalog) == "Ascend910B2C"
        assert ascend_soc_version("ascend310p3", catalog) == "Ascend310P3"
        assert ascend_soc_version("ascend910_9392", catalog) == "Ascend910_9392"
        # 950 mixed-case variants come back exactly as CANN spells them.
        assert ascend_soc_version("ascend950pr_957b", catalog) == "Ascend950PR_957b"
        assert ascend_soc_version("ascend950dt_950x", catalog) == "Ascend950DT_950x"
        assert ascend_soc_version("ascend950dt_95a1", catalog) == "Ascend950DT_95A1"
        # input case is irrelevant; only catalog membership matters
        assert ascend_soc_version("ASCEND950PR_957B", catalog) == "Ascend950PR_957b"
        # bare family / unknown / empty catalog -> None
        assert ascend_soc_version("ascend950pr", catalog) is None
        assert ascend_soc_version("unknown", catalog) is None
        assert ascend_soc_version("ascend910b4", frozenset()) is None

    def test_load_ascend_soc_catalog_reads_platform_config(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            pc = os.path.join(home, "compiler", "data", "platform_config", "x")
            os.makedirs(pc)
            for name in ("Ascend950PR_957b.ini", "Ascend910B3.ini"):
                open(os.path.join(pc, name), "w").close()
            prev = os.environ.get("ASCEND_HOME_PATH")
            os.environ["ASCEND_HOME_PATH"] = home
            try:
                load_ascend_soc_catalog.cache_clear()
                assert load_ascend_soc_catalog() == frozenset(
                    {"Ascend950PR_957b", "Ascend910B3"})
            finally:
                load_ascend_soc_catalog.cache_clear()
                if prev is None:
                    os.environ.pop("ASCEND_HOME_PATH", None)
                else:
                    os.environ["ASCEND_HOME_PATH"] = prev

    def test_ascend_direct_invoke_npu_arch_derives_from_arch(self):
        assert ascend_direct_invoke_npu_arch("ascend910b4") == "dav-2201"
        assert ascend_direct_invoke_npu_arch("ascend310p3") == "dav-2201"
        assert ascend_direct_invoke_npu_arch("ascend950dt_95a") == "dav-3510"


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
        # 现在统一使用 ModelNew 类格式，模块名带 _impl 后缀
        assert "from test_op_cpp_impl import ModelNew" in imports
    
    def test_call_impl(self):
        """Test call implementation code generation."""
        adapter = get_dsl_adapter("cpp")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        # 现在使用 impl_model 调用（ModelNew 实例）
        assert "impl_output = impl_model(*inputs)" in code
    
    def test_benchmark_impl(self):
        """Test benchmark code generation."""
        adapter = get_dsl_adapter("cpp")
        code = adapter.benchmark_impl("test_func", "inputs", 10, 100, "cpu", "test_op")
        assert "time.perf_counter" in code
        assert "warmup" in code.lower()


class TestDSLAdapterCudaC:
    """Test CUDA C DSL Adapter."""
    
    def test_get_import_statements(self):
        """CUDA C adapter should emit torch cpp extension imports."""
        adapter = get_dsl_adapter("cuda_c")
        imports = adapter.get_import_statements("torch")
        assert "from torch.utils.cpp_extension import load_inline" in imports
    
    def test_get_impl_import(self):
        """CUDA C adapter now imports ModelNew."""
        adapter = get_dsl_adapter("cuda_c")
        imports = adapter.get_impl_import("test_op", "test_func")
        # 模块名带 _impl 后缀
        assert "from test_op_cuda_c_impl import ModelNew" in imports
    
    def test_create_impl_module(self):
        """Impl model should be instantiated once and moved to device."""
        adapter = get_dsl_adapter("cuda_c")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.create_impl_module("torch", framework_adapter)
        assert "impl_model = ModelNew(*init_params)" in code
        assert "impl_model = impl_model.to(device)" in code
    
    def test_call_impl(self):
        """Call site should reuse impl_model."""
        adapter = get_dsl_adapter("cuda_c")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "impl_output = impl_model(*inputs)" in code
    
    def test_benchmark_impl(self):
        """Benchmark section should invoke impl_model."""
        adapter = get_dsl_adapter("cuda_c")
        code = adapter.benchmark_impl("test_func", "inputs", 5, 50, "cuda", "test_op")
        assert "def cuda_c_benchmark_fn()" in code
        assert "impl_model(*inputs)" in code
        assert "torch.cuda.synchronize()" in code


class TestDSLAdapterTilelangCuda:
    """Test TileLang CUDA DSL Adapter."""
    
    def test_get_import_statements(self):
        adapter = get_dsl_adapter("tilelang_cuda")
        imports = adapter.get_import_statements("torch")
        assert "import tilelang.language as T" in imports
    
    def test_get_impl_import(self):
        adapter = get_dsl_adapter("tilelang_cuda")
        imports = adapter.get_impl_import("test_op", "test_func")
        # 模块名带 _impl 后缀
        assert "from test_op_tilelang_cuda_impl import ModelNew" in imports
    
    def test_create_impl_module(self):
        adapter = get_dsl_adapter("tilelang_cuda")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.create_impl_module("torch", framework_adapter)
        assert "impl_model = ModelNew(*init_params)" in code
        assert "impl_model = impl_model.to(device)" in code
    
    def test_call_impl(self):
        adapter = get_dsl_adapter("tilelang_cuda")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "impl_output = impl_model(*inputs)" in code
    
    def test_benchmark_impl(self):
        adapter = get_dsl_adapter("tilelang_cuda")
        code = adapter.benchmark_impl("test_func", "inputs", 5, 50, "cuda", "test_op")
        assert "tilelang_benchmark_fn" in code
        assert "impl_model(*inputs)" in code
        assert "torch.cuda.synchronize()" in code
        assert "torch.cuda.Event" in code


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
    
    def test_get_dsl_adapter_ascendc(self):
        """Test getting AscendC adapter."""
        adapter = get_dsl_adapter("ascendc")
        assert adapter is not None
        assert adapter.__class__.__name__ == "DSLAdapterAscendC"
    
    def test_get_dsl_adapter_invalid(self):
        """Test getting invalid DSL adapter."""
        with pytest.raises(ValueError, match="Unsupported DSL"):
            get_dsl_adapter("invalid")


class TestDSLAdapterTilelangAscend:
    """Test TileLang-Ascend DSL Adapter."""

    def test_get_import_statements(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        imports = adapter.get_import_statements("torch")
        assert "import tilelang" in imports
        assert "import tilelang.language as T" in imports
        assert "import torch_npu" in imports
        assert "apply_tilelang_patches" in imports

    def test_get_impl_import(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        imports = adapter.get_impl_import("test_op", "test_func")
        assert "from test_op_tilelang_ascend_impl import ModelNew" in imports

    def test_create_impl_module(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.create_impl_module("torch", framework_adapter)
        assert "impl_model = ModelNew(*init_params)" in code
        assert "impl_model = impl_model.to(device)" in code

    def test_call_impl(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        framework_adapter = get_framework_adapter("torch")
        code = adapter.call_impl("test_func", "inputs", 0, framework_adapter, "test_op")
        assert "impl_output = impl_model(*inputs)" in code

    def test_benchmark_impl(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        code = adapter.benchmark_impl("test_func", "inputs", 5, 50, "ascend", "test_op")
        assert "tilelang_benchmark_fn" in code
        assert "impl_model(*inputs)" in code
        assert "torch.npu.synchronize()" in code

    def test_get_special_setup_code(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        code = adapter.get_special_setup_code()
        assert "tilelang.cache.clear_cache()" in code
        assert "apply_tilelang_patches" in code

    def test_needs_binary_io(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        assert adapter.needs_binary_io is False
        assert not callable(adapter.needs_binary_io)

    def test_needs_compilation(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        assert adapter.needs_compilation is False
        assert not callable(adapter.needs_compilation)


class TestDSLAdapterFactoryTilelangAscend:
    """Test Factory produces correct tilelang_ascend adapter."""

    def test_get_dsl_adapter_tilelang_ascend(self):
        adapter = get_dsl_adapter("tilelang_ascend")
        assert adapter is not None
        assert adapter.__class__.__name__ == "DSLAdapterTilelangAscend"
