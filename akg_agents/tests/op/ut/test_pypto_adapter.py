# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Unit test for PyPTO DSL adapter."""

from akg_agents.op.verifier.adapters.factory import get_dsl_adapter, get_framework_adapter


def test_pypto_adapter_basic_behavior():
    """Validate key codegen paths of the PyPTO adapter."""
    adapter = get_dsl_adapter("pypto")
    framework_adapter = get_framework_adapter("torch")

    imports = adapter.get_import_statements("torch")
    assert "import torch" in imports
    assert "import pypto" in imports
    assert "import os" in imports

    runtime_override = adapter.get_runtime_env_override_code(
        pypto_run_mode=0, pypto_runtime_debug_mode=1
    )
    assert 'AIKG_PYPTO_RUN_MODE' in runtime_override
    assert 'AIKG_PYPTO_RUNTIME_DEBUG_MODE' in runtime_override

    impl_import = adapter.get_impl_import("23_Softmax", "ModelNew")
    assert "import importlib.util" in impl_import
    assert "ModelNew = _impl_module.ModelNew" in impl_import

    create_impl = adapter.create_impl_module("torch", framework_adapter)
    assert "impl_model = ModelNew(*init_params)" in create_impl
    assert "impl_model = impl_model.to(device)" in create_impl

    call_impl = adapter.call_impl(
        "ModelNew", "inputs", 0, framework_adapter, "23_Softmax"
    )
    assert "impl_output = impl_model(*inputs)" in call_impl

    benchmark_code = adapter.benchmark_impl(
        "ModelNew", "inputs", 5, 20, "ascend", "23_Softmax"
    )
    assert "TILE_FWK_OUTPUT_DIR" in benchmark_code
    assert "trace_span" in benchmark_code
