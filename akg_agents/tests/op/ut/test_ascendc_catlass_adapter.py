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

"""Unit tests for ascendc_catlass DSL adapter (no NPU required)."""

import pytest

from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
from akg_agents.op.verifier.adapters.dsl.ascendc_catlass import arch_to_catlass_arch


def test_factory_returns_catlass_adapter():
    adapter = get_dsl_adapter("ascendc_catlass")
    assert adapter.get_impl_import("my_op", "ModelNew") == "from kernel import ModelNew\n"
    assert adapter.kernel_arg_is_directory is True
    assert adapter.kernel_project_dir_name == "catlass_op"
    assert "catlass_op/kernel/catlass_kernel.asc" in adapter.kernel_project_files


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("ascend910b4", "2201"),
        ("ascend950pr_9572", "3510"),
        ("ascend950dt_95a", "3510"),
    ],
)
def test_arch_to_catlass_arch(arch, expected):
    assert arch_to_catlass_arch(arch) == expected


def test_special_setup_embeds_cmake_arch_flags(tmp_path):
    # get_special_setup_code now uses the ABC-fixed signature (framework
    # only); arch + catlass_root are stashed by prepare_config which the
    # KernelVerifier calls before special-setup-code generation.
    adapter = get_dsl_adapter("ascendc_catlass")
    op_dir = tmp_path / "catlass_op"
    op_dir.mkdir()
    (op_dir / "CMakeLists.txt").write_text("", encoding="utf-8")
    cfg = {
        "task_dir": str(tmp_path),
        "catlass_op_dir": "catlass_op",
        "arch": "ascend910b3",
        "catlass_root": "/opt/catlass",
    }
    adapter.prepare_config(cfg)
    code = adapter.get_special_setup_code(framework="torch")
    assert '_catlass_arch = "2201"' in code
    assert "-DNPU_ARCH=" in code
    assert "-DCATLASS_ARCH=" in code
    assert "-DCATLASS_ROOT=" in code
    assert '"/opt/catlass"' in code or "'/opt/catlass'" in code


def test_merge_catlass_config_resolves_op_folder(tmp_path):
    from akg_agents.op.utils.catlass_paths import merge_catlass_config

    task_dir = tmp_path / "task"
    op_dir = task_dir / "catlass_op"
    op_dir.mkdir(parents=True)
    (op_dir / "CMakeLists.txt").write_text("", encoding="utf-8")

    cfg = {"task_dir": str(task_dir), "catlass_op_dir": "catlass_op"}
    merge_catlass_config(cfg, task_dir=str(task_dir))
    assert cfg["catlass_op_src"] == str(op_dir.resolve())
