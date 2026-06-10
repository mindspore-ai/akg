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

"""Round-trip test: scaffold_task_dir -> load_yaml_config.

Verifies that every field written by the scaffolder is correctly read
back by the config loader, catching "upstream writes / downstream
doesn't read" regressions.
"""

import os
from unittest.mock import patch

import pytest

from akg_agents.op.autoresearch.adapters.task_scaffolder import scaffold_task_dir
from akg_agents.op.autoresearch.framework import config_loader
from akg_agents.op.autoresearch.framework.config_loader import load_yaml_config


@pytest.fixture
def task_dir(tmp_path):
    """Scaffold a task dir with all adapter fields populated."""
    return scaffold_task_dir(
        base_dir=str(tmp_path),
        op_name="test_matmul",
        task_desc="# reference\nimport torch\n",
        editable_files={"kernel.py": "# kernel\n"},
        program_md="Optimize the kernel.",
        context_files={"api.md": "# API docs\n"},
        max_rounds=15,
        eval_timeout=90,
        dsl="triton_cuda",
        framework="torch",
        backend="cuda",
        arch="a100",
    )


class TestScaffoldRoundtrip:
    """scaffold_task_dir -> load_yaml_config preserves all fields."""

    def test_adapter_fields_roundtrip(self, task_dir):
        cfg = load_yaml_config(task_dir)
        assert cfg is not None
        assert cfg.dsl == "triton_cuda"
        assert cfg.framework == "torch"
        assert cfg.backend == "cuda"
        assert cfg.arch == "a100"

    def test_basic_fields_roundtrip(self, task_dir):
        cfg = load_yaml_config(task_dir)
        assert cfg.name == "test_matmul"
        assert cfg.max_rounds == 15
        assert cfg.eval_timeout == 90
        assert cfg.primary_metric == "latency_us"
        assert cfg.lower_is_better is True

    def test_agent_fields_roundtrip(self, task_dir):
        cfg = load_yaml_config(task_dir)
        assert cfg.program_file == "program.md"
        assert cfg.ref_file == "reference.py"
        assert "api.md" in cfg.context_files

    def test_arch_none_when_empty(self, tmp_path):
        """When arch is not provided, cfg.arch should be None."""
        td = scaffold_task_dir(
            base_dir=str(tmp_path),
            op_name="no_arch",
            task_desc="ref",
            editable_files={"k.py": ""},
            program_md="prog",
            context_files={},
        )
        cfg = load_yaml_config(td)
        assert cfg.arch is None

    def test_files_exist(self, task_dir):
        """Scaffolded files should physically exist."""
        assert os.path.isfile(os.path.join(task_dir, "task.yaml"))
        assert os.path.isfile(os.path.join(task_dir, "kernel.py"))
        assert os.path.isfile(os.path.join(task_dir, "reference.py"))
        assert os.path.isfile(os.path.join(task_dir, "program.md"))

    def test_multifile_dsl_project_files_are_editable(self, tmp_path):
        """Multi-file DSL scaffolds expose the project tree to AgentLoop."""
        project_src = tmp_path / "catlass_src"
        project_files = [
            "kernel/catlass_kernel.asc",
            "include/catlass_kernel.h",
            "src/catlass_torch.cpp",
            "CMakeLists.txt",
        ]
        for rel in project_files:
            path = project_src / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// catlass fixture\n", encoding="utf-8")

        td = scaffold_task_dir(
            base_dir=str(tmp_path),
            op_name="catlass_case",
            task_desc="# reference\n",
            editable_files={"kernel.py": "# wrapper\n"},
            dsl="ascendc_catlass",
            framework="torch",
            backend="ascend",
            arch="ascend910b4",
            kernel_project_src=str(project_src),
        )
        cfg = load_yaml_config(td)
        expected = [
            "kernel.py",
            "catlass_op/kernel/catlass_kernel.asc",
            "catlass_op/include/catlass_kernel.h",
            "catlass_op/src/catlass_torch.cpp",
            "catlass_op/CMakeLists.txt",
        ]
        for rel in expected:
            assert rel in cfg.editable_files
            assert os.path.isfile(os.path.join(td, rel))


# A fake guardrails dict with a hardware-scoped rule for "a100".
_FAKE_GUARDRAILS = {
    "global": {
        "diff": ["^\\s*#"],
    },
    "hardware": {
        "a100": {
            "diff_any": ["\\bforbidden_hw_pattern\\b"],
        },
    },
}


class TestGuardrailRoundtrip:
    """Top-level arch flows through to build_forbidden_patterns."""

    def test_toplevel_arch_activates_hardware_guardrails(self, tmp_path):
        """arch='a100' in task.yaml -> hardware.a100 rules in cfg.forbidden_patterns."""
        td = scaffold_task_dir(
            base_dir=str(tmp_path),
            op_name="hw_guard",
            task_desc="ref",
            editable_files={"k.py": ""},
            program_md="prog",
            context_files={},
            arch="a100",
        )
        with patch.object(config_loader, "_guardrails_cache", _FAKE_GUARDRAILS):
            cfg = load_yaml_config(td)
        assert "\\bforbidden_hw_pattern\\b" in cfg.forbidden_patterns.get("diff_any", [])
        # Global rule should also be present (merged).
        assert "^\\s*#" in cfg.forbidden_patterns.get("diff", [])

    def test_no_arch_no_hardware_guardrails(self, tmp_path):
        """Without arch, hardware-scoped rules should not appear."""
        td = scaffold_task_dir(
            base_dir=str(tmp_path),
            op_name="no_hw",
            task_desc="ref",
            editable_files={"k.py": ""},
            program_md="prog",
            context_files={},
        )
        with patch.object(config_loader, "_guardrails_cache", _FAKE_GUARDRAILS):
            cfg = load_yaml_config(td)
        # Hardware rule must NOT be present.
        assert "\\bforbidden_hw_pattern\\b" not in cfg.forbidden_patterns.get("diff_any", [])
        # Global rule still present.
        assert "^\\s*#" in cfg.forbidden_patterns.get("diff", [])
