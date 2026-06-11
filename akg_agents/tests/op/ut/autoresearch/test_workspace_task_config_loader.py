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

from pathlib import Path
import sys

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "workspace_autoresearch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from task_config.loader import load_task_config  # noqa: E402


def _write_task_yaml(task_dir: Path, extra: dict | None = None) -> None:
    data = {
        "name": "case",
        "editable_files": ["kernel.py"],
    }
    data.update(extra or {})
    (task_dir / "task.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def test_loader_does_not_emit_catlass_defaults_without_block(tmp_path):
    _write_task_yaml(tmp_path)

    cfg = load_task_config(str(tmp_path))

    assert cfg is not None
    assert cfg.dsl_config == {}


def test_loader_flattens_explicit_catlass_block(tmp_path):
    _write_task_yaml(
        tmp_path,
        {
            "catlass": {
                "root": "/opt/catlass",
                "op_dir": "custom_catlass_op",
            },
        },
    )

    cfg = load_task_config(str(tmp_path))

    assert cfg.dsl_config == {
        "catlass_root": "/opt/catlass",
        "catlass_op_dir": "custom_catlass_op",
    }


def test_loader_flattens_explicit_ascendc_block(tmp_path):
    _write_task_yaml(
        tmp_path,
        {
            "ascendc": {
                "op_dir": "custom_ascendc_op",
            },
        },
    )

    cfg = load_task_config(str(tmp_path))

    assert cfg.dsl_config == {"ascendc_op_dir": "custom_ascendc_op"}


def test_loader_defaults_only_present_dsl_blocks(tmp_path):
    _write_task_yaml(tmp_path, {"catlass": {}, "ascendc": {}})

    cfg = load_task_config(str(tmp_path))

    assert cfg.dsl_config == {
        "catlass_op_dir": "catlass_op",
        "ascendc_op_dir": "ascendc_op",
    }
