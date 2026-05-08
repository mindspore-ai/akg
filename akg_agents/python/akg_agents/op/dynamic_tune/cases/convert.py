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

from __future__ import annotations

from datetime import datetime
import shlex
import shutil
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OPENCODE_WORKSPACE = (_PROJECT_ROOT / "workspace").resolve()
POSTPROCESS_COMMAND = "opencode run --agent dynamic_tune"
POSTPROCESS_PROMPT = "Use the akg-modelnew-postprocess skill to patch impl.py in place."


def _make_stage_dir(stage_root: Path, stage_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stage_dir = stage_root / f"{stage_name}-{timestamp}"
    for index in range(100):
        candidate = stage_dir if index == 0 else stage_root / f"{stage_name}-{timestamp}-{index:02d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"无法创建 staging 目录: {stage_dir}")


def _stage_case(
    *,
    stage_name: str,
    raw_impl_path: Path,
    base_path: Path,
    sample_path: Path,
) -> tuple[Path, Path]:
    workspace = DEFAULT_OPENCODE_WORKSPACE
    if not workspace.is_dir():
        raise RuntimeError(f"opencode workspace 不存在: {workspace}")

    stage_root = workspace / ".tmp" / "txa_postprocess"
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_dir = _make_stage_dir(stage_root, stage_name)

    shutil.copyfile(raw_impl_path, stage_dir / "impl.py")
    shutil.copyfile(base_path, stage_dir / "base.py")
    shutil.copyfile(sample_path, stage_dir / "sample.json")
    return workspace, stage_dir


def postprocess_case(
    *,
    case_name: str,
    raw_impl_path: Path,
    base_path: Path,
    sample_path: Path,
    output_path: Path,
) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    workspace, stage_dir = _stage_case(
        stage_name=case_name,
        raw_impl_path=raw_impl_path,
        base_path=base_path,
        sample_path=sample_path,
    )
    print(
        f"[convert] case={case_name} mode=opencode workspace={workspace} "
        f"stage={stage_dir} output={output_path} cmd={POSTPROCESS_COMMAND}"
    )
    subprocess.run(
        shlex.split(POSTPROCESS_COMMAND) + [POSTPROCESS_PROMPT],
        check=True,
        cwd=str(stage_dir),
    )

    staged_output = stage_dir / "impl.py"
    if not staged_output.is_file():
        raise RuntimeError("opencode 未保留 `impl.py`")
    shutil.copyfile(staged_output, output_path)
    return {
        "mode": "opencode",
        "impl_path": str(output_path),
        "raw_impl_path": str(raw_impl_path),
    }


__all__ = ["postprocess_case"]
