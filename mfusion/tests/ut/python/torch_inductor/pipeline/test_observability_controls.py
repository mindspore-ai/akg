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

"""UTs for IR observability controls in torch inductor pipeline."""
# pylint: disable=protected-access

from pathlib import Path

from mfusion.torch import inductor as inductor_mod
from mfusion.torch._pipeline import PipelineRunner, _parse_mlir_module_from_text


_MIN_TORCH_MLIR = """
module {
  func.func @main(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
    return %arg0 : !torch.vtensor<[2],f32>
  }
}
"""


class _FakeRunner:
    def __init__(self, verbose: bool):
        self.enabled_verbose_internal_ir = verbose
        self.calls = []

    def run(self, pipeline: str, stage: str):
        self.calls.append((pipeline, stage))


def test_load_internal_passes_from_cpp_parses_expected_entries(tmp_path: Path):
    """Parse pass table entries from C++ source text."""
    cpp_rel = "lib/Conversion/TorchToMfuse/TorchFusion.cc"
    cpp_path = tmp_path / cpp_rel
    cpp_path.parent.mkdir(parents=True, exist_ok=True)
    cpp_path.write_text(
        """
std::vector<std::pair<const char *, PassCreator>> passes = {
  {"torch-fuse-rms-norm", []() { return createTorchFuseRmsNormPass(); }},
  {"torch-fuse-rope", []() { return createTorchFuseRoPEPass(); }},
};
""",
        encoding="utf-8",
    )

    loaded = inductor_mod._load_internal_passes_from_cpp(
        cpp_rel,
        base_dir=tmp_path,
    )

    assert loaded == ("torch-fuse-rms-norm", "torch-fuse-rope")


def test_load_internal_passes_from_cpp_returns_empty_when_file_missing(tmp_path: Path):
    """Return empty tuple when C++ source is unavailable."""
    loaded = inductor_mod._load_internal_passes_from_cpp(
        "lib/Dialect/Mfuse/Transforms/Fusion/MfuseFusion.cc",
        base_dir=tmp_path,
    )
    assert loaded == ()


def test_run_composite_fusion_stage_expands_internal_passes_when_verbose():
    """Expand composite stage into per-pass runs at verbose level."""
    runner = _FakeRunner(verbose=True)
    inductor_mod._run_composite_fusion_stage(
        runner=runner,
        stage_label="Mfuse Fusion",
        composite_pipeline="builtin.module(mfuse-fusion,canonicalize)",
        internal_passes=("fuse-a", "fuse-b"),
    )
    assert runner.calls == [
        ("builtin.module(fuse-a)", "Mfuse Fusion / fuse-a"),
        ("builtin.module(fuse-b)", "Mfuse Fusion / fuse-b"),
        ("builtin.module(canonicalize)", "Mfuse Fusion / canonicalize"),
    ]


def test_run_composite_fusion_stage_uses_composite_pipeline_when_not_verbose():
    """Keep one-shot composite pipeline when verbose mode is disabled."""
    runner = _FakeRunner(verbose=False)
    inductor_mod._run_composite_fusion_stage(
        runner=runner,
        stage_label="Torch Fusion",
        composite_pipeline="builtin.module(torch-fusion,canonicalize)",
        internal_passes=("torch-fuse-rope",),
    )
    assert runner.calls == [("builtin.module(torch-fusion,canonicalize)", "Torch Fusion")]


def test_pipeline_runner_env_levels_control_observability(monkeypatch):
    """Enable internal observability when either print/save level is 2."""
    monkeypatch.setenv("MFUSION_PRINT_IR", "1")
    monkeypatch.setenv("MFUSION_SAVE_IR", "2")

    module = _parse_mlir_module_from_text(_MIN_TORCH_MLIR)
    runner = PipelineRunner(module)

    assert runner.enabled_print_ir is True
    assert runner.enabled_save_ir is True
    assert runner.enabled_verbose_internal_ir is True


def test_pipeline_runner_level_one_keeps_stage_only_mode(monkeypatch):
    """Keep stage-level mode when only print level is set to 1."""
    monkeypatch.setenv("MFUSION_PRINT_IR", "1")
    monkeypatch.delenv("MFUSION_SAVE_IR", raising=False)

    module = _parse_mlir_module_from_text(_MIN_TORCH_MLIR)
    runner = PipelineRunner(module)

    assert runner.enabled_print_ir is True
    assert runner.enabled_save_ir is False
    assert runner.enabled_verbose_internal_ir is False
