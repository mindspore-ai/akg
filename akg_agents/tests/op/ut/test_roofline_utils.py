# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

import math

from akg_agents.op.verifier import roofline_utils


def test_augment_roofline_metrics():
    roofline = {
        "success": True,
        "time_us": 25.0,
    }

    augmented = roofline_utils.augment_roofline_metrics(
        roofline_result=roofline,
        gen_time_us=50.0,
        base_time_us=100.0,
    )

    assert math.isclose(augmented["speedup_vs_generated"], 0.5)
    assert math.isclose(augmented["gap_vs_generated"], 2.0)
    assert math.isclose(augmented["speedup_vs_baseline"], 0.25)


def test_resolve_arch_spec_generates_local_yaml(tmp_path):
    arch_spec = roofline_utils.resolve_arch_spec(
        arch="ascend950pr_9579",
        verify_dir=tmp_path,
    )

    assert arch_spec is not None
    arch_path = tmp_path / "_roofline_arch" / "ascend950_pr.yaml"
    assert arch_path.exists()
    assert arch_spec == str(arch_path)


def test_compute_roofline_profile_sol_uses_geomean(tmp_path, monkeypatch):
    verify_dir = tmp_path
    (verify_dir / "definition.json").write_text("{}", encoding="utf-8")
    (verify_dir / "reference.py").write_text("def run(*args):\n    return args[0]\n", encoding="utf-8")
    (verify_dir / "workload.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    monkeypatch.setattr(roofline_utils, "_import_solar_api", lambda: ({"dummy": True}, None))
    monkeypatch.setattr(
        roofline_utils,
        "resolve_arch_spec",
        lambda arch, verify_dir, explicit_arch_config=None: "Ascend910B4",
    )
    monkeypatch.setattr(
        roofline_utils,
        "_create_solbench_wrapper",
        lambda _verify_dir, wrapper_path, workload_idx: wrapper_path.write_text(
            f"# workload={workload_idx}\n",
            encoding="utf-8",
        ),
    )

    def _fake_single_case(**kwargs):
        case_label = kwargs["case_label"]
        idx = int(case_label[1:])
        if idx == 0:
            return {
                "success": True,
                "case_label": case_label,
                "precision": "fp16",
                "arch_name": "Ascend910B4",
                "time_us": 10.0,
                "compute_time_us": 6.0,
                "memory_time_us": 10.0,
                "bottleneck": "memory",
            }
        return {
            "success": True,
            "case_label": case_label,
            "precision": "fp16",
            "arch_name": "Ascend910B4",
            "time_us": 40.0,
            "compute_time_us": 24.0,
            "memory_time_us": 40.0,
            "bottleneck": "memory",
        }

    monkeypatch.setattr(roofline_utils, "_compute_single_case_roofline", _fake_single_case)

    result = roofline_utils.compute_roofline_profile(
        verify_dir=str(verify_dir),
        op_name="sol_case",
        task_id="task0",
        profile_settings={
            "backend": "ascend",
            "arch": "ascend910b4",
            "framework": "torch",
            "bench_type": "sol",
        },
    )

    assert result["success"] is True
    assert result["workload_count"] == 2
    assert result["case_labels"] == ["w000", "w001"]
    assert math.isclose(result["time_us"], 20.0)
    assert math.isclose(result["compute_time_us"], 12.0)
    assert math.isclose(result["memory_time_us"], 20.0)
    assert result["bottleneck"] == "memory"
