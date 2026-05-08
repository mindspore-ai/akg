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

import gc
import json
import shutil
from pathlib import Path
from typing import Any

from akg_agents.op.dynamic_tune.deploy.locator import (
    manifest_dir_for_source,
    manifest_dir_for_source_text,
)
from akg_agents.op.dynamic_tune.cases.autotune import _AutotuneSession
from akg_agents.op.dynamic_tune.cases.case import _CaseSpec, _RuntimeBundle
from akg_agents.op.dynamic_tune.cases.device import _maybe_empty_npu_cache
from akg_agents.op.dynamic_tune.cases.report import _render_markdown_report
from akg_agents.op.dynamic_tune.cases.verifier import _KernelVerifierRunner


def _reclaim_npu_memory_before_verify(npu_device: str) -> None:
    """autotune 与 verify 之间再收一轮：tune() 里局部 Module/adapter 可能尚未被 GC。"""
    gc.collect()
    try:
        import torch  # type: ignore
    except ImportError:
        return
    try:
        import torch_npu  # type: ignore

        torch_npu.npu.set_device(_KernelVerifierRunner._device_id_from(npu_device))
    except Exception:
        pass
    _maybe_empty_npu_cache(torch)


def _mirror_manifest_for_verifier_source(
    *,
    impl_path: Path,
    manifest_dir: Path,
    verifier_runner: _KernelVerifierRunner,
) -> Path:
    verifier_source = verifier_runner.verifier_impl_source(
        impl_path.read_text(encoding="utf-8")
    )
    verifier_manifest_dir = manifest_dir_for_source_text(verifier_source)
    if verifier_manifest_dir == manifest_dir:
        return verifier_manifest_dir

    verifier_manifest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(manifest_dir, verifier_manifest_dir, dirs_exist_ok=True)
    print(
        "[manifest] mirror "
        f"source={manifest_dir} "
        f"verifier_source={verifier_manifest_dir}"
    )
    return verifier_manifest_dir


def run_case(
    *,
    case_dir: str | Path,
    impl_path: str | Path,
    npu_device: str,
    cache_dir: Path,
    work_dir: Path,
    artifacts_root: Path,
    artifact_name: str | None = None,
) -> dict[str, Any]:
    # 1. Resolve inputs and prepare directories.
    case_spec = _CaseSpec.from_case_dir(case_dir)
    resolved_impl_path = Path(impl_path).expanduser().resolve()
    if not resolved_impl_path.is_file():
        raise FileNotFoundError(f"impl_path 不存在: {resolved_impl_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    case_artifacts_dir = artifacts_root / str(artifact_name or case_spec.name)
    case_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load the generated implementation and compute its manifest location.
    runtime_bundle = _RuntimeBundle.load(case_spec, resolved_impl_path)
    manifest_cache_dir = manifest_dir_for_source(resolved_impl_path)

    print(
        "[run_case] start "
        f"case={case_spec.name} "
        f"device={npu_device} "
        f"impl={resolved_impl_path} "
        f"artifacts_dir={case_artifacts_dir}"
    )

    # 3. Tune configs and release transient NPU memory before verifier/profile.
    autotune_result = _AutotuneSession(
        runtime_bundle=runtime_bundle,
        npu_device=npu_device,
        cache_dir=manifest_cache_dir,
        work_dir=work_dir,
    ).run()
    _reclaim_npu_memory_before_verify(npu_device)

    # 4. Run verifier/profile against the tuned implementation.
    verifier_runner = _KernelVerifierRunner()
    verifier_manifest_dir = _mirror_manifest_for_verifier_source(
        impl_path=resolved_impl_path,
        manifest_dir=autotune_result.manifest_dir,
        verifier_runner=verifier_runner,
    )
    print(
        "[verify] start "
        f"case={case_spec.name} "
        f"manifest_dir={autotune_result.manifest_dir} "
        f"verifier_manifest_dir={verifier_manifest_dir} "
        f"log_root={autotune_result.log_root}"
    )
    summary = verifier_runner.run(
        case_spec=case_spec,
        impl_code_path=resolved_impl_path,
        npu_device=npu_device,
        log_root=autotune_result.log_root,
        base_module=runtime_bundle.base_module,
        impl_module=runtime_bundle.runtime_module,
    )

    # 5. Attach autotune metadata to the verifier/profile summary.
    summary["manifest_dir"] = str(autotune_result.manifest_dir)
    summary["dynamic_shapes"] = autotune_result.dynamic_shapes
    summary["impl_path"] = str(resolved_impl_path)
    summary["autotune_matrix"] = autotune_result.matrix_summary
    # 把 autotune 时间合并进 timings; verify/profile 的时间由 verifier runner 写入.
    timings = dict(summary.get("timings") or {})
    timings["autotune_seconds"] = autotune_result.tune_seconds
    finite = [v for v in timings.values() if isinstance(v, (int, float))]
    timings["total_seconds"] = sum(finite) if finite else None
    summary["timings"] = timings

    # 6. Persist machine-readable and human-readable artifacts.
    out_path = case_artifacts_dir / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["artifact_path"] = str(out_path)

    report_path = case_artifacts_dir / "report.md"
    try:
        report_path.write_text(
            _render_markdown_report(case_spec=case_spec, summary=summary),
            encoding="utf-8",
        )
        summary["report_path"] = str(report_path)
    except Exception as exc:
        print(f"[report] 渲染 markdown 报告失败 case={case_spec.name}: {exc}")

    # 7. Print compact terminal summary.
    autotune_time = timings.get("autotune_seconds")
    verify_time = timings.get("verify_seconds")
    profile_time = timings.get("profile_seconds")
    total_time = timings.get("total_seconds")
    autotune_text = f"{float(autotune_time):.3f}s" if isinstance(autotune_time, (int, float)) else "n/a"
    verify_text = f"{float(verify_time):.3f}s" if isinstance(verify_time, (int, float)) else "n/a"
    profile_text = f"{float(profile_time):.3f}s" if isinstance(profile_time, (int, float)) else "n/a"
    total_text = f"{float(total_time):.3f}s" if isinstance(total_time, (int, float)) else "n/a"
    print(
        "[timings] "
        f"case={case_spec.name} "
        f"autotune={autotune_text} "
        f"verify={verify_text} "
        f"profile={profile_text} "
        f"total={total_text}"
    )
    print(
        "[run_case] done "
        f"case={case_spec.name} "
        f"verify_passed={summary['verify_passed']} "
        f"artifact={out_path} "
        f"report={report_path}"
    )
    return summary

__all__ = ["run_case"]
