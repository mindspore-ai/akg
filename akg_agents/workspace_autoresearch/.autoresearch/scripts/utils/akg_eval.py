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

"""Bridge: sync workspace eval entry → async ``akg_agents`` verifier + worker.

``eval_kernel(task_dir, config, device_id, worker_url)`` reads the current
``kernel.py`` + reference, registers a LocalWorker (or RemoteWorker), runs
``KernelVerifier.run()`` then ``run_profile()``, and returns a dict in the
schema ``phase_machine`` / ``workflow`` already consume (``outcome`` /
``correctness`` / ``metrics{...}`` / ``error`` / ``error_source``).
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring
from __future__ import annotations

import asyncio
import math
import os
import traceback
from typing import Any, Dict, Optional


def eval_kernel(task_dir: str, config, device_id: int = 0,
                worker_url: Optional[str] = None,
                current_step: int = 0) -> Dict[str, Any]:
    """Run akg verify + profile against current kernel.py.

    `worker_url` non-None → RemoteWorker; otherwise LocalWorker on `device_id`.
    `current_step` distinguishes akg log subdirs across rounds.
    """
    return asyncio.run(_eval_async(task_dir, config, device_id, worker_url,
                                   current_step))


def _load_seed_files(task_dir: str, ref_file: str):
    """Return (kernel_code, ref_code) or (None, infra_fail_dict)."""
    kernel_path = os.path.join(task_dir, "kernel.py")
    ref_path = os.path.join(task_dir, ref_file)
    if not os.path.exists(kernel_path):
        return None, _infra_fail(f"kernel.py not found in {task_dir}")
    if not os.path.exists(ref_path):
        return None, _infra_fail(
            f"reference file {ref_file} not found in {task_dir}")
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_code = f.read()
    return (kernel_code, ref_code), None


async def _acquire_worker(backend: str, arch: str, device_id: int,
                          worker_url: Optional[str]):
    """Register + select a worker. Returns (worker, manager) on success,
    or (None, infra_fail_dict) on failure."""
    from akg_agents.core.worker.manager import (
        register_local_worker, register_remote_worker, get_worker_manager,
    )
    try:
        if worker_url:
            await register_remote_worker(backend=backend, arch=arch,
                                         worker_url=worker_url)
        else:
            await register_local_worker(device_ids=[device_id],
                                        backend=backend, arch=arch)
    except Exception as e:
        return None, _infra_fail(f"worker registration failed: {e}")
    wm = get_worker_manager()
    worker = await wm.select(backend=backend, arch=arch)
    if worker is None:
        return None, _infra_fail(
            f"no worker available for backend={backend} arch={arch}")
    return (worker, wm), None


def _build_verifier(task_dir: str, config, ref_code: str, backend: str,
                    arch: str, framework: str, dsl: str, worker):
    from akg_agents.op.verifier.kernel_verifier import KernelVerifier
    log_dir = os.path.join(task_dir, ".ar_state", "akg_verify")
    os.makedirs(log_dir, exist_ok=True)
    return KernelVerifier(
        op_name=config.name,
        task_id=config.name,
        framework_code=ref_code,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        config={
            "log_dir": log_dir,
            "verify_timeout": int(config.eval_timeout or 600),
            "warmup_times": int(config.warmup_times or 10),
            "run_times": int(config.run_times or 100),
        },
        worker=worker,
    )


def _verify_fail_payload(verify_log: Optional[str]) -> Dict[str, Any]:
    return {
        "outcome": "kernel_fail",
        "correctness": False,
        "metrics": {},
        "error": (verify_log or "verify failed")[-2000:],
        "error_source": "kernel",
        "raw_output_tail": (verify_log or "")[-4000:],
    }


def _profile_fail_payload(exc: Exception) -> Dict[str, Any]:
    return {
        "outcome": "kernel_fail",
        "correctness": True,
        "metrics": {},
        "error": f"profile raised: {exc}",
        "error_source": "kernel",
        "raw_output_tail": traceback.format_exc()[-4000:],
    }


def _make_ok_payload(profile_result: dict, op_name: str) -> Dict[str, Any]:
    """Pack profile_result into the workspace's eval-result schema. Returns
    an infra_fail payload when gen_time is missing or non-positive."""
    gen_us = _float(profile_result.get("gen_time"))
    base_us = _float(profile_result.get("base_time"))
    if gen_us is None or gen_us <= 0:
        return _infra_fail(
            "profile returned invalid "
            f"gen_time={profile_result.get('gen_time')!r}"
        )
    speedup = _float(profile_result.get("speedup"))
    if speedup is None and base_us and base_us > 0:
        speedup = base_us / gen_us
    return {
        "outcome": "ok",
        "correctness": True,
        "metrics": {
            "latency_us": gen_us,
            "ref_latency_us": base_us,
            "speedup_vs_ref": speedup,
            "num_cases": 1,
            "per_shape_gen_us": [gen_us],
            "per_shape_gen_method": ["akg_kernel_verifier"],
            "timing_method_gen": "akg_kernel_verifier",
            "per_shape_base_us": [base_us] if base_us else [],
            "per_shape_base_method":
                ["akg_kernel_verifier"] if base_us else [],
            "timing_method_base":
                "akg_kernel_verifier" if base_us else None,
            "per_shape_speedup": [speedup] if speedup else [],
            "speedup_aggregation": "geomean",
            "per_shape_descs": [op_name],
        },
        "error": None,
        "error_source": None,
    }


async def _eval_async(task_dir: str, config, device_id: int,
                      worker_url: Optional[str],
                      current_step: int) -> Dict[str, Any]:
    seed, err = _load_seed_files(task_dir, config.ref_file)
    if err is not None:
        return err
    kernel_code, ref_code = seed

    backend = config.backend or "ascend"
    arch = config.arch or "ascend910b3"
    framework = config.framework or "torch"
    dsl = config.dsl or "triton_ascend"

    acq, err = await _acquire_worker(backend, arch, device_id, worker_url)
    if err is not None:
        return err
    worker, wm = acq

    verifier = _build_verifier(task_dir, config, ref_code, backend, arch,
                               framework, dsl, worker)
    task_info: Dict[str, Any] = {"coder_code": kernel_code}
    try:
        verify_ok, verify_log = await verifier.run(
            task_info, current_step=current_step)
        if not verify_ok:
            return _verify_fail_payload(verify_log)
        try:
            profile_result = await verifier.run_profile(
                task_info, current_step=current_step, profile_settings={},
            )
        except Exception as e:
            return _profile_fail_payload(e)
        return _make_ok_payload(profile_result, config.name)
    finally:
        try:
            await wm.release(worker)
        except Exception:
            pass


def _float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _infra_fail(msg: str) -> Dict[str, Any]:
    return {
        "outcome": "infra_fail",
        "correctness": False,
        "metrics": {},
        "error": msg,
        "error_source": None,
    }
