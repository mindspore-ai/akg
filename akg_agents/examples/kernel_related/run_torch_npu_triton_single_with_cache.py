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

"""Direct KernelVerifier demo for Triton Ascend with persistent data cache."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

from akg_agents.core.worker.manager import get_worker_manager, register_local_worker
from akg_agents.op.config.config_validator import load_config
from akg_agents.op.verifier.kernel_verifier import KernelVerifier


ARCH = "ascend910b4"
BACKEND = "ascend"
DEVICE_ID = 0
FRAMEWORK = "torch"
DSL = "triton_ascend"
OP_NAME = "relu"
TASK_ID = "cache_demo"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resource_dir() -> Path:
    return _repo_root() / "tests" / "op" / "resources" / f"{OP_NAME}_op"


def _load_framework_code() -> str:
    return textwrap.dedent((_resource_dir() / f"{OP_NAME}_{FRAMEWORK}.py").read_text(encoding="utf-8"))


def _load_kernel_code() -> str:
    return (_resource_dir() / f"{OP_NAME}_{DSL}_{FRAMEWORK}.py").read_text(encoding="utf-8")


def _build_config() -> dict:
    config = load_config(DSL, backend=BACKEND)
    config["data_cache"] = {
        "enabled": True,
        "cache_dir": "~/.akg/verifier_data_cache",
        "cache_reference_data": True,
        "cache_baseline_result": True,
    }
    return config


def _build_verifier(task_id: str, framework_code: str, worker) -> KernelVerifier:
    return KernelVerifier(
        op_name=OP_NAME,
        framework_code=framework_code,
        task_id=task_id,
        framework=FRAMEWORK,
        dsl=DSL,
        backend=BACKEND,
        arch=ARCH,
        impl_func_name="ModelNew",
        config=_build_config(),
        worker=worker,
    )


async def _run_once(task_id: str, framework_code: str, kernel_code: str, worker):
    verifier = _build_verifier(task_id, framework_code, worker)
    task_info = {"coder_code": kernel_code}

    verify_ok, verify_log = await verifier.run(task_info, current_step=0, device_id=DEVICE_ID)
    if not verify_ok:
        raise RuntimeError(f"verification failed: {verify_log}")

    profile_result = await verifier.run_profile(
        task_info,
        current_step=1,
        device_id=DEVICE_ID,
        profile_settings={
            "warmup_times": 5,
            "run_times": 20,
        },
    )
    return profile_result


async def main():
    framework_code = _load_framework_code()
    kernel_code = _load_kernel_code()

    await register_local_worker([DEVICE_ID], backend=BACKEND, arch=ARCH)
    worker = await get_worker_manager().select(backend=BACKEND, arch=ARCH)
    if not worker:
        raise RuntimeError(f"No available worker for backend={BACKEND}, arch={ARCH}")

    print("=== Run 1: populate verifier data cache ===")
    first = await _run_once(TASK_ID, framework_code, kernel_code, worker)
    print(
        f"Run 1 profile: base={first['base_time']:.2f} us, "
        f"gen={first['gen_time']:.2f} us, speedup={first['speedup']:.4f}x"
    )

    print("\n=== Run 2: reuse local verifier data cache ===")
    second = await _run_once(TASK_ID, framework_code, kernel_code, worker)
    print(
        f"Run 2 profile: base={second['base_time']:.2f} us, "
        f"gen={second['gen_time']:.2f} us, speedup={second['speedup']:.4f}x"
    )

    print("\nCache directory: ~/.akg/verifier_data_cache")
    print("Expected behavior:")
    print("- Run 1 populates reference data cache and baseline cache")
    print("- Run 2 reuses cached reference data and skips base profile")


if __name__ == "__main__":
    asyncio.run(main())
