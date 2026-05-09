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

import argparse
import asyncio
import shutil
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
CACHE_DIR = Path("~/.akg/verifier_data_cache_demo").expanduser()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resource_dir() -> Path:
    return _repo_root() / "tests" / "op" / "resources" / f"{OP_NAME}_op"


def _load_framework_code() -> str:
    return textwrap.dedent((_resource_dir() / f"{OP_NAME}_{FRAMEWORK}.py").read_text(encoding="utf-8"))


def _load_kernel_code() -> str:
    return (_resource_dir() / f"{OP_NAME}_{DSL}_{FRAMEWORK}.py").read_text(encoding="utf-8")


def _build_config(cache_dir: Path) -> dict:
    config = load_config(DSL, backend=BACKEND)
    config["data_cache"] = {
        "enabled": True,
        "cache_dir": str(cache_dir),
        "cache_reference_data": True,
        "cache_baseline_result": True,
    }
    return config


def _build_verifier(task_id: str, framework_code: str, worker, arch: str, cache_dir: Path) -> KernelVerifier:
    return KernelVerifier(
        op_name=OP_NAME,
        framework_code=framework_code,
        task_id=task_id,
        framework=FRAMEWORK,
        dsl=DSL,
        backend=BACKEND,
        arch=arch,
        impl_func_name="ModelNew",
        config=_build_config(cache_dir),
        worker=worker,
    )


async def _run_once(task_id: str, framework_code: str, kernel_code: str, worker, arch: str,
                    cache_dir: Path, device_id: int):
    verifier = _build_verifier(task_id, framework_code, worker, arch, cache_dir)
    task_info = {"coder_code": kernel_code}

    verify_ok, verify_log = await verifier.run(task_info, current_step=0, device_id=device_id)
    if not verify_ok:
        raise RuntimeError(f"verification failed: {verify_log}")

    profile_result = await verifier.run_profile(
        task_info,
        current_step=1,
        device_id=device_id,
        profile_settings={
            "warmup_times": 5,
            "run_times": 20,
        },
    )
    return profile_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Triton Ascend KernelVerifier flow with persistent verifier data cache.",
    )
    parser.add_argument("--device-id", type=int, default=DEVICE_ID, help="Ascend device id to use")
    parser.add_argument("--arch", default=ARCH, help="Ascend architecture name")
    parser.add_argument("--task-id", default=TASK_ID, help="Stable task id used in cache keys")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help="Verifier data cache directory")
    parser.add_argument("--runs", type=int, default=2, help="Number of verify/profile runs")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete the demo cache directory before running to show a miss followed by hits",
    )
    return parser.parse_args()


async def main():
    args = _parse_args()
    cache_dir = Path(args.cache_dir).expanduser()
    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    framework_code = _load_framework_code()
    kernel_code = _load_kernel_code()

    await register_local_worker([args.device_id], backend=BACKEND, arch=args.arch)
    worker = await get_worker_manager().select(backend=BACKEND, arch=args.arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={BACKEND}, arch={args.arch}")

    for run_idx in range(1, max(args.runs, 1) + 1):
        print(f"=== Run {run_idx}: verifier data cache enabled ===")
        result = await _run_once(
            args.task_id,
            framework_code,
            kernel_code,
            worker,
            args.arch,
            cache_dir,
            args.device_id,
        )
        print(
            f"Run {run_idx} profile: base={result['base_time']:.2f} us, "
            f"gen={result['gen_time']:.2f} us, speedup={result['speedup']:.4f}x"
        )

    print(f"\nCache directory: {cache_dir}")
    print("Use --clear-cache for a deterministic miss on the first run.")


if __name__ == "__main__":
    asyncio.run(main())
