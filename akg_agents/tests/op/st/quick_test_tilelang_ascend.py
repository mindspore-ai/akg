#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Quick standalone test for tilelang_ascend DSL adapter + KernelVerifier."""

import asyncio
import textwrap
from pathlib import Path

from akg_agents.op.utils.config_utils import normalize_dsl, check_dsl
from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.common_utils import create_log_dir
from akg_agents.core.worker.manager import register_local_worker, get_worker_manager

DEVICE_ID = 0


def test_dsl_registration():
    """Step 1: Verify DSL is registered."""
    print("=" * 60)
    print("Step 1: DSL Registration Check")
    print("=" * 60)

    dsl = normalize_dsl("tilelang_ascend", backend="ascend")
    assert dsl == "tilelang_ascend", f"normalize_dsl failed: {dsl}"
    print(f"  normalize_dsl('tilelang_ascend') -> '{dsl}' OK")

    try:
        check_dsl("tilelang_ascend")
        print("  check_dsl('tilelang_ascend') OK")
    except ValueError as e:
        print(f"  check_dsl FAILED: {e}")
        return False

    return True


def test_adapter():
    """Step 2: Verify DSL adapter works."""
    print("=" * 60)
    print("Step 2: DSL Adapter Check")
    print("=" * 60)

    adapter = get_dsl_adapter("tilelang_ascend")
    assert adapter is not None, "get_dsl_adapter returned None"
    print(f"  Adapter class: {adapter.__class__.__name__} OK")

    imports = adapter.get_import_statements("torch")
    assert "tilelang" in imports, f"Missing 'tilelang' in imports: {imports[:200]}"
    print("  get_import_statements('torch') OK")

    impl_import = adapter.get_impl_import("test_op", "ModelNew")
    assert "import ModelNew" in impl_import, f"Wrong impl_import: {impl_import}"
    print(f"  get_impl_import('test_op') -> '{impl_import.strip()}' OK")

    return True


async def test_kernel_verifier():
    """Step 3: Run KernelVerifier for relu with tilelang_ascend."""
    print("=" * 60)
    print("Step 3: KernelVerifier End-to-End Test")
    print("=" * 60)

    op_name = "relu"
    framework = "torch"
    dsl = "tilelang_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    config = load_config(dsl, backend=backend)
    print(f"  Config loaded OK")

    tests_dir = Path(__file__).resolve().parents[3] / "tests" / "op" / "resources"
    op_task_file = tests_dir / f"{op_name}_op" / f"{op_name}_{framework}.py"
    kernel_file = tests_dir / f"{op_name}_op" / f"{op_name}_{dsl}_{framework}.py"

    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())
    print(f"  Framework code loaded from {op_task_file}")

    with open(kernel_file, "r", encoding="utf-8") as f:
        kernel_code = f.read()
    print(f"  Kernel code loaded from {kernel_file}")

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_quick_test')
    print(f"  log_dir: {log_dir}")

    await register_local_worker([DEVICE_ID], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}")
    print(f"  Worker registered OK")

    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name="ModelNew",
        config=config,
        worker=worker
    )
    print(f"  KernelVerifier created OK")
    print(f"  impl_func_name: {verifier.impl_func_name}")

    task_info = {"coder_code": kernel_code}
    print(f"  Running verifier...")
    result, error_log = await verifier.run(task_info, device_id=DEVICE_ID)

    if result:
        print(f"  ✅ VERIFICATION PASSED!")
    else:
        print(f"  ❌ VERIFICATION FAILED!")
        print(f"  Error log: {error_log[:500]}")

    return result


def main():
    print("TileLang-Ascend AKG Agents Quick Test")
    print()

    if not test_dsl_registration():
        print("\n❌ Step 1 FAILED - fix DSL registration first")
        return False

    if not test_adapter():
        print("\n❌ Step 2 FAILED - fix adapter first")
        return False

    result = asyncio.run(test_kernel_verifier())
    print()
    if result:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Check logs above.")
    return result


if __name__ == "__main__":
    main()
