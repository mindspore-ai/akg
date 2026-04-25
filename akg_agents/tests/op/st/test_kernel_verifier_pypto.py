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

"""ST test for PyPTO KernelVerifier using relu resource case."""

import os
import textwrap
import pytest

from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import (
    register_local_worker,
    register_remote_worker,
    get_worker_manager,
)
from akg_agents.utils.environment_check import check_env_for_task
from ..utils import get_device_id


@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.pypto
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.asyncio
async def test_kernel_verifier_pypto_ascend910b4():
    """Verify relu PyPTO code with KernelVerifier directly."""
    framework = "torch"
    dsl = "pypto"
    backend = "ascend"
    arch = "ascend910b4"

    worker_mode = os.getenv("AKG_ST_WORKER_MODE", "local").strip().lower()
    remote_url = os.getenv("AKG_ST_WORKER_URL", "http://127.0.0.1:19001").strip()
    device_id = get_device_id()

    config = load_config(dsl=dsl, backend=backend)
    config["pypto_run_mode"] = 0
    check_env_for_task(framework, backend, dsl, config, is_remote=(worker_mode == "remote"))

    if worker_mode == "remote":
        await register_remote_worker(backend=backend, arch=arch, worker_url=remote_url)
    else:
        await register_local_worker([device_id], backend=backend, arch=arch)

    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}.")

    op_name = "relu"
    task_file = f"./tests/op/resources/{op_name}_op/{op_name}_{framework}.py"
    kernel_file = f"./tests/op/resources/{op_name}_op/{op_name}_{dsl}.py"
    with open(task_file, "r", encoding="utf-8") as f:
        task_desc = f.read()
    with open(kernel_file, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=textwrap.dedent(task_desc),
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name="ModelNew",
        config=config,
        worker=worker,
    )

    result, error_log = await verifier.run({"coder_code": kernel_code}, device_id=device_id)
    assert result, f"KernelVerifier pypto case failed: {error_log}"
