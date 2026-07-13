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

"""Shared CATLASS runtime preparation for verifier and local smoke tests."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from akg_agents.core.worker.eval_config import resolve_eval_timeout
from .catlass_paths import patch_catlass_op_cmake, resolve_catlass_root

_ARCH_TO_CATLASS_ARCH = {
    "ascend910b1": "2201",
    "ascend910b2": "2201",
    "ascend910b2c": "2201",
    "ascend910b3": "2201",
    "ascend910b4": "2201",
    "ascend310p3": "2201",
    "ascend910_9362": "2201",
    "ascend910_9372": "2201",
    "ascend910_9381": "2201",
    "ascend910_9382": "2201",
    "ascend910_9391": "2201",
    "ascend910_9392": "2201",
    "ascend950dt_95a": "3510",
}


def arch_to_catlass_arch(arch: str) -> str:
    """Map KernelVerifier arch strings to CATLASS CMake arch ids."""
    if arch in _ARCH_TO_CATLASS_ARCH:
        return _ARCH_TO_CATLASS_ARCH[arch]
    if arch.startswith("ascend950"):
        return "3510"
    if arch.startswith("ascend910") or arch.startswith("ascend310"):
        return "2201"
    raise ValueError(
        f"Unsupported arch for ascendc_catlass: {arch}. "
        f"Known keys include: {sorted(_ARCH_TO_CATLASS_ARCH.keys())} and ascend950* prefixes."
    )


def ensure_catlass_library(
    task_dir: str,
    *,
    arch: str,
    catlass_root: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """Build ``catlass_op/build/libcatlass.so`` when missing and return its path."""
    timeout = resolve_eval_timeout(timeout)
    import torch as _t
    import torch_npu as _tnp

    if not os.environ.get("ASCEND_HOME_PATH"):
        raise RuntimeError(
            "ASCEND_HOME_PATH is not set. Source the CANN environment before eval."
        )

    resolved_root = resolve_catlass_root(catlass_root=catlass_root)
    if not resolved_root:
        raise RuntimeError(
            "CATLASS_ROOT is not set. Set task.yaml catlass.root, export CATLASS_ROOT, "
            "or install catlass at <akg-root>/thirdparty/catlass via `bash download.sh --with_catlass`."
        )
    resolved_root = os.path.abspath(resolved_root)
    os.environ["CATLASS_ROOT"] = resolved_root

    catlass_op_dir = os.path.join(os.path.abspath(task_dir), "catlass_op")
    if not os.path.isdir(catlass_op_dir):
        raise RuntimeError(f"catlass_op directory not found: {catlass_op_dir}")
    patch_catlass_op_cmake(catlass_op_dir)

    lib_so = os.path.join(catlass_op_dir, "build", "libcatlass.so")
    if os.path.isfile(lib_so):
        return lib_so

    torch_npu_root = os.path.dirname(_tnp.__file__)
    torch_npu_inc = os.path.join(torch_npu_root, "include")
    torch_npu_lib = os.path.join(torch_npu_root, "lib")
    os.environ["CPLUS_INCLUDE_PATH"] = torch_npu_inc + ":" + os.environ.get("CPLUS_INCLUDE_PATH", "")
    os.environ["LIBRARY_PATH"] = torch_npu_lib + ":" + os.environ.get("LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = torch_npu_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    catlass_arch = arch_to_catlass_arch(arch)
    build_dir = os.path.join(catlass_op_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    cmake_shell = (
        f"cd {build_dir} && cmake .. "
        f"-DCMAKE_PREFIX_PATH={_t.utils.cmake_prefix_path} "
        f"-DPython_EXECUTABLE={sys.executable} "
        f"-DPython3_EXECUTABLE={sys.executable} "
        f"-DCATLASS_ROOT={resolved_root} "
        f"-DNPU_ARCH={catlass_arch} -DCATLASS_ARCH={catlass_arch} && make -j1"
    )
    result = subprocess.run(
        ["bash", "-c", cmake_shell],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        details = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise RuntimeError(f"catlass cmake build failed\n{details}".strip())
    if not os.path.isfile(lib_so):
        raise RuntimeError(f"catlass build finished without libcatlass.so: {lib_so}")
    return lib_so
