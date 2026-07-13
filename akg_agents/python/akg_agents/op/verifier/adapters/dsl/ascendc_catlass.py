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

"""AscendC + CATLASS pybind DSL adapter for KernelBench / AR verify."""

from __future__ import annotations

import logging
import os
import shutil
import textwrap
from typing import Any, Dict, Optional

from akg_agents.core.worker.eval_config import resolve_eval_timeout
from akg_agents.op.utils.catlass_runtime import arch_to_catlass_arch
from .base import DSLAdapter

logger = logging.getLogger(__name__)

# KernelVerifier arch → catlass cmake arch id (2201 / 3510).
# Pass BOTH -DNPU_ARCH and -DCATLASS_ARCH: pipeline CMakeLists vary
# (option(NPU_ARCH) vs if(NOT DEFINED CATLASS_ARCH)).
arch_to_npu_arch = arch_to_catlass_arch


class DSLAdapterAscendC_Catlass(DSLAdapter):
    """CATLASS pybind + ModelNew wrapper on Ascend NPU."""

    impl_func_name_template = "ModelNew"
    uses_cannbench_precision = True

    def get_import_statements(self, framework: str) -> str:
        return "import torch\nimport torch_npu\n"

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        return "from kernel import ModelNew\n"

    def create_impl_module(
        self,
        framework: str,
        framework_adapter: Any,
        init_params_var: str = "init_params",
        device_var: str = "device",
    ) -> str:
        code = f"impl_model = ModelNew(*{init_params_var})\n"
        if framework == "torch":
            code += f"impl_model = impl_model.to({device_var})\n"
        code += "impl_model.eval()\n"
        return code

    def call_impl(
        self,
        impl_func_name: str,
        inputs: str,
        device_id: int,
        framework_adapter: Any,
        op_name: str,
        data_dir: Optional[str] = None,
        framework_output: Optional[str] = None,
    ) -> str:
        # Runtime anti-cheat, same as ascendc: run the candidate under
        # compute_gate so core-compute delegation (Python / C++ torch::* / at::*
        # nested in the candidate's own catlass op) is disabled at the dispatch
        # layer for this forward. Static CodeChecker is the pre-flight for raw
        # aclnn* / torch_npu.npu_* (which never reach dispatch).
        return (
            "from akg_agents.op.utils.code_checker.runtime_guard import "
            "guarded_call as _akg_guarded_call\n"
            f"impl_output = _akg_guarded_call(lambda: impl_model(*{inputs}))\n"
        )

    # catlass kernel handoff is a directory: the catlass_op/ project
    # subtree sitting next to a Python wrapper (kernel.py).
    kernel_arg_is_directory = True
    kernel_project_dir_name = "catlass_op"
    kernel_project_files = [
        "catlass_op/kernel/catlass_kernel.asc",
        "catlass_op/include/catlass_kernel.h",
        "catlass_op/src/catlass_torch.cpp",
        "catlass_op/CMakeLists.txt",
    ]

    def benchmark_impl(
        self,
        impl_func_name: str,
        inputs: str,
        warmup: int,
        runs: int,
        backend: str,
        op_name: str,
        case_idx: int = 0,
        framework_model: Optional[str] = None,
        framework_adapter: Optional[Any] = None,
        device_id: Optional[int] = None,
        clear_l2_cache: bool = False,
        framework: str = "torch",
    ) -> str:
        """Profile via torch_npu.profiler (same path as triton_ascend / desktop bench)."""
        framework_arg = f', framework="{framework}"' if framework == "mindspore" else ""
        if backend == "ascend":
            return textwrap.dedent(
                f"""
                try:
                    from akg_agents.op.verifier.profiler import profiler_npu
                    patch_imported = True
                except ImportError:
                    patch_imported = False

                def catlass_benchmark_fn():
                    return impl_model(*{inputs})

                if patch_imported:
                    execution_time_us = profiler_npu(
                        catlass_benchmark_fn,
                        warmup={warmup},
                        active={runs},
                        prof_dir_name=f"prof_generation_output_case_{{case_idx}}",
                        keep_res=False,
                        suppress_warnings=True,
                        clear_l2_cache={clear_l2_cache},
                        dsl="other"{framework_arg}
                    )
                    execution_time_ms = execution_time_us / 1000
                    method = "profiler_npu"
                else:
                    import time
                    start_time = time.time()
                    for _ in range({warmup + runs}):
                        _ = catlass_benchmark_fn()
                        torch.npu.synchronize()
                    end_time = time.time()
                    execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}
                    method = "traditional_timing"
                """
            )
        sync_code = "torch.cuda.synchronize()" if backend == "cuda" else "pass"
        return textwrap.dedent(
            f"""
            import time
            def catlass_benchmark_fn():
                return impl_model(*{inputs})
            start_time = time.time()
            for _ in range({warmup + runs}):
                _ = catlass_benchmark_fn()
                {sync_code}
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}
            method = "traditional_timing"
            """
        )

    def get_special_setup_code(self, framework: str = "torch") -> str:
        # arch + catlass_root resolved at prepare_config() time into
        # self._setup_arch / self._setup_catlass_root. Fall back to
        # the ascend910b3 / env-driven defaults if prepare_config
        # was bypassed (e.g. unit tests).
        arch = getattr(self, "_setup_arch", None)
        if not arch:
            raise RuntimeError(
                "ascendc_catlass requires config['arch'] before special setup runs"
            )
        catlass_root = getattr(self, "_setup_catlass_root", None)
        catlass_arch = arch_to_catlass_arch(arch)
        catlass_root_repr = repr(catlass_root) if catlass_root else "None"
        timeout = resolve_eval_timeout()
        return textwrap.dedent(
            f"""
        # --- ascendc_catlass rebuild ---
        import subprocess
        import os
        import sys
        import torch as _t
        import torch_npu as _tnp
        # cmake invocation needs to find: (1) the active env's torch
        # (CMAKE_PREFIX_PATH), (2) the active env's Python — base
        # miniconda's Python 3.13 vs yyz env's 3.10 → undefined-symbol
        # at load time if cmake picks wrong (Python_EXECUTABLE), (3)
        # torch_npu headers/libs referenced by catlass_torch.cpp
        # (CPLUS_INCLUDE_PATH / LIBRARY_PATH).
        _torch_cmake = _t.utils.cmake_prefix_path
        _torch_npu_root = os.path.dirname(_tnp.__file__)
        _torch_npu_inc = os.path.join(_torch_npu_root, "include")
        _torch_npu_lib = os.path.join(_torch_npu_root, "lib")
        os.environ["CPLUS_INCLUDE_PATH"] = _torch_npu_inc + ":" + os.environ.get("CPLUS_INCLUDE_PATH", "")
        os.environ["LIBRARY_PATH"] = _torch_npu_lib + ":" + os.environ.get("LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _torch_npu_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

        if not os.environ.get("ASCEND_HOME_PATH"):
            raise RuntimeError(
                "ASCEND_HOME_PATH is not set. Source the CANN environment before eval."
            )

        try:
            from akg_agents.op.utils.catlass_paths import resolve_catlass_root as _resolve_catlass_root
        except Exception:
            _resolve_catlass_root = None
        _catlass_root = None
        if _resolve_catlass_root is not None:
            _catlass_root = _resolve_catlass_root(catlass_root={catlass_root_repr})
        if not _catlass_root:
            _catlass_root = {catlass_root_repr} or os.environ.get("CATLASS_ROOT")
        if not _catlass_root:
            raise RuntimeError(
                "CATLASS_ROOT is not set. Set task.yaml catlass.root, config catlass_root, "
                "export CATLASS_ROOT, or install catlass at <akg-root>/thirdparty/catlass "
                "via `bash download.sh --with_catlass` before eval."
            )
        _catlass_root = os.path.abspath(_catlass_root)
        os.environ["CATLASS_ROOT"] = _catlass_root

        _catlass_arch = "{catlass_arch}"
        catlass_op_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catlass_op")
        if not os.path.isdir(catlass_op_dir):
            raise RuntimeError(f"catlass_op directory not found: {{catlass_op_dir}}")
        _lib_so = os.path.join(catlass_op_dir, "build", "libcatlass.so")
        if os.path.isfile(_lib_so):
            print(f"[INFO]: catlass using existing build: {{_lib_so}}")
        else:
            build_dir = os.path.join(catlass_op_dir, "build")
            os.makedirs(build_dir, exist_ok=True)
            _cmake_shell = (
                f"cd {{build_dir}} && cmake .. "
                f"-DCMAKE_PREFIX_PATH={{_torch_cmake}} "
                f"-DPython_EXECUTABLE={{sys.executable}} "
                f"-DPython3_EXECUTABLE={{sys.executable}} "
                f"-DCATLASS_ROOT={{_catlass_root}} "
                f"-DNPU_ARCH={{_catlass_arch}} -DCATLASS_ARCH={{_catlass_arch}} && make -j1"
            )
            result = subprocess.run(
                ["bash", "-c", _cmake_shell],
                capture_output=True,
                text=True,
                timeout={timeout},
            )
            if result.returncode != 0:
                print("[ERROR]: catlass build failed!")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                raise RuntimeError("catlass cmake build failed")
            print("[INFO]: catlass build successful")
        """
        )

    # ------------------------------------------------------------------
    # Extension hooks (override DSLAdapter defaults)
    # ------------------------------------------------------------------

    def materialize_impl(self, impl_code: str, verify_dir: str,
                         op_name: str, framework: str,
                         dsl_name: str,
                         task_info: Optional[Dict[str, Any]] = None,
                         config: Optional[Dict[str, Any]] = None) -> None:
        """Write the primary wrapper + copy catlass_op tree into verify_dir."""
        kernel_file = os.path.join(
            verify_dir, self.entry_filename_template.format(op_name=op_name))
        with open(kernel_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        from akg_agents.op.utils.catlass_paths import merge_catlass_config
        cfg = dict(config) if config else {}
        merge_catlass_config(cfg, task_info=task_info)
        catlass_op_src = cfg.get("catlass_op_src")
        if not catlass_op_src or not os.path.isdir(catlass_op_src):
            raise ValueError(
                f"[{op_name}] catlass_op_src not found or not a directory. "
                f"Set config catlass_op_src / task_dir + catlass_op/, or task.yaml "
                f"catlass.op_dir. Got: {catlass_op_src!r}"
            )
        catlass_op_dst = os.path.join(verify_dir, "catlass_op")
        if os.path.isdir(catlass_op_dst):
            shutil.rmtree(catlass_op_dst)
        shutil.copytree(
            catlass_op_src,
            catlass_op_dst,
            ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc", "*.so"),
        )
        logger.debug("[%s] catlass_op copied: %s -> %s",
                     op_name, catlass_op_src, catlass_op_dst)

    def expected_artifacts(self, verify_dir: str, op_name: str,
                           framework: str, bench_type: str,
                           dsl_filename_hint: str) -> list:
        wrapper_name = self.entry_filename_template.format(op_name=op_name)
        return [
            os.path.join(verify_dir, wrapper_name),
            os.path.join(verify_dir, "catlass_op", "CMakeLists.txt"),
        ]

    def prepare_config(self, config: Dict[str, Any],
                       task_info: Optional[Dict[str, Any]] = None) -> None:
        """Resolve CATLASS_ROOT + catlass_op_src into config, and remember
        arch / catlass_root for get_special_setup_code (which has the
        ABC-fixed signature ``(framework)``)."""
        from akg_agents.op.utils.catlass_paths import merge_catlass_config
        merge_catlass_config(config, task_info=task_info)
        # Stash for get_special_setup_code; not a config key so that
        # cross-DSL config inspection stays clean.
        self._setup_arch = config.get("arch")
        self._setup_catlass_root = config.get("catlass_root")

    benchmark_requires_l2_clear = False
    profile_via_python_script = True

    def post_iteration_cleanup(self, verify_dir: str) -> None:
        """Drop catlass_op/build for this round; keep profile JSON + sources."""
        build_dir = os.path.join(verify_dir, "catlass_op", "build")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)

    def read_kernel_source(self, kernel_arg: str,
                           op_name: Optional[str] = None) -> tuple:
        """``kernel_arg`` is the catlass_op directory; the Python wrapper
        is the sibling primary editable (per adapter's
        ``entry_filename_template``) — or ``<op_name>_kernel.py`` when
        the canonical name isn't there: KernelBench dumps name the
        wrapper ``<op>_kernel.py``."""
        if not os.path.isdir(kernel_arg):
            raise FileNotFoundError(
                "ascendc_catlass kernel handoff must be a catlass_op "
                f"directory; got {kernel_arg!r}"
            )
        parent = os.path.dirname(kernel_arg)
        canonical = self.entry_filename_template.format(op_name=op_name or "")
        candidates = [canonical]
        if op_name:
            candidates.append(f"{op_name}_kernel.py")
        for name in candidates:
            sibling = os.path.join(parent, name)
            if os.path.isfile(sibling):
                with open(sibling, "r", encoding="utf-8") as f:
                    return f.read(), os.path.abspath(kernel_arg)
        raise FileNotFoundError(
            "ascendc_catlass kernel handoff is a directory; expected sibling "
            f"{' or '.join(candidates)} at {parent}"
        )

    def materialize_project_tree(self, dst_dir: str,
                                 project_src: Optional[str],
                                 project_dir_name: Optional[str] = None) -> None:
        """Copy catlass_op tree into ``dst_dir`` and patch its
        CMakeLists.txt for the AR task layout."""
        if not project_src:
            return
        from akg_agents.op.utils.catlass_paths import patch_catlass_op_cmake
        dst = os.path.join(dst_dir, project_dir_name or self.kernel_project_dir_name)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(
            project_src,
            dst,
            ignore=shutil.ignore_patterns("build", "__pycache__", "*.so"),
        )
        patch_catlass_op_cmake(dst)
