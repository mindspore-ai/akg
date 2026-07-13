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

"""AscendC direct-invoke DSL adapter.

The canonical ``ascendc`` format is now a CANNBot-style project tree next
to a small Python ``ModelNew`` wrapper:

    kernel.py
    ascendc_op/
        CMakeLists.txt
        op_kernel/
        op_host/
        op_extension/
        scripts/

The wrapper is responsible for loading the built shared object and calling
``torch.ops.npu.<op>(...)``.  The adapter owns project-tree copy, CMake
rebuild, arch patching, and KernelVerifier integration.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

from akg_agents.core.worker.eval_config import resolve_eval_timeout
from akg_agents.op.utils.arch_normalize import ascend_direct_invoke_npu_arch
from .base import DSLAdapter

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = {
    ".asc",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
_TEXT_FILENAMES = {"CMakeLists.txt", "CMakePresets.json", "run.sh"}
_IGNORE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "CMakeFiles",
    "output",
}
_EDITABLE_PROJECT_ROOTS = {
    "CMakeLists.txt",
    "CMakePresets.json",
    "cmake",
    "op_kernel",
    "op_extension",
    "op_host",
    "src",
    "include",
    "common",
}
_CMAKE_NPU_ARCH_FLAG_RE = re.compile(r"--npu-arch=[A-Za-z0-9_-]+")
_CMAKE_NPU_ARCH_VAR_RE = re.compile(r"\b(?:NPU_ARCH|ASCENDC_NPU_ARCH)\b")
_CMAKE_TEXT_SUFFIXES = {".cmake", ".txt"}


def _copy_project_tree(src: str, dst: str) -> None:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "build",
            "CMakeFiles",
            "output",
            "*.o",
            "*.pyc",
            "*.so",
        ),
    )


def _patch_cmake_npu_arch(project_dir: str, npu_arch: str) -> bool:
    """Rewrite literal ``--npu-arch=...`` flags in a copied project."""
    cmake_path = os.path.join(os.path.abspath(project_dir), "CMakeLists.txt")
    if not os.path.isfile(cmake_path):
        return False
    with open(cmake_path, "r", encoding="utf-8") as f:
        text = f.read()
    patched, count = _CMAKE_NPU_ARCH_FLAG_RE.subn(f"--npu-arch={npu_arch}", text)
    if count:
        with open(cmake_path, "w", encoding="utf-8") as f:
            f.write(patched)
    return bool(count)


def _iter_cmake_text_files(project_dir: str):
    root = Path(project_dir).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _IGNORE_DIRS for part in rel_parts):
            continue
        if path.name == "CMakeLists.txt" or path.suffix in _CMAKE_TEXT_SUFFIXES:
            yield path


def _project_consumes_cmake_npu_arch_vars(project_dir: str) -> bool:
    """Return whether project CMake files reference AKG's arch variables."""
    for path in _iter_cmake_text_files(project_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _CMAKE_NPU_ARCH_VAR_RE.search(text):
            return True
    return False


def _assert_cmake_has_npu_arch_channel(project_dir: str, op_name: str) -> None:
    """Fail before build when AKG cannot steer direct-invoke arch.

    A project is acceptable when either:
    - it has a literal ``--npu-arch=...`` flag for us to patch, or
    - its CMake files consume ``NPU_ARCH`` / ``ASCENDC_NPU_ARCH`` passed by
      the verifier runtime.
    """
    cmake_path = os.path.join(os.path.abspath(project_dir), "CMakeLists.txt")
    if not os.path.isfile(cmake_path):
        return
    with open(cmake_path, "r", encoding="utf-8") as f:
        text = f.read()
    if _CMAKE_NPU_ARCH_FLAG_RE.search(text):
        return
    if _project_consumes_cmake_npu_arch_vars(project_dir):
        return
    raise ValueError(
        f"[{op_name}] ascendc CMake has no controllable NPU arch channel. "
        "Add a literal --npu-arch=<dav-token> flag to CMakeLists.txt, or "
        "consume ${NPU_ARCH} / ${ASCENDC_NPU_ARCH} in the project CMake files."
    )


class DSLAdapterAscendC(DSLAdapter):
    """CANNBot-style AscendC direct-invoke project adapter."""

    impl_func_name_template = "ModelNew"
    profile_via_python_script = True
    benchmark_requires_l2_clear = False
    uses_cannbench_precision = True

    kernel_arg_is_directory = True
    kernel_project_dir_name = "ascendc_op"
    kernel_project_files = ["ascendc_op/CMakeLists.txt"]

    def get_import_statements(self, framework: str) -> str:
        if framework != "torch":
            raise ValueError(
                f"ascendc direct-invoke currently supports torch only, got {framework!r}"
            )
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
        # Runtime anti-cheat: run the candidate under compute_gate (block by
        # default; AKG_GUARD_MODE overrides), which DISABLES the core-compute
        # leaves + bulk D2H at the dispatch layer for this forward. Delegation via
        # Python torch.matmul, C++ torch::matmul, or nesting inside the candidate's
        # own custom op all raise — no watcher blind spot. Static CodeChecker stays
        # the pre-flight for raw aclnn* / torch_npu.npu_* (which never reach
        # dispatch). Emitted as a local import + single expression so it is
        # indentation neutral in the verify script.
        return (
            "from akg_agents.op.utils.code_checker.runtime_guard import "
            "guarded_call as _akg_guarded_call\n"
            f"impl_output = _akg_guarded_call(lambda: impl_model(*{inputs}))\n"
        )

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
        framework: str = "torch",
    ) -> str:
        framework_arg = f', framework="{framework}"' if framework == "mindspore" else ""
        if backend == "ascend":
            return textwrap.dedent(
                f"""
                try:
                    from akg_agents.op.verifier.profiler import profiler_npu
                    patch_imported = True
                except ImportError:
                    patch_imported = False

                def ascendc_benchmark_fn():
                    return impl_model(*{inputs})

                if patch_imported:
                    execution_time_us = profiler_npu(
                        ascendc_benchmark_fn,
                        warmup={warmup},
                        active={runs},
                        prof_dir_name=f"prof_generation_output_case_{{case_idx}}",
                        keep_res=False,
                        suppress_warnings=True,
                        clear_l2_cache=False,
                        dsl="other"{framework_arg},
                    )
                    execution_time_ms = execution_time_us / 1000
                    method = "profiler_npu"
                else:
                    import time
                    start_time = time.time()
                    for _ in range({warmup + runs}):
                        _ = ascendc_benchmark_fn()
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
            def ascendc_benchmark_fn():
                return impl_model(*{inputs})
            start_time = time.time()
            for _ in range({warmup + runs}):
                _ = ascendc_benchmark_fn()
                {sync_code}
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}
            method = "traditional_timing"
            """
        )

    def prepare_config(
        self,
        config: Dict[str, Any],
        task_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        task_info = task_info or {}
        project_dir_name = (
            config.get("ascendc_op_dir")
            or config.get("ascendc_project_dir")
            or task_info.get("ascendc_op_dir")
            or task_info.get("ascendc_project_dir")
            or self.kernel_project_dir_name
        )
        config["ascendc_op_dir"] = project_dir_name

        project_src = (
            config.get("ascendc_op_src")
            or config.get("ascendc_project_src")
            or task_info.get("ascendc_op_src")
            or task_info.get("ascendc_project_src")
            or task_info.get("kernel_project_src")
        )
        task_dir = task_info.get("task_dir")
        if not project_src and task_dir:
            project_src = os.path.join(task_dir, project_dir_name)
        if project_src:
            config["ascendc_op_src"] = os.path.abspath(project_src)

        arch = config.get("arch")
        self._setup_arch = arch
        self._setup_npu_arch = ascend_direct_invoke_npu_arch(arch or "")
        self._setup_timeout = int(
            config.get("verify_timeout")
            or config.get("timeout")
            or resolve_eval_timeout()
        )
        self._setup_project_dir_name = project_dir_name

    def get_special_setup_code(self, framework: str = "torch") -> str:
        arch = getattr(self, "_setup_arch", None)
        npu_arch = getattr(self, "_setup_npu_arch", None)
        if not arch:
            raise RuntimeError(
                "ascendc requires config['arch'] before special setup runs"
            )
        if not npu_arch:
            raise RuntimeError(
                f"ascendc cannot derive direct-invoke --npu-arch from {arch!r}"
            )
        project_dir_name = getattr(
            self, "_setup_project_dir_name", self.kernel_project_dir_name
        )
        timeout = int(getattr(self, "_setup_timeout", resolve_eval_timeout()))
        return textwrap.dedent(
            f"""
            # --- ascendc direct-invoke rebuild ---
            import os
            import re
            import subprocess
            import sys
            import torch as _t
            import torch_npu as _tnp

            if not os.environ.get("ASCEND_HOME_PATH"):
                raise RuntimeError(
                    "ASCEND_HOME_PATH is not set. Source the CANN environment before eval."
                )

            _project_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                {project_dir_name!r},
            )
            if not os.path.isdir(_project_dir):
                raise RuntimeError(f"ascendc project directory not found: {{_project_dir}}")

            _cmake_path = os.path.join(_project_dir, "CMakeLists.txt")
            if not os.path.isfile(_cmake_path):
                raise RuntimeError(f"ascendc CMakeLists.txt not found: {{_cmake_path}}")
            with open(_cmake_path, "r", encoding="utf-8") as _fp:
                _cmake_text = _fp.read()

            def _cmake_uses_arch_vars(_root):
                _var_re = re.compile(r"\\b(?:NPU_ARCH|ASCENDC_NPU_ARCH)\\b")
                for _dirpath, _dirnames, _filenames in os.walk(_root):
                    _dirnames[:] = [
                        _d for _d in _dirnames
                        if _d not in ("CMakeFiles", "__pycache__", "build", "output")
                    ]
                    for _name in _filenames:
                        if _name != "CMakeLists.txt" and not _name.endswith(".cmake"):
                            continue
                        _path = os.path.join(_dirpath, _name)
                        try:
                            with open(_path, "r", encoding="utf-8") as _fp:
                                _text = _fp.read()
                        except UnicodeDecodeError:
                            continue
                        if _var_re.search(_text):
                            return True
                return False

            _patched, _count = re.subn(
                r"--npu-arch=[A-Za-z0-9_-]+",
                "--npu-arch={npu_arch}",
                _cmake_text,
            )
            _uses_arch_vars = _cmake_uses_arch_vars(_project_dir)
            if _count:
                with open(_cmake_path, "w", encoding="utf-8") as _fp:
                    _fp.write(_patched)
            elif not _uses_arch_vars:
                raise RuntimeError(
                    "ascendc CMake has no controllable NPU arch channel. "
                    "Add a literal --npu-arch=<dav-token> flag to CMakeLists.txt, "
                    "or consume ${{NPU_ARCH}} / ${{ASCENDC_NPU_ARCH}} in project CMake files."
                )
            else:
                print(
                    "[WARN]: ascendc CMakeLists.txt has no literal --npu-arch flag; "
                    "passing -DNPU_ARCH / -DASCENDC_NPU_ARCH to cmake instead."
                )

            _torch_npu_root = os.path.dirname(_tnp.__file__)
            _torch_npu_inc = os.path.join(_torch_npu_root, "include")
            _torch_npu_lib = os.path.join(_torch_npu_root, "lib")
            os.environ["CPLUS_INCLUDE_PATH"] = (
                _torch_npu_inc + ":" + os.environ.get("CPLUS_INCLUDE_PATH", "")
            )
            os.environ["LIBRARY_PATH"] = (
                _torch_npu_lib + ":" + os.environ.get("LIBRARY_PATH", "")
            )
            os.environ["LD_LIBRARY_PATH"] = (
                _torch_npu_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            )

            _build_dir = os.path.join(_project_dir, "build")
            if os.path.isdir(_build_dir):
                import shutil as _shutil
                _shutil.rmtree(_build_dir, ignore_errors=True)
            os.makedirs(_build_dir, exist_ok=True)

            def _find_shared_objects(_root):
                _matches = []
                for _dirpath, _dirnames, _filenames in os.walk(_root):
                    _dirnames[:] = [
                        _d for _d in _dirnames
                        if _d not in ("CMakeFiles", "__pycache__")
                    ]
                    for _name in _filenames:
                        if _name.endswith(".so"):
                            _path = os.path.join(_dirpath, _name)
                            _matches.append(os.path.relpath(_path, _root))
                return sorted(_matches)

            _cmake_cmd = [
                "cmake",
                "..",
                f"-DCMAKE_PREFIX_PATH={{_t.utils.cmake_prefix_path}}",
                f"-DPython_EXECUTABLE={{sys.executable}}",
                f"-DPython3_EXECUTABLE={{sys.executable}}",
                "-DNPU_ARCH={npu_arch}",
                "-DASCENDC_NPU_ARCH={npu_arch}",
            ]
            _cfg = subprocess.run(
                _cmake_cmd,
                cwd=_build_dir,
                capture_output=True,
                text=True,
                timeout={timeout},
            )
            if _cfg.returncode != 0:
                _details = "\\n".join(
                    _part for _part in (_cfg.stdout, _cfg.stderr) if _part
                )
                raise RuntimeError("ascendc cmake configure failed\\n" + _details)
            if not _count:
                _cfg_log = "\\n".join(
                    _part for _part in (_cfg.stdout, _cfg.stderr) if _part
                )
                _unused_arch_vars = {{
                    _var for _var in ("NPU_ARCH", "ASCENDC_NPU_ARCH")
                    if re.search(rf"(?m)^\\s*{{_var}}\\s*$", _cfg_log)
                }}
                if (
                    "Manually-specified variables were not used by the project"
                    in _cfg_log
                    and _unused_arch_vars == {{"NPU_ARCH", "ASCENDC_NPU_ARCH"}}
                ):
                    raise RuntimeError(
                        "ascendc cmake ignored -DNPU_ARCH/-DASCENDC_NPU_ARCH; "
                        "the build may be using a stale hard-coded --npu-arch. "
                        "Add a literal --npu-arch=<dav-token> flag or consume "
                        "${{NPU_ARCH}} / ${{ASCENDC_NPU_ARCH}}.\\n" + _cfg_log
                    )
            _build = subprocess.run(
                ["cmake", "--build", ".", "--parallel", "1"],
                cwd=_build_dir,
                capture_output=True,
                text=True,
                timeout={timeout},
            )
            if _build.returncode != 0:
                _details = "\\n".join(
                    _part for _part in (_build.stdout, _build.stderr) if _part
                )
                raise RuntimeError("ascendc cmake build failed\\n" + _details)
            _built_so = _find_shared_objects(_build_dir)
            if not _built_so:
                raise RuntimeError(
                    f"ascendc build finished without a shared library in {{_build_dir}}"
                )
            print(f"[INFO]: ascendc build successful: {{', '.join(_built_so)}}")
            """
        )

    def materialize_impl(
        self,
        impl_code: str,
        verify_dir: str,
        op_name: str,
        framework: str,
        dsl_name: str,
        task_info: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        kernel_file = os.path.join(
            verify_dir, self.entry_filename_template.format(op_name=op_name)
        )
        with open(kernel_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        cfg = dict(config or {})
        project_src = cfg.get("ascendc_op_src")
        if not project_src and task_info:
            project_src = (
                task_info.get("ascendc_op_src")
                or task_info.get("ascendc_project_src")
                or task_info.get("kernel_project_src")
            )
        project_dir_name = cfg.get("ascendc_op_dir") or self.kernel_project_dir_name
        if (not project_src or not os.path.isdir(project_src)) and cfg.get("task_dir"):
            candidate = os.path.join(os.path.abspath(cfg["task_dir"]), project_dir_name)
            if os.path.isdir(candidate):
                project_src = candidate
        if not project_src or not os.path.isdir(project_src):
            raise ValueError(
                f"[{op_name}] ascendc_op_src not found or not a directory. "
                "Pass --kernel pointing to the ascendc_op project tree, or "
                "set config/task_info ascendc_op_src. "
                f"Got: {project_src!r}"
            )

        project_dst = os.path.join(verify_dir, project_dir_name)
        _copy_project_tree(project_src, project_dst)
        _assert_cmake_has_npu_arch_channel(project_dst, op_name)
        npu_arch = getattr(self, "_setup_npu_arch", None)
        if npu_arch:
            _patch_cmake_npu_arch(project_dst, npu_arch)
        logger.debug(
            "[%s] ascendc project copied: %s -> %s",
            op_name,
            project_src,
            project_dst,
        )

    def expected_artifacts(
        self,
        verify_dir: str,
        op_name: str,
        framework: str,
        bench_type: str,
        dsl_filename_hint: str,
    ) -> list:
        wrapper_name = self.entry_filename_template.format(op_name=op_name)
        return [
            os.path.join(verify_dir, wrapper_name),
            os.path.join(
                verify_dir,
                getattr(self, "_setup_project_dir_name", self.kernel_project_dir_name),
                "CMakeLists.txt",
            ),
        ]

    def post_iteration_cleanup(self, verify_dir: str) -> None:
        if os.environ.get("AR_KEEP_BATCH_VERIFY_TEMP") == "1":
            return
        project_dir = os.path.join(
            verify_dir,
            getattr(self, "_setup_project_dir_name", self.kernel_project_dir_name),
        )
        build_dir = os.path.join(project_dir, "build")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)

    def read_kernel_source(
        self,
        kernel_arg: str,
        op_name: Optional[str] = None,
    ) -> tuple:
        if not os.path.isdir(kernel_arg):
            raise FileNotFoundError(
                "ascendc kernel handoff must be an ascendc_op project "
                f"directory; got {kernel_arg!r}"
            )
        parent = os.path.dirname(os.path.abspath(kernel_arg))
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
            "ascendc kernel handoff is a directory; expected sibling "
            f"{' or '.join(candidates)} at {parent}"
        )

    def materialize_project_tree(
        self,
        dst_dir: str,
        project_src: Optional[str],
        project_dir_name: Optional[str] = None,
    ) -> None:
        if not project_src:
            return
        dst = os.path.join(
            dst_dir,
            project_dir_name or self.kernel_project_dir_name,
        )
        _copy_project_tree(project_src, dst)

    def list_kernel_project_files(
        self,
        project_src: Optional[str] = None,
        op_name: Optional[str] = None,
        project_dir_name: Optional[str] = None,
    ) -> list:
        if not project_src or not os.path.isdir(project_src):
            return list(self.kernel_project_files)
        project_src_path = Path(project_src).resolve()
        dst_dir = project_dir_name or self.kernel_project_dir_name
        files: list[str] = []
        for path in sorted(project_src_path.rglob("*")):
            if not path.is_file():
                continue
            project_rel_parts = path.relative_to(project_src_path).parts
            if any(part in _IGNORE_DIRS for part in project_rel_parts):
                continue
            if path.name not in _TEXT_FILENAMES and path.suffix not in _TEXT_SUFFIXES:
                continue
            # Keep the copied project broad, but the WA edit surface narrow:
            # only core direct-invoke implementation/build files are mutable.
            if not project_rel_parts or project_rel_parts[0] not in _EDITABLE_PROJECT_ROOTS:
                continue
            rel = path.relative_to(project_src_path).as_posix()
            files.append(f"{dst_dir}/{rel}")
        return files or list(self.kernel_project_files)
