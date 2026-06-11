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

"""Base class for DSL adapters."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict


class DSLAdapter(ABC):
    """Abstract base class for DSL adapters.
    
    DSL adapters provide a unified interface for different implementation languages
    (Triton, SWFT, AscendC, etc.) to handle function calls, benchmarking, and
    other DSL-specific operations. DSL adapters are unaware of autotune logic.
    """
    
    @abstractmethod
    def get_import_statements(self, framework: str) -> str:
        """Return import statements for the DSL.
        
        Args:
            framework: Framework name (torch, mindspore, numpy)
            
        Returns:
            str: Import statements as a string
        """
        pass
    
    @abstractmethod
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return import statement for implementation function.
        
        Args:
            op_name: Operator name
            impl_func_name: Implementation function name
            
        Returns:
            str: Import statement (e.g., "from {op_name}_triton_cuda_impl import {impl_func_name}\n")
        """
        pass
    
    def create_impl_module(self, framework: str,
                           framework_adapter: Any,
                           init_params_var: str = "init_params",
                           device_var: str = "device") -> str:
        """生成创建 impl_model 的代码（用于 ModelNew 类格式的 DSL）。

        对于使用 ModelNew 类格式的 DSL（如 triton_cuda, triton_ascend, cpp），
        需要先实例化模型。对于函数式 DSL，返回空字符串。

        Args:
            framework: Framework name (torch, mindspore, numpy)
            framework_adapter: Framework adapter instance
            init_params_var: Variable name for init_params (default: "init_params")
            device_var: Variable name for device (default: "device")

        Returns:
            str: Code string to create impl_model, or empty string if not needed
        """
        return ""  # 默认返回空字符串，ModelNew 类格式的 DSL 需要override
    
    @abstractmethod
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call implementation function.
        
        Args:
            impl_func_name: Implementation function name
            inputs: Input variable name (e.g., "inputs_for_impl")
            device_id: Device ID
            framework_adapter: Framework adapter instance (for generating code)
            op_name: Operator name
            data_dir: Data directory variable name (for swft)
            framework_output: Framework output variable name (for swft)
            
        Returns:
            str: Code string to call the implementation
        """
        pass
    
    # needs_binary_io: True iff impl uses file-based I/O (swft).
    # needs_compilation: True iff impl must be compiled before import/use.
    # static_check_via_python_ast: True iff LLM-submitted source is
    # parseable Python; CodeChecker skips when False.
    needs_binary_io: bool = False
    needs_compilation: bool = False
    static_check_via_python_ast: bool = True

    # ------------------------------------------------------------------
    # Kernel project structure — DSL-knowable data describing how this
    # DSL packages its kernel sources. Used by both akg (verifier needs
    # the file list for materialize_impl / expected_artifacts) and any
    # outer driver that has to know where to copy / what to expose as
    # editable. The driver derives its own policy (e.g. WA scaffold
    # builds task.yaml ``editable_files`` from this list); the adapter
    # only states what the project IS.
    # ------------------------------------------------------------------
    # True → the kernel handoff path is a directory containing a sibling
    # Python wrapper + a per-DSL project subtree (e.g. catlass_op/).
    # False → a single Python file IS the kernel handoff.
    kernel_arg_is_directory: bool = False
    # When kernel_arg_is_directory=True, the subdirectory name (relative
    # to the per-op root) holding the project subtree (catlass = "catlass_op").
    kernel_project_dir_name: Optional[str] = None
    # Files (relative to the kernel's Python wrapper) that belong to the
    # DSL's kernel project besides the wrapper itself — sources, headers,
    # build files. Single-file DSLs leave this empty.
    kernel_project_files: list = []
    # Op entry filename — the file the LLM mainly edits. Format-string
    # with optional ``{op_name}`` slot. ModelNew wrapper for triton /
    # tilelang / pypto / catlass / ascendc / torch; a pure-C++ DSL would
    # override to ``"{op_name}_kernel.cpp"`` etc.
    # Consumers should NOT assume Python — check
    # ``static_check_via_python_ast`` for that.
    entry_filename_template: str = "kernel.py"

    @abstractmethod
    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      framework: str = "torch") -> str:
        """Return code string to benchmark implementation function.
        
        Args:
            impl_func_name: Implementation function name
            inputs: Input variable name (e.g., "inputs")
            warmup: Number of warmup iterations
            runs: Number of benchmark iterations
            backend: Backend type
            op_name: Operator name
            case_idx: Case index (for dynamic shape)
            framework_model: Framework model variable name (for swft)
            framework_adapter: Framework adapter (for generating code)
            device_id: Device ID (for swft)
            framework: Framework type ("torch" or "mindspore")
            
        Returns:
            str: Code string for benchmarking
        """
        pass
    
    def get_special_setup_code(self, framework: str = "torch") -> str:
        """Return special setup code (e.g., tilelang cache clear).

        Args:
            framework: Framework type ("torch" or "mindspore")

        Returns:
            str: Setup code as string (empty if not needed)
        """
        return ""

    # ------------------------------------------------------------------
    # Extension hooks — KernelVerifier / akg_eval / LocalWorker delegate
    # per-DSL behavior here so new DSLs need only override these instead
    # of editing the call sites with if/elif chains.
    # ------------------------------------------------------------------

    def materialize_impl(self, impl_code: str, verify_dir: str,
                         op_name: str, framework: str,
                         dsl_name: str,
                         task_info: Optional[Dict[str, Any]] = None,
                         config: Optional[Dict[str, Any]] = None) -> None:
        """Write the generated kernel into verify_dir for this DSL.

        Default: write ``<op>_<dsl_name>_impl.py`` with the adapter's
        own import statements prepended (the convention used by Triton /
        Tilelang / SWFT / Torch). DSLs that need a project tree
        (AscendC-CATLASS) override to drop their own files.
        """
        import os
        impl_path = os.path.join(verify_dir, f"{op_name}_{dsl_name}_impl.py")
        imports = self.get_import_statements(framework)
        with open(impl_path, "w", encoding="utf-8") as f:
            f.write(imports + impl_code)

    def expected_artifacts(self, verify_dir: str, op_name: str,
                           framework: str, bench_type: str,
                           dsl_filename_hint: str) -> list:
        """Files that must exist in verify_dir before profile can run.

        Default: framework file + ``<op>_<dsl>_impl.py``. SOL / CANN /
        catlass override.
        """
        import os
        return [
            os.path.join(verify_dir, f"{op_name}_{framework}.py"),
            os.path.join(verify_dir, f"{op_name}_{dsl_filename_hint}_impl.py"),
        ]

    def prepare_config(self, config: Dict[str, Any],
                       task_info: Optional[Dict[str, Any]] = None) -> None:
        """Mutate ``config`` in place before verify/profile runs (e.g.
        resolve CATLASS_ROOT). Default: no-op."""
        return None

    # Pure per-adapter flags / templates (no runtime state). Override as
    # a class attribute on the subclass — don't wrap in a method just
    # to return a constant.
    #
    #   benchmark_requires_l2_clear: should the base benchmark template
    #     clear L2 cache between runs? AscendC-CATLASS = False (cmake-
    #     built kernel keeps state across iterations).
    #
    #   profile_via_python_script: LocalWorker dispatch — True → run
    #     profile scripts via Python + read JSON; False → compile-then-
    #     launch flow (AscendC / SWFT / tilelang etc.).
    #
    #   impl_func_name_template: default impl function name; KernelVerifier
    #     reads it when the caller didn't pin one. Format-string with
    #     ``{op_name}``, ``{dsl}``, ``{framework}`` slots. Override per
    #     DSL: "ModelNew" for class-style (triton / catlass / torch),
    #     "{op_name}_kernel" for AscendC, the default
    #     "{op_name}_{dsl}_{framework}" for everything else.
    benchmark_requires_l2_clear: bool = True
    profile_via_python_script: bool = False
    impl_func_name_template: str = "{op_name}_{dsl}_{framework}"

    def post_iteration_cleanup(self, verify_dir: str) -> None:
        """Drop per-round artifacts that should not survive into the
        next iteration (e.g. catlass build dir). Default: no-op."""
        return None

    def get_runtime_env_override_code(self, **kwargs) -> str:
        """Emit a `__main__`-side env override snippet (e.g. PyPTO's
        runtime mode / debug flags). Default: no-op so KernelVerifier
        can always call it without a hasattr() guard. Override on the
        DSL that actually needs it."""
        return ""

    def read_kernel_source(self, kernel_arg: str,
                           op_name: Optional[str] = None) -> tuple:
        """Resolve a kernel handoff path into ``(source_code, project_dir_or_None)``.

        ``source_code`` is the text of the Python wrapper exposing
        ModelNew (or whatever entry the DSL uses); ``project_dir`` is
        the supplementary source tree the wrapper sits next to when
        ``kernel_arg_is_directory=True``, or ``None`` for single-file
        DSLs. Callers that need the full project tree forward
        ``project_dir`` to :meth:`materialize_project_tree`.

        Default: single-file convention — ``kernel_arg`` is a ``.py``
        file; read it and return ``(code, None)``. catlass overrides to
        accept a directory and uses ``op_name`` for the
        ``kernel.py`` / ``<op>_kernel.py`` fallback.
        """
        import os
        if not os.path.isfile(kernel_arg):
            raise FileNotFoundError(
                f"kernel handoff must be a file for this DSL; got {kernel_arg!r}"
            )
        with open(kernel_arg, "r", encoding="utf-8") as f:
            return f.read(), None

    def materialize_project_tree(self, dst_dir: str,
                                 project_src: Optional[str],
                                 project_dir_name: Optional[str] = None) -> None:
        """Copy the DSL's project tree from ``project_src`` into ``dst_dir``
        with any DSL-specific patching (e.g. catlass cmake rewrite).
        Default: no-op (single-file DSLs have nothing to copy beyond the
        wrapper, which the caller already wrote out)."""
        return None

    def list_kernel_project_files(self, project_src: Optional[str] = None,
                                  op_name: Optional[str] = None,
                                  project_dir_name: Optional[str] = None) -> list:
        """Return editable files belonging to this DSL project tree.

        Most multi-file DSLs have a fixed project shape and can use the
        class-level ``kernel_project_files`` list. DSLs whose project files
        are operator-named (AscendC direct-invoke) override this to discover
        text sources from ``project_src`` after the handoff path is known.
        Entries are relative to the wrapper's directory.
        """
        files = list(self.kernel_project_files)
        src_dir = self.kernel_project_dir_name
        dst_dir = project_dir_name or src_dir
        if src_dir and dst_dir and dst_dir != src_dir:
            prefix = f"{src_dir}/"
            files = [
                f"{dst_dir}/{path[len(prefix):]}"
                if path.startswith(prefix) else path
                for path in files
            ]
        return files

    def get_autotune_info(self, case_idx: int) -> Optional[Dict]:
        """Get autotune information (only for triton_ascend in profiling).
        
        Args:
            case_idx: Case index
            
        Returns:
            dict or None: Autotune information
        """
        return None
    
    def get_binary_io_functions(self) -> str:
        """Get binary I/O functions code (only for swft).
        
        Returns:
            str: Function definitions as string (empty if not needed)
        """
        return ""
