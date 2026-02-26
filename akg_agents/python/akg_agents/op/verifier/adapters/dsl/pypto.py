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

"""PyPTO DSL adapter - 支持 ModelNew (KernelBench) 格式."""

from typing import Any, Optional
import re

from .base import DSLAdapter


class DSLAdapterPypto(DSLAdapter):
    """Adapter for PyPTO DSL.
    
    PyPTO 是一种用于生成 NPU 算子的新语言，使用 @pypto.jit 装饰器和切片语法。
    与 Triton 不同，PyPTO 使用 tensor[start:end] 语法而非 tl.load/store。
    """
    
    def get_import_statements(self, framework: str) -> str:
        """Return PyPTO import statements."""
        code = ""
        if framework == "torch":
            code += "import torch\n"
            code += "import pypto\n"
        elif framework == "mindspore":
            code += "import torch\n"
            code += "import pypto\n"
        elif framework == "numpy":
            code += "import numpy as np\n"
            code += "import torch\n"
            code += "import pypto\n"
        else:
            code += "import pypto\n"
        code += """import os
"""
        return code

    def get_runtime_env_override_code(
        self,
        pypto_run_mode: Optional[int] = None,
        pypto_runtime_debug_mode: Optional[int] = None,
    ) -> str:
        """Return code to inject per-task PyPTO runtime env overrides."""
        lines = []
        if pypto_run_mode is not None:
            lines.append(f'os.environ["AIKG_PYPTO_RUN_MODE"] = "{pypto_run_mode}"')
            lines.append(
                'print(f"[INFO] Task override: AIKG_PYPTO_RUN_MODE={os.environ[\'AIKG_PYPTO_RUN_MODE\']}")'
            )
        if pypto_runtime_debug_mode is not None:
            lines.append(
                f'os.environ["AIKG_PYPTO_RUNTIME_DEBUG_MODE"] = "{pypto_runtime_debug_mode}"'
            )
            lines.append(
                'print(f"[INFO] Task override: AIKG_PYPTO_RUNTIME_DEBUG_MODE={os.environ[\'AIKG_PYPTO_RUNTIME_DEBUG_MODE\']}")'
            )
        return "\n".join(lines) + ("\n" if lines else "")
    
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.
        
        统一使用 ModelNew 类格式（KernelBench 风格）。
        """
        module_name = re.sub(r"\W", "_", op_name)
        if not module_name or module_name[0].isdigit():
            module_name = f"op_{module_name}"
        return (
            "import importlib.util\n"
            "import os\n"
            f"_impl_module_name = '{module_name}_pypto_impl'\n"
            f"_impl_module_path = os.path.join(os.path.dirname(__file__), '{op_name}_pypto_impl.py')\n"
            "_impl_spec = importlib.util.spec_from_file_location(_impl_module_name, _impl_module_path)\n"
            "_impl_module = importlib.util.module_from_spec(_impl_spec)\n"
            "_impl_spec.loader.exec_module(_impl_module)\n"
            "ModelNew = _impl_module.ModelNew\n"
        )

    def create_impl_module(self, framework: str,
                          framework_adapter: Any, 
                          init_params_var: str = "init_params",
                          device_var: str = "device") -> str:
        """生成创建 impl_model 的代码（只实例化一次）。
        
        Args:
            framework: Framework name (torch, mindspore, numpy)
            framework_adapter: Framework adapter instance
            init_params_var: Variable name for init_params (default: "init_params")
            device_var: Variable name for device (default: "device")
            
        Returns:
            str: Code string to create impl_model
        """
        code = f"impl_model = ModelNew(*{init_params_var})\n"
        if framework == "torch":
            code += f"impl_model = impl_model.to({device_var})\n"
        
        return code
    
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str, 
                  data_dir: Optional[str] = None, 
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call PyPTO implementation function.

        调用已经实例化好的 ``impl_model``。
        """
        return (
            f"impl_output = impl_model(*{inputs})\n"
        )
    
    def needs_binary_io(self) -> bool:
        """PyPTO doesn't need binary I/O."""
        return False
    
    def needs_compilation(self) -> bool:
        """PyPTO doesn't need separate compilation step."""
        return False
    
    def benchmark_impl(self, impl_func_name: str, inputs: str, 
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None) -> str:
        """Return code string to benchmark PyPTO implementation.
        
        使用已经实例化好的 impl_model 进行性能测试。
        PyPTO 运行在 NPU 上，使用 NPU profiler 进行性能测试。
        """
        code = f"""        import os
        import json
        import sys
        import subprocess
        from pathlib import Path
        
        # 定义性能测试函数
        def pypto_benchmark_fn():
            result = impl_model(*{inputs})
            return result
        
        def _calc_trace_span_us(trace_path):
            try:
                from akg_agents import get_project_root
                proj_root = Path(get_project_root()).parent.parent
                script = proj_root / "python" / "akg_agents" / "op" / "tools" / "calc_trace_span.py"
                if script.exists():
                    out = subprocess.check_output(
                        [sys.executable, str(script), str(trace_path)],
                        text=True
                    )
                    return float(out.strip())
            except Exception:
                pass
            with open(trace_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            events = [e for e in data.get("traceEvents", []) if e.get("ph") == "X"]
            if not events:
                raise RuntimeError("未找到 ph==X 的事件")
            min_ts = min(float(e.get("ts", 0) or 0) for e in events)
            max_end = max(
                float(e.get("ts", 0) or 0) + float(e.get("dur", 0) or 0)
                for e in events
            )
            return max_end - min_ts
        
        def _find_latest_swimlane(base_dir):
            \"\"\"Find merged_swimlane.json from profiler output directory.\"\"\"
            import glob
            if not os.path.isdir(base_dir):
                raise FileNotFoundError(f"Profiler output dir not found: {{base_dir}}")

            patterns = [
                os.path.join(base_dir, "output_*", "merged_swimlane.json"),
                os.path.join(base_dir, "merged_swimlane.json"),
                os.path.join(base_dir, "**", "merged_swimlane.json"),
            ]
            candidates = []
            for pattern in patterns:
                candidates.extend(glob.glob(pattern, recursive=("**" in pattern)))
            candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
            if candidates:
                return candidates[0]
            try:
                entries = sorted(os.listdir(base_dir))
            except Exception:
                entries = []
            raise FileNotFoundError(
                f"merged_swimlane.json not found under {{base_dir}}. "
                f"Searched patterns: {{patterns}}. "
                f"Top-level entries: {{entries[:30]}}"
            )

        if backend == "ascend":
            # persistent 场景下必须每次重置输出目录与日志状态；
            # 否则 pypto 可能复用上一次缓存的 output_* 子目录，导致二次运行找不到文件。
            output_dir = os.path.abspath(f"prof_generation_output_case{case_idx}")
            os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir
            os.makedirs(output_dir, exist_ok=True)
            try:
                if hasattr(pypto, "pypto_impl") and hasattr(pypto.pypto_impl, "ResetLog"):
                    pypto.pypto_impl.ResetLog("")
            except Exception as e:
                print(f"[WARN] pypto ResetLog failed: {{e}}")
            # PyPTO profile 不需要 warmup
            pypto_benchmark_fn()
            trace_path = _find_latest_swimlane(output_dir)
            print(f"[INFO] PyPTO trace path: {{trace_path}}")
            execution_time_us = _calc_trace_span_us(trace_path)
            execution_time_ms = execution_time_us / 1000
            method = "trace_span"
        else:
            # 简单计时方式（无 warmup）
            import time
            times = []
            for _ in range({runs}):
                start = time.perf_counter()
                pypto_benchmark_fn()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            execution_time_ms = min(times) if times else 0.0
            method = "simple_timing"
"""
        return code
    
