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

"""
算子领域专用工具 (op/tools)

包含：
- verify_kernel: 验证生成的 Kernel 代码正确性
- profile_kernel: 对 Kernel 代码进行性能分析

这些工具依赖算子生成基础设施（KernelVerifier），属于 op 层领域特化内容。
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def verify_kernel(
    task_code: str,
    generated_code: str,
    op_name: str,
    task_id: str = "default_task",
    device_id: int = 0,
    timeout: int = 300,
    cur_path: str = "",
    framework: str = "torch",
    backend: str = "cuda",
    arch: str = "a100",
    dsl: str = "triton"
) -> Dict[str, Any]:
    """验证生成的 Kernel 代码的正确性"""
    logger.info(f"[verify_kernel] 验证算子: {op_name}, backend={backend}, arch={arch}, dsl={dsl}")

    try:
        if not task_code or not task_code.strip():
            return {"status": "error", "output": "",
                    "error_information": "task_code (框架实现代码) 不能为空"}
        if not generated_code or not generated_code.strip():
            return {"status": "error", "output": "",
                    "error_information": "generated_code (生成代码) 不能为空"}
        if not op_name or not op_name.strip():
            return {"status": "error", "output": "",
                    "error_information": "op_name (算子名称) 不能为空"}

        import asyncio
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier

        if cur_path:
            log_dir = Path(cur_path) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            log_dir = Path(tempfile.mkdtemp(prefix=f"verify_{op_name}_"))
        logger.info(f"[verify_kernel] 日志目录: {log_dir}")

        if dsl == "triton" and backend == "cuda":
            normalized_dsl = "triton_cuda"
        elif dsl == "triton" and backend == "ascend":
            normalized_dsl = "triton_ascend"
        else:
            normalized_dsl = dsl

        config = {"log_dir": str(log_dir), "verify_timeout": timeout}

        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=task_code,
            task_id=task_id,
            framework=framework,
            dsl=normalized_dsl,
            backend=backend,
            arch=arch,
            config=config
        )

        task_info = {"coder_code": generated_code, "framework_code": task_code}
        verify_result, verify_log = await verifier.run(task_info, current_step=0, device_id=device_id)

        if verify_result:
            logger.info(f"[verify_kernel] 验证成功: {op_name}")
            return {
                "status": "success",
                "output": f"验证通过！算子 {op_name} 的生成代码正确性验证成功。",
                "error_information": "",
                "verify_log": verify_log,
                "verify_dir": str(log_dir)
            }
        else:
            logger.warning(f"[verify_kernel] 验证失败: {op_name}")
            return {
                "status": "fail",
                "output": "",
                "error_information": f"验证失败！算子 {op_name} 的生成代码存在错误。\n\n{verify_log}",
                "verify_log": verify_log,
                "verify_dir": str(log_dir)
            }

    except ImportError as e:
        return {"status": "error", "output": "",
                "error_information": f"无法导入 KernelVerifier，请检查依赖: {str(e)}"}
    except Exception as e:
        logger.error(f"[verify_kernel] 验证失败: {e}", exc_info=True)
        return {"status": "error", "output": "",
                "error_information": f"验证过程出错: {type(e).__name__}: {str(e)}"}


async def profile_kernel(
    task_code: str,
    generated_code: str,
    op_name: str,
    task_id: str = "default_task",
    device_id: int = 0,
    run_times: int = 50,
    warmup_times: int = 5,
    cur_path: str = "",
    framework: str = "torch",
    backend: str = "cuda",
    arch: str = "a100",
    dsl: str = "triton"
) -> Dict[str, Any]:
    """对 Kernel 代码进行性能分析（仅性能测试，不做验证）"""
    logger.info(f"[profile_kernel] 性能分析: {op_name}, backend={backend}, arch={arch}")

    try:
        if not task_code or not task_code.strip():
            return {"status": "error", "output": "",
                    "error_information": "task_code (框架实现代码) 不能为空"}
        if not generated_code or not generated_code.strip():
            return {"status": "error", "output": "",
                    "error_information": "generated_code (生成代码) 不能为空"}
        if not op_name or not op_name.strip():
            return {"status": "error", "output": "",
                    "error_information": "op_name (算子名称) 不能为空"}

        import asyncio
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier

        if cur_path:
            log_dir = Path(cur_path) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            log_dir = Path(tempfile.mkdtemp(prefix=f"profile_{op_name}_"))

        if dsl == "triton" and backend == "cuda":
            normalized_dsl = "triton_cuda"
        elif dsl == "triton" and backend == "ascend":
            normalized_dsl = "triton_ascend"
        else:
            normalized_dsl = dsl

        config = {
            "log_dir": str(log_dir),
            "profile_settings": {"run_times": run_times, "warmup_times": warmup_times}
        }

        verifier = KernelVerifier(
            op_name=op_name, framework_code=task_code, task_id=task_id,
            framework=framework, dsl=normalized_dsl, backend=backend,
            arch=arch, config=config
        )

        task_info = {"coder_code": generated_code, "framework_code": task_code}
        profile_result = await verifier.run_profile(
            task_info, current_step=0, device_id=device_id,
            profile_settings=config["profile_settings"]
        )

        gen_time = profile_result.get('gen_time')
        base_time = profile_result.get('base_time')
        speedup = profile_result.get('speedup', 0.0)

        if gen_time is None or base_time is None or gen_time == float('inf') or base_time == float('inf'):
            error_msg = profile_result.get('error_log', '性能测试执行失败')
            return {
                "status": "fail", "output": "",
                "error_information": f"性能分析失败！{error_msg}",
                "gen_time_us": gen_time, "base_time_us": base_time,
                "speedup": 0.0, "profile_dir": str(log_dir)
            }

        if speedup >= 1.0:
            perf_status = f"加速 {speedup:.2f}x"
        elif speedup > 0:
            perf_status = f"减速 {1/speedup:.2f}x"
        else:
            perf_status = "无法计算加速比"

        return {
            "status": "success",
            "output": (
                f"性能分析完成！\n"
                f"算子: {op_name}\n"
                f"框架实现: {base_time:.2f} us\n"
                f"生成代码: {gen_time:.2f} us\n"
                f"性能对比: {perf_status}"
            ),
            "error_information": "",
            "gen_time_us": gen_time, "base_time_us": base_time,
            "speedup": speedup, "profile_dir": str(log_dir)
        }

    except ImportError as e:
        return {"status": "error", "output": "",
                "error_information": f"无法导入 KernelVerifier，请检查依赖: {str(e)}"}
    except Exception as e:
        logger.error(f"[profile_kernel] 性能分析失败: {e}", exc_info=True)
        return {"status": "error", "output": "",
                "error_information": f"性能分析过程出错: {type(e).__name__}: {str(e)}"}


# ==================== 工具注册 ====================

def _register_all():
    from akg_agents.core_v2.tools.tool_registry import ToolRegistry

    ToolRegistry.register(
        name="verify_kernel",
        description=(
            "验证生成的 Kernel 代码的正确性。\n"
            "对比框架实现（如 PyTorch）和生成实现（如 Triton）的输出结果，"
            "验证数值精度是否满足要求。\n\n"
            "使用场景：生成代码后确认正确性。\n"
            "参数获取：task_code 从 call_op_task_builder 返回，"
            "generated_code 从代码生成工具返回。"
        ),
        parameters={"type": "object", "properties": {
            "task_code": {"type": "string", "description": "PyTorch/MindSpore 框架实现代码"},
            "generated_code": {"type": "string", "description": "待验证的生成代码"},
            "op_name": {"type": "string", "description": "算子名称"},
            "task_id": {"type": "string", "description": "任务 ID", "default": "default_task"},
            "device_id": {"type": "integer", "description": "设备 ID", "default": 0},
            "timeout": {"type": "integer", "description": "超时秒数", "default": 300},
        }, "required": ["task_code", "generated_code", "op_name"]},
        func=verify_kernel,
        category="domain",
        scopes=["kernel_agent"],
    )

    ToolRegistry.register(
        name="profile_kernel",
        description=(
            "对 Kernel 代码进行性能分析（仅性能测试，不做验证）。\n"
            "返回框架实现和生成代码的执行时间及加速比。"
        ),
        parameters={"type": "object", "properties": {
            "task_code": {"type": "string", "description": "PyTorch/MindSpore 框架实现代码"},
            "generated_code": {"type": "string", "description": "待分析的生成代码"},
            "op_name": {"type": "string", "description": "算子名称"},
            "task_id": {"type": "string", "description": "任务 ID", "default": "default_task"},
            "device_id": {"type": "integer", "description": "设备 ID", "default": 0},
            "run_times": {"type": "integer", "description": "运行次数", "default": 50},
            "warmup_times": {"type": "integer", "description": "预热次数", "default": 5},
        }, "required": ["task_code", "generated_code", "op_name"]},
        func=profile_kernel,
        category="domain",
        scopes=["kernel_agent"],
    )


_register_all()
