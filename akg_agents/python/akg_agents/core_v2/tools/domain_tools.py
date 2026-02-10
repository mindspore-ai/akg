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
Domain Tools for core_v2 (领域专用工具)

包含：
- verify_kernel: 验证生成的 Kernel 代码正确性
- profile_kernel: 对 Kernel 代码进行性能分析
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
    # 从 agent_context 获取的参数
    framework: str = "torch",
    backend: str = "cuda",
    arch: str = "a100",
    dsl: str = "triton"
) -> Dict[str, Any]:
    """
    验证生成的 Kernel 代码的正确性
    
    Args:
        task_code: PyTorch/MindSpore 框架实现代码
        generated_code: 待验证的 Triton/CUDA/AscendC 生成代码
        op_name: 算子名称
        task_id: 任务 ID
        device_id: 设备 ID
        timeout: 验证超时时间（秒）
        cur_path: 当前节点路径（可选），指定后验证日志存放在 cur_path/logs/
        framework: 框架类型（从 agent_context 获取）
        backend: 计算后端（从 agent_context 获取）
        arch: 硬件架构（从 agent_context 获取）
        dsl: DSL 类型（从 agent_context 获取）
        
    Returns:
        包含验证结果的字典
    """
    logger.info(f"[verify_kernel] 验证算子: {op_name}, backend={backend}, arch={arch}, dsl={dsl}")
    
    try:
        # 参数验证
        if not task_code or not task_code.strip():
            return {
                "status": "error",
                "output": "",
                "error_information": "task_code (框架实现代码) 不能为空"
            }
        if not generated_code or not generated_code.strip():
            return {
                "status": "error",
                "output": "",
                "error_information": "generated_code (生成代码) 不能为空"
            }
        if not op_name or not op_name.strip():
            return {
                "status": "error",
                "output": "",
                "error_information": "op_name (算子名称) 不能为空"
            }
        
        # 动态导入（避免循环依赖）
        import asyncio
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier
        
        # 确定日志目录：优先使用 cur_path/logs，否则使用临时目录
        if cur_path:
            log_dir = Path(cur_path) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
        log_dir = Path(tempfile.mkdtemp(prefix=f"verify_{op_name}_"))
        logger.info(f"[verify_kernel] 日志目录: {log_dir}")
        
        # 规范化 DSL
        if dsl == "triton" and backend == "cuda":
            normalized_dsl = "triton_cuda"
        elif dsl == "triton" and backend == "ascend":
            normalized_dsl = "triton_ascend"
        else:
            normalized_dsl = dsl
        
        logger.info(f"[verify_kernel] DSL: {dsl} -> {normalized_dsl}")
        
        # 创建配置
        config = {
            "log_dir": str(log_dir),
            "verify_timeout": timeout
        }
        
        # 创建 KernelVerifier 实例
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
        
        # 准备 task_info
        task_info = {
            "coder_code": generated_code,
            "framework_code": task_code
        }
        
        # 执行异步验证
        verify_result, verify_log = await verifier.run(task_info, current_step=0, device_id=device_id)
        
        if verify_result:
            logger.info(f"[verify_kernel] ✓ 验证成功: {op_name}")
            return {
                "status": "success",
                "output": f"验证通过！算子 {op_name} 的生成代码正确性验证成功。",
                "error_information": "",
                "verify_log": verify_log,
                "verify_dir": str(log_dir)
            }
        else:
            logger.warning(f"[verify_kernel] ✗ 验证失败: {op_name}")
            return {
                "status": "fail",
                "output": "",
                "error_information": f"验证失败！算子 {op_name} 的生成代码存在错误。\n\n{verify_log}",
                "verify_log": verify_log,
                "verify_dir": str(log_dir)
            }
        
    except ImportError as e:
        logger.error(f"[verify_kernel] 导入 KernelVerifier 失败: {e}")
        return {
            "status": "error",
            "output": "",
            "error_information": f"无法导入 KernelVerifier，请检查依赖: {str(e)}"
        }
    except Exception as e:
        logger.error(f"[verify_kernel] 验证失败: {e}", exc_info=True)
        return {
            "status": "error",
            "output": "",
            "error_information": f"验证过程出错: {type(e).__name__}: {str(e)}"
        }


async def profile_kernel(
    task_code: str,
    generated_code: str,
    op_name: str,
    task_id: str = "default_task",
    device_id: int = 0,
    run_times: int = 50,
    warmup_times: int = 5,
    cur_path: str = "",
    # 从 agent_context 获取的参数
    framework: str = "torch",
    backend: str = "cuda",
    arch: str = "a100",
    dsl: str = "triton"
) -> Dict[str, Any]:
    """
    对 Kernel 代码进行性能分析（仅性能测试，不做验证）
    
    Args:
        task_code: PyTorch/MindSpore 框架实现代码
        generated_code: 待分析的 Triton/CUDA/AscendC 生成代码
        op_name: 算子名称
        task_id: 任务 ID
        device_id: 设备 ID
        run_times: 性能测试运行次数
        warmup_times: 预热次数
        cur_path: 当前节点路径（可选），指定后日志存放在 cur_path/logs/
        framework: 框架类型（从 agent_context 获取）
        backend: 计算后端（从 agent_context 获取）
        arch: 硬件架构（从 agent_context 获取）
        dsl: DSL 类型（从 agent_context 获取）
        
    Returns:
        包含性能分析结果的字典
    """
    logger.info(f"[profile_kernel] 性能分析: {op_name}, backend={backend}, arch={arch}, "
               f"run_times={run_times}, warmup_times={warmup_times}")
    
    try:
        # 参数验证
        if not task_code or not task_code.strip():
            return {
                "status": "error",
                "output": "",
                "error_information": "task_code (框架实现代码) 不能为空"
            }
        if not generated_code or not generated_code.strip():
            return {
                "status": "error",
                "output": "",
                "error_information": "generated_code (生成代码) 不能为空"
            }
        if not op_name or not op_name.strip():
            return {
                "status": "error",
                "output": "",
                "error_information": "op_name (算子名称) 不能为空"
            }
        
        # 动态导入（避免循环依赖）
        import asyncio
        from akg_agents.op.verifier.kernel_verifier import KernelVerifier
        
        # 确定日志目录：优先使用 cur_path/logs，否则使用临时目录
        if cur_path:
            log_dir = Path(cur_path) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
        log_dir = Path(tempfile.mkdtemp(prefix=f"profile_{op_name}_"))
        logger.info(f"[profile_kernel] 日志目录: {log_dir}")
        
        # 规范化 DSL
        if dsl == "triton" and backend == "cuda":
            normalized_dsl = "triton_cuda"
        elif dsl == "triton" and backend == "ascend":
            normalized_dsl = "triton_ascend"
        else:
            normalized_dsl = dsl
        
        logger.info(f"[profile_kernel] DSL: {dsl} -> {normalized_dsl}")
        
        # 创建配置
        config = {
            "log_dir": str(log_dir),
            "profile_settings": {
                "run_times": run_times,
                "warmup_times": warmup_times
            }
        }
        
        # 创建 KernelVerifier 实例
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
        
        # 准备 task_info
        task_info = {
            "coder_code": generated_code,
            "framework_code": task_code
        }
        
        # 执行异步性能分析
        profile_result = await verifier.run_profile(
            task_info, 
            current_step=0, 
            device_id=device_id,
            profile_settings=config["profile_settings"]
        )
        
        # 提取性能数据
        gen_time = profile_result.get('gen_time')
        base_time = profile_result.get('base_time')
        speedup = profile_result.get('speedup', 0.0)
        
        # 判断性能测试是否成功
        if gen_time is None or base_time is None or gen_time == float('inf') or base_time == float('inf'):
            error_msg = profile_result.get('error_log', '性能测试执行失败，未能获取有效的执行时间数据')
            logger.warning(f"[profile_kernel] ✗ 性能分析失败: {op_name}")
            return {
                "status": "fail",
                "output": "",
                "error_information": f"性能分析失败！{error_msg}",
                "gen_time_us": gen_time,
                "base_time_us": base_time,
                "speedup": 0.0,
                "profile_dir": str(log_dir)
            }
        
        logger.info(f"[profile_kernel] ✓ 性能分析完成: {op_name}")
        logger.info(f"  框架实现: {base_time:.2f} us")
        logger.info(f"  生成代码: {gen_time:.2f} us")
        logger.info(f"  加速比: {speedup:.2f}x")
        
        # 构建性能对比消息
        if speedup >= 1.0:
            perf_status = f"🚀 加速 {speedup:.2f}x"
        elif speedup > 0:
            perf_status = f"⚠️ 减速 {1/speedup:.2f}x"
        else:
            perf_status = "⚠️ 无法计算加速比"
        
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
            "gen_time_us": gen_time,
            "base_time_us": base_time,
            "speedup": speedup,
            "profile_dir": str(log_dir)
        }
        
    except ImportError as e:
        logger.error(f"[profile_kernel] 导入 KernelVerifier 失败: {e}")
        return {
            "status": "error",
            "output": "",
            "error_information": f"无法导入 KernelVerifier，请检查依赖: {str(e)}"
        }
    except Exception as e:
        logger.error(f"[profile_kernel] 性能分析失败: {e}", exc_info=True)
        return {
            "status": "error",
            "output": "",
            "error_information": f"性能分析过程出错: {type(e).__name__}: {str(e)}"
        }
