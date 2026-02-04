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
Domain Tools (领域专用工具)

包含：
- verify_kernel: 验证生成的 Kernel 代码正确性
- profile_kernel: 对 Kernel 代码进行性能分析
"""

import logging
from typing import Dict, Any, List, Optional

from langchain.tools import tool

from akg_agents.core.tools.tool_schemas import (
    VerifyToolInput,
    ProfileToolInput,
)

logger = logging.getLogger(__name__)


def _error_result(error_msg: str) -> Dict[str, Any]:
    """构造错误返回结果"""
    return {
        "success": False,
        "error_log": error_msg,
        "message": f"[ERROR] {error_msg}"
    }


def _create_verify_tool(
    config: dict,
    framework: str = "torch",
    backend: str = "cuda",
    arch: str = "a100",
    dsl: str = "triton"
):
    """创建 verify_kernel tool
    
    Args:
        config: 全局配置字典，包含 log_dir 等
        framework: 框架类型，从 Agent 配置继承
        backend: 计算后端，从 Agent 配置继承
        arch: 硬件架构，从 Agent 配置继承
        dsl: DSL 类型，从 Agent 配置继承
    """
    
    @tool("verify_kernel", args_schema=VerifyToolInput)
    async def verify_kernel(
        task_code: str,
        generated_code: str,
        op_name: str,
        task_id: str = "default_task",
        device_id: int = 0,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """验证生成的 Kernel 代码的正确性。"""
        logger.info(f"verify_kernel: op={op_name}, backend={backend}, arch={arch}")
        
        try:
            # 参数验证
            if not task_code or not task_code.strip():
                return _error_result("task_code (框架实现代码) 不能为空")
            if not generated_code or not generated_code.strip():
                return _error_result("generated_code (生成代码) 不能为空")
            if not op_name or not op_name.strip():
                return _error_result("op_name (算子名称) 不能为空")
            
            # 导入必要的模块
            from akg_agents.core.verifier.kernel_verifier import KernelVerifier
            from akg_agents.core.worker.manager import get_worker_manager
            from akg_agents.core.utils import normalize_dsl
            
            # 规范化 DSL
            normalized_dsl = normalize_dsl(dsl, backend)
            logger.info(f"Normalized DSL: {dsl} -> {normalized_dsl}")
            
            # 从 WorkerManager 获取 worker
            worker_manager = get_worker_manager()
            worker = await worker_manager.select(backend=backend, arch=arch)
            if not worker:
                return _error_result(
                    f"No available worker for backend={backend}, arch={arch}. "
                    "Please register a worker first using register_local_worker() or register_remote_worker()."
                )
            
            logger.info(f"Using worker: {worker}")
            
            # 构建 verifier 配置
            verifier_config = dict(config)
            verifier_config["verify_timeout"] = timeout
            
            # 创建 KernelVerifier 实例
            verifier = KernelVerifier(
                op_name=op_name,
                framework_code=task_code,
                task_id=task_id,
                framework=framework,
                dsl=normalized_dsl,
                backend=backend,
                arch=arch,
                impl_func_name="ModelNew",
                config=verifier_config,
                worker=worker
            )
            
            # 构建 task_info
            task_info = {"coder_code": generated_code}
            
            # 执行验证
            logger.info(f"Running verification for {op_name}...")
            verify_result, error_log = await verifier.run(task_info, device_id=device_id)
            
            if verify_result:
                logger.info(f"✓ Verification passed for {op_name}")
                return {
                    "success": True,
                    "error_log": "",
                    "verify_dir": verifier.log_dir,
                    "message": f"[SUCCESS] 验证通过！算子 {op_name} 的生成代码正确性验证成功。"
                }
            else:
                logger.error(f"✗ Verification failed for {op_name}: {error_log[:200]}...")
                return {
                    "success": False,
                    "error_log": error_log,
                    "verify_dir": verifier.log_dir,
                    "message": f"[FAILED] 验证失败！算子 {op_name} 的生成代码存在错误，请检查 error_log。"
                }
                
        except Exception as e:
            logger.error(f"verify_kernel failed: {e}", exc_info=True)
            return _error_result(f"验证过程发生异常: {str(e)}")
    
    return verify_kernel


def _create_profile_tool(
    config: dict,
    framework: str = "torch",
    backend: str = "cuda",
    arch: str = "a100",
    dsl: str = "triton"
):
    """创建 profile_kernel tool
    
    Args:
        config: 全局配置字典，包含 log_dir 等
        framework: 框架类型，从 Agent 配置继承
        backend: 计算后端，从 Agent 配置继承
        arch: 硬件架构，从 Agent 配置继承
        dsl: DSL 类型，从 Agent 配置继承
    """
    
    @tool("profile_kernel", args_schema=ProfileToolInput)
    async def profile_kernel(
        task_code: str,
        generated_code: str,
        op_name: str,
        task_id: str = "default_task",
        device_id: int = 0,
        run_times: int = 50,
        warmup_times: int = 5
    ) -> Dict[str, Any]:
        """对 Kernel 代码进行性能分析（仅性能测试，不做验证）。"""
        logger.info(f"profile_kernel: op={op_name}, backend={backend}, arch={arch}, "
                   f"run_times={run_times}, warmup_times={warmup_times}")
        
        try:
            # 参数验证
            if not task_code or not task_code.strip():
                return _error_result("task_code (框架实现代码) 不能为空")
            if not generated_code or not generated_code.strip():
                return _error_result("generated_code (生成代码) 不能为空")
            if not op_name or not op_name.strip():
                return _error_result("op_name (算子名称) 不能为空")
            
            # 导入必要的模块
            from akg_agents.core.verifier.kernel_verifier import KernelVerifier
            from akg_agents.core.worker.manager import get_worker_manager
            from akg_agents.core.utils import normalize_dsl
            
            # 规范化 DSL
            normalized_dsl = normalize_dsl(dsl, backend)
            logger.info(f"Normalized DSL: {dsl} -> {normalized_dsl}")
            
            # 从 WorkerManager 获取 worker
            worker_manager = get_worker_manager()
            worker = await worker_manager.select(backend=backend, arch=arch)
            if not worker:
                return _error_result(
                    f"No available worker for backend={backend}, arch={arch}. "
                    "Please register a worker first using register_local_worker() or register_remote_worker()."
                )
            
            logger.info(f"Using worker: {worker}")
            
            # 构建 verifier 配置
            verifier_config = dict(config)
            verifier_config["verify_timeout"] = 300
            
            # 创建 KernelVerifier 实例（用于调用 run_profile 方法）
            verifier = KernelVerifier(
                op_name=op_name,
                framework_code=task_code,
                task_id=task_id,
                framework=framework,
                dsl=normalized_dsl,
                backend=backend,
                arch=arch,
                impl_func_name="ModelNew",
                config=verifier_config,
                worker=worker
            )
            
            # 构建 task_info
            task_info = {"coder_code": generated_code}
            
            # 执行性能分析
            logger.info(f"Running performance analysis for {op_name}...")
            logger.info(f"Profile settings: run_times={run_times}, warmup_times={warmup_times}")
            
            profile_settings = {
                "run_times": run_times,
                "warmup_times": warmup_times
            }
            
            profile_result = await verifier.run_profile(
                task_info,
                current_step=0,
                device_id=device_id,
                profile_settings=profile_settings
            )
            
            # 提取性能数据（注意：当 key 存在但值为 None 时，.get() 返回 None 而不是默认值）
            raw_gen_time = profile_result.get('gen_time')
            raw_base_time = profile_result.get('base_time')
            speedup = profile_result.get('speedup') or 0.0
            
            # 判断各个时间值的状态（用于区分失败原因）
            gen_time_is_inf = raw_gen_time is not None and raw_gen_time == float('inf')
            base_time_is_inf = raw_base_time is not None and raw_base_time == float('inf')
            gen_time_valid = raw_gen_time is not None and raw_gen_time != float('inf')
            base_time_valid = raw_base_time is not None and raw_base_time != float('inf')
            
            # 转换为最终值
            gen_time = raw_gen_time if gen_time_valid else None
            base_time = raw_base_time if base_time_valid else None
            
            # 根据不同失败情况提供精确的诊断信息
            if not gen_time_valid or not base_time_valid:
                # 构建详细的错误诊断
                if base_time_is_inf or gen_time_is_inf:
                    # 环境问题
                    failure_type = "ENV_CRITICAL"
                    error_msg = (
                        f"基准框架或生成代码的性能测试失败（返回 inf）。\n"
                        f"可能原因：设备不可用、profiler 配置错误、或资源问题。\n"
                        f"建议：1) 检查设备是否正常；2) 重启设备或服务；3) 检查 profiler 依赖。"
                    )
                    message = (
                        f"[ENV_CRITICAL] 性能测试失败（环境问题）\n"
                        f"  算子: {op_name}\n"
                        f"  base_time: {raw_base_time}\n"
                        f"  gen_time: {raw_gen_time}\n"
                        f"  诊断: 设备可能不可用，请检查硬件和环境配置。"
                    )
                else:
                    # 其他情况
                    failure_type = "UNKNOWN"
                    error_msg = profile_result.get('error_log', '性能测试执行失败，未能获取有效的执行时间数据')
                    message = (
                        f"[FAILED] 性能分析失败\n"
                        f"  算子: {op_name}\n"
                        f"  base_time: {raw_base_time}\n"
                        f"  gen_time: {raw_gen_time}\n"
                        f"  请检查日志获取更多信息。"
                    )
                
                logger.warning(f"✗ Performance analysis failed for {op_name}: {failure_type}")
                return {
                    "success": False,
                    "gen_time_us": gen_time,
                    "base_time_us": base_time,
                    "speedup": 0.0,
                    "failure_type": failure_type,
                    "error_log": error_msg,
                    "profile_dir": verifier.log_dir,
                    "message": message
                }
            
            logger.info(f"✓ Performance analysis completed for {op_name}!")
            logger.info(f"  Base (framework) performance: {base_time:.2f} us")
            logger.info(f"  Generated code performance: {gen_time:.2f} us")
            logger.info(f"  Speedup: {speedup:.2f}x")
            
            # 构建结果消息
            if speedup >= 1.0:
                perf_status = f"🚀 加速 {speedup:.2f}x"
            elif speedup > 0:
                perf_status = f"⚠️ 减速 {1/speedup:.2f}x"
            else:
                perf_status = "⚠️ 无法计算加速比"
            
            return {
                "success": True,
                "gen_time_us": gen_time,
                "base_time_us": base_time,
                "speedup": speedup,
                "error_log": "",
                "profile_dir": verifier.log_dir,
                "message": (
                    f"[SUCCESS] 性能分析完成！\n"
                    f"  算子: {op_name}\n"
                    f"  框架实现: {base_time:.2f} us\n"
                    f"  生成代码: {gen_time:.2f} us\n"
                    f"  性能对比: {perf_status}"
                )
            }
            
        except Exception as e:
            logger.error(f"profile_kernel failed: {e}", exc_info=True)
            return _error_result(f"性能分析过程发生异常: {str(e)}")
    
    return profile_kernel


def create_domain_tools(
    config: Optional[dict] = None,
    framework: str = "torch",
    backend: str = "cuda",
    arch: str = "a100",
    dsl: str = "triton"
) -> List:
    """
    创建领域专用 Tools (verifyTool, profileTool)
    
    Args:
        config: 全局配置字典，包含以下可选字段：
            - log_dir: 日志目录路径（默认: ~/.akg/logs）
            - 其他 KernelVerifier 需要的配置
        framework: 框架类型 ('torch', 'mindspore', 'numpy')
        backend: 计算后端 ('cuda', 'ascend', 'cpu')
        arch: 硬件架构 ('a100', 'v100', 'h20', 'ascend910b4' 等)
        dsl: DSL 类型 ('triton', 'triton_cuda', 'triton_ascend', 'ascendc', 'cuda_c')
            
    Returns:
        List of LangChain Tools: [verify_kernel, profile_kernel]
    """
    if config is None:
        config = {}
    
    # 确保有默认的 log_dir
    if "log_dir" not in config:
        import os
        config["log_dir"] = os.path.expanduser("~/.akg/logs")
    
    logger.info(f"Creating domain tools with: framework={framework}, backend={backend}, "
               f"arch={arch}, dsl={dsl}")
    
    tools = [
        _create_verify_tool(config, framework, backend, arch, dsl),
        _create_profile_tool(config, framework, backend, arch, dsl),
    ]
    
    logger.info(f"Created {len(tools)} domain tools: {[t.name for t in tools]}")
    return tools


# 为了方便直接导入使用，提供带默认配置的工具实例
# 注意：这些是带默认配置的实例，如需自定义配置，请使用 create_domain_tools(config)
_default_tools = None


def get_default_domain_tools() -> List:
    """获取带默认配置的 domain tools（单例模式）"""
    global _default_tools
    if _default_tools is None:
        _default_tools = create_domain_tools()
    return _default_tools
