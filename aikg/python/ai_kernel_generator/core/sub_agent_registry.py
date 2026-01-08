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
子 Agent 注册中心

提供即插即用的子 Agent 管理机制：
1. 注册新的子 Agent
2. 动态发现可用的子 Agent
3. 统一的调用接口
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SubAgentBase(ABC):
    """
    子 Agent 基类
    
    所有子 Agent 必须继承这个类并实现 execute() 方法
    """
    
    def __init__(self, 
                 config: dict,
                 framework: str = "torch",
                 backend: str = "cuda",
                 arch: str = "a100",
                 dsl: str = "triton"):
        """
        初始化子 Agent
        
        Args:
            config: 配置字典
            framework: 框架类型
            backend: 后端类型
            arch: 硬件架构
            dsl: DSL 类型
        """
        self.config = config
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.dsl = dsl

    @abstractmethod
    async def execute(self, 
                     task_code: str,
                     op_name: str,
                     task_id: str,
                     **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        执行子 Agent
        
        Args:
            task_code: OpTaskBuildAgent 生成的 task 代码
            op_name: 算子名称
            task_id: 任务 ID
            **kwargs: 其他参数
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (是否成功, 结果字典)
                结果字典应包含：
                - generated_code: 生成的代码
                - verification_result: 验证结果          
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回子 Agent 的名称"""
        pass
    
    @abstractmethod
    def get_detailed_info(self) -> Dict[str, Any]:
        """
        返回子 Agent 的详细信息（用于 LLM 决策选择）
        
        Returns:
            Dict 包含：
            - name: 名称
            - description: 简短描述
            - workflow_steps: 工作流程步骤列表
            - use_cases: 适用场景列表
            - advantages: 优势列表
            - limitations: 限制/劣势列表
            - performance: 性能特点
        """
        pass


class CodeOnlySubAgent(SubAgentBase):
    """
    CodeOnly 子 Agent
    
    直接生成代码，跳过 Designer 阶段
    流程: Coder → Verifier
    """
    
    def get_name(self) -> str:
        return "codeonly"
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """返回详细信息用于 LLM 决策"""
        return {
            "name": "codeonly",
            "description": "标准的算子代码生成流程（生成 + 验证，可选性能测试）",
            "workflow_steps": [
                "Coder: 使用 LLM 直接生成 Triton 算子代码",
                "Verifier: 编译、运行、验证代码正确性",
                "Profile（可选）: 如果用户要求测性能，进行性能测试"
            ],
            "use_cases": [
                "【默认选择】用户没有明确的性能优化需求，只是要生成算子代码",
                "【标准生成】各种算子实现（ReLU、MatMul、LayerNorm 等）",
                "【快速生成】需要快速生成并验证代码的场景",
                "【性能测试】用户要求测试性能，但不需要性能优化（设置 profile 模式）",
                "用户说「生成算子」、「实现算子」、「写一个算子」等",
                "用户说「测一下性能」但没有要求优化（只测试，不优化）"
            ],
            "advantages": [
                "速度快：单次生成，流程简洁（30-90秒）",
                "流程简单：生成 + 验证（可选性能测试）",
                "适用范围广：适合大多数标准算子",
                "资源消耗低：单次 LLM 调用，不迭代优化"
            ],
            "limitations": [
                "不进行性能优化：生成的代码性能可能不是最优",
                "单次生成：不会通过多轮迭代提升性能",
                "如果用户明确要求「高性能」、「性能优化」，应该使用 evolve 或 adaptive_search"
            ],
            "performance": "快速（约 30-90 秒），适合标准算子生成和基础性能测试"
        }
    
    async def execute(self,
                     task_code: str,
                     op_name: str,
                     task_id: str,
                     **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        执行 codeonly workflow
        
        Args:
            task_code: OpTaskBuilder 生成的 task 代码
            op_name: 算子名称
            task_id: 任务 ID
            **kwargs: 其他参数，可选：
                - task_type: 任务类型（"precision_only" 或 "profile"）
                  - "precision_only": 只生成并验证代码（默认）
                  - "profile": 生成代码并进行性能测试
        """
        task_type = kwargs.get("task_type", "precision_only")
        logger.info(f"Executing CodeOnly sub-agent for {op_name}, task_type={task_type}")
        logger.info(f"[RAG] CodeOnlySubAgent.execute: config.rag={self.config.get('rag')}, config keys: {list(self.config.keys()) if self.config else 'None'}")

        try:
            # 延迟导入避免循环依赖
            from ai_kernel_generator.core.langgraph_task import LangGraphTask

            # 使用 LangGraphTask 调用 codeonly workflow
            cfg = dict(self.config or {})
            # 确保保留所有重要参数，特别是 rag
            if "rag" in self.config:
                cfg["rag"] = self.config["rag"]
            task_label = str(kwargs.get("task_label") or "").strip()
            if not task_label:
                from ai_kernel_generator.utils.task_label import resolve_task_label

                task_label = resolve_task_label(
                    op_name=op_name,
                    parallel_index=1,
                )
            if not task_label:
                raise ValueError("[CodeOnlySubAgent] missing task_label")
            cfg["task_label"] = task_label
            logger.info(f"[RAG] CodeOnlySubAgent: passing config with rag={cfg.get('rag')} to LangGraphTask")
            task = LangGraphTask(
                op_name=op_name,
                task_desc=task_code,
                task_id=task_id,
                backend=self.backend,
                arch=self.arch,
                dsl=self.dsl,
                config=cfg,
                framework=self.framework,
                workflow="coder_only_workflow",  # codeonly 对应的 workflow
                task_type=task_type,  # 传递任务类型
            )
            
            # 执行
            final_op_name, success, final_state = await task.run()
            
            # 提取结果
            profile_res = final_state.get("profile_res")
            result = {
                "generated_code": final_state.get("coder_code", ""),
                "verification_result": final_state.get("verifier_result", False),
                "verification_error": final_state.get("verifier_error", ""),
                "profile_result": profile_res,
                "sub_agent_type": "codeonly",  
                "final_state": final_state  # 保存完整状态
            }

            # 如果 codeonly 进行了 profile：把性能结果推送到面板历史（Top 5）
            # 说明：CLI 面板历史依赖 PanelDataMessage(action="move_to_history")。
            if task_type == "profile" and isinstance(profile_res, dict):
                try:
                    gen_time = float(profile_res.get("gen_time") or 0.0)
                    base_time = float(profile_res.get("base_time") or 0.0)
                    speedup = float(profile_res.get("speedup") or 0.0)
                except Exception:
                    gen_time = base_time = speedup = 0.0

                if gen_time > 0.0 or base_time > 0.0 or speedup > 0.0:
                    session_id = str((self.config or {}).get("session_id") or "").strip()
                    if session_id:
                        unique_dir = str(profile_res.get("unique_dir") or "").strip()
                        log_dir = ""
                        base_log_dir = str((self.config or {}).get("log_dir") or "").strip()
                        if unique_dir:
                            if base_log_dir:
                                log_dir = os.path.join(
                                    os.path.expanduser(base_log_dir), op_name, unique_dir
                                )
                            else:
                                log_dir = str(profile_res.get("log_dir") or unique_dir)
                        else:
                            log_dir = str(profile_res.get("log_dir") or "")

                        try:
                            from ai_kernel_generator.cli.runtime.message_sender import (
                                send_message,
                            )
                            from ai_kernel_generator.cli.messages import PanelDataMessage

                            send_message(
                                session_id,
                                PanelDataMessage(
                                    action="move_to_history",
                                    data={
                                        "speedup": speedup,
                                        "gen_time": gen_time,
                                        "base_time": base_time,
                                        "log_dir": log_dir,
                                    },
                                ),
                            )
                            logger.info(
                                f"[panel] Sent codeonly profile to history: speedup={speedup:.2f}x"
                            )
                        except Exception:
                            pass
            
            return success, result
            
        except Exception as e:
            logger.error(f"CodeOnly sub-agent failed: {e}")
            return False, {
                "generated_code": "",
                "verification_result": False,
                "verification_error": str(e),
                "profile_result": None
            }


class EvolveSubAgent(SubAgentBase):
    """
    Evolve 子 Agent
    
    进化优化算子性能
    流程: 多轮迭代优化
    
    """
    
    def get_name(self) -> str:
        return "evolve"
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """返回详细信息用于 LLM 决策"""
        return {
            "name": "evolve",
            "description": "进化式性能优化流程（多轮迭代优化，追求极致性能，耗时较长）",
            "workflow_steps": [
                "初始代码生成: 生成基础算子实现",
                "性能测试: 运行并收集性能数据",
                "Evolve Agent: 分析性能瓶颈，提出优化方案",
                "迭代优化: 多轮演化（5-20轮），持续提升性能",
                "最优选择: 选择性能最佳的版本"
            ],
            "use_cases": [
                "【关键场景】用户明确要求「高性能」、「性能优化」、「极致性能」、「最优性能」",
                "【关键场景】用户说「我要一个高性能的算子」、「帮我优化性能」",
                "用户对性能指标有明确要求（如「加速比大于 2x」）",
                "性能优化空间较大的算子（如矩阵运算、卷积等）",
                "用户允许较长的优化时间（3-10 分钟）以换取更好的性能",
                "需要探索多种优化策略的场景"
            ],
            "advantages": [
                "性能极致：通过多轮迭代寻找最佳方案",
                "自动调优：无需人工介入的性能优化",
                "探索性强：尝试多种优化策略",
                "适应性好：根据实际性能数据调整",
                "可追溯：保留每轮优化的历史"
            ],
            "limitations": [
                "【关键】耗时长：需要多轮迭代，通常需要 3-10 分钟",
                "【关键】如果用户要求「快速生成」或时间有限，应该使用 adaptive_search（更快）",
                "资源消耗高：多次编译、运行、LLM 调用",
                "不确定性：优化效果受迭代次数影响"
            ],
            "performance": "慢（约 180-600 秒），适合对性能要求极高且允许较长优化时间的场景"
        }
    
    def _get_evolve_config_path(self) -> str:
        """
        获取 evolve 配置文件路径
        
        Returns:
            evolve 配置文件的完整路径
        """
        from ai_kernel_generator import get_project_root
        import os
        
        # 统一使用 evolve_config.yaml，不根据 backend 区分
        config_path = os.path.join(get_project_root(), "config", "evolve_config.yaml")
        return config_path
    
    async def execute(self, 
                     task_code: str,
                     op_name: str,
                     task_id: str,
                     **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        执行 evolve 进化优化
        """
        logger.info(f"Executing Evolve sub-agent for {op_name}")
        
        try:
            # 导入 evolve 相关模块
            from ai_kernel_generator.core.evolve import evolve
            from ai_kernel_generator.core.async_pool.task_pool import TaskPool
            from ai_kernel_generator.core.utils import normalize_dsl
            from ai_kernel_generator.utils.common_utils import load_yaml
            from ai_kernel_generator import get_project_root
            import os
            
            # 加载 evolve 专用配置文件
            # 根据 backend 选择对应的配置文件
            evolve_config_file = self._get_evolve_config_path()
            logger.info(f"Loading evolve config from: {evolve_config_file}")
            
            try:
                evolve_yaml = load_yaml(evolve_config_file)
                logger.info(f"✓ Loaded evolve config: {evolve_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load evolve config {evolve_config_file}: {e}")
                logger.warning("Using default config and parameters")
                evolve_yaml = {}
            
            # 从 evolve_config.yaml 读取参数（如果没有则使用默认值）
            evolve_params = evolve_yaml.get("evolve", {})
            island_params = evolve_yaml.get("island", {})
            
            # 获取 evolve 相关参数
            max_rounds = evolve_params.get("max_rounds", 5)
            parallel_num = evolve_params.get("parallel_num", 4)
            
            # 岛屿模型参数（可选，默认禁用）
            num_islands = island_params.get("num_islands", 1)
            migration_interval = island_params.get("migration_interval", 0)
            elite_size = island_params.get("elite_size", 0)
            parent_selection_prob = island_params.get("parent_selection_prob", 0.5)
            handwrite_decay_rate = evolve_params.get("handwrite_decay_rate", 2.0)
            
            logger.info(f"Evolve parameters: max_rounds={max_rounds}, parallel_num={parallel_num}, "
                       f"num_islands={num_islands}, migration_interval={migration_interval}, "
                       f"elite_size={elite_size}")
            
            # 规范化 DSL
            normalized_dsl = normalize_dsl(self.dsl, self.backend)
            logger.info(f"Normalized DSL: {self.dsl} -> {normalized_dsl}")
            
            task_pool = TaskPool(max_concurrency=parallel_num)
            
            logger.info(f"Starting evolve with config: {evolve_config_file}")
            logger.info(f"Parameters: max_rounds={max_rounds}, parallel_num={parallel_num}")
            
            cfg = dict(self.config or {})
            # 确保保留所有重要参数，特别是 rag
            if "rag" in self.config:
                cfg["rag"] = self.config["rag"]
            task_label = str(kwargs.get("task_label") or "").strip()
            if not task_label:
                from ai_kernel_generator.utils.task_label import resolve_task_label

                task_label = resolve_task_label(
                    op_name=op_name,
                    parallel_index=1,
                )
            if not task_label:
                raise ValueError("[EvolveSubAgent] missing task_label")
            cfg["task_label"] = task_label
            logger.info(f"[RAG] EvolveSubAgent: passing config with rag={cfg.get('rag')} to evolve")
            evolution_result = await evolve(
                op_name=op_name,
                task_desc=task_code,
                dsl=normalized_dsl,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                config=cfg,
                task_pool=task_pool,
                max_rounds=max_rounds,
                parallel_num=parallel_num,
                num_islands=num_islands,
                migration_interval=migration_interval,
                elite_size=elite_size,
                parent_selection_prob=parent_selection_prob,
                handwrite_decay_rate=handwrite_decay_rate,
            )
            
            # 判断成功与否
            success = evolution_result.get("successful_tasks", 0) > 0
            best_implementations = evolution_result.get("best_implementations", [])
            
            # 提取最佳实现的代码
            generated_code = ""
            profile_result = None
            if best_implementations:
                best = best_implementations[0]
                generated_code = best.get("code", "")
                profile_result = best.get("profile", {})
            
            # 直接返回 evolve 提供的原始 log_dir（配置路径），由 main_op_agent 负责拼接 op_name
            result = {
                "generated_code": generated_code,
                "verification_result": success,
                "verification_error": "" if success else "No successful implementations",
                "profile_result": profile_result,
                "evolution_history": evolution_result.get("round_results", []),
                "best_implementations": best_implementations,
                "total_rounds": evolution_result.get("total_rounds", 0),
                "total_tasks": evolution_result.get("total_tasks", 0),
                "successful_tasks": evolution_result.get("successful_tasks", 0),
                "final_success_rate": evolution_result.get("final_success_rate", 0.0),
                "storage_dir": evolution_result.get("storage_dir", ""),  # 搜索状态元数据目录
                "log_dir": evolution_result.get("log_dir", ""),  # 基础 log 路径（如 ~/aikg_logs），需要拼接 op_name
                "task_folder": evolution_result.get("task_folder", ""),
                "sub_agent_type": "evolve",  # 标记子 Agent 类型
                "final_state": evolution_result
            }
            
            logger.info(f"Evolve completed: {evolution_result.get('successful_tasks', 0)}/{evolution_result.get('total_tasks', 0)} successful")
            
            return success, result
            
        except Exception as e:
            logger.error(f"Evolve sub-agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, {
                "generated_code": "",
                "verification_result": False,
                "verification_error": str(e),
                "profile_result": None,
                "storage_dir": "",
                "log_dir": "",
                "task_folder": "",
                "sub_agent_type": "evolve"
            }

class KernelVerifierSubAgent(SubAgentBase):
    """
    KernelVerifier 子 Agent
    
    对已生成的算子代码进行性能分析
    流程: 验证 → 性能分析 → 返回结果
    """
    
    def get_name(self) -> str:
        return "kernel_verifier"
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """返回详细信息用于 LLM 决策"""
        return {
            "name": "kernel_verifier",
            "description": "【仅用于已生成代码】对已有 Triton 代码进行性能测试和分析",
            "workflow_steps": [
                "前提检查: 确保已有生成的 Triton 代码",
                "性能测试: 运行多次测试收集性能数据",
                "结果对比: 对比生成代码与原始实现的性能",
                "生成报告: 返回详细的性能分析结果（耗时、加速比等）"
            ],
            "use_cases": [
                "【关键场景】用户在多轮对话中，对前面已生成的 Triton 代码要求测试性能",
                "用户说「测一下性能」、「看看加速比」、「性能怎么样」，且已有生成的代码",
                "用户想知道生成的 Triton 代码是否比原始 Torch 实现更快",
                "需要验证已有代码的优化效果"
            ],
            "advantages": [
                "专注性能测试：只测试，不生成新代码",
                "准确测量：多次运行，结果可靠",
                "完整对比：同时测试原始实现和生成代码",
                "详细报告：提供耗时、加速比等指标"
            ],
            "limitations": [
                "【关键限制】必须在多轮对话中，前面已经使用 codeonly 生成了 Triton 代码",
                "【关键限制】如果用户是第一次对话或还没生成代码，不能使用此 Agent",
                "【关键限制】只做性能测试，不会生成或优化代码",
                "如果用户要求「生成并测性能」，应该使用 codeonly（profile 模式）",
                "如果用户要求「高性能优化」，应该使用 evolve 或 adaptive_search"
            ],
            "performance": "快速（取决于测试轮数），只测试不生成"
        }
    
    async def execute(self, 
                     task_code: str,
                     op_name: str,
                     task_id: str,
                     **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        执行性能分析
        
        Args:
            task_code: OpTaskBuilder 生成的 task 代码（torch 实现）
            op_name: 算子名称
            task_id: 任务 ID
            **kwargs: 其他参数，应包含：
                - generated_code: 之前生成的 triton 代码
                - device_id: 设备 ID（可选，默认 0）
                - profile_settings: 性能测试配置（可选）
        """
        logger.info(f"Executing KernelVerifier sub-agent for {op_name}")
        
        try:
            # 导入必要的模块
            from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
            from ai_kernel_generator.core.worker.manager import get_worker_manager
            from ai_kernel_generator.core.utils import normalize_dsl
            
            # 获取已生成的代码
            generated_code = kwargs.get("generated_code", "")
            if not generated_code:
                error_msg = "No generated code found. Please generate code first before performance analysis."
                logger.error(error_msg)
                return False, {
                    "generated_code": "",
                    "verification_result": False,
                    "verification_error": error_msg,
                    "profile_result": None
                }
            
            # 获取设备 ID
            device_id = kwargs.get("device_id", 0)
            
            # 获取性能测试配置
            profile_settings = kwargs.get("profile_settings", {
                "run_times": 50,
                "warmup_times": 5
            })
            
            # 规范化 DSL
            normalized_dsl = normalize_dsl(self.dsl, self.backend)
            logger.info(f"Normalized DSL: {self.dsl} -> {normalized_dsl}")
            
            # 从 WorkerManager 获取 worker
            worker = await get_worker_manager().select(backend=self.backend, arch=self.arch)
            if not worker:
                error_msg = f"No available worker for backend={self.backend}, arch={self.arch}. Please register a worker first."
                logger.error(error_msg)
                return False, {
                    "generated_code": generated_code,
                    "verification_result": False,
                    "verification_error": error_msg,
                    "profile_result": None
                }
            
            logger.info(f"Using worker: {worker}")
            
            # 创建 KernelVerifier 实例
            impl_func_name = "ModelNew"
            cfg = dict(self.config or {})
            # 确保保留所有重要参数，特别是 rag
            if "rag" in self.config:
                cfg["rag"] = self.config["rag"]
            task_label = str(kwargs.get("task_label") or "").strip()
            if not task_label:
                from ai_kernel_generator.utils.task_label import resolve_task_label

                task_label = resolve_task_label(
                    op_name=op_name,
                    parallel_index=1,
                )
            if not task_label:
                raise ValueError("[KernelVerifierSubAgent] missing task_label")
            cfg["task_label"] = task_label
            logger.info(f"[RAG] KernelVerifierSubAgent: passing config with rag={cfg.get('rag')} to KernelVerifier")
            verifier = KernelVerifier(
                op_name=op_name,
                framework_code=task_code,
                task_id=task_id,
                framework=self.framework,
                dsl=normalized_dsl,
                backend=self.backend,
                arch=self.arch,
                impl_func_name=impl_func_name,
                config=cfg,
                worker=worker
            )
            
            task_info = {"coder_code": generated_code}
            
            # 步骤 1: 先进行验证
            logger.info(f"Step 1: Verifying generated code...")
            verify_result, error_log = await verifier.run(task_info, device_id=device_id)
            
            if not verify_result:
                logger.error(f"Verification failed: {error_log}")
                return False, {
                    "generated_code": generated_code,
                    "verification_result": False,
                    "verification_error": error_log,
                    "profile_result": None
                }
            
            logger.info(f"✓ Verification passed!")
            
            # 步骤 2: 进行性能分析
            logger.info(f"Step 2: Running performance analysis...")
            logger.info(f"Profile settings: run_times={profile_settings['run_times']}, warmup_times={profile_settings['warmup_times']}")
            
            profile_result = await verifier.run_profile(
                task_info, 
                current_step=0, 
                device_id=device_id, 
                profile_settings=profile_settings
            )
            
            # 提取性能数据
            gen_time = profile_result.get('gen_time', 0.0)
            base_time = profile_result.get('base_time', 0.0)
            speedup = profile_result.get('speedup', 0.0)
            
            logger.info(f"✓ Performance analysis completed!")
            logger.info(f"  Original performance: {base_time:.2f} us")
            logger.info(f"  Generated performance: {gen_time:.2f} us")
            logger.info(f"  Speedup: {speedup:.2f}x")
            
            # 立即发送性能结果到历史记录
            if isinstance(profile_result, dict):
                try:
                    gen_time_val = float(gen_time or 0.0)
                    base_time_val = float(base_time or 0.0)
                    speedup_val = float(speedup or 0.0)
                except Exception:
                    gen_time_val = base_time_val = speedup_val = 0.0
                
                if gen_time_val > 0.0 or base_time_val > 0.0 or speedup_val > 0.0:
                    session_id = str(self.config.get("session_id") or "").strip()
                    if session_id:
                        # 构造 log_dir：从 profile_result 中获取 unique_dir
                        unique_dir = profile_result.get('unique_dir', '')
                        log_dir = ""
                        if unique_dir:
                            # 从 config 中获取基础 log_dir，如果没有则使用默认路径
                            base_log_dir = self.config.get('log_dir', '')
                            if base_log_dir:
                                log_dir = os.path.join(os.path.expanduser(base_log_dir), op_name, unique_dir)
                            else:
                                # 如果没有 base_log_dir，尝试从 profile_result 中获取完整路径
                                log_dir = profile_result.get('log_dir', unique_dir)
                        
                        from ai_kernel_generator.cli.runtime.message_sender import send_message
                        from ai_kernel_generator.cli.messages import PanelDataMessage
                        
                        send_message(
                            session_id,
                            PanelDataMessage(
                                action="move_to_history",
                                data={
                                    "speedup": speedup_val,
                                    "gen_time": gen_time_val,
                                    "base_time": base_time_val,
                                    "log_dir": log_dir,
                                },
                            ),
                        )
                        logger.info(f"Sent profile result to history immediately: speedup={speedup_val:.2f}x")
            
            result = {
                "generated_code": generated_code,
                "verification_result": True,
                "verification_error": "",
                "profile_result": profile_result,
                "performance_summary": {
                    "base_time_us": base_time,
                    "gen_time_us": gen_time,
                    "speedup": speedup,
                    "run_times": profile_settings['run_times'],
                    "warmup_times": profile_settings['warmup_times']
                }
            }
            
            return True, result
            
        except Exception as e:
            logger.error(f"KernelVerifier sub-agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, {
                "generated_code": kwargs.get("generated_code", ""),
                "verification_result": False,
                "verification_error": str(e),
                "profile_result": None
            }


class AdaptiveSearchSubAgent(SubAgentBase):
    """
    AdaptiveSearch 子 Agent
    
    使用自适应树搜索优化算子性能
    """
    
    def get_name(self) -> str:
        return "adaptive_search"
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """返回详细信息用于 LLM 决策"""
        return {
            "name": "adaptive_search",
            "description": "基于自适应树搜索的智能优化流程（追求极致性能，比 evolve 更快）",
            "workflow_steps": [
                "初始种群: 并行生成多个初始算子实现",
                "UCB选择: 使用上界置信区间算法选择最有潜力的父代",
                "灵感采样: 从历史成功案例中采样优化灵感",
                "代码生成: 基于父代和灵感生成新的优化版本",
                "性能评估: 并行测试并收集性能数据",
                "迭代优化: 多轮自适应搜索，动态调整策略",
                "最优选择: 返回性能最佳的实现"
            ],
            "use_cases": [
                "【关键场景】用户要求「高性能」且希望「比 evolve 更快」、「快速优化」",
                "【关键场景】用户说「我要高性能算子，但时间不要太长」",
                "【关键场景】用户明确提到「树搜索」或「adaptive search」",
                "复杂算子的性能优化（如矩阵运算、卷积等）",
                "需要探索大量优化策略组合的场景",
                "对性能有明确指标要求，但希望比 evolve 更快完成"
            ],
            "advantages": [
                "【关键优势】比 evolve 更快：通过并行搜索和智能选择，通常比 evolve 快 30-50%",
                "智能搜索: 基于 UCB 算法，自动平衡探索与利用",
                "灵感驱动: 从历史成功案例学习优化模式",
                "高效收敛: 相比 evolve 的随机演化，更快找到最优解",
                "并行能力强: 支持高并发搜索，充分利用计算资源",
                "可追溯: 保留完整的搜索树和谱系图"
            ],
            "limitations": [
                "【关键】仍需要时间：虽然比 evolve 快，但仍需要 2-5 分钟",
                "【关键】如果用户只是要「生成代码」，不需要性能优化，应该使用 codeonly",
                "资源消耗高: 需要多次并发编译和运行测试",
                "复杂度高: 涉及 UCB 选择、灵感采样等复杂逻辑"
            ],
            "performance": "中等偏快（约 120-300 秒），比 evolve 快 30-50%，但仍需一定时间"
        }
    
    def _load_adaptive_search_config(self) -> Dict[str, Any]:
        """
        加载自适应搜索配置
        
        Returns:
            配置字典，包含所有自适应搜索参数
        """
        from ai_kernel_generator.utils.common_utils import load_yaml
        from ai_kernel_generator import get_project_root
        import os
        
        # 默认配置
        default_config = {
            "max_concurrent": 4,
            "initial_task_count": 4,
            "tasks_per_parent": 1,  #
            "max_total_tasks": 50,
            "exploration_coef": 1.414,
            "random_factor": 0.1,
            "use_softmax": False,
            "softmax_temperature": 1.0,
            "inspiration_sample_num": 3,
            "use_tiered_sampling": True,
            "handwrite_sample_num": 2,
            "handwrite_decay_rate": 2.0
        }
        
        # 尝试加载自适应搜索配置文件
        config_path = os.path.join(get_project_root(), "config", "adaptive_search_config.yaml")
        
        if not os.path.exists(config_path):
            logger.warning(f"Adaptive search config not found at {config_path}, using defaults")
            return default_config
        
        try:
            # 直接加载 yaml 配置
            config_dict = load_yaml(config_path)
            logger.info(f"✓ Loaded adaptive search config from: {config_path}")
            
            # 提取各个配置部分
            concurrency_config = config_dict.get("concurrency", {})
            stopping_config = config_dict.get("stopping", {})
            ucb_config = config_dict.get("ucb_selection", {})
            inspiration_config = config_dict.get("inspiration", {})
            handwrite_config = config_dict.get("handwrite", {})
            
            loaded_config = {
                # 并发配置
                "max_concurrent": concurrency_config.get("max_concurrent", default_config["max_concurrent"]),
                "initial_task_count": concurrency_config.get("initial_task_count", default_config["initial_task_count"]),
                "tasks_per_parent": concurrency_config.get("tasks_per_parent", default_config["tasks_per_parent"]), 
                
                # 停止条件
                "max_total_tasks": stopping_config.get("max_total_tasks", default_config["max_total_tasks"]),
                
                # UCB 参数
                "exploration_coef": ucb_config.get("exploration_coef", default_config["exploration_coef"]),
                "random_factor": ucb_config.get("random_factor", default_config["random_factor"]),
                "use_softmax": ucb_config.get("use_softmax", default_config["use_softmax"]),
                "softmax_temperature": ucb_config.get("softmax_temperature", default_config["softmax_temperature"]),
                
                # 灵感采样参数
                "inspiration_sample_num": inspiration_config.get("sample_num", default_config["inspiration_sample_num"]),
                "use_tiered_sampling": inspiration_config.get("use_tiered_sampling", default_config["use_tiered_sampling"]),
                
                # 手写建议参数
                "handwrite_sample_num": handwrite_config.get("sample_num", default_config["handwrite_sample_num"]),
                "handwrite_decay_rate": handwrite_config.get("decay_rate", default_config["handwrite_decay_rate"])
            }
            
            logger.info(f"Adaptive search config loaded successfully:")
            logger.info(f"  - max_concurrent: {loaded_config['max_concurrent']}")
            logger.info(f"  - initial_task_count: {loaded_config['initial_task_count']}")
            logger.info(f"  - tasks_per_parent: {loaded_config['tasks_per_parent']}")
            logger.info(f"  - max_total_tasks: {loaded_config['max_total_tasks']}")
            
            return loaded_config
            
        except Exception as e:
            logger.error(f"Failed to load adaptive search config from {config_path}: {e}")
            logger.warning("Using default configuration")
            return default_config
    
    async def execute(self, 
                     task_code: str,
                     op_name: str,
                     task_id: str,
                     **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        执行自适应搜索优化
        
        Args:
            task_code: OpTaskBuilder 生成的 task 代码
            op_name: 算子名称
            task_id: 任务 ID
            **kwargs: 其他参数
        """
        logger.info(f"Executing AdaptiveSearch sub-agent for {op_name}")
        
        try:
            # 导入自适应搜索模块
            from ai_kernel_generator.core.adaptive_search import adaptive_search
            from ai_kernel_generator.core.utils import normalize_dsl
            
            # 加载自适应搜索配置
            search_config = self._load_adaptive_search_config()
            
            logger.info(f"Adaptive search parameters:")
            logger.info(f"  - max_concurrent: {search_config['max_concurrent']}")
            logger.info(f"  - initial_task_count: {search_config['initial_task_count']}")
            logger.info(f"  - tasks_per_parent: {search_config['tasks_per_parent']}")
            logger.info(f"  - max_total_tasks: {search_config['max_total_tasks']}")
            logger.info(f"  - exploration_coef: {search_config['exploration_coef']}")
            logger.info(f"  - inspiration_sample_num: {search_config['inspiration_sample_num']}")
            
            # 规范化 DSL
            normalized_dsl = normalize_dsl(self.dsl, self.backend)
            logger.info(f"Normalized DSL: {self.dsl} -> {normalized_dsl}")
            
            # 准备配置
            cfg = dict(self.config or {})
            if "rag" in self.config:
                cfg["rag"] = self.config["rag"]
            task_label = str(kwargs.get("task_label") or "").strip()
            if not task_label:
                from ai_kernel_generator.utils.task_label import resolve_task_label
                task_label = resolve_task_label(
                    op_name=op_name,
                    parallel_index=1,
                )
            if not task_label:
                raise ValueError("[AdaptiveSearchSubAgent] missing task_label")
            cfg["task_label"] = task_label
            
            logger.info(f"[RAG] AdaptiveSearchSubAgent: passing config with rag={cfg.get('rag')} to adaptive_search")
            logger.info(f"Starting adaptive search for {op_name}...")
            
            # 执行自适应搜索
            search_result = await adaptive_search(
                op_name=op_name,
                task_desc=task_code,
                dsl=normalized_dsl,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                config=cfg,
                
                # 并发控制
                max_concurrent=search_config["max_concurrent"],
                initial_task_count=search_config["initial_task_count"],
                tasks_per_parent=search_config["tasks_per_parent"],  
                
                # 停止条件
                max_total_tasks=search_config["max_total_tasks"],
                
                # UCB 参数
                exploration_coef=search_config["exploration_coef"],
                random_factor=search_config["random_factor"],
                use_softmax=search_config["use_softmax"],
                softmax_temperature=search_config["softmax_temperature"],
                
                # 灵感采样参数
                inspiration_sample_num=search_config["inspiration_sample_num"],
                use_tiered_sampling=search_config["use_tiered_sampling"],
                handwrite_sample_num=search_config["handwrite_sample_num"],
                handwrite_decay_rate=search_config["handwrite_decay_rate"]
            )
            
            # 判断成功与否
            success = search_result.get("total_success", 0) > 0
            best_implementations = search_result.get("best_implementations", [])
            
            # 提取最佳实现的代码和性能数据
            generated_code = ""
            profile_result = None
            if best_implementations:
                best = best_implementations[0]
                generated_code = best.get("code", "")
                profile_result = best.get("profile", {})
            
            # 直接返回 adaptive_search 提供的原始 log_dir（配置路径），由 main_op_agent 负责拼接 op_name
            result = {
                "generated_code": generated_code,
                "verification_result": success,
                "verification_error": "" if success else "No successful implementations found",
                "profile_result": profile_result,
                "best_implementations": best_implementations,
                "total_submitted": search_result.get("total_submitted", 0),
                "total_completed": search_result.get("total_completed", 0),
                "total_success": search_result.get("total_success", 0),
                "total_failed": search_result.get("total_failed", 0),
                "success_rate": search_result.get("success_rate", 0.0),
                "elapsed_time": search_result.get("elapsed_time", 0.0),
                "stop_reason": search_result.get("stop_reason", ""),
                "storage_dir": search_result.get("storage_dir", ""),  # 搜索状态元数据目录
                "log_dir": search_result.get("log_dir", ""),  
                "task_folder": search_result.get("task_folder", ""),
                "lineage_graph": search_result.get("lineage_graph", ""),
                "sub_agent_type": "adaptive_search",  
                "final_state": search_result
            }
            
            logger.info(f"Adaptive search completed:")
            logger.info(f"  - Success: {search_result.get('total_success', 0)}/{search_result.get('total_completed', 0)}")
            logger.info(f"  - Success rate: {search_result.get('success_rate', 0.0):.1%}")
            logger.info(f"  - Elapsed time: {search_result.get('elapsed_time', 0.0):.1f}s")
            logger.info(f"  - Stop reason: {search_result.get('stop_reason', '')}")
            
            return success, result
            
        except Exception as e:
            logger.error(f"AdaptiveSearch sub-agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, {
                "generated_code": "",
                "verification_result": False,
                "verification_error": str(e),
                "profile_result": None,
                "storage_dir": "",
                "log_dir": "",
                "task_folder": "",
                "sub_agent_type": "adaptive_search"
            }


class OpTaskBuilderSubAgent(SubAgentBase):

    
    def __init__(self, 
                 config: dict,
                 framework: str = "torch",
                 backend: str = "cuda",
                 arch: str = "a100",
                 dsl: str = "triton"):
        super().__init__(config, framework, backend, arch, dsl)
        self._op_task_builder = None  
    
    def _get_op_task_builder(self):
        if self._op_task_builder is None:
            from ai_kernel_generator.core.agent.op_task_builder import OpTaskBuilder
            self._op_task_builder = OpTaskBuilder(config=self.config)
            self._op_task_builder.framework = self.framework
            self._op_task_builder.backend = self.backend
            self._op_task_builder.arch = self.arch
            self._op_task_builder.dsl = self.dsl
        return self._op_task_builder
    
    def get_name(self) -> str:
        return "op_task_builder"
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """返回详细信息用于 LLM 决策"""
        return {
            "name": "op_task_builder",
            "description": "将用户的自然语言需求转换为 KernelBench 格式的 Torch task 代码",
            "workflow_steps": [
                "理解需求: 分析用户的算子需求描述",
                "生成代码: 生成 KernelBench 格式的 task_desc 代码",
                "代码检查: 静态检查和运行时检查代码正确性",
                "返回结果: 返回生成的 task_desc 供后续子 Agent 使用"
            ],
            "use_cases": [
                "【首要调用】用户描述算子需求，但没有提供 task 代码",
                "用户说「生成 ReLU 算子」、「实现矩阵乘法」等自然语言需求",
                "用户要求修改已生成的 task 代码",
                "需要将自然语言转换为 KernelBench 格式代码"
            ],
            "advantages": [
                "自然语言理解：能理解用户的算子需求描述",
                "格式标准：生成标准的 KernelBench 格式代码",
                "代码验证：自动进行静态和运行时检查",
                "支持修改：可根据用户反馈修改已生成的代码"
            ],
            "limitations": [
                "仅生成 task_desc：不生成最终的 Triton 代码",
                "需要后续处理：生成的 task_desc 需要传给其他子 Agent 生成 Triton 代码"
            ],
            "performance": "快速（约 10-30 秒），主要耗时在 LLM 理解和代码验证"
        }
    
    async def execute(self, 
                     task_code: str,
                     op_name: str,
                     task_id: str,
                     **kwargs) -> Tuple[bool, Dict[str, Any]]:
        user_request = kwargs.get("user_request", "")
        user_feedback = kwargs.get("user_feedback", "")
        
        if not user_request:
            logger.error("OpTaskBuilder: user_request is required")
            return False, {
                "status": "ERROR",
                "generated_task_desc": "",
                "op_name": op_name or "",
                "agent_message": "缺少 user_request 参数，无法生成 task_desc",
                "sub_agent_type": "op_task_builder"
            }
        
        logger.info(f"Executing OpTaskBuilder sub-agent, user_request: {user_request[:50]}...")
        
        try:
            op_task_builder = self._get_op_task_builder()
            
            state = {
                "user_input": user_request,
                "user_feedback": user_feedback,
                "generated_task_desc": task_code, 
                "framework": self.framework,
                "backend": self.backend,
                "arch": self.arch,
                "dsl": self.dsl,
            }
            
            # 调用 OpTaskBuilder.run()
            result = await op_task_builder.run(state)
            
            from ai_kernel_generator.utils.langgraph.op_task_builder_state import OpTaskBuilderStatus
            status = result.get("status", OpTaskBuilderStatus.NEED_CLARIFICATION)
            success = (status == OpTaskBuilderStatus.READY)
            
            # 添加 sub_agent_type 标识
            result["sub_agent_type"] = "op_task_builder"
            
            return success, result
            
        except Exception as e:
            logger.error(f"OpTaskBuilder sub-agent failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, {
                "status": "ERROR",
                "generated_task_desc": "",
                "op_name": op_name or "",
                "agent_message": f"生成 task_desc 失败：{str(e)}",
                "sub_agent_type": "op_task_builder"
            }


class SubAgentRegistry:
    """
    
    管理所有可用的子 Agent
    """
    
    def __init__(self):
        """初始化注册中心"""
        self._agents: Dict[str, type] = {}
        
        # 自动注册内置的子 Agent
        self._register_builtin_agents()
    
    def _register_builtin_agents(self):
        """注册内置的子 Agent"""
        self.register(OpTaskBuilderSubAgent)
        self.register(CodeOnlySubAgent)
        self.register(EvolveSubAgent)
        self.register(KernelVerifierSubAgent)
        self.register(AdaptiveSearchSubAgent)
        
        logger.info(f"Registered {len(self._agents)} built-in sub-agents")
    
    def register(self, agent_class: type):
        """
        注册一个子 Agent
        
        Args:
            agent_class: 子 Agent 类
        """
        if not issubclass(agent_class, SubAgentBase):
            raise ValueError(f"{agent_class} must inherit from SubAgentBase")
        
        temp_instance = agent_class(config={})
        agent_name = temp_instance.get_name()
        
        if agent_name in self._agents:
            logger.warning(f"Sub-agent '{agent_name}' already registered, overwriting")
        
        self._agents[agent_name] = agent_class
        logger.info(f"Registered sub-agent: {agent_name}")
    
    def get_agent(self, 
                  agent_name: str,
                  config: dict,
                  framework: str = "torch",
                  backend: str = "cuda",
                  arch: str = "a100",
                  dsl: str = "triton") -> Optional[SubAgentBase]:
        """
        获取子 Agent 实例
        
        Args:
            agent_name: 子 Agent 名称
            config: 配置字典
            framework: 框架类型
            backend: 后端类型
            arch: 硬件架构
            dsl: DSL 类型
            
        Returns:
            子 Agent 实例，如果不存在返回 None
        """
        agent_class = self._agents.get(agent_name)
        if agent_class is None:
            logger.error(f"Sub-agent '{agent_name}' not found")
            return None
        
        return agent_class(
            config=config,
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl
        )
    
    def list_agents(self) -> Dict[str, str]:
        """
        列出所有可用的子 Agent
        
        Returns:
            Dict[agent_name, description]
        """
        result = {}
        for agent_name, agent_class in self._agents.items():
            temp_instance = agent_class(config={})
            detailed_info = temp_instance.get_detailed_info()
            result[agent_name] = detailed_info.get("description", "")
        return result
    
    def get_agents_detailed_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有子 Agent 的详细信息（用于 LLM 决策）
        
        Returns:
            Dict[agent_name, detailed_info]
        """
        result = {}
        for agent_name, agent_class in self._agents.items():
            temp_instance = agent_class(config={})
            result[agent_name] = temp_instance.get_detailed_info()
        return result
    
    def is_registered(self, agent_name: str) -> bool:
        """检查子 Agent 是否已注册"""
        return agent_name in self._agents


# 全局注册中心实例
_global_registry = SubAgentRegistry()


def get_registry() -> SubAgentRegistry:
    """获取全局注册中心"""
    return _global_registry


def register_sub_agent(agent_class: type):
    _global_registry.register(agent_class)
    return agent_class
