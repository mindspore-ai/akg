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
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from ai_kernel_generator.core.langgraph_task import LangGraphTask

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
            "description": "直接生成代码的标准流程",
            "workflow_steps": [
                "Coder: 使用 LLM 直接生成高性能算子代码",
                "Verifier: 编译、运行、验证生成的代码"
            ],
            "use_cases": [
                "各种算子实现（简单到中等复杂度）",
                "标准的数学运算算子（如 ReLU、Add、Mul、MatMul 等）",
                "大部分常见算子（如激活函数、归一化层等）",
                "需要快速生成和验证的场景",
                "算子逻辑清晰、实现直接的情况"
            ],
            "advantages": [
                "速度适中：直接生成代码，流程简洁",
                "流程简单：仅两个步骤（生成 + 验证）",
                "适用范围广：适合大多数标准算子",
                "资源消耗合理：LLM 调用次数适中",
                "稳定：基于 Coder Agent"
            ],
            "limitations": [
                "不支持迭代优化：无法通过多轮优化提升性能",
                "性能可能不是最优：相比 evolve，缺少性能调优过程"
            ],
            "performance": "适中（约 30-90 秒），适合大部分算子的标准生成场景"
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
        # 获取任务类型（默认为 precision_only）
        task_type = kwargs.get("task_type", "precision_only")
        logger.info(f"Executing CodeOnly sub-agent for {op_name}, task_type={task_type}")

        try:
            # 延迟导入避免循环依赖
            from ai_kernel_generator.core.langgraph_task import LangGraphTask

            # 使用 LangGraphTask 调用 codeonly workflow
            task = LangGraphTask(
                op_name=op_name,
                task_desc=task_code,
                task_id=task_id,
                backend=self.backend,
                arch=self.arch,
                dsl=self.dsl,
                config=self.config,
                framework=self.framework,
                workflow="coder_only_workflow",  # codeonly 对应的 workflow
                task_type=task_type,  # 传递任务类型
            )
            
            # 执行
            final_op_name, success, final_state = await task.run()
            
            # 提取结果
            result = {
                "generated_code": final_state.get("coder_code", ""),
                "verification_result": final_state.get("verifier_result", False),
                "verification_error": final_state.get("verifier_error", ""),
                "profile_result": final_state.get("profile_res"),
                "final_state": final_state  # 保存完整状态
            }
            
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
            "description": "进化式性能优化流程",
            "workflow_steps": [
                "初始代码生成: 生成基础算子实现",
                "性能测试: 运行并收集性能数据",
                "Evolve Agent: 分析性能瓶颈，提出优化方案",
                "迭代优化: 多轮演化，持续提升性能",
                "最优选择: 选择性能最佳的版本"
            ],
            "use_cases": [
                "性能要求极高的算子",
                "需要探索多种优化策略的场景",
                "性能优化空间较大的算子",
                "需要自动调优的算子",
                "已有基础实现但需要进一步优化"
            ],
            "advantages": [
                "性能最优：通过多轮迭代寻找最佳方案",
                "自动调优：无需人工介入的性能优化",
                "探索性强：尝试多种优化策略",
                "适应性好：根据实际性能数据调整",
                "可追溯：保留每轮优化的历史"
            ],
            "limitations": [
                "时间长：需要多轮迭代，耗时最多",
                "资源消耗极高：多次编译、运行、LLM 调用",
                "不确定性：优化效果受迭代次数影响",
                "可能过拟合：针对特定 shape 的优化可能不通用",
                "复杂度高：流程最复杂"
            ],
            "performance": "非常慢（约 180-600 秒，取决于迭代次数），适合对性能要求极高且允许较长优化时间的场景"
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
            
            evolution_result = await evolve(
                op_name=op_name,
                task_desc=task_code,
                dsl=normalized_dsl,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                config=self.config,
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
                "storage_dir": evolution_result.get("storage_dir", ""),
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
                "profile_result": None
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
            "description": "对已生成的算子代码进行性能分析",
            "workflow_steps": [
                "代码验证: 确保生成的代码可以正确编译和运行",
                "性能测试: 运行多次测试收集性能数据",
                "结果对比: 对比生成代码与原始实现的性能",
                "生成报告: 返回详细的性能分析结果（耗时、加速比等）"
            ],
            "use_cases": [
                "用户想了解生成算子的性能表现",
                "用户明确要求进行性能测试或性能分析",
                "用户询问生成代码的速度、加速比、性能对比",
                "用户想知道生成的 triton 代码是否比原始torch的描述实现更快",
                "需要验证优化效果的场景"
            ],
            "advantages": [
                "准确测量：多次运行，结果可靠",
                "完整对比：同时测试原始实现和生成代码",
                "详细报告：提供耗时、加速比等指标",
                "跨设备支持：支持 CUDA 和 Ascend 等不同硬件",
                "自动化：无需手动编写性能测试脚本"
            ],
            "limitations": [
                "【重要前置条件】必须先使用 codeonly生成 triton 代码后才能使用",
                "【重要前置条件】必须已有 torch task 代码（由 OpTaskBuilder 生成）",
                "只做性能分析：不会生成或修改代码，只测试已有代码的性能",
                "需要硬件资源：需要实际的 GPU/NPU 设备进行测试",
                "如果用户还没生成代码就要求性能测试，应该先引导用户生成代码"
            ],
            "performance": "取决于算子复杂度和测试轮数"
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
            verifier = KernelVerifier(
                op_name=op_name,
                framework_code=task_code,
                task_id=task_id,
                framework=self.framework,
                dsl=normalized_dsl,
                backend=self.backend,
                arch=self.arch,
                impl_func_name=impl_func_name,
                config=self.config,
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


class SubAgentRegistry:
    """
    子 Agent 注册中心
    
    管理所有可用的子 Agent
    """
    
    def __init__(self):
        """初始化注册中心"""
        self._agents: Dict[str, type] = {}
        
        # 自动注册内置的子 Agent
        self._register_builtin_agents()
    
    def _register_builtin_agents(self):
        """注册内置的子 Agent"""
        self.register(CodeOnlySubAgent)
        self.register(EvolveSubAgent)
        self.register(KernelVerifierSubAgent)
        
        logger.info(f"Registered {len(self._agents)} built-in sub-agents")
    
    def register(self, agent_class: type):
        """
        注册一个子 Agent
        
        Args:
            agent_class: 子 Agent 类（必须继承 SubAgentBase）
        """
        if not issubclass(agent_class, SubAgentBase):
            raise ValueError(f"{agent_class} must inherit from SubAgentBase")
        
        # 创建临时实例获取名称
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

