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

"""Evolve Workflow: 进化式算子生成"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.core_v2.workflows.registry import register_workflow

logger = logging.getLogger(__name__)


@register_workflow(scopes=["op"])
class EvolveWorkflow(OpBaseWorkflow):
    """Evolve Workflow：进化式算子生成工作流
    
    将 evolve() 异步函数封装为 LangGraph 单节点 workflow，
    以便通过 KernelAgent 的 ToolExecutor 统一调用。
    
    特点：
    - 使用进化算法进行多轮并行搜索
    - 支持岛屿模型和精英池
    - 自动选择父代进行变异
    """
    
    TOOL_NAME = "call_evolve_workflow"
    
    DESCRIPTION = """
使用进化式搜索生成优化的 kernel 代码。采用岛屿隔离模型，每轮同步迭代，搜索多样性更强，但速度慢于自适应搜索（adaptive_search）。

核心特点：
- **同步迭代**：每一轮需要等待所有并行任务全部完成后才进入下一轮选择-变异，轮间存在等待开销
- **岛屿隔离模型**：将种群划分为多个独立岛屿，各岛屿独立进化并定期迁移精英个体，保持种群多样性，避免早熟收敛
- **精英池机制**：维护全局精英池，从多个岛屿汇聚最优个体，变异来源更丰富

完整流程：
1. 初始化种群（并行生成多个初始实现，分配到各岛屿）
2. 评估适应度（验证 + 性能测试）
3. 选择和变异（基于精英池和岛屿内最佳实现生成新变体）
4. 等待本轮所有任务完成后，进入下一轮迭代
5. 定期在岛屿间迁移精英个体

适用场景：
- 有充足的计算资源和设备可以大规模并行搜索
- 搜索空间复杂，需要维持种群多样性防止陷入局部最优
- 对搜索时间不敏感，追求更广泛的解空间探索

注意事项：
- 执行时间较长，且因同步等待通常比自适应搜索更慢
- 需要多个可用设备进行并行验证
- 会生成大量中间文件
"""
    
    # ---- 进化参数默认值 ----
    _EVOLVE_DEFAULTS: Dict[str, Any] = {
        "max_rounds": 3,
        "parallel_num": 4,
        "num_islands": 2,
        "migration_interval": 2,
        "elite_size": 5,
        "parent_selection_prob": 0.5,
    }
    
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {
                "type": "string",
                "description": "算子名称"
            },
            "task_desc": {
                "type": "string",
                "description": "任务描述（框架代码）"
            },
            "dsl": {
                "type": "string",
                "description": "DSL 类型，如 'triton_ascend', 'triton_cuda'"
            },
            "framework": {
                "type": "string",
                "description": "框架，如 'torch'"
            },
            "backend": {
                "type": "string",
                "description": "后端，如 'cuda', 'ascend'"
            },
            "arch": {
                "type": "string",
                "description": "架构，如 'a100', 'ascend910b4'"
            },
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"]
    }
    
    def build_graph(self) -> StateGraph:
        """构建进化工作流图（单节点封装）"""
        workflow = StateGraph(KernelGenState)
        
        # 捕获 self 到闭包
        _self = self
        
        async def evolve_node(state: KernelGenState) -> dict:
            """进化节点：调用 evolve() 完成整个进化过程
            
            进化参数从 _self.config 中读取（由 prepare_config 设置默认值）。
            """
            from akg_agents.op.evolve import evolve
            from akg_agents.core.async_pool.task_pool import TaskPool
            from akg_agents.core.worker.manager import get_worker_manager, register_local_worker
            
            op_name = state.get("op_name", "")
            task_desc = state.get("task_desc", "")
            dsl = state.get("dsl", "")
            framework = state.get("framework", "")
            backend = state.get("backend", "")
            arch = state.get("arch", "")
            
            # 从 config 读取进化参数（prepare_config 已通过 _EVOLVE_DEFAULTS 设置好默认值）
            defaults = type(_self)._EVOLVE_DEFAULTS
            cfg = _self.config
            max_rounds = cfg.get("max_rounds", defaults["max_rounds"])
            parallel_num = cfg.get("parallel_num", defaults["parallel_num"])
            num_islands = cfg.get("num_islands", defaults["num_islands"])
            migration_interval = cfg.get("migration_interval", defaults["migration_interval"])
            elite_size = cfg.get("elite_size", defaults["elite_size"])
            parent_selection_prob = cfg.get("parent_selection_prob", defaults["parent_selection_prob"])
            
            logger.info(
                f"[EvolveWorkflow] 开始进化搜索: op_name={op_name}, backend={backend}, "
                f"max_rounds={max_rounds}, parallel_num={parallel_num}, "
                f"num_islands={num_islands}, migration_interval={migration_interval}, "
                f"elite_size={elite_size}"
            )
            
            # 确保 Worker 可用
            wm = get_worker_manager()
            if not await wm.has_worker(backend=backend, arch=arch):
                await register_local_worker([0], backend=backend, arch=arch)
                logger.info(f"[EvolveWorkflow] 已注册本地 worker: backend={backend}, arch={arch}")
            
            # 创建 TaskPool
            task_pool = TaskPool(max_concurrency=parallel_num)
            
            try:
                result = await evolve(
                    op_name=op_name,
                    task_desc=task_desc,
                    dsl=dsl,
                    framework=framework,
                    backend=backend,
                    arch=arch,
                    config=_self.config,
                    task_pool=task_pool,
                    max_rounds=max_rounds,
                    parallel_num=parallel_num,
                    num_islands=num_islands,
                    migration_interval=migration_interval,
                    elite_size=elite_size,
                    parent_selection_prob=parent_selection_prob,
                )
                
                # 提取最佳代码
                best_code = ""
                best_profile = {}
                if result.get("best_implementations"):
                    best = result["best_implementations"][0]
                    best_code = best.get("impl_code", "")
                    best_profile = best.get("profile", {})
                
                success = result.get("successful_tasks", 0) > 0
                
                return {
                    "coder_code": best_code,
                    "verifier_result": success,
                    "profile_res": best_profile,
                }
            except Exception as e:
                logger.error(f"[EvolveWorkflow] 进化搜索失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    "verifier_result": False,
                    "verifier_error": str(e),
                }
        
        workflow.add_node("evolve", evolve_node)
        workflow.set_entry_point("evolve")
        workflow.add_edge("evolve", END)
        
        return workflow
    
    @classmethod
    def prepare_config(cls, workflow_resources: Dict[str, Any], arguments: Dict[str, Any]):
        """根据 arguments 调整 workflow 配置
        
        1. 使用 build_langgraph_task_config() 补齐 docs_dir、agent_model_config 等
        2. 设置内部 LangGraphTask 使用 kernelgen_only_workflow（基于 Skill 系统的 KernelGen）
        3. 调用父类 prepare_config 完成 log_dir 重定向 + 创建 WorkflowLogger
        4. 用 _EVOLVE_DEFAULTS 补齐进化参数的默认值
        """
        # 补齐 LangGraphTask 所需的完整配置
        dsl = arguments.get("dsl", "")
        backend = arguments.get("backend", "")
        op_name = arguments.get("op_name", "")

        base_config = workflow_resources.get("config") or {}
        full_config = cls.build_langgraph_task_config(
            dsl=dsl,
            backend=backend,
            op_name=op_name,
            base_config=base_config,
        )

        full_config["default_workflow"] = "kernelgen_only_workflow"
        full_config["enable_sketch_generation"] = False
        workflow_resources["config"] = full_config
        
        # 调用父类：log_dir 重定向 + 创建 WorkflowLogger
        super().prepare_config(workflow_resources, arguments)
        
        # 用 _EVOLVE_DEFAULTS 补齐进化参数的默认值
        config = workflow_resources["config"]
        for key, default_val in cls._EVOLVE_DEFAULTS.items():
            config.setdefault(key, default_val)
    
    def format_result(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """格式化进化搜索结果"""
        code = final_state.get("coder_code", "")
        profile_res = final_state.get("profile_res", {})
        verifier_result = final_state.get("verifier_result", False)

        status = "success" if verifier_result else "fail"

        res = {
            "code": code,
            "profile": str(profile_res) if profile_res else "",
            "status": status,
        }

        if not verifier_result:
            verifier_error = final_state.get("verifier_error", "")
            conductor_suggestion = final_state.get("conductor_suggestion", "")
            if verifier_error:
                res["error_information"] = verifier_error
            if conductor_suggestion:
                res["conductor_suggestion"] = conductor_suggestion

        return res
