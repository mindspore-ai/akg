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

"""Adaptive Search Workflow: 自适应搜索式算子生成"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.core_v2.workflows.registry import register_workflow

logger = logging.getLogger(__name__)


@register_workflow(scopes=["op"])
class AdaptiveSearchWorkflow(OpBaseWorkflow):
    """Adaptive Search Workflow：自适应搜索式算子生成工作流
    
    将 adaptive_search() 异步函数封装为 LangGraph 单节点 workflow，
    以便通过 KernelAgent 的 ToolExecutor 统一调用。
    
    特点：
    - 使用 UCB 选择策略自适应探索
    - 基于历史结果动态调整搜索方向
    - 支持并发任务执行
    """
    
    TOOL_NAME = "call_adaptive_search_workflow"
    
    DESCRIPTION = """
使用自适应搜索生成优化的 kernel 代码。相比进化搜索（evolve），自适应搜索完成速度更快、资源利用率更高。

核心优势：
- **即时递补**：任意任务一完成，立即基于当前最优父代启动新任务，无需等待同批次其他任务结束，最大化资源利用率
- **UCB 智能选择**：每次扩展时基于 UCB（Upper Confidence Bound）策略选择最有潜力的父代进化，自动平衡探索与利用，趋向极致性能
- **收敛更快**：持续滚动式搜索，相同时间内可完成更多轮迭代，更快找到高性能实现

完整流程：
1. 初始化阶段：并行生成多个初始实现
2. 评估阶段：验证正确性 + 性能测试
3. 选择阶段：基于 UCB 策略选择最有潜力的父代
4. 扩展阶段：基于父代生成新的变体
5. 任务完成即触发下一轮选择-扩展，持续滚动直到达到最大任务数

适用场景：
- 希望在有限时间内尽快找到高质量实现（推荐首选）
- 设备资源有限，需要高效利用每个计算槽位
- 需要智能探索-利用平衡

注意事项：
- 执行时间较长（通常 10-30 分钟）
- 需要可用的验证环境
"""
    
    # ---- 搜索参数默认值 ----
    _SEARCH_DEFAULTS: Dict[str, Any] = {
        "max_total_tasks": 10,
        "max_concurrent": 2,
        "initial_task_count": 2,
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
        """构建自适应搜索工作流图（单节点封装）"""
        workflow = StateGraph(KernelGenState)
        
        # 捕获 self 到闭包
        _self = self
        
        async def adaptive_search_node(state: KernelGenState) -> dict:
            """自适应搜索节点：调用 adaptive_search() 完成整个搜索过程
            
            搜索参数从 _self.config 中读取（由 prepare_config 设置默认值）。
            """
            from akg_agents.op.adaptive_search.adaptive_search import adaptive_search
            from akg_agents.core.worker.manager import get_worker_manager, register_local_worker
            
            op_name = state.get("op_name", "")
            task_desc = state.get("task_desc", "")
            dsl = state.get("dsl", "")
            framework = state.get("framework", "")
            backend = state.get("backend", "")
            arch = state.get("arch", "")
            
            # 从 config 读取搜索参数（prepare_config 已通过 _SEARCH_DEFAULTS 设置好默认值）
            defaults = type(_self)._SEARCH_DEFAULTS
            cfg = _self.config
            max_total_tasks = cfg.get("max_total_tasks", defaults["max_total_tasks"])
            max_concurrent = cfg.get("max_concurrent", defaults["max_concurrent"])
            initial_task_count = cfg.get("initial_task_count", defaults["initial_task_count"])
            
            logger.info(
                f"[AdaptiveSearchWorkflow] 开始自适应搜索: op_name={op_name}, backend={backend}, "
                f"max_total_tasks={max_total_tasks}, max_concurrent={max_concurrent}, "
                f"initial_task_count={initial_task_count}"
            )
            
            # 确保 Worker 可用
            wm = get_worker_manager()
            if not await wm.has_worker(backend=backend, arch=arch):
                await register_local_worker([0], backend=backend, arch=arch)
                logger.info(f"[AdaptiveSearchWorkflow] 已注册本地 worker: backend={backend}, arch={arch}")
            
            try:
                result = await adaptive_search(
                    op_name=op_name,
                    task_desc=task_desc,
                    dsl=dsl,
                    framework=framework,
                    backend=backend,
                    arch=arch,
                    config=_self.config,
                    max_total_tasks=max_total_tasks,
                    max_concurrent=max_concurrent,
                    initial_task_count=initial_task_count,
                )
                
                # 提取最佳代码
                best_code = ""
                best_profile = {}
                if result.get("best_implementations"):
                    best = result["best_implementations"][0]
                    best_code = best.get("impl_code", "")
                    best_profile = best.get("profile", {})
                
                success = result.get("total_success", 0) > 0
                
                return {
                    "coder_code": best_code,
                    "verifier_result": success,
                    "profile_res": best_profile,
                }
            except Exception as e:
                logger.error(f"[AdaptiveSearchWorkflow] 自适应搜索失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    "verifier_result": False,
                    "verifier_error": str(e),
                }
        
        workflow.add_node("adaptive_search", adaptive_search_node)
        workflow.set_entry_point("adaptive_search")
        workflow.add_edge("adaptive_search", END)
        
        return workflow
    
    @classmethod
    def prepare_config(cls, workflow_resources: Dict[str, Any], arguments: Dict[str, Any]):
        """根据 arguments 调整 workflow 配置
        
        1. 使用 build_langgraph_task_config() 补齐 docs_dir、agent_model_config 等
        2. 设置内部 LangGraphTask 使用 default_workflow_v2（基于 Skill 系统的 KernelDesigner + KernelGen）
        3. 调用父类 prepare_config 完成 log_dir 重定向 + 创建 WorkflowLogger
        4. 用 _SEARCH_DEFAULTS 补齐搜索参数的默认值
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

        workflow_resources["config"] = full_config
        
        # 调用父类：log_dir 重定向 + 创建 WorkflowLogger
        super().prepare_config(workflow_resources, arguments)
        
        # 用 _SEARCH_DEFAULTS 补齐搜索参数的默认值
        config = workflow_resources["config"]
        for key, default_val in cls._SEARCH_DEFAULTS.items():
            config.setdefault(key, default_val)
    
    def format_result(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """格式化自适应搜索结果"""
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
