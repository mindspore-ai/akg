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

"""文档修复工作流

继承 BaseWorkflow，实现 TypoFixer → Beautifier 的处理流程。
"""

import logging
import sys
from pathlib import Path

# 添加 python 目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from langgraph.graph import StateGraph, END

from akg_agents.core_v2.langgraph_base.base_workflow import BaseWorkflow

# 使用绝对导入，避免 "attempted relative import with no known parent package" 错误
import importlib.util
state_spec = importlib.util.spec_from_file_location("state", Path(__file__).parent / "state.py")
state_module = importlib.util.module_from_spec(state_spec)
state_spec.loader.exec_module(state_module)
DocFixerState = state_module.DocFixerState

typo_fixer_spec = importlib.util.spec_from_file_location("typo_fixer", Path(__file__).parent / "agents" / "typo_fixer.py")
typo_fixer_module = importlib.util.module_from_spec(typo_fixer_spec)
typo_fixer_spec.loader.exec_module(typo_fixer_module)
TypoFixer = typo_fixer_module.TypoFixer

beautifier_spec = importlib.util.spec_from_file_location("beautifier", Path(__file__).parent / "agents" / "beautifier.py")
beautifier_module = importlib.util.module_from_spec(beautifier_spec)
beautifier_spec.loader.exec_module(beautifier_module)
Beautifier = beautifier_module.Beautifier

logger = logging.getLogger(__name__)


class DocFixerWorkflow(BaseWorkflow[DocFixerState]):
    """文档修复工作流
    
    继承通用 BaseWorkflow，实现文档修复的处理流程：
    
    流程:
        START → typo_fixer → beautifier → END
    
    Example:
        workflow = DocFixerWorkflow(config={})
        app = workflow.compile()
        result = await app.ainvoke(initial_state)
    """
    
    def __init__(self, config: dict, trace=None):
        """初始化工作流
        
        Args:
            config: 配置字典
            trace: Trace 实例（可选）
        """
        super().__init__(config, trace)
        
        # 初始化 agents
        self.typo_fixer = TypoFixer(config)
        self.beautifier = Beautifier(config)
    
    def build_graph(self) -> StateGraph:
        """构建工作流图
        
        Returns:
            StateGraph: 文档修复工作流图
        """
        workflow = StateGraph(DocFixerState)
        
        # 添加节点
        workflow.add_node("typo_fixer", self._create_typo_fixer_node())
        workflow.add_node("beautifier", self._create_beautifier_node())
        
        # 添加边: typo_fixer → beautifier → END
        workflow.add_edge("typo_fixer", "beautifier")
        workflow.add_edge("beautifier", END)
        
        # 设置入口点
        workflow.set_entry_point("typo_fixer")
        
        return workflow
    
    def _create_typo_fixer_node(self):
        """创建错别字修复节点"""
        async def typo_fixer_node(state: DocFixerState) -> dict:
            """TypoFixer 节点：修复错别字"""
            task_id = state.get("task_id", "unknown")
            logger.info(f"[Task {task_id}] Running TypoFixer...")
            
            # 获取原始内容
            content = state.get("original_content", "")
            document_type = state.get("document_type", "markdown")
            language = state.get("language", "zh")
            
            # 执行错别字修复
            corrected_content, corrections, reasoning = await self.typo_fixer.run(
                content=content,
                document_type=document_type,
                language=language
            )
            
            logger.info(f"[Task {task_id}] TypoFixer found {len(corrections)} corrections")
            
            return {
                "typo_fixed_content": corrected_content,
                "typo_corrections": corrections,
                "typo_fixer_reasoning": reasoning,
                "step_count": state.get("step_count", 0) + 1,
                "agent_history": ["typo_fixer"]
            }
        
        return typo_fixer_node
    
    def _create_beautifier_node(self):
        """创建文档美化节点"""
        async def beautifier_node(state: DocFixerState) -> dict:
            """Beautifier 节点：美化文档"""
            task_id = state.get("task_id", "unknown")
            logger.info(f"[Task {task_id}] Running Beautifier...")
            
            # 获取错别字修复后的内容
            content = state.get("typo_fixed_content", state.get("original_content", ""))
            document_type = state.get("document_type", "markdown")
            language = state.get("language", "zh")
            
            # 执行文档美化
            beautified_content, changes, reasoning = await self.beautifier.run(
                content=content,
                document_type=document_type,
                language=language
            )
            
            logger.info(f"[Task {task_id}] Beautifier made {len(changes)} changes")
            
            return {
                "beautified_content": beautified_content,
                "beautify_changes": changes,
                "beautifier_reasoning": reasoning,
                "step_count": state.get("step_count", 0) + 1,
                "agent_history": ["beautifier"],
                "success": True  # 标记任务成功完成
            }
        
        return beautifier_node

