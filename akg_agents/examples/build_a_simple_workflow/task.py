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

"""文档修复任务

继承 BaseLangGraphTask，封装文档修复工作流的执行。
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# 添加 python 目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from akg_agents.core_v2.langgraph_base.base_task import BaseLangGraphTask

# 使用绝对导入，避免 "attempted relative import with no known parent package" 错误
import importlib.util
workflow_spec = importlib.util.spec_from_file_location("workflow", Path(__file__).parent / "workflow.py")
workflow_module = importlib.util.module_from_spec(workflow_spec)
workflow_spec.loader.exec_module(workflow_module)
DocFixerWorkflow = workflow_module.DocFixerWorkflow

logger = logging.getLogger(__name__)


class DocFixerTask(BaseLangGraphTask):
    """文档修复任务
    
    继承 BaseLangGraphTask，提供文档修复的完整执行流程。
    
    Example:
        task = DocFixerTask(
            task_id="doc_fix_001",
            config={},
            document_content="这是一篇有错别子的文章...",
            document_type="markdown",
            language="zh"
        )
        success, result = await task.run()
        
        if success:
            print("修复后的文档:")
            print(result["beautified_content"])
    """
    
    def __init__(
        self,
        task_id: str,
        config: dict,
        document_content: str,
        document_type: str = "markdown",
        language: str = "zh"
    ):
        """初始化文档修复任务
        
        Args:
            task_id: 任务唯一标识
            config: 配置字典
            document_content: 待处理的文档内容
            document_type: 文档类型 ("markdown", "text", "code")
            language: 文档语言 ("zh", "en")
        """
        super().__init__(task_id, config, workflow_name="doc_fixer")
        
        # 文档修复专用参数
        self.document_content = document_content
        self.document_type = document_type
        self.language = language
        
        # 初始化工作流
        self._init_workflow()
        
        logger.info(f"DocFixerTask initialized: {task_id}, type={document_type}, lang={language}")
    
    def _init_workflow(self):
        """初始化工作流"""
        self.workflow = DocFixerWorkflow(config=self.config)
        self.app = self.workflow.compile()
    
    def _prepare_initial_state(self, init_info: Optional[dict] = None) -> Dict[str, Any]:
        """准备初始状态
        
        Args:
            init_info: 可选的初始化信息
            
        Returns:
            初始状态字典
        """
        max_iterations = self.config.get("max_step", 10)
        
        state = {
            # 基础字段 (来自 BaseState)
            "task_id": self.task_id,
            "task_label": f"doc_fix_{self.task_id}",
            "session_id": self.config.get("session_id", ""),
            "iteration": 0,
            "step_count": 0,
            "max_iterations": max_iterations,
            "agent_history": [],
            "success": False,
            "error_message": None,
            
            # 文档修复专用字段
            "original_content": self.document_content,
            "document_type": self.document_type,
            "language": self.language,
            "typo_fixed_content": "",
            "beautified_content": "",
            "typo_corrections": [],
            "beautify_changes": [],
            "typo_fixer_reasoning": "",
            "beautifier_reasoning": ""
        }
        
        # 合并额外的初始化信息
        if init_info:
            state.update(init_info)
        
        return state
    
    async def run(self, init_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, dict]:
        """执行文档修复任务
        
        Args:
            init_info: 可选的初始化信息
            
        Returns:
            Tuple[bool, dict]: (是否成功, 最终状态)
        """
        logger.info(f"[DocFixerTask {self.task_id}] Starting document fix...")
        
        success, final_state = await super().run(init_info)
        
        if success:
            # 打印摘要
            corrections = final_state.get("typo_corrections", [])
            changes = final_state.get("beautify_changes", [])
            logger.info(
                f"[DocFixerTask {self.task_id}] Completed: "
                f"{len(corrections)} typos fixed, {len(changes)} beautifications made"
            )
        else:
            logger.error(f"[DocFixerTask {self.task_id}] Failed: {final_state.get('error', 'Unknown error')}")
        
        return success, final_state
    
    def get_result_summary(self, final_state: dict) -> str:
        """获取结果摘要
        
        Args:
            final_state: 最终状态
            
        Returns:
            结果摘要字符串
        """
        lines = [
            "=" * 60,
            "文档修复结果摘要",
            "=" * 60,
            "",
            f"任务 ID: {self.task_id}",
            f"文档类型: {self.document_type}",
            f"语言: {self.language}",
            "",
            "--- 错别字修复 ---",
        ]
        
        corrections = final_state.get("typo_corrections", [])
        if corrections:
            for i, c in enumerate(corrections, 1):
                lines.append(f"  {i}. {c}")
        else:
            lines.append("  无错别字修复")
        
        lines.append("")
        lines.append("--- 美化修改 ---")
        
        changes = final_state.get("beautify_changes", [])
        if changes:
            for i, change in enumerate(changes, 1):
                lines.append(f"  {i}. {change}")
        else:
            lines.append("  无美化修改")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)

