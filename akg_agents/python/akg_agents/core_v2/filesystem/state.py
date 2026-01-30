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
AIKG FileSystem 状态管理

基于文件系统的状态持久化方案，核心原则：
1. 文件系统作为真相源：所有状态都持久化到文件系统
2. 人类可读：使用 JSON/Text 格式，便于调试和审计
3. 增量保存：避免重复，节省空间
4. 框架无关：不依赖特定框架，易于迁移
5. 支持断点续跑：保存足够信息以恢复执行
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from .models import (
    NodeState,
    ActionRecord,
    ActionHistoryFact,
    ActionHistoryCompressed,
    ThinkingState,
    PendingTool,
    PendingToolsState,
)
from .exceptions import (
    FileSystemStateError,
    NodeNotFoundError,
    InvalidNodeStateError,
)

logger = logging.getLogger(__name__)


class FileSystemState:
    """
    FileSystem 状态管理器
    
    管理任务的文件系统状态，包括：
    - 节点状态快照 (state.json)
    - 动作历史 (action_history.json, action_history_fact.json)
    - Thinking 固化 (thinking.json)
    - 待执行工具 (pending_tools.json)
    - 系统提示词 (system_prompts/)
    
    目录结构：
    ~/.akg_agents/conversations/{task_id}/
    ├─ trace.json
    ├─ current_node.txt
    ├─ nodes/
    │   ├─ root/
    │   │   └─ state.json
    │   ├─ node_001/
    │   │   ├─ state.json
    │   │   ├─ thinking.json
    │   │   ├─ actions/
    │   │   │   ├─ action_history.json
    │   │   │   ├─ action_history_fact.json
    │   │   │   └─ pending_tools.json
    │   │   ├─ code/
    │   │   └─ system_prompts/
    │   └─ ...
    └─ logs/
    """
    
    DEFAULT_BASE_DIR = "~/.akg_agents"
    
    def __init__(self, task_id: str, base_dir: str = None):
        """
        初始化 FileSystemState
        
        Args:
            task_id: 任务 ID
            base_dir: 基础目录，默认 ~/.akg_agents
        """
        self.task_id = task_id
        self.base_dir = Path(os.path.expanduser(base_dir or self.DEFAULT_BASE_DIR))
        self.task_dir = self.base_dir / "conversations" / task_id
        self.nodes_dir = self.task_dir / "nodes"
        self.logs_dir = self.task_dir / "logs"
        
        # 文件路径
        self.trace_file = self.task_dir / "trace.json"
        self.current_node_file = self.task_dir / "current_node.txt"
    
    # ==================== 目录管理 ====================
    
    def ensure_dir(self, path: Path) -> Path:
        """确保目录存在"""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_node_dir(self, node_id: str) -> Path:
        """获取节点目录路径"""
        return self.nodes_dir / node_id
    
    def get_actions_dir(self, node_id: str) -> Path:
        """获取节点动作目录路径"""
        return self.get_node_dir(node_id) / "actions"
    
    def get_code_dir(self, node_id: str) -> Path:
        """获取节点代码目录路径"""
        return self.get_node_dir(node_id) / "code"
    
    def get_system_prompts_dir(self, node_id: str) -> Path:
        """获取节点系统提示词目录路径"""
        return self.get_node_dir(node_id) / "system_prompts"
    
    def node_exists(self, node_id: str) -> bool:
        """检查节点是否存在"""
        return self.get_node_dir(node_id).exists()
    
    def task_exists(self) -> bool:
        """
        检查任务是否存在
        
        只检查任务目录是否存在。trace.json 由 TraceSystem 创建和管理。
        """
        return self.task_dir.exists()
    
    # ==================== 初始化 ====================
    
    def initialize_task(self, force: bool = False) -> None:
        """
        初始化任务目录结构
        
        Args:
            force: 是否强制重新初始化（删除现有目录）
        """
        if self.task_exists():
            if force:
                shutil.rmtree(self.task_dir)
                logger.info(f"Removed existing task directory: {self.task_dir}")
            else:
                logger.info(f"Task already exists: {self.task_id}")
                return
        
        # 创建目录结构
        self.ensure_dir(self.task_dir)
        self.ensure_dir(self.nodes_dir)
        self.ensure_dir(self.logs_dir)
        
        # 创建 root 节点
        self._create_root_node()
        
        # 设置当前节点为 root
        self.set_current_node("root")
        
        logger.info(f"Initialized task: {self.task_id} at {self.task_dir}")
    
    def _create_root_node(self) -> None:
        """创建 root 节点"""
        root_dir = self.get_node_dir("root")
        self.ensure_dir(root_dir)
        
        # 创建 root 节点状态
        root_state = NodeState(
            node_id="root",
            turn=0,
            status="init",
            task_info={"task_id": self.task_id},
        )
        self.save_node_state("root", root_state)
    
    # ==================== 当前节点管理 ====================
    
    def get_current_node(self) -> str:
        """
        获取当前节点 ID
        
        Returns:
            当前节点 ID，如果文件不存在或为空则返回 "root"
        """
        if not self.current_node_file.exists():
            return "root"
        node_id = self.current_node_file.read_text(encoding="utf-8").strip()
        # 如果文件为空或只有空白字符，返回默认值 "root"
        return node_id if node_id else "root"
    
    def set_current_node(self, node_id: str) -> None:
        """设置当前节点 ID"""
        self.current_node_file.write_text(node_id, encoding="utf-8")
    
    # ==================== 节点状态管理 ====================
    
    def save_node_state(self, node_id: str, state: NodeState) -> None:
        """
        保存节点状态
        
        Args:
            node_id: 节点 ID
            state: 节点状态
        """
        node_dir = self.get_node_dir(node_id)
        self.ensure_dir(node_dir)
        
        state_file = node_dir / "state.json"
        state_file.write_text(state.to_json(), encoding="utf-8")
        logger.debug(f"Saved node state: {node_id}")
    
    def load_node_state(self, node_id: str) -> NodeState:
        """
        加载节点状态
        
        Args:
            node_id: 节点 ID
            
        Returns:
            节点状态
            
        Raises:
            NodeNotFoundError: 节点不存在
        """
        state_file = self.get_node_dir(node_id) / "state.json"
        if not state_file.exists():
            raise NodeNotFoundError(node_id)
        
        content = state_file.read_text(encoding="utf-8")
        return NodeState.from_json(content)
    
    def update_node_state(
        self,
        node_id: str,
        turn: int = None,
        status: str = None,
        metrics: Dict = None,
        execution_info: Dict = None,
        file_state: Dict = None,
    ) -> NodeState:
        """
        更新节点状态
        
        Args:
            node_id: 节点 ID
            turn: 新的 turn 值
            status: 新的状态
            metrics: 新的指标
            execution_info: 新的执行信息
            file_state: 新的文件状态
            
        Returns:
            更新后的节点状态
        """
        state = self.load_node_state(node_id)
        
        if turn is not None:
            state.turn = turn
        if status is not None:
            state.status = status
        if metrics is not None:
            state.metrics.update(metrics)
        if execution_info is not None:
            state.execution_info.update(execution_info)
        if file_state is not None:
            state.file_state.update(file_state)
        
        from .models import _get_timestamp
        state.timestamp = _get_timestamp()
        
        self.save_node_state(node_id, state)
        return state
    
    # ==================== 动作历史管理 ====================
    
    def save_action_history_fact(
        self,
        node_id: str,
        history: ActionHistoryFact,
    ) -> None:
        """
        保存增量版动作历史
        
        每个 node 只保存自己新增的 action（避免重复）
        
        Args:
            node_id: 节点 ID
            history: 增量动作历史
        """
        actions_dir = self.get_actions_dir(node_id)
        self.ensure_dir(actions_dir)
        
        fact_file = actions_dir / "action_history_fact.json"
        fact_file.write_text(history.to_json(), encoding="utf-8")
        logger.debug(f"Saved action history fact: {node_id} ({len(history.actions)} actions)")
    
    def load_action_history_fact(self, node_id: str) -> ActionHistoryFact:
        """
        加载增量版动作历史
        
        Args:
            node_id: 节点 ID
            
        Returns:
            增量动作历史
        """
        fact_file = self.get_actions_dir(node_id) / "action_history_fact.json"
        if not fact_file.exists():
            # 返回空的历史
            return ActionHistoryFact(
                node_id=node_id,
                parent_node_id=None,
                turn=0,
            )
        
        content = fact_file.read_text(encoding="utf-8")
        return ActionHistoryFact.from_json(content)
    
    def append_action(
        self,
        node_id: str,
        action: ActionRecord,
        parent_node_id: str = None,
        turn: int = 0,
    ) -> ActionHistoryFact:
        """
        追加动作到增量历史
        
        Args:
            node_id: 节点 ID
            action: 动作记录
            parent_node_id: 父节点 ID
            turn: 当前 turn
            
        Returns:
            更新后的增量动作历史
        """
        history = self.load_action_history_fact(node_id)
        
        # 更新元数据
        if parent_node_id:
            history.parent_node_id = parent_node_id
        history.turn = turn
        
        # 添加动作
        history.add_action(action)
        
        # 保存
        self.save_action_history_fact(node_id, history)
        return history
    
    def save_action_history_compressed(
        self,
        node_id: str,
        history: ActionHistoryCompressed,
    ) -> None:
        """
        保存压缩版动作历史
        
        用于渲染、展示
        
        Args:
            node_id: 节点 ID
            history: 压缩动作历史
        """
        actions_dir = self.get_actions_dir(node_id)
        self.ensure_dir(actions_dir)
        
        compressed_file = actions_dir / "action_history_compressed.json"
        content = json.dumps(history.to_dict(), indent=2, ensure_ascii=False)
        compressed_file.write_text(content, encoding="utf-8")
        logger.debug(f"Saved compressed action history: {node_id}")
    
    def load_action_history_compressed(self, node_id: str) -> ActionHistoryCompressed:
        """
        加载压缩版动作历史
        
        Args:
            node_id: 节点 ID
            
        Returns:
            压缩动作历史
        """
        compressed_file = self.get_actions_dir(node_id) / "action_history_compressed.json"
        if not compressed_file.exists():
            return ActionHistoryCompressed(node_id=node_id, turn=0)
        
        content = compressed_file.read_text(encoding="utf-8")
        return ActionHistoryCompressed.from_dict(json.loads(content))
    
    # ==================== Thinking 管理 ====================
    
    def save_thinking(self, node_id: str, thinking: ThinkingState) -> None:
        """
        保存 Thinking 固化状态
        
        Args:
            node_id: 节点 ID
            thinking: Thinking 状态
        """
        node_dir = self.get_node_dir(node_id)
        self.ensure_dir(node_dir)
        
        thinking_file = node_dir / "thinking.json"
        thinking_file.write_text(thinking.to_json(), encoding="utf-8")
        logger.debug(f"Saved thinking: {node_id}")
    
    def load_thinking(self, node_id: str) -> Optional[ThinkingState]:
        """
        加载 Thinking 固化状态
        
        Args:
            node_id: 节点 ID
            
        Returns:
            Thinking 状态，不存在返回 None
        """
        thinking_file = self.get_node_dir(node_id) / "thinking.json"
        if not thinking_file.exists():
            return None
        
        content = thinking_file.read_text(encoding="utf-8")
        return ThinkingState.from_json(content)
    
    def update_thinking(
        self,
        node_id: str,
        latest_thinking: str = None,
        current_plan: Dict = None,
        decision: str = None,
    ) -> ThinkingState:
        """
        更新 Thinking 状态
        
        Args:
            node_id: 节点 ID
            latest_thinking: 最新思考
            current_plan: 当前计划
            decision: 决策
            
        Returns:
            更新后的 Thinking 状态
        """
        thinking = self.load_thinking(node_id)
        if thinking is None:
            state = self.load_node_state(node_id)
            thinking = ThinkingState(
                node_id=node_id,
                turn=state.turn,
            )
        
        if current_plan is not None:
            thinking.current_plan = current_plan
        
        if latest_thinking is not None and decision is not None:
            thinking.add_decision(latest_thinking, decision)
        elif latest_thinking is not None:
            thinking.latest_thinking = latest_thinking
        
        self.save_thinking(node_id, thinking)
        return thinking
    
    # ==================== Pending Tools 管理 ====================
    
    def save_pending_tools(self, node_id: str, pending: PendingToolsState) -> None:
        """
        保存待执行工具状态
        
        Args:
            node_id: 节点 ID
            pending: 待执行工具状态
        """
        actions_dir = self.get_actions_dir(node_id)
        self.ensure_dir(actions_dir)
        
        pending_file = actions_dir / "pending_tools.json"
        pending_file.write_text(pending.to_json(), encoding="utf-8")
        logger.debug(f"Saved pending tools: {node_id}")
    
    def load_pending_tools(self, node_id: str) -> PendingToolsState:
        """
        加载待执行工具状态
        
        Args:
            node_id: 节点 ID
            
        Returns:
            待执行工具状态
        """
        pending_file = self.get_actions_dir(node_id) / "pending_tools.json"
        if not pending_file.exists():
            return PendingToolsState(node_id=node_id, turn=0)
        
        content = pending_file.read_text(encoding="utf-8")
        return PendingToolsState.from_json(content)
    
    def add_pending_tool(
        self,
        node_id: str,
        tool: PendingTool,
        turn: int = 0,
    ) -> PendingToolsState:
        """
        添加待执行工具
        
        Args:
            node_id: 节点 ID
            tool: 待执行工具
            turn: 当前 turn
            
        Returns:
            更新后的待执行工具状态
        """
        pending = self.load_pending_tools(node_id)
        pending.turn = turn
        pending.add_pending_tool(tool)
        self.save_pending_tools(node_id, pending)
        return pending
    
    def mark_tool_completed(
        self,
        node_id: str,
        tool_call_id: str,
        raise_if_not_found: bool = False,
    ) -> PendingToolsState:
        """
        标记工具执行完成
        
        Args:
            node_id: 节点 ID
            tool_call_id: 工具调用 ID
            raise_if_not_found: 如果未找到工具是否抛出异常
            
        Returns:
            持久化后的待执行工具状态（始终反映磁盘上的实际状态）
            
        Raises:
            FileSystemStateError: 如果 raise_if_not_found=True 且未找到工具
        """
        pending = self.load_pending_tools(node_id)
        found = pending.mark_completed(tool_call_id)
        
        if not found:
            if raise_if_not_found:
                raise FileSystemStateError(
                    f"Tool '{tool_call_id}' not found in pending tools for node '{node_id}'"
                )
            logger.warning(
                f"Tool '{tool_call_id}' not found in pending tools for node '{node_id}'"
            )
            # 返回从磁盘重新加载的状态，确保返回值反映实际持久化的状态
            return self.load_pending_tools(node_id)
        
        # 保存修改并返回持久化后的状态
        self.save_pending_tools(node_id, pending)
        return pending
    
    def clear_pending_tools(self, node_id: str) -> None:
        """
        清空待执行工具
        
        Args:
            node_id: 节点 ID
        """
        state = self.load_node_state(node_id)
        pending = PendingToolsState(node_id=node_id, turn=state.turn)
        self.save_pending_tools(node_id, pending)
    
    # ==================== System Prompt 管理 ====================
    
    def save_system_prompt(
        self,
        node_id: str,
        turn: int,
        prompt: str,
    ) -> Path:
        """
        保存完整系统提示词（可选，用于调试）
        
        Args:
            node_id: 节点 ID
            turn: turn 编号
            prompt: 提示词内容
            
        Returns:
            保存的文件路径
        """
        prompts_dir = self.get_system_prompts_dir(node_id)
        self.ensure_dir(prompts_dir)
        
        prompt_file = prompts_dir / f"turn_{turn:03d}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        logger.debug(f"Saved system prompt: {node_id} turn {turn}")
        return prompt_file
    
    def load_system_prompt(self, node_id: str, turn: int) -> Optional[str]:
        """
        加载系统提示词
        
        Args:
            node_id: 节点 ID
            turn: turn 编号
            
        Returns:
            提示词内容，不存在返回 None
        """
        prompt_file = self.get_system_prompts_dir(node_id) / f"turn_{turn:03d}.txt"
        if not prompt_file.exists():
            return None
        return prompt_file.read_text(encoding="utf-8")
    
    def get_latest_system_prompt(self, node_id: str) -> Optional[str]:
        """
        获取最新的系统提示词
        
        Args:
            node_id: 节点 ID
            
        Returns:
            最新提示词内容，不存在返回 None
        """
        prompts_dir = self.get_system_prompts_dir(node_id)
        if not prompts_dir.exists():
            return None
        
        prompt_files = sorted(prompts_dir.glob("turn_*.txt"))
        if not prompt_files:
            return None
        
        return prompt_files[-1].read_text(encoding="utf-8")
    
    # ==================== 代码文件管理 ====================
    
    def save_code_file(
        self,
        node_id: str,
        filename: str,
        content: str,
    ) -> Path:
        """
        保存代码文件
        
        Args:
            node_id: 节点 ID
            filename: 文件名
            content: 文件内容
            
        Returns:
            保存的文件路径
        """
        code_dir = self.get_code_dir(node_id)
        self.ensure_dir(code_dir)
        
        code_file = code_dir / filename
        code_file.write_text(content, encoding="utf-8")
        
        # 更新文件状态
        import hashlib
        from .models import _get_timestamp
        
        checksum = hashlib.md5(content.encode()).hexdigest()
        file_info = {
            "size": len(content),
            "last_modified": _get_timestamp(),
            "checksum": checksum,
        }
        
        self.update_node_state(
            node_id,
            file_state={f"code/{filename}": file_info}
        )
        
        logger.debug(f"Saved code file: {node_id}/{filename}")
        return code_file
    
    def load_code_file(self, node_id: str, filename: str) -> Optional[str]:
        """
        加载代码文件
        
        Args:
            node_id: 节点 ID
            filename: 文件名
            
        Returns:
            文件内容，不存在返回 None
        """
        code_file = self.get_code_dir(node_id) / filename
        if not code_file.exists():
            return None
        return code_file.read_text(encoding="utf-8")
    
    def list_code_files(self, node_id: str) -> List[str]:
        """
        列出节点的所有代码文件
        
        Args:
            node_id: 节点 ID
            
        Returns:
            文件名列表
        """
        code_dir = self.get_code_dir(node_id)
        if not code_dir.exists():
            return []
        return [f.name for f in code_dir.iterdir() if f.is_file()]
    
    # ==================== 节点复制 ====================
    
    def copy_node_state(self, from_node_id: str, to_node_id: str) -> NodeState:
        """
        复制节点状态到新节点
        
        Args:
            from_node_id: 源节点 ID
            to_node_id: 目标节点 ID
            
        Returns:
            新节点的状态
        """
        source_state = self.load_node_state(from_node_id)
        
        # 创建新状态（更新 node_id 和 timestamp）
        from .models import _get_timestamp
        new_state = NodeState(
            node_id=to_node_id,
            turn=source_state.turn,
            status=source_state.status,
            agent_info=source_state.agent_info.copy(),
            task_info=source_state.task_info.copy(),
            execution_info=source_state.execution_info.copy(),
            file_state=source_state.file_state.copy(),
            metrics=source_state.metrics.copy(),
            timestamp=_get_timestamp(),
        )
        
        self.save_node_state(to_node_id, new_state)
        return new_state
    
    def copy_code_files(self, from_node_id: str, to_node_id: str) -> None:
        """
        复制代码文件到新节点
        
        Args:
            from_node_id: 源节点 ID
            to_node_id: 目标节点 ID
        """
        source_code_dir = self.get_code_dir(from_node_id)
        if not source_code_dir.exists():
            return
        
        target_code_dir = self.get_code_dir(to_node_id)
        self.ensure_dir(target_code_dir)
        
        for code_file in source_code_dir.iterdir():
            if code_file.is_file():
                shutil.copy2(code_file, target_code_dir / code_file.name)
        
        logger.debug(f"Copied code files from {from_node_id} to {to_node_id}")
    
    # ==================== 清理 ====================
    
    def delete_node(self, node_id: str) -> None:
        """
        删除节点及其所有数据
        
        Args:
            node_id: 节点 ID
        """
        if node_id == "root":
            raise FileSystemStateError("Cannot delete root node")
        
        node_dir = self.get_node_dir(node_id)
        if node_dir.exists():
            shutil.rmtree(node_dir)
            logger.info(f"Deleted node: {node_id}")
    
    def delete_task(self) -> None:
        """删除整个任务"""
        if self.task_dir.exists():
            shutil.rmtree(self.task_dir)
            logger.info(f"Deleted task: {self.task_id}")
