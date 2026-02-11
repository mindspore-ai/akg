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

import difflib
import hashlib
import json
import logging
import os
import shutil
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterator

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
# Git imports removed - using Snapshot Filesystem instead

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
        
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path.home() / ".akg_agents"
            
        self.task_dir = self.base_dir / "conversations" / task_id
        self.nodes_dir = self.task_dir / "nodes"
        self.workspace_dir = self.task_dir / "workspace"  # Git 代码工作区
        self.logs_dir = self.task_dir / "logs"
        
        # 文件路径
        self.trace_file = self.task_dir / "trace.json"
        self.current_node_file = self.task_dir / "current_node.txt"
    
    def get_code_snapshot_dir(self, node_id: str) -> Path:
        """获取节点代码快照目录"""
        return self.get_node_dir(node_id) / "code"
    
    def _hardlink_or_copy(self, src: Path, dst: Path) -> None:
        """
        尝试硬链接，失败则复制
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
        """
        try:
            os.link(src, dst)
        except (OSError, NotImplementedError):
            # 硬链接失败（跨设备、FS不支持等），回退到复制
            shutil.copy2(src, dst)

    # ==================== 目录管理 ====================
    
    def ensure_dir(self, path: Path) -> Path:
        """确保目录存在"""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_node_dir(self, node_id: str) -> Path:
        """获取节点目录路径 (元数据)"""
        return self.nodes_dir / node_id
    
    def get_actions_dir(self, node_id: str) -> Path:
        """获取节点动作目录路径"""
        return self.get_node_dir(node_id) / "actions"
    
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
        if force and self.task_dir.exists():
            shutil.rmtree(self.task_dir)
        
        self.ensure_dir(self.task_dir)
        self.ensure_dir(self.nodes_dir)
        self.ensure_dir(self.logs_dir)
        self.ensure_dir(self.workspace_dir)
        
        # 创建 root 节点
        self._create_root_node()
        
        # 设置当前节点为 root
        self.set_current_node("root")
        
        logger.info(f"Initialized task: {self.task_id} at {self.task_dir}")
    
    def _create_root_node(self) -> None:
        """创建 root 节点"""
        root_dir = self.get_node_dir("root")
        self.ensure_dir(root_dir)
        
        # 创建空的代码快照目录
        code_dir = self.get_code_snapshot_dir("root")
        self.ensure_dir(code_dir)

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
        """
        设置当前节点 ID，并恢复工作区到该节点的快照状态
        """
        current_node_file = self.task_dir / "current_node.txt"
        current_node_file.write_text(node_id, encoding="utf-8")
        
        # 恢复工作区到节点快照
        self._restore_workspace(node_id)
    
    def _restore_workspace(self, node_id: str) -> None:
        """
        从节点快照恢复工作区
        
        Args:
            node_id: 节点 ID
        """
        snapshot_dir = self.get_code_snapshot_dir(node_id)
        if not snapshot_dir.exists():
            logger.debug(f"Node {node_id} has no code snapshot, workspace unchanged")
            return
        
        # 清空工作区 (保留目录本身)
        for item in self.workspace_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        
        # 从快照复制文件到工作区 (使用复制而非硬链接，防止意外修改快照)
        for src_file in snapshot_dir.rglob("*"):
            if src_file.is_file():
                rel_path = src_file.relative_to(snapshot_dir)
                dst_file = self.workspace_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
        
        logger.debug(f"Restored workspace to node {node_id}")
    
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
    #
    # ⚠️ 重要：所有对 code/ 目录下文件的写入操作，必须通过 save_code_file() 方法进行。
    # 不要直接使用 Path.write_text() 或 open().write() 写入 code/ 目录下的文件！
    # 原因：code/ 目录下的文件可能通过硬链接与其他节点共享同一个 inode，
    # 直接写入会原地修改共享 inode 的内容，污染所有共享该快照的节点。
    # save_code_file() 会先 unlink() 打破硬链接，再写入新内容（Copy-on-Write）。
    #
    
    def verify_snapshot_integrity(self, node_id: str) -> List[str]:
        """
        验证节点代码快照的完整性（检测硬链接污染）
        
        检查 code/ 目录下的文件是否与 file_state 中记录的 checksum 一致。
        如果不一致，说明文件可能被外部直接写入（绕过了 save_code_file 的 CoW 机制）。
        
        Args:
            node_id: 节点 ID
            
        Returns:
            不一致的文件列表（空列表表示完整性正常）
        """
        import hashlib
        
        corrupted = []
        try:
            state = self.load_node_state(node_id)
        except Exception:
            return corrupted
        
        snapshot_dir = self.get_code_snapshot_dir(node_id)
        
        for file_key, file_info in state.file_state.items():
            if not file_key.startswith("code/"):
                continue
            filename = file_key[len("code/"):]
            file_path = snapshot_dir / filename
            
            if not file_path.exists():
                corrupted.append(f"{file_key}: file missing")
                continue
            
            expected_checksum = file_info.get("checksum")
            if expected_checksum:
                actual_content = file_path.read_text(encoding="utf-8")
                actual_checksum = hashlib.md5(actual_content.encode("utf-8")).hexdigest()
                if actual_checksum != expected_checksum:
                    corrupted.append(
                        f"{file_key}: checksum mismatch "
                        f"(expected={expected_checksum[:8]}..., actual={actual_checksum[:8]}..., "
                        f"inode={file_path.stat().st_ino}, nlink={file_path.stat().st_nlink})"
                    )
        
        if corrupted:
            logger.warning(
                f"[Integrity] Node {node_id} snapshot corrupted! "
                f"Files may have been written directly (bypassing CoW): {corrupted}"
            )
        
        return corrupted
    
    def save_code_file(self, node_id: str, filename: str, content: str) -> None:
        """
        保存代码文件 (Snapshot, Copy-on-Write)
        
        ⚠️ 这是写入 code/ 目录的唯一正确入口。不要直接用 Path.write_text() 写入
        code/ 目录下的文件，否则会污染硬链接共享的其他节点快照。
        
        Args:
            node_id: 节点 ID
            filename: 文件名
            content: 代码内容
        """
        # 1. 写入文件到 Workspace
        file_path = self.workspace_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        
        # 2. 创建/更新节点快照
        snapshot_dir = self.get_code_snapshot_dir(node_id)
        self.ensure_dir(snapshot_dir)
        
        # 复制到快照目录 (注意: 要先删除旧文件以打破硬链接, 防止污染其他节点的快照)
        snapshot_file = snapshot_dir / filename
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        if snapshot_file.exists():
            snapshot_file.unlink()  # 打破硬链接
        shutil.copy2(file_path, snapshot_file)
        
        # 3. 更新 Node 元数据
        state = self.load_node_state(node_id)
        
        # 更新 file_state (保持 API 兼容: code/{filename} with size)
        import hashlib
        md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
        file_key = f"code/{filename}"
        state.file_state[file_key] = {
            "path": str(snapshot_file.relative_to(self.task_dir)),
            "checksum": md5,
            "size": len(content)
        }
        
        self.save_node_state(node_id, state)
        logger.debug(f"Saved code file {filename} to snapshot (node {node_id})")
    
    def load_code_file(self, node_id: str, filename: str) -> str:
        """
        加载代码文件 (Snapshot)
        
        Args:
            node_id: 节点 ID
            filename: 文件名
            
        Returns:
            代码内容
        """
        snapshot_dir = self.get_code_snapshot_dir(node_id)
        file_path = snapshot_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {filename} not found in node {node_id}")
        
        return file_path.read_text(encoding="utf-8")
    
    # ==================== 节点复制 ====================
    
    
    def _path_matches_patterns(self, path: Path, include_patterns: List[str], exclude_patterns: List[str], base_dir: Path) -> bool:
        """
        检查路径是否匹配包含规则且不匹配排除规则
        """
        rel_path = path.relative_to(base_dir).as_posix()
        
        # 1. 检查排除规则 (优先级最高)
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return False
                
        # 2. 检查包含规则
        for pattern in include_patterns:
            # 支持目录匹配 (如 code/ 匹配 code/main.py)
            if pattern.endswith("/"):
                if rel_path.startswith(pattern) or rel_path == pattern.rstrip("/"):
                    return True
            elif fnmatch.fnmatch(rel_path, pattern):
                return True
                
        return False

    def copy_node_state(self, from_node_id: str, to_node_id: str, 
                       include_patterns: List[str] = None, 
                       exclude_patterns: List[str] = None) -> NodeState:
        """
        复制节点状态到新节点 (使用硬链接，支持配置化文件追踪)
        
        Args:
            from_node_id: 源节点 ID
            to_node_id: 目标节点 ID
            include_patterns: 需要包含的文件/目录模式列表 (Glob). 默认 ["code/**"]
            exclude_patterns: 需要排除的文件/目录模式列表 (Glob). 默认 []
            
        Returns:
            新节点的状态
        """
        source_state = self.load_node_state(from_node_id)
        
        # 默认规则: 只包含 code 目录
        if include_patterns is None:
            include_patterns = ["code/"]
        if exclude_patterns is None:
            exclude_patterns = []
            
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
        
        # 复制文件快照 (根据配置规则)
        source_dir = self.get_node_dir(from_node_id)
        target_dir = self.get_node_dir(to_node_id)
        
        if source_dir.exists():
            self.ensure_dir(target_dir)
            # 遍历源节点下所有文件
            for src_file in source_dir.rglob("*"):
                if src_file.is_file():
                    # 跳过元数据文件
                    if src_file.name in ["state.json", "thinking.json"]:
                        continue
                        
                    if self._path_matches_patterns(src_file, include_patterns, exclude_patterns, source_dir):
                        rel_path = src_file.relative_to(source_dir)
                        dst_file = target_dir / rel_path
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        self._hardlink_or_copy(src_file, dst_file)
        
        # 复制 Thinking 状态 (继承)
        source_thinking = self.load_thinking(from_node_id)
        if source_thinking:
            # 创建新 ThinkingState，继承内容但更新元数据
            new_thinking = ThinkingState(
                node_id=to_node_id,
                turn=new_state.turn,
                current_plan=source_thinking.current_plan.copy(),
                latest_thinking=source_thinking.latest_thinking,
                decision_history=source_thinking.decision_history.copy(),
            )
            self.save_thinking(to_node_id, new_thinking)
        
        self.save_node_state(to_node_id, new_state)
        logger.debug(f"Copied node state from {from_node_id} to {to_node_id} (hardlink)")
        return new_state
    
    
    # ==================== 代码访问便捷方法 ====================
    
    def list_code_files(self, node_id: str) -> List[str]:
        """
        列出节点的所有代码文件
        
        Snapshot 实现: 遍历 nodes/{node_id}/code/ 目录
        """
        snapshot_dir = self.get_code_snapshot_dir(node_id)
        if not snapshot_dir.exists():
            return []
        
        files = []
        for file_path in snapshot_dir.rglob("*"):
            if file_path.is_file():
                files.append(str(file_path.relative_to(snapshot_dir)))
        return files

    def export_node_code(self, node_id: str, target_dir: str) -> None:
        """
        导出节点代码到指定目录 (便利用于查看)
        
        Snapshot 实现: 直接复制 nodes/{node_id}/code/ 目录
        """
        snapshot_dir = self.get_code_snapshot_dir(node_id)
        if not snapshot_dir.exists():
            raise FileSystemStateError(f"Node {node_id} has no code snapshot")
            
        target_path = Path(target_dir).absolute()
        if target_path.exists():
             raise FileExistsError(f"Target directory {target_path} already exists")
        
        shutil.copytree(snapshot_dir, target_path)
        logger.info(f"Exported node {node_id} code to {target_path}")
    
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

    # ==================== Diff 功能 ====================

    def diff_file(self, node_a: str, node_b: str, filename: str) -> str:
        """
        比较两个节点之间指定文件的差异
        
        Args:
            node_a: 节点 A ID
            node_b: 节点 B ID
            filename: 文件名
            
        Returns:
            Unified Diff 格式的差异内容，如果没有差异则返回空字符串
        """
        content_a = self.load_code_file(node_a, filename).splitlines(keepends=True)
        content_b = self.load_code_file(node_b, filename).splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            content_a, content_b,
            fromfile=f"a/{filename} (node {node_a})",
            tofile=f"b/{filename} (node {node_b})"
        )
        return "".join(diff)

    def diff_nodes(self, node_a: str, node_b: str, output_path: Optional[Path] = None) -> Path:
        """
        比较两个节点之间的所有代码差异，并写入补丁文件
        
        默认存放在 workspace/.akg/diffs/{node_a}_to_{node_b}.patch
        
        Args:
            node_a: 节点 A ID
            node_b: 节点 B ID
            output_path: 输出路径（可选）
            
        Returns:
            生成的补丁文件路径
        """
        if output_path is None:
            diff_dir = self.workspace_dir / ".akg" / "diffs"
            self.ensure_dir(diff_dir)
            output_path = diff_dir / f"{node_a}_to_{node_b}.patch"
        
        state_a = self.load_node_state(node_a)
        state_b = self.load_node_state(node_b)
        
        all_files = set(state_a.file_state.keys()) | set(state_b.file_state.keys())
        processed_files = set()
        
        with open(output_path, "w", encoding="utf-8") as f:
            for file_key in sorted(all_files):
                filename = file_key.replace("code/", "")
                
                # 利用 Checksum 快速过滤未修改文件
                if file_key in state_a.file_state and file_key in state_b.file_state:
                    if state_a.file_state[file_key]["checksum"] == state_b.file_state[file_key]["checksum"]:
                        continue
                
                # 获取内容
                content_a = []
                if file_key in state_a.file_state:
                    content_a = self.load_code_file(node_a, filename).splitlines(keepends=True)
                
                content_b = []
                if file_key in state_b.file_state:
                    content_b = self.load_code_file(node_b, filename).splitlines(keepends=True)
                
                # 生成 Diff
                diff = difflib.unified_diff(
                    content_a, content_b,
                    fromfile=f"a/{filename}",
                    tofile=f"b/{filename}"
                )
                f.write("".join(diff))
                f.write("\n")
                processed_files.add(filename)
                
        logger.info(f"Generated diff between {node_a} and {node_b} to {output_path} ({len(processed_files)} files modified)")
        return output_path

