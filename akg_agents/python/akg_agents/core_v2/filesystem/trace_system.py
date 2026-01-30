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
AIKG Trace 系统（Tree 版本）

单一树结构，支持：
- 多分叉探索
- 节点切换
- 增量动作历史
- 断点续跑
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .models import (
    TraceTree,
    TraceNode,
    NodeState,
    ActionRecord,
    ActionHistoryFact,
    ActionHistoryCompressed,
)
from .compressor import ActionCompressor
from .state import FileSystemState
from .exceptions import (
    NodeNotFoundError,
    TraceSystemError,
    TraceNotInitializedError,
    TraceAlreadyExistsError,
)

logger = logging.getLogger(__name__)


def _get_timestamp() -> str:
    """获取当前 ISO 格式时间戳"""
    return datetime.now().isoformat()


class TraceSystem:
    """
    Trace 树管理器
    
    核心功能：
    - 单一树结构管理
    - 节点添加和切换
    - 自动分叉
    - 增量动作历史
    - 路径查询和对比
    
    设计理念：
    - 整个 Trace 就是一个树（Tree），不需要 "branch" 的概念
    - 一个节点可以有多个子节点（children），形成分叉
    - 用户可以切换到任意节点
    - 在已有子节点的节点上执行新操作时，自动创建新的子节点（分叉）
    """
    
    def __init__(self, task_id: str, base_dir: str = None):
        """
        初始化 TraceSystem
        
        Args:
            task_id: 任务 ID
            base_dir: 基础目录，默认 ~/.akg_agents
        """
        self.task_id = task_id
        self.fs = FileSystemState(task_id, base_dir)
        self._node_counter = 0
        self._action_counter = 0
        self._trace: Optional[TraceTree] = None
    
    @property
    def trace(self) -> TraceTree:
        """获取 Trace 树（自动加载）"""
        if self._trace is None:
            self._trace = self._load_trace()
        return self._trace
    
    # ==================== 初始化 ====================
    
    def initialize(self, force: bool = False) -> None:
        """
        初始化 Trace 系统
        
        创建 trace.json 和 root 节点
        
        Args:
            force: 是否强制重新初始化
        """
        if self.fs.task_exists() and self.fs.trace_file.exists() and not force:
            # 加载现有 trace
            self._trace = self._load_trace()
            self._update_counters()
            logger.info(f"Loaded existing trace: {self.task_id}")
            return
        
        # 初始化文件系统
        self.fs.initialize_task(force=force)
        
        # 创建 Trace 树
        self._trace = TraceTree(task_id=self.task_id)
        
        # 创建 root 节点
        root_node = TraceNode(
            node_id="root",
            parent_id=None,
            state_snapshot={"turn": 0, "status": "init"},
            action=None,
            result=None,
        )
        self._trace.add_node(root_node)
        
        # 保存
        self._save_trace()
        
        logger.info(f"Initialized trace system: {self.task_id}")
    
    def _load_trace(self) -> TraceTree:
        """加载 Trace 树"""
        if not self.fs.trace_file.exists():
            raise TraceNotInitializedError(self.task_id)
        
        content = self.fs.trace_file.read_text(encoding="utf-8")
        return TraceTree.from_json(content)
    
    def _save_trace(self) -> None:
        """保存 Trace 树"""
        self.fs.trace_file.write_text(self._trace.to_json(), encoding="utf-8")
    
    def _update_counters(self) -> None:
        """从现有节点和动作更新计数器"""
        for node_id in self.trace.tree.keys():
            # 更新节点计数器
            if node_id.startswith("node_"):
                try:
                    num = int(node_id.split("_")[1])
                    self._node_counter = max(self._node_counter, num)
                except (ValueError, IndexError):
                    pass
            
            # 更新动作计数器 - 扫描每个节点的动作历史
            if node_id != "root":
                try:
                    history = self.fs.load_action_history_fact(node_id)
                    for action in history.actions:
                        if action.action_id.startswith("action_"):
                            try:
                                num = int(action.action_id.split("_")[1])
                                self._action_counter = max(self._action_counter, num)
                            except (ValueError, IndexError):
                                pass
                except Exception:
                    # 如果加载失败，跳过该节点
                    pass
    
    # ==================== 节点 ID 生成 ====================
    
    def _generate_node_id(self) -> str:
        """生成新的节点 ID"""
        self._node_counter += 1
        return f"node_{self._node_counter:03d}"
    
    def _generate_action_id(self) -> str:
        """生成新的动作 ID"""
        self._action_counter += 1
        return f"action_{self._action_counter:03d}"
    
    # ==================== 节点操作 ====================
    
    def get_current_node(self) -> str:
        """获取当前节点 ID"""
        return self.fs.get_current_node()
    
    def get_node(self, node_id: str) -> TraceNode:
        """
        获取节点信息
        
        Args:
            node_id: 节点 ID
            
        Returns:
            TraceNode 对象
            
        Raises:
            NodeNotFoundError: 节点不存在
        """
        node = self.trace.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)
        return node
    
    def add_node(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        metrics: Dict[str, Any] = None,
        state_snapshot: Dict[str, Any] = None,
    ) -> str:
        """
        在当前节点添加子节点
        
        如果当前节点已有子节点，自动创建分叉
        
        Args:
            action: 执行的动作
            result: 执行结果
            metrics: 指标数据
            state_snapshot: 状态快照（可选，自动从父节点继承）
            
        Returns:
            新节点的 ID
        """
        current_node_id = self.get_current_node()
        current_node = self.get_node(current_node_id)
        
        # 生成新节点 ID
        new_node_id = self._generate_node_id()
        
        # 获取父节点状态
        try:
            parent_state = self.fs.load_node_state(current_node_id)
            new_turn = parent_state.turn + 1
        except NodeNotFoundError:
            new_turn = 1
        
        # 创建状态快照
        if state_snapshot is None:
            state_snapshot = {
                "turn": new_turn,
                "status": "running",
            }
        
        # 创建新节点
        new_node = TraceNode(
            node_id=new_node_id,
            parent_id=current_node_id,
            state_snapshot=state_snapshot,
            action=action,
            result=result,
            metrics=metrics or {},
        )
        
        # 添加到树
        self.trace.add_node(new_node)
        self.trace.set_current_node(new_node_id)
        
        # 保存 trace.json
        self._save_trace()
        
        # 更新 current_node.txt
        self.fs.set_current_node(new_node_id)
        
        # 创建节点目录和状态
        self._create_node_files(new_node_id, current_node_id, new_turn, action, result)
        
        # 记录分叉
        if len(current_node.children) > 1:
            logger.info(f"Created fork at {current_node_id}: {new_node_id} "
                       f"(now {len(current_node.children)} children)")
        else:
            logger.debug(f"Added node: {new_node_id} (parent: {current_node_id})")
        
        return new_node_id
    
    def _create_node_files(
        self,
        node_id: str,
        parent_id: str,
        turn: int,
        action: Dict,
        result: Dict,
    ) -> None:
        """创建节点的文件系统结构"""
        # 复制父节点状态
        if parent_id != "root":
            self.fs.copy_node_state(parent_id, node_id)
            self.fs.copy_code_files(parent_id, node_id)
        else:
            # 从 root 创建新状态
            state = NodeState(
                node_id=node_id,
                turn=turn,
                status="running",
                task_info={"task_id": self.task_id},
            )
            self.fs.save_node_state(node_id, state)
        
        # 更新节点状态
        self.fs.update_node_state(node_id, turn=turn, status="running")
        
        # 创建增量动作历史
        action_record = ActionRecord(
            action_id=self._generate_action_id(),
            tool_name=action.get("type", "unknown"),
            arguments=action.get("params", {}),
            result=result,
        )
        
        history = ActionHistoryFact(
            node_id=node_id,
            parent_node_id=parent_id,
            turn=turn,
        )
        history.add_action(action_record)
        self.fs.save_action_history_fact(node_id, history)
    
    def switch_node(self, node_id: str) -> None:
        """
        切换到指定节点
        
        Args:
            node_id: 节点 ID
            
        Raises:
            NodeNotFoundError: 节点不存在
        """
        # 验证节点存在
        if node_id not in self.trace.tree:
            raise NodeNotFoundError(node_id)
        
        # 更新 Trace 树
        self.trace.set_current_node(node_id)
        self._save_trace()
        
        # 更新 current_node.txt
        self.fs.set_current_node(node_id)
        
        logger.info(f"Switched to node: {node_id}")
    
    def update_node_result(
        self,
        node_id: str,
        result: Dict[str, Any],
        metrics: Dict[str, Any] = None,
    ) -> None:
        """
        更新节点的执行结果
        
        Args:
            node_id: 节点 ID
            result: 新的执行结果
            metrics: 新的指标数据
        """
        node = self.get_node(node_id)
        node.result = result
        if metrics:
            node.metrics.update(metrics)
        node.timestamp = _get_timestamp()
        
        self._save_trace()
    
    def mark_node_completed(self, node_id: str, metrics: Dict = None) -> None:
        """
        标记节点为已完成
        
        Args:
            node_id: 节点 ID
            metrics: 最终指标
        """
        node = self.get_node(node_id)
        node.state_snapshot["status"] = "completed"
        if metrics:
            node.metrics.update(metrics)
        
        self._save_trace()
        
        # 更新文件系统状态
        self.fs.update_node_state(node_id, status="completed", metrics=metrics)
    
    def mark_node_failed(self, node_id: str, error: str = None) -> None:
        """
        标记节点为失败
        
        Args:
            node_id: 节点 ID
            error: 错误信息
        """
        node = self.get_node(node_id)
        node.state_snapshot["status"] = "failed"
        if error:
            node.result = node.result or {}
            node.result["error"] = error
        
        self._save_trace()
        
        # 更新文件系统状态
        self.fs.update_node_state(node_id, status="failed")
    
    # ==================== 路径查询 ====================
    
    def get_path_to_node(self, node_id: str) -> List[str]:
        """
        获取从 root 到指定节点的路径
        
        Args:
            node_id: 节点 ID
            
        Returns:
            路径列表 ["root", "node_001", "node_002", ...]
        """
        return self.trace.get_path_to_node(node_id)
    
    def get_full_action_history(self, node_id: str) -> List[ActionRecord]:
        """
        获取从 root 到指定节点的完整动作历史
        
        每个 node 只保存自己的 action（增量），此方法沿 parent 链回溯重建完整历史
        
        Args:
            node_id: 节点 ID
            
        Returns:
            完整的动作记录列表
        """
        path = self.get_path_to_node(node_id)
        
        full_history = []
        for node in path:
            if node == "root":
                continue
            
            history = self.fs.load_action_history_fact(node)
            full_history.extend(history.actions)
        
        return full_history
    
    async def get_compressed_history_for_llm(
        self,
        llm_client,
        node_id: str,
        max_tokens: int = 2000,
        force_refresh: bool = False
    ) -> List[ActionRecord]:
        """
        获取用于 LLM 渲染的压缩历史（带缓存）
        
        Args:
            llm_client: LLM 客户端实例
            node_id: 节点 ID
            max_tokens: 目标摘要最大 token 数
            force_refresh: 是否强制刷新缓存
            
        Returns:
            压缩后的动作列表
        
        策略：
        1. 检查当前节点是否有有效缓存（source_path 匹配）
        2. 如果无缓存或失效，从 root 回溯重建完整历史
        3. 调用 ActionCompressor 压缩
        4. 保存新的压缩历史作为缓存
        """
        path = self.get_path_to_node(node_id)
        
        # 1. 尝试读取缓存
        if not force_refresh:
            cached = self.fs.load_action_history_compressed(node_id)
            # 校验缓存有效性 (source_path 必须匹配当前路径)
            if hasattr(cached, "is_valid") and cached.is_valid(path):
                return cached.actions
            # 兼容性检查：如果对象没有 is_valid 但有 source_path
            if hasattr(cached, "source_path") and cached.source_path == path:
                return cached.actions
        
        # 2. 重建完整历史
        full_history = self.get_full_action_history(node_id)
        
        # 3. 压缩
        compressor = ActionCompressor(llm_client)
        compressed_actions = await compressor.compress_history(full_history, max_tokens)
        
        # 4. 保存缓存
        new_compressed = ActionHistoryCompressed(
            node_id=node_id,
            turn=len(full_history),
            actions=compressed_actions,
            total_actions=len(full_history),
            compressed_count=len(full_history) - len(compressed_actions),
            source_path=path,
        )
        self.fs.save_action_history_compressed(node_id, new_compressed)
        
        return compressed_actions
    
    def get_node_depth(self, node_id: str) -> int:
        """
        获取节点深度
        
        Args:
            node_id: 节点 ID
            
        Returns:
            节点深度（root 为 0）
        """
        return len(self.get_path_to_node(node_id)) - 1
    
    # ==================== 节点对比 ====================
    
    def compare_nodes(self, node_1: str, node_2: str) -> Dict[str, Any]:
        """
        对比两个节点的路径
        
        Args:
            node_1: 第一个节点 ID
            node_2: 第二个节点 ID
            
        Returns:
            对比结果，包含：
            - path_1: 第一条路径
            - path_2: 第二条路径
            - fork_point: 分叉点（如果路径为空则为 None）
            - metrics_1: 第一条路径的指标
            - metrics_2: 第二条路径的指标
        """
        path_1 = self.get_path_to_node(node_1)
        path_2 = self.get_path_to_node(node_2)
        
        # 处理空路径情况
        if not path_1 or not path_2:
            return {
                "path_1": path_1,
                "path_2": path_2,
                "fork_point": None,
                "metrics_1": self._calculate_path_metrics(path_1),
                "metrics_2": self._calculate_path_metrics(path_2),
            }
        
        # 找到分叉点
        fork_point = None
        for i in range(min(len(path_1), len(path_2))):
            if path_1[i] != path_2[i]:
                fork_point = path_1[i - 1] if i > 0 else "root"
                break
        
        # 如果没找到分叉点，说明一个路径是另一个的前缀
        if fork_point is None:
            if len(path_1) < len(path_2):
                fork_point = path_1[-1]
            elif len(path_2) < len(path_1):
                fork_point = path_2[-1]
            # 如果两条路径完全相同，fork_point 保持为 None
        
        # 计算路径指标
        metrics_1 = self._calculate_path_metrics(path_1)
        metrics_2 = self._calculate_path_metrics(path_2)
        
        return {
            "path_1": path_1,
            "path_2": path_2,
            "fork_point": fork_point,
            "metrics_1": metrics_1,
            "metrics_2": metrics_2,
        }
    
    def _calculate_path_metrics(self, path: List[str]) -> Dict[str, Any]:
        """计算路径的累计指标"""
        total_token = 0
        total_duration = 0
        performance = None
        
        for node_id in path:
            if node_id == "root":
                continue
            
            node = self.trace.get_node(node_id)
            if node and node.metrics:
                total_token += node.metrics.get("token_used", 0)
                total_duration += node.metrics.get("duration_ms", 0)
                if "performance" in node.metrics:
                    performance = node.metrics["performance"]
        
        return {
            "steps": len(path) - 1,
            "total_token": total_token,
            "total_duration_ms": total_duration,
            "performance": performance,
        }
    
    # ==================== 叶节点查询 ====================
    
    def get_all_leaf_nodes(self) -> List[str]:
        """获取所有叶节点 ID"""
        return self.trace.get_all_leaf_nodes()
    
    def get_best_leaf_node(self, metric: str = "performance") -> Optional[str]:
        """
        获取最优叶节点
        
        Args:
            metric: 评价指标（默认 performance）
            
        Returns:
            最优叶节点 ID，如果没有叶节点返回 None
        """
        leaves = self.get_all_leaf_nodes()
        if not leaves:
            return None
        
        best_node = None
        best_value = None
        
        for leaf in leaves:
            node = self.trace.get_node(leaf)
            if node and node.metrics:
                value = node.metrics.get(metric)
                if value is not None:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_node = leaf
        
        return best_node or leaves[0]
    
    # ==================== 并行分叉 ====================
    
    def create_parallel_forks(
        self,
        n: int,
        action_template: Dict[str, Any],
    ) -> List[str]:
        """
        从当前节点创建 N 个并行分叉
        
        注意：这只创建节点结构，不执行实际操作
        
        Args:
            n: 分叉数量
            action_template: 动作模板
            
        Returns:
            新创建的节点 ID 列表
        """
        current_node_id = self.get_current_node()
        fork_nodes = []
        
        for i in range(n):
            # 生成新节点 ID
            new_node_id = self._generate_node_id()
            
            # 获取父节点状态
            try:
                parent_state = self.fs.load_node_state(current_node_id)
                new_turn = parent_state.turn + 1
            except NodeNotFoundError:
                new_turn = 1
            
            # 创建动作（添加分叉索引）
            action = action_template.copy()
            action["fork_index"] = i
            
            # 创建新节点
            new_node = TraceNode(
                node_id=new_node_id,
                parent_id=current_node_id,
                state_snapshot={"turn": new_turn, "status": "pending"},
                action=action,
                result=None,
            )
            
            # 添加到树（但不切换当前节点）
            self.trace.add_node(new_node)
            
            # 创建节点目录
            self._create_fork_node_files(new_node_id, current_node_id, new_turn)
            
            fork_nodes.append(new_node_id)
        
        # 保存 trace.json
        self._save_trace()
        
        logger.info(f"Created {n} parallel forks from {current_node_id}: {fork_nodes}")
        return fork_nodes
    
    def _create_fork_node_files(
        self,
        node_id: str,
        parent_id: str,
        turn: int,
    ) -> None:
        """为分叉节点创建文件"""
        # 复制父节点状态
        if parent_id != "root":
            self.fs.copy_node_state(parent_id, node_id)
            self.fs.copy_code_files(parent_id, node_id)
        else:
            state = NodeState(
                node_id=node_id,
                turn=turn,
                status="pending",
                task_info={"task_id": self.task_id},
            )
            self.fs.save_node_state(node_id, state)
        
        # 更新节点状态
        self.fs.update_node_state(node_id, turn=turn, status="pending")
    
    def complete_fork(
        self,
        node_id: str,
        result: Dict[str, Any],
        metrics: Dict[str, Any] = None,
    ) -> None:
        """
        完成分叉节点的执行
        
        Args:
            node_id: 节点 ID
            result: 执行结果
            metrics: 指标数据
        """
        node = self.get_node(node_id)
        node.result = result
        node.state_snapshot["status"] = "completed"
        if metrics:
            node.metrics.update(metrics)
        node.timestamp = _get_timestamp()
        
        # 创建动作记录
        action_record = ActionRecord(
            action_id=self._generate_action_id(),
            tool_name=node.action.get("type", "unknown") if node.action else "unknown",
            arguments=node.action or {},
            result=result,
        )
        
        history = ActionHistoryFact(
            node_id=node_id,
            parent_node_id=node.parent_id,
            turn=node.state_snapshot.get("turn", 0),
        )
        history.add_action(action_record)
        self.fs.save_action_history_fact(node_id, history)
        
        # 更新文件系统状态
        self.fs.update_node_state(node_id, status="completed", metrics=metrics)
        
        self._save_trace()
    
    # ==================== 树可视化 ====================
    
    def visualize_tree(self, show_full: bool = False) -> str:
        """
        可视化 Trace 树
        
        Args:
            show_full: 是否显示完整树（不折叠）
            
        Returns:
            树的字符串表示
        """
        current_node = self.trace.current_node
        lines = [f"🌳 Trace Tree (当前: {current_node}):\n"]
        
        # 从 root 开始递归构建
        tree_str = self._build_tree_str("root", current_node, indent=0, show_full=show_full)
        lines.append(tree_str)
        
        lines.append("\n💡 使用 /trace show <node> 查看节点详情")
        lines.append("💡 使用 /trace switch <node> 切换节点")
        
        return "\n".join(lines)
    
    def _build_tree_str(
        self,
        node_id: str,
        current_node: str,
        indent: int,
        show_full: bool,
    ) -> str:
        """递归构建树的字符串表示"""
        node = self.trace.get_node(node_id)
        if node is None:
            return ""
        
        prefix = "  " * indent
        is_current = " ⭐ 当前" if node_id == current_node else ""
        
        if node_id == "root":
            node_str = f"{prefix}root{is_current}\n"
        else:
            # 获取节点简短编号
            short_id = node_id.split("_")[1] if "_" in node_id else node_id
            action_type = node.action.get("type", "unknown") if node.action else "init"
            result_summary = self._summarize_result(node.result)
            
            node_str = f"{prefix}[{short_id}] {action_type}{is_current}\n"
            if result_summary:
                node_str += f"{prefix}    → {result_summary}\n"
        
        # 递归处理子节点
        children = node.children
        if not children:
            return node_str
        
        if len(children) == 1:
            node_str += f"{prefix}  ↓\n"
            node_str += self._build_tree_str(children[0], current_node, indent, show_full)
        else:
            # 多个子节点，显示分叉
            for i, child in enumerate(children):
                if i == 0:
                    node_str += f"{prefix}  ↓\n"
                    node_str += f"{prefix}  ├─ "
                elif i == len(children) - 1:
                    node_str += f"{prefix}  │\n"
                    node_str += f"{prefix}  └─ "
                else:
                    node_str += f"{prefix}  │\n"
                    node_str += f"{prefix}  ├─ "
                
                # 移除子节点的前缀缩进，因为已经在上面处理了
                child_str = self._build_tree_str(child, current_node, indent + 2, show_full)
                # 去掉第一行的缩进
                child_lines = child_str.split("\n")
                if child_lines:
                    child_lines[0] = child_lines[0].strip()
                    node_str += "\n".join(child_lines)
        
        return node_str
    
    def _summarize_result(self, result: Optional[Dict]) -> str:
        """总结执行结果"""
        if result is None:
            return ""
        
        summaries = []
        
        if "success" in result:
            status = "✅" if result["success"] else "❌"
            summaries.append(status)
        
        if "performance" in result:
            perf = result["performance"]
            if isinstance(perf, (int, float)):
                summaries.append(f"性能: {perf:.1%}" if perf <= 1 else f"性能: {perf}")
        
        if "output" in result:
            output = str(result["output"])
            if len(output) > 30:
                output = output[:30] + "..."
            summaries.append(output)
        
        if "lines" in result:
            summaries.append(f"{result['lines']} 行")
        
        return " | ".join(summaries) if summaries else ""
    
    def get_node_detail(self, node_id: str) -> str:
        """
        获取节点详情
        
        Args:
            node_id: 节点 ID
            
        Returns:
            节点详情字符串
        """
        node = self.get_node(node_id)
        
        lines = [f"📋 节点详情: {node_id}\n"]
        
        # 节点信息
        lines.append("📍 节点信息:")
        lines.append(f"  • ID: {node.node_id}")
        lines.append(f"  • 父节点: {node.parent_id or 'None'}")
        if node.children:
            lines.append(f"  • 子节点: [{', '.join(node.children)}] ({len(node.children)} 个)")
        else:
            lines.append("  • 子节点: 无（叶节点）")
        lines.append(f"  • 时间: {node.timestamp}")
        lines.append("")
        
        # 执行动作
        if node.action:
            lines.append("📝 执行动作:")
            lines.append(f"  • 类型: {node.action.get('type', 'unknown')}")
            if node.action.get("params"):
                lines.append(f"  • 参数: {json.dumps(node.action['params'], ensure_ascii=False)}")
            lines.append("")
        
        # 执行结果
        if node.result:
            lines.append("📊 执行结果:")
            for key, value in node.result.items():
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                lines.append(f"  • {key}: {value}")
            lines.append("")
        
        # 指标
        if node.metrics:
            lines.append("📈 指标:")
            for key, value in node.metrics.items():
                lines.append(f"  • {key}: {value}")
            lines.append("")
        
        lines.append(f"💡 /trace switch {node_id} 切换到此节点")
        
        return "\n".join(lines)
    
    def get_path_detail(self, node_id: str) -> str:
        """
        获取从 root 到指定节点的路径详情
        
        Args:
            node_id: 节点 ID
            
        Returns:
            路径详情字符串
        """
        path = self.get_path_to_node(node_id)
        metrics = self._calculate_path_metrics(path)
        
        lines = [f"📍 路径: root → {node_id}\n"]
        
        for i, nid in enumerate(path):
            node = self.trace.get_node(nid)
            if node is None:
                continue
            
            if nid == "root":
                lines.append("root (初始状态)")
            else:
                short_id = nid.split("_")[1] if "_" in nid else nid
                action_type = node.action.get("type", "unknown") if node.action else "init"
                result_summary = self._summarize_result(node.result)
                
                # 检查是否是分叉点
                parent = self.trace.get_node(node.parent_id) if node.parent_id else None
                is_fork = parent and len(parent.children) > 1
                fork_marker = " 🌿 分叉点" if is_fork else ""
                
                lines.append(f"  ↓")
                lines.append(f"[{short_id}] {action_type}{fork_marker}")
                if result_summary:
                    lines.append(f"    → {result_summary}")
                if node.metrics.get("token_used"):
                    lines.append(f"    Token: {node.metrics['token_used']}")
        
        lines.append("")
        lines.append("总计:")
        lines.append(f"  • 步数: {metrics['steps']}")
        lines.append(f"  • Token: {metrics['total_token']}")
        if metrics['performance'] is not None:
            lines.append(f"  • 性能: {metrics['performance']}")
        
        return "\n".join(lines)
    
    # ==================== 任务恢复 ====================
    
    def get_resume_info(self) -> Dict[str, Any]:
        """
        获取任务恢复所需的信息
        
        Returns:
            包含恢复所需信息的字典
        """
        current_node = self.get_current_node()
        
        try:
            state = self.fs.load_node_state(current_node)
        except NodeNotFoundError:
            state = None
        
        action_history = self.get_full_action_history(current_node)
        pending_tools = self.fs.load_pending_tools(current_node)
        thinking = self.fs.load_thinking(current_node)
        
        return {
            "task_id": self.task_id,
            "current_node": current_node,
            "state": state,
            "action_history": action_history,
            "pending_tools": pending_tools,
            "thinking": thinking,
            "path": self.get_path_to_node(current_node),
        }
