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

import difflib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

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
            base_dir: 基础目录，默认 ~/.akg
        """
        self.task_id = task_id
        self.fs = FileSystemState(task_id, base_dir)
        self._node_counter = 0
        self._action_counter = 0
        self._trace: Optional[TraceTree] = None
        
        # ---- 工作流日志上下文（通过 configure_logging 设置）----
        self._log_dir: Optional[str] = None       # 日志根目录
        self._log_category: str = ""               # 日志分类子目录（调用方自行指定）
        self._log_task_id: str = ""                # 日志中使用的 task_id
        self._step_records: List[Any] = []         # 步骤记录列表（用于步骤计数）
    
    @property
    def trace(self) -> TraceTree:
        """获取 Trace 树（自动加载）"""
        if self._trace is None:
            self.initialize()
        return self._trace
    
    # ==================== 初始化 ====================
    
    def initialize(self, task_input: str = "", force: bool = False) -> None:
        """
        初始化 Trace 系统
        
        创建 trace.json 和 root 节点
        
        Args:
            task_input: 任务输入（用户请求）
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
        # 注意：实际保存到 root 节点的 state.json 是在 _create_root_node 中做的（如果不在这里处理 task_input，那边就需要传参）
        # 这里我们在 TraceTree 的内存结构中保存，并随后触发 _save_trace 持久化
        state_snapshot = {"turn": 0, "status": "init"}
        if task_input:
            state_snapshot["task_info"] = {"task_input": task_input}
            
        root_node = TraceNode(
            node_id="root",
            parent_id=None,
            state_snapshot=state_snapshot,
            action=None,
            result=None,
        )
        self._trace.add_node(root_node)
        
        
        # 保存
        self._save_trace()
        
        # 加载 .traceconfig 配置
        self._load_trace_config()
        
        # 将 User Request 持久化为 root 的 Action
        if task_input:
            action_record = ActionRecord(
                action_id="task_init",
                tool_name="user_request",
                arguments={"content": task_input},
                result={"status": "received"},
                timestamp=root_node.timestamp
            )
            history = ActionHistoryFact(
                node_id="root",
                parent_node_id=None,
                turn=0,
            )
            history.add_action(action_record)
            self.fs.save_action_history_fact("root", history)
            
        logger.info(f"Initialized trace system: {self.task_id}")

    def _load_trace_config(self):
        """加载 .traceconfig 文件"""
        config_path = self.fs.base_dir / ".traceconfig"
        self.include_patterns = ["code/"]  # 默认值
        self.exclude_patterns = []
        
        if config_path.exists():
            try:
                content = config_path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                        
                    if line.startswith("!"):
                        self.exclude_patterns.append(line[1:])
                    else:
                        self.include_patterns.append(line)
                logger.info(f"Loaded trace config: include={self.include_patterns}, exclude={self.exclude_patterns}")
            except Exception as e:
                logger.warning(f"Failed to load .traceconfig: {e}")



    
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
        # 复制父节点状态 (应用 .traceconfig 配置)
        self.fs.copy_node_state(
            parent_id, 
            node_id,
            include_patterns=getattr(self, "include_patterns", None),
            exclude_patterns=getattr(self, "exclude_patterns", None)
        )
        # Snapshot Filesystem: copy_node_state 使用硬链接处理代码快照的继承
        
        # 更新节点状态
        self.fs.update_node_state(node_id, turn=turn, status="running")

        
        # 创建增量动作历史
        # 如果 action 中没有显式的 arguments 或 params，则将除 type 外的所有字段作为 arguments
        action_args = action.get("arguments", action.get("params"))
        if action_args is None:
            action_args = {k: v for k, v in action.items() if k != "type"}

        action_record = ActionRecord(
            action_id=self._generate_action_id(),
            tool_name=action.get("type", "unknown"),
            arguments=action_args,
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
    
    def fork_ask_user(self, node_id: str) -> str:
        """
        在 ask_user 节点上创建分叉
        
        语义：复制 agent 的提问（action），清空用户回答，
        新节点成为原节点父节点的新子节点。原有历史完全不动。
        
        Args:
            node_id: 目标 ask_user 节点 ID
            
        Returns:
            新创建的节点 ID
            
        Raises:
            NodeNotFoundError: 节点不存在
            TraceSystemError: 节点不是 ask_user 类型
        """
        # 验证节点存在
        node = self.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)
        
        # 验证是 ask_user 类型
        action_type = (node.action or {}).get("type", "")
        if action_type != "ask_user":
            raise TraceSystemError(
                f"只能对 ask_user 节点执行 fork，"
                f"节点 {node_id} 的类型是 '{action_type}'"
            )
        
        # 获取父节点
        parent_id = node.parent_id
        if not parent_id:
            raise TraceSystemError(f"节点 {node_id} 没有父节点，无法 fork")
        
        # 复制 action（agent 的提问不变）
        import copy
        from datetime import datetime
        new_action = copy.deepcopy(node.action)
        new_action["timestamp"] = datetime.now().isoformat()
        
        # 清空用户回答
        message = (node.action.get("arguments") or {}).get("message", "")
        new_result = {"status": "waiting", "message": message}
        
        # 临时切换到父节点，让 add_node 在父节点下创建子节点
        old_current = self.get_current_node()
        self.switch_node(parent_id)
        
        # 创建新节点
        new_node_id = self.add_node(
            action=new_action,
            result=new_result,
            metrics={"duration_ms": 0},
        )
        
        logger.info(
            f"Forked ask_user: {node_id} -> {new_node_id} "
            f"(parent: {parent_id}, siblings: {len(self.get_node(parent_id).children)})"
        )
        
        return new_node_id
    
    def update_node_result(
        self,
        node_id: str,
        result: Dict[str, Any],
        metrics: Dict[str, Any] = None,
    ) -> None:
        """
        更新节点的执行结果
        
        同时更新：
        1. trace.json 中的 TraceNode.result
        2. trace.json 中的 TraceNode.state_snapshot["status"]
        3. 文件系统中的 state.json（NodeState.status）
        4. action_history_fact.json 中最后一条 action 的 result
        
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
        
        # 根据 result 中的 status 推断节点状态
        result_status = result.get("status", "") if isinstance(result, dict) else ""
        if result_status in ("fail", "error", "failed"):
            node_status = "failed"
        else:
            node_status = "completed"
        
        # 1. 更新 TraceNode 的 state_snapshot
        node.state_snapshot["status"] = node_status
        
        # 2. 保存 trace.json
        self._save_trace()
        
        # 3. 更新文件系统中的 state.json
        self.fs.update_node_state(node_id, status=node_status, metrics=metrics)
        
        # 4. 更新 action_history_fact.json 中的 result（替换掉占位的 pending）
        try:
            history_fact = self.fs.load_action_history_fact(node_id)
            if history_fact.actions:
                last_action = history_fact.actions[-1]
                last_action.result = result
                if metrics and "duration_ms" in metrics:
                    last_action.duration_ms = metrics["duration_ms"]
                self.fs.save_action_history_fact(node_id, history_fact)
        except Exception as e:
            logger.warning(f"Failed to update action history fact for {node_id}: {e}")
    
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
            # 移除 root 跳过逻辑，因为 root 现在可能包含 user_request action
            # if node == "root":
            #     continue
            
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
            # 校验缓存有效性 (source_path 必须匹配当前路径，且缓存有内容)
            if cached.actions:
                if hasattr(cached, "is_valid") and cached.is_valid(path):
                    return cached.actions
        
        # 2. 重建完整历史
        full_history = self.get_full_action_history(node_id)
        
        # 如果没有历史，直接返回空列表
        if not full_history:
            return []
        
        # 如果历史很短，跳过压缩，直接返回原始历史（节省 LLM 调用）
        if len(full_history) <= 5:
            return full_history
        
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
            节点深度（root 为 0），路径无效时返回 0
        """
        path = self.get_path_to_node(node_id)
        return len(path) - 1 if path else 0
    
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
            "node_count": len(path),
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
            
        Note:
            如果所有叶节点都没有指定的 metric 字段，则默认返回 leaves[0]（第一个叶节点）
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
        
        # 如果没有找到带有 metric 的节点，返回第一个叶节点
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
        self.fs.copy_node_state(parent_id, node_id)
        
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
        # 处理 action 字段，映射到 tool_name 和 arguments
        action_dict = node.action or {}
        tool_name = action_dict.get("type", "unknown")
        
        action_args = action_dict.get("arguments", action_dict.get("params"))
        if action_args is None:
            action_args = {k: v for k, v in action_dict.items() if k != "type"}

        action_record = ActionRecord(
            action_id=self._generate_action_id(),
            tool_name=tool_name,
            arguments=action_args,
            result=result,
        )
        
        # 加载现有历史（如果存在），避免覆盖
        history = self.fs.load_action_history_fact(node_id)
        history.parent_node_id = node.parent_id
        history.turn = node.state_snapshot.get("turn", 0)
        history.add_action(action_record)
        self.fs.save_action_history_fact(node_id, history)
        
        # 更新文件系统状态
        self.fs.update_node_state(node_id, status="completed", metrics=metrics)
        
        self._save_trace()
    
    # ==================== 树可视化 ====================
    
    def visualize_tree(self, focus_node: str = None, depth: int = 4) -> str:
        """
        可视化 Trace 树（纯文本版本，用于非 Rich 环境）
        
        Args:
            focus_node: 焦点节点 ID，默认为当前节点
            depth: 向上/向下展示的层数，默认 4
        """
        from .trace_visualizer import visualize_text
        return visualize_text(self, focus_node=focus_node, depth=depth)
    
    def visualize_tree_rich(self, focus_node: str = None, depth: int = 4):
        """
        可视化 Trace 树（Rich Text 版本，用于 CLI 终端）
        
        Args:
            focus_node: 焦点节点 ID，默认为当前节点
            depth: 向上/向下展示的层数，默认 4
            
        Returns:
            rich.text.Text 对象，可直接传给 console.print() / Panel()
        """
        from .trace_visualizer import visualize_rich
        return visualize_rich(self, focus_node=focus_node, depth=depth)
    
    def get_node_detail(self, node_id: str) -> str:
        """
        获取节点详情
        
        ask_user 节点：分区展示 "Agent 提问" 和 "用户回答"
        工具节点：展示输入参数 + 执行结果 + 性能数据
        
        Args:
            node_id: 节点 ID
            
        Returns:
            节点详情字符串
        """
        node = self.get_node(node_id)
        
        action = node.action or {}
        result = node.result or {}
        action_type = action.get("type", "")
        args = action.get("arguments", {}) or {}
        
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
        
        if action_type == "ask_user":
            # ask_user 节点：分区展示
            message = args.get("message", result.get("message", ""))
            user_resp = result.get("user_response", "")
            status = result.get("status", "?")
            
            lines.append("🗣️ Agent 提问:")
            if message:
                for line in message.split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append("  (无消息)")
            lines.append("")
            
            if status == "responded" and user_resp:
                lines.append("💬 用户回答:")
                for line in user_resp.split("\n"):
                    lines.append(f"  {line}")
            elif status == "waiting":
                lines.append("⏳ 状态: 等待用户回答")
            elif status == "skipped":
                lines.append("⏭️ 状态: 已跳过（用户未回答）")
            else:
                lines.append(f"  状态: {status}")
            lines.append("")
            
            short_id = node_id.replace("node_", "")
            lines.append(f"💡 /trace fork {short_id} — 在此决策点创建新分支，重新回答")
        else:
            # 工具节点
            lines.append("📝 执行动作:")
            lines.append(f"  • 类型: {action_type or 'unknown'}")
            if args:
                lines.append("  • 参数:")
                for k, v in args.items():
                    v_str = str(v) if v is not None else "None"
                    if len(v_str) > 100:
                        v_str = v_str[:100] + "..."
                    lines.append(f"    - {k}: {v_str}")
            elif action.get("params"):
                lines.append(f"  • 参数: {json.dumps(action['params'], ensure_ascii=False)}")
            lines.append("")
            
            # 执行结果
            if result:
                lines.append("📊 执行结果:")
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    lines.append(f"  • {key}: {value}")
                lines.append("")
        
        # 指标
        if node.metrics:
            lines.append("📈 指标:")
            for key, value in node.metrics.items():
                lines.append(f"  • {key}: {value}")
            lines.append("")
        
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
        
        # 处理空路径情况
        if not path:
            return f"❌ 无法找到节点 '{node_id}' 的路径（节点不存在或路径断裂）"
        
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
                from .trace_visualizer import _summarize_result
                result_summary = _summarize_result(node.result)
                
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
    
    # ==================== 工作流日志记录 ====================
    
    def configure_logging(self, log_dir: str, category: str = "", task_id: str = "") -> None:
        """
        配置工作流日志上下文
        
        在 workflow 被调用时，由 prepare_config 调用此方法，
        使得后续的 log_record / log_merged_record 能正确写出日志文件。
        
        Args:
            log_dir: 日志文件根目录（如 node_00X/logs/）
            category: 分类子目录名（调用方自行指定），文件保存在 {log_dir}/{category}/ 下
            task_id: 日志文件名中使用的任务标识
        """
        self._log_dir = log_dir
        self._log_category = category
        self._log_task_id = task_id or self.task_id
        self._step_records = []
        logger.info(f"[TraceSystem] 日志上下文已配置: log_dir={log_dir}, category={category}, task_id={self._log_task_id}")
    
    def get_step_count(self) -> int:
        """获取当前日志步骤计数（已记录的 log_record 次数）"""
        return len(self._step_records)
    
    def get_log_task_id(self) -> str:
        """获取日志文件名中使用的 task_id（由 configure_logging 设置）"""
        return self._log_task_id or self.task_id
    
    def log_record(self, record_name: str, params: list,
                   subdirectory: str = None) -> int:
        """
        记录一个工作流步骤，并将参数保存为独立文件。
        
        每调用一次，内部步骤计数器自增。每个 (param_name, content) 对
        保存为一个独立的 .txt 文件，仅 None 会被跳过（空字符串也会保存）。
        
        Args:
            record_name: 记录名称，用于文件名中标识来源
            params: 参数列表，格式为 [('param_name', 'content'), ...]
            subdirectory: 可选的子目录名（在 {log_dir}/{category}/ 之下）
        
        Returns:
            当前步骤号（自增后）
        """
        # 即使未配置日志目录，也自增步骤计数器以保持编号一致
        self._step_records.append({"name": record_name})
        step = len(self._step_records)
        
        if not self._log_dir:
            logger.debug(f"[TraceSystem] log_record({record_name}): 日志上下文未配置，跳过文件保存")
            return step
        
        import os
        expanded_log_dir = os.path.expanduser(self._log_dir)
        target_dir = os.path.join(expanded_log_dir, self._log_category)
        if subdirectory:
            target_dir = os.path.join(target_dir, subdirectory)
        os.makedirs(target_dir, exist_ok=True)
        
        base_name = f"Iteration{self._log_task_id}_Step{step:02d}_{self._log_category}_{record_name}_"
        
        for param_name, content in params:
            if content is None:
                continue
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))
        
        return step
    
    def log_merged_record(self, record_name: str, params: list,
                          filename_suffix: str = "_parsed.txt",
                          subdirectory: str = "recorder") -> None:
        """
        将多个参数合并保存到单个文件（不自增步骤计数器）。
        
        适用于将解析后的结构化内容集中存储。
        
        文件命名:
            Iteration{task_id}_Step{NN:02d}_{category}_{record_name}{filename_suffix}
        
        文件目录:
            {log_dir}/{category}/{subdirectory}/
        
        Args:
            record_name: 记录名称
            params: 参数列表，格式为 [('section_name', 'content'), ...]
            filename_suffix: 文件名后缀（默认 "_parsed.txt"）
            subdirectory: 子目录名（默认 "recorder"）
        """
        if not self._log_dir or not params:
            return
        
        import os
        expanded_log_dir = os.path.expanduser(self._log_dir)
        target_dir = os.path.join(expanded_log_dir, self._log_category, subdirectory)
        os.makedirs(target_dir, exist_ok=True)
        
        step = len(self._step_records)
        base_name = f"Iteration{self._log_task_id}_Step{step:02d}_{self._log_category}_{record_name}{filename_suffix}"
        file_path = os.path.join(target_dir, base_name)
        
        content_parts = []
        for section_name, content in params:
            if content:
                content_parts.append(f"=== {section_name} ===\n{content}")
        
        if content_parts:
            combined_content = "\n\n\n".join(content_parts)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(combined_content)
    
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
            # 节点状态不存在时，返回一个最小化的默认状态
            state = NodeState(
                node_id=current_node,
                turn=0,
                status="unknown",
                task_info={"task_id": self.task_id},
            )
        
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

    # ==================== 合并功能 ====================

    def find_lca(self, node_id1: str, node_id2: str) -> str:
        """
        寻找两个节点的最近公共祖先 (Lowest Common Ancestor)
        
        Args:
            node_id1: 节点 1 ID
            node_id2: 节点 2 ID
            
        Returns:
            LCA 节点的 ID
        """
        path1 = self.get_path_to_node(node_id1)
        path2 = self.get_path_to_node(node_id2)
        
        lca = "root"
        for n1, n2 in zip(path1, path2):
            if n1 == n2:
                lca = n1
            else:
                break
        return lca

    def merge_nodes(self, target_node_id: str, source_node_id: str) -> str:
        """
        将 source_node 的变更合并到 target_node
        
        算法: 三路合并 (3-way Merge)，基准为两者的 LCA。
        
        Args:
            target_node_id: 目标节点 (Yours/HEAD)
            source_node_id: 源节点 (Theirs/Merging)
            
        Returns:
            新生成的合并节点 ID
        """
        # 1. 寻找 LCA
        lca_id = self.find_lca(target_node_id, source_node_id)
        logger.info(f"Merging {source_node_id} into {target_node_id} (LCA: {lca_id})")
        
        # 2. 如果 source 是 target 的祖先，无需合并
        if lca_id == source_node_id:
            logger.info("Source is an ancestor of target. No-op merge.")
            return target_node_id
            
        # 3. 如果 target 是 source 的祖先，执行 Fast-Forward (逻辑上即切换到 source 的新分支)
        # 这里为了 Trace 树的严谨性，我们还是创建一个新节点，但代码完全等于 source
        
        # 4. 创建合并节点
        action = {
            "type": "merge",
            "params": {
                "source_node": source_node_id,
                "target_node": target_node_id,
                "lca_node": lca_id
            }
        }
        
        # 临时切换到 target 节点以准备合并环境
        self.switch_node(target_node_id)
        
        # 添加合并节点
        merge_node_id = self.add_node(
            action=action,
            result={"status": "merging"}
        )
        
        # 5. 执行文件级 3-way 合并
        state_lca = self.fs.load_node_state(lca_id)
        state_yours = self.fs.load_node_state(target_node_id)
        state_theirs = self.fs.load_node_state(source_node_id)
        
        all_file_keys = (
            set(state_lca.file_state.keys()) | 
            set(state_yours.file_state.keys()) | 
            set(state_theirs.file_state.keys())
        )
        
        conflicts = []
        for file_key in all_file_keys:
            filename = file_key.replace("code/", "")
            
            # 读取三个版本的内容
            base_content = self._get_file_content_safe(lca_id, filename)
            yours_content = self._get_file_content_safe(target_node_id, filename)
            theirs_content = self._get_file_content_safe(source_node_id, filename)
            
            # 三路合并逻辑
            merged_content, has_conflict = self._three_way_merge(base_content, yours_content, theirs_content)
            
            if has_conflict:
                conflicts.append(filename)
                logger.warning(f"Merge conflict in {filename}")
            
            # 保存到合并节点
            self.fs.save_code_file(merge_node_id, filename, merged_content)
            
        # 6. 更新结果
        status = "completed" if not conflicts else "conflict"
        self.update_node_result(merge_node_id, {
            "status": status,
            "conflicts": conflicts,
            "merged_from": source_node_id,
            "base_lca": lca_id
        })
        
        # 同步状态到文件系统
        self.fs.update_node_state(merge_node_id, status=status)
        
        return merge_node_id

    def _get_file_content_safe(self, node_id: str, filename: str) -> Optional[str]:
        """安全读取文件内容，不存在则返回 None"""
        try:
            return self.fs.load_code_file(node_id, filename)
        except Exception:
            return None

    def _three_way_merge(self, base: Optional[str], yours: Optional[str], theirs: Optional[str]) -> Tuple[str, bool]:
        """
        核心三路合并算法 (无外部依赖版)

        能力:
        - 基于 difflib 实现，无需 merge3 库
        - 双方修改不同区域 → 自动合并，无冲突
        - 双方修改相同行 → 精确标记冲突区域
        - 一方删除/另一方修改 → 标记冲突

        Returns: (合并后内容, 是否有冲突)
        """
        # 情况 1: 两人修改一致，或都没改
        if yours == theirs:
            return yours or "", False

        # 情况 2: 只有一方改了（或者一方删了，另一方没动）
        if yours == base:
            return theirs or "", False
        if theirs == base:
            return yours or "", False

        # 情况 3: 双方都改了 — 使用行级合并
        if base is not None and yours is not None and theirs is not None:
            return self._line_level_merge(base, yours, theirs)

        # 情况 4: 一方删了文件，另一方修改了 → 冲突
        conflict_marker = f"<<<<<<< YOURS\n{yours or ''}\n=======\n{theirs or ''}\n>>>>>>> THEIRS\n"
        return conflict_marker, True

    def _line_level_merge(self, base: str, yours: str, theirs: str) -> Tuple[str, bool]:
        """
        行级三路合并 (基于 difflib 原生实现，无外部依赖)
        
        策略:
        1. 计算 Base -> Yours 的差异 (hunks_a)
        2. 计算 Base -> Theirs 的差异 (hunks_b)
        3. 归并两组 hunks，按照在 Base 中的起始行排序
        4. 检测重叠区域 -> 冲突；无重叠 -> 应用修改
        """
        lines_base = base.splitlines(True)
        lines_a = yours.splitlines(True)
        lines_b = theirs.splitlines(True)

        matcher_a = difflib.SequenceMatcher(None, lines_base, lines_a)
        matcher_b = difflib.SequenceMatcher(None, lines_base, lines_b)

        # 获取差异块 (tag, i1, i2, j1, j2)
        # i: base 索引, j: 修改后索引
        # 我们只关心非 'equal' 的块
        hunks_a = [op for op in matcher_a.get_opcodes() if op[0] != 'equal']
        hunks_b = [op for op in matcher_b.get_opcodes() if op[0] != 'equal']

        # 合并两组 chunks 并按 base 索引排序
        # 结构: (i1, i2, 'A'/'B', op, lines_branch)
        events = []
        for op in hunks_a:
            events.append({
                'start': op[1], 'end': op[2], 
                'source': 'A', 'op': op, 'lines': lines_a[op[3]:op[4]]
            })
        for op in hunks_b:
            events.append({
                'start': op[1], 'end': op[2], 
                'source': 'B', 'op': op, 'lines': lines_b[op[3]:op[4]]
            })
        
        # 按 base 的 start 排序
        events.sort(key=lambda x: x['start'])

        output_lines = []
        base_idx = 0
        has_conflict = False
        
        i = 0
        while i < len(events):
            evt = events[i]
            
            # 1. 先把 base 中未修改的部分补上 (从当前 base_idx 到 evt['start'])
            # 注意：如果有重叠冲突，base_idx 可能已经超过了 evt['start']，这时不补
            if base_idx < evt['start']:
                output_lines.extend(lines_base[base_idx : evt['start']])
                base_idx = evt['start']
            
            # 2. 检查 Op 是否与下一个 Op 冲突 (重叠)
            # 冲突条件: 下一个事件的 start < 当前事件的 end
            conflict_group = [evt]
            max_end = evt['end']
            
            j = i + 1
            while j < len(events):
                next_evt = events[j]
                if next_evt['start'] < max_end:
                    # 发现重叠！加入冲突组
                    conflict_group.append(next_evt)
                    max_end = max(max_end, next_evt['end'])
                    j += 1
                else:
                    break
            
            # 处理这一组
            if len(conflict_group) == 1:
                # 无冲突，直接应用修改
                output_lines.extend(evt['lines'])
                base_idx = max(base_idx, evt['end'])
            else:
                # 有冲突
                has_conflict = True
                # 注意：这里简化处理，直接把冲突组涉及的 base 区域标记为冲突
                # 实际上 A 和 B 可能动了同一个 base 区域，但也可能只是部分重叠
                
                # 区分 A 和 B 的修改内容
                # 简单粗暴：把冲突区域内的 A 的改动和 B 的改动分别提取
                # 这里的逻辑比较复杂，为了简化，我们将所有涉及的 base 区域视为冲突区
                # 并把 A 和 B 对应的 target 内容全部放进去（可能包含重复，但能保证不丢代码）
                
                # 更精细的做法是 merge3 的算法，这里我们做一个简化版：
                # 既然冲突了，我们把 A 的修改作为 YOURS，B 的修改作为 THEIRS
                # 但 A 和 B 可能对应 base 的不同子区域，直接拼接可能会乱
                
                # 妥协方案: 
                # 对于冲突组，我们取出 A 所有的 hunk 内容拼在一起，B 所有 hunk 拼在一起
                
                content_a = []
                content_b = []
                for c in conflict_group:
                    if c['source'] == 'A': content_a.extend(c['lines'])
                    if c['source'] == 'B': content_b.extend(c['lines'])
                
                output_lines.append("<<<<<<< YOURS\n")
                output_lines.extend(content_a)
                output_lines.append("=======\n")
                output_lines.extend(content_b)
                output_lines.append(">>>>>>> THEIRS\n")
                
                base_idx = max(base_idx, max_end)
            
            i = j
            
        # 补上最后的 base
        if base_idx < len(lines_base):
            output_lines.extend(lines_base[base_idx:])
            
        return "".join(output_lines), has_conflict

    def blame_file(self, node_id: str, filename: str) -> List[Dict[str, Any]]:
        """
        追踪单个文件在从 root 到指定节点路径上的演化历史

        沿 parent 链遍历，比对相邻节点中同一文件的 checksum，
        记录文件被创建、修改或删除的节点。

        Args:
            node_id: 目标节点 ID
            filename: 文件名（相对于 code/ 目录，如 "main.py"）

        Returns:
            变更记录列表，每条包含:
            - node_id: 发生变更的节点 ID
            - action: 该节点执行的动作类型（如 "generate", "modify" 等）
            - change_type: "created" | "modified" | "deleted"
            - checksum: 变更后的文件 checksum（deleted 时为 None）

        Raises:
            NodeNotFoundError: node_id 不存在
        """
        path = self.get_path_to_node(node_id)
        file_key = f"code/{filename}"
        changes: List[Dict[str, Any]] = []
        prev_checksum = None

        for nid in path:
            try:
                state = self.fs.load_node_state(nid)
            except Exception:
                # 如果节点状态加载失败（损坏等），跳过
                logger.warning(f"blame_file: 无法加载节点 {nid} 的状态，跳过")
                continue

            file_info = state.file_state.get(file_key)
            cur_checksum = file_info.get("checksum") if file_info else None

            if cur_checksum == prev_checksum:
                # 文件未变化，跳过
                continue

            # 确定变更类型
            if prev_checksum is None and cur_checksum is not None:
                change_type = "created"
            elif prev_checksum is not None and cur_checksum is None:
                change_type = "deleted"
            else:
                change_type = "modified"

            # 获取动作类型
            try:
                node = self.get_node(nid)
                action_type = (
                    node.action.get("type", "unknown") if node.action else "init"
                )
            except Exception:
                action_type = "unknown"

            changes.append({
                "node_id": nid,
                "action": action_type,
                "change_type": change_type,
                "checksum": cur_checksum,
            })

            prev_checksum = cur_checksum

        return changes

    def blame_all_files(self, node_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        追踪指定节点上所有文件的演化历史

        Args:
            node_id: 目标节点 ID

        Returns:
            {filename: [blame_records]} 的字典
        """
        # 收集路径上所有节点出现过的文件名
        path = self.get_path_to_node(node_id)
        all_filenames: set = set()
        for nid in path:
            try:
                state = self.fs.load_node_state(nid)
            except Exception:
                continue
            for key in state.file_state:
                if key.startswith("code/"):
                    all_filenames.add(key[len("code/"):])

        result: Dict[str, List[Dict[str, Any]]] = {}
        for filename in sorted(all_filenames):
            records = self.blame_file(node_id, filename)
            if records:
                result[filename] = records
        return result
