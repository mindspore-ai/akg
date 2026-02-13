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
AIKG FileSystem 数据模型

定义了 Trace 系统和状态管理所需的所有数据结构。
所有模型都支持 JSON 序列化/反序列化。
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


def _get_timestamp() -> str:
    """获取当前 ISO 格式时间戳"""
    return datetime.now().isoformat()


@dataclass
class AgentInfo:
    """Agent 信息"""
    agent_name: str
    agent_id: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AgentInfo":
        return cls(**data)


@dataclass
class TaskInfo:
    """任务信息（通用，不含领域特定字段）"""
    task_id: str
    task_input: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain: str = ""  # 会话所属领域，如 "op", "common", "graph" 等
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TaskInfo":
        # 把领域特定字段迁移到 metadata
        known_fields = {"task_id", "task_input", "metadata"}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        if extra:
            metadata = filtered.get("metadata", {})
            if isinstance(metadata, dict):
                metadata.update(extra)
            filtered["metadata"] = metadata
        return cls(**filtered)


@dataclass
class ExecutionInfo:
    """执行信息"""
    tool_call_counter: int = 0
    first_thinking_done: bool = False
    current_turn: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ExecutionInfo":
        return cls(**data)


@dataclass
class FileInfo:
    """文件信息"""
    size: int
    last_modified: str
    checksum: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FileInfo":
        return cls(**data)


@dataclass
class Metrics:
    """指标数据"""
    token_used: int = 0
    duration_ms: int = 0
    performance: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = {
            "token_used": self.token_used,
            "duration_ms": self.duration_ms,
        }
        if self.performance is not None:
            result["performance"] = self.performance
        if self.extra:
            result.update(self.extra)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Metrics":
        return cls(
            token_used=data.get("token_used", 0),
            duration_ms=data.get("duration_ms", 0),
            performance=data.get("performance"),
            extra={k: v for k, v in data.items() 
                   if k not in ("token_used", "duration_ms", "performance")}
        )


@dataclass
class NodeState:
    """
    节点状态快照
    
    对应 nodes/{node_id}/state.json
    """
    node_id: str
    turn: int
    status: str  # init, running, completed, failed
    agent_info: Dict[str, Any] = field(default_factory=dict)
    task_info: Dict[str, Any] = field(default_factory=dict)
    execution_info: Dict[str, Any] = field(default_factory=dict)
    file_state: Dict[str, Dict] = field(default_factory=dict)  # path -> FileInfo
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_get_timestamp)
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "turn": self.turn,
            "status": self.status,
            "agent_info": self.agent_info,
            "task_info": self.task_info,
            "execution_info": self.execution_info,
            "file_state": self.file_state,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "NodeState":
        return cls(
            node_id=data["node_id"],
            turn=data.get("turn", 0),
            status=data.get("status", "init"),
            agent_info=data.get("agent_info", {}),
            task_info=data.get("task_info", {}),
            execution_info=data.get("execution_info", {}),
            file_state=data.get("file_state", {}),
            metrics=data.get("metrics", {}),
            timestamp=data.get("timestamp", _get_timestamp()),
        )
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "NodeState":
        return cls.from_dict(json.loads(json_str))


@dataclass
class TraceNode:
    """
    Trace 树节点
    
    存储在 trace.json 的 tree 字段中
    """
    node_id: str
    parent_id: Optional[str]
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    action: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=_get_timestamp)
    children: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpointer_thread: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "state_snapshot": self.state_snapshot,
            "action": self.action,
            "result": self.result,
            "timestamp": self.timestamp,
            "children": self.children,
            "metrics": self.metrics,
            "checkpointer_thread": self.checkpointer_thread,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TraceNode":
        return cls(
            node_id=data["node_id"],
            parent_id=data.get("parent_id"),
            state_snapshot=data.get("state_snapshot", {}),
            action=data.get("action"),
            result=data.get("result"),
            timestamp=data.get("timestamp", _get_timestamp()),
            children=data.get("children", []),
            metrics=data.get("metrics", {}),
            checkpointer_thread=data.get("checkpointer_thread"),
        )


@dataclass
class ActionRecord:
    """
    动作记录
    
    单个工具调用的记录
    """
    action_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_get_timestamp)
    duration_ms: Optional[int] = None
    compressed: bool = False
    
    def to_dict(self) -> Dict:
        data = {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "timestamp": self.timestamp,
            "compressed": self.compressed,
        }
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ActionRecord":
        return cls(
            action_id=data["action_id"],
            tool_name=data["tool_name"],
            arguments=data.get("arguments", {}),
            result=data.get("result", {}),
            timestamp=data.get("timestamp", _get_timestamp()),
            duration_ms=data.get("duration_ms"),
            compressed=data.get("compressed", False),
        )


@dataclass
class ActionHistoryFact:
    """
    增量版动作历史
    
    对应 nodes/{node_id}/actions/action_history_fact.json
    每个 node 只保存自己新增的 action（避免重复）
    """
    node_id: str
    parent_node_id: Optional[str]
    turn: int
    actions: List[ActionRecord] = field(default_factory=list)
    actions_count: int = 0
    last_updated: str = field(default_factory=_get_timestamp)
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "turn": self.turn,
            "actions": [a.to_dict() for a in self.actions],
            "actions_count": self.actions_count,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ActionHistoryFact":
        actions = [ActionRecord.from_dict(a) for a in data.get("actions", [])]
        return cls(
            node_id=data["node_id"],
            parent_node_id=data.get("parent_node_id"),
            turn=data.get("turn", 0),
            actions=actions,
            actions_count=len(actions),
            last_updated=data.get("last_updated", _get_timestamp()),
        )
    
    def add_action(self, action: ActionRecord):
        """添加一个动作记录"""
        self.actions.append(action)
        self.actions_count = len(self.actions)
        self.last_updated = _get_timestamp()
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ActionHistoryFact":
        return cls.from_dict(json.loads(json_str))


@dataclass
class ActionHistoryCompressed:
    """
    压缩版动作历史
    
    对应 nodes/{node_id}/actions/action_history_compressed.json
    用于渲染、展示（减少 token 消耗）
    
    source_path 用于校验缓存是否有效（parent 链是否变化）
    """
    node_id: str
    turn: int
    actions: List[ActionRecord] = field(default_factory=list)
    total_actions: int = 0
    compressed_count: int = 0
    source_path: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=_get_timestamp)
    
    def is_valid(self, current_path: List[str]) -> bool:
        """
        校验缓存是否有效
        
        Args:
            current_path: 当前从 root 到此节点的路径
            
        Returns:
            True 如果 source_path 与当前路径匹配
        """
        return self.source_path == current_path
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "turn": self.turn,
            "actions": [a.to_dict() for a in self.actions],
            "total_actions": self.total_actions,
            "compressed_count": self.compressed_count,
            "source_path": self.source_path,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ActionHistoryCompressed":
        actions = [ActionRecord.from_dict(a) for a in data.get("actions", [])]
        return cls(
            node_id=data["node_id"],
            turn=data.get("turn", 0),
            actions=actions,
            total_actions=data.get("total_actions", len(actions)),
            compressed_count=data.get("compressed_count", 0),
            source_path=data.get("source_path", []),
            last_updated=data.get("last_updated", _get_timestamp()),
        )


@dataclass
class DecisionRecord:
    """决策记录"""
    turn: int
    thinking: str
    decision: str
    timestamp: str = field(default_factory=_get_timestamp)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DecisionRecord":
        return cls(**data)


@dataclass
class PlanState:
    """计划状态"""
    goal: str = ""
    steps: List[str] = field(default_factory=list)
    current_step: int = 0
    status: str = "pending"  # pending, in_progress, completed
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PlanState":
        return cls(**data)


@dataclass
class ThinkingState:
    """
    PlanAgent 固化状态
    
    对应 nodes/{node_id}/thinking.json
    """
    node_id: str
    turn: int
    current_plan: Dict[str, Any] = field(default_factory=dict)
    latest_thinking: str = ""
    decision_history: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=_get_timestamp)
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "turn": self.turn,
            "current_plan": self.current_plan,
            "latest_thinking": self.latest_thinking,
            "decision_history": self.decision_history,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ThinkingState":
        return cls(
            node_id=data["node_id"],
            turn=data.get("turn", 0),
            current_plan=data.get("current_plan", {}),
            latest_thinking=data.get("latest_thinking", ""),
            decision_history=data.get("decision_history", []),
            timestamp=data.get("timestamp", _get_timestamp()),
        )
    
    def add_decision(self, thinking: str, decision: str):
        """添加决策记录"""
        record = DecisionRecord(
            turn=self.turn,
            thinking=thinking,
            decision=decision,
        )
        self.decision_history.append(record.to_dict())
        self.latest_thinking = thinking
        self.timestamp = _get_timestamp()
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ThinkingState":
        return cls.from_dict(json.loads(json_str))


@dataclass
class PendingTool:
    """
    待执行工具
    
    用于断点续跑
    """
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, executing, completed, failed
    created_at: str = field(default_factory=_get_timestamp)
    
    def to_dict(self) -> Dict:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PendingTool":
        return cls(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            arguments=data.get("arguments", {}),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", _get_timestamp()),
        )


@dataclass
class PendingToolsState:
    """
    待执行工具状态
    
    对应 nodes/{node_id}/actions/pending_tools.json
    """
    node_id: str
    turn: int
    pending_tools: List[PendingTool] = field(default_factory=list)
    last_updated: str = field(default_factory=_get_timestamp)
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "turn": self.turn,
            "pending_tools": [t.to_dict() for t in self.pending_tools],
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PendingToolsState":
        tools = [PendingTool.from_dict(t) for t in data.get("pending_tools", [])]
        return cls(
            node_id=data["node_id"],
            turn=data.get("turn", 0),
            pending_tools=tools,
            last_updated=data.get("last_updated", _get_timestamp()),
        )
    
    def add_pending_tool(self, tool: PendingTool):
        """添加待执行工具"""
        self.pending_tools.append(tool)
        self.last_updated = _get_timestamp()
    
    def mark_completed(self, tool_call_id: str) -> bool:
        """
        标记工具执行完成
        
        Args:
            tool_call_id: 工具调用 ID
            
        Returns:
            True 如果找到并标记了工具，False 如果未找到
        """
        for tool in self.pending_tools:
            if tool.tool_call_id == tool_call_id:
                tool.status = "completed"
                self.last_updated = _get_timestamp()
                return True
        return False
    
    def get_pending(self) -> List[PendingTool]:
        """获取所有待执行的工具"""
        return [t for t in self.pending_tools if t.status == "pending"]
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PendingToolsState":
        return cls.from_dict(json.loads(json_str))


@dataclass
class TraceTree:
    """
    Trace 树结构
    
    对应 trace.json
    """
    task_id: str
    created_at: str = field(default_factory=_get_timestamp)
    current_node: str = "root"
    tree: Dict[str, TraceNode] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "current_node": self.current_node,
            "tree": {k: v.to_dict() for k, v in self.tree.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TraceTree":
        tree = {}
        for node_id, node_data in data.get("tree", {}).items():
            tree[node_id] = TraceNode.from_dict(node_data)
        return cls(
            task_id=data["task_id"],
            created_at=data.get("created_at", _get_timestamp()),
            current_node=data.get("current_node", "root"),
            tree=tree,
        )
    
    def add_node(self, node: TraceNode):
        """添加节点到树"""
        self.tree[node.node_id] = node
        # 更新父节点的 children
        if node.parent_id and node.parent_id in self.tree:
            if node.node_id not in self.tree[node.parent_id].children:
                self.tree[node.parent_id].children.append(node.node_id)
    
    def get_node(self, node_id: str) -> Optional[TraceNode]:
        """获取节点"""
        return self.tree.get(node_id)
    
    def set_current_node(self, node_id: str):
        """设置当前节点"""
        if node_id not in self.tree:
            raise ValueError(f"Node '{node_id}' not found in tree")
        self.current_node = node_id
    
    def get_path_to_node(self, node_id: str) -> List[str]:
        """
        获取从 root 到指定节点的路径
        
        Args:
            node_id: 目标节点 ID
            
        Returns:
            路径列表 ["root", "node_001", ...]，若节点不存在、路径断裂或存在环则返回 []。
            
        Note:
            只有当路径完整（从目标节点沿 parent 链到 root）且无环时才返回有效路径，
            否则返回空列表以保持一致性并避免返回不完整或错误数据。
        """
        if node_id not in self.tree:
            return []
        
        path = []
        current = node_id
        seen = set()  # 防止 parent 链成环导致死循环
        while current is not None:
            if current in seen:
                return []  # 成环，视为非法数据
            seen.add(current)
            node = self.tree.get(current)
            if node is None:
                return []
            path.append(current)
            current = node.parent_id
        path.reverse()
        return path
    
    def get_all_leaf_nodes(self) -> List[str]:
        """获取所有叶节点"""
        return [
            node_id for node_id, node in self.tree.items()
            if not node.children
        ]
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TraceTree":
        return cls.from_dict(json.loads(json_str))
