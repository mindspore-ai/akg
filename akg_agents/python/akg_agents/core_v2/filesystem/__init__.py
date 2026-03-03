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
AIKG FileSystem 状态管理与 Trace 系统

基于文件系统的状态持久化方案，支持:
- 树状 Trace 管理
- 节点状态快照
- 增量动作历史保存
- 断点续跑
"""

from .models import (
    NodeState,
    TraceNode,
    ActionRecord,
    ThinkingState,
    PendingTool,
    PendingToolsState,
    ActionHistoryFact,
    ActionHistoryCompressed,
    TraceTree,
)
from .exceptions import (
    FileSystemStateError,
    NodeNotFoundError,
    TraceSystemError,
    InvalidNodeStateError,
    TraceNotInitializedError,
    TraceAlreadyExistsError,
    SessionResumeError,
)
from .state import FileSystemState
from .trace_system import TraceSystem
from .compressor import ActionCompressor

__all__ = [
    # Models
    "NodeState",
    "TraceNode",
    "ActionRecord",
    "ThinkingState",
    "PendingTool",
    "PendingToolsState",
    "ActionHistoryFact",
    "ActionHistoryCompressed",
    "TraceTree",
    # Exceptions
    "FileSystemStateError",
    "NodeNotFoundError",
    "TraceSystemError",
    "InvalidNodeStateError",
    "TraceNotInitializedError",
    "TraceAlreadyExistsError",
    "SessionResumeError",
    # Core classes
    "FileSystemState",
    "TraceSystem",
    "ActionCompressor",
]
