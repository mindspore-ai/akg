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
AIKG FileSystem 自定义异常类
"""


class FileSystemStateError(Exception):
    """FileSystemState 基础异常"""
    pass


class NodeNotFoundError(FileSystemStateError):
    """节点不存在异常"""
    
    def __init__(self, node_id: str, message: str = None):
        self.node_id = node_id
        if message is None:
            message = f"Node '{node_id}' not found"
        super().__init__(message)


class InvalidNodeStateError(FileSystemStateError):
    """无效的节点状态异常"""
    
    def __init__(self, node_id: str, reason: str):
        self.node_id = node_id
        self.reason = reason
        super().__init__(f"Invalid state for node '{node_id}': {reason}")


class TraceSystemError(Exception):
    """TraceSystem 基础异常"""
    pass


class TraceNotInitializedError(TraceSystemError):
    """Trace 系统未初始化异常"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Trace system for task '{task_id}' is not initialized")


class TraceAlreadyExistsError(TraceSystemError):
    """Trace 已存在异常"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Trace for task '{task_id}' already exists")
