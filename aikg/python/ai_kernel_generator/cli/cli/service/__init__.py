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

"""CLI Service 子包（从 core/services.py 拆分而来）。"""

from .app_services import CLIAppServices
from .config import CLIConfigManager, ResolvedValue
from .jobs import JobInspector
from .kernelbench import KernelBenchStaticChecker
from .normalization import (
    normalize_backend,
    normalize_dsl,
    normalize_framework,
    validate_basic,
    validate_target_config,
)
from .notify import BarkNotifier
from .processes import WorkflowServerProcessManager, WorkerServiceProcessManager
from .workers import WorkerRegistry

__all__ = [
    "ResolvedValue",
    "normalize_backend",
    "normalize_framework",
    "normalize_dsl",
    "validate_basic",
    "validate_target_config",
    "CLIConfigManager",
    "KernelBenchStaticChecker",
    "WorkerRegistry",
    "WorkflowServerProcessManager",
    "WorkerServiceProcessManager",
    "BarkNotifier",
    "JobInspector",
    "CLIAppServices",
]
