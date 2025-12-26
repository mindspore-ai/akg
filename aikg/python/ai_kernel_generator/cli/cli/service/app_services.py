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

from __future__ import annotations

import os
from dataclasses import dataclass

from ai_kernel_generator.cli.cli.constants import EnvVar
from .config import CLIConfigManager
from .jobs import JobInspector
from .kernelbench import KernelBenchStaticChecker
from .processes import WorkflowServerProcessManager, WorkerServiceProcessManager
from .workers import WorkerRegistry


@dataclass
class CLIAppServices:
    config: CLIConfigManager
    workers: WorkerRegistry
    server: WorkflowServerProcessManager
    worker_service: WorkerServiceProcessManager
    kernelbench: KernelBenchStaticChecker
    jobs: JobInspector

    def get_server_url(self) -> str | None:
        cfg_server_url = self.config.get("server_url")
        if isinstance(cfg_server_url, str) and cfg_server_url.strip():
            return cfg_server_url.strip()
        env_url = os.environ.get(EnvVar.WORKFLOW_SERVER_URL)
        if isinstance(env_url, str) and env_url.strip():
            return env_url.strip()
        return None
