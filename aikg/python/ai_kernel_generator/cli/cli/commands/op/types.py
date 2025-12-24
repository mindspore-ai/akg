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

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpCommandArgs:
    intent: Optional[str]
    framework: str
    backend: str
    arch: str
    dsl: str
    notify: bool
    bark_key: str
    auto_yes: bool

    server_url: Optional[str]
    worker_url: Optional[str]
    devices: Optional[str]
    stream: bool


@dataclass(frozen=True)
class ResolvedRuntimeOptions:
    stream: bool
    notify: bool
    bark_key: str

    server_url: str | None


@dataclass(frozen=True)
class ResolvedTargetConfig:
    framework: str
    backend: str
    arch: str
    dsl: str
