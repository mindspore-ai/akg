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
from typing import Optional, Any


class UIIntent:
    """UI -> Presenter 的用户意图事件（类型化）。"""


@dataclass(frozen=True)
class WatchNext(UIIntent):
    pass


@dataclass(frozen=True)
class WatchPrev(UIIntent):
    pass


@dataclass(frozen=True)
class WatchSet(UIIntent):
    task_id: str


@dataclass(frozen=True)
class TraceJump(UIIntent):
    task_id: str
    event_idx: Optional[int] = None


@dataclass(frozen=True)
class LangChanged(UIIntent):
    lang: str


@dataclass(frozen=True)
class AppMounted(UIIntent):
    pass


@dataclass(frozen=True)
class WriteMainContent(UIIntent):
    """写入主内容区域（会同时更新缓存）。"""
    content: Any  # 可以是 Text、str 或其他 MainContent 类型
    task_id: Optional[str] = None
