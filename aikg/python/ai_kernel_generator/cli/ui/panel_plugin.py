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

"""Panel plugin interface for AKG CLI"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class PanelPlugin(ABC):
    """Abstract base class for panel plugins"""

    @abstractmethod
    def get_name(self) -> str:
        """返回插件名称"""
        pass

    @abstractmethod
    def render_fragments(self, width: int) -> List[Tuple[str, str]]:
        """
        渲染面板内容，返回 (style_class, text) 元组列表
        用于 prompt_toolkit 的 FormattedTextControl
        """
        pass

    def on_data_update(self, data: dict) -> None:
        """接收数据更新（可选实现）"""
        pass

    def get_height(self) -> int:
        """返回面板高度（可选，默认返回固定高度）"""
        return 10  # 默认高度
