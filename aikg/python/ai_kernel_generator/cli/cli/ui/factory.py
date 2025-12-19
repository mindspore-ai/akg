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

from .protocols import LayoutManager


def create_default_layout_manager() -> LayoutManager:
    # 延迟导入：避免 presenter/core 在 import 阶段把 Textual 相关依赖都拉起来。
    from ai_kernel_generator.cli.cli.ui.tui.manager import TextualLayoutManager

    return TextualLayoutManager()
