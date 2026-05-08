# Copyright 2026 Huawei Technologies Co., Ltd
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

"""shape→config 选择器子包。

设计成可扩展：通过 selector/registry.py 注册新算法实现，扩展 kmeans /
predictive_model 等只需新增一个文件、注册名字即可。

MVP 默认实现 tree（决策树贪心切分 shape 空间），纯 numpy 实现。
"""

from akg_agents.op.dynamic_tune.selector.base import (
    SelectorArtifact,
    SelectorBase,
    SelectorTrainingInputs,
)
from akg_agents.op.dynamic_tune.selector.registry import (
    list_selectors,
    register_selector,
    resolve_selector,
)

# 触发内建算法注册。
from akg_agents.op.dynamic_tune.selector import tree  # noqa: F401

__all__ = [
    "SelectorBase",
    "SelectorArtifact",
    "SelectorTrainingInputs",
    "register_selector",
    "resolve_selector",
    "list_selectors",
]
