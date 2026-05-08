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

"""manifest 持久化与部署侧加载子包。

调优阶段产出：manifest.json（+ 可选 model.pkl）。
部署阶段消费：DeployedSelector 懒加载 manifest，按 shape 走 selector 推理。
"""

from akg_agents.op.dynamic_tune.deploy.loader import (
    DeployedSelector,
    load_deployed_selector,
)
from akg_agents.op.dynamic_tune.deploy.manifest import (
    Manifest,
    dump_manifest,
    load_manifest,
)

__all__ = [
    "Manifest",
    "dump_manifest",
    "load_manifest",
    "DeployedSelector",
    "load_deployed_selector",
]
