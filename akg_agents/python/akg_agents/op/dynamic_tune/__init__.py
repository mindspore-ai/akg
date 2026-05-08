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

"""akg_agents.op.dynamic_tune

Triton on Ascend NPU 上的 dynamic-shape 离线调优框架，遵循"single ModelNew，
forward(*inputs, config=None)"约定，与 akg_agents.KernelVerifier 无缝衔接。

约定（ModelNew）：
    class ModelNew(nn.Module):
        def __init__(self, ...):
            super().__init__()
            self._selector = None

        def forward(self, *inputs, config=None):
            if config is None:
                if self._selector is None:
                    self._selector = load_deployed_selector()
                config = self._selector.select_config(_axis_from_inputs(inputs))
            ...
            return out

    - tune 阶段：`tune_configs` 在每条 (shape, config) 显式 `module(*inputs, config=cfg)`。
    - KernelVerifier / 生产：`module(*inputs)`，触发内部 selector，与基线 Model 接口一致。

Public API：
    Config                   - 单条 kernel 候选配置
    tune_configs             - 显式执行离线调优并写出 manifest
    load_deployed_selector   - 从 manifest 加载 selector（ModelNew.forward 内部用）

"""

from akg_agents.op.dynamic_tune.config import Config
from akg_agents.op.dynamic_tune.deploy.loader import (
    DeployedSelector,
    load_deployed_selector,
)
from akg_agents.op.dynamic_tune.tune.tuner import (
    TuneOutcome,
    tune_configs,
)

__all__ = [
    "Config",
    "DeployedSelector",
    "TuneOutcome",
    "load_deployed_selector",
    "tune_configs",
]
__version__ = "0.3.0"
