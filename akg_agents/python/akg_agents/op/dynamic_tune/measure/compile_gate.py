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

"""Compile gate：调优开始前，剔除编译/运行报错的 config。

为什么需要它：
    批量测时把 N×M 个 launch 拼成一次 profile，只要其中任何一个 launch 编译或
    运行失败（如存储单元超限、shared memory 超限、register 不够），整批 profile
    都会污染。所以必须**先**对每个 config 单独做一次最小 shape sanity launch，
    把通不过的剔掉，再进入批量测时主流程。

接口契约：
    CompileGate.filter(configs, sanity_shape, launch_factory) -> CompileGateOutcome
        kept_configs    : 通过 sanity 的 config（保持原顺序）
        rejections      : 被剔除的 config + 原因
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from akg_agents.op.dynamic_tune.config import Config

Shape = tuple[int, ...]
LaunchFactory = Callable[[Shape, Config], Callable[[], None]]


@dataclass(frozen=True)
class CompileGateRejection:
    config: Config
    reason: str


@dataclass(frozen=True)
class CompileGateOutcome:
    kept_configs: tuple[Config, ...]
    rejections: tuple[CompileGateRejection, ...]

    def kept_config_ids(self) -> tuple[str, ...]:
        return tuple(config.config_id for config in self.kept_configs)


class CompileGate:
    """Sanity launch 守门员。

    `synchronize`：每次 sanity launch 后立即调用，确保异步 kernel 异常被
    归因到**当前**这个 config，而不是滞后污染下一个 launch。NPU/CUDA 上
    都强烈建议传，否则一个有问题的 config 会让整批 sanity 都失真。
    """

    def __init__(
        self,
        *,
        on_reject: Callable[[CompileGateRejection], None] | None = None,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        self._on_reject = on_reject
        self._synchronize = synchronize

    def filter(
        self,
        *,
        configs: Sequence[Config],
        sanity_shape: Shape,
        launch_factory: LaunchFactory,
    ) -> CompileGateOutcome:
        if not configs:
            raise ValueError("compile gate 收到空 configs")
        normalized_shape = tuple(int(value) for value in sanity_shape)
        kept: list[Config] = []
        rejections: list[CompileGateRejection] = []
        for config in configs:
            verdict = self._sanity_one(normalized_shape, config, launch_factory)
            if verdict is None:
                kept.append(config)
            else:
                rejection = CompileGateRejection(config=config, reason=verdict)
                rejections.append(rejection)
                if self._on_reject is not None:
                    self._on_reject(rejection)
        return CompileGateOutcome(
            kept_configs=tuple(kept), rejections=tuple(rejections)
        )

    def _sanity_one(
        self,
        sanity_shape: Shape,
        config: Config,
        launch_factory: LaunchFactory,
    ) -> str | None:
        try:
            launch = launch_factory(sanity_shape, config)
            launch()
            if self._synchronize is not None:
                self._synchronize()
        except BaseException as exc:  # noqa: BLE001
            return _format_reason(exc)
        return None


def _format_reason(exc: BaseException) -> str:
    """把异常压成单行原因字符串，方便落 manifest 与日志。"""

    head = f"{type(exc).__name__}"
    message = str(exc).strip()
    if not message:
        return head
    flat = message.replace("\n", " | ")
    if len(flat) > 240:
        flat = flat[:237] + "..."
    return f"{head}: {flat}"


__all__ = [
    "CompileGate",
    "CompileGateOutcome",
    "CompileGateRejection",
]
