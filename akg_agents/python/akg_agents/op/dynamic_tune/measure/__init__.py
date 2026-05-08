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

"""批量测时子包。

核心价值：N×M (shapes × configs) launch 共享一次 NPU profiler 启停。

主要导出：
    BatchProfiler           - 批量测时入口，自带 a 主路径 + b 回退
    CompileGate             - 调优前剔除编译/运行报错的 config
    LatencyMatrix           - (shape × config) → latency 矩阵
    BatchProfilerOOMError   - 触发 b 回退的标志异常
"""

from akg_agents.op.dynamic_tune.measure.batch_profiler import (
    BatchProfiler,
    BatchProfilerOOMError,
    LatencyMatrix,
)
from akg_agents.op.dynamic_tune.measure.compile_gate import (
    CompileGate,
    CompileGateOutcome,
    CompileGateRejection,
)

__all__ = [
    "BatchProfiler",
    "BatchProfilerOOMError",
    "LatencyMatrix",
    "CompileGate",
    "CompileGateOutcome",
    "CompileGateRejection",
]
