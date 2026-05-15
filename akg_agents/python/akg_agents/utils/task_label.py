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

import threading
import uuid

_LABEL_LOCK = threading.Lock()


def _gen_short_uuid() -> str:
    return f"{uuid.uuid4().int % 10000:04d}"


def resolve_task_label(
    *,
    op_name: str | None = None,
    parallel_index: int | None = None,
    uuid_prefix: str | None = None,
) -> str:
    """生成 task label（server 侧来源，client 仅展示）。

    规则：
    - 若提供 op_name：label 为 "{uuid4}-{op_name}-parallelN"
      - uuid4：4 位数字
      - N：并行序号（默认 1）
    - 否则返回空字符串
    """
    oname = str(op_name or "").strip()
    idx = int(parallel_index or 1)
    if idx <= 0:
        idx = 1
    if not oname:
        return ""

    if uuid_prefix:
        short_uuid = str(uuid_prefix).strip()
        if not short_uuid:
            short_uuid = _gen_short_uuid()
        return f"{short_uuid}-{oname}-parallel{idx}"

    with _LABEL_LOCK:
        short_uuid = _gen_short_uuid()
        return f"{short_uuid}-{oname}-parallel{idx}"
