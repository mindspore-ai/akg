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


def parse_devices(devices_str: str) -> list[int]:
    devices_str = (devices_str or "").strip()
    if not devices_str:
        raise ValueError("--devices 不能为空（例如 0 或 0,1,2,3）")

    try:
        device_ids = [int(x.strip()) for x in devices_str.split(",") if x.strip()]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--devices 格式非法: {devices_str}（期望逗号分隔整数）"
        ) from exc

    if not device_ids:
        raise ValueError(f"--devices 解析结果为空: {devices_str}")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"--devices 不允许重复: {devices_str}")
    if any(d < 0 for d in device_ids):
        raise ValueError(f"--devices 不能包含负数: {devices_str}")

    return device_ids
