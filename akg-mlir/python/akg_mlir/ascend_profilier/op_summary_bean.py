# Copyright 2023-2026 Huawei Technologies Co., Ltd
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
"""Op summary bean module."""
from .op_summary_headers import OpSummaryHeaders

class OpSummaryBean:
    """Op summary bean."""
    headers = []

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        """Get profiling data as a row of values."""
        row = []
        read_headers = OpSummaryBean.headers if OpSummaryBean.headers else self._data.keys()
        for field_name in read_headers:
            row.append(self._data.get(field_name, ""))
        return row

    @property
    def ts(self) -> float:
        """Get the start timestamp of the op."""
        return float(self._data.get(OpSummaryHeaders.TASK_START_TIME, 0))

    @property
    def all_headers(self) -> list:
        """get all headers."""
        return list(self._data.keys())
