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

"""Empty-code check for CodeChecker."""

import logging
from typing import Dict, List

from akg_agents.op.utils.code_checker.base import BlockingCodeChecker

logger = logging.getLogger(__name__)


class EmptyCodeChecker(BlockingCodeChecker):
    """Detect missing or whitespace-only generated code."""

    name = "empty_code"

    def check(self, code: str) -> List[Dict]:
        if code and code.strip():
            return []

        logger.warning("CodeChecker: Empty code provided")
        return [
            {
                "line": 0,
                "error_type": "empty_code",
                "detail": "代码为空，无法进行检查",
                "suggestion": "请生成有效的代码",
                "code_snippet": "",
            }
        ]
