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

"""文档修复状态定义

继承 BaseState，添加文档处理专用字段。
"""

import sys
from pathlib import Path
from typing import Optional, List

# 添加 python 目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from akg_agents.core_v2.langgraph_base.base_state import BaseState


class DocFixerState(BaseState, total=False):
    """文档修复工作流状态
    
    继承通用 BaseState，添加文档处理专用字段。
    
    工作流程:
        原始文档 → TypoFixer(修复错别字) → Beautifier(美化格式) → 最终文档
    """
    
    # === 文档内容 ===
    original_content: str           # 原始文档内容
    typo_fixed_content: str         # 错别字修复后的内容
    beautified_content: str         # 美化后的最终内容
    
    # === 处理结果 ===
    typo_corrections: List[dict]    # 错别字修正记录 [{"original": "...", "corrected": "...", "position": ...}]
    beautify_changes: List[str]     # 美化修改说明列表
    
    # === Agent 输出 ===
    typo_fixer_reasoning: str       # TypoFixer 的推理过程
    beautifier_reasoning: str       # Beautifier 的推理过程
    
    # === 配置 ===
    document_type: str              # 文档类型: "markdown", "text", "code"
    language: str                   # 文档语言: "zh", "en"

