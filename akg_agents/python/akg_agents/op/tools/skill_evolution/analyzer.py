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

"""Skill 自进化系统 - LLM prompt 变量 + 输出解析"""

import re
from dataclasses import asdict
from typing import Dict, Any, Tuple

from .models import CompressedData


def compressed_to_prompt_vars(compressed: CompressedData) -> Dict[str, Any]:
    """将 CompressedData 转换为 Jinja2 模板变量"""
    return {
        "op_name": compressed.op_name,
        "dsl": compressed.dsl,
        "backend": compressed.backend,
        "arch": compressed.arch,
        "total_tasks": compressed.total_tasks,
        "success_count": compressed.success_count,
        "best_speedup": compressed.best_speedup,
        "best_task_id": compressed.best_task_id,
        "best_gen_time": compressed.best_gen_time,
        "best_code": compressed.best_code,
        "evolution_chains": [asdict(s) for s in compressed.evolution_chains],
        "performance_summary": compressed.performance_summary,
    }


def parse_skill_output(llm_output: str) -> Tuple[str, str, str]:
    """从 LLM 输出中提取 skill_name、description 和正文

    Returns:
        (skill_name, description, markdown_body)
    """
    if not llm_output:
        return "", "", ""

    text = llm_output.strip()
    skill_name = ""
    description = ""

    lines = text.splitlines()
    consumed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            consumed += 1
            continue
        m_name = re.match(r"skill_name:\s*(.+)", stripped)
        if m_name:
            skill_name = m_name.group(1).strip()
            consumed += 1
            continue
        m_desc = re.match(r"description:\s*(.+)", stripped)
        if m_desc:
            description = m_desc.group(1).strip()
            consumed += 1
            continue
        break

    text = "\n".join(lines[consumed:]).strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    return skill_name, description, text.strip()
