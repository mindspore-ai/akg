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

"""
KernelConductor Agent - 基于 Skill 系统的错误分析与修复建议 Agent

负责：
- 分析 Verifier 报错信息，判断错误类型
- 复用 KernelGen 已选择的 skill（fundamental + guide + example + fix），
  作为错误分析的知识背景
- 决策下一步应交给代码生成 Agent 还是终止
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from akg_agents import get_project_root
from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper
from akg_agents.core_v2.skill import SkillLoader
from akg_agents.core_v2.skill.metadata import dsl_to_dir_key

project_root = Path(get_project_root())
SKILLS_DIR = project_root / "op" / "resources" / "skills"

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class KernelConductor(AgentBase):
    """基于 Skill 系统的 Conductor Agent

    优先复用 KernelGen 已选择的 skill（通过 node 层传入 skill_contents），
    这样 conductor 能看到完整的上下文（fundamental + guide + example + fix）。

    如果没有外部传入的 skill_contents（兜底），则自行加载 fundamental 类 skill。
    """

    TOOL_NAME = "call_kernel_conductor"
    DESCRIPTION = "分析 Verifier 错误并基于 Skill 知识提供修复建议"

    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {"type": "string"},
            "task_desc": {"type": "string"},
            "dsl": {"type": "string"},
            "framework": {"type": "string"},
            "backend": {"type": "string"},
            "error_log": {"type": "string"},
            "agent_result": {"type": "string", "default": ""},
            "history_attempts": {
                "type": "array",
                "items": {"type": "object"},
                "default": [],
            },
            "valid_next_agents": {"type": "string", "default": "kernel_gen, finish"},
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend", "error_log"],
    }

    FALLBACK_SKILL_CATEGORIES = {"fundamental", "reference"}

    def __init__(self):
        context = {"agent_name": "kernel_conductor", "task_label": "main"}
        super().__init__(context=context)

        self._loader = SkillLoader()
        self._skills_cache: Dict[str, list] = {}

        self.system_prompt_template = self.load_template("kernel_conductor/system_prompt.j2")
        self.user_prompt_template = self.load_template("kernel_conductor/user_prompt.j2")

    def _load_fundamental_skills_fallback(self, dsl: str) -> str:
        """兜底：自行加载 fundamental/reference skill 并拼接为文本"""
        dsl_key = dsl_to_dir_key(dsl)
        cache_key = f"conductor_{dsl_key}"
        if cache_key in self._skills_cache:
            skills = self._skills_cache[cache_key]
        else:
            dsl_dir = SKILLS_DIR / dsl_key
            if not dsl_dir.exists():
                logger.warning(f"[KernelConductor] Skill directory not found: {dsl_dir}")
                self._skills_cache[cache_key] = []
                return ""
            try:
                all_skills = self._loader.load_from_directory(dsl_dir)
                skills = [
                    s for s in all_skills
                    if (getattr(s, "category", "") or "") in self.FALLBACK_SKILL_CATEGORIES
                ]
                self._skills_cache[cache_key] = skills
                logger.info(
                    f"[KernelConductor] Fallback: loaded {len(skills)} fundamental skills "
                    f"from {dsl_dir} (total scanned: {len(all_skills)})"
                )
            except Exception as e:
                logger.error(f"[KernelConductor] Failed to load skills: {e}")
                self._skills_cache[cache_key] = []
                return ""

        if not skills:
            return ""
        sorted_skills = sorted(skills, key=lambda s: getattr(s, "name", ""))
        return "\n\n---\n\n".join(s.content for s in sorted_skills)

    async def run(
        self,
        op_name: str,
        task_desc: str,
        dsl: str,
        framework: str,
        backend: str,
        error_log: str,
        agent_result: str = "",
        history_attempts: Optional[List[dict]] = None,
        valid_next_agents: str = "kernel_gen, finish",
        model_level: str = "standard",
        skill_contents: str = "",
    ) -> Tuple[str, str, str, str]:
        """执行错误分析并给出修复建议

        Args:
            skill_contents: 由 node 层从 KernelGen 缓存中获取并拼接好的 skill 文本。
                            如果为空，conductor 自行加载 fundamental skill 作为兜底。

        Returns:
            (decision, suggestion, formatted_prompt, reasoning)
        """
        from akg_agents.utils.common_utils import ParserFactory
        from akg_agents.op.utils.result_processor import ResultProcessor

        if history_attempts is None:
            history_attempts = []

        if not skill_contents:
            logger.info("[KernelConductor] 未收到外部 skill_contents，使用 fallback 加载 fundamental")
            skill_contents = self._load_fundamental_skills_fallback(dsl)

        conductor_parser = ParserFactory.get_conductor_parser()
        format_instructions = conductor_parser.get_format_instructions()

        error_for_prompt = error_log
        if error_log and len(error_log) > 4000:
            error_for_prompt = "... (前面省略) ...\n" + error_log[-4000:]

        system_prompt = self.system_prompt_template.format(
            dsl=dsl,
            framework=framework,
            backend=backend,
        )

        user_prompt = self.user_prompt_template.format(
            skill_contents=skill_contents,
            op_name=op_name,
            framework=framework,
            task_desc=task_desc,
            agent_result=agent_result,
            error_log=error_for_prompt,
            history_attempts=history_attempts[-5:],
            valid_next_agents=valid_next_agents,
            format_instructions=format_instructions,
        )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        template = Jinja2TemplateWrapper("{{ prompt }}")

        response_text, formatted_prompt, reasoning = await self.run_llm(
            template, {"prompt": full_prompt}, model_level
        )

        valid_options_set = {a.strip() for a in valid_next_agents.split(",")}
        decision, suggestion = ResultProcessor.parse_conductor_decision(
            response_text, conductor_parser, valid_options_set
        )

        logger.info(
            f"[KernelConductor] decision={decision}, "
            f"suggestion_len={len(suggestion or '')}, "
            f"skill_contents_len={len(skill_contents)}"
        )

        return decision or "kernel_gen", suggestion or "", formatted_prompt, reasoning
