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
SkillEvolutionBase - Skill 自进化基类

提供 Skill Evolution Agent 的通用能力：
- 工作区初始化与文件持久化
- 日志打印
- LLM 调用 → 保存的通用骨架
- 统一的 fail / error 返回格式

具体的数据收集、压缩、prompt 模板由子类（如算子层的 SkillEvolutionAgent）实现。
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from akg_agents.core_v2.agents.base import AgentBase, Jinja2TemplateWrapper

logger = logging.getLogger(__name__)


class SkillEvolutionBase(AgentBase):
    """Skill Evolution 基类"""

    # ==================== 日志 ====================

    @staticmethod
    def _print(mode: str, msg: str, log_lines: List[str]) -> None:
        """统一日志打印：同时写入 logger 和 log_lines 列表"""
        logger.info(f"[SkillEvolution:{mode}] {msg}")
        log_lines.append(msg)

    # ==================== 工作区与文件 ====================

    @staticmethod
    def _init_workspace(cur_path: str, fallback_dir: str, name: str) -> str:
        if cur_path:
            base = Path(cur_path) / "logs" / "skill_evolution"
        elif fallback_dir:
            base = Path(fallback_dir) / "skill_evolution"
        else:
            base = Path(".") / "skill_evolution" / name
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    @staticmethod
    def _save_text(work_dir: str, filename: str, content: str) -> None:
        try:
            with open(os.path.join(work_dir, filename), "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"[SkillEvolution] 保存 {filename} 失败: {e}")

    @staticmethod
    def _save_json(work_dir: str, filename: str, data: Any) -> None:
        try:
            with open(os.path.join(work_dir, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[SkillEvolution] 保存 {filename} 失败: {e}")

    @staticmethod
    def _save_session_log(work_dir: str, lines: List[str]) -> None:
        try:
            path = os.path.join(work_dir, "session.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"=== Skill Evolution ({datetime.now().isoformat()}) ===\n\n")
                for line in lines:
                    f.write(line + "\n")
        except Exception as e:
            logger.warning(f"[SkillEvolution] 保存日志失败: {e}")

    # ==================== LLM 调用 ====================

    async def _call_llm_and_save(
        self,
        mode: str,
        template: Jinja2TemplateWrapper,
        prompt_vars: Dict[str, Any],
        work_dir: str,
        log_lines: List[str],
    ) -> str:
        """渲染 prompt → 调用 LLM → 保存 prompt / response / reasoning，返回 LLM 输出"""
        rendered = template.format(**prompt_vars)
        self._print(
            mode,
            f"LLM prompt: {rendered.count(chr(10))+1} 行, {len(rendered)} 字符",
            log_lines,
        )
        self._save_text(work_dir, "llm_prompt.txt", rendered)

        passthrough = Jinja2TemplateWrapper("{{ prompt }}")
        content, _, reasoning = await self.run_llm(
            passthrough, {"prompt": rendered}, "standard",
        )
        self._save_text(work_dir, "llm_response.txt", content or "")
        if reasoning:
            self._save_text(work_dir, "llm_reasoning.txt", reasoning)

        return content

    # ==================== 统一返回 ====================

    def _fail_result(
        self, mode: str, msg: str, work_dir: str, log_lines: List[str],
    ) -> Dict[str, Any]:
        """记录失败日志并返回 fail dict"""
        self._print(mode, msg, log_lines)
        self._save_session_log(work_dir, log_lines)
        return {"status": "fail", "output": "", "error_information": msg}

    def _error_result(
        self, mode: str, error: Exception, work_dir: str, log_lines: List[str],
    ) -> Dict[str, Any]:
        """记录异常日志并返回 error dict"""
        logger.error(f"[SkillEvolution:{mode}] 失败: {error}", exc_info=True)
        self._print(mode, f"失败: {error}", log_lines)
        self._save_session_log(work_dir, log_lines)
        return {
            "status": "error", "output": "",
            "error_information": f"Skill 自进化失败: {error}",
        }
