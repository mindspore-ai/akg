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
SkillEvolutionAgent - Skill 自进化 Agent

支持两种模式：
  Mode A (adaptive_search): 从搜索日志中收集进化链 diff → LLM 生成 SKILL.md
  Mode B (feedback):        从对话历史中提取人工经验 → LLM 生成 SKILL.md

注册为 Agent 工具，由 KernelAgent 通过 ToolExecutor 调用。
"""

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class SkillEvolutionAgent(AgentBase):
    """Skill 自进化 Agent"""

    TOOL_NAME = "call_skill_evolution"

    DESCRIPTION = """从搜索日志或对话历史中总结优化技术，生成可复用的 SKILL.md 文档。

支持两种模式：
- adaptive_search（默认）：从搜索日志中提取进化链 diff，总结代码层面的优化模式
- feedback：从对话历史中提取人工调优经验，总结"用户建议 → 性能提升"的因果链

输出：生成的 SKILL.md 文件路径"""

    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["adaptive_search", "feedback"],
                "description": "skill 进化模式: adaptive_search(搜索日志) 或 feedback(对话反馈)",
                "default": "adaptive_search",
            },
            "log_dir": {
                "type": "string",
                "description": "adaptive_search 的日志目录（mode=adaptive_search 时必填）",
                "default": "",
            },
            "op_name": {
                "type": "string",
                "description": "算子名称（如 relu、l1norm）",
            },
            "task_desc": {
                "type": "string",
                "description": "算子任务描述（可选）",
                "default": "",
            },
            "output_dir": {
                "type": "string",
                "description": "SKILL.md 输出目录（可选）",
                "default": "",
            },
            "conversation_dir": {
                "type": "string",
                "description": "对话目录路径，如 .akg/conversations/cli_xxx（mode=feedback 时必填）",
                "default": "",
            },
        },
        "required": ["op_name"],
    }

    def __init__(self):
        context = {"agent_name": "skill_evolution"}
        super().__init__(context=context)
        self.prompt_template = self.load_template("skill_evolution/analyze.j2")
        self.feedback_prompt_template = self.load_template(
            "skill_evolution/analyze_feedback.j2"
        )

    async def run(
        self,
        op_name: str,
        mode: str = "adaptive_search",
        log_dir: str = "",
        task_desc: str = "",
        output_dir: str = "",
        cur_path: str = "",
        conversation_dir: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        if mode == "feedback":
            return await self._run_feedback(
                op_name=op_name,
                conversation_dir=conversation_dir,
                output_dir=output_dir,
                cur_path=cur_path,
            )
        return await self._run_adaptive_search(
            log_dir=log_dir,
            op_name=op_name,
            task_desc=task_desc,
            output_dir=output_dir,
            cur_path=cur_path,
        )

    # ==================== Mode A: adaptive_search ====================

    async def _run_adaptive_search(
        self,
        log_dir: str,
        op_name: str,
        task_desc: str = "",
        output_dir: str = "",
        cur_path: str = "",
    ) -> Dict[str, Any]:
        from akg_agents.op.tools.skill_evolution.collector import collect
        from akg_agents.op.tools.skill_evolution.compressor import compress
        from akg_agents.op.tools.skill_evolution.analyzer import (
            compressed_to_prompt_vars, parse_skill_output,
        )
        from akg_agents.op.tools.skill_evolution.writer import SkillWriter

        work_dir = self._init_workspace(cur_path, log_dir, op_name)
        log_lines: List[str] = []

        def log(msg: str) -> None:
            logger.info(f"[SkillEvolution:ModeA] {msg}")
            log_lines.append(msg)

        try:
            log(f"开始: op_name={op_name}, log_dir={log_dir}")
            t0 = time.time()

            records, metadata = collect(log_dir, op_name)
            metadata["op_name"] = op_name

            if not records:
                log("未找到任何任务记录")
                self._save_session_log(work_dir, log_lines)
                return {"status": "fail", "output": "",
                        "error_information": "未找到任何任务记录"}

            log(f"收集: {len(records)} 条记录")
            self._save_json(work_dir, "collected_data.json", {
                "metadata": metadata,
                "records": [
                    {"task_id": r.task_id, "parent_id": r.parent_id,
                     "generation": r.generation, "speedup": r.speedup,
                     "gen_time": r.gen_time if r.gen_time < float("inf") else None,
                     "has_code": bool(r.code)}
                    for r in sorted(records, key=lambda r: (r.gen_time, -r.speedup))
                ],
            })

            compressed = compress(records, metadata)

            log(f"压缩: best={compressed.best_task_id} "
                f"({compressed.best_gen_time}us/{compressed.best_speedup:.2f}x), "
                f"进化链={len(compressed.evolution_chains)} 步")

            self._save_json(work_dir, "compressed_data.json", {
                "best_task_id": compressed.best_task_id,
                "best_speedup": compressed.best_speedup,
                "best_gen_time": compressed.best_gen_time,
                "has_code": bool(compressed.best_code),
                "evolution_chains": [asdict(s) for s in compressed.evolution_chains],
            })

            if task_desc:
                compressed.task_desc = task_desc
            prompt_vars = compressed_to_prompt_vars(compressed)
            template = Jinja2TemplateWrapper("{{ prompt }}")
            rendered = self.prompt_template.format(**prompt_vars)

            log(f"LLM prompt: {rendered.count(chr(10))+1} 行, {len(rendered)} 字符")
            self._save_text(work_dir, "llm_prompt.txt", rendered)

            content, _, reasoning = await self.run_llm(
                template, {"prompt": rendered}, "standard"
            )
            self._save_text(work_dir, "llm_response.txt", content or "")
            if reasoning:
                self._save_text(work_dir, "llm_reasoning.txt", reasoning)

            skill_name, description, body = parse_skill_output(content)
            log(f"LLM 输出: skill_name={skill_name}, {body.count(chr(10))+1} 行")

            if not body:
                log("LLM 未生成有效正文")
                self._save_session_log(work_dir, log_lines)
                return {"status": "fail", "output": "",
                        "error_information": "LLM 未生成有效正文"}

            writer = SkillWriter()
            skill_path = writer.write(
                skill_name, description, body, compressed,
                output_dir or None,
            )

            elapsed = time.time() - t0
            log(f"完成: {skill_path} ({elapsed:.1f}s)")

            self._save_json(work_dir, "result.json", {
                "status": "success", "skill_path": skill_path,
                "skill_name": skill_name, "elapsed_seconds": round(elapsed, 1),
            })
            self._save_session_log(work_dir, log_lines)

            return {
                "status": "success",
                "output": (
                    f"Skill 已生成: {skill_path}\n"
                    f"- 名称: {skill_name}\n"
                    f"- 数据: {len(records)} 任务, 最佳 {compressed.best_speedup:.2f}x\n"
                    f"- 工作区: {work_dir}"
                ),
                "error_information": "",
                "skill_path": skill_path,
            }

        except Exception as e:
            logger.error(f"[SkillEvolution:ModeA] 失败: {e}", exc_info=True)
            log(f"失败: {e}")
            self._save_session_log(work_dir, log_lines)
            return {"status": "error", "output": "",
                    "error_information": f"Skill 自进化失败: {e}"}

    # ==================== Mode B: feedback ====================

    async def _run_feedback(
        self,
        op_name: str,
        conversation_dir: str,
        output_dir: str = "",
        cur_path: str = "",
    ) -> Dict[str, Any]:
        from akg_agents.op.tools.skill_evolution.feedback_collector import (
            collect_feedback, build_timeline,
        )
        from akg_agents.op.tools.skill_evolution.analyzer import (
            feedback_to_prompt_vars, parse_skill_output,
        )
        from akg_agents.op.tools.skill_evolution.writer import SkillWriter

        work_dir = self._init_workspace(cur_path, "", op_name)
        log_lines: List[str] = []

        def log(msg: str) -> None:
            logger.info(f"[SkillEvolution:ModeB] {msg}")
            log_lines.append(msg)

        try:
            log(f"开始(feedback): op_name={op_name}, "
                f"conversation_dir={conversation_dir}")
            t0 = time.time()

            if not conversation_dir:
                log("未提供对话目录")
                self._save_session_log(work_dir, log_lines)
                return {"status": "fail", "output": "",
                        "error_information": "未提供对话目录 (conversation_dir)"}

            # 1. 读取所有 action 并格式化为 section 列表
            sections, metadata = collect_feedback(conversation_dir, op_name)
            metadata["op_name"] = op_name
            metadata["source"] = "user_feedback"

            if not sections:
                log("未读取到任何 action 记录")
                self._save_session_log(work_dir, log_lines)
                return {"status": "fail", "output": "",
                        "error_information": "对话目录中无 action 记录"}

            log(f"收集到 {len(sections)} 个 section, "
                f"dsl={metadata.get('dsl')}, arch={metadata.get('arch')}")

            # 2. 增量构建时间线（超阈值时 LLM 压缩）
            async def _llm_compress(prompt: str) -> str:
                compress_tpl = Jinja2TemplateWrapper("{{ prompt }}")
                content, _, _ = await self.run_llm(
                    compress_tpl, {"prompt": prompt}, "standard",
                )
                return content or ""

            timeline = await build_timeline(
                sections, _llm_compress, work_dir=work_dir,
            )
            self._save_text(work_dir, "action_timeline.md", timeline)
            log(f"时间线: {len(timeline)} 字符")

            # 3. 交由 LLM 分析
            prompt_vars = feedback_to_prompt_vars(timeline, metadata)
            template = Jinja2TemplateWrapper("{{ prompt }}")
            rendered = self.feedback_prompt_template.format(**prompt_vars)

            log(f"LLM prompt: {rendered.count(chr(10))+1} 行, {len(rendered)} 字符")
            self._save_text(work_dir, "llm_prompt.txt", rendered)

            content, _, reasoning = await self.run_llm(
                template, {"prompt": rendered}, "standard"
            )
            self._save_text(work_dir, "llm_response.txt", content or "")
            if reasoning:
                self._save_text(work_dir, "llm_reasoning.txt", reasoning)

            skill_name, description, body = parse_skill_output(content)
            log(f"LLM 输出: skill_name={skill_name}, {body.count(chr(10))+1} 行")

            if not body:
                log("LLM 未生成有效正文")
                self._save_session_log(work_dir, log_lines)
                return {"status": "fail", "output": "",
                        "error_information": "LLM 未生成有效正文"}

            # 3. 写入 SKILL.md
            writer = SkillWriter()
            skill_path = writer.write(
                skill_name, description, body, metadata,
                output_dir or None,
            )

            elapsed = time.time() - t0
            log(f"完成: {skill_path} ({elapsed:.1f}s)")

            self._save_json(work_dir, "result.json", {
                "status": "success", "skill_path": skill_path,
                "skill_name": skill_name, "elapsed_seconds": round(elapsed, 1),
                "mode": "feedback",
            })
            self._save_session_log(work_dir, log_lines)

            return {
                "status": "success",
                "output": (
                    f"Skill 已生成（人工经验）: {skill_path}\n"
                    f"- 名称: {skill_name}\n"
                    f"- 工作区: {work_dir}"
                ),
                "error_information": "",
                "skill_path": skill_path,
            }

        except Exception as e:
            logger.error(f"[SkillEvolution:ModeB] 失败: {e}", exc_info=True)
            log(f"失败: {e}")
            self._save_session_log(work_dir, log_lines)
            return {"status": "error", "output": "",
                    "error_information": f"Skill 反馈进化失败: {e}"}

    # ==================== 工具方法 ====================

    @staticmethod
    def _init_workspace(cur_path: str, log_dir: str, op_name: str) -> str:
        if cur_path:
            base = Path(cur_path) / "logs" / "skill_evolution"
        else:
            base = Path(log_dir) / "skill_evolution"
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
