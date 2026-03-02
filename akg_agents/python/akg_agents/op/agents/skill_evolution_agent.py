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

从 adaptive_search 日志中收集数据 → 单调栈进化链 diff → LLM 生成 SKILL.md。
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

    DESCRIPTION = """从 adaptive_search 的搜索日志中总结优化技术，生成可复用的 SKILL.md 文档。

功能：
- 从 verification_results / speed_up_record / lineage_graph 收集数据
- 对每条进化路径维护单调栈，提取性能递增的关键代码 diff
- 通过 LLM 分析进化链，提炼可复用的优化技术
- 生成标准 SKILL.md 文件

输出：生成的 SKILL.md 文件路径"""

    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "log_dir": {
                "type": "string",
                "description": "adaptive_search 的日志目录（节点 logs 路径）",
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
        },
        "required": ["log_dir", "op_name"],
    }

    def __init__(self):
        context = {"agent_name": "skill_evolution"}
        super().__init__(context=context)
        self.prompt_template = self.load_template("skill_evolution/analyze.j2")

    async def run(
        self,
        log_dir: str,
        op_name: str,
        task_desc: str = "",
        output_dir: str = "",
        cur_path: str = "",
        **kwargs,
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
            logger.info(f"[SkillEvolution] {msg}")
            log_lines.append(msg)

        try:
            log(f"开始: op_name={op_name}, log_dir={log_dir}")
            t0 = time.time()

            # 1. 收集
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

            # 2. 压缩
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

            # 3. LLM 生成
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

            # 4. 写入 SKILL.md
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
            logger.error(f"[SkillEvolution] 失败: {e}", exc_info=True)
            log(f"失败: {e}")
            self._save_session_log(work_dir, log_lines)
            return {"status": "error", "output": "",
                    "error_information": f"Skill 自进化失败: {e}"}

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
