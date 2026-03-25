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
SkillEvolutionAgent - 算子 Skill 自进化 Agent

支持四种模式：
  search_log:     从搜索日志中收集进化链 diff → LLM 生成 SKILL.md
  expert_tuning:  从对话历史中提取人工调优经验 → LLM 生成 SKILL.md
  error_fix:      从错误修复记录中提取调试经验 → LLM 生成 SKILL.md
  organize:       整理 evolved skills（fix=按错误类型拆分，improvement=按主题合并）

注册为 Agent 工具，由 KernelAgent 通过 ToolExecutor 调用。
"""

import logging
import os
import time
from dataclasses import asdict
from typing import Dict, Any, List, Union

from akg_agents.core_v2.agents import register_agent, Jinja2TemplateWrapper
from akg_agents.core_v2.agents.skill_evolution_base import SkillEvolutionBase

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class SkillEvolutionAgent(SkillEvolutionBase):
    """算子 Skill 自进化 Agent"""

    TOOL_NAME = "call_skill_evolution"

    DESCRIPTION = """
将算子优化经验、调试经验、人工调优经验沉淀为可复用的 SKILL.md 文档。

功能：
- 从搜索日志中提取进化链diff，总结能带来性能提升的优化经验（search_log 模式）
- 从对话历史中提取人工调优经验，“用户建议 → 代码变更 → 性能变化”因果链（expert_tuning 模式）
- 从错误修复记录中提取调试经验，“错误类型 → 修复策略”（error_fix 模式）
- 整理 evolved skills：fix 按错误类型拆分，improvement 按主题合并去重（organize 模式）

适用场景：
- 使用过adaptive_search，想要收集能带来性能提升的优化经验
- 用户手动指导优化后，想要固化专家经验
- 使用过adaptive_search、evolve、kernelgen等代码生成工具，想要收集代码生成过程中出现多次失败后成功修复的调试经验
- evolved skills 数量较多且内容重复时，进行合并精简

输出：生成或更新的 SKILL.md 文件路径
"""

    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["search_log", "expert_tuning", "error_fix", "organize"],
                "description": "模式选择：search_log=搜索日志优化经验，expert_tuning=人工调优经验，error_fix=错误修复经验，organize=整理evolved skills",
                "default": "search_log",
            },
            "op_name": {
                "type": "string",
                "description": "算子名称（如 relu、l1norm）",
            },
            "log_dir": {
                "type": "string",
                "description": "日志目录路径。只有 search_log/error_fix 模式需要，通常为 ~/.akg/conversations/cli_xxx/nodes/xxx/logs/，需要保证当前节点的result.json里面'status'字段为'success'",
                "default": "",
            },
            "conversation_dir": {
                "type": "string",
                "description": "对话目录路径。只有 expert_tuning 模式需要，通常为 ~/.akg/conversations/cli_xxx",
                "default": "",
            },
            "task_desc": {
                "type": "string",
                "description": "算子任务描述（可选）",
                "default": "",
            },
            "skills_dir": {
                "type": "string",
                "description": "evolved skill 目录（organize 模式必填），dsl/backend 等信息从 skill metadata 中自动提取",
                "default": "",
            },
            "output_dir": {
                "type": "string",
                "description": "SKILL.md 输出目录（可选，默认按 source 类型写入 evolved_fix/ 或 evolved_improvement/），非特殊需求不需要指定",
                "default": "",
            },
        },
        "required": [],
    }

    def __init__(self):
        context = {"agent_name": "skill_evolution"}
        super().__init__(context=context)
        self.search_log_template = self.load_template(
            "skill_evolution/analyze_search_log.j2"
        )
        self.expert_tuning_template = self.load_template(
            "skill_evolution/analyze_expert_tuning.j2"
        )
        self.error_fix_template = self.load_template(
            "skill_evolution/analyze_error_fix.j2"
        )
        self.dedup_error_fix_template = self.load_template(
            "skill_evolution/dedup_error_fix.j2"
        )
        self.classify_skills_template = self.load_template(
            "skill_evolution/classify_skills.j2"
        )
        self.merge_cluster_template = self.load_template(
            "skill_evolution/merge_cluster.j2"
        )

    async def run(
        self,
        op_name: str = "",
        mode: str = "search_log",
        log_dir: str = "",
        task_desc: str = "",
        output_dir: str = "",
        cur_path: str = "",
        conversation_dir: str = "",
        skills_dir: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        if mode == "search_log":
            return await self._run_search_log(
                log_dir=log_dir,
                op_name=op_name,
                task_desc=task_desc,
                output_dir=output_dir,
                cur_path=cur_path,
            )
        elif mode == "expert_tuning":
            return await self._run_expert_tuning(
                op_name=op_name,
                conversation_dir=conversation_dir,
                output_dir=output_dir,
                cur_path=cur_path,
            )
        elif mode == "error_fix":
            return await self._run_error_fix(
                log_dir=log_dir,
                op_name=op_name,
                output_dir=output_dir,
                cur_path=cur_path,
            )
        elif mode in ("organize", "merge_skills"):
            return await self._run_merge_skills(
                skills_dir=skills_dir,
                output_dir=output_dir,
                cur_path=cur_path,
            )
        else:
            return {
                "status": "error",
                "output": "",
                "error_information": f"不支持的模式: {mode}，可选: search_log, expert_tuning, error_fix, organize",
            }

    # ==================== 公共：解析 + 写入 SKILL.md ====================

    def _parse_and_write_skill(
        self,
        mode: str,
        content: str,
        compressed_data: Union[Any, Dict[str, str]],
        output_dir: str,
        work_dir: str,
        t0: float,
        log_lines: List[str],
        extra_output: str = "",
    ) -> Dict[str, Any]:
        """解析 LLM 输出 → 写入 SKILL.md → 保存 result.json，返回 success / fail dict"""
        from akg_agents.op.tools.skill_evolution.common import (
            parse_skill_output, SkillWriter,
        )

        skill_name, description, body = parse_skill_output(content)
        self._print(
            mode,
            f"LLM 输出: skill_name={skill_name}, {body.count(chr(10))+1} 行",
            log_lines,
        )

        if not body:
            return self._fail_result(mode, "LLM 未生成有效正文", work_dir, log_lines)

        writer = SkillWriter()
        try:
            skill_path = writer.write(
                skill_name, description, body, compressed_data,
                output_dir or None,
            )
        except OSError as e:
            return self._fail_result(mode, f"SKILL.md 写入失败: {e}", work_dir, log_lines)

        elapsed = time.time() - t0
        self._print(mode, f"完成: {skill_path} ({elapsed:.1f}s)", log_lines)

        result_meta: Dict[str, Any] = {
            "status": "success", "skill_path": skill_path,
            "skill_name": skill_name, "elapsed_seconds": round(elapsed, 1),
            "mode": mode,
        }
        self._save_json(work_dir, "result.json", result_meta)
        self._save_session_log(work_dir, log_lines)

        output_parts = [f"Skill 已生成: {skill_path}", f"- 名称: {skill_name}"]
        if extra_output:
            output_parts.append(extra_output)
        output_parts.append(f"- 工作区: {work_dir}")

        return {
            "status": "success",
            "output": "\n".join(output_parts),
            "error_information": "",
            "skill_path": skill_path,
        }

    # ==================== search_log 模式 ====================

    async def _run_search_log(
        self,
        log_dir: str,
        op_name: str,
        task_desc: str = "",
        output_dir: str = "",
        cur_path: str = "",
    ) -> Dict[str, Any]:
        from akg_agents.op.tools.skill_evolution.search_log_utils import (
            collect, compress, to_prompt_vars,
        )

        work_dir = self._init_workspace(cur_path, log_dir, op_name)
        log_lines: List[str] = []

        try:
            if not log_dir:
                return self._fail_result(
                    "search_log", "未提供搜索日志目录 (log_dir)",
                    work_dir, log_lines,
                )
            if not os.path.isdir(log_dir):
                return self._fail_result(
                    "search_log", f"搜索日志目录不存在: {log_dir}",
                    work_dir, log_lines,
                )

            self._print("search_log", f"开始: op_name={op_name}, log_dir={log_dir}", log_lines)
            t0 = time.time()

            records, metadata = collect(log_dir, op_name)
            metadata["op_name"] = op_name

            if not records:
                return self._fail_result("search_log", "未找到任何任务记录", work_dir, log_lines)

            self._print("search_log", f"收集: {len(records)} 条记录", log_lines)
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
            self._print(
                "search_log",
                f"压缩: best={compressed.best_task_id} "
                f"({compressed.best_gen_time}us/{compressed.best_speedup:.2f}x), "
                f"进化链={len(compressed.evolution_chains)} 步",
                log_lines,
            )
            self._save_json(work_dir, "compressed_data.json", {
                "best_task_id": compressed.best_task_id,
                "best_speedup": compressed.best_speedup,
                "best_gen_time": compressed.best_gen_time,
                "has_code": bool(compressed.best_code),
                "evolution_chains": [asdict(s) for s in compressed.evolution_chains],
            })

            if task_desc:
                compressed.task_desc = task_desc
            prompt_vars = to_prompt_vars(compressed)

            content = await self._call_llm_and_save(
                "search_log", self.search_log_template, prompt_vars,
                work_dir, log_lines,
            )

            return self._parse_and_write_skill(
                "search_log", content, compressed, output_dir,
                work_dir, t0, log_lines,
                extra_output=f"- 数据: {len(records)} 任务, 最佳 {compressed.best_speedup:.2f}x",
            )

        except Exception as e:
            return self._error_result("search_log", e, work_dir, log_lines)

    # ==================== expert_tuning 模式 ====================

    async def _run_expert_tuning(
        self,
        op_name: str,
        conversation_dir: str,
        output_dir: str = "",
        cur_path: str = "",
    ) -> Dict[str, Any]:
        from akg_agents.op.tools.skill_evolution.expert_tuning_utils import (
            collect, build_timeline, to_prompt_vars,
        )

        work_dir = self._init_workspace(cur_path, "", op_name)
        log_lines: List[str] = []

        try:
            self._print(
                "expert_tuning",
                f"开始: op_name={op_name}, conversation_dir={conversation_dir}",
                log_lines,
            )
            t0 = time.time()

            if not conversation_dir:
                return self._fail_result(
                    "expert_tuning", "未提供对话目录 (conversation_dir)",
                    work_dir, log_lines,
                )

            sections, metadata = collect(conversation_dir, op_name)
            metadata["op_name"] = op_name
            metadata["source"] = "expert_tuning"

            if not sections:
                return self._fail_result(
                    "expert_tuning", "对话目录中无 action 记录",
                    work_dir, log_lines,
                )

            self._print(
                "expert_tuning",
                f"收集到 {len(sections)} 个 section, "
                f"dsl={metadata.get('dsl')}, arch={metadata.get('arch')}",
                log_lines,
            )

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
            self._print("expert_tuning", f"时间线: {len(timeline)} 字符", log_lines)

            prompt_vars = to_prompt_vars(timeline, metadata)

            content = await self._call_llm_and_save(
                "expert_tuning", self.expert_tuning_template, prompt_vars,
                work_dir, log_lines,
            )

            return self._parse_and_write_skill(
                "expert_tuning", content, metadata, output_dir,
                work_dir, t0, log_lines,
            )

        except Exception as e:
            return self._error_result("expert_tuning", e, work_dir, log_lines)

    # ==================== error_fix 模式 ====================

    async def _run_error_fix(
        self,
        log_dir: str,
        op_name: str,
        output_dir: str = "",
        cur_path: str = "",
    ) -> Dict[str, Any]:
        from akg_agents.op.tools.skill_evolution.error_fix_utils import (
            collect, to_prompt_vars,
        )
        from akg_agents.op.tools.skill_evolution.common import SkillWriter

        work_dir = self._init_workspace(cur_path, log_dir, op_name)
        log_lines: List[str] = []

        try:
            if not log_dir:
                return self._fail_result(
                    "error_fix", "未提供搜索日志目录 (log_dir)",
                    work_dir, log_lines,
                )
            if not os.path.isdir(log_dir):
                return self._fail_result(
                    "error_fix", f"搜索日志目录不存在: {log_dir}",
                    work_dir, log_lines,
                )

            self._print("error_fix", f"开始: op_name={op_name}, log_dir={log_dir}", log_lines)
            t0 = time.time()

            records, metadata = collect(log_dir, op_name)
            metadata["op_name"] = op_name
            metadata["source"] = "error_fix"

            if not records:
                return self._fail_result(
                    "error_fix", "未找到任何成功修复记录", work_dir, log_lines,
                )

            self._print("error_fix", f"收集到 {len(records)} 个成功修复记录", log_lines)
            self._save_json(work_dir, "collected_fix_records.json", {
                "metadata": metadata,
                "records": [
                    {"task_id": r.task_id, "error_step": r.error_step,
                     "diff_lines": r.diff.count("\n")}
                    for r in records
                ],
            })

            prompt_vars = to_prompt_vars(records, metadata)

            body = await self._call_llm_and_save(
                "error_fix", self.error_fix_template, prompt_vars,
                work_dir, log_lines,
            )

            if not body.strip():
                return self._fail_result(
                    "error_fix", "LLM 未生成有效内容", work_dir, log_lines,
                )

            writer = SkillWriter()
            skill_path = writer.get_error_fix_skill_path(
                metadata, output_dir or None,
            )
            existing_body = writer.read_error_fix_body(skill_path)

            if existing_body:
                self._print(
                    "error_fix",
                    f"已有 SKILL.md ({len(existing_body)} 字符)，LLM 去重",
                    log_lines,
                )
                self._save_text(work_dir, "existing_skill_body.md", existing_body)
                dedup_vars = {
                    "existing_body": existing_body,
                    "new_body": body,
                }
                increment = await self._call_llm_and_save(
                    "error_fix_dedup", self.dedup_error_fix_template,
                    dedup_vars, work_dir, log_lines,
                )
                no_new = not increment.strip() or "无新增内容" in increment
                if no_new:
                    elapsed = time.time() - t0
                    self._print("error_fix", f"无新增内容，跳过写入 ({elapsed:.1f}s)", log_lines)
                    result_meta: Dict[str, Any] = {
                        "status": "success", "skill_path": skill_path,
                        "skill_name": SkillWriter._error_fix_skill_name(metadata.get("dsl", "")),
                        "elapsed_seconds": round(elapsed, 1),
                        "mode": "error_fix",
                    }
                    self._save_json(work_dir, "result.json", result_meta)
                    self._save_session_log(work_dir, log_lines)
                    return {
                        "status": "success",
                        "output": f"无新增内容，已有 SKILL.md 未变更: {skill_path}",
                        "error_information": "",
                        "skill_path": skill_path,
                    }
                body = increment

            try:
                skill_path = writer.write_error_fix(
                    body, metadata, output_dir or None,
                )
            except OSError as e:
                return self._fail_result(
                    "error_fix", f"SKILL.md 写入失败: {e}", work_dir, log_lines,
                )

            elapsed = time.time() - t0
            action = "追加" if existing_body else "新建"
            self._print("error_fix", f"完成({action}): {skill_path} ({elapsed:.1f}s)", log_lines)

            result_meta_final: Dict[str, Any] = {
                "status": "success", "skill_path": skill_path,
                "skill_name": SkillWriter._error_fix_skill_name(metadata.get("dsl", "")),
                "elapsed_seconds": round(elapsed, 1),
                "mode": "error_fix",
            }
            self._save_json(work_dir, "result.json", result_meta_final)
            self._save_session_log(work_dir, log_lines)

            return {
                "status": "success",
                "output": "\n".join([
                    f"Skill 已{action}: {skill_path}",
                    f"- 数据: {len(records)} 个成功修复案例",
                    f"- 工作区: {work_dir}",
                ]),
                "error_information": "",
                "skill_path": skill_path,
            }

        except Exception as e:
            return self._error_result("error_fix", e, work_dir, log_lines)

    # ==================== organize / merge_skills 模式 ====================

    async def _run_merge_skills(
        self,
        skills_dir: str = "",
        output_dir: str = "",
        cur_path: str = "",
    ) -> Dict[str, Any]:
        from akg_agents.op.tools.skill_evolution.merge_utils import (
            scan_evolved_skills, build_summaries, parse_classify_output,
            archive_skills, write_merged_skill,
            split_large_cluster,
        )
        from akg_agents.op.tools.skill_evolution.common import parse_skill_output

        if not skills_dir:
            return {
                "status": "error",
                "output": "",
                "error_information": "merge_skills 模式需要提供 skills_dir 参数",
            }

        evolved_dir = skills_dir
        work_dir = self._init_workspace(cur_path, "", "merge_skills")
        log_lines: List[str] = []

        try:
            self._print(
                "merge_skills",
                f"开始: evolved_dir={evolved_dir}",
                log_lines,
            )
            t0 = time.time()

            skills = scan_evolved_skills(evolved_dir)
            if len(skills) < 2:
                return self._fail_result(
                    "merge_skills",
                    f"evolved 目录下只有 {len(skills)} 个 skill，无需合并",
                    work_dir, log_lines,
                )

            self._print("merge_skills", f"扫描到 {len(skills)} 个 skill", log_lines)
            name_to_skill = {s.name: s for s in skills}

            # --- Phase 1: 摘要聚类 ---
            summaries = build_summaries(skills)
            self._save_json(work_dir, "skill_summaries.json", summaries)

            classify_vars = {
                "skill_count": len(summaries),
                "summaries": summaries,
            }
            classify_output = await self._call_llm_and_save(
                "merge_classify", self.classify_skills_template,
                classify_vars, work_dir, log_lines,
            )

            clusters = parse_classify_output(classify_output)
            if not clusters:
                return self._fail_result(
                    "merge_skills", "LLM 聚类输出解析失败", work_dir, log_lines,
                )

            self._save_json(work_dir, "clusters.json", clusters)
            self._print(
                "merge_skills",
                f"聚类结果: {len(clusters)} 个簇 — "
                + ", ".join(f"簇{i}({len(c['skills'])}个)" for i, c in enumerate(clusters)),
                log_lines,
            )

            # --- Phase 2: 逐簇合并 ---
            merged_paths: List[str] = []
            skills_to_archive: List = []

            for ci, cluster in enumerate(clusters):
                reason = cluster.get("reason", "")
                skill_names = cluster["skills"]
                cluster_label = f"簇{ci}"

                valid_names = [n for n in skill_names if n in name_to_skill]
                if not valid_names:
                    self._print("merge_skills", f"{cluster_label} 无有效 skill，跳过", log_lines)
                    continue

                if len(valid_names) == 1:
                    self._print("merge_skills", f"{cluster_label} 只有 1 个 skill，保留原样", log_lines)
                    continue

                cluster_skills = [name_to_skill[n] for n in valid_names]
                batches = split_large_cluster(valid_names)

                dsl = ""
                backend = ""
                for s in cluster_skills:
                    if not dsl:
                        dsl = s.metadata.get("dsl", "")
                    if not backend:
                        backend = s.metadata.get("backend", "")
                    if dsl and backend:
                        break
                dsl_prefix = dsl.replace("_", "-").lower() if dsl else "unknown"

                merged_body = None
                merged_name = ""
                merged_desc = ""
                for batch_idx, batch_names in enumerate(batches):
                    batch_skills = [name_to_skill[n] for n in batch_names]

                    if merged_body is not None:
                        documents = [{"name": "已合并文档", "content": merged_body}]
                        documents.extend(
                            {"name": s.name, "content": s.content}
                            for s in batch_skills
                        )
                    else:
                        documents = [
                            {"name": s.name, "content": s.content}
                            for s in batch_skills
                        ]

                    merge_vars = {
                        "cluster_reason": reason,
                        "dsl_prefix": dsl_prefix,
                        "doc_count": len(documents),
                        "documents": documents,
                    }

                    suffix = f"_batch{batch_idx}" if len(batches) > 1 else ""
                    merge_output = await self._call_llm_and_save(
                        f"merge_cluster{ci}{suffix}", self.merge_cluster_template,
                        merge_vars, work_dir, log_lines,
                    )

                    name, desc, body = parse_skill_output(merge_output)
                    if not body:
                        self._print(
                            "merge_skills",
                            f"{cluster_label} batch {batch_idx} 合并输出为空，跳过",
                            log_lines,
                        )
                        continue

                    merged_body = body
                    if name:
                        merged_name = name
                    if desc:
                        merged_desc = desc

                if not merged_name:
                    merged_name = f"{dsl_prefix}-merged-cluster{ci}"
                    logger.info(f"[merge_skills] {cluster_label}: LLM 未返回 skill_name，使用默认: {merged_name}")

                if merged_body:
                    skill_path = write_merged_skill(
                        name=merged_name,
                        description=merged_desc,
                        body=merged_body,
                        dsl=dsl,
                        backend=backend,
                        evolved_dir=output_dir or evolved_dir,
                    )
                    merged_paths.append(skill_path)
                    skills_to_archive.extend(cluster_skills)
                    self._print(
                        "merge_skills",
                        f"{cluster_label} 合并完成: {skill_path}",
                        log_lines,
                    )

            # --- 归档原始 skill ---
            if skills_to_archive:
                archive_path = archive_skills(skills_to_archive, evolved_dir)
                self._print(
                    "merge_skills",
                    f"已归档 {len(skills_to_archive)} 个原始 skill 至 {archive_path}",
                    log_lines,
                )

            elapsed = time.time() - t0
            self._print("merge_skills", f"完成: {len(merged_paths)} 个合并 skill ({elapsed:.1f}s)", log_lines)

            result_meta: Dict[str, Any] = {
                "status": "success",
                "merged_count": len(merged_paths),
                "merged_paths": merged_paths,
                "archived_count": len(skills_to_archive),
                "elapsed_seconds": round(elapsed, 1),
                "mode": "merge_skills",
            }
            self._save_json(work_dir, "result.json", result_meta)
            self._save_session_log(work_dir, log_lines)

            return {
                "status": "success",
                "output": "\n".join([
                    f"Skill 合并完成: {len(merged_paths)} 个主题",
                    *[f"  - {p}" for p in merged_paths],
                    f"- 归档: {len(skills_to_archive)} 个原始 skill",
                    f"- 工作区: {work_dir}",
                ]),
                "error_information": "",
            }

        except Exception as e:
            return self._error_result("merge_skills", e, work_dir, log_lines)
