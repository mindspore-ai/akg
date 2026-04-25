#!/usr/bin/env python3
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
run_skill_evolution.py - Skill Evolution 独立 CLI 工具（支持四模式）

此工具用于从 adaptive_search 搜索日志、akg_cli 对话历史或错误修复记录中提取优化/调试经验，
生成可复用的 SKILL.md 文档。还支持将已有的 evolved skills 按主题合并去重。

前置条件：
  - search_log 模式需要先运行 `akg_cli op` 执行 adaptive_search，产生搜索日志，对应日志保存在 ~/.akg/conversations/cli_<session_id>/nodes/<node_id>/logs/
  - expert_tuning 模式需要先运行 `akg_cli op` 进行交互式调优，产生对话记录，对应对话记录保存在 ~/.akg/conversations/cli_<session_id>/
  - error_fix 模式需要搜索日志中包含失败→成功的修复记录，对应日志保存在 ~/.akg/conversations/cli_<session_id>/nodes/<node_id>/logs/
  - organize 模式需要 ~/.akg/evolved_skills/{dsl}/evolved-fix/ 和 ~/.akg/evolved_skills/{dsl}/evolved-improvement/ 下有 SKILL.md

search_log 模式:
    python examples/kernel_related/skill_evolution/run_skill_evolution.py search_log <log_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

expert_tuning 模式:
    python examples/kernel_related/skill_evolution/run_skill_evolution.py expert_tuning <conversation_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

error_fix 模式:
    python examples/kernel_related/skill_evolution/run_skill_evolution.py error_fix <log_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

organize 模式（整理：fix=split, improvement=merge）:
    python examples/kernel_related/skill_evolution/run_skill_evolution.py organize <dsl> [--skills-dir DIR] [--output-dir DIR] [--model-level LEVEL]

兼容旧用法（默认 search_log）:
    python examples/kernel_related/skill_evolution/run_skill_evolution.py <log_dir> <op_name>

示例:
    python examples/kernel_related/skill_evolution/run_skill_evolution.py search_log /path/to/node_004/logs relu
    python examples/kernel_related/skill_evolution/run_skill_evolution.py expert_tuning ~/.akg/conversations/cli_xxx rope -o ./output
    python examples/kernel_related/skill_evolution/run_skill_evolution.py error_fix /path/to/node_005/logs matmul -o ./output
    python examples/kernel_related/skill_evolution/run_skill_evolution.py organize triton_ascend
    python examples/kernel_related/skill_evolution/run_skill_evolution.py organize triton_ascend -o ./organized_output
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_USAGE_HINT = """
╔══════════════════════════════════════════════════════════════════════╗
║  Skill Evolution 需要 akg_cli 产生的日志数据作为输入               ║
║                                                                      ║
║  search_log 模式:                                                    ║
║    请先执行 `akg_cli op` 并使用 adaptive_search workflow，           ║
║    搜索日志保存在 nodes/<node_id>/logs/ 目录下                       ║
║                                                                      ║
║  expert_tuning 模式:                                                 ║
║    请先执行 `akg_cli op` 进行交互式调优，                            ║
║    对话记录保存在 ~/.akg/conversations/cli_<session_id>/ 目录下      ║
║                                                                      ║
║  error_fix 模式:                                                     ║
║    与 search_log 共用日志目录，提取失败→成功的错误修复经验           ║
║                                                                      ║
║  organize 模式:                                                      ║
║    整理 evolved skills: fix=按错误类型拆分, improvement=按主题合并   ║
║    参数: organize <dsl> [--skills-dir DIR]                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _load_template(name: str):
    """加载指定的 prompt 模板"""
    from akg_agents.utils.common_utils import get_prompt_path
    template_path = Path(get_prompt_path()) / "skill_evolution" / name
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    from akg_agents.core_v2.agents import Jinja2TemplateWrapper
    return Jinja2TemplateWrapper(template_path.read_text(encoding="utf-8"))


async def _call_llm(prompt: str, model_level: str) -> str:
    """调用 LLM 并返回生成内容"""
    from akg_agents.core_v2.llm.factory import create_llm_client

    client = create_llm_client(model_level=model_level)
    messages = [
        {"role": "system", "content": "你是AKG Agent，你是一个AI助手，你可以帮助用户完成任务。"},
        {"role": "user", "content": prompt},
    ]
    result = await client.generate(messages, stream=False, agent_name="skill_evolution_cli")

    content = result.get("content", "")
    reasoning = result.get("reasoning_content", "")
    if "</think>" in content:
        pos = content.find("</think>")
        content = content[pos + len("</think>"):].lstrip()
    return content or reasoning


async def run_search_log(log_dir: str, op_name: str, output_dir: str, model_level: str, task_desc: str = ""):
    """search_log 模式: 从搜索日志生成 SKILL.md"""
    from akg_agents.op.tools.skill_evolution.search_log_utils import (
        collect, compress, to_prompt_vars,
    )
    from akg_agents.op.tools.skill_evolution.common import (
        parse_skill_output, SkillWriter,
    )

    t0 = time.time()

    work_dir = output_dir or _default_work_dir("search_log", op_name)
    os.makedirs(work_dir, exist_ok=True)

    if not os.path.isdir(log_dir):
        logger.error(f"log_dir 不存在: {log_dir}")
        logger.error("提示: log_dir 应指向 adaptive_search 产生的节点 logs 目录")
        logger.error("      例如: ~/.akg/conversations/cli_xxx/nodes/node_004/logs")
        sys.exit(1)

    logger.info(f"[1/4] 收集数据: log_dir={log_dir}, op_name={op_name}")
    records, metadata = collect(log_dir, op_name)
    metadata["op_name"] = op_name
    if not records:
        logger.error("未找到任何任务记录")
        logger.error("提示: log_dir 应包含 verification_results.jsonl、speed_up_record.txt、lineage_graph.md")
        sys.exit(1)
    logger.info(f"  → {len(records)} 条记录")

    logger.info("[2/4] 压缩进化链")
    compressed = compress(records, metadata)
    if task_desc:
        compressed.task_desc = task_desc
    logger.info(
        f"  → best={compressed.best_task_id} "
        f"({compressed.best_gen_time}us / {compressed.best_speedup:.2f}x), "
        f"进化链={len(compressed.evolution_chains)} 步"
    )

    logger.info(f"[3/4] 调用 LLM (model_level={model_level})")
    template = _load_template("analyze_search_log.j2")
    prompt_vars = to_prompt_vars(compressed)
    rendered = template.format(**prompt_vars)
    _save_file(work_dir, "llm_prompt.txt", rendered)
    logger.info(f"  → prompt: {len(rendered)} 字符")
    logger.info(f"  → 已保存: {work_dir}/llm_prompt.txt")

    content = await _call_llm(rendered, model_level)
    if not content:
        logger.error("LLM 返回为空")
        sys.exit(1)
    _save_file(work_dir, "llm_response.txt", content)
    logger.info(f"  → 生成: {len(content)} 字符")
    logger.info(f"  → 已保存: {work_dir}/llm_response.txt")

    skill_name, description, body = parse_skill_output(content)
    if not body:
        logger.error("LLM 未生成有效正文，原始输出已打印到 stderr")
        print(content, file=sys.stderr)
        sys.exit(1)

    logger.info("[4/4] 写入 SKILL.md")
    writer = SkillWriter()
    skill_path = writer.write(
        skill_name, description, body, compressed,
        output_dir or None,
    )

    elapsed = time.time() - t0
    logger.info(f"完成 ({elapsed:.1f}s): {skill_path}")
    print(f"\n{'='*60}")
    print(f"SKILL.md 已生成: {skill_path}")
    print(f"  skill_name : {skill_name}")
    print(f"  description: {description}")
    print(f"  数据       : {len(records)} 任务, 最佳 {compressed.best_speedup:.2f}x")
    print(f"  耗时       : {elapsed:.1f}s")
    print(f"  中间文件   : {work_dir}/")
    print(f"{'='*60}")


async def run_expert_tuning(conversation_dir: str, op_name: str, output_dir: str, model_level: str):
    """expert_tuning 模式: 从对话目录提取专家调优经验，生成 SKILL.md"""
    from akg_agents.op.tools.skill_evolution.expert_tuning_utils import (
        collect, build_timeline, to_prompt_vars,
    )
    from akg_agents.op.tools.skill_evolution.common import (
        parse_skill_output, SkillWriter,
    )

    t0 = time.time()

    work_dir = output_dir or _default_work_dir("expert_tuning", op_name)
    os.makedirs(work_dir, exist_ok=True)

    if not os.path.isdir(conversation_dir):
        logger.error(f"conversation_dir 不存在: {conversation_dir}")
        logger.error("提示: conversation_dir 应指向 akg_cli 的会话目录")
        logger.error("      例如: ~/.akg/conversations/cli_<session_id>")
        sys.exit(1)

    logger.info(f"[1/4] 读取所有 action: {conversation_dir}")
    sections, metadata = collect(conversation_dir, op_name)
    metadata["op_name"] = op_name
    metadata["source"] = "expert_tuning"

    if not sections:
        logger.error("未读取到任何 action 记录")
        logger.error("提示: conversation_dir 应包含 nodes/ 子目录和 action_history_fact.json 文件")
        sys.exit(1)

    full_text = "\n".join(sections)
    _save_file(work_dir, "action_sections_full.md", full_text)
    logger.info(f"  → {len(sections)} 个 section, {len(full_text)} 字符")
    logger.info(f"  → 已保存: {work_dir}/action_sections_full.md")

    logger.info("[2/4] 增量构建时间线（按需 LLM 压缩）")

    async def _llm_compress(prompt: str) -> str:
        return await _call_llm(prompt, model_level)

    timeline = await build_timeline(sections, _llm_compress, work_dir=work_dir)
    _save_file(work_dir, "action_timeline.md", timeline)
    logger.info(f"  → 最终时间线: {len(timeline)} 字符")
    logger.info(f"  → 已保存: {work_dir}/action_timeline.md")

    logger.info(f"[3/4] 调用 LLM 分析 (model_level={model_level})")
    template = _load_template("analyze_expert_tuning.j2")
    prompt_vars = to_prompt_vars(timeline, metadata)
    rendered = template.format(**prompt_vars)
    _save_file(work_dir, "llm_prompt.txt", rendered)
    logger.info(f"  → prompt: {len(rendered)} 字符")
    logger.info(f"  → 已保存: {work_dir}/llm_prompt.txt")

    content = await _call_llm(rendered, model_level)
    if not content:
        logger.error("LLM 返回为空")
        sys.exit(1)
    _save_file(work_dir, "llm_response.txt", content)
    logger.info(f"  → 生成: {len(content)} 字符")
    logger.info(f"  → 已保存: {work_dir}/llm_response.txt")

    skill_name, description, body = parse_skill_output(content)
    if not body:
        logger.error("LLM 未生成有效正文，原始输出已打印到 stderr")
        print(content, file=sys.stderr)
        sys.exit(1)

    logger.info("[4/4] 写入 SKILL.md")
    writer = SkillWriter()
    skill_path = writer.write(
        skill_name, description, body, metadata,
        output_dir or None,
    )

    elapsed = time.time() - t0
    logger.info(f"完成 ({elapsed:.1f}s): {skill_path}")
    print(f"\n{'='*60}")
    print(f"SKILL.md 已生成（专家调优经验）: {skill_path}")
    print(f"  skill_name : {skill_name}")
    print(f"  description: {description}")
    print(f"  时间线     : {len(timeline)} 字符")
    print(f"  耗时       : {elapsed:.1f}s")
    print(f"  中间文件   : {work_dir}/")
    print(f"{'='*60}")


async def run_error_fix(log_dir: str, op_name: str, output_dir: str, model_level: str):
    """error_fix 模式: 从错误修复记录生成 SKILL.md（已有时 LLM 去重合并）"""
    from akg_agents.op.tools.skill_evolution.error_fix_utils import (
        collect, to_prompt_vars,
    )
    from akg_agents.op.tools.skill_evolution.common import SkillWriter
    import json

    t0 = time.time()

    work_dir = output_dir or _default_work_dir("error_fix", op_name)
    os.makedirs(work_dir, exist_ok=True)

    if not os.path.isdir(log_dir):
        logger.error(f"log_dir 不存在: {log_dir}")
        logger.error("提示: log_dir 应指向节点的 logs 目录")
        logger.error("      例如: ~/.akg/conversations/cli_xxx/nodes/node_005/logs")
        sys.exit(1)

    logger.info(f"[1/4] 收集成功修复记录: log_dir={log_dir}, op_name={op_name}")
    records, metadata = collect(log_dir, op_name)
    metadata["op_name"] = op_name
    metadata["source"] = "error_fix"
    if not records:
        logger.error("未找到任何成功修复记录")
        logger.error("提示: 需要 log_dir 中有 verification_results.jsonl 且包含失败→成功的记录")
        sys.exit(1)
    logger.info(f"  -> {len(records)} 个成功修复记录")
    for r in records:
        logger.info(f"     {r.task_id}: error_step={r.error_step}")

    _save_file(work_dir, "collected_fix_records.json", json.dumps({
        "metadata": metadata,
        "records": [
            {"task_id": r.task_id, "error_step": r.error_step,
             "diff_lines": r.diff.count("\n"),
             "error_log_preview": r.error_log[:200] if r.error_log else ""}
            for r in records
        ],
    }, ensure_ascii=False, indent=2))
    logger.info(f"  -> saved: {work_dir}/collected_fix_records.json")

    logger.info(f"[2/4] 调用 LLM 提取修复经验 (model_level={model_level})")
    template = _load_template("analyze_error_fix.j2")
    prompt_vars = to_prompt_vars(records, metadata)
    rendered = template.format(**prompt_vars)
    _save_file(work_dir, "llm_prompt.txt", rendered)
    logger.info(f"  -> prompt: {len(rendered)} chars")
    logger.info(f"  -> saved: {work_dir}/llm_prompt.txt")

    body = await _call_llm(rendered, model_level)
    if not body:
        logger.error("LLM 返回为空")
        sys.exit(1)
    _save_file(work_dir, "llm_response.txt", body)
    logger.info(f"  -> response: {len(body)} chars")
    logger.info(f"  -> saved: {work_dir}/llm_response.txt")

    writer = SkillWriter()
    skill_path = writer.get_error_fix_skill_path(metadata, output_dir or None)
    existing_body = writer.read_error_fix_body(skill_path)

    if existing_body:
        logger.info(f"[3/4] 已有 SKILL.md ({len(existing_body)} 字符)，LLM 去重")
        _save_file(work_dir, "existing_skill_body.md", existing_body)
        dedup_template = _load_template("dedup_error_fix.j2")
        dedup_rendered = dedup_template.format(
            existing_body=existing_body,
            new_body=body,
        )
        _save_file(work_dir, "llm_dedup_prompt.txt", dedup_rendered)
        logger.info(f"  -> dedup prompt: {len(dedup_rendered)} chars")

        increment = await _call_llm(dedup_rendered, model_level)
        _save_file(work_dir, "llm_dedup_response.txt", increment or "")
        logger.info(f"  -> dedup response: {len(increment or '')} chars")

        no_new = not increment or not increment.strip() or "无新增内容" in increment
        if no_new:
            elapsed = time.time() - t0
            logger.info(f"无新增内容，已有 SKILL.md 未变更 ({elapsed:.1f}s)")
            print(f"\n{'='*60}")
            print(f"无新增内容，已有 SKILL.md 未变更: {skill_path}")
            print(f"  fix records: {len(records)}")
            print(f"  elapsed    : {elapsed:.1f}s")
            print(f"{'='*60}")
            return
        body = increment
    else:
        logger.info("[3/4] 无已有 SKILL.md，跳过去重")

    logger.info("[4/4] 写入 SKILL.md")
    skill_path = writer.write_error_fix(body, metadata, output_dir or None)

    action = "追加" if existing_body else "新建"
    elapsed = time.time() - t0
    logger.info(f"完成 ({elapsed:.1f}s): {skill_path}")
    print(f"\n{'='*60}")
    print(f"SKILL.md 已{action}（错误修复经验）: {skill_path}")
    print(f"  fix records: {len(records)}")
    print(f"  elapsed    : {elapsed:.1f}s")
    print(f"  work_dir   : {work_dir}/")
    print(f"{'='*60}")


async def _merge_one_category(
    dsl: str, case_type: str, evolved_dir: str,
    output_dir: str, work_dir: str, model_level: str,
) -> Tuple[List[str], List]:
    """对单个 case_type (fix / improvement) 目录执行扫描→聚类→合并"""
    from akg_agents.op.tools.skill_evolution.merge_utils import (
        scan_evolved_skills, build_summaries, parse_classify_output,
        archive_skills, write_merged_skill,
        split_large_cluster,
    )
    from akg_agents.op.tools.skill_evolution.common import parse_skill_output
    from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
    import json

    label = f"[{case_type}]"
    logger.info(f"{label} 扫描目录: {evolved_dir}")
    skills = scan_evolved_skills(evolved_dir)
    if len(skills) < 2:
        logger.info(f"{label} 目录下只有 {len(skills)} 个 skill，无需合并，跳过")
        return [], []

    logger.info(f"{label} -> {len(skills)} 个 skill")
    name_to_skill = {s.name: s for s in skills}
    summaries = build_summaries(skills)
    _save_file(work_dir, f"skill_summaries_{case_type}.json",
               json.dumps(summaries, ensure_ascii=False, indent=2))

    classify_template = _load_template("classify_skills.j2")
    classify_rendered = classify_template.format(
        skill_count=len(summaries), summaries=summaries,
    )
    _save_file(work_dir, f"classify_{case_type}_prompt.txt", classify_rendered)

    classify_output = await _call_llm(classify_rendered, model_level)
    _save_file(work_dir, f"classify_{case_type}_response.txt", classify_output or "")

    clusters = parse_classify_output(classify_output)
    if not clusters:
        logger.warning(f"{label} LLM 聚类输出解析失败")
        return [], []
    _save_file(work_dir, f"clusters_{case_type}.json",
               json.dumps(clusters, ensure_ascii=False, indent=2))
    logger.info(f"{label} {len(clusters)} 个簇: " +
                ", ".join(f"簇{i}({len(c['skills'])}个)" for i, c in enumerate(clusters)))

    merge_template = _load_template("merge_cluster.j2")
    merged_paths = []
    skills_to_archive = []

    for ci, cluster in enumerate(clusters):
        reason = cluster.get("reason", "")
        valid_names = [n for n in cluster["skills"] if n in name_to_skill]
        cluster_label = f"{label}簇{ci}"

        if len(valid_names) < 2:
            logger.info(f"  {cluster_label} 只有 {len(valid_names)} 个 skill，保留原样")
            continue

        cluster_skills = [name_to_skill[n] for n in valid_names]
        batches = split_large_cluster(valid_names)

        cluster_dsl, backend = "", ""
        for s in cluster_skills:
            if not cluster_dsl:
                cluster_dsl = s.metadata.get("dsl", "")
            if not backend:
                backend = s.metadata.get("backend", "")
            if cluster_dsl and backend:
                break
        dsl_prefix = dsl_to_dir_key(cluster_dsl) if cluster_dsl else dsl_to_dir_key(dsl)

        merged_body, merged_name, merged_desc = None, "", ""

        for batch_idx, batch_names in enumerate(batches):
            batch_skills = [name_to_skill[n] for n in batch_names]
            if merged_body is not None:
                documents = [{"name": "已合并文档", "content": merged_body}]
                documents.extend({"name": s.name, "content": s.content} for s in batch_skills)
            else:
                documents = [{"name": s.name, "content": s.content} for s in batch_skills]

            merge_rendered = merge_template.format(
                cluster_reason=reason, dsl_prefix=dsl_prefix,
                doc_count=len(documents), documents=documents,
            )
            suffix = f"_batch{batch_idx}" if len(batches) > 1 else ""
            _save_file(work_dir, f"merge_{case_type}_cluster{ci}{suffix}_prompt.txt", merge_rendered)

            merge_output = await _call_llm(merge_rendered, model_level)
            _save_file(work_dir, f"merge_{case_type}_cluster{ci}{suffix}_response.txt", merge_output or "")

            name, desc, body = parse_skill_output(merge_output)
            if not body:
                logger.warning(f"  {cluster_label}{suffix}: 合并输出为空，跳过")
                continue
            merged_body = body
            if name:
                merged_name = name
            if desc:
                merged_desc = desc

        if not merged_body:
            continue
        if not merged_name:
            merged_name = f"{dsl_prefix}-merged-{case_type}-cluster{ci}"

        target_dir = output_dir or evolved_dir
        skill_path = write_merged_skill(
            name=merged_name, description=merged_desc, body=merged_body,
            dsl=cluster_dsl or dsl, backend=backend, evolved_dir=target_dir,
        )
        merged_paths.append(skill_path)
        skills_to_archive.extend(cluster_skills)
        logger.info(f"  {cluster_label} 合并完成: {skill_path}")

    if skills_to_archive and not output_dir:
        archive_path = archive_skills(skills_to_archive, evolved_dir)
        logger.info(f"{label} 已归档 {len(skills_to_archive)} 个原始 skill 至 {archive_path}")

    return merged_paths, skills_to_archive


def _find_raw_fix_file(fix_dir: str, dsl: str) -> Optional[Path]:
    """定位 error_fix 追加模式产生的原始文件。

    error_fix 模式始终写入固定路径 evolved-fix/{dsl}-error-fix/SKILL.md，
    split 只处理这个文件，不碰已 split 过的产物。
    """
    from akg_agents.op.tools.skill_evolution.common import SkillWriter
    raw_name = SkillWriter._error_fix_skill_name(dsl)
    candidate = Path(fix_dir) / raw_name / "SKILL.md"
    if candidate.is_file():
        return candidate
    return None


async def _split_fix_skills(
    dsl: str, fix_dir: str,
    output_dir: str, work_dir: str, model_level: str,
) -> List[str]:
    """读取 error_fix 原始追加文件 → LLM 按错误根因拆分 → 写入多个 fix skill。

    只处理 {dsl}-error-fix/SKILL.md（error_fix 追加模式的唯一产物），
    不碰已 split 过的产物（如 triton-ascend-fix-cbuf-overflow/ 等）。
    split 完成后归档原始文件，下次 error_fix 会重新创建。
    """
    from akg_agents.op.tools.skill_evolution.common import SkillWriter
    from akg_agents.op.tools.skill_evolution.merge_utils import archive_skills
    from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
    from akg_agents.core_v2.skill.loader import SkillLoader
    import json

    label = "[fix-split]"

    raw_file = _find_raw_fix_file(fix_dir, dsl)
    if raw_file is None:
        logger.info(f"{label} 未找到待 split 的原始 fix 文件，跳过")
        return []

    loader = SkillLoader()
    loaded = loader.load_single(raw_file)
    if not loaded or not loaded.content or len(loaded.content) < 50:
        logger.info(f"{label} 原始 fix 文件内容过短，跳过 split")
        return []

    raw_body = loaded.content
    logger.info(f"{label} 读取原始 fix 文件: {raw_file} ({len(raw_body)} chars)")
    _save_file(work_dir, "split_fix_input.md", raw_body)

    # 同时收集已有 split 产物的摘要，供 LLM 去重
    existing_summaries = _collect_existing_split_summaries(fix_dir, dsl)
    if existing_summaries:
        n_existing = existing_summaries.count("\n- **") + (1 if existing_summaries.startswith("- **") else 0)
        logger.info(f"{label} 发现 {n_existing} 个已有 split 产物，将传入 LLM 去重")

    dsl_prefix = dsl_to_dir_key(dsl)
    split_template = _load_template("split_fix.j2")
    split_rendered = split_template.format(
        fix_body=raw_body,
        dsl_prefix=dsl_prefix,
        existing_summaries=existing_summaries,
    )
    _save_file(work_dir, "split_fix_prompt.txt", split_rendered)
    logger.info(f"{label} split prompt: {len(split_rendered)} chars")

    split_output = await _call_llm(split_rendered, model_level)
    _save_file(work_dir, "split_fix_response.txt", split_output or "")

    groups = _parse_split_output(split_output)
    if not groups:
        logger.warning(f"{label} LLM split 输出解析失败，保留原文件")
        return []

    logger.info(f"{label} LLM 拆分为 {len(groups)} 组")
    _save_file(work_dir, "split_fix_groups.json",
               json.dumps(groups, ensure_ascii=False, indent=2))

    target_dir = output_dir or fix_dir
    writer = SkillWriter()
    written_paths = []

    for gi, group in enumerate(groups):
        skill_name = group.get("skill_name", f"{dsl_prefix}-fix-group{gi}")
        description = group.get("description", "")
        content = group.get("content", "")
        if not content:
            continue

        backend_guess = "ascend" if "ascend" in dsl else ""

        skill_dir = os.path.join(target_dir, skill_name)
        os.makedirs(skill_dir, exist_ok=True)
        fm = writer._make_error_fix_frontmatter(dsl, backend_guess, description)
        fm_patched = fm.replace(
            f"name: {writer._error_fix_skill_name(dsl)}",
            f"name: {skill_name}",
        )
        skill_path = os.path.join(skill_dir, "SKILL.md")
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(fm_patched + "\n\n" + content.strip() + "\n")
        written_paths.append(skill_path)
        logger.info(f"{label} 写入: {skill_path}")

    # 只归档原始追加文件，不碰已有 split 产物
    if written_paths and not output_dir:
        original = [loaded]
        archive_skills(original, fix_dir)
        logger.info(f"{label} 已归档原始 fix 文件: {raw_file.parent.name}")

    return written_paths


def _collect_existing_split_summaries(fix_dir: str, dsl: str) -> str:
    """收集 evolved-fix/ 下已有 split 产物的 name + description，供 LLM 去重。

    跳过原始追加文件和 .archive 目录。
    """
    from akg_agents.op.tools.skill_evolution.common import SkillWriter
    from akg_agents.core_v2.skill.loader import SkillLoader

    raw_name = SkillWriter._error_fix_skill_name(dsl)
    fix_path = Path(fix_dir)
    if not fix_path.exists():
        return ""

    loader = SkillLoader()
    lines = []
    for md in fix_path.rglob("SKILL.md"):
        rel = md.relative_to(fix_path)
        if ".archive" in rel.parts:
            continue
        if rel.parts[0] == raw_name:
            continue
        loaded = loader.load_single(md)
        if loaded:
            lines.append(f"- **{loaded.name}**: {loaded.description or '(无描述)'}")

    return "\n".join(lines)


def _parse_split_output(llm_output: str) -> List[dict]:
    """解析 split_fix LLM 输出，返回 [{skill_name, description, content}, ...]"""
    import json as _json
    import re

    text = (llm_output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        data = _json.loads(text)
        if isinstance(data, list):
            return data
    except _json.JSONDecodeError:
        pass

    arr_match = re.search(r'\[.*\]', text, re.DOTALL)
    if arr_match:
        try:
            data = _json.loads(arr_match.group())
            if isinstance(data, list):
                return data
        except _json.JSONDecodeError:
            pass
    return []


async def run_organize_skills(dsl: str, skills_dir: str, output_dir: str, model_level: str):
    """organize 模式: 对 evolved-fix 执行 split，对 evolved-improvement 执行 merge

    skills_dir 如果指定，视为 dsl 根目录（如 triton-ascend/），自动派生子目录。
    """
    from akg_agents.op.tools.skill_evolution.common import get_default_evolved_dir

    t0 = time.time()
    work_dir = output_dir or _default_work_dir("organize", dsl)
    os.makedirs(work_dir, exist_ok=True)

    all_paths: List[str] = []
    total_archived = 0

    if skills_dir:
        fix_dir = os.path.join(skills_dir, "evolved-fix")
        imp_dir = os.path.join(skills_dir, "evolved-improvement")
    else:
        fix_dir = get_default_evolved_dir(dsl, case_type="fix")
        imp_dir = get_default_evolved_dir(dsl, case_type="improvement")

    fix_exists = os.path.isdir(fix_dir)
    imp_exists = os.path.isdir(imp_dir)
    if not fix_exists and not imp_exists:
        logger.error(f"evolved-fix 和 evolved-improvement 目录均不存在:")
        logger.error(f"  fix: {fix_dir}")
        logger.error(f"  improvement: {imp_dir}")
        logger.error("提示: 请先使用 error_fix/search_log/expert_tuning 模式生成 evolved skill")
        sys.exit(1)

    # --- fix: split ---
    logger.info(f"[organize] === Phase 1: split fix skills ({fix_dir}) ===")
    if not fix_exists:
        logger.info(f"[organize] evolved-fix 目录不存在，跳过 Phase 1")
        fix_paths = []
    else:
        fix_paths = await _split_fix_skills(dsl, fix_dir, output_dir, work_dir, model_level)
    all_paths.extend(fix_paths)

    # --- improvement: merge ---
    logger.info(f"[organize] === Phase 2: merge improvement skills ({imp_dir}) ===")
    if not imp_exists:
        logger.info(f"[organize] evolved-improvement 目录不存在，跳过 Phase 2")
        merged, archived = [], []
    else:
        merged, archived = await _merge_one_category(
            dsl, "improvement", imp_dir, output_dir, work_dir, model_level,
        )
    all_paths.extend(merged)
    total_archived += len(archived)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Skill 整理完成:")
    print(f"  fix split   : {len(fix_paths)} 个新 skill")
    print(f"  improvement : {len(merged)} 个合并 skill（归档 {total_archived} 个）")
    for p in all_paths:
        print(f"    - {p}")
    print(f"  耗时        : {elapsed:.1f}s")
    print(f"  中间文件    : {work_dir}/")
    print(f"{'='*60}")


def _default_work_dir(mode: str, op_name: str) -> str:
    """统一的默认工作目录：~/.akg/skill_evolution/{mode}_{op_name}"""
    base = os.path.join(os.path.expanduser("~"), ".akg", "skill_evolution")
    return os.path.join(base, f"{mode}_{op_name}")


def _save_file(directory: str, filename: str, content: str) -> None:
    try:
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logger.warning(f"保存 {filename} 失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Skill Evolution CLI — 从搜索日志、对话历史或错误修复记录生成 SKILL.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    sp_a = subparsers.add_parser("search_log", help="从搜索日志生成 SKILL.md")
    sp_a.add_argument("log_dir", help="搜索日志目录（节点 logs 路径）")
    sp_a.add_argument("op_name", help="算子名称（如 relu、l1norm）")
    sp_a.add_argument("--task-desc", "-t", default="", help="算子任务描述（可选，注入到 LLM prompt）")
    sp_a.add_argument("--output-dir", "-o", default="", help="SKILL.md 输出目录")
    sp_a.add_argument("--model-level", "-m", default="standard", help="LLM 模型级别")

    sp_b = subparsers.add_parser("expert_tuning", help="从对话目录提取专家调优经验，生成 SKILL.md")
    sp_b.add_argument("conversation_dir", help="对话目录路径（如 .akg/conversations/cli_xxx）")
    sp_b.add_argument("op_name", help="算子名称")
    sp_b.add_argument("--output-dir", "-o", default="", help="SKILL.md 输出目录")
    sp_b.add_argument("--model-level", "-m", default="standard", help="LLM 模型级别")

    sp_c = subparsers.add_parser("error_fix", help="从错误修复记录生成/合并 SKILL.md")
    sp_c.add_argument("log_dir", help="搜索日志目录（节点 logs 路径）")
    sp_c.add_argument("op_name", help="算子名称")
    sp_c.add_argument("--output-dir", "-o", default="", help="SKILL.md 输出目录")
    sp_c.add_argument("--model-level", "-m", default="standard", help="LLM 模型级别")

    sp_d = subparsers.add_parser("organize", help="整理 evolved skills：fix 执行 split，improvement 执行 merge")
    sp_d.add_argument("dsl", help="DSL 类型（如 triton_cuda、triton_ascend）")
    sp_d.add_argument("--skills-dir", "-s", default="", help="evolved skill 目录（默认 ~/.akg/evolved_skills/{dsl}/evolved-fix 和 evolved-improvement）")
    sp_d.add_argument("--output-dir", "-o", default="", help="输出目录（默认覆写到原目录）")
    sp_d.add_argument("--model-level", "-m", default="standard", help="LLM 模型级别")

    args = parser.parse_args()

    if args.mode is None:
        if len(sys.argv) >= 3 and os.path.isdir(sys.argv[1]):
            legacy = argparse.ArgumentParser()
            legacy.add_argument("log_dir")
            legacy.add_argument("op_name")
            legacy.add_argument("--output-dir", "-o", default="")
            legacy.add_argument("--model-level", "-m", default="standard")
            args = legacy.parse_args()
            args.mode = "search_log"
        else:
            print(_USAGE_HINT)
            parser.print_help()
            sys.exit(1)

    if args.mode == "search_log":
        if not os.path.isdir(args.log_dir):
            logger.error(f"log_dir 不存在: {args.log_dir}")
            print(_USAGE_HINT)
            sys.exit(1)
        asyncio.run(run_search_log(args.log_dir, args.op_name, args.output_dir, args.model_level, getattr(args, "task_desc", "")))
    elif args.mode == "expert_tuning":
        if not os.path.isdir(args.conversation_dir):
            logger.error(f"conversation_dir 不存在: {args.conversation_dir}")
            print(_USAGE_HINT)
            sys.exit(1)
        asyncio.run(run_expert_tuning(args.conversation_dir, args.op_name, args.output_dir, args.model_level))
    elif args.mode == "error_fix":
        if not os.path.isdir(args.log_dir):
            logger.error(f"log_dir 不存在: {args.log_dir}")
            print(_USAGE_HINT)
            sys.exit(1)
        asyncio.run(run_error_fix(args.log_dir, args.op_name, args.output_dir, args.model_level))
    elif args.mode == "organize":
        asyncio.run(run_organize_skills(args.dsl, args.skills_dir, args.output_dir, args.model_level))


if __name__ == "__main__":
    main()
