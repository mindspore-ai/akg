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
  - search_log 模式需要先运行 `akg_cli op` 执行 adaptive_search，产生搜索日志
  - expert_tuning 模式需要先运行 `akg_cli op` 进行交互式调优，产生对话记录
  - error_fix 模式需要搜索日志中包含失败→成功的修复记录
  - merge_skills 模式需要 evolved 目录下有至少 2 个 SKILL.md
  日志默认保存在 ~/.akg/conversations/cli_<session_id>/nodes/<node_id>/logs/

search_log 模式:
    python run_skill_evolution.py search_log <log_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

expert_tuning 模式:
    python run_skill_evolution.py expert_tuning <conversation_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

error_fix 模式:
    python run_skill_evolution.py error_fix <log_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

merge_skills 模式:
    python run_skill_evolution.py merge_skills <dsl> [--skills-dir DIR] [--output-dir DIR] [--model-level LEVEL]

兼容旧用法（默认 search_log）:
    python run_skill_evolution.py <log_dir> <op_name>

示例:
    python run_skill_evolution.py search_log /path/to/node_004/logs relu
    python run_skill_evolution.py expert_tuning ~/.akg/conversations/cli_xxx rope -o ./output
    python run_skill_evolution.py error_fix /path/to/node_005/logs matmul -o ./output
    python run_skill_evolution.py merge_skills triton_cuda
    python run_skill_evolution.py merge_skills triton_cuda --skills-dir /path/to/evolved -o ./merged_output
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

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
║  merge_skills 模式:                                                  ║
║    将 evolved 目录下的多个 skill 按主题合并去重                      ║
║    参数: merge_skills <dsl> [--skills-dir DIR]                       ║
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

    logger.info(f"[1/4] 收集数据: log_dir={log_dir}, op_name={op_name}")
    records, metadata = collect(log_dir, op_name)
    metadata["op_name"] = op_name
    if not records:
        logger.error("未找到任何任务记录，请检查 log_dir 是否正确")
        logger.error("提示: log_dir 应指向 adaptive_search 产生的节点 logs 目录")
        logger.error("      例如: ~/.akg/conversations/cli_xxx/nodes/node_004/logs")
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

    logger.info(f"[1/4] 读取所有 action: {conversation_dir}")
    sections, metadata = collect(conversation_dir, op_name)
    metadata["op_name"] = op_name
    metadata["source"] = "expert_tuning"

    if not sections:
        logger.error("未读取到任何 action 记录")
        logger.error("提示: conversation_dir 应指向 akg_cli 的会话目录")
        logger.error("      例如: ~/.akg/conversations/cli_<session_id>")
        logger.error("      该目录应包含 nodes/ 子目录")
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

    logger.info(f"[1/4] 收集成功修复记录: log_dir={log_dir}, op_name={op_name}")
    records, metadata = collect(log_dir, op_name)
    metadata["op_name"] = op_name
    metadata["source"] = "error_fix"
    if not records:
        logger.error("未找到任何成功修复记录")
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


async def run_merge_skills(dsl: str, skills_dir: str, output_dir: str, model_level: str):
    """merge_skills 模式: 将 evolved skills 按主题合并去重"""
    from akg_agents.op.tools.skill_evolution.merge_utils import (
        scan_evolved_skills, build_summaries, parse_classify_output,
        archive_skills, write_merged_skill,
        split_large_cluster,
    )
    from akg_agents.op.tools.skill_evolution.common import (
        parse_skill_output,
        get_default_evolved_dir,
    )
    import json

    t0 = time.time()

    evolved_dir = skills_dir or get_default_evolved_dir(dsl)
    work_dir = output_dir or _default_work_dir("merge_skills", dsl)
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"[1/3] 扫描 evolved skills: {evolved_dir}")
    skills = scan_evolved_skills(evolved_dir)
    if len(skills) < 2:
        logger.error(f"evolved 目录下只有 {len(skills)} 个 skill，无需合并")
        sys.exit(1)
    logger.info(f"  -> {len(skills)} 个 skill")

    name_to_skill = {s.name: s for s in skills}
    summaries = build_summaries(skills)
    _save_file(work_dir, "skill_summaries.json", json.dumps(summaries, ensure_ascii=False, indent=2))

    # Phase 1: 摘要聚类
    logger.info(f"[2/3] Phase 1: 摘要聚类 (model_level={model_level})")
    classify_template = _load_template("classify_skills.j2")
    classify_rendered = classify_template.format(
        skill_count=len(summaries),
        summaries=summaries,
    )
    _save_file(work_dir, "classify_prompt.txt", classify_rendered)
    logger.info(f"  -> classify prompt: {len(classify_rendered)} chars")

    classify_output = await _call_llm(classify_rendered, model_level)
    _save_file(work_dir, "classify_response.txt", classify_output or "")

    clusters = parse_classify_output(classify_output)
    if not clusters:
        logger.error("LLM 聚类输出解析失败")
        sys.exit(1)
    _save_file(work_dir, "clusters.json", json.dumps(clusters, ensure_ascii=False, indent=2))
    logger.info(f"  -> {len(clusters)} 个簇: " + ", ".join(f"簇{i}({len(c['skills'])}个)" for i, c in enumerate(clusters)))

    # Phase 2: 逐簇合并
    logger.info("[3/3] Phase 2: 逐簇合并")
    merge_template = _load_template("merge_cluster.j2")
    merged_paths = []
    skills_to_archive = []

    for ci, cluster in enumerate(clusters):
        reason = cluster.get("reason", "")
        valid_names = [n for n in cluster["skills"] if n in name_to_skill]
        cluster_label = f"簇{ci}"

        if len(valid_names) < 2:
            logger.info(f"  {cluster_label} 只有 {len(valid_names)} 个 skill，保留原样")
            continue

        cluster_skills = [name_to_skill[n] for n in valid_names]
        batches = split_large_cluster(valid_names)

        cluster_dsl = ""
        backend = ""
        for s in cluster_skills:
            if not cluster_dsl:
                cluster_dsl = s.metadata.get("dsl", "")
            if not backend:
                backend = s.metadata.get("backend", "")
            if cluster_dsl and backend:
                break
        from akg_agents.core.utils import dsl_to_dir_key
        dsl_prefix = dsl_to_dir_key(cluster_dsl) if cluster_dsl else dsl_to_dir_key(dsl)

        merged_body = None
        merged_name = ""
        merged_desc = ""

        for batch_idx, batch_names in enumerate(batches):
            batch_skills = [name_to_skill[n] for n in batch_names]

            if merged_body is not None:
                documents = [{"name": "已合并文档", "content": merged_body}]
                documents.extend({"name": s.name, "content": s.content} for s in batch_skills)
            else:
                documents = [{"name": s.name, "content": s.content} for s in batch_skills]

            merge_rendered = merge_template.format(
                cluster_reason=reason,
                dsl_prefix=dsl_prefix,
                doc_count=len(documents),
                documents=documents,
            )
            suffix = f"_batch{batch_idx}" if len(batches) > 1 else ""
            _save_file(work_dir, f"merge_cluster{ci}{suffix}_prompt.txt", merge_rendered)
            logger.info(f"  {cluster_label}{suffix}: merge prompt {len(merge_rendered)} chars")

            merge_output = await _call_llm(merge_rendered, model_level)
            _save_file(work_dir, f"merge_cluster{ci}{suffix}_response.txt", merge_output or "")

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
            merged_name = f"{dsl_prefix}-merged-cluster{ci}"
            logger.info(f"  {cluster_label}: LLM 未返回 skill_name，使用默认: {merged_name}")

        target_dir = output_dir or evolved_dir
        skill_path = write_merged_skill(
            name=merged_name,
            description=merged_desc,
            body=merged_body,
            dsl=cluster_dsl or dsl,
            backend=backend,
            evolved_dir=target_dir,
        )
        merged_paths.append(skill_path)
        skills_to_archive.extend(cluster_skills)
        logger.info(f"  {cluster_label} 合并完成: {skill_path}")

    # 仅当 output_dir 未指定时才归档原始文件
    if skills_to_archive and not output_dir:
        archive_path = archive_skills(skills_to_archive, evolved_dir)
        logger.info(f"已归档 {len(skills_to_archive)} 个原始 skill 至 {archive_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Skill 合并完成:")
    print(f"  合并主题 : {len(merged_paths)} 个")
    for p in merged_paths:
        print(f"    - {p}")
    print(f"  归档     : {len(skills_to_archive)} 个原始 skill")
    print(f"  耗时     : {elapsed:.1f}s")
    print(f"  中间文件 : {work_dir}/")
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

    sp_d = subparsers.add_parser("merge_skills", help="将 evolved skills 按主题合并去重")
    sp_d.add_argument("dsl", help="DSL 类型（如 triton_cuda、triton_ascend）")
    sp_d.add_argument("--skills-dir", "-s", default="", help="evolved skill 目录（默认 op/resources/skills/{dsl}/evolved/）")
    sp_d.add_argument("--output-dir", "-o", default="", help="合并输出目录（默认覆写到 evolved/）")
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
    elif args.mode == "merge_skills":
        asyncio.run(run_merge_skills(args.dsl, args.skills_dir, args.output_dir, args.model_level))


if __name__ == "__main__":
    main()
