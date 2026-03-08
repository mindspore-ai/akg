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
run_skill_evolution.py - Skill Evolution 独立 CLI 工具（支持双模式）

此工具用于从 adaptive_search 搜索日志或 akg_cli 对话历史中提取优化经验，
生成可复用的 SKILL.md 文档。

前置条件：
  - search_log 模式需要先运行 `akg_cli op` 执行 adaptive_search，产生搜索日志
  - expert_tuning 模式需要先运行 `akg_cli op` 进行交互式调优，产生对话记录
  日志默认保存在 ~/.akg/conversations/cli_<session_id>/nodes/<node_id>/logs/

search_log 模式:
    python run_skill_evolution.py search_log <log_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

expert_tuning 模式:
    python run_skill_evolution.py expert_tuning <conversation_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

兼容旧用法（默认 search_log）:
    python run_skill_evolution.py <log_dir> <op_name>

示例:
    python run_skill_evolution.py search_log /path/to/node_004/logs relu
    python run_skill_evolution.py expert_tuning ~/.akg/conversations/cli_xxx rope -o ./output
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

    work_dir = output_dir or os.path.join(log_dir, "_skill_evolution_debug")
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

    work_dir = output_dir or os.path.join(conversation_dir, "_skill_evolution_debug")
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


def _save_file(directory: str, filename: str, content: str) -> None:
    try:
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logger.warning(f"保存 {filename} 失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Skill Evolution CLI — 从搜索日志或对话历史生成 SKILL.md",
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


if __name__ == "__main__":
    main()
