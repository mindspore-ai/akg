#!/usr/bin/env python3
"""
test_skill_evolution.py - 从 adaptive_search 日志一键生成 SKILL.md

用法:
    python akg_agents/tests/op/st/test_skill_evolution.py <log_dir> <op_name> [--output-dir DIR] [--model-level LEVEL]

示例:
    python akg_agents/tests/op/st/test_skill_evolution.py /path/to/node_004/logs relu
    python akg_agents/tests/op/st/test_skill_evolution.py /path/to/logs l1norm -o ./output -m complex
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


def _load_template():
    """加载 analyze.j2 模板"""
    from akg_agents.utils.common_utils import get_prompt_path
    template_path = Path(get_prompt_path()) / "skill_evolution" / "analyze.j2"
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


async def run(log_dir: str, op_name: str, output_dir: str, model_level: str):
    from akg_agents.op.tools.skill_evolution.collector import collect
    from akg_agents.op.tools.skill_evolution.compressor import compress
    from akg_agents.op.tools.skill_evolution.analyzer import (
        compressed_to_prompt_vars, parse_skill_output,
    )
    from akg_agents.op.tools.skill_evolution.writer import SkillWriter

    t0 = time.time()

    # 1. 收集
    logger.info(f"[1/4] 收集数据: log_dir={log_dir}, op_name={op_name}")
    records, metadata = collect(log_dir, op_name)
    metadata["op_name"] = op_name
    if not records:
        logger.error("未找到任何任务记录，请检查 log_dir 是否正确")
        sys.exit(1)
    logger.info(f"  → {len(records)} 条记录")

    # 2. 压缩
    logger.info("[2/4] 压缩进化链")
    compressed = compress(records, metadata)
    logger.info(
        f"  → best={compressed.best_task_id} "
        f"({compressed.best_gen_time}us / {compressed.best_speedup:.2f}x), "
        f"进化链={len(compressed.evolution_chains)} 步"
    )

    # 3. LLM 生成
    logger.info(f"[3/4] 调用 LLM (model_level={model_level})")
    template = _load_template()
    prompt_vars = compressed_to_prompt_vars(compressed)
    rendered = template.format(**prompt_vars)
    logger.info(f"  → prompt: {len(rendered)} 字符")

    content = await _call_llm(rendered, model_level)
    if not content:
        logger.error("LLM 返回为空")
        sys.exit(1)
    logger.info(f"  → 生成: {len(content)} 字符")

    skill_name, description, body = parse_skill_output(content)
    if not body:
        logger.error("LLM 未生成有效正文，原始输出已打印到 stderr")
        print(content, file=sys.stderr)
        sys.exit(1)

    # 4. 写入
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
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="从 adaptive_search 日志生成 SKILL.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("log_dir", help="adaptive_search 的日志目录（节点 logs 路径）")
    parser.add_argument("op_name", help="算子名称（如 relu、l1norm）")
    parser.add_argument("--output-dir", "-o", default="", help="SKILL.md 输出目录（默认写入项目 skills 目录）")
    parser.add_argument("--model-level", "-m", default="standard", help="LLM 模型级别 (default: standard)")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        logger.error(f"log_dir 不存在: {args.log_dir}")
        sys.exit(1)

    asyncio.run(run(args.log_dir, args.op_name, args.output_dir, args.model_level))


if __name__ == "__main__":
    main()
