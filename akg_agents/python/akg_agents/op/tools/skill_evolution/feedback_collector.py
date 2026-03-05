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
Skill 自进化系统 - 对话历史收集器 (Mode B)

从 conversation 目录读取所有 node 的 action_history_fact.json，
按节点编号排序后格式化为 Markdown 时间线。

当增量构建时间线超出字符阈值时，用 LLM 压缩已积累的部分
（保留用户优化建议、代码生成工具产出的完整代码、性能数据），
然后继续追加新节点内容，以此类推。
"""

import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_TIMELINE_CHARS = 60000

_COMPRESS_PROMPT = """你是一个对话历史压缩助手。请压缩以下 action 时间线，为后续新增内容腾出空间。

## 必须完整保留

1. **用户回复**（`ask_user` 中的"用户回复"内容）——包含用户的优化建议和需求
2. **生成的代码**（`call_kernelgen_workflow` / `call_evolve_workflow` / `call_kernel_gen` 等代码生成工具产出的完整代码块）
3. **性能数据**（`profile_kernel` 中的 gen_time、base_time、speedup 等所有数值）
4. 各工具调用的**执行状态**（success/fail）

## 可以压缩或删除

1. Agent 发给用户的消息（解释性文字、确认提问等）
2. 工具调用的冗余参数，read_file 等无用工具的历史
3. 重复的错误信息
4. 不影响理解"用户建议 → 代码变更 → 性能变化"因果关系的中间内容

## 输出要求

- 保持 Turn 编号和 Markdown 格式
- 压缩后总长度控制在 **{target_chars}** 字符以内
- 直接输出压缩后的时间线，不要添加任何解释或前言

## 待压缩的时间线

{timeline}"""


# ==================== 公开接口 ====================


def collect_feedback(
    conversation_dir: str,
    op_name: str = "",
) -> Tuple[List[str], Dict[str, str]]:
    """从 conversation 目录读取所有 node 的 action，格式化为 section 列表

    读取 {conversation_dir}/nodes/*/actions/action_history_fact.json，
    按节点编号排序，每个 action 格式化为一个 Markdown section。

    Args:
        conversation_dir: 对话目录路径
        op_name: 算子名称（用于日志和 metadata）

    Returns:
        (sections, metadata)
        - sections: 每个 action 对应的格式化文本列表（按时间顺序）
        - metadata: 环境信息 dict (op_name, dsl, backend, arch)
    """
    actions = _load_all_actions(conversation_dir)
    logger.info(
        f"[FeedbackCollector] conversation_dir={conversation_dir}, "
        f"{len(actions)} 条 action, op_name={op_name}"
    )

    metadata: Dict[str, str] = {
        "op_name": op_name, "dsl": "", "backend": "", "arch": "",
    }
    sections: List[str] = []

    for action in actions:
        _try_fill_metadata(metadata, action.get("arguments", {}))
        section = _format_action(action)
        if section:
            sections.append(section)

    logger.info(
        f"[FeedbackCollector] 生成 {len(sections)} 个 section, "
        f"总计 {sum(len(s) for s in sections)} 字符"
    )
    return sections, metadata


async def build_timeline(
    sections: List[str],
    llm_fn: Callable[[str], Awaitable[str]],
    max_chars: int = MAX_TIMELINE_CHARS,
    work_dir: str = "",
) -> str:
    """增量构建时间线，超出阈值时用 LLM 压缩已积累部分

    逐个添加 section，每次添加后检查总长度：
    - 未超阈值：继续
    - 超出阈值：LLM 压缩当前已积累的文本，再追加新 section

    Args:
        sections: action 格式化文本列表
        llm_fn: 异步 LLM 调用函数
        max_chars: 时间线最大字符数
        work_dir: 中间文件输出目录（可选，用于调试）

    Returns:
        最终时间线文本
    """
    accumulated = ""
    compress_count = 0

    for i, section in enumerate(sections):
        candidate = accumulated + "\n" + section if accumulated else section

        if len(candidate) <= max_chars:
            accumulated = candidate
            continue

        # 超出阈值：压缩已积累的部分
        compress_count += 1
        target = max(2000, max_chars - len(section) - 500)
        logger.info(
            f"[FeedbackCollector] 第 {compress_count} 次压缩: "
            f"已积累 {len(accumulated)} 字符 + 新 section {len(section)} 字符 "
            f"> {max_chars}，目标压缩到 {target} 字符"
        )

        if work_dir:
            _save_debug(work_dir, f"pre_compress_{compress_count}.md", accumulated)

        prompt = _COMPRESS_PROMPT.format(
            target_chars=target,
            timeline=accumulated,
        )

        try:
            compressed = await llm_fn(prompt)
            if compressed and compressed.strip():
                accumulated = compressed.strip()
                logger.info(
                    f"[FeedbackCollector] 压缩完成: → {len(accumulated)} 字符"
                )
            else:
                logger.warning(
                    "[FeedbackCollector] LLM 压缩返回空，保留原文"
                )
        except Exception as e:
            logger.error(f"[FeedbackCollector] LLM 压缩失败: {e}")

        if work_dir:
            _save_debug(work_dir, f"post_compress_{compress_count}.md", accumulated)

        accumulated = accumulated + "\n" + section

    if compress_count > 0:
        logger.info(
            f"[FeedbackCollector] 共压缩 {compress_count} 次, "
            f"最终 {len(accumulated)} 字符"
        )
    else:
        logger.info(
            f"[FeedbackCollector] 无需压缩, {len(accumulated)} 字符"
        )

    return accumulated


# ==================== 数据加载 ====================


def _load_all_actions(conversation_dir: str) -> List[Dict[str, Any]]:
    """读取所有 node 的 action_history_fact.json，按节点编号排序"""
    nodes_dir = os.path.join(conversation_dir, "nodes")
    if not os.path.isdir(nodes_dir):
        raise FileNotFoundError(f"nodes 目录不存在: {nodes_dir}")

    node_facts: List[Tuple[str, Dict[str, Any]]] = []

    for node_name in os.listdir(nodes_dir):
        if node_name == "root":
            continue
        fact_path = os.path.join(
            nodes_dir, node_name, "actions", "action_history_fact.json",
        )
        if not os.path.isfile(fact_path):
            continue
        try:
            with open(fact_path, "r", encoding="utf-8") as f:
                fact = json.load(f)
            node_facts.append((node_name, fact))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[FeedbackCollector] 读取失败 {fact_path}: {e}")

    # 按节点编号排序（node_001, node_002, ...）
    node_facts.sort(key=lambda x: _node_sort_key(x[0]))

    all_actions: List[Dict[str, Any]] = []
    for node_name, fact in node_facts:
        turn = fact.get("turn", 0)
        for action in fact.get("actions", []):
            action["_turn"] = turn
            action["_node"] = node_name
            all_actions.append(action)

    logger.info(
        f"[FeedbackCollector] 从 {len(node_facts)} 个 node "
        f"读取了 {len(all_actions)} 条 action"
    )
    return all_actions


def _node_sort_key(node_name: str) -> int:
    """提取节点编号用于排序: node_001 → 1"""
    m = re.search(r"(\d+)$", node_name)
    return int(m.group(1)) if m else 0


# ==================== 格式化 ====================


def _format_action(action: Dict[str, Any]) -> str:
    """将单个 action 格式化为 Markdown 段落"""
    tool_name = action.get("tool_name", "")
    arguments = action.get("arguments", {})
    result = action.get("result", {})
    turn = action.get("_turn", "?")
    node = action.get("_node", "")
    ts = action.get("timestamp", "")
    ts_short = ts[:19] if ts else ""

    lines: List[str] = []
    lines.append(f"### {node} Turn {turn} — {tool_name} ({ts_short})")

    # 参数（过滤超长值）
    for k, v in arguments.items():
        v_str = str(v)
        if len(v_str) > 500:
            lines.append(f"**{k}**: ({len(v_str)} 字符，已省略)")
        else:
            lines.append(f"**{k}**: {v_str}")

    # 结果
    if isinstance(result, dict):
        for k, v in result.items():
            if k.startswith("_"):
                continue
            v_str = str(v)
            if len(v_str) > 1000:
                # 代码等长内容保留，用代码块包裹
                lines.append(f"**{k}** ({len(v_str)} 字符):")
                lines.append(f"```\n{v_str}\n```")
            elif len(v_str) > 0:
                lines.append(f"**{k}**: {v_str}")

    return "\n".join(lines) + "\n"


# ==================== 工具 ====================


def _try_fill_metadata(metadata: Dict[str, str], arguments: Dict) -> None:
    for key in ("dsl", "backend", "arch"):
        if not metadata.get(key) and arguments.get(key):
            metadata[key] = arguments[key]


def _save_debug(work_dir: str, filename: str, content: str) -> None:
    try:
        path = os.path.join(work_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logger.warning(f"[FeedbackCollector] 保存 {filename} 失败: {e}")
