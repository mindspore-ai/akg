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
Skill 自进化系统 - 专家调优经验提取 (expert_tuning 模式)

通过 trace.json 获取对话树结构，DFS 找到所有 root→leaf 路径，
每条路径作为一个独立分支，按顺序读取 action_history_fact.json
并格式化为 Markdown 时间线。

多分支场景下每个分支包含完整的共享前缀节点，确保因果链完整。
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


def collect(
    conversation_dir: str,
    op_name: str = "",
) -> Tuple[List[str], Dict[str, str]]:
    """从 conversation 目录读取对话树，按分支组织为 section 列表

    通过 trace.json 获取树结构，DFS 找到所有 root→leaf 路径，
    每条路径作为一个独立分支。单分支时不加标签，多分支时用分支标题分隔。
    若 trace.json 不存在则回退到按节点编号排序。

    Returns:
        (sections, metadata)
        - sections: 每个 action 对应的格式化文本列表（按分支 + 路径顺序）
        - metadata: 环境信息 dict (op_name, dsl, backend, arch)
    """
    metadata: Dict[str, str] = {
        "op_name": op_name, "dsl": "", "backend": "", "arch": "",
    }

    node_facts = _load_node_facts(conversation_dir)
    paths = _get_branch_paths(conversation_dir, set(node_facts.keys()))

    multi_branch = len(paths) > 1
    sections: List[str] = []
    total_actions = 0

    for branch_idx, path in enumerate(paths):
        if multi_branch:
            sections.append(
                f"\n---\n## 分支 {branch_idx + 1} "
                f"(共 {len(path)} 个节点)\n"
            )

        for node_id in path:
            fact = node_facts.get(node_id)
            if not fact:
                continue
            turn = fact.get("turn", 0)
            for action in fact.get("actions", []):
                action["_turn"] = turn
                action["_node"] = node_id
                _try_fill_metadata(metadata, action.get("arguments", {}))
                section = _format_action(action)
                if section:
                    sections.append(section)
                    total_actions += 1

    branch_info = f"{len(paths)} 个分支" if multi_branch else "单分支"
    logger.info(
        f"[ExpertTuning:Collect] conversation_dir={conversation_dir}, "
        f"{branch_info}, {total_actions} 条 action, op_name={op_name}"
    )
    logger.info(
        f"[ExpertTuning:Collect] 生成 {len(sections)} 个 section, "
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
    """
    accumulated = ""
    compress_count = 0

    for i, section in enumerate(sections):
        candidate = accumulated + "\n" + section if accumulated else section

        if len(candidate) <= max_chars:
            accumulated = candidate
            continue

        compress_count += 1
        target = max(2000, max_chars - len(section) - 500)
        logger.info(
            f"[ExpertTuning:Timeline] 第 {compress_count} 次压缩: "
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
                    f"[ExpertTuning:Timeline] 压缩完成: → {len(accumulated)} 字符"
                )
            else:
                logger.warning(
                    "[ExpertTuning:Timeline] LLM 压缩返回空，保留原文"
                )
        except Exception as e:
            logger.error(f"[ExpertTuning:Timeline] LLM 压缩失败: {e}")

        if work_dir:
            _save_debug(work_dir, f"post_compress_{compress_count}.md", accumulated)

        accumulated = accumulated + "\n" + section

    if compress_count > 0:
        logger.info(
            f"[ExpertTuning:Timeline] 共压缩 {compress_count} 次, "
            f"最终 {len(accumulated)} 字符"
        )
    else:
        logger.info(
            f"[ExpertTuning:Timeline] 无需压缩, {len(accumulated)} 字符"
        )

    return accumulated


# ==================== 树结构与路径 ====================


def _get_branch_paths(
    conversation_dir: str, available_nodes: set,
) -> List[List[str]]:
    """获取所有分支路径，优先用 trace.json 树结构，回退到节点编号排序"""
    trace_path = os.path.join(conversation_dir, "trace.json")
    if os.path.isfile(trace_path):
        try:
            paths = _paths_from_trace(trace_path, available_nodes)
            if paths:
                logger.info(
                    f"[ExpertTuning] 从 trace.json 获取 {len(paths)} 条分支路径"
                )
                return paths
        except Exception as e:
            logger.warning(f"[ExpertTuning] trace.json 解析失败，回退: {e}")

    sorted_nodes = sorted(available_nodes, key=_node_sort_key)
    logger.info(
        f"[ExpertTuning] 无 trace.json，按节点编号排序: {len(sorted_nodes)} 个节点"
    )
    return [sorted_nodes]


def _paths_from_trace(
    trace_path: str, available_nodes: set,
) -> List[List[str]]:
    """从 trace.json 解析树结构，DFS 返回所有 root→leaf 路径（不含 root）"""
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    tree = trace.get("tree", {})
    if not tree or "root" not in tree:
        return []

    paths: List[List[str]] = []

    def _dfs(node_id: str, path: List[str]) -> None:
        if node_id != "root":
            path.append(node_id)
        children = tree.get(node_id, {}).get("children", [])
        real_children = [c for c in children if c in tree]
        if not real_children:
            if path:
                paths.append(list(path))
        else:
            for child_id in real_children:
                _dfs(child_id, path)
        if node_id != "root":
            path.pop()

    _dfs("root", [])

    # 过滤路径中没有 action 数据的节点
    filtered: List[List[str]] = []
    for path in paths:
        clean = [n for n in path if n in available_nodes]
        if clean:
            filtered.append(clean)

    return filtered


def _load_node_facts(conversation_dir: str) -> Dict[str, Dict[str, Any]]:
    """读取所有 node 的 action_history_fact.json，返回 {node_id: fact_data}"""
    nodes_dir = os.path.join(conversation_dir, "nodes")
    if not os.path.isdir(nodes_dir):
        raise FileNotFoundError(f"nodes 目录不存在: {nodes_dir}")

    result: Dict[str, Dict[str, Any]] = {}

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
                result[node_name] = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[ExpertTuning] 读取失败 {fact_path}: {e}")

    logger.info(
        f"[ExpertTuning] 从 {len(result)} 个 node 读取了 action 数据"
    )
    return result


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

    for k, v in arguments.items():
        v_str = str(v)
        if len(v_str) > 500:
            lines.append(f"**{k}**: ({len(v_str)} 字符，已省略)")
        else:
            lines.append(f"**{k}**: {v_str}")

    if isinstance(result, dict):
        for k, v in result.items():
            if k.startswith("_"):
                continue
            v_str = str(v)
            if len(v_str) > 1000:
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
        logger.warning(f"[ExpertTuning] 保存 {filename} 失败: {e}")


# ==================== Prompt 变量构建 ====================


def to_prompt_vars(
    timeline: str, metadata: Dict[str, str],
) -> Dict[str, Any]:
    """将 action 时间线文本和元数据转换为 analyze_expert_tuning.j2 的模板变量"""
    return {
        "op_name": metadata.get("op_name", ""),
        "dsl": metadata.get("dsl", ""),
        "backend": metadata.get("backend", ""),
        "arch": metadata.get("arch", ""),
        "timeline": timeline,
    }
