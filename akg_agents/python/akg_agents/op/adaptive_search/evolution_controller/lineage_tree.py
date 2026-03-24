# Copyright 2025 Huawei Technologies Co., Ltd
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
Lineage Tree

从 SuccessDB 重建的有根森林，提供谱系统计信息。
每条初始任务（generation=0, parent_id=None）为一棵树的根。
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from akg_agents.op.adaptive_search.success_db import SuccessDB, SuccessRecord

logger = logging.getLogger(__name__)


@dataclass
class LineageStats:
    """谱系统计信息"""
    root_id: str
    size: int = 0
    best_gen_time: float = float('inf')
    best_speedup: float = 0.0
    best_record_id: str = ""
    max_depth: int = 0
    share: float = 0.0
    total_selections: int = 0
    recent_improvement: float = 0.0


class LineageTree:
    """
    谱系树：从 SuccessDB 重建的有根森林。

    每个初始任务（parent_id=None）是一条谱系的根节点。
    通过 parent_id 字段构建父子关系，形成多棵树。
    """

    def __init__(self):
        self._roots: List[str] = []
        self._children: Dict[str, List[str]] = {}
        self._root_of: Dict[str, str] = {}
        self._depth_of: Dict[str, int] = {}
        self._lineage_stats: Dict[str, LineageStats] = {}
        self._record_map: Dict[str, SuccessRecord] = {}

    def rebuild(self, db: SuccessDB) -> None:
        """
        从 SuccessDB 完整重建谱系树。

        复杂度: O(N)，N 为记录数。
        """
        self._roots.clear()
        self._children.clear()
        self._root_of.clear()
        self._depth_of.clear()
        self._lineage_stats.clear()
        self._record_map.clear()

        records = db.get_all()
        if not records:
            return

        record_ids = set()
        for r in records:
            self._record_map[r.id] = r
            record_ids.add(r.id)

        # 1. 识别根节点，构建 children 映射
        for r in records:
            if r.parent_id is None or r.parent_id not in record_ids:
                self._roots.append(r.id)
            else:
                self._children.setdefault(r.parent_id, []).append(r.id)

        # 2. BFS 填充 root_of 和 depth_of
        for root_id in self._roots:
            self._root_of[root_id] = root_id
            self._depth_of[root_id] = 0

            queue = deque([root_id])
            while queue:
                node_id = queue.popleft()
                for child_id in self._children.get(node_id, []):
                    self._root_of[child_id] = root_id
                    self._depth_of[child_id] = self._depth_of[node_id] + 1
                    queue.append(child_id)

        # 3. 计算谱系统计
        self._compute_lineage_stats(db)

    def get_lineage_root(self, record_id: str) -> str:
        """获取记录所属谱系的根节点 ID"""
        return self._root_of.get(record_id, record_id)

    def get_descendants(self, record_id: str) -> Set[str]:
        """DFS 获取所有后代"""
        descendants = set()
        stack = list(self._children.get(record_id, []))
        while stack:
            node_id = stack.pop()
            descendants.add(node_id)
            stack.extend(self._children.get(node_id, []))
        return descendants

    def get_lineage_stats(self, root_id: str) -> Optional[LineageStats]:
        """获取指定谱系的统计信息"""
        return self._lineage_stats.get(root_id)

    def get_all_lineage_stats(self) -> Dict[str, LineageStats]:
        """获取所有谱系的统计信息"""
        return dict(self._lineage_stats)

    def get_diversity_index(self) -> float:
        """
        计算归一化基因多样性指数 D ∈ [0, 1]。

        D = H / ln(K)，其中 H = -Σ a_i * ln(a_i) 为 Shannon 熵。
        D=0 表示完全坍缩（所有记录属于同一谱系）。
        D=1 表示完美均匀分布。
        """
        K = len(self._roots)
        if K <= 1:
            return 0.0 if K <= 1 else 1.0

        shares = [s.share for s in self._lineage_stats.values()]
        H = 0.0
        for a in shares:
            if a > 0:
                H -= a * math.log(a)

        max_H = math.log(K)
        if max_H < 1e-9:
            return 0.0
        return H / max_H

    def get_lineage_depth(self, record_id: str) -> int:
        """获取记录在谱系中的深度"""
        return self._depth_of.get(record_id, 0)

    def get_lineage_records(self, root_id: str) -> List[SuccessRecord]:
        """获取指定谱系的所有记录"""
        result = []
        for record_id, rid in self._root_of.items():
            if rid == root_id and record_id in self._record_map:
                result.append(self._record_map[record_id])
        return result

    @property
    def roots(self) -> List[str]:
        return list(self._roots)

    @property
    def num_lineages(self) -> int:
        return len(self._roots)

    def _compute_lineage_stats(self, db: SuccessDB) -> None:
        """计算各谱系的统计信息"""
        N = len(self._record_map)
        if N == 0:
            return

        # 按谱系分组
        lineage_records: Dict[str, List[SuccessRecord]] = {}
        for record_id, root_id in self._root_of.items():
            r = self._record_map.get(record_id)
            if r:
                lineage_records.setdefault(root_id, []).append(r)

        for root_id, records in lineage_records.items():
            sorted_by_perf = sorted(records, key=lambda r: r.gen_time)
            best = sorted_by_perf[0]
            max_depth = max(self._depth_of.get(r.id, 0) for r in records)

            self._lineage_stats[root_id] = LineageStats(
                root_id=root_id,
                size=len(records),
                best_gen_time=best.gen_time,
                best_speedup=best.speedup,
                best_record_id=best.id,
                max_depth=max_depth,
                share=len(records) / max(N, 1),
                total_selections=sum(r.selection_count for r in records),
                recent_improvement=self._compute_recent_improvement(records),
            )

    @staticmethod
    def _compute_recent_improvement(records: List[SuccessRecord]) -> float:
        """
        计算谱系的近期改善率。

        比较最新一代和前一代的最优性能差异。
        """
        if len(records) < 2:
            return 0.0

        max_gen = max(r.generation for r in records)
        if max_gen == 0:
            return 0.0

        current_gen = [r for r in records if r.generation == max_gen]
        prev_gen = [r for r in records if r.generation == max_gen - 1]

        if not current_gen or not prev_gen:
            return 0.0

        best_current = min(r.gen_time for r in current_gen)
        best_prev = min(r.gen_time for r in prev_gen)

        if best_prev < 1e-9:
            return 0.0
        return (best_prev - best_current) / best_prev
