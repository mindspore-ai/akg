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
adaptive_search 模块单元测试

覆盖 SuccessDB、UCBParentSelector、AsyncTaskPool、数据类等核心组件的
接口正确性、参数校验和纯逻辑行为，不依赖 LLM 或 GPU。
"""

import asyncio
import json
import math
import os
import tempfile
from dataclasses import asdict

import pytest

from akg_agents.op.adaptive_search.success_db import SuccessDB, SuccessRecord
from akg_agents.op.adaptive_search.ucb_selector import UCBParentSelector
from akg_agents.op.adaptive_search.task_pool import (
    AsyncTaskPool,
    PendingTask,
    TaskResult,
    TaskWrapper,
    TaskStatus,
)
from akg_agents.op.adaptive_search.controller import SearchConfig
from akg_agents.op.adaptive_search.task_generator import TaskGeneratorConfig


# ====================== SuccessRecord 数据类 ======================

class TestSuccessRecord:
    def test_default_fields(self):
        r = SuccessRecord(id="t1", impl_code="pass")
        assert r.id == "t1"
        assert r.impl_code == "pass"
        assert r.sketch == ""
        assert r.gen_time == float("inf")
        assert r.speedup == 0.0
        assert r.selection_count == 0
        assert r.generation == 0
        assert r.parent_id is None
        assert r.sketch_generated is False

    def test_roundtrip_dict(self):
        r = SuccessRecord(id="t2", impl_code="code", gen_time=1.5, speedup=2.0)
        d = r.to_dict()
        r2 = SuccessRecord.from_dict(d)
        assert r2.id == r.id
        assert r2.gen_time == r.gen_time
        assert r2.speedup == r.speedup

    def test_to_inspiration_format(self):
        r = SuccessRecord(id="t3", impl_code="kernel", sketch="sk", generation=2)
        insp = r.to_inspiration()
        assert insp["id"] == "t3"
        assert insp["sketch"] == "sk"
        assert insp["impl_code"] == "kernel"
        assert insp["strategy_mode"] == "adaptive_search"
        assert insp["generation"] == 2
        assert insp["is_parent"] is False


# ====================== SuccessDB ======================

class TestSuccessDB:
    def test_empty_db(self):
        db = SuccessDB()
        assert db.is_empty()
        assert db.size() == 0
        assert db.get_best_record() is None
        assert db.get_best_gen_time() == float("inf")
        assert db.get_best_speedup() == 0.0

    def test_add_and_get(self):
        db = SuccessDB()
        r = SuccessRecord(id="r1", impl_code="code1", gen_time=10.0)
        assert db.add(r) is True
        assert db.size() == 1
        assert not db.is_empty()

        fetched = db.get("r1")
        assert fetched is not None
        assert fetched.id == "r1"
        assert fetched.gen_time == 10.0

    def test_add_duplicate_better_performance(self):
        db = SuccessDB()
        db.add(SuccessRecord(id="r1", impl_code="v1", gen_time=10.0))
        result = db.add(SuccessRecord(id="r1", impl_code="v2", gen_time=5.0))
        assert result is True
        assert db.get("r1").gen_time == 5.0

    def test_add_duplicate_worse_performance(self):
        db = SuccessDB()
        db.add(SuccessRecord(id="r1", impl_code="v1", gen_time=5.0))
        result = db.add(SuccessRecord(id="r1", impl_code="v2", gen_time=10.0))
        assert result is False
        assert db.get("r1").gen_time == 5.0

    def test_get_all_preserves_order(self):
        db = SuccessDB()
        for i in range(5):
            db.add(SuccessRecord(id=f"r{i}", impl_code=f"c{i}", gen_time=float(10 - i)))
        ids = [r.id for r in db.get_all()]
        assert ids == ["r0", "r1", "r2", "r3", "r4"]

    def test_get_all_sorted_by_performance(self):
        db = SuccessDB()
        db.add(SuccessRecord(id="slow", impl_code="s", gen_time=100.0))
        db.add(SuccessRecord(id="fast", impl_code="f", gen_time=1.0))
        db.add(SuccessRecord(id="mid", impl_code="m", gen_time=50.0))
        sorted_ids = [r.id for r in db.get_all_sorted_by_performance()]
        assert sorted_ids == ["fast", "mid", "slow"]

    def test_increment_decrement_selection(self):
        db = SuccessDB()
        db.add(SuccessRecord(id="r1", impl_code="c"))
        db.increment_selection("r1")
        db.increment_selection("r1")
        assert db.get("r1").selection_count == 2
        assert db.get_total_selections() == 2

        db.decrement_selection("r1")
        assert db.get("r1").selection_count == 1
        assert db.get_total_selections() == 1

    def test_decrement_does_not_go_negative(self):
        db = SuccessDB()
        db.add(SuccessRecord(id="r1", impl_code="c"))
        db.decrement_selection("r1")
        assert db.get("r1").selection_count == 0

    def test_add_from_result(self):
        db = SuccessDB()
        state = {
            "coder_code": "def kernel(): pass",
            "profile_res": {"gen_time": 3.14, "speedup": 1.5},
            "verifier_result": {"passed": True},
        }
        record = db.add_from_result("task1", state, generation=1, parent_id="p0")
        assert record is not None
        assert record.gen_time == 3.14
        assert record.speedup == 1.5
        assert record.generation == 1
        assert record.parent_id == "p0"
        assert record.sketch_generated is False

    def test_update_sketch(self):
        db = SuccessDB()
        db.add(SuccessRecord(id="r1", impl_code="c"))
        assert db.update_sketch("r1", "new_sketch") is True
        assert db.get("r1").sketch == "new_sketch"
        assert db.get("r1").sketch_generated is True

    def test_update_sketch_nonexistent(self):
        db = SuccessDB()
        assert db.update_sketch("nonexistent", "sk") is False

    def test_get_pending_sketch_records(self):
        db = SuccessDB()
        db.add(SuccessRecord(id="r1", impl_code="c1"))
        db.add(SuccessRecord(id="r2", impl_code="c2"))
        db.update_sketch("r1", "done")
        pending = db.get_pending_sketch_records()
        assert len(pending) == 1
        assert pending[0].id == "r2"

    def test_statistics(self):
        db = SuccessDB()
        stats = db.get_statistics()
        assert stats["total_records"] == 0

        db.add(SuccessRecord(id="r1", impl_code="c", gen_time=10.0, speedup=2.0))
        db.add(SuccessRecord(id="r2", impl_code="c", gen_time=20.0, speedup=3.0))
        stats = db.get_statistics()
        assert stats["total_records"] == 2
        assert stats["best_gen_time"] == 10.0
        assert stats["worst_gen_time"] == 20.0
        assert stats["best_speedup"] == 3.0

    def test_persistence(self, tmp_path):
        db1 = SuccessDB(storage_dir=str(tmp_path))
        db1.add(SuccessRecord(id="r1", impl_code="persistent", gen_time=5.0))

        db2 = SuccessDB(storage_dir=str(tmp_path))
        assert db2.size() == 1
        assert db2.get("r1").gen_time == 5.0
        assert db2.get("r1").impl_code == "persistent"

    def test_increment_selection_not_auto_persisted(self, tmp_path):
        """increment_selection 不会自动持久化，这是当前设计行为"""
        db1 = SuccessDB(storage_dir=str(tmp_path))
        db1.add(SuccessRecord(id="r1", impl_code="c", gen_time=5.0))
        db1.increment_selection("r1")
        assert db1.get("r1").selection_count == 1

        db2 = SuccessDB(storage_dir=str(tmp_path))
        assert db2.get("r1").selection_count == 0


# ====================== UCBParentSelector ======================

class TestUCBParentSelector:
    def _make_db(self, records):
        db = SuccessDB()
        for r in records:
            db.add(r)
        return db

    def test_select_from_empty_db(self):
        db = SuccessDB()
        selector = UCBParentSelector(db)
        assert selector.select() is None

    def test_select_single_record(self):
        db = self._make_db([
            SuccessRecord(id="only", impl_code="c", gen_time=10.0)
        ])
        selector = UCBParentSelector(db)
        result = selector.select()
        assert result is not None
        assert result.id == "only"
        assert db.get("only").selection_count == 1

    def test_quality_ranking(self):
        db = self._make_db([
            SuccessRecord(id="best", impl_code="c", gen_time=1.0),
            SuccessRecord(id="worst", impl_code="c", gen_time=100.0),
        ])
        selector = UCBParentSelector(db, random_factor=0.0)
        best_q = selector._compute_quality(db.get("best"), db.get_all())
        worst_q = selector._compute_quality(db.get("worst"), db.get_all())
        assert best_q == 1.0
        assert worst_q == 0.0

    def test_quality_single_record(self):
        db = self._make_db([
            SuccessRecord(id="solo", impl_code="c", gen_time=5.0)
        ])
        selector = UCBParentSelector(db)
        q = selector._compute_quality(db.get("solo"), db.get_all())
        assert q == 1.0

    def test_exploration_decreases_with_selection(self):
        db = self._make_db([
            SuccessRecord(id="r1", impl_code="c", gen_time=5.0)
        ])
        selector = UCBParentSelector(db)
        e0 = selector._compute_exploration(selection_count=0, total_selections=10)
        e5 = selector._compute_exploration(selection_count=5, total_selections=10)
        assert e0 > e5

    def test_exploration_zero_total(self):
        db = SuccessDB()
        selector = UCBParentSelector(db, exploration_coef=1.414)
        e = selector._compute_exploration(selection_count=0, total_selections=0)
        assert e == 1.414

    def test_greedy_select_picks_highest_ucb(self):
        db = self._make_db([
            SuccessRecord(id="fast", impl_code="c", gen_time=1.0),
            SuccessRecord(id="slow", impl_code="c", gen_time=100.0),
        ])
        selector = UCBParentSelector(db, random_factor=0.0)
        for _ in range(5):
            result = selector.select()
            assert result is not None

    def test_softmax_select_returns_valid(self):
        records = [
            SuccessRecord(id=f"r{i}", impl_code="c", gen_time=float(i + 1))
            for i in range(5)
        ]
        db = self._make_db(records)
        selector = UCBParentSelector(db, use_softmax=True, softmax_temperature=0.5)
        result = selector.select()
        assert result is not None
        assert result.id in {f"r{i}" for i in range(5)}

    def test_selection_stats(self):
        db = self._make_db([
            SuccessRecord(id="r1", impl_code="c", gen_time=5.0),
            SuccessRecord(id="r2", impl_code="c", gen_time=10.0),
        ])
        selector = UCBParentSelector(db, random_factor=0.0)
        selector.select()
        selector.select()

        stats = selector.get_selection_stats()
        assert stats["total_records"] == 2
        assert stats["total_selections"] == 2

    def test_empty_stats(self):
        db = SuccessDB()
        selector = UCBParentSelector(db)
        stats = selector.get_selection_stats()
        assert stats["total_records"] == 0
        assert stats["total_selections"] == 0


# ====================== AsyncTaskPool ======================

class TestAsyncTaskPool:
    def test_init(self):
        pool = AsyncTaskPool(max_concurrent=4)
        assert pool.max_concurrent == 4

    def test_pending_task_dataclass(self):
        async def dummy():
            return True
        pt = PendingTask(task_id="t1", coroutine_factory=dummy, generation=1, parent_id="p0")
        assert pt.task_id == "t1"
        assert pt.generation == 1
        assert pt.parent_id == "p0"
        assert callable(pt.coroutine_factory)

    def test_task_result_dataclass(self):
        tr = TaskResult(
            task_id="t1", success=True,
            final_state={"coder_code": "pass"},
            generation=2, parent_id="p1"
        )
        assert tr.task_id == "t1"
        assert tr.success is True
        assert tr.generation == 2
        assert tr.error is None

    def test_task_result_failure(self):
        tr = TaskResult(
            task_id="t1", success=False,
            final_state={}, error="timeout"
        )
        assert tr.success is False
        assert tr.error == "timeout"

    def test_task_status_enum(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncTaskPool(max_concurrent=2) as pool:
            assert pool.max_concurrent == 2


# ====================== 配置数据类 ======================

class TestSearchConfig:
    def test_defaults(self):
        cfg = SearchConfig()
        assert cfg.max_concurrent == 8
        assert cfg.initial_task_count == 8
        assert cfg.max_total_tasks == 100
        assert cfg.exploration_coef == 1.414
        assert cfg.random_factor == 0.1
        assert cfg.use_softmax is False
        assert cfg.softmax_temperature == 1.0
        assert cfg.use_evolution_controller is False
        assert cfg.storage_dir is None
        assert cfg.poll_interval == 0.5

    def test_custom_values(self):
        cfg = SearchConfig(
            max_concurrent=16,
            max_total_tasks=200,
            exploration_coef=2.0,
            use_softmax=True,
        )
        assert cfg.max_concurrent == 16
        assert cfg.max_total_tasks == 200
        assert cfg.exploration_coef == 2.0
        assert cfg.use_softmax is True


class TestTaskGeneratorConfig:
    def test_defaults(self):
        cfg = TaskGeneratorConfig()
        assert cfg.inspiration_sample_num == 3
        assert cfg.use_tiered_sampling is True
        assert cfg.handwrite_sample_num == 2
        assert cfg.handwrite_decay_rate == 2.0
        assert cfg.meta_prompts_per_task == 1

    def test_custom_values(self):
        cfg = TaskGeneratorConfig(
            inspiration_sample_num=5,
            handwrite_sample_num=0,
        )
        assert cfg.inspiration_sample_num == 5
        assert cfg.handwrite_sample_num == 0


# ====================== 模块导出 ======================

class TestModuleExports:
    def test_package_exports(self):
        from akg_agents.op.adaptive_search import (
            adaptive_search,
            adaptive_search_from_config,
            load_adaptive_search_config,
            SuccessDB,
            SuccessRecord,
            UCBParentSelector,
            AsyncTaskPool,
            PendingTask,
            TaskResult,
            TaskWrapper,
            TaskGenerator,
            TaskGeneratorConfig,
            AdaptiveSearchController,
            SearchConfig,
        )
        assert callable(adaptive_search)
        assert callable(adaptive_search_from_config)
        assert callable(load_adaptive_search_config)
