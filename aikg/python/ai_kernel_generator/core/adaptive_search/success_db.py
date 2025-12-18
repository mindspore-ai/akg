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
Success Database for Adaptive Search

存储成功任务的数据库，支持 UCB 选择所需的统计信息。
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SuccessRecord:
    """成功任务记录"""
    id: str                                     # 任务唯一标识
    impl_code: str                              # 实现代码（最终 coder_code）
    sketch: str = ""                            # 设计草图（根据最终代码重新生成）
    profile: Dict[str, Any] = field(default_factory=dict)  # 性能数据
    gen_time: float = float('inf')              # 生成时间 (ms)
    speedup: float = 0.0                        # 加速比
    selection_count: int = 0                    # 被选择次数
    generation: int = 0                         # 代数（0=初始任务）
    parent_id: Optional[str] = None             # 父代 ID
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    meta_info: Dict[str, Any] = field(default_factory=dict)  # 其他元信息
    sketch_generated: bool = False              # sketch 是否已根据最终代码重新生成
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessRecord":
        """从字典创建"""
        return cls(**data)
    
    def to_inspiration(self) -> Dict[str, Any]:
        """转换为灵感格式（用于 LangGraphTask）"""
        return {
            'id': self.id,
            'sketch': self.sketch,
            'impl_code': self.impl_code,
            'profile': self.profile,
            'strategy_mode': 'adaptive_search',
            'generation': self.generation,
            'is_parent': False
        }


class SuccessDB:
    """
    成功任务数据库
    
    只存储成功验证并测试性能的任务，失败任务直接丢弃。
    支持 UCB 选择所需的统计信息（选择次数、性能数据等）。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化 SuccessDB
        
        Args:
            storage_dir: 可选的存储目录，用于持久化
        """
        self._records: Dict[str, SuccessRecord] = {}
        self._record_ids: List[str] = []  # 有序列表，保持插入顺序
        self._total_selections: int = 0   # 全局总选择次数
        self._storage_dir = storage_dir
        
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            self._db_path = os.path.join(storage_dir, "success_db.json")
            self._load_from_disk()
        
        logger.info(f"SuccessDB initialized with {len(self._records)} records")
    
    def add(self, record: SuccessRecord) -> bool:
        """
        添加成功任务记录
        
        Args:
            record: 成功任务记录
            
        Returns:
            bool: 是否成功添加（如果 ID 已存在且性能更差，返回 False）
        """
        if record.id in self._records:
            existing = self._records[record.id]
            # 如果新记录性能更好，更新
            if record.gen_time < existing.gen_time:
                logger.info(f"Updating record {record.id}: gen_time {existing.gen_time:.4f} -> {record.gen_time:.4f}")
                record.selection_count = existing.selection_count  # 保留选择次数
                self._records[record.id] = record
                self._save_to_disk()
                return True
            else:
                logger.debug(f"Record {record.id} already exists with better performance, skipping")
                return False
        
        self._records[record.id] = record
        self._record_ids.append(record.id)
        logger.info(f"Added record {record.id} to SuccessDB. gen_time={record.gen_time:.4f}ms, speedup={record.speedup:.2f}x. DB size: {len(self._records)}")
        self._save_to_disk()
        return True
    
    def add_from_result(self, 
                        task_id: str, 
                        final_state: Dict[str, Any],
                        generation: int = 0,
                        parent_id: Optional[str] = None) -> Optional[SuccessRecord]:
        """
        从任务结果创建并添加记录
        
        注意：sketch 字段暂时为空，需要后续调用 update_sketch 方法更新。
        因为 coder 经过多次修改后代码会和初始 designer_code 不一样，
        需要根据最终 impl_code 重新生成 sketch。
        
        Args:
            task_id: 任务 ID
            final_state: LangGraphTask 的最终状态
            generation: 代数
            parent_id: 父代 ID
            
        Returns:
            SuccessRecord: 成功添加的记录，或 None
        """
        profile_res = final_state.get("profile_res", {})
        
        record = SuccessRecord(
            id=task_id,
            impl_code=final_state.get("coder_code", ""),
            sketch="",  # 暂时为空，后续异步生成
            profile=profile_res,
            gen_time=profile_res.get("gen_time", float('inf')),
            speedup=profile_res.get("speedup", 0.0),
            generation=generation,
            parent_id=parent_id,
            sketch_generated=False,  # 标记 sketch 未生成
            meta_info={
                "task_info": final_state,  # 保存完整 task_info 用于后续生成 sketch
                "verifier_result": final_state.get("verifier_result", {}),
                "inspirations_used": final_state.get("inspirations", []),
                "meta_prompts_used": final_state.get("meta_prompts", "")
            }
        )
        
        if self.add(record):
            return record
        return None
    
    def update_sketch(self, record_id: str, sketch: str) -> bool:
        """
        更新记录的 sketch（根据最终代码重新生成后调用）
        
        Args:
            record_id: 记录 ID
            sketch: 新生成的 sketch
            
        Returns:
            bool: 是否更新成功
        """
        if record_id not in self._records:
            logger.warning(f"Record {record_id} not found, cannot update sketch")
            return False
        
        self._records[record_id].sketch = sketch
        self._records[record_id].sketch_generated = True
        self._save_to_disk()
        logger.debug(f"Updated sketch for record {record_id}")
        return True
    
    def get_pending_sketch_records(self) -> List[SuccessRecord]:
        """获取需要生成 sketch 的记录列表"""
        return [r for r in self._records.values() if not r.sketch_generated and r.impl_code]
    
    def get(self, record_id: str) -> Optional[SuccessRecord]:
        """获取指定记录"""
        return self._records.get(record_id)
    
    def get_all(self) -> List[SuccessRecord]:
        """获取所有记录（按插入顺序）"""
        return [self._records[rid] for rid in self._record_ids if rid in self._records]
    
    def get_all_sorted_by_performance(self) -> List[SuccessRecord]:
        """获取所有记录（按性能排序，gen_time 小的在前）"""
        return sorted(self._records.values(), key=lambda r: r.gen_time)
    
    def increment_selection(self, record_id: str) -> None:
        """增加选择次数"""
        if record_id in self._records:
            self._records[record_id].selection_count += 1
            self._total_selections += 1
            logger.debug(f"Record {record_id} selection count: {self._records[record_id].selection_count}")
    
    def get_total_selections(self) -> int:
        """获取全局总选择次数"""
        return self._total_selections
    
    def is_empty(self) -> bool:
        """检查是否为空"""
        return len(self._records) == 0
    
    def size(self) -> int:
        """获取记录数量"""
        return len(self._records)
    
    def get_best_record(self) -> Optional[SuccessRecord]:
        """获取性能最好的记录"""
        if not self._records:
            return None
        return min(self._records.values(), key=lambda r: r.gen_time)
    
    def get_best_gen_time(self) -> float:
        """获取最佳 gen_time"""
        best = self.get_best_record()
        return best.gen_time if best else float('inf')
    
    def get_best_speedup(self) -> float:
        """获取最佳 speedup"""
        if not self._records:
            return 0.0
        return max(r.speedup for r in self._records.values())
    
    def get_all_as_inspirations(self) -> List[Dict[str, Any]]:
        """获取所有记录作为灵感格式"""
        return [r.to_inspiration() for r in self.get_all_sorted_by_performance()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._records:
            return {
                "total_records": 0,
                "total_selections": 0
            }
        
        gen_times = [r.gen_time for r in self._records.values() if r.gen_time != float('inf')]
        speedups = [r.speedup for r in self._records.values()]
        
        return {
            "total_records": len(self._records),
            "total_selections": self._total_selections,
            "best_gen_time": min(gen_times) if gen_times else None,
            "worst_gen_time": max(gen_times) if gen_times else None,
            "avg_gen_time": sum(gen_times) / len(gen_times) if gen_times else None,
            "best_speedup": max(speedups) if speedups else None,
            "avg_speedup": sum(speedups) / len(speedups) if speedups else None,
            "generations": list(set(r.generation for r in self._records.values()))
        }
    
    def _save_to_disk(self) -> None:
        """保存到磁盘"""
        if not self._storage_dir:
            return
        
        try:
            data = {
                "records": {rid: r.to_dict() for rid, r in self._records.items()},
                "record_ids": self._record_ids,
                "total_selections": self._total_selections
            }
            with open(self._db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save SuccessDB to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        if not self._storage_dir or not os.path.exists(self._db_path):
            return
        
        try:
            with open(self._db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._records = {
                rid: SuccessRecord.from_dict(rdata) 
                for rid, rdata in data.get("records", {}).items()
            }
            self._record_ids = data.get("record_ids", list(self._records.keys()))
            self._total_selections = data.get("total_selections", 0)
            
            logger.info(f"Loaded {len(self._records)} records from disk")
        except Exception as e:
            logger.warning(f"Failed to load SuccessDB from disk: {e}")

