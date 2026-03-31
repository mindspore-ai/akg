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

"""工作流日志记录器

统一的日志记录类，用于记录 workflow 中各节点产生的中间文件。
每个 workflow 实例持有一个独立的 WorkflowLogger，互不干扰。

Step 编号规则：
- log_record()  → append 到内部记录列表，step = len(列表)，自动自增
- write_record() → 在当前 step 写文件，不 append，不自增
- 只有推进 state step_count 的节点（kernel_gen、verifier 等）调 log_record
- 不推进 step_count 的节点（conductor）调 write_record
- 这样 get_step_count() == state step_count，天然对齐

文件命名规则：
    Iteration{task_id}_Step{NN:02d}_{category}_{record_name}_{param_name}.txt

目录结构：
    {log_dir}/{category}/                     ← 主目录
    {log_dir}/{category}/{subdirectory}/      ← 可选子目录（如 conductor/）
    {log_dir}/{category}/recorder/            ← 合并文件默认目录
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class WorkflowLogger:
    """工作流日志记录器

    每个 workflow 实例持有独立的 WorkflowLogger，
    通过 _step_records 列表长度作为 step 编号，与 state step_count 自然对齐。
    """

    def __init__(self, log_dir: str, category: str, task_id: str):
        self.log_dir = log_dir
        self.category = category
        self.task_id = task_id
        self._step_records: List[dict] = []

        logger.info(
            f"[WorkflowLogger] 初始化: log_dir={log_dir}, "
            f"category={category}, task_id={task_id}"
        )

    # ==================== 核心接口 ====================

    def log_record(self, record_name: str, params: list,
                   subdirectory: str = None) -> int:
        """记录一个工作流步骤，append 后写文件。step 自增。

        用于推进 step_count 的节点（kernel_gen、verifier、coder 等）。

        Returns:
            自增后的步骤号
        """
        self._step_records.append({"name": record_name})
        step = len(self._step_records)
        self._write_files(record_name, params, subdirectory, step)
        return step

    def write_record(self, record_name: str, params: list,
                     subdirectory: str = None) -> int:
        """在当前 step 写文件，不 append，不自增。

        用于不推进 step_count 的节点（conductor 等），
        与上一个 log_record 共享同一 step 编号。

        Returns:
            当前步骤号（未自增）
        """
        step = len(self._step_records)
        self._write_files(record_name, params, subdirectory, step)
        return step

    def _write_files(self, record_name: str, params: list,
                     subdirectory: str, step: int) -> None:
        """内部：将参数写入磁盘文件"""
        if not self.log_dir:
            logger.debug(
                f"[WorkflowLogger] _write_files({record_name}): "
                f"log_dir 未设置，跳过文件保存"
            )
            return

        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.category)
        if subdirectory:
            target_dir = os.path.join(target_dir, subdirectory)
        os.makedirs(target_dir, exist_ok=True)

        base_name = (
            f"Iteration{self.task_id}_Step{step:02d}_"
            f"{self.category}_{record_name}_"
        )

        for param_name, content in params:
            if content is None:
                continue
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))

    def log_merged_record(self, record_name: str, params: list,
                          filename_suffix: str = "_parsed.txt",
                          subdirectory: str = "recorder") -> None:
        """将多个参数合并保存到单个文件（不自增）"""
        if not self.log_dir or not params:
            return

        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.category, subdirectory)
        os.makedirs(target_dir, exist_ok=True)

        step = len(self._step_records)
        base_name = (
            f"Iteration{self.task_id}_Step{step:02d}_"
            f"{self.category}_{record_name}{filename_suffix}"
        )
        file_path = os.path.join(target_dir, base_name)

        content_parts = []
        for section_name, content in params:
            if content:
                content_parts.append(f"=== {section_name} ===\n{content}")

        if content_parts:
            combined_content = "\n\n\n".join(content_parts)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(combined_content)

    # ==================== 查询接口 ====================

    def get_step_count(self) -> int:
        """获取当前步骤计数（= len(_step_records)，与 state step_count 对齐）"""
        return len(self._step_records)

    def get_log_task_id(self) -> str:
        """获取日志文件名中使用的 task_id"""
        return self.task_id
