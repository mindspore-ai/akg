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
Task Generator for Adaptive Search

生成新的 LangGraphTask，复用现有的灵感采样、meta_prompts 和 handwrite 机制。
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.utils.evolve.evolution_core import sample_inspirations
from akg_agents.op.utils.evolve.evolution_utils import load_meta_prompts
from akg_agents.op.utils.handwrite_loader import HandwriteLoader, HandwriteSampler
from akg_agents.op.adaptive_search.success_db import SuccessDB, SuccessRecord

logger = logging.getLogger(__name__)


@dataclass
class TaskGeneratorConfig:
    """任务生成器配置"""
    inspiration_sample_num: int = 3      # 灵感采样数量（不含父代）
    use_tiered_sampling: bool = True     # 是否使用层次化采样
    handwrite_sample_num: int = 2        # 手写建议采样数量
    handwrite_decay_rate: float = 2.0    # 手写建议衰减率
    meta_prompts_per_task: int = 1       # 每个任务的 meta_prompts 数量


class TaskGenerator:
    """
    任务生成器
    
    负责生成新的 LangGraphTask，包括初始任务和进化任务。
    复用现有的灵感采样、meta_prompts 和 handwrite 机制。
    """
    
    def __init__(self,
                 op_name: str,
                 task_desc: str,
                 dsl: str,
                 framework: str,
                 backend: str,
                 arch: str,
                 config: Dict[str, Any],
                 db: SuccessDB,
                 generator_config: Optional[TaskGeneratorConfig] = None):
        """
        初始化任务生成器
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述
            dsl: DSL 类型
            framework: 框架
            backend: 后端
            arch: 架构
            config: 全局配置
            db: 成功任务数据库
            generator_config: 生成器配置
        """
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.config = config
        self.db = db
        self.gen_config = generator_config or TaskGeneratorConfig()
        
        self._task_counter = 0
        self._handwrite_loader: Optional[HandwriteLoader] = None
        self._handwrite_sampler: Optional[HandwriteSampler] = None
        self._handwrite_initialized = False
        
        logger.info(f"TaskGenerator initialized for {op_name}")
    
    async def _ensure_handwrite_initialized(self) -> None:
        """确保 handwrite 组件已初始化"""
        if self._handwrite_initialized:
            return
        
        try:
            self._handwrite_loader = HandwriteLoader(
                dsl=self.dsl,
                op_name=self.op_name,
                task_desc=self.task_desc,
                config=self.config,
                arch=self.arch,
                backend=self.backend,
                rag=self.config.get('rag', False)  # 从 config 读取 rag
            )
            await self._handwrite_loader.select_relevant_pairs()
            
            self._handwrite_sampler = HandwriteSampler(
                loader=self._handwrite_loader,
                sample_num=self.gen_config.handwrite_sample_num,
                decay_rate=self.gen_config.handwrite_decay_rate
            )
            
            self._handwrite_initialized = True
            logger.info("Handwrite components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize handwrite components: {e}")
            self._handwrite_initialized = True  # 标记为已初始化，避免重复尝试
    
    def _generate_task_id(self, prefix: str = "task") -> str:
        """生成唯一任务 ID"""
        self._task_counter += 1
        return f"{prefix}_{self._task_counter}"
    
    def _get_meta_prompts(self) -> str:
        """获取 meta_prompts"""
        try:
            prompts = load_meta_prompts(self.dsl, self.gen_config.meta_prompts_per_task)
            return prompts[0] if prompts else ""
        except Exception as e:
            logger.warning(f"Failed to load meta_prompts: {e}")
            return ""
    
    def _get_handwrite_suggestions(self) -> List[Dict[str, Any]]:
        """获取 handwrite 建议"""
        if self._handwrite_sampler:
            try:
                return self._handwrite_sampler.sample()
            except Exception as e:
                logger.warning(f"Failed to sample handwrite suggestions: {e}")
        return []
    
    def _prepare_inspirations(self,
                              parent: Optional[SuccessRecord] = None,
                              exclude_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        准备灵感列表
        
        Args:
            parent: 父代记录（可选）
            exclude_ids: 要排除的记录 ID 列表
            
        Returns:
            List[Dict[str, Any]]: 灵感列表
        """
        inspirations = []
        exclude_ids = exclude_ids or []
        
        # 如果有父代，将其作为第一个灵感
        if parent:
            parent_inspiration = parent.to_inspiration()
            parent_inspiration['is_parent'] = True
            inspirations.append(parent_inspiration)
            exclude_ids.append(parent.id)
        
        # 从 DB 中采样其他灵感
        if not self.db.is_empty():
            all_records = self.db.get_all_sorted_by_performance()
            
            # 过滤掉要排除的记录
            available_records = [r for r in all_records if r.id not in exclude_ids]
            
            if available_records:
                # 将记录转换为 sample_inspirations 所需的格式
                implementations = []
                for record in available_records:
                    impl = {
                        'id': record.id,
                        'sketch': record.sketch,
                        'impl_code': record.impl_code,
                        'profile': record.profile,
                        'gen_time': record.gen_time,
                        'strategy_mode': 'adaptive_search'
                    }
                    implementations.append(impl)
                
                # 使用现有的层次化采样函数
                sampled = sample_inspirations(
                    implementations=implementations,
                    sample_num=self.gen_config.inspiration_sample_num,
                    use_tiered_sampling=self.gen_config.use_tiered_sampling
                )
                
                inspirations.extend(sampled)
        
        logger.debug(f"Prepared {len(inspirations)} inspirations (parent={parent is not None})")
        return inspirations
    
    async def generate_initial_task(self) -> LangGraphTask:
        """
        生成初始任务（无父代，无灵感）
        
        Returns:
            LangGraphTask: 新任务
        """
        await self._ensure_handwrite_initialized()
        
        task_id = self._generate_task_id("init")
        meta_prompts = self._get_meta_prompts()
        handwrite_suggestions = self._get_handwrite_suggestions()
        
        # 从 config 中获取 user_requirements（来自 ReAct 多轮对话）
        user_requirements = self.config.get("user_requirements", "")
        task = LangGraphTask(
            op_name=self.op_name,
            task_desc=self.task_desc,
            task_id=task_id,
            backend=self.backend,
            arch=self.arch,
            dsl=self.dsl,
            config=self.config,
            framework=self.framework,
            task_type="profile",
            workflow="default_workflow",
            inspirations=[],  # 初始任务无灵感
            meta_prompts=meta_prompts,
            handwrite_suggestions=handwrite_suggestions,
            user_requirements=user_requirements,  # 用户额外需求
        )
        
        logger.info(f"Generated initial task: {task_id}")
        return task
    
    async def generate_evolved_task(self,
                                    parent: SuccessRecord,
                                    generation: int) -> LangGraphTask:
        """
        生成进化任务（有父代，有灵感）
        
        Args:
            parent: 父代记录
            generation: 新任务的代数
            
        Returns:
            LangGraphTask: 新任务
        """
        await self._ensure_handwrite_initialized()
        
        task_id = self._generate_task_id(f"gen{generation}")
        meta_prompts = self._get_meta_prompts()
        handwrite_suggestions = self._get_handwrite_suggestions()
        inspirations = self._prepare_inspirations(parent=parent)
        
        # 从 config 中获取 user_requirements（来自 ReAct 多轮对话）
        user_requirements = self.config.get("user_requirements", "")
        task = LangGraphTask(
            op_name=self.op_name,
            task_desc=self.task_desc,
            task_id=task_id,
            backend=self.backend,
            arch=self.arch,
            dsl=self.dsl,
            config=self.config,
            framework=self.framework,
            task_type="profile",
            workflow="default_workflow",
            inspirations=inspirations,
            meta_prompts=meta_prompts,
            handwrite_suggestions=handwrite_suggestions,
            user_requirements=user_requirements,  # 用户额外需求
        )
        
        logger.info(f"Generated evolved task: {task_id} (parent={parent.id[:16]}, gen={generation})")
        return task
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        return {
            "total_generated": self._task_counter,
            "handwrite_initialized": self._handwrite_initialized,
            "db_size": self.db.size()
        }

