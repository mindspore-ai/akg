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
手写优化建议和实现加载工具
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.agent.selector import Selector
from ai_kernel_generator.database.coder_database import CoderDatabase

logger = logging.getLogger(__name__)

DEFAULT_HANDWRITE_DATABASE_PATH = Path(get_project_root()).parent.parent / "handwrite_database" / "local"

class HandwriteLoader:
    """手写优化建议和实现加载器
    
    支持两种模式：
    1. RAG模式：基于CoderDatabase进行语义检索
    2. 文件系统模式：遍历文件系统直接加载
    """
    
    def __init__(self, 
        dsl: str = "triton_ascend", 
        framework: str = "torch",
        task_desc: str = None, 
        arch: str = None,
        backend: str = None,
        database_path: str = None, 
        config: dict = None,
        op_name: str = None,
        rag: bool = False,
    ):
        """初始化加载器
        
        Args:
            dsl: DSL类型，默认为"triton_ascend"（也支持"triton_cuda"）
            framework: 框架类型，默认为"torch"
            task_desc: 任务描述
            arch: 架构类型
            backend: 后端类型
            database_path: 手写优化建议数据库路径
            config: 配置字典
            op_name: 算子名称（文件系统模式需要）
            rag: 是否使用RAG模式，默认为False（使用文件系统模式）
        """
        self.dsl = dsl
        self.framework = framework
        self.task_desc = task_desc
        self.config = config
        self.arch = arch
        self.backend = backend
        self.op_name = op_name
        self.rag = rag
        self.database_path = database_path or str(DEFAULT_HANDWRITE_DATABASE_PATH)
        self.database = None
        self._all_data_pairs = []
        self._selected_data_pairs = []

        self.load_num = 20

    async def select_relevant_pairs(self) -> None:
        """根据rag参数决定使用RAG模式或文件系统模式
        
        如果rag=True，优先使用RAG模式，失败后自动降级到文件系统模式
        如果rag=False，直接使用文件系统模式
        """
        if not (self.task_desc and self.config):
            raise ValueError("task_desc and config must be provided")

        # 如果rag=False，直接使用文件系统模式
        if not self.rag:
            await self._select_relevant_pairs_filesystem()
            return

        # rag=True时，尝试使用RAG模式
        if not (self.arch and self.backend):
            logger.warning("arch or backend not provided, using filesystem mode")
            await self._select_relevant_pairs_filesystem()
            return
        
        # 尝试创建 CoderDatabase，如果失败则让异常向上传播
        # VectorStore 初始化时会检查依赖，如果缺少会抛出明确的错误信息
        self.database = CoderDatabase(
            database_path=self.database_path,
            config=self.config
        )
        
        try:
            await self.database.auto_update(
                dsl=self.dsl, 
                framework=self.framework, 
                backend=self.backend,
                arch=self.arch, 
                ref_type="docs", 
                update_mode="skip"
            )
            self._selected_data_pairs = await self.database.samples(
                output_content=["name", "framework_code", "impl_code", "improvement_doc"],
                framework_code=self.task_desc, 
                backend=self.backend,
                arch=self.arch, 
                dsl=self.dsl, 
                framework=self.framework,
                sample_num=self.load_num
            )
            
            logger.info(f"RAG mode: Loaded {len(self._selected_data_pairs)} pairs: "
                        f"{', '.join([pair['name'] for pair in self._selected_data_pairs])}")
            # 如果RAG检索没有结果，降级到文件系统模式
            if not self._selected_data_pairs:
                logger.warning("RAG retrieval found no relevant documents, falling back to filesystem mode")
                await self._select_relevant_pairs_filesystem()
                return
        except Exception as e:
            # RAG检索失败，降级到文件系统模式
            logger.warning(f"RAG sampling failed: {e}, falling back to filesystem mode")
            await self._select_relevant_pairs_filesystem()

    def _init_filesystem_mode(self) -> None:
        """初始化文件系统模式的路径"""
        aikg_root = Path(get_project_root()).parent.parent
        self.aikgbench_root = aikg_root / "benchmark" / "aikgbench"
        self.torch_base_dir = self.aikgbench_root
        self.triton_impl_base_dir = self.aikgbench_root / self.dsl / "impl"
        self.triton_docs_base_dir = self.aikgbench_root / self.dsl / "docs"

    def _load_data_pairs(self) -> None:
        """加载所有可用的数据对（torch文件，triton文件，优化建议文件）
        
        遍历 triton_ascend/docs 下的所有.md文件，查找对应的triton实现和torch文件
        """
        if not self.triton_docs_base_dir.exists():
            logger.warning(f"Triton docs directory not found: {self.triton_docs_base_dir}")
            return
        
        if not self.triton_impl_base_dir.exists():
            logger.warning(f"Triton impl directory not found: {self.triton_impl_base_dir}")
            return
        
        if not self.torch_base_dir.exists():
            logger.warning(f"Torch base directory not found: {self.torch_base_dir}")
            return
        
        # 递归遍历docs目录下的所有.md文件
        for improvement_path in self.triton_docs_base_dir.rglob("*.md"):
            # 获取相对路径和文件名
            # 例如: dynamic_shape/reduction/softmax_001.md
            rel_path = improvement_path.relative_to(self.triton_docs_base_dir)
            file_stem = improvement_path.stem  # softmax_001
            
            # 构建对应的triton实现文件路径
            # 例如: impl/dynamic_shape/reduction/softmax_001.py
            impl_path = self.triton_impl_base_dir / rel_path.parent / f"{file_stem}.py"
            if not impl_path.exists():
                logger.debug(f"Triton implementation not found for {rel_path}, skipping")
                continue
            
            # 构建对应的torch文件路径
            # 例如: aikgbench/dynamic_shape/reduction/softmax_001.py
            framework_path = self.torch_base_dir / rel_path.parent / f"{file_stem}.py"
            if not framework_path.exists():
                logger.debug(f"Torch task file not found for {rel_path}, skipping")
                continue
            
            # 构建唯一标识符：包含shape_type和category
            # 例如: dynamic_shape/reduction/softmax_001
            unique_name = str(rel_path.with_suffix('')).replace('\\', '/')
            
            # 添加数据对
            data_pair = {
                'name': unique_name,  # 使用完整路径作为唯一标识
                'framework_path': framework_path,
                'impl_path': impl_path,
                'improvement_path': improvement_path,
            }
            self._all_data_pairs.append(data_pair)
            logger.debug(f"Loaded data pair: {unique_name}")
        
        logger.info(f"Loaded {len(self._all_data_pairs)} hand-write data pairs")
        
        # 默认使用所有数据对，筛选需要显式调用select_relevant_pairs()
        self._selected_data_pairs = self._all_data_pairs
    
    def read_pair_content(self, data_pair: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """读取数据对的文件内容
        
        Args:
            data_pair: 数据对字典
            
        Returns:
            包含name, framework_code, impl_code, improvement_doc字典，失败返回None
        """
        try:
            # 如果data_pair中已有impl_code和improvement_doc字段（RAG模式），直接返回
            if 'impl_code' in data_pair and 'improvement_doc' in data_pair:
                return data_pair
            
            # 否则从文件读取（文件系统模式）
            framework_code = data_pair['framework_path'].read_text(encoding='utf-8')
            impl_code = data_pair['impl_path'].read_text(encoding='utf-8')
            improvement_doc = data_pair['improvement_path'].read_text(encoding='utf-8')
            
            return {
                'name': data_pair['name'],
                'framework_code': framework_code,
                'impl_code': impl_code,
                'improvement_doc': improvement_doc
            }
        except Exception as e:
            logger.debug(f"Failed to read pair content for {data_pair['name']}: {e}")
            return None

    def _load_all_candidates(self) -> List[Dict[str, str]]:
        """加载所有候选文档的完整内容，用于LLM筛选
        
        Returns:
            候选文档列表，每个包含name和完整内容
        """
        candidates = []
        for pair in self._all_data_pairs:
            content = self.read_pair_content(pair)
            if content:
                candidates.append(content)
        return candidates

    async def _select_relevant_pairs_filesystem(self) -> None:
        """使用文件系统模式筛选相关的数据对"""
        if not self.op_name or not self.task_desc or not self.config:
            logger.warning("op_name, task_desc, or config not provided, using all pairs in filesystem mode")
            return

        self._init_filesystem_mode()
        self._load_data_pairs()
        
        # 加载所有候选文档
        candidates = self._load_all_candidates()
        
        if not candidates:
            logger.warning("No valid candidates for selection, using all pairs")
            return
        
        logger.info(f"Filesystem mode: Running Selector Agent for {self.op_name}...")
        
        try:
            # 创建Selector Agent
            selector = Selector(
                op_name=self.op_name,
                task_desc=self.task_desc,
                dsl=self.dsl,
                config=self.config
            )
            
            # 调用LLM进行筛选
            selected_names = await selector.run(candidates)
            self._selected_data_pairs = [p for p in self._all_data_pairs if p['name'] in selected_names]
            logger.info(f"Filesystem mode: Selected {len(self._selected_data_pairs)}/{len(self._all_data_pairs)} pairs: "
                        f"{', '.join([pair['name'] for pair in self._selected_data_pairs])}")
        except ImportError as e:
            logger.warning(f"Selector not available: {e}, using all pairs")
            self._selected_data_pairs = self._all_data_pairs
    
    def get_selected_pairs(self) -> List[Dict[str, Any]]:
        """获取筛选后的数据对列表
        
        Returns:
            数据对列表的副本
        """
        return self._selected_data_pairs.copy()


class HandwriteSampler:
    """手写优化建议采样器
    
    支持基于相关性的加权采样，用完后自动重置
    
    采样策略：
    - 文档列表已按相关性排序（由LLM筛选时排序）
    - 使用指数衰减权重：weight(i) = exp(-decay_rate * i / total_count)
    - 索引越小（相关性越高）权重越大，被选中概率越高
    - 索引越大（相关性越低）权重越小，但仍有被选中的机会
    """
    
    def __init__(self, loader: HandwriteLoader, sample_num: int = 2, decay_rate: float = 2.0):
        """初始化采样器
        
        Args:
            loader: HandwriteLoader实例
            sample_num: 每次采样的数量
            decay_rate: 权重衰减率，值越大衰减越快（默认2.0）
                       - 2.0: 第1个文档权重约为最后一个的7.4倍
                       - 3.0: 第1个文档权重约为最后一个的20倍
                       - 1.0: 第1个文档权重约为最后一个的2.7倍
        """
        self.loader = loader
        self.sample_num = sample_num
        self.decay_rate = decay_rate
        
        # 获取可用的数据对（已按相关性排序）
        self._available_pairs = self.loader.get_selected_pairs()
        self._total_count = len(self._available_pairs)
        
        # 记录已使用的索引
        self._used_indices = set()
        
        # 预计算权重
        self._weights = self._compute_weights()
        
        if self._total_count == 0:
            logger.warning("No hand-write data pairs available")
        else:
            logger.info(f"HandwriteSampler initialized with {self._total_count} pairs, "
                       f"sample_num={sample_num}, decay_rate={decay_rate}")
            if self._total_count > 1:
                weight_ratio = self._weights[0] / self._weights[-1]
                logger.info(f"Weight ratio (first/last): {weight_ratio:.2f}x")
    
    def _compute_weights(self) -> np.ndarray:
        """计算每个索引的采样权重
        
        使用指数衰减：weight(i) = exp(-decay_rate * i / total_count)
        
        Returns:
            权重数组，已归一化为概率分布
        """
        if self._total_count == 0:
            return np.array([])
        
        # 计算指数衰减权重
        indices = np.arange(self._total_count)
        weights = np.exp(-self.decay_rate * indices / self._total_count)
        
        # 归一化为概率分布
        weights = weights / weights.sum()
        
        return weights
    
    def sample(self) -> List[Dict[str, str]]:
        """基于相关性权重采样建议
        
        使用加权随机采样，相关性高的文档被选中概率更大
        
        Returns:
            采样的建议列表，每个包含name, task_desc, impl_code, improvement及其文件路径等
        """
        if self._total_count == 0:
            return []
        
        # 计算可用索引
        available_indices = list(set(range(self._total_count)) - self._used_indices)
        
        # 如果可用索引不足，重置
        if len(available_indices) == 0:
            logger.debug("All pairs used, resetting sampler")
            self._used_indices.clear()
            available_indices = list(range(self._total_count))
        
        # 准备加权采样
        actual_sample_num = min(self.sample_num, len(available_indices))
        
        # 提取可用索引对应的权重并重新归一化
        available_weights = self._weights[available_indices]
        available_probs = available_weights / available_weights.sum()
        
        # 使用numpy进行加权随机采样（不放回）
        sampled_indices = np.random.choice(
            available_indices,
            size=actual_sample_num,
            replace=False,
            p=available_probs
        )
        
        # 标记为已使用
        self._used_indices.update(sampled_indices)
        
        # 读取内容并记录导入的文档
        suggestions = []
        imported_doc_names = []
        
        for idx in sampled_indices:
            pair = self._available_pairs[idx]
            content = self.loader.read_pair_content(pair)
            if content:
                suggestions.append(content)
                # 记录文档名称用于日志
                imported_doc_names.append(pair['name'])
        
        # 输出导入的优化建议文档日志
        if imported_doc_names:
            logger.info(f"HandwriteSampler: 已导入 {len(imported_doc_names)} 个手写优化文档（加权采样）")
            for i, name in enumerate(imported_doc_names, 1):
                logger.info(f"  [{i}] {name}")
        
        return suggestions
    
    def reset(self):
        """重置采样器，清空已使用记录"""
        self._used_indices.clear()
        logger.debug("Sampler reset")
