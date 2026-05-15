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

import shutil
import json
import random
import logging
from typing import List, Dict, Any
from pathlib import Path
import asyncio
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from akg_agents.op.database.coder_vector_store import CoderVectorStore
from akg_agents.database.database import Database, RetrievalStrategy
from akg_agents import get_project_root
from akg_agents.core.agent.utils.feature_extractor import FeatureExtractor
from akg_agents.utils.common_utils import get_md5_hash

logger = logging.getLogger(__name__)

DEFAULT_CODER_DATABASE_PATH = Path(get_project_root()).parent.parent / "coder_database" / "local"
DEFAULT_BENCHMARK_PATH = Path(get_project_root()).parent.parent / "benchmark" / "akg_kernels_bench"

class CoderDatabase(Database):
    """
    算子代码数据库，专门用于存储和检索算子实现代码。
    继承自通用的 Database 基类，添加了算子特定的功能。
    """
    # 单例模式实现
    _instances: Dict[str, 'CoderDatabase'] = {}
    _lock = False  # 简单的锁机制避免并发问题
    
    def __new__(cls, database_path: str = "", benchmark_path: str = "", config: dict = None):
        database_path = database_path or str(DEFAULT_CODER_DATABASE_PATH)
        # 使用数据库路径作为实例的唯一标识
        instance_key = get_md5_hash(database_path=database_path)
        
        # 检查实例是否已存在
        if instance_key not in cls._instances or cls._instances[instance_key] is None:
            # 简单锁机制
            while cls._lock:
                pass
            cls._lock = True
            try:
                # 双重检查锁定模式
                if instance_key not in cls._instances or cls._instances[instance_key] is None:
                    cls._instances[instance_key] = super(CoderDatabase, cls).__new__(cls)
            finally:
                cls._lock = False
        
        return cls._instances[instance_key]
        
    def __init__(self, database_path: str = "", benchmark_path: str = "", config: dict = None):
        # 防止重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.database_path = database_path or str(DEFAULT_CODER_DATABASE_PATH)
        self.benchmark_path = benchmark_path or DEFAULT_BENCHMARK_PATH
        self.computation_vector_store = CoderVectorStore(
            database_path=self.database_path,
            index_name="computation_vector_store",
            features=["op_name", "computation"],
            config=config
        )
        self.vector_stores = [self.computation_vector_store]
        super().__init__(self.database_path, self.vector_stores, config)
        self._auto_update_completed = set() # 用于追踪已完成的auto_update配置（基于参数hash）
        self._initialized = True

    async def extract_features(self, impl_code: str, framework_code: str, backend: str, arch: str, dsl: str, profile=float('inf')):
        """提取算子任务特征"""
        # 特征提取
        feature_extractor = FeatureExtractor(
            model_config=self.model_config,
            impl_code=impl_code,
            framework_code=framework_code,
            config=self.config  # 传递完整 config，包含 session_id
        )
        feature_content, _, _ = await feature_extractor.run()
        parsed_content = feature_extractor.feature_parser.parse(feature_content)
        extracted_features = {
            "op_name": parsed_content.op_name,
            "op_type": parsed_content.op_type,
            "input_specs": parsed_content.input_specs,
            "output_specs": parsed_content.output_specs,
            "computation": parsed_content.computation,
            "schedule": parsed_content.schedule,
            "profile": profile,
            "backend": backend,
            "arch": arch,
            "dsl": dsl
        }
        return extracted_features

    def get_case_content(self, output_content: List[str], case_path: str, strategy_mode: RetrievalStrategy = None, dsl: str = "", framework: str = ""):
        """获取算子案例内容"""
        case_path = Path(case_path)
        if not case_path.exists():
            raise FileNotFoundError(f"Case path not found: {case_path}")
        
        res_dict = {}
        for content in output_content:
            if content == "strategy_mode":
                res_dict[content] = strategy_mode
                continue

            if content == "impl_code":
                if not dsl:
                    raise ValueError("dsl is required for impl_code")
                code_file_path = case_path / f"{dsl}.py"
                if not code_file_path.exists():
                    raise FileNotFoundError(f"Code file not found: {code_file_path}")
                with open(code_file_path, "r", encoding="utf-8") as f:
                    impl_code = f.read()
                res_dict[content] = impl_code
                continue
                
            if content == "framework_code":
                if not framework:
                    raise ValueError("framework is required for framework_code")
                code_file_path = case_path / f"{framework}.py"
                if not code_file_path.exists():
                    raise FileNotFoundError(f"Code file not found: {code_file_path}")
                with open(code_file_path, "r", encoding="utf-8") as f:
                    framework_code = f.read()
                res_dict[content] = framework_code
                continue

            if content == "improvement_doc":
                doc_file = case_path / "doc.md"
                if not doc_file.exists():
                    raise FileNotFoundError(f"Document file not found: {doc_file}")
                with open(doc_file, "r", encoding="utf-8") as f:
                    improvement_doc = f.read()
                res_dict[content] = improvement_doc
                continue
            
            metadata_file = case_path / "metadata.json"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if content in metadata:
                res_dict[content] = metadata[content]
            else:
                raise ValueError(f"Content '{content}' not found. Available keys: strategy_mode, impl_code, framework_code, {', '.join(metadata.keys())}")

        return res_dict

    def randomicity_search(self, output_content: List[str], k: int = 5, backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        """随机搜索算子实现"""
        if not arch or not dsl:
            raise ValueError("arch and dsl must be provided in random mode")
            
        sample_path = Path(self.database_path) / arch / dsl
        if not sample_path.exists() or not sample_path.is_dir():
            raise ValueError(f"Sample path {sample_path} does not exist or is not a directory")
        
        cases = [f for f in sample_path.iterdir() if f.is_dir()]
        k = min(k, len(cases))
        selected_cases = random.sample(cases, k) if k > 0 else []
        result = []
        for case_path in selected_cases:
            result.append(self.get_case_content(output_content, case_path, RetrievalStrategy.RANDOMICITY, dsl, framework))
        return result
    
    def get_output_content(self, output_content: List[str], strategy_mode: RetrievalStrategy, docs, dsl, framework):
        """获取输出内容"""
        result = []
        for doc in docs:
            case_path = doc.metadata["file_path"]
            result.append(self.get_case_content(output_content, case_path, strategy_mode, dsl, framework))
        return result
    
    async def update_single_ref_file(self, file_path, dsl, framework, backend, arch, ref_type, update_mode, 
                                      impl_code: str = None, md5_hash: str = None):
        """处理单个参考文件
        
        Args:
            file_path: 文件路径（Path对象）
            dsl: DSL类型（如 'triton_ascend'）
            framework: 框架类型（如 'torch'）
            backend: 后端类型
            arch: 架构类型
            ref_type: 参考类型（'docs' 或 'impl'）
            update_mode: 更新模式（'skip' 或 'overwrite'）
            impl_code: 预读取的实现代码（可选，避免重复读取）
            md5_hash: 预计算的 md5_hash（可选，避免重复计算）
        """
        benchmark_path = Path(self.benchmark_path)
        relative_path = file_path.relative_to(benchmark_path)
        parts = relative_path.parts
        shape_type = parts[2]
        manual_op_type = parts[3]
        manual_op_name = file_path.stem

        try:
            if impl_code is None:
                impl_path = benchmark_path / dsl / "impl" / shape_type / manual_op_type / f"{manual_op_name}.py"
                with open(impl_path, 'r', encoding='utf-8') as f:
                    impl_code = f.read()
            
            framework_path = benchmark_path / shape_type / manual_op_type / f"{manual_op_name}.py"
            with open(framework_path, 'r', encoding='utf-8') as f:
                framework_code = f.read()
            
            # 读取优化文档（如果是docs类型）
            improvement_doc = ""
            if ref_type == "docs":
                doc_path = benchmark_path / dsl / "docs" / shape_type / manual_op_type / f"{manual_op_name}.md"
                if doc_path.exists():
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        improvement_doc = f.read()
                else:
                    logger.warning(f"Document file not found: {doc_path}")
            
            # 构建自定义元数据
            custom = {
                "name": f"{shape_type}/{manual_op_type}/{manual_op_name}",
                "shape_type": shape_type
            }

            # 插入数据库
            await self.insert(impl_code, framework_code, backend, arch, dsl, framework, 
                              improvement_doc=improvement_doc, custom=custom, mode=update_mode, md5_hash=md5_hash)
            logger.debug(f"Successfully processed: {shape_type}/{manual_op_type}/{manual_op_name}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    async def process_ref_file(self, file_path, dsl, framework, backend, arch, ref_type, semaphore, update_mode,
                               impl_code: str = None, md5_hash: str = None):
        """通过信号量控制并发处理文件"""
        async with semaphore:
            await self.update_single_ref_file(file_path, dsl, framework, backend, arch, ref_type, update_mode, impl_code, md5_hash)

    async def auto_update(self, dsl: str, framework: str, backend: str, arch: str, 
                          ref_type: str = "docs", max_concurrency: int = 4, update_mode: str = "skip",
                          max_files: int = None):
        """从benchmark目录自动更新数据库
        
        Args:
            dsl: DSL类型（如 'triton_ascend'）
            framework: 框架类型（如 'torch'）
            backend: 后端类型
            arch: 架构类型
            ref_type: 参考类型，'docs' 或 'impl'
                - 'docs': 从docs目录加载，包含优化文档
                - 'impl': 从impl目录加载，不包含优化文档
            max_concurrency: 最大并发处理数，默认4
            update_mode: 更新模式
                - 'skip': 已存在的条目跳过
                - 'overwrite': 已存在的条目覆盖
            max_files: 最大处理文件数，用于测试时限制数量
        
        Returns:
            list: 处理结果列表，每个元素对应一个文件的处理结果（None表示成功或跳过）
        """
        # 生成配置唯一标识
        config_key = get_md5_hash(
            dsl=dsl, 
            framework=framework, 
            backend=backend, 
            arch=arch, 
            ref_type=ref_type
        )
        
        # 检查是否已经执行过相同配置的auto_update
        if config_key in self._auto_update_completed:
            logger.info(f"auto_update already completed for dsl={dsl}, framework={framework}, "
                       f"backend={backend}, arch={arch}, ref_type={ref_type}, skipping")
            return
        
        # 验证参数
        if ref_type not in ('docs', 'impl'):
            raise ValueError("ref_type must be either 'docs' or 'impl'")
        
        # 检查目录是否存在
        benchmark_path = Path(self.benchmark_path)
        ref_dir = benchmark_path / dsl / ref_type
        ref_files = list(ref_dir.rglob("*.md")) if ref_type == "docs" else list(ref_dir.rglob("*.py"))
        
        # 收集 benchmark 中所有有效文件的 impl_code 和 md5_hash（避免重复计算）
        benchmark_hashes = set()
        valid_files = []  # (file_path, impl_code, md5_hash)
        for file_path in ref_files:
            if any(part.startswith('.') for part in file_path.parts):
                continue
            relative_path = file_path.relative_to(benchmark_path)
            parts = relative_path.parts
            if len(parts) < 5:
                continue
            shape_type, manual_op_type, manual_op_name = parts[2], parts[3], file_path.stem
            impl_path = benchmark_path / dsl / "impl" / shape_type / manual_op_type / f"{manual_op_name}.py"
            try:
                with open(impl_path, 'r', encoding='utf-8') as f:
                    impl_code = f.read()
                md5_hash = get_md5_hash(impl_code=impl_code, backend=backend, arch=arch, dsl=dsl)
                benchmark_hashes.add(md5_hash)
                valid_files.append((file_path, impl_code, md5_hash))
            except Exception:
                continue
        
        # 限制文件数量（用于测试）
        if max_files is not None and len(valid_files) > max_files:
            valid_files = valid_files[:max_files]
            benchmark_hashes = {h for _, _, h in valid_files}
        
        # 删除过期 case
        db_dir = Path(self.database_path) / arch / dsl
        if db_dir.exists():
            db_hashes = {d.name for d in db_dir.iterdir() if d.is_dir()}
            for stale_hash in db_hashes - benchmark_hashes:
                self.delete_by_hash(stale_hash, arch, dsl)
        
        # 处理 benchmark 文件（复用预读取的 impl_code 和预计算的 md5_hash）
        tasks = []
        semaphore = asyncio.Semaphore(max_concurrency)
        for file_path, impl_code, md5_hash in valid_files:
            tasks.append(self.process_ref_file(file_path, dsl, framework, backend, arch, ref_type, semaphore, update_mode, impl_code, md5_hash))
        
        await asyncio.gather(*tasks)
        
        # 标记为已完成
        self._auto_update_completed.add(config_key)
        logger.info(f"auto_update completed for dsl={dsl}, framework={framework}, "
                    f"backend={backend}, arch={arch}, ref_type={ref_type}")
        
        return

    def hierarchy_search(self, features: dict, feature_invariants: str, k: int = 5):
        """层次检索：先按计算逻辑检索，再按shape检索"""
        op_type = features["op_type"]
        computation_query = ", ".join([f"{key}: {features[key]}" for key in self.computation_vector_store.features])
        computation_docs = self.computation_vector_store.vector_store.similarity_search(
            query=computation_query, 
            k=max(20, 5 * k),
            fetch_k=max(100, 20 * k),
            filter={"feature_invariants": feature_invariants, "op_type": op_type}
        )
        if not computation_docs:
            raise ValueError(f"No {op_type} operator found")
        shape_features = ["input_specs", "output_specs"]
        shape_docs = []
        for doc in computation_docs:
            doc_path = doc.metadata.get("file_path", "")
            doc_content = self.get_case_content(shape_features, doc_path)
            shape_doc = Document(
                page_content=", ".join([f"{key}: {doc_content[key]}" for key in shape_features]),
                metadata={"file_path": doc_path}
            )
            shape_docs.append(shape_doc)
        shape_vector_store = FAISS.from_documents(shape_docs, self.computation_vector_store.embedding_model)
        docs = shape_vector_store.similarity_search(
            query=", ".join([f"{key}: {features[key]}" for key in shape_features]),
            k=k
        )
        return docs

    async def samples(self, output_content: List[str], sample_num: int = 5, impl_code: str = "", framework_code: str = "",
                      backend: str = "", arch: str = "", dsl: str = "", framework: str = ""):
        """
        Coder采样方案，多层级混合检索
        """
        need_extract_features = False
        for vector_store in self.vector_stores:
            if vector_store.enable_vector_store:
                need_extract_features = True
                break
        
        if need_extract_features:
            features = await self.extract_features(impl_code, framework_code, backend, arch, dsl)
            feature_invariants = get_md5_hash(backend=backend, arch=arch, dsl=dsl)
            
            docs = self.hierarchy_search(features, feature_invariants, sample_num)
            result = self.get_output_content(output_content, RetrievalStrategy.HIERARCHY, docs, dsl, framework)
        else:
            random_res = self.randomicity_search(output_content, sample_num, backend, arch, dsl, framework)
            result = random_res
        
        return result

    async def _do_insert(self, doc_id: str, content: Any, file_path: Path, **kwargs):
        """执行具体的算子插入逻辑
        
        Args:
            doc_id: 文档ID（格式：arch/dsl/md5_hash）
            content: 内容字典，包含 impl_code, framework_code 等
            file_path: 存储路径
            **kwargs: 其他参数
        """
        impl_code = content.get("impl_code", "")
        framework_code = content.get("framework_code", "")
        dsl = content.get("dsl", "")
        framework = content.get("framework", "")
        backend = content.get("backend", "")
        arch = content.get("arch", "")
        improvement_doc = content.get("improvement_doc")
        profile = content.get("profile", float('inf'))
        custom = content.get("custom")
        
        # 提取特征
        features = await self.extract_features(impl_code, framework_code, backend, arch, dsl, profile)
        if custom:
            features.update(custom)

        # 创建目录并保存文件
        file_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = file_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=4)
        
        framework_file = file_path / f"{framework}.py"
        with open(framework_file, "w", encoding="utf-8") as f:
            f.write(framework_code)
            
        impl_file = file_path / f"{dsl}.py"
        with open(impl_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        if improvement_doc:
            doc_file = file_path / "doc.md"
            with open(doc_file, "w", encoding="utf-8") as f:
                f.write(improvement_doc)

    def _do_delete(self, doc_id: str, file_path: Path):
        """执行具体的算子删除逻辑"""
        shutil.rmtree(file_path)

    async def insert(self, impl_code: str, framework_code: str, backend: str, arch: str, dsl: str, framework: str, 
                     improvement_doc: str = None, profile=float('inf'), custom: dict = None, mode: str = "skip", md5_hash: str = None):
        """
        插入新的算子实现
        
        Args:
            impl_code: 实现代码
            framework_code: 框架代码
            backend: 后端类型
            arch: 架构类型
            dsl: DSL类型
            framework: 框架类型
            improvement_doc: 优化文档（可选）
            profile: 性能数据（可选）
            custom: 自定义元数据（可选）
            mode: 插入模式，'skip' 跳过已存在的，'overwrite' 覆盖已存在的
            md5_hash: 预计算的哈希值（可选）
        """
        md5_hash = md5_hash or get_md5_hash(impl_code=impl_code, backend=backend, arch=arch, dsl=dsl)
        doc_id = f"{arch}/{dsl}/{md5_hash}"
        
        content = {
            "impl_code": impl_code,
            "framework_code": framework_code,
            "backend": backend,
            "arch": arch,
            "dsl": dsl,
            "framework": framework,
            "improvement_doc": improvement_doc,
            "profile": profile,
            "custom": custom,
        }
        
        await self._insert_with_vectors(doc_id, content, mode=mode)

    def delete_by_hash(self, md5_hash: str, arch: str, dsl: str):
        """
        根据 md5_hash 删除算子实现
        """
        doc_id = f"{arch}/{dsl}/{md5_hash}"
        self._delete_with_vectors(doc_id)

    def delete(self, impl_code: str, backend: str, arch: str, dsl: str):
        """
        删除算子实现
        
        Args:
            impl_code: 实现代码
            backend: 后端类型
            arch: 架构类型
            dsl: DSL类型
        """
        md5_hash = get_md5_hash(impl_code=impl_code, backend=backend, arch=arch, dsl=dsl)
        self.delete_by_hash(md5_hash, arch, dsl)
