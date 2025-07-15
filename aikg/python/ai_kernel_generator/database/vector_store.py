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

import yaml
import json
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ai_kernel_generator import get_project_root

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DEFAULT_DATABASE_PATH = Path(get_project_root()).parent.parent / "database"
DEFAULT_INDEX_PATH = DEFAULT_DATABASE_PATH / "vector_store"
DEFAULT_CONFIG_PATH = DEFAULT_DATABASE_PATH / "rag_config.yaml"

class VectorStore:
    """
    基于RAG的优化方案检索器，检索最相似的算子调度方案
    """
    def __init__(
        self, 
        database_path: str = "",
        index_path: str = "",
        config_path: str = ""):
        self.database_path = database_path if database_path else str(database_path)
        self.index_path = index_path if index_path else str(DEFAULT_INDEX_PATH)
        self.config_path = config_path if config_path else str(DEFAULT_CONFIG_PATH)
        self.embedding_model = self.load_embedding_model()
        self.vector_store = self.load_or_create_vector_store()
        self.setup_file_monitor()
        
    def load_embedding_model(self):
        """从配置文件加载嵌入模型"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        embedding_model = config.get('embedding_model', 'GanymedeNil/text2vec-large-chinese')
        return HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # 如有GPU可改为'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_or_create_vector_store(self):
        """加载或创建向量存储"""
        index_path = Path(self.index_path)
        
        # 如果索引不存在则创建
        if not (index_path / "index.faiss").exists():
            print("构建算子特征向量库...")
            return self.build_vector_store()
        
        # 加载现有索引
        print("加载现有向量索引...")
        return FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # 注意安全性
        )
    
    def build_vector_store(self):
        """从算子元数据构建向量存储"""
        root_dir = Path(self.database_path)
        documents = []
        
        # 递归查找所有metadata.json文件（支持任意目录结构）
        for metadata_file in root_dir.rglob("metadata.json"):
            op_subdir = metadata_file.parent  # 算子目录为元数据文件所在目录
            op_name = op_subdir.name  # 算子名称为目录名
            
            # 跳过隐藏目录中的文件
            if any(part.startswith('.') for part in op_subdir.parts):
                continue
                
            # 加载算子元数据
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # 创建检索文档
            doc = Document(
                page_content=metadata['description'],
                metadata={
                    "operator_name": op_name,
                    "arch": metadata.get('arch', ''),
                    "operator_type": metadata.get('type', ''),
                    "operator_shape": metadata.get('shape', ''),
                    "file_path": str(op_subdir),
                }
            )
            documents.append(doc)
        
        # 创建向量存储
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        # 保存索引
        vector_store.save_local(self.index_path)
        return vector_store

    def insert(self, op_name: str, arch: str):
        """向向量存储添加新的算子特征文档
        Args:
            op_name (str): 算子名称。
            arch (str): 架构名称。
        """
        metadata_path = Path(self.database_path) / arch / op_name / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError(f"算子元数据文件 {str(metadata_path)} 不存在")
        
        with open(str(metadata_path), 'r') as f:
            metadata = json.load(f)
        
        # 获取算子目录
        op_dir = metadata_path.parent

        # 创建文档对象
        doc = Document(
            page_content=metadata['description'],
            metadata={
                "operator_name": op_name,
                "arch": arch,
                "operator_type": metadata.get('type', ''),
                "operator_shape": metadata.get('shape', ''),
                "file_path": str(op_dir),
            }
        )

        # 检查是否已存在相同算子的文档（去重并覆盖）
        self.delete(op_name, arch)
        
        # 添加到向量存储并保存
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(self.index_path)
        print(f"成功添加算子 {op_name} {arch}架构到向量索引")

    def delete(self, op_name: str, arch: str):
        existing_ids = list(self.vector_store.index_to_docstore_id.values())
        for doc_id in existing_ids:
            existing_doc = self.vector_store.docstore.search(doc_id)
            if existing_doc.metadata.get("operator_name") == op_name and existing_doc.metadata.get("arch") == arch:
                # 已存在相同算子的文档，删除旧文档
                self.vector_store.delete([doc_id])
                return 
        print(f"算子 {op_name} {arch}架构不存在于向量索引中")
    
    def setup_file_monitor(self):
        """设置文件系统监控器，监听新的metadata.json文件"""
        event_handler = VectorStoreEventHandler(self)
        self.observer = Observer()
        self.observer.schedule(
            event_handler,
            path=self.database_path,
            recursive=True  # 监控所有子目录
        )
        self.observer.start()
        print(f"已启动文件监控，监听目录: {self.database_path}")

    def stop_monitor(self):
        """停止文件监控器"""
        if hasattr(self, 'observer') and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            print("文件监控已停止")


class VectorStoreEventHandler(FileSystemEventHandler):
    """文件系统事件处理器，用于检测新的metadata.json文件"""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.processed_files = set()  # 用于去重已处理的文件

    def on_created(self, event):
        """当文件或目录被创建时触发"""
        if not event.is_directory and event.src_path.endswith('metadata.json'):
            # 等待文件写入完成（处理文件创建事件可能早于内容写入完成的情况）
            time.sleep(0.5)
            file_path = Path(event.src_path)

            # 去重处理
            if str(file_path) in self.processed_files:
                return
            self.processed_files.add(str(file_path))

            try:
                # 调用添加向量存储的方法
                # 解析算子名称和架构
                op_dir = file_path.parent
                op_name = op_dir.name
                arch = op_dir.parent.name
                # 调用添加向量存储的方法
                self.vector_store.insert(op_name, arch)
                print(f"检测到新文件并添加到向量索引: {file_path}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")

    def on_deleted(self, event):
        """当文件或目录被删除时触发"""
        # 处理目录删除
        if event.is_directory:
            dir_path = Path(event.src_path)
            op_name = dir_path.name
            # 跳过隐藏目录
            if any(part.startswith('.') for part in dir_path.parts):
                return
            try:
                op_name = dir_path.name
                arch = dir_path.parent.name
                self.vector_store.delete(op_name, arch)
                print(f"检测到算子目录删除，已从向量索引中移除: {op_name}")
            except ValueError:
                print(f"算子目录 {op_name} 已删除，无法找到metadata.json，跳过向量索引移除")
        # 处理metadata.json文件删除
        elif event.src_path.endswith('metadata.json'):
            file_path = Path(event.src_path)
            op_name = file_path.parent.name
            # 跳过隐藏目录
            if any(part.startswith('.') for part in file_path.parent.parts):
                return
            try:
                op_dir = file_path.parent
                op_name = op_dir.name
                arch = op_dir.parent.name
                self.vector_store.delete(op_name, arch)
                print(f"检测到metadata.json删除，已从向量索引中移除算子: {op_name}")
            except ValueError:
                print(f"metadata.json文件 {file_path} 已删除，跳过向量索引移除")