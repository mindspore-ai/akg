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

class VectorStore:
    """
    基于RAG的优化方案检索器，检索最相似的算子调度方案
    """
    def __init__(
        self, 
        database_path="database", 
        config_path = "",
        index_path="",
        ):
        self.database_path = database_path
        self.config_path = config_path
        self.index_path = index_path
        self.embedding_model = self._load_embedding_model()
        self.vector_store = self.load_or_create_vector_store()
        self._setup_file_monitor()
        
    def _load_embedding_model(self):
        """从配置文件加载嵌入模型"""
        # 构建配置文件路径
        config_path = Path(self.database_path / self.config_path)
        
        # 验证配置文件存在性
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 加载并解析YAML配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"配置文件解析失败: {str(e)}") from e
        
        # 提取嵌入模型名称
        model_name = config.get('embedding_model')
        if not model_name:
            raise ValueError("配置文件中未找到embedding_model配置项")
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # 如有GPU可改为'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_or_create_vector_store(self):
        """加载或创建向量存储"""
        index_path = Path(self.database_path / self.index_path)
        
        # 如果索引不存在则创建
        if not (index_path / "index.faiss").exists():
            print("构建算子特征向量库...")
            return self._build_vector_store()
        
        # 加载现有索引
        print("加载现有向量索引...")
        return FAISS.load_local(
            folder_path=str(index_path),
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # 注意安全性
        )
    
    def _build_vector_store(self):
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
            
            # 查找算子实现文件
            py_files = list(op_subdir.glob("*.py"))
            if not py_files:
                print(f"警告: 算子 {op_name} 目录下未找到Python实现文件，跳过该算子")
                continue
            
            # 创建检索文档
            doc = Document(
                page_content=metadata['description'],
                metadata={
                    "operator_name": op_name,
                    "operator_type": metadata.get('type', ''),
                    "operator_shape": metadata.get('shape', ''),
                    "file_path": str(py_files[0]),
                }
            )
            documents.append(doc)
        
        # 创建向量存储
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        # 保存索引
        vector_store.save_local(str(Path(self.database_path / self.index_path)))
        return vector_store

    def add_to_vector_store(self, metadata_file: str):
        """向向量存储添加新的算子特征文档
        Args:
            metadata_file: 算子元数据文件路径
        """
        if not Path(metadata_file).exists():
            raise ValueError(f"算子元数据文件 {metadata_file} 不存在")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # 获取算子目录并查找Python实现文件
        op_dir = Path(metadata_file).parent
        op_name = op_dir.name
        py_files = list(op_dir.glob("*.py"))
        if not py_files:
            raise ValueError(f"算子 {op_name} 目录下未找到Python实现文件")

        # 创建文档对象
        doc = Document(
            page_content=metadata['description'],
            metadata={
                "operator_name": op_name,
                "operator_type": metadata.get('type', ''),
                "operator_shape": metadata.get('shape', ''),
                "file_path": str(py_files[0]),
            }
        )

        # 检查是否已存在相同算子的文档（去重并覆盖）
        self.delete_from_vector_store(op_name)
        
        # 添加到向量存储并保存
        self.vector_store.add_documents([doc])
        self.vector_store.save_local(self.index_path)
        print(f"成功添加算子 {op_name} 到向量索引")
    
    def _setup_file_monitor(self):
        """设置文件系统监控器，监听新的metadata.json文件"""
        event_handler = VectorStoreEventHandler(self)
        self.observer = Observer()
        self.observer.schedule(
            event_handler,
            path=str(self.database_path),
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

    def delete_from_vector_store(self, op_name: str):
        existing_ids = list(self.vector_store.index_to_docstore_id.values())
        for doc_id in existing_ids:
            existing_doc = self.vector_store.docstore.search(doc_id)
            if existing_doc.metadata.get("operator_name") == op_name:
                # 已存在相同算子的文档，删除旧文档
                self.vector_store.delete([doc_id])
                return 
        print(f"算子 {op_name} 不存在于向量索引中")


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
                self.vector_store.add_to_vector_store(str(file_path))
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
            self.vector_store.delete_from_vector_store(op_name)
            print(f"检测到算子目录删除，已从向量索引中移除: {op_name}")