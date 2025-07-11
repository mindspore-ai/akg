import os
import re
import ast
import numpy as np
# import faiss
from pathlib import Path
import spacy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import List, Dict, Any

"""读取磁盘的json文件并且向量化"""
# 初始化模型
# nlp = spacy.load("en_core_web_sm")
sbert = SentenceTransformer('all-MiniLM-L6-v2')

class AULKnowledgeBase:
    def __init__(self):
        self.aul_files = []  # 存储所有AUL文件元数据
        self.file_index = {}  # 文件路径到索引的映射
        self.op_index = defaultdict(list)  # 操作类型到文件索引的映射
        self.torch_op_index = defaultdict(list)  # Torch操作到文件索引的映射
        self.faiss_index = None
        self.index_embeddings = None
        self.index_descriptions = []
    
    def index_directory(self, root_dir: str):
        """遍历目录索引metadata文件"""
        print(f"开始索引metadata知识库: {root_dir}")
        file_paths = list(Path(root_dir).rglob("metadata.json"))
        print(f"找到 {len(file_paths)} 个metadata文件")
        
        for idx, path in enumerate(file_paths):
            metadata = self.parse_aul_file(path)
            if metadata:
                self.aul_files.append(metadata)
                self.file_index[str(path)] = idx

        # 构建语义索引
        self.build_faiss_index()
        print(f"知识库索引完成，共 {len(self.aul_files)} 个有效文件")
    
    def parse_aul_file(self, file_path: Path) -> Dict[str, Any]:
        """解析AUL文件元数据"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # 1. 从文件名提取前缀 (xxx_aul.py → xxx)
            file_stem = file_path.stem
            if '_' in file_stem:
                file_prefix = file_stem.split('_')[0]  # 提取第一个下划线前的部分
            else:
                file_prefix = file_stem
            
            # 2. 提取函数名前缀 (def relu_op_im_xxx → relu)
            func_prefix = ""
            # 匹配第一个函数定义
            func_match = re.search(r"def\s+(\w+)_", content)
            if func_match:
                full_func_name = func_match.group(1)
                # 提取第一个下划线前的部分
                if '_' in full_func_name:
                    func_prefix = full_func_name.split('_')[0]
                else:
                    func_prefix = full_func_name
            
            return {
                'file_path': str(file_path),
                'file_prefix': file_prefix,
                'func_prefix': func_prefix,
                'content': content
            }
        except Exception as e:
            print(f"解析文件 {file_path} 失败: {e}")
            return None
    
    def build_faiss_index(self):
        """构建语义索引 - 基于描述文本"""
        if not self.aul_files:
            return
        
        # 创建组合文本索引
        texts = []
        for file in self.aul_files:
            # 组合文件名前缀、函数名前缀和描述
            text = f"{file['file_prefix']} {file['func_prefix']} {file['description']}"
            self.index_descriptions.append(text)
            texts.append(text)
        
        # 生成嵌入向量
        self.index_embeddings = sbert.encode(texts, normalize_embeddings=True)

        # self.faiss_index = faiss.IndexFlatIP(self.index_embeddings.shape[1])
        # self.faiss_index.add(self.index_embeddings)
        print(f"FAISS索引构建完成，维度: {self.index_embeddings.shape[1]}, 数量: {len(self.aul_files)}")

if __name__ == "__main__":
    root_dir = "/home/zqs/latest-aikg/ai_kernel_generator/database/swft/ascend310p3"
    aul = AULKnowledgeBase()
    aul.index_directory(root_dir)
    

