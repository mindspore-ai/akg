# RAG (检索增强生成) 模块设计文档

## 概述
AIKG中的RAG模块目前通过VectorStore类实现，提供基于向量的文档检索能力。通过向量检索，可以快速找到相似的相关文档内容。

## 核心功能
- **向量存储**：基于FAISS的高效向量索引
- **嵌入模型**：支持HuggingFace嵌入模型
- **自动文档生成**：从算子元数据自动生成检索文档
- **多种检索方式**：相似度搜索、最大边际相关性搜索
- **索引管理**：支持插入、删除、清空等操作

## 核心组件

### VectorStore (抽象基类)
RAG系统的基础，提供向量存储和检索能力。

**关键特性：**
- 所有向量存储实现的抽象基类
- 单例模式提高资源效率
- HuggingFace嵌入模型支持（默认：GanymedeNil/text2vec-large-chinese）
- 基于FAISS的向量索引
- 从算子元数据自动生成文档

### 专用向量存储

#### CoderVectorStore
专门用于代码生成场景的向量存储。

**核心特性：**
- 专注于计算相关特征：["op_name", "op_type", "input_specs", "output_specs", "computation"]
- 实现层次搜索能力
- 支持代码相似性匹配

#### EvolveVectorStore
专门用于进化优化场景的向量存储。

**核心特性：**
- 处理调度相关特征：["base", "pass", "text"]
- 支持多种调度方面以实现多样化优化
- 专门处理schedule块字段

## 嵌入模型支持

### 模型加载机制
VectorStore支持灵活的嵌入模型加载策略：

**加载优先级：**
1. **指定模型**：优先加载配置中指定的HuggingFace模型
2. **环境变量**：如果指定模型失败，尝试从EMBEDDING_MODEL_PATH环境变量加载本地模型
3. **优雅降级**：如果所有加载方式都失败，自动禁用向量存储功能

### 设备配置
- **CPU模式**：默认配置，适合开发和测试
- **CUDA模式**：通过配置文件中的`embedding_device: "cuda"`启用GPU加速

## 文档生成

### 自动文档创建
VectorStore从算子元数据自动生成检索文档，子类需要实现`gen_document`方法来定义具体的文档生成逻辑。

### 专用文档生成

#### CoderVectorStore文档生成
- 从元数据中提取计算相关特征
- 构建包含算子类型、文件路径等信息的文档
- 支持特征不变量的过滤

#### EvolveVectorStore文档生成
- 专门处理schedule块字段
- 将调度信息展开为键值对格式
- 支持多种调度方面的特征提取

## 索引管理

### 自动索引构建
- 索引从metadata.json文件自动构建
- 支持递归目录遍历
- 优雅处理空数据库
- 使用FAISS持久化存储

## 使用指南

### 文档存储结构
每个文档及其元数据文件存储在一个独立的文件夹中：
```
{doc_path}/
├── metadata.json    # 元数据文件
└── {document_file}       # 文档内容文件
```

`doc_path`参数指向包含文档的文件夹路径，相对于database_path。

### 索引操作接口

#### insert
**功能**：向向量存储添加新的文档  
**参数**：
- `doc_path`: 要插入的文档路径

#### delete
**功能**：从向量存储中删除指定文档  
**参数**：
- `doc_path`: 要删除的文档路径

#### clear
**功能**：清空向量存储中的所有文档  

### 检索接口

#### similarity_search
**功能**：执行语义搜索并返回匹配的文档  
**参数**：
- `query`: 查询字符串
- `k`: 返回的文档数量（默认：5）
- `fetch_k`: 候选文档数量（默认：20，用于提高召回率）

**返回**：匹配的Document对象列表

#### max_marginal_relevance_search
**功能**：执行最大边际相关搜索，平衡相似度和多样性  
**参数**：
- `query`: 查询字符串
- `k`: 返回的文档数量（默认：5）

**返回**：经过MMR重排序的Document对象列表

**特点**：
- `lambda_mult=0.2`：极致多样性设置
- `fetch_k=max(20, 5 * k)`：动态候选数量

#### similarity_search_with_score
**功能**：执行语义搜索并返回匹配的文档及其相似度得分  
**参数**：
- `query`: 查询字符串
- `k`: 返回的文档数量（默认：5）
- `fetch_k`: 候选文档数量（默认：20）

**返回**：(Document, score)元组列表

## 使用示例

### 基础向量搜索
```python
from ai_kernel_generator.database.coder_vector_store import CoderVectorStore

# 初始化向量存储
vector_store = CoderVectorStore(
    database_path="/path/to/database",
    config=config
)

# 执行相似度搜索
docs = vector_store.similarity_search(query, k=5)

# 执行MMR搜索以获得多样性
docs = vector_store.max_marginal_relevance_search(query, k=5)
```

### 索引管理操作
```python
# 插入新文档
vector_store.insert("path/to/your/document")

# 删除文档
vector_store.delete("path/to/your/document")

# 清空所有文档
vector_store.clear()
```

## 未来扩展

### 潜在扩展
- **API文档集成**: 支持AscendC API手册
- **多源RAG**: 与外部知识源集成
- **自定义文档适配器**: 支持PDF、markdown和其他格式
- **高级融合策略**: 更复杂的结果组合方法
- **查询扩展**: 自动查询增强以提高检索效果