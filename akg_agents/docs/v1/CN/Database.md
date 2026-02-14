# Database 模块设计文档

## 概述
Database 模块是数据库框架：

- **通用基类**（`database/`）：提供 `Database` 抽象基类和 `VectorStore` 抽象基类，定义了文档存储、检索、向量索引的通用框架
- **算子专用实现**（`op/database/`）：`CoderDatabase` 和 `CoderVectorStore`，继承通用基类，实现算子方案的特定逻辑

## 架构

```
database/                           # 通用基类
├── database.py                     # Database 抽象基类
└── vector_store.py                 # VectorStore 抽象基类

op/database/                        # 算子专用实现
├── coder_database.py               # CoderDatabase（算子代码数据库）
└── coder_vector_store.py           # CoderVectorStore（算子向量存储）
```

## 通用基类

### Database（抽象基类）

提供文档管理的通用框架，子类需实现 `_do_insert` 和 `_do_delete` 方法。

**核心方法：**
- `_insert_with_vectors(doc_id, content, mode)`: 通用插入方法（含向量存储同步），支持 `skip` 和 `overwrite` 模式
- `_delete_with_vectors(doc_id)`: 通用删除方法（含向量存储同步和空目录清理）
- `clear()`: 清空数据库和所有向量存储

**初始化参数：**

| 参数名称 | 类型/必选 | 描述 |
|---------|---------|---------|
| database_path | str (可选) | 数据库存储根目录 |
| vector_stores | List[VectorStore] (可选) | 向量存储列表 |
| config | dict (必选) | 配置字典，需包含 `agent_model_config` |

### VectorStore（抽象基类）

提供向量存储和检索的通用框架，子类需实现 `gen_document` 方法。

**核心特性：**
- 基于 FAISS 的向量索引
- 支持 OpenAI 兼容嵌入模型（通过 `core_v2` 的 `create_embedding_model`）
- 自动从 metadata.json 构建索引
- 支持相似度搜索和 MMR 搜索

**嵌入模型加载：**
VectorStore 现在通过 `core_v2.llm.create_embedding_model()` 加载嵌入模型，支持 OpenAI 兼容的 Embedding API（详见 [RAG 文档](./RAG.md)）。加载优先级：
1. 配置中的 OpenAI 兼容 Embedding 模型
2. 指定本地 HuggingFace 模型
3. 自动降级禁用向量存储功能

**检索接口：**

| 方法 | 描述 |
|------|------|
| `similarity_search(query, k, fetch_k)` | 语义相似度搜索 |
| `max_marginal_relevance_search(query, k)` | MMR 搜索（平衡相似度和多样性） |
| `similarity_search_with_score(query, k, fetch_k)` | 带得分的语义搜索 |
| `insert(doc_path)` | 向向量存储添加文档 |
| `delete(doc_path)` | 从向量存储删除文档 |
| `clear()` | 清空向量存储 |

## 算子专用实现

### CoderDatabase

继承自 `Database`，专门用于算子代码的存储和检索。

**核心特性：**
- 单例模式，避免资源重复创建
- 基于计算逻辑的层次搜索（计算相似 → 形状匹配）
- 支持特征提取（通过 LLM 从代码中提取算子特征）
- 支持从 benchmark 目录自动更新数据库

**检索策略：**

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| RANDOMICITY | 随机采样 | 测试、基线对比 |
| HIERARCHY | 层次检索（计算→形状） | 代码生成场景（默认） |
| NAIVETY | 直接向量相似度 | 简单匹配 |
| MMR | 最大边际相关性 | 多样性需求 |

**核心方法：**

```python
# 检索相似算子方案
results = await coder_db.samples(
    output_content=["impl_code"],
    sample_num=3,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton_ascend"
)

# 插入新方案
await coder_db.insert(
    impl_code=impl_code,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton_ascend",
    framework="torch"
)

# 删除方案
coder_db.delete(
    impl_code=impl_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton_ascend"
)

# 从 benchmark 目录自动更新
await coder_db.auto_update(
    dsl="triton_ascend",
    framework="torch",
    backend="ascend",
    arch="ascend910b4",
    ref_type="docs"  # 或 "impl"
)
```

### CoderVectorStore

继承自 `VectorStore`，专注于算子计算特征的向量索引。

**核心特性：**
- 专注于计算相关特征：`["op_name", "computation"]`
- 单例模式
- 支持 feature_invariants 过滤

## 存储结构

```
database/
├── ascend910b4/
│   ├── triton/
│   │   ├── {md5_hash_1}/
│   │   │   ├── metadata.json     # 算子特征
│   │   │   ├── triton.py          # DSL 实现代码
│   │   │   ├── torch.py           # 框架适配代码
│   │   │   └── doc.md             # 优化文档（可选）
│   │   └── {md5_hash_2}/
│   │       └── ...
│   └── swft/
│       └── ...
└── a100/
    └── triton/
        └── ...
```

## 配置说明

```python
config = {
    "agent_model_config": {
        "feature_extraction": "standard"  # 特征提取模型
    },
    "database_config": {
        "enable_rag": True,          # 是否启用 RAG 功能
        "sample_num": 2,             # 默认采样数量
    }
}
```

## 使用示例

```python
from akg_agents.op.database.coder_database import CoderDatabase

# 初始化（单例模式）
coder_db = CoderDatabase(config=config)

# 从 benchmark 自动更新数据库
await coder_db.auto_update(
    dsl="triton_ascend",
    framework="torch",
    backend="ascend",
    arch="ascend910b4"
)

# 检索相似方案
results = await coder_db.samples(
    output_content=["impl_code", "framework_code"],
    sample_num=3,
    impl_code=my_impl_code,
    framework_code=my_framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton_ascend",
    framework="torch"
)
```

## 相关文档
- [RAG 模块文档](./RAG.md)
- [RAG 使用指南](./RAG_Usage.md)
