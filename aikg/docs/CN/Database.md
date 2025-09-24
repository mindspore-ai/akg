# Database 模块设计文档

## 概述
Database模块是算子优化方案管理数据库框架，主要负责算子优化方案的存储、检索、验证和管理。它提供了跨不同硬件架构和DSL的算子实现的结构化组织和访问方法，通过特征提取和向量检索实现算子方案的快速匹配。

## 核心功能
- **多策略检索**：支持随机采样、相似度搜索、规则过滤等多种检索策略
- **方案生命周期管理**：提供算子方案的插入、更新、删除、查找完整操作
- **特征提取**：从实现代码中自动提取算子特征
- **层次化组织**：按架构、DSL和唯一标识符进行结构化存储

## 核心组件

### Database (基类)
提供算子方案管理的核心功能的主数据库类。

**关键方法：**
- `extract_features()`: 从实现代码中提取算子特征
- `samples()`: 使用指定策略检索相似算子方案
- `insert()`: 向数据库插入新算子实现
- `delete()`: 从数据库删除算子实现
- `get_case_content()`: 从特定算子案例中检索内容

### CoderDatabase (代码生成专用)
继承自Database，专门用于代码生成场景。

**核心特性：**
- 单例模式实现高效资源管理
- 专注于计算的向量存储，用于代码相似性
- 层次搜索：计算逻辑 → 形状匹配

### EvolveDatabase (进化优化专用)
继承自Database，专门用于进化优化场景。

**核心特性：**
- 多个向量存储处理不同调度方面（base、pass、text）
- 使用倒数排名融合（RRF）的融合搜索
- 最大边际相关性（MMR）重排序
- 基于性能的最优性搜索

## 存储结构

数据库使用层次化文件系统结构：

```
database/
├── ascend910b4/
│   ├── triton/
│   │   ├── {md5_hash_1}/
│   │   │   ├── metadata.json
│   │   │   ├── triton.py
│   │   │   └── torch.py
│   │   └── {md5_hash_2}/
│   │       ├── metadata.json
│   │       ├── triton.py
│   │       └── mindspore.py
│   └── swft/
│       └── {md5_hash_3}/
│           ├── metadata.json
│           ├── swft.py
│           └── numpy.py
└── cuda/
    └── triton/
        └── {md5_hash_4}/
            ├── metadata.json
            ├── triton.py
            └── torch.py
```

## 检索策略

Database模块支持多种检索策略来查找算子优化方案：

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **RANDOMICITY** | 从数据库中随机抽取算子 | 测试、基线对比、无偏采样 |
| **NAIVETY** | 基于特征向量的直接相似度搜索 | 简单相似度匹配、直接检索 |
| **MMR** | 最大边际相关性，平衡相似度和多样性 | 避免冗余结果、确保方案多样性 |
| **OPTIMALITY** | 性能最优的检索策略 | 高性能场景、需要最快检索速度 |
| **RULE** | 基于指定规则的搜索 | 自定义过滤逻辑、领域特定需求 |
| **HIERARCHY** | 跨不同抽象层次的层次检索 | 多级分析、渐进式细化 |
| **FUSION** | 多策略融合，结合不同方法 | 综合搜索、利用多种方法 |

## 使用指南

### 初始化参数
| 参数名称 | 类型/必选 | 默认值 | 参数说明 |
|---------|---------|-------|---------|
| database_path | str (可选) | ../database | 算子方案存储根目录 |
| vector_stores | List[VectorStore] (可选) | [] | 用于相似度搜索的VectorStore列表 |
| config | dict (必选) | None | 包含agent_model_config的配置字典 |

### 配置说明
```python
config = {
    "agent_model_config": {
        "feature_extraction": "deepseek_r1_default"  # 特征提取模型配置
    },
    "database_config": {
        "enable_rag": True,          # 是否启用RAG功能
        "sample_num": 2,             # 默认采样数量
        "embedding_device": "cpu"    # 嵌入模型设备：cpu 或 cuda
    }
}
```

**配置参数说明：**
- `feature_extraction`: 指定用于特征提取的模型
- `enable_rag`: 控制是否启用向量检索功能
- `sample_num`: 设置默认的检索样本数量
- `embedding_device`: 指定嵌入模型运行的设备

### 核心方法

#### samples
**功能**：使用指定策略检索相似算子优化方案  
**参数**：
- `output_content`: 要检索的内容类型列表（如["impl_code", "framework_code"]）
- `strategy_modes`: 要使用的检索策略列表
- `sample_num`: 要检索的样本数量（默认：5）
- `impl_code`: 算子实现代码
- `framework_code`: 框架适配器代码
- `backend`: 计算后端(ascend/cuda/cpu)
- `arch`: 硬件架构(如ascend910b4)
- `dsl`: 领域特定语言(triton/swft)
- `framework`: 框架名称(mindspore/pytorch)

**返回**：检索到的算子方案列表

#### insert
**功能**：向数据库中插入新的算子实现  
**参数**：
- `impl_code`: 算子实现代码
- `framework_code`: 框架适配器代码
- `backend`: 计算后端(ascend/cuda/cpu)
- `arch`: 硬件架构
- `dsl`: 领域特定语言
- `framework`: 框架名称
- `profile`: 性能配置文件（默认：inf）

#### delete
**功能**：从数据库中删除算子实现  
**参数**：
- `impl_code`: 算子实现代码
- `backend`: 计算后端
- `arch`: 硬件架构
- `dsl`: 领域特定语言

#### extract_features
**功能**：从实现代码中提取算子特征  
**参数**：
- `impl_code`: 算子实现代码
- `framework_code`: 框架适配器代码
- `backend`: 计算后端
- `arch`: 硬件架构
- `dsl`: 领域特定语言
- `profile`: 性能配置文件

**返回**：包含提取特征的字典（op_name, op_type, input_specs, output_specs, computation, schedule等）

## 使用示例

### 基础数据库操作
```python
from ai_kernel_generator.database.database import Database, RetrievalStrategy

# 初始化数据库
database = Database(
    database_path="/path/to/database",
    vector_stores=[vector_store],
    config=config
)

# 插入新方案
await database.insert(
    impl_code=impl_code,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton",
    framework="mindspore"
)

# 检索相似方案
results = await database.samples(
    output_content=["impl_code"],
    strategy_modes=[RetrievalStrategy.NAIVETY],
    sample_num=5,
    impl_code=impl_code,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton",
    framework="mindspore"
)

# 删除方案
await database.delete(
    impl_code=impl_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton"
)
```

### 专用数据库使用

#### CoderDatabase 示例
```python
from ai_kernel_generator.database.coder_database import CoderDatabase

# 初始化代码生成数据库
coder_db = CoderDatabase(config=config)

# 代码生成的层次搜索
results = await coder_db.samples(
    output_content=["impl_code"],
    sample_num=3,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton"
)
```

#### EvolveDatabase 示例
```python
from ai_kernel_generator.database.evolve_database import EvolveDatabase

# 初始化进化优化数据库
evolve_db = EvolveDatabase(config=config)

# 融合搜索优化
results = await evolve_db.samples(
    output_content=["impl_code"],
    sample_num=5,
    impl_code=impl_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton"
)
```

## 性能优化

- **单例模式**：数据库类使用单例模式避免资源重复
- **懒加载**：向量存储仅在需要时构建
- **高效索引**：基于FAISS的向量索引实现快速相似度搜索
- **内存管理**：删除时自动清理空目录
- **并发安全**：带锁机制的线程安全单例实现