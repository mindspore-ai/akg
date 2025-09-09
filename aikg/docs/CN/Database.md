# Database 模块设计文档

## 概述
Database模块是算子优化方案管理数据库框架，主要负责算子优化方案的存储、检索、验证和管理。它提供了跨不同硬件架构和DSL的算子实现的结构化组织和访问方法，通过特征提取和向量检索实现算子方案的快速匹配

## 核心功能
- **多策略检索**：支持随机采样、相似度搜索、规则过滤等多种检索策略
- **方案生命周期管理**：提供算子方案的插入、更新、删除、查找完整操作
- **特征提取**：从实现代码中自动提取算子特征
- **层次化组织**：按架构、DSL和唯一标识符进行结构化存储

### 检索策略说明
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

## 初始化参数
| 参数名称 | 类型/必选 | 默认值 | 参数说明 |
|---------|---------|-------|---------|
| database_path | str (可选) | ../database | 算子方案存储根目录 |
| vector_stores | List[VectorStore] (可选) | [] | 用于相似度搜索的VectorStore列表 |
| config | dict (必选) | None | 包含agent_model_config的配置字典 |

### 配置结构：
```python
config = {
    "agent_model_config": {
        "feature_extraction": "deepseek_r1_default"
    }
}
```

## 核心方法说明

### samples
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

### insert
**功能**：向数据库中插入新的算子实现  
**参数**：
- `impl_code`: 算子实现代码
- `framework_code`: 框架适配器代码
- `backend`: 计算后端(ascend/cuda/cpu)
- `arch`: 硬件架构
- `dsl`: 领域特定语言
- `framework`: 框架名称
- `profile`: 性能配置文件（默认：inf）

**存储结构**：
1. 使用`get_md5_hash()`生成md5_hash
2. 目录结构：`{database_path}/{arch}/{dsl}/{md5_hash}/`
3. 保存元数据为metadata.json
4. 保存实现代码为{dsl}.py
5. 保存框架代码为{framework}.py

### delete
**功能**：从数据库中删除算子实现  
**参数**：
- `impl_code`: 算子实现代码
- `backend`: 计算后端
- `arch`: 硬件架构
- `dsl`: 领域特定语言

**删除过程**：
1. 生成md5_hash定位目录
2. 删除算子目录和文件
3. 级联删除空父目录
4. 更新VectorStore索引

### extract_features
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
```python
from ai_kernel_generator.database.database import Database, RetrievalStrategy
from ai_kernel_generator.database.vector_store import VectorStore

# 初始化带VectorStore支持的数据库
vector_store = VectorStore(
    database_path="/path/to/database",
    embedding_model_name="GanymedeNil/text2vec-large-chinese",
    index_name="operator_vector_store"
)

config = {
    "agent_model_config": {
        "feature_extraction": "deepseek_r1_default"
    }
}

database = Database(
    database_path="/path/to/database",
    vector_stores=[vector_store],
    config=config
)

# 插入新方案
await database.insert(
    impl_code=matmul_impl,
    framework_code=mindspore_adapter,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton",
    framework="mindspore"
)

# 检索相似方案
results = await database.samples(
    output_content=["impl_code", "framework_code"],
    strategy_modes=[RetrievalStrategy.NAIVETY],
    sample_num=5,
    impl_code=new_matmul_impl,
    framework_code=new_framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton",
    framework="mindspore"
)
```
