# Database 模块设计文档

## 概述
Database模块是算子优化方案管理数据库框架，主要负责算子优化方案的存储、检索、验证和管理，通过特征提取和向量检索实现相似方案的快速匹配。

## 核心功能
- **多策略相似检索**：支持COSINE/EUCLIDEAN_DISTANCE等多种相似度计算策略
- **方案生命周期管理**：提供算子方案的插入、更新、删除、查找完整操作
- **双阶段检索验证**：首次检索+二次验证机制确保结果准确性

## 初始化参数
| 参数名称 | 类型/必选 | 默认值 | 参数说明 |
|---------|---------|-------|---------|
| config_path | str (可选) | database_config.yaml | 配置文件路径，其中 embedding_model 表示可以是模型名字，也可以是本地下载的模型路径；distance_strategy 和 verify_distance_strategy 表示向量距离，当前支持 EUCLIDEAN_DISTANCE，MAX_INNER_PRODUCT，DOT_PRODUCT，JACCARD和COSINE |
| database_path | str (可选) | ../database | 算子方案存储根目录 |

### database_config.yaml示例：
```yaml
# 嵌入模型配置
embedding_model: "GanymedeNil/text2vec-large-chinese"  # 模型名字，也可以是本地下载的模型路径, 如："xxx/thirdparty/text2vec-large-chinese"

# 向量数据库配置
distance_strategy: "COSINE"
verify_distance_strategy: "EUCLIDEAN_DISTANCE"

# 特征提取模型预设配置
agent_model_config:
  feature_extraction: deepseek_r1_default
```

## 核心方法说明
### sample
**功能**：检索相似算子优化方案  
**参数**：
- `impl_code`: 算子实现代码
- `backend`: 计算后端(ascend/cuda/cpu)
- `arch`: 硬件架构(如ascend910b4)
- `impl_type`: 实现类型(triton/swft)

**返回**：
- 召回率、包含相似度得分的方案列表

### insert
**特征生成规则**：
1. 使用`get_md5_hash()`生成特征不变量
2. 目录结构：`{database_path}/operators/{arch}/{impl_type}/{md5_hash}/`
3. 元数据保存为metadata.json

### delete
**删除逻辑**：
1. 根据代码内容生成md5_hash定位目录
2. 级联删除空父目录
3. 同步更新向量存储索引

### verify
**验证流程**：
1. 使用备用距离策略重新检索
2. 计算两次检索结果的召回率
3. 返回验证通过率

## 使用示例
```python
# 初始化RAG系统
rag = DatabaseRAG(config_path="custom_rag.yaml")

# 插入新方案
rag.insert(
    impl_code=matmul_impl,
    framework_code=mindspore_adapter,
    backend="ascend",
    arch="ascend910b4",
    impl_type="triton",
    framework="mindspore"
)

# 检索相似方案
recall, results = rag.sample(
    impl_code=new_matmul_impl,
    backend="ascend",
    arch="ascend910b4",
    impl_type="triton"
)
```
