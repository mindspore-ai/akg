# RAG 使用指南

RAG（检索增强生成）功能通过向量检索从历史算子代码库中查找相似实现，为算子生成提供参考。本文档介绍如何安装和使用 RAG 功能。

## 1. 安装依赖

在 `aikg` 目录下执行以下命令安装 RAG 相关依赖：

```bash
pip install -r requirements_rag.txt
```

## 2. 下载嵌入模型

RAG 功能需要嵌入模型来生成向量表示。执行以下命令下载模型：

```bash
bash download.sh --with_local_model
```

该命令会将 `text2vec-large-chinese` 模型下载到 `~/.aikg/text2vec-large-chinese` 目录。

**说明：**
- 模型下载需要网络连接，首次下载可能需要较长时间
- 如果模型目录已存在，脚本会跳过下载
- 如需重新下载，请先删除 `~/.aikg/text2vec-large-chinese` 目录

## 3. AKG CLI 使用 RAG 功能

在生成算子时，通过 `--rag` 参数启用 RAG 功能：

```bash
# Ascend 910B2
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001 --rag

# CUDA A100: --backend cuda --arch a100 --dsl triton_cuda
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --worker-url 127.0.0.1:9001 --rag
```

**说明：**
- `--rag` 启用 RAG 功能，`--no-rag` 关闭（默认关闭）
- RAG 功能会在生成算子时自动检索相似的历史实现作为参考
- 其他参数的使用方式与普通算子生成相同，详见 [AKG CLI 文档](./AKG_CLI.md)

**完整示例：**

```bash
# Ascend 910B2 使用 RAG 和 intent
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001 --rag --intent "实现 fused softmax，输入为 [batch, head, seq, dim]"

# CUDA A100 使用 RAG 和 intent
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --worker-url 127.0.0.1:9001 --rag --intent "实现 fused softmax，输入为 [batch, head, seq, dim]"
```

## 4. 注意事项

- RAG 功能需要向量数据库已建立索引，首次使用前可能需要构建索引
- 检索效果取决于向量数据库中的历史实现质量和数量
- 如果未找到相似实现，系统会加载默认示例正常进行算子生成，不受影响

