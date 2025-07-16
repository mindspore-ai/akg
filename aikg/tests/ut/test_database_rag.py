import pytest
import json
import os
from pathlib import Path
from ai_kernel_generator import get_project_root
from ai_kernel_generator.database.database_rag import DatabaseRAG

@pytest.mark.level0
def test_database_rag():
    # 初始化系统
    config_path = Path(get_project_root()) / "database" / "rag_config.yaml"
    db_system = DatabaseRAG(str(config_path))

    # 准备查询特征
    query_features = {
        "type": "reduce+elementwise融合",
        "name": "custom_softmax",
        "shape": "reduce轴:64, 非reduce轴:8192",
        "description": "包含exp和sum操作的融合算子",
        "arch": "ascend310p3"
    }

    # 检索优化方案
    results = db_system.find(query_features)
    # 输出结果
    print("===================================================")
    print(f"找到 {len(results)} 个匹配的优化方案:")
    for i, res in enumerate(results, 1):
        print(f"\n#{i} 相似度: {res['similarity_score']:.4f}")
        print(f"算子名称: {res['operator_name']}")
        print(f"文件路径: {res['file_path']}")
        print(f"特征描述: {res['description'][:100]}...")
    print("===================================================")

    # 添加示例（向量）
    new_metadata = {
        "op_type": "reduce+elementwise融合",
        "op_name": "custom_softmax",
        "op_shape": "reduce轴:64, 非reduce轴:8192",
        "description": "包含exp和sum操作的融合算子",
        "arch": "ascend310p3"
    }
    meta_path = Path(get_project_root()).parent.parent / "database" / "operators" / "ascend310p3" / "custom_softmax" / "metadata.json"
    os.makedirs(meta_path.parent, exist_ok=True)
    print(str(meta_path.parent))
    with open(str(meta_path), 'w', encoding='utf-8') as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=4)
    db_system.test_insert("custom_softmax", "ascend310p3")

    # 检索优化方案
    results = db_system.find(query_features)
    # 输出结果
    print("===================================================")
    print(f"找到 {len(results)} 个匹配的优化方案:")
    for i, res in enumerate(results, 1):
        print(f"\n#{i} 相似度: {res['similarity_score']:.4f}")
        print(f"算子名称: {res['operator_name']}")
        print(f"文件路径: {res['file_path']}")
        print(f"特征描述: {res['description'][:100]}...")
    print("===================================================")

    # 删除示例
    db_system.delete("custom_softmax", "ascend310p3")

    # 检索优化方案
    results = db_system.find(query_features)
    # 输出结果
    print("===================================================")
    print(f"找到 {len(results)} 个匹配的优化方案:")
    for i, res in enumerate(results, 1):
        print(f"\n#{i} 相似度: {res['similarity_score']:.4f}")
        print(f"算子名称: {res['operator_name']}")
        print(f"文件路径: {res['file_path']}")
        print(f"特征描述: {res['description'][:100]}...")
    print("===================================================")
