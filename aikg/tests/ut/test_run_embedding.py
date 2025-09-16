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

import asyncio
from ai_kernel_generator.core.llm.model_loader import create_embedding_model


async def test_simple_run_embedding():
    """简单的硅流embedding API测试"""
    
    # 指定model name
    model_name = "sflow_qwen3_embedding_8b"
    
    try:
        # 创建embedding模型
        embedding_model = create_embedding_model(model_name)
        print(f"成功创建embedding模型: {model_name}")
        print(f"模型类型: {type(embedding_model).__name__}")
        
        # 测试文本
        test_texts = [
            "这是一个测试文本，用于验证embedding功能",
            "另一个测试文本，包含不同的内容",
            "Silicon flow embedding online: fast, affordable, and high-quality embedding services"
        ]
        
        print(f"\n=== 开始生成embedding ===")
        print(f"输入文本数量: {len(test_texts)}")
        
        # 使用LangChain的embed_documents方法
        embeddings = embedding_model.embed_documents(test_texts)
        
        # 输出结果
        print(f"API响应状态: 成功")
        print(f"生成向量数量: {len(embeddings)}")
        print(f"每个向量维度: {len(embeddings[0])}")
        
        # 显示前几个向量的前5个值
        for i, embedding in enumerate(embeddings):
            vector_preview = embedding[:5]
            print(f"文本{i+1}向量预览: {vector_preview}...")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_single_text_embedding():
    """测试单个文本的embedding生成"""
    
    try:
        # 创建embedding模型
        embedding_model = create_embedding_model("sflow_qwen3_embedding_4b")
        
        # 单个文本
        single_text = "这是一个单独的测试文本"
        
        print(f"\n=== 单个文本embedding测试 ===")
        print(f"输入文本: {single_text}")
        
        # 使用LangChain的embed_query方法
        embedding = embedding_model.embed_query(single_text)
        
        print(f"生成成功，向量维度: {len(embedding)}")
        print(f"向量前10个值: {embedding[:10]}")
        
    except Exception as e:
        print(f"单个文本测试失败: {e}")


if __name__ == "__main__":
    print("开始测试硅流embedding API...")
    asyncio.run(test_simple_run_embedding())
    asyncio.run(test_single_text_embedding())
