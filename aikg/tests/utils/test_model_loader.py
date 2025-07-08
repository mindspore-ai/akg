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

import os
import pytest
from pathlib import Path

from langchain.prompts import PromptTemplate
from ai_kernel_generator.core.llm.model_loader import create_model, OLLAMA_API_BASE_ENV, VLLM_API_BASE_ENV


def test_verify_all_models():
    """验证所有模型的可用性"""
    # 获取配置文件路径
    config_path = Path(__file__).parent.parent.parent / "ai_kernel_generator" / "core" / "llm" / "llm_config.yaml"

    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)

    presets = [k for k in config.keys() if k != "default_preset"]

    # 创建简单的提示模板
    template = "你好"
    prompt = PromptTemplate.from_template(template)

    # 存储结果
    working_models = []
    failed_models = []

    print("\n开始验证所有模型...")
    for preset in presets:
        try:
            # 创建模型
            model = create_model(preset)

            # 创建链
            chain = prompt | model

            # 测试调用
            response = chain.invoke({})

            # 如果成功，添加到工作模型列表
            working_models.append(preset)
            print(f"{preset}: 验证成功")

        except Exception as e:
            # 如果失败，添加到失败模型列表
            failed_models.append((preset, str(e)))
            print(f"{preset}: 验证失败 - {str(e)}")

    # 打印总结
    print("\n=== 验证结果总结 ===")
    print(f"\n成功模型 ({len(working_models)}):")
    for model in working_models:
        print(f"- {model}")

    print(f"\n失败模型 ({len(failed_models)}):")
    for model, error in failed_models:
        print(f"- {model}: {error}")

    print(f"有 {len(failed_models)} 个模型验证失败")


def test_ollama_model_with_env():
    """测试带环境变量的Ollama模型"""
    # 保存原始环境变量
    original_env = os.environ.get(OLLAMA_API_BASE_ENV)

    try:
        # 设置测试环境变量
        test_api_base = os.getenv('AIKG_OLLAMA_API_BASE', 'http://localhost:11434')
        os.environ[OLLAMA_API_BASE_ENV] = test_api_base

        # 创建模型

        model = create_model("ollama_qwen_coder_0.5b_default")

        # 验证模型配置
        assert model.base_url == test_api_base

        # 创建简单的提示模板并测试
        template = "你好"
        prompt = PromptTemplate.from_template(template)
        chain = prompt | model
        response = chain.invoke({})

        # 验证响应
        assert response is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0

        print("\n=== Ollama环境变量测试响应 ===")
        print(f"输入: {template}")
        print(f"输出: {response.content}")
        print("============================")

        print(f"Ollama环境变量测试成功: {test_api_base}")

    except Exception as e:
        pytest.fail(f"Ollama环境变量测试失败: {str(e)}")

    finally:
        # 恢复环境变量
        if original_env is not None:
            os.environ[OLLAMA_API_BASE_ENV] = original_env
        else:
            os.environ.pop(OLLAMA_API_BASE_ENV, None)


def test_vllm_model_with_env():
    """测试带环境变量的VLLM模型"""
    # 保存原始环境变量
    original_env = os.environ.get(VLLM_API_BASE_ENV)

    try:
        # 设置测试环境变量
        test_api_base = os.getenv('AIKG_VLLM_API_BASE', 'http://localhost:8001/v1')
        os.environ[VLLM_API_BASE_ENV] = test_api_base

        # 创建模型

        model = create_model("vllm_deepseek_r1_default")

        # 创建简单的提示模板并测试
        template = "你好"
        prompt = PromptTemplate.from_template(template)
        chain = prompt | model
        response = chain.invoke({})

        # # 直接读取文件内容作为 prompt 字符串
        # with open("1.txt", "r") as f:
        #     template = f.read().strip()  # 读取并去除首尾空字符
        # chain = model  # 直接使用模型，无需模板链
        # response = chain.invoke(template)  # 将纯文本作为输入传递给模型

        # 验证响应
        assert response is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0

        print("\n=== VLLM环境变量测试响应 ===")
        print(f"输入: {template}")
        print(f"输出: {response.content}")
        print("============================")

        print(f"VLLM环境变量测试成功: {test_api_base}")

    except Exception as e:
        pytest.fail(f"VLLM环境变量测试失败: {str(e)}")

    finally:
        # 恢复环境变量
        if original_env is not None:
            os.environ[VLLM_API_BASE_ENV] = original_env
        else:
            os.environ.pop(VLLM_API_BASE_ENV, None)


if __name__ == "__main__":
    test_verify_all_models()
    test_ollama_model_with_env()
    test_vllm_model_with_env()
