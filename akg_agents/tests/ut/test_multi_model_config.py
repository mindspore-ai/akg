# Copyright 2026 Huawei Technologies Co., Ltd
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

"""
测试多模型配置功能
"""

import pytest
from akg_agents.core_v2.config import ModelConfig, get_settings
from akg_agents.core_v2.llm import create_llm_client


class TestMultiModelConfig:
    """测试多模型配置"""
    
    def test_model_config_creation(self):
        """测试 ModelConfig 创建"""
        config = ModelConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="gpt-4",
            temperature=0.0
        )
        
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.0
    
    def test_model_config_from_dict(self):
        """测试从字典创建 ModelConfig"""
        data = {
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model_name": "deepseek-chat",
            "temperature": 0.5
        }
        
        config = ModelConfig.from_dict(data)
        
        assert config.model_name == "deepseek-chat"
        assert config.temperature == 0.5
    
    def test_settings_with_multiple_models(self, monkeypatch, tmp_path):
        """测试多模型配置加载"""
        # 设置环境变量
        monkeypatch.setenv("AKG_AGENTS_COMPLEX_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("AKG_AGENTS_COMPLEX_API_KEY", "sk-complex")
        monkeypatch.setenv("AKG_AGENTS_COMPLEX_MODEL_NAME", "gpt-4")
        
        monkeypatch.setenv("AKG_AGENTS_STANDARD_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("AKG_AGENTS_STANDARD_API_KEY", "sk-standard")
        monkeypatch.setenv("AKG_AGENTS_STANDARD_MODEL_NAME", "deepseek-chat")
        
        monkeypatch.setenv("AKG_AGENTS_FAST_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("AKG_AGENTS_FAST_API_KEY", "sk-fast")
        monkeypatch.setenv("AKG_AGENTS_FAST_MODEL_NAME", "gpt-3.5-turbo")
        
        settings = get_settings()
        
        # 验证三个级别都加载了
        assert "complex" in settings.models
        assert "standard" in settings.models
        assert "fast" in settings.models
        
        # 验证配置内容
        assert settings.models["complex"].model_name == "gpt-4"
        assert settings.models["standard"].model_name == "deepseek-chat"
        assert settings.models["fast"].model_name == "gpt-3.5-turbo"


class TestLLMClientFactory:
    """测试 LLM Client 工厂函数"""
    
    def test_create_client_with_level(self, monkeypatch):
        """测试使用模型级别创建 client"""
        monkeypatch.setenv("AKG_AGENTS_STANDARD_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("AKG_AGENTS_STANDARD_API_KEY", "sk-test")
        monkeypatch.setenv("AKG_AGENTS_STANDARD_MODEL_NAME", "gpt-4")
        
        client = create_llm_client(model_level="standard")
        
        assert client.provider.model_name == "gpt-4"
    
    def test_create_client_with_direct_params(self):
        """测试直接指定参数创建 client"""
        client = create_llm_client(
            model_name="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-test"
        )
        
        assert client.provider.model_name == "deepseek-chat"
        assert "deepseek" in str(client.provider.client.base_url)
    
    def test_backward_compatibility(self, monkeypatch):
        """测试向后兼容性（旧的单模型配置）"""
        # 旧版配置方式
        monkeypatch.setenv("AKG_AGENTS_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("AKG_AGENTS_API_KEY", "sk-test")
        monkeypatch.setenv("AKG_AGENTS_MODEL_NAME", "deepseek-chat")
        
        settings = get_settings()
        
        # 应该自动设置为 standard 级别
        assert "standard" in settings.models
        assert settings.models["standard"].model_name == "deepseek-chat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
