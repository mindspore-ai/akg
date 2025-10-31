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

"""
Selector Agent的单元测试 - 精简版
只测试核心流程
"""

import pytest
from unittest.mock import patch, AsyncMock
from ai_kernel_generator.core.agent.selector import Selector


@pytest.fixture
def mock_config():
    """Mock配置"""
    return {
        'agent_model_config': {
            'default': {
                'model_name': 'test-model',
                'temperature': 0.3,
                'max_tokens': 1000
            }
        }
    }


@pytest.fixture
def sample_candidates():
    """示例候选文档"""
    return [
        {
            'name': 'relu_001',
            'torch_code': 'def relu_torch(x): return torch.relu(x)',
            'triton_code': '@triton.jit\ndef relu_kernel(): pass',
            'improvement': '# ReLU优化建议\n使用向量化操作'
        },
        {
            'name': 'gelu_001',
            'torch_code': 'def gelu_torch(x): return F.gelu(x)',
            'triton_code': '@triton.jit\ndef gelu_kernel(): pass',
            'improvement': '# GELU优化建议\n使用近似计算'
        },
        {
            'name': 'matmul_001',
            'torch_code': 'def matmul_torch(a, b): return torch.matmul(a, b)',
            'triton_code': '@triton.jit\ndef matmul_kernel(): pass',
            'improvement': '# MatMul优化建议\n使用tile优化'
        }
    ]


class TestSelectorCore:
    """测试Selector核心功能"""
    
    def test_initialization(self, mock_config):
        """测试1: 初始化和基本属性"""
        selector = Selector(
            op_name="relu_op",
            task_desc="ReLU activation function",
            dsl="triton",
            config=mock_config
        )
        
        assert selector.op_name == "relu_op"
        assert selector.task_desc == "ReLU activation function"
        assert selector.dsl == "triton"
        assert selector.llm_step_count == 0
    
    @pytest.mark.asyncio
    async def test_run_with_valid_selection(self, mock_config, sample_candidates):
        """测试2: 正常选择流程"""
        selector = Selector(
            op_name="relu_op",
            task_desc="ReLU activation function",
            dsl="triton",
            config=mock_config
        )
        
        # Mock LLM响应
        mock_llm_response = '{"selected_names": ["relu_001", "gelu_001"]}'
        
        with patch.object(selector, 'run_llm', new_callable=AsyncMock) as mock_run_llm:
            mock_run_llm.return_value = (mock_llm_response, None, None)
            
            selected_names = await selector.run(sample_candidates)
            
            assert selected_names == ["relu_001", "gelu_001"]
            assert selector.llm_step_count == 1
            mock_run_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_filter_invalid_names(self, mock_config, sample_candidates):
        """测试3: 过滤无效文档名"""
        selector = Selector(
            op_name="relu_op",
            task_desc="ReLU activation function",
            dsl="triton",
            config=mock_config
        )
        
        # Mock LLM响应，包含无效名称
        mock_llm_response = '{"selected_names": ["relu_001", "invalid_001", "gelu_001"]}'
        
        with patch.object(selector, 'run_llm', new_callable=AsyncMock) as mock_run_llm:
            mock_run_llm.return_value = (mock_llm_response, None, None)
            
            selected_names = await selector.run(sample_candidates)
            
            # 只返回有效名称
            assert set(selected_names) == {"relu_001", "gelu_001"}
            assert "invalid_001" not in selected_names
    
    @pytest.mark.asyncio
    async def test_fallback_on_empty_selection(self, mock_config, sample_candidates):
        """测试4: 空选择时的fallback"""
        selector = Selector(
            op_name="relu_op",
            task_desc="ReLU activation function",
            dsl="triton",
            config=mock_config
        )
        
        # Mock LLM返回空列表
        mock_llm_response = '{"selected_names": []}'
        
        with patch.object(selector, 'run_llm', new_callable=AsyncMock) as mock_run_llm:
            mock_run_llm.return_value = (mock_llm_response, None, None)
            
            selected_names = await selector.run(sample_candidates)
            
            # Fallback: 返回所有候选文档名
            assert len(selected_names) == 3
            assert set(selected_names) == {"relu_001", "gelu_001", "matmul_001"}
    
    @pytest.mark.asyncio
    async def test_parse_error_fallback(self, mock_config, sample_candidates):
        """测试5: 解析错误时的fallback"""
        selector = Selector(
            op_name="relu_op",
            task_desc="ReLU activation function",
            dsl="triton",
            config=mock_config
        )
        
        # Mock LLM返回无效JSON
        mock_llm_response = 'invalid json response'
        
        with patch.object(selector, 'run_llm', new_callable=AsyncMock) as mock_run_llm:
            mock_run_llm.return_value = (mock_llm_response, None, None)
            
            selected_names = await selector.run(sample_candidates)
            
            # Fallback: 返回所有候选文档名
            assert len(selected_names) == 3
            assert set(selected_names) == {"relu_001", "gelu_001", "matmul_001"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
