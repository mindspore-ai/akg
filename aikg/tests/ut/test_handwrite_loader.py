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
HandwriteLoader和HandwriteSampler的单元测试 - 精简版
测试新的目录结构
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from ai_kernel_generator.utils.handwrite_loader import HandwriteLoader, HandwriteSampler


@pytest.fixture
def temp_handwrite_dir():
    """创建临时的手写文件目录结构"""
    temp_dir = tempfile.mkdtemp()
    
    # 模拟实际项目结构:
    # temp_dir 模拟 aikg/python/ai_kernel_generator (project_root)
    # 需要创建 aikg/benchmark/aikgbench/
    # 
    # temp_dir/          <- aikg/python/ai_kernel_generator (project_root)
    # ../../benchmark/   <- aikg/benchmark/
    #   └── aikgbench/
    #       ├── triton_ascend/
    #       │   ├── impl/
    #       │   └── docs/
    #       ├── dynamic_shape/
    #       └── static_shape/
    
    # 从 temp_dir 往上2级，然后创建 benchmark
    aikg_root = Path(temp_dir).parent.parent
    benchmark_root = aikg_root / "benchmark"
    aikgbench_root = benchmark_root / "aikgbench"
    torch_base = aikgbench_root
    triton_impl_base = aikgbench_root / "triton_ascend" / "impl"
    triton_docs_base = aikgbench_root / "triton_ascend" / "docs"
    
    # 创建测试文件（包含dynamic_shape和static_shape）
    test_files = [
        ("dynamic_shape", "reduction", "softmax_001", "Softmax"),
        ("dynamic_shape", "reduction", "layernorm_001", "LayerNorm"),
        ("static_shape", "reduction", "relu_001", "ReLU"),
        ("static_shape", "sorting", "topk_001", "TopK"),
        ("static_shape", "reduction", "gelu_001", "GELU"),
    ]
    
    for shape_type, category, name, desc in test_files:
        # 创建torch文件
        torch_dir = torch_base / shape_type / category
        torch_dir.mkdir(parents=True, exist_ok=True)
        (torch_dir / f"{name}.py").write_text(f"# Torch {desc}\ndef {name}_torch(): pass")
        
        # 创建triton实现文件
        triton_impl_dir = triton_impl_base / shape_type / category
        triton_impl_dir.mkdir(parents=True, exist_ok=True)
        (triton_impl_dir / f"{name}.py").write_text(f"# Triton {desc}\n@triton.jit\ndef {name}_kernel(): pass")
        
        # 创建优化建议文件
        triton_docs_dir = triton_docs_base / shape_type / category
        triton_docs_dir.mkdir(parents=True, exist_ok=True)
        (triton_docs_dir / f"{name}.md").write_text(f"# {desc} Optimization\nSuggestions for {name}")
    
    yield {
        'temp_dir': temp_dir,
        'benchmark_root': benchmark_root,
        'aikgbench_root': aikgbench_root,
        'torch_base': torch_base,
        'triton_impl_base': triton_impl_base,
        'triton_docs_base': triton_docs_base,
        'test_files': test_files
    }
    
    # 清理
    shutil.rmtree(temp_dir)
    if benchmark_root.exists():
        shutil.rmtree(benchmark_root, ignore_errors=True)


class TestHandwriteLoaderCore:
    """测试HandwriteLoader核心功能"""
    
    def test_load_and_read(self, temp_handwrite_dir):
        """测试1: 加载所有文件并读取内容"""
        with patch('ai_kernel_generator.utils.handwrite_loader.get_project_root') as mock_root:
            mock_root.return_value = str(temp_handwrite_dir['temp_dir'])
            
            loader = HandwriteLoader(dsl="triton")
            
            # 验证加载
            assert len(loader._all_data_pairs) == 5
            assert len(loader._selected_data_pairs) == 5
            
            # 验证数据对结构
            first_pair = loader._all_data_pairs[0]
            assert 'name' in first_pair
            assert 'file_stem' in first_pair
            assert 'shape_type' in first_pair
            assert 'category' in first_pair
            assert first_pair['shape_type'] in ['dynamic_shape', 'static_shape']
            assert first_pair['category'] in ['reduction', 'sorting']
            
            # 验证读取
            content = loader.read_pair_content(first_pair)
            
            assert isinstance(content, dict)
            assert all(k in content for k in ['name', 'file_stem', 'shape_type', 'category', 
                                               'torch_code', 'triton_code', 'improvement'])
            assert 'Triton' in content['triton_code']
            assert 'Torch' in content['torch_code']
            assert 'Optimization' in content['improvement']
    
    @pytest.mark.asyncio
    async def test_select_with_mock_llm(self, temp_handwrite_dir):
        """测试2: LLM筛选功能（Mock）"""
        with patch('ai_kernel_generator.utils.handwrite_loader.get_project_root') as mock_root:
            mock_root.return_value = str(temp_handwrite_dir['temp_dir'])
            
            loader = HandwriteLoader(
                dsl="triton",
                op_name="relu_op",
                task_desc="ReLU activation",
                config={'agent_model_config': {'default': {}}}
            )
            
            with patch('ai_kernel_generator.utils.handwrite_loader.Selector') as MockSelector:
                mock_selector = MockSelector.return_value
                # LLM返回完整路径名称
                mock_selector.run = AsyncMock(return_value=[
                    'static_shape/reduction/relu_001',
                    'static_shape/reduction/gelu_001'
                ])
                
                await loader.select_relevant_pairs()
                
                assert len(loader._selected_data_pairs) == 2
                selected_names = {p['name'] for p in loader._selected_data_pairs}
                assert 'static_shape/reduction/relu_001' in selected_names
                assert 'static_shape/reduction/gelu_001' in selected_names


class TestHandwriteSamplerCore:
    """测试HandwriteSampler核心功能"""
    
    @pytest.fixture
    def mock_loader(self):
        """创建Mock的HandwriteLoader"""
        loader = Mock(spec=HandwriteLoader)
        
        mock_pairs = [
            {
                'name': f'static_shape/reduction/opt_{i:02d}',
                'file_stem': f'opt_{i:02d}',
                'shape_type': 'static_shape',
                'category': 'reduction',
                'torch_file': Path(f'/tmp/opt_{i:02d}.py'),
                'triton_file': Path(f'/tmp/opt_{i:02d}.py'),
                'improvement_file': Path(f'/tmp/opt_{i:02d}.md')
            }
            for i in range(10)
        ]
        
        loader.get_selected_pairs.return_value = mock_pairs
        loader.read_pair_content.side_effect = lambda p: {
            'name': p['name'],
            'file_stem': p['file_stem'],
            'shape_type': p['shape_type'],
            'category': p['category'],
            'torch_code': f"torch {p['name']}",
            'triton_code': f"triton {p['name']}",
            'improvement': f"improve {p['name']}"
        }
        
        return loader
    
    def test_basic_sampling(self, mock_loader):
        """测试3: 基本采样和不重复"""
        sampler = HandwriteSampler(loader=mock_loader, sample_num=3)
        
        # 第一次采样
        s1 = sampler.sample()
        assert len(s1) == 3
        
        # 第二次采样不重复
        s2 = sampler.sample()
        assert len(s2) == 3
        names1 = {x['name'] for x in s1}
        names2 = {x['name'] for x in s2}
        assert len(names1 & names2) == 0  # 无交集
    
    def test_reset_when_exhausted(self, mock_loader):
        """测试4: 用完后自动重置"""
        sampler = HandwriteSampler(loader=mock_loader, sample_num=3)
        
        # 采样3次 = 9个
        for _ in range(3):
            sampler.sample()
        
        # 第4次只剩1个
        s4 = sampler.sample()
        assert len(s4) == 1
        
        # 第5次重置后又是3个
        s5 = sampler.sample()
        assert len(s5) == 3
    
    def test_independent_samplers(self, mock_loader):
        """测试5: 多个sampler独立性"""
        samplers = [HandwriteSampler(loader=mock_loader, sample_num=2) for _ in range(3)]
        
        # 每个独立采样
        results = [s.sample() for s in samplers]
        
        # 都能正常采样
        assert all(len(r) == 2 for r in results)
        
        # 各自的状态独立
        assert all(len(s._used_indices) == 2 for s in samplers)


class TestMultiRoundScenario:
    """测试多轮采样场景"""
    
    @pytest.fixture
    def setup_samplers(self):
        """设置测试用的samplers"""
        loader = Mock(spec=HandwriteLoader)
        
        mock_pairs = [
            {
                'name': f'static_shape/reduction/opt_{i:02d}',
                'file_stem': f'opt_{i:02d}',
                'shape_type': 'static_shape',
                'category': 'reduction',
                'torch_file': Path(f'/tmp/opt_{i:02d}.py'),
                'triton_file': Path(f'/tmp/opt_{i:02d}.py'),
                'improvement_file': Path(f'/tmp/opt_{i:02d}.md')
            }
            for i in range(6)
        ]
        
        loader.get_selected_pairs.return_value = mock_pairs
        loader.read_pair_content.side_effect = lambda p: {
            'name': p['name'],
            'file_stem': p['file_stem'],
            'shape_type': p['shape_type'],
            'category': p['category'],
            'torch_code': f"code {p['name']}",
            'triton_code': f"code {p['name']}",
            'improvement': f"improve {p['name']}"
        }
        
        return loader
    
    def test_multi_island_multi_round(self, setup_samplers):
        """测试6: 多岛屿多轮采样"""
        num_islands = 3
        num_rounds = 2
        sample_num = 2
        
        # 每个岛屿独立sampler
        island_samplers = [
            HandwriteSampler(loader=setup_samplers, sample_num=sample_num)
            for _ in range(num_islands)
        ]
        
        # 模拟2轮
        for round_idx in range(num_rounds):
            for island_idx in range(num_islands):
                suggestions = island_samplers[island_idx].sample()
                assert 1 <= len(suggestions) <= sample_num
        
        # 验证每个岛屿都采样到了多个不同的建议
        for sampler in island_samplers:
            # 2轮 × 2个 = 4个采样
            assert len(sampler._used_indices) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
