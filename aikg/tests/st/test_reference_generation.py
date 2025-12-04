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
测试参考数据生成功能

用于验证 CUDA-to-Ascend 转换场景中的参考数据生成和传输功能。
在 CPU 后端上运行，验证基础功能的正确性。
"""

import pytest
import asyncio
import tarfile
import io
import os
import tempfile
import torch

from ai_kernel_generator.core.worker.local_worker import LocalWorker
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier


# 简单的 ReLU task_desc 用于测试
RELU_TASK_DESC = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """Simple ReLU model for testing."""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

batch_size = 4
dim = 16

def get_inputs():
    torch.manual_seed(0)  # 固定种子确保可复现
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
'''


def create_reference_generation_package(op_name: str, task_desc: str) -> bytes:
    """
    创建参考数据生成的 TAR 包
    
    Args:
        op_name: 算子名称
        task_desc: task_desc 代码
        
    Returns:
        bytes: TAR 包数据
    """
    # 生成参考数据的脚本
    gen_ref_script = f'''
import torch
import sys
import os

sys.path.append(os.getcwd())

def generate_reference():
    print("Starting reference data generation...")
    try:
        from reference import Model, get_inputs, get_init_inputs
        print("Successfully imported Model and helper functions.")
        
        device = "cpu"
        print(f"Using device: {{device}}")
        
        # Fixed seed
        torch.manual_seed(0)
        print("[INFO] Random seed: 0")
        
        # Instantiate model
        init_inputs = get_init_inputs()
        model = Model(*init_inputs)
        model.eval()
        
        # Get inputs
        torch.manual_seed(0)
        inputs = get_inputs()
        
        # Run forward
        with torch.no_grad():
            outputs = model(*inputs)
        
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        
        # Save reference data
        ref_data = {{
            'op_name': '{op_name}',
            'seed': 0,
            'outputs': outputs,
            'output_shapes': [x.shape if isinstance(x, torch.Tensor) else None for x in outputs],
        }}
        
        ref_file = os.path.join(os.getcwd(), "{op_name}_reference.pt")
        torch.save(ref_data, ref_file)
        print(f"[INFO] Reference data saved to: {{ref_file}}")
        print(f"[INFO] Output count: {{len(outputs)}}")
        
        return True
    except Exception as e:
        print(f"Error: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_reference()
    if success:
        print("REFERENCE_GENERATION_SUCCESS")
        sys.exit(0)
    else:
        print("REFERENCE_GENERATION_FAILED")
        sys.exit(1)
'''
    
    # 创建 TAR 包
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
        # 添加 reference.py
        ref_info = tarfile.TarInfo(name="reference.py")
        ref_bytes = task_desc.encode('utf-8')
        ref_info.size = len(ref_bytes)
        tar_file.addfile(tarinfo=ref_info, fileobj=io.BytesIO(ref_bytes))
        
        # 添加 verify_{op_name}.py
        script_info = tarfile.TarInfo(name=f"verify_{op_name}.py")
        script_bytes = gen_ref_script.encode('utf-8')
        script_info.size = len(script_bytes)
        tar_file.addfile(tarinfo=script_info, fileobj=io.BytesIO(script_bytes))
    
    return tar_buffer.getvalue()


class TestReferenceGeneration:
    """测试参考数据生成功能"""
    
    @pytest.mark.asyncio
    async def test_local_worker_generate_reference_success(self):
        """测试 LocalWorker 成功生成参考数据"""
        op_name = "test_relu"
        package_data = create_reference_generation_package(op_name, RELU_TASK_DESC)
        
        # 创建 Worker
        device_pool = DevicePool([0])
        worker = LocalWorker(device_pool, backend="cpu")
        
        # 生成参考数据
        task_id = "test_gen_ref_001"
        success, log, ref_bytes = await worker.generate_reference(
            package_data, task_id, op_name, timeout=30
        )
        
        # 验证结果
        assert success is True, f"Reference generation failed: {log}"
        assert "REFERENCE_GENERATION_SUCCESS" in log
        assert len(ref_bytes) > 0, "Reference data bytes should not be empty"
        
        # 验证 .pt 文件内容
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            f.write(ref_bytes)
            temp_path = f.name
        
        try:
            ref_data = torch.load(temp_path)
            assert 'op_name' in ref_data
            assert 'seed' in ref_data
            assert 'outputs' in ref_data
            assert ref_data['seed'] == 0
            assert len(ref_data['outputs']) > 0
            
            # 验证输出形状
            output = ref_data['outputs'][0]
            assert output.shape == (4, 16)  # batch_size=4, dim=16
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_local_worker_generate_reference_invalid_task_desc(self):
        """测试 LocalWorker 处理无效的 task_desc"""
        op_name = "test_invalid"
        invalid_task_desc = "this is not valid python code !!!"
        package_data = create_reference_generation_package(op_name, invalid_task_desc)
        
        device_pool = DevicePool([0])
        worker = LocalWorker(device_pool, backend="cpu")
        
        success, log, ref_bytes = await worker.generate_reference(
            package_data, "test_gen_ref_002", op_name, timeout=30
        )
        
        assert success is False
        assert ref_bytes == b''
    
    @pytest.mark.asyncio
    async def test_reference_data_reproducibility(self):
        """测试参考数据的可复现性（使用相同 seed 应产生相同结果）"""
        op_name = "test_repro"
        package_data = create_reference_generation_package(op_name, RELU_TASK_DESC)
        
        device_pool = DevicePool([0])
        worker = LocalWorker(device_pool, backend="cpu")
        
        # 生成两次参考数据
        success1, log1, ref_bytes1 = await worker.generate_reference(
            package_data, "test_repro_001", op_name, timeout=30
        )
        success2, log2, ref_bytes2 = await worker.generate_reference(
            package_data, "test_repro_002", op_name, timeout=30
        )
        
        assert success1 is True
        assert success2 is True
        
        # 加载并比较输出
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f1:
            f1.write(ref_bytes1)
            path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f2:
            f2.write(ref_bytes2)
            path2 = f2.name
        
        try:
            data1 = torch.load(path1)
            data2 = torch.load(path2)
            
            # 输出应该完全相同
            output1 = data1['outputs'][0]
            output2 = data2['outputs'][0]
            assert torch.allclose(output1, output2), "Outputs should be identical with same seed"
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestReferenceDataTransfer:
    """测试参考数据传输功能"""
    
    @pytest.mark.asyncio
    async def test_reference_bytes_serialization(self):
        """测试参考数据的序列化和反序列化"""
        import base64
        
        # 创建测试数据
        test_data = {
            'op_name': 'test_op',
            'seed': 0,
            'outputs': [torch.randn(4, 16)],
            'output_shapes': [(4, 16)],
        }
        
        # 保存到 bytes
        buffer = io.BytesIO()
        torch.save(test_data, buffer)
        original_bytes = buffer.getvalue()
        
        # Base64 编码（模拟 HTTP 传输）
        encoded = base64.b64encode(original_bytes).decode('utf-8')
        
        # Base64 解码
        decoded_bytes = base64.b64decode(encoded)
        
        # 验证数据完整性
        assert original_bytes == decoded_bytes
        
        # 加载并验证内容
        buffer2 = io.BytesIO(decoded_bytes)
        loaded_data = torch.load(buffer2)
        
        assert loaded_data['op_name'] == test_data['op_name']
        assert loaded_data['seed'] == test_data['seed']
        assert torch.allclose(loaded_data['outputs'][0], test_data['outputs'][0])
    
    @pytest.mark.asyncio
    async def test_config_reference_data_injection(self):
        """测试将参考数据注入到 config 中"""
        # 模拟生成参考数据
        op_name = "test_inject"
        package_data = create_reference_generation_package(op_name, RELU_TASK_DESC)
        
        device_pool = DevicePool([0])
        worker = LocalWorker(device_pool, backend="cpu")
        
        success, log, ref_bytes = await worker.generate_reference(
            package_data, "test_inject_001", op_name, timeout=30
        )
        
        assert success is True
        
        # 模拟 JobManager 将参考数据注入 config
        config = {
            'log_dir': '/tmp/aikg_test',
        }
        config['use_reference_data'] = True
        config['reference_data'] = ref_bytes
        
        # 验证 config 中的数据
        assert config['use_reference_data'] is True
        assert len(config['reference_data']) > 0
        
        # 模拟 KernelVerifier 从 config 读取并写入文件
        with tempfile.TemporaryDirectory() as verify_dir:
            ref_file = os.path.join(verify_dir, f"{op_name}_reference.pt")
            with open(ref_file, 'wb') as f:
                f.write(config['reference_data'])
            
            # 验证文件可以被正确加载
            assert os.path.exists(ref_file)
            loaded = torch.load(ref_file)
            assert 'outputs' in loaded
            assert loaded['seed'] == 0


class TestKernelVerifierGenerateReference:
    """测试 KernelVerifier.generate_reference_data 方法"""
    
    @pytest.mark.asyncio
    async def test_verifier_generate_reference_data(self):
        """测试 KernelVerifier 的 generate_reference_data 方法"""
        with tempfile.TemporaryDirectory() as log_dir:
            config = {'log_dir': log_dir}
            
            # 创建 Worker
            device_pool = DevicePool([0])
            worker = LocalWorker(device_pool, backend="cpu")
            
            # 创建 Verifier
            verifier = KernelVerifier(
                op_name="test_relu_verifier",
                framework_code=RELU_TASK_DESC,
                task_id="test_verifier_001",
                framework="torch",
                dsl="triton_cuda",  # dsl 在这个测试中不重要
                backend="cpu",
                arch="x86_64",
                config=config,
                worker=worker
            )
            
            # 生成参考数据
            success, log, ref_bytes = await verifier.generate_reference_data(
                RELU_TASK_DESC, timeout=60
            )
            
            # 验证成功
            assert success is True, f"generate_reference_data failed: {log}"
            assert len(ref_bytes) > 0, "Reference bytes should not be empty"
            
            # 验证 .pt 文件内容
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                f.write(ref_bytes)
                temp_path = f.name
            
            try:
                ref_data = torch.load(temp_path)
                assert 'op_name' in ref_data
                assert ref_data['op_name'] == "test_relu_verifier"
                assert 'outputs' in ref_data
                assert len(ref_data['outputs']) > 0
            finally:
                os.unlink(temp_path)


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])

