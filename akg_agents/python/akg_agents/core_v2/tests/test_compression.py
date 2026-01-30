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
Action Compression 单元测试
"""

import pytest
import shutil
import tempfile
import asyncio
import json
from typing import Dict, List, Any
from pathlib import Path

from akg_agents.core_v2.filesystem import (
    TraceSystem,
    ActionRecord,
    ActionHistoryCompressed,
)
from akg_agents.core_v2.filesystem.compressor import ActionCompressor
from akg_agents.core_v2.llm.client import LLMClient

# Realistic Action Data Generators
def create_op_task_builder_action(action_id="act_001"):
    return ActionRecord(
        action_id=action_id,
        tool_name="call_op_task_builder",
        arguments={
            "user_request": "生成一个 ReLU 算子，输入 shape (1024, 1024), float16",
        },
        result={
            "task_code": "import torch\n...",
            "op_name": "ReluCustom"
        }
    )

def create_designer_action(action_id="act_002"):
    return ActionRecord(
        action_id=action_id,
        tool_name="call_designer",
        arguments={
            "op_name": "ReluCustom",
            "task_code": "..."
        },
        result={
            "design_doc": "## 算子设计\n采用 Vector 向量化计算..."
        }
    )

def create_coder_action(action_id="act_003"):
    return ActionRecord(
        action_id=action_id,
        tool_name="call_coder_only",
        arguments={
            "op_name": "ReluCustom",
            "guidelines": "使用 Vector 计算"
        },
        result={
            "kernel_code": "@triton.jit\ndef relu_kernel...",
            "success": True
        }
    )

def create_verifier_action(action_id="act_004"):
    return ActionRecord(
        action_id=action_id,
        tool_name="call_kernel_verifier",
        arguments={
            "kernel_code": "..."
        },
        result={
            "performance": 120.5,
            "accuracy": True,
            "error_msg": ""
        }
    )

# Mock LLM Client
class MockLLMClient:
    def __init__(self):
        self.generate_calls = []
        
    async def generate(self, messages, **kwargs):
        self.generate_calls.append({
            "messages": messages,
            "kwargs": kwargs
        })
        # Simulate a realistic summary
        return {
            "content": "用户请求生成 ReLU 算子。首先调用 op_task_builder 生成了 PyTorch 任务代码。接着 Designer 设计了基于向量化的实现方案。Coder 根据设计生成了初始 Triton 代码。",
            "usage": {"total_tokens": 150}
        }

@pytest.fixture
def mock_llm_client():
    return MockLLMClient()

class TestActionCompressor:
    """测试 ActionCompressor 类"""
    
    @pytest.mark.asyncio
    async def test_compress_short_history(self, mock_llm_client):
        """测试短历史不压缩"""
        compressor = ActionCompressor(mock_llm_client)
        
        # 模拟只有两步：TaskBuilder -> Designer
        history = [
            create_op_task_builder_action("a1"),
            create_designer_action("a2"),
        ]
        
        compressed = await compressor.compress_history(history)
        
        # 历史太短（<=3），应该原样返回
        assert len(compressed) == 2
        assert len(mock_llm_client.generate_calls) == 0
        
    @pytest.mark.asyncio
    async def test_compress_long_history(self, mock_llm_client):
        """测试长历史压缩"""
        compressor = ActionCompressor(mock_llm_client)
        
        # 模拟长历史：Builder -> Designer -> Coder -> Verifier -> Coder(Fix) -> Verifier
        history = [
            create_op_task_builder_action("a1"),
            create_designer_action("a2"),
            create_coder_action("a3"),
            ActionRecord("a4", "call_kernel_verifier", {"code": "..."}, {"error": "syntax error"}), # 第一次验证失败
            create_coder_action("a5"), # 修复
            create_verifier_action("a6") # 最终成功
        ]
        
        compressed = await compressor.compress_history(history)
        
        # 策略：保留最后 1 个动作 (Verifier)，前面的总结
        # 结果应该是 [SummaryAction, VerifierAction]
        assert len(compressed) == 2
        assert compressed[0].tool_name == "history_summary"
        assert compressed[0].result["_is_summary"] is True
        assert compressed[1].tool_name == "call_kernel_verifier"
        assert compressed[1].action_id == "a6"
        
        assert len(mock_llm_client.generate_calls) == 1
        
        # 验证 prompt 包含真实工具名
        call_args = mock_llm_client.generate_calls[0]
        prompt = call_args["messages"][1]["content"]
        assert "call_op_task_builder" in prompt
        assert "call_designer" in prompt
        assert "call_coder_only" in prompt


class TestTraceSystemCompression:
    """测试 TraceSystem 的压缩集成"""
    
    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
        
    @pytest.fixture
    def trace(self, temp_dir):
        trace = TraceSystem("test_compression_real", base_dir=temp_dir)
        trace.initialize()
        return trace
        
    @pytest.mark.asyncio
    async def test_get_compressed_history_flow(self, trace, mock_llm_client):
        """测试完整流程：从无缓存到有缓存"""
        # 1. 构建一个逼真的 Trace 路径
        # Root -> Builder -> Designer -> Coder -> Verifier
        
        # Node 1: Task Builder
        n1_action = {"type": "call_op_task_builder", "params": {"req": "ReLU"}}
        n1_result = {"task_code": "..."}
        node1 = trace.add_node(n1_action, n1_result)
        
        # Node 2: Designer
        trace.switch_node(node1)
        n2_action = {"type": "call_designer", "params": {"op": "Relu"}}
        n2_result = {"doc": "design..."}
        node2 = trace.add_node(n2_action, n2_result)
        
        # Node 3: Coder
        trace.switch_node(node2)
        n3_action = {"type": "call_coder_only", "params": {"design": "..."}}
        n3_result = {"code": "..."}
        node3 = trace.add_node(n3_action, n3_result)
        
        # Node 4: Verifier
        trace.switch_node(node3)
        n4_action = {"type": "call_kernel_verifier", "params": {"code": "..."}}
        n4_result = {"pass": True}
        node4 = trace.add_node(n4_action, n4_result)
        
        # 2. 获取压缩历史 (应该触发压缩，因为 len=4 > 3)
        compressed = await trace.get_compressed_history_for_llm(
            mock_llm_client, 
            node4, 
            max_tokens=2000
        )
        
        # 验证
        assert len(compressed) == 2
        # 第一个是 summary
        assert compressed[0].tool_name == "history_summary"
        summary_text = compressed[0].result["summary"]
        assert "用户请求生成 ReLU 算子" in summary_text
        
        # 第二个是最后一个动作 (Verifier)
        assert compressed[1].tool_name == "call_kernel_verifier"
        assert compressed[1].result["pass"] is True
        
        # 3. 验证缓存文件
        actions_dir = trace.fs.get_actions_dir(node4)
        assert (actions_dir / "action_history_compressed.json").exists()
        
        # 4. 再次获取 (应该命中缓存)
        # 清除 mock 记录
        mock_llm_client.generate_calls = []
        
        compressed_2 = await trace.get_compressed_history_for_llm(mock_llm_client, node4)
        
        assert len(compressed_2) == 2
        assert len(mock_llm_client.generate_calls) == 0  # 没调用 LLM
