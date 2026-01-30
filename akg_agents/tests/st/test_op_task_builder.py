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

"""OpTaskBuilderAgent ST测试用例

测试多轮交互将用户文字需求转换为KernelBench格式的完整流程。

运行方式：
    cd akg_agents && source env.sh
    pytest tests/st/test_op_task_builder.py -v -s
"""

import pytest
import os

from akg_agents.op.workflows.op_task_builder_workflow import OpTaskBuilderWorkflow, run_op_task_builder
from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderStatus
from akg_agents.utils.common_utils import ParserFactory
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_remote_worker

cuda_worker_url = os.environ.get("CUDA_WORKER_URL", "http://localhost:9001")


async def ensure_worker_registered():
    """确保 worker 已注册（如果已注册则跳过）"""
    from akg_agents.core.worker.manager import get_worker_manager
    if not await get_worker_manager().has_worker(backend="cuda", arch="a100"):
        await register_remote_worker(backend="cuda", arch="a100", worker_url=cuda_worker_url)


def get_test_config() -> dict:
    """获取测试配置
    
    优先使用vllm_triton_cuda_coderonly_config.yaml，
    如果加载失败则使用最小化测试配置
    """
    try:
        config = load_config(config_path="./python/akg_agents/op/config/vllm_triton_cuda_coderonly_config.yaml")
        # 确保op_task_builder模型配置存在（使用coder模型作为fallback）
        if "agent_model_config" in config and "op_task_builder" not in config["agent_model_config"]:
            config["agent_model_config"]["op_task_builder"] = config["agent_model_config"].get("coder", "default")
        return config
    except Exception as e:
        print(f"Warning: Failed to load config file: {e}, using minimal config")
        return {
            "agent_model_config": {
                "op_task_builder": "default",
                "coder": "default",
            },
            "log_dir": "/tmp/akg_agents_test",
            "op_task_builder_max_iterations": 5,
        }


class TestOpTaskBuilderParser:
    """测试OpTaskBuilder解析器"""
    
    def test_parser_creation(self):
        """测试解析器创建"""
        parser = ParserFactory.get_op_task_builder_parser()
        assert parser is not None
        
        # 检查format_instructions
        instructions = parser.get_format_instructions()
        assert "op_name" in instructions
        assert "status" in instructions
        assert "task_code" in instructions
    
    def test_parser_parse_ready(self):
        """测试解析ready状态的输出"""
        parser = ParserFactory.get_op_task_builder_parser()
        
        # 模拟LLM输出
        llm_output = '''```json
{
    "op_name": "relu",
    "status": "ready",
    "task_code": "import torch\\nimport torch.nn as nn\\n\\nclass Model(nn.Module):\\n    def __init__(self):\\n        super(Model, self).__init__()\\n    \\n    def forward(self, x):\\n        return torch.relu(x)\\n\\nbatch_size = 16\\ndim = 16384\\n\\ndef get_inputs():\\n    return [torch.randn(batch_size, dim)]\\n\\ndef get_init_inputs():\\n    return []",
    "description": "已生成ReLU算子代码",
    "reasoning": "用户需要ReLU激活函数，这是一个简单的元素级操作"
}
```'''
        
        result = ParserFactory.robust_parse(llm_output, parser)
        assert result is not None
        assert result.op_name == "relu"
        assert result.status == "ready"
        assert "class Model" in result.task_code
    
    def test_parser_parse_need_clarification(self):
        """测试解析need_clarification状态的输出"""
        parser = ParserFactory.get_op_task_builder_parser()
        
        llm_output = '''```json
{
    "op_name": "",
    "status": "need_clarification",
    "task_code": "",
    "description": "请问您需要的矩阵乘法的输入形状是什么？例如：两个1024x1024的方阵",
    "reasoning": "用户只说了矩阵乘法，没有指定具体的形状"
}
```'''
        
        result = ParserFactory.robust_parse(llm_output, parser)
        assert result is not None
        assert result.status == "need_clarification"
        assert "形状" in result.description


class TestOpTaskBuilderStaticCheck:
    """测试静态检查功能"""
    
    def test_valid_task_desc(self):
        """测试有效的task_desc"""
        from akg_agents.core.agent.op_task_builder import OpTaskBuilder
        
        config = get_test_config()
        agent = OpTaskBuilder(config)
        
        valid_code = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
'''
        
        passed, error = agent._check_task_desc_static(valid_code)
        assert passed, f"Valid code should pass static check, but got error: {error}"
    
    def test_missing_model_class(self):
        """测试缺少Model类"""
        from akg_agents.core.agent.op_task_builder import OpTaskBuilder
        
        config = get_test_config()
        agent = OpTaskBuilder(config)
        
        invalid_code = '''import torch

def get_inputs():
    return [torch.randn(16, 16384)]

def get_init_inputs():
    return []
'''
        
        passed, error = agent._check_task_desc_static(invalid_code)
        assert not passed
        assert "Model" in error
    
    def test_missing_forward_method(self):
        """测试缺少forward方法"""
        from akg_agents.core.agent.op_task_builder import OpTaskBuilder
        
        config = get_test_config()
        agent = OpTaskBuilder(config)
        
        invalid_code = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def compute(self, x):  # 错误：应该是forward
        return torch.relu(x)

def get_inputs():
    return [torch.randn(16, 16384)]

def get_init_inputs():
    return []
'''
        
        passed, error = agent._check_task_desc_static(invalid_code)
        assert not passed
        assert "forward" in error
    
    def test_missing_get_inputs(self):
        """测试缺少get_inputs函数"""
        from akg_agents.core.agent.op_task_builder import OpTaskBuilder
        
        config = get_test_config()
        agent = OpTaskBuilder(config)
        
        invalid_code = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return torch.relu(x)

def get_init_inputs():
    return []
'''
        
        passed, error = agent._check_task_desc_static(invalid_code)
        assert not passed
        assert "get_inputs" in error
    
    def test_syntax_error(self):
        """测试语法错误"""
        from akg_agents.core.agent.op_task_builder import OpTaskBuilder
        
        config = get_test_config()
        agent = OpTaskBuilder(config)
        
        invalid_code = '''import torch
class Model(nn.Module)
    def forward(self, x):
        return x
'''
        
        passed, error = agent._check_task_desc_static(invalid_code)
        assert not passed
        assert "Syntax error" in error or "Error" in error


class TestOpTaskBuilderWorkflow:
    """测试OpTaskBuilderWorkflow"""
    
    def test_workflow_creation(self):
        """测试workflow创建"""
        config = get_test_config()
        workflow = OpTaskBuilderWorkflow(config)
        
        assert workflow.op_task_builder_agent is not None
        assert workflow.max_iterations == config.get("op_task_builder_max_iterations", 5)
    
    def test_workflow_graph_build(self):
        """测试workflow图构建"""
        config = get_test_config()
        workflow = OpTaskBuilderWorkflow(config)
        
        graph = workflow.build_graph()
        assert graph is not None
        
        # 编译图
        app = workflow.compile()
        assert app is not None
    
    def test_workflow_visualize(self):
        """测试workflow可视化"""
        config = get_test_config()
        workflow = OpTaskBuilderWorkflow(config)
        
        mermaid = workflow.visualize()
        assert mermaid is not None
        assert "op_task_builder" in mermaid


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_simple_relu_request():
    """测试简单需求：ReLU算子（一轮直接生成）
    
    用户输入明确的需求，应该能一轮生成ready的task_desc
    """
    await ensure_worker_registered()
    config = get_test_config()
    
    result = await run_op_task_builder(
        user_input="我需要一个ReLU激活函数的算子，输入是16x16384的张量",
        config=config,
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Test Result ===")
    print(f"Status: {result.get('status')}")
    print(f"Op Name: {result.get('op_name')}")
    print(f"Message: {result.get('agent_message')}")
    
    # 验证结果
    assert result.get("status") in [OpTaskBuilderStatus.READY, OpTaskBuilderStatus.NEED_CLARIFICATION]
    
    if result.get("status") == OpTaskBuilderStatus.READY:
        assert result.get("op_name") is not None
        assert result.get("generated_task_desc") is not None
        assert "class Model" in result.get("generated_task_desc", "")
        print(f"\n=== Generated Task Desc ===")
        print(result.get("generated_task_desc"))


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_matmul_request():
    """测试矩阵乘法需求"""
    await ensure_worker_registered()
    config = get_test_config()
    
    result = await run_op_task_builder(
        user_input="实现两个1024x1024矩阵的乘法运算",
        config=config,
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Test Result ===")
    print(f"Status: {result.get('status')}")
    print(f"Op Name: {result.get('op_name')}")
    print(f"Message: {result.get('agent_message')}")
    
    assert result.get("status") in [OpTaskBuilderStatus.READY, OpTaskBuilderStatus.NEED_CLARIFICATION]


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_layernorm_request():
    """测试LayerNorm需求"""
    await ensure_worker_registered()
    config = get_test_config()
    
    result = await run_op_task_builder(
        user_input="我需要一个Layer Normalization层，处理shape为(batch=32, seq=512, hidden=768)的输入",
        config=config,
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Test Result ===")
    print(f"Status: {result.get('status')}")
    print(f"Op Name: {result.get('op_name')}")
    print(f"Message: {result.get('agent_message')}")
    
    assert result.get("status") in [OpTaskBuilderStatus.READY, OpTaskBuilderStatus.NEED_CLARIFICATION]


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_vague_request_need_clarification():
    """测试模糊需求：应该返回need_clarification"""
    await ensure_worker_registered()
    config = get_test_config()
    
    result = await run_op_task_builder(
        user_input="帮我优化一下性能",  # 模糊需求
        config=config,
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Test Result ===")
    print(f"Status: {result.get('status')}")
    print(f"Message: {result.get('agent_message')}")
    
    # 模糊需求应该返回need_clarification或unsupported
    assert result.get("status") in [OpTaskBuilderStatus.NEED_CLARIFICATION, OpTaskBuilderStatus.UNSUPPORTED]
    # 应该有提示消息
    assert result.get("agent_message") is not None and len(result.get("agent_message", "")) > 0


@pytest.mark.level0
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_unsupported_request():
    """测试不支持的需求：应该返回unsupported"""
    await ensure_worker_registered()
    config = get_test_config()
    
    result = await run_op_task_builder(
        user_input="帮我写一个网页登录功能",  # 非算子需求
        config=config,
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Test Result ===")
    print(f"Status: {result.get('status')}")
    print(f"Message: {result.get('agent_message')}")
    
    # 非算子需求应该返回unsupported
    assert result.get("status") == OpTaskBuilderStatus.UNSUPPORTED
    assert result.get("agent_message") is not None


@pytest.mark.level1
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_multi_turn_interaction():
    """测试多轮交互流程
    
    模拟用户先给出模糊需求，然后补充信息的场景
    """
    await ensure_worker_registered()
    config = get_test_config()
    workflow = OpTaskBuilderWorkflow(config)
    
    # 第一轮：模糊需求
    result1 = await workflow.run(
        user_input="矩阵乘法",
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Round 1 ===")
    print(f"Status: {result1.get('status')}")
    print(f"Message: {result1.get('agent_message')}")
    
    # 如果第一轮就ready了，测试也通过
    if result1.get("status") == OpTaskBuilderStatus.READY:
        print("First round already returned READY")
        return
    
    # 第二轮：用户补充信息
    result2 = await workflow.run(
        user_input="矩阵乘法",
        user_feedback="两个1024x1024的方阵相乘，使用float32类型",
        previous_state=result1,
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Round 2 ===")
    print(f"Status: {result2.get('status')}")
    print(f"Message: {result2.get('agent_message')}")
    
    # 第二轮应该更接近ready
    assert result2.get("iteration", 0) > result1.get("iteration", 0)


@pytest.mark.level1
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_softmax_request():
    """测试Softmax需求"""
    await ensure_worker_registered()
    config = get_test_config()
    
    result = await run_op_task_builder(
        user_input="实现一个Softmax函数，输入shape是(batch=64, classes=1000)",
        config=config,
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Test Result ===")
    print(f"Status: {result.get('status')}")
    print(f"Op Name: {result.get('op_name')}")
    print(f"Message: {result.get('agent_message')}")
    
    assert result.get("status") in [OpTaskBuilderStatus.READY, OpTaskBuilderStatus.NEED_CLARIFICATION]


@pytest.mark.level1
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_workflow_run_until_ready():
    """测试自动多轮交互直到ready
    
    使用mock的用户反馈回调
    """
    await ensure_worker_registered()
    config = get_test_config()
    workflow = OpTaskBuilderWorkflow(config)
    
    # 模拟用户反馈
    feedback_count = [0]
    
    async def mock_get_feedback(message: str) -> str:
        feedback_count[0] += 1
        print(f"\n[Mock User] Received message: {message}")
        
        if feedback_count[0] == 1:
            return "输入shape是(16, 16384)的float32张量"
        elif feedback_count[0] == 2:
            return "确认，就是这样"
        else:
            return ""  # 不再提供反馈
    
    result = await workflow.run_until_ready(
        user_input="我要一个简单的ReLU",
        get_user_feedback_callback=mock_get_feedback,
        framework="torch",
        backend="cuda",
        arch="a100"
    )
    
    print(f"\n=== Final Result ===")
    print(f"Status: {result.get('status')}")
    print(f"Total rounds: {result.get('iteration', 0)}")
    print(f"Op Name: {result.get('op_name')}")
    
    # 最终应该达到某个终止状态
    assert result.get("status") in [
        OpTaskBuilderStatus.READY, 
        OpTaskBuilderStatus.UNSUPPORTED, 
        OpTaskBuilderStatus.NEED_CLARIFICATION,
        OpTaskBuilderStatus.NEED_MODIFICATION
    ]


# 运行所有测试的入口
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-k", "test_simple_relu_request"])
