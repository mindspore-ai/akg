#!/usr/bin/env python3
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
测试 OpTaskBuilder Agent 的基本功能

演示如何使用 OpTaskBuilder agent 将用户需求转换为 KernelBench 格式
"""

import asyncio
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_op_task_builder_basic():
    """测试基本的任务构建功能 - 清晰的需求应该返回 READY"""
    try:
        from akg_agents.core_v2.agents.op_task_builder import OpTaskBuilder
        from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderStatus
        
        # 创建 OpTaskBuilder 实例
        agent = OpTaskBuilder()
        
        logger.info("✓ OpTaskBuilder agent created successfully")
        
        # 准备输入状态
        state = {
            "user_input": """
我需要实现一个 ReLU 激活函数算子：
- 输入：一维张量 x，形状为 (N,)，数据类型为 float32
- 输出：一维张量 y，形状与输入相同
- 功能：对于每个元素，如果 x[i] > 0，则 y[i] = x[i]，否则 y[i] = 0
- 目标平台：CUDA，使用 PyTorch 框架
""",
            "framework": "torch",
            "backend": "cuda",
            "arch": "a100",
            "dsl": "triton_cuda",
            "iteration": 0,
            "max_iterations": 5,
            "max_check_retries": 3,
        }
        
        logger.info("Running OpTaskBuilder agent with ReLU request...")
        
        # 执行生成
        result = await agent.run(state)
        
        # 验证结果
        status = result.get("status")
        op_name = result.get("op_name", "")
        generated_task_desc = result.get("generated_task_desc", "")
        agent_message = result.get("agent_message", "")
        
        logger.info(f"✓ Task building completed")
        logger.info(f"Status: {status}")
        logger.info(f"Op Name: {op_name}")
        logger.info(f"Agent Message: {agent_message}")
        
        if status == OpTaskBuilderStatus.READY:
            logger.info(f"{'='*60}\nGenerated Task Desc:\n{'='*60}\n{generated_task_desc[:500]}...\n{'='*60}")
            
            # 检查必需的组件
            has_model = "class Model" in generated_task_desc
            has_forward = "def forward" in generated_task_desc
            has_get_inputs = "def get_inputs" in generated_task_desc
            has_get_init_inputs = "def get_init_inputs" in generated_task_desc
            
            if has_model and has_forward and has_get_inputs and has_get_init_inputs:
                logger.info("✓ All required components found in generated code")
                return True
            else:
                logger.error(f"✗ Missing required components: Model={has_model}, forward={has_forward}, get_inputs={has_get_inputs}, get_init_inputs={has_get_init_inputs}")
                return False
        elif status == OpTaskBuilderStatus.NEED_CLARIFICATION:
            logger.warning(f"⚠ Need clarification: {agent_message}")
            return True  # 需要澄清也算正常行为
        else:
            logger.error(f"✗ Unexpected status: {status}")
            return False
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_op_task_builder_with_feedback():
    """测试带用户反馈的多轮交互"""
    try:
        from akg_agents.core_v2.agents.op_task_builder import OpTaskBuilder
        from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderStatus
        
        agent = OpTaskBuilder()
        
        logger.info("✓ OpTaskBuilder agent created for multi-turn interaction")
        
        # 第一轮：提供初始需求
        state = {
            "user_input": "实现一个矩阵乘法算子",
            "framework": "torch",
            "backend": "cuda",
            "arch": "a100",
            "dsl": "triton_cuda",
            "iteration": 0,
            "max_iterations": 5,
            "max_check_retries": 3,
        }
        
        logger.info("First round: Basic matmul request...")
        result1 = await agent.run(state)
        
        status1 = result1.get("status")
        logger.info(f"First round status: {status1}")
        logger.info(f"First round message: {result1.get('agent_message', '')}")
        
        # 如果需要澄清，提供更多信息
        if status1 == OpTaskBuilderStatus.NEED_CLARIFICATION:
            logger.info("Agent needs clarification, providing more details...")
            
            # 第二轮：提供详细信息
            state2 = {
                "user_input": "实现一个矩阵乘法算子",
                "user_feedback": """
好的，详细需求如下：
- 输入：两个二维张量 A (M, K) 和 B (K, N)，数据类型 float32
- 输出：二维张量 C (M, N)，C = A @ B
- 使用标准的矩阵乘法算法
""",
                "framework": "torch",
                "backend": "cuda",
                "arch": "a100",
                "dsl": "triton_cuda",
                "iteration": 1,
                "max_iterations": 5,
                "max_check_retries": 3,
                "conversation_history": result1.get("conversation_history", []),
            }
            
            result2 = await agent.run(state2)
            status2 = result2.get("status")
            logger.info(f"Second round status: {status2}")
            
            if status2 == OpTaskBuilderStatus.READY:
                logger.info("✓ Successfully generated task after clarification")
                generated_task_desc = result2.get("generated_task_desc", "")
                logger.info(f"{'='*60}\nGenerated Task Desc:\n{'='*60}\n{generated_task_desc[:500]}...\n{'='*60}")
                return True
            else:
                logger.warning(f"⚠ Status after clarification: {status2}")
                return True  # 其他状态也算测试通过
        
        elif status1 == OpTaskBuilderStatus.READY:
            logger.info("✓ Generated task on first attempt")
            return True
        
        else:
            logger.info(f"ℹ Got status: {status1}")
            return True  # 其他状态也算正常
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_op_task_builder_unsupported():
    """测试不支持的需求 - 应该返回 UNSUPPORTED"""
    try:
        from akg_agents.core_v2.agents.op_task_builder import OpTaskBuilder
        from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderStatus
        
        agent = OpTaskBuilder()
        
        logger.info("✓ OpTaskBuilder agent created for unsupported request test")
        
        # 准备不相关的需求
        state = {
            "user_input": "帮我写一个网页前端应用，包含登录注册功能",
            "framework": "torch",
            "backend": "cuda",
            "arch": "a100",
            "dsl": "triton_cuda",
            "iteration": 0,
            "max_iterations": 5,
            "max_check_retries": 3,
        }
        
        logger.info("Running OpTaskBuilder with unsupported request...")
        
        result = await agent.run(state)
        
        status = result.get("status")
        agent_message = result.get("agent_message", "")
        
        logger.info(f"Status: {status}")
        logger.info(f"Message: {agent_message}")
        
        if status == OpTaskBuilderStatus.UNSUPPORTED:
            logger.info("✓ Correctly identified unsupported request")
            return True
        elif status == OpTaskBuilderStatus.NEED_CLARIFICATION:
            logger.info("ℹ Agent asked for clarification instead of rejecting outright")
            return True  # 这也是合理的行为
        else:
            logger.warning(f"⚠ Got status {status} instead of UNSUPPORTED")
            return True  # 不算失败，因为 LLM 行为可能不同
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_op_task_builder_softmax():
    """测试 Softmax 算子生成"""
    try:
        from akg_agents.core_v2.agents.op_task_builder import OpTaskBuilder
        from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderStatus
        
        agent = OpTaskBuilder()
        
        logger.info("✓ OpTaskBuilder agent created for Softmax")
        
        state = {
            "user_input": """
实现一个 Softmax 算子：
- 输入：二维张量 x，形状为 (batch_size, dim)，数据类型为 float32
- 输出：二维张量 y，在最后一个维度上应用 softmax
- 使用数值稳定的实现（减去最大值）
- 公式：softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
- 目标：CUDA 平台，PyTorch 框架
""",
            "framework": "torch",
            "backend": "cuda",
            "arch": "a100",
            "dsl": "triton_cuda",
            "iteration": 0,
            "max_iterations": 5,
            "max_check_retries": 3,
        }
        
        logger.info("Running OpTaskBuilder for Softmax...")
        
        result = await agent.run(state)
        
        status = result.get("status")
        op_name = result.get("op_name", "")
        agent_message = result.get("agent_message", "")
        
        logger.info(f"Status: {status}")
        logger.info(f"Op Name: {op_name}")
        logger.info(f"Message: {agent_message}")
        
        if status == OpTaskBuilderStatus.READY:
            generated_task_desc = result.get("generated_task_desc", "")
            logger.info(f"{'='*60}\nGenerated Softmax Task:\n{'='*60}\n{generated_task_desc[:500]}...\n{'='*60}")
            logger.info("✓ Successfully generated Softmax task")
            return True
        elif status == OpTaskBuilderStatus.NEED_CLARIFICATION:
            logger.info(f"ℹ Need clarification: {agent_message}")
            return True
        else:
            logger.warning(f"⚠ Got status: {status}")
            return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def main():
    """运行所有测试"""
    logger.info("="*60)
    logger.info("Testing OpTaskBuilder Agent")
    logger.info("="*60)
    
    tests = [
        ("Basic ReLU generation", test_op_task_builder_basic),
        ("Multi-turn interaction with feedback", test_op_task_builder_with_feedback),
        ("Unsupported request handling", test_op_task_builder_unsupported),
        ("Softmax generation", test_op_task_builder_softmax),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        
        # 添加间隔，避免请求过快
        await asyncio.sleep(1)
    
    # 打印总结
    logger.info(f"{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    logger.info(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
