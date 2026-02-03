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
测试 KernelGen Agent 的基本功能

演示如何使用新的 KernelGen agent 生成内核代码
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


async def test_kernel_gen_basic():
    """测试基本的 kernel 生成功能"""
    try:
        # 导入 KernelGen 和 ActionRecord
        from akg_agents.core_v2.agents.kernel_gen import KernelGen
        from akg_agents.core_v2.filesystem import ActionRecord
        
        # 创建 KernelGen 实例（新 API：不需要业务参数）
        agent = KernelGen()
        
        logger.info("✓ KernelGen agent created successfully")
        
        # 准备历史（使用 ActionRecord 对象）
        history_compress = [
            ActionRecord(
                action_id="summary",
                tool_name="history_summary",
                arguments={},  # 必须提供，即使为空
                result={"summary": "用户请求实现一个向量加法算子，目标是 Triton Ascend"}
            ),
            ActionRecord(
                action_id="act_001",
                tool_name="op_task_build",
                arguments={"user_input": "向量加法"},
                result={"task_spec": "..."}
            ),
        ]
        
        logger.info("Running KernelGen agent...")
        
        # 执行生成（新 API：直接传参数而不是 dict）
        generated_code, formatted_prompt, reasoning = await agent.run(
            op_name="vector_add",
            task_desc="""
实现一个简单的向量加法内核：
- 输入：两个大小为 N 的一维张量 A 和 B
- 输出：张量 C = A + B
- 要求：
  * 处理任意大小
  * 使用高效的内存访问模式
  * 包含边界检查
""",
            dsl="triton_ascend",
            framework="torch",
            backend="ascend",
            arch="ascend910b4",
            task_id="test_vector_add_001",
            history_compress=history_compress
        )
        
        logger.info("✓ Code generation completed")
        logger.info(f"{'='*60}\nGenerated Code:\n{'='*60}\n{generated_code[:500]}...\n{'='*60}")
        
        if reasoning:
            logger.info(f"\nReasoning:\n{reasoning[:300]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_kernel_gen_with_error():
    """测试带错误反馈的迭代生成"""
    try:
        from akg_agents.core_v2.agents.kernel_gen import KernelGen
        from akg_agents.core_v2.filesystem import ActionRecord
        
        # 创建 KernelGen 实例（新 API：不需要业务参数）
        agent = KernelGen()
        
        logger.info("✓ KernelGen agent created for softmax")
        
        # 准备历史（使用 ActionRecord 对象，包含所有必需字段）
        history_compress = [
            ActionRecord(
                action_id="summary1",
                tool_name="history_summary",
                arguments={},  # 必须提供
                result={"summary": "用户请求实现 softmax 算子，第一次生成的代码有编译错误"}
            ),
            ActionRecord(
                action_id="act_001",
                tool_name="kernel_gen",
                arguments={"task_desc": "softmax implementation"},
                result={"code": "code_v1"}
            ),
            ActionRecord(
                action_id="act_002",
                tool_name="verifier",
                arguments={"code": "code_v1"},
                result={
                    "passed": "False",  # 使用字符串而不是布尔值，避免 truncate 过滤器失败
                    "error": "Error: Compilation failed\n  Line 42: undefined variable 'max_val'\n  Hint: You need to compute max before exp"
                }
            ),
        ]
        
        logger.info("Running KernelGen agent with error feedback...")
        
        # 执行生成（新 API：直接传参数）
        generated_code, _, _ = await agent.run(
            op_name="softmax",
            task_desc="""
实现一个 softmax 内核：
- 输入：二维张量 (batch_size, seq_len)
- 输出：在最后一个维度上应用 softmax
- 使用数值稳定的实现
""",
            dsl="triton_cuda",
            framework="torch",
            backend="cuda",
            arch="a100",
            user_requirements="修复之前的编译错误：需要先计算 max_val",
            task_id="test_softmax_001",
            history_compress=history_compress
        )
        
        logger.info("✓ Code generation with error feedback completed")
        logger.info(f"{'='*60}\nGenerated Code (v2):\n{'='*60}\n{generated_code[:500]}...\n{'='*60}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def main():
    """运行所有测试"""
    logger.info("="*60)
    logger.info("Testing KernelGen Agent")
    logger.info("="*60)
    
    tests = [
        ("Basic generation", test_kernel_gen_basic),
        ("Generation with error feedback", test_kernel_gen_with_error)
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
