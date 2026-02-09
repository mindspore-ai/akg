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
测试 KernelDesigner Agent 的基本功能

演示如何使用新的 KernelDesigner agent 生成算法草图
"""

import asyncio
import logging
import json

from akg_agents.utils.common_utils import ParserFactory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_code(raw_output: str, keys: list[str] = ["code"]):
    """打印 code 结果"""
    try:
        extracted_json = ParserFactory._extract_json_comprehensive(raw_output)
        if extracted_json:
            parsed = json.loads(extracted_json)
            if isinstance(parsed, dict) and all(key in parsed for key in keys):
                parsed = {key: parsed[key] for key in keys}
    except (json.JSONDecodeError, Exception):
        raise ValueError(f"Failed to extract code from raw output: {raw_output}")

    print(f"{'=' * 50}")
    print(f"\n{'-' * 50}\n".join([f"📋 [{k.upper()}]\n{v}" for k, v in parsed.items()]))
    print("=" * 50)


async def test_kernel_designer_basic():
    """测试基本的 sketch 生成功能"""
    try:
        from akg_agents.op.agents import KernelDesigner
        
        # 创建 KernelDesigner 实例（新 API：不需要业务参数）
        agent = KernelDesigner()
        
        logger.info("✓ KernelDesigner agent created successfully")
        
        logger.info("Running KernelDesigner agent...")
        
        # 执行生成（新 API：直接传参数而不是 dict）
        sketch, formatted_prompt, reasoning = await agent.run(
            op_name="vector_add",
            task_desc="""
实现一个简单的向量加法算子：
- 输入：两个大小为 N 的一维张量 A 和 B
- 输出：张量 C = A + B
- 要求：
  * 使用合适的 tiling 策略
  * 考虑 GPU 并行化
  * 标注并行化的机会
""",
            dsl="triton_ascend",
            backend="ascend",
            arch="ascend910b4",
            task_id="test_vector_add_001"
        )
        
        logger.info("✓ Sketch generation completed")
        print_code(sketch, ["code"])
        
        # 验证输出是否为 sketch DSL 格式
        if 'sketch ' in sketch and '{' in sketch:
            logger.info("✓ Output is in sketch DSL format")
        else:
            logger.warning("⚠ Output may not be in correct sketch DSL format")
        
        if reasoning:
            logger.info(f"\n💡 Reasoning (前300字符):\n{reasoning[:300]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_kernel_designer_hint_mode():
    """测试 Hint 模式（生成参数空间配置）"""
    try:
        from akg_agents.op.agents import KernelDesigner
        
        agent = KernelDesigner()
        
        logger.info("✓ KernelDesigner agent created for matmul with Hint mode")
        
        logger.info("Running KernelDesigner agent in Hint mode...")
        
        # 执行生成（Hint 模式）
        result, _, reasoning = await agent.run(
            op_name="matmul",
            task_desc="""
实现矩阵乘法：
- 输入：A (M x K) 和 B (K x N)
- 输出：C = A @ B
- @hint: 需要优化 BLOCK_M, BLOCK_K, BLOCK_N 参数
- @hint: BLOCK_M in [64, 128, 256]
- @hint: BLOCK_K in [32, 64, 128]
- @hint: BLOCK_N in [64, 128, 256]
""",
            dsl="triton_cuda",
            backend="cuda",
            arch="a100",
            task_id="test_matmul_hint",
            enable_hint_mode=True
        )
        
        logger.info("✓ Sketch generation in Hint mode completed")
        print_code(result, ["code", "space_config"])
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_kernel_designer_triton_ascend():
    """测试 Triton Ascend 后端的 sketch 生成"""
    try:
        from akg_agents.op.agents import KernelDesigner
        
        agent = KernelDesigner()
        
        logger.info("✓ KernelDesigner agent created for Triton Ascend")
        
        logger.info("Running KernelDesigner agent for Triton Ascend...")
        
        # 执行生成（Triton Ascend）
        sketch, _, _ = await agent.run(
            op_name="layernorm",
            task_desc="""
实现 LayerNorm 算子：
- 输入：X (batch_size, seq_len, hidden_size)
- 输出：Y = (X - mean) / sqrt(var + eps)
- 要求：
  * 考虑 Ascend NPU 的 CUBE 和 Vector 指令
  * 优化 L0 Buffer 使用
  * 标注并行化策略
""",
            dsl="triton_ascend",
            backend="ascend",
            arch="ascend910b4",
            task_id="test_layernorm_ascend"
        )
        
        logger.info("✓ Sketch generation for Triton Ascend completed")
        print_code(sketch, ["code"])
        
        # 验证是否包含 Ascend 相关的优化提示
        ascend_keywords = ['CUBE', 'L0', 'aicoreidx', 'coreidx', 'l1_buffer', 'l0c']
        found_keywords = [kw for kw in ascend_keywords if kw.lower() in sketch.lower()]
        if found_keywords:
            logger.info(f"✓ Sketch contains Ascend-specific optimizations: {found_keywords}")
        else:
            logger.info("ℹ Sketch may benefit from more Ascend-specific optimizations")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def main():
    """运行所有测试"""
    logger.info("="*50)
    logger.info("Testing KernelDesigner Agent")
    logger.info("="*50)
    
    tests = [
        ("Basic sketch generation", test_kernel_designer_basic),
        ("Hint mode (parameter space config)", test_kernel_designer_hint_mode),
        ("Triton Ascend backend", test_kernel_designer_triton_ascend)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"{'='*50}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # 打印总结
    logger.info(f"{'='*50}")
    logger.info("Test Summary")
    logger.info(f"{'='*50}")
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    logger.info(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
