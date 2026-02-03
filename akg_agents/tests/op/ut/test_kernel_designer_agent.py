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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_kernel_designer_basic():
    """测试基本的 sketch 生成功能"""
    try:
        from akg_agents.core_v2.agents.kernel_designer import KernelDesigner
        
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
            dsl="triton_cuda",
            backend="cuda",
            arch="a100",
            task_id="test_vector_add_001"
        )
        
        logger.info("✓ Sketch generation completed")
        logger.info(f"{'='*60}\nGenerated Sketch:\n{'='*60}\n{sketch[:500]}...\n{'='*60}")
        
        # 验证输出是否为 sketch DSL 格式
        if 'sketch ' in sketch and '{' in sketch:
            logger.info("✓ Output is in sketch DSL format")
        else:
            logger.warning("⚠ Output may not be in correct sketch DSL format")
        
        if reasoning:
            logger.info(f"\nReasoning:\n{reasoning[:300]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_kernel_designer_with_inspirations():
    """测试带 inspirations 的 sketch 生成"""
    try:
        from akg_agents.core_v2.agents.kernel_designer import KernelDesigner
        
        agent = KernelDesigner()
        
        logger.info("✓ KernelDesigner agent created for softmax with inspirations")
        
        # 准备 inspirations（进化优化的参考方案）
        inspirations = [
            {
                'strategy_mode': 'baseline',
                'sketch': """sketch softmax {
  symbols: B, N;
  tensors: X[B, N]: f32; Y[B, N]: f32;
  
  N_tile = 512
  
  @llm_hint("parallel", "blockidx")
  for b in range(B):
    for n_outer in range(0, ceil(N, N_tile)):
      max_val = max(X[b, n_outer*N_tile:(n_outer+1)*N_tile])
      exp_sum = sum(exp(X[b, n_outer*N_tile:(n_outer+1)*N_tile] - max_val))
      Y[b, n_outer*N_tile:(n_outer+1)*N_tile] = exp(X[b, n_outer*N_tile:(n_outer+1)*N_tile] - max_val) / exp_sum
}""",
                'profile': {
                    'gen_time': 125.5,
                    'base_time': 200.0,
                    'speedup': 1.59
                },
                'is_parent': True  # 标记为父代方案
            },
            {
                'strategy_mode': 'optimized',
                'sketch': """sketch softmax {
  symbols: B, N;
  tensors: X[B, N]: f32; Y[B, N]: f32;
  
  N_tile = 1024
  
  @llm_hint("parallel", "blockidx")
  for b in range(B):
    for n_outer in range(0, ceil(N, N_tile)):
      max_val = max(X[b, n_outer*N_tile:(n_outer+1)*N_tile])
      exp_sum = sum(exp(X[b, n_outer*N_tile:(n_outer+1)*N_tile] - max_val))
      Y[b, n_outer*N_tile:(n_outer+1)*N_tile] = exp(X[b, n_outer*N_tile:(n_outer+1)*N_tile] - max_val) / exp_sum
}""",
                'profile': {
                    'gen_time': 98.3,
                    'base_time': 200.0,
                    'speedup': 2.03
                },
                'is_parent': False
            }
        ]
        
        logger.info("Running KernelDesigner agent with inspirations...")
        
        # 执行生成
        sketch, _, _ = await agent.run(
            op_name="softmax",
            task_desc="""
实现一个 softmax 算子：
- 输入：二维张量 (batch_size, seq_len)
- 输出：在最后一个维度上应用 softmax
- 使用数值稳定的实现（减去 max）
""",
            dsl="triton_cuda",
            backend="cuda",
            arch="a100",
            task_id="test_softmax_001",
            inspirations=inspirations
        )
        
        logger.info("✓ Sketch generation with inspirations completed")
        logger.info(f"{'='*60}\nGenerated Sketch (optimized):\n{'='*60}\n{sketch[:500]}...\n{'='*60}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_kernel_designer_hint_mode():
    """测试 Hint 模式（生成参数空间配置）"""
    try:
        from akg_agents.core_v2.agents.kernel_designer import KernelDesigner
        
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
        
        # 检查是否返回 JSON 格式（Hint 模式）
        try:
            result_dict = json.loads(result)
            
            if "code" in result_dict:
                logger.info(f"✓ Found 'code' field (sketch)")
                logger.info(f"Sketch preview:\n{result_dict['code'][:300]}...")
            
            if "space_config_code" in result_dict:
                logger.info(f"✓ Found 'space_config_code' field")
                logger.info(f"Space config preview:\n{result_dict['space_config_code'][:300]}...")
            else:
                logger.warning("⚠ No 'space_config_code' field found (expected in Hint mode)")
            
            return True
        
        except json.JSONDecodeError:
            logger.error(f"✗ Result is not JSON format: {result[:200]}...")
            return False
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def test_kernel_designer_triton_ascend():
    """测试 Triton Ascend 后端的 sketch 生成"""
    try:
        from akg_agents.core_v2.agents.kernel_designer import KernelDesigner
        
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
        logger.info(f"{'='*60}\nGenerated Sketch:\n{'='*60}\n{sketch[:500]}...\n{'='*60}")
        
        # 验证是否包含 Ascend 相关的优化提示
        if 'CUBE' in sketch or 'L0' in sketch or 'aicoreidx' in sketch:
            logger.info("✓ Sketch contains Ascend-specific optimizations")
        else:
            logger.info("ℹ Sketch may benefit from more Ascend-specific optimizations")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


async def main():
    """运行所有测试"""
    logger.info("="*60)
    logger.info("Testing KernelDesigner Agent")
    logger.info("="*60)
    
    tests = [
        ("Basic sketch generation", test_kernel_designer_basic),
        ("Sketch generation with inspirations", test_kernel_designer_with_inspirations),
        ("Hint mode (parameter space config)", test_kernel_designer_hint_mode),
        ("Triton Ascend backend", test_kernel_designer_triton_ascend)
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
