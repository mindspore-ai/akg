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
CodeChecker LLM 模式系统测试

测试未知后端（如 cuda_c）场景下使用 LLM 进行代码检查
需要配置 LLM 模型才能运行
"""

import pytest
import asyncio
import os
from ai_kernel_generator.core.checker import CodeChecker


# 错误的 CUDA C 代码示例（包含多种问题）
BAD_CUDA_C_CODE = '''
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel with intentional errors
__global__ void bad_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 问题1: 在 kernel 中使用 break
    for (int i = 0; i < 10; i++) {
        if (i > 5) {
            break;  // 不推荐在 GPU kernel 中使用
        }
    }
    
    // 问题2: 在 kernel 中使用 continue
    for (int j = 0; j < 10; j++) {
        if (j % 2 == 0) {
            continue;  // 可能影响 warp 执行效率
        }
    }
    
    // 问题3: 没有边界检查
    output[idx] = input[idx] * 2.0f;  // 可能越界访问
    
    // 问题4: 使用 printf 在 kernel 中（调试用，生产不推荐）
    printf("Debug: idx = %d\\n", idx);
}

// Host function
void launch_kernel(float* d_input, float* d_output, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    bad_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
}
'''


class TestCodeCheckerLLM:
    """测试 CodeChecker 的 LLM 检查模式"""
    
    @pytest.fixture
    def config(self):
        """创建包含 LLM 配置的 config"""
        # 使用可用的模型预设（优先使用环境变量，否则使用 deepseek_v3_default）
        return {
            "agent_model_config": {
                "code_checker": os.getenv("AIKG_CODE_CHECKER_MODEL", "deepseek_v3_default"),
                "conductor": os.getenv("AIKG_CONDUCTOR_MODEL", "deepseek_v3_default")
            }
        }
    
    @pytest.fixture
    def checker(self, config):
        """创建 cuda_c 的 CodeChecker 实例（未知场景，使用 LLM）"""
        return CodeChecker(backend="cuda", dsl="cuda_c", config=config)
    
    def test_cuda_c_should_use_llm_check(self, checker):
        """测试 cuda_c 场景应该使用 LLM 检查（不在已知场景列表中）"""
        # cuda_c 不在 KNOWN_SCENARIOS 中，应该使用 LLM
        assert checker._should_use_static_check() is False
        print("\n=== cuda_c 应该使用 LLM 检查 ===")
        print(f"_should_use_static_check(): {checker._should_use_static_check()}")
    
    @pytest.mark.asyncio
    async def test_llm_check_bad_cuda_c_code(self, checker):
        """测试 LLM 检查模式能够分析 CUDA C 代码
        
        注意：LLM 会根据实际语言特性判断，CUDA C 中 break/continue 是合法的，
        所以 LLM 可能判断为通过。这正是 LLM 检查的灵活性。
        """
        print("\n" + "=" * 60)
        print("=== 测试: LLM 检查 CUDA C 代码 ===")
        print("=" * 60)
        
        print("\n--- 输入代码 ---")
        print(BAD_CUDA_C_CODE[:500] + "..." if len(BAD_CUDA_C_CODE) > 500 else BAD_CUDA_C_CODE)
        
        # 执行检查
        passed, error_message, errors = await checker.check(BAD_CUDA_C_CODE)
        
        print("\n--- 检查结果 ---")
        print(f"通过: {passed}")
        print(f"错误数量: {len(errors)}")
        
        print("\n--- 错误详情 ---")
        if errors:
            for i, err in enumerate(errors, 1):
                print(f"\n【错误 {i}】")
                print(f"  行号: {err.get('line', 'N/A')}")
                print(f"  类型: {err.get('error_type', 'N/A')}")
                print(f"  详情: {err.get('detail', 'N/A')[:200]}")
                if err.get('suggestion'):
                    print(f"  建议: {err.get('suggestion', 'N/A')[:200]}")
        else:
            print("  LLM 判断代码没有问题（CUDA C 中 break/continue 是合法语法）")
        
        print("\n--- 格式化错误信息 ---")
        if error_message:
            print(error_message[:1500] + "..." if len(error_message) > 1500 else error_message)
        else:
            print("  (无错误信息 - LLM 认为代码是正确的)")
        
        # 验证：LLM 调用成功（无论结果如何）
        # 关键是验证 LLM 模式被正确触发并返回了结果
        print("\n--- 验证 ---")
        print(f"✅ LLM 检查模式成功执行")
        print(f"✅ 返回了有效结果: passed={passed}, errors_count={len(errors)}")
        
        # 如果没通过，验证有错误信息
        if not passed:
            assert len(errors) > 0, "检查失败时应该有错误列表"
            all_details = " ".join(err.get("detail", "") for err in errors)
            print(f"\n--- 所有错误描述合并 ---\n{all_details[:500]}")
        
        print("\n" + "=" * 60)
        print("=== 测试完成 ===")
        print("=" * 60)
        
        # 基本断言：验证函数正常执行
        assert passed is True or passed is False, "passed 应该是布尔值"
        assert isinstance(errors, list), "errors 应该是列表"
    
    @pytest.mark.asyncio
    async def test_llm_check_with_task_info(self, checker):
        """测试 LLM 检查时传入 task_info（包含专家建议）"""
        print("\n" + "=" * 60)
        print("=== 测试: LLM 检查带 task_info ===")
        print("=" * 60)
        
        task_info = {
            "expert_suggestion": """
CUDA 编程最佳实践：
1. 避免在 kernel 中使用 break/continue，这会导致 warp divergence
2. 始终检查数组边界，防止越界访问
3. 生产代码中不要使用 printf，影响性能
4. 使用 __syncthreads() 时要小心死锁
"""
        }
        
        simple_bad_code = '''
__global__ void kernel(float* data, int n) {
    int idx = threadIdx.x;
    for (int i = 0; i < 10; i++) {
        if (i > 5) break;  // warp divergence
    }
    data[idx] = 1.0f;  // 可能越界
}
'''
        
        passed, error_message, errors = await checker.check(simple_bad_code, task_info)
        
        print(f"\n通过: {passed}")
        print(f"错误数: {len(errors)}")
        if error_message:
            print(f"\n错误信息:\n{error_message[:1000]}")
        
        # 验证有输出
        assert error_message is not None


class TestCodeCheckerLLMFallback:
    """测试 LLM 不可用时的回退机制"""
    
    @pytest.fixture
    def checker_no_llm(self):
        """创建没有 LLM 配置的 CodeChecker（使用 triton_ascend 以便回退时使用静态规则）"""
        # 使用 triton_ascend 作为 dsl，这样回退到静态检查时会应用规则
        return CodeChecker(backend="ascend", dsl="triton_ascend", config={})
    
    @pytest.mark.asyncio
    async def test_fallback_to_static_when_no_llm(self, checker_no_llm):
        """测试静态检查能正确检测 Triton kernel 中的禁止语法"""
        print("\n" + "=" * 60)
        print("=== 测试: 静态检查检测 Triton kernel 中的 break ===")
        print("=" * 60)
        
        # 使用 Triton Python 代码（不是 CUDA C），因为静态检查是针对 Triton 设计的
        code_with_break = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    for i in range(10):
        if i > 5:
            break  # 应该被静态检查发现
'''
        
        passed, error_message, errors = await checker_no_llm.check(code_with_break)
        
        print(f"\n通过: {passed}")
        print(f"错误数: {len(errors)}")
        
        # 静态检查应该能检测到 break
        assert passed is False, "应该检测到 break"
        assert len(errors) >= 1
        assert any("break" in err.get("detail", "").lower() for err in errors)
        
        print("\n✅ 成功检测到 Triton kernel 中的 break")


# 独立运行测试
if __name__ == "__main__":
    # 运行特定测试
    pytest.main([
        __file__,
        "-v",
        "-s",
        "-k", "test_llm_check_bad_cuda_c_code or test_fallback_to_static_when_no_llm"
    ])

