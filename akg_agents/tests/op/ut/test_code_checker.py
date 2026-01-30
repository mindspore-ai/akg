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
CodeChecker 单元测试

测试 triton-ascend 场景下的静态代码检查功能
"""

import pytest
import asyncio
from akg_agents.core.checker import CodeChecker


class TestCodeCheckerTritonAscend:
    """测试 triton-ascend 场景的 CodeChecker"""
    
    @pytest.fixture
    def checker(self):
        """创建 triton-ascend 的 CodeChecker 实例"""
        return CodeChecker(backend="ascend", dsl="triton_ascend")
    
    def test_should_use_static_check(self, checker):
        """测试 triton-ascend 场景应该使用静态检查"""
        assert checker._should_use_static_check() is True
    
    @pytest.mark.asyncio
    async def test_code_with_break(self, checker):
        """测试包含 break 的代码应该被检测出来"""
        code_with_break = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    for i in range(10):
        if i > 5:
            break  # 这里使用了禁止的 break
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)
'''
        passed, error_message, errors = await checker.check(code_with_break)
        
        # 应该检测到 break
        assert passed is False
        assert len(errors) >= 1
        assert any("break" in err["detail"].lower() for err in errors)
        assert "break" in error_message.lower()
        
        print("\n=== 测试: 检测 break ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        print(f"错误信息:\n{error_message}")
    
    @pytest.mark.asyncio
    async def test_code_with_continue(self, checker):
        """测试包含 continue 的代码应该被检测出来"""
        code_with_continue = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    for i in range(10):
        if i % 2 == 0:
            continue  # 这里使用了禁止的 continue
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)
'''
        passed, error_message, errors = await checker.check(code_with_continue)
        
        # 应该检测到 continue
        assert passed is False
        assert len(errors) >= 1
        assert any("continue" in err["detail"].lower() for err in errors)
        assert "continue" in error_message.lower()
        
        print("\n=== 测试: 检测 continue ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        print(f"错误信息:\n{error_message}")
    
    @pytest.mark.asyncio
    async def test_code_with_break_and_continue(self, checker):
        """测试同时包含 break 和 continue 的代码"""
        code_with_both = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    for i in range(10):
        if i % 2 == 0:
            continue  # 禁止的 continue
        if i > 8:
            break  # 禁止的 break
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)
'''
        passed, error_message, errors = await checker.check(code_with_both)
        
        # 应该检测到 break 和 continue
        assert passed is False
        assert len(errors) >= 2
        
        error_types = [err["detail"].lower() for err in errors]
        has_break = any("break" in e for e in error_types)
        has_continue = any("continue" in e for e in error_types)
        
        assert has_break, "应该检测到 break"
        assert has_continue, "应该检测到 continue"
        
        print("\n=== 测试: 同时检测 break 和 continue ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        print(f"错误信息:\n{error_message}")
    
    @pytest.mark.asyncio
    async def test_code_with_while(self, checker):
        """测试包含 while 的代码应该被检测出来（Ascend 特有）"""
        code_with_while = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    i = 0
    while i < 10:  # Ascend 后端禁止 while
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)
        i += 1
'''
        passed, error_message, errors = await checker.check(code_with_while)
        
        # 应该检测到 while
        assert passed is False
        assert len(errors) >= 1
        assert any("while" in err["detail"].lower() for err in errors)
        
        print("\n=== 测试: 检测 while (Ascend 特有) ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        print(f"错误信息:\n{error_message}")
    
    @pytest.mark.asyncio
    async def test_code_with_return(self, checker):
        """测试包含 return 的代码应该被检测出来"""
        code_with_return = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid > 100:
        return  # 禁止的 return
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)
'''
        passed, error_message, errors = await checker.check(code_with_return)
        
        # 应该检测到 return
        assert passed is False
        assert len(errors) >= 1
        assert any("return" in err["detail"].lower() for err in errors)
        
        print("\n=== 测试: 检测 return ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        print(f"错误信息:\n{error_message}")
    
    @pytest.mark.asyncio
    async def test_clean_code_passes(self, checker):
        """测试符合规范的代码应该通过检查"""
        clean_code = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 使用 for 循环和 mask 控制流程（正确做法）
    for i in range(10):
        # 使用 mask 而非 break/continue
        valid_mask = mask & (i < 8)
        data = tl.load(input_ptr + offsets, mask=valid_mask, other=0.0)
        result = data * 2.0
        tl.store(output_ptr + offsets, result, mask=valid_mask)
'''
        passed, error_message, errors = await checker.check(clean_code)
        
        # 应该通过检查
        assert passed is True
        assert len(errors) == 0
        assert error_message == ""
        
        print("\n=== 测试: 规范代码应该通过 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
    
    @pytest.mark.asyncio
    async def test_break_in_comment_should_pass(self, checker):
        """测试注释中的 break 不应该被检测"""
        code_with_comment = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 注意：不要使用 break 或 continue，这在 Triton 中是禁止的
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)
'''
        passed, error_message, errors = await checker.check(code_with_comment)
        
        # 注释中的 break 不应该触发检测
        assert passed is True
        assert len(errors) == 0
        
        print("\n=== 测试: 注释中的关键字应该被忽略 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
    
    @pytest.mark.asyncio
    async def test_break_in_string_should_pass(self, checker):
        """测试字符串中的 break 不应该被检测"""
        code_with_string = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    This kernel processes data.
    Note: we don't use break or continue here.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)
'''
        passed, error_message, errors = await checker.check(code_with_string)
        
        # docstring 中的 break 不应该触发检测
        assert passed is True
        assert len(errors) == 0
        
        print("\n=== 测试: 字符串中的关键字应该被忽略 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
    
    @pytest.mark.asyncio
    async def test_empty_code(self, checker):
        """测试空代码应该通过"""
        passed, error_message, errors = await checker.check("")
        assert passed is True
        assert len(errors) == 0
        
        passed, error_message, errors = await checker.check("   \n\n  ")
        assert passed is True
        assert len(errors) == 0
        
        print("\n=== 测试: 空代码应该通过 ===")
        print(f"通过: {passed}")
    
    @pytest.mark.asyncio
    async def test_multiple_errors_with_line_numbers(self, checker):
        """测试多个错误应该报告正确的行号"""
        code_with_multiple_errors = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i = 0
    while i < 10:  # 第 10 行: while 错误
        if i % 2 == 0:
            continue  # 第 12 行: continue 错误
        if i > 8:
            break  # 第 14 行: break 错误
        i += 1
    
    if pid > 100:
        return  # 第 18 行: return 错误
'''
        passed, error_message, errors = await checker.check(code_with_multiple_errors)
        
        assert passed is False
        assert len(errors) >= 4  # while, continue, break, return
        
        # 验证每个错误都有行号
        for err in errors:
            assert "line" in err
            assert err["line"] > 0
        
        print("\n=== 测试: 多个错误应该报告行号 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        for err in errors:
            print(f"  Line {err['line']}: {err['detail'][:50]}...")
        print(f"\n格式化错误信息:\n{error_message}")


class TestCodeCheckerTritonCuda:
    """测试 triton-cuda 场景的 CodeChecker"""
    
    @pytest.fixture
    def checker(self):
        """创建 triton-cuda 的 CodeChecker 实例"""
        return CodeChecker(backend="cuda", dsl="triton_cuda")
    
    def test_should_use_static_check(self, checker):
        """测试 triton-cuda 场景应该使用静态检查"""
        assert checker._should_use_static_check() is True
    
    @pytest.mark.asyncio
    async def test_cuda_allows_while(self, checker):
        """测试 CUDA 后端不禁止 while（只有 Ascend 禁止）"""
        code_with_while = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    i = 0
    while i < 10:  # CUDA 应该允许 while
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)
        i += 1
'''
        passed, error_message, errors = await checker.check(code_with_while)
        
        # CUDA 后端应该允许 while
        # 检查是否有 while 相关的错误
        has_while_error = any("while" in err["detail"].lower() for err in errors)
        assert has_while_error is False, "CUDA 后端不应该禁止 while"
        
        print("\n=== 测试: CUDA 后端允许 while ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
    
    @pytest.mark.asyncio
    async def test_cuda_still_forbids_break(self, checker):
        """测试 CUDA 后端仍然禁止 break"""
        code_with_break = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    for i in range(10):
        if i > 5:
            break  # 所有后端都禁止 break
'''
        passed, error_message, errors = await checker.check(code_with_break)
        
        assert passed is False
        assert any("break" in err["detail"].lower() for err in errors)
        
        print("\n=== 测试: CUDA 后端仍然禁止 break ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")


class TestCodeCheckerUnknownBackend:
    """测试未知后端场景的 CodeChecker"""
    
    @pytest.fixture
    def checker(self):
        """创建未知后端的 CodeChecker 实例"""
        return CodeChecker(backend="unknown", dsl="unknown_dsl")
    
    def test_should_use_llm_check(self, checker):
        """测试未知后端应该使用 LLM 检查"""
        assert checker._should_use_static_check() is False
    
    @pytest.mark.asyncio
    async def test_unknown_backend_fallback_to_static(self, checker):
        """测试未知后端在 LLM 不可用时回退到静态检查"""
        code_with_break = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel():
    for i in range(10):
        if i > 5:
            break  # 应该被静态检查发现
'''
        # 由于没有配置 LLM，应该回退到静态检查
        passed, error_message, errors = await checker.check(code_with_break)
        
        # 即使是未知后端，回退到静态检查也应该检测到 break
        assert passed is False
        assert len(errors) >= 1
        
        print("\n=== 测试: 未知后端回退到静态检查 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")


class TestCodeCheckerErrorFormat:
    """测试错误信息格式化"""
    
    @pytest.fixture
    def checker(self):
        return CodeChecker(backend="ascend", dsl="triton_ascend")
    
    @pytest.mark.asyncio
    async def test_error_format_contains_required_fields(self, checker):
        """测试错误信息包含必要字段"""
        code_with_error = '''
@triton.jit
def my_kernel():
    for i in range(10):
        break
'''
        passed, error_message, errors = await checker.check(code_with_error)
        
        assert len(errors) >= 1
        for err in errors:
            assert "line" in err
            assert "error_type" in err
            assert "detail" in err
            assert "suggestion" in err
            assert "code_snippet" in err
        
        print("\n=== 测试: 错误信息包含必要字段 ===")
        for err in errors:
            print(f"  line: {err['line']}")
            print(f"  error_type: {err['error_type']}")
            print(f"  detail: {err['detail']}")
            print(f"  suggestion: {err['suggestion']}")
            print(f"  code_snippet: {err['code_snippet']}")
    
    @pytest.mark.asyncio
    async def test_formatted_error_message_is_readable(self, checker):
        """测试格式化的错误信息可读性"""
        code_with_errors = '''
@triton.jit
def my_kernel():
    for i in range(10):
        if i > 5:
            break
        continue
'''
        passed, error_message, errors = await checker.check(code_with_errors)
        
        assert passed is False
        # 格式化消息应该包含分隔线和问题编号
        assert "问题" in error_message or "错误" in error_message.lower()
        
        print("\n=== 测试: 格式化错误信息可读性 ===")
        print(error_message)


class TestPythonSyntaxCheck:
    """测试 Python 语法检查功能"""
    
    @pytest.fixture
    def checker(self):
        """创建 triton-ascend 的 CodeChecker 实例"""
        return CodeChecker(backend="ascend", dsl="triton_ascend")
    
    @pytest.mark.asyncio
    async def test_detect_unclosed_parenthesis(self, checker):
        """测试检测未闭合的括号"""
        code_with_syntax_error = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
'''
        passed, error_message, errors = await checker.check(code_with_syntax_error)
        
        print("\n=== 测试: 检测未闭合的括号 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        print(f"错误信息:\n{error_message}")
        
        assert passed is False
        assert len(errors) >= 1
        # 应该检测到语法错误
        assert any(err["error_type"] == "syntax_error" for err in errors)
    
    @pytest.mark.asyncio
    async def test_detect_indentation_error(self, checker):
        """测试检测缩进错误"""
        code_with_indent_error = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
  offsets = pid * BLOCK_SIZE  # 缩进错误
'''
        passed, error_message, errors = await checker.check(code_with_indent_error)
        
        print("\n=== 测试: 检测缩进错误 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        print(f"错误信息:\n{error_message}")
        
        assert passed is False
        assert len(errors) >= 1
        assert any(err["error_type"] == "syntax_error" for err in errors)
    
    @pytest.mark.asyncio
    async def test_syntax_error_combined_with_rule_error(self, checker):
        """测试语法错误和规则错误一起汇总"""
        # 这个代码既有语法错误（未闭合括号）又有规则错误（break）
        # 但由于语法错误，ast.parse 会失败，无法继续检测 break
        # 所以只会报告语法错误
        code_with_both_errors = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0
'''
        passed, error_message, errors = await checker.check(code_with_both_errors)
        
        print("\n=== 测试: 语法错误检测 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        print(f"错误信息:\n{error_message}")
        
        assert passed is False
        assert len(errors) >= 1
        # 应该有语法错误
        assert any(err["error_type"] == "syntax_error" for err in errors)
    
    @pytest.mark.asyncio
    async def test_valid_code_passes_syntax_check(self, checker):
        """测试正确的代码通过语法检查"""
        valid_code = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)
'''
        passed, error_message, errors = await checker.check(valid_code)
        
        print("\n=== 测试: 正确代码通过检查 ===")
        print(f"通过: {passed}")
        print(f"错误数: {len(errors)}")
        
        assert passed is True
        assert len(errors) == 0
    
    def test_check_python_syntax_method(self, checker):
        """测试 _check_python_syntax 方法"""
        # 测试未闭合括号
        code1 = "def foo(x:\n    return x"
        errors1 = checker._check_python_syntax(code1)
        assert len(errors1) == 1
        assert errors1[0]["error_type"] == "syntax_error"
        print(f"\n括号未闭合错误: {errors1[0]['detail']}")
        
        # 测试正确代码
        code2 = "def foo(x):\n    return x"
        errors2 = checker._check_python_syntax(code2)
        assert len(errors2) == 0
        print("正确代码: 无错误")


class TestComplexKernelScenarios:
    """测试复杂的 kernel 场景（多行定义、多装饰器、混合函数等）"""
    
    @pytest.fixture
    def checker(self):
        return CodeChecker(backend="ascend", dsl="triton_ascend")
    
    @pytest.mark.asyncio
    async def test_multiline_function_definition(self, checker):
        """测试多行函数定义中的 continue/break 能被检测"""
        code = '''
import triton
import triton.language as tl

@triton.jit
def complex_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride_x, stride_y, stride_z,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    for i in range(BLOCK_M):
        if i > 5: continue
        if i < 2: break
'''
        passed, error_message, errors = await checker.check(code)
        
        assert passed is False
        assert len(errors) == 2
        error_details = [e["detail"].lower() for e in errors]
        assert any("continue" in d for d in error_details)
        assert any("break" in d for d in error_details)
        
        print("\n=== 测试: 多行函数定义 ===")
        print(f"检测到 {len(errors)} 个错误 ✓")
    
    @pytest.mark.asyncio
    async def test_multiple_decorators(self, checker):
        """测试多个装饰器叠加的 kernel"""
        code = '''
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 64})],
    key=['n_elements'],
)
@triton.jit
def autotuned_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_M: tl.constexpr,
):
    for i in range(10):
        if i > 5: continue
'''
        passed, error_message, errors = await checker.check(code)
        
        assert passed is False
        assert len(errors) >= 1
        assert any("continue" in e["detail"].lower() for e in errors)
        
        print("\n=== 测试: 多装饰器叠加 ===")
        print(f"检测到 {len(errors)} 个错误 ✓")
    
    @pytest.mark.asyncio
    async def test_multiple_kernels_in_file(self, checker):
        """测试同一文件中多个 kernel 函数"""
        code = '''
import triton
import triton.language as tl

@triton.jit
def kernel_one(ptr, N: tl.constexpr):
    for i in range(N):
        if i > 5: continue  # 第一个 kernel 的 continue

@triton.jit
def kernel_two(ptr, M: tl.constexpr):
    for j in range(M):
        if j < 2: break     # 第二个 kernel 的 break
'''
        passed, error_message, errors = await checker.check(code)
        
        assert passed is False
        assert len(errors) == 2
        
        # 验证两个 kernel 的错误都被检测到
        error_lines = [e["line"] for e in errors]
        assert len(set(error_lines)) == 2  # 两个不同的行
        
        print("\n=== 测试: 多个 kernel 函数 ===")
        print(f"检测到 {len(errors)} 个错误，分别在行 {error_lines} ✓")
    
    @pytest.mark.asyncio
    async def test_kernel_and_helper_mixed(self, checker):
        """测试 kernel 和普通函数混合，只检测 kernel 内的错误"""
        code = '''
import triton
import triton.language as tl

def helper_function(x):
    """普通的 Python 函数"""
    for i in range(10):
        if i > 5:
            continue  # 普通函数中的 continue，不应该报错
        if i < 2:
            break     # 普通函数中的 break，不应该报错
    return x * 2

@triton.jit
def my_kernel(ptr, N: tl.constexpr):
    for i in range(N):
        if i > 5: continue  # kernel 中的 continue，应该报错

def launcher(x, y):
    """launcher 函数"""
    for i in range(len(x)):
        if i > 10:
            continue  # launcher 中的 continue，不应该报错
    my_kernel[(1,)](x, 64)
'''
        passed, error_message, errors = await checker.check(code)
        
        assert passed is False
        assert len(errors) == 1  # 只有 kernel 内的一个 continue
        assert "continue" in errors[0]["detail"].lower()
        
        print("\n=== 测试: kernel 和普通函数混合 ===")
        print(f"只检测到 kernel 内的 1 个错误（行 {errors[0]['line']}） ✓")
    
    @pytest.mark.asyncio
    async def test_decorator_with_arguments(self, checker):
        """测试带参数的装饰器"""
        code = '''
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['n_elements'])
def kernel_with_args(
    ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    for i in range(BLOCK):
        if i > 5: continue
'''
        passed, error_message, errors = await checker.check(code)
        
        assert passed is False
        assert len(errors) >= 1
        assert any("continue" in e["detail"].lower() for e in errors)
        
        print("\n=== 测试: 带参数的装饰器 ===")
        print(f"检测到 {len(errors)} 个错误 ✓")
    
    @pytest.mark.asyncio
    async def test_same_line_statements(self, checker):
        """测试同一行的各种写法"""
        code = '''
import triton
import triton.language as tl

@triton.jit
def kernel(ptr, N: tl.constexpr):
    for i in range(N):
        if i > 5: continue
        if i < 2: break
        if i == 3: return
'''
        passed, error_message, errors = await checker.check(code)
        
        assert passed is False
        assert len(errors) == 3  # continue, break, return
        
        error_types = [e["detail"].lower() for e in errors]
        assert any("continue" in t for t in error_types)
        assert any("break" in t for t in error_types)
        assert any("return" in t for t in error_types)
        
        print("\n=== 测试: 同一行的语句 ===")
        print(f"检测到 continue, break, return 共 {len(errors)} 个错误 ✓")
    
    @pytest.mark.asyncio
    async def test_ascend_specific_while(self, checker):
        """测试 Ascend 特有的规则（while）"""
        code = '''
import triton
import triton.language as tl

@triton.jit
def kernel(ptr, N: tl.constexpr):
    i = 0
    while i < 10:
        i += 1
'''
        passed, error_message, errors = await checker.check(code)
        
        assert passed is False
        assert len(errors) == 1  # while
        
        error_details = [e["detail"].lower() for e in errors]
        assert any("while" in d for d in error_details)
        
        print("\n=== 测试: Ascend 特有规则 (while) ===")
        print(f"检测到 while 共 {len(errors)} 个错误 ✓")
    
    @pytest.mark.asyncio
    async def test_kernel_range_detection_with_ast(self, checker):
        """直接测试 AST kernel 范围检测"""
        code = '''
import triton

def helper():
    pass

@triton.jit
def kernel_one(ptr):
    x = 1

@triton.jit
def kernel_two(ptr):
    y = 2

def another_helper():
    pass
'''
        ranges = checker._find_kernel_ranges(code)
        
        assert len(ranges) == 2  # 应该找到 2 个 kernel
        
        # 验证范围合理（kernel_one 和 kernel_two）
        print("\n=== 测试: AST kernel 范围检测 ===")
        print(f"检测到 {len(ranges)} 个 kernel 范围: {ranges} ✓")
    
    @pytest.mark.asyncio
    async def test_very_long_kernel(self, checker):
        """测试很长的 kernel 函数（模拟真实场景）"""
        # 构造一个较长的 kernel，中间有 continue
        body_lines = []
        for i in range(50):
            body_lines.append(f"    x{i} = tl.load(ptr + {i})")
        body_lines.insert(25, "    if condition: continue  # 在中间位置")
        
        code = f'''
import triton
import triton.language as tl

@triton.jit
def long_kernel(
    ptr,
    condition,
    BLOCK: tl.constexpr,
):
{chr(10).join(body_lines)}
'''
        passed, error_message, errors = await checker.check(code)
        
        assert passed is False
        assert len(errors) >= 1
        assert any("continue" in e["detail"].lower() for e in errors)
        
        print("\n=== 测试: 很长的 kernel 函数 ===")
        print(f"在 {len(code.splitlines())} 行代码中检测到 {len(errors)} 个错误 ✓")


# 运行测试的简便方法
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

