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

测试纯静态检查流程：
1. ast.parse 语法检查
2. py_compile 编译检查
3. import 可用性检查
4. 中文文本混入检测
5. DSL 合规性检测（反作弊）
"""

import pytest
from akg_agents.op.utils.code_checker import CodeChecker


@pytest.fixture
def checker():
    """triton_cuda checker — 会启用 DSL 合规性检测"""
    return CodeChecker(backend="cuda", dsl="triton_cuda")


@pytest.fixture
def checker_no_dsl():
    """dsl='torch' checker — 跳过 DSL 合规性检测，用于测试语法/import/中文等基础功能"""
    return CodeChecker(backend="cuda", dsl="torch")


# ============================================================
# 语法错误检测 (ast.parse)
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_unclosed_parenthesis(checker):
    """括号不匹配应被检测"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert len(errors) == 1
    assert errors[0]["error_type"] == "syntax_error"
    assert errors[0]["line"] == 10


@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_indentation_error(checker):
    """缩进错误应被检测"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert len(errors) == 1
    assert errors[0]["error_type"] == "syntax_error"
    assert "indent" in errors[0]["detail"].lower()


@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_missing_colon(checker):
    """if 后缺少冒号应被检测"""
    code = '''\
import torch

def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE=1024):
    pid = 0
    offsets = pid * BLOCK_SIZE
    mask = offsets < n_elements
    x = 1.0
    y = 2.0
    if mask
        out = x + y
    return out
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"
    assert errors[0]["line"] == 9


@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_fstring_syntax_error(checker):
    """f-string 语法错误应被检测"""
    code = '''\
import torch

def launch(x, eps=1e-5):
    N = x.shape[-1]
    out = torch.empty_like(x)
    grid = (x.numel() // N,)
    print(f"grid={grid!r"}")
    return out
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert len(errors) >= 1


@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_fullwidth_chinese_punctuation(checker):
    """全角中文标点（括号、逗号）混入代码应被检测"""
    code = '''\
import torch

def relu_kernel(x_ptr, out_ptr, n_elements):
    pid = tl.program_id（axis=0）
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets， mask=mask)
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"
    assert "U+FF08" in errors[0]["detail"]


@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_trailing_xml_tag(checker):
    """结尾多出的 XML 标签应被检测"""
    code = '''\
import torch

def add(x, y):
    return x + y
<triton.language>
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"
    assert errors[0]["line"] == 5


@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_trailing_markdown_fence(checker):
    """结尾多出的 markdown code fence 应被检测"""
    code = '''\
import torch

def softmax(x):
    max_val = x.max()
    exps = torch.exp(x - max_val)
    return exps / exps.sum()
```
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"
    assert errors[0]["line"] == 7


# ============================================================
# import 可用性检测
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_import_typo(checker_no_dsl):
    """拼写错误的模块名应被检测"""
    code = '''\
import torch
from triton_ascned import autotune

def foo():
    pass
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    assert passed is False
    import_errors = [e for e in errors if e["error_type"] == "import_error"]
    modules = [e["detail"] for e in import_errors]
    assert any("triton_ascned" in m for m in modules)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_relative_import_skipped(checker_no_dsl):
    """相对导入不应触发 import 检测"""
    code = '''\
from . import utils
from .core import helper

def foo():
    return 1
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    import_errors = [e for e in errors if e["error_type"] == "import_error"]
    assert len(import_errors) == 0


# ============================================================
# 中文文本混入检测 (stray_chinese_text)
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_detect_bare_chinese_sentence(checker_no_dsl):
    """代码中混入的裸中文句子（无 # 注释符）应被检测"""
    code = '''\
import torch

def add(x, y):
    result = x + y
    这里计算两个张量的和
    return result
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    assert passed is False
    chinese_errors = [e for e in errors if e["error_type"] == "stray_chinese_text"]
    assert len(chinese_errors) == 1
    assert chinese_errors[0]["line"] == 5
    assert "这里计算两个张量的和" in chinese_errors[0]["detail"]


@pytest.mark.level0
@pytest.mark.asyncio
async def test_chinese_in_comment_not_detected(checker_no_dsl):
    """注释中的中文不应触发检测"""
    code = '''\
import os

def compute(x):
    # 这里计算两个张量的和
    result = x * 2
    return result
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    chinese_errors = [e for e in errors if e["error_type"] == "stray_chinese_text"]
    assert len(chinese_errors) == 0


@pytest.mark.level0
@pytest.mark.asyncio
async def test_chinese_in_string_not_detected(checker_no_dsl):
    """字符串中的中文不应触发检测"""
    code = '''\
import os

def compute(x):
    name = "计算两个张量的和"
    result = x * 2
    return result
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    chinese_errors = [e for e in errors if e["error_type"] == "stray_chinese_text"]
    assert len(chinese_errors) == 0


@pytest.mark.level0
@pytest.mark.asyncio
async def test_short_chinese_variable_not_detected(checker_no_dsl):
    """短中文变量名（<3 个汉字）不应触发检测"""
    code = '''\
import os

def forward(x):
    输出 = x * 2
    return 输出
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    chinese_errors = [e for e in errors if e["error_type"] == "stray_chinese_text"]
    assert len(chinese_errors) == 0


# ============================================================
# 空代码 / 正确代码
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_empty_code_returns_false(checker):
    """空代码应返回 False"""
    passed, error_message, errors = await checker.check("")
    assert passed is False
    assert len(errors) == 1
    assert errors[0]["error_type"] == "empty_code"


@pytest.mark.level0
@pytest.mark.asyncio
async def test_whitespace_only_returns_false(checker):
    """纯空白代码应返回 False"""
    passed, error_message, errors = await checker.check("   \n\n  ")
    assert passed is False
    assert len(errors) == 1
    assert errors[0]["error_type"] == "empty_code"


@pytest.mark.level0
@pytest.mark.asyncio
async def test_correct_code_passes(checker_no_dsl):
    """正确的代码应全部通过（基础语法/import/中文检测）"""
    code = '''\
import os
import sys
import math
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def relu(values: List[float]) -> List[float]:
    return [max(0.0, v) for v in values]

def softmax(values: List[float]) -> List[float]:
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [e / total for e in exps]

def compute(data: List[float], mode: str = "relu") -> Tuple[List[float], str]:
    """
    计算两个张量的和

    Args:
        data: 输入数据
        mode: 计算模式

    Returns:
        result: 计算结果
        summary: 计算总结
    """
    if mode == "relu":
        result = relu(data)
    elif mode == "softmax":
        result = softmax(data)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    summary = f"Processed {len(data)} elements with {mode}"
    logger.info(summary)
    return result, summary
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    assert passed is True
    assert len(errors) == 0
    assert error_message == ""


# ============================================================
# 运行时错误（静态检查不可捕获，确认不误报）
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_forward_arg_mismatch_not_detected(checker_no_dsl):
    """函数参数数量不匹配是运行时错误，静态检查不应报错"""
    code = '''\
import os

class Model:
    def forward(self, x, y, z):
        return x + y + z

def run():
    model = Model()
    result = model.forward(1)
    return result
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    assert passed is True
    assert len(errors) == 0


# ============================================================
# 错误输出格式验证
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_error_dict_has_required_fields(checker):
    """每个错误 dict 必须包含 line/error_type/detail/suggestion/code_snippet"""
    code = "def foo(\n    return 1"
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    for err in errors:
        assert "line" in err
        assert "error_type" in err
        assert "detail" in err
        assert "suggestion" in err
        assert "code_snippet" in err


@pytest.mark.level0
@pytest.mark.asyncio
async def test_error_message_contains_report_header(checker):
    """格式化的错误消息应包含报告标题"""
    code = "def foo(\n    return 1"
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert "CodeChecker" in error_message
    assert "问题" in error_message


@pytest.mark.level0
@pytest.mark.asyncio
async def test_error_message_contains_follow_up_hint(checker):
    """格式化的错误消息末尾应提示可能有后续问题"""
    code = "def foo(\n    return 1"
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    assert "修复后可能还有后续问题" in error_message


# ============================================================
# DSL 合规性检测 (反作弊)
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_dsl_compliant_triton_code_passes(checker):
    """完整合规的 triton 代码不应触发任何 DSL 合规性错误"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
        return out
'''
    passed, error_message, errors = await checker.check(code)
    dsl_errors = [e for e in errors if e["error_type"] not in ("import_error",)]
    assert len(dsl_errors) == 0


@pytest.mark.level0
@pytest.mark.asyncio
async def test_no_triton_kernel_detected(checker):
    """dsl=triton_cuda 但代码中无 @triton.jit kernel 应被检测"""
    code = '''\
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    error_types = [e["error_type"] for e in errors]
    assert "no_triton_kernel" in error_types


@pytest.mark.level0
@pytest.mark.asyncio
async def test_triton_kernel_defined_but_not_called(checker):
    """定义了 @triton.jit kernel 但未通过 kernel[grid](...) 调用应被检测"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    error_types = [e["error_type"] for e in errors]
    assert "triton_kernel_not_called" in error_types


@pytest.mark.level0
@pytest.mark.asyncio
async def test_hard_torch_api_rejected_even_with_kernel(checker):
    """即使 triton kernel 被调用，forward() 中的 HARD 类 torch API（如 matmul）也应被打回"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def preprocess_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        temp = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        preprocess_kernel[grid](x, temp, n, BLOCK=1024)
        return torch.matmul(temp, w)
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    error_types = [e["error_type"] for e in errors]
    assert "torch_api_instead_of_kernel" in error_types


@pytest.mark.level0
@pytest.mark.asyncio
async def test_soft_torch_api_allowed_when_kernel_called(checker):
    """triton kernel 被调用时，SOFT 类 torch API（如 exp）仅警告不打回"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def compute_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        compute_kernel[grid](x, out, n, BLOCK=1024)
        return torch.exp(out)
'''
    passed, error_message, errors = await checker.check(code)
    dsl_errors = [e for e in errors if e["error_type"] not in ("import_error",)]
    assert len(dsl_errors) == 0


@pytest.mark.level0
@pytest.mark.asyncio
async def test_soft_torch_api_rejected_without_kernel(checker):
    """triton kernel 未调用时，SOFT 类 torch API 也应硬失败"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def unused_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(torch.sigmoid(x))
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    error_types = [e["error_type"] for e in errors]
    assert "triton_kernel_not_called" in error_types
    assert "torch_api_without_kernel" in error_types


@pytest.mark.level0
@pytest.mark.asyncio
async def test_matmul_operator_detected(checker):
    """@ 矩阵乘法运算符应被检测为 HARD 类"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def noop_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        noop_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out @ out.T
'''
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    error_types = [e["error_type"] for e in errors]
    assert "torch_api_instead_of_kernel" in error_types


@pytest.mark.level0
@pytest.mark.asyncio
async def test_dsl_check_skipped_for_torch_dsl(checker_no_dsl):
    """dsl='torch' 时不应触发 DSL 合规性检测"""
    code = '''\
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    assert passed is True
    assert len(errors) == 0


@pytest.mark.level0
@pytest.mark.asyncio
async def test_autotune_decorator_recognized(checker):
    """@triton.autotune 装饰的 kernel 应被正确识别"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 128}), triton.Config({"BLOCK_SIZE": 256})],
    key=["n_elements"],
)
@triton.jit
def tuned_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        tuned_kernel[grid](x, out, n)
        return out
'''
    passed, error_message, errors = await checker.check(code)
    dsl_errors = [e for e in errors if e["error_type"] not in ("import_error",)]
    assert len(dsl_errors) == 0


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
