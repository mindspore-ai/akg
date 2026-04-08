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

覆盖：语法检查、import 检测、中文混入、空代码、错误格式、DSL 合规、Autotune 规范。
"""

import pytest
from akg_agents.op.utils.code_checker import CodeChecker


@pytest.fixture
def checker():
    return CodeChecker(backend="cuda", dsl="triton_cuda")


@pytest.fixture
def checker_no_dsl():
    return CodeChecker(backend="cuda", dsl="torch")


# ============================================================
# 语法错误
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_syntax_error_unclosed_paren(checker):
    """括号不匹配"""
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
    passed, _, errors = await checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"


@pytest.mark.level0
@pytest.mark.asyncio
async def test_syntax_error_fullwidth_punctuation(checker):
    """全角中文标点混入"""
    code = '''\
import torch

def relu_kernel(x_ptr, out_ptr, n_elements):
    pid = tl.program_id（axis=0）
    x = tl.load(x_ptr + 0， mask=True)
'''
    passed, _, errors = await checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"
    assert "U+FF08" in errors[0]["detail"]


@pytest.mark.level0
@pytest.mark.asyncio
async def test_syntax_error_trailing_markdown_fence(checker):
    """结尾 markdown fence"""
    code = '''\
import torch

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()
```
'''
    passed, _, errors = await checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"


# ============================================================
# import 检测
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_import_typo_detected(checker_no_dsl):
    """拼写错误的模块名"""
    code = '''\
import torch
from triton_ascned import autotune

def foo():
    pass
'''
    passed, _, errors = await checker_no_dsl.check(code)
    assert passed is False
    import_errors = [e for e in errors if e["error_type"] == "import_error"]
    assert any("triton_ascned" in e["detail"] for e in import_errors)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_relative_import_skipped(checker_no_dsl):
    """相对导入不应报错"""
    code = '''\
from . import utils
from .core import helper

def foo():
    return 1
'''
    passed, _, errors = await checker_no_dsl.check(code)
    assert all(e["error_type"] != "import_error" for e in errors)


# ============================================================
# 中文混入
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_bare_chinese_sentence_detected(checker_no_dsl):
    """裸中文句子应被检测"""
    code = '''\
import torch

def add(x, y):
    result = x + y
    这里计算两个张量的和
    return result
'''
    passed, _, errors = await checker_no_dsl.check(code)
    assert passed is False
    assert any(e["error_type"] == "stray_chinese_text" for e in errors)


# ============================================================
# 空代码 / 正确代码
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_empty_code(checker):
    passed, _, errors = await checker.check("")
    assert passed is False
    assert errors[0]["error_type"] == "empty_code"


@pytest.mark.level0
@pytest.mark.asyncio
async def test_correct_code_passes(checker_no_dsl):
    code = '''\
import os
import math

def relu(values):
    return [max(0.0, v) for v in values]
'''
    passed, error_message, errors = await checker_no_dsl.check(code)
    assert passed is True
    assert len(errors) == 0


# ============================================================
# 错误输出格式
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_error_dict_fields(checker):
    """错误 dict 必须包含 line/error_type/detail/suggestion/code_snippet"""
    code = "def foo(\n    return 1"
    passed, error_message, errors = await checker.check(code)
    assert passed is False
    for err in errors:
        for key in ("line", "error_type", "detail", "suggestion", "code_snippet"):
            assert key in err
    assert "CodeChecker" in error_message


# ============================================================
# DSL 合规性检测
# ============================================================

TRITON_KERNEL_SNIPPET = '''\
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


@pytest.mark.level0
@pytest.mark.asyncio
async def test_dsl_compliant_triton_passes(checker):
    """合规的 triton 代码不应触发 DSL 错误"""
    passed, _, errors = await checker.check(TRITON_KERNEL_SNIPPET)
    dsl_errors = [e for e in errors if e["error_type"] not in ("import_error",)]
    assert len(dsl_errors) == 0


@pytest.mark.level0
@pytest.mark.asyncio
async def test_no_triton_kernel(checker):
    """dsl=triton_cuda 但无 kernel"""
    code = '''\
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.matmul(x, y)
'''
    passed, _, errors = await checker.check(code)
    assert passed is False
    assert any(e["error_type"] == "no_triton_kernel" for e in errors)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_hard_torch_api_rejected(checker):
    """kernel 调用了但 forward 用 matmul 应被打回"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, w):
        tmp = torch.empty_like(x)
        k[(1,)](x, tmp, x.numel(), BLOCK=1024)
        return torch.matmul(tmp, w)
'''
    passed, _, errors = await checker.check(code)
    assert passed is False
    assert any(e["error_type"] == "torch_api_instead_of_kernel" for e in errors)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_kernel_not_called_with_torch_api(checker):
    """kernel 定义了但没调用，且 forward 用 torch API"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def unused(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.exp(torch.sigmoid(x))
'''
    passed, _, errors = await checker.check(code)
    assert passed is False
    error_types = {e["error_type"] for e in errors}
    assert "triton_kernel_not_called" in error_types
    assert "torch_api_without_kernel" in error_types


@pytest.mark.level0
@pytest.mark.asyncio
async def test_dsl_check_skipped_for_torch(checker_no_dsl):
    """dsl='torch' 跳过 DSL 检测"""
    code = '''\
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.matmul(x, y)
'''
    passed, _, errors = await checker_no_dsl.check(code)
    dsl_errors = [e for e in errors if e["error_type"] not in ("import_error",)]
    assert len(dsl_errors) == 0


# ============================================================
# Autotune 规范
# ============================================================

@pytest.mark.level0
@pytest.mark.asyncio
async def test_autotune_missing_restore_value(checker):
    """缺少 restore_value"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 128}), triton.Config({"BLOCK_SIZE": 256})],
    key=["n_elements"],
)
@triton.jit
def tuned(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        tuned[grid](x, out, x.numel())
        return out
'''
    passed, _, errors = await checker.check(code)
    assert passed is False
    assert any(e["error_type"] == "autotune_missing_restore_value" for e in errors)


@pytest.mark.level0
@pytest.mark.asyncio
async def test_autotune_with_restore_value_passes(checker):
    """有 restore_value 时多 config 应通过"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=["n_elements"],
    restore_value=["out_ptr"],
)
@triton.jit
def tuned(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        tuned[grid](x, out, x.numel())
        return out
'''
    passed, _, errors = await checker.check(code)
    assert not any(e["error_type"].startswith("autotune_") for e in errors)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
