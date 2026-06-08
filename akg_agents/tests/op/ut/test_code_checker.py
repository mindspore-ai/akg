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


@pytest.fixture
def checker_a5():
    return CodeChecker(backend="ascend", dsl="triton_ascend", arch="ascend950pr_9589")

# ============================================================
# 语法错误
# ============================================================

@pytest.mark.level0
def test_syntax_error_unclosed_paren(checker):
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
    passed, _, errors = checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"


@pytest.mark.level0
def test_syntax_error_fullwidth_punctuation(checker):
    """全角中文标点混入"""
    code = '''\
import torch

def relu_kernel(x_ptr, out_ptr, n_elements):
    pid = tl.program_id（axis=0）
    x = tl.load(x_ptr + 0， mask=True)
'''
    passed, _, errors = checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"
    assert "U+FF08" in errors[0]["detail"]


@pytest.mark.level0
def test_syntax_error_trailing_markdown_fence(checker):
    """结尾 markdown fence"""
    code = '''\
import torch

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()
```
'''
    passed, _, errors = checker.check(code)
    assert passed is False
    assert errors[0]["error_type"] == "syntax_error"


# ============================================================
# import 检测
# ============================================================

@pytest.mark.level0
def test_import_typo_detected(checker_no_dsl):
    """拼写错误的模块名"""
    code = '''\
import torch
from triton_ascned import autotune

def foo():
    pass
'''
    passed, _, errors = checker_no_dsl.check(code)
    assert passed is False
    import_errors = [e for e in errors if e["error_type"] == "import_error"]
    assert any("triton_ascned" in e["detail"] for e in import_errors)


@pytest.mark.level0
def test_relative_import_skipped(checker_no_dsl):
    """相对导入不应报错"""
    code = '''\
from . import utils
from .core import helper

def foo():
    return 1
'''
    passed, _, errors = checker_no_dsl.check(code)
    assert all(e["error_type"] != "import_error" for e in errors)


# ============================================================
# 中文混入
# ============================================================

@pytest.mark.level0
def test_bare_chinese_sentence_detected(checker_no_dsl):
    """裸中文句子应被检测"""
    code = '''\
import torch

def add(x, y):
    result = x + y
    这里计算两个张量的和
    return result
'''
    passed, _, errors = checker_no_dsl.check(code)
    assert passed is False
    assert any(e["error_type"] == "stray_chinese_text" for e in errors)


# ============================================================
# 空代码 / 正确代码
# ============================================================

@pytest.mark.level0
def test_empty_code(checker):
    passed, _, errors = checker.check("")
    assert passed is False
    assert errors[0]["error_type"] == "empty_code"


@pytest.mark.level0
def test_correct_code_passes(checker_no_dsl):
    code = '''\
import os
import math

def relu(values):
    return [max(0.0, v) for v in values]
'''
    passed, error_message, errors = checker_no_dsl.check(code)
    assert passed is True
    assert len(errors) == 0


# ============================================================
# 错误输出格式
# ============================================================

@pytest.mark.level0
def test_error_dict_fields(checker):
    """错误 dict 必须包含 line/error_type/detail/suggestion/code_snippet"""
    code = "def foo(\n    return 1"
    passed, error_message, errors = checker.check(code)
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
def test_dsl_compliant_triton_passes(checker):
    """合规的 triton 代码不应触发 DSL 错误"""
    passed, _, errors = checker.check(TRITON_KERNEL_SNIPPET)
    dsl_errors = [e for e in errors if e["error_type"] not in ("import_error",)]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_no_triton_kernel(checker):
    """dsl=triton_cuda 但无 kernel"""
    code = '''\
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.matmul(x, y)
'''
    passed, _, errors = checker.check(code)
    assert passed is False
    assert any(e["error_type"] == "no_triton_kernel" for e in errors)


@pytest.mark.level0
def test_hard_torch_api_rejected(checker):
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
    passed, _, errors = checker.check(code)
    assert passed is False
    assert any(e["error_type"] == "torch_api_instead_of_kernel" for e in errors)


@pytest.mark.level0
def test_dotted_torch_nn_functional_hard_api_rejected(checker):
    """torch.nn.functional.conv2d should be rejected like F.conv2d."""
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
        return torch.nn.functional.conv2d(tmp, w)
'''
    passed, _, errors = checker.check(code)
    assert passed is False
    assert any(
        e["error_type"] == "torch_api_instead_of_kernel"
        and "torch.nn.functional.conv2d" in e["detail"]
        for e in errors
    )


@pytest.mark.level0
def test_dotted_torch_linalg_hard_api_rejected(checker):
    """torch.linalg.matmul should be rejected through the dotted namespace path."""
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
        return torch.linalg.matmul(tmp, w)
'''
    passed, _, errors = checker.check(code)
    assert passed is False
    assert any(
        e["error_type"] == "torch_api_instead_of_kernel"
        and "torch.linalg.matmul" in e["detail"]
        for e in errors
    )


@pytest.mark.level0
def test_kernel_not_called_with_torch_api(checker):
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
    passed, _, errors = checker.check(code)
    assert passed is False
    error_types = {e["error_type"] for e in errors}
    assert "triton_kernel_not_called" in error_types
    assert "torch_api_without_kernel" in error_types


@pytest.mark.level0
def test_dsl_check_skipped_for_torch(checker_no_dsl):
    """dsl='torch' 跳过 DSL 检测"""
    code = '''\
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.matmul(x, y)
'''
    passed, _, errors = checker_no_dsl.check(code)
    dsl_errors = [e for e in errors if e["error_type"] not in ("import_error",)]
    assert len(dsl_errors) == 0


# ============================================================
# Autotune 规范
# ============================================================

@pytest.mark.level0
def test_autotune_missing_restore_value(checker):
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
    passed, _, errors = checker.check(code)
    assert passed is False
    assert any(e["error_type"] == "autotune_missing_restore_value" for e in errors)


@pytest.mark.level0
def test_autotune_with_restore_value_passes(checker):
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
    passed, _, errors = checker.check(code)
    assert not any(e["error_type"].startswith("autotune_") for e in errors)


# ============================================================
# YAML 策略加载
# ============================================================

@pytest.mark.level0
def test_policy_loaded_from_yaml():
    """CodeChecker 的关键词集合均来自 op/config/code_checker.yaml"""
    from akg_agents.op.utils import code_checker as cc

    checker = CodeChecker(backend="cuda", dsl="triton_cuda")
    assert "matmul" in checker.torch_compute_ops_hard
    # layer_norm 在当前策略下被降到 soft（对齐 Ascend seeds 约束）
    assert "layer_norm" in checker.torch_compute_ops_soft
    assert "relu" in checker.torch_compute_ops_soft
    assert "torch" in checker.torch_call_prefixes
    assert "torch.nn.functional" in checker.torch_call_prefixes
    assert "nn.functional" in checker.torch_call_prefixes
    assert "torch.linalg" in checker.torch_call_prefixes
    assert "scaled_dot_product_attention" in checker.torch_compute_ops_hard
    assert "jit" in checker.triton_decorators
    # 身份字符串直接走模块级 _POLICY
    assert cc._POLICY["kernel_class_name"] == "ModelNew"
    assert cc._POLICY["triton_module_name"] == "triton"


@pytest.mark.level0
def test_config_dict_parameter_is_ignored():
    """CodeChecker(config=...) 不再影响策略（YAML 是唯一真源）"""
    c1 = CodeChecker(backend="cuda", dsl="triton_cuda", config=None)
    c2 = CodeChecker(
        backend="cuda",
        dsl="triton_cuda",
        config={"code_checker": {"torch_compute_ops_hard": ["only_this_one"]}},
    )
    assert c1.torch_compute_ops_hard == c2.torch_compute_ops_hard
    assert "matmul" in c2.torch_compute_ops_hard

# A5 API 合规性检测
# ============================================================

A5_COMPLIANT_KERNEL = '''\
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl

@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    c_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)

    with al.scope(core_mode="cube"):
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        acc = tl.dot(a, b)
        al.fixpipe(acc, bl.to_buffer(c_ub, al.ascend_address_space.UB),
                   al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        c = bl.to_tensor(c_ub)
        tl.store(C_ptr, c)
'''


@pytest.mark.level0
def test_a5_compliant_kernel_emits_no_a5_errors(checker_a5):
    """合规的 A5 kernel：al/bl import + al.scope + al.fixpipe + bl.alloc 齐备时
    不应该触发任何 a5_* 错误
    """
    passed, _, errors = checker_a5.check(A5_COMPLIANT_KERNEL)
    a5_errors = [e for e in errors if e["error_type"].startswith("a5_")]
    assert a5_errors == [], (
        f"A5_COMPLIANT_KERNEL 不应触发任何 a5_* 错误，但触发了：{a5_errors}"
    )


@pytest.mark.level0
def test_a5_check_skipped_for_non_a5_arch(checker):
    """非 A5 架构（如 cuda）不应触发 A5 检测"""
    code = '''\
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = torch.empty_like(x)
        kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    passed, _, errors = checker.check(code)
    a5_errors = [e for e in errors if e["error_type"].startswith("a5_")]
    assert len(a5_errors) == 0


@pytest.mark.level0
def test_a5_affinity_check_disabled_by_yaml(checker_a5):
    """yaml 中 a5_compliance.enable_triton_ascend_affinity_check 默认关闭：
    在 A5 + triton_ascend 路径上，即使 kernel 完全不使用任何亲和接口
    （al.scope / al.fixpipe / bl.alloc 全缺），Step 7 也应整段跳过、
    不产生任何 a5_* 错误。
    """
    code = '''\
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr,
    L: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    scale: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_l = tl.arange(0, L)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + offs_l[:, None] + offs_d[None, :])
    k = tl.load(k_ptr + offs_l[:, None] + offs_d[None, :])
    v = tl.load(v_ptr + offs_l[:, None] + offs_d[None, :])
    qk = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16))) * scale
    p = tl.exp(qk - tl.max(qk, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    pv = tl.dot(p.to(tl.float16), v.to(tl.float16))
    tl.store(out_ptr + offs_l[:, None] + offs_d[None, :], pv.to(tl.float16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        out = torch.empty_like(q)
        B, H, L, D = q.shape
        S = k.shape[2]
        scale = 1.0 / (D ** 0.5)
        attention_kernel[(B * H,)](q, k, v, out, B, H, L, S, D, scale)
        return out
'''
    _, _, errors = checker_a5.check(code)
    a5_errors = [e for e in errors if e["error_type"].startswith("a5_")]
    assert a5_errors == [], (
        "yaml 中 enable_triton_ascend_affinity_check=false 时 Step 7 "
        f"应整段跳过，但仍触发了 a5_* 错误：{a5_errors}"
    )


# TileLang DSL 合规性检测
# ============================================================

TILELANG_COMPLIANT_SOFTMAX = '''\
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def softmax_kernel(M, N, dtype="float32"):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
        shape: T.int32,
    ):
        with T.Kernel(M, is_npu=True) as (row_id, _):
            x_ub = T.alloc_ub([1, N], dtype)
            max_ub = T.alloc_ub([1, 1], dtype)
            sum_ub = T.alloc_ub([1, 1], dtype)
            with T.Scope("Vector"):
                T.copy(X[row_id, 0], x_ub[0, 0], [1, N])
                T.npuir_reduce(x_ub, max_ub, dims=[1], reduce_mode="max")
                T.npuir_sub(x_ub, max_ub, x_ub)
                T.npuir_exp(x_ub, x_ub)
                T.npuir_reduce(x_ub, sum_ub, dims=[1], reduce_mode="sum")
                T.npuir_div(x_ub, sum_ub, x_ub)
                T.copy(x_ub[0, 0], Y[row_id, 0], [1, N])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        M, N = x.shape
        Y = torch.zeros_like(x)
        func = softmax_kernel(M, N, "float32")
        compiled = tilelang.compile(func, target="npuir")
        compiled(x, Y, M)
        return Y
'''

TILELANG_COMPLIANT_MATMUL_CUDA = '''\
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul_kernel(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M, K = A.shape
        K2, N = B.shape
        block_M, block_N, block_K = 128, 128, 32
        C = torch.zeros([M, N], dtype=torch.float32, device=A.device)
        kernel = matmul_kernel(M, N, K, block_M, block_N, block_K)
        kernel(A, B, C)
        return C
'''

TILELANG_TORCH_DEGRADATION = '''\
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)
'''

TILELANG_KERNEL_NOT_CALLED = '''\
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def unused_kernel(M, N, dtype="float32"):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, is_npu=True) as (row_id, _):
            T.copy(X[row_id], Y[row_id])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)
'''

TILELANG_KERNEL_CALLED_WITH_HARD_TORCH = '''\
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def helper_kernel(M, N, dtype="float32"):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, is_npu=True) as (row_id, _):
            T.copy(X[row_id], Y[row_id])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        tmp = torch.empty_like(x)
        func = helper_kernel(x.shape[0], x.shape[1], "float32")
        compiled = tilelang.compile(func, target="npuir")
        compiled(x, tmp, x.shape[0])
        return torch.matmul(tmp, w)
'''

TILELANG_KERNEL_CALLED_WITH_SOFT_TORCH = '''\
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def relu_kernel(M, N, dtype="float32"):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, is_npu=True) as (row_id, _):
            T.copy(X[row_id], Y[row_id])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        func = relu_kernel(x.shape[0], x.shape[1], "float32")
        compiled = tilelang.compile(func, target="npuir")
        compiled(x, out, x.shape[0])
        return torch.relu(out) + torch.exp(out)
'''

TILELANG_KERNEL_NOT_CALLED_SOFT_TORCH_ONLY = '''\
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def unused_kernel(M, N, dtype="float32"):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, is_npu=True) as (row_id, _):
            T.copy(X[row_id], Y[row_id])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)
'''

TILELANG_DIRECT_CALL_PATTERN = '''\
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def add_kernel(M, N):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            T.copy(A[by * 128, bx * 128], C[by * 128, bx * 128])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M, N = A.shape
        C = torch.empty_like(A)
        kernel = add_kernel(M, N)
        kernel(A, B, C)
        return C
'''


@pytest.fixture
def checker_tilelang_ascend():
    return CodeChecker(backend="ascend", dsl="tilelang_ascend", arch="ascend910b4")


@pytest.fixture
def checker_tilelang_cuda():
    return CodeChecker(backend="cuda", dsl="tilelang_cuda", arch="a100")


@pytest.mark.level0
def test_tilelang_dsl_compliant_ascend_passes(checker_tilelang_ascend):
    """合规的 tilelang_ascend softmax 代码不应触发 DSL 错误"""
    passed, _, errors = checker_tilelang_ascend.check(TILELANG_COMPLIANT_SOFTMAX)
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_tilelang", "tilelang_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_tilelang_dsl_compliant_cuda_passes(checker_tilelang_cuda):
    """合规的 tilelang_cuda matmul 代码不应触发 DSL 错误"""
    passed, _, errors = checker_tilelang_cuda.check(TILELANG_COMPLIANT_MATMUL_CUDA)
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_tilelang", "tilelang_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_no_tilelang_kernel(checker_tilelang_ascend):
    """dsl=tilelang_ascend 但无 kernel — torch退化"""
    passed, _, errors = checker_tilelang_ascend.check(TILELANG_TORCH_DEGRADATION)
    assert passed is False
    assert any(e["error_type"] == "no_tilelang_kernel" for e in errors)


@pytest.mark.level0
def test_tilelang_kernel_not_called(checker_tilelang_ascend):
    """定义了 tilelang kernel 但没调用 + 用 torch.softmax — 硬失败"""
    passed, _, errors = checker_tilelang_ascend.check(TILELANG_KERNEL_NOT_CALLED)
    assert passed is False
    error_types = {e["error_type"] for e in errors}
    assert "tilelang_kernel_not_called" in error_types
    # softmax is a hard torch op
    assert "torch_api_instead_of_tilelang_kernel" in error_types


@pytest.mark.level0
def test_tilelang_hard_torch_api_rejected(checker_tilelang_ascend):
    """kernel 调用了但 forward 用 torch.matmul — 硬失败"""
    passed, _, errors = checker_tilelang_ascend.check(TILELANG_KERNEL_CALLED_WITH_HARD_TORCH)
    assert passed is False
    assert any(e["error_type"] == "torch_api_instead_of_tilelang_kernel" for e in errors)


@pytest.mark.level0
def test_tilelang_soft_torch_api_with_kernel_warning(checker_tilelang_ascend):
    """kernel 调用了但 forward 有 torch.relu/torch.exp — 仅警告，不应硬失败"""
    passed, _, errors = checker_tilelang_ascend.check(TILELANG_KERNEL_CALLED_WITH_SOFT_TORCH)
    # soft torch API with kernel called = only warning, not hard failure
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_tilelang", "tilelang_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_tilelang_soft_torch_api_without_kernel_rejected(checker_tilelang_ascend):
    """kernel 定义了但没调用 + forward 用 torch.relu — 硬失败"""
    passed, _, errors = checker_tilelang_ascend.check(TILELANG_KERNEL_NOT_CALLED_SOFT_TORCH_ONLY)
    assert passed is False
    error_types = {e["error_type"] for e in errors}
    assert "tilelang_kernel_not_called" in error_types
    assert "torch_api_without_tilelang_kernel" in error_types


@pytest.mark.level0
def test_tilelang_direct_call_pattern_passes(checker_tilelang_cuda):
    """直接调用 kernel = kernel_func(params)(inputs) 模式 — 应通过"""
    passed, _, errors = checker_tilelang_cuda.check(TILELANG_DIRECT_CALL_PATTERN)
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_tilelang", "tilelang_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_tilelang_yaml_policy_loaded():
    """TileLang compliance keys are loaded from YAML"""
    from akg_agents.op.utils import code_checker as cc
    assert cc._TL_MODULE_NAME == "tilelang"
    assert "jit" in cc._TL_DECORATORS
    assert cc._TL_PRIM_FUNC_ATTR == "prim_func"
    assert cc._TL_NAMESPACE == "T"
    assert "triton" in cc._DSL_COMPLIANCE_PREFIXES
    assert "tilelang" in cc._DSL_COMPLIANCE_PREFIXES


@pytest.mark.level0
def test_tilelang_check_skipped_for_torch(checker_no_dsl):
    """dsl='torch' 跳过 tilelang DSL 检测"""
    code = '''\
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.matmul(x, y)
'''
    passed, _, errors = checker_no_dsl.check(code)
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_tilelang", "tilelang_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


# Import alias (bare-name decorator) detection
# ============================================================

TILELANG_BARE_JIT = '''\
import torch
from tilelang import jit
import tilelang.language as T

@jit
def softmax_kernel(M, N, dtype="float32"):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
        shape: T.int32,
    ):
        with T.Kernel(M, is_npu=True) as (row_id, _):
            x_ub = T.alloc_ub([1, N], dtype)
            with T.Scope("Vector"):
                T.copy(X[row_id, 0], x_ub[0, 0], [1, N])
                T.copy(x_ub[0, 0], Y[row_id, 0], [1, N])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        M, N = x.shape
        Y = torch.zeros_like(x)
        func = softmax_kernel(M, N, "float32")
        compiled = tilelang.compile(func, target="npuir")
        compiled(x, Y, M)
        return Y
'''

TILELANG_BARE_JIT_ALIAS = '''\
import torch
from tilelang import jit as tl_jit
import tilelang.language as T

@tl_jit
def add_kernel(M, N):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, is_npu=True) as (row_id, _):
            T.copy(A[row_id], C[row_id])
    return main

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        M, N = A.shape
        C = torch.empty_like(A)
        kernel = add_kernel(M, N)
        kernel(A, B, C)
        return C
'''

TRITON_BARE_JIT = '''\
import torch
from triton import jit
import triton.language as tl

@jit
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

TRITON_BARE_JIT_ALIAS = '''\
import torch
from triton import jit as triton_jit
import triton.language as tl

@triton_jit
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

TILELANG_FALSE_BARE_JIT = '''\
import torch
# @jit here is NOT from tilelang — it's from some other module
from some_other_lib import jit

@jit
def my_func(x):
    return x

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.matmul(x, x)
'''


@pytest.mark.level0
def test_tilelang_bare_jit_detected(checker_tilelang_ascend):
    """from tilelang import jit; @jit — bare-name 形式应被识别为合法 kernel"""
    passed, _, errors = checker_tilelang_ascend.check(TILELANG_BARE_JIT)
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_tilelang", "tilelang_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_tilelang_bare_jit_alias_detected(checker_tilelang_cuda):
    """from tilelang import jit as tl_jit; @tl_jit — aliased bare-name 应被识别"""
    passed, _, errors = checker_tilelang_cuda.check(TILELANG_BARE_JIT_ALIAS)
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_tilelang", "tilelang_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_triton_bare_jit_detected(checker):
    """from triton import jit; @jit — bare-name 形式应被识别"""
    passed, _, errors = checker.check(TRITON_BARE_JIT)
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_triton", "triton_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_triton_bare_jit_alias_detected(checker):
    """from triton import jit as triton_jit; @triton_jit — aliased bare-name 应被识别"""
    passed, _, errors = checker.check(TRITON_BARE_JIT_ALIAS)
    dsl_errors = [e for e in errors if e["error_type"].startswith(("no_triton", "triton_kernel", "torch_api"))]
    assert len(dsl_errors) == 0


@pytest.mark.level0
def test_false_bare_jit_not_detected(checker_tilelang_ascend):
    """from some_other_lib import jit; @jit — 不是 tilelang 的 @jit 不应被误识别"""
    passed, _, errors = checker_tilelang_ascend.check(TILELANG_FALSE_BARE_JIT)
    assert passed is False
    assert any(e["error_type"] == "no_tilelang_kernel" for e in errors)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
