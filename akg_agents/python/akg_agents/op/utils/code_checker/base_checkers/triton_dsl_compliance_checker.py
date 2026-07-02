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

"""Triton DSL compliance check for CodeChecker."""

import ast
import logging
from typing import Dict, List, Optional

from akg_agents.op.utils.code_checker.base import BlockingCodeChecker

logger = logging.getLogger(__name__)

_TRITON_DECORATORS = frozenset({"jit", "autotune", "heuristics"})

# HARD: 高级复杂 API，无论 kernel 是否调用都硬失败（不应出现在 forward() 中）
_TORCH_COMPUTE_OPS_HARD = frozenset(
    {
        # Matrix / linear algebra — 这些是核心计算，必须由 kernel 完成
        "matmul",
        "mm",
        "bmm",
        "addmm",
        "addmv",
        "addbmm",
        "baddbmm",
        "einsum",
        "dot",
        "mv",
        "inner",
        "outer",
        "linear",
        # Convolution
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        # Normalization — 整个 norm 语义应在 kernel 内实现
        "layer_norm",
        "batch_norm",
        "group_norm",
        "instance_norm",
        # Compound activations — softmax 含 reduction + exp，属于完整子计算图
        "softmax",
        "log_softmax",
        "logsumexp",
        # Pooling
        "max_pool1d",
        "max_pool2d",
        "max_pool3d",
        "avg_pool1d",
        "avg_pool2d",
        "avg_pool3d",
        "adaptive_avg_pool2d",
        # Other complex ops
        "embedding",
        "interpolate",
        "cumsum",
        "cumprod",
    }
)

# SOFT: 简单元素级/辅助操作，仅在 kernel 未调用时硬失败，已调用时仅警告。
_TORCH_COMPUTE_OPS_SOFT = frozenset(
    {
        # Simple activations — 融合算子可能在 kernel 外做 pre/post 激活
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "leaky_relu",
        "elu",
        "hardswish",
        "mish",
        # Simple elementwise math — pre/post 处理可能合理使用
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "pow",
        "sin",
        "cos",
        "abs",
        "clamp",
        "clamp_min",
        "clamp_max",
        # Simple reductions — 可能用于 grid 计算或后处理
        "sum",
        "mean",
        "prod",
        "norm",
        "amax",
        "amin",
        "argmax",
        "argmin",
    }
)

_TORCH_CALL_PREFIXES = frozenset({"torch", "F"})


def _is_triton_decorator(node: ast.expr) -> bool:
    """判断 decorator 节点是否为 @triton.jit / @triton.autotune 等"""
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "triton"
            and node.attr in _TRITON_DECORATORS
        )
    if isinstance(node, ast.Call):
        return _is_triton_decorator(node.func)
    return False


def _find_model_new_class(tree: ast.Module) -> Optional[ast.ClassDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            return node
    return None


def _find_forward(cls_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
            return item
    return None


class TritonDslComplianceChecker(BlockingCodeChecker):
    """Verify that Triton DSL code defines and launches a Triton kernel."""

    name = "triton_dsl_compliance"

    def __init__(self, dsl: str):
        self.dsl = dsl.lower() if dsl else ""

    def check(self, code: str) -> List[Dict]:
        """
        检测生成代码是否真正使用了指定的 DSL 实现（纯 AST 静态分析）。

        仅对 triton 系列 DSL 生效（triton_cuda / triton_ascend），检查三类问题：

        A. 没有定义任何 @triton.jit kernel            → 硬失败
        B. 定义了 kernel 但没有通过 kernel[grid](...) 调用 → 硬失败
        C. forward() 中使用了 torch 高层计算 API          → 结合 B 判断
           - kernel 未调用 + torch API → 硬失败（明确作弊）
           - kernel 已调用 + torch API → 仅日志警告（可能是合理辅助操作）
        """
        if not self.dsl.startswith("triton"):
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        errors: List[Dict] = []

        triton_kernels: set = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if _is_triton_decorator(dec):
                        triton_kernels.add(node.name)
                        break

        if not triton_kernels:
            errors.append(
                {
                    "line": 0,
                    "error_type": "no_triton_kernel",
                    "detail": (
                        f"DSL 指定为 {self.dsl}，但代码中未找到任何 @triton.jit 装饰的 kernel 函数。"
                        f"代码可能使用了 torch 高层 API 替代 triton kernel 实现。"
                    ),
                    "suggestion": (
                        "请确保代码中包含至少一个 @triton.jit 装饰的 kernel 函数，"
                        "并在 ModelNew.forward() 中通过 kernel[grid](...) 语法调用它。"
                    ),
                    "code_snippet": "",
                }
            )
            return errors

        launched_kernels: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                value = node.func.value
                if isinstance(value, ast.Name) and value.id in triton_kernels:
                    launched_kernels.add(value.id)

        kernels_not_launched = not launched_kernels

        if kernels_not_launched:
            errors.append(
                {
                    "line": 0,
                    "error_type": "triton_kernel_not_called",
                    "detail": (
                        f"定义了 triton kernel 函数 {sorted(triton_kernels)}，"
                        f"但代码中未找到任何 kernel[grid](...) 形式的调用。"
                        f"kernel 函数可能只是装饰性的，实际计算未使用 triton。"
                    ),
                    "suggestion": (
                        "请在 ModelNew.forward() 或其辅助方法中，"
                        "通过 kernel_name[grid_size](...) 语法启动 triton kernel。"
                    ),
                    "code_snippet": "",
                }
            )

        model_cls = _find_model_new_class(tree)
        if model_cls is None:
            return errors

        forward_node = _find_forward(model_cls)
        if forward_node is None:
            return errors

        hard_calls: List[tuple] = []
        soft_calls: List[tuple] = []

        for node in ast.walk(forward_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                mod = node.func.value
                method = node.func.attr
                if isinstance(mod, ast.Name) and mod.id in _TORCH_CALL_PREFIXES:
                    label = f"{mod.id}.{method}"
                    if method in _TORCH_COMPUTE_OPS_HARD:
                        hard_calls.append((node.lineno, label))
                    elif method in _TORCH_COMPUTE_OPS_SOFT:
                        soft_calls.append((node.lineno, label))

            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                hard_calls.append((node.lineno, "@ (matmul operator)"))

        def _fmt_calls(calls: List[tuple], limit: int = 5) -> str:
            summary = ", ".join(f"{name}(第{line}行)" for line, name in calls[:limit])
            if len(calls) > limit:
                summary += f" 等（共 {len(calls)} 处）"
            return summary

        if hard_calls:
            errors.append(
                {
                    "line": hard_calls[0][0],
                    "error_type": "torch_api_instead_of_kernel",
                    "detail": (
                        f"forward() 中使用了 {len(hard_calls)} 个不允许的 torch 高层计算 API: "
                        f"{_fmt_calls(hard_calls)}。"
                        f"这些操作（矩阵乘法、卷积、归一化、池化等）必须完全在 DSL kernel 内实现。"
                    ),
                    "suggestion": (
                        "请将这些核心计算操作移入 triton kernel 中实现，"
                        "forward() 仅负责准备输入、启动 kernel 和返回输出。"
                    ),
                    "code_snippet": "",
                }
            )

        if soft_calls:
            if kernels_not_launched:
                errors.append(
                    {
                        "line": soft_calls[0][0],
                        "error_type": "torch_api_without_kernel",
                        "detail": (
                            f"forward() 中使用了 {len(soft_calls)} 个 torch 计算 API: "
                            f"{_fmt_calls(soft_calls)}。"
                            f"同时 triton kernel 未被调用，代码很可能用 torch API 替代了 kernel 实现。"
                        ),
                        "suggestion": (
                            "请将核心计算逻辑用 triton kernel 实现。"
                            "这些简单操作（exp/relu/sum 等）如果只是 kernel 的前后处理可以保留，"
                            "但前提是必须有 kernel 承担主要计算。"
                        ),
                        "code_snippet": "",
                    }
                )
            else:
                logger.warning(
                    f"CodeChecker DSL compliance: forward() 调用了 triton kernel，"
                    f"同时包含 {len(soft_calls)} 处 torch 辅助计算 API: "
                    f"{_fmt_calls(soft_calls)}。（融合算子可能合理，仅记录警告）"
                )

        return errors
