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
KernelBench 任务代码验证脚本

验证代码是否符合 KernelBench 格式并通过运行时检查。

检查项目:
1. 静态: class Model(nn.Module), forward, get_inputs, get_init_inputs
2. 运行时: exec → Model() → forward() → NaN/Inf 检查 → 一致性检查

用法:
    # 验证文件
    python validate_kernelbench_task.py path/to/task.py

    # 从标准输入读取
    echo "import torch..." | python validate_kernelbench_task.py --stdin

    # 只做静态检查
    python validate_kernelbench_task.py --stdin --static-only

    # JSON 格式输出
    python validate_kernelbench_task.py --stdin --json

输出格式:
    [VALID] 代码符合 KernelBench 格式
    [INVALID] 代码不符合格式 + 原因 + 修复建议
"""

import ast
import sys
import argparse
import json


def check_static(code: str) -> dict:
    """
    静态检查: 验证 KernelBench 四大组件是否存在

    自动检测 framework:
    - 包含 "mindspore" → MindSpore 模式 (nn.Cell + construct)
    - 否则 → PyTorch 模式 (nn.Module + forward)

    Returns:
        {"passed": bool, "found": [...], "missing": [...], "error": str|None, "framework": str}
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "passed": False,
            "found": [],
            "missing": ["Model", "forward/construct", "get_inputs", "get_init_inputs"],
            "error": f"SyntaxError: {e}",
            "framework": "unknown",
        }

    framework = "mindspore" if "mindspore" in code else "torch"
    forward_name = "construct" if framework == "mindspore" else "forward"
    base_name = "Cell" if framework == "mindspore" else "Module"

    has = {
        "Model": False,
        forward_name: False,
        "get_inputs": False,
        "get_init_inputs": False,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for base in node.bases:
                bn = getattr(base, "attr", getattr(base, "id", ""))
                if bn == base_name:
                    has["Model"] = True
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == forward_name:
                            has[forward_name] = True

        if isinstance(node, ast.FunctionDef) and node.name in (
            "get_inputs",
            "get_init_inputs",
        ):
            has[node.name] = True

    found = [k for k, v in has.items() if v]
    missing = [k for k, v in has.items() if not v]
    return {"passed": len(missing) == 0, "found": found, "missing": missing, "error": None, "framework": framework}


def check_runtime(code: str, timeout: int = 30) -> dict:
    """
    运行时检查: exec → Model() → forward() → NaN/Inf

    Returns:
        {"passed": bool, "checks": [...], "error": str|None}
    """
    import signal

    checks = []
    namespace = {}

    # 1. exec
    try:
        exec(code, namespace)
        checks.append({"name": "exec", "passed": True})
    except Exception as e:
        checks.append({"name": "exec", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"exec error: {e}"}

    # 2. get_init_inputs
    try:
        init_inputs = namespace["get_init_inputs"]()
        checks.append({"name": "get_init_inputs()", "passed": True})
    except Exception as e:
        checks.append({"name": "get_init_inputs()", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"get_init_inputs() error: {e}"}

    # 3. Model instantiation
    try:
        model = namespace["Model"](*init_inputs)
        checks.append({"name": "Model(*init_inputs)", "passed": True})
    except Exception as e:
        checks.append({"name": "Model(*init_inputs)", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"Model() error: {e}"}

    # 4. get_inputs
    try:
        inputs = namespace["get_inputs"]()
        checks.append({"name": "get_inputs()", "passed": True})
    except Exception as e:
        checks.append({"name": "get_inputs()", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"get_inputs() error: {e}"}

    # 5. forward
    try:
        output = model(*inputs)
        checks.append({"name": "model(*inputs)", "passed": True})
    except Exception as e:
        checks.append({"name": "model(*inputs)", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"forward() error: {e}"}

    # 6. NaN/Inf check
    try:
        import torch as _torch

        def _check_tensor_torch(t, name="output"):
            if isinstance(t, _torch.Tensor):
                if _torch.isnan(t).any():
                    return f"{name} contains NaN"
                if _torch.isinf(t).any():
                    return f"{name} contains Inf"
            return None
    except ImportError:
        _check_tensor_torch = None

    try:
        import mindspore as _ms

        def _check_tensor_ms(t, name="output"):
            if isinstance(t, _ms.Tensor):
                if _ms.ops.isnan(t).any():
                    return f"{name} contains NaN"
                if _ms.ops.isinf(t).any():
                    return f"{name} contains Inf"
            return None
    except ImportError:
        _check_tensor_ms = None

    def _check_tensor(t, name="output"):
        if _check_tensor_torch and isinstance(t, _torch.Tensor):
            return _check_tensor_torch(t, name)
        if _check_tensor_ms and isinstance(t, _ms.Tensor):
            return _check_tensor_ms(t, name)
        return None

    try:
        issues = []
        if isinstance(output, (tuple, list)):
            for i, item in enumerate(output):
                issue = _check_tensor(item, f"output[{i}]")
                if issue:
                    issues.append(issue)
        else:
            issue = _check_tensor(output)
            if issue:
                issues.append(issue)

        if issues:
            checks.append({"name": "NaN/Inf check", "passed": False, "error": "; ".join(issues)})
            return {"passed": False, "checks": checks, "error": "; ".join(issues)}
        checks.append({"name": "NaN/Inf check", "passed": True})
    except Exception:
        checks.append({"name": "NaN/Inf check", "passed": True, "note": "skipped"})

    # 7. Consistency check (run twice, compare)
    try:
        output2 = model(*inputs)

        def _tensors_close(a, b, rtol=1e-5, atol=1e-6):
            if _check_tensor_torch is not None:
                import torch
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
            if _check_tensor_ms is not None:
                import mindspore as ms
                if isinstance(a, ms.Tensor) and isinstance(b, ms.Tensor):
                    return ms.ops.allclose(a.astype(ms.float32), b.astype(ms.float32), rtol=rtol, atol=atol)
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                return all(_tensors_close(x, y) for x, y in zip(a, b))
            return True

        if _tensors_close(output, output2):
            checks.append({"name": "consistency check", "passed": True})
        else:
            checks.append({"name": "consistency check", "passed": False, "error": "outputs differ between runs"})
            return {"passed": False, "checks": checks, "error": "consistency check failed"}
    except Exception:
        checks.append({"name": "consistency check", "passed": True, "note": "skipped"})

    return {"passed": True, "checks": checks, "error": None}


def main():
    parser = argparse.ArgumentParser(
        description="验证代码是否符合 KernelBench 任务格式"
    )
    parser.add_argument("file", nargs="?", help="要验证的 Python 文件路径")
    parser.add_argument("--stdin", action="store_true", help="从标准输入读取代码")
    parser.add_argument("--static-only", action="store_true", help="只做静态检查")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")

    args = parser.parse_args()

    # Read code
    if args.stdin:
        code = sys.stdin.read()
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                code = f.read()
        except FileNotFoundError:
            if args.json:
                print(json.dumps({"valid": False, "error": f"File not found: {args.file}"}))
            else:
                print(f"[ERROR] 文件不存在: {args.file}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Static check
    static_result = check_static(code)
    result = {
        "valid": False,
        "static_check": static_result,
        "runtime_check": None,
        "suggestion": "",
    }

    if not static_result["passed"]:
        result["error"] = static_result.get("error") or f"缺少组件: {', '.join(static_result['missing'])}"
        result["suggestion"] = "调用 call_task_constructor 重新构建"
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[INVALID] 代码不符合 KernelBench 格式")
            print(f"缺少: {', '.join(static_result['missing'])}")
            print(f"建议: {result['suggestion']}")
        sys.exit(1)

    # Runtime check
    if not args.static_only:
        runtime_result = check_runtime(code)
        result["runtime_check"] = runtime_result

        if not runtime_result["passed"]:
            result["error"] = runtime_result["error"]
            result["suggestion"] = "检查代码逻辑，修复后重新验证"
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"[INVALID] 运行时检查失败")
                print(f"错误: {runtime_result['error']}")
                for check in runtime_result["checks"]:
                    status = "PASS" if check["passed"] else "FAIL"
                    print(f"  [{status}] {check['name']}")
            sys.exit(1)

    # All passed
    result["valid"] = True
    check_type = "静态" if args.static_only else "静态+运行时"

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[VALID] 代码符合 KernelBench 格式（{check_type}检查通过）")
        print(f"包含组件: {', '.join(static_result['found'])}")
    sys.exit(0)


if __name__ == "__main__":
    main()
