#!/usr/bin/env python3
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
Torch Task 代码格式验证脚本

验证代码是否符合 KernelBench 格式（4 个必需组件）：
1. class Model(nn.Module)
2. def forward(self, ...)
3. def get_inputs()
4. def get_init_inputs()

用法：
    # 从命令行参数读取代码文件
    python check_torch_code.py path/to/code.py
    
    # 从标准输入读取代码（推荐用于 LLM 调用）
    echo "import torch..." | python check_torch_code.py --stdin
    
    # 只做静态检查（不执行代码）
    python check_torch_code.py --stdin --static-only
    
    # 输出 JSON 格式
    python check_torch_code.py --stdin --json

输出格式：
    [VALID] 代码符合 KernelBench 格式
    [INVALID] 代码不符合格式 + 原因
"""

import ast
import sys
import argparse
import json


def check_static(code: str) -> tuple[bool, list[str], list[str]]:
    """
    静态检查代码是否包含必需组件
    
    Returns:
        (is_valid, missing_components, found_components)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"], []
    
    has = {
        "Model": False,
        "forward": False,
        "get_inputs": False,
        "get_init_inputs": False
    }
    
    for node in ast.walk(tree):
        # 检查 class Model(nn.Module)
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for base in node.bases:
                base_name = getattr(base, 'attr', getattr(base, 'id', ''))
                if base_name == "Module":
                    has["Model"] = True
                    # 检查 forward 方法
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "forward":
                            has["forward"] = True
        
        # 检查顶层函数 get_inputs / get_init_inputs
        if isinstance(node, ast.FunctionDef) and node.name in ("get_inputs", "get_init_inputs"):
            has[node.name] = True
    
    found = [k for k, v in has.items() if v]
    missing = [k for k, v in has.items() if not v]
    
    return len(missing) == 0, missing, found


def check_runtime(code: str) -> tuple[bool, str]:
    """
    运行时检查代码是否能正确执行
    
    检查流程：
    1. exec(code)
    2. get_init_inputs()
    3. Model(*init_inputs)
    4. get_inputs()
    5. model.forward(*inputs)
    
    Returns:
        (is_valid, error_message)
    """
    namespace = {}
    
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"exec error: {type(e).__name__}: {e}"
    
    # 检查 get_init_inputs
    if "get_init_inputs" not in namespace:
        return False, "get_init_inputs not defined"
    try:
        init_inputs = namespace["get_init_inputs"]()
    except Exception as e:
        return False, f"get_init_inputs() error: {type(e).__name__}: {e}"
    
    # 检查 Model
    if "Model" not in namespace:
        return False, "Model not defined"
    try:
        model = namespace["Model"](*init_inputs)
    except Exception as e:
        return False, f"Model(*get_init_inputs()) error: {type(e).__name__}: {e}"
    
    # 检查 get_inputs
    if "get_inputs" not in namespace:
        return False, "get_inputs not defined"
    try:
        inputs = namespace["get_inputs"]()
    except Exception as e:
        return False, f"get_inputs() error: {type(e).__name__}: {e}"
    
    # 检查 forward
    try:
        model(*inputs)
    except Exception as e:
        return False, f"model.forward(*get_inputs()) error: {type(e).__name__}: {e}"
    
    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="验证 Torch 代码是否符合 KernelBench 格式"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="要验证的 Python 文件路径"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="从标准输入读取代码"
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="只进行静态检查，不执行代码"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结果"
    )
    
    args = parser.parse_args()
    
    # 读取代码
    if args.stdin:
        code = sys.stdin.read()
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                code = f.read()
        except FileNotFoundError:
            if args.json:
                print(json.dumps({
                    "valid": False,
                    "error": f"File not found: {args.file}"
                }))
            else:
                print(f"[ERROR] 文件不存在: {args.file}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    # 静态检查
    static_valid, missing, found = check_static(code)
    
    result = {
        "valid": False,
        "static_check": {
            "passed": static_valid,
            "found_components": found,
            "missing_components": missing
        },
        "runtime_check": None,
        "suggestion": ""
    }
    
    if not static_valid:
        if missing and missing[0].startswith("SyntaxError"):
            result["error"] = missing[0]
            result["suggestion"] = "调用 call_op_task_builder 重新生成"
        else:
            result["error"] = f"缺少组件: {', '.join(missing)}"
            result["suggestion"] = "调用 call_op_task_builder 补全代码"
        
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[INVALID] 代码不符合 KernelBench 格式")
            print(f"缺少组件: {', '.join(missing)}")
            print(f"建议: {result['suggestion']}")
        sys.exit(1)
    
    # 运行时检查
    if not args.static_only:
        runtime_valid, runtime_error = check_runtime(code)
        result["runtime_check"] = {
            "passed": runtime_valid,
            "error": runtime_error if not runtime_valid else None
        }
        
        if not runtime_valid:
            result["error"] = runtime_error
            result["suggestion"] = "调用 call_op_task_builder 修复代码"
            
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"[INVALID] 代码运行时检查失败")
                print(f"错误信息: {runtime_error}")
                print(f"建议: {result['suggestion']}")
            sys.exit(1)
    
    # 检查通过
    result["valid"] = True
    check_type = "静态" if args.static_only else "静态+运行时"
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[VALID] 代码符合 KernelBench 格式（{check_type}检查通过）")
        print(f"包含组件: {', '.join(found)}")
        print(f"可直接用于生成 kernel")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

