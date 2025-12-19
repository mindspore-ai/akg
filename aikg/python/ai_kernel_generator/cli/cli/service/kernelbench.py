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

from __future__ import annotations

import ast


class KernelBenchStaticChecker:
    """KernelBench 形式的最小静态检查。"""

    def check(self, task_desc: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(task_desc)
            has_model = False
            has_get_inputs = False
            has_get_init_inputs = False

            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == "Model":
                    has_model = True
                elif isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
                    has_get_inputs = True
                elif (
                    isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs"
                ):
                    has_get_init_inputs = True

            missing = []
            if not has_model:
                missing.append("class Model")
            if not has_get_inputs:
                missing.append("function get_inputs")
            if not has_get_init_inputs:
                missing.append("function get_init_inputs")
            if missing:
                return False, f"缺少 KernelBench 必需组件: {', '.join(missing)}"
            return True, ""
        except SyntaxError as e:
            return False, f"语法错误: {e}"
        except Exception as e:
            return False, f"静态检查失败: {e}"
