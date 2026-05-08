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

from __future__ import annotations

import ast


class _ModelNewContractValidator:
    def __init__(self, code: str, *, output_path: str = "<string>") -> None:
        self._code = code
        self._output_path = output_path

    @classmethod
    def validate_code(
        cls,
        code: str,
        *,
        output_path: str = "<string>",
    ) -> tuple[str, ...]:
        return cls(code, output_path=output_path).validate()

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            compile(self._code, self._output_path, "exec")
            tree = ast.parse(self._code, filename=self._output_path)
        except SyntaxError as exc:
            return (f"Python 语法错误: line {exc.lineno}: {exc.msg}",)

        model_new = self._find_modelnew(tree)
        if model_new is None:
            return ("缺少 `class ModelNew`",)

        forward = self._find_method(model_new, "forward")
        select_config = self._find_method(model_new, "_select_config")

        if forward is None:
            errors.append("缺少 `ModelNew.forward`")
        if select_config is None:
            errors.append("缺少 `ModelNew._select_config`")
        if forward is None:
            return tuple(errors)

        if not self._function_has_config_default_none(forward):
            errors.append("`ModelNew.forward` 必须声明 `config=None`")
        if not self._calls_method(forward, "_select_config"):
            errors.append("`ModelNew.forward` 缺少 `_select_config(...)` 路径")
        errors.extend(self._validate_load_deployed_selector_imports(tree))
        return tuple(errors)

    def _find_modelnew(self, tree: ast.Module) -> ast.ClassDef | None:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
                return node
        return None

    def _find_method(
        self,
        cls: ast.ClassDef,
        name: str,
    ) -> ast.FunctionDef | None:
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def _function_has_config_default_none(self, fn: ast.FunctionDef) -> bool:
        positional = list(fn.args.args)
        positional_defaults = list(fn.args.defaults)
        if positional_defaults:
            default_offset = len(positional) - len(positional_defaults)
            for index, arg in enumerate(positional):
                if arg.arg != "config":
                    continue
                default_index = index - default_offset
                if default_index < 0:
                    return False
                default = positional_defaults[default_index]
                return isinstance(default, ast.Constant) and default.value is None
        for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            if arg.arg != "config":
                continue
            return isinstance(default, ast.Constant) and default.value is None
        return False

    def _calls_method(self, node: ast.AST, method_name: str) -> bool:
        for item in ast.walk(node):
            if not isinstance(item, ast.Call):
                continue
            func = item.func
            if isinstance(func, ast.Attribute) and func.attr == method_name:
                return True
            if isinstance(func, ast.Name) and func.id == method_name:
                return True
        return False

    def _validate_load_deployed_selector_imports(self, tree: ast.Module) -> list[str]:
        errors: list[str] = []
        has_direct_call = False
        has_valid_import = False
        for item in ast.walk(tree):
            if isinstance(item, ast.ImportFrom):
                imports_loader = any(
                    alias.name == "load_deployed_selector" for alias in item.names
                )
                if not imports_loader:
                    continue
                if item.module == "akg_agents.op.dynamic_tune":
                    has_valid_import = True
                    continue
                errors.append(
                    "`load_deployed_selector` 必须从 `akg_agents.op.dynamic_tune` 导入"
                )
            if isinstance(item, ast.Call):
                func = item.func
                if isinstance(func, ast.Name) and func.id == "load_deployed_selector":
                    has_direct_call = True
        if has_direct_call and not has_valid_import:
            errors.append("缺少 `from akg_agents.op.dynamic_tune import load_deployed_selector`")
        return errors


__all__ = ["_ModelNewContractValidator"]
