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

"""MLIR checker utilities for UTs."""

from __future__ import annotations

from typing import Iterable, List, Optional
from mfusion import ir
from mfusion.dialects import torch as torch_d


class MlirChecker:
    """Helper to parse MLIR and validate IR content in UTs."""

    def __init__(self, module: ir.Module):
        """Create a checker for an already-parsed module."""
        self.module = module
        self._error = ""

    @staticmethod
    def parse_torch_module(text: str) -> "MlirChecker":
        """Parse Torch MLIR text and wrap it with a checker."""
        ctx = ir.Context()
        # Ensure context is preserved or managed if necessary
        torch_d.register_dialect(ctx)
        try:
            module = ir.Module.parse(text, ctx)
            return MlirChecker(module)
        except Exception as e:
            # Handle parsing errors gracefully for UTs
            raise ValueError(f"Failed to parse MLIR text: {e}") from e

    @property
    def error(self) -> str:
        """Return the latest error message."""
        return self._error

    def check_has_op(self, op_name: str, count: int | None = None) -> bool:
        """Check that an op appears, optionally with an exact count."""
        ops = self._filter_ops(op_name)
        if count is None:
            if not ops:
                return self._set_error(f"Expected op '{op_name}', but not found.")
        else:
            actual = len(ops)
            if actual != count:
                return self._set_error(f"Expected {count} ops of '{op_name}', but got {actual}.")
        return self._clear_error()

    def check_no_op(self, op_name: str) -> bool:
        """Check that an op does not appear."""
        ops = self._filter_ops(op_name)
        if ops:
            return self._set_error(f"Unexpected op '{op_name}' found (count: {len(ops)}).")
        return self._clear_error()

    def check_top_level_ops(self, expected: List[str]) -> bool:
        """Check the exact sequence of top-level op names."""
        # Use op.operation.name to get the full dialect.op name
        actual = [op.operation.name for op in self.module.body.operations]
        if actual != expected:
            return self._set_error(f"Top-level ops mismatch.\nExpected: {expected}\nActual:   {actual}")
        return self._clear_error()

    def check_has_function(self, func_name: str) -> bool:
        """Check that a function with the given name exists."""
        if self._find_func_op(func_name) is None:
            return self._set_error(f"Function '@{func_name}' not found.")
        return self._clear_error()

    def check_func_op_sequence(self, func_name: str, expected: List[str]) -> bool:
        """Check op name sequence inside the specified function block."""
        func_op = self._find_func_op(func_name)
        if func_op is None:
            return self._set_error(f"Function '@{func_name}' not found.")

        # A func.func usually has one region and one block
        if not func_op.regions or not func_op.regions[0].blocks:
            return self._set_error(f"Function '@{func_name}' has no body block.")

        block = func_op.regions[0].blocks[0]
        actual = [op.operation.name for op in block.operations]
        if actual != expected:
            return self._set_error(
                f"Ops mismatch in '@{func_name}'.\nExpected: {expected}\nActual:   {actual}"
            )
        return self._clear_error()

    def check_text_contains(self, text: str) -> bool:
        """Check that the module string contains the expected text."""
        module_text = str(self.module)
        if text not in module_text:
            return self._set_error(f"Substring '{text}' not found in IR.")
        return self._clear_error()

    def check_total_op_count(self, expected: int) -> bool:
        """Check the total number of operations in the module."""
        actual = sum(1 for _ in self._walk_ops())
        if actual != expected:
            return self._set_error(f"Total op count mismatch. Expected {expected}, got {actual}.")
        return self._clear_error()

    def _filter_ops(self, op_name: str) -> List[ir.Operation]:
        """Return ops matching the given name."""
        return [op for op in self._walk_ops() if op.name == op_name]

    def _walk_ops(self) -> Iterable[ir.Operation]:
        """Yield all operations in the module using a manual walk if necessary."""
        # The standard Python binding walk works with a callback.
        def _recursive_walk(operation):
            yield operation
            for region in operation.regions:
                for block in region.blocks:
                    for op in block.operations:
                        yield from _recursive_walk(op)

        # Start walking from the top-level operation of the module
        yield from _recursive_walk(self.module.operation)

    def _find_func_op(self, func_name: str) -> Optional[ir.Operation]:
        """Find a func.func or similar op by symbol name."""
        for op in self.module.body.operations:
            # Note: Depending on dialect, it could be 'func.func' or 'func'
            if "func" not in op.operation.name:
                continue
            if "sym_name" in op.attributes:
                if ir.StringAttr(op.attributes["sym_name"]).value == func_name:
                    return op
        return None

    def _set_error(self, msg: str) -> bool:
        """Set the error message and return False."""
        self._error = self._format_error(msg)
        return False

    def _clear_error(self) -> bool:
        """Clear the error message and return True."""
        self._error = ""
        return True

    def _format_error(self, msg: str) -> str:
        """Format error message with a module snippet."""
        snippet = self._module_snippet()
        return f"[MlirChecker Error]: {msg}\n--- IR Snippet ---\n{snippet}\n------------------"

    def _module_snippet(self, max_lines: int = 40) -> str:
        """Build a truncated module text snippet."""
        lines = str(self.module).splitlines()
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[:max_lines]) + f"\n... (truncated, total {len(lines)} lines)"
