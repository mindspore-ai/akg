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

"""Unit tests for Triton autotune patch helpers."""

import pytest

from akg_agents.op.utils import triton_autotune_patch as patch


class TestTritonAutotunePatch:
    """Test restore helpers used by Triton autotune patching."""

    def test_wrap_kernel_call_restores_before_and_after(self, monkeypatch):
        """Benchmark wrapper should restore outputs on both sides of the kernel."""
        calls = []
        arg = {"value": "dirty"}
        restore_info = {"saved": {0: "clean"}, "args": [arg]}

        def fake_restore(dst, src):
            calls.append(("restore", dst["value"], src))
            dst["value"] = src

        def kernel_call():
            calls.append(("kernel", arg["value"]))
            arg["value"] = "kernel_result"
            return "ok"

        monkeypatch.setattr(patch, "akg_restore_copy", fake_restore)

        wrapped = patch._wrap_kernel_call_with_restore(kernel_call, restore_info)
        result = wrapped()

        assert result == "ok"
        assert arg["value"] == "clean"
        assert calls == [
            ("restore", "dirty", "clean"),
            ("kernel", "clean"),
            ("restore", "kernel_result", "clean"),
        ]

    def test_wrap_kernel_call_restores_after_exception(self, monkeypatch):
        """Even failing configs should leave outputs restored for later configs."""
        arg = {"value": "dirty"}
        restore_info = {"saved": {0: "clean"}, "args": [arg]}

        def fake_restore(dst, src):
            dst["value"] = src

        def kernel_call():
            arg["value"] = "broken"
            raise RuntimeError("boom")

        monkeypatch.setattr(patch, "akg_restore_copy", fake_restore)

        wrapped = patch._wrap_kernel_call_with_restore(kernel_call, restore_info)

        with pytest.raises(RuntimeError, match="boom"):
            wrapped()

        assert arg["value"] == "clean"

    def test_wrap_kernel_call_prevents_partial_write_pollution(self, monkeypatch):
        """A later config should not inherit untouched tail values from a previous run."""
        output = {"value": [9.0, 9.0, 9.0, 9.0]}
        restore_info = {"saved": {0: [0.0, 0.0, 0.0, 0.0]}, "args": [output]}

        def fake_restore(dst, src):
            dst["value"] = list(src)

        def bad_kernel_call():
            output["value"][0] = 1.0
            output["value"][1] = 1.0

        monkeypatch.setattr(patch, "akg_restore_copy", fake_restore)

        wrapped = patch._wrap_kernel_call_with_restore(bad_kernel_call, restore_info)
        wrapped()

        assert output["value"] == [0.0, 0.0, 0.0, 0.0]

    def test_wrap_kernel_call_without_restore_info_returns_original(self):
        """No restore_value should keep the original benchmark callback untouched."""

        def kernel_call():
            return "ok"

        wrapped = patch._wrap_kernel_call_with_restore(kernel_call, None)

        assert wrapped is kernel_call
