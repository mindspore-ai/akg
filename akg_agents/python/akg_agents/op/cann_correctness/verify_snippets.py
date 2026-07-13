# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""Verify-scaffold code the cannbench precision path injects into akg's generic
verify template through its neutral hooks (reference_call_code / compare_outputs_code).

The template itself carries zero cannbench logic; these are the only lines that
mention the standard, and they live here (the package), not in akg core. The
DSLAdapter base delegates to these when uses_cannbench_precision is True."""


def reference_call_snippet() -> str:
    """Lines for the verify loop's reference else-branch: FP64-CPU golden +
    target-precision native reference from one callable (NPU has no fp64, so the
    golden runs on CPU). Sets ``framework_output`` (golden) and the template's
    optional ``_akg_secondary_ref`` (native, for the small-value / cancellation
    excusal). Vars in scope: ``framework_model``, ``inputs_for_framework``."""
    return (
        "import cann_correctness as _akg_cc\n"
        "framework_output, _akg_secondary_ref = _akg_cc.dual_reference("
        "framework_model, inputs_for_framework)\n"
        "print('[INFO] cannbench precision: FP64 golden + native reference computed')\n"
    )


def compare_snippet() -> str:
    """Per-output compare for the verify loop via cann_correctness.assert_outputs.
    Vars in scope: ``i`` (output index), ``fw_out`` (FP64 golden), ``impl_out``,
    ``_akg_secondary_ref_i`` (native ref for excusal), ``_akg_op_name``,
    ``inputs_for_impl``, ``impl_output``. op_name + full inputs/outputs are
    forwarded so a registered index output (TopK/Cummin idx) runs the
    tie-independent ``x.gather(dim,idx)==values`` check instead of element-wise
    compare — that logic all lives in cann_correctness."""
    return (
        "import cann_correctness\n"
        "cann_correctness.assert_outputs("
        "impl_out, fw_out, index=i, native_out=_akg_secondary_ref_i, "
        "op_name=_akg_op_name, impl_inputs=inputs_for_impl, "
        "all_impl_outputs=impl_output)\n"
    )
