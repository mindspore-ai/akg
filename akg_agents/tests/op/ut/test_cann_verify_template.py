# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from pathlib import Path

from akg_agents.op.cann_correctness import TEMPLATES_DIR


def _read_cann_verify_template() -> str:
    template_path = Path(TEMPLATES_DIR) / "verify_cann.j2"
    return template_path.read_text(encoding="utf-8")


def test_cann_verify_template_preserves_integer_golden_inputs():
    content = _read_cann_verify_template()

    assert "_to_cpu_golden_dtype" in content
    assert "is_floating_point()" in content
    assert "val.detach().cpu().double()" not in content
    assert "v.detach().cpu().double()" not in content


def test_cann_verify_template_matches_kernel_eval_input_flow():
    content = _read_cann_verify_template()

    assert 'get_input_fn = getattr(golden, "get_input", None)' in content
    assert "_build_params_from_signature(golden_fn, input_tensors, attrs)" in content
    assert "native_outputs = golden_fn(**native_kwargs)" in content
    assert "native_output=native_outputs" in content
