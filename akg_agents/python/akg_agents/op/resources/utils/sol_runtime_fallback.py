"""Minimal SOL-ExecBench runtime fallback used by generated verifier scripts.

This module is intentionally small.  Official ``sol_execbench`` is preferred
when installed; the fallback only covers the common tensor-input path so local
mock cases can still run in environments where the external runtime package is
not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch


@dataclass
class Definition:
    raw: Dict[str, Any]

    @classmethod
    def model_validate(cls, data: Mapping[str, Any]) -> "Definition":
        return cls(dict(data))

    @property
    def axes(self) -> Dict[str, Any]:
        return dict(self.raw.get("axes") or {})

    @property
    def inputs(self) -> Dict[str, Any]:
        return dict(self.raw.get("inputs") or {})

    @property
    def custom_inputs_entrypoint(self) -> Optional[str]:
        return self.raw.get("custom_inputs_entrypoint")


@dataclass
class Workload:
    raw: Dict[str, Any]

    @classmethod
    def model_validate(cls, data: Mapping[str, Any]) -> "Workload":
        return cls(dict(data))

    @property
    def axes(self) -> Dict[str, Any]:
        return dict(self.raw.get("axes") or {})

    @property
    def inputs(self) -> Dict[str, Any]:
        return dict(self.raw.get("inputs") or {})


def gen_inputs(
    *,
    definition: Definition,
    workload: Workload,
    device,
    safe_tensors=None,
    custom_inputs_fn=None,
):
    if custom_inputs_fn is not None:
        return _call_custom_inputs_fn(
            custom_inputs_fn,
            definition=definition,
            workload=workload,
            device=device,
            safe_tensors=safe_tensors,
        )

    axes = _resolve_axes(definition, workload)
    inputs = []
    for name, input_def in definition.inputs.items():
        workload_input = workload.inputs.get(name, {})
        inputs.append(_make_input(input_def, workload_input, axes, device))
    return inputs


def _resolve_axes(definition: Definition, workload: Workload) -> Dict[str, Any]:
    axes: Dict[str, Any] = {}
    for key, spec in definition.axes.items():
        if isinstance(spec, Mapping) and spec.get("type") == "const":
            axes[key] = spec.get("value")
    axes.update(workload.axes)
    return axes


def _make_input(
    input_def: Mapping[str, Any],
    workload_input: Mapping[str, Any],
    axes: Mapping[str, Any],
    device,
):
    dtype = _to_torch_dtype(str(input_def.get("dtype") or "float32"))
    shape = _resolve_shape(input_def.get("shape") or [], axes)

    if "value" in workload_input:
        return torch.as_tensor(workload_input["value"], dtype=dtype, device=device)

    input_type = str(workload_input.get("type") or "random").lower()
    if input_type in {"zero", "zeros"}:
        return torch.zeros(shape, dtype=dtype, device=device)
    if input_type in {"one", "ones"}:
        return torch.ones(shape, dtype=dtype, device=device)
    if input_type in {"empty"}:
        return torch.empty(shape, dtype=dtype, device=device)
    if input_type in {"randint", "random_int", "int"} or not dtype.is_floating_point:
        low = int(workload_input.get("low", 0))
        high = int(workload_input.get("high", 10))
        if dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
        return torch.randint(low, high, shape, dtype=dtype, device=device)
    if input_type in {"randn", "normal"}:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.rand(shape, dtype=dtype, device=device)


def _resolve_shape(shape_spec, axes: Mapping[str, Any]) -> tuple[int, ...]:
    resolved = []
    for dim in shape_spec:
        if isinstance(dim, str):
            if dim not in axes:
                raise KeyError(f"Missing axis value for SOL dimension {dim!r}")
            dim = axes[dim]
        resolved.append(int(dim))
    return tuple(resolved)


def _to_torch_dtype(dtype_name: str):
    mapping = {
        "bool": torch.bool,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported SOL fallback dtype: {dtype_name}")
    return mapping[dtype_name]


def _call_custom_inputs_fn(custom_inputs_fn, **kwargs):
    attempts = (
        kwargs,
        {
            "definition": kwargs["definition"],
            "workload": kwargs["workload"],
            "device": kwargs["device"],
        },
        {"device": kwargs["device"]},
        {},
    )
    for call_kwargs in attempts:
        try:
            return custom_inputs_fn(**call_kwargs)
        except TypeError:
            continue
    raise TypeError("Unable to call SOL custom_inputs_fn with supported signatures")
