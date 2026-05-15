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

"""SOLAR roofline 集成工具。

设计约束：
1. 不修改 / patch SOLAR 仓库。
2. AKG 运行时只依赖“已安装的 solar Python 包”，不依赖本地 SOLAR 工作树。
3. 之前只存在于本地 SOLAR 改动里的辅助逻辑（如 solbench wrapper、Ascend arch config）
   迁移到 AKG 自己维护。
4. roofline 失败只能降级，不能影响原有 profile 主流程。
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

ROOFLINE_MODEL = "fused"
ROOFLINE_RESULT_JSON = "roofline_profile_result.json"
SOLAR_INSTALL_HINT = "bash ./download.sh --with_solar"

ARCH_ALIAS_TO_CONFIG_KEY = {
    "a100": "a100",
    "v100": "v100",
    "ascend910b1": "ascend910b1",
    "ascend910b2": "ascend910b2",
    "ascend910b2c": "ascend910b2",
    "ascend910b3": "ascend910b3",
    "ascend910b4": "ascend910b4",
    "ascend910_9362": "ascend910b4",
    "ascend910_9372": "ascend910b4",
    "ascend910_9381": "ascend910b4",
    "ascend910_9382": "ascend910b4",
    "ascend910_9391": "ascend910b4",
    "ascend910_9392": "ascend910b4",
    "ascend950dt_95a": "ascend950_pr",
    "ascend950pr_950z": "ascend950_pr",
    "ascend950pr_9572": "ascend950_pr",
    "ascend950pr_9574": "ascend950_pr",
    "ascend950pr_9575": "ascend950_pr",
    "ascend950pr_9576": "ascend950_pr",
    "ascend950pr_9577": "ascend950_pr",
    "ascend950pr_9578": "ascend950_pr",
    "ascend950pr_9579": "ascend950_pr",
    "ascend950pr_957b": "ascend950_pr",
    "ascend950pr_957d": "ascend950_pr",
    "ascend950pr_9581": "ascend950_pr",
    "ascend950pr_9582": "ascend950_pr",
    "ascend950pr_9584": "ascend950_pr",
    "ascend950pr_9587": "ascend950_pr",
    "ascend950pr_9588": "ascend950_pr",
    "ascend950pr_9589": "ascend950_pr",
    "ascend950pr_958a": "ascend950_pr",
    "ascend950pr_958b": "ascend950_pr",
    "ascend950pr_9591": "ascend950_pr",
    "ascend950pr_9592": "ascend950_pr",
    "ascend950pr_9599": "ascend950_pr",
}

# 这些配置原先依赖本地 SOLAR 改动；现在由 AKG 自己维护，避免依赖 drop commit。
AKG_ROOFLINE_ARCH_CONFIGS = {
    "a100": {
        "name": "A100",
        "freq_GHz": 1.41,
        "DRAM_byte_per_cycle": 1935e9 / 1.41e9,
        "MAC_per_cycle_fp16_tc": 312e12 / (2 * 1.41e9),
        "MAC_per_cycle_fp32_tc": 19.5e12 / (2 * 1.41e9),
    },
    "v100": {
        "name": "V100",
        "freq_GHz": 1.38,
        "DRAM_byte_per_cycle": 900e9 / 1.38e9,
        "MAC_per_cycle_fp16_tc": 112e12 / (2 * 1.38e9),
        "MAC_per_cycle_fp32_tc": 14e12 / (2 * 1.38e9),
    },
    "ascend910b1": {
        "name": "Ascend910B1",
        "freq_GHz": 1.5,
        "DRAM_byte_per_cycle": 1.8e12 / 1.5e9,
        "MAC_per_cycle_fp16_tc": 245e12 / (2 * 1.5e9),
        "MAC_per_cycle_fp32_tc": 61e12 / (2 * 1.5e9),
    },
    "ascend910b2": {
        "name": "Ascend910B2",
        "freq_GHz": 1.5,
        "DRAM_byte_per_cycle": 1.8e12 / 1.5e9,
        "MAC_per_cycle_fp16_tc": 245e12 / (2 * 1.5e9),
        "MAC_per_cycle_fp32_tc": 61e12 / (2 * 1.5e9),
    },
    "ascend910b3": {
        "name": "Ascend910B3",
        "freq_GHz": 1.5,
        "DRAM_byte_per_cycle": 1.6e12 / 1.5e9,
        "MAC_per_cycle_fp16_tc": 245e12 / (2 * 1.5e9),
        "MAC_per_cycle_fp32_tc": 61e12 / (2 * 1.5e9),
    },
    "ascend910b4": {
        "name": "Ascend910B4",
        "freq_GHz": 1.5,
        "DRAM_byte_per_cycle": 0.8e12 / 1.5e9,
        "MAC_per_cycle_fp16_tc": 245e12 / (2 * 1.5e9),
        "MAC_per_cycle_fp32_tc": 61e12 / (2 * 1.5e9),
    },
    "ascend950_pr": {
        "name": "Ascend950_PR",
        "freq_GHz": 1.65,
        "DRAM_byte_per_cycle": 1.6e12 / 1.65e9,
        "MAC_per_cycle_fp16_tc": 380e12 / (2 * 1.65e9),
        "MAC_per_cycle_fp32_tc": 27e12 / (2 * 1.65e9),
    },
}

_PRECISION_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "float": "fp32",
    "torch.float32": "fp32",
    "torch.float": "fp32",
    "tf32": "tf32",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "torch.float16": "fp16",
    "torch.half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "fp64": "fp64",
    "float64": "fp64",
    "double": "fp64",
    "torch.float64": "fp64",
    "torch.double": "fp64",
    "int8": "int8",
    "torch.int8": "int8",
    "fp8": "fp8",
    "float8": "fp8",
    "nvfp4": "nvfp4",
    "fp4": "nvfp4",
    "float4": "nvfp4",
}

_PREFERRED_PRECISION_ORDER = ["nvfp4", "fp8", "bf16", "fp16", "fp32", "int8", "fp64"]


def compute_roofline_profile(
    verify_dir: str,
    op_name: str,
    task_id: str,
    profile_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """为当前 verify_dir 计算 SOLAR roofline。"""
    verify_path = Path(verify_dir)
    bench_type = _infer_bench_type(verify_path, profile_settings.get("bench_type"))
    backend = str(profile_settings.get("backend", "")).lower()
    arch = str(profile_settings.get("arch", "")).lower()
    framework = str(profile_settings.get("framework", "torch")).lower()
    precision_override = profile_settings.get("roofline_precision")

    if not profile_settings.get("enable_roofline", True):
        return _skipped_result("roofline 已显式关闭", bench_type=bench_type, arch=arch)

    if backend not in {"cuda", "ascend"}:
        return _skipped_result(
            f"backend={backend} 当前不支持 roofline",
            bench_type=bench_type,
            arch=arch,
        )

    solar_api, import_error = _import_solar_api()
    if solar_api is None:
        return _skipped_result(
            "未安装 solar Python 包。"
            f"可执行: {SOLAR_INSTALL_HINT}. 原始错误: {import_error}",
            bench_type=bench_type,
            arch=arch,
        )

    arch_spec = resolve_arch_spec(
        arch=arch,
        verify_dir=verify_path,
        explicit_arch_config=profile_settings.get("roofline_arch_config"),
    )
    if arch_spec is None:
        return _skipped_result(
            f"arch={arch} 当前没有可用的 roofline 架构配置",
            bench_type=bench_type,
            arch=arch,
        )

    try:
        if bench_type == "sol":
            result = _compute_sol_roofline(
                verify_dir=verify_path,
                op_name=op_name,
                task_id=task_id,
                solar_api=solar_api,
                arch_spec=arch_spec,
                precision_override=precision_override,
            )
        else:
            result = _compute_kernelbench_roofline(
                verify_dir=verify_path,
                op_name=op_name,
                framework=framework,
                task_id=task_id,
                solar_api=solar_api,
                arch_spec=arch_spec,
                precision_override=precision_override,
            )
        result.setdefault("bench_type", bench_type)
        result.setdefault("arch", arch)
        result.setdefault("task_id", task_id)
        result.setdefault("op_name", op_name)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s:%s] roofline 计算失败: %s", task_id, op_name, exc, exc_info=True)
        return {
            "success": False,
            "skipped": False,
            "source": "solar",
            "model": ROOFLINE_MODEL,
            "bench_type": bench_type,
            "arch": arch,
            "op_name": op_name,
            "task_id": task_id,
            "error": str(exc),
        }


def augment_roofline_metrics(
    roofline_result: Dict[str, Any],
    gen_time_us: Optional[float],
    base_time_us: Optional[float] = None,
) -> Dict[str, Any]:
    """补充与 profile 实测时间相关的 roofline 指标。"""
    augmented = dict(roofline_result or {})
    roofline_time = augmented.get("time_us")
    gen_valid = _is_valid_positive_number(gen_time_us)
    base_valid = _is_valid_positive_number(base_time_us)
    roofline_valid = _is_valid_positive_number(roofline_time)

    if roofline_valid and gen_valid:
        augmented["speedup_vs_generated"] = float(roofline_time) / float(gen_time_us)
        augmented["gap_vs_generated"] = float(gen_time_us) / float(roofline_time)
    else:
        augmented["speedup_vs_generated"] = 0.0
        augmented["gap_vs_generated"] = None

    if roofline_valid and base_valid:
        augmented["speedup_vs_baseline"] = float(roofline_time) / float(base_time_us)
    else:
        augmented["speedup_vs_baseline"] = 0.0

    return augmented


def write_roofline_profile_result(verify_dir: str, roofline_result: Dict[str, Any]) -> str:
    """将 roofline 结果写入 verify_dir/roofline_profile_result.json。"""
    output_path = Path(verify_dir) / ROOFLINE_RESULT_JSON
    output_path.write_text(
        json.dumps(_sanitize_for_json(roofline_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path)


def resolve_arch_spec(
    arch: str,
    verify_dir: Path,
    explicit_arch_config: Optional[str] = None,
) -> Optional[str]:
    """将 AKG arch 解析为 roofline arch-config 参数。"""
    if explicit_arch_config:
        explicit_path = Path(os.path.expanduser(explicit_arch_config))
        return str(explicit_path.resolve()) if explicit_path.exists() else explicit_arch_config

    config_key = ARCH_ALIAS_TO_CONFIG_KEY.get(arch)
    arch_config = AKG_ROOFLINE_ARCH_CONFIGS.get(config_key or "")
    if arch_config is None:
        return None

    custom_dir = verify_dir / "_roofline_arch"
    custom_dir.mkdir(parents=True, exist_ok=True)
    custom_path = custom_dir / f"{config_key}.yaml"
    custom_path.write_text(yaml.safe_dump(arch_config, sort_keys=False), encoding="utf-8")
    return str(custom_path)


def _compute_kernelbench_roofline(
    verify_dir: Path,
    op_name: str,
    framework: str,
    task_id: str,
    solar_api: Dict[str, Any],
    arch_spec: str,
    precision_override: Optional[str] = None,
) -> Dict[str, Any]:
    source_file = _find_framework_source_file(verify_dir, op_name, framework)
    case_output_dir = verify_dir / "_roofline_kernelbench"
    case_result = _compute_single_case_roofline(
        source_file=source_file,
        case_output_dir=case_output_dir,
        solar_api=solar_api,
        arch_spec=arch_spec,
        task_id=task_id,
        case_label=source_file.stem,
        precision_override=precision_override,
    )
    return _aggregate_case_results([case_result], bench_type="kernelbench")


def _compute_sol_roofline(
    verify_dir: Path,
    op_name: str,
    task_id: str,
    solar_api: Dict[str, Any],
    arch_spec: str,
    precision_override: Optional[str] = None,
) -> Dict[str, Any]:
    workload_path = verify_dir / "workload.jsonl"
    if not workload_path.is_file():
        raise FileNotFoundError(f"SOL workload 文件不存在: {workload_path}")

    workloads = [line for line in workload_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not workloads:
        raise ValueError(f"SOL workload 为空: {workload_path}")

    wrappers_dir = verify_dir / "_roofline_sol_wrappers"
    wrappers_dir.mkdir(parents=True, exist_ok=True)

    case_results = []
    for workload_idx in range(len(workloads)):
        wrapper_file = wrappers_dir / f"{op_name}_w{workload_idx:03d}.py"
        _create_solbench_wrapper(verify_dir, wrapper_file, workload_idx)

        case_output_dir = verify_dir / "_roofline_sol" / f"w{workload_idx:03d}"
        case_results.append(
            _compute_single_case_roofline(
                source_file=wrapper_file,
                case_output_dir=case_output_dir,
                solar_api=solar_api,
                arch_spec=arch_spec,
                task_id=task_id,
                case_label=f"w{workload_idx:03d}",
                precision_override=precision_override,
            )
        )

    result = _aggregate_case_results(case_results, bench_type="sol")
    result["workload_count"] = len(case_results)
    return result


def _compute_single_case_roofline(
    source_file: Path,
    case_output_dir: Path,
    solar_api: Dict[str, Any],
    arch_spec: str,
    task_id: str,
    case_label: str,
    precision_override: Optional[str] = None,
) -> Dict[str, Any]:
    graph_dir = case_output_dir / "graph"
    einsum_dir = case_output_dir / "einsum"
    analysis_dir = case_output_dir / "analysis"
    perf_dir = case_output_dir / "perf"
    graph_dir.mkdir(parents=True, exist_ok=True)
    einsum_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    processing_config = solar_api["ProcessingConfig"](
        save_graph=False,
        force_rerun=True,
        timeout=600,
        output_dir=str(graph_dir),
        debug=False,
        safe_mode=True,
    )
    processor = solar_api["PyTorchProcessor"](processing_config)
    if not processor.process_model_file(str(source_file), str(graph_dir)):
        raise RuntimeError(f"[process_model] 处理失败: {source_file}")

    graph_path = graph_dir / "pytorch_graph.yaml"
    if not graph_path.is_file():
        raise FileNotFoundError(f"未生成 pytorch_graph.yaml: {graph_path}")

    precision = _normalize_precision_name(precision_override) if precision_override else _infer_graph_precision(graph_path)

    converter = solar_api["PyTorchToEinsum"](
        debug=False,
        enable_agent=False,
        cache_dir=str(case_output_dir / "solar_handlers_cache"),
    )
    convert_result = converter.convert(
        graph_path,
        einsum_dir,
        copy_graph=True,
        expand_complex_ops=True,
        enable_rename=True,
    )
    if convert_result is None:
        raise RuntimeError(f"[toeinsum_model] 转换失败: {graph_path}")

    einsum_graph_path = einsum_dir / "einsum_graph_renamed.yaml"
    if not einsum_graph_path.is_file():
        raise FileNotFoundError(f"未生成 einsum_graph_renamed.yaml: {einsum_graph_path}")

    analyzer = solar_api["EinsumGraphAnalyzer"](debug=False)
    analysis_result = analyzer.analyze_graph(
        einsum_graph_path,
        analysis_dir,
        precision=precision,
        copy_graph=True,
    )
    if analysis_result is None:
        raise RuntimeError(f"[analyze_model] 分析失败: {einsum_graph_path}")

    analysis_path = analysis_dir / "analysis.yaml"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"未生成 analysis.yaml: {analysis_path}")

    perf_model = solar_api["EinsumGraphPerfModel"](debug=False)
    perf_result = perf_model.predict(
        analysis_path,
        perf_dir,
        arch_config=arch_spec,
        precision=precision,
        copy_analysis=True,
    )
    if perf_result is None:
        raise RuntimeError(f"[predict_perf_model] 预测失败: {analysis_path}")

    perf_data = perf_result
    arch_info = perf_data.get("arch") or {}
    fused_info = perf_data.get(ROOFLINE_MODEL) or {}
    freq_ghz = float(arch_info.get("freq_GHz") or 0.0)
    runtime_ms = float(fused_info.get("runtime_ms") or 0.0)
    compute_cycles = float(fused_info.get("compute_cycles") or 0.0)
    memory_cycles = float(fused_info.get("memory_cycles") or 0.0)

    return {
        "success": True,
        "case_label": case_label,
        "precision": precision,
        "arch_name": str(arch_info.get("name") or arch_spec),
        "time_us": runtime_ms * 1000.0,
        "compute_time_us": _cycles_to_us(compute_cycles, freq_ghz),
        "memory_time_us": _cycles_to_us(memory_cycles, freq_ghz),
        "bottleneck": str(fused_info.get("bottleneck") or ""),
    }


def _aggregate_case_results(case_results: list[Dict[str, Any]], bench_type: str) -> Dict[str, Any]:
    if not case_results:
        raise ValueError("没有可聚合的 roofline case 结果")

    failures = [item for item in case_results if not item.get("success")]
    if failures:
        first = failures[0]
        return {
            "success": False,
            "skipped": False,
            "source": "solar",
            "model": ROOFLINE_MODEL,
            "bench_type": bench_type,
            "error": first.get("error") or "roofline case failed",
            "case_count": len(case_results),
        }

    case_labels = [str(item["case_label"]) for item in case_results]
    case_times_us = [float(item["time_us"]) for item in case_results]
    compute_times_us = [float(item["compute_time_us"]) for item in case_results]
    memory_times_us = [float(item["memory_time_us"]) for item in case_results]
    bottlenecks = [str(item.get("bottleneck") or "") for item in case_results if item.get("bottleneck")]
    precisions = [str(item.get("precision") or "") for item in case_results if item.get("precision")]
    arch_names = [str(item.get("arch_name") or "") for item in case_results if item.get("arch_name")]

    return {
        "success": True,
        "skipped": False,
        "source": "solar",
        "model": ROOFLINE_MODEL,
        "bench_type": bench_type,
        "case_count": len(case_results),
        "case_labels": case_labels,
        "case_times_us": case_times_us,
        "time_us": _geomean(case_times_us),
        "compute_time_us": _geomean(compute_times_us),
        "memory_time_us": _geomean(memory_times_us),
        "bottleneck": _merge_strings_keep_mixed(bottlenecks),
        "precision": _merge_strings_keep_mixed(precisions),
        "arch_name": _merge_strings_keep_mixed(arch_names),
    }


def _create_solbench_wrapper(verify_dir: Path, wrapper_path: Path, workload_idx: int) -> None:
    """为单个 workload 生成 SOLBench reference wrapper。

    这段逻辑来自原本本地 SOLAR 改动中的 `scripts/solbench.py`，现迁到 AKG 内部维护。
    """
    definition_path = verify_dir / "definition.json"
    reference_path = verify_dir / "reference.py"
    workload_path = verify_dir / "workload.jsonl"

    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    input_order = list((definition.get("inputs") or {}).keys())
    entrypoint = definition.get("custom_inputs_entrypoint") or "get_inputs"
    axes_spec = definition.get("axes") or {}
    inputs_spec = definition.get("inputs") or {}

    code = f"""
import importlib.util
import json
import math
from pathlib import Path

import torch

_REFERENCE_PATH = Path(r"{reference_path}")
_WORKLOAD_PATH = Path(r"{workload_path}")
_WORKLOAD_IDX = {workload_idx}
_INPUT_ORDER = {input_order!r}
_INPUT_ENTRYPOINT = {entrypoint!r}
_AXES_SPEC = {axes_spec!r}
_INPUTS_SPEC = {inputs_spec!r}

def _load_reference():
    spec = importlib.util.spec_from_file_location("solbench_reference_module", _REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

_reference = _load_reference()

def _resolve_axes(workload_axes):
    resolved = dict(workload_axes)
    pending = dict(_AXES_SPEC)
    for _ in range(len(pending) + 4):
        changed = False
        for name, spec in list(pending.items()):
            axis_type = spec.get("type")
            if axis_type == "var":
                pending.pop(name, None)
                continue
            if axis_type == "const":
                resolved[name] = spec.get("value")
                pending.pop(name, None)
                changed = True
                continue
            if axis_type == "expr":
                expr = spec.get("expression")
                try:
                    resolved[name] = eval(expr, {{"__builtins__": {{}}, "math": math}}, resolved)  # noqa: S307
                except Exception:
                    continue
                pending.pop(name, None)
                changed = True
        if not pending or not changed:
            break
    return resolved

def _map_dtype(dtype_name):
    name = str(dtype_name).lower()
    mapping = {{
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float64": torch.float64,
        "double": torch.float64,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "int": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }}
    return mapping.get(name, torch.float32)

def _resolve_shape(shape_spec, axes):
    if shape_spec is None:
        return None
    dims = []
    for dim in shape_spec:
        if isinstance(dim, int):
            dims.append(dim)
        elif isinstance(dim, str):
            if dim in axes:
                dims.append(int(axes[dim]))
            else:
                dims.append(int(eval(dim, {{"__builtins__": {{}}, "math": math}}, axes)))  # noqa: S307
        else:
            dims.append(int(dim))
    return dims

def _make_scalar(spec):
    dtype_name = str(spec.get("dtype", "float32")).lower()
    desc = str(spec.get("description", "")).lower()
    if "eps" in desc or "epsilon" in desc:
        return 1e-5
    if "dropout" in desc:
        return 0.1
    if "bool" in dtype_name:
        return False
    if dtype_name.startswith("int") or dtype_name in {{"long", "uint8"}}:
        return 1
    return 1.0

def _make_tensor(shape, dtype, device):
    shape_tuple = tuple(shape)
    if dtype == torch.bool:
        if len(shape_tuple) == 0:
            return torch.rand((), device=device) > 0.5
        return torch.rand(*shape_tuple, device=device) > 0.5
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        high = max(2, min(1024, shape_tuple[-1] if shape_tuple else 16))
        if len(shape_tuple) == 0:
            return torch.randint(0, high, (), dtype=dtype, device=device)
        return torch.randint(0, high, shape_tuple, dtype=dtype, device=device)
    if len(shape_tuple) == 0:
        return torch.randn((), dtype=dtype, device=device)
    return torch.randn(*shape_tuple, dtype=dtype, device=device)

def _build_inputs_from_definition(axes, device):
    built = {{}}
    for name in _INPUT_ORDER:
        spec = _INPUTS_SPEC[name]
        shape = _resolve_shape(spec.get("shape"), axes)
        dtype = _map_dtype(spec.get("dtype", "float32"))
        if shape is None:
            built[name] = _make_scalar(spec)
        else:
            built[name] = _make_tensor(shape, dtype, device)
    return built

class ReferenceModel(torch.nn.Module):
    def forward(self, *args):
        return _reference.run(*args)

def get_inputs():
    with _WORKLOAD_PATH.open() as f:
        for idx, line in enumerate(f):
            if idx == _WORKLOAD_IDX:
                workload = json.loads(line)
                break
        else:
            raise IndexError(f"workload index {{_WORKLOAD_IDX}} out of range for {{_WORKLOAD_PATH}}")
    axes = _resolve_axes(workload.get("axes") or {{}})
    ref_get_inputs = getattr(_reference, _INPUT_ENTRYPOINT, None)
    if ref_get_inputs is not None:
        generated = ref_get_inputs(axes, torch.device("cpu"))
    else:
        generated = _build_inputs_from_definition(axes, torch.device("cpu"))
    if isinstance(generated, dict):
        return [generated[name] for name in _INPUT_ORDER]
    return generated

def launch_reference_implementation(model, inputs):
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)
"""
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(code.strip() + "\n", encoding="utf-8")


def _import_solar_api() -> tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        common_types = importlib.import_module("solar.common.types")
        graph_mod = importlib.import_module("solar.graph")
        einsum_mod = importlib.import_module("solar.einsum")
        analysis_mod = importlib.import_module("solar.analysis")
        perf_mod = importlib.import_module("solar.perf")
        return (
            {
                "ProcessingConfig": common_types.ProcessingConfig,
                "PyTorchProcessor": graph_mod.PyTorchProcessor,
                "PyTorchToEinsum": einsum_mod.PyTorchToEinsum,
                "EinsumGraphAnalyzer": analysis_mod.EinsumGraphAnalyzer,
                "EinsumGraphPerfModel": perf_mod.EinsumGraphPerfModel,
            },
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return None, exc


def _find_framework_source_file(verify_dir: Path, op_name: str, framework: str) -> Path:
    exact = verify_dir / f"{op_name}_{framework}.py"
    if exact.is_file():
        return exact

    matches = sorted(verify_dir.glob(f"*_{framework}.py"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"未找到 framework 源文件: expected={exact} or any '*_{framework}.py' in {verify_dir}"
    )


def _infer_bench_type(verify_dir: Path, explicit_bench_type: Optional[str]) -> str:
    if explicit_bench_type in {"sol", "kernelbench"}:
        return explicit_bench_type
    if (verify_dir / "definition.json").is_file() and (verify_dir / "workload.jsonl").is_file():
        return "sol"
    return "kernelbench"


def _normalize_precision_name(raw: Optional[str]) -> str:
    if raw is None:
        return "fp32"
    text = str(raw).strip().lower()
    if text not in _PRECISION_ALIASES:
        raise ValueError(f"不支持的 precision: {raw}")
    return _PRECISION_ALIASES[text]


def _infer_graph_precision(graph_path: Path) -> str:
    graph = yaml.safe_load(graph_path.read_text(encoding="utf-8")) or {}
    counts: Dict[str, int] = {}

    for layer in (graph.get("layers") or {}).values():
        for key in ("input_dtypes", "output_dtypes"):
            for dtype in layer.get(key) or []:
                mapped = _map_graph_dtype(dtype)
                if mapped is None:
                    continue
                counts[mapped] = counts.get(mapped, 0) + 1

    if not counts:
        return "fp32"

    return sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            _PREFERRED_PRECISION_ORDER.index(item[0]) if item[0] in _PREFERRED_PRECISION_ORDER else 999,
            item[0],
        ),
    )[0][0]


def _map_graph_dtype(raw: Any) -> Optional[str]:
    text = str(raw).strip().lower()
    if not text or text in {"torch.bool", "bool"}:
        return None
    if text.startswith("torch.int") or text.startswith("torch.uint") or text in {"int", "long", "torch.long"}:
        return None
    try:
        return _normalize_precision_name(text)
    except ValueError:
        return None


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("几何平均输入为空")
    safe_values = [max(float(v), 1e-12) for v in values]
    return math.exp(sum(math.log(v) for v in safe_values) / len(safe_values))


def _merge_strings_keep_mixed(values: list[str]) -> Optional[str]:
    filtered = [value for value in values if value]
    if not filtered:
        return None
    if all(value == filtered[0] for value in filtered):
        return filtered[0]
    return "mixed"


def _cycles_to_us(cycles: float, freq_ghz: float) -> float:
    return cycles / (freq_ghz * 1e3) if freq_ghz > 0 else 0.0


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _is_valid_positive_number(value: Optional[float]) -> bool:
    return value is not None and isinstance(value, (int, float)) and float(value) > 0 and math.isfinite(float(value))


def _skipped_result(reason: str, bench_type: str, arch: str) -> Dict[str, Any]:
    return {
        "success": False,
        "skipped": True,
        "source": "solar",
        "model": ROOFLINE_MODEL,
        "bench_type": bench_type,
        "arch": arch,
        "error": reason,
    }
