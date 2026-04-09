import ast
import logging
import re
from typing import Dict, Optional

import torch
from torch.fx import symbolic_trace
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.shape_prop import ShapeProp

logger = logging.getLogger(__name__)

_DANGEROUS_MODULES = {
    "os", "subprocess", "shutil", "sys", "importlib", "ctypes", "socket",
    "http", "ftplib", "smtplib", "pathlib", "tempfile", "signal", "multiprocessing",
}
_DANGEROUS_CALLS = {
    "system", "popen", "exec", "eval", "compile", "execfile", "__import__",
    "globals", "locals", "breakpoint",
}


def _validate_code_safety(code_str: str) -> ast.AST:
    """Run minimal AST checks before exec to block obvious dangerous patterns."""
    if not isinstance(code_str, str) or not code_str.strip():
        raise ValueError("code_str must be a non-empty Python source string.")

    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        raise ValueError(f"Code string is not valid Python: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in _DANGEROUS_MODULES:
                    raise ValueError(
                        f"Importing dangerous module '{alias.name}' is not allowed."
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_root = node.module.split(".")[0]
            if module_root in _DANGEROUS_MODULES:
                raise ValueError(
                    f"Importing from dangerous module '{node.module}' is not allowed."
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_CALLS:
                raise ValueError(
                    f"Calling dangerous function '{func.id}' is not allowed."
                )
            if isinstance(func, ast.Attribute) and func.attr in _DANGEROUS_CALLS:
                raise ValueError(
                    f"Calling dangerous attribute '{func.attr}' is not allowed."
                )
    return tree


# ── AST 静态语义识别词表 ────────────────────────────────────────────────────
# KernelBench 算子代码中常见的 nn.Xxx 类名 → 语义标签
_AST_NN_TAGS: Dict[str, list] = {
    # 矩阵类
    "Linear":             ["is_matrix", "is_compute_heavy"],
    "Bilinear":           ["is_matrix", "is_compute_heavy"],
    "MultiheadAttention": ["is_matrix", "is_compute_heavy", "has_online_reduction", "is_reduction"],
    # 卷积类
    "Conv1d":             ["is_conv", "is_compute_heavy"],
    "Conv2d":             ["is_conv", "is_compute_heavy"],
    "Conv3d":             ["is_conv", "is_compute_heavy"],
    "ConvTranspose1d":    ["is_conv", "is_compute_heavy"],
    "ConvTranspose2d":    ["is_conv", "is_compute_heavy"],
    "ConvTranspose3d":    ["is_conv", "is_compute_heavy"],
    # 归一化类（在线归约）
    "BatchNorm1d":        ["is_reduction", "has_online_reduction"],
    "BatchNorm2d":        ["is_reduction", "has_online_reduction"],
    "BatchNorm3d":        ["is_reduction", "has_online_reduction"],
    "LayerNorm":          ["is_reduction", "has_online_reduction"],
    "GroupNorm":          ["is_reduction", "has_online_reduction"],
    "InstanceNorm1d":     ["is_reduction", "has_online_reduction"],
    "InstanceNorm2d":     ["is_reduction", "has_online_reduction"],
    "InstanceNorm3d":     ["is_reduction", "has_online_reduction"],
    "RMSNorm":            ["is_reduction", "has_online_reduction"],
    "Softmax":            ["is_reduction", "has_online_reduction"],
    "LogSoftmax":         ["is_reduction", "has_online_reduction"],
    # Epilogue 类
    "ReLU":               ["has_epilogue"],
    "ReLU6":              ["has_epilogue"],
    "LeakyReLU":          ["has_epilogue"],
    "GELU":               ["has_epilogue"],
    "SiLU":               ["has_epilogue"],
    "Sigmoid":            ["has_epilogue"],
    "Tanh":               ["has_epilogue"],
    "Hardswish":          ["has_epilogue"],
    "Hardsigmoid":        ["has_epilogue"],
    "Dropout":            ["has_epilogue"],
    "ELU":                ["has_epilogue"],
    "PReLU":              ["has_epilogue"],
    "Mish":               ["has_epilogue"],
}

# F.xxx / torch.nn.functional.xxx 函数调用 → 语义标签
_AST_F_TAGS: Dict[str, list] = {
    "linear":              ["is_matrix", "is_compute_heavy"],
    "conv1d":              ["is_conv", "is_compute_heavy"],
    "conv2d":              ["is_conv", "is_compute_heavy"],
    "conv3d":              ["is_conv", "is_compute_heavy"],
    "conv_transpose1d":    ["is_conv", "is_compute_heavy"],
    "conv_transpose2d":    ["is_conv", "is_compute_heavy"],
    "conv_transpose3d":    ["is_conv", "is_compute_heavy"],
    "batch_norm":          ["is_reduction", "has_online_reduction"],
    "layer_norm":          ["is_reduction", "has_online_reduction"],
    "group_norm":          ["is_reduction", "has_online_reduction"],
    "instance_norm":       ["is_reduction", "has_online_reduction"],
    "normalize":           ["is_reduction", "has_online_reduction"],
    "softmax":             ["is_reduction", "has_online_reduction"],
    "log_softmax":         ["is_reduction", "has_online_reduction"],
    "max_pool1d":          ["is_reduction"],
    "max_pool2d":          ["is_reduction"],
    "max_pool3d":          ["is_reduction"],
    "avg_pool1d":          ["is_reduction"],
    "avg_pool2d":          ["is_reduction"],
    "avg_pool3d":          ["is_reduction"],
    "adaptive_avg_pool1d": ["is_reduction"],
    "adaptive_avg_pool2d": ["is_reduction"],
    "adaptive_avg_pool3d": ["is_reduction"],
    "adaptive_max_pool1d": ["is_reduction"],
    "adaptive_max_pool2d": ["is_reduction"],
    "adaptive_max_pool3d": ["is_reduction"],
    "relu":                ["has_epilogue"],
    "relu6":               ["has_epilogue"],
    "leaky_relu":          ["has_epilogue"],
    "gelu":                ["has_epilogue"],
    "silu":                ["has_epilogue"],
    "sigmoid":             ["has_epilogue"],
    "tanh":                ["has_epilogue"],
    "hardswish":           ["has_epilogue"],
    "hardtanh":            ["has_epilogue"],
    "elu":                 ["has_epilogue"],
    "dropout":             ["has_epilogue"],
    "mish":                ["has_epilogue"],
}

# torch.xxx 顶层函数调用 → 语义标签
_AST_TORCH_FUNC_TAGS: Dict[str, list] = {
    "matmul":      ["is_matrix", "is_compute_heavy", "is_reduction"],
    "mm":          ["is_matrix", "is_compute_heavy", "is_reduction"],
    "bmm":         ["is_matrix", "is_compute_heavy", "is_reduction"],
    "addmm":       ["is_matrix", "is_compute_heavy", "is_reduction"],
    "einsum":      ["is_matrix", "is_compute_heavy", "is_reduction"],
    "dot":         ["is_matrix", "is_compute_heavy", "is_reduction"],
    "inner":       ["is_matrix", "is_compute_heavy", "is_reduction"],
    "outer":       ["is_matrix", "is_compute_heavy"],
    "linear":      ["is_matrix", "is_compute_heavy"],
    "conv1d":      ["is_conv", "is_compute_heavy"],
    "conv2d":      ["is_conv", "is_compute_heavy"],
    "conv3d":      ["is_conv", "is_compute_heavy"],
    "sum":         ["is_reduction"],
    "mean":        ["is_reduction"],
    "max":         ["is_reduction"],
    "min":         ["is_reduction"],
    "amax":        ["is_reduction"],
    "amin":        ["is_reduction"],
    "prod":        ["is_reduction"],
    "norm":        ["is_reduction"],
    "cumsum":      ["is_reduction"],
    "cumprod":     ["is_reduction"],
    "softmax":     ["is_reduction", "has_online_reduction"],
    "log_softmax": ["is_reduction", "has_online_reduction"],
}


def _ast_extract_op_hints(tree: ast.AST) -> Dict[str, bool]:
    """从 KernelBench 算子代码的 AST 中静态提取算子语义 hints。

    扫描 nn.Xxx 类实例化（__init__）及 F.xxx / torch.xxx 函数调用（forward），
    无需执行代码，不受动态控制流影响。
    - FX trace 失败时：作为唯一语义来源，避免全部 False 的信息丢失。
    - FX trace 成功时：作为补充信号，与图语义取 OR（两者均可信）。
    """
    hints: Dict[str, bool] = {
        "is_matrix": False,
        "is_conv": False,
        "is_reduction": False,
        "has_online_reduction": False,
        "has_epilogue": False,
        "is_compute_heavy": False,
    }

    # 解析 import 别名（KernelBench 标准：import torch.nn as nn / import torch.nn.functional as F）
    nn_aliases: set = {"nn"}
    f_aliases: set = {"F"}
    torch_aliases: set = {"torch"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                asname = alias.asname or alias.name.split(".")[-1]
                if alias.name == "torch":
                    torch_aliases.add(asname)
                elif alias.name == "torch.nn":
                    nn_aliases.add(asname)
                elif alias.name in ("torch.nn.functional", "torch.functional"):
                    f_aliases.add(asname)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                asname = alias.asname or alias.name
                if mod == "torch" and alias.name == "nn":
                    nn_aliases.add(asname)
                elif mod in ("torch.nn", "torch") and alias.name == "functional":
                    f_aliases.add(asname)

    def _apply(tag_list):
        for tag in tag_list:
            hints[tag] = True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        attr = func.attr
        val = func.value

        if isinstance(val, ast.Name):
            cid = val.id
            if cid in nn_aliases:           # nn.Xxx(...)
                _apply(_AST_NN_TAGS.get(attr, []))
            elif cid in f_aliases:          # F.xxx(...)
                _apply(_AST_F_TAGS.get(attr, []))
            elif cid in torch_aliases:      # torch.xxx(...)
                _apply(_AST_TORCH_FUNC_TAGS.get(attr, []))

        # torch.nn.Xxx(...)  无别名完整路径
        elif (
            isinstance(val, ast.Attribute)
            and val.attr == "nn"
            and isinstance(val.value, ast.Name)
            and val.value.id in torch_aliases
        ):
            _apply(_AST_NN_TAGS.get(attr, []))

        # torch.nn.functional.xxx(...)  无别名完整路径
        elif (
            isinstance(val, ast.Attribute)
            and val.attr == "functional"
            and isinstance(val.value, ast.Attribute)
            and val.value.attr == "nn"
            and isinstance(val.value.value, ast.Name)
            and val.value.value.id in torch_aliases
        ):
            _apply(_AST_F_TAGS.get(attr, []))

    return hints


def _find_callable(ns, names):
    for n in names:
        obj = ns.get(n, None)
        if callable(obj):
            return obj
    return None


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _normalize_module_type_name(module_type_name: str) -> str:
    """Map nn.Module class names to stable semantic op names."""
    n = _camel_to_snake(module_type_name)

    # Convolution family
    if re.fullmatch(r"conv_transpose[123]d", n):
        return n
    if re.fullmatch(r"conv[123]d", n):
        return n

    # Normalization family (strip spatial suffix for semantic stability)
    if re.fullmatch(r"batch_norm[123]d", n) or n == "batch_norm":
        return "batch_norm"
    if re.fullmatch(r"instance_norm[123]d", n) or n == "instance_norm":
        return "instance_norm"
    if n in {"layer_norm", "group_norm", "rms_norm"}:
        return n

    # Common loss family
    if n.endswith("_loss"):
        return n

    return n


def _get_node_semantic_info(node, gm):
    """Return (raw_target, canonical_target, module_type) for one FX node."""
    raw_target = str(node.target)
    canonical_target = raw_target
    module_type = None

    if node.op == "call_module":
        try:
            submod = gm.get_submodule(raw_target)
            module_type = type(submod).__name__
            canonical_target = _normalize_module_type_name(module_type)
        except Exception:
            pass

    return raw_target, canonical_target, module_type


def _shape_numel(shape):
    if not isinstance(shape, (tuple, list)):
        return 0
    numel = 1
    for d in shape:
        if not isinstance(d, int):
            return 0
        numel *= d
    return numel


def _semantic_tags(canonical_target: str, module_type: Optional[str]):
    """Build coarse semantic tags from static canonical/module type names."""
    tags = set()
    text = f"{canonical_target or ''} {_camel_to_snake(module_type) if module_type else ''}".lower()

    if any(k in text for k in ["matmul", "gemm", "linear", "bmm", "addmm", "einsum", " mm"]):
        tags.update(["matrix", "compute_heavy", "reduction"])

    if any(k in text for k in ["conv1d", "conv2d", "conv3d", "conv_transpose", "deconv", "depthwise", "pointwise"]):
        tags.update(["conv", "compute_heavy", "reduction"])

    if any(k in text for k in ["softmax", "log_softmax", "layer_norm", "group_norm", "batch_norm", "instance_norm", "rms_norm", "normalize"]):
        tags.update(["reduction", "online_reduction"])

    if any(k in text for k in ["pool", "sum", "mean", "max", "min", "argmax", "argmin", "cum", "reduce"]):
        tags.add("reduction")

    if "loss" in text:
        tags.add("reduction")

    if any(k in text for k in ["relu", "gelu", "silu", "tanh", "sigmoid", "clamp", "hardswish", "leaky_relu"]):
        tags.add("epilogue")

    return sorted(tags)


def _build_parameter_stats(params_meta):
    total_numel = 0
    max_tensor_numel = 0
    max_weight_rank = 0
    for meta in params_meta.values():
        shape = meta.get("shape")
        numel = _shape_numel(shape)
        total_numel += numel
        if numel > max_tensor_numel:
            max_tensor_numel = numel
        if isinstance(shape, (tuple, list)):
            max_weight_rank = max(max_weight_rank, len(shape))
    return {
        "total_numel": total_numel,
        "max_tensor_numel": max_tensor_numel,
        "max_weight_rank": max_weight_rank,
    }


def _build_static_feature_hints(graph_tensors, parameter_stats, ast_hints=None):
    tag_set = set()
    for node in graph_tensors:
        for tag in node.get("semantic_tags", []):
            tag_set.add(tag)

    total_numel = int(parameter_stats.get("total_numel", 0) or 0)
    max_tensor_numel = int(parameter_stats.get("max_tensor_numel", 0) or 0)

    # 参数规模提示：当参数规模较大时可作为 compute-heavy 辅助信号。
    param_scale_heavy = total_numel >= 1_000_000 or max_tensor_numel >= 1_000_000

    # FX trace 语义标签（当图不为空时有效）
    hints = {
        "is_matrix": "matrix" in tag_set,
        "is_conv": "conv" in tag_set,
        "is_reduction": "reduction" in tag_set,
        "has_online_reduction": "online_reduction" in tag_set,
        "has_epilogue": "epilogue" in tag_set,
        # compute_heavy 仅由算子语义决定（如 matmul/conv），参数规模单独暴露。
        "is_compute_heavy": ("compute_heavy" in tag_set),
        "param_scale_heavy": param_scale_heavy,
    }

    # AST 静态分析作为补充/兜底：与 FX 结果取 OR，不覆盖已有 True 值。
    # FX trace 失败（graph_tensors 为空）时 AST 是唯一语义来源。
    if ast_hints:
        for key in ("is_matrix", "is_conv", "is_reduction", "has_online_reduction",
                    "has_epilogue", "is_compute_heavy"):
            hints[key] = hints[key] or bool(ast_hints.get(key, False))

    return hints


def _to_tuple_inputs(x):
    # KernelBench 常见：Tensor / (Tensor, ...) / [Tensor,...]
    if isinstance(x, torch.Tensor):
        return (x,)
    if isinstance(x, (list, tuple)):
        return tuple(x)
    raise TypeError(f"Unsupported inputs type: {type(x)}")


def _make_fake_arg(arg, device="cuda"):
    # Tensor -> FakeTensor（保留 shape/stride/dtype）
    if isinstance(arg, torch.Tensor):
        # 这里不真正分配数据，empty_strided + FakeTensorMode 会走 meta
        return torch.empty_strided(
            size=tuple(arg.shape),
            stride=tuple(arg.stride()),
            device=device if arg.device.type != "meta" else "meta",
            dtype=arg.dtype,
        )
    # 其他类型（int/float/bool/None/tuple/list/dict）原样保留
    return arg


def _infer_output_shape_via_forward(model, real_inputs):
    """Infer output shape via meta forward when FX tracing fails."""
    try:
        meta_inputs = tuple(_make_fake_arg(arg, device="meta") for arg in real_inputs)
        with torch.no_grad():
            output = model(*meta_inputs)
        if isinstance(output, torch.Tensor):
            return tuple(output.shape)
        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return tuple(item.shape)
    except Exception as exc:  # best-effort fallback path
        logger.warning("Meta forward fallback failed for output shape inference: %s", exc)
    return None


def extract_kernelbench_shapes_dtypes(code_str: str, device="cuda"):
    # 1) exec 字符串前先做静态安全检查；同时获取 AST 树供后续语义分析复用
    tree = _validate_code_safety(code_str)
    ast_hints = _ast_extract_op_hints(tree)

    # 限制 __builtins__，只保留 KernelBench 代码所需的安全内建函数，
    # 防止通过 __builtins__.__import__("os") 等方式绕过 AST 检查。
    _SAFE_BUILTINS = {
        k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
        for k in (
            "__build_class__",  # Python 内部用于 class 定义
            "range", "len", "int", "float", "bool", "str", "list", "tuple", "dict", "set",
            "frozenset", "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
            "delattr", "property", "staticmethod", "classmethod", "super",
            "print", "enumerate", "zip", "map", "filter", "sorted", "reversed", "min", "max",
            "sum", "abs", "round", "pow", "divmod", "hash", "id", "callable", "repr",
            "True", "False", "None",
            "ValueError", "TypeError", "RuntimeError", "KeyError", "IndexError",
            "AttributeError", "NotImplementedError", "StopIteration", "Exception",
            "ImportError", "object", "slice",
        )
        if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))
    }
    # KernelBench 代码需要 import torch / math 等安全模块，通过受控 __import__ 放行。
    _ALLOWED_IMPORT_ROOTS = {"torch", "math", "numpy", "collections", "dataclasses", "typing", "functools", "itertools"}

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in _ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"Importing '{name}' is not allowed in sandboxed execution.")
        return __builtins__["__import__"](name, globals, locals, fromlist, level) if isinstance(__builtins__, dict) \
            else __builtins__.__import__(name, globals, locals, fromlist, level)

    _SAFE_BUILTINS["__import__"] = _restricted_import

    ns = {"__builtins__": _SAFE_BUILTINS, "__name__": "__kernelbench_sandbox__"}
    exec(code_str, ns, ns)

    # 2) 找 Model
    Model = ns.get("Model", None)
    if Model is None:
        raise RuntimeError("No `Model` class found in code_str.")

    model = None
     # 2.1) 找初始化函数（KernelBench 常见命名）
    get_init_inputs = ns.get("get_init_inputs", None)
    if callable(get_init_inputs):
        init_args = _to_tuple_inputs(get_init_inputs())
        model = Model(*init_args)
    else:
        model = Model()
    model.eval()

    model.to("meta")
    
    # 3) 找输入函数（KernelBench 常见命名）
    get_inputs = _find_callable(ns, ["get_inputs", "get_example_inputs", "get_input"])
    if get_inputs is None:
        get_inputs = getattr(model, "get_inputs", None)
    if not callable(get_inputs):
        raise RuntimeError("No input generator found (get_inputs/get_example_inputs/model.get_inputs).")

    # 4) 先拿一次真实输入（只为拿 shape/dtype/参数结构；不跑 forward）
    real_inputs = _to_tuple_inputs(get_inputs())

    # 5) Fake forward + FX：收集 tensor_meta
    # with FakeTensorMode(allow_non_fake_inputs=True):
    #     fake_inputs = tuple(_make_fake_arg(a, device=device) for a in real_inputs)

    #     gm = symbolic_trace(model)
    #     _ = gm(*fake_inputs)
    
    gm = None
    try:
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_inputs = tuple(_make_fake_arg(a, device="meta") for a in real_inputs)

            gm = symbolic_trace(model)
            ShapeProp(gm).propagate(*fake_inputs)
    except (RuntimeError, TypeError, AttributeError) as exc:
        logger.error("Symbolic trace failed, skipping graph tensors: %s", exc, exc_info=True)

    # 6) 汇总：输入 / 参数 / buffer
    inputs_meta = []
    for i, a in enumerate(real_inputs):
        if isinstance(a, torch.Tensor):
            inputs_meta.append({
                "idx": i,
                "shape": tuple(a.shape),
                "dtype": str(a.dtype),
                "stride": tuple(a.stride()),
                "device": str(a.device),
            })
        else:
            inputs_meta.append({
                "idx": i,
                "non_tensor_type": type(a).__name__,
                "value_repr": repr(a)[:200],
            })

    params_meta = {
        name: {"shape": tuple(p.shape), "dtype": str(p.dtype), "device": str(p.device)}
        for name, p in model.named_parameters(recurse=True)
    }
    buffers_meta = {
        name: {"shape": tuple(b.shape), "dtype": str(b.dtype), "device": str(b.device)}
        for name, b in model.named_buffers(recurse=True)
    }

    # 7) 图中每个节点的输出 tensor_meta（中间张量/最终输出都在这里）
    graph_tensors = []
    if gm is not None:
        for node in gm.graph.nodes:
            tm = node.meta.get("tensor_meta", None)
            if tm is not None:
                # Fix: TensorMetadata is a NamedTuple (subclass of tuple). 
                # We must distinguish between "a tuple of TensorMetadata" and "a single TensorMetadata (which is a tuple)".
                if hasattr(tm, 'shape') and hasattr(tm, 'dtype'):
                     tm_list = [tm]
                elif isinstance(tm, (list, tuple)):
                     tm_list = tm
                else:
                     tm_list = [tm]

                for i, item in enumerate(tm_list):
                    if hasattr(item, 'shape') and hasattr(item, 'dtype'):
                        suffix = f"_{i}" if len(tm_list) > 1 else ""
                        raw_target, canonical_target, module_type = _get_node_semantic_info(node, gm)
                        graph_tensors.append({
                            "name": f"{node.name}{suffix}",
                            "op": node.op,
                            "target": raw_target,
                            "raw_target": raw_target,
                            "canonical_target": canonical_target,
                            "module_type": module_type,
                            "semantic_tags": _semantic_tags(canonical_target, module_type),
                            "shape": tuple(item.shape),
                            "dtype": str(item.dtype),
                            "stride": tuple(item.stride) if hasattr(item, 'stride') else None,
                        })

    fallback_output_shape = None
    if not graph_tensors:
        fallback_output_shape = _infer_output_shape_via_forward(model, real_inputs)

    parameter_stats = _build_parameter_stats(params_meta)
    static_feature_hints = _build_static_feature_hints(graph_tensors, parameter_stats, ast_hints)

    return {
        "inputs": inputs_meta,
        "parameters": params_meta,
        "parameter_stats": parameter_stats,
        "buffers": buffers_meta,
        "graph_tensors": graph_tensors,
        "static_feature_hints": static_feature_hints,
        "fallback_output_shape": fallback_output_shape,
    }
