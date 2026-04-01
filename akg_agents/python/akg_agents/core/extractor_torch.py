import torch
from torch.fx import symbolic_trace
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.shape_prop import ShapeProp

def _find_callable(ns, names):
    for n in names:
        obj = ns.get(n, None)
        if callable(obj):
            return obj
    return None


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


def extract_kernelbench_shapes_dtypes(code_str: str, device="cuda"):
    # 1) exec 字符串
    ns = {}
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
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Symbolic trace failed, skipping graph tensors: {e}")

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
                        graph_tensors.append({
                            "name": f"{node.name}{suffix}",
                            "op": node.op,
                            "target": str(node.target),
                            "shape": tuple(item.shape),
                            "dtype": str(item.dtype),
                            "stride": tuple(item.stride) if hasattr(item, 'stride') else None,
                        })

    return {
        "inputs": inputs_meta,
        "parameters": params_meta,
        "buffers": buffers_meta,
        "graph_tensors": graph_tensors,
    }
