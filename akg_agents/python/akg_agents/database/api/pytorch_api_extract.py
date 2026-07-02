"""PyTorch API extraction and indexing for the Triton API database."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


_SIG_LINE_RE = re.compile(r"^[A-Za-z_]\w*\s*\(.*\)\s*(?:->\s*.*)?$")


@dataclass(frozen=True)
class ApiHit:
    canonical: str
    raw: str
    lineno: int


@dataclass
class TorchApi:
    qualname: str
    name: str
    module: str
    kind: str
    redsig: str
    doc: str
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AtenApiHit:
    canonical: str
    raw: str
    redsig: str
    doc: str
    name: str
    overload: str
    source_kind: str = "aten_dispatch"


def _first_nonempty_docline(obj: Any) -> Optional[str]:
    doc = getattr(obj, "__doc__", None)
    if not isinstance(doc, str):
        return None
    return next((line.strip() for line in doc.splitlines() if line.strip()), None)


def _red_signature_line(canonical: str, obj: Any) -> str:
    docline = _first_nonempty_docline(obj)
    if isinstance(docline, str) and _SIG_LINE_RE.match(docline):
        return re.sub(r"^[A-Za-z_]\w*", canonical, docline)
    try:
        return f"{canonical}{inspect.signature(obj)}"
    except Exception:
        return f"{canonical}(<signature unavailable>)"


def _kind_of(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isbuiltin(obj):
        return "builtin_function"
    if inspect.isfunction(obj):
        return "function"
    if inspect.ismethod(obj):
        return "method"
    return "callable"


def _torch_tags_for(qualname: str, name: str) -> List[str]:
    tags = set()
    ql = qualname.lower()
    if "functional" in ql:
        tags.add("functional")
    if ".nn." in ql:
        tags.add("nn")
    if ".tensor." in ql:
        tags.add("tensor")
    for word, tag in (
        ("conv", "convolution"),
        ("norm", "normalization"),
        ("softmax", "softmax"),
        ("matmul", "matmul"),
        ("mm", "matmul"),
        ("sum", "reduction"),
        ("mean", "reduction"),
        ("max", "reduction"),
        ("min", "reduction"),
        ("exp", "elementwise"),
        ("log", "elementwise"),
        ("relu", "activation"),
        ("gelu", "activation"),
        ("mish", "activation"),
    ):
        if word in name.lower():
            tags.add(tag)
    return sorted(tags)


class TorchApiExtractor:
    """Extract PyTorch calls from a KernelBench-style ``Model`` class."""

    def __init__(self, *, class_name: str = "Model") -> None:
        self.class_name = class_name
        self.alias: Dict[str, str] = {}
        self.self_map: Dict[str, str] = {}
        self.forward_hits: List[ApiHit] = []

    def extract(self, code: str, *, only_in_forward: bool = True, dedup: bool = True) -> List[ApiHit]:
        tree = ast.parse(code)
        self._collect_imports(tree)
        self._collect_self_assignments(tree)
        self._collect_calls(tree, only_in_forward=only_in_forward)
        hits = self.forward_hits
        if dedup:
            seen: Set[str] = set()
            uniq: List[ApiHit] = []
            for hit in hits:
                if hit.canonical not in seen:
                    uniq.append(hit)
                    seen.add(hit.canonical)
            hits = uniq
        return hits

    def _collect_imports(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    asname = alias.asname or alias.name.split(".")[-1]
                    if alias.name in {"torch", "torch.nn", "torch.nn.functional"}:
                        self.alias[asname] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    asname = alias.asname or alias.name
                    if node.module == "torch" and alias.name == "nn":
                        self.alias[asname] = "torch.nn"
                    elif node.module == "torch.nn" and alias.name == "functional":
                        self.alias[asname] = "torch.nn.functional"
                    elif node.module.startswith("torch"):
                        self.alias[asname] = f"{node.module}.{alias.name}"

    def _collect_self_assignments(self, tree: ast.AST) -> None:
        model = self._find_class(tree, self.class_name)
        init_fn = self._find_method(model, "__init__") if model else None
        if init_fn is None:
            return
        for stmt in init_fn.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            name = self._self_attr_name(stmt.targets[0])
            if not name or not isinstance(stmt.value, ast.Call):
                continue
            callee = self._resolve_callee(stmt.value.func)
            if callee and callee.startswith("torch."):
                self.self_map[name] = callee

    def _collect_calls(self, tree: ast.AST, *, only_in_forward: bool) -> None:
        scope: ast.AST = tree
        if only_in_forward:
            model = self._find_class(tree, self.class_name)
            forward = self._find_method(model, "forward") if model else None
            if forward is None:
                return
            scope = forward
        for node in ast.walk(scope):
            if isinstance(node, ast.Call):
                canonical = self._resolve_callee(node.func)
                if canonical and canonical.startswith("torch."):
                    raw = canonical
                    self.forward_hits.append(ApiHit(canonical=canonical, raw=raw, lineno=getattr(node, "lineno", 0)))

    @staticmethod
    def _find_class(tree: ast.AST, name: str) -> Optional[ast.ClassDef]:
        return next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name), None)

    @staticmethod
    def _find_method(cls: Optional[ast.ClassDef], name: str) -> Optional[ast.FunctionDef]:
        if cls is None:
            return None
        return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None)

    @staticmethod
    def _self_attr_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return node.attr
        return None

    def _resolve_callee(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return self.alias.get(node.id)
        if not isinstance(node, ast.Attribute):
            return None
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return self.self_map.get(node.attr)
        base = self._resolve_callee(node.value)
        if base:
            return f"{base}.{node.attr}"
        if isinstance(node.value, ast.Name):
            name = node.value.id
            if name in self.alias:
                return f"{self.alias[name]}.{node.attr}"
            if name in {"x", "input", "inputs", "tensor", "a", "b"}:
                return f"torch.Tensor.{node.attr}"
        if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
            if node.value.value.id == "self" and node.value.attr in self.self_map:
                return self.self_map[node.value.attr]
        return None


def pytorch_code_to_doc_list(code: str, *, only_in_forward: bool = True, class_name: str = "Model") -> List[Dict[str, Any]]:
    extractor = TorchApiExtractor(class_name=class_name)
    hits = extractor.extract(code, only_in_forward=only_in_forward)
    docs: List[Dict[str, Any]] = []
    for hit in hits:
        obj = _resolve_object(hit.canonical)
        docs.append(
            {
                "canonical": hit.canonical,
                "raw": hit.raw,
                "lineno": hit.lineno,
                "redsig": _red_signature_line(hit.canonical, obj) if obj is not None else hit.canonical,
                "doc": (inspect.getdoc(obj) or "") if obj is not None else "",
            }
        )
    return docs


def aten_dispatch_code_to_doc_list(code: str, *, class_name: str = "Model") -> List[Dict[str, Any]]:
    """Extract ATen operators by running a KernelBench forward under fake/meta dispatch."""
    hits = AtenDispatchExtractor(class_name=class_name).extract(code)
    return [
        {
            "canonical": hit.canonical,
            "raw": hit.raw,
            "lineno": 0,
            "redsig": hit.redsig,
            "doc": hit.doc,
            "source_kind": hit.source_kind,
            "aten_name": hit.name,
            "aten_overload": hit.overload,
        }
        for hit in hits
    ]


class _AtenDispatchRecorder:
    def __init__(self) -> None:
        self.ops: List[Any] = []

    def __enter__(self) -> "_AtenDispatchRecorder":
        from torch.utils._python_dispatch import TorchDispatchMode

        recorder = self

        class _RecorderMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
                recorder.ops.append(func)
                return func(*args, **(kwargs or {}))

        self._mode = _RecorderMode()
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return self._mode.__exit__(exc_type, exc, tb)


class AtenDispatchExtractor:
    """Capture local ATen operators triggered by a KernelBench-style forward."""

    def __init__(self, *, class_name: str = "Model") -> None:
        self.class_name = class_name

    def extract(self, code: str, *, dedup: bool = True) -> List[AtenApiHit]:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode

        ns = self._exec_kernelbench_code(code)
        Model = ns.get(self.class_name)
        if Model is None:
            raise RuntimeError(f"No `{self.class_name}` class found in code.")

        init_fn = ns.get("get_init_inputs")
        init_args = self._to_tuple_inputs(init_fn()) if callable(init_fn) else ()
        model = Model(*init_args)
        model.eval()
        model.to("meta")

        get_inputs = self._find_callable(ns, ["get_inputs", "get_example_inputs", "get_input"])
        if get_inputs is None:
            get_inputs = getattr(model, "get_inputs", None)
        if not callable(get_inputs):
            raise RuntimeError("No input generator found (get_inputs/get_example_inputs/model.get_inputs).")

        real_inputs = self._to_tuple_inputs(get_inputs())
        with torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True):
            fake_inputs = tuple(self._make_fake_arg(arg) for arg in real_inputs)
            with _AtenDispatchRecorder() as recorder:
                model(*fake_inputs)

        hits = [self._op_to_hit(op) for op in recorder.ops if self._is_aten_op(op)]
        if not dedup:
            return hits
        seen: Set[str] = set()
        out: List[AtenApiHit] = []
        for hit in hits:
            if hit.canonical in seen:
                continue
            seen.add(hit.canonical)
            out.append(hit)
        return out

    @staticmethod
    def _is_aten_op(op: Any) -> bool:
        return str(op).startswith("aten.")

    @staticmethod
    def _op_to_hit(op: Any) -> AtenApiHit:
        raw = str(op)
        canonical = f"torch.ops.{raw}"
        schema = str(getattr(op, "_schema", "") or raw)
        name, overload = AtenDispatchExtractor._split_aten_name(raw)
        doc = (
            "ATen dispatcher operator captured from KernelBench fake/meta forward.\n"
            f"Operator: {raw}\n"
            f"Schema: {schema}\n"
            f"Name: {name}\n"
            f"Overload: {overload}"
        )
        return AtenApiHit(
            canonical=canonical,
            raw=raw,
            redsig=schema,
            doc=doc,
            name=name,
            overload=overload,
        )

    @staticmethod
    def _split_aten_name(raw: str) -> tuple[str, str]:
        tail = raw[len("aten.") :] if raw.startswith("aten.") else raw
        if "." not in tail:
            return tail, ""
        name, overload = tail.split(".", 1)
        return name, overload

    @staticmethod
    def _to_tuple_inputs(value: Any) -> tuple:
        import torch

        if isinstance(value, torch.Tensor):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise TypeError(f"Unsupported inputs type: {type(value)}")

    @staticmethod
    def _make_fake_arg(arg: Any) -> Any:
        import torch

        if isinstance(arg, torch.Tensor):
            return torch.empty_strided(
                size=tuple(arg.shape),
                stride=tuple(arg.stride()),
                device="meta",
                dtype=arg.dtype,
            )
        return arg

    @staticmethod
    def _find_callable(ns: Dict[str, Any], names: List[str]) -> Optional[Any]:
        for name in names:
            candidate = ns.get(name)
            if callable(candidate):
                return candidate
        return None

    def _exec_kernelbench_code(self, code: str) -> Dict[str, Any]:
        tree = ast.parse(code)
        self._validate_imports(tree)
        safe_builtins = self._safe_builtins()
        ns: Dict[str, Any] = {"__builtins__": safe_builtins, "__name__": "__kernelbench_sandbox__"}
        exec(code, ns, ns)
        return ns

    @staticmethod
    def _validate_imports(tree: ast.AST) -> None:
        allowed_roots = {"torch", "math", "numpy", "collections", "dataclasses", "typing", "functools", "itertools"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in allowed_roots:
                        raise ImportError(f"Importing '{alias.name}' is not allowed in sandboxed execution.")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root not in allowed_roots:
                    raise ImportError(f"Importing '{node.module}' is not allowed in sandboxed execution.")

    @staticmethod
    def _safe_builtins() -> Dict[str, Any]:
        builtin_source = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        names = (
            "__build_class__",
            "range",
            "len",
            "int",
            "float",
            "bool",
            "str",
            "list",
            "tuple",
            "dict",
            "set",
            "frozenset",
            "type",
            "isinstance",
            "issubclass",
            "hasattr",
            "getattr",
            "setattr",
            "delattr",
            "property",
            "staticmethod",
            "classmethod",
            "super",
            "print",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "min",
            "max",
            "sum",
            "abs",
            "round",
            "pow",
            "divmod",
            "hash",
            "id",
            "callable",
            "repr",
            "True",
            "False",
            "None",
            "ValueError",
            "TypeError",
            "RuntimeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "NotImplementedError",
            "StopIteration",
            "Exception",
            "ImportError",
            "object",
            "slice",
        )
        safe = {name: builtin_source[name] for name in names if name in builtin_source}
        allowed_roots = {"torch", "math", "numpy", "collections", "dataclasses", "typing", "functools", "itertools"}
        real_import = builtin_source["__import__"]

        def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            root = name.split(".")[0]
            if root not in allowed_roots:
                raise ImportError(f"Importing '{name}' is not allowed in sandboxed execution.")
            return real_import(name, globals, locals, fromlist, level)

        safe["__import__"] = _restricted_import
        return safe


def _resolve_object(canonical: str) -> Any:
    try:
        import torch

        if canonical.startswith("torch.Tensor."):
            return getattr(torch.Tensor, canonical.split(".", 2)[2], None)
        cur: Any = torch
        for part in canonical.split(".")[1:]:
            cur = getattr(cur, part, None)
            if cur is None:
                return None
        return cur
    except Exception:
        return None


def scan_torch_api(include_tensor_methods: bool = True) -> List[TorchApi]:
    import torch

    modules = [("torch", torch), ("torch.nn", torch.nn), ("torch.nn.functional", torch.nn.functional)]
    seen: Dict[str, TorchApi] = {}
    for mod_name, mod in modules:
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            try:
                obj = getattr(mod, attr)
            except Exception:
                continue
            if not callable(obj):
                continue
            qn = f"{mod_name}.{attr}"
            seen[qn] = TorchApi(qn, attr, mod_name, _kind_of(obj), _red_signature_line(qn, obj), inspect.getdoc(obj) or "", _torch_tags_for(qn, attr))
    if include_tensor_methods:
        for attr in dir(torch.Tensor):
            if attr.startswith("_"):
                continue
            obj = getattr(torch.Tensor, attr, None)
            if callable(obj):
                qn = f"torch.Tensor.{attr}"
                seen[qn] = TorchApi(qn, attr, "torch.Tensor", _kind_of(obj), _red_signature_line(qn, obj), inspect.getdoc(obj) or "", _torch_tags_for(qn, attr))
    return sorted(seen.values(), key=lambda item: item.qualname)


def push_torch_api_to_qdrant(
    entries: List[TorchApi],
    *,
    host: str = "localhost",
    port: int = 6333,
    collection: str = "torch_api",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder: Optional[str] = None,
    recreate: bool = True,
    batch: int = 256,
) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model, **({"cache_folder": cache_folder} if cache_folder else {}))
    dim = model.get_sentence_embedding_dimension()
    texts = [f"{entry.redsig}\n{entry.doc[:1500]}".strip() for entry in entries]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
    client = QdrantClient(host=host, port=port, trust_env=False)
    if recreate:
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass
        client.create_collection(collection_name=collection, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
    points = [qm.PointStruct(id=i, vector=v, payload=asdict(e)) for i, (e, v) in enumerate(zip(entries, vecs))]
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i : i + batch])
    print(f"[qdrant] upserted {len(points)} torch APIs -> '{collection}'")


def retrieve_torch_apis_by_canonical(
    canonicals: List[str],
    *,
    host: str = "localhost",
    port: int = 6333,
    collection: str = "torch_api",
) -> Dict[str, Dict[str, Any]]:
    if not canonicals:
        return {}
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    client = QdrantClient(host=host, port=port, trust_env=False)
    flt = qm.Filter(must=[qm.FieldCondition(key="qualname", match=qm.MatchAny(any=canonicals))])
    out: Dict[str, Dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = client.scroll(collection_name=collection, scroll_filter=flt, limit=256, offset=offset, with_payload=True)
        for point in points:
            payload = point.payload or {}
            qn = payload.get("qualname")
            if qn:
                out[qn] = payload
        if offset is None:
            break
    return out
