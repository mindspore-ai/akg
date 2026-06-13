"""Triton API scanning and retrieval backed by Qdrant."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import threading
import types
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


BACKEND_ALIASES = {
    "cuda": ("cuda", "nvidia"),
    "amd": ("amd", "hip"),
    "ascend": ("ascend", "npu"),
    "intel": ("intel",),
}
CANDIDATE_BACKENDS = list(BACKEND_ALIASES)
BACKEND_IMPORT_ROOTS = {
    "cuda": [
        "triton.language.extra.cuda.libdevice",
        "triton.language.extra.cuda",
        "triton.backends.nvidia",
    ],
    "amd": [
        "triton.language.extra.amd.libdevice",
        "triton.language.extra.amd",
        "triton.backends.amd",
    ],
    "ascend": [
        "triton.language.extra.ascend.libdevice",
        "triton.language.extra.ascend",
        "triton.backends.ascend",
    ],
    "intel": [
        "triton.language.extra.intel.libdevice",
        "triton.language.extra.intel",
        "triton.backends.intel",
    ],
}
BACKEND_DETECTION_ROOTS = {
    backend: tuple(roots) for backend, roots in BACKEND_IMPORT_ROOTS.items()
}
BASE_ROOTS = [
    "triton.language",
    "triton.language.math",
    "triton.language.standard",
    "triton.language.extra",
    "triton.runtime",
    "triton.compiler",
    "triton",
]
NPU_INTRINSICS = {"gather_out_to_ub", "index_put", "index_select_simd", "set_element", "embedding_gather"}
NOISY_RECALL_PATTERNS = (
    ".random.",
    ".compiler.",
    ".runtime.",
    ".backends.",
    ".is_builtin",
    ".inline_asm",
)


@dataclass
class TritonApi:
    qualname: str
    name: str
    module: str
    kind: str
    redsig: str
    doc: str
    tags: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    source_lineno: Optional[int] = None


def _has_import_spec(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _normalize_backend_name(name: str) -> str:
    name = (name or "").strip().lower()
    for backend, aliases in BACKEND_ALIASES.items():
        if name == backend or name in aliases:
            return backend
    return name


def normalize_target_backends(target_backend: Optional[str]) -> List[str]:
    if not target_backend or str(target_backend).strip().lower() == "auto":
        return detect_active_backends()
    backends: List[str] = []
    for raw in re.split(r"[,/\s]+", str(target_backend)):
        backend = _normalize_backend_name(raw)
        if backend and backend in BACKEND_ALIASES and backend not in backends:
            backends.append(backend)
    return backends or detect_active_backends()


def _detect_runtime_backend() -> str:
    try:
        from triton.runtime import driver

        target = driver.active.get_current_target()
        return _normalize_backend_name(getattr(target, "backend", ""))
    except Exception:
        return ""


def detect_active_backends() -> List[str]:
    runtime_backend = _detect_runtime_backend()
    if runtime_backend in BACKEND_ALIASES:
        return [runtime_backend]
    active: List[str] = []
    for backend, roots in BACKEND_DETECTION_ROOTS.items():
        if any(_has_import_spec(root) for root in roots):
            active.append(backend)
    return active


def _backend_roots(backends: List[str]) -> List[str]:
    roots: List[str] = []
    for backend in backends:
        roots.extend(BACKEND_IMPORT_ROOTS.get(_normalize_backend_name(backend), []))
    return roots


class TritonApiScanner:
    """Scan importable Triton namespaces and keep canonical callable entries."""

    def __init__(self, backends: List[str], skip_private: bool = True) -> None:
        self.backends = normalize_target_backends(",".join(backends)) if backends else detect_active_backends()
        self.skip_private = skip_private
        self._allowed_backend_aliases = {
            alias
            for backend in self.backends
            for alias in BACKEND_ALIASES.get(backend, (backend,))
        }
        self._seen_obj_ids: Set[int] = set()
        self._collected: Dict[str, TritonApi] = {}
        self._modules_walked: Set[str] = set()

    def scan(self) -> List[TritonApi]:
        for root in BASE_ROOTS + _backend_roots(self.backends):
            try:
                mod = importlib.import_module(root)
            except Exception:
                continue
            self._walk(mod, qualname_prefix=self._short_alias(root))
        return sorted(self._collected.values(), key=lambda entry: entry.qualname)

    @staticmethod
    def _short_alias(full_module_name: str) -> str:
        name = full_module_name
        for aliases in BACKEND_ALIASES.values():
            for backend in aliases:
                name = name.replace(f"triton.language.extra.{backend}.libdevice", f"tl.extra.{backend}.libdevice")
                name = name.replace(f"triton.language.extra.{backend}", f"tl.extra.{backend}")
        return (
            name.replace("triton.language.extra.libdevice", "tl.extra.libdevice")
            .replace("triton.language.extra", "tl.extra")
            .replace("triton.language.math", "tl.math")
            .replace("triton.language.standard", "tl.standard")
            .replace("triton.language", "tl")
        )

    def _walk(self, mod: types.ModuleType, qualname_prefix: str) -> None:
        if mod.__name__ in self._modules_walked:
            return
        if not self._is_backend_module_allowed(mod.__name__):
            return
        self._modules_walked.add(mod.__name__)
        for attr_name in dir(mod):
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            if self.skip_private and attr_name.startswith("_"):
                continue
            try:
                obj = getattr(mod, attr_name)
            except Exception:
                continue
            child_qualname = f"{qualname_prefix}.{attr_name}"
            if isinstance(obj, types.ModuleType):
                obj_name = getattr(obj, "__name__", "")
                if not self._is_backend_module_allowed(obj_name):
                    continue
                if obj_name.startswith("triton") and obj_name == f"{mod.__name__}.{attr_name}":
                    self._walk(obj, qualname_prefix=child_qualname)
                continue
            if callable(obj):
                self._record(child_qualname, attr_name, mod, obj)

    def _is_backend_module_allowed(self, module_name: str) -> bool:
        lower_name = (module_name or "").lower()
        if lower_name.startswith("triton.language.extra.libdevice"):
            return False
        for aliases in BACKEND_ALIASES.values():
            for alias in aliases:
                if f".{alias}" in lower_name or f"backends.{alias}" in lower_name:
                    return alias in self._allowed_backend_aliases
        return True

    def _record(self, qualname: str, name: str, defining_module: types.ModuleType, obj: Any) -> None:
        qualname = self._public_alias_for(qualname, name, obj)
        obj_id = id(obj)
        if obj_id in self._seen_obj_ids:
            return
        self._seen_obj_ids.add(obj_id)
        source_file, source_lineno = self._source_loc(obj)
        self._collected[qualname] = TritonApi(
            qualname=qualname,
            name=name,
            module=defining_module.__name__,
            kind=self._kind_of(obj),
            redsig=self._make_redsig(qualname, obj),
            doc=(inspect.getdoc(obj) or "").strip(),
            tags=self._tags_for(qualname, name, defining_module.__name__),
            source_file=source_file,
            source_lineno=source_lineno,
        )

    @staticmethod
    def _public_alias_for(qualname: str, name: str, obj: Any) -> str:
        if not qualname.startswith(("tl.core.", "tl.standard.")):
            return qualname
        try:
            import triton.language as tl

            public_obj = getattr(tl, name, None)
        except Exception:
            public_obj = None
        if public_obj is obj:
            return f"tl.{name}"
        return qualname

    @staticmethod
    def _kind_of(obj: Any) -> str:
        if inspect.isclass(obj):
            return "class"
        if obj.__class__.__name__ == "JITFunction":
            return "jit_function"
        if inspect.isbuiltin(obj):
            return "builtin_function"
        if inspect.isfunction(obj):
            return "function"
        if inspect.ismethod(obj):
            return "method"
        return "callable"

    @staticmethod
    def _make_redsig(qualname: str, obj: Any) -> str:
        doc = inspect.getdoc(obj) or ""
        first = next((line.strip() for line in doc.splitlines() if line.strip()), "")
        if re.match(r"^[A-Za-z_]\w*\s*\(.*\).*$", first):
            return re.sub(rf"^{re.escape(qualname.rsplit('.', 1)[-1])}", qualname, first)
        try:
            return f"{qualname}{inspect.signature(obj)}"
        except Exception:
            return f"{qualname}(<signature unavailable>)"

    @staticmethod
    def _source_loc(obj: Any) -> Tuple[Optional[str], Optional[int]]:
        try:
            return inspect.getsourcefile(obj), inspect.getsourcelines(obj)[1]
        except Exception:
            return None, None

    @staticmethod
    def _tags_for(qualname: str, name: str, defining_module: str) -> List[str]:
        tags = set()
        ql = qualname.lower()
        if ql.startswith("tl."):
            tags.add("tl")
        if "math" in ql:
            tags.add("math")
        if "standard" in ql:
            tags.add("standard")
        if "libdevice" in ql:
            tags.update({"libdevice", "low_level"})
        if "runtime" in defining_module:
            tags.add("runtime")
        if "compiler" in defining_module:
            tags.add("compiler")
        for backend in CANDIDATE_BACKENDS:
            aliases = BACKEND_ALIASES.get(backend, (backend,))
            if any(f".{alias}." in ql or f"backends.{alias}" in defining_module for alias in aliases):
                tags.add(backend)
                if backend == "ascend":
                    tags.add("npu")
        if name in NPU_INTRINSICS:
            tags.update({"npu_intrinsic", "npu"})
        for keyword, tag in (
            ("dot", "matmul"),
            ("load", "memory"),
            ("store", "memory"),
            ("atomic", "memory"),
            ("sum", "reduction"),
            ("max", "reduction"),
            ("min", "reduction"),
            ("exp", "elementwise"),
            ("log", "elementwise"),
            ("sin", "trig"),
            ("cos", "trig"),
            ("tan", "trig"),
            ("sigmoid", "activation"),
            ("relu", "activation"),
            ("arange", "index"),
        ):
            if keyword in name.lower():
                tags.add(tag)
        return sorted(tags)


def push_to_qdrant(
    entries: List[TritonApi],
    *,
    host: str,
    port: int,
    collection: str,
    embed_model: str,
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
    print(f"[qdrant] upserted {len(points)} points -> '{collection}'")


_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_CACHE: Dict[Tuple[str, str], Any] = {}


def _get_embedder(model_name: str, cache_folder: Optional[str] = None) -> Any:
    from sentence_transformers import SentenceTransformer

    key = (model_name, cache_folder or "")
    with _EMBEDDER_LOCK:
        if key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[key] = SentenceTransformer(model_name, **({"cache_folder": cache_folder} if cache_folder else {}))
        return _EMBEDDER_CACHE[key]


DEFAULT_KEYWORD_MAP: Dict[str, List[str]] = {
    "mish": ["tanh", "softplus", "exp", "log", "log1p"],
    "gelu": ["erf", "tanh", "exp"],
    "silu": ["sigmoid"],
    "swish": ["sigmoid"],
    "softplus": ["log", "exp", "log1p"],
    "relu6": ["min", "max", "minimum", "maximum"],
    "hardswish": ["min", "max", "minimum", "maximum"],
    "softmax": ["exp", "max", "sum"],
    "logsoftmax": ["exp", "max", "sum", "log"],
    "convolution": ["program_id", "arange", "load", "store", "dot", "where"],
    "max_pool2d_with_indices": ["program_id", "arange", "load", "store", "max", "maximum", "where"],
    "add": ["add"],
    "mul": ["mul"],
    "layernorm": ["rsqrt", "sqrt", "sum"],
    "batchnorm": ["rsqrt", "sqrt"],
    "groupnorm": ["rsqrt", "sqrt", "sum"],
    "instancenorm": ["rsqrt", "sqrt"],
    "rmsnorm": ["rsqrt", "sqrt"],
    "exp": ["exp", "exp2"],
    "log": ["log", "log1p", "log2"],
    "erf": ["erf", "erfc"],
    "rsqrt": ["rsqrt"],
    "sqrt": ["sqrt"],
    "abs": ["abs"],
    "sin": ["sin"],
    "cos": ["cos"],
    "tanh": ["exp", "tanh"],
    "tan": ["tan"],
    "clamp": ["clamp", "minimum", "maximum"],
}
_TORCH_PREFIX_FRAGS = (
    "torch.ops.aten.",
    "torch.nn.functional.",
    "torch.nn.",
    "torch.Tensor.",
    "torch.special.",
    "torch.linalg.",
    "torch.",
    "nn.functional.",
    "nn.",
    "F.",
    "Tensor.",
)


def normalize_torch_leaf(canonical: str) -> str:
    s = (canonical or "").strip()
    if s.startswith("torch.ops.aten."):
        return s[len("torch.ops.aten.") :].split(".", 1)[0].lower()
    if s.startswith("aten."):
        return s[len("aten.") :].split(".", 1)[0].lower()
    changed = True
    while changed:
        changed = False
        for frag in _TORCH_PREFIX_FRAGS:
            if s.startswith(frag):
                s = s[len(frag) :]
                changed = True
                break
    return s.rsplit(".", 1)[-1].lower()


def _keyword_leaves_for_source(src_canon: str, keyword_map: Dict[str, List[str]]) -> Set[str]:
    leaf = normalize_torch_leaf(src_canon)
    if leaf in keyword_map:
        return {item.lower() for item in keyword_map[leaf]}
    for key, values in keyword_map.items():
        if len(key) >= 5 and leaf.startswith(key):
            return {item.lower() for item in values}
    return set()


def _is_keyword_hit(entry: Dict[str, Any], keyword_leaves: Set[str]) -> Optional[str]:
    triton_name = (entry.get("triton_name") or "").lower()
    tail = (entry.get("triton_qualname") or "").lower().rsplit(".", 1)[-1]
    return next((kw for kw in keyword_leaves if kw == triton_name or kw == tail), None)


def _is_noisy_recall(qn: str, name: str) -> bool:
    qn_lc = (qn or "").lower()
    name_lc = (name or "").lower()
    if any(pattern in qn_lc for pattern in NOISY_RECALL_PATTERNS):
        return True
    if name_lc in {"jit"} and ".random." in qn_lc:
        return True
    return False


def _entry_public_rank(entry: Dict[str, Any]) -> Tuple[int, float]:
    qn = (entry.get("triton_qualname") or "").lower()
    score = float(entry.get("score", 0.0) or 0.0)
    if qn.startswith("tl.") and ".core." not in qn and ".libdevice." not in qn:
        public_rank = 0
    elif ".core." in qn:
        public_rank = 1
    elif ".libdevice." in qn:
        public_rank = 2
    else:
        public_rank = 3
    return public_rank, -score


def _dedupe_keyword_hits(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_keyword: Dict[str, Dict[str, Any]] = {}
    passthrough: List[Dict[str, Any]] = []
    for entry in entries:
        keyword = (entry.get("keyword_hit") or "").lower()
        if not keyword:
            passthrough.append(entry)
            continue
        current = best_by_keyword.get(keyword)
        if current is None or _entry_public_rank(entry) < _entry_public_rank(current):
            best_by_keyword[keyword] = entry
    out = list(best_by_keyword.values()) + passthrough
    out.sort(key=lambda r: (_entry_public_rank(r)[0], -float(r.get("score", 0.0) or 0.0)))
    return out


def _qdrant_filter(tags: Optional[List[str]] = None):
    if not tags:
        return None
    from qdrant_client.http import models as qm

    return qm.Filter(must=[qm.FieldCondition(key="tags", match=qm.MatchAny(any=tags))])


def _elbow_cut(scores: List[float], *, min_gap_ratio: float, min_keep: int, max_keep: int, relative_decay: float) -> int:
    if not scores:
        return 0
    if len(scores) <= min_keep:
        return len(scores)
    if scores[0] <= 0:
        return min(len(scores), max_keep)
    cuts: List[int] = []
    threshold = scores[0] * min_gap_ratio
    for idx in range(len(scores) - 1):
        if scores[idx] - scores[idx + 1] >= threshold:
            cuts.append(idx + 1)
            break
    if 0 < relative_decay < 1:
        floor = scores[0] * relative_decay
        for idx, score in enumerate(scores):
            if score < floor:
                cuts.append(idx)
                break
    cut = min(cuts) if cuts else min(len(scores), max_keep)
    return max(min_keep, min(cut, max_keep))


def _keyword_fallback_query(
    *,
    client: Any,
    triton_collection: str,
    missing_kws: List[str],
    base_filter: Any,
    exclude_backend_patterns: List[str],
    already_qns: Set[str],
    src_canon: str,
    limit_per_kw: int,
) -> List[Dict[str, Any]]:
    from qdrant_client.http import models as qm

    name_cond = qm.FieldCondition(key="name", match=qm.MatchAny(any=missing_kws))
    flt = qm.Filter(must=(list(getattr(base_filter, "must", None) or []) + [name_cond]) if base_filter else [name_cond])
    points, _ = client.scroll(
        collection_name=triton_collection,
        scroll_filter=flt,
        limit=max(8, limit_per_kw * max(1, len(missing_kws)) * 4),
        with_payload=True,
    )
    out: List[Dict[str, Any]] = []
    per_kw_count = {kw: 0 for kw in missing_kws}
    for point in points:
        payload = point.payload or {}
        qn = payload.get("qualname") or payload.get("name") or ""
        name = payload.get("name") or ""
        if (
            not qn
            or qn in already_qns
            or any(pattern in qn for pattern in exclude_backend_patterns)
            or _is_noisy_recall(qn, name)
        ):
            continue
        name_lc = name.lower()
        tail_lc = qn.lower().rsplit(".", 1)[-1]
        match = next((kw for kw in missing_kws if kw == name_lc or kw == tail_lc), None)
        if match is None or per_kw_count[match] >= limit_per_kw:
            continue
        per_kw_count[match] += 1
        already_qns.add(qn)
        out.append(
            {
                "score": 0.0,
                "triton_qualname": qn,
                "triton_name": payload.get("name"),
                "tags": payload.get("tags"),
                "doc": payload.get("doc", ""),
                "redsig": payload.get("redsig", ""),
                "matched_from": [src_canon],
                "payload": payload,
                "keyword_hit": match,
                "fallback": True,
            }
        )
    return out


def _build_query_text(item: Dict[str, Any], *, doc_head_chars: int = 800) -> str:
    parts = []
    if item.get("redsig"):
        parts.append(item["redsig"])
    if item.get("doc"):
        parts.append(item["doc"][:doc_head_chars])
    text = "\n".join(parts)
    text = re.sub(r"\b(torch|tensor)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def retrieve_triton_apis(
    pytorch_docs: List[Dict[str, Any]],
    *,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    triton_collection: str = "triton_api",
    torch_collection: str = "torch_api",
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_cache_folder: Optional[str] = None,
    topk_per_query: int = 32,
    filter_tags: Optional[List[str]] = None,
    target_backend: str = "cuda",
    group_by_source: bool = True,
    enable_keyword_recall: bool = True,
    keyword_map: Optional[Dict[str, List[str]]] = None,
    min_keep: int = 2,
    max_keep: int = 10,
    elbow_min_gap_ratio: float = 0.15,
    relative_decay: float = 0.90,
    keyword_fallback_qdrant: bool = True,
    keyword_fallback_limit_per_kw: int = 4,
) -> Any:
    from qdrant_client import QdrantClient
    from akg_agents.database.api.pytorch_api_extract import retrieve_torch_apis_by_canonical

    if not pytorch_docs:
        return {} if group_by_source else []
    client = QdrantClient(host=qdrant_host, port=qdrant_port, trust_env=False)
    embedder = _get_embedder(embed_model_name, embed_cache_folder)
    flt = _qdrant_filter(filter_tags)
    target_backends = normalize_target_backends(target_backend)
    allowed_backend_aliases = {
        alias
        for backend in target_backends
        for alias in BACKEND_ALIASES.get(backend, (backend,))
    }
    all_backend_aliases = {alias for aliases in BACKEND_ALIASES.values() for alias in aliases}
    exclude_backend_patterns = [f".{backend}." for backend in all_backend_aliases - allowed_backend_aliases]
    exclude_backend_patterns.append("tl.extra.libdevice.")

    canonicals = [doc.get("canonical") for doc in pytorch_docs if doc.get("canonical")]
    pivot_payloads = retrieve_torch_apis_by_canonical(canonicals, host=qdrant_host, port=qdrant_port, collection=torch_collection)
    queries: List[Tuple[str, str, str]] = []
    for item in pytorch_docs:
        canon = item.get("canonical", "")
        if not canon:
            continue
        pivot = pivot_payloads.get(canon)
        query_text = _build_query_text(pivot or item)
        if query_text:
            queries.append((canon, normalize_torch_leaf(canon), query_text))
    if not queries:
        return {} if group_by_source else []

    q_vecs = embedder.encode([query for _, _, query in queries], normalize_embeddings=True).tolist()
    raw_limit = max(topk_per_query * 4, 32)
    active_keyword_map = keyword_map if keyword_map is not None else DEFAULT_KEYWORD_MAP
    per_source: Dict[str, List[Dict[str, Any]]] = {}
    for (src_canon, src_leaf, _query_text), qvec in zip(queries, q_vecs):
        hits = client.query_points(
            collection_name=triton_collection,
            query=qvec,
            limit=raw_limit,
            query_filter=flt,
            with_payload=True,
        )
        kept: List[Dict[str, Any]] = []
        seen_qn: Set[str] = set()
        for hit in hits.points:
            payload = hit.payload or {}
            qn = payload.get("qualname") or payload.get("name") or ""
            name = payload.get("name") or ""
            score = float(getattr(hit, "score", 0.0) or 0.0)
            if (
                not qn
                or qn in seen_qn
                or any(pattern in qn for pattern in exclude_backend_patterns)
                or _is_noisy_recall(qn, name)
            ):
                continue
            seen_qn.add(qn)
            kept.append(
                {
                    "score": score,
                    "triton_qualname": qn,
                    "triton_name": name,
                    "tags": payload.get("tags"),
                    "doc": payload.get("doc", ""),
                    "redsig": payload.get("redsig", ""),
                    "matched_from": [src_canon],
                    "payload": payload,
                }
            )
        kept.sort(key=lambda r: (0 if src_leaf and src_leaf in (r.get("triton_name") or "").lower() else 1, -r["score"]))
        kw_leaves = _keyword_leaves_for_source(src_canon, active_keyword_map) if enable_keyword_recall else set()
        must_keep: List[Dict[str, Any]] = []
        rest: List[Dict[str, Any]] = []
        for entry in kept:
            kw = _is_keyword_hit(entry, kw_leaves)
            if kw:
                entry["keyword_hit"] = kw
                must_keep.append(entry)
            else:
                rest.append(entry)
        must_keep.sort(key=lambda r: -r["score"])
        if kw_leaves and keyword_fallback_qdrant:
            hit_kws = {entry.get("keyword_hit", "").lower() for entry in must_keep if entry.get("keyword_hit")}
            missing_kws = sorted(kw_leaves - hit_kws)
            if missing_kws:
                already_qns = {entry.get("triton_qualname", "") for entry in must_keep + rest}
                must_keep.extend(
                    _keyword_fallback_query(
                        client=client,
                        triton_collection=triton_collection,
                        missing_kws=missing_kws,
                        base_filter=flt,
                        exclude_backend_patterns=exclude_backend_patterns,
                        already_qns=already_qns,
                        src_canon=src_canon,
                        limit_per_kw=keyword_fallback_limit_per_kw,
                    )
                )
                must_keep = _dedupe_keyword_hits(must_keep)
        if kw_leaves and must_keep:
            must_keep = _dedupe_keyword_hits(must_keep)
            per_source[src_canon] = must_keep[:max_keep]
            continue
        rest_budget = max(0, max_keep - len(must_keep))
        rest_min_keep = max(0, min_keep - len(must_keep))
        cut = _elbow_cut(
            [entry["score"] for entry in rest],
            min_gap_ratio=elbow_min_gap_ratio,
            min_keep=rest_min_keep,
            max_keep=rest_budget,
            relative_decay=relative_decay,
        )
        per_source[src_canon] = (must_keep + rest[:cut])[:max_keep]

    if group_by_source:
        return per_source
    flat: Dict[str, Dict[str, Any]] = {}
    for src_canon, entries in per_source.items():
        for entry in entries:
            qn = entry["triton_qualname"]
            current = flat.get(qn)
            if current is None:
                flat[qn] = {**entry, "matched_from": list(entry.get("matched_from", []))}
            else:
                current["score"] = max(current.get("score", 0.0), entry.get("score", 0.0))
                if src_canon not in current["matched_from"]:
                    current["matched_from"].append(src_canon)
                if entry.get("keyword_hit") and not current.get("keyword_hit"):
                    current["keyword_hit"] = entry["keyword_hit"]
    return sorted(flat.values(), key=lambda item: item.get("score", 0.0), reverse=True)
