"""High-level helpers for Triton API database retrieval and prompt rendering."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from akg_agents import get_project_root

logger = logging.getLogger(__name__)


_TRITON_QUALNAME_RE = re.compile(r"^(tl|triton)\.[A-Za-z_][\w.]*$")
STRICT_INSTRUCTION_LINE = "请严格按照API文档的使用方法和约束生成对应的代码API"
TRITON_RECALL_DOC_MAX_CHARS = 600
BASE_API_DOC_SOURCE = "基础 API 文档"
_DOC_API_NAME_RE = re.compile(r"(?m)^(?:#{2,4}\s+|API name:\s*)`?(@?(?:tl|triton)\.[A-Za-z_][\w.]*)")


def resolve_embed_cache_folder(path: Optional[str]) -> Optional[str]:
    """Resolve API database embedding cache paths relative to the akg_agents repo root."""
    if not path:
        return path
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return expanded
    repo_root = os.path.abspath(os.path.join(get_project_root(), "..", ".."))
    return os.path.normpath(os.path.join(repo_root, expanded))


def looks_like_triton_qualname(value: str) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip().lstrip("@").split("(", 1)[0].rstrip(":,*")
    return bool(_TRITON_QUALNAME_RE.match(value))


def _normalize_triton_doc_name(value: str) -> str:
    if not isinstance(value, str):
        return ""
    qualname = value.strip().strip("`").lstrip("@").split("(", 1)[0].rstrip(":,*")
    if not _TRITON_QUALNAME_RE.match(qualname):
        return ""
    return _public_triton_qualname(qualname)


def extract_documented_triton_apis(doc_text: str) -> List[str]:
    """Return public Triton API names already shown in a markdown/API docs block."""
    documented: List[str] = []
    seen = set()
    for match in _DOC_API_NAME_RE.finditer(doc_text or ""):
        qualname = _normalize_triton_doc_name(match.group(1))
        if not qualname or qualname in seen:
            continue
        seen.add(qualname)
        documented.append(qualname)
    return documented


def resolve_triton_api(qualname: str) -> Tuple[Optional[Any], bool]:
    qualname = (qualname or "").strip().lstrip("@").split("(", 1)[0].rstrip(":,*")
    try:
        if qualname.startswith("tl."):
            import triton.language as tl

            obj: Any = tl
        elif qualname.startswith("triton."):
            import triton

            obj = triton
        else:
            return None, False
        for part in qualname.split(".")[1:]:
            obj = getattr(obj, part, None)
            if obj is None:
                return None, False
        return obj, True
    except Exception:
        return None, False


def verify_triton_api_runtime(qualname: str) -> bool:
    if not looks_like_triton_qualname(qualname):
        return True
    if importlib.util.find_spec("triton") is None:
        return True
    _, ok = resolve_triton_api(qualname)
    return ok


def filter_api_doc_blocks(doc_text: str) -> str:
    if not doc_text or not doc_text.strip():
        return doc_text
    if re.search(r"^API name:", doc_text, re.M):
        filtered = _filter_compressed_format(doc_text)
    else:
        filtered = _filter_markdown_format(doc_text)
    if doc_text.strip() and not filtered.strip():
        logger.warning("[api_doc_verify] filter dropped all entries; falling back to original docs")
        return doc_text
    return filtered


def _filter_compressed_format(text: str) -> str:
    out: List[str] = []
    for block in re.split(r"(?m)(?=^API name:\s*)", text):
        if not block.strip():
            continue
        match = re.match(r"^API name:\s*(\S+)", block.strip())
        if match and looks_like_triton_qualname(match.group(1)) and not verify_triton_api_runtime(match.group(1)):
            continue
        out.append(block.rstrip())
    return "\n\n".join(out)


def _filter_markdown_format(text: str) -> str:
    out: List[str] = []
    for part in re.split(r"(?m)(?=^#{2,3} )", text):
        if not part.strip():
            continue
        match = re.match(r"^### +(@?\S+)", part)
        if match and looks_like_triton_qualname(match.group(1)) and not verify_triton_api_runtime(match.group(1)):
            continue
        out.append(part)
    return "".join(out)


def _pack_triton_recall_entry(api: dict) -> dict:
    payload = api.get("payload") or {}
    raw_qualname = api.get("triton_qualname", "")
    qualname = _public_triton_qualname(raw_qualname)
    signature = api.get("redsig") or api.get("signature") or payload.get("redsig") or payload.get("signature") or ""
    if raw_qualname and qualname != raw_qualname and signature.startswith(raw_qualname):
        signature = qualname + signature[len(raw_qualname) :]
    return {
        "triton_qualname": qualname,
        "triton_name": api.get("triton_name", ""),
        "tags": api.get("tags") or payload.get("tags") or [],
        "signature": signature,
        "doc": _clean_doc(api.get("doc", "") or payload.get("doc", ""), max_chars=TRITON_RECALL_DOC_MAX_CHARS),
        "matched_from": api.get("matched_from") or [],
        "keyword_hit": api.get("keyword_hit"),
        "fallback": bool(api.get("fallback", False)),
    }


def _public_triton_qualname(qualname: str) -> str:
    if not isinstance(qualname, str):
        return ""
    for prefix in ("tl.core.", "tl.standard."):
        if qualname.startswith(prefix):
            candidate = "tl." + qualname.rsplit(".", 1)[-1]
            if resolve_triton_api(candidate)[1]:
                return candidate
    return qualname


def _clean_doc(doc: str, *, max_chars: Optional[int] = None) -> str:
    doc = (doc or "").replace("\r\n", "\n")
    doc = re.sub(r"\n{3,}", "\n\n", doc).strip()
    if max_chars is not None and len(doc) > max_chars:
        snippet = doc[:max_chars].rstrip()
        line_cut = snippet.rfind("\n")
        if line_cut > int(max_chars * 0.65):
            snippet = snippet[:line_cut].rstrip()
        return snippet + "\n..."
    return doc


def filter_recall_entries(entries: List[dict], *, verify_runtime: bool = False) -> Tuple[List[dict], List[str]]:
    kept: List[dict] = []
    dropped: List[str] = []
    for entry in entries or []:
        qn = entry.get("triton_qualname", "")
        if verify_runtime and not verify_triton_api_runtime(qn):
            dropped.append(qn)
            continue
        kept.append(_pack_triton_recall_entry(entry))
    return kept, dropped


def render_triton_recall(
    task_info: dict,
    *,
    load_template_fn: Optional[Callable[[str], Any]] = None,
    verify_runtime: bool = False,
    documented_apis: Optional[List[str]] = None,
) -> str:
    by_source = task_info.get("triton_api_recall_by_source") or {}
    flat_list = task_info.get("triton_api_recall") or []
    documented_apis = documented_apis or task_info.get("documented_triton_apis") or []
    if not by_source and not flat_list:
        return ""
    if by_source:
        recall_by_source: Dict[str, List[dict]] = {}
        for source, candidates in by_source.items():
            kept, dropped = filter_recall_entries(candidates, verify_runtime=verify_runtime)
            if dropped:
                logger.info("[api_doc_verify] dropped invalid recall APIs: %s", dropped)
            if kept:
                recall_by_source[source] = kept
        if not recall_by_source:
            return ""
        marked_by_source = _mark_recall_duplicates(
            recall_by_source,
            documented_apis=documented_apis,
        )
        source_groups = _build_recall_source_groups(marked_by_source)
        if load_template_fn:
            return load_template_fn("database/triton_doc_gen_by_source.j2").format(
                recall_by_source=source_groups
            )
        return _render_recall_by_source(source_groups)
    kept_flat, dropped = filter_recall_entries(flat_list, verify_runtime=verify_runtime)
    if dropped:
        logger.info("[api_doc_verify] dropped invalid recall APIs: %s", dropped)
    return _render_recall_flat(kept_flat, documented_apis=documented_apis)


def _documented_api_source_map(documented_apis: Optional[List[str]]) -> Dict[str, str]:
    source_map: Dict[str, str] = {}
    for api in documented_apis or []:
        qualname = _normalize_triton_doc_name(api)
        if qualname:
            source_map[qualname] = BASE_API_DOC_SOURCE
    return source_map


def _mark_recall_duplicates(
    recall_by_source: Dict[str, List[dict]],
    *,
    documented_apis: Optional[List[str]] = None,
) -> Dict[str, List[dict]]:
    seen_sources: Dict[str, str] = _documented_api_source_map(documented_apis)
    marked_by_source: Dict[str, List[dict]] = {}
    for source, candidates in recall_by_source.items():
        marked_candidates: List[dict] = []
        for api in candidates:
            marked_api = dict(api)
            api_key = _normalize_triton_doc_name(
                marked_api.get("triton_qualname") or marked_api.get("triton_name") or ""
            )
            if api_key and api_key in seen_sources:
                marked_api["detail_already_rendered"] = True
                marked_api["first_rendered_source"] = seen_sources[api_key]
            else:
                marked_api["detail_already_rendered"] = False
                marked_api["first_rendered_source"] = ""
                if api_key:
                    seen_sources[api_key] = source
            marked_candidates.append(marked_api)
        marked_by_source[source] = marked_candidates
    return marked_by_source


def _build_recall_source_groups(marked_by_source: Dict[str, List[dict]]) -> Dict[str, dict]:
    groups: Dict[str, dict] = {}
    for source, candidates in marked_by_source.items():
        displayed = [api for api in candidates if api.get("detail_already_rendered")]
        supplemental = [api for api in candidates if not api.get("detail_already_rendered")]
        groups[source] = {
            "candidates": candidates,
            "displayed_apis": displayed,
            "supplemental_apis": supplemental,
        }
    return groups


def _format_api_list(apis: List[dict]) -> str:
    if not apis:
        return "无"
    return "、".join(f"`{api.get('triton_qualname') or api.get('triton_name')}`" for api in apis)


def _render_recall_by_source(recall_by_source: Dict[str, dict]) -> str:
    lines = ["## 可能用到的 Triton API（按 PyTorch API 分组，来自本地 Triton API database）", ""]
    for source, group in recall_by_source.items():
        candidates = group.get("candidates", [])
        displayed = group.get("displayed_apis", [])
        supplemental = group.get("supplemental_apis", [])
        lines.append(f"### 来源 PyTorch API: `{source}`（共 {len(candidates)} 条候选）")
        lines.append(f"已展示: {_format_api_list(displayed)}")
        lines.append(f"补充API: {_format_api_list(supplemental)}")
        if not supplemental:
            lines.append("- 说明: 本组召回 API 已由基础 API 文档或前文召回覆盖，此处不重复展开。")
            continue
        for idx, api in enumerate(supplemental, 1):
            lines.append(f"- {idx}. Triton API: `{api['triton_qualname']}` / `{api['triton_name']}`")
            lines.extend(
                [
                    f"  - Tags: {api['tags']}",
                    f"  - Triton 函数签名: {api['signature']}",
                    f"  - Triton 函数文档: {api['doc']}",
                ]
            )
    return "\n".join(lines)


def _render_recall_flat(entries: List[dict], *, documented_apis: Optional[List[str]] = None) -> str:
    seen_sources = _documented_api_source_map(documented_apis)
    lines = ["## 可能用到的 Triton API（来自本地 Triton API database）", ""]
    for api in entries:
        api_key = _normalize_triton_doc_name(api.get("triton_qualname") or api.get("triton_name") or "")
        if api_key and api_key in seen_sources:
            lines.extend(
                [
                    f"- Triton API: {api['triton_qualname']} / {api['triton_name']}",
                    f"- 说明: 详细内容已在 `{seen_sources[api_key]}` 中展示，此处仅保留召回命中关系。",
                    "",
                ]
            )
            continue
        if api_key:
            seen_sources[api_key] = "前文召回"
        lines.extend(
            [
                f"- Triton API: {api['triton_qualname']} / {api['triton_name']}",
                f"- Tags: {api['tags']}",
                f"- 或许可以匹配的 PyTorch API: {api['matched_from']}",
                f"- Triton 函数签名: {api['signature']}",
                f"- Triton 函数文档: {api['doc']}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def compose_api_docs_block(base_block: str, recall_block: str) -> str:
    sections: List[str] = []
    if base_block and base_block.strip():
        sections.append("## 基础 API 文档\n" + base_block.strip())
    if recall_block and recall_block.strip():
        sections.append(recall_block.strip())
    if sections:
        sections.append(STRICT_INSTRUCTION_LINE)
    return "\n\n".join(sections)


def persist_api_recall_artifacts(
    *,
    task_info: dict,
    log_dir: str,
    op_name: Optional[str] = None,
    rendered_text: Optional[str] = None,
) -> Tuple[str, str]:
    """Persist structured and rendered API recall artifacts for downstream prompts."""
    if not log_dir:
        return "", ""
    expanded_log_dir = os.path.expanduser(log_dir)
    op_name = op_name or task_info.get("op_name") or "unknown_op"
    save_dir = os.path.join(expanded_log_dir, op_name, "api_recall")
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "api_recall_structured.json")
    docs_path = os.path.join(save_dir, "api_recall_rendered.md")
    if rendered_text is None:
        rendered_text = render_triton_recall(task_info, verify_runtime=True)

    structured = {
        "op_name": op_name,
        "task_id": task_info.get("task_id", ""),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_database_enabled": task_info.get("api_database_enabled", False),
        "api_database_status": task_info.get("api_database_status", ""),
        "api_database_recall_hash": task_info.get("api_database_recall_hash", ""),
        "api_database_source_kind": task_info.get("api_database_source_kind", ""),
        "api_database_source_apis": task_info.get("api_database_source_apis", []),
        "api_database_embed_cache_folder": task_info.get("api_database_embed_cache_folder", ""),
        "documented_triton_apis": task_info.get("documented_triton_apis", []),
        "triton_api_recall": task_info.get("triton_api_recall", []),
        "triton_api_recall_by_source": task_info.get("triton_api_recall_by_source", {}),
        "rendered_docs_path": docs_path,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(structured, handle, ensure_ascii=False, indent=2, default=str)
    with open(docs_path, "w", encoding="utf-8") as handle:
        handle.write(rendered_text or "")
    return json_path, docs_path


def _load_recall_block_from_file(path: str) -> Tuple[str, Optional[tuple]]:
    if not path:
        return "", None
    try:
        stat = os.stat(path)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        logger.warning("[api_doc_recall_file] read failed: %s: %s", path, exc)
        return "", None
    if not text.strip():
        return "", None
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    signature = ("file", os.path.abspath(path), stat.st_size, stat.st_mtime_ns, digest)
    return text, signature


try:
    from akg_agents.core_v2.agents import AgentBase as _AgentBase
    from akg_agents.utils.common_utils import ParserFactory as _ParserFactory

    _AGENT_BASE_AVAILABLE = True
except Exception:
    _AgentBase = object  # type: ignore
    _ParserFactory = None  # type: ignore
    _AGENT_BASE_AVAILABLE = False


class ApiDocsAgent(_AgentBase):
    """Generate database-aware API docs for Coder prompts."""

    LLM_COMPRESS_THRESHOLD = 24000

    def _get_agent_type(self) -> str:
        return "coder"

    def __init__(
        self,
        *,
        op_name: str,
        dsl: str,
        framework: str,
        backend: str,
        arch: str,
        config: dict,
        base_doc: dict,
    ):
        if not _AGENT_BASE_AVAILABLE:
            raise RuntimeError("ApiDocsAgent requires akg_agents.core_v2.agents.AgentBase")
        context = {
            "agent_name": "api",
            "dsl": dsl,
            "op_name": op_name,
            "framework": framework,
            "backend": backend,
            "arch": arch,
        }
        super().__init__(context=context, config=config)
        self.op_name = op_name
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.base_doc = base_doc
        self.model_config = (config or {}).get("agent_model_config", {})
        api_docs_cfg = (config or {}).get("api_docs", {}) or {}
        raw_compress_enabled = api_docs_cfg.get("llm_compress_enabled", False)
        if isinstance(raw_compress_enabled, str):
            self.llm_compress_enabled = raw_compress_enabled.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self.llm_compress_enabled = bool(raw_compress_enabled)
        try:
            self.llm_compress_threshold = int(
                api_docs_cfg.get("llm_compress_threshold", self.LLM_COMPRESS_THRESHOLD)
            )
        except (TypeError, ValueError):
            self.llm_compress_threshold = self.LLM_COMPRESS_THRESHOLD
        self.api_docs_prompt = self.load_template("utils/api_gen_template.j2")
        self._call_count = 0
        self.step_count = 0
        self._cached_filtered_base: Optional[str] = None
        self._cached_recall_signature: Optional[tuple] = None
        self._cached_recall_block = ""
        self._final_doc_cache: Dict[str, str] = {}
        self._final_doc_cache_meta: Dict[str, Dict[str, Any]] = {}

    async def generate(self, *, task_info: dict, sketch: str = "", conductor_suggestion: str = "") -> str:
        self._call_count += 1

        raw_base = self.base_doc.get("api_docs") or ""
        if self._cached_filtered_base is None:
            self._cached_filtered_base = filter_api_doc_blocks(raw_base)
        filtered_base = self._cached_filtered_base
        documented_triton_apis = extract_documented_triton_apis(filtered_base)

        recall_source = "state"
        recall_file_path = task_info.get("api_recall_docs_path") or ""
        recall_block, file_signature = _load_recall_block_from_file(recall_file_path)
        if file_signature is not None:
            recall_source = "file"
            recall_signature = file_signature
        else:
            by_source = task_info.get("triton_api_recall_by_source") or {}
            flat = task_info.get("triton_api_recall") or []
            recall_signature = (
                "state",
                tuple(sorted(by_source.keys())),
                sum(len(v) for v in by_source.values()) if by_source else 0,
                len(flat),
            )
            if recall_signature != self._cached_recall_signature:
                self._cached_recall_block = render_triton_recall(
                    task_info,
                    load_template_fn=self.load_template,
                    verify_runtime=True,
                    documented_apis=documented_triton_apis,
                )
                self._cached_recall_signature = recall_signature
            recall_block = self._cached_recall_block
        base_for_prompt = filtered_base
        combined = compose_api_docs_block(base_for_prompt, recall_block)

        error_log = task_info.get("verifier_error", "") or ""
        cache_key = self._make_final_doc_cache_key(
            recall_signature,
            sketch,
            conductor_suggestion,
            error_log,
        )
        cached_final = self._final_doc_cache.get(cache_key)
        cache_hit = cached_final is not None
        if cache_hit:
            final_doc = cached_final or ""
            compressed_used = self._final_doc_cache_meta.get(cache_key, {}).get(
                "compressed_used",
                False,
            )
            meta_entry = self._final_doc_cache_meta.setdefault(
                cache_key,
                {"hits": 0, "compressed_used": compressed_used},
            )
            meta_entry["hits"] = meta_entry.get("hits", 0) + 1
        else:
            compressed_used = False
            if (
                self.llm_compress_enabled
                and len(combined) > self.llm_compress_threshold
                and base_for_prompt.strip()
            ):
                compressed = await self._llm_compress(
                    base_for_prompt,
                    sketch,
                    conductor_suggestion,
                    task_info,
                )
                final_doc = compose_api_docs_block(
                    filter_api_doc_blocks(compressed),
                    recall_block,
                )
                compressed_used = True
            else:
                final_doc = combined
            self._final_doc_cache[cache_key] = final_doc
            self._final_doc_cache_meta[cache_key] = {
                "hits": 0,
                "compressed_used": compressed_used,
            }

        try:
            self._persist(
                task_info=task_info,
                final_doc=final_doc,
                raw_base_len=len(raw_base),
                filtered_base_len=len(base_for_prompt),
                recall_len=len(recall_block),
                combined_len=len(combined),
                final_len=len(final_doc),
                compressed_used=compressed_used,
                cache_key=cache_key,
                cache_hit=cache_hit,
                sketch_len=len(sketch or ""),
                suggestion_len=len(conductor_suggestion or ""),
                error_log_len=len(error_log),
                recall_source=recall_source,
            )
        except Exception as exc:
            logger.warning("[api_doc_persist] save failed: %s: %s", type(exc).__name__, exc)
        return final_doc

    @staticmethod
    def _make_final_doc_cache_key(
        recall_signature: tuple,
        sketch: str,
        conductor_suggestion: str,
        error_log: str,
    ) -> str:
        digest = hashlib.md5()
        digest.update(repr(recall_signature).encode("utf-8"))
        digest.update(b"\x00sketch\x00")
        digest.update((sketch or "").encode("utf-8"))
        digest.update(b"\x00suggestion\x00")
        digest.update((conductor_suggestion or "").encode("utf-8"))
        digest.update(b"\x00error_log\x00")
        digest.update((error_log or "").encode("utf-8"))
        return digest.hexdigest()

    def _persist(
        self,
        *,
        task_info: dict,
        final_doc: str,
        raw_base_len: int,
        filtered_base_len: int,
        recall_len: int,
        combined_len: int,
        final_len: int,
        compressed_used: bool,
        cache_key: str = "",
        cache_hit: bool = False,
        sketch_len: int = 0,
        suggestion_len: int = 0,
        error_log_len: int = 0,
        recall_source: str = "state",
    ) -> None:
        log_dir = (self.config or {}).get("log_dir", "") if self.config else ""
        if not log_dir:
            return
        log_dir = os.path.expanduser(log_dir)
        op_name = self.op_name or task_info.get("op_name") or "unknown_op"
        task_id = task_info.get("task_id") or "00"
        save_dir = os.path.join(log_dir, op_name, "api_docs")
        os.makedirs(save_dir, exist_ok=True)

        base_name = f"I{task_id}_C{self._call_count:03d}"
        md_path = os.path.join(save_dir, f"{base_name}_api_docs.md")
        meta_path = os.path.join(save_dir, f"{base_name}_api_meta.json")
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(final_doc or "")

        by_source = task_info.get("triton_api_recall_by_source") or {}
        flat = task_info.get("triton_api_recall") or []
        meta = {
            "op_name": op_name,
            "task_id": task_id,
            "call_seq": self._call_count,
            "llm_compress_step": self.step_count,
            "compressed_used": compressed_used,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "cache_hits_for_key": (
                self._final_doc_cache_meta.get(cache_key, {}).get("hits", 0)
                if cache_key
                else 0
            ),
            "lengths": {
                "raw_base": raw_base_len,
                "filtered_base": filtered_base_len,
                "recall": recall_len,
                "combined": combined_len,
                "final": final_len,
                "sketch": sketch_len,
                "conductor_suggestion": suggestion_len,
                "error_log": error_log_len,
                "llm_compress_enabled": self.llm_compress_enabled,
                "llm_compress_threshold": self.llm_compress_threshold,
            },
            "recall_by_source_qualnames": {
                source: [entry.get("triton_qualname", "") for entry in entries]
                for source, entries in by_source.items()
            },
            "recall_flat_qualnames": [entry.get("triton_qualname", "") for entry in flat],
            "recall_by_source_sizes": {
                source: len(entries) for source, entries in by_source.items()
            },
            "api_database_source_kind": task_info.get("api_database_source_kind", ""),
            "api_database_source_apis": task_info.get("api_database_source_apis", []),
            "api_database_status": task_info.get("api_database_status", ""),
            "api_database_embed_cache_folder": task_info.get("api_database_embed_cache_folder", ""),
            "api_recall_source": recall_source,
            "api_recall_json_path": task_info.get("api_recall_json_path", ""),
            "api_recall_docs_path": task_info.get("api_recall_docs_path", ""),
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        logger.info(
            "[api_doc_persist] saved %s (cache_hit=%s, compressed=%s)",
            md_path,
            cache_hit,
            compressed_used,
        )

    async def _llm_compress(
        self,
        filtered_base: str,
        sketch: str,
        conductor_suggestion: str,
        task_info: dict,
    ) -> str:
        if _ParserFactory is None:
            return filtered_base
        api_parser = _ParserFactory.get_api_parser()
        format_api_instructions = api_parser.get_format_instructions()

        by_source = task_info.get("triton_api_recall_by_source") or {}
        must_keep_apis: List[str] = []
        seen = set()
        for entries in by_source.values():
            for entry in entries:
                if not entry.get("keyword_hit"):
                    continue
                qualname = entry.get("triton_qualname", "")
                if qualname and qualname not in seen:
                    seen.add(qualname)
                    must_keep_apis.append(qualname)

        api_input_data = {
            **self.base_doc,
            "api_docs": filtered_base,
            "sketch": sketch,
            "llm_suggestions": conductor_suggestion,
            "error_log": task_info.get("verifier_error", ""),
            "must_keep_apis": must_keep_apis,
            "format_instructions": format_api_instructions,
        }
        self.step_count += 1
        self.context.update(
            {
                "agent_name": "api",
                "hash": task_info.get("task_id", "Api") + "@" + str(self.step_count),
                "task_id": task_info.get("task_id", ""),
                "step": self.step_count,
            }
        )
        api_docs_json, _, _ = await self.run_llm(
            self.api_docs_prompt,
            api_input_data,
            self.model_config.get("api_generator", "standard"),
        )
        parsed_content = api_parser.parse(api_docs_json)
        names = list(parsed_content.api_name)
        descriptions = list(parsed_content.api_desc)
        examples = list(parsed_content.api_example)

        if must_keep_apis:
            present_names = {name.strip() for name in names if isinstance(name, str)}
            for qualname in must_keep_apis:
                if qualname in present_names:
                    continue
                entry = None
                for entries in by_source.values():
                    for candidate in entries:
                        if candidate.get("triton_qualname") == qualname and candidate.get("keyword_hit"):
                            entry = candidate
                            break
                    if entry is not None:
                        break
                names.append(qualname)
                if entry is None:
                    descriptions.append("keyword_hit recall; see triton_api_recall")
                    examples.append("<no example>")
                else:
                    doc_lines = (entry.get("doc") or "").strip().splitlines()[:2]
                    descriptions.append(
                        " ".join(doc_lines)
                        if doc_lines
                        else f"keyword_hit={entry.get('keyword_hit')}"
                    )
                    examples.append(entry.get("redsig") or entry.get("signature") or "<no example>")

        compressed = "\n\n".join(
            f"API name: {name}\nAPI description:{desc}\nAPI implement：\n{example}"
            for name, desc, example in zip(names, descriptions, examples)
        )
        return filter_api_doc_blocks(compressed)


def retrieve_and_store_triton_apis(
    *,
    task_info: dict,
    task_desc: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    triton_collection: str = "triton_api",
    torch_collection: str = "torch_api",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_cache_folder: Optional[str] = None,
    force_rebuild: bool = False,
    force_rebuild_triton: bool = False,
    force_rebuild_torch: bool = False,
    topk_per_query: int = 32,
    filter_tags: Optional[List[str]] = None,
    target_backend: str = "cuda",
    enable_keyword_recall: bool = True,
    keyword_map: Optional[Dict[str, List[str]]] = None,
    min_keep: int = 2,
    max_keep: int = 10,
    elbow_min_gap_ratio: float = 0.15,
    relative_decay: float = 0.90,
    keyword_fallback_qdrant: bool = True,
    keyword_fallback_limit_per_kw: int = 4,
) -> None:
    from akg_agents.database.api.api_db_bootstrap import ensure_qdrant_databases
    from akg_agents.database.api.pytorch_api_extract import aten_dispatch_code_to_doc_list, pytorch_code_to_doc_list
    from akg_agents.database.api.triton_api_extract import retrieve_triton_apis

    embed_cache_folder = resolve_embed_cache_folder(embed_cache_folder)
    task_info["api_database_embed_cache_folder"] = embed_cache_folder or ""
    ensure_qdrant_databases(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        triton_collection=triton_collection,
        torch_collection=torch_collection,
        embed_model=embed_model,
        embed_cache_folder=embed_cache_folder,
        force_rebuild=force_rebuild,
        force_rebuild_triton=force_rebuild_triton,
        force_rebuild_torch=force_rebuild_torch,
        target_backend=target_backend,
    )
    source_kind = "aten_dispatch"
    try:
        docs = aten_dispatch_code_to_doc_list(task_desc, class_name="Model")
    except Exception as exc:
        logger.warning("[api_database] ATen dispatch extraction failed, falling back to AST extraction: %s: %s", type(exc).__name__, exc)
        docs = []
    if not docs:
        source_kind = "ast"
        docs = pytorch_code_to_doc_list(task_desc, only_in_forward=True, class_name="Model")
    task_info["api_database_source_kind"] = source_kind
    task_info["api_database_source_apis"] = [doc.get("canonical", "") for doc in docs if doc.get("canonical")]
    results_by_source = retrieve_triton_apis(
        docs,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        triton_collection=triton_collection,
        torch_collection=torch_collection,
        embed_model_name=embed_model,
        embed_cache_folder=embed_cache_folder,
        filter_tags=filter_tags or ["tl"],
        topk_per_query=topk_per_query,
        target_backend=target_backend,
        group_by_source=True,
        enable_keyword_recall=enable_keyword_recall,
        keyword_map=keyword_map,
        min_keep=min_keep,
        max_keep=max_keep,
        elbow_min_gap_ratio=elbow_min_gap_ratio,
        relative_decay=relative_decay,
        keyword_fallback_qdrant=keyword_fallback_qdrant,
        keyword_fallback_limit_per_kw=keyword_fallback_limit_per_kw,
    )
    flat_by_qn: Dict[str, Dict[str, Any]] = {}
    for source, entries in results_by_source.items():
        for entry in entries:
            qn = entry.get("triton_qualname", "")
            current = flat_by_qn.get(qn)
            if current is None:
                flat_by_qn[qn] = {**entry, "matched_from": list(entry.get("matched_from", []))}
            else:
                current["score"] = max(current.get("score", 0.0), entry.get("score", 0.0))
                if source not in current["matched_from"]:
                    current["matched_from"].append(source)
                if entry.get("keyword_hit") and not current.get("keyword_hit"):
                    current["keyword_hit"] = entry["keyword_hit"]
    task_info["triton_api_recall"] = sorted(flat_by_qn.values(), key=lambda item: item.get("score", 0.0), reverse=True)
    task_info["triton_api_recall_by_source"] = results_by_source


def is_api_database_available(config: dict) -> Tuple[bool, str]:
    if importlib.util.find_spec("qdrant_client") is None:
        return False, "missing dependency: qdrant-client"
    if importlib.util.find_spec("sentence_transformers") is None:
        return False, "missing dependency: sentence-transformers"
    db_cfg = (config or {}).get("api_database", {}) or {}
    host = db_cfg.get("qdrant_host", "localhost")
    port = int(db_cfg.get("qdrant_port", 6333))
    try:
        from qdrant_client import QdrantClient

        QdrantClient(host=host, port=port, trust_env=False).get_collections()
        return True, ""
    except Exception as exc:
        return False, f"qdrant unavailable at {host}:{port}: {type(exc).__name__}: {exc}"


def maybe_retrieve_and_store_triton_apis(
    *,
    task_info: dict,
    task_desc: str,
    target_backend: str,
    config: dict,
) -> None:
    db_cfg = (config or {}).get("api_database", {}) or {}
    if not db_cfg.get("enabled", False):
        task_info["api_database_enabled"] = False
        return
    if os.environ.get("AKG_AGENTS_DISABLE_API_DATABASE", "").lower() in {"1", "true", "yes", "on"}:
        task_info["api_database_enabled"] = False
        task_info["api_database_status"] = "disabled by AKG_AGENTS_DISABLE_API_DATABASE"
        return
    available, reason = is_api_database_available(config)
    if not available:
        task_info["api_database_enabled"] = False
        task_info["api_database_status"] = reason
        logger.warning("[api_database] skip retrieval: %s", reason)
        return
    retrieve_and_store_triton_apis(
        task_info=task_info,
        task_desc=task_desc,
        qdrant_host=db_cfg.get("qdrant_host", "localhost"),
        qdrant_port=int(db_cfg.get("qdrant_port", 6333)),
        triton_collection=db_cfg.get("triton_collection", "triton_api"),
        torch_collection=db_cfg.get("torch_collection", "torch_api"),
        embed_model=db_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        embed_cache_folder=db_cfg.get("embed_cache_folder"),
        force_rebuild=bool(db_cfg.get("force_rebuild", False)),
        force_rebuild_triton=bool(db_cfg.get("force_rebuild_triton", False)),
        force_rebuild_torch=bool(db_cfg.get("force_rebuild_torch", False)),
        topk_per_query=int(db_cfg.get("topk_per_query", 32)),
        filter_tags=db_cfg.get("filter_tags", ["tl"]),
        target_backend=db_cfg.get("target_backend", target_backend),
        enable_keyword_recall=bool(db_cfg.get("enable_keyword_recall", True)),
        min_keep=int(db_cfg.get("min_keep", 2)),
        max_keep=int(db_cfg.get("max_keep", 10)),
        elbow_min_gap_ratio=float(db_cfg.get("elbow_min_gap_ratio", 0.15)),
        relative_decay=float(db_cfg.get("relative_decay", 0.90)),
        keyword_fallback_qdrant=bool(db_cfg.get("keyword_fallback_qdrant", True)),
        keyword_fallback_limit_per_kw=int(db_cfg.get("keyword_fallback_limit_per_kw", 4)),
    )
    task_info["api_database_enabled"] = True
    task_info["api_database_status"] = "ok"
    task_info["api_database_recall_hash"] = _recall_hash(task_info)


def _recall_hash(task_info: dict) -> str:
    digest = hashlib.md5()
    for source, entries in sorted((task_info.get("triton_api_recall_by_source") or {}).items()):
        digest.update(source.encode("utf-8"))
        for entry in entries:
            digest.update((entry.get("triton_qualname", "") or "").encode("utf-8"))
    return digest.hexdigest()
