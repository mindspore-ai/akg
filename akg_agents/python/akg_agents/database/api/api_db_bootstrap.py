"""Bootstrap Qdrant collections used by Triton API retrieval."""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

from akg_agents import get_project_root


def _resolve_cache_folder(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(get_project_root(), "..", ".."))
    return os.path.normpath(os.path.join(repo_root, path))


def _collection_status(client, name: str) -> Tuple[bool, int]:
    try:
        info = client.get_collection(collection_name=name)
        count = getattr(info, "points_count", None)
        if count is None:
            count = client.count(collection_name=name, exact=True).count
        return True, int(count or 0)
    except Exception:
        return False, 0


def _ensure_triton_api(
    client,
    *,
    collection: str,
    embed_model: str,
    embed_cache_folder: Optional[str],
    qdrant_host: str,
    qdrant_port: int,
    force_rebuild: bool,
    target_backend: Optional[str],
    min_points: int,
) -> bool:
    from akg_agents.database.api.triton_api_extract import TritonApiScanner, detect_active_backends, normalize_target_backends, push_to_qdrant

    exists, count = _collection_status(client, collection)
    if exists and count >= min_points and not force_rebuild:
        print(f"[bootstrap] '{collection}' OK (points={count}), skip")
        return False
    backends = normalize_target_backends(target_backend) if target_backend else detect_active_backends()
    print(f"[bootstrap] build '{collection}', triton backends={backends or '(none)'}")
    entries = TritonApiScanner(backends=backends, skip_private=True).scan()
    push_to_qdrant(
        entries,
        host=qdrant_host,
        port=qdrant_port,
        collection=collection,
        embed_model=embed_model,
        cache_folder=embed_cache_folder,
        recreate=True,
    )
    return True


def _ensure_torch_api(
    client,
    *,
    collection: str,
    embed_model: str,
    embed_cache_folder: Optional[str],
    qdrant_host: str,
    qdrant_port: int,
    force_rebuild: bool,
    min_points: int,
) -> bool:
    from akg_agents.database.api.pytorch_api_extract import push_torch_api_to_qdrant, scan_torch_api

    exists, count = _collection_status(client, collection)
    if exists and count >= min_points and not force_rebuild:
        print(f"[bootstrap] '{collection}' OK (points={count}), skip")
        return False
    entries = scan_torch_api()
    print(f"[bootstrap] build '{collection}', scanned {len(entries)} torch APIs")
    push_torch_api_to_qdrant(
        entries,
        host=qdrant_host,
        port=qdrant_port,
        collection=collection,
        embed_model=embed_model,
        cache_folder=embed_cache_folder,
        recreate=True,
    )
    return True


_BOOTSTRAPPED_KEYS = set()


def ensure_qdrant_databases(
    *,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    triton_collection: str = "triton_api",
    torch_collection: str = "torch_api",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_cache_folder: Optional[str] = "python/akg_agents/database/api/embed_model_cache",
    force_rebuild: bool = False,
    force_rebuild_triton: bool = False,
    force_rebuild_torch: bool = False,
    target_backend: Optional[str] = "cuda",
    min_points: int = 50,
    skip_if_already_checked: bool = True,
) -> None:
    """Ensure both Triton and PyTorch API collections exist and are non-empty."""
    rebuild_triton = force_rebuild or force_rebuild_triton
    rebuild_torch = force_rebuild or force_rebuild_torch
    embed_cache_folder = _resolve_cache_folder(embed_cache_folder)
    bootstrap_key = (
        qdrant_host,
        qdrant_port,
        triton_collection,
        torch_collection,
        embed_model,
        embed_cache_folder,
        target_backend,
        bool(rebuild_triton),
        bool(rebuild_torch),
        min_points,
    )
    if skip_if_already_checked and bootstrap_key in _BOOTSTRAPPED_KEYS:
        return

    from qdrant_client import QdrantClient

    started = time.time()
    client = QdrantClient(host=qdrant_host, port=qdrant_port, trust_env=False)
    wrote_triton = _ensure_triton_api(
        client,
        collection=triton_collection,
        embed_model=embed_model,
        embed_cache_folder=embed_cache_folder,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        force_rebuild=rebuild_triton,
        target_backend=target_backend,
        min_points=min_points,
    )
    wrote_torch = _ensure_torch_api(
        client,
        collection=torch_collection,
        embed_model=embed_model,
        embed_cache_folder=embed_cache_folder,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        force_rebuild=rebuild_torch,
        min_points=min_points,
    )
    print(
        f"[bootstrap] qdrant ready in {time.time() - started:.1f}s "
        f"[triton={'+' if wrote_triton else '='} torch={'+' if wrote_torch else '='}]"
    )
    _BOOTSTRAPPED_KEYS.add(bootstrap_key)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ensure triton_api and torch_api collections exist in Qdrant")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--triton-collection", default="triton_api")
    parser.add_argument("--torch-collection", default="torch_api")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed-cache", default="python/akg_agents/database/api/embed_model_cache")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild both Triton and Torch collections.")
    parser.add_argument("--force-rebuild-triton", action="store_true", help="Rebuild only the Triton API collection.")
    parser.add_argument("--force-rebuild-torch", action="store_true", help="Rebuild only the Torch API collection.")
    parser.add_argument("--target-backend", default="cuda", help="Logical Triton target backend, e.g. cuda, amd, ascend, intel, or auto.")
    parser.add_argument("--min-points", type=int, default=50)
    args = parser.parse_args()
    ensure_qdrant_databases(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        triton_collection=args.triton_collection,
        torch_collection=args.torch_collection,
        embed_model=args.embed_model,
        embed_cache_folder=args.embed_cache,
        force_rebuild=args.force_rebuild,
        force_rebuild_triton=args.force_rebuild_triton,
        force_rebuild_torch=args.force_rebuild_torch,
        target_backend=args.target_backend,
        min_points=args.min_points,
        skip_if_already_checked=False,
    )


if __name__ == "__main__":
    main()
