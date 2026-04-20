# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Layering invariants after the Phase 2 decoupling.

Two properties are pinned here:

  1. ``adapters/llm_adapter.py`` does NOT import from ``agent/*``.
     Adapters sit *below* agent in the dependency graph (agent wires
     adapters into AgentLoop); a reverse import would reintroduce the
     circular risk that the Phase 2 audit flagged.

  2. ``skill_adapter._get_catalog`` is a true lru-cached factory — no
     mutable module-level ``_catalog`` global — and repeated calls
     share one instance.

Both invariants are source-level / behavioral; they do not require a
running LLM backend.
"""
import ast
import pathlib

from akg_agents.op.autoresearch.agent import skill_adapter as sa


# -- Layering: adapters must not import from agent ---------------------------


class TestAdaptersLayering:
    def test_llm_adapter_has_no_agent_imports(self):
        """Static scan of the adapter source: no ``from ..agent`` or
        ``import ...agent.tools`` anywhere (including inside functions)."""
        adapter_path = (
            pathlib.Path(sa.__file__).parents[1]
            / "adapters"
            / "llm_adapter.py"
        )
        tree = ast.parse(adapter_path.read_text(encoding="utf-8"))

        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                # "..agent" relative or "akg_agents.op.autoresearch.agent" absolute
                is_rel_agent = node.level >= 2 and mod.startswith("agent")
                is_abs_agent = "autoresearch.agent" in mod
                if is_rel_agent or is_abs_agent:
                    offenders.append(
                        f"line {node.lineno}: from "
                        f"{'.' * node.level}{mod} import ..."
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "autoresearch.agent" in (alias.name or ""):
                        offenders.append(
                            f"line {node.lineno}: import {alias.name}"
                        )

        assert not offenders, (
            "adapters/llm_adapter.py must not reach back into agent/* "
            "(adapters live below agent in the dep graph).\n"
            + "\n".join(offenders)
        )


# -- Catalog is an lru_cache, not a mutable global --------------------------


class TestCatalogFactory:
    def test_module_has_no_mutable_catalog_global(self):
        """The old ``_catalog = None`` module attribute is gone — the
        factory now owns its own cache."""
        assert not hasattr(sa, "_catalog"), (
            "Module-level ``_catalog`` singleton was replaced by "
            "functools.lru_cache on _get_catalog."
        )

    def test_get_catalog_is_cached_factory(self):
        """lru_cache-decorated callable exposes cache_clear / cache_info."""
        assert hasattr(sa._get_catalog, "cache_clear")
        assert hasattr(sa._get_catalog, "cache_info")

    def test_get_catalog_returns_same_instance(self):
        sa._get_catalog.cache_clear()
        try:
            assert sa._get_catalog() is sa._get_catalog()
        finally:
            sa._get_catalog.cache_clear()

    def test_cache_clear_forces_reconstruction(self):
        sa._get_catalog.cache_clear()
        try:
            first = sa._get_catalog()
            sa._get_catalog.cache_clear()
            second = sa._get_catalog()
            assert first is not second
        finally:
            sa._get_catalog.cache_clear()
