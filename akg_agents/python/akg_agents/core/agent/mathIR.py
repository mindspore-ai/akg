# Copyright 2026 Huawei Technologies Co., Ltd
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

"""MathIR agent for extracting expression-level math semantics."""

from __future__ import annotations

import importlib
import json
import logging
import os
from functools import partial
from typing import Optional, Tuple

from akg_agents import get_project_root
from akg_agents.core_v2.agents import AgentBase, register_agent
from akg_agents.utils.common_utils import ParserFactory, remove_copyright_from_text

logger = logging.getLogger(__name__)


def ir_resource_json_path(op_name: str) -> str:
    """Return primary MathIR preset path.

    Preset MathIR files live under
    ``op/resources/mathIR/{op_name}.json``.
    """
    safe_op = os.path.basename(str(op_name))
    return os.path.join(
        get_project_root(),
        "op",
        "resources",
        "mathIR",
        f"{safe_op}.json",
    )


def try_load_ir_resource_for_op(
    op_name: Optional[str],
) -> Optional[Tuple[str, str]]:
    """Load preset MathIR text when it exists.

    The canonical location is a flat ``op/resources/mathIR`` directory.
    Callers are expected to provide globally unique ``op_name`` values.
    """
    if not op_name:
        return None

    path = ir_resource_json_path(op_name)
    if not os.path.isfile(path):
        logger.debug("No MathIR preset file: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read(), path


@register_agent
class MathIR(AgentBase):
    """Generate MathIR JSON with a single multi prompt chain."""

    def __init__(
        self,
        op_name: str,
        task_desc: str,
        dsl: str,
        framework: str,
        backend: str,
        arch: str = "",
        parser_config_path: str = None,
        config: dict = None,
    ):
        if config is None:
            raise ValueError("config is required for MathIR")

        self.op_name = op_name
        self.task_desc = remove_copyright_from_text(task_desc)
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.parser_config_path = parser_config_path
        self.config = config
        self.model_config = config.get("agent_model_config", {})
        self.mathir_config = dict(config)
        self.mathir_config.update(config.get("mathIR_config", {}))
        self.codegen_step_count = 0

        context = {
            "dsl": dsl,
            "op_name": op_name,
            "framework": framework,
            "backend": backend,
            "arch": arch,
            "task_desc": self.task_desc,
        }
        if config.get("session_id"):
            context["session_id"] = config["session_id"]
        super().__init__(context=context, config=config)

        ParserFactory.register_parser(
            "mathIR_parser",
            {
                "output_fields": {
                    "code": {
                        "field_type": "dict",
                        "mandatory": True,
                        "field_description": "Expression-level math IR",
                    }
                }
            },
        )

        self.mathIR_multi_prompt = self.load_template("mathIR/mathIRgen_multi.j2")

    def _update_codegen_context(
        self,
        task_info: dict,
        agent_name: str = "mathIR",
        hash_prefix: str = "",
    ) -> None:
        self.codegen_step_count += 1
        self.context.update(
            {
                "agent_name": agent_name,
                "hash": (
                    task_info.get("task_id", "MathIR")
                    + "@"
                    + hash_prefix
                    + str(self.codegen_step_count)
                ),
                "task_id": task_info.get("task_id", ""),
                "step": self.codegen_step_count,
                "workflow_name": task_info.get("workflow_name", ""),
            }
        )

    async def run_preset(self, task_info: dict) -> Optional[Tuple[str, str, str]]:
        preset_result = try_load_ir_resource_for_op(self.op_name)
        if preset_result is None:
            return None

        preset, preset_path = preset_result
        try:
            json.loads(preset)
        except json.JSONDecodeError as exc:
            logger.warning("[%s] invalid preset IR %s: %s", self.op_name, preset_path, exc)
            task_info.pop("preset_ir_json", None)
            task_info.pop("preset_ir_path", None)
            return None

        task_info["preset_ir_json"] = preset
        task_info["preset_ir_path"] = preset_path
        logger.info("[MathIR] using preset IR: %s", preset_path)
        return preset, f"(skipped LLM; preset from {preset_path})", ""

    async def run_mlir_export(self, task_info: dict) -> None:
        task_info["mlir"] = False
        task_info["mlir_compile_code"] = ""
        try:
            from akg_agents.op.utils.mlir_export import export_one

            mlir = export_one(
                py_context=self.task_desc,
                output_type=self.mathir_config.get("mlir_output_type", "scf"),
                device=self.mathir_config.get("mlir_device", "cpu"),
                op_name=self.op_name,
                mlir_opt=self.mathir_config.get("mlir_opt", "mlir-opt"),
                timeout_s=int(self.mathir_config.get("mlir_timeout_s", 120)),
                isolate=bool(self.mathir_config.get("mlir_isolate", True)),
                start_method=self.mathir_config.get("mlir_start_method", "spawn"),
            )
        except Exception as exc:
            logger.info("[MathIR] MLIR export failed, fallback to non-MLIR prompt: %s", exc)
            return None

        if mlir:
            task_info["mlir"] = True
            task_info["mlir_compile_code"] = mlir
            logger.info("[MathIR] MLIR export succeeded")
        return None

    async def run_doc_string(self, task_info: dict) -> None:
        task_info["pytorch_doc_string"] = ""
        try:
            module = importlib.import_module("akg_agents.database.api.doc_extract")
            pytorch_code_to_red_signatures = getattr(
                module,
                "pytorch_code_to_red_signatures",
            )

            task_info["pytorch_doc_string"] = pytorch_code_to_red_signatures(
                self.task_desc,
                only_in_forward=True,
                class_name="Model",
            )
        except Exception as exc:
            logger.info("[MathIR] doc string extraction skipped: %s", exc)
        return None

    async def run_standard_formula(self, task_info: dict) -> None:
        if task_info.get("standard_formula"):
            return None
        try:
            task_info["standard_formula"] = self.load_template("mathIR/defaultIR.j2").format()
        except Exception as exc:
            logger.warning("[MathIR] failed to load defaultIR prompt: %s", exc)
            task_info["standard_formula"] = ""
        return None

    async def run_db_math_query(self, task_info: dict) -> None:
        """Optional placeholder for deployments that provide math formula DB."""
        # The fork does not currently carry aikg_new's math database embedding
        # stack. Keep the hook non-fatal and rely on defaultIR until that module
        # is merged as the database feature.
        logger.info("[MathIR] db math query not configured in this fork; using default formula")
        return None

    async def run(self, task_info: dict, config: Optional[dict] = None) -> Tuple[str, str, str]:
        config = config or self.mathir_config

        strategies = []
        if config.get("mathir_use_preset", False):
            strategies.append(("preset", partial(self.run_preset, task_info=task_info)))
        if config.get("mathir_mlir_export", False):
            strategies.append(("mlir_export", partial(self.run_mlir_export, task_info=task_info)))
        if config.get("mathir_doc_string", False):
            strategies.append(("doc_string", partial(self.run_doc_string, task_info=task_info)))
        if config.get("mathir_db_math_query", False):
            strategies.append(("db_math_query", partial(self.run_db_math_query, task_info=task_info)))

        strategies.append(("standard_formula", partial(self.run_standard_formula, task_info=task_info)))
        strategies.append(("llm", partial(self._run_with_llm, task_info=task_info)))

        for name, fn in strategies:
            logger.info("[MathIR] pipeline: %s", name)
            result = await fn()
            if result is not None:
                return result

        raise RuntimeError(f"[{self.op_name}] all MathIR strategies failed")

    async def _run_with_llm(self, task_info: dict) -> Tuple[str, str, str]:
        self._update_codegen_context(task_info)
        mlir_compile_code = task_info.get("mlir_compile_code", "")
        input_data = {
            "task_desc": self.context.get("task_desc", ""),
            "extra": task_info.get("extra", ""),
            "code_file_content": self.task_desc,
            "mlir": bool(task_info.get("mlir", False) and mlir_compile_code),
            "mlir_compile_code": mlir_compile_code,
            "pytorch_doc_string": task_info.get("pytorch_doc_string", ""),
            "standard_formula": task_info.get("standard_formula", ""),
            "multi_kernel_gen": task_info.get("multi_kernel_gen", True),
            "framework": self.framework,
            "backend": self.backend,
            "arch_name": self.arch,
            "dsl": self.dsl,
        }
        return await self.run_llm(
            self.mathIR_multi_prompt,
            input_data,
            self.model_config.get("mathIR") or self.model_config.get("coder") or "standard",
        )
