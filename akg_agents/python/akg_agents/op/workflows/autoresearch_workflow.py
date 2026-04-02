"""
AutoresearchWorkflow — 自主迭代深度优化 workflow.

将 autoresearch 框架封装为 LangGraph 单节点 workflow，
通过 KernelAgent 的 ToolExecutor 统一调用。

核心流程:
  1. Seed: KernelGen → CodeChecker 稳定化
  2. 知识组装: skill 系统 + hardware docs + API docs
  3. scaffold_task_dir: 生成任务目录
  4. AgentLoop: 自主迭代优化 (eval_fn → KernelVerifier)
  5. 读取最优结果返回
"""

import copy
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from akg_agents.op.workflows.base_workflow import OpBaseWorkflow, _DSL_DOCS_DIR_MAP
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.core_v2.workflows.registry import register_workflow

logger = logging.getLogger(__name__)


@register_workflow(scopes=["op"])
class AutoresearchWorkflow(OpBaseWorkflow):
    """Autoresearch Workflow: 自主迭代深度优化

    使用 autoresearch 框架对已有 kernel 进行多轮自主迭代优化。
    Agent 驱动的 ReAct 循环: 读取代码 → 分析 → 编辑 → 评测 → keep/discard。
    """

    TOOL_NAME = "call_autoresearch_workflow"

    DESCRIPTION = """
使用 autoresearch 框架对已有 kernel 进行自主迭代深度优化。Agent 在多轮循环中自主决策编辑方向，
每轮编辑后通过 KernelVerifier 验证 + profile，自动 keep/discard。

核心特点：
- **自主迭代**：Agent 自主规划优化策略，根据评测反馈调整方向
- **结构化计划**：Agent 提交优化计划，按计划逐项执行和评估
- **自动回滚**：评测失败或无改进时自动回滚到最优版本
- **深度优化**：适合需要多轮迭代才能达到最优的场景

支持所有 DSL：triton_cuda, triton_ascend, torch, cuda_c, cpp, ascendc, swft, tilelang_cuda, tilelang_npuir

适用场景：
- 已有初始 kernel，需要深度性能优化
- 优化空间复杂，需要多次试错
- 对单设备长时间运行可接受
"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Scale workflow_timeout before task.py reads it for asyncio.wait_for.
        max_rounds = self.config.get("max_step", 20)
        eval_timeout = self.config.get("eval_timeout", 120)
        self.config["workflow_timeout"] = max(
            self.config.get("workflow_timeout", 1800),
            max_rounds * (eval_timeout + 60) + 300,
        )

    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {
                "type": "string",
                "description": "算子名称",
            },
            "task_desc": {
                "type": "string",
                "description": "框架代码（Model/get_inputs）",
            },
            "previous_code": {
                "type": "string",
                "description": "初始 kernel 代码（可选）",
            },
            "max_rounds": {
                "type": "integer",
                "description": "最大评测轮数（默认 20）",
            },
            "dsl": {
                "type": "string",
                "description": "DSL 类型",
            },
            "framework": {
                "type": "string",
                "description": "框架，如 'torch'",
            },
            "backend": {
                "type": "string",
                "description": "后端，如 'cuda', 'ascend'",
            },
            "arch": {
                "type": "string",
                "description": "架构，如 'a100', 'ascend910b4'",
            },
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"],
    }

    def build_graph(self) -> StateGraph:
        """构建 autoresearch 工作流图（单节点封装）"""
        workflow = StateGraph(KernelGenState)

        _self = self

        async def autoresearch_node(state: KernelGenState) -> dict:
            """Autoresearch 主节点: seed → scaffold → AgentLoop → result"""
            from akg_agents.op.autoresearch.agent.loop import AgentLoop
            from akg_agents.op.autoresearch.adapters.llm_adapter import AkgLLMAdapter
            from akg_agents.op.autoresearch.adapters.task_scaffolder import scaffold_task_dir
            from akg_agents.op.autoresearch.framework.config import EvalResult
            from akg_agents.op.verifier.kernel_verifier import KernelVerifier
            from akg_agents.core.worker.manager import (
                get_worker_manager, register_local_worker,
            )
            from akg_agents.core_v2.llm.factory import create_llm_client

            op_name = state.get("op_name", "")
            task_desc = state.get("task_desc", "")
            dsl = state.get("dsl", "")
            framework = state.get("framework", "")
            backend = state.get("backend", "")
            arch = state.get("arch", "")
            # max_rounds: ToolExecutor path sets state["max_rounds"] via build_initial_state;
            # LangGraphTask direct path does NOT — fall back to config["max_step"].
            max_rounds = state.get("max_rounds") or _self.config.get("max_step", 20)

            logger.info(
                f"[AutoresearchWorkflow] Starting: op_name={op_name}, "
                f"dsl={dsl}, backend={backend}, max_rounds={max_rounds}"
            )

            # ---- 1. Worker + KernelVerifier setup ----
            # Created early so the seed stabilization loop can use verifier
            # for runtime validation (not just static CodeChecker).
            wm = get_worker_manager()
            if not await wm.has_worker(backend=backend, arch=arch):
                await register_local_worker(
                    [0], backend=backend, arch=arch,
                )
                logger.info(
                    f"[AutoresearchWorkflow] Registered local worker: "
                    f"backend={backend}, arch={arch}"
                )

            _raw_tid = str(state.get("task_id", ""))
            task_id = (
                _raw_tid
                if (_raw_tid and _raw_tid != "0")
                else f"ar_{uuid.uuid4().hex[:8]}"
            )

            workflow_config = _self.config
            verifier = KernelVerifier(
                op_name=op_name,
                task_id=task_id,
                framework_code=task_desc,
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                config=workflow_config,
                worker=None,
            )

            # ---- 2. Preflight: validate reference + resolve seed ----
            # Guarantee: AgentLoop only starts when BOTH reference and seed
            # have passed runtime verification via KernelVerifier.
            #
            # Steps:
            #   a) Acquire a worker (mandatory — fail closed if unavailable)
            #   b) Validate reference (task_desc) via check_task_desc_static +
            #      check_task_desc_runtime
            #   c) Resolve seed: use previous_code if provided and passes
            #      runtime verify; otherwise generate via KernelGen with
            #      CodeChecker + runtime verify loop
            #   d) Release preflight worker

            # a) Acquire worker — fail closed
            _pf_worker = _self.private_worker
            _pf_borrowed = False
            if not _pf_worker and _self.worker_manager:
                _pf_worker = await _self.worker_manager.select(
                    backend=backend, arch=arch,
                )
                _pf_borrowed = True
            if not _pf_worker:
                return {
                    "verifier_result": False,
                    "verifier_error": (
                        "No worker available — cannot verify reference and "
                        "seed before starting optimization. Register a worker "
                        "or check backend/arch configuration."
                    ),
                }

            try:
                # b) Validate reference
                ref_ok, ref_err = verifier.check_task_desc_static(task_desc)
                if not ref_ok:
                    return {
                        "verifier_result": False,
                        "verifier_error": f"Reference static check failed: {ref_err}",
                    }

                verifier.worker = _pf_worker
                ref_ok, ref_err = await verifier.check_task_desc_runtime(
                    task_desc, timeout=300)
                verifier.worker = None
                if not ref_ok:
                    return {
                        "verifier_result": False,
                        "verifier_error": (
                            f"Reference runtime check failed: {ref_err}. "
                            f"The torch reference (task_desc) must execute "
                            f"successfully before optimization can start."
                        ),
                    }
                logger.info("[AutoresearchWorkflow] Reference validated OK")

                # Runtime verify helper for seed candidates.
                async def _verify_seed(code: str) -> tuple[bool, str, str]:
                    """Returns (ok, log, verified_code)."""
                    task_info = {"coder_code": code}
                    verifier.worker = _pf_worker
                    try:
                        ok, log = await verifier.run(task_info, current_step=0)
                    except Exception as e:
                        ok, log = False, str(e)
                    finally:
                        verifier.worker = None
                    return ok, (log or ""), task_info.get("coder_code", code)

                # c) Resolve seed
                seed = state.get("previous_code", "")

                if seed:
                    logger.info("[AutoresearchWorkflow] Verifying provided kernel …")
                    ok, log, verified = await _verify_seed(seed)
                    if ok:
                        seed = verified
                        logger.info("[AutoresearchWorkflow] Provided kernel OK")
                    else:
                        logger.warning(
                            f"[AutoresearchWorkflow] Provided kernel failed "
                            f"({log[:200]}), falling back to KernelGen"
                        )
                        seed = ""

                if not seed:
                    kernel_gen = _self.agents.get("kernel_gen")
                    if kernel_gen:
                        session_id = str(state.get("session_id") or "").strip()
                        if session_id:
                            kernel_gen.context["session_id"] = session_id

                        from akg_agents.op.utils.code_checker import CodeChecker
                        checker = CodeChecker(
                            backend=backend, dsl=dsl, config=_self.config,
                        )

                        max_seed_retries = workflow_config.get(
                            "gen_retries", 5)
                        code_check_errors = ""
                        for attempt in range(max_seed_retries + 1):
                            logger.info(
                                f"[AutoresearchWorkflow] Seed generation "
                                f"attempt {attempt + 1}/{max_seed_retries + 1}"
                            )
                            try:
                                seed_code, _, _ = await kernel_gen.run(
                                    op_name=op_name,
                                    task_desc=task_desc,
                                    dsl=dsl,
                                    framework=framework,
                                    backend=backend,
                                    arch=arch,
                                    previous_code=state.get("previous_code", ""),
                                    code_check_errors=code_check_errors,
                                    model_level=_self.config.get(
                                        "agent_model_config", {},
                                    ).get("coder", "standard"),
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[AutoresearchWorkflow] KernelGen failed: {e}"
                                )
                                seed_code = None
                                break

                            if not seed_code:
                                logger.warning(
                                    "[AutoresearchWorkflow] KernelGen returned empty code"
                                )
                                break

                            passed, error_msg, _ = await checker.check(seed_code)
                            if not passed:
                                logger.warning(
                                    f"[AutoresearchWorkflow] Static check failed "
                                    f"(attempt {attempt + 1}): {error_msg[:200]}"
                                )
                                code_check_errors = error_msg
                                continue

                            ok, log, verified_code = await _verify_seed(seed_code)
                            if ok:
                                seed = verified_code
                                logger.info(
                                    "[AutoresearchWorkflow] Seed verified OK"
                                )
                                break
                            logger.warning(
                                f"[AutoresearchWorkflow] Runtime verify failed "
                                f"(attempt {attempt + 1}): {log[:200]}"
                            )
                            code_check_errors = (
                                f"Runtime verification failed: {log[:500]}"
                            )

            finally:
                verifier.worker = None
                if _pf_borrowed and _self.worker_manager:
                    await _self.worker_manager.release(_pf_worker)

            if not seed:
                return {
                    "verifier_result": False,
                    "verifier_error": (
                        "Seed kernel generation failed after all retries. "
                        "Check that the reference (task_desc) is correct — "
                        "KernelGen derives the initial kernel from it."
                    ),
                }

            _profile_settings = workflow_config.get("profile_settings", {})
            _private_worker = _self.private_worker
            _worker_manager = _self.worker_manager

            # Base profile cache: reference implementation doesn't change across
            # rounds, so we run base profile once and reuse the result.
            # Uses KernelVerifier's built-in use_reference_data + override_base_time_us.
            _base_time_cache = {"value": None}

            # ---- 3. Construct eval_fn (per-eval worker borrow) ----
            async def eval_fn(task_dir, config, round_num=0):
                """Evaluate kernel via KernelVerifier: verify + profile.

                Called once per round.  Profiler's run_times controls
                internal repetition.

                Base profile (reference impl) is cached after the first
                successful profile and reused via use_reference_data +
                override_base_time_us.
                """
                worker = None
                borrowed = False
                if _private_worker:
                    worker = _private_worker
                elif _worker_manager:
                    worker = await _worker_manager.select(
                        backend=backend, arch=arch,
                    )
                    borrowed = True

                if not worker:
                    return EvalResult(
                        correctness=False, error="no available worker",
                    )

                try:
                    verifier.worker = worker

                    editable = config.editable_files
                    main_file = os.path.join(task_dir, editable[0])
                    with open(main_file, "r", encoding="utf-8") as f:
                        code = f.read()
                    task_info = {"coder_code": code}

                    # Verify
                    verifier.config.pop("use_reference_data", None)
                    try:
                        success, log = await verifier.run(
                            task_info, current_step=round_num,
                        )
                    except Exception as e:
                        return EvalResult(
                            correctness=False,
                            error=f"verifier exception: {e}",
                        )

                    if not success:
                        reason = (log or "verification failed")[:500]
                        return EvalResult(
                            correctness=False,
                            error=reason,
                            raw_output=log or "",
                        )

                    # Autotune writeback
                    if task_info.get("coder_code", "") != code:
                        with open(main_file, "w", encoding="utf-8") as f:
                            f.write(task_info["coder_code"])

                    # Profile: skip base if already cached
                    cur_profile_settings = _profile_settings
                    if _base_time_cache["value"] is not None:
                        cur_profile_settings = {
                            **_profile_settings,
                            "override_base_time_us": _base_time_cache["value"],
                        }
                        verifier.config["use_reference_data"] = True
                    else:
                        verifier.config.pop("use_reference_data", None)

                    try:
                        profile = await verifier.run_profile(
                            task_info,
                            current_step=round_num,
                            profile_settings=cur_profile_settings,
                        )
                    except Exception as e:
                        return EvalResult(
                            correctness=False,
                            error=f"profile exception: {e}",
                        )

                    if profile.get("gen_time") is None:
                        return EvalResult(
                            correctness=False,
                            error="profile failed: gen_time is None",
                        )

                    # Cache base_time on first successful profile.
                    # Reject inf — it means profiling failed, and we should
                    # retry on the next round instead of locking in a bad value.
                    base_time_val = profile.get("base_time")
                    if (_base_time_cache["value"] is None
                            and base_time_val is not None
                            and base_time_val < float('inf')):
                        _base_time_cache["value"] = base_time_val

                    ref_latency = (
                        base_time_val
                        if base_time_val is not None and base_time_val < float('inf')
                        else _base_time_cache["value"] or 0
                    )
                    return EvalResult(
                        correctness=True,
                        metrics={
                            "latency_us": profile["gen_time"],
                            "ref_latency_us": ref_latency,
                            "speedup_vs_ref": profile.get("speedup", 0),
                        },
                    )
                finally:
                    verifier.worker = None
                    if borrowed and _worker_manager:
                        await _worker_manager.release(worker)

            # ---- 4. Knowledge assembly ----
            program_md, context_files, extra_files = await _assemble_knowledge(
                dsl=dsl,
                framework=framework,
                backend=backend,
                arch=arch,
                op_name=op_name,
                task_desc=task_desc,
                worker_manager=_worker_manager,
            )

            # ---- 5. Scaffold task_dir ----
            log_dir = workflow_config.get("log_dir", "/tmp/autoresearch")
            task_dir = scaffold_task_dir(
                base_dir=log_dir,
                op_name=op_name,
                task_desc=task_desc,
                editable_files={"kernel.py": seed},
                program_md=program_md,
                context_files=context_files,
                extra_files=extra_files,
                max_rounds=max_rounds,
                dsl=dsl,
            )
            logger.info(f"[AutoresearchWorkflow] task_dir: {task_dir}")
            _self._autoresearch_task_dir = task_dir

            # ---- 6. Create LLM adapter ----
            session_id = str(state.get("session_id") or "").strip()
            llm_client = create_llm_client(
                model_level=workflow_config.get(
                    "agent_model_config", {},
                ).get("coder", "standard"),
                session_id=session_id or None,
            )
            adapter = AkgLLMAdapter(llm_client)

            # ---- 7. Run AgentLoop ----
            # Suppress verbose verifier/worker INFO logs during agent loop.
            # AgentLoop already prints structured eval summaries.
            _quiet_loggers = [
                "akg_agents.op.verifier",
                "akg_agents.core.worker",
                "akg_agents.utils.process_utils",
            ]
            _saved_levels = {}
            for _name in _quiet_loggers:
                _lg = logging.getLogger(_name)
                _saved_levels[_name] = _lg.level
                _lg.setLevel(logging.WARNING)

            loop_result = None
            try:
                loop_result = await AgentLoop(
                    task_dir,
                    llm_adapter=adapter,
                    eval_fn=eval_fn,
                    skip_branch_switch=True,
                    max_rounds=max_rounds,
                ).run()
            except Exception as e:
                logger.error(f"[AutoresearchWorkflow] AgentLoop failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                for _name, _lvl in _saved_levels.items():
                    logging.getLogger(_name).setLevel(_lvl)
                # Runs on ANY exit — normal, exception, or CancelledError
                # from workflow_timeout. Generates report and salvages the
                # best result so progress is never silently lost.
                if loop_result is None:
                    loop_result = _self.finalize_on_timeout()

            # ---- 8. Extract result ----
            best_metrics = loop_result.get("best_metrics")
            has_valid_result = best_metrics is not None

            final_code_path = os.path.join(task_dir, "kernel.py")
            final_code = ""
            if os.path.exists(final_code_path):
                with open(final_code_path, "r", encoding="utf-8") as f:
                    final_code = f.read()

            profile_res = {}
            if best_metrics:
                profile_res = {
                    "gen_time": best_metrics.get("latency_us"),
                    "base_time": best_metrics.get("ref_latency_us"),
                    "speedup": best_metrics.get("speedup_vs_ref", 0),
                }

            return {
                "coder_code": final_code,
                "verifier_result": has_valid_result,
                "profile_res": profile_res,
                "verifier_error": (
                    ""
                    if has_valid_result
                    else loop_result.get("final_status", "no valid result")
                ),
            }

        workflow.add_node("autoresearch", autoresearch_node)
        workflow.set_entry_point("autoresearch")
        workflow.add_edge("autoresearch", END)

        return workflow

    @classmethod
    def build_initial_state(
        cls,
        arguments: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Override: add max_rounds to state."""
        state = super().build_initial_state(arguments, agent_context)
        state["max_rounds"] = arguments.get("max_rounds", 20)
        return state

    @classmethod
    def prepare_config(
        cls,
        workflow_resources: Dict[str, Any],
        arguments: Dict[str, Any],
    ):
        """Ensure workflow has complete LangGraph config.

        Deep-copies config to prevent nested dict leakage via
        KernelAgent's cached workflow_resources.
        """
        base_config = copy.deepcopy(
            workflow_resources.get("config") or {},
        )

        dsl = arguments.get("dsl", "")
        backend = arguments.get("backend", "")
        op_name = arguments.get("op_name", "")

        full_config = cls.build_langgraph_task_config(
            dsl=dsl,
            backend=backend,
            op_name=op_name,
            base_config=base_config,
        )
        workflow_resources["config"] = full_config

        super().prepare_config(workflow_resources, arguments)

    def finalize_on_timeout(self) -> Dict[str, Any]:
        """Salvage after workflow_timeout kills the agent loop.

        Generates a report from round history on disk and returns a
        loop_result-compatible dict (same shape as AgentLoop.run()),
        so the normal extract-result path in the workflow node can
        handle it uniformly.
        """
        task_dir = getattr(self, "_autoresearch_task_dir", None)
        if not task_dir or not os.path.exists(task_dir):
            return {"best_metrics": None, "final_status": "timeout before task_dir was created"}

        print(
            "\n[AutoresearchWorkflow] Workflow time limit reached. "
            "Saving best result and generating report.",
            flush=True,
        )

        from akg_agents.op.autoresearch.framework.runner import load_task_config
        from akg_agents.op.autoresearch.framework.logger import RoundLogger
        from akg_agents.op.autoresearch.framework.report import generate_report

        try:
            config = load_task_config(task_dir)
        except Exception:
            return {"best_metrics": None, "final_status": "timeout — failed to load task config"}

        try:
            generate_report(task_dir, config)
        except Exception as e:
            logger.warning(f"[AutoresearchWorkflow] Report generation failed: {e}")

        best = RoundLogger(task_dir, config).get_best()
        return {
            "best_metrics": dict(best["metrics"]) if best else None,
            "final_status": "timed out — best result preserved" if best else "timed out — no valid result",
        }

    def format_result(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Format result for ToolExecutor (matches evolve/adaptive_search)."""
        code = final_state.get("coder_code", "")
        profile_res = final_state.get("profile_res", {})
        verifier_result = final_state.get("verifier_result", False)

        status = "success" if verifier_result else "fail"
        res = {
            "code": code,
            "profile": str(profile_res) if profile_res else "",
            "status": status,
        }

        if not verifier_result:
            verifier_error = final_state.get("verifier_error", "")
            if verifier_error:
                res["error_information"] = verifier_error

        return res


# ---------------------------------------------------------------------------
# Knowledge assembly helpers
# ---------------------------------------------------------------------------

async def _assemble_knowledge(
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    op_name: str,
    task_desc: str,
    worker_manager=None,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Assemble DSL knowledge for autoresearch.

    Returns:
        program_md: Agent instructions + core reference (enters system prompt).
        context_files: Listed in task.yaml (enters system prompt, must be compact).
        extra_files: Written to task_dir/docs/ (agent reads via read_file tool).
    """
    try:
        from akg_agents.core_v2.skill import SkillLoader
        from akg_agents.op.skill.operator_selector import (
            OperatorSkillSelector, OperatorSelectionContext,
        )
    except ImportError:
        SkillLoader = None
        OperatorSkillSelector = None

    try:
        from akg_agents.utils.hardware_utils import get_hardware_doc
    except ImportError:
        get_hardware_doc = None

    try:
        from akg_agents.op.utils.triton_ascend_api_docs import (
            resolve_triton_ascend_api_docs,
        )
    except ImportError:
        resolve_triton_ascend_api_docs = None

    try:
        from akg_agents import get_project_root
        project_root = Path(get_project_root())
    except ImportError:
        project_root = None

    # Initialize output containers
    extra_files: dict[str, str] = {}
    context_files: dict[str, str] = {}

    # Try skill system first
    all_skills = []
    if SkillLoader and OperatorSkillSelector and project_root:
        SKILLS_DIR = project_root / "op" / "resources" / "skills"
        dsl_key = dsl.lower().replace("_", "-")

        loader = SkillLoader()
        selector = OperatorSkillSelector()
        all_skills = loader.load_from_directory(SKILLS_DIR / dsl_key)

        if all_skills:
            ctx = OperatorSelectionContext(
                backend=backend, dsl=dsl, framework=framework, hardware=arch,
            )
            all_skills = selector.coarse_filter(all_skills, ctx)

    # Fallback: no skill package → use legacy _DSL_DOCS_DIR_MAP
    fallback_docs_text = ""
    if not all_skills and project_root:
        docs_entry = _DSL_DOCS_DIR_MAP.get(dsl, {}).get("coder")
        if docs_entry:
            docs_path = project_root / docs_entry
            if docs_path.is_dir():
                for f in sorted(docs_path.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(docs_path).as_posix()
                        try:
                            content = f.read_text(errors="ignore")
                        except Exception:
                            continue
                        extra_files[f"docs/{rel}"] = content
                fallback_docs_text = f"[Using legacy docs from {docs_entry}]"

    # Layer 0: fundamental + reference skills → program.md
    core_skills = [
        s for s in all_skills if s.category in {"fundamental", "reference"}
    ]
    core_text = (
        _format_skills(core_skills) if core_skills else fallback_docs_text
    )
    program_md = _build_program_md(
        dsl, framework, backend, arch, op_name, core_text,
    )

    # context_files (compact, enters system prompt)
    if get_hardware_doc:
        hw_docs = get_hardware_doc(backend, arch)
        if hw_docs:
            context_files["hardware_info.md"] = hw_docs

    # Layer 1: large docs + guide/example → docs/ (not in system prompt)
    if dsl.lower() == "triton_ascend" and resolve_triton_ascend_api_docs:
        try:
            api_docs = await resolve_triton_ascend_api_docs(
                backend=backend, arch=arch, worker_manager=worker_manager,
            )
            if api_docs:
                extra_files["docs/api.md"] = api_docs
        except Exception as e:
            logger.warning(f"Failed to resolve triton_ascend API docs: {e}")

    # Guide/example skills → agent reads via read_file
    guide_skills = [s for s in all_skills if s.category == "guide"]
    example_skills = [s for s in all_skills if s.category == "example"]
    for skill in guide_skills:
        extra_files[f"docs/guide_{skill.name}.md"] = skill.content
    for skill in example_skills:
        extra_files[f"docs/example_{skill.name}.md"] = skill.content

    # Generate index file
    if extra_files:
        index_lines = ["# Available DSL Documentation\n"]
        index_lines.append(
            "Use `read_file` to access these documents as needed.\n",
        )
        for path, content in extra_files.items():
            index_lines.append(f"- `{path}` ({len(content)} chars)")
        extra_files["docs/README.md"] = "\n".join(index_lines)

    return program_md, context_files, extra_files


def _format_skills(skills) -> str:
    """Format skill objects into a single text block."""
    parts = []
    for s in skills:
        parts.append(f"### {s.name}\n{s.content}")
    return "\n\n".join(parts)


def _build_program_md(
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    op_name: str,
    core_reference: str,
) -> str:
    """Build the program.md agent instructions."""
    lines = [
        f"# Optimization Task: {op_name}",
        "",
        f"DSL: {dsl} | Framework: {framework} | Backend: {backend} | Arch: {arch}",
        "",
        "## Objective",
        "Optimize the kernel implementation for maximum performance while maintaining correctness.",
        "The evaluation measures `latency_us` (lower is better).",
        "",
        "## Rules",
        "- Edit only the editable files listed in task.yaml",
        "- Each edit is automatically evaluated: correct + faster → KEEP, otherwise → DISCARD",
        "- Plan your optimizations as a structured list using update_plan()",
        "- After each plan item, submit the edit for evaluation",
        "",
    ]

    if core_reference:
        lines.extend([
            "## DSL Reference",
            core_reference,
            "",
        ])

    lines.extend([
        "## Available Reference Documents",
        "The `docs/` directory contains DSL guides, examples, and API documentation.",
        "Use `read_file` to access them when you need specific optimization techniques.",
        "Start with `docs/README.md` for an overview.",
    ])

    return "\n".join(lines)
