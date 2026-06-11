"""
核心数据结构定义 — 与具体任务类型无关
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """
    Agent 行为参数 — 框架内部默认值, 任务作者一般不需要碰.

    这些参数控制 AgentLoop 的循环策略、prompt 组装、日志截断等行为.
    与"任务是什么"无关, 只跟"agent 怎么跑"有关.

    如果某个任务确实需要覆盖 (比如复杂任务需要更高的失败容忍度),
    在 task.yaml 的 agent.config 块中设置:
        agent:
          config:
            max_consecutive_failures: 20
    """

    # -- 实验控制 ------------------------------------------------------------
    max_consecutive_failures: int = 10
    max_no_edit_turns: int = 3
    max_turns_multiplier: int = 8

    # -- Prompt 截断 ----------------------------------------------------------
    chars_per_token: int = 3
    editable_file_truncate: int = 8_000
    system_context_file_truncate: int = 15_000
    system_context_total_truncate: int = 40_000
    system_fundamentals_max_chars: int = 20_000
    plan_max_chars: int = 4_000
    finish_hint_threshold: int = 2

    # -- 日志截断 ------------------------------------------------------------
    log_arg_truncate: int = 500
    log_result_truncate: int = 1_000
    cumulative_diff_truncate: int = 10_000
    smoke_output_limit: int = 2_000

    # -- 工具参数 ----------------------------------------------------------
    raw_output_tail: int = 2_048
    """eval 结果中 raw_output 尾部截断长度."""

    # -- LLM 调用 ----------------------------------------------------------
    llm_max_tokens: int = 8_192
    thinking_budget: int = 8_000
    """Anthropic extended thinking budget (tokens). 0 = disabled.
    When enabled, max_tokens is auto-raised to thinking_budget + llm_max_tokens."""
    call_timeout: float = 120.0
    retry_initial_backoff: float = 5.0
    retry_max_backoff_rate_limit: float = 120.0
    retry_max_backoff_other: float = 60.0
    llm_max_retries: int = 5
    llm_connection_check_timeout: float = 15.0

    # -- 上下文压缩 --------------------------------------------------------
    context_limit: int | None = 150_000
    """模型 context window token 数. 150K 对大多数模型安全."""
    compression_threshold: float = 0.75
    """估算 token 数超过 context_limit × 此值时触发 auto_compact."""
    microcompact_min_chars: int = 200
    """microcompact 清理阈值: 低于此长度的 tool_result 不清理."""
    microcompact_keep_recent: int = 1
    """microcompact 保留最近 N 条 tool_result 完整内容."""
    compact_min_messages: int = 4
    """compact 工具: 消息数低于此值时不执行压缩."""
    compact_max_retries: int = 3
    """每个 compact LLM 调用的 PTL 重试次数."""
    compact_diagnosis_truncate: int = 2_000
    """bootstrap 中 last_diagnosis (must_replan 警告) 截断字符数."""
    compact_post_check_ratio: float = 0.9
    """post-compact 二次兜底 = context_limit × 此值."""
    compact_max_failures: int = 3
    """PTL circuit breaker 次数上限."""
    compact_emergency_keep_rounds: int = 1
    """PTL level 1 auto_compact 保留最近 round 数."""
    compact_keep_recent_rounds: int = 3
    """auto_compact 正常路径保留最近 N 个 round."""
    # --- Multi-step compact pipeline (operator summary + plan analysis) ---
    compact_op_summary_max_tokens: int = 500
    """LLM call #1 (operator summary from historical keywords) 的 max_tokens."""
    compact_plan_analysis_max_tokens: int = 1_500
    """LLM call #2 (structured plan.md analysis) 的 max_tokens."""
    compact_kernel_sanity_cap: int = 80_000
    """Normal auto_compact path: per-editable-file char cap. Dominated
    by the LLM context window; tight cap isn't the goal here, safety is."""
    compact_rebuild_kernel_cap: int = 20_000
    """PTL-recovery force_rebuild path: much tighter per-file cap. Ensures
    the rebuilt buffer is strictly smaller than the one that just tripped
    PTL. Size independent from compact_kernel_sanity_cap because the
    recovery path has to shrink; the normal path doesn't."""
    compact_rebuild_ranking_cap: int = 8_000
    """PTL-recovery force_rebuild path: char cap for ranking.md. Normal
    auto_compact keeps ranking.md uncapped (full performance landscape);
    force_rebuild trims it because the whole point of force_rebuild is
    to shed bytes."""
    """单个 editable file 在 state attachment 中的硬性字符上限 (兜底, 正常
    kernel 远小于此值). 超过时截断并打 WARNING 日志."""
    compact_plan_raw_fallback_chars: int = 6_000
    """当 LLM plan analysis 调用失败时, raw plan.md fallback 的截断字符数."""
    replanning_max_idle_turns: int = 2
    """replanning 阶段连续无工具调用超过此值则退出."""

    # -- 反馈/排名截断 -------------------------------------------------------
    eval_feedback_tail: int = 1_000
    """eval feedback 消息中 raw_output 尾部截断."""
    log_raw_output_truncate: int = 4_096
    """jsonl 日志中 raw_output 截断."""
    history_summary_last_n: int = 10
    """get_history_summary 返回最近 N 轮."""
    ranking_description_truncate: int = 100
    """ranking 中 description 截断字符数."""
    ranking_error_truncate: int = 120
    """ranking 中 error 截断字符数."""
    compact_ranking_max_entries: int = 5
    """压缩后 ranking 中 correct/failed 各自最多保留条目数."""

    # -- Skill 注入 ---------------------------------------------------------
    skill_block_max_chars: int = 8_000
    """主 Agent 初始 prompt 中完整 skill block 的字符预算."""
    skill_block_top_k: int = 5
    skill_keyword_max_per_item: int = 5
    """`update_plan.items[].keywords` 鍗曚釜 item 鏈€澶氫繚鐣欑殑鍏抽敭璇嶆暟."""
    """主 Agent 初始 prompt 中完整展开的 top-k skill 数量."""
    skill_narrow_timeout: float = 30.0
    """LLM 关键词生成的硬超时 (秒)."""

    # -- Plan item rationale (forced reflection) ----------------------------
    plan_item_rationale_min_chars: int = 30
    """Minimum length for the rationale field on each update_plan item.
    Plans containing items with shorter rationale are rejected whole."""
    plan_item_rationale_max_chars: int = 400
    """Maximum length; longer rationale is truncated with ellipsis."""

    # -- Plan breadth (forced diversification) ------------------------------
    min_items_per_plan: int = 3
    """Minimum number of items a fresh `update_plan` call must contain.
    Forces the agent to articulate several distinct directions up front,
    so settled_history builds pattern signal inside a single plan version
    (instead of each plan being a 1-item reactive tweak of the previous
    outcome). Only applies to fresh submissions; replace_active_item
    (the must_replan surgical path) ignores this floor."""

    # -- Skill content injection (on skill-backed item activation) -----------
    skill_inject_max_chars: int = 6_000
    """Max chars of SKILL.md content injected into the conversation when
    a skill-backed plan item activates. No LLM call — raw content dump."""

    # -- Diagnose ----------------------------------------------------------
    diagnose_suggest_threshold: int = 3
    """连续失败多少次后触发 diagnose 子 Agent."""
    subagent_code_truncate: int = 8_000
    """诊断 subagent 中代码内容截断长度."""
    subagent_result_truncate: int = 10_000
    """诊断 subagent read_file 结果截断长度."""
    subagent_max_iterations: int = 15
    """诊断 subagent 最大迭代次数."""

    # -- 运行时文件 ----------------------------------------------------------
    session_dir: str = "agent_session"
    heartbeat_file: str = "RUNNING"


@dataclass
class TaskConfig:
    """
    一个优化任务的完整配置.

    通过 task.yaml 声明式定义, 与框架代码解耦.

    框架只关心: 哪些文件可编辑, 怎么评测, 什么指标, 怎么判定好坏.
    """

    # 基本信息
    name: str
    description: str

    # ---- 适配器声明 (adapter-based eval) ----
    # dsl/framework/backend/arch 同时指定时, 框架自动生成评测脚本
    dsl: Optional[str] = None
    framework: Optional[str] = None
    backend: Optional[str] = None
    arch: Optional[str] = None
    dsl_config: dict = field(default_factory=dict)

    # 自定义评测脚本 (覆盖 adapter 生成)
    eval_script: Optional[str] = None
    editable_files: list[str] = field(default_factory=list)

    # 评测参数
    eval_timeout: int = 600
    primary_metric: str = "score"
    lower_is_better: bool = True
    improvement_threshold: float = 0.0

    # Hard constraints: {metric_name: (operator, threshold)}
    constraints: dict = field(default_factory=dict)

    # Preflight smoke test
    smoke_test_script: Optional[str] = None
    smoke_test_timeout: int = 10
    import_timeout: int = 15

    # 编辑护栏
    max_patch_size: int = 15_000
    # forbidden_patterns is built by config_loader.build_forbidden_patterns
    # from edit_guardrails.yaml (global + dsl + hardware + framework scopes)
    # merged with any task.yaml override. The default here is empty because
    # the YAML file is the single source of truth for defaults.
    forbidden_patterns: dict = field(default_factory=lambda: {
        "content": [],
        "diff": [],
        "diff_any": [],
    })

    # Agent 上下文文件
    program_file: Optional[str] = None
    ref_file: Optional[str] = None
    context_files: list[str] = field(default_factory=list)

    # Git 控制
    git_push: bool = False
    git_branch: Optional[str] = None

    # 实验控制
    max_rounds: int = 30

    # Agent 行为 — 框架默认值, 绝大多数任务不需要覆盖
    agent: AgentConfig = field(default_factory=AgentConfig)

    # 任务特定元数据
    metadata: dict = field(default_factory=dict)


@dataclass
class CommitResult:
    """git_commit 的结构化返回值 — 消灭 Optional[str] 的歧义."""

    hash: Optional[str] = None
    nothing_to_commit: bool = False
    error: Optional[str] = None

    @property
    def committed(self) -> bool:
        return self.hash is not None

    @property
    def ok(self) -> bool:
        """True if committed or nothing to commit (not an error)."""
        return self.committed or self.nothing_to_commit


@dataclass
class EvalResult:
    """单次评测的结果 — 通用, 不预设具体指标"""

    correctness: bool
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    raw_output: str = ""

    def get_metric(self, key: str, default=None):
        return self.metrics.get(key, default)


@dataclass
class RoundRecord:
    """一轮实验的完整记录"""

    round_num: int
    description: str
    result: EvalResult
    accepted: bool
    commit_hash: Optional[str] = None
    duration_sec: float = 0.0
    constraint_violations: list[str] = field(default_factory=list)
