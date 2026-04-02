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
    """连续失败上限 (quick_check 或 eval 失败都计数, KEEP 时重置)."""
    max_no_edit_turns: int = 3
    """连续无编辑轮数上限 — 触发强制编辑提示."""
    max_turns_multiplier: int = 8
    """安全上限倍数. 最大 LLM 调用次数 = max_rounds * max_turns_multiplier."""

    # -- Prompt 截断 ----------------------------------------------------------
    chars_per_token: int = 4
    editable_file_truncate: int = 8_000
    system_context_file_truncate: int = 15_000
    system_context_total_truncate: int = 40_000
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
    """主 LLM 调用的 max_tokens."""
    thinking_budget: int = 8000
    """Anthropic extended thinking budget (tokens). 0 = disabled.
    When enabled, max_tokens is auto-raised to thinking_budget + llm_max_tokens."""
    call_timeout: float = 120.0
    """API 调用超时 (秒)."""
    retry_initial_backoff: float = 5.0
    """重试初始退避 (秒)."""
    retry_max_backoff_rate_limit: float = 120.0
    """RateLimitError 最大退避 (秒)."""
    retry_max_backoff_other: float = 60.0
    """其他可重试错误最大退避 (秒)."""

    # -- 上下文压缩 --------------------------------------------------------
    context_limit: int | None = None
    """模型 context window token 数. None = 禁用自动压缩."""
    compression_threshold: float = 0.75
    """估算 token 数超过 context_limit × 此值时触发 auto_compact."""
    microcompact_min_chars: int = 200
    """microcompact 清理阈值: 低于此长度的 tool_result 不清理."""
    microcompact_keep_recent: int = 3
    """microcompact 保留最近 N 条 tool_result 完整内容."""
    auto_compact_text_limit: int = 80_000
    """auto_compact 传给 LLM 摘要的最大字符数."""
    compact_summary_max_tokens: int = 2_000
    """auto_compact 摘要 LLM 调用的 max_tokens."""
    compact_min_messages: int = 4
    """compact 工具: 消息数低于此值时不执行压缩."""

    # -- 诊断子 Agent ------------------------------------------------------
    diagnose_suggest_threshold: int = 3
    """连续失败多少次后建议调用 diagnose."""
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
    # 三个字段同时指定时, 框架自动生成评测脚本
    dsl: Optional[str] = None           # triton_cuda, triton_ascend, pytorch, cuda_c, cpp
    framework: Optional[str] = None     # torch, mindspore, numpy
    backend: Optional[str] = None       # cuda, ascend, cpu

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
    max_patch_size: int = 15000
    forbidden_patterns: dict = field(default_factory=lambda: {
        "content": [],
        "diff": ["^\\s*#"],
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