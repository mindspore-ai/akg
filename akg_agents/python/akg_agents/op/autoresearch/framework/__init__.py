"""
Autoresearch Framework — 自主迭代优化框架

核心抽象：
  - Frozen eval harness: 评测代码不可修改，防止 agent 作弊
  - Limited edit scope: agent 只能编辑指定文件
  - Experiment loop: propose → edit → test → keep/discard
  - Structured logging: 每轮结果记录为 JSONL + Markdown
  - Git rollback: 失败时自动回滚
  - Adapter-based eval: 多 DSL / 多框架 / 多后端的评测脚本自动生成
"""

from .config import TaskConfig, EvalResult, RoundRecord
from .evaluator import run_eval, is_improvement
from .logger import RoundLogger
