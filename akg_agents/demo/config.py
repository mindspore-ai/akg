"""
配置管理 - 复用 akg_agents 的 LLM 客户端，同时支持独立配置
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# 将 akg_agents 加入 sys.path 以便复用
AKG_AGENTS_ROOT = Path(__file__).resolve().parent.parent / "python"
if str(AKG_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(AKG_AGENTS_ROOT))

# demo 项目根目录
DEMO_ROOT = Path(__file__).resolve().parent
# 默认输出目录
OUTPUT_DIR = DEMO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
# 日志目录
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
# 工作区目录 - 存储提取的代码片段，供最终拼凑使用
WORKSPACE_DIR = OUTPUT_DIR / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)

# ReAct 循环配置
MAX_REACT_STEPS = 50
MAX_RETRIES_PER_STEP = 3

# 代码执行配置
CODE_EXEC_TIMEOUT = 60  # 秒
PYTHON_EXECUTABLE = sys.executable

# 文件读取配置
READ_FILE_MAX_LINES = 300  # read_file 单次最大返回行数


def get_llm_client(model_level: str = None):
    """复用 akg_agents 的 LLM 客户端工厂"""
    try:
        from akg_agents.core_v2.llm.factory import create_llm_client
        return create_llm_client(model_level=model_level)
    except Exception as e:
        raise RuntimeError(
            f"无法创建 LLM 客户端: {e}\n"
            f"请确保 akg_agents 配置正确 (settings.json 或环境变量)"
        )


def get_run_id() -> str:
    """生成本次运行的唯一ID（用于日志目录）"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
