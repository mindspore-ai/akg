# Copyright 2025 Huawei Technologies Co., Ltd
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

"""常量定义 - 统一管理 AIKG Workflow CLI 中的所有常量"""

import logging

logger = logging.getLogger(__name__)


# ===== Logo =====
def make_gradient_logo():
    """读取并返回带薄荷绿渐变的 logo（从资源文件加载 ANSI）"""
    from rich.text import Text
    from pathlib import Path

    try:
        current_dir = Path(__file__).parent.resolve()
        # constants.py 在 cli/ 目录下，需要回到 ai_kernel_generator 目录
        logo_path = current_dir.parent / "resources" / "logo.ans"

        return Text.from_ansi(logo_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load logo: {e}", exc_info=True)
        return Text("AKG CLI (Logo Missing)", style="bold red")


# ===== 节点名称 =====
class NodeName:
    """LangGraph 节点名称常量"""

    TASK_INIT = "task_init"
    CODER = "coder"
    VERIFIER = "verifier"
    CONDUCTOR = "conductor"
    DESIGNER = "designer"  # 新增 designer


# ===== 框架 =====
class Framework:
    """支持的框架常量"""

    TORCH = "torch"
    MINDSPORE = "mindspore"


# ===== 后端 =====
class Backend:
    """支持的后端常量"""

    CUDA = "cuda"
    ASCEND = "ascend"


# ===== 默认值 =====
class Defaults:
    """默认值常量"""

    WORKFLOW_NAME = "coder_only_workflow"
    LOG_DIR = "~/aikg_logs"
    STREAM_UPDATE_INTERVAL = 0.3  # 流式渲染更新间隔（秒）- 真流式渲染


# ===== 语法高亮语言 =====
class SyntaxLanguage:
    """语法高亮支持的语言"""

    PYTHON = "python"
    TEXT = "text"


# ===== 显示样式 =====
class DisplayStyle:
    """Rich 控制台显示样式"""

    # 基础样式
    BOLD = "bold"
    DIM = "dim"
    ITALIC = "italic"
    REVERSE = "reverse"

    # 统一强调色：尽量只用黑白 + 一种颜色（终端里用 bright_blue 避免部分终端把 blue 显示成偏紫）
    PRIMARY = "bright_blue"

    # 复合样式 - Bold
    BOLD_CYAN = f"bold {PRIMARY}"
    BOLD_YELLOW = f"bold {PRIMARY}"
    BOLD_GREEN = f"bold {PRIMARY}"
    BOLD_BLUE = f"bold {PRIMARY}"
    BOLD_WHITE = "bold"
    BOLD_RED = f"bold {PRIMARY}"

    # 单色样式
    GREEN = PRIMARY
    RED = PRIMARY
    YELLOW = PRIMARY
    CYAN = PRIMARY
    BLUE = PRIMARY
    MAGENTA = PRIMARY
    WHITE = "default"


# ===== UI 符号 =====
class UISymbol:
    """UI 符号常量"""

    # 分隔符
    VERTICAL_BAR = "│"

    # 树形结构
    TREE_END = "└─"

    # 状态标记
    DONE = "[DONE]"
    ERROR = "[ERR]"
    STOP = "[STOP]"
    WARNING = "[!]"

    # Emoji 风格标记 (可选)
    ROCKET = "📡"

    # 列表标记
    BULLET = "•"
