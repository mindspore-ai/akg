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

from enum import Enum

from textual import log


# ===== Logo =====
def make_gradient_logo():
    """读取并返回带薄荷绿渐变的 logo（从资源文件加载 ANSI）"""
    from rich.text import Text
    from pathlib import Path

    try:
        current_dir = Path(__file__).parent.resolve()
        # constants.py is in python/ai_kernel_generator/cli/cli/
        # resources is in python/ai_kernel_generator/resources/
        logo_path = current_dir.parent.parent / "resources" / "logo.ans"

        return Text.from_ansi(logo_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Failed to load logo: {e}")
        return Text("AKG CLI (Logo Missing)", style="bold red")


AKG_CLI_LOGO = make_gradient_logo().plain


# ===== 节点名称 =====
class NodeName:
    """LangGraph 节点名称常量"""

    TASK_INIT = "task_init"
    CODER = "coder"
    VERIFIER = "verifier"
    CONDUCTOR = "conductor"
    DESIGNER = "designer"  # 新增 designer


# ===== TaskInit 状态 =====
class TaskInitStatus:
    """TaskInit 阶段状态常量"""

    READY = "ready"
    NEED_CLARIFICATION = "need_clarification"
    UNSUPPORTED = "unsupported"


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


# ===== DSL =====
class DSL:
    """支持的 DSL 常量"""

    TRITON_CUDA = "triton_cuda"
    TRITON = "triton"
    TILELANG = "tilelang"
    CUDA = "cuda"
    CPP = "cpp"


# ===== 架构 =====
class Arch:
    """常用架构常量"""

    A100 = "a100"
    ASCEND_910B4 = "ascend910b4"


# ===== 环境变量 =====
class EnvVar:
    """环境变量名称常量"""

    WORKFLOW_SERVER_URL = "WORKFLOW_SERVER_URL"
    WORKFLOW_FRAMEWORK = "WORKFLOW_FRAMEWORK"
    WORKFLOW_BACKEND = "WORKFLOW_BACKEND"
    WORKFLOW_ARCH = "WORKFLOW_ARCH"
    WORKFLOW_DSL = "WORKFLOW_DSL"
    WORKFLOW_USE_STREAM = "WORKFLOW_USE_STREAM"


# ===== 预定义任务 =====
class PredefinedTasks:
    """预定义的测试任务"""

    # 简单任务
    RELU = "我需要一个 ReLU 激活函数算子，输入形状是 (32, 4096)"
    SIGMOID = "我需要一个 Sigmoid 激活函数算子，输入形状是 (16, 8192)"

    # 中等任务
    RMSNORM = "我需要一个 RmsNorm 算子，输入形状是 (16, 16384)"
    LAYERNORM = "我需要一个 LayerNorm 算子，输入形状是 (32, 1024)，归一化维度是最后一维"
    SOFTMAX = (
        "我需要一个 Softmax 算子，输入形状是 (16, 512, 512)，在最后一维上做 softmax"
    )
    GELU = "我需要一个 GELU 激活函数算子，输入形状是 (64, 2048)"

    # 中等偏难任务
    MATMUL = (
        "我需要一个矩阵乘法算子，输入 A 的形状是 (128, 256)，输入 B 的形状是 (256, 512)"
    )
    ADD = "我需要一个向量加法算子，两个输入形状都是 (1024, 4096)"

    # 困难任务
    FLASH_ATTENTION = "我需要一个 Flash Attention 算子，Q/K/V 的形状都是 (8, 16, 128, 64)，其中 batch=8, heads=16, seq_len=128, head_dim=64"

    # 任务名称到描述的映射
    TASK_MAP = {
        "relu": RELU,
        "sigmoid": SIGMOID,
        "rmsnorm": RMSNORM,
        "layernorm": LAYERNORM,
        "softmax": SOFTMAX,
        "gelu": GELU,
        "matmul": MATMUL,
        "add": ADD,
        "flash_attention": FLASH_ATTENTION,
    }

    # 任务列表（用于命令行帮助）
    TASK_CHOICES = list(TASK_MAP.keys())

    # 默认任务
    DEFAULT_TASK = "rmsnorm"


# ===== 默认值 =====
class Defaults:
    """默认值常量"""

    FRAMEWORK = Framework.TORCH
    BACKEND = Backend.CUDA
    ARCH = Arch.A100
    DSL = DSL.TRITON_CUDA
    WORKFLOW_NAME = "coder_only_workflow"
    SERVER_URL = "http://localhost:8000"
    LOG_DIR = "~/aikg_logs"
    DEFAULT_USER_INPUT = PredefinedTasks.RMSNORM  # 使用预定义任务
    STREAM_UPDATE_INTERVAL = 0.3  # 流式渲染更新间隔（秒）- 真流式渲染
    MAX_DISPLAY_LINES = 100  # 流式渲染最大显示行数（增量渲染模式下可以更大）
    BARK_KEY = "DM4kSUFaw33Sy7iFe4ujXM"  # 默认 Bark 推送 Key


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
