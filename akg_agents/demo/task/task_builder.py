"""
任务构建器 - 将提取的代码片段构建为 KernelBench 标准格式

目标格式（参考 36_RMSNorm_.py）:
  - import torch / torch.nn
  - class Model(nn.Module): __init__ + forward
  - get_inputs()    -> 返回 forward 的输入
  - get_init_inputs() -> 返回 __init__ 的参数
"""

TASK_TEMPLATE = '''\
import torch
import torch.nn as nn

{model_class}

{shape_constants}

def get_inputs():
{get_inputs_body}

def get_init_inputs():
{get_init_inputs_body}
'''

TASK_TEMPLATE_MINIMAL = '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    {description}
    """
    def __init__(self{init_params}):
        super(Model, self).__init__()
{init_body}

    def forward(self{forward_params}):
{forward_body}

{shape_constants}

def get_inputs():
{get_inputs_body}

def get_init_inputs():
{get_init_inputs_body}
'''


class TaskBuilder:
    """
    将提取的算子代码转化为 KernelBench 标准格式。
    
    核心能力由 LLM 在 ReAct 循环中驱动，此类提供模板和验证。
    """

    @staticmethod
    def get_template() -> str:
        """返回完整模板说明，供 LLM 参考"""
        return TASK_TEMPLATE

    @staticmethod
    def get_minimal_template() -> str:
        """返回最小模板"""
        return TASK_TEMPLATE_MINIMAL

    @staticmethod
    def validate_task_code(code: str) -> dict:
        """
        验证任务代码是否符合 KernelBench 格式

        Returns:
            {"valid": bool, "issues": [str]}
        """
        issues = []

        if "class Model" not in code:
            issues.append("缺少 class Model 定义")
        if "def forward" not in code:
            issues.append("缺少 forward 方法")
        if "def get_inputs" not in code:
            issues.append("缺少 get_inputs() 函数")
        if "def get_init_inputs" not in code:
            issues.append("缺少 get_init_inputs() 函数")
        if "import torch" not in code:
            issues.append("缺少 import torch")

        # 检查 get_inputs 返回 list
        if "def get_inputs" in code:
            try:
                get_inputs_body = code.split("def get_inputs")[1].split("\ndef ")[0]
                has_return_list = ("return [" in get_inputs_body or
                                   "return list(" in get_inputs_body)
                if not has_return_list:
                    issues.append("get_inputs() 应返回列表 (return [...] 或 return list(...))")
            except IndexError:
                pass  # 格式不标准，跳过此检查

        return {"valid": len(issues) == 0, "issues": issues}

    @staticmethod
    def format_reference() -> str:
        """返回一个格式参考示例"""
        return '''\
# === KernelBench 任务格式参考 (RMSNorm) ===
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x / rms

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
'''
