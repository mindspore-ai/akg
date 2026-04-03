"""
设备后端抽象 — 统一 CUDA / NPU / CPU 的检测、同步、环境变量。

所有设备相关的 if/else 集中在这里，其他模块通过调用本模块实现设备无关。
"""

import os


def _probe_device(device_type: str) -> bool:
    """在目标设备上做一次小 tensor 运算，验证设备真正可用。"""
    import torch
    try:
        dev = torch.device(device_type)
        a = torch.tensor([1.0, 2.0], device=dev)
        b = (a + a).sum().item()
        return b == 6.0
    except Exception:
        return False


def detect_device_type() -> str:
    """按优先级检测可用设备后端: cuda → npu → cpu。

    不仅检查 is_available()，还会在设备上执行一次小运算验证健康状态。
    """
    try:
        import torch
        if torch.cuda.is_available() and _probe_device("cuda"):
            return "cuda"
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available() and _probe_device("npu"):
                return "npu"
        except ImportError:
            pass
    except ImportError:
        pass
    return "cpu"


def setup_device(device_str: str = "auto"):
    """解析设备字符串，返回 torch.device。

    auto → 按优先级自动选择 (cuda → npu → cpu)。
    """
    import torch
    if device_str == "auto":
        device_str = detect_device_type()
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    elif device.type == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    return device


def synchronize(device):
    """设备无关的同步。CPU 上为 no-op。"""
    if device is None:
        return
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize()
    elif device.type == "npu":
        import torch
        torch.npu.synchronize()


def get_device_info() -> dict:
    """返回当前设备的元信息（name, type, 及后端特有属性）。

    CUDA 额外返回 compute_cap, multi_processor_count；
    NPU / CPU 仅返回 name + type。
    """
    dtype = detect_device_type()
    if dtype == "cuda":
        import torch
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "type": "cuda",
            "compute_cap": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
    if dtype == "npu":
        import torch
        return {
            "name": torch.npu.get_device_name(0),
            "type": "npu",
        }
    return {"name": "CPU", "type": "cpu"}


def get_device_env(device_id=None, device_type: str | None = None) -> dict:
    """返回带设备可见性环境变量的 env dict。

    Args:
        device_id: 设备编号 (0, 1, ...)。None 时不设置可见性变量。
        device_type: "cuda" / "npu" / None (自动检测)。
    """
    env = os.environ.copy()
    if device_id is None:
        return env
    if device_type is None:
        device_type = detect_device_type()
    if device_type == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    elif device_type == "npu":
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    return env
