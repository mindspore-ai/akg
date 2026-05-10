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

"""
L2 Cache 清除模块。

提供 NPU L2 cache 清除功能，用于性能测试时确保测量结果不受缓存影响。

支持两种清除方式：
1. triton_ascend: 使用专用 triton kernel（推荐，可精确过滤）
2. 其他 DSL: 使用 tensor.zero_()（fallback，有误判风险）
"""

from typing import Literal, List

# 延迟导入 torch/torch_npu，避免在 mindspore 环境下触发 aclInit 冲突
_torch = None
_torch_npu = None


def _ensure_torch():
    global _torch, _torch_npu
    if _torch is None:
        import torch
        import torch_npu
        _torch = torch
        _torch_npu = torch_npu
    return _torch, _torch_npu

# 尝试导入 triton（可能未安装）
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# L2 cache 清除相关常量
L2_CACHE_SIZE_DEFAULT = 192 * 1024 * 1024  # 192MB 默认值
L2_CACHE_CLEAR_KERNEL_NAME = "AKG_l2cache_clear"  # 专用 kernel 名称，用于过滤

# DSL 类型定义
DslType = Literal["triton_ascend", "triton_cuda", "torch", "tilelang_npuir", "ascendc", "other"]

# ============================================================================
# L2 Cache 警告消息收集（绕过 suppress_output）
# ============================================================================

_l2_cache_warnings: List[str] = []


def _add_l2_cache_warning(message: str):
    """添加警告消息到收集列表（绕过 suppress_output）"""
    _l2_cache_warnings.append(message)


def get_l2_cache_warnings() -> List[str]:
    """获取所有收集的 L2 cache 警告消息"""
    return _l2_cache_warnings.copy()


def clear_l2_cache_warnings():
    """清空所有收集的 L2 cache 警告消息"""
    global _l2_cache_warnings
    _l2_cache_warnings = []


# ============================================================================
# L2 Cache 大小检测
# ============================================================================

_l2_cache_size_detected = None


def _get_l2_cache_size(device_id: int = 0) -> int:
    """
    从 NPU 设备属性获取 L2 cache 大小。
    
    Args:
        device_id: NPU 设备 ID
        
    Returns:
        int: L2 cache 大小（字节）
    """
    global _l2_cache_size_detected
    
    if _l2_cache_size_detected is not None:
        return _l2_cache_size_detected
    
    try:
        _, torch_npu = _ensure_torch()
        device_props = torch_npu.npu.get_device_properties(device_id)
        l2_size = getattr(device_props, 'L2_cache_size', None)
        if l2_size is not None and l2_size > 0:
            _l2_cache_size_detected = l2_size
            return l2_size
    except Exception:
        pass
    
    _l2_cache_size_detected = L2_CACHE_SIZE_DEFAULT
    return _l2_cache_size_detected


# ============================================================================
# 获取核心数
# ============================================================================

_core_nums_cache = None


def _get_core_nums(vec_default=40, cube_default=20):
    """
    获取 NPU 核心数（VEC + CUBE）。

    通过 triton runtime API 获取，结果缓存避免重复调用。

    Returns:
        tuple[int, int]: (vec_core_num, cube_core_num)
    """
    global _core_nums_cache

    if _core_nums_cache is not None:
        return _core_nums_cache

    vec, cube = vec_default, cube_default
    try:
        import torch
        import triton
        device = torch.npu.current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(device)
        vec = properties.get("num_vectorcore", vec_default)
        cube = properties.get("num_aicore", cube_default)
    except Exception:
        pass

    _core_nums_cache = (vec, cube)
    return _core_nums_cache


# ============================================================================
# Triton-Ascend 专用 L2 Cache 清除 Kernel（模块级别定义）
# ============================================================================

# 在模块级别定义 triton kernel，避免 JIT 编译时作用域问题
# 性能优化：使用大 BLOCK_SIZE 减少循环次数，提升带宽利用率
if _TRITON_AVAILABLE:
    @triton.jit
    def AKG_l2cache_clear(
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        CORE_NUM: tl.constexpr,
    ):
        """
        专用 L2 cache 清除 kernel。
        
        通过写入一个大 buffer 来强制刷新 L2 cache。
        kernel 名称为 AKG_l2cache_clear，便于在 profiler 结果中识别和过滤。
        
        使用交错循环处理，grid 大小等于核心数，参考 triton-ascend 编写规范。
        大 BLOCK_SIZE 确保高带宽利用率，减少循环开销。
        """
        pid = tl.program_id(0)
        
        # 计算总块数
        num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
        
        # 交错循环处理：每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM, ... 块
        # 这样所有数据都会被恰好处理一次，负载均衡
        for block_idx in range(pid, num_blocks, CORE_NUM):
            block_start = block_idx * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            # 写入 0 值来清除 cache
            tl.store(output_ptr + offsets, tl.zeros([BLOCK_SIZE], dtype=tl.int32), mask=mask)


# ============================================================================
# L2 Cache Buffer 管理
# ============================================================================

_l2_cache_buffer = None


def _get_l2_cache_buffer(device_id: int = 0):
    """
    获取用于清除 L2 cache 的 buffer（PyTorch 版）。
    使用惰性初始化，避免重复分配内存。
    
    Args:
        device_id: NPU 设备 ID
        
    Returns:
        torch.Tensor: 足以覆盖 L2 cache 大小的 int32 tensor
    """
    global _l2_cache_buffer
    
    if _l2_cache_buffer is None:
        torch, _ = _ensure_torch()
        l2_size = _get_l2_cache_size(device_id)
        n_elements = l2_size // 4
        _l2_cache_buffer = torch.empty(n_elements, dtype=torch.int32, device='npu')
    
    return _l2_cache_buffer


# ============================================================================
# L2 Cache 清除函数
# ============================================================================

def clear_l2_cache_triton():
    """
    使用 triton-ascend kernel 清除 L2 cache。
    
    这是推荐的清除方式，因为：
    1. 使用专用 kernel 名称 (AKG_l2cache_clear)，便于在 profiler 中精确识别和过滤
    2. 避免与用户代码中的 zeros/zero_ 操作混淆
    
    性能优化：
    - 使用大 BLOCK_SIZE (32768) 减少循环次数
    - grid 大小等于 VEC 核心数，充分利用并行度
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available for L2 cache clearing")
    
    torch, _ = _ensure_torch()
    buffer = _get_l2_cache_buffer()
    n_elements = buffer.numel()
    
    core_num, _ = _get_core_nums()
    
    BLOCK_SIZE = 32768
    
    grid = (core_num,)
    
    AKG_l2cache_clear[grid](buffer, n_elements, BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=core_num)
    torch.npu.synchronize()


def clear_l2_cache_zero():
    """
    使用 tensor.zero_() 清除 L2 cache（fallback 方式）。
    
    警告：此方式会在 profiler 中记录为 "ZerosLike" 类型，
    如果用户代码中也使用了 zeros_like/zero_()，可能导致误过滤。
    """
    torch, _ = _ensure_torch()
    buffer = _get_l2_cache_buffer()
    buffer.zero_()
    torch.npu.synchronize()


# ============================================================================
# MindSpore 版 L2 Cache Buffer 管理
# ============================================================================

_l2_cache_buffer_ms = None


def _get_l2_cache_buffer_ms():
    """
    获取用于清除 L2 cache 的 buffer（MindSpore 版）。
    使用惰性初始化，避免重复分配内存。
    
    MindSpore 的 AscendDeviceProperties 不提供 L2_cache_size，
    因此使用默认值 L2_CACHE_SIZE_DEFAULT。
    
    Returns:
        mindspore.Tensor: 足以覆盖 L2 cache 大小的 int32 tensor
    """
    global _l2_cache_buffer_ms
    
    if _l2_cache_buffer_ms is None:
        import mindspore as ms
        n_elements = L2_CACHE_SIZE_DEFAULT // 4
        _l2_cache_buffer_ms = ms.ops.zeros((n_elements,), dtype=ms.int32)
    
    return _l2_cache_buffer_ms


def clear_l2_cache_zero_ms():
    """
    使用 MindSpore tensor.zero_() 清除 L2 cache。
    
    警告：此方式会在 profiler 中记录为 "ZerosLike" 类型，
    如果用户代码中也使用了 zeros_like/zero_()，可能导致误过滤。
    """
    import mindspore as ms
    buffer = _get_l2_cache_buffer_ms()
    buffer.zero_()
    ms.runtime.synchronize()


def clear_l2_cache(dsl: DslType = "other", framework: str = "torch"):
    """
    清除 L2 cache 的统一入口函数。
    
    Args:
        dsl: DSL 类型，决定使用哪种清除方式
             - "triton_ascend": 使用专用 triton kernel（推荐，仅 torch 框架支持）
             - 其他: 使用 tensor.zero_()（fallback）
        framework: 框架类型 ("torch" 或 "mindspore")，决定使用哪套 tensor 接口
    
    Returns:
        None
    """
    if framework == "mindspore":
        clear_l2_cache_zero_ms()
        return

    if dsl == "triton_ascend":
        try:
            clear_l2_cache_triton()
        except Exception as e:
            _add_l2_cache_warning(
                f"[L2 Cache] Triton kernel call failed ({e}), falling back to zero_() method. "
                "Results may have false positive filtering risk."
            )
            clear_l2_cache_zero()
    else:
        if not hasattr(clear_l2_cache, '_warned_for_dsl'):
            clear_l2_cache._warned_for_dsl = set()
        
        if dsl not in clear_l2_cache._warned_for_dsl:
            _add_l2_cache_warning(
                f"[L2 Cache] Current DSL ({dsl}) has no dedicated L2 cache clear method. "
                "Using tensor.zero_() for clearing. "
                "Note: This will be recorded as 'ZerosLike' type in profiler. "
                "If the target operator also uses zeros_like/zero_(), timing may be inaccurate. "
                "For precise results, please analyze the specific operator manually."
            )
            clear_l2_cache._warned_for_dsl.add(dsl)
        
        clear_l2_cache_zero()
