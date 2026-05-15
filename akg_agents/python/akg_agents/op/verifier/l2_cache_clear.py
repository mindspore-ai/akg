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
import torch
import torch_npu

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
        device_props = torch_npu.npu.get_device_properties(device_id)
        # L2_cache_size 单位是字节
        l2_size = getattr(device_props, 'L2_cache_size', None)
        if l2_size is not None and l2_size > 0:
            _l2_cache_size_detected = l2_size
            return l2_size
    except Exception:
        pass
    
    # 回退到默认值
    _l2_cache_size_detected = L2_CACHE_SIZE_DEFAULT
    return _l2_cache_size_detected


# ============================================================================
# 获取核心数
# ============================================================================

_vec_core_num = None


def _get_vec_core_num(device_id: int = 0) -> int:
    """
    获取 NPU 的 VEC 核心数。
    
    根据 triton-ascend 编写规范，向量计算类算子使用 VEC 核心数。
    
    Args:
        device_id: NPU 设备 ID
        
    Returns:
        int: VEC 核心数
    """
    global _vec_core_num
    
    if _vec_core_num is not None:
        return _vec_core_num
    
    try:
        _vec_core_num = torch_npu.npu.npu_config.get_device_limit(device_id).get("vector_core_num", 40)
    except Exception:
        # Ascend 910B4 默认: 20个AI Core × 2个VEC/Core = 40
        _vec_core_num = 40
    
    return _vec_core_num


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
    获取用于清除 L2 cache 的 buffer。
    使用惰性初始化，避免重复分配内存。
    
    Args:
        device_id: NPU 设备 ID
        
    Returns:
        torch.Tensor: 足以覆盖 L2 cache 大小的 int32 tensor
    """
    global _l2_cache_buffer
    
    if _l2_cache_buffer is None:
        l2_size = _get_l2_cache_size(device_id)
        # l2_size / 4 bytes = int32 元素个数
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
    
    buffer = _get_l2_cache_buffer()
    n_elements = buffer.numel()
    
    # 获取 VEC 核心数作为 grid 大小，参考 triton-ascend 编写规范
    core_num = _get_vec_core_num()
    
    # 使用大 BLOCK_SIZE 提升性能
    # 32768 个 int32 = 128KB，符合 UB 大小限制
    # 对于 192MB = 48M 元素，只需要 ~1465 个块，每核心处理约 37 个块
    BLOCK_SIZE = 32768
    
    # grid 设置为核心数，使用交错循环处理
    grid = (core_num,)
    
    # 调用 kernel
    AKG_l2cache_clear[grid](buffer, n_elements, BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=core_num)
    torch.npu.synchronize()


def clear_l2_cache_zero():
    """
    使用 tensor.zero_() 清除 L2 cache（fallback 方式）。
    
    警告：此方式会在 profiler 中记录为 "ZerosLike" 类型，
    如果用户代码中也使用了 zeros_like/zero_()，可能导致误过滤。
    """
    buffer = _get_l2_cache_buffer()
    buffer.zero_()
    torch.npu.synchronize()


def clear_l2_cache(dsl: DslType = "other"):
    """
    清除 L2 cache 的统一入口函数。
    
    Args:
        dsl: DSL 类型，决定使用哪种清除方式
             - "triton_ascend": 使用专用 triton kernel（推荐）
             - 其他: 使用 tensor.zero_()（fallback）
    
    Returns:
        None
    """
    if dsl == "triton_ascend":
        try:
            clear_l2_cache_triton()
        except Exception as e:
            # triton kernel 失败，fallback 到 zero_()
            _add_l2_cache_warning(
                f"[L2 Cache] Triton kernel call failed ({e}), falling back to zero_() method. "
                "Results may have false positive filtering risk."
            )
            clear_l2_cache_zero()
    else:
        # 非 triton_ascend DSL，使用 zero_() 方式
        # 每个 DSL 只警告一次
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
