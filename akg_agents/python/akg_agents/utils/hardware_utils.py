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

import logging
import platform
import subprocess
import os
from typing import Dict, List, Set

from akg_agents import get_project_root

logger = logging.getLogger(__name__)


# ==================== 指令集定义与优化建议 ====================

# 指令集优先级与分类（用于大模型选择最优指令集）
INSTRUCTION_SET_INFO: Dict[str, Dict] = {
    # ===== x86/x64 SIMD 指令集（按性能等级排序） =====
    "avx512": {
        "category": "SIMD",
        "vector_width": 512,
        "priority": 100,  # 最高优先级
        "flags": ["avx512f", "avx512dq", "avx512cd", "avx512bw", "avx512vl", 
                  "avx512_vnni", "avx512_bf16", "avx512_fp16", "avx512_vbmi", "avx512_vbmi2"],
        "description": "AVX-512指令集，512位向量宽度，最强SIMD能力",
        "optimization_hints": [
            "优先使用512位向量操作，单指令处理16个float32或8个double",
            "支持更复杂的mask操作，可高效处理非对齐数据",
            "avx512_vnni适合INT8/INT16量化推理",
            "avx512_bf16/fp16适合混合精度训练",
            "注意：某些CPU在运行AVX-512时会降频，需权衡"
        ]
    },
    "avx2": {
        "category": "SIMD",
        "vector_width": 256,
        "priority": 80,
        "flags": ["avx2"],
        "description": "AVX2指令集，256位向量宽度，主流高性能SIMD",
        "optimization_hints": [
            "单指令处理8个float32或4个double",
            "支持FMA（融合乘加）操作，减少计算延迟",
            "广泛支持，是目前最通用的高性能选择",
            "gather指令支持非连续内存访问"
        ]
    },
    "avx": {
        "category": "SIMD",
        "vector_width": 256,
        "priority": 60,
        "flags": ["avx"],
        "description": "AVX指令集，256位向量宽度",
        "optimization_hints": [
            "单指令处理8个float32或4个double",
            "比SSE提供更宽的向量寄存器",
            "需要数据256位对齐以获得最佳性能"
        ]
    },
    "sse4": {
        "category": "SIMD",
        "vector_width": 128,
        "priority": 40,
        "flags": ["sse4_1", "sse4_2", "sse4.1", "sse4.2", "sse41", "sse42"],
        "description": "SSE4指令集，128位向量宽度",
        "optimization_hints": [
            "单指令处理4个float32或2个double",
            "包含点积、字符串处理等实用指令",
            "几乎所有现代x86 CPU都支持"
        ]
    },
    "sse2": {
        "category": "SIMD",
        "vector_width": 128,
        "priority": 20,
        "flags": ["sse2"],
        "description": "SSE2指令集，128位向量宽度（基础SIMD）",
        "optimization_hints": [
            "x86-64的基线SIMD指令集",
            "支持64位整数SIMD操作",
            "作为fallback选项，兼容性最好"
        ]
    },
    
    # ===== 融合乘加指令 =====
    "fma": {
        "category": "FMA",
        "vector_width": 256,
        "priority": 85,
        "flags": ["fma", "fma3", "fma4"],
        "description": "FMA融合乘加指令，单指令完成乘法和加法",
        "optimization_hints": [
            "a*b+c 单指令完成，提升吞吐量和精度",
            "矩阵乘法、卷积等计算密集型操作的关键优化",
            "配合AVX2使用效果最佳"
        ]
    },
    
    # ===== ARM NEON 指令集 =====
    "neon": {
        "category": "SIMD",
        "vector_width": 128,
        "priority": 70,
        "flags": ["neon", "asimd", "asimdhp", "asimddp"],
        "description": "ARM NEON指令集，128位向量宽度",
        "optimization_hints": [
            "ARM平台的主要SIMD指令集",
            "asimdhp支持半精度浮点(FP16)",
            "asimddp支持点积操作，适合矩阵运算",
            "适用于Apple Silicon、高通等ARM处理器"
        ]
    },
    
    # ===== ARM SVE/SVE2 指令集 =====
    "sve": {
        "category": "SIMD",
        "vector_width": 2048,  # 可变，最高2048
        "priority": 95,
        "flags": ["sve", "sve2"],
        "description": "ARM SVE/SVE2可扩展向量指令集，向量宽度可变（128-2048位）",
        "optimization_hints": [
            "向量长度无关编程，同一代码适配不同硬件",
            "比NEON更强的向量处理能力",
            "主要用于服务器和高性能计算场景",
            "SVE2增加了更多整数和密码学指令"
        ]
    },
    
    # ===== 位操作指令 =====
    "bmi": {
        "category": "Bit Manipulation",
        "vector_width": 64,
        "priority": 30,
        "flags": ["bmi1", "bmi2", "abm", "popcnt", "lzcnt", "tzcnt"],
        "description": "位操作指令集，高效位运算",
        "optimization_hints": [
            "popcnt用于快速计算popcount（位1的数量）",
            "lzcnt/tzcnt用于前导/尾随零计数",
            "适合位图操作、哈希计算等场景"
        ]
    },
    
    # ===== 加密指令 =====
    "aes": {
        "category": "Crypto",
        "vector_width": 128,
        "priority": 25,
        "flags": ["aes", "aes-ni", "aesni", "vaes"],
        "description": "AES加密指令，硬件加速加解密",
        "optimization_hints": [
            "AES加解密性能提升10-20倍",
            "vaes支持向量化AES操作（AVX-512）"
        ]
    },
    "sha": {
        "category": "Crypto",
        "vector_width": 128,
        "priority": 25,
        "flags": ["sha_ni", "sha1", "sha256", "sha512", "sha3"],
        "description": "SHA哈希指令，硬件加速哈希计算",
        "optimization_hints": [
            "SHA-1/SHA-256硬件加速",
            "适合需要大量哈希计算的场景"
        ]
    },
}


def get_instruction_set_flags(raw_flags: str) -> Dict[str, List[str]]:
    """从CPU flags字符串中解析指令集信息
    
    Args:
        raw_flags: CPU flags字符串（空格分隔）
        
    Returns:
        Dict: 分类后的指令集信息
    """
    flags_set: Set[str] = set(flag.lower() for flag in raw_flags.split())
    
    detected_instruction_sets: Dict[str, List[str]] = {}
    
    for inst_name, inst_info in INSTRUCTION_SET_INFO.items():
        matched_flags = [f for f in inst_info["flags"] if f.lower() in flags_set]
        if matched_flags:
            detected_instruction_sets[inst_name] = matched_flags
    
    return detected_instruction_sets


def format_instruction_set_info(detected_sets: Dict[str, List[str]]) -> str:
    """格式化指令集信息，供大模型参考
    
    Args:
        detected_sets: 检测到的指令集字典
        
    Returns:
        str: 格式化的指令集信息文本
    """
    if not detected_sets:
        return ""
    
    lines = ["## CPU 指令集信息", ""]
    
    # 按优先级排序
    sorted_sets = sorted(
        detected_sets.items(),
        key=lambda x: INSTRUCTION_SET_INFO.get(x[0], {}).get("priority", 0),
        reverse=True
    )
    
    # 推荐的最优指令集
    if sorted_sets:
        best_simd = None
        best_fma = None
        for name, _ in sorted_sets:
            info = INSTRUCTION_SET_INFO.get(name, {})
            if info.get("category") == "SIMD" and not best_simd:
                best_simd = name
            elif info.get("category") == "FMA" and not best_fma:
                best_fma = name
        
        lines.append("### 推荐的优化指令集")
        if best_simd:
            info = INSTRUCTION_SET_INFO[best_simd]
            lines.append(f"- **SIMD首选**: {best_simd.upper()} ({info['vector_width']}位向量宽度)")
        if best_fma:
            lines.append(f"- **融合乘加**: {best_fma.upper()} (a*b+c单指令完成)")
        lines.append("")
    
    # 详细的指令集列表
    lines.append("### 可用指令集详情")
    lines.append("")
    
    for name, matched_flags in sorted_sets:
        info = INSTRUCTION_SET_INFO.get(name, {})
        lines.append(f"#### {name.upper()}")
        lines.append(f"- **类别**: {info.get('category', 'Unknown')}")
        lines.append(f"- **向量宽度**: {info.get('vector_width', 'N/A')}位")
        lines.append(f"- **描述**: {info.get('description', '')}")
        lines.append(f"- **检测到的flags**: {', '.join(matched_flags)}")
        
        hints = info.get("optimization_hints", [])
        if hints:
            lines.append("- **优化建议**:")
            for hint in hints:
                lines.append(f"  - {hint}")
        lines.append("")
    
    # 代码生成指导
    lines.append("### 代码生成指导")
    lines.append("")
    lines.append("请根据上述指令集信息生成优化代码：")
    lines.append("1. **优先使用最高优先级的SIMD指令集**进行向量化")
    lines.append("2. **如果支持FMA**，将乘加操作融合为FMA指令")
    lines.append("3. **考虑数据对齐**，确保数据按向量宽度对齐以获得最佳性能")
    lines.append("4. **提供多版本实现**（可选）：针对不同指令集提供多个代码路径")
    lines.append("")
    
    return "\n".join(lines)


def _get_windows_cpu_flags() -> str:
    """获取Windows系统的CPU指令集flags
    
    Returns:
        str: 空格分隔的CPU flags字符串
    """
    flags = []
    
    try:
        # 方法1: 使用WMIC获取基本特性
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'Caption', '/format:list'],
            capture_output=True, text=True, timeout=5, 
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        
        # 方法2: 通过注册表或cpuid检测常见指令集
        # 使用Python内置功能检测部分指令集
        import struct
        
        # 检测基本的x86指令集支持（通过尝试执行相关代码来间接检测）
        # 这里使用更可靠的方式：检查CPU型号名称来推断支持的指令集
        cpu_name_result = subprocess.run(
            ['wmic', 'cpu', 'get', 'Name', '/format:list'],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        
        cpu_name = ""
        if cpu_name_result.returncode == 0:
            for line in cpu_name_result.stdout.split('\n'):
                if line.startswith('Name='):
                    cpu_name = line.split('=', 1)[1].strip().lower()
                    break
        
        # 基于CPU型号推断指令集支持
        # 现代Intel/AMD CPU通用支持
        flags.extend(['sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2'])
        
        # Intel CPU
        if 'intel' in cpu_name:
            # Haswell及更新 (2013+)
            if any(x in cpu_name for x in ['i3', 'i5', 'i7', 'i9', 'xeon', 'core']):
                flags.extend(['avx', 'avx2', 'fma', 'bmi1', 'bmi2', 'popcnt'])
                
                # Skylake-X/Cannon Lake及更新支持AVX-512
                if any(x in cpu_name for x in ['platinum', 'gold', 'silver', 'bronze', 
                                                'w-', '10th gen', '11th gen', '12th gen', 
                                                '13th gen', '14th gen']):
                    flags.extend(['avx512f', 'avx512dq', 'avx512cd', 'avx512bw', 'avx512vl'])
        
        # AMD CPU
        elif 'amd' in cpu_name:
            if any(x in cpu_name for x in ['ryzen', 'epyc', 'threadripper']):
                flags.extend(['avx', 'avx2', 'fma', 'bmi1', 'bmi2', 'popcnt'])
                # Zen4及更新支持AVX-512
                if any(x in cpu_name for x in ['7000', '9000', '8000', 'genoa', 'bergamo']):
                    flags.extend(['avx512f', 'avx512dq', 'avx512cd', 'avx512bw', 'avx512vl'])
        
        # 尝试使用PowerShell获取更详细信息（Windows 10+）
        try:
            ps_cmd = '''
            try {
                $cpu = Get-CimInstance -ClassName Win32_Processor
                $features = @()
                if ($cpu.VMMonitorModeExtensions) { $features += "vmx" }
                if ($cpu.SecondLevelAddressTranslationExtensions) { $features += "slat" }
                $features -join " "
            } catch { "" }
            '''
            ps_result = subprocess.run(
                ['powershell', '-Command', ps_cmd],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            if ps_result.returncode == 0 and ps_result.stdout.strip():
                flags.extend(ps_result.stdout.strip().split())
        except Exception:
            pass
            
    except Exception as e:
        logger.warning(f"获取Windows CPU flags时发生错误: {e}")
    
    return ' '.join(set(flags))


def get_cpu_info() -> str:
    """动态获取当前CPU配置信息，包含指令集优化建议"""
    try:
        system = platform.system()
        basic_info = []
        raw_flags = ""

        if system == "Linux":
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_keys = [
                    'Architecture', 'CPU(s)', 'Thread(s) per core', 'Core(s) per socket',
                    'Socket(s)', 'Model name', 'CPU MHz', 'CPU max MHz',
                    'L1d cache', 'L1i cache', 'L2 cache', 'L3 cache'
                ]

                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if key in useful_keys:
                            basic_info.append(f"{key}: {value}")
                        elif key == 'Flags':
                            raw_flags = value

        elif system == "Darwin":
            result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_keys = [
                    'hw.ncpu', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.cpufrequency',
                    'hw.cpufrequency_max', 'hw.l1dcachesize', 'hw.l1icachesize',
                    'hw.l2cachesize', 'hw.l3cachesize', 'machdep.cpu.brand_string',
                    'machdep.cpu.core_count', 'machdep.cpu.thread_count'
                ]
                feature_keys = ['machdep.cpu.features', 'machdep.cpu.leaf7_features']

                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if key in useful_keys:
                            basic_info.append(f"{key}: {value}")
                        elif key in feature_keys:
                            raw_flags += " " + value

        elif system == "Windows":
            result = subprocess.run(
                ['wmic', 'cpu', 'get', '*', '/format:list'],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_keys = [
                    'Name', 'NumberOfCores', 'NumberOfLogicalProcessors',
                    'L2CacheSize', 'L3CacheSize', 'MaxClockSpeed',
                    'Architecture', 'Family', 'Manufacturer'
                ]

                for line in lines:
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        if key in useful_keys and value and value != 'None' and value != '':
                            basic_info.append(f"{key}: {value}")
            
            # Windows需要特殊处理来获取CPU flags
            raw_flags = _get_windows_cpu_flags()

        # 构建输出
        output_parts = []
        
        # 基础信息部分
        if basic_info:
            output_parts.append(f"# {system} CPU信息\n")
            output_parts.append("## 基础配置\n")
            output_parts.append('\n'.join(basic_info))
            output_parts.append("\n")
        
        # 指令集信息部分（使用新的解析和格式化功能）
        if raw_flags:
            detected_sets = get_instruction_set_flags(raw_flags)
            if detected_sets:
                instruction_set_info = format_instruction_set_info(detected_sets)
                output_parts.append(instruction_set_info)
            else:
                # 如果没有匹配到已知指令集，显示原始flags
                output_parts.append("## CPU Flags (原始)\n")
                output_parts.append(raw_flags[:500])  # 限制长度
                output_parts.append("\n")
        
        return '\n'.join(output_parts) if output_parts else ""

    except Exception as e:
        logger.error(f"获取CPU信息失败: {e}")
        return ""


def get_hardware_doc(backend: str, arch: str) -> str:
    """根据backend和architecture获取硬件信息

    Args:
        backend: 后端类型
        arch: 架构类型  

    Returns:
        str: 硬件文档内容

    Raises:
        ValueError: 不支持的backend或architecture时抛出异常
    """
    hardware_mapping = {
        "ascend": {
            "ascend910b1": "hardware/Ascend910B1.md",
            "ascend910b2": "hardware/Ascend910B2.md",
            "ascend910b2c": "hardware/Ascend910B2C.md",
            "ascend910b3": "hardware/Ascend910B3.md",
            "ascend910b4": "hardware/Ascend910B4.md",
            "ascend310p3": "hardware/Ascend310P3.md"
        },
        "cuda": {
            "a100": "hardware/CUDA_A100.md",
            "v100": "hardware/CUDA_V100.md",
            "h20": "hardware/CUDA_H20.md",
            "l20": "hardware/CUDA_L20.md",
            "rtx3090": "hardware/CUDA_RTX3090.md"
        }
    }

    if backend.lower() == "cpu":
        # 对CPU后端使用动态检测
        return get_cpu_info()

    if backend.lower() not in hardware_mapping:
        raise ValueError(f"不支持的backend: {backend}")

    architecture_mapping = hardware_mapping[backend.lower()]
    arch_lower = arch.lower()
    
    if arch_lower in architecture_mapping:
        hardware_doc_path = architecture_mapping[arch_lower]
    else:
        supported_architectures = list(architecture_mapping.keys())
        raise ValueError(f"不支持的architecture: {arch}，支持的architecture: {supported_architectures}")
    full_path = os.path.join(get_project_root(), "op", "resources", "docs", hardware_doc_path)

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"硬件文档不存在: {full_path}")
    except Exception as e:
        raise ValueError(f"读取硬件文档失败: {full_path}, 错误: {e}")


if __name__ == "__main__":
    print(get_cpu_info())