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

from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)


def get_cpu_info() -> str:
    """动态获取当前CPU配置信息"""
    try:
        system = platform.system()

        if system == "Linux":
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_info = []
                useful_keys = [
                    'Architecture', 'CPU(s)', 'Thread(s) per core', 'Core(s) per socket',
                    'Socket(s)', 'Model name', 'CPU MHz', 'CPU max MHz',
                    'L1d cache', 'L1i cache', 'L2 cache', 'L3 cache', 'Flags'
                ]

                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key = line.split(':', 1)[0].strip()
                        if key in useful_keys:
                            useful_info.append(line)

                if useful_info:
                    return f"# Linux CPU信息 (关键参数)\n" + '\n'.join(useful_info)

        elif system == "Darwin":
            result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_info = []
                useful_keys = [
                    'hw.ncpu', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.cpufrequency',
                    'hw.cpufrequency_max', 'hw.l1dcachesize', 'hw.l1icachesize',
                    'hw.l2cachesize', 'hw.l3cachesize', 'machdep.cpu.brand_string',
                    'machdep.cpu.core_count', 'machdep.cpu.thread_count',
                    'machdep.cpu.features', 'machdep.cpu.leaf7_features'
                ]

                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key = line.split(':', 1)[0].strip()
                        if key in useful_keys:
                            useful_info.append(line)

                if useful_info:
                    return f"# macOS CPU信息 (关键参数)\n" + '\n'.join(useful_info)

        elif system == "Windows":
            result = subprocess.run(['wmic', 'cpu', 'get', '*', '/format:list'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                useful_info = []
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
                            useful_info.append(f"{key}={value}")

                if useful_info:
                    return f"# Windows CPU信息 (关键参数)\n" + '\n'.join(useful_info)

        return ""

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
            "ascend910b3": "hardware/Ascend910B3.md",
            "ascend910b4": "hardware/Ascend910B4.md",
            "ascend310p3": "hardware/Ascend310P3.md"
        },
        "cuda": {
            "a100": "hardware/CUDA_A100.md",
            "v100": "hardware/CUDA_V100.md"
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
    full_path = os.path.join(get_project_root(), "resources", "docs", hardware_doc_path)

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"硬件文档不存在: {full_path}")
    except Exception as e:
        raise ValueError(f"读取硬件文档失败: {full_path}, 错误: {e}")
