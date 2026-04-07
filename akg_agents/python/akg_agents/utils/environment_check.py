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

import importlib
import subprocess
import logging

logger = logging.getLogger(__name__)


def check_env(framework=None, backend=None, dsl=None, config_path=None, config=None, is_remote=False):
    """
    检查环境是否满足要求（仅检查软件包和硬件，不检查LLM API）

    Args:
        framework: 框架类型 (mindspore/torch/numpy)
        backend: 后端类型 (ascend/cuda/cpu)
        dsl: DSL类型 (triton_cuda/triton_ascend/swft)
        config_path: 保留参数，不再使用
        config: 保留参数，不再使用
        is_remote: 是否为远程模式，若为True，则跳过本地软件和硬件检查

    Returns:
        bool: 环境检查是否通过
    """
    print("🔍 正在检查环境...")
    if is_remote:
        print("ℹ️  使用远程模式，跳过本地软件和硬件检查")
        print("✅ 环境检查通过!")
        return True

    issues = []

    # 1. 检查基础包
    base_packages = ['numpy', 'jinja2', 'httpx']
    for pkg in base_packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            issues.append(f"❌ 缺少基础包: {pkg}")

    # 2. 检查框架
    if framework == 'mindspore':
        try:
            importlib.import_module('mindspore')
        except ImportError:
            issues.append("❌ 缺少 mindspore")
    elif framework == 'torch':
        try:
            importlib.import_module('torch')
        except ImportError:
            issues.append("❌ 缺少 torch")

    # 3. 检查DSL
    if dsl in ['triton_cuda', 'triton_ascend']:
        try:
            importlib.import_module('triton')
        except ImportError:
            issues.append(f"❌ 缺少 triton (required for {dsl})")
    elif dsl == 'triton':
        if backend:
            if backend == 'cuda':
                dsl = 'triton_cuda'
            elif backend == 'ascend':
                dsl = 'triton_ascend'
            return check_env(framework, backend, dsl, config_path, config, is_remote)
        else:
            issues.append("❌ dsl='triton' is no longer supported. Please use 'triton_cuda' or 'triton_ascend' explicitly, or provide backend parameter for automatic conversion.")
    elif dsl == 'torch':
        try:
            importlib.import_module('torch')
        except ImportError:
            issues.append(f"❌ 缺少 torch (required for {dsl})")
    elif dsl == 'swft':
        try:
            importlib.import_module('swft')
        except ImportError:
            issues.append("❌ 缺少 swft")
    elif dsl == 'pypto':
        try:
            importlib.import_module('pypto')
        except ImportError:
            issues.append("❌ 缺少 pypto")

    # 4. 检查硬件
    if backend == 'cuda':
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=30, shell=True)
            if result.returncode != 0:
                issues.append("⚠️ CUDA设备可能不可用")
        except subprocess.TimeoutExpired:
            issues.append("⚠️ nvidia-smi 执行超时，GPU驱动可能异常")
        except FileNotFoundError:
            issues.append("⚠️ 未找到 nvidia-smi 命令")
        except Exception as e:
            issues.append(f"⚠️ nvidia-smi 检查失败: {str(e)}")
    elif backend == 'ascend':
        try:
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, timeout=30)
            if result.returncode != 0:
                issues.append("⚠️ 昇腾设备可能不可用")
        except Exception:
            issues.append("⚠️ 未找到 npu-smi")

    # 输出结果
    if issues:
        print("🚨 发现问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n请解决上述问题后重试。")
        return False

    print("✅ 环境检查通过!")
    return True


def check_env_for_task(framework, backend, dsl, config, is_remote=False):
    """
    为tests目录提供的便捷环境检查函数
    失败时直接抛出异常

    Args:
        framework: 框架类型
        backend: 后端类型
        dsl: DSL类型
        config: 通过load_config()加载的配置字典
        is_remote: 是否为远程模式，若为True，则跳过本地软件和硬件检查

    Raises:
        RuntimeError: 环境检查失败时抛出异常
    """
    success = check_env(
        framework=framework,
        backend=backend,
        dsl=dsl,
        config=config,
        is_remote=is_remote
    )
    if not success:
        raise RuntimeError("环境检查失败，请解决上述问题后重试")
