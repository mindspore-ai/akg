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

import os
import importlib
import subprocess
import logging
import yaml
import requests
from pathlib import Path
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


def check_env(framework=None, backend=None, dsl=None, config_path=None, config=None, is_remote=False):
    """
    检查环境是否满足要求

    Args:
        framework: 框架类型 (mindspore/torch/numpy)
        backend: 后端类型 (ascend/cuda/cpu)
        dsl: DSL类型 (triton_cuda/triton_ascend/swft)
        config_path: 任务配置文件路径，用于检查具体使用的模型API
        config: 已加载的配置字典，优先于config_path使用
        is_remote: 是否为远程模式，若为True，则跳过本地软件和硬件检查，仅检查LLM API

    Returns:
        bool: 环境检查是否通过
    """
    print("🔍 正在检查环境...")
    if is_remote:
        print("ℹ️  使用远程模式，跳过本地软件和硬件检查，仅检查LLM API连接")
    issues = []

    # 1. 检查基础包
    if not is_remote:
        base_packages = ['numpy', 'jinja2', 'httpx']
        for pkg in base_packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                issues.append(f"❌ 缺少基础包: {pkg}")

    # 2. 检查框架
    if not is_remote:
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
    if not is_remote:
        if dsl in ['triton_cuda', 'triton_ascend']:
            try:
                importlib.import_module('triton')
            except ImportError:
                issues.append(f"❌ 缺少 triton (required for {dsl})")
        elif dsl == 'triton':
            # 自动转换逻辑：根据backend推断
            if backend:
                if backend == 'cuda':
                    dsl = 'triton_cuda'
                elif backend == 'ascend':
                    dsl = 'triton_ascend'
                # 递归检查转换后的DSL
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

    # 4. 检查硬件
    if not is_remote:
        if backend == 'cuda':
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                if result.returncode != 0:
                    issues.append("⚠️ CUDA设备可能不可用")
            except:
                issues.append("⚠️ 未找到 nvidia-smi")
        elif backend == 'ascend':
            try:
                result = subprocess.run(['npu-smi', 'info'], capture_output=True, timeout=5)
                if result.returncode != 0:
                    issues.append("⚠️ 昇腾设备可能不可用")
            except:
                issues.append("⚠️ 未找到 npu-smi")

    # 5. 检查API配置
    api_ok = _check_llm_api(config_path, config)
    if not api_ok:
        issues.append("❌ LLM API配置或连接有问题")

    # 输出结果
    if issues:
        print("🚨 发现问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n请解决上述问题后重试。")
        return False
    else:
        print("✅ 环境检查通过!")
        return True


def _load_llm_config():
    """
    加载LLM配置（使用 core_v2 配置系统）
    
    Returns:
        AKGSettings 对象，包含所有模型配置
    """
    try:
        from akg_agents.core_v2.config import get_settings
        settings = get_settings()
        return settings
    except Exception as e:
        print(f"⚠️ 读取LLM配置失败: {e}")
        return None


def _check_llm_api(config_path=None, config=None):
    """检查LLM API配置和连接（使用 core_v2 配置系统）"""
    # 【最高优先级】检查环境变量覆盖模式
    env_base_url = os.getenv("AKG_AGENTS_BASE_URL")
    env_model_name = os.getenv("AKG_AGENTS_MODEL_NAME")
    env_api_key = os.getenv("AKG_AGENTS_API_KEY")
    
    if env_base_url and env_model_name and env_api_key:
        print("=" * 60)
        print("🔍 检测到环境变量覆盖模式（最高优先级）")
        print(f"  AKG_AGENTS_BASE_URL: {env_base_url}")
        print(f"  AKG_AGENTS_MODEL_NAME: {env_model_name}")
        # 脱敏显示API密钥
        masked_key = env_api_key[:8] + "*" * (len(env_api_key) - 12) + \
            env_api_key[-4:] if len(env_api_key) > 12 else "***"
        print(f"  AKG_AGENTS_API_KEY: {masked_key}")
        print("=" * 60)
        
        # 简单的连通性检查（尝试访问base_url）
        try:
            import requests
            # 尝试访问根路径或models端点
            test_url = env_base_url.rstrip('/') + '/models'
            response = requests.get(test_url, timeout=5, verify=False)
            print(f"✅ API连接测试成功: {env_base_url}")
            return True
        except Exception as e:
            print(f"⚠️ API连接测试失败: {e}")
            print("ℹ️ 将在实际调用时再次尝试连接")
            # 不强制失败，因为有些API的models端点可能不存在
            return True
    
    # 加载 core_v2 配置
    settings = _load_llm_config()
    if not settings:
        print("⚠️ 无法加载 LLM 配置，跳过 API 检查")
        return True

    # 获取使用的 model_level（从 agent_model_config）
    used_levels = set()

    if config:
        # 优先使用已加载的配置字典
        if 'agent_model_config' in config:
            agent_configs = config['agent_model_config']
            # agent_model_config 中的值现在是 model_level（如 "standard"）
            used_levels.update(agent_configs.values())
            print(f"🔍 从配置字典读取到 model_level: {', '.join(used_levels)}")
        else:
            print("⚠️ 配置字典中未找到 agent_model_config，使用默认 model_level")
            used_levels.add(settings.default_model)
    elif config_path:
        # 如果没有配置字典，尝试从文件加载
        task_config = _load_task_config(config_path)
        if task_config and 'agent_model_config' in task_config:
            agent_configs = task_config['agent_model_config']
            used_levels.update(agent_configs.values())
            print(f"🔍 从任务配置文件读取到 model_level: {', '.join(used_levels)}")
        else:
            print("⚠️ 任务配置文件中未找到 agent_model_config，使用默认 model_level")
            used_levels.add(settings.default_model)
    else:
        print("⚠️ 未提供任务配置，使用默认 model_level")
        used_levels.add(settings.default_model)

    # 检查每个使用的 model_level
    all_ok = True
    for level in used_levels:
        if level not in settings.models:
            print(f"❌ 未找到 model_level '{level}' 的配置")
            print(f"   可用的 model_level: {', '.join(settings.models.keys())}")
            all_ok = False
            continue

        model_config = settings.models[level]
        print(f"🔍 检查 model_level: {level}")
        print(f"   模型: {model_config.model_name}")
        print(f"   API: {model_config.base_url}")

        # 检查 API 密钥
        if not model_config.api_key:
            print(f"❌ model_level '{level}' 未配置 API 密钥")
            all_ok = False
            continue
        
        # 脱敏显示
        masked_key = model_config.api_key[:8] + "*" * (len(model_config.api_key) - 12) + \
            model_config.api_key[-4:] if len(model_config.api_key) > 12 else "***"
        print(f"   ✅ API密钥已配置: {masked_key}")

        # 测试 API 连接
        if not _test_api_connection_v2(model_config):
            all_ok = False

    return all_ok


def _test_api_connection_v2(model_config) -> bool:
    """
    测试 API 连接（core_v2 版本）
    
    Args:
        model_config: ModelConfig 对象
    
    Returns:
        bool: 连接是否成功
    """
    try:
        # 准备请求头
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_config.api_key}"
        }

        # 统一处理API地址：去掉所有路径，只保留协议和域名
        from urllib.parse import urlparse
        parsed = urlparse(model_config.base_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # GET请求端点（通常用于获取信息）
        get_endpoints = [
            "/models",
            "/v1/models",
            "/v3/models",
            "/api/v1/models",
            "/api/v3/models",
            "/health",
            "/status",
            "/"
        ]

        # POST请求端点（聊天补全，需要发送测试数据）
        post_endpoints = [
            ("/chat/completions", {"model": "test", "messages": []}),
            ("/v1/chat/completions", {"model": "test", "messages": []}),
            ("/v3/chat/completions", {"model": "test", "messages": []}),
            ("/api/v3/chat/completions", {"model": "test", "messages": []}),
        ]

        success_endpoint = None

        # 先尝试GET请求
        for endpoint in get_endpoints:
            try:
                url = f"{base_url}{endpoint}"
                response = requests.get(url, headers=headers, timeout=30, verify=False)

                if response.status_code in [200, 201, 202, 204]:
                    success_endpoint = url
                    break
                elif response.status_code == 401:
                    success_endpoint = url + " (需要认证)"
                    break

            except:
                continue

        # 如果GET请求都失败，尝试POST请求到聊天端点
        if not success_endpoint:
            for endpoint, test_data in post_endpoints:
                try:
                    url = f"{base_url}{endpoint}"
                    response = requests.post(url, headers=headers, json=test_data, timeout=30, verify=False)

                    if response.status_code in [200, 201, 202, 204, 400, 422]:
                        success_endpoint = url + " (POST)"
                        break
                    elif response.status_code == 401:
                        success_endpoint = url + " (需要认证)"
                        break

                except:
                    continue

        if success_endpoint:
            print(f"   ✅ API连接正常: {success_endpoint}")
            return True
        else:
            print(f"   ⚠️ API连接测试失败，请检查 {base_url} 是否可访问")
            return False

    except Exception as e:
        print(f"   ❌ API连接测试异常: {e}")
        return False


def _load_task_config(config_path):
    """加载任务配置文件"""
    try:
        # 支持相对路径和绝对路径
        if not config_path.startswith('/'):
            # 相对路径，从项目根目录开始
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent  # 回到项目根目录
            config_path = project_root / config_path
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            print(f"⚠️ 未找到任务配置文件: {config_path}")
            return None

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        print(f"⚠️ 读取任务配置失败: {e}")
        return None


def check_env_for_task(framework, backend, dsl, config, is_remote=False):
    """
    为tests目录提供的便捷环境检查函数
    失败时直接抛出异常

    Args:
        framework: 框架类型
        backend: 后端类型
        dsl: DSL类型
        config: 通过load_config()加载的配置字典
        is_remote: 是否为远程模式，若为True，则跳过本地软件和硬件检查，仅检查LLM API

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
