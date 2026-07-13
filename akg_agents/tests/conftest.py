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
import logging
import pytest


def pytest_configure(config):
    """
    pytest启动时的配置钩子，根据AKG_AGENTS_LOG_LEVEL环境变量设置日志级别
    """
    # 根据AKG_AGENTS_LOG_LEVEL环境变量设置日志级别
    glog_level = os.getenv('AKG_AGENTS_LOG_LEVEL', '1')  # 默认为1 (INFO)
    level_map = {
        '0': logging.DEBUG,
        '1': logging.INFO,
        '2': logging.WARNING,
        '3': logging.ERROR
    }
    log_level = level_map.get(glog_level, logging.INFO)

    # 设置根日志记录器的级别
    logging.getLogger().setLevel(log_level)

    # 更安全的方式：检查并设置pytest的日志配置
    level_name = logging.getLevelName(log_level)
    if hasattr(config, 'option'):
        if hasattr(config.option, 'log_cli_level'):
            config.option.log_cli_level = level_name
        if hasattr(config.option, 'log_level'):
            config.option.log_level = level_name

    print(f"pytest配置: 根据环境变量AKG_AGENTS_LOG_LEVEL={glog_level}设置日志级别为{level_name}")


def _cuda_available() -> bool:
    """True only on a host with a working NVIDIA CUDA device. Degrades to
    False if torch is absent / CUDA init fails, so ``cuda``-marked tests skip
    rather than error on non-NVIDIA hosts (e.g. the Ascend boxes)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``cuda`` / ``a100`` tests when no NVIDIA device is present —
    these run real triton-CUDA kernels and otherwise fail with "no NVIDIA
    driver" on Ascend hardware. Single place that gives those markers their
    skip semantics."""
    if _cuda_available():
        return
    skip_cuda = pytest.mark.skip(reason="no NVIDIA CUDA device available")
    for item in items:
        if "cuda" in item.keywords or "a100" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(autouse=True)
def manage_test_dir():
    """
    自动管理测试目录
    """
    # 保存初始工作目录
    original_dir = os.getcwd()
    yield
    # 测试结束后恢复到初始目录
    os.chdir(original_dir)
