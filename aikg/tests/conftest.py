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
    pytest启动时的配置钩子，根据AIKG_LOG_LEVEL环境变量设置日志级别
    """
    # 根据AIKG_LOG_LEVEL环境变量设置日志级别
    glog_level = os.getenv('AIKG_LOG_LEVEL', '1')  # 默认为1 (INFO)
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

    print(f"pytest配置: 根据环境变量AIKG_LOG_LEVEL={glog_level}设置日志级别为{level_name}")
    print(f"DEBUG: 根日志记录器级别: {logging.getLogger().level}")


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
