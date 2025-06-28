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
import sys
import os

# 定义通用的日志格式
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
log_datefmt = '%Y-%m-%d %H:%M:%S'

# 根据AIKG_LOG_LEVEL环境变量设置日志级别
glog_level = os.getenv('AIKG_LOG_LEVEL', '1')  # 默认为1 (INFO)
level_map = {
    '0': logging.DEBUG,
    '1': logging.INFO,
    '2': logging.WARNING,
    '3': logging.ERROR
}
log_level = level_map.get(glog_level, logging.INFO)  # 如果环境变量值无效，默认使用INFO

logging.basicConfig(
    level=log_level,
    format=log_format,
    datefmt=log_datefmt
)

logger = logging.getLogger(__name__)


def get_project_root():
    """获取项目根目录的绝对路径

    Returns:
        str: 项目根目录的绝对路径
    """
    return os.path.dirname(os.path.abspath(__file__))
