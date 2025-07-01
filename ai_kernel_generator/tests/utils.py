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
import re


def get_op_task_str(op_name):
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    task_path = os.path.join(commom_path, 'resources', 'task', op_name + '_task.py')
    with open(task_path, "r", encoding="utf-8") as f:
        op_task_str = f.read()
    return op_task_str


def get_benchmark_task(op_name, framework="mindspore"):
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    task_path = os.path.join(os.path.dirname(commom_path), 'benchmark',
                             'kernelbench', framework,
                             op_name, op_name + f'_{framework}.py')
    with open(task_path, "r", encoding="utf-8") as f:
        benchmark_task_str = f.read()
    return benchmark_task_str


def get_benchmark_name(task_index_list, framework="mindspore"):
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    # 在kernelbench的torch_benchmark目录中查找
    task_path = os.path.join(os.path.dirname(commom_path), 'benchmark', 'kernelbench', framework)
    task_prefix_list = [f"{task_index}_" for task_index in task_index_list]
    matched_folders = []

    if os.path.exists(task_path):
        for file in os.listdir(task_path):
            if any(file.startswith(task_prefix) for task_prefix in task_prefix_list):
                matched_folders.append(file)

    return matched_folders if matched_folders else None


def remove_alnum_from_benchmark_name(benchmark_name):
    return re.sub(r'^\d+_', '', benchmark_name).rstrip('_') + "_op"
