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


def get_op_task_str(op_name):
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    task_path = os.path.join(commom_path, 'resources',
                             'task', op_name + '_task.py')
    with open(task_path, "r", encoding="utf-8") as f:
        op_task_str = f.read()
    return op_task_str


def get_benchmark_task(op_name, framework="mindspore", benchmark="KernelBench"):
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    aikg_path = os.path.dirname(commom_path)

    if framework == "torch":
        if benchmark == "multiKernelBench":
            # Path for multiKernelBench benchmarks from reference directory
            base_path = os.path.join(
                aikg_path, 'thirdparty', 'multiKernelBench', 'reference')

            # Find the file in any category
            for category in os.listdir(base_path):
                category_path = os.path.join(base_path, category)
                if os.path.isdir(category_path):
                    task_path = os.path.join(category_path, op_name + '.py')
                    if os.path.exists(task_path):
                        with open(task_path, "r", encoding="utf-8") as f:
                            return f.read()

            # If not found in any category, raise an error
            raise FileNotFoundError(
                f"Operation {op_name} not found in multiKernelBench reference directory")

        elif benchmark == "KernelBench":
            # Path for torch benchmarks from the KernelBench submodule.
            # The submodule is at `aikg/thirdparty/KernelBench`, and benchmark files are inside `KernelBench/level1/` subdirectory.
            base_dir = os.path.join(
                aikg_path, 'thirdparty', 'KernelBench', 'KernelBench', 'level1')
            # Files are directly in level1 directory with naming pattern: {number}_{name}.py
            task_path = os.path.join(base_dir, op_name + '.py')

        else:
            # 支持其他第三方库的扩展
            # 可以在这里添加其他 benchmark 的处理逻辑
            raise FileNotFoundError(f"Benchmark {benchmark} not supported")
    else:
        # Original path for mindspore and numpy benchmarks
        base_dir = os.path.join(aikg_path, 'benchmark',
                                'kernelbench', framework)
        task_path = os.path.join(
            base_dir, op_name, op_name + f'_{framework}.py')

    with open(task_path, "r", encoding="utf-8") as f:
        benchmark_task_str = f.read()
    return benchmark_task_str


def get_benchmark_name(task_index_list=None, framework="mindspore", benchmark="KernelBench", category="all", op_name=None):
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    aikg_path = os.path.dirname(commom_path)

    if framework == "torch":
        if benchmark == "multiKernelBench":
            # For multiKernelBench, use category-based reading
            base_path = os.path.join(
                aikg_path, 'thirdparty', 'multiKernelBench', 'reference')
            matched_files = []

            if os.path.exists(base_path):
                if category == "all":
                    # Get all available categories
                    categories = [d for d in os.listdir(
                        base_path) if os.path.isdir(os.path.join(base_path, d))]
                else:
                    categories = [category] if os.path.isdir(
                        os.path.join(base_path, category)) else []

                for cat in categories:
                    category_path = os.path.join(base_path, cat)
                    if os.path.exists(category_path):
                        for file in os.listdir(category_path):
                            if file.endswith('.py'):
                                # Remove .py extension to get the benchmark name
                                operation_name = file[:-3]

                                # 如果指定了 op_name，只返回匹配的 case
                                if op_name is not None:
                                    if operation_name == op_name:
                                        matched_files.append(operation_name)
                                        break  # 找到匹配的就停止搜索
                                else:
                                    matched_files.append(operation_name)

            return matched_files if matched_files else None

        elif benchmark == "KernelBench":
            # For KernelBench, use index-based reading (original method)
            if task_index_list is None:
                return None

            task_path = os.path.join(
                aikg_path, 'thirdparty', 'KernelBench', 'KernelBench', 'level1')
            task_prefix_list = [
                f"{task_index}_" for task_index in task_index_list]
            matched_files = []

            if os.path.exists(task_path):
                for file in os.listdir(task_path):
                    if file.endswith('.py') and any(file.startswith(task_prefix) for task_prefix in task_prefix_list):
                        # Remove .py extension to get the benchmark name
                        benchmark_name = file[:-3]
                        matched_files.append(benchmark_name)

            return matched_files if matched_files else None

        else:
            # 不支持的 benchmark 类型
            return None
    else:
        # Original logic for mindspore and numpy benchmarks
        if task_index_list is None:
            return None

        task_path = os.path.join(
            aikg_path, 'benchmark', 'kernelbench', framework)
        task_prefix_list = [f"{task_index}_" for task_index in task_index_list]
        matched_folders = []

        if os.path.exists(task_path):
            for file in os.listdir(task_path):
                if any(file.startswith(task_prefix) for task_prefix in task_prefix_list):
                    matched_folders.append(file)

        return matched_folders if matched_folders else None


def add_op_prefix(benchmark_name):
    return "aikg_" + benchmark_name


def get_folder_names(folder_path):
    python_files = []

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.py') and os.path.isfile(os.path.join(folder_path, file)):
                python_files.append(file[:-3])

    return python_files


def get_task_content(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name + '.py')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return ""
