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


def _raise_submodule_error(path_name, path_value):
    """通用的子模块错误提示函数"""
    error_msg = f"找不到 {path_name}: {path_value}\n"
    error_msg += "请确保已正确下载相关子模块。\n"
    error_msg += "解决方案：\n"
    error_msg += "1. 如果已克隆仓库，请执行：git submodule update --init --recursive\n"
    error_msg += "2. 或者在aikg目录下执行：git submodule update --init --remote 'thirdparty/*'"
    raise FileNotFoundError(error_msg)


def get_op_task_str(op_name):
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    task_path = os.path.join(commom_path, 'resources',
                             'task', op_name + '_task.py')
    with open(task_path, "r", encoding="utf-8") as f:
        op_task_str = f.read()
    return op_task_str


def get_kernelbench_task_desc(op_name, framework="torch"):
    """获取 KernelBench 任务描述"""
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    aikg_path = os.path.dirname(commom_path)

    if framework == "torch":
        # Path for torch benchmarks from the KernelBench submodule.
        # The submodule is at `aikg/thirdparty/KernelBench`, and benchmark files are inside `KernelBench/level1/` subdirectory.
        base_dir = os.path.join(
            aikg_path, 'thirdparty', 'KernelBench', 'KernelBench', 'level1')
        # Files are directly in level1 directory with naming pattern: {number}_{name}.py
        task_path = os.path.join(base_dir, op_name + '.py')
    else:
        # Original path for mindspore and numpy benchmarks
        base_dir = os.path.join(aikg_path, 'benchmark',
                                'kernelbench', framework)
        task_path = os.path.join(
            base_dir, op_name, op_name + f'_{framework}.py')

    # 检查文件是否存在
    if not os.path.exists(task_path):
        if framework == "torch":
            _raise_submodule_error("KernelBench 任务文件", task_path)
        else:
            _raise_submodule_error(f"{framework} benchmark 任务文件", task_path)

    with open(task_path, "r", encoding="utf-8") as f:
        benchmark_task_str = f.read()
    return benchmark_task_str


def get_multikernelbench_task_desc(op_name, framework="torch"):
    """获取 MultiKernelBench 任务描述"""
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    aikg_path = os.path.dirname(commom_path)

    # Path for MultiKernelBench benchmarks from reference directory
    base_path = os.path.join(
        aikg_path, 'thirdparty', 'MultiKernelBench', 'reference')

    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        _raise_submodule_error("MultiKernelBench 目录", base_path)

    # Find the file in any category
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            task_path = os.path.join(category_path, op_name + '.py')
            if os.path.exists(task_path):
                with open(task_path, "r", encoding="utf-8") as f:
                    return f.read()

    # If not found in any category, raise an error
    _raise_submodule_error(f"MultiKernelBench 中的操作 {op_name}", f"已搜索目录: {base_path}")


def get_kernelbench_op_name(task_index_list, framework="torch"):
    """获取 KernelBench 操作名称列表"""
    if task_index_list is None:
        return None

    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    aikg_path = os.path.dirname(commom_path)

    if framework == "torch":
        task_path = os.path.join(
            aikg_path, 'thirdparty', 'KernelBench', 'KernelBench', 'level1')
        # PyTorch: 直接查找文件
        task_prefix_list = [f"{task_index}_" for task_index in task_index_list]
        matched_files = []

        if os.path.exists(task_path):
            for file in os.listdir(task_path):
                if file.endswith('.py') and any(file.startswith(task_prefix) for task_prefix in task_prefix_list):
                    benchmark_name = file[:-3]
                    matched_files.append(benchmark_name)
    else:
        # MindSpore/NumPy: 查找子目录
        task_path = os.path.join(
            aikg_path, 'benchmark', 'kernelbench', framework)
        task_prefix_list = [f"{task_index}_" for task_index in task_index_list]
        matched_files = []

        if os.path.exists(task_path):
            for dir_name in os.listdir(task_path):
                dir_path = os.path.join(task_path, dir_name)
                if os.path.isdir(dir_path) and any(dir_name.startswith(task_prefix) for task_prefix in task_prefix_list):
                    # 对于MindSpore，返回目录名作为benchmark_name
                    matched_files.append(dir_name)

    return matched_files if matched_files else None


def get_multikernelbench_op_name(category="all", framework="torch", op_name=None):
    """获取 MultiKernelBench 操作名称列表"""
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)
    aikg_path = os.path.dirname(commom_path)

    # For MultiKernelBench, use category-based reading
    base_path = os.path.join(
        aikg_path, 'thirdparty', 'MultiKernelBench', 'reference')

    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        _raise_submodule_error("MultiKernelBench 目录", base_path)

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


def add_op_prefix(op_name, benchmark="KernelBench"):
    """为操作名称添加前缀，格式：aikg_{benchmark}_{op_name}"""
    return f"aikg_{benchmark.lower()}_{op_name}"


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


def process_task_results(results, print_summary=True):
    """
    处理任务运行结果，验证每个op_name是否至少有一次成功。

    Args:
        results: task_pool.wait_all() 返回的结果列表，格式为 [(op_name, result, _), ...]
        print_summary: 是否打印结果摘要

    Returns:
        bool: 如果所有op_name都至少成功一次返回True，否则返回False
    """
    from collections import defaultdict

    # 收集结果到字典
    result_dict = defaultdict(int)
    for op_name, result, _ in results:
        result_dict[op_name] += result

    # 检查失败的case - 同一个op_name只要有一次成功就算通过
    failed_cases = []
    for op_name, success_count in result_dict.items():
        if success_count == 0:  # 如果成功次数为0，说明所有尝试都失败了
            failed_cases.append(op_name)

    # 可选的结果摘要
    if print_summary:
        total_ops = len(result_dict)
        passed_ops = total_ops - len(failed_cases)
        pass_rate = passed_ops / total_ops if total_ops > 0 else 0.0

        print('-' * 60)
        print(f"结果字典: {dict(result_dict)}")
        print(f"通过的操作数: {passed_ops}/{total_ops}")
        print(f"通过率: {pass_rate:.2%}")
        if failed_cases:
            print(f"失败的测试case: {failed_cases}")
        print('-' * 60)

    return len(failed_cases) == 0
