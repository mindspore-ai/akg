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


def get_device_id(default=0):
    """统一获取 DEVICE_ID，默认值为 0。"""
    return int(os.getenv("DEVICE_ID", default))


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


def get_kernelbench_task_desc(op_name, framework="torch", level="level1"):
    """获取 KernelBench 任务描述"""
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)  # tests/op/
    tests_path = os.path.dirname(commom_path)  # tests/
    akg_agents_path = os.path.dirname(tests_path)  # akg_agents/

    if framework == "torch":
        # Path for torch benchmarks from the KernelBench submodule.
        # The submodule is at `akg_agents/thirdparty/KernelBench`, and benchmark files are inside `KernelBench/{level}/` subdirectory.
        base_dir = os.path.join(
            akg_agents_path, 'thirdparty', 'KernelBench', 'KernelBench', level)
        # Files are directly in level directory with naming pattern: {number}_{name}.py
        task_path = os.path.join(base_dir, op_name + '.py')
    else:
        # Original path for mindspore and numpy benchmarks
        base_dir = os.path.join(akg_agents_path, 'benchmark',
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
    commom_path = os.path.dirname(current_file_path)  # tests/op/
    tests_path = os.path.dirname(commom_path)  # tests/
    akg_agents_path = os.path.dirname(tests_path)  # akg_agents/

    # Path for MultiKernelBench benchmarks from reference directory
    base_path = os.path.join(
        akg_agents_path, 'thirdparty', 'MultiKernelBench', 'reference')

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


def get_akg_kernels_bench_task_desc(op_name, category=None, framework="torch"):
    """获取 AIKGBench 任务描述
    
    Args:
        op_name: 算子名称
        framework: 框架名称，默认为 "torch"
        category: 类别名称，可选值：'dynamic' 或 'static'，如果不指定则搜索所有类别
    """
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)  # tests/op/
    tests_path = os.path.dirname(commom_path)  # tests/
    akg_agents_path = os.path.dirname(tests_path)  # akg_agents/

    # Path for AIKGBench benchmarks
    base_path = os.path.join(akg_agents_path, 'benchmark', 'akg_kernels_bench')

    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        _raise_submodule_error("AIKGBench 目录", base_path)

    # 确定要搜索的类别列表
    if category:
        if category not in ['dynamic', 'static']:
            raise ValueError(f"无效的类别参数: {category}，有效值为 'dynamic' 或 'static'")
        categories = [f"{category}_shape"]
    else:
        # 搜索所有类别
        categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # 搜索文件
    for category_dir in categories:
        category_path = os.path.join(base_path, category_dir)
        if not os.path.exists(category_path):
            continue
            
        # 遍历子分类
        for subcategory in os.listdir(category_path):
            subcategory_path = os.path.join(category_path, subcategory)
            if os.path.isdir(subcategory_path):
                task_path = os.path.join(subcategory_path, op_name + '.py')
                if os.path.exists(task_path):
                    with open(task_path, "r", encoding="utf-8") as f:
                        return f.read()

    # 如果未找到，抛出错误
    if category:
        _raise_submodule_error(f"AIKGBench 类别 {categories[0]} 中的操作 {op_name}", f"已搜索目录: {os.path.join(base_path, categories[0])}")
    else:
        _raise_submodule_error(f"AIKGBench 中的操作 {op_name}", f"已搜索目录: {base_path}")


def get_kernelbench_op_name(task_index_list, framework="torch", level="level1"):
    """获取 KernelBench 操作名称列表"""
    if task_index_list is None:
        return None

    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)  # tests/op/
    tests_path = os.path.dirname(commom_path)  # tests/
    akg_agents_path = os.path.dirname(tests_path)  # akg_agents/

    if framework == "torch":
        task_path = os.path.join(
            akg_agents_path, 'thirdparty', 'KernelBench', 'KernelBench', level)
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
            akg_agents_path, 'benchmark', 'kernelbench', framework)
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
    commom_path = os.path.dirname(current_file_path)  # tests/op/
    tests_path = os.path.dirname(commom_path)  # tests/
    akg_agents_path = os.path.dirname(tests_path)  # akg_agents/

    # For MultiKernelBench, use category-based reading
    base_path = os.path.join(
        akg_agents_path, 'thirdparty', 'MultiKernelBench', 'reference')

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


def get_akg_kernels_bench_op_name(category="all", subcategory="all", framework="torch", op_name=None):
    """获取 AIKGBench 操作名称列表

    Args:
        category: 主分类 ("all", "dynamic", "static")
        subcategory: 子分类 ("all", "attention", "elemwise", "fused", "index", "norm", "reduction", "sorting", "tensor_manipulation", "matmul")
        framework: 框架名称 (默认 "torch")
        op_name: 特定操作名称 (可选)

    Returns:
        list: 匹配的操作名称列表，如果没有找到则返回 None
    """
    current_file_path = os.path.abspath(__file__)
    commom_path = os.path.dirname(current_file_path)  # tests/op/
    tests_path = os.path.dirname(commom_path)  # tests/
    akg_agents_path = os.path.dirname(tests_path)  # akg_agents/

    # For AIKGBench, use category-based reading
    base_path = os.path.join(akg_agents_path, 'benchmark', 'akg_kernels_bench')

    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        _raise_submodule_error("AIKGBench 目录", base_path)

    matched_files = []

    if os.path.exists(base_path):
        if category == "all":
            # Get all available categories (dynamic and static)
            categories = [d for d in os.listdir(
                base_path) if os.path.isdir(os.path.join(base_path, d))]
        else:
            categories = [category] if os.path.isdir(
                os.path.join(base_path, f"{category}_shape")) else []

        for cat in categories:
            category_path = os.path.join(base_path, f"{cat}_shape")
            if os.path.exists(category_path):
                # 遍历子分类 (attention, elemwise, fused, etc.)
                for subcat in os.listdir(category_path):
                    subcat_path = os.path.join(category_path, subcat)
                    if os.path.isdir(subcat_path):
                        # 如果指定了子分类，只处理匹配的子分类
                        if subcategory != "all" and subcat != subcategory:
                            continue

                        for file in os.listdir(subcat_path):
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
    """为操作名称添加前缀，格式：akg_agents_{benchmark}_{op_name}"""
    return f"akg_agents_{benchmark.lower()}_{op_name}"


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


def generate_beautiful_test_report(results, config, framework, dsl, backend, arch,
                                   save_to_file=True, output_dir=None):
    """
    生成美观的测试结果报告，包含控制台输出和文件保存。

    Args:
        results: task_pool.wait_all() 返回的结果列表，格式为 [(op_name, result, _), ...]
        config: 配置字典，需要包含 'log_dir' 键
        framework: 框架名称 (如 "torch", "mindspore")
        dsl: DSL名称 (如 "triton_cuda", "triton_ascend")
        backend: 后端名称 (如 "cuda", "ascend")
        arch: 架构名称 (如 "a100", "910b")
        save_to_file: 是否保存结果到文件
        output_dir: 输出目录，如果为None则使用config['log_dir']

    Returns:
        dict: 包含统计信息的字典
    """
    from pathlib import Path

    # 确定输出目录
    if output_dir is None:
        result_dir = Path(os.path.expanduser(config['log_dir']))
    else:
        result_dir = Path(output_dir)

    # 统计每个算子的通过次数 (pass@n 统计) - 从原始results统计
    op_stats = {}
    for op_name, result, _ in results:
        base_op_name = op_name  # 可能需要根据实际情况调整算子名称提取逻辑
        if base_op_name not in op_stats:
            op_stats[base_op_name] = {'passed': 0, 'total': 0}
        op_stats[base_op_name]['total'] += 1
        if result:
            op_stats[base_op_name]['passed'] += 1

    # 分类算子：有通过的 vs 完全失败的
    passed_ops = []  # 至少通过1次的算子
    failed_ops = []  # 完全失败的算子

    for op_name, stats in op_stats.items():
        if stats['passed'] > 0:
            passed_ops.append((op_name, stats['passed'], stats['total']))
        else:
            failed_ops.append((op_name, stats['passed'], stats['total']))

    # 提取order并按order排序的函数
    def extract_order(op_name):
        try:
            # 从 akg_agents_{order}_{op_name} 格式中提取 order
            if op_name.startswith('akg_agents_'):
                parts = op_name.split('_', 2)  # 分割成 ['akg_agents', 'order', 'op_name']
                if len(parts) >= 2:
                    return int(parts[1])
            return float('inf')  # 如果格式不匹配，放到最后
        except (ValueError, IndexError):
            return float('inf')  # 如果解析失败，放到最后

    # 按order排序
    passed_ops.sort(key=lambda x: extract_order(x[0]))
    failed_ops.sort(key=lambda x: extract_order(x[0]))

    # 控制台输出
    print('=' * 80)
    print(f"🚀 Pass@N 测试结果报告 - {framework.upper()} + {dsl.upper()} ({backend.upper()}/{arch.upper()})")
    print('=' * 80)
    print(f"📊 总体统计:")
    print(f"   • 测试算子总数: {len(op_stats)}")
    print(f"   • 通过算子数量: {len(passed_ops)} ")
    print(f"   • 失败算子数量: {len(failed_ops)} ")
    print(f"   • 算子通过率: {len(passed_ops)/len(op_stats)*100:.1f}%")
    print('-' * 80)

    if passed_ops:
        print("✅ 通过的算子:")
        for i, (op, passed, total) in enumerate(passed_ops, 1):
            print(f"   {i:2d}. {op} (pass num: {passed}/{total})")

    if failed_ops:
        print(f"\n❌ 完全失败的算子:")
        for i, (op, passed, total) in enumerate(failed_ops, 1):
            print(f"   {i:2d}. {op} (pass num: {passed}/{total})")

    print('=' * 80)

    # 保存详细结果到文件
    if save_to_file:
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "test_results.txt", "w", encoding="utf-8") as f:
            f.write("🚀 Pass@N 测试结果报告\n")
            f.write("=" * 80 + "\n\n")

            # 测试配置信息
            f.write("📋 测试配置:\n")
            f.write(f"   • 框架: {framework.upper()}\n")
            f.write(f"   • DSL: {dsl.upper()}\n")
            f.write(f"   • 后端: {backend.upper()}\n")
            f.write(f"   • 架构: {arch.upper()}\n\n")

            # 统计信息
            f.write("📊 Pass@N 统计:\n")
            f.write(f"   • 测试算子总数: {len(op_stats)}\n")
            f.write(f"   • 通过算子数量: {len(passed_ops)} ✅\n")
            f.write(f"   • 失败算子数量: {len(failed_ops)} ❌\n")
            f.write(f"   • 算子通过率: {len(passed_ops)/len(op_stats)*100:.1f}%\n\n")

            # 详细结果
            f.write("📝 详细结果:\n")
            f.write("-" * 60 + "\n")

            if passed_ops:
                f.write("✅ 通过的算子 (按order排序):\n")
                for i, (op, passed, total) in enumerate(passed_ops, 1):
                    f.write(f"   {i:2d}. {op} (pass num: {passed}/{total})\n")
                f.write("\n")

            if failed_ops:
                f.write("❌ 完全失败的算子:\n")
                for i, (op, passed, total) in enumerate(failed_ops, 1):
                    f.write(f"   {i:2d}. {op} (pass num: {passed}/{total})\n")
                f.write("\n")

            # Pass@N 统计表格
            f.write("📊 Pass@N 统计表:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'算子名称':<30} {'PassNum':<15} {'状态':<8}\n")
            f.write("-" * 50 + "\n")

            all_ops = passed_ops + failed_ops
            all_ops.sort(key=lambda x: extract_order(x[0]))  # 按order排序

            for op, passed, total in all_ops:
                pass_at_n = f"{passed}/{total}"
                status = "✅ 通过" if passed > 0 else "❌ 失败"
                f.write(f"{op:<30} {pass_at_n:<15} {status}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("报告生成完成! 🎉\n")

        print(f"📄 详细结果已保存到 {result_dir}/test_results.txt")
        print('=' * 80)

    # 返回统计信息供进一步处理
    return {
        'total_ops': len(op_stats),
        'passed_ops': len(passed_ops),
        'failed_ops': len(failed_ops),
        'pass_rate': len(passed_ops) / len(op_stats) if len(op_stats) > 0 else 0.0,
        'op_stats': op_stats,
        'passed_list': passed_ops,
        'failed_list': failed_ops
    }
