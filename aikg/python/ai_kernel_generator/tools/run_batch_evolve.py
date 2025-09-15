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
import sys
import subprocess
import json
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import traceback
from ai_kernel_generator import get_project_root

# ============================================================================
# 批量执行配置参数 - 在此处修改批量执行的配置
# ============================================================================

# 基础进化参数配置 - 对所有任务生效
EVOLVE_BASE_CONFIG = {
    "dsl": "triton",           # 实现类型: triton, swft, etc.
    "framework": "torch",      # 框架: torch, numpy, mindspore, etc.
    "backend": "ascend",       # 后端: ascend, cuda, etc.
    "arch": "ascend910b4"      # 架构: a100, ascend910b4, etc.
}

# 批量并行配置
BATCH_PARALLEL_NUM = 2  # batch级别的并行数（同时运行的evolve任务数）

# 任务目录和输出目录配置
TASK_DIR = "Path/to/your/tasks"  # 任务文件目录 - 请修改为实际路径
OUTPUT_DIR = "Path/to/your/batch_results"  # 输出目录 - 请修改为实际路径

# 设备池配置（循环分配给不同任务，避免并行冲突）
# 每个任务会分配一个设备，数量需要大于等于并行数
DEVICE_POOL = [4, 5]  # 可用设备列表

# 默认任务配置
DEFAULT_TASK_CONFIG = {
    "max_rounds": 2,
    "parallel_num": 2
}

# 每个任务的自定义配置（可选）
# 格式：{任务名: EvolveConfig参数字典}
TASK_CUSTOM_CONFIGS = {
    # 示例：为特定任务配置不同参数
    # "relu_task": {"max_rounds": 3, "parallel_num": 1},
    # "add_task": {"max_rounds": 2, "parallel_num": 2},
}


class BatchTaskPool:
    """批量任务池，用于管理并行执行的evolve任务"""

    def __init__(self, max_concurrency: int, device_pool: List[int]):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        # 动态设备池管理
        self.available_devices = asyncio.Queue()
        self.device_lock = asyncio.Lock()

        # 初始化设备池
        for device in device_pool:
            self.available_devices.put_nowait(device)

    async def acquire_device(self) -> int:
        """获取可用设备"""
        async with self.device_lock:
            device = await self.available_devices.get()
            return device

    async def release_device(self, device: int):
        """释放设备回池中"""
        async with self.device_lock:
            await self.available_devices.put(device)

    async def run_task_async(self, task_file: Path, output_dir: Path, index: int, total: int,
                             use_compact_output: bool = False) -> Dict[str, Any]:
        """异步运行单个任务"""
        # 获取设备
        device = await self.acquire_device()

        if not use_compact_output:
            print(f"🎯 任务 [{index}/{total}] {task_file.stem} 分配到设备 {device}")

        try:
            async with self.semaphore:
                # 在线程池中运行同步的任务
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    run_single_task_subprocess,
                    task_file, output_dir, index, total, use_compact_output, device
                )
                return result
        finally:
            # 确保设备被释放
            await self.release_device(device)
            if not use_compact_output:
                print(f"♻️  任务 {task_file.stem} 完成，设备 {device} 已回收")

    async def run_batch_parallel(self, task_files: List[Path], output_dir: Path) -> List[Dict[str, Any]]:
        """并行运行批量任务"""
        use_compact_output = self.max_concurrency > 1

        if use_compact_output:
            print(f"🚀 启动并行执行，最大并发数: {self.max_concurrency}")

        # 使用工作者队列模式避免设备数不足时的死锁
        task_queue = asyncio.Queue()
        results = [None] * len(task_files)

        # 将所有任务放入队列
        for i, task_file in enumerate(task_files):
            await task_queue.put((i, task_file))

        # 工作者函数
        async def worker():
            while True:
                try:
                    # 从队列获取任务，超时退出避免无限等待
                    task_index, task_file = await asyncio.wait_for(
                        task_queue.get(), timeout=1.0
                    )

                    # 执行任务
                    result = await self.run_task_async(
                        task_file, output_dir, task_index + 1, len(task_files), use_compact_output
                    )

                    # 保存结果
                    results[task_index] = result
                    task_queue.task_done()

                except asyncio.TimeoutError:
                    # 队列为空，工作者退出
                    break
                except Exception as e:
                    # 处理异常，确保任务标记完成
                    if 'task_index' in locals():
                        results[task_index] = {
                            'op_name': task_file.stem,
                            'task_file': str(task_file),
                            'success': False,
                            'execution_time': 0,
                            'error': f"Worker exception: {str(e)}",
                            'start_time': datetime.now().isoformat(),
                            'end_time': datetime.now().isoformat()
                        }
                        task_queue.task_done()

        # 创建工作者
        workers = []
        for _ in range(self.max_concurrency):
            workers.append(asyncio.create_task(worker()))

        # 等待所有任务完成
        await task_queue.join()

        # 清理工作者
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # 返回结果（异常已在工作者内部处理）
        return [r for r in results if r is not None]


def discover_task_files(task_dir: str) -> List[Path]:
    """发现所有任务文件"""
    task_path = Path(task_dir)
    if not task_path.exists():
        raise FileNotFoundError(f"任务目录不存在: {task_dir}")

    task_files = list(task_path.glob("*.py"))
    task_files.sort()  # 按文件名排序

    print(f"📁 发现 {len(task_files)} 个任务文件:")
    for i, file_path in enumerate(task_files, 1):
        print(f"  {i}. {file_path.name}")

    return task_files


def run_single_task_subprocess(task_file: Path, output_dir: Path, index: int, total: int,
                               use_compact_output: bool = False, device: int = 5) -> Dict[str, Any]:
    """使用subprocess方式运行单个任务"""
    op_name = "aikg_" + task_file.stem

    if not use_compact_output:
        print(f"\n" + "="*80)
        print(f"📋 开始执行任务 [{index}/{total}]: {op_name}")
        print("="*80)
    else:
        print(f"🔄 [{index}/{total}] {op_name}")

    start_time = datetime.now()

    # 准备输出文件路径（任务完成后保存）
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"output_{op_name}_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"

    try:
        # 设置环境变量
        env = os.environ.copy()
        # 检查必要的目录和文件
        tools_dir = Path(get_project_root()) / "tools"
        single_evolve_script = tools_dir / "single_evolve_runner.py"

        if not tools_dir.exists():
            raise FileNotFoundError(f"tools目录不存在: {tools_dir}")

        if not single_evolve_script.exists():
            raise FileNotFoundError(f"single_evolve_runner.py不存在: {single_evolve_script}")

        # 使用绝对路径
        absolute_task_file = Path(task_file).resolve()

        # 获取任务配置
        max_rounds = DEFAULT_TASK_CONFIG["max_rounds"]
        parallel_num = DEFAULT_TASK_CONFIG["parallel_num"]

        # 应用任务特定配置
        if op_name in TASK_CUSTOM_CONFIGS:
            custom_config = TASK_CUSTOM_CONFIGS[op_name]
            max_rounds = custom_config.get('max_rounds', max_rounds)
            parallel_num = custom_config.get('parallel_num', parallel_num)

        # 构建命令 - 传递完整的配置参数
        cmd = [
            sys.executable, str(single_evolve_script),
            op_name,                                    # 1. 算子名称
            str(absolute_task_file),                   # 2. 任务文件路径
            str(device),                               # 3. 设备ID
            str(max_rounds),                           # 4. 最大轮数
            str(parallel_num),                         # 5. 并行数
            EVOLVE_BASE_CONFIG["dsl"],                 # 6. DSL类型
            EVOLVE_BASE_CONFIG["framework"],           # 7. 框架
            EVOLVE_BASE_CONFIG["backend"],             # 8. 后端
            EVOLVE_BASE_CONFIG["arch"]                 # 9. 架构
        ]

        # 根据输出模式选择执行方式
        if use_compact_output:
            # 精简模式：静默执行，只捕获输出用于解析
            subprocess_result = subprocess.run(cmd, capture_output=True, text=True, env=env, errors='replace')
            result = {
                'returncode': subprocess_result.returncode,
                'stdout': subprocess_result.stdout,
                'stderr': subprocess_result.stderr
            }
        else:
            # 详细模式：实时显示所有输出并实时写入文件
            print(f"🔄 开始进化过程，实时输出:")
            print("-" * 60)

            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, env=env, errors='replace', bufsize=1, universal_newlines=True)

                stdout_lines = []
                for line in iter(process.stdout.readline, ''):
                    line = line.rstrip()
                    if line:
                        print(f"  {line}")
                        stdout_lines.append(line)

                process.wait()
                # 直接使用字典存储结果
                result = {
                    'returncode': process.returncode,
                    'stdout': '\n'.join(stdout_lines),
                    'stderr': ""
                }
            except Exception as e:
                print(f"执行过程中发生错误: {e}")
                subprocess_result = subprocess.run(cmd, capture_output=True, text=True, env=env, errors='replace')
                result = {
                    'returncode': subprocess_result.returncode,
                    'stdout': subprocess_result.stdout,
                    'stderr': subprocess_result.stderr
                }

                # 如果是fallback模式，也要写入文件
                if output_file and result['stdout']:
                    try:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(result['stdout'])
                            f.flush()
                    except Exception:
                        pass

            print("-" * 60)

        # 在精简模式下，解析并显示关键信息，同时实时写入文件
        if use_compact_output and result['stdout']:
            lines = result['stdout'].split('\n')
            for line in lines:
                line_clean = line.strip()
                # 只显示关键的轮次结果和最终结果，过滤掉纯分隔线和带标识码的行
                if any(keyword in line_clean for keyword in ['轮次', '最终全局最佳加速比', '🚀 加速比统计汇总', '进化完成！最终结果汇总']):
                    print(f"  {line_clean}")
                elif '进化完成' in line_clean and '最终结果汇总' in line_clean:
                    print(f"  {line_clean}")
                elif line_clean.startswith('算子名称:') or line_clean.startswith('总轮数:') or line_clean.startswith('成功任务数:'):
                    print(f"  {line_clean}")
                # 跳过纯分隔线、带标识码的行、以及其他格式化输出
                # elif line_clean and not (line_clean.startswith('=') or len(line_clean.split()) == 1 and '=' in line_clean):
                #     pass  # 其他行暂时不显示

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # 解析输出中的关键信息
        output_lines = result['stdout'].split('\n') if result['stdout'] else []
        success_rate = None
        best_speedup = None

        for line in output_lines:
            if "最终成功率:" in line or "完成" in line and "%" in line:
                try:
                    # 尝试从不同格式中提取成功率
                    if "最终成功率:" in line:
                        success_rate = float(line.split("最终成功率:")[1].split("%")[0].strip()) / 100
                    elif "完成" in line and "(" in line and "%" in line:
                        # 匹配 "完成 2/4(50%)" 格式
                        match = re.search(r'(\d+\.?\d*)%', line)
                        if match:
                            success_rate = float(match.group(1)) / 100
                except:
                    pass
            elif "最终全局最佳加速比:" in line or "最佳:" in line:
                try:
                    if "最终全局最佳加速比:" in line:
                        best_speedup = float(line.split("最终全局最佳加速比:")[1].split("x")[0].strip())
                    elif "最佳:" in line:
                        # 匹配 "最佳:1.23x" 格式
                        match = re.search(r'最佳:(\d+\.?\d*)x', line)
                        if match:
                            best_speedup = float(match.group(1))
                except:
                    pass

        # 判断任务是否真正成功：进程正常退出 且 有有效的加速比结果
        task_success = (result['returncode'] == 0 and
                        best_speedup is not None and
                        best_speedup > 0.0)

        # 任务完成后保存完整输出到文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"任务名称: {op_name}\n")
                f.write(f"任务文件: {task_file}\n")
                f.write(f"开始时间: {start_time.isoformat()}\n")
                f.write(f"结束时间: {end_time.isoformat()}\n")
                f.write(f"执行时间: {execution_time:.2f}秒\n")
                f.write(f"返回码: {result['returncode']}\n")
                f.write(f"任务成功: {'是' if task_success else '否'}\n")
                if best_speedup is not None:
                    f.write(f"最佳加速比: {best_speedup:.2f}x\n")
                f.write("\n" + "="*50 + " 完整输出 " + "="*50 + "\n")
                f.write(result['stdout'] or "")
                if result['stderr']:
                    f.write("\n" + "="*50 + " 错误输出 " + "="*50 + "\n")
                    f.write(result['stderr'])
        except Exception as e:
            print(f"⚠️  无法保存输出文件: {e}")
            output_file = None

        if result['returncode'] == 0:
            if task_success:
                if not use_compact_output:
                    print(f"✅ 任务 {op_name} 执行成功，最佳加速比: {best_speedup:.2f}x")
            else:
                if not use_compact_output:
                    print(f"⚠️  任务 {op_name} 进程正常结束，但未生成有效算子 (加速比: {best_speedup or 0.0:.2f}x)")

            return {
                'op_name': op_name,
                'task_file': str(task_file),
                'success': task_success,
                'execution_time': execution_time,
                'return_code': result['returncode'],
                'success_rate': success_rate,
                'best_speedup': best_speedup or 0.0,
                'output_file': str(output_file) if output_file else None,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
        else:
            if use_compact_output:
                print(f"❌ {op_name}: 执行失败(码:{result['returncode']})")
            else:
                print(f"❌ 任务 {op_name} 执行失败，返回码: {result['returncode']}")
                # 显示简要错误信息
                if result['stderr']:
                    stderr_lines = result['stderr'].strip().split('\n')[-3:]
                    for line in stderr_lines:
                        if line.strip():
                            print(f"   {line}")

            return {
                'op_name': op_name,
                'task_file': str(task_file),
                'success': False,
                'execution_time': execution_time,
                'return_code': result['returncode'],
                'error': f"Process failed with return code {result['returncode']}",
                'output_file': str(output_file) if output_file else None,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }

    except Exception as e:

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        error_msg = f"执行异常: {str(e)}"
        print(f"❌ {error_msg}")

        # 打印详细错误信息到控制台（调试模式）
        print("🔧 详细错误堆栈:")
        traceback.print_exc()

        # 保存异常信息到文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"任务名称: {op_name}\n")
                f.write(f"任务文件: {task_file}\n")
                f.write(f"开始时间: {start_time.isoformat()}\n")
                f.write(f"结束时间: {end_time.isoformat()}\n")
                f.write(f"执行时间: {execution_time:.2f}秒\n")
                f.write(f"任务成功: 否\n")
                f.write(f"错误信息: {error_msg}\n")
                f.write("\n" + "="*50 + " 错误详情 " + "="*50 + "\n")
                f.write(traceback.format_exc())
            print(f"📁 错误信息已保存到: {output_file}")
            error_file_path = str(output_file)
        except Exception as file_error:
            print(f"⚠️  无法保存错误文件: {file_error}")
            error_file_path = None

        return {
            'op_name': op_name,
            'task_file': str(task_file),
            'success': False,
            'execution_time': execution_time,
            'error': error_msg,
            'error_file': error_file_path,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }


def print_batch_summary(batch_results: List[Dict[str, Any]], total_start_time: datetime):
    """打印批量执行摘要"""
    successful_tasks = [r for r in batch_results if r.get('success', False)]
    failed_tasks = [r for r in batch_results if not r.get('success', False)]

    total_time = (datetime.now() - total_start_time).total_seconds()

    print("\n" + "="*100)
    print("🎯 批量执行完成！最终统计报告")
    print("="*100)
    print(f"总任务数: {len(batch_results)}")
    print(f"成功任务数: {len(successful_tasks)}")
    print(f"失败任务数: {len(failed_tasks)}")
    print(f"成功率: {len(successful_tasks)/len(batch_results):.2%}")
    print(f"总执行时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")

    # 显示成功任务的性能统计
    if successful_tasks:
        print(f"\n🏆 成功任务性能统计:")
        performance_data = []
        for task in successful_tasks:
            if task.get('best_speedup'):
                performance_data.append({
                    'name': task['op_name'],
                    'speedup': task['best_speedup'],
                    'success_rate': task.get('success_rate', 0),
                    'time': task.get('execution_time', 0)
                })

        if performance_data:
            performance_data.sort(key=lambda x: x['speedup'], reverse=True)
            for i, perf in enumerate(performance_data[:10], 1):
                print(f"  {i:2d}. {perf['name']:<20} - {perf['speedup']:6.2f}x "
                      f"(成功率: {perf['success_rate']:.1%}, 时间: {perf['time']:.1f}s)")

    # 显示失败任务
    if failed_tasks:
        print(f"\n❌ 失败任务列表:")
        for task in failed_tasks:
            error_msg = task.get('error', 'Unknown error')
            print(f"  • {task['op_name']}: {error_msg}")

    print("="*100)


def main():
    """主函数"""
    # 使用硬编码配置
    task_dir = TASK_DIR
    output_dir = Path(OUTPUT_DIR)
    parallel_num = BATCH_PARALLEL_NUM

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 开始批量进化执行")
    print("="*80)
    print(f"任务目录: {task_dir}")
    print(f"输出目录: {output_dir}")
    print(f"并行数: {parallel_num}")
    print(f"设备池: {DEVICE_POOL}")
    print("="*80)

    total_start_time = datetime.now()
    batch_results = []

    try:
        # 发现所有任务文件
        task_files = discover_task_files(task_dir)

        if not task_files:
            print("❌ 未找到任何.py文件")
            return

        # 创建批量任务池（传入设备池）
        batch_pool = BatchTaskPool(max_concurrency=parallel_num, device_pool=DEVICE_POOL)

        if parallel_num <= 1:
            print(f"\n📋 将按顺序执行 {len(task_files)} 个算子的进化流程...")
        else:
            print(f"\n🚀 将并行执行 {len(task_files)} 个算子的进化流程...")
            print(f"📱 设备动态分配：{DEVICE_POOL} (任务完成后自动回收)")

        # 运行任务
        try:
            batch_results = asyncio.run(batch_pool.run_batch_parallel(task_files, output_dir))
        except Exception as e:
            print(f"❌ 执行过程中发生错误: {e}")
            traceback.print_exc()
            return

        # 生成并保存批量摘要
        summary_data = {
            'batch_info': {
                'total_tasks': len(batch_results),
                'successful_tasks': len([r for r in batch_results if r.get('success', False)]),
                'failed_tasks': len([r for r in batch_results if not r.get('success', False)]),
                'total_execution_time_seconds': (datetime.now() - total_start_time).total_seconds(),
                'start_time': total_start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            },
            'task_results': batch_results
        }

        summary_file = output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print_batch_summary(batch_results, total_start_time)
        print(f"\n💾 批量摘要已保存到: {summary_file}")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\n❌ 批量执行过程中发生错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
