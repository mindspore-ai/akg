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

"""
Runner管理模块

整合单任务和批量执行的配置管理、CLI解析和执行逻辑
"""

import os
import sys
import asyncio
import subprocess
import json
import re
import yaml
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from ai_kernel_generator import get_project_root
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.evolve import evolve
from ai_kernel_generator.core.worker.manager import get_worker_manager
from ai_kernel_generator.utils.environment_check import check_env_for_task
from .result_collector import RealtimeResultCollector


# ============================================================================
# 配置管理
# ============================================================================

class RunnerConfig:
    """进化Runner配置参数类"""

    def __init__(self):
        # 基本参数
        self.dsl = "triton_cuda"
        self.framework = "torch"
        self.backend = "cuda"
        self.arch = "a100"

        # 进化参数
        self.max_rounds = 5
        self.parallel_num = 4

        # 岛屿模型参数
        self.num_islands = 1
        self.migration_interval = 0
        self.elite_size = 0
        self.parent_selection_prob = 0.5
        
        # 手写建议采样参数
        self.handwrite_decay_rate = 2.0

        # 设备配置
        self.device_list = [0]

        # 配置文件路径
        self.config_path = "config/default_evolve_config.yaml"

        # 任务配置
        self.op_name = "relu_op"
        self.task_desc = "Path/to/your/tasks/relu_task.py"

    @classmethod
    def from_yaml(cls, config_path: str, skip_task_config: bool = False) -> 'RunnerConfig':
        """从YAML配置文件加载配置"""
        config = cls()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            # 基础配置
            if 'base' in yaml_config:
                base = yaml_config['base']
                config.dsl = base.get('dsl', config.dsl)
                config.framework = base.get('framework', config.framework)
                config.backend = base.get('backend', config.backend)
                config.arch = base.get('arch', config.arch)
                config_path_value = base.get('config_path', config.config_path)
                if config_path_value and not Path(config_path_value).is_absolute():
                    config.config_path = str(Path(get_project_root()) / config_path_value)
                else:
                    config.config_path = config_path_value
            else:
                raise ValueError("base section not found in config file")

            # 进化参数
            if 'evolve' in yaml_config:
                evolve_config = yaml_config['evolve']
                config.max_rounds = evolve_config.get('max_rounds', config.max_rounds)
                config.parallel_num = evolve_config.get('parallel_num', config.parallel_num)
            else:
                raise ValueError("evolve section not found in config file")

            # 岛屿模型配置
            if 'island' in yaml_config:
                island_config = yaml_config['island']
                config.num_islands = island_config.get('num_islands', config.num_islands)
                config.migration_interval = island_config.get('migration_interval', config.migration_interval)
                config.elite_size = island_config.get('elite_size', config.elite_size)
                config.parent_selection_prob = island_config.get('parent_selection_prob', config.parent_selection_prob)
            else:
                raise ValueError("island section not found in config file")

            # 设备配置
            if 'devices' in yaml_config:
                config.device_list = yaml_config['devices'].get('device_list', config.device_list)
            else:
                raise ValueError("devices section not found in config file")

            # 任务配置（仅在非批量调用模式下加载）
            if not skip_task_config and 'task' in yaml_config:
                task_config = yaml_config['task']
                config.op_name = task_config.get('op_name', config.op_name)

                task_desc_value = task_config.get('task_desc', config.task_desc)
                if task_desc_value and isinstance(task_desc_value, str):
                    try:
                        with open(task_desc_value, 'r', encoding='utf-8') as f:
                            config.task_desc = f.read().strip()
                    except Exception as e:
                        print(f"Error: Failed to read task description file {task_desc_value}: {e}")
                        config.task_desc = None

            return config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}, using default config: {e}")
            return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'dsl': self.dsl,
            'framework': self.framework,
            'backend': self.backend,
            'arch': self.arch,
            'max_rounds': self.max_rounds,
            'parallel_num': self.parallel_num,
            'num_islands': self.num_islands,
            'migration_interval': self.migration_interval,
            'elite_size': self.elite_size,
            'parent_selection_prob': self.parent_selection_prob,
            'device_list': self.device_list,
            'config_path': self.config_path,
            'op_name': self.op_name,
            'task_desc': self.task_desc
        }


def apply_custom_task_config(config: RunnerConfig, config_path: str, op_name: str) -> None:
    """应用自定义任务配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)

        if 'custom_tasks' in yaml_config and yaml_config['custom_tasks']:
            if op_name in yaml_config['custom_tasks']:
                custom_config = yaml_config['custom_tasks'][op_name]
                print(f"发现自定义配置 for {op_name}: {custom_config}")

                config_mapping = {
                    'max_rounds': 'max_rounds',
                    'parallel_num': 'parallel_num',
                    'num_islands': 'num_islands',
                    'migration_interval': 'migration_interval',
                    'elite_size': 'elite_size',
                    'parent_selection_prob': 'parent_selection_prob'
                }

                for config_key, attr_name in config_mapping.items():
                    if config_key in custom_config:
                        setattr(config, attr_name, custom_config[config_key])
                        print(f"   自定义 {config_key}: {custom_config[config_key]}")

                print(f"已应用自定义配置")

    except Exception as e:
        print(f"提示: 无法解析custom_tasks配置: {e}")


def load_task_description(task_file: str) -> str:
    """加载任务描述文件"""
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"任务文件不存在: {task_file}")
    except Exception as e:
        raise Exception(f"读取任务文件失败: {e}")


# ============================================================================
# 结果打印函数
# ============================================================================

def print_evolve_config(op_name: str, evolve_config: RunnerConfig) -> None:
    """打印进化配置信息"""
    print("="*80)
    print("AI KERNEL GENERATOR - 统一进化式算子生成")
    print("="*80)
    print(f"算子名称: {op_name}")
    print(f"实现类型: {evolve_config.dsl}")
    print(f"框架: {evolve_config.framework}")
    print(f"后端: {evolve_config.backend}")
    print(f"架构: {evolve_config.arch}")
    print(f"进化轮数: {evolve_config.max_rounds}")
    print(f"并行任务数: {evolve_config.parallel_num}")

    if evolve_config.num_islands <= 1:
        print("岛屿模型: 禁用（简单进化模式）")
    else:
        print(f"岛屿数量: {evolve_config.num_islands}")
        if evolve_config.migration_interval <= 0:
            print("迁移: 禁用")
        else:
            print(f"迁移间隔: {evolve_config.migration_interval}")

    if evolve_config.elite_size <= 0:
        print("精英机制: 禁用")
    else:
        print(f"精英数量: {evolve_config.elite_size}")

    if evolve_config.num_islands > 1 and evolve_config.elite_size > 0:
        print(f"父代选择概率: {evolve_config.parent_selection_prob}")
    print("="*80)


def print_evolution_result(evolution_result: Dict[str, Any], evolve_config: RunnerConfig) -> Dict[str, Any]:
    """打印进化结果信息"""
    if not evolution_result:
        print("\n进化过程返回空结果")
        return {}

    print("\n" + "="*80)
    print("进化完成！最终结果汇总:")
    print("="*80)
    print(f"算子名称: {evolution_result.get('op_name', 'Unknown')}")
    print(f"总轮数: {evolution_result.get('total_rounds', 0)}")
    print(f"总任务数: {evolution_result.get('total_tasks', 0)}")
    print(f"成功任务数: {evolution_result.get('successful_tasks', 0)}")
    print(f"最终成功率: {evolution_result.get('final_success_rate', 0.0):.2%}")
    print(f"最佳成功率: {evolution_result.get('best_success_rate', 0.0):.2%}")
    print(f"实现类型: {evolution_result.get('implementation_type', 'Unknown')}")
    print(f"框架: {evolution_result.get('framework', 'Unknown')}")
    print(f"后端: {evolution_result.get('backend', 'Unknown')}")
    print(f"架构: {evolution_result.get('architecture', 'Unknown')}")

    # 岛屿信息
    island_info = evolution_result.get('island_info', {})
    if island_info:
        num_islands_used = island_info.get('num_islands', 'N/A')
        if num_islands_used <= 1:
            print("进化模式: 简单进化（无岛屿模型）")
        else:
            print(f"岛屿数量: {num_islands_used}")
            migration_interval_used = island_info.get('migration_interval', 'N/A')
            if migration_interval_used <= 0:
                print("迁移: 禁用")
            else:
                print(f"迁移间隔: {migration_interval_used}")

            elite_size_used = island_info.get('elite_size', 'N/A')
            if elite_size_used <= 0:
                print("精英机制: 禁用")
            else:
                print(f"精英数量: {elite_size_used}")

    # 显示存储目录信息
    storage_dir = evolution_result.get('storage_dir', '')
    if storage_dir:
        print(f"存储目录: {storage_dir}")
    
    task_folder = evolution_result.get('task_folder', '')
    if task_folder:
        print(f"Task文件夹: {task_folder}")
    
    log_dir = evolution_result.get('log_dir', '')
    if log_dir:
        print(f"Log目录: {log_dir}")

    # 显示最佳实现
    best_implementations = evolution_result.get('best_implementations', [])
    if best_implementations:
        print(f"\n最佳实现 (前{len(best_implementations)}个):")
        for i, impl in enumerate(best_implementations, 1):
            profile_data = impl.get('profile', {})

            if isinstance(profile_data, dict):
                gen_time = profile_data.get('gen_time', float('inf'))
                base_time = profile_data.get('base_time', 0.0)
                speedup = profile_data.get('speedup', 0.0)
                
                if gen_time != float('inf'):
                    profile_str = f"生成代码: {gen_time:.4f}us, 基准代码: {base_time:.4f}us, 加速比: {speedup:.2f}x"
                else:
                    profile_str = "性能: N/A"
            else:
                profile_str = "性能: N/A"

            info_parts = [f"{impl.get('op_name', 'Unknown')} (轮次 {impl.get('round', 'N/A')}"]

            if evolution_result.get('island_info', {}).get('num_islands', 1) > 1:
                source_island = impl.get('source_island', 'N/A')
                info_parts.append(f"来源岛屿 {source_island}")
            
            unique_dir = impl.get('unique_dir', 'N/A')
            info_parts.append(f"个体路径: {unique_dir}")

            info_parts.append(profile_str)
            print(f"  {i}. {', '.join(info_parts)}")
    else:
        print("\n没有找到成功的实现")

    # 显示每轮详细结果
    round_results = evolution_result.get('round_results', [])
    if round_results:
        print(f"\n每轮详细结果:")
        for round_result in round_results:
            round_num = round_result.get('round', 'N/A')
            success_rate = round_result.get('success_rate', 0.0)
            successful = round_result.get('successful_tasks', 0)
            total = round_result.get('total_tasks', 0)
            print(f"  轮次 {round_num}: {successful}/{total} 成功 ({success_rate:.2%})")

    print("="*80)

    return evolution_result


# ============================================================================
# 单任务执行
# ============================================================================

async def run_single_evolve(op_name: str = None, task_desc: str = None, evolve_config: RunnerConfig = None) -> Dict[str, Any]:
    """运行自定义任务的进化过程"""
    if evolve_config is None:
        evolve_config = RunnerConfig()

    if op_name is None:
        op_name = evolve_config.op_name
    if task_desc is None:
        task_desc = evolve_config.task_desc

    print_evolve_config(op_name, evolve_config)

    # 初始化资源
    task_pool = TaskPool(max_concurrency=evolve_config.parallel_num)

    config = load_config(config_path=evolve_config.config_path)
    
    # 判断是否为远程模式（通过环境变量）
    is_remote = os.getenv("AIKG_WORKER_URL") is not None
    check_env_for_task(evolve_config.framework, evolve_config.backend, evolve_config.dsl, config, is_remote=is_remote)

    # 确保在运行任务前已注册 Worker
    if not await get_worker_manager().has_worker(backend=evolve_config.backend, arch=evolve_config.arch):
        raise RuntimeError(
            f"未检测到可用的 Worker。请先注册 Worker 后再调用 run_single_evolve：\n"
            f"  from ai_kernel_generator.core.worker.manager import register_worker\n"
            f"  await register_worker(backend='{evolve_config.backend}', arch='{evolve_config.arch}', device_ids=[0])\n"
            f"或设置环境变量 AIKG_WORKER_URL 指向远程 Worker 服务。"
        )

    # 运行进化过程
    print("开始进化过程...")
    evolution_result = await evolve(
        op_name=op_name,
        task_desc=task_desc,
        dsl=evolve_config.dsl,
        framework=evolve_config.framework,
        backend=evolve_config.backend,
        arch=evolve_config.arch,
        config=config,
        task_pool=task_pool,
        max_rounds=evolve_config.max_rounds,
        parallel_num=evolve_config.parallel_num,
        num_islands=evolve_config.num_islands,
        migration_interval=evolve_config.migration_interval,
        elite_size=evolve_config.elite_size,
        parent_selection_prob=evolve_config.parent_selection_prob,
        handwrite_decay_rate=evolve_config.handwrite_decay_rate
    )

    return print_evolution_result(evolution_result, evolve_config)


# ============================================================================
# 批量执行
# ============================================================================

class BatchTaskPool:
    """批量任务池，用于管理并行执行的 evolve 任务"""

    def __init__(
        self,
        max_concurrency: int,
        config_path: Optional[str] = None,
        collector: Optional[RealtimeResultCollector] = None
    ):
        self.max_concurrency = max_concurrency
        self.config_path = config_path
        self.collector = collector
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run_task_async(
        self,
        task_file: Path,
        output_dir: Path,
        index: int,
        total: int,
        use_compact_output: bool = False
    ) -> Dict[str, Any]:
        """异步运行单个任务"""
        if not use_compact_output:
            print(f"任务 [{index}/{total}] {task_file.stem} 开始执行")

        async with self.semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                run_single_task_subprocess,
                task_file,
                output_dir,
                index,
                total,
                use_compact_output,
                self.config_path
            )

            # 任务完成后立即收集结果
            if self.collector and result.get('success', False):
                try:
                    output_file = result.get('output_file')
                    output_content = ""
                    if output_file and Path(output_file).exists():
                        with open(output_file, 'r', encoding='utf-8') as f:
                            output_content = f.read()

                    await loop.run_in_executor(
                        None,
                        self.collector.collect_task_result,
                        result['op_name'],
                        output_content
                    )
                except Exception as e:
                    print(f"收集结果失败: {e}")

            return result

    async def run_batch_parallel(self, task_files: List[Path], output_dir: Path) -> List[Dict[str, Any]]:
        """并行运行批量任务"""
        use_compact_output = self.max_concurrency > 1

        if use_compact_output:
            print(f"启动并行执行，最大并发数: {self.max_concurrency}")

        task_queue = asyncio.Queue()
        results = [None] * len(task_files)

        for i, task_file in enumerate(task_files):
            await task_queue.put((i, task_file))

        async def worker():
            while True:
                try:
                    task_index, task_file = await asyncio.wait_for(
                        task_queue.get(), timeout=1.0
                    )

                    result = await self.run_task_async(
                        task_file, output_dir, task_index + 1, len(task_files), use_compact_output
                    )

                    results[task_index] = result
                    task_queue.task_done()

                except asyncio.TimeoutError:
                    break
                except Exception as e:
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

        workers = []
        for _ in range(self.max_concurrency):
            workers.append(asyncio.create_task(worker()))

        await task_queue.join()

        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        return [r for r in results if r is not None]


def discover_task_files(task_dir: str) -> List[Path]:
    """发现所有任务文件"""
    task_path = Path(task_dir)
    if not task_path.exists():
        raise FileNotFoundError(f"任务目录不存在: {task_dir}")

    task_files = list(task_path.glob("*.py"))
    task_files.sort()

    print(f"发现 {len(task_files)} 个任务文件:")
    for i, file_path in enumerate(task_files, 1):
        print(f"  {i}. {file_path.name}")

    return task_files


def run_single_task_subprocess(task_file: Path, output_dir: Path, index: int, total: int,
                               use_compact_output: bool = False, config_path: Optional[str] = None) -> Dict[str, Any]:
    """使用subprocess方式运行单个任务"""
    op_name = "aikg_" + task_file.stem

    if not use_compact_output:
        print(f"\n" + "="*80)
        print(f"开始执行任务 [{index}/{total}]: {op_name}")
        print("="*80)
    else:
        print(f"[{index}/{total}] {op_name}")

    start_time = datetime.now()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"output_{op_name}_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"

    try:
        env = os.environ.copy()
        # get_project_root() 返回 python/ai_kernel_generator，需要向上两级到达真正的项目根目录
        project_root = Path(get_project_root()).parent.parent
        tools_dir = project_root / "tools"
        single_evolve_script = tools_dir / "run_single_evolve.py"

        if not tools_dir.exists():
            raise FileNotFoundError(f"tools目录不存在: {tools_dir}")

        if not single_evolve_script.exists():
            raise FileNotFoundError(f"run_single_evolve.py不存在: {single_evolve_script}")

        absolute_task_file = Path(task_file).resolve()

        cmd = [
            sys.executable,
            str(single_evolve_script),
            op_name,
            str(absolute_task_file)
        ]

        if config_path:
            cmd.append(config_path)

        if use_compact_output:
            subprocess_result = subprocess.run(cmd, capture_output=True, text=True, env=env, errors='replace')
            result = {
                'returncode': subprocess_result.returncode,
                'stdout': subprocess_result.stdout,
                'stderr': subprocess_result.stderr
            }
        else:
            print(f"开始进化过程，实时输出:")
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

                if output_file and result['stdout']:
                    try:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(result['stdout'])
                            f.flush()
                    except Exception:
                        pass

            print("-" * 60)

        # 在精简模式下显示关键信息
        if use_compact_output and result['stdout']:
            lines = result['stdout'].split('\n')
            for line in lines:
                line_clean = line.strip()
                if any(keyword in line_clean for keyword in ['轮次', '最终全局最佳加速比', '加速比统计汇总', '进化完成！最终结果汇总']):
                    print(f"  {line_clean}")
                elif '进化完成' in line_clean and '最终结果汇总' in line_clean:
                    print(f"  {line_clean}")
                elif line_clean.startswith('算子名称:') or line_clean.startswith('总轮数:') or line_clean.startswith('成功任务数:'):
                    print(f"  {line_clean}")

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # 解析输出中的关键信息
        output_lines = result['stdout'].split('\n') if result['stdout'] else []
        success_rate = None
        best_speedup = None

        for line in output_lines:
            if "最终成功率:" in line or "完成" in line and "%" in line:
                try:
                    if "最终成功率:" in line:
                        success_rate = float(line.split("最终成功率:")[1].split("%")[0].strip()) / 100
                    elif "完成" in line and "(" in line and "%" in line:
                        match = re.search(r'(\d+\.?\d*)%', line)
                        if match:
                            success_rate = float(match.group(1)) / 100
                except:
                    pass
            elif "最终全局最佳加速比:" in line or "最佳:" in line or "加速比:" in line:
                try:
                    if "最终全局最佳加速比:" in line:
                        best_speedup = float(line.split("最终全局最佳加速比:")[1].split("x")[0].strip())
                    elif "最佳:" in line:
                        match = re.search(r'最佳:(\d+\.?\d*)x', line)
                        if match:
                            best_speedup = float(match.group(1))
                    elif "加速比:" in line:
                        match = re.search(r'加速比:\s*(\d+\.?\d*)x', line)
                        if match:
                            speedup_value = float(match.group(1))
                            if best_speedup is None or speedup_value > best_speedup:
                                best_speedup = speedup_value
                except:
                    pass

        task_success = (result['returncode'] == 0 and
                        best_speedup is not None and
                        best_speedup > 0.0)

        # 保存完整输出
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
            print(f"无法保存输出文件: {e}")
            output_file = None

        if result['returncode'] == 0:
            if task_success:
                if not use_compact_output:
                    print(f"任务 {op_name} 执行成功，最佳加速比: {best_speedup:.2f}x")
            else:
                if not use_compact_output:
                    print(f"任务 {op_name} 进程正常结束，但未生成有效算子 (加速比: {best_speedup or 0.0:.2f}x)")

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
                print(f"{op_name}: 执行失败(码:{result['returncode']})")
            else:
                print(f"任务 {op_name} 执行失败，返回码: {result['returncode']}")
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
        print(f"{error_msg}")

        print("详细错误堆栈:")
        traceback.print_exc()

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"任务名称: {op_name}\n")
                f.write(f"任务文件: {task_file}\n")
                f.write(f"开始时间: {start_time.isoformat()}\n")
                f.write(f"结束时间: {end_time.isoformat()}\n")
                f.write(f"执行时间: {execution_time:.2f}微秒\n")
                f.write(f"任务成功: 否\n")
                f.write(f"错误信息: {error_msg}\n")
                f.write("\n" + "="*50 + " 错误详情 " + "="*50 + "\n")
                f.write(traceback.format_exc())
            print(f"错误信息已保存到: {output_file}")
            error_file_path = str(output_file)
        except Exception as file_error:
            print(f"无法保存错误文件: {file_error}")
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


def load_batch_config(config_path: str = None) -> Tuple[Dict[str, Any], str]:
    """加载批量执行配置

    Returns:
        Tuple[Dict[str, Any], str]: 批量配置字典和实际使用的配置文件路径
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = os.path.join(project_root, "config", "evolve_config.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if 'batch' not in config:
                raise ValueError("配置文件中缺少 'batch' 部分")

            batch_config = config['batch']

            required_keys = ['parallel_num', 'device_pool', 'task_dir', 'output_dir']
            missing_keys = [key for key in required_keys if key not in batch_config]
            if missing_keys:
                raise ValueError(f"配置文件中 'batch' 部分缺少必要的配置项: {missing_keys}")

            config_dict = {
                "batch_parallel_num": batch_config['parallel_num'],
                "task_dir": batch_config['task_dir'],
                "output_dir": batch_config['output_dir'],
                "device_pool": batch_config['device_pool']
            }

            print(f"成功加载配置文件: {config_path}")
            print(f"   任务目录: {config_dict['task_dir']}")
            print(f"   输出目录: {config_dict['output_dir']}")
            print(f"   设备池: {config_dict['device_pool']}")
            print(f"   批量并行数: {config_dict['batch_parallel_num']}")

        except Exception as e:
            print(f"错误: 无法加载配置文件 {config_path}: {e}")
            raise
    else:
        print(f"错误: 配置文件不存在: {config_path}")
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    return config_dict, config_path


def print_batch_summary(batch_results: List[Dict[str, Any]], total_start_time: datetime):
    """打印批量执行摘要"""
    successful_tasks = [r for r in batch_results if r.get('success', False)]
    failed_tasks = [r for r in batch_results if not r.get('success', False)]

    total_time = (datetime.now() - total_start_time).total_seconds()

    print("\n" + "="*100)
    print("批量执行完成！最终统计报告")
    print("="*100)
    print(f"总任务数: {len(batch_results)}")
    print(f"成功任务数: {len(successful_tasks)}")
    print(f"失败任务数: {len(failed_tasks)}")
    print(f"成功率: {len(successful_tasks)/len(batch_results):.2%}")
    print(f"总执行时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")

    # 显示成功任务的性能统计
    if successful_tasks:
        print(f"\n成功任务性能统计:")
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
        print(f"\n失败任务列表:")
        for task in failed_tasks:
            error_msg = task.get('error', 'Unknown error')
            print(f"  • {task['op_name']}: {error_msg}")

    print("="*100)


async def run_batch_evolve(config_path: str = None) -> None:
    """批量执行进化任务"""
    config, resolved_config_path = load_batch_config(config_path)

    task_dir = os.path.expanduser(config["task_dir"]) if config["task_dir"] else os.path.expanduser("~/aikg_tasks")
    output_dir = Path(os.path.expanduser(config["output_dir"])) if config["output_dir"] else Path(
        os.path.expanduser("~/aikg_batch_results"))
    parallel_num = config["batch_parallel_num"] if config["batch_parallel_num"] > 0 else 2

    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始批量进化执行")
    print("="*80)
    print(f"任务目录: {task_dir}")
    print(f"输出目录: {output_dir}")
    print(f"并行数: {parallel_num}")
    print(f"设备配置: {config['device_pool'] or '使用 AIKG_WORKER_URL 指定远程 Worker'}")
    print("="*80)

    total_start_time = datetime.now()
    batch_results = []

    try:
        task_files = discover_task_files(task_dir)

        if not task_files:
            print("未找到任何.py文件")
            return

        # 尝试检查是否有可用 Worker，但不强制依赖 WorkerManager 实例
        # (因为此时可能还未连接到 Remote，但为了兼容性，我们假设外部已注册)
        # 这里不再主动调用 register_worker

        # 初始化实时结果收集器
        collector = RealtimeResultCollector(output_dir)
        print(f"实时结果收集器已启动")
        print(f"   TXT输出: {collector.txt_file}")
        print(f"   CSV输出: {collector.csv_file}")
        print("="*80)

        # 创建批量任务池
        batch_pool = BatchTaskPool(
            max_concurrency=parallel_num,
            config_path=resolved_config_path,
            collector=collector
        )

        if parallel_num <= 1:
            print(f"\n将按顺序执行 {len(task_files)} 个算子的进化流程...")
        else:
            print(f"\n将并行执行 {len(task_files)} 个算子的进化流程...")
            print("Worker 分配：由 WorkerManager 统一调度")

        # 运行任务
        try:
            batch_results = await batch_pool.run_batch_parallel(task_files, output_dir)
        except Exception as e:
            print(f"执行过程中发生错误: {e}")
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
        print(f"\n批量摘要已保存到: {summary_file}")

    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n批量执行过程中发生错误: {e}")
        traceback.print_exc()

