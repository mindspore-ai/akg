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
批量自适应搜索执行脚本

支持多算子并行调优，自动管理设备分配，避免设备冲突。

用法:
  python run_batch_adaptive_search.py                    # 使用默认配置
  python run_batch_adaptive_search.py <config_file>      # 使用自定义配置文件

配置参数说明:
  batch.parallel_num: 同时运行的算子数量（需 <= device_pool 长度）
  batch.device_pool: 设备池（如 [0, 1, 2, 3]）
  batch.task_dir: 任务文件目录
  batch.output_dir: 结果输出目录
  
  concurrency.max_concurrent: 每个算子搜索时的最大并发任务数
  concurrency.initial_task_count: 每个算子搜索时的初始任务数
  stopping.max_total_tasks: 每个算子搜索时的最大任务数
"""

import sys
import os
import asyncio
import subprocess
import json
import re
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from akg_agents import get_project_root
from akg_agents.core.worker.manager import register_worker
from akg_agents.utils.common_utils import load_yaml


# ============================================================================
# 配置定义
# ============================================================================

@dataclass
class BatchAdaptiveSearchConfig:
    """批量自适应搜索配置"""
    # 环境配置
    dsl: str = "triton_ascend"
    framework: str = "torch"
    backend: str = "ascend"
    arch: str = "ascend910b4"
    
    # 批量执行配置
    batch_parallel_num: int = 2          # 同时执行的算子数量
    device_pool: List[int] = field(default_factory=lambda: [0, 1])
    task_dir: str = ""                   # 任务文件目录
    output_dir: str = ""                 # 输出目录
    
    # 搜索参数（每个算子的搜索配置）
    max_concurrent: int = 4              # 每个算子搜索时的最大并发任务数
    initial_task_count: int = 4          # 初始生成的任务数
    max_total_tasks: int = 50            # 每个算子的最大搜索任务数
    
    # UCB 参数
    exploration_coef: float = 1.414
    random_factor: float = 0.1
    use_softmax: bool = False
    softmax_temperature: float = 1.0
    
    # 灵感采样参数
    inspiration_sample_num: int = 3
    use_tiered_sampling: bool = True
    handwrite_sample_num: int = 2
    handwrite_decay_rate: float = 2.0
    
    # 基础配置文件路径
    config_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "BatchAdaptiveSearchConfig":
        """从 YAML 文件加载配置"""
        config_dict = load_yaml(config_path)
        
        instance = cls()
        
        # 环境配置
        env_config = config_dict.get("environment", {})
        instance.dsl = env_config.get("dsl", instance.dsl)
        instance.framework = env_config.get("framework", instance.framework)
        instance.backend = env_config.get("backend", instance.backend)
        instance.arch = env_config.get("arch", instance.arch)
        
        # 批量执行配置
        batch_config = config_dict.get("batch", {})
        instance.batch_parallel_num = batch_config.get("parallel_num", instance.batch_parallel_num)
        instance.device_pool = batch_config.get("device_pool", instance.device_pool)
        instance.task_dir = batch_config.get("task_dir", instance.task_dir)
        instance.output_dir = batch_config.get("output_dir", instance.output_dir)
        
        # 并发配置
        concurrency_config = config_dict.get("concurrency", {})
        instance.max_concurrent = concurrency_config.get("max_concurrent", instance.max_concurrent)
        instance.initial_task_count = concurrency_config.get("initial_task_count", instance.initial_task_count)
        
        # 停止条件
        stopping_config = config_dict.get("stopping", {})
        instance.max_total_tasks = stopping_config.get("max_total_tasks", instance.max_total_tasks)
        
        # UCB 参数
        ucb_config = config_dict.get("ucb_selection", {})
        instance.exploration_coef = ucb_config.get("exploration_coef", instance.exploration_coef)
        instance.random_factor = ucb_config.get("random_factor", instance.random_factor)
        instance.use_softmax = ucb_config.get("use_softmax", instance.use_softmax)
        instance.softmax_temperature = ucb_config.get("softmax_temperature", instance.softmax_temperature)
        
        # 灵感采样参数
        inspiration_config = config_dict.get("inspiration", {})
        instance.inspiration_sample_num = inspiration_config.get("sample_num", instance.inspiration_sample_num)
        instance.use_tiered_sampling = inspiration_config.get("use_tiered_sampling", instance.use_tiered_sampling)
        
        # 手写建议参数
        handwrite_config = config_dict.get("handwrite", {})
        instance.handwrite_sample_num = handwrite_config.get("sample_num", instance.handwrite_sample_num)
        instance.handwrite_decay_rate = handwrite_config.get("decay_rate", instance.handwrite_decay_rate)
        
        # 基础配置文件路径
        instance.config_path = config_dict.get("config_path")
        
        return instance


# ============================================================================
# 结果收集器
# ============================================================================

class AdaptiveSearchResultCollector:
    """实时收集和保存自适应搜索结果"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 汇总文件
        self.summary_file = output_dir / f"batch_summary_{timestamp}.txt"
        self.csv_file = output_dir / f"batch_results_{timestamp}.csv"
        self.json_file = output_dir / f"batch_results_{timestamp}.json"
        
        # 初始化 CSV 文件
        self._init_csv_file()
        
        # 存储所有结果
        self.results: List[Dict[str, Any]] = []
    
    def _init_csv_file(self):
        """初始化 CSV 文件"""
        headers = [
            "op_name", "success", "total_submitted", "total_success", "total_failed",
            "success_rate", "elapsed_time", "best_speedup", "best_gen_time",
            "storage_dir", "execution_time", "start_time", "end_time", "error"
        ]
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def add_result(self, result: Dict[str, Any]):
        """添加一个任务结果"""
        self.results.append(result)
        
        # 格式化 gen_time（处理 0 或 inf 的情况）
        gen_time = result.get('best_gen_time', 0)
        if gen_time > 0 and gen_time != float('inf'):
            gen_time_str = f"{gen_time:.4f}us"
        else:
            gen_time_str = "N/A"
        
        # 追加到 CSV
        row = [
            result.get('op_name', ''),
            result.get('success', False),
            result.get('total_submitted', 0),
            result.get('total_success', 0),
            result.get('total_failed', 0),
            f"{result.get('success_rate', 0):.2%}",
            f"{result.get('elapsed_time', 0):.1f}s",
            f"{result.get('best_speedup', 0):.2f}x",
            gen_time_str,
            result.get('storage_dir', ''),
            f"{result.get('execution_time', 0):.1f}s",
            result.get('start_time', ''),
            result.get('end_time', ''),
            result.get('error', '')
        ]
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # 打印进度
        print(f"[{len(self.results)}] {result.get('op_name', 'Unknown')}: "
              f"{'✅ 成功' if result.get('success') else '❌ 失败'} | "
              f"成功率 {result.get('success_rate', 0):.1%} | "
              f"最佳加速比 {result.get('best_speedup', 0):.2f}x | "
              f"gen_time {gen_time_str}")
    
    def save_final_summary(self, total_start_time: datetime):
        """保存最终汇总"""
        total_time = (datetime.now() - total_start_time).total_seconds()
        successful_tasks = [r for r in self.results if r.get('success', False)]
        failed_tasks = [r for r in self.results if not r.get('success', False)]
        
        # 保存 JSON
        summary_data = {
            'batch_info': {
                'total_tasks': len(self.results),
                'successful_tasks': len(successful_tasks),
                'failed_tasks': len(failed_tasks),
                'success_rate': len(successful_tasks) / max(1, len(self.results)),
                'total_execution_time_seconds': total_time,
                'start_time': total_start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            },
            'task_results': self.results
        }
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # 保存文本汇总
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("批量自适应搜索执行报告\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"总任务数: {len(self.results)}\n")
            f.write(f"成功任务数: {len(successful_tasks)}\n")
            f.write(f"失败任务数: {len(failed_tasks)}\n")
            f.write(f"成功率: {len(successful_tasks)/max(1, len(self.results)):.2%}\n")
            f.write(f"总执行时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)\n\n")
            
            if successful_tasks:
                f.write("-" * 80 + "\n")
                f.write("成功任务性能排名（按加速比）:\n")
                f.write("-" * 80 + "\n")
                sorted_tasks = sorted(successful_tasks, key=lambda x: x.get('best_speedup', 0), reverse=True)
                for i, task in enumerate(sorted_tasks, 1):
                    gen_time = task.get('best_gen_time', 0)
                    gen_time_str = f"{gen_time:8.4f}us" if gen_time > 0 and gen_time != float('inf') else "     N/A"
                    f.write(f"{i:3d}. {task['op_name']:<30} "
                           f"加速比: {task.get('best_speedup', 0):6.2f}x | "
                           f"gen_time: {gen_time_str} | "
                           f"成功率: {task.get('success_rate', 0):.1%}\n")
            
            if failed_tasks:
                f.write("\n" + "-" * 80 + "\n")
                f.write("失败任务列表:\n")
                f.write("-" * 80 + "\n")
                for task in failed_tasks:
                    error = task.get('error', 'Unknown error')
                    f.write(f"  • {task['op_name']}: {error}\n")
        
        return summary_data


# ============================================================================
# 批量任务池
# ============================================================================

class BatchAdaptiveSearchPool:
    """批量自适应搜索任务池"""
    
    def __init__(
        self,
        max_concurrency: int,
        config_path: Optional[str],
        collector: AdaptiveSearchResultCollector,
        device_pool: List[int]
    ):
        self.max_concurrency = max_concurrency
        self.config_path = config_path
        self.collector = collector
        self.device_pool = device_pool
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        # 设备分配队列（用于避免设备冲突）
        self.device_queue: Optional[asyncio.Queue] = None
        if self.device_pool and not os.getenv("AKG_AGENTS_WORKER_URL"):
            self.device_queue = asyncio.Queue()
            for device_id in self.device_pool:
                self.device_queue.put_nowait(device_id)
    
    async def run_task_async(
        self,
        task_file: Path,
        output_dir: Path,
        index: int,
        total: int
    ) -> Dict[str, Any]:
        """异步运行单个任务"""
        # 获取设备
        assigned_device = None
        if self.device_queue:
            assigned_device = await self.device_queue.get()
            print(f"任务 [{index}/{total}] {task_file.stem} 分配设备: {assigned_device}")
        
        try:
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._run_single_task_subprocess,
                    task_file,
                    output_dir,
                    index,
                    total,
                    assigned_device
                )
                
                # 收集结果
                self.collector.add_result(result)
                
                return result
        finally:
            # 释放设备
            if self.device_queue and assigned_device is not None:
                await self.device_queue.put(assigned_device)
    
    def _run_single_task_subprocess(
        self,
        task_file: Path,
        output_dir: Path,
        index: int,
        total: int,
        assigned_device: Optional[int]
    ) -> Dict[str, Any]:
        """使用子进程运行单个任务"""
        op_name = "akg_agents_" + task_file.stem
        start_time = datetime.now()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"output_{op_name}_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            env = os.environ.copy()
            project_root = Path(get_project_root()).parent.parent
            tools_dir = project_root / "tools"
            single_script = tools_dir / "run_single_adaptive_search.py"
            
            if not single_script.exists():
                raise FileNotFoundError(f"run_single_adaptive_search.py 不存在: {single_script}")
            
            absolute_task_file = Path(task_file).resolve()
            device_arg = str(assigned_device) if assigned_device is not None else "0"
            
            cmd = [
                sys.executable,
                str(single_script),
                op_name,
                str(absolute_task_file),
                device_arg
            ]
            
            if self.config_path:
                cmd.append(self.config_path)
            
            # 运行子进程
            subprocess_result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, errors='replace'
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 解析输出
            output_lines = subprocess_result.stdout.split('\n') if subprocess_result.stdout else []
            result_data = self._parse_output(output_lines)
            
            # 保存输出
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"任务名称: {op_name}\n")
                f.write(f"任务文件: {task_file}\n")
                f.write(f"开始时间: {start_time.isoformat()}\n")
                f.write(f"结束时间: {end_time.isoformat()}\n")
                f.write(f"执行时间: {execution_time:.2f}秒\n")
                f.write(f"返回码: {subprocess_result.returncode}\n")
                f.write("\n" + "=" * 50 + " 完整输出 " + "=" * 50 + "\n")
                f.write(subprocess_result.stdout or "")
                if subprocess_result.stderr:
                    f.write("\n" + "=" * 50 + " 错误输出 " + "=" * 50 + "\n")
                    f.write(subprocess_result.stderr)
            
            task_success = (
                subprocess_result.returncode == 0 and
                result_data.get('total_success', 0) > 0
            )
            
            return {
                'op_name': op_name,
                'task_file': str(task_file),
                'success': task_success,
                'execution_time': execution_time,
                'return_code': subprocess_result.returncode,
                'total_submitted': result_data.get('total_submitted', 0),
                'total_success': result_data.get('total_success', 0),
                'total_failed': result_data.get('total_failed', 0),
                'success_rate': result_data.get('success_rate', 0),
                'elapsed_time': result_data.get('elapsed_time', 0),
                'best_speedup': result_data.get('best_speedup', 0),
                'best_gen_time': result_data.get('best_gen_time', float('inf')),
                'storage_dir': result_data.get('storage_dir', ''),
                'output_file': str(output_file),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                'op_name': op_name,
                'task_file': str(task_file),
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
    
    def _parse_output(self, lines: List[str]) -> Dict[str, Any]:
        """解析输出获取结果数据"""
        result = {
            'total_submitted': 0,
            'total_success': 0,
            'total_failed': 0,
            'success_rate': 0,
            'elapsed_time': 0,
            'best_speedup': 0,
            'best_gen_time': 0,  # 初始化为 0，表示未找到
            'storage_dir': ''
        }
        
        for line in lines:
            line = line.strip()
            
            # 解析任务统计行
            # 格式：任务统计：提交20 / 完成20 / 成功15 / 失败5 | 成功率75.0% | 耗时120.5s
            if '任务统计' in line and '提交' in line:
                try:
                    # 提取数字
                    match = re.search(r'提交(\d+)', line)
                    if match:
                        result['total_submitted'] = int(match.group(1))
                    match = re.search(r'成功(\d+)', line)
                    if match:
                        result['total_success'] = int(match.group(1))
                    match = re.search(r'失败(\d+)', line)
                    if match:
                        result['total_failed'] = int(match.group(1))
                    match = re.search(r'成功率([\d.]+)%', line)
                    if match:
                        result['success_rate'] = float(match.group(1)) / 100
                    match = re.search(r'耗时([\d.]+)s', line)
                    if match:
                        result['elapsed_time'] = float(match.group(1))
                except:
                    pass
            
            # 解析存储目录
            if '存储目录' in line:
                try:
                    result['storage_dir'] = line.split('：')[-1].strip()
                except:
                    pass
            
            # 解析最佳实现行（同时包含加速比和生成代码时间）
            # 格式：1. xxx（初始，个体路径：xxx，生成代码：0.1234us，基准代码：0.5678us，加速比：4.50x）
            # 注意：加速比和生成代码可能在同一行，使用 if 而非 elif 确保两者都被解析
            
            # 解析加速比
            if '加速比' in line:
                try:
                    match = re.search(r'加速比[：:]\s*([\d.]+)x', line)
                    if match:
                        speedup = float(match.group(1))
                        if speedup > result['best_speedup']:
                            result['best_speedup'] = speedup
                except:
                    pass
            
            # 解析 gen_time（生成代码时间）
            if '生成代码' in line:
                try:
                    match = re.search(r'生成代码[：:]\s*([\d.]+)us', line)
                    if match:
                        gen_time = float(match.group(1))
                        # 取最小的 gen_time（最佳性能）
                        if result['best_gen_time'] == 0 or gen_time < result['best_gen_time']:
                            result['best_gen_time'] = gen_time
                except:
                    pass
        
        return result
    
    async def run_batch_parallel(
        self,
        task_files: List[Path],
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """并行运行批量任务"""
        print(f"启动并行执行，最大并发数: {self.max_concurrency}")
        print(f"设备池: {self.device_pool}")
        
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
                        task_file, output_dir, task_index + 1, len(task_files)
                    )
                    
                    results[task_index] = result
                    task_queue.task_done()
                    
                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    print(f"Worker 异常: {e}")
                    if 'task_index' in locals():
                        results[task_index] = {
                            'op_name': task_file.stem,
                            'success': False,
                            'error': str(e),
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


# ============================================================================
# 主函数
# ============================================================================

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


def load_batch_adaptive_search_config(config_path: Optional[str] = None) -> Tuple[BatchAdaptiveSearchConfig, str]:
    """加载批量自适应搜索配置"""
    if config_path is None:
        project_root = get_project_root()
        config_path = os.path.join(project_root, "config", "adaptive_search_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = BatchAdaptiveSearchConfig.from_yaml(config_path)
    
    print(f"成功加载配置文件: {config_path}")
    print(f"   环境: {config.dsl}/{config.framework}/{config.backend}/{config.arch}")
    print(f"   批量并行数: {config.batch_parallel_num}")
    print(f"   设备池: {config.device_pool}")
    print(f"   任务目录: {config.task_dir}")
    print(f"   输出目录: {config.output_dir}")
    print(f"   每算子配置: 最大并发={config.max_concurrent}, 初始任务={config.initial_task_count}, 最大任务={config.max_total_tasks}")
    
    return config, config_path


def print_batch_summary(results: List[Dict[str, Any]], total_start_time: datetime):
    """打印批量执行摘要"""
    successful_tasks = [r for r in results if r.get('success', False)]
    failed_tasks = [r for r in results if not r.get('success', False)]
    
    total_time = (datetime.now() - total_start_time).total_seconds()
    
    print("\n" + "=" * 100)
    print("批量自适应搜索执行完成！最终统计报告")
    print("=" * 100)
    print(f"总任务数: {len(results)}")
    print(f"成功任务数: {len(successful_tasks)}")
    print(f"失败任务数: {len(failed_tasks)}")
    print(f"成功率: {len(successful_tasks)/max(1, len(results)):.2%}")
    print(f"总执行时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
    
    if successful_tasks:
        print(f"\n成功任务性能排名（按加速比）:")
        sorted_tasks = sorted(successful_tasks, key=lambda x: x.get('best_speedup', 0), reverse=True)
        for i, task in enumerate(sorted_tasks[:10], 1):
            gen_time = task.get('best_gen_time', 0)
            gen_time_str = f"{gen_time:8.4f}us" if gen_time > 0 and gen_time != float('inf') else "     N/A"
            print(f"  {i:2d}. {task['op_name']:<25} "
                  f"加速比: {task.get('best_speedup', 0):6.2f}x | "
                  f"gen_time: {gen_time_str} | "
                  f"成功率: {task.get('success_rate', 0):.1%}")
    
    if failed_tasks:
        print(f"\n失败任务列表:")
        for task in failed_tasks:
            error = task.get('error', 'Unknown error')
            print(f"  • {task['op_name']}: {error}")
    
    print("=" * 100)


async def run_batch_adaptive_search(config_path: Optional[str] = None) -> None:
    """批量执行自适应搜索"""
    # 加载配置
    config, resolved_config_path = load_batch_adaptive_search_config(config_path)
    
    # 验证配置
    if not config.task_dir:
        raise ValueError("请在配置文件中指定 batch.task_dir（任务文件目录）")
    if not config.output_dir:
        raise ValueError("请在配置文件中指定 batch.output_dir（输出目录）")
    
    task_dir = os.path.expanduser(config.task_dir)
    output_dir = Path(os.path.expanduser(config.output_dir))
    parallel_num = min(config.batch_parallel_num, len(config.device_pool)) if config.device_pool else config.batch_parallel_num
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("开始批量自适应搜索")
    print("=" * 80)
    print(f"任务目录: {task_dir}")
    print(f"输出目录: {output_dir}")
    print(f"并行数: {parallel_num}")
    print(f"设备配置: {config.device_pool}")
    print("=" * 80)
    
    total_start_time = datetime.now()
    
    try:
        # 发现任务文件
        task_files = discover_task_files(task_dir)
        
        if not task_files:
            print("未找到任何 .py 任务文件")
            return
        
        # 注册 Worker（仅用于验证环境，实际任务由子进程独立注册）
        await register_worker(
            backend=config.backend,
            arch=config.arch,
            device_ids=config.device_pool
        )
        
        # 创建结果收集器
        collector = AdaptiveSearchResultCollector(output_dir)
        print(f"\n结果收集器已启动:")
        print(f"   TXT汇总: {collector.summary_file}")
        print(f"   CSV结果: {collector.csv_file}")
        print(f"   JSON结果: {collector.json_file}")
        print("=" * 80)
        
        # 创建批量任务池
        batch_pool = BatchAdaptiveSearchPool(
            max_concurrency=parallel_num,
            config_path=resolved_config_path,
            collector=collector,
            device_pool=config.device_pool
        )
        
        print(f"\n将并行执行 {len(task_files)} 个算子的自适应搜索...")
        
        # 运行任务
        results = await batch_pool.run_batch_parallel(task_files, output_dir)
        
        # 保存最终汇总
        collector.save_final_summary(total_start_time)
        
        # 打印摘要
        print_batch_summary(results, total_start_time)
        print(f"\n结果已保存到: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n批量执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        asyncio.run(run_batch_adaptive_search(config_path))
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n批量执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

