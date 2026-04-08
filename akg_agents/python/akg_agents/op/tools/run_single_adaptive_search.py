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
单任务自适应搜索执行脚本

用法:
  python run_single_adaptive_search.py                                    # 使用默认配置
  python run_single_adaptive_search.py <config_file>                      # 使用YAML配置文件
  python run_single_adaptive_search.py <op_name> <task_file> <device> [config_file]  # 批量runner模式
"""

import sys
import asyncio
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from akg_agents import get_project_root
from akg_agents.op.adaptive_search import adaptive_search
from akg_agents.core.worker.manager import register_worker
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task
from akg_agents.utils.common_utils import load_yaml


@dataclass
class AdaptiveSearchRunnerConfig:
    """自适应搜索运行器配置"""
    # 基础信息
    op_name: str = "akg_agents_test"
    task_desc: str = ""
    
    # 环境配置
    dsl: str = "triton_ascend"
    framework: str = "torch"
    backend: str = "ascend"
    arch: str = "ascend910b4"
    
    # 设备配置
    device_list: List[int] = field(default_factory=lambda: [0])
    
    # 搜索参数
    max_concurrent: int = 4
    initial_task_count: int = 4
    max_total_tasks: int = 50
    
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
    
    # 配置文件路径
    config_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str, skip_task_config: bool = False) -> "AdaptiveSearchRunnerConfig":
        """从 YAML 文件加载配置"""
        config_dict = load_yaml(config_path)
        config_dir = os.path.dirname(os.path.abspath(config_path))
        
        instance = cls()
        
        # 任务配置
        task_config = config_dict.get("task", {})
        if not skip_task_config:
            instance.op_name = task_config.get("op_name", instance.op_name)
            
            # SOL 数据集目录
            sol_problem_dir = task_config.get("sol_problem_dir", "")
            if sol_problem_dir:
                if not os.path.isabs(sol_problem_dir):
                    sol_problem_dir = os.path.normpath(os.path.join(config_dir, sol_problem_dir))
                instance._sol_problem_dir = sol_problem_dir
            
            # 从文件路径加载任务描述
            task_desc = task_config.get("task_desc", "")
            if task_desc:
                if not os.path.isabs(task_desc):
                    task_desc = os.path.normpath(os.path.join(config_dir, task_desc))
                if os.path.isfile(task_desc):
                    with open(task_desc, 'r', encoding='utf-8') as f:
                        instance.task_desc = f.read()
                else:
                    raise FileNotFoundError(f"Task description file not found: {task_desc}")
        
        # 环境配置
        env_config = config_dict.get("environment", {})
        instance.dsl = env_config.get("dsl", instance.dsl)
        instance.framework = env_config.get("framework", instance.framework)
        instance.backend = env_config.get("backend", instance.backend)
        instance.arch = env_config.get("arch", instance.arch)
        instance.device_list = env_config.get("device_list", instance.device_list)
        
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
        
        # 基础配置文件路径（必填）
        instance.config_path = config_dict.get("config_path")
        if not instance.config_path:
            raise ValueError("config_path 是必填项，请在配置文件中指定 LLM 配置文件路径，如: config_path: 'config/triton_ascend_evolve_config.yaml'")
        
        if not os.path.isabs(instance.config_path):
            instance.config_path = os.path.normpath(os.path.join(config_dir, instance.config_path))
        
        return instance


def print_usage():
    """打印使用说明"""
    print("用法:")
    print("  python run_single_adaptive_search.py                                                    # 使用默认配置")
    print("  python run_single_adaptive_search.py <config_file>                                      # 使用YAML配置文件")
    print("  python run_single_adaptive_search.py <op_name> <task_file> <device> [config_file]       # batch runner模式")


def print_config(op_name: str, config: AdaptiveSearchRunnerConfig):
    """打印配置信息"""
    print("=" * 60)
    print("自适应搜索配置")
    print("=" * 60)
    print(f"算子名称: {op_name}")
    print(f"DSL: {config.dsl}")
    print(f"框架: {config.framework}")
    print(f"后端: {config.backend}")
    print(f"架构: {config.arch}")
    print("-" * 60)
    print(f"设备列表: {config.device_list}")
    print(f"最大并发数: {config.max_concurrent}")
    print(f"初始任务数: {config.initial_task_count}")
    print(f"最大总任务数: {config.max_total_tasks}")
    print("-" * 60)
    print(f"UCB 探索系数: {config.exploration_coef}")
    print(f"随机扰动: {config.random_factor}")
    print(f"使用 Softmax: {config.use_softmax}")
    print(f"灵感采样: 父代 + {config.inspiration_sample_num}个（层次化={config.use_tiered_sampling}）")
    print("=" * 60)


def print_result(result):
    """打印搜索结果"""
    print("\n" + "=" * 100)
    print("自适应搜索结果")
    print("=" * 100)
    
    print(f"算子名称：{result['op_name']}")
    print(f"终止原因：{result.get('stop_reason', 'Unknown')}")
    print(f"任务统计：提交{result['total_submitted']} / 完成{result['total_completed']} / 成功{result['total_success']} / 失败{result['total_failed']} | 成功率{result['success_rate']:.1%} | 耗时{result['elapsed_time']:.1f}s")
    print(f"存储目录：{result.get('storage_dir', 'N/A')}")
    
    # 打印 Task 文件夹和 Log 目录
    task_folder = result.get('task_folder', '')
    if task_folder:
        print(f"Task文件夹：{task_folder}")
    
    log_dir = result.get('log_dir', '')
    if log_dir:
        print(f"Log目录：{log_dir}")
    
    # 打印谱系图路径
    lineage_graph = result.get('lineage_graph', '')
    if lineage_graph:
        print(f"谱系图：{lineage_graph}")
    
    # 打印最佳实现
    print("\n最佳实现（前5个）：")
    
    best_impls = result.get('best_implementations', [])
    if best_impls:
        for i, impl in enumerate(best_impls[:5], 1):
            task_id = impl.get('id', 'unknown')
            gen_time = impl.get('gen_time', 0)
            profile = impl.get('profile', {})
            base_time = profile.get('base_time', 0) if profile else 0
            speedup = impl.get('speedup', 0)
            generation = impl.get('generation', 0)
            parent_id = impl.get('parent_id', None)
            # verify_dir 现在是目录名（如 Iteration_Init_Task1_Step02_verify）而不是完整路径
            verify_dir = impl.get('verify_dir', '')
            
            # 父代描述
            if generation == 0:
                parent_desc = "初始"
            else:
                parent_desc = f"父代 {parent_id}" if parent_id else f"G{generation}"
            
            # 格式：序号. 任务ID（父代信息，个体路径：xxx，生成代码：xxxus，基准代码：xxxus，加速比：x.xxx）
            print(f"{i}. {task_id}（{parent_desc}，个体路径：{verify_dir}，生成代码：{gen_time:.4f}us，基准代码：{base_time:.4f}us，加速比：{speedup:.2f}x）")
    else:
        print("未找到成功的实现")
    
    print("\n" + "=" * 100)


def load_task_description(task_file: str) -> str:
    """加载任务描述文件"""
    if os.path.isfile(task_file):
        with open(task_file, 'r', encoding='utf-8') as f:
            return f.read()
    return task_file


def load_sol_task(sol_dir: str):
    """Load SOL dataset: returns (op_name, task_desc, sol_problem_dir)"""
    case_dir = Path(sol_dir).resolve()
    if not (case_dir / "definition.json").exists():
        raise FileNotFoundError(f"SOL dataset missing definition.json: {case_dir}")
    
    with open(case_dir / "definition.json", "r", encoding="utf-8") as f:
        def_json = f.read()
    definition = json.loads(def_json)
    op_name = definition.get("name", case_dir.name)
    
    with open(case_dir / "reference.py", "r", encoding="utf-8") as f:
        ref_py = f.read()
    
    workload_sample = ""
    workload_file = case_dir / "workload.jsonl"
    if workload_file.exists():
        with open(workload_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            first = json.loads(lines[0])
            workload_sample = (
                f"\n\n## workload 示例（共 {len(lines)} 组，以下为第 1 组）\n"
                f"```json\n{json.dumps(first, indent=2)}\n```"
            )
    
    task_desc = (
        f"请实现一个 Triton Ascend 算子。\n\n"
        f"## definition.json\n```json\n{def_json}\n```\n\n"
        f"## reference.py\n```python\n{ref_py}\n```"
        f"{workload_sample}\n\n"
        f"注意：请使用 Triton 编写 kernel，并将其封装在 ModelNew 类的 forward 方法中。"
    )
    
    return op_name, task_desc, str(case_dir)


def parse_default_config():
    """解析默认配置"""
    project_root = get_project_root()
    config_path = os.path.join(project_root, "op", "config", "adaptive_search_config.yaml")
    
    if os.path.exists(config_path):
        try:
            config = AdaptiveSearchRunnerConfig.from_yaml(config_path)
            print(f"使用默认配置文件: {config_path}")
            print(f"算子名称: {config.op_name}")
            return config.op_name, config.task_desc, config
        except Exception as e:
            print(f"无法加载默认配置文件 {config_path}: {e}")
    
    print("使用内置默认配置")
    config = AdaptiveSearchRunnerConfig()
    config.task_desc = """
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.rand(batch_size, dim, device='npu')
    return [x]

def get_init_inputs():
    return []
"""
    return config.op_name, config.task_desc, config


def parse_config_file_mode(config_path: str):
    """解析配置文件模式"""
    try:
        config = AdaptiveSearchRunnerConfig.from_yaml(config_path)
        print(f"使用配置文件: {config_path}")
        print(f"算子名称: {config.op_name}")
        return config.op_name, config.task_desc, config
    except Exception as e:
        print(f"无法加载配置文件 {config_path}: {e}")
        sys.exit(1)


def parse_batch_runner_mode(args):
    """解析批量运行器模式"""
    op_name = args[1]
    task_file = args[2]
    device = int(args[3])
    
    config = AdaptiveSearchRunnerConfig()
    config.device_list = [device]
    
    # 如果提供了配置文件路径
    if len(args) >= 5:
        config_path = args[4]
        try:
            file_config = AdaptiveSearchRunnerConfig.from_yaml(config_path, skip_task_config=True)
            # 复制配置（除了任务相关的）
            config.dsl = file_config.dsl
            config.framework = file_config.framework
            config.backend = file_config.backend
            config.arch = file_config.arch
            config.max_concurrent = file_config.max_concurrent
            config.initial_task_count = file_config.initial_task_count
            config.max_total_tasks = file_config.max_total_tasks
            config.exploration_coef = file_config.exploration_coef
            config.random_factor = file_config.random_factor
            config.use_tiered_sampling = file_config.use_tiered_sampling
            config.config_path = file_config.config_path
        except Exception as e:
            print(f"警告: 无法加载配置文件 {config_path}: {e}")
    
    # 检测 SOL 数据集：如果 task_file 是目录且包含 definition.json
    if os.path.isdir(task_file) and os.path.exists(os.path.join(task_file, "definition.json")):
        sol_op_name, task_desc, sol_problem_dir = load_sol_task(task_file)
        op_name = sol_op_name
        config._sol_problem_dir = sol_problem_dir
        print(f"SOL 模式: 数据集目录={sol_problem_dir}")
    else:
        task_desc = load_task_description(task_file)
    
    print(f"任务: {op_name}")
    print(f"任务文件: {task_file}")
    print(f"设备: {config.device_list}")
    print(f"配置: {config.max_total_tasks}总/{config.max_concurrent}并行")
    print(f"基础参数: {config.dsl}/{config.framework}/{config.backend}/{config.arch}")
    
    return op_name, task_desc, config


async def run_wrapper(op_name: str, task_desc: str, config: AdaptiveSearchRunnerConfig):
    """包装运行函数，负责注册Worker"""
    # 注册 Worker
    await register_worker(
        backend=config.backend,
        arch=config.arch,
        device_ids=config.device_list
    )
    
    # 加载配置
    if config.config_path and os.path.exists(config.config_path):
        loaded_config = load_config(config_path=config.config_path)
    else:
        loaded_config = load_config(dsl=config.dsl, backend=config.backend)
    
    # 兜底补齐 agent_model_config（各 Agent 需要通过此配置获取模型名称
    if "agent_model_config" not in loaded_config or not isinstance(
        loaded_config.get("agent_model_config"), dict
    ):
        loaded_config["agent_model_config"] = {}
    mc = loaded_config["agent_model_config"]
    
    # 设置默认模型级别（优先使用配置中的 default，否则使用 "standard"）
    default_level = mc.get("default") or "standard"
    mc.setdefault("default", default_level)
    
    # 为各个 agent 设置默认模型级别（如果未配置则使用 default）
    for agent_name in ["designer", "coder", "conductor", "verifier", "selector", "op_task_builder"]:
        mc.setdefault(agent_name, mc["default"])
    
    # 添加 task_label 到配置（adaptive_search 内部的 LangGraphTask 需要）
    from akg_agents.utils.task_label import resolve_task_label
    loaded_config["task_label"] = resolve_task_label(
        op_name=op_name,
        parallel_index=1,
    )
    
    # 检查环境
    check_env_for_task(config.framework, config.backend, config.dsl, loaded_config)
    
    # SOL 模式：注入 bench_type 和 sol_problem_dir
    sol_problem_dir = getattr(config, '_sol_problem_dir', None)
    if sol_problem_dir:
        loaded_config["bench_type"] = "sol"
        loaded_config["sol_problem_dir"] = sol_problem_dir
    
    # 运行自适应搜索
    print("\n开始自适应搜索...")
    result = await adaptive_search(
        op_name=op_name,
        task_desc=task_desc,
        dsl=config.dsl,
        framework=config.framework,
        backend=config.backend,
        arch=config.arch,
        config=loaded_config,
        
        # 并发控制
        max_concurrent=config.max_concurrent,
        initial_task_count=config.initial_task_count,
        
        # UCB 参数
        exploration_coef=config.exploration_coef,
        random_factor=config.random_factor,
        use_softmax=config.use_softmax,
        softmax_temperature=config.softmax_temperature,
        
        # 停止条件
        max_total_tasks=config.max_total_tasks,
        
        # 灵感采样参数
        inspiration_sample_num=config.inspiration_sample_num,
        use_tiered_sampling=config.use_tiered_sampling,
        handwrite_sample_num=config.handwrite_sample_num,
        handwrite_decay_rate=config.handwrite_decay_rate
    )
    
    return result


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) == 1:
        # 无参数模式：使用默认配置文件
        op_name, task_desc, config = parse_default_config()
    
    elif len(sys.argv) == 2:
        # 配置文件模式：从YAML配置文件加载
        config_path = sys.argv[1]
        op_name, task_desc, config = parse_config_file_mode(config_path)
    
    elif len(sys.argv) >= 4:
        # batch_runner模式: op_name task_file device [config_file]
        op_name, task_desc, config = parse_batch_runner_mode(sys.argv)
    
    else:
        print_usage()
        sys.exit(1)
    
    # 打印配置
    print_config(op_name, config)
    
    # 运行任务
    try:
        result = asyncio.run(run_wrapper(op_name=op_name, task_desc=task_desc, config=config))
        
        if result:
            print_result(result)
            
            if result.get('total_success', 0) > 0:
                print("\n✅ 自适应搜索成功完成!")
                print(f"   成功生成了 {result['total_success']} 个有效的算子实现")
            else:
                print("\n⚠️ 未能生成成功的算子实现，请检查配置和任务描述")
        else:
            print("\n❌ 搜索过程失败，请检查日志获取详细信息")
    
    except Exception as e:
        print(f"Error occurred during adaptive search: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
