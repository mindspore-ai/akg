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
Client-Server-Worker 架构使用示例

此示例展示如何使用 AKGAgentsClient 与远程 Server-Worker 架构交互。

架构说明：
    Client (本地) --> Server (GPU服务器:8000) --> Worker Service (GPU服务器:9001) --> GPU

部署步骤：

1. 【在 GPU 服务器上】启动 Worker Service
   ```bash
   cd /path/to/akg/akg_agents
   source env.sh
   ./scripts/server_related/start_worker_service.sh cuda a100 0 9001
   ```
   参数说明：
   - cuda: 后端类型 (cuda/ascend)
   - a100: 硬件架构 (a100/ascend910b4)
   - 0: GPU 设备编号
   - 9001: Worker Service 端口

2. 【在 GPU 服务器上】启动 AIKG Server
   ```bash
   cd /path/to/akg/akg_agents
   source env.sh
   ./scripts/server_related/start_server.sh 8000
   ```
   参数说明：
   - 8000: Server 端口（默认）

3. 【在 GPU 服务器上】注册 Worker 到 Server
   ```bash
   ./scripts/server_related/register_worker_to_server.sh \
       http://localhost:8000 \
       http://localhost:9001 \
       cuda a100 1
   ```
   参数说明：
   - http://localhost:8000: Server URL
   - http://localhost:9001: Worker Service URL
   - cuda: 后端类型
   - a100: 硬件架构
   - 1: Worker 容量（并发能力）

4. 【在本地机器上】（可选）建立 SSH 隧道（如果 Server 在远程）
   ```bash
   # 方式1: 使用提供的脚本（需要修改脚本中的 SSH 配置）
   ./scripts/server_related/setup_ssh_tunnel.sh 8000 8000
   
   # 方式2: 手动建立 SSH 隧道
   ssh -N -L 8000:localhost:8000 user@gpu-server
   ```

5. 【在本地机器上】运行此示例
   ```bash
   cd /path/to/akg/akg_agents
   source env.sh
   python examples/run_client_server_worker.py
   ```

快速检查环境：
   运行以下脚本检查 Server 和 Worker 是否就绪：
   ```bash
   ./scripts/server_related/check_e2e_setup.sh http://localhost:8000 http://localhost:9001
   ```
"""

import os
import sys
import argparse
from akg_agents.client.akg_agents_client import AKGAgentsClient


def get_task_desc():
    """获取测试任务描述"""
    return '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
'''


def example_single_job(client: AKGAgentsClient):
    """
    示例 1: 提交单个 Job（CoderOnly 流程）
    
    此示例展示如何提交一个简单的 single job，Server 会：
    1. 调用 LLM 生成代码
    2. 将代码打包发送给 Worker Service 验证
    3. 返回验证结果
    """
    print("\n" + "=" * 60)
    print("示例 1: Single Job (CoderOnly 流程)")
    print("=" * 60)
    
    op_name = "relu_example"
    task_desc = get_task_desc()
    
    print(f"📝 提交 Job: {op_name}")
    print(f"   类型: single (coder_only_workflow)")
    print(f"   后端: cuda/a100/triton_cuda")
    
    # 提交 job
    job_id = client.submit_job(
        op_name=op_name,
        task_desc=task_desc,
        job_type="single",
        backend="cuda",
        arch="a100",
        dsl="triton_cuda",
        framework="torch",
        workflow="coder_only_workflow"
    )
    
    print(f"✅ Job 已提交: {job_id}")
    
    # 等待完成
    print(f"\n⏳ 等待 Job 完成...")
    print(f"   （这可能需要几分钟，包括 LLM 调用和 GPU 验证）")
    status = client.wait_for_completion(job_id, interval=5, timeout=1800)  # 30分钟超时
    
    # 显示结果
    print(f"\n📊 Job 结果:")
    print(f"   状态: {status.get('status')}")
    print(f"   类型: {status.get('job_type')}")
    
    if status.get('status') == 'completed':
        result = status.get('result')
        if isinstance(result, bool):
            print(f"   结果: {'✅ 成功' if result else '❌ 失败'}")
        else:
            print(f"   结果: {result}")
        return True
    else:
        error = status.get('error', 'Unknown error')
        print(f"   ❌ 失败: {error}")
        return False


def example_evolve_job(client: AKGAgentsClient):
    """
    示例 2: 提交 Evolve Job（进化优化流程）
    
    此示例展示如何提交一个 evolve job，Server 会：
    1. 调用 LLM 生成多个候选代码
    2. 并行发送给 Worker Service 验证
    3. 根据结果进行多轮进化优化
    4. 返回最佳结果
    """
    print("\n" + "=" * 60)
    print("示例 2: Evolve Job（进化优化）")
    print("=" * 60)
    
    op_name = "relu_evolve_example"
    task_desc = get_task_desc()
    
    print(f"📝 提交 Job: {op_name}")
    print(f"   类型: evolve")
    print(f"   后端: cuda/a100/triton_cuda")
    print(f"   参数: max_rounds=2, parallel_num=2")
    
    # 提交 job
    job_id = client.submit_job(
        op_name=op_name,
        task_desc=task_desc,
        job_type="evolve",
        backend="cuda",
        arch="a100",
        dsl="triton_cuda",
        framework="torch",
        max_rounds=2,
        parallel_num=2,
        num_islands=1,
        migration_interval=0,
        elite_size=0,
        parent_selection_prob=0.5
    )
    
    print(f"✅ Job 已提交: {job_id}")
    
    # 等待完成
    print(f"\n⏳ 等待 Evolve Job 完成...")
    print(f"   （这可能需要较长时间，包含多轮 LLM 调用和多次验证）")
    status = client.wait_for_completion(job_id, interval=10, timeout=3600)  # 1小时超时
    
    # 显示结果
    print(f"\n📊 Evolve Job 结果:")
    print(f"   状态: {status.get('status')}")
    print(f"   类型: {status.get('job_type')}")
    
    if status.get('status') == 'completed':
        result = status.get('result')
        if isinstance(result, dict):
            print(f"   ✅ Evolve 完成")
            # 只打印关键信息，避免打印过长的 code 和 full_result
            profile = result.get('profile', {})
            print(f"   性能数据: {profile}")
            
            # 简略显示代码信息
            code = result.get('code', '')
            code_preview = (code[:100] + '...') if len(code) > 100 else code
            print(f"   生成的代码(前100字符): {code_preview}")
            
            if code:
                # 可选：保存到文件
                save_path = f"{op_name}_best.py"
                with open(save_path, "w") as f:
                    f.write(code)
                print(f"   完整代码已保存至: {save_path}")
        else:
            print(f"   结果: {result}")
        return True
    else:
        error = status.get('error', 'Unknown error')
        print(f"   ❌ 失败: {error}")
        return False


def check_environment(client: AKGAgentsClient):
    """
    检查 Server 和 Worker 环境
    
    验证：
    1. Server 是否可访问
    2. Worker Service 是否已注册
    3. Worker 状态是否正常
    """
    print("\n" + "=" * 60)
    print("检查 Server 和 Worker 环境")
    print("=" * 60)
    
    try:
        # 检查 Worker 状态
        workers = client.get_workers_status()
        print(f"\n📋 已注册的 Workers: {len(workers)}")
        
        if len(workers) == 0:
            print("   ⚠️  没有注册的 Worker！")
            print("\n   💡 请先注册 Worker Service:")
            print("      ./scripts/server_related/register_worker_to_server.sh \\")
            print("          http://localhost:8000 \\")
            print("          http://localhost:9001 \\")
            print("          cuda a100 1")
            return False
        
        for i, worker in enumerate(workers, 1):
            print(f"\n   Worker {i}:")
            print(f"     Backend: {worker.get('backend')}")
            print(f"     Arch: {worker.get('arch')}")
            print(f"     Load: {worker.get('load')}/{worker.get('capacity')}")
            print(f"     Tags: {worker.get('tags', [])}")
        
        print("\n✅ 环境检查通过！")
        return True
        
    except Exception as e:
        print(f"\n   ❌ 检查失败: {e}")
        print("\n   💡 请确认:")
        print("      1. Server 正在运行: ./scripts/server_related/start_server.sh")
        print("      2. Worker Service 正在运行: ./scripts/server_related/start_worker_service.sh")
        print("      3. Worker 已注册: ./scripts/server_related/register_worker_to_server.sh")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Client-Server-Worker 架构使用示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认 Server URL (http://localhost:8000)
  python examples/run_client_server_worker.py
  
  # 指定 Server URL（通过 SSH 隧道）
  python examples/run_client_server_worker.py --server-url http://localhost:8000
  
  # 只运行 single job 示例
  python examples/run_client_server_worker.py --example single
  
  # 只运行 evolve job 示例
  python examples/run_client_server_worker.py --example evolve
  
  # 只检查环境
  python examples/run_client_server_worker.py --example check

部署步骤请查看文件开头的注释说明。
        """
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="AIKG Server 的 URL。也可通过环境变量 SERVER_URL 设置。默认: http://localhost:8000"
    )
    parser.add_argument(
        "--example",
        choices=["single", "evolve", "check", "all"],
        default="all",
        help="选择要运行的示例: single, evolve, check, 或 all (默认)"
    )
    
    args = parser.parse_args()
    
    # 获取 Server URL
    server_url = args.server_url or os.getenv("SERVER_URL", "http://localhost:8000")
    
    print("=" * 60)
    print("Client-Server-Worker 架构使用示例")
    print("=" * 60)
    print(f"\n🔗 Server URL: {server_url}")
    print("\n📖 部署说明:")
    print("   请查看文件开头的注释，了解如何启动 Server 和 Worker Service")
    print("=" * 60)
    
    # 创建 Client
    try:
        client = AKGAgentsClient(server_url)
        print(f"\n✅ 成功连接到 Server: {server_url}")
    except Exception as e:
        print(f"\n❌ 连接 Server 失败: {e}")
        print("\n💡 提示:")
        print("   - 确认 Server 正在运行")
        print("   - 确认 SSH 隧道已建立（如果使用）")
        print("   - 测试: curl http://localhost:8000/docs")
        sys.exit(1)
    
    # 运行示例
    results = []
    
    # 检查环境
    if args.example in ["check", "all"]:
        env_ok = check_environment(client)
        if not env_ok and args.example == "check":
            sys.exit(1)
        if not env_ok and args.example == "all":
            print("\n⚠️  环境检查未通过，但继续运行示例...")
    
    # 运行 single job 示例
    if args.example in ["single", "all"]:
        try:
            result = example_single_job(client)
            results.append(("Single Job", result))
        except Exception as e:
            print(f"\n❌ Single Job 示例异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(("Single Job", False))
    
    # 运行 evolve job 示例
    if args.example in ["evolve", "all"]:
        try:
            result = example_evolve_job(client)
            results.append(("Evolve Job", result))
        except Exception as e:
            print(f"\n❌ Evolve Job 示例异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(("Evolve Job", False))
    
    # 总结
    if results:
        print("\n" + "=" * 60)
        print("示例运行总结")
        print("=" * 60)
        for example_name, success in results:
            status = "✅ 成功" if success else "❌ 失败"
            print(f"  {example_name}: {status}")
        
        all_passed = all(result for _, result in results)
        if all_passed:
            print("\n🎉 所有示例运行成功！")
            sys.exit(0)
        else:
            print("\n❌ 部分示例运行失败")
            sys.exit(1)


if __name__ == "__main__":
    main()

