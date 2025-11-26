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
批量任务执行脚本

用法:
  python run_batch_evolve.py                    # 使用默认配置
  python run_batch_evolve.py <config_file>      # 使用自定义配置文件
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_kernel_generator.utils.evolve.runner_manager import run_batch_evolve


def main():
    """主函数"""
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    try:
        asyncio.run(run_batch_evolve(config_path))
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n批量执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

