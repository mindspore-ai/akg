"""
Kernel泛化性随机Case测试
配置并运行: python tools/random_cases_test/random_cases_test.py
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "python"))

from tools.random_cases_test.single_kernel_tester import SingleKernelTester


# ============================================================================
# 配置区域
# ============================================================================

# TODO: 修改为您的kernel所在文件夹
KERNEL_DIR = "path/to/your/kernel_folder/"  # 需要修改此路径
KERNEL_NAME = "kernel.py"                   # 需要修改为实际的kernel文件名
SPACE_CONFIG_NAME = "space_config.py"

# 实验参数
NUM_CASES = 10                # 测试case数量
SAMPLING_STRATEGY = "mixed"   # 采样策略: 'random', 'boundary', 'mixed'
TIMEOUT_SECONDS = 10.0        # 超时阈值（秒）
RANDOM_SEED = 42              # 随机种子（确保可复现）
OUTPUT_DIR = None             # 输出目录（None表示使用KERNEL_DIR/results）
DEVICE_ID = None              # 设备ID（None表示从环境变量DEVICE_ID获取，默认0）

# ============================================================================


def main():
    print("=" * 80)
    print("Kernel泛化性测试")
    print("=" * 80)
    
    kernel_dir = os.path.expanduser(KERNEL_DIR)
    kernel_path = os.path.join(kernel_dir, KERNEL_NAME)
    space_config_path = os.path.join(kernel_dir, SPACE_CONFIG_NAME)
    print(f"文件夹: {kernel_dir}")
    
    # 确定输出目录
    if OUTPUT_DIR is None:
        output_dir = os.path.join(kernel_dir, "results")
        print(f"输出目录: {output_dir} (自动设置)")
    else:
        output_dir = OUTPUT_DIR
        print(f"输出目录: {output_dir}")
    
    # 设备ID
    device_id = DEVICE_ID if DEVICE_ID is not None else int(os.getenv("DEVICE_ID", "0"))
    
    print()
    
    # 检查文件
    for name, path in [("Kernel", kernel_path), ("Space config", space_config_path)]:
        if not os.path.exists(path):
            print(f"[ERROR] {name}不存在: {path}")
            return
    
    # 加载配置
    config = {
        'dsl': 'triton',
        'backend': 'cuda',
        'arch': 'a100',
        'log_dir': '~/aikg_logs'
    }
    
    # 运行测试
    try:
        print("\n开始测试...\n")
        
        tester = SingleKernelTester(
            kernel_path=kernel_path,
            space_config_path=space_config_path,
            num_cases=NUM_CASES,
            output_dir=output_dir,
            sampling_strategy=SAMPLING_STRATEGY,
            timeout_seconds=TIMEOUT_SECONDS,
            seed=RANDOM_SEED,
            config=config,
            device_id=device_id
        )
        
        result = tester.run_test()
        tester.print_summary(result)
        tester.save_report(result)
        
        print("\n" + "=" * 80)
        print("[SUCCESS] 测试完成")
        print("=" * 80)
        print(f"结果: {output_dir}/")
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] 测试失败")
        print("=" * 80)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
