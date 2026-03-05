"""
算子鲁棒性测试 — 矩阵乘法 (Standard_matrix_multiplication)

基于 akg_agents 已验证通过的 case，直接导入 Model / ModelNew，
构造不同 shape/dtype 的输入进行精度验证和性能对比。

已验证配置: M=1024, K=4096, N=2048, dtype=float32, backend=cpu, dsl=cpp

使用方法:
    source <AKG_AGENTS_PATH>/env.sh && conda activate <CONDA_ENV>
    python robustness_test_matmul.py

    # 多设备并行:
    AKG_AGENTS_DEVICES_LIST="0,1,2,3" python robustness_test_matmul.py
"""

import torch
from robustness_test_runner import RobustnessTestRunner


# ============================================================
# 算子特定配置 — 根据实际 case 修改
# ============================================================

CONFIG = {
    # 已通过验证的 case 路径和模块名
    # 注意：task_code 和 kernel_code 不一定在同一个文件夹，
    #       verify_dir 应该是包含两个文件的目录，或用 sys.path 添加多个路径
    "verify_dir": "~/akg_agents_logs/Task_ih0at2l3/passed_cases/"
                  "akg_agents_kernelbench_2_Standard_matrix_multiplication_/"
                  "Iteration0_Step01_verify",
    "task_module": "akg_agents_kernelbench_2_Standard_matrix_multiplication__torch",
    "kernel_module": "akg_agents_kernelbench_2_Standard_matrix_multiplication__cpp_impl",

    # 算子信息
    "op_name": "Standard_matrix_multiplication",
    "framework": "torch",
    "dsl": "cpp",
    "backend": "cpu",            # "ascend" / "cuda" / "cpu"
    "arch": "x86_64",           # "Ascend910B" / "sm_80" / "x86_64"
    "device_ids": [0],          # 多设备: [0, 1, 2, 3]
    "seed": 42,
    "warmup_times": 5,
    "run_times": 50,
    "verify_timeout": 300,
}


# ============================================================
# 测试 case 定义 — 根据算子 tensor 签名构造
# ============================================================
#
# 算子分析:
#   forward(A, B) -> torch.matmul(A, B)
#   A=(M, K), B=(K, N), Output=(M, N)
#   get_init_inputs() = [] -> 所有维度均为自由维度
#   kernel: AVX2 float32, VECTOR_SIZE=8 -> N 对齐边界重要
#
# 元素数分级 (以最大 tensor 元素数计):
#   小 <= 1e3, 中 1e4~1e7, 大 >= 1e8

TEST_CASES = [
    # (tag, params_dict, dtype, description)
    ("original",     {"M": 1024, "K": 4096, "N": 2048}, torch.float32,
     "原始通过 shape (中等, max ~4M 元素)"),
    ("small",        {"M": 4,    "K": 16,   "N": 4},    torch.float32,
     "小 (max tensor 64 元素)"),
    ("medium",       {"M": 128,  "K": 512,  "N": 256},  torch.float32,
     "中等 (max tensor ~64K 元素)"),
    ("large",        {"M": 4096, "K": 4096, "N": 4096}, torch.float32,
     "大 (max tensor ~16M 元素)"),
    ("min_edge",     {"M": 1,    "K": 1,    "N": 1},    torch.float32,
     "最小边界: 所有维度=1"),
    ("single_row",   {"M": 1,    "K": 4096, "N": 2048}, torch.float32,
     "单行 batch (极端纵横比)"),
    ("non_align_N",  {"M": 64,   "K": 128,  "N": 127},  torch.float32,
     "N=127, 非 VECTOR_SIZE(8) 对齐"),
    ("dtype_fp16",   {"M": 1024, "K": 4096, "N": 2048}, torch.float16,
     "dtype 变异: float16"),
]


def make_inputs(params, dtype, device):
    """根据 matmul 的 tensor 签名构造输入: A=(M,K), B=(K,N)"""
    M, K, N = params["M"], params["K"], params["N"]
    return [
        torch.randn(M, K, dtype=dtype, device=device),
        torch.randn(K, N, dtype=dtype, device=device),
    ]


# ============================================================
# 以下无需修改 — 通用逻辑在 robustness_test_runner.py 中
# ============================================================

if __name__ == "__main__":
    runner = RobustnessTestRunner(CONFIG, TEST_CASES, make_inputs)
    runner.run()
