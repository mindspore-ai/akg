"""TileLang-Ascend 矩阵乘法示例 (PyTorch)"""

import torch
import torch_npu
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()


def matmul_kernel(M, N, K, block_m=16, block_n=16, block_k=256,
                  dtype="float16", accum_dtype="float32"):
    """创建矩阵乘法内核

    计算 C[M, N] = A[M, K] @ B[K, N]

    Args:
        M, N, K: 矩阵维度
        block_m, block_n, block_k: Tile大小
        dtype: 输入数据类型（float16）
        accum_dtype: 累加器类型（float32）
    """
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        """矩阵乘法内核"""
        # 每个block处理一个M维度的tile
        with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (cid, _):
            # 计算当前block的实际大小（边界处理）
            tail_m = cid * block_m
            tail_m = M - tail_m
            tail_m = T.min(block_m, tail_m)

            # 分配NPU内存
            l1_a = T.alloc_L1([block_m, block_k], dtype)      # A的L1缓存
            l1_b = T.alloc_L1([block_k, block_n], dtype)      # B的L1缓存
            l0_c = T.alloc_L0C([block_m, block_n], accum_dtype)  # C的累加器

            # Cube核心执行矩阵乘法
            with T.Scope("Cube"):
                # 遍历N维度
                for n_idx in T.serial(T.ceildiv(N, block_n)):
                    tail_n = n_idx * block_n
                    tail_n = N - tail_n
                    tail_n = T.min(block_n, tail_n)

                    # 遍历K维度（累加维度）
                    for k_idx in T.serial(T.ceildiv(K, block_k)):
                        tail_k = k_idx * block_k
                        tail_k = K - tail_k
                        tail_k = T.min(block_k, tail_k)

                        # 从GM加载到L1（ND→NZ格式）
                        T.npuir_load_nd2nz(
                            A[cid * block_m, k_idx * block_k],
                            l1_a,
                            [tail_m, tail_k]
                        )
                        T.npuir_load_nd2nz(
                            B[k_idx * block_k, n_idx * block_n],
                            l1_b,
                            [tail_k, tail_n]
                        )

                        # 矩阵乘法
                        # 首次: 初始化C为0后累加
                        # 后续: 直接累加到C
                        T.npuir_dot(
                            l1_a, l1_b, l0_c,
                            initC=(k_idx == 0),
                            size=[tail_m, tail_k, tail_n]
                        )

                    # 存储结果到GM（NZ→ND格式）
                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(
                            l0_c,
                            C[cid * block_m, n_idx * block_n],
                            size=[tail_m, tail_n],
                            enable_nz2nd=True
                        )
                        T.sync_block_set(0)

    return main


def matmul_tilelang_torch(A, B):
    """TileLang矩阵乘法的host侧接口

    Args:
        A: 输入矩阵A [M, K], dtype=float16
        B: 输入矩阵B [K, N], dtype=float16

    Returns:
        C: 输出矩阵 C = A @ B [M, N], dtype=float32
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"矩阵维度不匹配: A.shape={A.shape}, B.shape={B.shape}"

    # Block配置
    block_m, block_n, block_k = 16, 16, 256
    dtype = "float16"
    accum_dtype = "float32"

    C = torch.zeros([M, N], dtype=torch.float32, device=A.device)

    # 创建并编译kernel
    func = matmul_kernel(M, N, K, block_m, block_n, block_k, dtype, accum_dtype)
    compiled = tilelang.compile(func, target="npuir")

    # 运行kernel
    compiled(A, B, C)

    return C


# if __name__ == "__main__":
#     # 测试矩阵乘法
#     print("="*60)
#     print("TileLang-Ascend 矩阵乘法测试")
#     print("="*60)

#     torch.npu.set_device(0)

#     # 参数配置
#     M, N, K = 512, 512, 512

#     print(f"矩阵维度: A[{M}, {K}] @ B[{K}, {N}] = C[{M}, {N}]")

#     # 准备输入数据
#     A = torch.randn([M, K], dtype=torch.float16).npu()
#     B = torch.randn([K, N], dtype=torch.float16).npu()

#     # 运行TileLang实现
#     print("\n运行TileLang kernel...")
#     C = matmul_tilelang_torch(A, B)
#     print("运行完成")

#     # PyTorch参考实现
#     C_ref = torch.matmul(A.float(), B.float())

#     # 验证结果
#     print("\n" + "="*60)
#     print("结果验证")
#     print("="*60)

#     abs_diff = torch.abs(C - C_ref)
#     max_diff = abs_diff.max().item()
#     mean_diff = abs_diff.mean().item()
#     rel_diff = (abs_diff / (torch.abs(C_ref) + 1e-8)).mean().item()

#     print(f"最大绝对误差: {max_diff:.6f}")
#     print(f"平均绝对误差: {mean_diff:.6f}")
#     print(f"平均相对误差: {rel_diff:.6f}")

#     # 显示一些输出值
#     print(f"\nC[0, :5] = {C[0, :5]}")
#     print(f"Expected = {C_ref[0, :5]}")

#     # 判断测试结果
#     print("\n" + "="*60)
#     threshold = 1e-2
#     if max_diff < threshold:
#         print(f"测试通过！最大误差 {max_diff:.6f} < {threshold}")
#     else:
#         print(f"测试失败！最大误差 {max_diff:.6f} >= {threshold}")
#     print("="*60)
