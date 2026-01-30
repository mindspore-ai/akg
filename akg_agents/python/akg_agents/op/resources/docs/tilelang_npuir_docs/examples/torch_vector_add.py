"""TileLang-Ascend 向量加法示例 (PyTorch)"""

import torch
import torch_npu
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()


def vec_add_kernel(N, block_N, dtype="float32"):
    """创建向量加法内核

    Args:
        N: 向量长度
        block_N: 块大小
        dtype: 数据类型
    """
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        B: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
        shape: T.int32,
    ):
        """向量加法内核: C = A + B"""
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # 分配UB缓冲区
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)

            # 计算边界（官方模式）
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)

            # 数据加载
            T.copy(A[cid * block_N], A_VEC, [tail_size])
            T.copy(B[cid * block_N], B_VEC, [tail_size])

            # 向量加法
            T.npuir_add(A_VEC, B_VEC, C_VEC)

            # 存储结果
            T.copy(C_VEC, C[cid * block_N], [tail_size])

    return main


def vector_add_tilelang_torch(A, B):
    """TileLang向量加法的host侧接口

    Args:
        A: 输入向量A
        B: 输入向量B

    Returns:
        C: 输出向量 C = A + B
    """
    N = A.shape[0]
    C = torch.empty_like(A)

    # 创建并编译kernel
    func = vec_add_kernel(N, N, A.dtype)
    compiled = tilelang.compile(func, target="npuir")

    # 运行kernel
    compiled(A, B, C, N)

    return C


# if __name__ == "__main__":
#     # 测试向量加法
#     torch.npu.set_device(0)

#     # 参数配置
#     seq_len = 4096
#     dtype = "float32"

#     # 准备输入数据
#     v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
#     v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()

#     # 运行TileLang实现
#     v3 = vector_add_tilelang_torch(v1, v2)

#     # 计算参考结果
#     y_ref = v1 + v2

#     # 验证结果
#     max_diff = torch.max(torch.abs(v3 - y_ref)).item()
#     print(f"Max difference: {max_diff}")
#     assert max_diff < 1e-5, f"Result mismatch! Max diff: {max_diff}"
#     print("Test passed!")
