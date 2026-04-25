"""TileLang-Ascend Softmax示例 (PyTorch)

实现行级Softmax操作，支持自动广播。

遵循的关键规则：
1. 函数名必须为main
2. Tensor形状用元组 (M, N)
3. 所有buffer统一为2D: [1, N] 和 [1, 1]
4. 使用T.Scope("Vector")包裹向量操作
5. 数值稳定：先减去最大值再exp
6. dims=[1] 对第二维度reduce: [1,N] → [1,1]
7. 利用自动广播：[1,N] op [1,1]
"""

import torch
import torch_npu
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()


def softmax_kernel(M, N, dtype="float32"):
    """创建Softmax内核

    支持自动广播，代码简洁高效。

    实现公式: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Args:
        M: 批次大小（行数）
        N: 特征维度（列数）
        dtype: 数据类型

    Returns:
        编译就绪的TileLang内核函数
    """
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
        shape: T.int32,
    ):
        """Softmax内核: 对每行独立计算softmax

        流程:
        1. 找最大值: max_val = max(x)（数值稳定性）
        2. 减去最大值: x' = x - max_val
        3. 指数运算: exp_x = exp(x')
        4. 求和: sum_exp = sum(exp_x)
        5. 归一化: y = exp_x / sum_exp
        """
        with T.Kernel(M, is_npu=True) as (row_id, _):
            # 分配UB缓冲区（统一为2D格式）
            x_ub = T.alloc_ub([1, N], dtype)   # [1, N] 存储一行数据
            max_ub = T.alloc_ub([1, 1], dtype)  # [1, 1] 存储最大值
            sum_ub = T.alloc_ub([1, 1], dtype)  # [1, 1] 存储指数和

            with T.Scope("Vector"):
                # 加载一行数据
                T.copy(X[row_id, 0], x_ub[0, 0], [1, N])

                # Step 1: 找最大值 [1,N] → [1,1]
                T.npuir_reduce(x_ub, max_ub, dims=[1], reduce_mode="max")

                # Step 2: x = x - max（自动广播 [1,N] - [1,1]）
                T.npuir_sub(x_ub, max_ub, x_ub)

                # Step 3: x = exp(x)
                T.npuir_exp(x_ub, x_ub)

                # Step 4: 计算指数和 [1,N] → [1,1]
                T.npuir_reduce(x_ub, sum_ub, dims=[1], reduce_mode="sum")

                # Step 5: 归一化（自动广播 [1,N] / [1,1]）
                T.npuir_div(x_ub, sum_ub, x_ub)

                # 存储结果
                T.copy(x_ub[0, 0], Y[row_id, 0], [1, N])

    return main


def softmax_tilelang_torch(X):
    """TileLang Softmax的host侧接口

    Args:
        X: 输入矩阵 [M, N], dtype=float32

    Returns:
        Y: 输出矩阵（归一化后） [M, N], dtype=float32
    """
    M, N = X.shape
    Y = torch.zeros_like(X)

    # 创建并编译kernel
    func = softmax_kernel(M, N, "float32")
    compiled = tilelang.compile(func, target="npuir")

    # 运行kernel
    compiled(X, Y, M)

    return Y


# if __name__ == "__main__":
#     # 测试Softmax实现
#     print("="*60)
#     print("TileLang-Ascend Softmax 测试")
#     print("="*60)

#     # 设置NPU设备
#     torch.npu.set_device(0)

#     # 参数配置
#     M, N = 64, 512  # 64行，每行512个元素

#     print(f"配置: M={M}, N={N}")

#     # 准备输入数据
#     X = torch.randn(size=[M, N], dtype=torch.float32).npu()

#     # 运行TileLang实现
#     print("\n运行TileLang kernel...")
#     Y = softmax_tilelang_torch(X)
#     print("运行完成")

#     # 计算参考结果（PyTorch）
#     Y_ref = torch.nn.functional.softmax(X, dim=1)

#     # 验证结果
#     print("\n" + "="*60)
#     print("结果验证")
#     print("="*60)

#     # 1. 检查输出范围
#     print(f"输入范围: [{X.min().item():.4f}, {X.max().item():.4f}]")
#     print(f"输出范围: [{Y.min().item():.4f}, {Y.max().item():.4f}]")
#     print(f"期望范围: [{Y_ref.min().item():.4f}, {Y_ref.max().item():.4f}]")

#     # 2. 检查每行求和是否为1
#     Y_sum = Y.sum(dim=1)
#     Y_ref_sum = Y_ref.sum(dim=1)
#     print(f"\n每行求和（前5行）:")
#     print(f"TileLang: {Y_sum[:5]}")
#     print(f"PyTorch:  {Y_ref_sum[:5]}")

#     # 3. 计算误差
#     abs_diff = torch.abs(Y - Y_ref)
#     max_diff = abs_diff.max().item()
#     mean_diff = abs_diff.mean().item()

#     print(f"\n误差统计:")
#     print(f"最大误差: {max_diff:.6f}")
#     print(f"平均误差: {mean_diff:.6f}")

#     # 4. 显示第一行详细对比
#     print(f"\n第一行前5个元素对比:")
#     print(f"TileLang: {Y[0, :5]}")
#     print(f"PyTorch:  {Y_ref[0, :5]}")

#     # 5. 判断测试结果
#     print("\n" + "="*60)
#     if max_diff < 1e-5:
#         print("测试通过！")
#         print(f"最大误差 {max_diff:.6f} < 1e-5")
#     else:
#         print("测试失败！")
#         print(f"最大误差 {max_diff:.6f} >= 1e-5")

#         # 显示错误最大的位置
#         max_idx = abs_diff.argmax()
#         row, col = max_idx // N, max_idx % N
#         print(f"\n最大误差位置: [{row}, {col}]")
#         print(f"TileLang值: {Y[row, col].item():.6f}")
#         print(f"PyTorch值:  {Y_ref[row, col].item():.6f}")
#     print("="*60)
