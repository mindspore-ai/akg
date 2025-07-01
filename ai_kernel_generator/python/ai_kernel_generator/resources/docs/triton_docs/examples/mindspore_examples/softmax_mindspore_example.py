import torch
import triton
import triton.language as tl
import mindspore as ms


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Triton softmax 内核
    每个程序处理一行数据
    """
    # 获取当前程序处理的行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        # 计算当前行的起始指针
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # 创建列偏移
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        # 加载数据，使用掩码处理边界
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # 数值稳定性：减去最大值
        row_minus_max = row - tl.max(row, axis=0)

        # 计算指数
        numerator = tl.exp(row_minus_max)

        # 计算分母（归一化因子）
        denominator = tl.sum(numerator, axis=0)

        # 计算 softmax
        softmax_output = numerator / denominator

        # 存储结果
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton_mindspore(x):
    """
    Triton softmax 启动函数
    """
    n_rows, n_cols = x.shape

    # 块大小是大于 n_cols 的最小 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 分配输出张量
    y = ms.mint.empty_like(x)

    # 启动内核
    num_programs = min(32, n_rows)  # 限制程序数量

    softmax_kernel[(num_programs, 1, 1)](
        y,                    # output_ptr
        x,                    # input_ptr
        x.stride(0),          # input_row_stride
        y.stride(0),          # output_row_stride
        n_rows,               # n_rows
        n_cols,               # n_cols
        BLOCK_SIZE            # BLOCK_SIZE
    )

    return y

# if __name__ == "__main__":
#     x = ms.mint.randn(1823, 781)
#     res = softmax_triton_mindspore(x)
#     print(x)
