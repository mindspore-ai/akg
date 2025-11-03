# 任务特征
**操作类型**：broadcast类型，broadcast第二根轴；3D Tensor输入，3D Tensor输出
**数据尺寸**：(64, 128, 128)、(64, 1, 1)，数据shape较小
**数据类型**：float16
**任务特点**：操作类型为elementwise，可以向量化操作；triton kernel里面可以直接load一个向量进行单词操作；需要对第二维和第三维进行广播，考虑将这两维摊平；选择切分B，并在 kernel 内部通过 for 循环分块处理HW，为 double buffering 和指令流水提供了优化空间。同时，将 grid size 限制在物理核数以内，以匹配硬件并行能力，提升整体执行效率。

# 关键代码切片

## 优化1
```python
# 优化Triton切分配置：
# BLOCK_B和SUB_HW控制B和HW维度的切分
triton.Config({'BLOCK_B': 2, 'SUB_HW': 16384}), # 最优，核数=32，被shape整除
triton.Config({'BLOCK_B': 2, 'SUB_HW': 8096}), # HW偏小
triton.Config({'BLOCK_B': 4, 'SUB_HW': 16384}), # 核数=16，太小
triton.Config({'BLOCK_B': 1, 'SUB_HW': 16384}), # 核数=64，大于40
# Grid: 根据BLOCK_B计算需要多少个kernel
grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
```
**优化内容**：通过设置grid大小小为B/BLOCK_B，控制其小于等于物理核数40，降低调度开销；并且让SUB_HW尽可能大，去用满ub。

**总结**：[通用优化] 在Ascend平台上，当triton kernel的grid数较高时，应调整BLOCK_B参数（grid = B/BLOCK_B），使grid降低至真实物理核（AI Vector）数（通常为40核），避免过多核数带来的调度开销。SUB_HW参数用于控制内部HW维度切分粒度，尽可能大以提升UB利用率。

## 优化2
```python
# 将H和W两个维度摊平为一维
HW = H * W  # 128 * 128 = 16384

# 循环处理摊平后的HW维度
for hw_block in range(0, HW, SUB_HW):
    hw_offs = hw_block + tl.arange(0, SUB_HW)
    mask_hw = hw_offs < HW
    
    # 使用线性索引访问：offset = b * stride_b + hw
    input1_offs = b_indices[:, None] * stride_input1_b + hw_offs[None, :]
```
**优化内容**：由于input2在H和W两个维度都需要广播（shape为(B, 1, 1)），将H和W维度摊平为一维HW（H×W=16384），简化了地址计算和循环控制逻辑，只需一层循环即可处理所有空间位置，避免了嵌套两层循环的开销。

**总结**：[通用优化] 在Ascend平台上，当多个连续维度都需要广播或处理方式相同时，可以将这些维度摊平为一维，简化循环结构和地址计算，提升kernel执行效率。