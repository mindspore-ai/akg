---
name: triton-ascend-error-fix
description: triton-ascend 常见错误修复：CBUF溢出、BiShengHIR编译失败、语法错误、变量名拼写错误、GroupNorm循环逻辑错误、张量连续性问题
category: case
version: 1.0.0
metadata:
  source: error_fix
  case_type: fix
  backend: ascend
  dsl: triton_ascend
---

### CBUF 溢出

- **报错特征**: `cbuf overflow`
- **根因**: 分块参数（BLOCK_M/N/K）过大，超出 Ascend UB 容量
- **修复**: 减小分块参数，通常 BLOCK_M=64, BLOCK_N=128 是安全起点

```python
# 错误：分块过大
BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

# 修复：缩小分块
BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 256
```

### BiShengHIR 编译失败

- **报错特征**: `Failed to run BiShengHIR pipeline`
- **根因**: `tl.store`/`tl.load` 中内联复杂地址计算导致编译器无法处理
- **修复**: 将地址计算拆分为中间变量

```python
# 错误：内联复杂地址
tl.store(c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn, acc, mask=mask)

# 修复：拆分为中间变量
c_ptrs = c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
tl.store(c_ptrs, acc, mask=mask)
```

### 语法与拼写错误

- **报错特征**: `SyntaxError: invalid syntax` / `NameError`
- **根因**: LLM 生成代码时出现重复 token、变量名拼写不一致
- **修复**: 检查重复关键字和变量名拼写一致性

```python
# 常见错误模式
offsets offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 重复变量名
C = torch.empty((M, N), dtype=torch=torch.float32, device=device)   # 重复 torch=
w_val = tl.load(ptr + c_clobal, ...)   # c_clobal 应为 c_global

# 修复：确保每个变量声明唯一，拼写一致
offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
C = torch.empty((M, N), dtype=torch.float32, device=device)
w_val = tl.load(ptr + c_global, ...)
```

### GroupNorm 循环与索引错误

- **报错特征**: 计算结果不正确
- **根因**: 任务映射方式过于复杂，多维偏移计算容易出错
- **修复**: 使用清晰的循环嵌套 + `local_idx // hw_size` 局部索引

```python
# 错误：复杂的一维展平映射
total_tasks = N * G
for task_idx in range(pid, total_tasks, CORE_NUM):
    n_idx = task_idx // G
    g_idx = task_idx % G
    c_local = offsets // S
    hw = offsets - c_local * S

# 修复：清晰的多层循环
for n in range(pid, N, CORE_NUM):
    for g_idx in range(num_groups):
        for i in range(0, group_elems, BLOCK_SIZE):
            local_idx = i + tl.arange(0, BLOCK_SIZE)
            c_local = local_idx // hw_size
            spatial_idx = local_idx % hw_size
```

### 张量连续性

- **报错特征**: 无明确报错，但结果可能错误
- **修复**: 在 kernel wrapper 入口处强制 `.contiguous()`

```python
if not x.is_contiguous():
    x = x.contiguous()
```

---
## Quick Checklist

生成代码前请逐项检查：

- [ ] CBUF 溢出
- [ ] BiShengHIR 编译失败
- [ ] 语法与拼写错误
- [ ] GroupNorm 循环与索引错误
- [ ] 张量连续性
