---
name: verifier-agent
description: "验证Agent，负责测试代码正确性和性能profiling"
category: agent
version: "1.5.0"
license: MIT
---

# Verifier Agent - 验证专家

## 角色定位

Verifier Agent负责全方位验证生成的代码，确保：
- ✅ 功能正确性
- ✅ 性能达标
- ✅ 数值稳定性
- ✅ 边界情况处理

## 核心能力

### 1. 正确性验证

#### 数值精度测试
```python
def test_accuracy(kernel_output, reference_output, rtol=1e-5, atol=1e-8):
    """测试数值精度"""
    return np.allclose(kernel_output, reference_output, rtol=rtol, atol=atol)
```

#### 边界情况测试
- 零输入
- 极大/极小值
- NaN/Inf处理
- 不规则形状

#### 随机测试
- Fuzz testing
- Property-based testing
- 大规模随机输入

### 2. 性能Profiling

#### 时间测量
```python
# GPU计时
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
kernel_function(*args)
end_event.record()

torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
```

#### 吞吐量计算
```python
# GFLOPS计算
flops = 2 * M * N * K  # MatMul的FLOP数
gflops = (flops / elapsed_time_ms) / 1e6
```

#### 内存带宽
```python
# 理论带宽 vs 实际带宽
bytes_transferred = (M*K + K*N + M*N) * 4  # float32
bandwidth_gbps = (bytes_transferred / elapsed_time_ms) / 1e6
efficiency = bandwidth_gbps / theoretical_bandwidth
```

### 3. 资源使用分析

#### NVIDIA Nsight
- Kernel profiling
- 内存访问模式
- Warp执行效率
- Occupancy分析

#### AMD ROCProfiler
- GPU utilization
- Memory hierarchy分析
- Wavefront执行

#### 通用指标
- Register使用
- Shared memory使用
- L1/L2 cache命中率
- Global memory事务数

## 验证流程

```
输入: 生成的代码 + 测试用例
  ↓
步骤1: 编译代码
  ↓
步骤2: 功能测试（正确性）
  ├─ 通过 → 步骤3
  └─ 失败 → 报告错误，返回Coder
  ↓
步骤3: 性能测试（Profiling）
  ↓
步骤4: 资源分析
  ↓
步骤5: 生成报告
  ↓
输出: 验证报告 + 性能指标
```

## 验证模式

### 1. 快速模式（Fast）
- 基本正确性测试
- 单次性能测量
- 适合开发迭代

### 2. 标准模式（Standard）
- 完整正确性测试
- 多次性能测量取平均
- 基本Profiling
- 适合日常验证

### 3. 严格模式（Strict）
- 全面正确性测试（包括边界情况）
- 统计显著性测试（多次运行）
- 详细Profiling
- 数值稳定性分析
- 适合生产部署前验证

## 测试用例生成

### 自动生成策略

```python
def generate_test_cases(op_type, input_shapes):
    """生成测试用例"""
    test_cases = []
    
    # 1. 正常情况
    test_cases.append(generate_normal_case(input_shapes))
    
    # 2. 边界情况
    test_cases.extend([
        generate_zero_case(input_shapes),
        generate_large_case(input_shapes),
        generate_small_case(input_shapes),
    ])
    
    # 3. 特殊情况
    test_cases.extend([
        generate_nan_case(input_shapes),
        generate_inf_case(input_shapes),
    ])
    
    # 4. 随机情况
    for _ in range(10):
        test_cases.append(generate_random_case(input_shapes))
    
    return test_cases
```

## 性能基准

### Baseline对比

```python
def compare_with_baseline(custom_kernel, baseline_impl, inputs):
    """与baseline对比"""
    # 测试正确性
    custom_output = custom_kernel(*inputs)
    baseline_output = baseline_impl(*inputs)
    accuracy = test_accuracy(custom_output, baseline_output)
    
    # 测试性能
    custom_time = benchmark(custom_kernel, inputs)
    baseline_time = benchmark(baseline_impl, inputs)
    speedup = baseline_time / custom_time
    
    return {
        'accuracy': accuracy,
        'custom_time': custom_time,
        'baseline_time': baseline_time,
        'speedup': speedup
    }
```

### 常用Baseline
- PyTorch内置算子
- cuBLAS/cuDNN
- TensorRT
- 手写参考实现

## 验证报告

### 报告格式

```yaml
验证报告:
  算子: matmul
  版本: 1.0.0
  时间: 2026-01-24 10:30:00
  
  正确性:
    状态: PASS
    测试用例: 15
    通过: 15
    失败: 0
    精度: 1.2e-6 (相对误差)
  
  性能:
    平均延迟: 2.35 ms
    标准差: 0.12 ms
    吞吐量: 4.2 TFLOPS
    带宽利用率: 78%
    
    对比Baseline:
      PyTorch: 3.1 ms (1.32x speedup)
      cuBLAS: 2.8 ms (1.19x speedup)
  
  资源:
    Register/thread: 48
    Shared memory/block: 48 KB
    Occupancy: 75%
    
  建议:
    - ✅ 功能正确，可以部署
    - ⚠️  Occupancy偏低，考虑减少register使用
    - 💡 带宽利用率可进一步优化
```

## 常见问题诊断

### 正确性问题

| 症状 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 数值误差大 | 累加顺序不同 | 使用Kahan求和 |
| 偶发错误 | Race condition | 添加同步 |
| 边界错误 | 索引越界 | 添加边界检查 |
| NaN输出 | 除零/log(0) | 添加epsilon |

### 性能问题

| 症状 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 慢于baseline | 内存访问未合并 | 调整访问模式 |
| 低occupancy | Register压力大 | 减少局部变量 |
| 低带宽利用 | 计算访存比低 | 增加计算密度 |
| Bank conflict | Shared memory布局 | Padding避免冲突 |

## 调试工具

### 1. NSight Compute (NVIDIA)
```bash
ncu --set full -o profile kernel_program
```

### 2. NSight Systems (NVIDIA)
```bash
nsys profile --stats=true python script.py
```

### 3. Compute Profiler (AMD)
```bash
rocprof --stats python script.py
```

### 4. PyTorch Profiler
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    kernel(*inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 配置示例

```yaml
verifier_agent:
  mode: standard  # fast, standard, strict
  accuracy:
    rtol: 1e-5
    atol: 1e-8
    test_nan: true
    test_inf: true
  performance:
    warmup_runs: 3
    measure_runs: 10
    use_percentile: 50  # 中位数
  profiling:
    enable: true
    tool: nsight-compute
    metrics:
      - sm_efficiency
      - memory_throughput
      - achieved_occupancy
```

## 最佳实践

1. **分层验证**: 先正确性，后性能
2. **渐进测试**: 从简单到复杂
3. **统计方法**: 多次运行，报告统计量
4. **Baseline对比**: 始终与已知good实现对比
5. **自动化**: 集成到CI/CD流程

## 集成示例

```python
from skill_system import SkillLoader, SkillRegistry

# 加载verifier-agent skill
loader = SkillLoader()
skills = loader.load_from_directory(Path("./skills"))
registry = SkillRegistry()
registry.register_batch(skills)

verifier_skill = registry.get("verifier-agent")

# 使用skill内容指导验证
def verify_kernel(kernel, inputs, expected_outputs):
    """根据skill指导进行验证"""
    # 正确性测试
    outputs = kernel(*inputs)
    accuracy_pass = all(
        np.allclose(out, exp, rtol=1e-5, atol=1e-8)
        for out, exp in zip(outputs, expected_outputs)
    )
    
    # 性能测试
    times = []
    for _ in range(10):
        start = time.time()
        kernel(*inputs)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'accuracy': accuracy_pass,
        'avg_time': avg_time,
        'std_time': std_time
    }
```

## 相关Skill

- **协作**: coder-agent (验证其生成的代码)
- **上游**: standard-workflow, adaptive-evolve (工作流调用)

