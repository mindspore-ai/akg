# Ascend C API 使用限制与替代方案

> **重要**：使用任何 API 前必读，避免编译错误和运行时问题

---

## 1. 编译期限制

### 1.1 禁止使用 std:: 计算函数

**原因**：Kernel 侧不支持 C++ 标准库，必须使用 Ascend C 提供的专用 API

**触发场景**：所有数学计算、比较操作

**禁止列表**：

| std:: 函数 | ❌ 错误用法 | ✅ Ascend C 替代 | 说明 |
|-----------|----------|----------------|------|
| `std::abs` | `std::abs(x)` | `AscendC::Abs(dst, src, count)` | 绝对值 |
| `std::min/max` | `std::min(a, b)` | `(a < b) ? a : b` 或 `AscendC::Min/Max` | 最小/最大值 |
| `std::sqrt` | `std::sqrt(x)` | `AscendC::Sqrt(dst, src, count)` | 平方根 |
| `std::pow` | `std::pow(x, y)` | `AscendC::Power(dst, src, count)` | 幂运算 |
| `std::exp` | `std::exp(x)` | `AscendC::Exp(dst, src, count)` | 指数 |
| `std::log/log2/log10` | `std::log(x)` | `AscendC::Log/Log2/Log10(dst, src, count)` | 对数 |
| `std::sin/cos/tan` | `std::sin(x)` | `AscendC::Sin/Cos/Tan(dst, src, count)` | 三角函数 |
| `std::floor/ceil/round` | `std::floor(x)` | `AscendC::Floor/Ceil/Round(dst, src, count)` | 取整 |
| `std::isnan/isinf` | `std::isnan(x)` | 手动检查 | 特殊值判断 |

**错误示例**：
```cpp
#include <algorithm>
#include <cmath>

uint32_t result = std::min(a, b);  // ❌ 编译错误
float val = std::sqrt(x);          // ❌ 编译错误
float val = std::exp(x);           // ❌ 编译错误
```

**正确替代**：
```cpp
// min/max：使用三元操作符
uint32_t result = (a < b) ? a : b;  // ✅ min
uint32_t result = (a > b) ? a : b;  // ✅ max

// 或使用 Ascend C API（批量操作）
AscendC::LocalTensor<T> minLocal = minBuf.Get<T>();
AscendC::LocalTensor<T> srcLocal = srcBuf.Get<T>();
AscendC::Min<T>(minLocal, srcLocal, src2Local, count);  // ✅ 批量最小值

// sqrt/exp/log 等：使用 Ascend C API
AscendC::LocalTensor<T> dstLocal = dstBuf.Get<T>();
AscendC::LocalTensor<T> srcLocal = srcBuf.Get<T>();
AscendC::Sqrt<T>(dstLocal, srcLocal, count);  // ✅ 平方根
AscendC::Exp<T>(dstLocal, srcLocal, count);   // ✅ 指数
AscendC::Log<T>(dstLocal, srcLocal, count);   // ✅ 对数
```

**⚠️ 重要**：所有数学计算都必须使用 Ascend C API，不能混用 std:: 函数！

### 1.2 禁止动态内存分配

**原因**：AI Core 无动态内存管理能力

**触发场景**：创建数组、缓冲区等

**错误示例**：
```cpp
std::vector<int> vec;       // ❌ 动态分配
int* ptr = new int[10];     // ❌ 动态分配
int* arr = malloc(100);     // ❌ 动态分配
```

**正确替代**：使用静态分配
```cpp
int arr[10];                          // ✅ 栈分配（Host 侧）
constexpr uint32_t SIZE = 1024;       // ✅ 编译期常量
pipe.InitBuffer(inQueue, 2, SIZE);    // ✅ UB 静态分配（Kernel 侧）
```

### 1.3 Host/Kernel 头文件隔离

**规则**：
- **Host 侧**（`.cpp`）：禁止包含 `kernel_operator.h`
- **Kernel 侧**（`.asc/.h`）：可包含 `kernel_operator.h`

**错误示例**：
```cpp
// host/tiling.cpp
#include "kernel_operator.h"  // ❌ Host 侧禁止
```

**正确用法**：
```cpp
// host/tiling.cpp
#include "tiling.h"  // ✅ 仅必要头文件
#include <cstring>

// kernel/operator.h
#include "kernel_operator.h"  // ✅ Kernel 侧允许
```

---

## 2. API 使用限制索引

以下限制在各专题文档中详细说明：

| 限制类型 | 详细文档 | 核心要点 |
|---------|---------|---------|
| **GM 数据搬运** | [api-datacopy.md](api-datacopy.md) | 禁用 SetValue/GetValue，强制 DataCopyPad |
| **Reduce API** | [api-reduce.md](api-reduce.md) | dst ≠ tmpBuffer，禁用低阶 API |
| **Compare 256字节对齐** | 见下文 2.1 | count 需 256B 对齐，padding 策略 |
| **repeatTime 限制** | [api-repeat-limits.md](api-repeat-limits.md) | uint8_t 最大 255，需分批处理 |
| **流水线同步** | [api-pipeline.md](api-pipeline.md) | MTE/Vector 必须用 EnQue/DeQue 同步 |

### 2.1 Compare API 256字节对齐约束

**约束**：`count` 个元素所占空间必须 **256 字节对齐**

**处理方案**：Padding 策略

```cpp
// 1. 计算对齐大小（float 类型：64 的倍数）
constexpr uint32_t A0 = 32;
constexpr uint32_t A0_ALIGN = (A0 + 63) / 64 * 64;  // = 64

// 2. UB Buffer 使用对齐大小
pipe.InitBuffer(inQueue, 1, R * A0_ALIGN * sizeof(float));

// 3. CopyIn 时填充极值
Duplicate(xLocal, -FLT_MAX, R * A0_ALIGN);  // ArgMax 用极小值
// 再拷贝实际数据到前 A0 个位置

// 4. API 调用使用对齐大小
Compare(cmpLocal, srcLocal, maxLocal, CMPMODE::GT, A0_ALIGN);

// 5. CopyOut 只输出有效数据
DataCopy(dstGm, yLocal, A0);  // 只输出 A0 个
```

**极值选择**：
- ArgMax / 找最大值：`-FLT_MAX` 或 `-INFINITY`
- ArgMin / 找最小值：`FLT_MAX` 或 `INFINITY`

---

## 3. 类型与常量规范

### 3.1 编译期常量

**规则**：Buffer 大小、循环次数等使用 `constexpr`

```cpp
// ✅ 正确：编译期常量
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;

// ❌ 不推荐：运行期常量
const uint32_t buffer_num = 2;  // 可能影响性能
```

### 3.2 类型转换

**规则**：显式类型转换，避免隐式精度损失

```cpp
// ✅ 正确：显式转换
T sumVal = scalarLocal.GetValue(0);
T invSumVal = (T)1.0 / sumVal;  // 显式转换为 T
Muls<T>(dst, src, invSumVal, count);

// ❌ 错误：隐式转换
float val = 1.0 / sumVal;  // 若 T 是 half，精度损失
```

---

## 4. 快速诊断清单

遇到编译错误时，检查：

- [ ] 是否使用了 **任何 std:: 计算函数**（min/max/abs/sqrt/exp/log等）→ 改用 Ascend C API 或基础操作
- [ ] 是否使用了动态内存（`std::vector`, `new`）→ 改用静态分配
- [ ] Host 侧是否包含了 `kernel_operator.h` → 移除该包含
- [ ] Reduce API 的 dst 和 tmp 是否是同一 buffer → 使用不同 buffer
- [ ] 是否使用了 WholeReduce 等低阶 API → 改用高阶 Reduce API
- [ ] 是否使用了 `const` 而非 `constexpr` → 改用 `constexpr`
- [ ] 是否使用了不存在的类型（如 `TensorShape`）→ 查阅正确 API
- [ ] Compare API 的 count 是否满足 256 字节对齐 → 使用 padding 策略

---

## 5. 相关文档

- [api-datacopy.md](api-datacopy.md)：DataCopyPad 使用规范
- [api-reduce.md](api-reduce.md)：Reduce API 详细用法
- [api-repeat-limits.md](api-repeat-limits.md)：repeatTime 限制与处理
- [api-buffer.md](api-buffer.md)：Buffer 管理最佳实践
- [api-precision.md](api-precision.md)：精度转换规范
