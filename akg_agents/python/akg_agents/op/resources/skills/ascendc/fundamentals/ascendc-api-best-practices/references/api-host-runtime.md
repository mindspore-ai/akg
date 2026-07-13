# Host 侧 Runtime API 使用规范

> **适用范围**：Kernel 直调模式下的 Host 侧代码（`.asc` 中的 `main()` 函数）

---

## 1. 设备初始化 API 调用顺序 ⚠️ **强制**

### 1.1 核数获取 API 选择 ⚠️ **关键**

**根据算子类型选择正确的核数获取 API**：

| 算子类型 | 使用的 API | 说明 |
|---------|-----------|------|
| **纯向量计算**（Add/Mul/Div/Reduce等） | `ACL_DEV_ATTR_VECTOR_CORE_NUM` | 使用 Vector Core 数量 |
| **矩阵计算**（MatMul/Conv等） | `ACL_DEV_ATTR_CUBE_CORE_NUM` | 使用 Cube Core 数量 |
| **混合计算** | `ACL_DEV_ATTR_AICORE_CORE_NUM` | 使用 AI Core 数量 |

**910B3 芯片核数参考**：
- AI Core: 20
- Cube Core: 20
- Vector Core: 40（每个 AI Core 有 2 个 Vector Core）

### 1.2 aclrtGetDeviceInfo 调用要求

**规则**：`aclrtGetDeviceInfo` **必须**在 `aclrtSetDevice` 之后调用

**原因**：获取设备资源前必须先设置设备上下文

**正确示例**（纯向量算子）：
```cpp
int32_t main() {
    // 1. 初始化 ACL
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed, ret=%d\n", ret);
        return ret;
    }

    // 2. 获取设备核数（必须在 aclrtSetDevice 之后）
    int64_t availableCoreNum = 8;  // 默认值
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    if (ret != ACL_SUCCESS) {
        printf("aclrtGetDeviceInfo failed, ret=%d\n", ret);
        aclrtResetDevice(deviceId);
        return ret;
    }

    // 3. 计算使用核数
    uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

    // 4. 后续处理...
}
```

**矩阵算子示例**：
```cpp
// 矩阵计算算子使用 Cube Core 数量
int64_t availableCoreNum = 8;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &availableCoreNum);
```

**错误示例**：
```cpp
int32_t main() {
    // ❌ 错误：未调用 aclrtSetDevice 就调用 aclrtGetDeviceInfo
    int64_t availableCoreNum = 8;
    aclError ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    // 可能返回错误或获取到错误值
}
```

---

## 2. 常见错误

| 错误类型 | 错误示例 | 后果 | 正确做法 |
|---------|---------|------|---------|
| **调用顺序错误** | 未调用 `aclrtSetDevice` 就调用 `aclrtGetDeviceInfo` | 获取核数失败或返回错误值 | 先 `aclrtSetDevice`，再获取资源 |
| **写死核数** | `uint32_t numBlocks = 8;` | 不同设备性能不匹配 | 使用 `aclrtGetDeviceInfo` 动态获取 |
| **API 选择错误** | 纯向量算子用 `ACL_DEV_ATTR_AICORE_CORE_NUM` | 未充分利用 Vector Core | 根据算子类型选择正确的 API |

---

## 3. 完整 Host 侧初始化流程

```cpp
int32_t main() {
    // Step 1: 初始化 ACL
    aclInit(nullptr);
    
    // Step 2: 设置设备
    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    CHECK_ACL(ret);

    // Step 3: 获取设备核数（根据算子类型选择）
    int64_t availableCoreNum = 8;
    // 纯向量算子
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    // 或矩阵算子：ACL_DEV_ATTR_CUBE_CORE_NUM
    // 或混合算子：ACL_DEV_ATTR_AICORE_CORE_NUM
    CHECK_ACL(ret);

    // Step 4: 分配 GM 内存
    size_t gmSize = ...;
    void* gmPtr = nullptr;
    ret = aclrtMalloc(&gmPtr, gmSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL(ret);

    // Step 5: 计算 Tiling 参数
    MyTilingData tiling;
    uint32_t numBlocks = (uint32_t)availableCoreNum;
    computeTiling(tiling, totalRows, numBlocks);

    // Step 6: 启动 Kernel
    KernelCall(..., (uint8_t*)&tiling);

    // Step 7: 清理资源
    aclrtFree(gmPtr);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```

---

## 4. 相关文档

- **代码审查检查项**：[code-review-checklist.md](../../ascendc-kernel-develop-workflow/references/code-review-checklist.md) §0.2.1
