# Group Reduce（跨核归约）

## 3. Group Reduce（跨核归约）

**适用场景**: R 太大，单核无法遍历完；同时 A 轴太小不能充分利用多核

### 3.1 两阶段执行模型

```
Phase 1（各核独立）:
  ┌──────┐  ┌──────┐  ┌──────┐
  │Core 0│  │Core 1│  │Core 2│
  │R[0:K]│  │R[K:2K]│ │R[2K:N]│
  └──┬───┘  └──┬───┘  └──┬───┘
     │          │          │
     ↓          ↓          ↓
  workspace[0] workspace[1] workspace[2]
     ↓          ↓          ↓
  ┌────────────────────────────┐
  │        SyncAll()           │
  └────────────────────────────┘
     ↓
Phase 2（合并核）:
  read workspace[0..coreNum]
  merge all partials → final output
```

### 3.2 Phase 1 实现模板

```cpp
void GroupReducePhase1() {
    int myRStart = rGroupIdx * rPerGroup;
    int myREnd = min(myRStart + rPerGroup, totalR);

    // 初始化 partial
    Duplicate(partialBuf, initValue, outSize);  // 0 for sum, -inf for max

    for (int r = myRStart; r < myREnd; r += cutRSize) {
        int curR = min(cutRSize, myREnd - r);
        CopyIn(xLocal, r, curR);
        // 局部归约
        ReduceOp(partialBuf, partialBuf, xLocal, curR);
    }

    // 写 partial 到 workspace
    int wsOffset = blockIdx * SLOT_STRIDE;  // 64B 对齐防 bank conflict
    DataCopyPad(workspaceGm[wsOffset], partialBuf, {1, outSize * sizeof(float), 0, 0});
}
```

### 3.3 Phase 2 实现模板

```cpp
void GroupReducePhase2() {
    SyncAll();  // 等所有核完成 Phase1

    // 合并所有 partial
    Duplicate(finalBuf, initValue, outSize);

    for (int g = 0; g < groupR; g++) {
        int wsOffset = (aBlockIdx * groupR + g) * SLOT_STRIDE;
        CopyIn(partialLocal, workspaceGm[wsOffset], outSize);
        ReduceOp(finalBuf, finalBuf, partialLocal, outSize);
    }

    // 写出最终结果
    CopyOut(yGm[myAStart], finalBuf, outSize);
}
```

### 3.4 Welford Group Reduce（统计归约专用）

对于 reduce_var，Phase 1 输出的是 (partial_mean, partial_M2, partial_count) 三元组，
Phase 2 用 Welford 合并公式合并：

```cpp
void WelfordGroupReducePhase2() {
    SyncAll();

    // 读第一组作为初始值
    float totalMean = workspace_mean[0];
    float totalM2 = workspace_M2[0];
    int totalCount = workspace_count[0];

    // 逐组合并
    for (int g = 1; g < groupR; g++) {
        float gMean = workspace_mean[g];
        float gM2 = workspace_M2[g];
        int gCount = workspace_count[g];

        float delta = gMean - totalMean;
        int newCount = totalCount + gCount;
        totalMean += delta * gCount / newCount;
        totalM2 += gM2 + delta * delta * totalCount * gCount / newCount;
        totalCount = newCount;
    }

    float var = totalM2 / (totalCount - correction);
}
```

