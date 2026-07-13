 # Double Buffering / 双缓冲流水线优化设计
 	 
 	 ## 1. 优化目标
 	 
 	 在 memory-bound 的逐元素算子中，单缓冲（InitBuffer num=1）导致 MTE2（数据搬入）、Vector（计算）、MTE3（数据搬出）三个阶段串行执行，计算单元大量空闲。本优化通过双缓冲（InitBuffer num=2）或多级缓冲策略，使数据搬运与计算流水线并行，**隐藏内存访问延迟**。
 	 
 	 | 指标 | naive | optimized | 收益 |
 	 |------|-------|-----------|------|
 	 | MTE2/VECTOR 重叠度 | 0%（串行） | ~80-90%（流水线并行） | 吞吐量提升 20-50% |
 	 | 计算单元利用率 | 低（大量 idle） | 高（持续有数据可算） | 硬件利用率最大化 |
 	 | UB 内存占用 | `tile_size × num_tensors` | `2 × tile_size × num_tensors` | 增加一倍，需精确预算 |
 	 | 代码复杂度 | 低 | 中（需管理同步事件） | 预取循环 + 事件同步 |
 	 
 	 > 适用算子族：`elementwise`（`sin`, `cos`, `abs`, `exp`, `foreach_add` 等所有 memory-bound 逐元素算子），以及其它 `omni-ops` 通用场景。
 	 
 	 ## 2. 架构概览
 	 
 	 ### 2.1 存储层级与数据流
 	 
 	 > 直接引用 `double_buffering.md` Pipeline Timeline Comparison：
 	 
 	 **单缓冲：**
 	 ```
 	 Single buffer:
 	   MTE2: [LOAD0]          [LOAD1]          [LOAD2]
 	   VEC:         [COMP0]          [COMP1]          [COMP2]
 	   MTE3:               [STORE0]        [STORE1]        [STORE2]
 	   Time: =========================>
 	 ```
 	 
 	 **双缓冲：**
 	 ```
 	 Double buffer:
 	   MTE2: [LOAD0][LOAD1][LOAD2][LOAD3]...
 	   VEC:         [COMP0][COMP1][COMP2]...
 	   MTE3:               [STORE0][STORE1]...
 	   Time: ============>  (significantly shorter)
 	 ```
 	 
 	 ### 2.3 事件同步模型
 	 
 	 | 事件类型 | 含义 | 用途 |
 	 |---------|------|------|
 	 | `HardEvent::MTE2_V` | MTE2 搬入完成 → Vector 可读取 | 数据就绪通知 |
 	 | `HardEvent::V_MTE3` | Vector 完成 → MTE3 可写回 | 计算完成通知 |
 	 | `PipeBarrier<PIPE_V>` | Vector PIPE 内同步 | 指令间数据依赖 |
 	 
 	 ### 2.4 缓冲策略扩展
 	 
 	 | 策略 | Buffer 数量 | 适用场景 | 流水阶段 |
 	 |------|------------|---------|---------|
 	 | Double Buffer | 2 | 通用 memory-bound | 搬入/计算/搬出三级重叠 |
 	 | Triple Buffer | 3 | AIC/AIV 混合核（Cube+Vector） | 搬运/Cube/Vector 三级同时 |
 	 | 2×2 Matrix Buffer | 4 | Matmul 切 M+K 双维度 | M/K 双维度同时流水 |
 	 | TQueBind | 1（VECIN/VECOUT 共享） | 读-改-写 in-place | 节省 50% buffer |
 	 | Custom PingPong | 2+（手写 flag） | FlashAttention 多级存储 | L1/L0 多级 PingPong |
 	 
 	 ## 3. 关键参数配置
 	 
 	 ```cpp
 	 // Host 侧 TilingData
 	 struct DoubleBufferTiling {
 	     uint32_t tileSize;        // 单 tile 数据量（元素数）
 	     uint32_t bufferNum;       // 缓冲数量：2（双缓冲）或 3（三缓冲）
 	     uint32_t ubFactor;        // UB 分块因子，bufferNum 翻倍时减半
 	 };
 	 ```
 	 
 	 ### 3.1 Tile 大小选取原则
 	 
 	 | 参数 | 典型值 | 说明 |
 	 |------|--------|------|
 	 | `tileSize` | 2048 / 4096 / 12288 | 需 32B 对齐；双缓冲时减半 |
 	 | `bufferNum` | 2（通用）/ 3（CV 融合） | 双缓冲为标准，三缓冲用于 Cube+Vector |
 	 | `ubFactor` | 原值 / `bufferNum` | InitBuffer 的 num 翻倍时 ub_factor 必须减半 |
 	 
 	 **UB 内存预算校验：**
 	 ```
 	 Required UB = tile_size × sizeof(dtype) × buffer_num × num_tensors
 	 Available UB = 192KB (Ascend 910B3)
 	 
 	 Example: FP16, 2 tensors (in+out), double buffer
 	   = 12288 × 2 × 2 × 2 = 98304 bytes = 96KB [PASS]
 	 ```
 	 
 	 ### 3.2 自适应双缓冲决策
 	 
 	 当 UB 足够大时可一次性容纳所有数据时，**无需双缓冲**（减少同步开销）；否则启用双缓冲隐藏延迟。
 	 
 	 ```cpp
 	 // Host 侧自适应决策
 	 bool useDb = true;
 	 if (maxRow > rowPerHeadCore) {  // UB 足够大
 	     useDb = false;                // 不使用双缓冲
 	 }
 	 SetTilingKey(context, xDtype, useDb);
 	 ```
 	 
 	 ## 4. 核心计算循环
 	 
 	 ### 4.1 naive 版本（优化前）
 	 
 	 ```cpp
 	 // 单缓冲，串行执行
 	 TQue<QuePosition::VECIN, 1> inQueue;
 	 TQue<QuePosition::VECOUT, 1> outQueue;
 	 pipe.InitBuffer(inQueue, 1, tileSize * sizeof(T));
 	 pipe.InitBuffer(outQueue, 1, tileSize * sizeof(T));
 	 
 	 for (uint32_t i = 0; i < tileNum; i++) {
 	     // 阶段 1：MTE2 搬入（Vector idle!）
 	     LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
 	     DataCopy(inLocal, inGm[i * tileSize], tileSize);
 	     inQueue.EnQue(inLocal);
 	 
 	     // 阶段 2：Vector 计算（MTE2 idle!）
 	     LocalTensor<T> computeLocal = inQueue.DeQue<T>();
 	     Compute(computeLocal, tileSize);
 	 
 	     // 阶段 3：MTE3 搬出（Vector idle!）
 	     LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
 	     DataCopy(outGm[i * tileSize], computeLocal, tileSize);
 	     outQueue.EnQue(outLocal);
 	     outQueue.FreeTensor(outLocal);
 	     inQueue.FreeTensor(computeLocal);
 	 }
 	 ```
 	 
 	 ### 4.2 optimized 版本（优化后）：标准双缓冲
 	 
 	 ```cpp
 	 // 双缓冲 + 预取循环重构
 	 // 注意：TQue 模板参数 depth 为队列深度（连续 EnQue 次数），与 double buffer 无关。
 	 //       非原地操作场景下 depth 推荐为 1（编译器有特殊优化）。
 	 //       double buffer 仅通过 InitBuffer 的 num 参数控制（num=2 开启）。
 	 TQue<QuePosition::VECIN,  1> inQueue;   // depth = 1（推荐值）
 	 TQue<QuePosition::VECOUT, 1> outQueue;  // depth = 1（推荐值）
 	 pipe.InitBuffer(inQueue,  2, tileSize * sizeof(T));  // num = 2 开启 double buffer
 	 pipe.InitBuffer(outQueue, 2, tileSize * sizeof(T));  // num = 2 开启 double buffer
 	 
 	 // 预取：加载第一个 tile
 	 CopyIn(0);
 	 
 	 // 主循环：计算 i，同时预取 i+1，写回 i-1
 	 // 实现 MTE2 / Vector / MTE3 三级硬件流水线并行
 	 for (uint32_t i = 0; i < tileNum - 1; i++) {
 	     CopyIn(i + 1);      // MTE2: 预取下一个 tile（与本轮 Vector 并行）
 	     Compute(i);         // Vector: 计算当前 tile（与上一轮 MTE3 并行）
 	     if (i > 0) {
 	         CopyOut(i - 1); // MTE3: 写回前一个结果（与下一轮 MTE2 并行）
 	     }
 	 }
 	 
 	 // 收尾：处理最后两个 tile（循环内未写回的尾部）
 	 Compute(tileNum - 1);
 	 for (uint32_t i = (tileNum > 1) ? tileNum - 2 : 0; i < tileNum; i++) {
 	     CopyOut(i);
 	 }
 	 ```
 	 
 	 ### 4.3 optimized 版本：事件同步双缓冲（精细控制）
 	 
 	 ```cpp
 	 // 使用 HardEvent 实现更精细的流水线同步
 	 // 注意：以下使用 TBuf 显式分配两份物理 buffer，通过 event 手动同步
 	 TBuf<TPosition::VECCALC> pingBuf, pongBuf;
 	 pipe.InitBuffer(pingBuf, tileSize * sizeof(T));
 	 pipe.InitBuffer(pongBuf, tileSize * sizeof(T));
 	 
 	 event_t pingId = EVENT_ID6;
 	 event_t pongId = EVENT_ID7;
 	 
 	 for (uint32_t idx = 0; idx < tileNum; idx++) {
 	     auto pipeId = (idx % 2 == 0) ? pingId : pongId;
 	     LocalTensor<T> local = (idx % 2 == 0) ? pingBuf.Get<T>() : pongBuf.Get<T>();
 	 
 	     // 等待上一轮该 pipe 的 MTE3 写回完成（避免覆盖未写出的数据）
 	     WaitFlag<HardEvent::MTE3_MTE2>(pipeId);
 	     CopyIn(local, idx * tileSize, tileSize);  // MTE2 搬入
 	     SetFlag<HardEvent::MTE2_V>(pipeId);       // 通知 Vector 数据就绪
 	 
 	     WaitFlag<HardEvent::MTE2_V>(pipeId);      // 等待数据就绪
 	     Compute(local, tileSize);                  // Vector 计算
 	     SetFlag<HardEvent::V_MTE3>(pipeId);       // 通知 MTE3 计算完成
 	 }
 	 ```
 	 
 	 ### 4.4 optimized 版本：三缓冲（Cube+Vector 融合场景）
 	 
 	 ```cpp
 	 // 搬运 / Cube / Vector 三级流水
 	 template<BufferType bufferType, SyncType syncType>
 	 class BuffersPolicy3buff {
 	     Buffer<bufferType, syncType> a_, b_, c_;
 	     uint32_t flag1_ = 0, flag1_vec1_ = 0, flag1_bmm2_ = 0;
 	 public:
 	     Buffer<bufferType, syncType>& Get() {
 	         if (flag1_ == 0) { flag1_ = 1; return a_; }
 	         else if (flag1_ == 1) { flag1_ = 2; return b_; }
 	         else { flag1_ = 0; return c_; }
 	     }
 	     Buffer<bufferType, syncType>& GetVec() {
 	         if (flag1_vec1_ == 0) { flag1_vec1_ = 1; return a_; }
 	         else if (flag1_vec1_ == 1) { flag1_vec1_ = 2; return b_; }
 	         else { flag1_vec1_ = 0; return c_; }
 	     }
 	     Buffer<bufferType, syncType>& GetCube() {
 	         if (flag1_bmm2_ == 0) { flag1_bmm2_ = 1; return a_; }
 	         else if (flag1_bmm2_ == 1) { flag1_bmm2_ = 2; return b_; }
 	         else { flag1_bmm2_ = 0; return c_; }
 	     }
 	 };
 	 ```
 	 
 	 ## 5. 从 naive 到 double_buffer 的关键修改点
 	 
 	 | 修改项 | naive（优化前） | double_buffer（优化后） |
 	 |--------|---------------|----------------------|
 	 | InitBuffer 内存块数 | `num = 1` | `num = 2`（或 3/4） |
 	 | 循环结构 | `for(i) { CopyIn(i); Compute(i); CopyOut(i); }` | 预取 + `for(i) { CopyIn(i+1); Compute(i); CopyOut(i-1); }` |
 	 | UB 用量 | `tile_size × num_tensors` | `2 × tile_size × num_tensors` |
 	 | 同步方式 | 隐式（EnQue/DeQue）或 无 | HardEvent 显式同步 + PipeBarrier |
 	 | Tiling 联动 | 固定 tile_size | tile_size 减半（保持总 UB 预算） |
 	 | 适用场景 | 所有场景（但性能低） | memory-bound、大数据量场景 |
 	 
 	 ## 6. 注意事项 / 约束
 	 
 	 1. **仅改 InitBuffer 的 num 不够！必须重构循环为预取模式**。
 	    - 错误：仅改 `InitBuffer(..., num, ...)` 的 `num=1→2`，循环不变 → 无性能提升。
 	    - 正确：必须实现预取（prefetch）循环。
 	 
 	 2. **Tiling 联动：InitBuffer 的 num 翻倍时 ub_factor 必须减半**。双缓冲使队列内存翻倍，若不调整 ub_factor，总 UB 用量超限会导致编译失败或运行时越界。
 	 
 	 3. **同步指令开销**：HardEvent `SetFlag/WaitFlag` 有轻微开销，仅在必要路径使用。过度同步会抵消流水收益。
 	 
 	 4. **Tail tile 处理**：tile_size 减半后，tile 数量翻倍，需确保尾 tile 处理逻辑正确更新。
 	 
 	 5. **不同 buffer 策略的适用边界**：
 	    - 纯 Vector 算子 → Double Buffer（2 个 buffer）
 	    - Cube+Vector 融合 → Triple Buffer（3 个 buffer）
 	    - Matmul 双维度切分 → 2×2 Matrix Buffer（4 个 buffer）
 	    - 读-改-写 in-place → TQueBind（共享物理 buffer）
 	 
 	 6. **自适应策略**：UB 足够大时（可一次容纳所有数据），不使用双缓冲，减少同步开销。
 	 
 	 7. **同一 TPosition 上 QUE Buffer 数量受限**（参考官方文档）。
 	 - Atlas A2 训练系列产品（如 910B1）：同一 TPosition 上 QUE Buffer 不超过 **8** 块。
 	 - Atlas 训练/推理系列产品：同一 TPosition 上 QUE Buffer 不超过 **4** 块。
 	 - `QuePosition::VECIN` 与 `QuePosition::VECOUT` 底层均映射到 `TPosition::VECCALC`（UB），因此 `inQueue(2) + outQueue(2)` 合计占用 **4 块**。虽未达到 910B1 的上限，但扩展时需注意。若需更多 buffer，应按官方建议**合并到一块 buffer 通过偏移使用**，而非继续增加独立队列。
 	 
 	 8. **事件 ID 管理**：自定义 PingPong 需手动管理 event ID 数组，避免复用冲突。
 	 
 	 ## 7. 实施常见问题与解决方案
 	 
 	 | 问题 | 根因 | 解决方案 |
 	 |------|------|---------|
 	 | 启用双缓冲后无性能提升 | 循环未重构为预取模式 | 必须实现 `CopyIn(i+1); Compute(i); CopyOut(i-1)` 三级流水结构 |
 	 | UB 溢出编译失败 | tile_size 未减半 | InitBuffer 的 num: 1→2 时，ub_factor 必须同步减半 |
 	 | 数据竞争/静默错误 | 同步缺失或事件 ID 复用 | 每对 SetFlag/WaitFlag 使用独立 event ID |
 	 | 尾 tile 处理异常 | tile 数量翻倍后尾 tile 逻辑未更新 | 更新 Host 侧 tiling 计算：tileNum = ceil(total / newTileSize) |
 	 | 三缓冲调试困难 | 三套独立 flag 管理复杂 | 使用类封装（BuffersPolicy3buff），统一 Get/GetVec/GetCube 接口 |
 	 
 	 ## 8. 选型决策与自检清单
 	 
 	 ### 8.1 选型决策
 	 
 	 ```
 	 if (算子是 memory-bound && MTE2 active >> VECTOR active):
 	     if (纯 Vector 算子):
 	         → 启用 Double Buffer（InitBuffer num=2）
 	         → tile_size 减半，保持 UB 预算
 	     elif (Cube+Vector 融合):
 	         → 启用 Triple Buffer（InitBuffer num=3）
 	         → 搬运/Cube/Vector 三级流水
 	     elif (UB 足够大，可一次容纳所有数据):
 	         → 不使用双缓冲（减少同步开销）
 	 else:
 	     → 单缓冲即可（compute-bound 场景双缓冲无收益）
 	 ```
 	 
 	 ### 8.2 自检清单
 	 
 	 - [ ] 循环重构为预取模式：`CopyIn(0)` + `for(i) { CopyIn(i+1); Compute(i); CopyOut(i-1); }`
 	 - [ ] `InitBuffer num: 1→2` 时 `ub_factor` 同步减半
 	 - [ ] 使用 `HardEvent` 或 `PipeBarrier` 确保数据依赖正确
 	 - [ ] 尾 tile 数量随 tile_size 减半而翻倍，尾 tile 逻辑已更新
 	 - [ ] UB 内存预算校验通过：`tile_size × sizeof(dtype) × buffer_num × num_tensors < UB_capacity`
 	 - [ ] 事件 ID 无复用冲突（每对 SetFlag/WaitFlag 独立）