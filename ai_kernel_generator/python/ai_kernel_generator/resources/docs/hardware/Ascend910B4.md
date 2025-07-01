# NPU硬件说明
## memory_system
├── gm (GlobalMemory/DeviceMemory):
|    └── size: 40GB
├── Buffers(per npu_core): // 整个芯片有20个物理npu_core
|    └── vector_core:
|         ├── 数量: 2块 // 每个npu_core有2个vector_core，所以一共有40个vector_core
|         └── vector_buffer: // 一个vector_core独享一块vector_buffer
|              ├── size: 192KB
|              ├── from_data: [\"gm\"]
|              ├── to_data: [\"gm\"]
|              └── data_align: 256B

## compute_system 
├── vector_compute_unit // 每vector_core内的vector_compute_unit、vector_buffer互不关联:
|    ├── 输入Buffer: vector_buffer
|    ├── 输入Buffer: vector_buffer
|    ├── 约束: 需要vector_buffer能放下
|    ├── 约束: 只能做连续、带mask的vector计算
|    └── 功能: numpy常规的vector计算，默认npu都有，例如vector_add、reduce_sum等
└── scalar_compute_unit
     └── 功能：相当于一个小cpu，可以读gm、vector_buffer里的数据，并完成一定的scalar计算


## 搬移Pipeline
- MTE2：从gm搬到vector_buffer // 一次只能处理一个搬运任务
- MTE3：从vector_buffer搬到gm // 一次只能处理一个搬运任务
## 执行逻辑
全部unit（数据搬移Pipeline、计算单元）都是并行的，它们的同步通过set_flag/wait_flag给出，即：前置unit完成执行后set flag，后续unit wait后接收到flag后才开始执行。用户需要显式管理这些flag。

前置需求：
请你思考，并理解上面的npu系统，包括存储结构、数据流向、计算方式等等。