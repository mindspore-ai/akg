# msSanitizer 内存检测工具指南

> **文档版本**: MindStudio 80.RC1
> **适用产品**: Atlas 系列开发者套件/模组

## 1. 概述

内存检测（Memory Check）是针对用户程序运行时的一种异常检测功能。**msSanitizer** 工具可以检测并报告算子运行中对外部存储（Global Memory）和内部存储（Local Memory）的越界及未对齐等内存访问异常。

> **⚠️ 注意**:
> *   msSanitizer 工具**不支持**对加速库（Ascend Transformer Boost）的算子仓进行内存检测。
> *   当用户使用 PyTorch 等框架接入算子时，框架内部可能会通过内存池管理 GM 内存。此时需使用手动上报接口（`SanitizerReportMalloc` / `SanitizerReportFree`）以确保检测准确性。

---

## 2. 支持的内存异常类型

内存检测能够识别并报告以下六类核心异常：

| 异常名称 | 描述 | 发生位置 | 支持地址空间 |
| :--- | :--- | :--- | :--- |
| **非法读写**<br>(Illegal Read/Write) | 访问了未分配的内存区域。 | Kernel, Host | GM, UB, L0{A,B,C}, L1 |
| **多核踩踏**<br>(Multi-core Overwrite) | 多个 AI Core 访问了重叠的内存区域，且至少有一个核进行了写入操作。 | Kernel | GM |
| **非对齐访问**<br>(Misaligned Access) | DMA 搬运数据的地址未满足硬件最小访问粒度对齐要求。 | Kernel | GM, UB, L0{A,B,C}, L1 |
| **非法释放**<br>(Illegal Free) | 尝试释放未分配或已释放的内存地址。 | Host | GM |
| **内存泄漏**<br>(Memory Leak) | 申请内存后未释放，导致运行过程中内存占用持续增加。 | Host | GM |
| **分配内存未使用**<br>(Unused Memory) | 内存分配后直到程序结束都未被访问或使用。 | Kernel, Host | GM |

---

## 3. 启用内存检测

运行 `msSanitizer` 工具时，默认启用基础内存检测功能（memcheck）。

### 3.1 基础检测命令
开启默认的非法读写、多核踩踏、非对齐访问和非法释放检测：
```bash
mssanitizer --tool=memcheck <application>
```

### 3.2 高级检测选项

*   **开启内存泄漏检测**:
    若需检测内存泄漏，需显式添加 `--leak-check=yes` 参数：
    ```bash
    mssanitizer --tool=memcheck --leak-check=yes <application>
    ```

*   **开启分配内存未使用检测**:
    若需检测已分配但未使用的内存，需显式添加 `--check-unused-memory=yes` 参数：
    ```bash
    mssanitizer --tool=memcheck --check-unused-memory=yes <application>
    ```

> **💡 提示**:
> *   异常报告将在用户程序运行完成后打印到终端。
> *   该工具也支持对 HCCL 通信接口（如 AllReduce, AllGather 等）及通算融合类算子的非法读写检测。

---

## 4. 内存异常报告解析

以下是各类异常的典型报告格式及解读方法。

### 4.1 非法读写 (Illegal Read/Write)
**含义**: 算子访问了未分配的 GM 或片上内存（超出硬件容量）。

**示例报告**:
```text
====== ERROR : illegal read of size 224 
====== at 0x12c0c0015000 on GM in add_custom_kernel 
====== in block aiv(0) on device 0 
====== code in pc current 0x77c (serialNo: 10) 
====== #0  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_data_copy_impl.h:58:9 
====== #1 ...
====== #3 illegal_read_and_write/add_custom.cpp:18:5 
```


解读:

错误类型: 非法读取 224 字节。

位置: GM 地址 0x12c0c0015000，发生在 add_custom_kernel 核函数中。

代码定位: 对应源文件 add_custom.cpp 第 18 行。

注意: 若编译时未添加调试选项，可能不会显示 #0 到 #3 的调用栈信息。


### 4.2 多核踩踏

AI Core 是昇腾 AI 处理器中的计算核心，AI 处理器内部有多个 AI Core，算子运行就在这些 AI Core 上。这些 AI Core 会在计算过程中从 GM 上搬入或搬出数据。当没有显式地进行核间同步时，如果各个核之间访问的 GM 内存存在重叠并且至少有一个核对重叠地址进行写入时，则会发生多核踩踏问题。这里我们通过所有者的概念来保证多核之间不会发生踩踏问题，当一块内存被某一个核写入后，这块内存就由该核所有。当其他核对这块内存进行访问时就会产生 out of bounds 异常。

**示例报告：**

```text
====== WARNING : out of bounds of size 256 
// 异常的基本信息，包含发生踩踏的字节数
====== at 0x12c0c00150fc on GM when writing data in add_custom_kernel 
// 异常发生的内存位置信息，包含发生的核函数名、地址空间与内存地址，此处的内存地址指一次内存访问中的首地址
====== in block aiv(9) on device 0 
// 异常代码对应 vector 核的 block 索引
====== code in pc current 0x7b8 (serialNo: 22) 
// 当前异常发生的 pc 指针和调用 api 行为的序列号
====== #0  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_data_copy_impl.h:103:9 
// 以下为异常发生代码的调用栈，包含文件名、行号和列号
====== #1  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/inner_interface/inner_kernel_operator_data_copy_intf.cppm:155:9
====== #2  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/inner_interface/inner_kernel_operator_data_copy_intf.cppm:461:5
====== #3 out_of_bound/add_custom.cpp:21:5
```
以上示例中，共有 256 个字节的访问发生踩踏，对 GM 上的“0x12c0c00150fc”地址进行访问时存在多核踩踏，且导致异常发生的指令对应于算子实现文件 add_custom.cpp 的第 21 行。

### 4.3 非对齐访问 (Misaligned Access)

含义: 访问地址不符合 DMA 搬运的最小粒度要求（如 32 字节或 128 字节对齐），可能导致数据错误或 AI Core 异常。

示例报告:
```text
====== ERROR : misaligned access of size 13 
// 异常的基本信息，包含发生对齐异常操作的字节数
====== at 0x6 on UB in add_custom_kernel 
// 异常发生的内存位置信息，包含发生的核函数名、地址空间与内存地址
====== in block aiv(0) on device 0 
// 异常代码对应 vector 核的 block 索引
====== code in pc current 0x780 (serialNo: 33) 
// 当前异常发生的 pc 指针和调用 api 行为的序列号
====== #0  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_data_copy_impl.h:103:9 
// 以下为异常发生代码的调用栈，包含文件名、行号和列号
====== #1  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/inner_interface/inner_kernel_operator_data_copy_intf.cppm:155:9
====== #2  $ {ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/inner_interface/inner_kernel_operator_data_copy_intf.cppm:461:5
====== #3 illegal_align/add_custom.cpp:18:5
```
以上示例中，共有针对 13 个字节的对齐异常访问，对 UB 上的“0x6”地址进行访问时存在对齐问题，且导致异常发生的指令对应于算子实现文件 add_custom.cpp 的第 18 行。
注意：不添加编译选项的情况下，异常报告将不会出现调用栈信息。

### 4.4 内存泄漏
内存检测可以检测出 Device 侧的内存泄漏问题，这些问题通常是开发者没有正确释放使用 AscendCL 接口申请的内存导致的，由于内部存储（Local Memory）目前不存在内存分配的概念，因此内存泄漏只可能出现在 GM 上。通过指定命令行参数 --leak-check=yes 可以开启内存泄漏检测。

示例报告:
```text
====== ERROR : LeakCheck: detected memory leaks 
// 检测到内存泄漏
====== Direct leak of 100 byte(s) 
// 具体每次的内存泄漏信息
====== at 0x124080013000 on GM allocated in add_custom.cpp:14 (serialNo: 37)
====== Direct leak of 1000 byte(s)
====== at 0x124080014000 on GM allocated in add_custom.cpp:15 (serialNo: 55)
====== SUMMARY: 1100 byte(s) leaked in 2 allocation(s) 
// 全部内存泄漏的总结，包括发生泄漏的次数以及总共泄漏了多少字节等信息
```

以上示例中，第一个内存泄漏信息包含了地址空间、内存地址、内存长度以及代码定位信息，代码定位信息指向具体分配这块内存的调用所在的文件名和行号。
### 4.5 非法释放
非法释放是指对一个未分配的地址或者已释放的地址进行了释放操作，一般发生在 GM 上。

示例报告：
```text
====== ERROR: illegal free()     
// 异常的基本信息，表明发生了非法释放异常
====== at 0x124080013000 on GM      
// 异常发生的内存位置信息，包含发生的地址空间与内存地址
====== code in add_custom.cpp:84 (serialNo:63)    
// 异常发生的代码定位信息，包含文件名、行号和调用 api 行为的序列号
```
以上示例中，对 GM 上的“0x124080013000”地址进行了非法释放，且导致异常发生的指令对应于算子实现文件 add_custom.cpp 的第 84 行。

### 4.6 分配内存未使用
分配内存未使用是指算子运行时申请了内存，但直到算子运行完成，都没有使用该内存。该异常场景一般是算子使用了错误的内存或算子逻辑存在问题，一般发生在 GM 上。

示例报告：
```text
====== WARNING : Unused memory of 1000 byte(s) 
// 异常的基本信息，表明检测到内存分配未使用异常
====== at 1240c0016000 on GM 
// 异常发生的内存位置信息，包含发生的地址空间与内存地址
====== code in add_custom.cpp:2 (serialNo: 69) 
// 异常发生的代码定位信息，包含文件名、行号和调用 api 行为的序列号
====== SUMMARY: 1100 byte(s) unused memory in 2 allocation(s) 
// 内存分配未使用的总结信息，包括未使用内存块的个数及字节等信息
```