/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __CCE_RUNTIME_MEM_H__
#define __CCE_RUNTIME_MEM_H__

#include <stddef.h>
#include "base.h"
#include "config.h"
#include "stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup dvrt_mem
 * @brief memory type
 */
#define RT_MEMORY_DEFAULT ((uint32_t)0x0)   // default memory on device
#define RT_MEMORY_HBM ((uint32_t)0x2)       // HBM memory on device
#define RT_MEMORY_DDR ((uint32_t)0x4)       // DDR memory on device
#define RT_MEMORY_SPM ((uint32_t)0x8)       // shared physical memory on device
#define RT_MEMORY_P2P_HBM ((uint32_t)0x10)  // HBM memory on other 4P device
#define RT_MEMORY_P2P_DDR ((uint32_t)0x11)  // DDR memory on other device
#define RT_MEMORY_DDR_NC ((uint32_t)0x20)   // DDR memory of non-cache
#define RT_MEMORY_TS_4G ((uint32_t)0x40)
#define RT_MEMORY_TS ((uint32_t)0x80)
#define RT_MEMORY_RESERVED ((uint32_t)0x100)

#define RT_MEMORY_L1 ((uint32_t)0x1<<16)
#define RT_MEMORY_L2 ((uint32_t)0x1<<17)

/**
 * @ingroup dvrt_mem
 * @brief memory Policy
 */
#define RT_MEMORY_POLICY_NONE ((uint32_t)0x0)                     // Malloc mem prior hage page, then default page
#define RT_MEMORY_POLICY_HUGE_PAGE_FIRST ((uint32_t)0x1 << 10)    // Malloc mem prior hage page, then default page
#define RT_MEMORY_POLICY_HUGE_PAGE_ONLY ((uint32_t)0x1 << 11)     // Malloc mem only use hage page
#define RT_MEMORY_POLICY_DEFAULT_PAGE_ONLY ((uint32_t)0x1 << 12)  // Malloc mem only use default page

#define MEM_ALLOC_TYPE_BIT ((uint32_t)0x3FF)  // mem type bit in <0, 9>

/**
 * @ingroup dvrt_mem
 * @brief memory type | memory Policy
 */
typedef uint32_t rtMemType_t;

/**
 * @ingroup dvrt_mem
 * @brief memory advise type
 */
#define RT_MEMORY_ADVISE_EXE (0x02)
#define RT_MEMORY_ADVISE_THP (0x04)
#define RT_MEMORY_ADVISE_PLE (0x08)
#define RT_MEMORY_ADVISE_PIN (0x16)

/**
 * @ingroup dvrt_mem
 * @brief memory copy type
 */
typedef enum tagRtMemcpyKind {
  RT_MEMCPY_HOST_TO_HOST = 0,  // host to host
  RT_MEMCPY_HOST_TO_DEVICE,    // host to device
  RT_MEMCPY_DEVICE_TO_HOST,    // device to host
  RT_MEMCPY_DEVICE_TO_DEVICE,  // device to device, 1P && P2P
  RT_MEMCPY_MANAGED,           // managed memory
  RT_MEMCPY_ADDR_DEVICE_TO_DEVICE,
  RT_MEMCPY_HOST_TO_DEVICE_EX, // host  to device ex (only used for 8 bytes)
  RT_MEMCPY_RESERVED,
} rtMemcpyKind_t;

typedef enum tagRtRecudeKind {
  RT_MEMCPY_SDMA_AUTOMATIC_ADD = 10,  // D2D, SDMA inline reduce, include 1P, and P2P
  RT_RECUDE_KIND_END
} rtRecudeKind_t;

typedef enum tagRtDataType {
  RT_DATA_TYPE_FP32 = 0,  // fp32
  RT_DATA_TYPE_FP16 = 1,  // fp16
  RT_DATA_TYPE_INT16 = 2, // int16
  RT_DATA_TYPE_END
} rtDataType_t;

/**
 * @ingroup dvrt_mem
 * @brief memory copy channel  type
 */
typedef enum tagRtMemcpyChannelType {
  RT_MEMCPY_CHANNEL_TYPE_INNER = 0,  // 1P
  RT_MEMCPY_CHANNEL_TYPE_PCIe,
  RT_MEMCPY_CHANNEL_TYPE_HCCs,  // not support now
  RT_MEMCPY_CHANNEL_TYPE_RESERVED,
} rtMemcpyChannelType_t;

/**
 * @ingroup rt_kernel
 * @brief ai core memory size
 */
typedef struct rtAiCoreMemorySize {
  uint32_t l0ASize;
  uint32_t l0BSize;
  uint32_t l0CSize;
  uint32_t l1Size;
  uint32_t ubSize;
  uint32_t l2Size;
  uint32_t l2PageNum;
  uint32_t blockSize;
  uint64_t bankSize;
  uint64_t bankNum;
  uint64_t burstInOneBlock;
  uint64_t bankGroupNum;
} rtAiCoreMemorySize_t;

/**
 * @ingroup dvrt_mem
 * @brief memory type
 */
typedef enum tagRtMemoryType { RT_MEMORY_TYPE_HOST = 1, RT_MEMORY_TYPE_DEVICE = 2 } rtMemoryType_t;

/**
 * @ingroup dvrt_mem
 * @brief memory attribute
 */
typedef struct tagRtPointerAttributes {
  rtMemoryType_t memoryType;  // host memory or device memory
  uint32_t deviceID;          // device ID
  uint32_t isManaged;
  uint32_t pageSize;
} rtPointerAttributes_t;

/**
 * @ingroup dvrt_mem
 * @brief alloc device memory
 * @param [in|out] devPtr   memory pointer
 * @param [in] size   memory size
 * @param [in] type   memory type
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_MEMORY_ALLOCATION for memory allocation failed
 */
RTS_API rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type);

/**
 * @ingroup dvrt_mem
 * @brief free device memory
 * @param [in|out] devPtr   memory pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error device memory pointer
 */
RTS_API rtError_t rtFree(void *devPtr);

/**
 * @ingroup dvrt_mem
 * @brief alloc device memory for dvpp
 * @param [in|out] devPtr   memory pointer
 * @param [in] size   memory size
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_MEMORY_ALLOCATION for memory allocation failed
 */
RTS_API rtError_t rtDvppMalloc(void **devPtr, uint64_t size);

/**
 * @ingroup dvrt_mem
 * @brief free device memory for dvpp
 * @param [in|out] devPtr   memory pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error device memory pointer
 */
RTS_API rtError_t rtDvppFree(void *devPtr);

/**
 * @ingroup dvrt_mem
 * @brief alloc host memory
 * @param [in|out] hostPtr   memory pointer
 * @param [in] size   memory size
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_MEMORY_ALLOCATION for memory allocation failed
 */
RTS_API rtError_t rtMallocHost(void **hostPtr, uint64_t size);

/**
 * @ingroup dvrt_mem
 * @brief free host memory
 * @param [in] hostPtr   memory pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error device memory pointer
 */
RTS_API rtError_t rtFreeHost(void *hostPtr);

/**
 * @ingroup dvrt_mem
 * @brief alloc managed memory
 * @param [in|out] ptr   memory pointer
 * @param [in] size   memory size
 * @param [in] flag   reserved, set to 0.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_MEMORY_ALLOCATION for memory allocation failed
 */
RTS_API rtError_t rtMemAllocManaged(void **ptr, uint64_t size, uint32_t flag);

/**
 * @ingroup dvrt_mem
 * @brief free managed memory
 * @param [in] ptr   memory pointer
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error device memory pointer
 */
RTS_API rtError_t rtMemFreeManaged(void *ptr);

/**
 * @ingroup dvrt_mem
 * @brief advise memory
 * @param [in] ptr    memory pointer
 * @param [in] size   memory size
 * @param [in] advise memory advise
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error device memory pointer
 */
RTS_API rtError_t rtMemAdvise(void *ptr, uint64_t size, uint32_t advise);

/**
 * @ingroup dvrt_mem
 * @brief flush device mempory
 * @param [in] base   virtal base address
 * @param [in] len    memory size
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtFlushCache(uint64_t base, uint32_t len);

/**
 * @ingroup dvrt_mem
 * @brief invalid device mempory
 * @param [in] base   virtal base address
 * @param [in] len    memory size
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtInvalidCache(uint64_t base, uint32_t len);

/**
 * @ingroup dvrt_mem
 * @brief synchronized memcpy
 * @param [in] dst   destination address pointer
 * @param [in] Max length of destination address memory
 * @param [in] src   source address pointer
 * @param [in] count   the number of byte to copy
 * @param [in] kind   memcpy type
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of count
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error input memory pointer of dst,src
 * @return RT_ERROR_INVALID_MEMCPY_DIRECTION for error copy direction of kind
 */
RTS_API rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind);

/**
 * @ingroup dvrt_mem
 * @brief asynchronized memcpy
 * @param [in] dst   destination address pointer
 * @param [in] Max length of destination address memory
 * @param [in] src   source address pointer
 * @param [in] count   the number of byte to copy
 * @param [in] kind   memcpy type
 * @param [in] stream   asynchronized task stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of count,stream
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error input memory pointer of dst,src
 * @return RT_ERROR_INVALID_MEMCPY_DIRECTION for error copy direction of kind
 */
RTS_API rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                                rtStream_t stream);

/**
 * @ingroup dvrt_mem
 * @brief asynchronized reduce memcpy
 * @param [in] dst   destination address pointer
 * @param [in] Max length of destination address memory
 * @param [in] src   source address pointer
 * @param [in] count   the number of byte to copy
 * @param [in] kind   memcpy type
 * @param [in] type   data type
 * @param [in] stream   asynchronized task stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of count,stream
 * @return RT_ERROR_INVALID_DEVICE_POINTER for error input memory pointer of dst,src
 * @return RT_ERROR_INVALID_MEMCPY_DIRECTION for error copy direction of kind
 */
RTS_API rtError_t rtReduceAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtRecudeKind_t kind,
                                rtDataType_t type, rtStream_t stream);

/**
 * @ingroup dvrt_mem
 * @brief query memory size
 * @param [in] aiCoreMemorySize
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t *aiCoreMemorySize);

/**
 * @ingroup dvrt_mem
 * @brief set memory size, Setting before model reasoning, Bright screen to prevent model can not be fully
       integrated network due to memory limitations.Requirement come from JiaMinHu.Only use for Tiny.
 * @param [in] aiCoreMemorySize
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtSetAiCoreMemorySizes(rtAiCoreMemorySize_t *aiCoreMemorySize);

/**
 * @ingroup dvrt_mem
 * @brief set memory with uint32_t value
 * @param [in] devPtr
 * @param [in] Max length of destination address memory
 * @param [in] value
 * @param [in] count byte num
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtMemset(void *devPtr, uint64_t destMax, uint32_t value, uint64_t count);

/**
 * @ingroup dvrt_mem
 * @brief set memory with uint32_t value async
 * @param [in] devPtr
 * @param [in] Max length of destination address memory
 * @param [in] value
 * @param [in] count byte num
 * @param [in] stream
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtMemsetAsync(void *ptr, uint64_t destMax, uint32_t value, uint64_t count, rtStream_t stream);

/**
 * @ingroup dvrt_mem
 * @brief get current device memory total and free
 * @param [out] free
 * @param [out] total
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtMemGetInfo(size_t *free, size_t *total);

/**
 * @ingroup dvrt_mem
 * @brief set memory with uint32_t value
 * @param [in] devPtr
 * @param [in] len
 * @param [in] device
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtMemPrefetchToDevice(void *devPtr, uint64_t len, int32_t device);

/**
 * @ingroup dvrt_mem
 * @brief get memory attribute:Host or Device
 * @param [in] ptr
 * @param [out] attributes
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtPointerGetAttributes(rtPointerAttributes_t *attributes, const void *ptr);

/**
 * @ingroup dvrt_mem
 * @brief make memory shared interprocess and assigned a name
 * @param [in] ptr    device memory address pointer
 * @param [in] name   identification name
 * @param [in] byteCount   identification byteCount
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of ptr, name, byteCount
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtIpcSetMemoryName(const void *ptr, uint64_t byteCount, char *name, uint32_t len);

/**
 * @ingroup dvrt_mem
 * @brief destroy a interprocess shared memory
 * @param [in] name   identification name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of name
 * @return RT_ERROR_DRV_ERR for driver error
 */
rtError_t rtIpcDestroyMemoryName(const char *name);

/**
 * @ingroup dvrt_mem
 * @brief open a interprocess shared memory
 * @param [in|out] ptr    device memory address pointer
 * @param [in] name   identification name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of ptr, name
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtIpcOpenMemory(void **ptr, const char *name);

/**
 * @ingroup dvrt_mem
 * @brief close a interprocess shared memory
 * @param [in] ptr    device memory address pointer
 * @param [in] name   identification name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of ptr, name
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtIpcCloseMemory(const void *ptr);

/**
 * @ingroup dvrt_mem
 * @brief HCCL Async memory cpy
 * @param [in] index sq index
 * @param [in] wqe_index moudle index
 * @param [in] stream asynchronized task stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of ptr, name
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtRDMASend(uint32_t index, uint32_t wqe_index, rtStream_t stream);

/**
 * @ingroup dvrt_mem
 * @brief Set the memory readCount value
 * @param [in] devPtr memory pointer
 * @param [in] size  memory size
 * @param [in] readCount  readCount value
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for invalid resource handle
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtMemSetRC(const void *devPtr, uint64_t size, uint32_t readCount);

/**
 * @ingroup dvrt_mem
 * @brief Ipc set mem pid
 * @param [in] name name to be queried
 * @param [in] pid  process id
 * @param [in] num  length of pid[]
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for invalid resource handle
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtSetIpcMemPid(const char *name, int32_t pid[], int num);

/**
 * @ingroup dvrt_mem
 * @brief HCCL Async memory cpy
 * @param [in] dbindex single device 0
 * @param [in] dbinfo doorbell info
 * @param [in] stream asynchronized task stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of ptr, name
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtRDMADBSend(uint32_t dbIndex, uint64_t dbInfo, rtStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // __CCE_RUNTIME_MEM_H__
