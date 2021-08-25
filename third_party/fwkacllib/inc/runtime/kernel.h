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

#ifndef __CCE_RUNTIME_KERNEL_H__
#define __CCE_RUNTIME_KERNEL_H__

#include "base.h"
#include "stream.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @ingroup rt_kernel
 * @brief shared memory data control
 */
typedef struct tagRtSmData {
    uint64_t L2_mirror_addr;          // preload or swap source address
    uint32_t L2_data_section_size;    // every data size
    uint8_t L2_preload;               // 1 - preload from mirrorAddr, 0 - no preload
    uint8_t modified;                 // 1 - data will be modified by kernel, 0 - no modified
    uint8_t priority;                 // data priority
    int8_t prev_L2_page_offset_base;  // remap source section offset
    uint8_t L2_page_offset_base;      // remap destination section offset
    uint8_t L2_load_to_ddr;           // 1 - need load out, 0 - no need
    uint8_t reserved[2];              // reserved
} rtSmData_t;

/**
 * @ingroup rt_kernel
 * @brief shared memory description
 */
typedef struct tagRtSmCtrl {
    rtSmData_t data[8];  // data description
    uint64_t size;       // max page Num
    uint8_t remap[64];   /* just using for static remap mode, default:0xFF
                          array index: virtual l2 page id, array value: physic l2 page id */
    uint8_t l2_in_main;  // 0-DDR, 1-L2, default:0xFF
    uint8_t reserved[3];
} rtSmDesc_t;

typedef rtSmDesc_t rtL2Ctrl_t;

/**
 * @ingroup rt_kernel
 * @brief device binary type
 */
typedef struct tagRtDevBinary {
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    const void *data;  // binary data
    uint64_t length;   // binary length
} rtDevBinary_t;

/**
  * @ingroup rt_kernel
  * @brief function mode type
  */
#define ONLINE_PROF_MAX_PMU_NUM (8)

typedef struct ProfilefDataInfo {
    const void *stubFunc;
    uint32_t blockDim;
    const void *args;
    uint32_t argsSize;
    rtSmDesc_t *smDesc;
    rtStream_t stream;
    uint64_t totalcycle;
    uint64_t ovcycle;
    uint64_t pmu_cnt[ONLINE_PROF_MAX_PMU_NUM];
} rtProfDataInfo_t;

/**
 * @ingroup rt_kernel
 * @brief function mode type
 */
typedef enum {
    FUNC_MODE_NORMAL = 0,
    FUNC_MODE_PCTRACE_USERPROFILE_RECORDLOOP,
    FUNC_MODE_PCTRACE_USERPROFILE_SKIPLOOP,
    FUNC_MODE_PCTRACE_CYCLECNT_RECORDLOOP,
    FUNC_MODE_PCTRACE_CYCLECNT_SKIPLOOP,
    FUNC_MODE_BUTT
} rtFuncModeType_t;

/**
 * @ingroup rt_kernel
 * @brief kernel info
 */
typedef struct rtKernelInfo {
    uint64_t task_offset;  // kernel offset in module
    /* flowtable */
    void *arg;  // launch kernel arg
    uint32_t arg_size;
    /* module */
    void *module_addr;  // module::baseaddr_
    uint32_t module_size;
} *rtKernelInfo_t;

/**
 * @ingroup rt_kernel
 * @brief op name
 */
typedef struct rtKernelLaunchNames {
    const char *soName;      // defined for so name
    const char *kernelName;  // defined for kernel type name
    const char *opName;      // defined for operator name
} rtKernelLaunchNames_t;

/**
 * @ingroup rt_kernel
 * @brief args struct
 */
typedef struct tagRtArgsWithTiling {
    void *args;                     // args host mem addr
    uint32_t argsSize;              // input + output + tiling addr size + tiling data size
    uint32_t argsSizeWithoutTiling; // input + output + tiling addr size
    uint16_t tilingAddrOffset;      // tiling addr offset
    uint16_t tilingDataOffset;      // tiling data offset
    uint16_t reserved[2];
} rtArgsWithTiling_t;

/**
 * @ingroup rt_KernelConfigDump
 * @brief device dump type
 */
typedef enum tagRtDumpKind {
    RT_DATA_DUMP_KIND_INVALID = -1,
    RT_DATA_DUMP_KIND_DUMP = 0,
    RT_DATA_DUMP_KIND_RESERVED
} rtDumpKind_t;

/**
 * @ingroup rt_kernel
 * @brief report callback
 */
typedef rtError_t (*rtKernelReportCallback)(rtStream_t stream, rtKernelInfo_t kernelInfo);

/**
 * @ingroup rt_kernel
 * @brief stream report callback
 */
typedef void (*rtCallback_t)(void *fnData);

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aicore
 */
#define RT_DEV_BINARY_MAGIC_PLAIN 0xabceed50

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aicpu
 */
#define RT_DEV_BINARY_MAGIC_PLAIN_AICPU 0xabceed51

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aivector
 */
#define RT_DEV_BINARY_MAGIC_PLAIN_AIVEC 0xabceed52

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicore
 */
#define RT_DEV_BINARY_MAGIC_ELF 0x43554245

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicpu
 */
#define RT_DEV_BINARY_MAGIC_ELF_AICPU 0x41415243

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aivector
 */
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC 0x41415246

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicube
 */
#define RT_DEV_BINARY_MAGIC_ELF_AICUBE 0x41494343

/**
 * @ingroup rt_kernel_flags
 * @brief kernel op bit flags
 */
#define RT_KERNEL_DEFAULT (0x00)
#define RT_KERNEL_CONVERT (0x01)
#define RT_KERNEL_DUMPFLAG (0x02)
#define RT_FUSION_KERNEL_DUMPFLAG (0x04)
#define RT_KERNEL_CUSTOM_AICPU (0x08)

// STARS topic scheduler sqe : topic_type
#define RT_KERNEL_DEVICE_FIRST (0x10)
#define RT_KERNEL_HOST_ONLY (0x20)
#define RT_KERNEL_HOST_FIRST (0x40)

/**
 * @ingroup rt_kernel
 * @brief kernel mode
**/
#define RT_DEFAULT_KERNEL_MODE (0x00)
#define RT_NORMAL_KERNEL_MODE (0x01)
#define RT_ALL_KERNEL_MODE (0x02)

/**
 * @ingroup rt_kernel
 * @brief kernel L1 Fusion Dump bit flags
 */
#define RT_DDR_ADDR (0x0)

/**
 * @ingroup rt_kernel
 * @brief register device binary
 * @param [in] bin   device binary description
 * @param [out] handle   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle);

/**
 * @ingroup rt_kernel
 * @brief register device binary with all kernel
 * @param [in] bin   device binary description
 * @param [out] handle   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle);

/**
 * @ingroup rt_kernel
 * @brief register fast memeory device binary
 * @param [in] handle   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtBinaryRegisterToFastMemory(void *handle);

/**
 * @ingroup rt_kernel
 * @brief unregister device binary
 * @param [in] handle   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDevBinaryUnRegister(void *handle);

/**
 * @ingroup rt_kernel
 * @brief register device binary metadata
 * @param [in] handle    device binary description
 * @param [in] metadata  device binary metadata
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtMetadataRegister(void *handle, const char *metadata);

/**
 * @ingroup rt_kernel
 * @brief register device binary dependency
 * @param [in] mHandle   master device binary description
 * @param [in] sHandle   slave device binary description
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDependencyRegister(void *mHandle, void *sHandle);

/**
 * @ingroup rt_kernel
 * @brief register device function
 * @param [in] binHandle   device binary handle
 * @param [in] stubFunc   stub function
 * @param [in] stubName   stub function name
 * @param [in] devFunc   device function description. symbol name or address
 *                       offset, depending binary type.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
                                     uint32_t funcMode);

/**
 * @ingroup rt_kernel
 * @brief find stub function by name
 * @param [in] stubName   stub function name
 * @param [out] stubFunc   stub function
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetFunctionByName(const char *stubName, void **stubFunc);

/**
 * @ingroup rt_kernel
 * @brief find addr by stub func
 * @param [in] stubFunc   stub function
 * @param [out] addr
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetAddrByFun(const void *stubFunc, void **addr);
/**
 * @ingroup rt_kernel
 * @brief query registered or not by stubName
 * @param [in] stubName   stub function name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtQueryFunctionRegistered(const char *stubName);

/**
 * @ingroup rt_kernel
 * @brief config data dump
 * @param [in] dumpSizePerBlock  dump size
 * @param [in] blockDim   block dimentions
 * @param [in] dumpBaseAddr   dump base address
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelConfigDump(uint32_t kind, uint32_t dumpSizePerBlock, uint32_t blockDim, void **dumpBaseAddr,
                                     rtStream_t stream);

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] args   argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] smDesc   shared memory description
 * @param [in] stream   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                 rtSmDesc_t *smDesc, rtStream_t stream);

/**
 * @ingroup rt_kernel
 * @brief launch kernel with handle to device
 * @param [in] handle   program
 * @param [in] devFunc   device function description.
 * @param [in] blockDim   block dimentions
 * @param [in] args   argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] smDesc   shared memory description
 * @param [in] stream   associated stream
 * @param [in] kernelInfo   kernel info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithHandle(void *handle, const void *devFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                            rtSmDesc_t *smDesc, rtStream_t stream_, const void *kernelInfo);

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] args   argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] smDesc   shared memory description
 * @param [in] stream   associated stream
 * @param [in] flag   dump flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                         rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags);

/**
 * @ingroup rt_kernel(abandoned)
 * @brief launch kernel to device
 * @param [in] args       argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] flags      launch flags
 * @param [in] stream     associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchEx(void *args, uint32_t argsSize, uint32_t flags, rtStream_t stream);

/**
 * @ingroup rt_kernel(in use)
 * @brief launch kernel to device
 * @param [in] opName     opkernel name
 * @param [in] args       argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] flags      launch flags
 * @param [in] stream     associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchFwk(const char *opName, void *args, uint32_t argsSize, uint32_t flags,
                                    rtStream_t rtStream);

/**
 * @ingroup rt_kernel(abandoned)
 * @brief launch cpu kernel to device
 * @param [in] soName        so name
 * @param [in] kernelName    kernel name
 * @param [in] blockDim      block dimentions
 * @param [in] args          argments address for kernel function
 * @param [in] argsSize      argments size
 * @param [in] smDesc        shared memory description
 * @param [in] stream        associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtCpuKernelLaunch(const void *soName, const void *kernelName, uint32_t blockDim, const void *args,
                                    uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stream);

/**
 * @ingroup rt_kernel(in use)
 * @brief launch cpu kernel to device
 * @param [in] launchNames   names for kernel launch
 * @param [in] blockDim      block dimentions
 * @param [in] args          argments address for kernel function
 * @param [in] argsSize      argments size
 * @param [in] smDesc        shared memory description
 * @param [in] stream        associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtAicpuKernelLaunch(const rtKernelLaunchNames_t *launchNames,
    uint32_t blockDim, const void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stream);

/**
 * @ingroup rt_kernel(abandoned)
 * @brief launch cpu kernel to device  with dump identifier
 * @param [in] soName        so name
 * @param [in] kernelName    kernel name
 * @param [in] blockDim      block dimentions
 * @param [in] args          argments address for kernel function
 * @param [in] argsSize      argments size
 * @param [in] smDesc        shared memory description
 * @param [in] stream        associated stream
 * @param [in] flag          dump flag or others function flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtCpuKernelLaunchWithFlag(const void *soName, const void *kernelName, uint32_t blockDim,
                                            const void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stream,
                                            uint32_t flags);

/**
 * @ingroup rt_kernel(in use)
 * @brief launch cpu kernel to device  with dump identifier
 * @param [in] launchNames   names for kernel launch
 * @param [in] blockDim      block dimentions
 * @param [in] args          argments address for kernel function
 * @param [in] argsSize      argments size
 * @param [in] smDesc        shared memory description
 * @param [in] stream        associated stream
 * @param [in] flag          dump flag or others function flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtAicpuKernelLaunchWithFlag(const rtKernelLaunchNames_t *launchNames, uint32_t blockDim,
    const void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags);

/**
 * @ingroup rt_kernel
 * @brief L1 fusion dump addr transfered to device
 * @param [in] model    handle info
 * @param [in] addr     ddr address of L1 Fusion Dump
 * @param [in] dumpSize memory size
 * @param [in] flag     memory flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDumpAddrSet(rtModel_t model, void *addr, uint32_t dumpSize, uint32_t flag);

/**
 * @ingroup rt_kernel
 * @brief load dump info to aicpu
 * @param [in] dumpInfo   dump info
 * @param [in] length   length of  dump info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDatadumpInfoLoad(const void *dumpInfo, uint32_t length);

#ifndef __CLANG_CCE_RUNTIME_H__
#define __CLANG_CCE_RUNTIME_H__
/**
 * @ingroup rt_kernel
 * @brief configure call argment for next rtLaunch in current thread
 * @param [in] numBlocks   block dimentions
 * @param [in] smDesc   shared memory description
 * @param [in] stream   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
#ifdef __cplusplus
RTS_API rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc = nullptr, rtStream_t stream = nullptr);
#else
RTS_API rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc, rtStream_t stream);

#endif
#endif  // __CLANG_CCE_RUNTIME_H__

/**
 * @ingroup rt_kernel
 * @brief setup argment for next rtLaunch in current thread
 * @param [in] arg   argment address for kernel function
 * @param [in] size   argment size
 * @param [in] offset  argment table offset
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetupArgument(const void *arg, uint32_t size, uint32_t offset);

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device with previous setting kernel argment
 *        and call argment
 * @param [in] stubFunc   stub function
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLaunch(const void *stubFunc);

/**
 * @ingroup rt_kernel
 * @brief implicitly transfered data to device.
 *        lifecycle end after next kernel task finish
 * @param [in] ptr   host memory
 * @param [in] size   host memory size
 * @param [in] flag   reserved. set to 0
 * @param [out] arg   returned arg. used for next kernel's arg.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelConfigTransArg(const void *ptr, uint64_t size, uint32_t flag, void **arg);

/**
 * @ingroup rt_kernel
 * @brief start fusion kernels.
 * @param [in] stream   stream for fusion kernels
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelFusionStart(rtStream_t stream);

/**
 * @ingroup rt_kernel
 * @brief end fusion kernels.
 * @param [in] stream   stream for fusion kernels
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelFusionEnd(rtStream_t stream);

/**
 * @ingroup rt_kernel
 * @brief set kernelinfo callback
 * @param [in] callback
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetKernelReportCallback(rtKernelReportCallback callBack);

/**
 * @ingroup rt_kernel
 * @brief subscribe stream callback report.
 * @param [in] threadId   thread id for stream
 * @param [in] stream   stream for subscribe
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSubscribeReport(uint64_t threadId, rtStream_t stream);

/**
 * @ingroup rt_kernel
 * @brief add callback launch task in stream.
 * @param [in] callBackFunc   app callback function
 * @param [in] fnData   user data
 * @param [in] stream   subscribed stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtCallbackLaunch(rtCallback_t callBackFunc, void *fnData, rtStream_t stream, bool isBlock);

/**
 * @ingroup rt_kernel
 * @brief process callback report.
 * @param [in] timeout   if timeout=-1, while(1); else timeout
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtProcessReport(int32_t timeout);

/**
 * @ingroup rt_kernel
 * @brief unsubscribe callback report.
 * @param [in] threadId   thread id for stream
 * @param [in] stream   stream for subscribe
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtUnSubscribeReport(uint64_t threadId, rtStream_t stream);

/**
 * @ingroup profiling_base
 * @brief start online prof.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStartOnlineProf(rtStream_t stream, uint32_t sampleNum);

/**
 * @ingroup profiling_base
 * @brief stop online prof.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStopOnlineProf(rtStream_t stream);

/**
 * @ingroup profiling_base
 * @brief get online prof.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetOnlineProfData(rtStream_t stream, rtProfDataInfo_t *pProfData, uint32_t profDataNum);

/**
 * @ingroup profiling_base
 * @brief start mdc profiler.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStartMDCProfiler(void **addr, uint32_t length);

/**
 * @ingroup profiling_base
 * @brief stop mdc profiler.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStopMDCProfiler(void *addr);

/**
 * @ingroup rt_kernel
 * @brief launch kernel with tiling data to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] argsInfo   argments info address for kernel function
 * @param [in] smDesc   shared memory description
 * @param [in] stream   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithTiling(const void *stubFunc, uint32_t blockDim,
    rtArgsWithTiling_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stream_);

/**
 * @ingroup rt_kernel
 * @brief launch kernel with handle and tiling data to device
 * @param [in] handle   program
 * @param [in] devFunc   device function description.
 * @param [in] blockDim   block dimentions
 * @param [in] argsInfo   argments info address for kernel function
 * @param [in] smDesc   shared memory description
 * @param [in] stream   associated stream
 * @param [in] kernelInfo   kernel info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithHandleAndTiling(void *handle, const void *devFunc, uint32_t blockDim,
    rtArgsWithTiling_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stream_, const void* kernelInfo);

#if defined(__cplusplus)
}
#endif

#endif  // __CCE_RUNTIME_KERNEL_H__

