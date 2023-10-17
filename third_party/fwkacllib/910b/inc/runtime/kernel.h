/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: kernel.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_KERNEL_H
#define CCE_RUNTIME_KERNEL_H

#include "base.h"
#include "stream.h"
#include "rt_stars_define.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @ingroup rt_kernel
 * @brief shared memory data control
 */
typedef struct tagRtSmData {
    uint64_t L2_mirror_addr;          // preload or swap source addr
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
    const char_t *soName;      // defined for so name
    const char_t *kernelName;  // defined for kernel type name
    const char_t *opName;      // defined for operator name
} rtKernelLaunchNames_t;

typedef struct rtFunctionInfo {
    void *pcAddr;
    uint32_t prefetchCnt;
    uint8_t mixType;                  // 0:NO_MIX; 1:MIX_AIC; 2:MIX_AIV; 3:MIX_AIC_AIV
    uint8_t reserved[3];
} rtFunctionInfo_t;

typedef struct tagRtKernelInfo {
    uint8_t functionInfoNum;
    uint8_t reserved[3];
    rtFunctionInfo_t functionInfo[2];
} rtKernelDetailInfo_t;

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
    uint16_t hostInputAddrOffset;   // index of host_memory input in inputs_addrs list
    uint16_t hostInputDataOffset;   // host_mem input data offset
    uint8_t hasHostMemInput;        // has host_memory input data in args or not: 0 means no host_memory input data,
                                    // others means has host_memory input data.
    uint8_t isNoNeedH2DCopy;        // is no need host to device copy: 0 means need H2D copy,
                                    // others means doesn't need H2D copy.
    uint8_t reserved[6];
} rtArgsWithTiling_t;

/**
 * @ingroup rt_kernel
 * @brief host memory input struct
 */
typedef struct rtHostInputInfo {
    uint16_t addrOffset;
    uint16_t dataOffset;
} rtHostInputInfo_t;

/**
 * @ingroup rt_kernel
 * @brief args struct
 */
typedef struct tagRtArgsEx {
    void *args;                     // args host mem addr
    rtHostInputInfo_t *hostInputInfoPtr;     // nullptr means no host mem input
    uint32_t argsSize;              // input + output + tiling addr size + tiling data size + host mem
    uint16_t tilingAddrOffset;      // tiling addr offset
    uint16_t tilingDataOffset;      // tiling data offset
    uint16_t hostInputInfoNum;      // hostInputInfo num
    uint8_t hasTiling;              // if has tiling: 0 means no tiling
    uint8_t isNoNeedH2DCopy;        // is no need host to device copy: 0 means need H2D copy,
                                    // others means doesn't need H2D copy.
    uint8_t reserved[4];
} rtArgsEx_t;

typedef struct tagRtAicpuArgsEx {
    void *args; // args host mem addr
    rtHostInputInfo_t *hostInputInfoPtr; // nullptr means no host mem input
    rtHostInputInfo_t *kernelOffsetInfoPtr; // KernelOffsetInfo, it is different for CCE Kernel and fwk kernel
    uint32_t argsSize;
    uint16_t hostInputInfoNum; // hostInputInfo num
    uint16_t kernelOffsetInfoNum; // KernelOffsetInfo num
    uint16_t soNameAddrOffset; // just for CCE Kernel, default value is 0xffff for FWK kernel
    uint16_t kernelNameAddrOffset; // just for CCE Kernel, default value is 0xffff for FWK kernel
    bool isNoNeedH2DCopy; // is no need host to device copy: 0 means need H2D copy,
                               // other means doesn't need H2D copy.
    uint8_t reserved[3];
} rtAicpuArgsEx_t;

typedef struct tagRtDvppTaskDesc {
    rtStarsCommonSqe_t sqe;
    uint16_t aicpuTaskPos ; // rtsq max dep is 1024
    uint16_t reserved;
} rtDvppTaskDesc_t;

typedef struct tagRtAicpuTaskDesc {
    rtKernelLaunchNames_t kernelLaunchNames;
    uint16_t blockDim;
    uint16_t isUnderstudyOp : 1; // dvpp op exist, set 1; otherwise set 0
    uint16_t resverved : 15;
    rtArgsEx_t argsInfo;
} rtAicpuTaskDesc_t;

typedef enum tagRtMultipleTaskType {
    RT_MULTIPLE_TASK_TYPE_DVPP = 0,
    RT_MULTIPLE_TASK_TYPE_AICPU = 1,
    RT_MULTIPLE_TASK_TYPE_MAX
} rtMultipleTaskType_t;

typedef struct tagRtTaskDesc {
    rtMultipleTaskType_t type; // only support AICPU or DVPP, will be checked in runtime api_error.
    union {
        rtDvppTaskDesc_t dvppTaskDesc;
        rtAicpuTaskDesc_t aicpuTaskDesc;
    } u;
} rtTaskDesc_t;

typedef struct tagRtMultipleTaskInfo {
    uint32_t taskNum;
    rtTaskDesc_t *taskDesc; // must memset0 after new obj
} rtMultipleTaskInfo_t;

/**
 * @ingroup rt_KernelConfigDump
 * @brief device dump type
 */
typedef enum tagRtDumpKind {
    RT_DATA_DUMP_KIND_INVALID = -1,
    RT_DATA_DUMP_KIND_DUMP = 0,
    RT_DATA_DUMP_KIND_RESERVED = 1,
} rtDumpKind_t;

/**
 * @ingroup rt_kernel
 * @brief rt kernel type
 */
typedef enum rtKernelType {
    KERNEL_TYPE_CCE = 0,
    KERNEL_TYPE_FWK = 1,
    KERNEL_TYPE_AICPU = 2,
    KERNEL_TYPE_AICPU_CUSTOM = 4,
    KERNEL_TYPE_HWTS = 10,
    KERNEL_TYPE_RESERVED = 99,
} rtKernelType_t;

/**
 * @ingroup rt_kernel
 * @brief report callback
 */
typedef rtError_t (*rtKernelReportCallback)(rtStream_t stm, rtKernelInfo_t kernelInfo);

/**
 * @ingroup rt_kernel
 * @brief stream report callback
 */
typedef void (*rtCallback_t)(void *fnData);

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aicore
 */
#define RT_DEV_BINARY_MAGIC_PLAIN 0xabceed50U

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aicpu
 */
#define RT_DEV_BINARY_MAGIC_PLAIN_AICPU 0xabceed51U

/**
 * @ingroup rt_kernel
 * @brief magic number of plain binary for aivector
 */
#define RT_DEV_BINARY_MAGIC_PLAIN_AIVEC 0xabceed52U

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicore
 */
#define RT_DEV_BINARY_MAGIC_ELF 0x43554245U

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicpu
 */
#define RT_DEV_BINARY_MAGIC_ELF_AICPU 0x41415243U

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aivector
 */
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC 0x41415246U

/**
 * @ingroup rt_kernel
 * @brief magic number of elf binary for aicube
 */
#define RT_DEV_BINARY_MAGIC_ELF_AICUBE 0x41494343U

/**
 * @ingroup rt_kernel_flags
 * @brief kernel op bit flags
 */
#define RT_KERNEL_DEFAULT (0x00U)
#define RT_KERNEL_CONVERT (0x01U)
#define RT_KERNEL_DUMPFLAG (0x02U)
#define RT_FUSION_KERNEL_DUMPFLAG (0x04U)
#define RT_KERNEL_CUSTOM_AICPU (0x08U)
#define RT_KERNEL_FFTSPLUS_DYNAMIC_SHAPE_DUMPFLAG (0x10U)
#define RT_KERNEL_FFTSPLUS_STATIC_SHAPE_DUMPFLAG  (0x20U)

// STARS topic scheduler sqe : topic_type
#define RT_KERNEL_DEVICE_FIRST (0x10U)
#define RT_KERNEL_HOST_ONLY (0x20U)
#define RT_KERNEL_HOST_FIRST (0x40U)
#define RT_KERNEL_BIUPERF_FLAG (0x80U)

/**
 * @ingroup rt_kernel
 * @brief kernel mode
**/
#define RT_DEFAULT_KERNEL_MODE (0x00U)
#define RT_NORMAL_KERNEL_MODE (0x01U)
#define RT_ALL_KERNEL_MODE (0x02U)

/**
 * @ingroup rt_kernel
 * @brief SHAPE kernel type
**/
#define RT_STATIC_SHAPE_KERNEL (0x00U)
#define RT_DYNAMIC_SHAPE_KERNEL (0x01U)

/**
 * @ingroup rt_kernel
 * @brief kernel L1 Fusion Dump bit flags
 */
#define RT_DDR_ADDR (0x0U)

/**
 * @ingroup rt_kernel
 * @brief register device binary
 * @param [in] bin   device binary description
 * @param [out] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl);

RTS_API rtError_t  rtGetNotifyAddress(rtNotify_t notify, uint64_t * const notifyAddres);

/**
 * @ingroup rt_kernel
 * @brief register device binary with all kernel
 * @param [in] bin   device binary description
 * @param [out] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **hdl);

/**
 * @ingroup rt_kernel
 * @brief register fast memeory device binary
 * @param [in] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtBinaryRegisterToFastMemory(void *hdl);

/**
 * @ingroup rt_kernel
 * @brief unregister device binary
 * @param [in] hdl   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDevBinaryUnRegister(void *hdl);

/**
 * @ingroup rt_kernel
 * @brief register device binary metadata
 * @param [in] hdl    device binary description
 * @param [in] metadata  device binary metadata
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtMetadataRegister(void *hdl, const char_t *metadata);

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
 * @param [in] kernelInfoExt   kernel Info extension. device function description or tiling key,
 *                             depending static shape or dynmaic shape.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName,
                                     const void *kernelInfoExt, uint32_t funcMode);

/**
 * @ingroup rt_kernel
 * @brief find stub function by name
 * @param [in] stubName   stub function name
 * @param [out] stubFunc   stub function
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetFunctionByName(const char_t *stubName, void **stubFunc);

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
RTS_API rtError_t rtQueryFunctionRegistered(const char_t *stubName);

/**
 * @ingroup rt_kernel
 * @brief config data dump
 * @param [in] dumpSizePerBlock  dump size
 * @param [in] blockDim   block dimentions
 * @param [in] dumpBaseAddr   dump base addr
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelConfigDump(uint32_t kind, uint32_t dumpSizePerBlock, uint32_t blockDim, void **dumpBaseAddr,
                                     rtStream_t stm);

/**
* @ingroup rt_kernel
* @brief get kernel address and prefetchCnt
* @param [in] hdl           program for dynamic shape
* @param [in] tilingKey     tilingKey for dynamic shape
* @param [in] stubFunc      stubFunc for static shape
* @param [in] flag          flag for distinguishing between dynamic shape and static shape
* @param [out] addr         address of kernel function
* @param [out] prefetchCnt  prefetchCnt of kernel function
* @return RT_ERROR_NONE for ok
* @return RT_ERROR_INVALID_VALUE for error input
*/
RTS_API rtError_t rtKernelGetAddrAndPrefCnt(void *hdl, const uint64_t tilingKey, const void * const stubFunc,
                                            const uint32_t flag, void **addr, uint32_t *prefetchCnt);

/**
* @ingroup rt_kernel
* @brief get kernel address and prefetchCnt
* @param [in] hdl           program for dynamic shape
* @param [in] tilingKey     tilingKey for dynamic shape
* @param [in] stubFunc      stubFunc for static shape
* @param [in] flag          flag for distinguishing between dynamic shape and static shape
* @param [out] kernelInfo   address & prefetchCnt of kernel function
* @return RT_ERROR_NONE for ok
* @return RT_ERROR_INVALID_VALUE for error input
*/
RTS_API rtError_t rtKernelGetAddrAndPrefCntV2(void *hdl, const uint64_t tilingKey, const void * const stubFunc,
                                              const uint32_t flag, rtKernelDetailInfo_t *kernelInfo);

/**
 * @ingroup rt_kernel
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] args   argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] smDesc   shared memory description
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                 rtSmDesc_t *smDesc, rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief launch kernel with handle to device
 * @param [in] hdl             program
 * @param [in] tilingKey       tilingKey
 * @param [in] blockDim        block dimentions
 * @param [in] argsInfo        argments address for kernel function
 * @param [in] smDesc          shared memory description
 * @param [in] stm             associated stream
 * @param [in] kernelInfo      kernel info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithHandle(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
                                           rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                           const void *kernelInfo);

/**
 * @ingroup rt_kernel
 * @brief launch kernel with handle to device
 * @param [in] hdl             program
 * @param [in] tilingKey       tilingKey
 * @param [in] blockDim        block dimentions
 * @param [in] argsInfo        argments address for kernel function
 * @param [in] smDesc          shared memory description
 * @param [in] stm             associated stream
 * @param [in] cfgInfo      task config
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
    rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo);

/**
 * @ingroup rtKernelLaunchWithFlag
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] argsInfo   argments address for kernel function
 * @param [in] smDesc     shared memory description
 * @param [in] stm        associated stream
 * @param [in] flags      dump flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                         rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags);

/**
 * @ingroup rtKernelLaunchWithFlag
 * @brief launch kernel to device
 * @param [in] stubFunc   stub function
 * @param [in] blockDim   block dimentions
 * @param [in] argsInfo   argments address for kernel function
 * @param [in] smDesc     shared memory description
 * @param [in] stm        associated stream
 * @param [in] flags      dump flag
 * @param [in] cfgInfo      task config info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
    rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo);

/**
 * @ingroup rt_kernel(abandoned)
 * @brief launch kernel to device
 * @param [in] args       argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] flags      launch flags
 * @param [in] stm     associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchEx(void *args, uint32_t argsSize, uint32_t flags, rtStream_t stm);

/**
 * @ingroup rt_kernel(in use)
 * @brief launch kernel to device
 * @param [in] opName     opkernel name
 * @param [in] args       argments address for kernel function
 * @param [in] argsSize   argements size
 * @param [in] flags      launch flags
 * @param [in] stm     associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelLaunchFwk(const char_t *opName, void *args, uint32_t argsSize, uint32_t flags,
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
 * @param [in] stm        associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtCpuKernelLaunch(const void *soName, const void *kernelName, uint32_t blockDim, const void *args,
                                    uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm);

/**
 * @ingroup rt_kernel(in use)
 * @brief launch cpu kernel to device
 * @param [in] launchNames   names for kernel launch
 * @param [in] blockDim      block dimentions
 * @param [in] args          argments address for kernel function
 * @param [in] argsSize      argments size
 * @param [in] smDesc        shared memory description
 * @param [in] stm        associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtAicpuKernelLaunch(const rtKernelLaunchNames_t *launchNames,
    uint32_t blockDim, const void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm);

/**
 * @ingroup rtCpuKernelLaunchWithFlag(abandoned)
 * @brief launch cpu kernel to device  with dump identifier
 * @param [in] soName        so name
 * @param [in] kernelName    kernel name
 * @param [in] blockDim      block dimentions
 * @param [in] argsInfo      argments address for kernel function
 * @param [in] smDesc        shared memory description
 * @param [in] stm           associated stream
 * @param [in] flag          dump flag or others function flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtCpuKernelLaunchWithFlag(const void *soName, const void *kernelName, uint32_t blockDim,
                                            const rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                            uint32_t flags);

/**
 * @ingroup rtAicpuKernelLaunchWithFlag(in use)
 * @brief launch cpu kernel to device  with dump identifier
 * @param [in] launchNames   names for kernel launch
 * @param [in] blockDim      block dimentions
 * @param [in] args          argments address for kernel function
 * @param [in] smDesc        shared memory description
 * @param [in] stm           associated stream
 * @param [in] flags          dump flag or others function flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtAicpuKernelLaunchWithFlag(const rtKernelLaunchNames_t *launchNames, uint32_t blockDim,
                                              const rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                              uint32_t flags);

/**
 * @ingroup rtAicpuKernelLaunchEx
 * @brief launch cpu kernel to device  with dump identifier and kernelType
 * @param [in] kernelType    aicpu kernel type
 * @param [in] soName        address of op name
 * @param [in] blockDim      block dimentions
 * @param [in] argsInfo      argments address for kernel function
 * @param [in] smDesc        shared memory description
 * @param [in] stm           associated stream
 * @param [in] flags         dump flag or others function flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtAicpuKernelLaunchEx(uint32_t kernelType, const rtKernelLaunchNames_t *launchNames,
                                        uint32_t blockDim, const rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                        rtStream_t stm, uint32_t flags);
/**
 * @ingroup rtAicpuKernelLaunchExWithArgs
 * @brief launch cpu kernel to device  with dump identifier and kernelType
 * @param [in] kernelType    aicpu kernel type
 * @param [in] op Name        address of op name
 * @param [in] blockDim      block dimentions
 * @param [in] argsInfo      argments address for kernel function
 * @param [in] smDesc        shared memory description
 * @param [in] stm           associated stream
 * @param [in] flags         dump flag or others function flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t * const opName,
                                                const uint32_t blockDim, const rtAicpuArgsEx_t *argsInfo,
                                                rtSmDesc_t * const smDesc, const rtStream_t stm,
                                                const uint32_t flags);

/**
 * @ingroup rt_kernel
 * @brief L1 fusion dump addr transfered to device
 * @param [in] mdl    handle info
 * @param [in] addr     ddr address of L1 Fusion Dump
 * @param [in] dumpSize memory size
 * @param [in] flag     memory flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDumpAddrSet(rtModel_t mdl, void *addr, uint32_t dumpSize, uint32_t flag);

/**
 * @ingroup rt_kernel
 * @brief load dump info to aicpu
 * @param [in] dumpInfo   dump info
 * @param [in] length   length of  dump info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDatadumpInfoLoad(const void *dumpInfo, uint32_t length);

/**
 * @ingroup rt_kernel
 * @brief load dump info to aicpu
 * @param [in] dumpInfo   dump info
 * @param [in] length     length of  dump info
 * @param [in] flag       RT_KERNEL_DEFAULT or RT_KERNEL_CUSTOM_AICPU
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDatadumpInfoLoadWithFlag(const void *dumpInfo, const uint32_t length, const uint32_t flag);

/**
 * @ingroup rt_kernel
 * @brief launch npu get float status task
 * @param [in] outputAddrPtr  pointer to op output addr
 * @param [in] outputSize   op output size
 * @param [in] checkMode   check mode
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNpuGetFloatStatus(void *outputAddrPtr, uint64_t outputSize, uint32_t checkMode, rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief launch npu clear float status task
 * @param [in] checkMode   check mode
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNpuClearFloatStatus(uint32_t checkMode, rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief launch npu get float status task
 * @param [in] outputAddrPtr  pointer to op output addr
 * @param [in] outputSize   op output size
 * @param [in] checkMode   check mode
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNpuGetFloatDebugStatus(void *outputAddrPtr, uint64_t outputSize, uint32_t checkMode,
                                           rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief launch npu clear float status task
 * @param [in] checkMode   check mode
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNpuClearFloatDebugStatus(uint32_t checkMode, rtStream_t stm);

#ifndef __CLANG_CCE_RUNTIME_H__
#define __CLANG_CCE_RUNTIME_H__
/**
 * @ingroup rt_kernel
 * @brief configure call argment for next rtLaunch in current thread
 * @param [in] numBlocks   block dimentions
 * @param [in] smDesc   shared memory description
 * @param [in] stm   associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
#ifdef __cplusplus
RTS_API rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc = nullptr, rtStream_t stm = nullptr);
#else
RTS_API rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc, rtStream_t stm);

#endif
#endif  // __CLANG_CCE_RUNTIME_H__

/**
 * @ingroup rt_kernel
 * @brief setup argment for next rtLaunch in current thread
 * @param [in] args   argment address for kernel function
 * @param [in] size   argment size
 * @param [in] offset  argment table offset
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetupArgument(const void *args, uint32_t size, uint32_t offset);

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
 * @param [out] args   returned arg. used for next kernel's arg.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelConfigTransArg(const void *ptr, uint64_t size, uint32_t flag, void **args);

/**
 * @ingroup rt_kernel
 * @brief start fusion kernels.
 * @param [in] stm   stream for fusion kernels
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelFusionStart(rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief end fusion kernels.
 * @param [in] stm   stream for fusion kernels
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtKernelFusionEnd(rtStream_t stm);

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
 * @param [in] stm   stream for subscribe
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSubscribeReport(uint64_t threadId, rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief add callback launch task in stream.
 * @param [in] callBackFunc   app callback function
 * @param [in] fnData   user data
 * @param [in] stm   subscribed stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtCallbackLaunch(rtCallback_t callBackFunc, void *fnData, rtStream_t stm, bool isBlock);

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
 * @param [in] stm   stream for subscribe
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtUnSubscribeReport(uint64_t threadId, rtStream_t stm);

/**
 * @ingroup profiling_base
 * @brief start online prof.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStartOnlineProf(rtStream_t stm, uint32_t sampleNum);

/**
 * @ingroup profiling_base
 * @brief stop online prof.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStopOnlineProf(rtStream_t stm);

/**
 * @ingroup profiling_base
 * @brief get online prof.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetOnlineProfData(rtStream_t stm, rtProfDataInfo_t *pProfData, uint32_t profDataNum);

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
 * @brief Calculate ArgsSize.
 * @param [in] argsSize   args Size
 * @param [in] hostInfoTotalSize   hostInfoTotal Size
 * @param [in] hostInfoNum   hostInfo num
 * @param [out] launchArgsSize   launch Args Size
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtCalcLaunchArgsSize(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum,
                               size_t *launchArgsSize);

/**
 * @ingroup rt_kernel
 * @brief Create Args Handle.
 * @param [in] argsSize   args Size
 * @param [in] hostInfoTotalSize   hostInfoTotal Size
 * @param [in] hostInfoNum   hostInfo num
 * @param [in] argsData   args Data
 * @param [out] argsHandle   args Handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtCreateLaunchArgs(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum,
                             void* argsData, rtLaunchArgsHandle* argsHandle);

/**
 * @ingroup rt_kernel
 * @brief Destroy Args Handle.
 * @param [in] argsHandle   args Handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtDestroyLaunchArgs(rtLaunchArgsHandle argsHandle);

/**
 * @ingroup rt_kernel
 * @brief Reset Args Handle Info.
 * @param [in] argsHandle   args Handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtResetLaunchArgs(rtLaunchArgsHandle argsHandle);

/**
 * @ingroup rt_kernel
 * @brief Append address info to Args Handle.
 * @param [in] argsHandle   args Handle
 * @param [in] addrInfo   address info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtAppendLaunchAddrInfo(rtLaunchArgsHandle argsHandle, void *addrInfo);

/**
 * @ingroup rt_kernel
 * @brief Append Host info to args  Handle.
 * @param [in] argsHandle   args Handle
 * @param [in] hostInfoSize   host Info Size
 * @param [out] hostInfo host info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtAppendLaunchHostInfo(rtLaunchArgsHandle argsHandle, size_t hostInfoSize, void **hostInfo);

/**
 * @ingroup rt_kernel
 * @brief Registers and parses the bin file and loads it to the device.
 * @param [in] bin   device binary description
 * @param [out] binHandle   device binary handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtBinaryLoad(const rtDevBinary_t *bin, rtBinHandle *binHandle);

/**
 * @ingroup rt_kernel
 * @brief Find funcHandle based on binHandle and tilingKey.
 * @param [in] binHandle  funcHandle
  * @param [in] tilingKey   tilingKey
 * @param [out] funcHandle   funcHandle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtBinaryGetFunction(const rtBinHandle binHandle, const uint64_t tilingKey, rtFuncHandle *funcHandle);

/**
 * @ingroup rt_kernel
 * @brief UnLoad binary
 * @param [in] binHandle  Binary Handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtBinaryUnLoad(rtBinHandle binHandle);

/**
 * @ingroup rt_kernel
 * @brief Kernel Launch to device
 * @param [in] funcHandle  function Handle
 * @param [in] blockDim  block dimentions
 * @param [in] argsHandle  args Handle
 * @param [in] stm  associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtLaunchKernelByFuncHandle(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                     rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief Kernel Launch to device
 * @param [in] funcHandle  function Handle
 * @param [in] blockDim  block dimentions
 * @param [in] argsHandle  args Handle
 * @param [in] stm  associated stream
 * @param [in] cfgInfo task config info
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtLaunchKernelByFuncHandleV2(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                       rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo);

/**
 * @ingroup rt_kernel
 * @brief get Saturation Status task
 * @param [in] outputAddrPtr  pointer to op output addr
 * @param [in] outputSize   op output size
 * @param [in] stm  associated stream
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtGetDeviceSatStatus(void * const outputAddrPtr, const uint64_t outputSize, rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief clear Saturation Status task
 * @param [in] stm  associated stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtCleanDeviceSatStatus(rtStream_t stm);

/**
 * @ingroup rt_kernel
 * @brief Get ConditionKernel Bin
 * @param [in] binFileName  binFileName
 * @param [out] buffer   bin buffer
 * @param [out] length   buffer length
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetConditionKernelBin(const char_t * const binFileName, char_t **const buffer, uint32_t *length);

/**
 * @ingroup dvrt_mem
 * @brief HCCL copy ffts args
 * @param [in] stm task stream
 * @param [in] argsInfo args info
 * @param [out] devArgsAddr device mem addr for args
 * @param [out] argsHandle copy handler
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtGetDevArgsAddr(rtStream_t stm, rtArgsEx_t *argsInfo, void **devArgsAddr, void **argsHandle);

/**
 * @ingroup rt_kernel
 * @brief subscribe stream for hostFunc thread.
 * @param [in] threadId   thread id for stream
 * @param [in] stream     stream for subscribe
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSubscribeHostFunc(uint64_t threadId, rtStream_t stream);

/**
 * @ingroup rt_kernel
 * @brief process hostFunc callback report and hostFunc callback function.
 * @param [in] timeout           if timeout=-1, while(1); else timeout
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtProcessHostFunc(int32_t timeout);

/**
 * @ingroup rt_kernel
 * @brief unsubscribe hostFunc callback report.
 * @param [in] threadId   thread id for stream
 * @param [in] stream     stream for subscribe
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtUnSubscribeHostFunc(uint64_t threadId, rtStream_t stream);

#if defined(__cplusplus)
}
#endif

#endif  // CCE_RUNTIME_KERNEL_H

