/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: config.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_CONFIG_H
#define CCE_RUNTIME_CONFIG_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define PLAT_COMBINE(arch, chip, ver) (((arch) << 16U) | ((chip) << 8U) | (ver))
#define PLAT_GET_ARCH(type)           (((type) >> 16U) & 0xffffU)
#define PLAT_GET_CHIP(type)           (((type) >> 8U) & 0xffU)
#define PLAT_GET_VER(type)            ((type) & 0xffU)

typedef enum tagRtArchType {
    ARCH_BEGIN = 0,
    ARCH_V100 = ARCH_BEGIN,
    ARCH_V200 = 1,
    ARCH_V300 = 2,
    ARCH_END = 3,
} rtArchType_t;

typedef enum tagRtChipType {
    CHIP_BEGIN = 0,
    CHIP_MINI = CHIP_BEGIN,
    CHIP_CLOUD = 1,
    CHIP_MDC = 2,
    CHIP_LHISI = 3,
    CHIP_DC = 4,
    CHIP_CLOUD_V2 = 5,
    CHIP_NO_DEVICE = 6,
    CHIP_MINI_V3 = 7,
    CHIP_5612 = 8, /* 1911T */
    CHIP_END = 9,
} rtChipType_t;

typedef enum tagRtAicpuScheType {
    SCHEDULE_SOFTWARE = 0, /* Software Schedule */
    SCHEDULE_SOFTWARE_OPT,
    SCHEDULE_HARDWARE, /* HWTS Schedule */
} rtAicpuScheType;

typedef enum tagRtDeviceCapabilityType {
    RT_SCHEDULE_SOFTWARE = 0, // Software Schedule
    RT_SCHEDULE_SOFTWARE_OPT,
    RT_SCHEDULE_HARDWARE, // HWTS Schedule
    RT_AICPU_BLOCKING_OP_NOT_SUPPORT,
    RT_AICPU_BLOCKING_OP_SUPPORT, // 1910/1980/1951 ts support AICPU blocking operation
    RT_MODE_NO_FFTS, // no ffts
    RT_MODE_FFTS, // 1981 get ffts work mode, ffts
    RT_MODE_FFTS_PLUS, // 1981 get ffts work mode, ffts plus
} rtDeviceCapabilityType;

typedef enum tagRtVersion {
    VER_BEGIN = 0,
    VER_NA = VER_BEGIN,
    VER_ES = 1,
    VER_CS = 2,
    VER_SD3403 = 3,
    VER_END = 4,
} rtVersion_t;

/* match rtChipType_t */
typedef enum tagRtPlatformType {
    PLATFORM_BEGIN = 0,
    PLATFORM_MINI_V1 = PLATFORM_BEGIN,
    PLATFORM_CLOUD_V1 = 1,
    PLATFORM_MINI_V2 = 2,
    PLATFORM_LHISI_ES = 3,
    PLATFORM_LHISI_CS = 4,
    PLATFORM_DC = 5,
    PLATFORM_CLOUD_V2 = 6,
    PLATFORM_LHISI_SD3403 = 7,
    PLATFORM_MINI_V3 = 8,
    PLATFORM_MINI_5612 = 9,
    PLATFORM_CLOUD_V2_910B1 = 10,
    PLATFORM_CLOUD_V2_910B2 = 11,
    PLATFORM_CLOUD_V2_910B3 = 12,
    PLATFORM_CLOUD_V2_910B4 = 13,
    PLATFORM_MDC_PG2 = 14,
    PLATFORM_END = 15,
} rtPlatformType_t;

typedef enum tagRtCubeFracMKNFp16 {
    RT_CUBE_MKN_FP16_2_16_16 = 0,
    RT_CUBE_MKN_FP16_4_16_16,
    RT_CUBE_MKN_FP16_16_16_16,
    RT_CUBE_MKN_FP16_Default,
} rtCubeFracMKNFp16_t;

typedef enum tagRtCubeFracMKNInt8 {
    RT_CUBE_MKN_INT8_2_32_16 = 0,
    RT_CUBE_MKN_INT8_4_32_4,
    RT_CUBE_MKN_INT8_4_32_16,
    RT_CUBE_MKN_INT8_16_32_16,
    RT_CUBE_MKN_INT8_Default,
} rtCubeFracMKNInt8_t;

typedef enum tagRtVecFracVmulMKNFp16 {
    RT_VEC_VMUL_MKN_FP16_1_16_16 = 0,
    RT_VEC_VMUL_MKN_FP16_Default,
} rtVecFracVmulMKNFp16_t;

typedef enum tagRtVecFracVmulMKNInt8 {
    RT_VEC_VMUL_MKN_INT8_1_32_16 = 0,
    RT_VEC_VMUL_MKN_INT8_Default,
} rtVecFracVmulMKNInt8_t;

typedef struct tagRtAiCoreSpec {
    uint32_t cubeFreq;
    uint32_t cubeMSize;
    uint32_t cubeKSize;
    uint32_t cubeNSize;
    rtCubeFracMKNFp16_t cubeFracMKNFp16;
    rtCubeFracMKNInt8_t cubeFracMKNInt8;
    rtVecFracVmulMKNFp16_t vecFracVmulMKNFp16;
    rtVecFracVmulMKNInt8_t vecFracVmulMKNInt8;
} rtAiCoreSpec_t;

typedef struct tagRtAiCoreRatesPara {
    uint32_t ddrRate;
    uint32_t l2Rate;
    uint32_t l2ReadRate;
    uint32_t l2WriteRate;
    uint32_t l1ToL0ARate;
    uint32_t l1ToL0BRate;
    uint32_t l0CToUBRate;
    uint32_t ubToL2;
    uint32_t ubToDDR;
    uint32_t ubToL1;
} rtAiCoreMemoryRates_t;

typedef struct tagRtMemoryConfig {
    uint32_t flowtableSize;
    uint32_t compilerSize;
} rtMemoryConfig_t;

typedef struct tagRtPlatformConfig {
    uint32_t platformConfig;
} rtPlatformConfig_t;

typedef enum tagRTTaskTimeoutType {
    RT_TIMEOUT_TYPE_OP_WAIT = 0,
    RT_TIMEOUT_TYPE_OP_EXECUTE,
} rtTaskTimeoutType_t;

/**
 * @ingroup
 * @brief get AI core count
 * @param [in] aiCoreCnt
 * @return aiCoreCnt
 */
RTS_API rtError_t rtGetAiCoreCount(uint32_t *aiCoreCnt);

/**
 * @ingroup
 * @brief get AI cpu count
 * @param [in] aiCpuCnt
 * @return aiCpuCnt
 */
RTS_API rtError_t rtGetAiCpuCount(uint32_t *aiCpuCnt);

/**
 * @ingroup
 * @brief get AI core frequency
 * @param [in] aiCoreSpec
 * @return aiCoreSpec
 */
RTS_API rtError_t rtGetAiCoreSpec(rtAiCoreSpec_t *aiCoreSpec);

/**
 * @ingroup
 * @brief AI get core band Info
 * @param [in] aiCoreMemoryRates
 * @return aiCoreMemoryRates
 */
RTS_API rtError_t rtGetAiCoreMemoryRates(rtAiCoreMemoryRates_t *aiCoreMemoryRates);

/**
 * @ingroup
 * @brief AI get core buffer Info,FlowTable Size,Compiler Size
 * @param [in] memoryConfig
 * @return memoryConfig
 */
RTS_API rtError_t rtGetMemoryConfig(rtMemoryConfig_t *memoryConfig);

/**
 * @ingroup
 * @brief get float overflow mode
 * @param [out] floatOverflowMode
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetFloatOverflowMode(rtFloatOverflowMode_t * const floatOverflowMode);

/**
 * @ingroup
 * @brief get l2 buffer Info,virtual baseaddr,Size
 * @param [in] stm
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtMemGetL2Info(rtStream_t stm, void **ptr, uint32_t *size);

/**
 * @ingroup
 * @brief get runtime version. The version is returned as (1000 major + 10 minor). For example, RUNTIME 9.2 would be
 *        represented by 9020.
 * @param [out] runtimeVersion
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetRuntimeVersion(uint32_t *runtimeVersion);


/**
 * @ingroup
 * @brief get device feature ability by device id, such as task schedule ability.
 * @param [in] deviceId
 * @param [in] moduleType
 * @param [in] featureType
 * @param [out] val
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetDeviceCapability(int32_t deviceId, int32_t moduleType, int32_t featureType, int32_t *val);

/**
 * @ingroup
 * @brief set event wait task timeout time.
 * @param [in] timeout
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetOpWaitTimeOut(uint32_t timeout);

/**
 * @ingroup
 * @brief set op execute task timeout time.
 * @param [in] timeout
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetOpExecuteTimeOut(uint32_t timeout);

/**
 * @ingroup
 * @brief get is Heterogenous.
 * @param [out] heterogenous=1 Heterogenous Mode: read isHeterogenous=1 in ini file.
 * @param [out] heterogenous=0 NOT Heterogenous Mode:
 *      1:not found ini file, 2:error when reading ini, 3:Heterogenous value is not 1
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtGetIsHeterogenous(int32_t *heterogenous);

#if defined(__cplusplus)
}
#endif

#endif // CCE_RUNTIME_CONFIG_H
