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

#ifndef __CCE_RUNTIME_CONFIG_H__
#define __CCE_RUNTIME_CONFIG_H__

#include "base.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#define PLAT_COMBINE(arch, chip, ver) ((arch << 16) | (chip << 8) | (ver))
#define PLAT_GET_ARCH(type) ((type >> 16) & 0xffff)
#define PLAT_GET_CHIP(type) ((type >> 8) & 0xff)
#define PLAT_GET_VER(type) (type & 0xff)

typedef enum tagRtArchType {
  ARCH_BEGIN = 0,
  ARCH_V100 = ARCH_BEGIN,
  ARCH_V200,
  ARCH_END,
} rtArchType_t;

typedef enum tagRtChipType {
  CHIP_BEGIN = 0,
  CHIP_MINI = CHIP_BEGIN,
  CHIP_CLOUD,
  CHIP_MDC,
  CHIP_LHISI,
  CHIP_DC,
  CHIP_END,
} rtChipType_t;

typedef enum tagRtVersion {
  VER_BEGIN = 0,
  VER_NA = VER_BEGIN,
  VER_ES,
  VER_CS,
  VER_END,
} rtVersion_t;

/* match rtChipType_t */
typedef enum tagRtPlatformType {
  PLATFORM_BEGIN = 0,
  PLATFORM_MINI_V1 = PLATFORM_BEGIN,
  PLATFORM_CLOUD_V1,
  PLATFORM_MINI_V2,
  PLATFORM_LHISI_ES,
  PLATFORM_LHISI_CS,
  PLATFORM_DC,
  PLATFORM_END,
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

typedef struct tagRtPlatformConfig { uint32_t platformConfig; } rtPlatformConfig_t;

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
 * @brief get l2 buffer Info,virtual baseaddr,Size
 * @param [in] stream
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtMemGetL2Info(rtStream_t stream, void **ptr, uint32_t *size);

/**
 * @ingroup
 * @brief get runtime version. The version is returned as (1000 major + 10 minor). For example, RUNTIME 9.2 would be represented by 9020.
 * @param [out] runtimeVersion
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetRuntimeVersion(uint32_t *runtimeVersion);
#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif

#endif  // __CCE_RUNTIME_STREAM_H__
