/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: ffts plus interface
 */

#ifndef CCE_RUNTIME_RT_FFTS_PLUS_H
#define CCE_RUNTIME_RT_FFTS_PLUS_H

#include "base.h"
#include "rt_ffts_plus_define.h"
#include "rt_stars_define.h"
#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#pragma pack(push)
#pragma pack (1)

// context desc addr type
#define RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_HOST   (0x0U)
#define RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_DEVICE (0x1U)

typedef struct tagFftsPlusDumpInfo {
    const void *loadDumpInfo;
    const void *unloadDumpInfo;
    uint32_t loadDumpInfolen;
    uint32_t unloadDumpInfolen;
} rtFftsPlusDumpInfo_t;

typedef struct tagFftsPlusTaskInfo {
    const rtFftsPlusSqe_t *fftsPlusSqe;
    const void *descBuf;                   // include total context
    size_t      descBufLen;                // the length of descBuf
    rtFftsPlusDumpInfo_t fftsPlusDumpInfo; // used only in the dynamic shape
    uint32_t descAddrType;                 // 0:host addr 1:device addr
    uint32_t argsHandleInfoNum;
    void **argsHandleInfoPtr;
} rtFftsPlusTaskInfo_t;

#pragma pack(pop)

RTS_API rtError_t rtGetAddrAndPrefCntWithHandle(void *hdl, const void *kernelInfoExt, void **addr,
    uint32_t *prefetchCnt);

RTS_API rtError_t rtFftsPlusTaskLaunch(rtFftsPlusTaskInfo_t *fftsPlusTaskInfo, rtStream_t stm);

RTS_API rtError_t rtFftsPlusTaskLaunchWithFlag(rtFftsPlusTaskInfo_t *fftsPlusTaskInfo, rtStream_t stm,
                                               uint32_t flag);

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif // CCE_RUNTIME_RT_FFTS_PLUS_H
