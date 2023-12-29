/**
* @file adx_datadump_callback.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef ADX_DATADUMP_CALLBACK_H
#define ADX_DATADUMP_CALLBACK_H
#include <cstdint>
#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define ADX_API __declspec(dllexport)
#else
#define ADX_API __attribute__((visibility("default")))
#endif
namespace Adx {
const uint32_t MAX_FILE_PATH_LENGTH          = 4096;
struct DumpChunk {
    char       fileName[MAX_FILE_PATH_LENGTH];   // file name, absolute path
    uint32_t   bufLen;                           // dataBuf length
    uint32_t   isLastChunk;                      // is last chunk. 0: not 1: yes
    int64_t    offset;                           // Offset in file. -1: append write
    int32_t    flag;                             // flag
    uint8_t    dataBuf[0];                       // data buffer
};

ADX_API int AdxRegDumpProcessCallBack(int (* const messageCallback)(const Adx::DumpChunk *, int));
ADX_API void AdxUnRegDumpProcessCallBack();
}

#endif
