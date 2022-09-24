/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef ADX_DATADUMP_CALLBACK_H
#define ADX_DATADUMP_CALLBACK_H
#include <cstdint>
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

    int AdxRegDumpProcessCallBack(int (* const messageCallback) (const Adx::DumpChunk *, int));
    void AdxUnRegDumpProcessCallBack();
}

#endif  // ADX_DATADUMP_CALLBACK_H