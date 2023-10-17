/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#ifndef AICPU_TASK_STRUCT_H
#define AICPU_TASK_STRUCT_H

#include <cstdint>

namespace aicpu {

using char_t = char;

#pragma pack(push, 1)
struct AicpuParamHead {
    uint32_t        length;                    // Total length: include cunstom message
    uint32_t        ioAddrNum;                 // Input and output address number
    uint32_t        extInfoLength;             // extInfo struct Length
    uint64_t        extInfoAddr;               // extInfo address
};

enum class AicpuConfigMsgType {
    AICPU_CONFIG_MSG_TYPE_BUF_FREE      = 0,  /* free buf */
    AICPU_CONFIG_MSG_TYPE_BUF_RESET     = 1,  /* reset buf */
    AICPU_CONFIG_MSG_TYPE_BUF_SET_ADDR  = 2,  /* set buf addr to aicpu */
};

enum class AicpuErrMsgType {
    ERR_MSG_TYPE_NULL   = 0,
    ERR_MSG_TYPE_AICORE = 1,
    ERR_MSG_TYPE_AICPU  = 2,
};

enum class AicpuExtInfoMsgType {
    EXT_MODEL_ID_MSG_TYPE = 0,
};

struct AicpuConfigMsg {
    uint8_t msgType;
    uint8_t reserved1;
    uint16_t bufLen;
    uint32_t offset;
    uint64_t bufAddr;
    uint32_t tsId;
    uint32_t reserved2;
};

struct AicpuModelIdInfo {
    uint32_t modelId;
    uint32_t extendModelId;
    uint32_t extendInfo[13];
};

// 64 bytes
struct AicpuExtendInfo {
    uint8_t msgType;
    uint8_t version;
    uint8_t reserved[2];
    union {
        AicpuModelIdInfo modelIdMap;
    };
};

struct AicoreErrMsgInfo {
    uint8_t errType;
    uint8_t version;
    uint8_t reserved1[2];    /* reserved1, 4 byte alignment */
    uint32_t errorCode;
    uint32_t modelId;
    uint32_t taskId;
    uint32_t streamId;
    uint64_t transactionId;
    uint8_t reserved2[228];  /* the total byte is 256, reserved2 len = 256 - other lens */
};

struct AicpuErrMsgInfo {
    uint8_t errType;
    uint8_t version;
    uint8_t reserved1[2];    /* reserved1, 4 byte alignment */
    uint32_t errorCode;
    uint32_t modelId;
    uint32_t streamId;
    uint64_t transactionId;
    char_t opName[64];      /* op name str */
    char_t errDesc[128];    /* err msg desc info */
    uint8_t reserved2[40];  /* the total byte is 256, reserved2 len = 256 - other lens */
};
#pragma pack(pop)

}  // namespace aicpu

#endif  // AICPU_TASK_STRUCT_H

