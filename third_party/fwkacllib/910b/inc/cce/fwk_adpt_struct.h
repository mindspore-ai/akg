/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef FWK_ADPT_STRUCT_H
#define FWK_ADPT_STRUCT_H

#include <cstdint>

namespace aicpu {
namespace FWKAdapter {
using char_t = char;
// API RETURN CODE
enum FWKAdptAPIRetCode {
    FWK_ADPT_SUCCESS = 0,                  // success
    FWK_ADPT_NOT_INIT = 1,                 // not init
    FWK_ADPT_ALLOC_FAILED = 2,             // allocate memory failed
    FWK_ADPT_PARAM_INVALID = 3,            // invalid input param
    FWK_ADPT_PARAM_PARSE_FAILED = 4,       // parase input param failed
    FWK_ADPT_NATIVE_ERROR = 5,             // error code
    FWK_ADPT_NOT_SUPPORT_OPTYPE = 6,       // unsupport operate type
    FWK_ADPT_INTERNAL_ERROR = 7,           // adpter internal error
    FWK_ADPT_NOT_SUPPORT_DATATYPE = 8,     // unsupport input/output data type
    FWK_ADPT_KERNEL_ALREADY_RUNING = 9,    // kernel already runing, not support parallel run
    FWK_ADPT_SESSION_NOT_EXIST = 10,       // session id not exist
    FWK_ADPT_SESSION_ALREADY_EXIST = 11,   // session id alread exist for create session
    FWK_ADPT_NATIVE_END_OF_SEQUENCE = 12,  // end of sequence
    FWK_ADPT_EXTEND_TYPE_NOT_EXIST = 13,   // extend info type not exist
    FWK_ADPT_UNKNOWN_ERROR = 99            // unknown error code
};

// FWKAdapter operate type
// Notice: add new operate type  need check with OMM, and make sure append to the end line.
enum FWKOperateType {
    FWK_ADPT_SESSION_CREATE = 0,
    FWK_ADPT_KERNEL_RUN,
    FWK_ADPT_KERNEL_DESTROY,
    FWK_ADPT_SESSION_DESTROY,
    FWK_ADPT_SINGLE_OP_RUN,
    FWK_ADPT_KERNEL_RUN_NO_SESS,
};

// Extend Info type for task
enum FWKTaskExtInfoType {
    FWK_ADPT_EXT_SHAPE_TYPE = 0,
    FWK_ADPT_EXT_INPUT_SHAPE,
    FWK_ADPT_EXT_OUTPUT_SHAPE,
    FWK_ADPT_EXT_UPDATE_ADDR,
    FWK_ADPT_EXT_OP_NAME,
    FWK_ADPT_EXT_SESSION_INFO,
    FWK_ADPT_EXT_BITMAP,
    FWK_ADPT_EXT_TOPIC_TYPE,
    FWK_ADPT_EXT_ASYNCWAIT,
    FWK_ADPT_EXT_UNKNOWN_SHAPE_INPUT_INDEX,
    FWK_ADPT_EXT_UNKNOWN_SHAPE_OUTPUT_INDEX,
    FWK_ADPT_EXT_WORKSPACE_INFO,
    FWK_ADPT_EXT_INVALID
};

enum FWKExtTopicType {
    FWK_ADPT_TOPIC_DEVICE_ONLY = 0,
    FWK_ADPT_TOPIC_DEVICE_FIRST,
    FWK_ADPT_TOPIC_HOST_ONLY,
    FWK_ADPT_TOPIC_HOST_FIRST,
    FWK_ADPT_TOPIC_INVALID
};

enum FWKExtUpdateAddrType {
    FWK_ADPT_UPDATE_NULL = 0,
    FWK_ADPT_UPDATE_INPUT,
    FWK_ADPT_UPDATE_OUTPUT,
    FWK_ADPT_UPDATE_INPUT_OUTPUT
};

enum FWKExtWaitType {
    FWK_ADPT_WAIT_TYPE_NULL = 0,
    FWK_ADPT_WAIT_TYPE_EVENT,
    FWK_ADPT_WAIT_TYPE_INVALID
};

#pragma pack(push, 1)
// API Parameter Structure
struct StrFWKKernel {
    FWKOperateType opType;
    uint64_t sessionID;  // unique

    uint64_t stepIDAddr;    // step id addr
    uint64_t kernelID;      // run kernel id, unique in session
    uint64_t nodeDefLen;    // nodeDef protobuf len
    uint64_t nodeDefBuf;    // NodeDef protobuf offset addr, need convert to void*
    uint64_t funDefLibLen;  // FunctionDefLibrary protobuf len
    uint64_t funDefLibBuf;  // FunctionDefLibrary protobuf addr which use in NodeDef, need convert to void*

    uint64_t inputOutputLen;     // InputOutput shap protobuf len
    uint64_t inputOutputBuf;     // InputOutput shap protobuf addr, need convert to void*
    uint64_t workspaceBaseAddr;  // Workspace base addr, need convert to void*
    uint64_t inputOutputAddr;    // InputOutput addr, need convert to void*

    uint64_t extInfoLen;         // extend info total length
    uint64_t extInfoAddr;        // extend info addr, ExtInfo structure
};
#pragma pack(pop)

using FWKOperateParam = StrFWKKernel;

// Extent info ShapeAndType
const uint32_t kMaxShapeDims = 8U;
#pragma pack(push, 1)
struct ShapeAndType {
    int32_t type;
    int64_t dims[kMaxShapeDims];
};
#pragma pack(pop)

// Extend info structure for extInfoAddr
const uint32_t kExtInfoHeadSize = 8U;

#pragma pack(push, 1)
struct ExtInfo {
    int32_t  infoType;    // extend type
    uint32_t infoLen;     // length for infoMsg
    char_t  infoMsg[0];  // extend value
};
#pragma pack(pop)

#pragma pack(push, 1)
struct ResultSummary {
    uint64_t shape_data_ptr;   // shape data addr, need convert to void*
    uint64_t shape_data_size;  // num of dims
    uint64_t raw_data_ptr;     // raw data addr,  need convert to void*
    uint64_t raw_data_size;    // size of raw data
};
#pragma pack(pop)

#pragma pack(push, 1)
struct AsyncWait {
    uint8_t waitType;  // wait type, FWk_ADPT_WAIT_TPYE_EVENT: event wait
    uint32_t waitId;  // wait id, GE refresh
    uint32_t timeOut;  // reserved
    uint64_t reserved;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct WorkSpaceInfo {
    uint64_t size;
    uint64_t addr;
};
#pragma pack(pop)
}  // end  namespace FWKAdapter
}  // namespace aicpu

#endif  // FWK_ADPT_STRUCT_H
