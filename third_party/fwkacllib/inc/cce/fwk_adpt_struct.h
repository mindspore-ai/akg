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

#ifndef FWK_ADPT_STRUCT_H__
#define FWK_ADPT_STRUCT_H__

#include <cstdint>

namespace aicpu {
namespace FWKAdapter {

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
  FWK_ADPT_UNKNOWN_ERROR = 99            // unknown error code
};

// FWKAdapter operate type
// Notice: add new operate type  need check with OMM, and make sure append to the end line.
enum FWKOperateType {
  FWK_ADPT_SESSION_CREATE = 0,
  FWK_ADPT_KERNEL_RUN,
  FWK_ADPT_KERNEL_DESTROY,
  FWK_ADPT_SESSION_DESTROY,
  FWK_ADPT_SINGLE_OP_RUN
};

// Extend Info type for task
enum FWKTaskExtInfoType {
  FWK_ADPT_EXT_SHAPE_TYPE = 0,
  FWK_ADPT_EXT_INPUT_SHAPE,
  FWK_ADPT_EXT_OUTPUT_SHAPE,
  FWK_ADPT_EXT_INVALID
};

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

  uint64_t extInfoNum;         // extend info number
  uint64_t extInfoAddr;        // extend info addr list, ExtInfo structure, num equal to extInfoNum
} __attribute__((packed));

typedef StrFWKKernel FWKOperateParam;

// Extend info structure for extInfoAddr
struct ExtInfo{
  int32_t  infoType;
  uint32_t infoLen;
  uint64_t infoAddr;
} __attribute__((packed));

struct ResultSummary {
  uint64_t shape_data_ptr;   // shape data addr, need convert to void*
  uint64_t shape_data_size;  // num of dims
  uint64_t raw_data_ptr;     // raw data addr,  need convert to void*
  uint64_t raw_data_size;    // size of raw data
} __attribute__((packed));
}  // end  namespace FWKAdapter
}  // namespace aicpu

#endif  // FWK_ADPT_STRUCT_H__
