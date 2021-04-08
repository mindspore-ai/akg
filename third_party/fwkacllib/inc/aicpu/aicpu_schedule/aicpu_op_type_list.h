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

#ifndef AICPU_OP_TYPE_LIST_H_
#define AICPU_OP_TYPE_LIST_H_

enum OpKernelType {
    TF_KERNEL,
    CPU_KERNEL
};

enum ReturnCode {
    OP_TYPE_NOT_SUPPORT,
    FORMAT_NOT_SUPPORT,
    DTYPE_NOT_SUPPORT
};

#pragma pack(push, 1)
//One byte alignment
struct SysOpInfo {
    uint64_t opLen;
    uint64_t opType;
    OpKernelType kernelsType;
};

struct OpParamInfo {
    uint64_t num;
    uint64_t dtypeList;
    uint64_t formatList;
};

struct SysOpCheckInfo {
    uint64_t opListNum;
    uint64_t offSetLen;
    uint64_t sysOpInfoList;
    uint64_t opParamInfoList;
};

struct SysOpCheckResp {
    uint64_t opListNum;
    bool isWithoutJson;
    uint64_t returnCodeList;
    uint64_t sysOpInfoList;
    uint64_t opParamInfoList;
};
#pragma pack(pop)
#endif  // AICPU_OP_TYPE_LIST_H_
