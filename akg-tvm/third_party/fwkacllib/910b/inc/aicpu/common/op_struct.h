/**
* Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*
* Description: define op info struct
*/
#ifndef OP_STRUCT_H
#define OP_STRUCT_H

#include <string>
#include <map>
#include <vector>

namespace aicpu {

struct OpFullInfo {
    std::string engine;       // which engine
    std::string opKernelLib;  // which opsKernelStore
    int computeCost;     // compute cost
    bool flagPartial;    // whether to support is related to shape
    bool flagAsync;      // Whether to support asynchronous
    std::string kernelSo;     // kernel so
    std::string functionName; // function name
    bool userDefined;    // user defined
    std::map<std::string, std::string> inOutFormat; // input output format
    std::string opsFlag;      // opsFlag[0] indicates constant folding
    int shapeType;
    std::map<std::string, std::string> inOutDataType; // input output DataType
    std::map<std::string, std::string> inOutRealName; // input output name
    bool formatAgnostic; // set format agnostic
    int workspaceSize;   // workspace size
    std::map<std::string, std::string> castSrcType;
    std::map<std::string, std::string> castDstType;
};

struct OpInfoDesc {
    std::string opName;  // op name
    OpFullInfo opInfo;   // engine information that the op
};

struct OpInfoDescs {
    std::vector<OpInfoDesc> opInfos;  // op info
    std::string libName;              // kernel lib name
};
}

#endif
