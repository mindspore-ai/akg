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

#ifndef CCE_DEF_H__
#define CCE_DEF_H__

#include "runtime/rt.h"

namespace cce {

/**
 * @ingroup cce
 * @brief memory configure for fusion
 */
typedef struct TagCceFusionMemCfg {
  uint64_t memAddr;        /**< memAddr */
  uint32_t memSize;        /**< memSize */
  uint32_t addrChangeFlag; /**< op data addr change flag. value:0,valid;1,not valid */
  uint32_t poolFlag;       /**< mempool flag : value:0,is valid; value: 1, not valid */
  TagCceFusionMemCfg() {
    memAddr = 0;
    memSize = 0;
    addrChangeFlag = 0;
    poolFlag = 0;
  }
} CceFusionMemCfg_t;
/**
 * @ingroup cce
 * @brief return value
 */
typedef enum tagCcStatus {
  CC_STATUS_SUCCESS = 0,         /**< succ */
  CC_STATUS_NOT_INITIALIZED = 1, /**< not init */
  CC_STATUS_ALLOC_FAILED = 2,    /**< alloc mem failed */
  CC_STATUS_BAD_PARAM = 3,       /**< para check failed */
  CC_STATUS_INTERNAL_ERROR = 4,  /**< internal error */
  CC_STATUS_KERNEL_ERROR = 5,    /**< kernel error */
  CC_STATUS_RUNTIME_ERROR = 6,   /**< runtime error */
  CC_STATUS_NOT_SUPPORTED = 7,   /**< unsupport error */
  CC_STATUS_INVALID_VALUE = 7,   /**< invalid value error for blas*/
  CC_STATUS_RESERVED             /**< just for check */
} ccStatus_t;

/**
 * @ingroup cce
 * @brief original data type
 */
typedef enum tagCcDataType {
  CC_DATA_FLOAT = 0,            /**< float type */
  CC_DATA_HALF,                 /**< fp16 type */
  CC_DATA_INT8,                 /**< int8 type */
  CC_DATA_INT32,                /**< int32 type */
  CC_DATA_UINT8,                /**< uint8 type */
  CC_DATA_HALF_UINT16_PROPOSAL, /**<mixed type for proposal*/
  CC_DATA_INT16,                /**< int16 type */
  CC_DATA_UINT16,               /**< uint16 type */
  CC_DATA_UINT32,               /**< uint32 type */
  CC_DATA_INT64,                /**< int64 type */
  CC_DATA_UINT64,               /**< uint64 type */
  CC_DATA_DOUBLE,               /**< double type */
  CC_DATA_BOOL,                 /**< bool type */
  CC_DATA_DUAL,                 /**< dual output type */
  CC_DATA_DUAL_SUB_INT8,        /**< dual output int8 type */
  CC_DATA_DUAL_SUB_UINT8,       /**< dual output uint8 type */
  CC_DATA_COMPLEX64,
  CC_DATA_COMPLEX128,
  CC_DATA_QINT8,
  CC_DATA_QINT16,
  CC_DATA_QINT32,
  CC_DATA_QUINT8,
  CC_DATA_QUINT16,
  CC_DATA_RESERVED
} ccDataType_t;

/**
 * @ingroup cce
 * @brief save context of cce library
 */
typedef struct tagCcContext {
  rtStream_t streamId;
  uint32_t opIndex;
} ccContext_t;

typedef struct tagCcContext *ccHandle_t;

/**
 * @ingroup cce
 * @brief mode of data type transform
 */
typedef enum tagCcDataTypeTransMode {
  CC_DATATYPE_TRANS_FLOAT_NO_TRANS = 0, /**< origin data is float, no trans */
  CC_DATATYPE_TRANS_FP16_NO_TRANS,      /**< origin data is fp16, no trans */
  CC_DATATYPE_TRANS_INT8_NO_TRANS,      /**< origin data is int8, no trans */
  CC_DATATYPE_TRANS_FLOAT_TO_FP16,      /**< data type float trans to fp16 */
  CC_DATATYPE_TRANS_FP16_TO_FLOAT,      /**< data type fp16 trans to float */
  CC_DATATYPE_TRANS_FLOAT_TO_INT8,      /**< data type float trans to int8 */
  CC_DATATYPE_TRANS_INT8_TO_FLOAT,      /**< data type int8 trans to float */
  CC_DATATYPE_TRANS_UINT8_TO_FLOAT,     /**< data type uint8 trans to float */
  CC_DATATYPE_TRANS_UINT8_NO_TRANS,     /**< origin data is uint8, no trans */
  CC_DATATYPE_TRANS_INT32_NO_TRANS,     /**< data type uint8 trans to float */
  CC_DATATYPE_TRANS_UINT16_NO_TRANS,    /** < origin data is uint16, no trans*/
  CC_DATATYPE_TRANS_UINT16_TO_FLOAT,    /** < data type uint16 trans to float*/
  CC_DATATYPE_TRANS_MODE_RESERVED
} ccDataTypeTransMode_t;

typedef struct tagContextInfo {
  ccHandle_t handle;
  rtStream_t stream;
  uint8_t *memBase;
  uint64_t totalMemSize;
  uint8_t *weightsMemBase;
  uint64_t weightsMemSize;
  uint8_t *weightsMemBaseHost;
} ContextInfo;

/**
 * @ingroup cce
 * @brief cce function parameter type
 */
typedef enum tagCcFuncType {
  CC_FUSION_L2,
  GLOBAL_MEMORY_CLEAR,
  MAX_NUM,
} ccFuncParamType_t;

/**
 * @ingroup cce
 * @brief cce set function point state
 */
ccStatus_t ccSetFuncState(ccFuncParamType_t type, bool isOpen);

/**
 * @ingroup cce
 * @brief cce get function point state
 */
bool ccGetFuncState(ccFuncParamType_t type);

}  // namespace cce
#endif  // CCE_DEF_H__
