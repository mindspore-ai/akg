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

#ifndef CC_CUSTOMIZE_API__
#define CC_CUSTOMIZE_API__

#include <stdint.h>

#define CC_DEVICE_DIM_MAX 8
typedef enum tagOpTensorFormat {
    OP_TENSOR_FORMAT_NC1HWC0 = 0,
    OP_TENSOR_FORMAT_ND,
    OP_TENSOR_FORMAT_RESERVED,
} opTensorFormat_t;


typedef enum tagOpDataType {
    OP_DATA_FLOAT = 0,             /**< float type */
    OP_DATA_HALF,                  /**< fp16 type */
    OP_DATA_INT8,                  /**< int8 type */
    OP_DATA_INT32,                 /**< int32 type */
    OP_DATA_UINT8,                 /**< uint8 type */
    OP_DATA_HALF_UINT16_PROPOSAL,  /**<mixed type for proposal*/
    OP_DATA_RESERVED
} opDataType_t;

typedef struct tagOpTensor {
    // real dim info
    opTensorFormat_t format;
    opDataType_t data_type;
    int32_t dim_cnt;
    int32_t mm;
    int32_t dim[CC_DEVICE_DIM_MAX];
} opTensor_t;

typedef opTensor_t tagCcAICPUTensor;
typedef void *rtStream_t;
typedef void (*aicpu_run_func)(opTensor_t **, void **, int32_t,
                               opTensor_t **, void **, int32_t, void *, rtStream_t);


#endif  // CC_CUSTOMIZE_API__

