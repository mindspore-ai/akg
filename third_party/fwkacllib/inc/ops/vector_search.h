/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

/*!
 * \file vector_search.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
#define OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Generate ADC(asymmetric distance computation) table. \n
*
* @par Inputs:
* Four inputs, including:
* @li query: A Tensor. Must be one of the following types: float16, float32.
* @li code_book: A Tensor. Must be one of the following types: float16, float32.
* @li centroids: A Tensor. Must be one of the following types: float16, float32.
* @li bucket_list: A Tensor. Must be one of the following types: int32, int64.
*
* @par Outputs:
* adc_tables: A Tensor. Must be one of the following types: float16, float32.
*
* @par Attributes:
* distance_type: The string indicates the distance type of ADC tables. Examples: `"l2sqr", "inner_product"`.
The default value is "l2sqr".
*/
REG_OP(GenADC)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(code_book, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(centroids, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(adc_tables, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(distance_type, String, "l2sqr")
    .OP_END_FACTORY_REG(GenADC)

/**
* @brief Finds values and indices of the "k" largest or least elements for the last dimension. \n
*
* @par Inputs:
* Dynamin inputs, including:
* @li actual_count: A Tensor of type int32, the actual number of pq_distance.
* @li pq_distance: A Tensor, Will be updated after calculation. Must be one of the following types: float32, float16. 
* @li grouped_extreme_distance: A Tensor, the extremum in each group. Must be one of the following types: float32, float16.
* @li pq_index: A Tensor of type int32, index corresponding to pq_distance.
* @li pq_ivf: A Tensor of type int32 , the bucket number corresponding to pq_distance.
*
* @par Attributes:
* @li order: A string, indicates the sorting method of topk_pq_distance. \n
* @li k: Int, k maximum or minimum values. \n
* @li group_size: Int, the group size of the extremum. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(TopKPQDistance)
    .DYNAMIC_INPUT(actual_count, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(pq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_INPUT(grouped_extreme_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DYNAMIC_INPUT(pq_ivf, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(pq_index, TensorType({DT_INT32}))
    .OUTPUT(topk_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(topk_ivf, TensorType({DT_INT32}))
    .OUTPUT(topk_index, TensorType({DT_INT32}))
    .ATTR(order, String, "ASC")
    .REQUIRED_ATTR(k, Int)
    .REQUIRED_ATTR(group_size, Int)
    .OP_END_FACTORY_REG(TopKPQDistance)

/**
* @brief Calculate PQ distance. \n
*
* @par Inputs:
* Six inputs, including:
* @li ivf: A Tensor, dtype is uint8.
* @li bucket_list: A Tensor, dtype is int32.
* @li bucket_base_distance: A Tensor, dtype is float16.
* @li bucket_limits: A Tensor, dtype is int32.
* @li bucket_offsets: A Tensor, dtype is int32.
* @li adc_tables: A Tensor. dtype is float16. \n
*
* @par Outputs:
* Five outputs, including:
* @li actual_count: A Tensor, dtype is int32, the first element means the length of processed ivf.
* @li pq_distance: A Tensor, dtype is float16.
* @li grouped_extreme_distance: A Tensor, dtype is float16.
* @li pq_ivf: A Tensor, dtype is int32.
* @li pq_index: A Tensor, dtype is int32. \n
*
* @par Attributes:
* Five attributes, including:
* @li group_size: A Scalar, indicates the group size when compute grouped_extreme_distance.
* @li total_limit: A Scalar, indicates the total length of the outputs.
* @li extreme_mode: A Scalar, indicates the type of extremum, 0 means minimum, and 1 means maximum.
* @li split_count: A Scalar.
* @li split_index: A Scalar. \n
*
*/
REG_OP(ScanPQCodes)
    .INPUT(ivf, TensorType({DT_UINT8}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_base_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_limits, TensorType({DT_INT32}))
    .INPUT(bucket_offsets, TensorType({DT_INT64}))
    .INPUT(adc_tables, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(actual_count, TensorType({DT_INT32}))
    .OUTPUT(pq_distance, TensorType({DT_FLOAT16}))
    .OUTPUT(grouped_extreme_distance, TensorType({DT_FLOAT16}))
    .OUTPUT(pq_ivf, TensorType({DT_INT32}))
    .OUTPUT(pq_index, TensorType({DT_INT32}))
    .REQUIRED_ATTR(total_limit, Int)
    .ATTR(group_size, Int, 64)
    .ATTR(extreme_mode, Int, 0)
    .ATTR(split_count, Int, 1)
    .ATTR(split_index, Int, 0)
    .OP_END_FACTORY_REG(ScanPQCodes)

/**
* @brief Calculate buckets limit and offset. \n

* @par Inputs:
* Three inputs, including:
* @li bucket_list: A 1-D tensor of type int32 with the value of ivf_counts and ivf_offset index. \n
* @li ivf_counts: A 1-D tensor of type int32 with the value of ivf counts. \n
* @li ivf_offset: A 1-D tensor of type int32 or int64 with the value of ivf offset. \n

* @par Attributes:
* total_limit: A int64 type maximum value of the sum of ivf_counts corresponding to bucket_list. \n

* @par Outputs:
* @li buckets_limit: A 1-D tensor of type int32 with the sum <= total_limit. \n
* @li buckets_offset: A 1-D tensor of type int32 or int64 with the value of ivf_offset corresponding to bucket_list. \n
*/
REG_OP(CalcBucketsLimitAndOffset)
    .INPUT(bucket_list, TensorType({DT_INT32}))
    .INPUT(ivf_counts, TensorType({DT_INT32}))
    .INPUT(ivf_offset, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(buckets_limit, TensorType({DT_INT32}))
    .OUTPUT(buckets_offset, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(total_limit, Int)
    .OP_END_FACTORY_REG(CalcBucketsLimitAndOffset)

/**
*@brief get block tensor according to base addr tensor, for hccl remote read to use.
*@par Inputs:
*@li base_addr: A Tensor of type int64/uint64. \n
*@li row:A Tensor of type int64/uint64. \n
*@li col: A Tensor of type int64/uint64.

*@par Outputs:
*addr_table: list of [rank id, host addr, device addr, read size]

*@par Attributes:
*@li ori_shape: An required list int. Shape of base tensor.
*@li block_size: An required list int. Shape of split block tensor.
*@li ori_storage_mode: An optional string from: '"Matrix", "UT"'. Defaults to
"Matrix". Currently only support Matrix storage
*@li block_storage_mode: An optional string from: '"Matrix", "UT"'. Defaults to
"Matrix". Currently only support Matrix storage
*@li rank_id: An optional int of rank id. Defaults is 0
*@li dtype: An optional Type of base tensor. Defaults is DT_FLOAT
*/
REG_OP(IndexToAddr)
    .INPUT(base_addr, TensorType({DT_INT64, DT_UINT64}))
    .INPUT(x, TensorType({DT_INT64, DT_UINT64}))
    .OUTPUT(addrs_table, TensorType({DT_INT64, DT_UINT64}))
    .REQUIRED_ATTR(ori_shape, ListInt)
    .REQUIRED_ATTR(block_size, ListInt)
    .ATTR(ori_storage_mode, String, "Matrix")
    .ATTR(block_storage_mode, String, "Matrix")
    .ATTR(rank_id, Int, 0)
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(IndexToAddr)

/**
*@brief Convert one-dimensional coordinates to two-dimensional coordinates.
*@par Inputs:
*@li x: A Tensor of type int32/int64/uint64. One-dimensional coordinates.
*@li shape: A Tensor of type int32/int64/uint64. 4D tensor [N,C,H,W].
*@par Outputs:
*@li row: row of two-dimensional
*@li col: col of two-dimensional
*@li n: col number of two-dimensional
*/
REG_OP(Coordinates1DTo2D)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(row, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(col, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(n, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OP_END_FACTORY_REG(Coordinates1DTo2D)

/**
*@brief x[0] is i, x[1] is j and x[2] is k when algorithm is LU,
y = 0 when i >= k && j < k,
y = 1 when i == k && j == k,
y = 2 when i > k && j == k,
y = 3 when i == k && j > k,
y = 4 when i > k && j > k,
default y = 5
use for lu decomposition
*@par Inputs:
*x: A Tensor of type int32/int64/uint64. \n

*@par Attributes:
*algorithm: A string, only support LU now
*@par Outputs:
*y: A Tensor of type int32
*/
REG_OP(CaseCondition)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(algorithm, String, "LU")
    .OP_END_FACTORY_REG(CaseCondition)

/**
*@brief write tensor value to tensor x.
*@par Inputs:
*x: A Tensor of type float16/float/double/int32/int64. \n
*begin:A Tensor of type int32/int64. \n
*value: A Tensor of type float16/float/double/int32/int64.
*@par Outputs:
*x: same tensor with input x
*/
REG_OP(SliceWrite)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .INPUT(begin, TensorType({DT_INT32, DT_INT64}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(SliceWrite)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_VECTOR_SEARCH_H_
