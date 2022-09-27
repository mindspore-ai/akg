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

/*!
 * \file map_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MAP_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MAP_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Returns whether the given key exists in the map. \n

* @par Inputs:
* @li input_handle: A scalar Tensor of type variant. The original map.
* @li key: The key to check. Supports int32, int64, string. \n

* @par Outputs:
* has_key: A scalar Tensor of type bool. Whether the key is already in the map or not. \n

* @par Third-party framework compatibility.
* Compatible with tensorflow TensorMapHasKey operator.
*/
REG_OP(TensorMapHasKey)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(key, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(has_key, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(TensorMapHasKey)

/**
* @brief Returns a tensor map with item from given key erased. \n

* @par Inputs:
* @li input_handle: A scalar Tensor of type variant. The original map.
* @li key: The key of the value to be erased. Supports int32, int64, string. \n

* @par Outputs:
* output_handle: A scalar Tensor of type variant. The map with value from given key removed. \n

* @par Third-party framework compatibility.
* Compatible with tensorflow TensorMapErase operator.
*/    
REG_OP(TensorMapErase)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(key, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .OP_END_FACTORY_REG(TensorMapErase)

/**
* @brief Returns a map that is the 'input_handle'
  with the given key-value pair inserted. \n

* @par Inputs:
* @li input_handle: The original map, Must be type: DT_VARIANT.
* @li key: A Tensor,the key to be inserted.Must be one of
  the following types: int32, int64, string.
* @li value: A Tensor,the value to be inserted.Must be
  one of BasicType types. \n

* @par Outputs:
* output_handle: The map with key and value inserted.
  Must be type: DT_VARIANT. \n
*/
REG_OP(TensorMapInsert)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(key, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(value, BasicType)
    .OUTPUT(output_handle, TensorType({DT_VARIANT}))
    .OP_END_FACTORY_REG(TensorMapInsert)

/**
* @brief Returns the value from a given key in a tensor map . \n

* @par Inputs:
* @li input_handle: The input map. Must be type: DT_VARIANT.
* @li key: A Tensor, the key to be looked up. Must be one of
  the following types: int32,int64,string . \n

* @par Attributes:
* value_dtype: A int. Representing the type of value . \n

* @par Outputs:
* value: A Tensor,the value found from the given key.
*/
REG_OP(TensorMapLookup)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .INPUT(key, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(value, BasicType)
    .REQUIRED_ATTR(value_dtype, Type)
    .OP_END_FACTORY_REG(TensorMapLookup)

/**
* @brief return TensorMap Size. \n
*
* @par Inputs:
* input_handle: A Tensor. Must be one of the following types: variant. \n
*
* @par Outputs:
* size: A Tensor. Must be one of the following types: int32. \n
*/
REG_OP(TensorMapSize)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .OUTPUT(size, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(TensorMapSize)

/**
 * @brief Return TensorMapStackKeys \n
 *
 * @par Inputs:
 * input_handle: A Tensor. Must be one of the following types: variant. \n
 *
 * @par Outputs:
 * keys: A Tensor. Must be one of the following types: int32, int64, string. \n
 * 
 * @par Attributes:
 * key_dtype: An required param. It is the dtype of the key.
 */
REG_OP(TensorMapStackKeys)
    .INPUT(input_handle, TensorType({DT_VARIANT}))
    .OUTPUT(keys, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .REQUIRED_ATTR(key_dtype, Type)
    .OP_END_FACTORY_REG(TensorMapStackKeys)

/**
* @brief Creates and returns an empty tensor map. \n

* @par Outputs:
* handle: An empty tensor map . \n

* @par Third-party framework compatibility.
* Compatible with tensorflow EmptyTensorMap operator.
*/
REG_OP(EmptyTensorMap)
    .OUTPUT(handle, TensorType({DT_VARIANT}))
    .OP_END_FACTORY_REG(EmptyTensorMap)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_MAP_OPS_H_
