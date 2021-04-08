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

/*!
 * \file lookup_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_LOOKUP_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_LOOKUP_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Replaces the contents of the table with the specified keys and values . \n

*@par Inputs:
*The dtype of input handle must be resource. Inputs include:
*@li handle: A Tensor of type resource. Handle to the table.
*@li keys: A Tensor. Any shape. Keys to look up.
*@li values: A Tensor. Values to associate with keys . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow LookupTableImport operator.
*/

REG_OP(LookupTableImport)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(keys, TensorType({DT_STRING, DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_DOUBLE, \
        DT_FLOAT, DT_INT32, DT_INT64, DT_STRING}))
    .OP_END_FACTORY_REG(LookupTableImport)

/**
*@brief Updates the table to associates keys with values . \n

*@par Inputs:
*The dtype of input handle must be resource. Inputs include:
*@li handle: A Tensor of type resource. Handle to the table.
*@li keys: A Tensor. Any shape. Keys to look up.
*@li values: A Tensor. Values to associate with keys . \n

*@attention Constraints:
*@li The tensor keys must be of the same type as the keys of the table.
*@li The tensor values must be of the type of the table values.

*@par Third-party framework compatibility.
*Compatible with tensorflow LookupTableInsert operator.
*/

REG_OP(LookupTableInsert)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(keys, TensorType({DT_STRING, DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_DOUBLE, DT_FLOAT, \
        DT_INT32, DT_INT64, DT_STRING}))
    .OP_END_FACTORY_REG(LookupTableInsert)

/**
*@brief Outputs all keys and values in the table . \n

*@par Inputs:
*The dtype of input handle must be resource. Inputs include:
*handle: A Tensor of type resource. Handle to the table . \n

*@par Attributes:
*@li Tkeys: A DType.
*@li Tvalues: A DType . \n

*@par Outputs:
*@li keys: A Tensor of type Tkeys.
*@li values: A Tensor of type Tvalues . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow LookupTableExport operator.
*/

REG_OP(LookupTableExport)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(keys, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(values, TensorType({DT_BOOL, DT_DOUBLE, DT_FLOAT, \
        DT_INT32, DT_INT64, DT_STRING}))
    .REQUIRED_ATTR(Tkeys, Type)
    .REQUIRED_ATTR(Tvalues, Type)
    .OP_END_FACTORY_REG(LookupTableExport)

/**
*@brief Computes the number of elements in the given table . \n

*@par Inputs:
*The dtype of input handle must be resource. Inputs include:
*handle: A Tensor of type resource. Handle to the table . \n

*@par Outputs:
*size: A Tensor of type int64 . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow LookupTableSize operator.
*/

REG_OP(LookupTableSize)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(size, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(LookupTableSize)

/**
*@brief Looks up keys in a table, outputs the corresponding values . \n

*@par Inputs:
*The dtype of input handle must be resource. Inputs include:
*@li handle: A Tensor of type resource. Handle to the table.
*@li keys: A Tensor. Any shape. Keys to look up.
*@li default_value: A Tensor . \n

*@par Attributes:
*Tout: Specified type of ouput values . \n

*@par Outputs:
*values: A Tensor. Has the same type as default_value . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow LookupTableFind operator.
*/

REG_OP(LookupTableFind)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(keys, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(default_value, TensorType({DT_DOUBLE, DT_FLOAT, \
        DT_INT32, DT_INT64, DT_STRING, DT_BOOL}))
    .OUTPUT(values, TensorType({DT_DOUBLE, DT_FLOAT, DT_INT32, \
        DT_INT64, DT_STRING, DT_BOOL}))
    .REQUIRED_ATTR(Tout, Type)
    .OP_END_FACTORY_REG(LookupTableFind)

/**
*@brief Creates a non-initialized hash table . \n

*@par Attributes:
*@li container: An optional string. Defaults to "". If non-empty, this table
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". If non-empty, this
table is shared under the given name across multiple sessions.
*@li use_node_name_sharing: An optional bool. Defaults to False. If true and
shared_name is empty, the table is shared using the node name.
*@li key_dtype: A DType. Type of the table keys.
*@li value_dtype: A DType. Type of the table values . \n

*@par Outputs:
*handle: A Tensor of type resource. Handle to the table . \n

*@attention Constraints:
*The implementation for HashTable on Ascend uses ai cpu, with bad performance.

*@par Third-party framework compatibility.
*Compatible with tensorflow HashTable operator.
*/

REG_OP(HashTable)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(use_node_name_sharing, Bool, false)
    .REQUIRED_ATTR(key_dtype, Type)
    .REQUIRED_ATTR(value_dtype, Type)
    .OP_END_FACTORY_REG(HashTable)

/**
*@brief Table initializer that takes two tensors for keys and values
respectively . \n

*@par Inputs:
*The dtype of input handle must be resource. Inputs include:
*@li handle: A Tensor of type resource. Handle to a table which will be
initialized.
*@li keys: A Tensor. Keys of type Tkey.
*@li values: A Tensor. Values of type Tval . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow InitializeTable operator.
*/

REG_OP(InitializeTable)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(keys, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(values, TensorType({DT_INT32, DT_INT64, DT_FLOAT, \
        DT_DOUBLE, DT_BOOL, DT_STRING}))
    .OP_END_FACTORY_REG(InitializeTable)

/**
*@brief Creates an empty hash table that uses tensors as the backing store . \n

*@par Inputs:
*The input deleted_key must have the same type as empty_key. Inputs include:
*@li empty_key: A Tensor. The key used to represent empty key buckets
internally. Must not be used in insert or lookup operations.
*@li deleted_key: A Tensor. Must have the same type as empty_key . \n

*@par Attributes:
*@li container: An optional string. Defaults to "". If non-empty, this table
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". If non-empty, this
table is shared under the given name across multiple sessions.
*@li use_node_name_sharing: An optional bool. Defaults to False. If true and
shared_name is empty, the table is shared using the node name.
*@li value_dtype: A DType. Type of the table values.
*@li value_shape: An optional TensorShape or list of ints. Defaults to [].
The shape of each value.
*@li initial_num_buckets: An optional int. Defaults to 131072. The initial
number of hash table buckets. Must be a power to 2.
*@li max_load_factor: An optional float. Defaults to 0.8. The maximum ratio
between number of entries and number of buckets before growing the table.
Must be between 0 and 1 . \n

*@par Outputs:
*handle: A Tensor of type resource. Handle to the table . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow MutableDenseHashTable operator.
*/

REG_OP(MutableDenseHashTable)
    .INPUT(empty_key, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(deleted_key, TensorType({DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(use_node_name_sharing, Bool, false)
    .REQUIRED_ATTR(value_dtype, Type)
    .ATTR(value_shape, ListInt, {})
    .ATTR(initial_num_buckets, Int, 131072)
    .ATTR(max_load_factor, Float, 0.8)
    .OP_END_FACTORY_REG(MutableDenseHashTable)

/**
*@brief Creates an empty hash table . \n

*@par Attributes:
*@li container: An optional string. Defaults to "". If non-empty, this table
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". If non-empty, this
table is shared under the given name across multiple sessions.
*@li use_node_name_sharing: An optional bool. Defaults to False. If true and
shared_name is empty, the table is shared using the node name.
*@li key_dtype: A DType. Type of the table keys.
*@li value_dtype: A DType. Type of the table values.
*@li value_shape: An optional TensorShape or list of ints. Defaults to [] . \n

*@par Outputs:
*handle: A Tensor of type resource. Handle to the table . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow MutableHashTableOfTensors operator.
*/

REG_OP(MutableHashTableOfTensors)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(use_node_name_sharing, Bool, false)
    .REQUIRED_ATTR(key_dtype, Type)
    .REQUIRED_ATTR(value_dtype, Type)
    .ATTR(value_shape, ListInt, {})
    .OP_END_FACTORY_REG(MutableHashTableOfTensors)

/**
*@brief Creates an empty hash table . \n

*@par Attributes:
*@li container: An optional string. Defaults to "". If non-empty, this table
is placed in the given container. Otherwise, a default container is used.
*@li shared_name: An optional string. Defaults to "". If non-empty, this
table is shared under the given name across multiple sessions.
*@li use_node_name_sharing: An optional bool. Defaults to False. If true and
shared_name is empty, the table is shared using the node name.
*@li key_dtype: A DType. Type of the table keys.
*@li value_dtype: A DType. Type of the table values . \n

*@par Outputs:
*handle: A Tensor of type resource. Handle to the table . \n

*@par Third-party framework compatibility.
*Compatible with tensorflow MutableHashTable operator.
*/

REG_OP(MutableHashTable)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(use_node_name_sharing, Bool, false)
    .REQUIRED_ATTR(key_dtype, Type)
    .REQUIRED_ATTR(value_dtype, Type)
    .OP_END_FACTORY_REG(MutableHashTable)
}   // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_LOOKUP_OPS_H_
