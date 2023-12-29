/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file parsing_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Converts each string in the input Tensor to the specified numeric type . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: string . \n

*@par Attributes:
*out_type: The numeric type to interpret each string in string_tensor as . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU, with bad performance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow StringToNumber operator.
*/
REG_OP(StringToNumber)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .ATTR(out_type, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StringToNumber)

/**
*@brief Convert serialized tensorflow.TensorProto prototype to Tensor.
*@brief Parse an Example prototype. 
*@par Inputs:
*@li serialized: A Tensor of type string.
*@li dense_defaults:  DYNAMIC INPUT Tensor type as string, float, int64. \n

*@par Attributes:
*@li num_sparse: type int num of inputs sparse_indices , sparse_values, sparse_shapes
*@li sparse_keys: ListString
*@li sparse_types: types of sparse_values
*@li dense_keys: ListString
*@li Tdense: output of dense_defaults type
*@li dense_shapes: output of dense_defaults shape  \n

*@par Outputs:
*@li sparse_indices: A Tensor of type string. 
*@li sparse_values:  Has the same type as sparse_types.
*@li sparse_shapes: A Tensor of type int64
*@li dense_values:  Has the same type as dense_defaults.

*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ParseSingleExample)
    .INPUT(serialized, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(dense_defaults, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_values, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(dense_values, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .ATTR(num_sparse, Int, 0)
    .ATTR(sparse_keys, ListString, {})
    .ATTR(dense_keys, ListString, {})
    .ATTR(sparse_types, ListType, {})
    .ATTR(Tdense, ListType, {})
    .ATTR(dense_shapes, ListListInt, {})
    .OP_END_FACTORY_REG(ParseSingleExample)

/**
*@brief Decodes raw file into  tensor . \n
*@par Inputs:
*bytes: A Tensor of type string.

*@par Attributes:
*@li little_endian: bool ture
*@li out_type: output type

*@par Outputs:
*Output: A Tensor
*/
REG_OP(DecodeRaw)
    .INPUT(bytes, TensorType({DT_STRING}))
    .OUTPUT(output, TensorType({DT_BOOL,DT_FLOAT16,DT_DOUBLE,DT_FLOAT,
                                    DT_INT64,DT_INT32,DT_INT8,DT_UINT8,DT_INT16,
                                    DT_UINT16,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(out_type, Type, DT_FLOAT)
    .ATTR(little_endian, Bool, true)
    .OP_END_FACTORY_REG(DecodeRaw)

/**
*@brief Convert serialized tensorflow.TensorProto prototype to Tensor. \n

*@par Inputs:
*serialized: A Tensor of string type. Scalar string containing serialized
*TensorProto prototype. \n

*@par Attributes:
*out_type: The type of the serialized tensor. The provided type must match the
*type of the serialized tensor and no implicit conversion will take place. \n

*@par Outputs:
*output: A Tensor of type out_type. \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU,
*with badperformance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow ParseTensor operator.
*/
REG_OP(ParseTensor)
    .INPUT(serialized, TensorType({DT_STRING}))
    .OUTPUT(output, TensorType(DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                          DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(out_type, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(ParseTensor)

/**
*@brief Converts each string in the input Tensor to the specified numeric
*type . \n

*@par Inputs:
*Inputs include:
*@li records: Each string is a record/row in the csv and all records should have the
*same format. \n
*@li record_defaults: One tensor per column of the input record, with either a
*scalar default value for that column or an empty vector if the column is
*required. \n

*@par Attributes:
*@li OUT_TYPE: The numeric type to interpret each string in string_tensor as . \n
*@li field_delim: char delimiter to separate fields in a record. \n
*@li use_quote_delim: If false, treats double quotation marks as regular characters
*inside of the string fields (ignoring RFC 4180, Section 2, Bullet 5). \n
*@li na_value: Additional string to recognize as NA/NaN. \n
*@li select_cols: Optional sorted list of column indices to select. If specified,
only this subset of columns will be parsed and returned.

*@par Outputs:
*output: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU, with bad
*performance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow StringToNumber operator.
*/
REG_OP(DecodeCSV)
    .INPUT(records, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(record_defaults, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32,
                                        DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32,
                                        DT_INT64, DT_STRING}))
    .ATTR(OUT_TYPE, ListType, {})
    .ATTR(field_delim, String, ",")
    .ATTR(use_quote_delim, Bool, true)
    .ATTR(na_value, String, ",")
    .ATTR(select_cols, ListInt, {})
    .OP_END_FACTORY_REG(DecodeCSV)

/**
*@brief Convert serialized tensorflow.TensorProto prototype to Tensor.
*@brief Parse an Example prototype.
*@par Inputs:
*@li serialized: A Tensor of type string. \n
*@li name:A Tensor of type string. \n
*@li sparse_keys: Dynamic input tensor of string. \n
*@li dense_keys: Dynamic input tensor of string \n
*@li dense_defaults:  Dynamic input tensor type as string, float, int64. \n

*@par Attributes:
*@li Nsparse: Number of sparse_keys, sparse_indices and sparse_shapes \n
*@li Ndense: Number of dense_keys \n
*@li sparse_types: types of sparse_values \n
*@li Tdense: Type of dense_defaults dense_defaults and dense_values \n
*@li dense_shapes: output of dense_defaults shape  \n

*@par Outputs:
*@li sparse_indices: A Tensor of type string. \n
*@li sparse_values:  Has the same type as sparse_types. \n
*@li sparse_shapes: A Tensor of type int64 \n
*@li dense_values:  Has the same type as dense_defaults. \n
*@par Third-party framework compatibility \n
*@li compatible with tensorflow StringToNumber operator. \n
*/
REG_OP(ParseExample)
    .INPUT(serialized, TensorType({DT_STRING}))
    .INPUT(name, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(sparse_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(dense_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(dense_defaults, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(dense_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .ATTR(Nsparse, Int, 0)
    .ATTR(Ndense, Int, 0)
    .ATTR(sparse_types, ListType, {})
    .ATTR(Tdense, ListType, {})
    .ATTR(dense_shapes, ListListInt, {})
    .OP_END_FACTORY_REG(ParseExample)

/**
*@brief Transforms a scalar brain.SequenceExample proto (as strings) into typed
*tensors.
*@par Inputs:
*@li serialized: A Tensor of type string. \n
*@li feature_list_dense_missing_assumed_empty:A Tensor of type string. \n
*@li context_sparse_keys: Dynamic input tensor of string. \n
*@li context_dense_keys: Dynamic input tensor of string \n
*@li feature_list_sparse_keys:  Dynamic input tensor of string \n
*@li feature_list_dense_keys:  Dynamic input tensor of string \n
*@li context_dense_defaults:  Dynamic input tensor of string, float, int64 \n
*@li debug_name: A Tensor of type string. \n

*@par Attributes:
*@li Ncontext_sparse: Number of context_sparse_keys, context_sparse_indices and context_sparse_shapes \n
*@li Ncontext_dense: Number of context_dense_keys \n
*@li Nfeature_list_sparse: Number of feature_list_sparse_keys \n
*@li Nfeature_list_dense: Number of feature_list_dense_keys \n
*@li context_sparse_types: Types of context_sparse_values \n
*@li Tcontext_dense: Number of dense_keys \n
*@li feature_list_dense_types: Types of feature_list_dense_values \n
*@li context_dense_shapes: Shape of context_dense \n
*@li feature_list_sparse_types: Type of feature_list_sparse_values \n
*@li feature_list_dense_shapes: Shape of feature_list_dense \n

*@par Outputs:
*@li context_sparse_indices: Dynamic output tensor of type int64. \n
*@li context_sparse_values:  Dynamic output tensor of type string, float, int64. \n
*@li context_sparse_shapes: Dynamic output tensor of type int64 \n
*@li context_dense_values:  Dynamic output tensor of type string, float, int64. \n
*@li feature_list_sparse_indices: Dynamic output tensor of type int64. \n
*@li feature_list_sparse_values:  Dynamic output tensor of type string, float, int64. \n
*@li feature_list_sparse_shapes: Dynamic output tensor of type int64 \n
*@li feature_list_dense_values:  Dynamic output tensor of type string, float, int64. \n
*@par Third-party framework compatibility \n
*@li compatible with tensorflow StringToNumber operator. \n
*/
REG_OP(ParseSingleSequenceExample)
    .INPUT(serialized, TensorType({DT_STRING}))
    .INPUT(feature_list_dense_missing_assumed_empty, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(context_sparse_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(context_dense_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(feature_list_sparse_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(feature_list_dense_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(context_dense_defaults, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .INPUT(debug_name, TensorType({DT_STRING}))
    .DYNAMIC_OUTPUT(context_sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(context_sparse_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(context_sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(context_dense_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(feature_list_sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(feature_list_sparse_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(feature_list_sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(feature_list_dense_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .ATTR(Ncontext_sparse, Int, 0)
    .ATTR(Ncontext_dense, Int, 0)
    .ATTR(Nfeature_list_sparse, Int, 0)
    .ATTR(Nfeature_list_dense, Int, 0)
    .ATTR(context_sparse_types, ListType, {})
    .ATTR(Tcontext_dense, ListType, {})
    .ATTR(feature_list_dense_types, ListType, {})
    .ATTR(context_dense_shapes, ListListInt, {})
    .ATTR(feature_list_sparse_types, ListType, {})
    .ATTR(feature_list_dense_shapes, ListListInt, {})
    .OP_END_FACTORY_REG(ParseSingleSequenceExample)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_
