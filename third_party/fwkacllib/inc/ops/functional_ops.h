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
 * \file functional_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUNCTIONAL_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUNCTIONAL_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
 *@brief Select one of the subgraphs to pass the input tensors and return the output tensors.
 *       If "cond" means True, the selected subgraph is "then_branch".
 *       Otherwise, the selected subgraph is "else_branch" . \n

 *@par Inputs:
 *@li cond: A Tensor. If "cond" is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if "cond" is a numerical scalar, non-zero means True and zero means False;
 *          if "cond" is a string scalar, non-empty means True and empty means False;
 *          if "cond" is not a scalar, non-empty means True and empty means False.
 *@li input: The input tensors . It's a dynamic input. \n

 *@par Graphs:
 *@li then_branch: A subgraph takes 'input' and returns a list of tensors,
 *                 whose types are the same as what else_branch returns.
 *@li else_branch: A subgraph takes 'input' and returns a list of tensors,
 *                 whose types are the same as what then_branch returns . \n

 *@par Outputs:
 *output: The output tensors returned by either then_branch(input) or else_branch(input) . \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator _If.
 */
REG_OP(_If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(_If)

/**
 *@brief Select one of the subgraphs to pass the input tensors and return the output tensors.
 *       If "cond" means True, the selected subgraph is "then_branch".
 *       Otherwise, the selected subgraph is "else_branch" . \n

 *@par Inputs:
 *@li cond: A Tensor. If "cond" is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if "cond" is a numerical scalar, non-zero means True and zero means False;
 *          if "cond" is a string scalar, non-empty means True and empty means False;
 *          if "cond" is not a scalar, non-empty means True and empty means False.
 *@li input: The input tensors . It's a dynamic input. \n

 *@par Graphs:
 *@li then_branch: A subgraph takes 'input' and returns a list of tensors,
 *                 whose types are the same as what else_branch returns.
 *@li else_branch: A subgraph takes 'input' and returns a list of tensors,
 *                 whose types are the same as what then_branch returns . \n

 *@par Outputs:
 *output: The output tensors returned by either then_branch(input) or else_branch(input) . \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator StatelessIf.
 */
REG_OP(StatelessIf)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(StatelessIf)

/**
 *@brief Select one of the subgraphs to pass the input tensors and return the output tensors.
 *       If "cond" means True, the selected subgraph is "then_branch".
 *       Otherwise, the selected subgraph is "else_branch" . \n

 *@par Inputs:
 *@li cond: A Tensor. If "cond" is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if "cond" is a numerical scalar, non-zero means True and zero means False;
 *          if "cond" is a string scalar, non-empty means True and empty means False;
 *          if "cond" is not a scalar, non-empty means True and empty means False.
 *@li input: The input tensors . It's a dynamic input. \n

 *@par Graphs:
 *@li then_branch: A subgraph takes 'input' and returns a list of tensors,
 *                 whose types are the same as what else_branch returns.
 *@li else_branch: A subgraph takes 'input' and returns a list of tensors,
 *                 whose types are the same as what then_branch returns . \n

 *@par Outputs:
 *output: The output tensors returned by either then_branch(input) or else_branch(input) . \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator If.
 */
REG_OP(If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(If)

/**
 *@brief Select one of the subgraphs to pass the input tensors and return the output tensors . \n

 *@par Inputs:
 *@li branch_index: A int32 scalar which determines the selected subgraph.
 *@li input: The input tensors, which will be passed to the subgraph . It's a dynamic input. \n

 *@par Graphs:
 *branches: A list of subgraphs, each of which takes 'input' and returns a list of tensors,
 *          whose types are the same as what every other subgraph returns . \n

 *@par Outputs:
 *output: The output tensors returned by one of branches . It's a dynamic output. \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator Case.
 */
REG_OP(Case)
    .INPUT(branch_index, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .DYNAMIC_GRAPH(branches)
    .OP_END_FACTORY_REG(Case)

/**
 *@brief Cyclic execute the "body" subgraph until the return tensor of "cond" subgraph means False . \n

 *@par Inputs:
 *input: The input tensors . It's a dynamic input. \n

 *@par Graphs:
 *@li cond: A subgraph takes 'input' and returns a tensor.
 *          If the tensor is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if it is a numerical scalar, non-zero means True and zero means False;
 *          if it is a string scalar, non-empty means True and empty means False;
 *          if it is not a scalar, non-empty means True and empty means False.
 *@li body: A subgraph takes 'input' and returns a another list of tensors .  \n

 *@par Attributes:
 *parallel_iterations: An optional int, default as 10 . \n

 *@par Outputs:
 *output: The output tensors returned by "body". Has the same type as "input" . \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator _While.
 */
REG_OP(_While)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(cond)
    .GRAPH(body)
    .OP_END_FACTORY_REG(_While)

/**
 *@brief Cyclic execute the "body" subgraph until the return tensor of "cond" subgraph means False . \n

 *@par Inputs:
 *input: The input tensors . It's a dynamic input. \n

 *@par Graphs:
 *@li cond: A subgraph takes 'input' and returns a tensor.
 *          If the tensor is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if it is a numerical scalar, non-zero means True and zero means False;
 *          if it is a string scalar, non-empty means True and empty means False;
 *          if it is not a scalar, non-empty means True and empty means False.
 *@li body: A subgraph takes 'input' and returns a another list of tensors . \n

 *@par Attributes:
 *parallel_iterations: An optional int, default as 10 . \n

 *@par Outputs:
 *output: The output tensors returned by "body". Has the same type as "input" . It's a dynamic output. \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator While.
 */
REG_OP(While)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(cond)
    .GRAPH(body)
    .ATTR(parallel_iterations, Int, 10)
    .OP_END_FACTORY_REG(While)

/**
 *@brief Cyclic execute the "body" subgraph until the return tensor of "cond" subgraph means False . \n

 *@par Inputs:
 *input: The input tensors . It's a dynamic input. \n

 *@par Graphs:
 *@li cond: A subgraph takes 'input' and returns a tensor.
 *          If the tensor is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if it is a numerical scalar, non-zero means True and zero means False;
 *          if it is a string scalar, non-empty means True and empty means False;
 *          if it is not a scalar, non-empty means True and empty means False.
 *@li body: A subgraph takes 'input' and returns a another list of tensors . \n

 *@par Attributes:
 *parallel_iterations: An optional int, default as 10 . \n

 *@par Outputs:
 *output: The output tensors returned by "body". Has the same type as "input" . It's a dynamic output. \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator StatelessWhile.
 */
REG_OP(StatelessWhile)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(cond)
    .GRAPH(body)
    .ATTR(parallel_iterations, Int, 10)
    .OP_END_FACTORY_REG(StatelessWhile)

/**
 *@brief Cyclic execute the "body" subgraph until the first input of For op exceed upper bound . \n

 *@par Inputs:
 *@li start: A int32 scalar. The lower bound.
 *@li limit: A int32 scalar. The upper bound.
 *@li delta: A int32 scalar. The step size.
 *@li input: The input tensors, which will be passed to "body" . It's a dynamic input. \n

 *@par Graphs:
 *body: A subgraph takes 'input' and returns a another list of tensors . \n

 *@par Outputs:
 *output: The output tensors returned by "body". Has the same type as "input" . It's a dynamic output. \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator For.
 */
REG_OP(For)
    .INPUT(start, DT_INT32)
    .INPUT(limit, DT_INT32)
    .INPUT(delta, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(body)
    .OP_END_FACTORY_REG(For)

/**
 *@brief Pass the input tensors to the subgraph "f" and return the output tensors . \n

 *@par Inputs:
 *args: The input tensors, which will be passed to "f" . It's a dynamic input. \n

 *@par Graphs:
 *f: A subgraph takes 'args' and returns a another list of tensors . \n

 *@par Attributes:
 *@li config: An optional string, default as "".
 *@li config_proto: An optional int, default as "".
 *@li executor_type: An optional int, default as "" . \n

 *@par Outputs:
 *output: The output tensors returned by "f" . It's a dynamic output. \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator PartitionedCall.
 */
REG_OP(PartitionedCall)
    .DYNAMIC_INPUT(args, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(f)
    .ATTR(config, String, "")
    .ATTR(config_proto, String, "")
    .ATTR(executor_type, String, "")
    .OP_END_FACTORY_REG(PartitionedCall)

/**
 *@brief Pass the input tensors to the subgraph "f" and return the output tensors . \n

 *@par Inputs:
 *args: The input tensors, which will be passed to "f" . It's a dynamic input. \n

 *@par Graphs:
 *f: A subgraph takes 'args' and returns a another list of tensors . \n

 *@par Attributes:
 *@li config: An optional string, default as "".
 *@li config_proto: An optional int, default as "".
 *@li executor_type: An optional int, default as "" . \n

 *@par Outputs:
 *output: The output tensors returned by "f" . It's a dynamic output. \n

 *@par Third-party framework compatibility
 *@Compatible with the TensorFlow operator StatefulPartitionedCall.
 */
REG_OP(StatefulPartitionedCall)
    .DYNAMIC_INPUT(args, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(f)
    .ATTR(config, String, "")
    .ATTR(config_proto, String, "")
    .ATTR(executor_type, String, "")
    .OP_END_FACTORY_REG(StatefulPartitionedCall)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUNCTIONAL_OPS_H_
