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

#ifndef PASS_AUTODIFF_CCE_H_
#define PASS_AUTODIFF_CCE_H_

#include <topi/reduction.h>
#include "codegen/util.h"

namespace akg {
namespace ir {
constexpr int AD_GROUP_CONV_DEFAULT_cutK = 48;
constexpr int AD_PASS_MIN_DEPENDANTS_TO_ISOLATE = 2;
constexpr int AD_PASS_MIN_DISTANCE_TO_ISOLATE = 2;

enum ADConvType { UNKNOWN = -1, NORMAL, GROUP, DEPTHWISE };

/**
 * This class is used to apply simplifications required exclusievley by the Autodiff module.
 * The simplifications included in this class wouldn't be included in Simplify or Simplify_cce
 * */
class AutodiffSimplify : public IRMutator {
 public:
  template <typename T1>
  Expr SimplifyMaxMin(Expr lhs, Expr rhs, const Expr &e);
  Range GetRangeOfVar(const std::string &var_name);
  std::unordered_set<const Variable *> GetExprVars(const Expr &expr) const;

 private:
  Expr Mutate_(const Mod *op, const Expr &e) final;
  Expr Mutate_(const EQ *op, const Expr &e) final;
  Map<Var, Range> cond_ranges_;
};

// This class is used to transform mad as add
class SimplifyMad : public IRMutator {
 public:
  SimplifyMad() = default;
  ~SimplifyMad() override = default;

 private:
  Expr Mutate_(const Call *op, const Expr &e) final;
};

/*!
 * \brief Check if the given expr is a const of any type equal to the given integer value.
 * \param e The expression.
 * \return Whether the expression is a const equal to the value.
 * \tparam ValueType The value type
 */
inline bool is_const_ad(const Expr &e) {
  if (e.as<air::ir::IntImm>()) {
    return true;
  } else if (e.as<air::ir::UIntImm>()) {
    return true;
  } else if (e.as<air::ir::FloatImm>()) {
    return true;
  } else if (const auto c = e.as<air::ir::Cast>()) {
    return is_const_ad(c->value);
  } else if (const auto b = e.as<air::ir::Broadcast>()) {
    return is_const_ad(b->value);
  } else {
    return false;
  }
}

/*!
 * \brief The function check convolution type of a given tensor
 *
 * \param tensor: the tensor to be checked
 *
 * \return The enum value of convolution type
 */
ADConvType ConvType(const Tensor &tensor);

/*!
 * \brief The function check if a given tensor is a Conv op or not. If it's a Conv op, then it check in the given name
 * matches the name of data input or the name of weight input.
 *
 * \param tensor: the tensor to be checked if it is a Conv op
 * \param is_name_data: set to True if tensor is a Conv op AND name is the name of data input of the Conv
 * \param is_name_weight: set to True if tensor is a Conv op AND name is the name of the weight (filter/kernel) input of
 * the Conv
 * \param name: the name to be checked if matches the data or weight input name of a Conv op
 *
 * \return The logic value false if tensor is not a Conv op (in this case both is_name_data and is_name_weight are false
 * too) The logic value true if tensor is a Conv op
 */
bool IsReduceConv(const Tensor &tensor, bool &is_name_data, bool &is_name_weight, const std::string &name = "");

/*!
 * \brief Creates the backward computation of a Convolution wrt [Data OR Weight].
 *
 *  This function evaluates the value of an expression \p expr with the boundaries
 *  of the Range of its (single) Variable. If the expression is True for both cases,
 *  this function returns True.
 *
 *  If \p expr has more than one Variable, this function return false.
 *
 * \param output The Tensor compute of the requested Backward kernel.
 * \param input The Tensor of the Data input to the Convolution.
 * \param head The Tensor of the Input gradient (Gradient of the Loss wrt output of the Convolution).
 * \param attrs The additional set of attributes used for the AD algorithms
 * \param new_pld_array Array of Tensors for the placeholders.
 *
 * \return A Tensor compute with the backward kernel if it could be generated, an empty Tensor otherwise.
 */
Tensor DiffConv(const Tensor &output, const Tensor &input, const Tensor &head, const Map<std::string, NodeRef> &attrs,
                const Array<Tensor> &new_pld_array);

/*!
 * \brief Create mmad computation node
 *
 * \param source: input expr
 * \param rdom: space of iter vars
 *
 * \return created mmad node
 */
Expr Mmad(Expr source, const Array<IterVar> &rdom);

/*!
 * \brief check if has mad Call
 *
 * \param output input tensor
 * \return true if input tensor is mad call, otherwise returns false.
 */
bool HasMad(const Tensor &output);

/*!
 * \brief A generalization of matrix multiplication to tensors.
 *
 * \param A The tensor A
 * \param B The tensor B
 * \param axes The number of the dimensions to reduce over
 * \param name The name of the operation
 * \param has_mad, has mad Call or not
 *
 * \return A Tensor computing the result
 */
Tensor TensorDot(const Tensor &A, const Tensor &B, int axes, std::string name, bool has_mad);

// Functions for optimization passes for AD

/*!
 * \brief Comparing two tensors based on a given set of equal tensors.
 * \param left_tensor The first tensor to compare.
 * \param right_tensor The second tensor to compare.
 * \param equal_set  The set containing equal tensors.
 * \return True if both left_tensor and right_tensor belong to the equal_set.
 */
bool CompareTensors(const Tensor &left_tensor, const Tensor &right_tensor, const std::unordered_set<Tensor> &equal_set);

/*!
 * \brief Comparing two tensors based on a number of given sets of equal tensors (a vector of sets).
 * \param left_tensor The first tensor to compare.
 * \param right_tensor The second tensor to compare.
 * \param all_equal_sets  The vector containing sets of equal tensors.
 * \return True if both left_tensor and right_tensor belong to one of the equal_sets given.
 */
bool CompareTensors(const Tensor &left_tensor, const Tensor &right_tensor,
                    const std::vector<std::unordered_set<Tensor>> &all_equal_sets);

/*!
 * \brief `Deep` comparing two tensors based on their parameters and computations.
 * \param left_tensor The first tensor to compare.
 * \param right_tensor The second tensor to compare.
 * \return True if both left_tensor and right_tensor have the same set of parameters.
 */
bool DeepCompareTensorsNodes(const Tensor &left_tensor, const Tensor &right_tensor);

/*!
 * \brief `Deep` comparing two tensors based on their parameters and computations and their input
 * tensors.
 *
 * \param left_tensor The first tensor to compare.
 *
 * \param right_tensor The second tensor to compare.
 *
 * \return True if both left_tensor and right_tensor have the same set of parameters and
 * recurrently for their input tensors, For commutative operators, "A op B" is equal "B op A" too.
 */
bool DeepCompareTensors(const Tensor &left_tensor, const Tensor &right_tensor);

/*!
 * \brief `Shallow` comparing two tensors based on their parameters and computations and their input
 * tensors.
 *
 * \param left_tensor The first tensor to compare.
 *
 * \param right_tensor The second tensor to compare.
 *
 * \return True if both left_tensor and right_tensor have the same set of parameters. This
 * function doesn't recurrently check the equalities of the input tensors of the given tensors but
 * only check from known sets of equal tensors. For commutative operators, "A op B" is equal "B op
 * A" too.
 */
bool ShallowCompareTensors(const Tensor &left_tensor, const Tensor &right_tensor,
                           std::vector<std::unordered_set<Tensor>> &equal_tensors_at_distance);

/*!
 * \brief Walk recursively the given tensor, collect and group subtensors by their "distance" to
 * placeholder inputs.
 *
 * \param tensor: The input tensor.
 *
 * \param distances (returned values): the set of distances from the input tensor to the
 * placeholders used in the computation.
 *
 * \param all_tensors_at_distance: A map of grouped subtensors by their distances to placeholders
 * (can have repetitions).
 */
void CollectAllTensorsByDistance(const Tensor &tensor, std::unordered_set<int> &distances,
                                 std::unordered_map<int, std::unordered_set<Tensor>> &all_tensors_at_distance);

/*!
 * \brief Walk recursively the computation tree of the given tensor, collect the map of all
 * subtensors and their reverse dependents.
 *
 * \param input_tensors: The array of input tensors.
 *
 * \param reverse_dependencies (returned values): the map of subtensors and their reverse dependent.
 */
void CollectReverseDependencies(const Array<Tensor> &input_tensors,
                                std::unordered_map<Tensor, std::unordered_set<Tensor>> &reverse_dependencies);

/*!
 * \brief Walk depth-first recursively the given tensor, collect subtensors and map them with their
 * orders of appearance in the walk.
 *
 * \param root: The input tensor.
 *
 * \param dfs_order (returned values): the map of subtensors and their orders of appearance.
 */
void CollectDFSOrder(const Tensor &root, std::unordered_map<Tensor, int> &dfs_order);

/*!
 * \brief Find all sets of equal tensors, grouped by their "distance" to placeholder inputs.
 *
 * \param all_tensors_at_distance: A map of grouped subtensors by their distances to
 * placeholders (can have repetitions).
 *
 * \param equal_tensors (returned values): the vector containing possible sets of equal tensors,
 * grouped by their "distance" to placeholder inputs
 */
void FindEqualTensorsByDistance(const std::unordered_map<int, std::unordered_set<Tensor>> &all_tensors_at_distance,
                                std::unordered_map<int, std::vector<std::unordered_set<Tensor>>> &equal_tensors);

/*!
 * \brief Recursively visit a computation tree and replace all the tensors' operations listed in a map.
 *
 * \param tensor: the "root" tensor of the computation tree.
 *
 * \param replace_map: the map of tensors' operations to-be-replaced paired with new operations.
 *
 * \param visited: set of the tensor nodes visited during the recurrent tree walking.
 *
 * \param new_operators_marked: set of tensors' operator marked (as replaced) to avoid double replacing.
 *
 * \return The (new) operation for the root tensor.
 */
Operation ReplaceTensorRecursively(const Tensor &tensor, std::unordered_map<Operation, Operation> &replace_map,
                                   std::unordered_set<Tensor> &visited,
                                   std::unordered_set<Operation> &new_operators_marked);

/*!
 * \brief Replace all the tensor listed in a map for an array of computation trees (used for cases of multiple
 * outputs).
 *
 * \param input_tensors: the "root" tensors of the computation trees.
 *
 * \param output_tensors (returned): the new "root" tensors.
 *
 * \param replace_map: the map of tensors to-be-replaced paired with new tensors.
 */
void ADPassReplaceArrayTensorsUsingOperations(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors,
                                              std::unordered_map<Tensor, Tensor> &replace_map);

/*!
 * \brief Replace all the tensor listed in a map for a computation trees.
 *
 * \param root: the "root" tensor of the computation tree.
 *
 * \param matching_array: the array of tensors to-be-replaced interleaved with new tensors, i.e. array containing
 * [tensor1, new_tensor1, tensor2, new_tensor2,...].
 */
Tensor ADPassReplaceExternalTensorsUsingOperations(const Tensor &root, const Array<Tensor> &matching_array);

/*!
 * \brief For all tensors from the array input_tensors, automatically find the equivalent
 * nodes and merge them into a single node.
 *
 * \param input_tensors: the array of input tensors.
 *
 * \param output_tensors (returned): the array of new tensors after merging.
 */
void ADPassMergeInternalArrayTensors(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors);

/*!
 * \brief Check if a tensor is a broadcasting computation.
 *
 * \param tensor: the input tensor.
 *
 * \return `True` if the tensor is a broadcasting computation (limit to the cases where output shape has the same
 * length as input shapes, and there is at least one dimension, in which input shape is 1 and output shape is >1).
 */
bool IsBroadcast(const Tensor &tensor);

/*!
 * \brief Check if a tensor is a broadcasting computation at (only) specific dimensions.
 *
 * \param tensor: the input tensor.
 *
 * \param axes: the vector of indices of broadcasted dimensions.
 *
 * \return `True` if the tensor is a broadcasting computation (limit to the cases where output shape has the same
 * length as input shapes) and at exactly the dimensions given in `axes`.
 */
bool IsBroadcastAt(const Tensor &tensor, const std::vector<size_t> &axes);

/*!
 * \brief Check if two tensors are broadcasting computations at the same dimensions.
 *
 * \param left_tensor: the 1st input tensor to be compared.
 *
 * \param right_tensor: the 2nd input tensor to be compared.
 *
 * \return `True` if the two tensors are broadcasting computations (limit to the cases where output shape has the same
 * length as input shapes) and at the same dimensions.
 */
bool IsSameBroadcast(const Tensor &left_tensor, const Tensor &right_tensor);

/*!
 * \brief Check if a tensor is a Reduce computation.
 *
 * \param tensor: the input tensor.
 *
 * \param reduction_axes (returned): the vector of indices of reduction axes.
 *
 * \return `True` if the given tensor is a Reduce computation (limit to the cases where keep_dims = True).
 */
bool IsReduceSum(const Tensor &root, std::vector<size_t> &reduction_axes);

/*!
 * \brief Check if a tensor match the computing pattern reduce(A * (B * broadcast(C))).
 *
 * \param tensor: the input tensor.
 *
 * \param result: the array containing tensors A, B, C and the new tensor of reduce(A * B) * C.
 */
bool IsReducePattern_1(const Tensor &root, std::vector<Tensor> &result);

/*!
 * \brief Check if a tensor match the computing pattern reduce((A * broadcast(B)) * broadcast(C)).
 *
 * \param tensor: the input tensor.
 *
 * \param result: the array containing tensors A, B, C and the new tensor of reduce(A) * B * C.
 */
bool IsReducePattern_2(const Tensor &root, std::vector<Tensor> &result);

/*!
 * \brief Check if a tensor match the computing pattern reduce((A * broadcast(B)) * C).
 *
 * \param tensor: the input tensor.
 *
 * \param result: the array containing tensors A, B, C and the new tensor of reduce(A * C) * B.
 */
bool IsReducePattern_3(const Tensor &root, std::vector<Tensor> &result);

/*!
 * \brief Check in the computation tree for Reduce op matching with the predefined patterns and map them
 * to the new, optimized computations
 *
 * \param tensor: the input tensor.
 *
 * \param result: the array containing Reduce tensors matched with one of the predefined patterns.
 *
 * \param map_new_reduce (returned): the map of found Reduce tensors paired with their new, optimized
 * computation tensor
 */
void CollectAllReduce(const Tensor &tensor, std::vector<Tensor> &result,
                      std::unordered_map<Tensor, Tensor> &map_new_reduce);

/*!
 * \brief For all tensors from array input_tensors, automatically find the Reduce nodes that matched one of
 * the predefined patterns and replace them with new, optimized computations.
 *
 * \param input_tensors: the array containing input tensors.
 *
 * \param output_tensors (returned): The new tensors with matched Reduce nodes optimized.
 */
void ADPassReduceBroadcastSimplify(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors);

/*!
 * \brief For a computation tensor, automatically check if there are nodes of form (A * const_1) * (B * const_2)
 * an replace them with (A * B) * const_3 where const_3 = const_1 * const_2.
 *
 * \param tensor: the input tensor.
 *
 * \param new_const (returned): the value of const_3
 *
 * \param map_new_muls: a map of tensors paired with a pair (Tensor, const_fl) such that the key tensor
 * equal the product of the value Tensor and the value const_fl.
 */
void PullConstFromMul(const Tensor &tensor, float &new_const,
                      std::unordered_map<Tensor, std::pair<Tensor, float>> &map_new_muls);

/*!
 * \brief For all tensors from array input_tensors, automatically find the chains of multiplication operations
 * with constants and replace them with new, optimized computations with only one constant multiplication.
 *
 * \param input_tensors: the array containing input tensors.
 *
 * \param output_tensors (returned): The new tensors with constant multiplications simplified.
 */
void ADPassSimplifyConstMultiply(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors);

/*!
 * \brief For all tensors from array input_tensors, automatically find the chains of multiplication operations,
 * which can be swept to have the multiplication of the same pair of tensors as other nodes.
 *
 * \param input_tensors: the array containing input tensors.
 *
 * \param output_tensors (returned): The new tensors with multiplication orders swept for reducing intermediate nodes.
 */
void ADPassSwapMultiplyOrder(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors);

/*!
 * \brief For all tensors from array input_tensors, automatically find the multiplication of two broadcasted tensors
 * and replace it with a broadcasted multiplication of two tensors.
 *
 * \param input_tensors: the array containing input tensors.
 *
 * \param output_tensors (returned): The new tensors with multiplication of broadcasted tensors replaced.
 */
void ADPassMergeMultipleBroadcast(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors);

/*!
 * \brief Clone a tensor and set additional attribute "no_inline" = 1 in order to isolate this tensor during code-gen.
 *
 * \param tensor: the input tensor.
 *
 * \return The new tensor cloned from input tensor with an additional attribute "no_inline" = 1.
 */
Tensor IsolateTensor(const Tensor &tensor);

/*!
 * \brief For all tensors from array input_tensors, automatically find the common nodes for at least two nodes and
 * isolate them to avoid the repetition in code-gen.
 *
 * \param input_tensors: the array containing input tensors.
 *
 * \param output_tensors (returned): The new tensors with some internal nodes isolated.
 */
void ADPassIsolateTensors(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors);

/*!
 * \brief The function to call all the available optimizing passes in AD for a set of input tensors.
 *
 * \param input_tensors: the input tensors to be optimized.
 *
 * \param output_tensors (returned): the new tensors after optimizations
 *
 * \param in_attrs: a map of attributes to be used in optimizing passes.
 *
 * \param new_pld_array: an array of tensors (placeholders) passed from external environment to be used in
 * the optimizing passes.
 *
 * \param DOT_prefix: a prefix used to generate DOT file for computation tree of tensors.
 */
void ADRunAllPasses(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors, AttrMap &in_attrs,
                    const Array<Tensor> &new_pld_array, const std::string &DOT_prefix);

/*!
 * \brief The main function to call all the available optimizing passes in AD.
 *
 * \param input_tensors: the input tensors to be optimized.
 *
 * \param output_tensors (returned): the new tensors after optimizations
 *
 * \param attrs: a map of attributes to be used in optimizing passes.
 *
 * \param new_pld_array: an array of tensors (placeholders) passed from external environment to be used in
 * the optimizing passes.
 */
void ADOptimizePasses(Array<Tensor> &input_tensors, Array<Tensor> &output_tensors,
                      const Map<std::string, NodeRef> &attrs, const Array<Tensor> &new_pld_array);

/*!
 * \brief Broadcasts a Tensor `t` to output_shape only where condition is True, otherwise
 *        it broadcasts else_value.
 * \param t The tensor to be broadcasted.
 * \param condition The condition to check to select if the values of `t` will be broadcasted
 *                  or the else_val instead.
 * \param iter_vars An Array of IterVar with the iteration variables of the new Tensor.
 * \param fw_axis An Array of Expr corresponding to the indexation of the Forward Compute.
 * \param else_val The value to set in the broadcasted tensor when the condition is False.
 * \param name Name of the broadcasted tensor.
 * \param tag A Tag for the Tensor.
 * \return A new tensor with the shape `output_shape` with the values of `t` when `condition`
 *         is True and with value `else_value` when the condition is False.
 */
Tensor BroadcastToCond(const Tensor &t, const Expr &condition, const Array<IterVar> &iter_vars,
                       const Array<IterVar> &fw_axis, const Expr &else_val, const std::string &name = "T_bcast_cond",
                       const std::string &tag = topi::kBroadcast);

/*!
 * \brief Duplicates an Array of IterVar appending a suffix to the var names.
 * \param axis The Array to be duplicated.
 * \param suffix An optional suffix to be appended to the newly created IterVars.
 * \return A new Array of IterVars
 */
Array<IterVar> DuplicateIterArray(const Array<IterVar> &axis, const std::string &suffix = "");

/*!
 * \brief Creates a Tensor from the first step of a Reduce.
 * \param red The Reduce object from where to extract the forward computation.
 *            Obtained by the Jacobian mutator.
 * \param op A compute op corresponding to the output being differentiated.
 * \param input The input Tensor wrt which the output is beinf differentiated.
 * \param output The output Tensor that is being differentiated.
 * \param source_call The Call object to the input within the `red` Reduce.
 * \return A new Tensor corresponding to the forward operation included in the Reduce.
 */
Tensor GetForwardCompute(const Reduce *red, const ComputeOpNode *op, const Tensor &input, const Tensor &output,
                         const Call *source_call);

/*!
 * \brief Creates a Tensor from the first step of a Reduce.
 * \param op A compute op corresponding to the output being differentiated.
 * \param input The input Tensor wrt which the output is being differentiated.
 * \param fw_bdc A broadcast tensor of the forward compute.
 * \param jac_axis An Array of IterVars of the Jacobian Tensor.
 * \param args An Array of Expr corresponding to the indexation of the Jacobian.
 * \param input_vars_eqsel An Array of Expr corresponding to the indexation of the input tensor.
 * \param fw_axis An Array of Expr corresponding to the indexation of the Forward Compute.
 * \param head The adjoint of the output, in other words, some tensor, by which the Jacobians
 *             will be multiplied. Its shape must be of the form `prefix + output.shape`. If the
 *             null pointer is provided, the identity tensor of shape
 *             `output.shape + output.shape` will be used.
 * \param used_head Indicates if the head was used while building the Jacobian.
 * \return A new Select Expr that finds when the input is equal to the broadcasted forward operation.
 */
Tensor BuildSelectFromBdcast(const ComputeOpNode *op, const Tensor &input, const Tensor &fw_bdc,
                             const Array<IterVar> &jac_axis, const Array<Expr> &args,
                             const Array<Expr> &input_vars_eqsel, const Array<IterVar> &fw_axis, const Tensor &head,
                             bool &used_head);

/*!
 * \brief Creates a Tensor by broadcasting `input` constrained to `condition`.
 * \param input The input Tensor wrt which the output is being differentiated.
 * \param condition The condition to use when creating the Select to decide to broadcast or not.
 * \param head The adjoint of the output, in other words, some tensor, by which the Jacobians
 *             will be multiplied. Its shape must be of the form `prefix + output.shape`. If the
 *             null pointer is provided, the identity tensor of shape
 *             `output.shape + output.shape` will be used.
 * \param used_head  if the head was used while building the Jacobian.
 * \param fw_axis An Array of IterVars of the Forward output Tensor.
 * \param jac_axis An Array of IterVars of the Jacobian Tensor.
 * \return A new Select Expr that finds when the input is equal to the broadcasted forward operation.
 */
Tensor BroadcastAndSelect(const Tensor &input, const Expr &condition, const Tensor &head, bool &used_head,
                          const Array<IterVar> &fw_axis, const Array<IterVar> &jac_axis);
/*!
 * \brief Optimizes the autodifferentiated compute of a Reduction operation.
 * \param tensor The initial AD tensor for output wrt input.
 * \param op The Compute node that's being autodifferentiated (output).
 * \param red The Reduce produced by the Jacobian function which needs to be optimized.
 * \param output The output Tensor that was autodifferentiated.
 * \param input The input Tensor which `output` was autodifferentiated wrt.
 * \param new_shape The shape of the Jacobian.
 * \param new_axis  The iterators for the Jacobian.
 * \param head The adjoint of the output, in other words, some tensor, by which the Jacobians
 *             will be multiplied. Its shape must be of the form `prefix + output.shape`. If the
 *             null pointer is provided, the identity tensor of shape
 *             `output.shape + output.shape` will be used.
 * \param used_head Boolean the indicates if the head was already used inside one of the
 *                  optimizations.
 * \return An optimized Tensor that doesn't have two combined reductions.
 *         If Head is given, it will set Head values in the corresponding locations to
 *         avoid the Head * Jacobian afterwards. If the Reduction is not optimizable, it'll return
 *         the original `tensor`.
 */
Tensor OptimizeReduction(Tensor &tensor, const ComputeOpNode *op, const Reduce *red, const Tensor &output,
                         const Tensor &input, const Array<Expr> &new_shape, const Array<IterVar> &new_axis,
                         const Tensor &head, bool &used_head);
}  // namespace ir
}  // namespace akg
#endif  // PASS_AUTODIFF_CCE_H_
