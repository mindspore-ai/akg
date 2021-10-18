/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef INCLUDE_AKG_IR_PASS_H_
#define INCLUDE_AKG_IR_PASS_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include "tvm.h"
#include "pass/rewrite_simplify_cce.h"

namespace akg {
namespace ir {
/*!
 * \brief Simplify just the combiner of the given reduce node.
 *
 *  This function applies Simplify to the components of the top reduction's
 *  combiner, but not to the source or condition of the reduction.
 *  By default it also removes all components which are not used to
 *  compute the resulting value (the value_index-th value).
 *
 * \param expr The expression to be simplifed. Must be a reduce expression.
 * \param prune_unused_components Whether to remove components which are not really used.
 * \return Simplified expression.
 */
Expr SimplifyCombiner(const Expr &expr, bool prune_unused_components = true);

/*!
 * \brief replace point word separators in variables with underscore.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt ReplaceSeparator(Stmt stmt);

/*!
 * \brief rewrite tensor.value[0] to tensor_v0.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt RewriteMultiValueFunc(Stmt stmt);

/*!
 * \brief Rename the attr in LocalUB.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt RenameRealize(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer, const Map<Tensor, Tensor> &replace);

Array<NodeRef> AutoPoly(const Stmt &body, const Map<Tensor, Buffer> &extern_buffer, std::string target,
                        const bool is_dynamic, const Map<std::string, NodeRef> &spec_gemm_attrs = {},
                        Schedule sch = Schedule());

NodeRef GenTuningSpace(const Stmt &body, std::string target, const Map<Tensor, Buffer> &extern_buffer,
                       const Map<std::string, NodeRef> &spec_gemm_attrs = {}, Schedule sch = Schedule());

Expr CastNormalize(const Expr &expr, const air::DataType cast_type);

Stmt InjectDoubleBufferScopeOnGpu(Stmt stmt);
/*!
 * \brief  Define the scope of data prefetch using transfer buffer and the scope of using thread group, then insert
 * corresponding attributes. Detect if there exists data promotion to shared memory and calculate local memory resource
 * and thread resource to decide if enable data prefetch and thread group.
 *
 * \param expr The statement to be transformed.
 * \return The statement after transformed.
 */
Stmt InjectTransferBufferScope(Stmt stmt);

/*!
 * \brief  Rearrange the buffer of shared memory to eliminate the bank conflict.
 *
 * \param expr The statement to be rearranged.
 * \return The statement after rearranged.
 */
Stmt ReconstructLayout(const Stmt &stmt);

Stmt ElementwiseFlatten(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer,
                        const Map<Tensor, Buffer> &new_extern_buffer);

Array<NodeRef> FuseAxis(Stmt stmt, const Array<NodeRef> &arg_list, const Map<Tensor, Buffer> &extern_buffer);

Expr CastNormalize(const Expr &expr, const air::DataType cast_type);

Stmt TestInferBoundWithCond(const Expr &expr, const Array<Expr> &constraints);

Stmt TestReduceInequality(const air::Expr &e, const Var &reduce_var, bool scale, bool getlarger);

Stmt TestSimplify(const Expr &expr);

Stmt TestCanProveWithPosParam(const air::Expr &e);

Stmt RemoveFakeOp(const Stmt &stmt);

Stmt AtomicAddHoist(Stmt stmt);

Stmt AtomicAddClean(Stmt stmt);

Stmt ToMLIR(Stmt stmt);

Stmt AlgebraSimplify(const Stmt &stmt);

Array<NodeRef> CastKernelParams(const Stmt &stmt, const Array<NodeRef> &arg_list);

Stmt CheckShapeParams(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Array<NodeRef> CollectExternalCall(const Stmt &stmt);

Stmt ConvertDivModToShift(const Stmt &stmt);

Stmt ConvertExtentToCond(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt ConvertIfToSelect(Stmt stmt);

Stmt CopyPropagation(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt FixBindBuffer(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt FixLoopExtent(Stmt stmt);

Stmt FixRealizeShape(Stmt stmt);

Stmt ForEliminate(Stmt stmt);

Stmt IsolateLoops(const Stmt &stmt, bool enable_isolate_min_max);

Stmt PromoteCommonExpr(const Stmt &stmt);

Stmt PromoteConstExpr(const Stmt &stmt);

/*!
 * \brief Promote IF stmt as possible.
 * \param stmt The stmt to be transformed.
 * \return Transformed stmt.
 */
Stmt PromoteIfStmt(Stmt stmt, bool is_dynamic = false);

Stmt PromoteLetStmt(const Stmt &stmt, const Array<NodeRef> &arg_list);

Stmt RemoveAssert(const Stmt &stmt);

Stmt RewriteFloorDiv(const Stmt &stmt);

Stmt HalfReduceSumRewrite(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt ScalarComputeRewrite(const Stmt &stmt);

Stmt AddAttrForLayoutOp(Stmt stmt);

Stmt RewriteTensorIndex(Stmt stmt);

Stmt RewriteVarTensorIdx(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

/*!
 * \brief Sink IF stmt as possible.
 * \param stmt The stmt to be transformed.
 * \return Transformed stmt.
 */
Stmt SinkIfStmt(const Stmt &stmt);

Stmt SpecialValueReplacer(Stmt stmt);

Stmt SpecifyMinMaxDataType(Stmt stmt);

/*!
 * \brief Use pattern match for simple statement rewrite
 */
Stmt StmtPatternRewrite(Stmt stmt);

Stmt SubstituteDivVar(Stmt stmt);

/*!
 * Unrolls the loops if the extent is non-constant.
 */
Stmt UnrollNonConstantExtent(Stmt s);

Stmt ValueNumbering(Stmt stmt);

Stmt TensorAccessRewrite(const Stmt stmt);

Stmt SwizzleGPU(const Stmt &stmt, const Map<std::string, NodeRef> &attrs);

Stmt AlignLastAxisLoopExtent(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt AlignPartitionCCE(Stmt stmt);

Stmt UnifyAllocate(const Stmt &stmt);

Stmt EliminateAtomicDma(Stmt stmt);

Stmt AutoReorder(Stmt stmt);

Stmt BypassL1(const Stmt &stmt);

Stmt ExpandC0(Stmt stmt);

Stmt CastFilter(const Stmt &stmt);

Stmt ConvertCondToExtent(Stmt stmt);

Stmt StmtCSE(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt DeadCodeElim(Stmt stmt);

Stmt ExprPatternRewrite(Stmt stmt);

Stmt FeatureLibTransform(Stmt stmt);

Stmt GatherLoopInfo(Stmt stmt);

Stmt EliminateIf(Stmt stmt);

Stmt InjectAttr(Stmt stmt);

Stmt LoopSwitchHoist(Stmt stmt, bool hoist_allocate = false);

Stmt LowerWith(Stmt stmt);

Stmt MathIntrinRewrite(Stmt stmt);

Stmt PostFusion(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer, bool is_dynamic);

Stmt PostProcessImg2col(Stmt stmt);

Stmt ModDivEliminate(Stmt stmt);

Stmt RealizeCompress(Stmt stmt);

Stmt ReduceFusionOpt(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt SinkAllocate(const Stmt &stmt);

Stmt StrideKernelOp(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer, bool is_dynamic = false);

/*!
 * \brief Split complicated expression to three address code and do instruction selection
 *        + reuse_variable = True: will try to minimize the newly generated variables
 *        + minimum_split - use with "reuse_variable" to reuse newly generated tensors when exceeding this threshold
 */
Stmt ToThreeAddress(Stmt stmt, bool reuse_variable = false, int minimum_split = 10, bool cross_stmt_simplify = false);

Stmt UnifyLoopVars(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer, const Array<NodeRef> &arg_list);

Stmt TileCoverCorrect(Stmt stmt);
}  // namespace ir
}  // namespace akg
#endif  // INCLUDE_AKG_IR_PASS_H_
