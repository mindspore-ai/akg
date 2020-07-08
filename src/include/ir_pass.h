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

#ifndef INCLUDE_AKG_IR_PASS_H_
#define INCLUDE_AKG_IR_PASS_H_

#include <tvm/expr.h>
#include <tvm/buffer.h>
#include <tvm/schedule.h>
#include <tvm/lowered_func.h>
#include <tvm.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

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
 * \brief Detect and insert sync points to d-processor.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt InjectSync(Stmt stmt);

/*!
 * \brief emit insn for D.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt EmitInsn(Stmt stmt, bool enable_bisect, bool enable_cover_protect, const Map<Tensor, Buffer> &extern_buffer,
              bool is_dynamic);

/*!
 * \brief emit insn debugger.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt EmitInsnDebug(Stmt stmt);

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

/*!
 * \brief auto inject pip info for D.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt InjectPipe(Stmt stmt);

/*!
 * \brief hoist insn for D.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */

Stmt HoistInsn(Stmt stmt);

/*!
 * \brief Inject tvm_access_ptr message buffer trasnform
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt InjectAccessPtrMSG(Stmt stmt);

/*!
 * \brief Remove tvm_access_ptr message buffer trasnform
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt RemoveAccessPtrMSG(Stmt stmt);

/*!
 * \brief Rewrite storage allocation pattern for CCE platform.
 *  Trying to share space between allocations to make
 *  a static allocation plan when possible.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt StorageRewriteCCE(Stmt stmt, const std::string &maxsat_filename, bool use_BC_opt = true, bool no_limits = false,
                       int maxsat_timeout = 4);

/*!
 * \brief Rewrite storage allocation pattern in Ubuf for
 *  CCE platform.
 *  Trying to share space between allocations to make
 *  a static allocation plan when possible.
 *  Also, try to minimize read-read and read-write bank
 *  conflicts for vector operations in Ubuf.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt StorageRewriteBC(Stmt stmt, const std::string &maxsat_filename, bool no_limits, int maxsat_timeout);

/*!
 * \brief Allow operators run on multi-core
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt InjectMultiCore(Stmt stmt, int max_block_dim, int merge_outer_loop = 1, bool is_dynamic = false,
                     bool scalar_rearrange = false);

Array<NodeRef> InjectMultiCoreVar(Stmt stmt, const Var &block_dim, int merge_outer_loop = 1);

Stmt AlignPartitionCCE(Stmt stmt);

Stmt FixMadAttrs(Stmt s);
/*!
 * Unrolls the loops if the extent is non-constant.
 */
Stmt UnrollNonConstantExtent(Stmt s);

/*!
 * \brief Compress tensor after autopoly.
 *
 * \param stmt The stmt to be transformed.
 * \return Transformed stmt.
 */
Stmt RealizeCompress(Stmt stmt);

/*!
 * \brief Normlize loop after autopoly.
 *
 * \param stmt The stmt to be transformed.
 * \return Transformed stmt.
 */
Stmt LoopNormlize(Stmt stmt);

/*!
 * \brief Rewrite expression to fit cce's intrinsics.
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt MathIntrinRewrite(Stmt stmt);

/*!
 * \brief Sink IF stmt as possible.
 * \param stmt The stmt to be transformed.
 * \return Transformed stmt.
 */
Stmt SinkIfStmt(const Stmt &stmt);

Array<NodeRef> AutoPoly(const Stmt &body, const Map<Tensor, Buffer> &extern_buffer,
                        const Map<std::string, NodeRef> &attrs, const bool is_specgemm, const bool is_dynamic);

/*!
 * \brief Promote IF stmt as possible.
 * \param stmt The stmt to be transformed.
 * \return Transformed stmt.
 */
Stmt PromoteIfStmt(Stmt stmt, bool is_dynamic = false);

Stmt PromoteLetStmt(const Stmt &stmt, const Array<NodeRef> &arg_list);

NodeRef GenTuningSpace(const Stmt &body, const Map<Tensor, Buffer> &extern_buffer,
                       const Map<std::string, NodeRef> &attrs, const bool is_specgemm);

Stmt Load3dTrans(Stmt stmt, bool is_dynamic);

Stmt PostFusion(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer, bool is_dynamic);

Stmt DmaFlatten(Stmt stmt, bool all_dynamic_conv);

Stmt ReduceFusionOpt(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt PostProcessImg2col(Stmt stmt);

/*!
 * \brief Lower the high-level cce_img2col intrinsic
 * to the D-intrinsics set_fmatrix and img2col_cbuf_to_ca.
 */
Stmt CoarsenImg2Col(Stmt stmt);

/*!
 * \brief direct copy filter from outer to L0, bypass L1
 *
 * \param stmt The stmt to be transformed.
 * \return Transformed stmt.
 */
Stmt BypassL1(const Stmt &stmt);

Stmt StrideKernelOp(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer, bool is_dynamic = false);

Stmt InvariantHoist(Stmt stmt);

Stmt ElimDMA(Stmt stmt);

Stmt InjectAttr(Stmt stmt);

Stmt UnifyLoopVars(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer, const Array<NodeRef> &arg_list);

Stmt IsolateLoops(const Stmt &stmt, bool enable_isolate_min_max);

Stmt CheckShapeParams(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt ForEliminate(Stmt stmt);

Stmt SubstituteDivVar(Stmt stmt);

Stmt FixLoopExtent(Stmt stmt);

Stmt FixRealizeShape(Stmt stmt);

Stmt EliminateIf(Stmt stmt);

Stmt MergeLoops(const Stmt &stmt, bool is_dynamic = false);

Stmt AnalyzeMinAlignStatic(Stmt stmt);

Stmt AnalyzeMinAlignDynamic(Stmt stmt, bool enable_conv_analyze_align, bool set_scalar_align = false);

Stmt RewriteByAlignStatic(Stmt stmt);

Stmt RewriteBroadcastVector(Stmt stmt);

Stmt OptimizePragma(Stmt stmt);

Stmt RewriteByAlignDynamic(Stmt stmt);

Stmt EliminateAtomicDma(Stmt stmt);

Stmt ExpandC0(Stmt stmt);

Stmt DTypeAdapter(Stmt stmt);

Stmt SetVectorMaskDefault(const Stmt &stmt);

Stmt ElimVectorMask(Stmt stmt);

Stmt TileCoverCorrect(Stmt stmt);

Stmt SelectLower(Stmt stmt);

Stmt RewriteTensorIndex(Stmt stmt);

Stmt LowerWith(Stmt stmt);

Stmt ModDivEliminate(Stmt stmt);

Stmt AutoDoubleBuffer(Stmt stmt);

Stmt ConvertExtentToCond(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt ConvertCondToExtent(Stmt stmt);

Stmt RewriteVarTensorIdx(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt AlignLastAxisLoopExtent(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt ConvertIfToSelect(Stmt stmt);

/*!
 * \brief Split complicated expression to three address code and do instruction selection
 *        + reuse_variable = True: will try to minimize the newly generated variables
 *        + minimum_split - use with "reuse_variable" to reuse newly generated tensors when exceeding this threshold
 */
Stmt ToThreeAddress(Stmt stmt, bool reuse_variable = false, int minimum_split = 10, bool cross_stmt_simplify = false);

/*!
 * \brief Use pattern match for simple statement rewrite
 */
Stmt StmtPatternRewrite(Stmt stmt);
Stmt ExprPatternRewrite(Stmt stmt);

Stmt AutoPragma(Stmt stmt);

Stmt DMASink(Stmt stmt);

Stmt SpecialValueReplacer(Stmt stmt);

Stmt ReplaceFargmaxCasts(Stmt stmt);

Stmt GatherLoopInfo(Stmt stmt);

Stmt CoverProtection(Stmt stmt, size_t th_block, size_t th_protect);

Stmt AutoMadPragmaAttr(Stmt stmt, bool man_schedule = false);

Stmt LowerStorageAccessInfoCCE(Stmt stmt);

Stmt StmtCSE(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt ValueNumbering(Stmt stmt);

Stmt MultiLastAxisReductions(Stmt stmt, bool is_dynamic);

Stmt AutoReorder(Stmt stmt);
Stmt SplitTail(Stmt stmt);

Stmt CopyPropagation(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Expr CastNormalize(const Expr &expr, const ktvm::DataType cast_type);

std::string DumpC(const Stmt &stmt, const Array<Buffer> &extern_buffer);

/*!
 * \brief Simplify expr using custom cce simplifiers.
 *
 * \param expr The expression to be simplified.
 * \return The result.
 *
 * \note Analyzer will call into sub-analyzers to get the result.
 */

Expr Simplify_cce(Expr expr, const Map<Var, Range> &vrange = Map<Var, Range>());
/*!
 * \brief Simplify stmt using custom cce simplifiers.
 *
 * \param expr The statement to be simplified.
 * \return The result.
 *
 * \note Analyzer will call into sub-analyzers to get the result.
 */
Stmt Simplify_cce(const Stmt &stmt, const Map<Var, Range> &vrange = Map<Var, Range>());

Stmt ElementwiseFlatten(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer,
                        const Map<Tensor, Buffer> &new_extern_buffer);

Stmt MultiCorePartition(const Stmt &stmt);

Stmt MultiCoreLoopSwitchHoist(Stmt stmt);

Stmt LoopSwitchHoist(Stmt stmt, bool hoist_allocate = false);

Stmt DeadCodeElim(Stmt stmt);

Stmt PoolingTransform(Stmt stmt, bool is_dynamic);

Stmt PreProcess4Multicore(Stmt stmt);

Stmt HalfReduceSumRewrite(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt FeatureLibTransform(Stmt stmt);

Stmt SpecifyMinMaxDataType(Stmt stmt);

Stmt RemoveAssert(const Stmt &stmt);

Stmt RewriteFloorDiv(const Stmt &stmt);

Expr CastNormalize(const Expr &expr, const ktvm::DataType cast_type);

Stmt TestInferBoundWithCond(const Expr &expr, const Array<Expr> &constraints);

Stmt TestReduceInequality(const ktvm::Expr &e, const Var &reduce_var, bool scale, bool getlarger);

Stmt TestSimplify(const Expr &expr);

Stmt TestCanProveWithPosParam(const ktvm::Expr &e);

Stmt PromoteCommonExpr(const Stmt &stmt);

Stmt PromoteConstExpr(const Stmt &stmt);

Array<NodeRef> CollectExternalCall(const Stmt &stmt);

Array<NodeRef> CastKernelParams(const Stmt &stmt, const Array<NodeRef> &arg_list);

Stmt AlgebraSimplify(const Stmt &stmt);

Stmt ConvertDivModToShift(const Stmt &stmt);

Stmt UnifyAllocate(const Stmt &stmt);

Stmt SinkAllocate(const Stmt &stmt);

Stmt FixBindBuffer(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer);

Stmt CastFilter(const Stmt &stmt);

Stmt ScalarComputeRewrite(const Stmt &stmt);
}  // namespace ir
}  // namespace akg

namespace ktvm {
namespace ir {
/** Substitute variables with the given pointer with the replacement
 * expression within expr. */
Expr substitute(const Variable *var, Expr replacement, Expr expr);

/** Substitute variables with the given pointer with the replacement
 * expression within stmt. */
Stmt substitute(const Variable *var, Expr replacement, Stmt stmt);

inline Expr substitute(const VarExpr &var, const Expr replacement, const Expr expr) {
  return substitute(var.get(), replacement, expr);
}

inline Stmt substitute(const VarExpr &var, const Expr replacement, const Stmt stmt) {
  return substitute(var.get(), replacement, stmt);
}

/** Substitute variables with pointers in the map. */
// @{
Expr substitute(const std::map<const Variable *, Expr> &replacements, Expr expr);
Stmt substitute(const std::map<const Variable *, Expr> &replacements, Stmt stmt);
// @}

/** Substitute expressions for other expressions. */
// @{
Expr substitute(Expr find, Expr replacement, Expr expr);
Stmt substitute(Expr find, Expr replacement, Stmt stmt);
// @}

/* align_partition.cc needs to call this function from tvm */
Stmt AppendStmts(const Stmt &a, const Stmt &b);

/* simplify_passes_cce.cc needs to call this function from tvm */
bool ExprUseVars(const Expr &expr, const std::unordered_set<const Variable *> &vars);

/*!
 * \brief partition loops in the stmt
 * \param stmt The stmt to do loop partition
 * \param split_const_loop flag to enable partition for const loop
 * \param remove_div_mod removes the division and modulo in the indexing of a tensor by partitioning the loop
 * \param partition_conv: whether to partition the convolution or not
 * \return Transformed stmt.
 */
Stmt LoopPartitionCCE(Stmt stmt, bool split_const_loop, bool remove_div_mod = false, bool partition_conv = false);
}  // namespace ir
}  // namespace ktvm

#endif  // INCLUDE_AKG_IR_PASS_H_
