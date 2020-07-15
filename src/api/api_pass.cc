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

#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/api_registry.h>

namespace akg {
namespace ir {
using air::ir::LoopPartitionCCE;
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

TVM_REGISTER_API("ir_pass.LoopPartitionCCE").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() == 2) {
    *ret = LoopPartitionCCE(args[0], args[1]);
  } else if (args.size() == 3) {
    *ret = LoopPartitionCCE(args[0], args[1], args[2]);
  } else {
    CHECK_EQ(args.size(), 4);
    *ret = LoopPartitionCCE(args[0], args[1], args[2], args[3]);
  }
});

TVM_REGISTER_API("ir_pass.StorageRewriteCCE").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() == 2) {
    *ret = StorageRewriteCCE(args[0], args[1]);
  } else if (args.size() == 3) {
    *ret = StorageRewriteCCE(args[0], args[1], args[2]);
  } else if (args.size() == 4) {
    *ret = StorageRewriteCCE(args[0], args[1], args[2], args[3]);
  } else {
    CHECK_EQ(args.size(), 5);
    *ret = StorageRewriteCCE(args[0], args[1], args[2], args[3], args[4]);
  }
});

TVM_REGISTER_API("ir_pass.EmitInsn").set_body([](const TVMArgs args, TVMRetValue *ret) {
  CHECK_EQ(args.size(), 5);
  *ret = EmitInsn(args[0], args[1], args[2], args[3], args[4]);
});

TVM_REGISTER_API("ir_pass.LoopSwitchHoist").set_body([](const TVMArgs args, TVMRetValue *ret) {
  CHECK_EQ(args.size(), 2);
  *ret = LoopSwitchHoist(args[0], args[1]);
});

TVM_REGISTER_API("ir_pass.AutoMadPragmaAttr").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() == 1) {
    *ret = AutoMadPragmaAttr(args[0]);
  } else {
    CHECK_EQ(args.size(), 2);
    *ret = AutoMadPragmaAttr(args[0], args[1]);
  }
});

TVM_REGISTER_API("ir_pass.ToThreeAddress").set_body([](const TVMArgs args, TVMRetValue *ret) {
  if (args.size() == 1) {
    *ret = ToThreeAddress(args[0]);
  } else if (args.size() == 2) {
    *ret = ToThreeAddress(args[0], args[1]);
  } else if (args.size() == 3) {
    *ret = ToThreeAddress(args[0], args[1], args[2]);
  } else {
    CHECK_EQ(args.size(), 4);
    *ret = ToThreeAddress(args[0], args[1], args[2], args[3]);
  }
});

#define REGISTER_PASS(PassName) TVM_REGISTER_API("ir_pass." #PassName).set_body_typed(PassName);

REGISTER_PASS(SinkIfStmt);
REGISTER_PASS(AutoPoly);
REGISTER_PASS(PromoteIfStmt);
REGISTER_PASS(PromoteLetStmt);
REGISTER_PASS(GenTuningSpace);
REGISTER_PASS(Load3dTrans);
REGISTER_PASS(PostFusion);
REGISTER_PASS(DmaFlatten);
REGISTER_PASS(ReduceFusionOpt);
REGISTER_PASS(PostProcessImg2col);
REGISTER_PASS(BypassL1);
REGISTER_PASS(StrideKernelOp);
REGISTER_PASS(AlignPartitionCCE);
REGISTER_PASS(EmitInsnDebug);
REGISTER_PASS(RewriteMultiValueFunc);
REGISTER_PASS(RenameRealize);
REGISTER_PASS(InjectSync);
REGISTER_PASS(MathIntrinRewrite);
REGISTER_PASS(InvariantHoist);
REGISTER_PASS(ElimDMA);
REGISTER_PASS(InjectAttr);
REGISTER_PASS(InjectPipe);
REGISTER_PASS(HoistInsn);
REGISTER_PASS(UnifyLoopVars);
REGISTER_PASS(IsolateLoops);
REGISTER_PASS(CheckShapeParams);
REGISTER_PASS(EliminateIf);
REGISTER_PASS(FixMadAttrs);
REGISTER_PASS(UnrollNonConstantExtent)
REGISTER_PASS(AnalyzeMinAlignStatic);
REGISTER_PASS(AnalyzeMinAlignDynamic);
REGISTER_PASS(RewriteBroadcastVector);
REGISTER_PASS(OptimizePragma);
REGISTER_PASS(ExpandC0);
REGISTER_PASS(ForEliminate);
REGISTER_PASS(FixLoopExtent);
REGISTER_PASS(FixRealizeShape);
REGISTER_PASS(DTypeAdapter);
REGISTER_PASS(SetVectorMaskDefault);
REGISTER_PASS(ElimVectorMask);
REGISTER_PASS(TileCoverCorrect);
REGISTER_PASS(SelectLower);
REGISTER_PASS(RewriteTensorIndex);
REGISTER_PASS(InjectMultiCore);
REGISTER_PASS(InjectMultiCoreVar);
REGISTER_PASS(InjectAccessPtrMSG);
REGISTER_PASS(RemoveAccessPtrMSG);
REGISTER_PASS(LowerWith);
REGISTER_PASS(ModDivEliminate);
REGISTER_PASS(RealizeCompress);
REGISTER_PASS(LoopNormlize);
REGISTER_PASS(AutoDoubleBuffer);
REGISTER_PASS(ConvertExtentToCond);
REGISTER_PASS(ConvertCondToExtent);
REGISTER_PASS(RewriteVarTensorIdx);
REGISTER_PASS(AlignLastAxisLoopExtent);
REGISTER_PASS(ConvertIfToSelect);
REGISTER_PASS(StmtPatternRewrite);
REGISTER_PASS(ExprPatternRewrite);
REGISTER_PASS(AutoPragma);
REGISTER_PASS(EliminateAtomicDma);
REGISTER_PASS(DMASink);
REGISTER_PASS(SpecialValueReplacer);
REGISTER_PASS(ReplaceFargmaxCasts);
REGISTER_PASS(GatherLoopInfo);
REGISTER_PASS(CoverProtection);
REGISTER_PASS(CoarsenImg2Col);
REGISTER_PASS(LowerStorageAccessInfoCCE);
REGISTER_PASS(StmtCSE);
REGISTER_PASS(ValueNumbering);
REGISTER_PASS(MultiLastAxisReductions);
REGISTER_PASS(AutoReorder);
REGISTER_PASS(CopyPropagation);
REGISTER_PASS(ElementwiseFlatten);
REGISTER_PASS(MultiCorePartition);
REGISTER_PASS(MultiCoreLoopSwitchHoist);
REGISTER_PASS(DeadCodeElim);
REGISTER_PASS(PoolingTransform);
REGISTER_PASS(PreProcess4Multicore);
REGISTER_PASS(HalfReduceSumRewrite);
REGISTER_PASS(FeatureLibTransform);
REGISTER_PASS(SpecifyMinMaxDataType);
REGISTER_PASS(RemoveAssert);
REGISTER_PASS(PromoteCommonExpr);
REGISTER_PASS(PromoteConstExpr);
REGISTER_PASS(CollectExternalCall);
REGISTER_PASS(CastKernelParams);
REGISTER_PASS(AlgebraSimplify);
REGISTER_PASS(ConvertDivModToShift);
REGISTER_PASS(UnifyAllocate);
REGISTER_PASS(TestInferBoundWithCond);
REGISTER_PASS(TestReduceInequality);
REGISTER_PASS(TestSimplify);
REGISTER_PASS(TestCanProveWithPosParam);
REGISTER_PASS(RewriteFloorDiv);
REGISTER_PASS(FixBindBuffer);
REGISTER_PASS(SinkAllocate);
REGISTER_PASS(SubstituteDivVar);
REGISTER_PASS(CastFilter);
REGISTER_PASS(ScalarComputeRewrite);
REGISTER_PASS(SplitTail);
}  // namespace ir
}  // namespace akg
