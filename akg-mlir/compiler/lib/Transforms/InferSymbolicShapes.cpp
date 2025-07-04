/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Transforms/Passes.h"
#include "akg/Utils/AKGGlobalVars.hpp"

#include <optional>
#include <string>
#include "symengine/expression.h"

namespace mlir {
#ifndef GEN_PASS_DECL_INFERSYMBOLICSHAPES
#ifndef GEN_PASS_DEF_INFERSYMBOLICSHAPES
#define GEN_PASS_DECL_INFERSYMBOLICSHAPES
#define GEN_PASS_DEF_INFERSYMBOLICSHAPES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
}  // namespace mlir
using namespace akgglobal;
using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::mindspore;
namespace {
static const SymEngine::Expression constOneExpr = 1;
static const SymEngine::Expression constZeroExpr = 0;
static constexpr char constOneStr[] = "1";
static constexpr char constZeroStr[] = "0";
static const uint64_t kDimIdx0 = 0;
static const uint64_t kDimIdx1 = 1;
static const uint64_t kDimIdx2 = 2;
static const uint64_t kDimIdx3 = 3;

static std::optional<NamedAttribute> getSymbolicShapeFromFrontend(Operation *op, StringRef &key) {
  if (!op->hasAttr(getFrontendSymbolAttrName())) {
    return std::nullopt;
  }
  DictionaryAttr dict = op->getAttr(getFrontendSymbolAttrName()).dyn_cast_or_null<DictionaryAttr>();
  std::optional<NamedAttribute> namedAttr = dict.getNamed(key);
  if (namedAttr == std::nullopt) {
    return std::nullopt;
  }
  (*namedAttr).setName(StringAttr::get(op->getContext(), getSymbolShapeAttrName()));
  return (*namedAttr);
}

template <typename OpTy>
struct PropagateMindsporeReduceOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &) const override {
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    mlir::Value resVal = op.getOperation()->getResults()[0];
    if (analysis.hasSymbolicShape(resVal.getType())) {
      return success();
    }
    // operand
    mlir::Value opnd0 = op.getOperation()->getOperands()[0];
    opnd0.setType(analysis.createNewSymbolicShape(opnd0.getType()));
    std::optional<llvm::SmallVector<std::string>> symShape = analysis.getSymbolicShape(opnd0.getType());
    if (!symShape) {
      return success();
    }
    // result
    for (uint64_t i = 0; i < op.getAxis().size(); i++) {
      (*symShape)[op.getAxis()[i]] = constOneStr;
    }
    resVal.setType(analysis.updateSymbolicShape(resVal.getType(), *symShape));
    return success();
  }
};

template <typename OpTy>
struct PropagateMindsporeCastOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    mlir::Value resVal = op.getOperation()->getResults()[0];
    if (analysis.hasSymbolicShape(resVal.getType())) {
      return success();
    }
    // operand
    mlir::Value opnd0 = op.getOperation()->getOperands()[0];
    opnd0.setType(analysis.createNewSymbolicShape(opnd0.getType()));
    std::optional<NamedAttribute> namedAttr = analysis.getSymbolShapeNamedAttr(opnd0.getType());
    if (!namedAttr) {
      return success();
    }
    // result

    resVal.setType(analysis.updateSymbolicShape(resVal.getType(), *namedAttr));
    return success();
  }
};

template <typename OpTy>
struct PropagateSameOprandsAndResultsShapeTosaOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    mlir::Value resVal = op.getOperation()->getResults()[0];
    if (analysis.hasSymbolicShape(resVal.getType())) {
      return success();
    }
    // operand0
    mlir::Value opnd0 = op.getOperation()->getOperands()[0];
    opnd0.setType(analysis.createNewSymbolicShape(opnd0.getType()));
    std::optional<NamedAttribute> namedAttr = analysis.getSymbolShapeNamedAttr(opnd0.getType());
    if (!namedAttr) {
      return success();
    }
    // operand1~n
    for (uint i = 1; i < op.getOperation()->getOperands().size(); i++) {
      mlir::Value opndN = op.getOperation()->getOperands()[i];
      opndN.setType(analysis.updateSymbolicShape(opndN.getType(), *namedAttr));
    }
    // result
    resVal.setType(analysis.updateSymbolicShape(resVal.getType(), *namedAttr));
    return success();
  }
};

static SymEngine::Expression GetBroadCastDim(const SymEngine::Expression &lhs, const SymEngine::Expression &rhs) {
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  if (lhs == rhs) {
    return lhs;
  }
  //           | lhs = 1 ; lhs != 1
  // --------- | -------------------------------
  // rhs  = 1  |    1        lhs
  // rhs != 1  |    rhs      newSymbolicDim()
  return (lhs == constOneExpr) ? ((rhs == constOneExpr) ? constOneExpr : rhs)
                               : ((rhs == constOneExpr) ? lhs : analysis.getNewSymbolicDimExpr());
}

// todo: To handle complex expressions, the parameters will be changed to SymEngine::Expression.
static std::optional<llvm::SmallVector<std::string>> GetInferenceShape(const llvm::SmallVector<std::string> &longShape,
                                                                       const llvm::SmallVector<std::string> &shortShape,
                                                                       const llvm::ArrayRef<int64_t> &res) {
  // Scenario 1.1: ShortShape's dims are all '1'
  // (n, m) + (1) => (n, m)
  // (n, m, k) + (1, 1) => (n, m, k)
  auto isShortShapeAllOne = [&](const llvm::SmallVector<std::string> &shortShape) -> bool {
    for (auto u : shortShape) {
      if (u != "1") {
        return false;
      }
    }
    return true;
  };
  if (isShortShapeAllOne(shortShape)) {
    return longShape;
  }
  // Scenario 1.2: symbolic infer is required for the remaining scenarios.
  // Scenario 1.2.1: The res dim has been determined to be a static shape.
  // (n, m) + (m) = (4, m) => n == 4 ('n' will be updated at the end of this infer Pass)
  // Scenario 1.2.2:
  // (n, m) + (m) => (n, m)
  // (n, m) + (n) => (n, m)
  // (n, m) + (k) => fail
  llvm::SmallVector<std::string> resShape;
  uint64_t longRank = longShape.size();
  uint64_t shortRank = shortShape.size();
  uint64_t shortIdx = 0, longIdx = 0;
  while (longIdx < longRank && shortIdx < shortRank) {
    (void)resShape.emplace_back(res[longIdx] == ShapedType::kDynamic ? longShape[longIdx]
                                                                     : std::to_string(res[longIdx]));
    if (shortShape[shortIdx] == longShape[longIdx]) {
      longIdx++;
      shortIdx++;
    } else {
      longIdx++;
    }
  }
  if (shortIdx < shortRank) {
    return std::nullopt;
  }
  if (longIdx < longRank) {
    for (uint i = longIdx; i < longRank; i++) {
      (void)resShape.emplace_back(res[longIdx] == ShapedType::kDynamic ? longShape[longIdx]
                                                                       : std::to_string(res[longIdx]));
    }
  }
  assert(resShape.size() == longRank);
  return resShape;
}

template <typename OpTy>
struct PropagateElementWiseOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    mlir::Value opnd0 = op.getOperation()->getOperands()[0];
    mlir::Value opnd1 = op.getOperation()->getOperands()[1];
    int64_t lhsRank = opnd0.getType().cast<ShapedType>().getRank();
    int64_t rhsRank = opnd1.getType().cast<ShapedType>().getRank();
    opnd0.setType(analysis.createNewSymbolicShape(opnd0.getType()));
    opnd1.setType(analysis.createNewSymbolicShape(opnd1.getType()));
    std::optional<llvm::SmallVector<std::string>> lSymShape = analysis.getSymbolicShape(opnd0.getType());
    std::optional<llvm::SmallVector<std::string>> rSymShape = analysis.getSymbolicShape(opnd1.getType());
    assert(lSymShape && rSymShape);
    mlir::Value resVal = op.getOperation()->getResults()[0];
    if (analysis.hasSymbolicShape(resVal.getType())) {
      return success();
    }
    // construct a symbolic shape according to lhs and rhs
    // Examples:
    // (n, m) + (1) => (n, m)
    // (n, m, k) + (1, 1) => (n, m, k)
    // (n, m) + (1,1) => (n, m)
    // (n, m) + (n,1) => (n, m)
    // (n, m) + (m) => (n, m)
    // todo : (n, 1, m) + (2, m) => (n, 2, m)

    // Scenario 1: The dims of the lhs and rhs are not equal.
    // lhs as longShape
    if (lhsRank > rhsRank) {
      std::optional<llvm::SmallVector<std::string>> resShape =
        GetInferenceShape(*lSymShape, *rSymShape, resVal.getType().cast<ShapedType>().getShape());
      if (resShape == std::nullopt) {
        resVal.setType(analysis.createNewSymbolicShape(resVal.getType()));
        return success();
      }
      resVal.setType(analysis.updateSymbolicShape(resVal.getType(), *resShape));
      return success();
    }
    // rhs as longShape
    if (lhsRank < rhsRank) {
      std::optional<llvm::SmallVector<std::string>> resShape =
        GetInferenceShape(*rSymShape, *lSymShape, resVal.getType().cast<ShapedType>().getShape());
      if (resShape == std::nullopt) {
        resVal.setType(analysis.createNewSymbolicShape(resVal.getType()));
        return success();
      }
      resVal.setType(analysis.updateSymbolicShape(resVal.getType(), *resShape));
      return success();
    }
    // Scenario 2: The dims of the lhs and rhs are equal.
    llvm::SmallVector<std::string> symShape;
    for (int i = 0; i < lhsRank; i++) {
      // Scenario 2.1: The res dim has been determined to be a static shape.
      //          e.g. (? or 4) + (? or 4) => (4)
      if (resVal.getType().cast<ShapedType>().getShape()[i] != ShapedType::kDynamic) {
        (void)symShape.emplace_back(std::to_string(resVal.getType().cast<ShapedType>().getShape()[i]));
        continue;
      }
      // Scenario 2.2: (4) + (?) => (4)
      if (opnd0.getType().cast<ShapedType>().getShape()[i] > 1 &&
          opnd1.getType().cast<ShapedType>().getShape()[i] == ShapedType::kDynamic) {
        (void)symShape.emplace_back(std::to_string(opnd0.getType().cast<ShapedType>().getShape()[i]));
        continue;
      }
      //            or (?) + (4) => (4)
      if (opnd1.getType().cast<ShapedType>().getShape()[i] > 1 &&
          opnd0.getType().cast<ShapedType>().getShape()[i] == ShapedType::kDynamic) {
        (void)symShape.emplace_back(std::to_string(opnd1.getType().cast<ShapedType>().getShape()[i]));
        continue;
      }
      // Scenario 2.3: symbolic infer is required for the remaining scenarios.
      std::optional<SymEngine::Expression> lhs = analysis.getSymbolicDimExpr(opnd0.getType(), i);
      std::optional<SymEngine::Expression> rhs = analysis.getSymbolicDimExpr(opnd1.getType(), i);
      SymEngine::Expression bs = GetBroadCastDim(*lhs, *rhs);
      (void)symShape.emplace_back(analysis.getSymbolicDimFromExpression(bs));
    }
    resVal.setType(analysis.updateSymbolicShape(resVal.getType(), symShape));
    return success();
  }
};

template <typename OpTy>
struct PropagateTosaBatchMatMulOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    mlir::Value opnd0 = op.getOperation()->getOperands()[0];
    mlir::Value opnd1 = op.getOperation()->getOperands()[1];
    std::optional<llvm::SmallVector<std::string>> symShape0 = analysis.getSymbolicShape(opnd0.getType());
    std::optional<llvm::SmallVector<std::string>> symShape1 = analysis.getSymbolicShape(opnd1.getType());
    if (!symShape0 && !symShape1) {
      return success();
    }
    mlir::Value resVal = op.getOperation()->getResults()[0];
    if (analysis.hasSymbolicShape(resVal.getType())) {
      return success();
    }
    int64_t rank = opnd0.getType().cast<ShapedType>().getRank();
    int64_t fourthRank = 4;
    if (rank == fourthRank && op.getOperation()->getAttr("transpose_b")) {
      if (!symShape1) {
        symShape1 = symShape0;
        (*symShape1)[kDimIdx2] = analysis.newSymbolicDim();
        opnd1.setType(analysis.updateSymbolicShape(opnd1.getType(), *symShape1));
      }
      if (!symShape0) {
        symShape0 = symShape1;
        (*symShape0)[kDimIdx2] = analysis.newSymbolicDim();
        opnd0.setType(analysis.updateSymbolicShape(opnd0.getType(), *symShape1));
      }

      // result
      llvm::SmallVector<std::string> resSymShape(*symShape0);
      resSymShape[kDimIdx3] = (*symShape1)[kDimIdx2];
      resVal.setType(analysis.updateSymbolicShape(resVal.getType(), resSymShape));
      return success();
    } else {
      llvm::errs() << "unsupported now";
    }
    return success();
  }
};

struct PropagateMindSporeReshapeOp : public OpRewritePattern<mindspore::ReshapeOp> {
  using OpRewritePattern<mindspore::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::ReshapeOp op, PatternRewriter &rewriter) const override {
    if (op.getNewShapeValue() != nullptr) {
      return rewriter.notifyMatchFailure(op, "new shape value is unsupported now");
    }
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    mlir::Value opnd = op.getOperation()->getOperands()[0];
    mlir::Value resVal = op.getOperation()->getResults()[0];
    if (analysis.hasSymbolicShape(resVal.getType())) {
      return success();
    }
    // init symbolic info
    opnd.setType(analysis.createNewSymbolicShape(opnd.getType()));
    resVal.setType(analysis.createNewSymbolicShape(resVal.getType()));
    // second infer
    // If two dimensions of the output shape are dynamic, the Op semantics are ambiguous or illegal. And symbolic
    // information cannot be deduced here.
    auto rankType = resVal.getType().dyn_cast<RankedTensorType>();
    if (rankType == nullptr || rankType.getNumDynamicDims() >= 2 || rankType.getNumDynamicDims() == 0) {
      return success();
    }
    std::optional<llvm::SmallVector<std::string>> opndShape = analysis.getSymbolicShape(opnd.getType());
    std::optional<llvm::SmallVector<std::string>> resShape = analysis.getSymbolicShape(resVal.getType());

    std::string intermediateShape("1");
    for (auto sym : *opndShape) {
      intermediateShape += "*" + sym;
    }
    uint dimIdx = 0, inferDim = 0;
    for (auto sym : *resShape) {
      if (resVal.getType().cast<ShapedType>().getShape()[dimIdx] == ShapedType::kDynamic) {
        inferDim = dimIdx;
        dimIdx++;
        continue;
      }
      intermediateShape += "/" + sym;
      dimIdx++;
    }

    SymEngine::Expression expr(intermediateShape);
    intermediateShape = analysis.getSymbolicDimFromExpression(expr);
    (*resShape)[inferDim] = intermediateShape;
    resVal.setType(analysis.updateSymbolicShape(resVal.getType(), *resShape));
    return success();
  }
};

void InferSymbolicShapesInFunc(func::FuncOp &func, bool isFinalInference) {
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  llvm::SmallVector<Type, 2> newInTys;
  llvm::SmallVector<Type, 2> newResTys;

  // func parameters
  Block &entryBlock = func.getBody().front();
  uint64_t i = 0;
  for (mlir::Value opnd : entryBlock.getArguments()) {
    if (isFinalInference) {
      Type newType = analysis.createNewSymbolicShape(opnd.getType());
      opnd.setType(newType);
      (void)newInTys.emplace_back(newType);
      continue;
    }
    StringRef key("input_" + std::to_string(i++));
    std::optional<NamedAttribute> symbol = getSymbolicShapeFromFrontend(func.getOperation(), key);
    if (symbol != std::nullopt) {
      Type newTy = analysis.updateSymbolicShape(opnd.getType(), *symbol);
      opnd.setType(newTy);
      (void)newInTys.emplace_back(newTy);
    }
  }

  // func body
  for (auto &block : func.getBody()) {
    for (Operation &op : block) {
      if (!isa<mlir::mindspore::MindSporeOp>(op)) {
        continue;
      }
      uint64_t j = 0;
      for (mlir::Value opnd : op.getOperands()) {
        if (isFinalInference) {
          opnd.setType(analysis.createNewSymbolicShape(opnd.getType()));
          continue;
        }
        StringRef key("input_" + std::to_string(j++));
        std::optional<NamedAttribute> symbol = getSymbolicShapeFromFrontend(&op, key);
        if (symbol != std::nullopt) {
          opnd.setType(analysis.updateSymbolicShape(opnd.getType(), *symbol));
        }
      }
      j = 0;
      for (mlir::Value resVal : op.getResults()) {
        if (isFinalInference) {
          resVal.setType(analysis.createNewSymbolicShape(resVal.getType()));
          continue;
        }
        StringRef key("output_" + std::to_string(j++));
        std::optional<NamedAttribute> symbol = getSymbolicShapeFromFrontend(&op, key);
        if (symbol != std::nullopt) {
          resVal.setType(analysis.updateSymbolicShape(resVal.getType(), *symbol));
        }
      }
      (void)op.removeAttr(getFrontendSymbolAttrName());
    }
  }
  // func return
  func.walk([&](func::ReturnOp op) {
    uint64_t i = 0;
    for (mlir::Value opnd : op.getOperation()->getOperands()) {
      if (isFinalInference) {
        (void)newResTys.emplace_back(opnd.getType());
        continue;
      }
      StringRef key("output_" + std::to_string(i++));
      std::optional<NamedAttribute> symbol = getSymbolicShapeFromFrontend(func.getOperation(), key);
      if (symbol != std::nullopt) {
        Type newTy = analysis.updateSymbolicShape(opnd.getType(), *symbol);
        opnd.setType(newTy);
        (void)newResTys.emplace_back(newTy);
      }
    }
  });
  (void)func->removeAttr(getFrontendSymbolAttrName());
  // update func type
  auto newFuncTy = mlir::FunctionType::get(func.getContext(), newInTys, newResTys);
  func.setType(newFuncTy);
}

/// Pass that performs shape inference across mindspore operations.
struct InferSymbolicShapes : public impl::InferSymbolicShapesBase<InferSymbolicShapes> {
 public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    MLIRContext *ctx = func.getContext();

    // 1.pre propagation from op.
    InferSymbolicShapesInFunc(func, false);

    // 2.infer symbolic shapes in mlir
    // Add the generated patterns to the list.
    (void)patterns.add<PropagateElementWiseOp<mindspore::AddOp>>(ctx);
    (void)patterns.add<PropagateElementWiseOp<mindspore::SubOp>>(ctx);
    (void)patterns.add<PropagateElementWiseOp<mindspore::MulOp>>(ctx);
    (void)patterns.add<PropagateElementWiseOp<mindspore::DivOp>>(ctx);
    (void)patterns.add<PropagateMindsporeReduceOp<mindspore::ReduceAllOp>>(ctx);
    (void)patterns.add<PropagateMindsporeReduceOp<mindspore::ReduceAnyOp>>(ctx);
    (void)patterns.add<PropagateMindsporeReduceOp<mindspore::ReduceMaxOp>>(ctx);
    (void)patterns.add<PropagateMindsporeReduceOp<mindspore::ReduceMinOp>>(ctx);
    (void)patterns.add<PropagateMindsporeReduceOp<mindspore::ReduceProdOp>>(ctx);
    (void)patterns.add<PropagateMindsporeReduceOp<mindspore::ReduceSumOp>>(ctx);
    (void)patterns.add<PropagateMindsporeCastOp<mindspore::CastOp>>(ctx);
    (void)patterns.add<PropagateSameOprandsAndResultsShapeTosaOp<mindspore::ExpOp>>(ctx);
    (void)patterns.add<PropagateSameOprandsAndResultsShapeTosaOp<mindspore::AddNOp>>(ctx);
    (void)patterns.add<PropagateSameOprandsAndResultsShapeTosaOp<mindspore::AssignOp>>(ctx);
    (void)patterns.add<PropagateMindSporeReshapeOp>(ctx);

    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), grc);

    // 3.final inference
    InferSymbolicShapesInFunc(func, true);

    // 4.build GlobalHostShapeInfo
    initGlobalHostShapeInfo();
  }

 private:
  void initGlobalHostShapeInfo() {
    func::FuncOp func = getOperation();
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    ShapeAlignTool &tool = ShapeAlignTool::getInstance();
    std::map<size_t, ShapeInfo> hostShapes = {};

    auto convertToShapeInfo = [&](std::optional<llvm::SmallVector<std::string>> symShape) -> ShapeInfo {
      ShapeInfo record;
      for (auto shape : *symShape) {
        record.push_back(shape);
      }
      return record;
    };
    // 1. init inputs
    for (size_t argIdx = 0; argIdx < func.getBody().front().getArguments().size(); ++argIdx) {
      auto arg = func.getBody().front().getArgument(argIdx);
      auto symShape = analysis.getSymbolicShape(arg.getType());
      auto record = convertToShapeInfo(symShape);
      hostShapes[argIdx] = record;
    }

    // 2. init outputs and indices
    std::unordered_set<size_t> outputIndices;
    func.walk([&](func::ReturnOp op) {
      for (mlir::Value opnd : op.getOperation()->getOperands()) {
        auto symShape = analysis.getSymbolicShape(opnd.getType());
        auto record = convertToShapeInfo(symShape);
        auto outIdx = hostShapes.size();
        (void)outputIndices.insert(outIdx);
        hostShapes[outIdx] = record;
      }
    });
    tool.initHost(hostShapes, outputIndices);
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> mlir::createInferSymbolicShapesPass() { return std::make_unique<InferSymbolicShapes>(); }
