/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <cstddef>
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mfusion/Dialect/Mfuse/Transforms/Cluster/BaseCluster.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/Utils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"
#include "mfusion/Support/Logging.h"

namespace mlir {

namespace mfuse {

namespace {
constexpr int64_t kMaxDimSize = UINT16_MAX - UINT8_MAX;
constexpr int64_t kMinDimSize = 512;

/// Get element type from a value or type
Type getElementType(Type type) {
  if (auto tensorType = dyn_cast<TensorType>(type)) {
    return tensorType.getElementType();
  }
  if (auto rankedTensor = dyn_cast<RankedTensorType>(type)) {
    return rankedTensor.getElementType();
  }
  return Type();
}

bool isComplexDataType(Operation *op) {
  if (op->getNumResults() > 0) {
    Type outputType = op->getResult(0).getType();
    if (mlir::isa<ComplexType>(outputType)) {
      return true;
    }
  }

  if (op->getName().getStringRef() != "mfuse.cast" || op->getNumOperands() < 1) {
    return false;
  }

  // Get the output type of op's first input's defining op
  // This assumes the first operand is produced by another op
  Value input0 = op->getOperand(0);
  Operation *defOp = input0.getDefiningOp();
  if (!defOp) {
    return false;
  }

  // Get the output type of the defining op
  if (auto outputType = getElementType(defOp->getResult(0).getType())) {
    if (mlir::isa<ComplexType>(outputType)) {
      return true;
    }
  }
  return false;
}

/// DvmSupportChecker - Singleton class for checking DVM operator support.
class DvmSupportChecker {
 public:
  static DvmSupportChecker &instance() {
    static DvmSupportChecker instance;
    return instance;
  }

  /// Check if operation is supported by DVM
  bool check(Operation *op) const {
    // Check if operation format is supported by DVM, currently not implemented.
    if (!checkFormat(op)) {
      return false;
    }
    std::string opName = op->getName().getStringRef().str();

    // Find check function for this operation
    auto it = checkFunc_.find(opName);
    if (it != checkFunc_.end()) {
      // Execute all check functions for this operation
      const auto &funcs = it->second;
      for (const auto &func : funcs) {
        if (!func(op)) {
          return false;
        }
      }
      return true;
    }

    // Default check: output must be float/int type
    Type outputType = getElementType(op->getResult(0).getType());
    if (!outputType) {
      return false;  // Can't check without element type
    }
    return isFloatType(outputType) && inputCheck(op, {});
  }

 private:
  DvmSupportChecker() { initializeCheckFunc(); }

  // Check if operation format is supported by DVM, currently not implemented.
  bool checkFormat(Operation *op) const { return true; }

  void initializeCheckFunc() {
    auto inputSameType = [](Operation *op) { return isElementTypesConsistent(op); };
    auto inputCheckAll = [](Operation *op) { return inputCheck(op, {}); };
    auto inputCheckFirst = [](Operation *op) { return inputCheck(op, {0}); };
    auto castCheck = [](Operation *op) { return castCheckFunc(op); };
    auto intOpCheck = [](Operation *op) { return intOpCheckFunc(op); };
    auto compareCheck = [](Operation *op) { return compareCheckFunc(op); };
    auto transposeOpCheck = [](Operation *op) { return transposeOpCheckFunc(op); };
    // Should add collective_comm_op_check when support AllReduce.

    // cast op
    checkFunc_["mfuse.cast"] = {castCheck};
    // reduce sum op
    checkFunc_["mfuse.reduce_sum"] = {reduceSumCheck, inputCheckFirst};
    // cmp op
    checkFunc_["mfuse.eq"] = {compareCheck, inputSameType};
    checkFunc_["mfuse.ne"] = {compareCheck, inputSameType};
    checkFunc_["mfuse.gt"] = {compareCheck, inputSameType};
    checkFunc_["mfuse.ge"] = {compareCheck, inputSameType};
    checkFunc_["mfuse.lt"] = {compareCheck, inputSameType};
    checkFunc_["mfuse.le"] = {compareCheck, inputSameType};
    checkFunc_["mfuse.is_finite"] = {compareCheck, isFiniteOpCheckFunc};
    // select op
    checkFunc_["mfuse.select"] = {selectOpCheck, [](Operation *op) { return inputCheck(op, {kIndex1, kIndex2}); }};
    // int op
    checkFunc_["mfuse.add"] = {intOpCheck, inputCheckAll};
    checkFunc_["mfuse.sub"] = {intOpCheck, inputCheckAll};
    checkFunc_["mfuse.mul"] = {mulOpCheck};
    checkFunc_["mfuse.maximum"] = {intOpCheck, inputCheckAll};
    checkFunc_["mfuse.minimum"] = {intOpCheck, inputCheckAll};
    checkFunc_["mfuse.neg"] = {intOpCheck, inputCheckAll};
    checkFunc_["mfuse.abs"] = {intOpCheck, inputCheckAll};
    // Should add Assign check. There is no corresponding op in aten.
    checkFunc_["mfuse.broadcast_to"] = {intOpCheck, inputCheckFirst};
    // slice op
    checkFunc_["mfuse.slice"] = {sliceSupported, inputCheckFirst};
    // Should add StridedSlice check. There is no corresponding op in aten.
    //  matmul op
    checkFunc_["mfuse.matmul"] = {matmulOpCheck, inputCheckAll};
    checkFunc_["mfuse.batch_matmul"] = {matmulOpCheck, inputCheckAll};
    checkFunc_["mfuse.grouped_matmul"] = {groupedMatmulOpCheck, inputCheckAll};
    // transpose op
    checkFunc_["mfuse.permute"] = {transposeOpCheck, inputCheckAll};
    checkFunc_["mfuse.reshape"] = {reshapeOpCheck};
  }

  static bool isCastTypeSupported(Type type) {
    return type.isF16() || type.isF32() || type.isInteger(1) || type.isInteger(32) || type.isBF16();
  }

  static bool castCheckFunc(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    Type inputType = getElementType(op->getOperand(0).getType());

    // Check input type
    if (!inputType || !outputType) {
      return false;
    }

    // Check if both input and output types are supported
    return isCastTypeSupported(inputType) && isCastTypeSupported(outputType);
  }

  static bool reduceSumCheck(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    return outputType && isFloatType(outputType);
  }

  static bool compareCheckFunc(Operation *op) {
    Type inputType = getElementType(op->getOperand(0).getType());
    return inputType && isFloatIntType(inputType);
  }

  static bool isFiniteOpCheckFunc(Operation *op) {
    Type inputType = getElementType(op->getOperand(0).getType());
    return inputType && !inputType.isInteger(32);
  }

  static bool selectOpCheck(Operation *op) {
    // Check first operand is bool type
    Type condType = getElementType(op->getOperand(0).getType());
    if (!condType || (!condType.isInteger(1) && !condType.isSignlessInteger(1))) {
      MLOG(DEBUG) << "Select op condition not bool";
      return false;
    }
    Type outputType = getElementType(op->getResult(0).getType());
    // Only support float type
    return outputType && isFloatType(outputType);
  }

  static bool intOpCheckFunc(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    return outputType && isFloatIntType(outputType);
  }

  static bool mulOpCheck(Operation *op) {
    return mixTypeCheck(op, [](Type type) { return isFloatIntType(type); }, {});
  }

  static bool transposeOpCheckFunc(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    return outputType && (outputType.isF32() || outputType.isF16());
  }

  static bool matmulOpCheck(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    // Only support float16 and bfloat16
    if (!outputType.isF16() && !outputType.isBF16()) {
      MLOG(DEBUG) << "MatMul op not float16/bfloat16";
      return false;
    }

    // Check if input or output has dynamic shape
    if (isDynamicShapeNode(op)) {
      return false;
    }

    // Check shape constraints
    return matMulShapeCheck(op);
  }

  static bool checkSplitItemAttr(Operation *op) {
    constexpr int64_t kSplitNumType3 = 3;
    auto splitItemAttr = op->getAttr("split_item");
    if (!splitItemAttr) {
      MLOG(DEBUG) << "GroupedMatmul: missing split_item attr";
      return false;
    }
    auto splitItem = dyn_cast<IntegerAttr>(splitItemAttr);
    if (!splitItem || splitItem.getInt() != kSplitNumType3) {
      MLOG(DEBUG) << "GroupedMatmul: split_item must be " << kSplitNumType3;
      return false;
    }
    return true;
  }

  static bool checkGroupTypeAttr(Operation *op) {
    constexpr int64_t kGroupTypeK = 2;
    constexpr int64_t kGroupTypeM = 0;
    auto groupTypeAttr = op->getAttr("group_type");
    if (!groupTypeAttr) {
      MLOG(DEBUG) << "GroupedMatmul: missing group_type attr";
      return false;
    }
    auto groupType = dyn_cast<IntegerAttr>(groupTypeAttr);
    if (!groupType || (groupType.getInt() != kGroupTypeM && groupType.getInt() != kGroupTypeK)) {
      MLOG(DEBUG) << "GroupedMatmul: group_type must be " << kGroupTypeM << " or " << kGroupTypeK;
      return false;
    }
    return true;
  }

  static bool checkOutputType(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    if (!outputType.isF16() && !outputType.isBF16()) {
      MLOG(DEBUG) << "GroupedMatmul: output type must be float16 or bfloat16";
      return false;
    }
    return true;
  }

  static bool checkOptionalInput(Operation *op, size_t index) {
    if (index >= op->getNumOperands()) {
      return true;
    }
    Value operand = op->getOperand(index);
    Type operandType = operand.getType();
    if (isa<TensorType>(operandType)) {
      auto tensorType = dyn_cast<TensorType>(operandType);
      if (!tensorType || !tensorType.hasStaticShape()) {
        MLOG(DEBUG) << "GroupedMatmul: optional input at index " << index << " must have static shape";
        return false;
      }
      auto shape = tensorType.getShape();
      if (shape.size() != 1 || shape[0] != 0) {
        MLOG(DEBUG) << "GroupedMatmul: optional input at index " << index << " must be empty tensor {0}";
        return false;
      }
    } else if (!isa<NoneType>(operandType)) {
      MLOG(DEBUG) << "GroupedMatmul: optional input at index " << index << " must be Tensor or None";
      return false;
    }
    return true;
  }

  static bool checkOptionalInputs(Operation *op) {
    for (size_t i = kIndex3; i <= kIndex6; ++i) {
      if (!checkOptionalInput(op, i)) {
        return false;
      }
    }
    return true;
  }

  static bool checkDimensionConstraints(Operation *op) {
    auto input1TypeTensor = dyn_cast<TensorType>(op->getOperand(0).getType());
    auto input2TypeTensor = dyn_cast<TensorType>(op->getOperand(1).getType());
    if (!input1TypeTensor || !input2TypeTensor) {
      return false;
    }
    auto aShape = input1TypeTensor.getShape();
    auto bShape = input2TypeTensor.getShape();
    if (aShape.empty() || bShape.empty()) {
      return false;
    }
    if (aShape.back() > kMaxDimSize || bShape.back() > kMaxDimSize) {
      MLOG(DEBUG) << "GroupedMatmul: dimension size exceeds max " << kMaxDimSize;
      return false;
    }
    return true;
  }

  static bool groupedMatmulOpCheck(Operation *op) {
    if (!checkSplitItemAttr(op)) {
      return false;
    }
    if (!checkGroupTypeAttr(op)) {
      return false;
    }
    if (!checkOutputType(op)) {
      return false;
    }
    if (!checkOptionalInputs(op)) {
      return false;
    }
    return checkDimensionConstraints(op);
  }

  static bool reshapeOpCheck(Operation *op) {
    auto tensorNode = op->getOperand(0);
    Operation *cubeOp = nullptr;
    if (op->getName().getStringRef() == "mfuse.matmul" || op->getName().getStringRef() == "mfuse.batch_matmul" ||
        op->getName().getStringRef() == "mfuse.grouped_matmul") {
      cubeOp = tensorNode.getDefiningOp();
    }
    return cubeOp && DVMCluster::canClusterableOp(DVMCluster::getClusterableOps(), cubeOp);
  }

  /// MatMul shape check helper
  static bool matMulShapeCheck(Operation *op) {
    auto outputTypeTensor = dyn_cast<TensorType>(op->getResult(0).getType());
    auto input1TypeTensor = dyn_cast<TensorType>(op->getOperand(0).getType());
    auto input2TypeTensor = dyn_cast<TensorType>(op->getOperand(1).getType());

    // Check tensor types
    if (!outputTypeTensor || !input1TypeTensor || !input2TypeTensor) {
      return false;
    }
    auto aShape = input1TypeTensor.getShape();
    auto bShape = input2TypeTensor.getShape();
    auto cShape = outputTypeTensor.getShape();
    if (aShape.back() > kMaxDimSize || bShape.back() > kMaxDimSize) {
      return false;
    }
    if (op->getName().getStringRef() == "mfuse.matmul" && cShape.back() <= kMinDimSize && cShape.size() >= 2 &&
        cShape[cShape.size() - 2] <= kMinDimSize) {
      return false;
    }
    if (op->getName().getStringRef() == "mfuse.batch_matmul" && cShape.size() > 4) {
      return false;
    }
    return true;
  }

  static bool sliceSupported(Operation *op) {
    constexpr size_t kMaxRank = 4;
    if (op->getNumOperands() < 1) {
      return false;
    }

    Type outputType = getElementType(op->getResult(0).getType());
    // Check output type is float/int
    if (!isFloatIntType(outputType)) {
      return false;
    }

    auto outputTypeTensor = dyn_cast<TensorType>(op->getResult(0).getType());
    auto inputTypeTensor = dyn_cast<TensorType>(op->getOperand(0).getType());

    // Check tensor types
    if (!outputTypeTensor || !inputTypeTensor) {
      return false;
    }

    auto inputShape = inputTypeTensor.getShape();
    if (inputShape.size() > kMaxRank) {
      return false;
    }
    auto outputShape = outputTypeTensor.getShape();
    auto rank = outputShape.size();
    for (size_t i = 3; i < rank && rank >= i + 1; i++) {
      if (inputShape[rank - 1 - i] != outputShape[rank - 1 - i]) {
        return false;
      }
    }

    // To check StridedSlice specific: step_vector must be all 1.
    return true;
  }

  /// Check input types
  static bool inputCheck(Operation *op, const std::vector<size_t> &inputsToCheck) {
    Type outputType = getElementType(op->getResult(0).getType());
    size_t inputNum = op->getNumOperands();

    std::vector<size_t> inputs;
    if (inputsToCheck.empty()) {
      for (size_t i = 0; i < inputNum; ++i) {
        inputs.push_back(i);
      }
    } else {
      inputs = inputsToCheck;
    }

    for (size_t idx : inputs) {
      if (idx >= op->getNumOperands()) {
        continue;
      }
      Value operand = op->getOperand(idx);
      Type operandType = operand.getType();
      // Skip tensor type check for non-tensor inputs
      if (!isa<TensorType>(operandType)) {
        continue;
      }
      Type inputType = getElementType(operandType);
      if (inputType != outputType) {
        return false;
      }
    }
    return true;
  }

  /// Check element type mix
  static bool mixTypeCheck(Operation *op, const std::function<bool(Type)> &typeCheck,
                           const std::vector<size_t> &inputsToCheck) {
    Type outputType = getElementType(op->getResult(0).getType());
    if (!typeCheck(outputType)) {
      return false;
    }

    size_t inputNum = op->getNumOperands();
    std::vector<size_t> inputs;
    if (inputsToCheck.empty()) {
      for (size_t i = 0; i < inputNum; ++i) {
        inputs.push_back(i);
      }
    } else {
      inputs = inputsToCheck;
    }

    for (size_t idx : inputs) {
      if (idx >= op->getNumOperands()) {
        continue;
      }
      Type inputType = getElementType(op->getOperand(idx).getType());
      if (!typeCheck(inputType)) {
        return false;
      }
    }
    return true;
  }

  /// Check if element types are consistent
  static bool isElementTypesConsistent(Operation *op) {
    if (op->getNumOperands() <= 1) {
      return true;
    }

    Type firstElemType = getElementType(op->getOperand(0).getType());
    if (!firstElemType) {
      return false;  // Can't check without element type
    }

    for (size_t i = 1; i < op->getNumOperands(); ++i) {
      Type elemType = getElementType(op->getOperand(i).getType());
      if (!elemType || elemType != firstElemType) {
        return false;
      }
    }
    return true;
  }

  static bool isFloatType(Type type) { return type.isF32() || type.isF16() || type.isBF16(); }

  /// Check if output type is float/int type
  static bool isFloatIntType(Type type) { return isFloatType(type) || type.isInteger(32); }

  std::unordered_map<std::string, std::vector<std::function<bool(Operation *)>>> checkFunc_;
};
}  // namespace

llvm::DenseSet<llvm::StringRef> DVMCluster::getClusterableOps() {
  // Clusterable operations for DVM backend.
  // Currently, we only support dvm supported operations.
  return llvm::DenseSet<llvm::StringRef>({
    "mfuse.abs",          "mfuse.add",
    "mfuse.broadcast_to", "mfuse.cast",
    "mfuse.exp",          "mfuse.log",
    "mfuse.maximum",      "mfuse.minimum",
    "mfuse.mul",          "mfuse.neg",
    "mfuse.pow",          "mfuse.div",
    "mfuse.real_div",     "mfuse.reciprocal",
    "mfuse.rsqrt",        "mfuse.sqrt",
    "mfuse.sub",          "mfuse.eq",
    "mfuse.ne",           "mfuse.gt",
    "mfuse.ge",           "mfuse.lt",
    "mfuse.le",           "mfuse.logical_and",
    "mfuse.logical_or",   "mfuse.logical_not",
    "mfuse.select",       "mfuse.assign",
    "mfuse.reduce_sum",   "mfuse.is_finite",
    "mfuse.reshape",      "mfuse.permute",
    "mfuse.floor",        "mfuse.ceil",
    "mfuse.trunc",        "mfuse.matmul",
    "mfuse.batch_matmul", "mfuse.grouped_matmul",
  });
}

// Currently, we only support dvm supported operations.
bool DVMCluster::canClusterableOp(const llvm::DenseSet<llvm::StringRef> &opList, Operation *op) {
  if (op == nullptr) {
    return false;
  }
  // Check if operation is in clusterable list
  StringRef opName = op->getName().getStringRef();
  if (opList.find(opName) == opList.end()) {
    MLOG(DEBUG) << "Op not in cluster list: " << opName;
    return false;
  }

  // Check if output type is complex type
  if (isComplexDataType(op)) {
    MLOG(DEBUG) << "Op has complex output type: " << opName;
    return false;
  }

  // Check DVM-specific constraints
  if (!DvmSupportChecker::instance().check(op)) {
    MLOG(DEBUG) << "Op is not DVM supported: " << opName;
    return false;
  }

  if (hasZeroShape(op)) {
    MLOG(DEBUG) << "Op has zero shape: " << opName;
    return false;
  }

  return true;
}

llvm::DenseSet<llvm::StringRef> DVMCluster::getClusterableOpList() { return getClusterableOps(); }

bool DVMCluster::isClusterableOp(Operation *op) { return canClusterableOp(opList_, op); }

std::string DVMCluster::getFusionType() { return "dvm"; }

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
#define GEN_PASS_DEF_DVMCLUSTER
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

struct DVMClusterPass : public impl::DVMClusterBase<DVMClusterPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    DVMCluster cluster;
    if (cluster.run(funcOp)) {
      MLOG(DEBUG) << "DVMCluster modified function: " << funcOp.getName();
    }
  }
};

}  // namespace mfuse

std::unique_ptr<Pass> createDVMClusterPass() { return std::make_unique<mfuse::DVMClusterPass>(); }

}  // namespace mlir
