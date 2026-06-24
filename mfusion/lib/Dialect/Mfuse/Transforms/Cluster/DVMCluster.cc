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
#include <cmath>
#include <string>
#include <unordered_map>
#include <functional>
#include <limits>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/BaseCluster.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/Utils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"
#include "mfusion/Support/Logging.h"

namespace mlir {

namespace mfuse {

namespace {
constexpr int64_t kMaxDimSize = UINT16_MAX - UINT8_MAX;
constexpr int64_t kMinDimSize = 512;
constexpr int64_t kReduceSumNonReduceAxisSizeLimit = 100000;

bool isProductOfTwoPrimes(int64_t value) {
  if (value < 4) {
    return false;
  }

  size_t primeFactorNum = 0;
  while (value % 2 == 0) {
    value /= 2;
    ++primeFactorNum;
    if (primeFactorNum > 2) {
      return false;
    }
  }
  for (int64_t factor = 3; factor <= value / factor; factor += 2) {
    while (value % factor == 0) {
      value /= factor;
      ++primeFactorNum;
      if (primeFactorNum > 2) {
        return false;
      }
    }
  }
  if (value > 1) {
    ++primeFactorNum;
  }
  return primeFactorNum == 2;
}

bool hasLargeSemiprimeNonReduceAxis(ReduceSumOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!inputType || !inputType.hasStaticShape()) {
    return false;
  }

  const int64_t rank = inputType.getRank();
  std::vector<bool> reduceDims(rank, false);
  for (auto dimAttr : op.getDimensions().getValue()) {
    auto dim = cast<IntegerAttr>(dimAttr).getValue().getSExtValue();
    if (dim < 0 || dim >= rank) {
      return false;
    }
    reduceDims[dim] = true;
  }

  for (int64_t i = 0; i < rank; ++i) {
    if (reduceDims[i]) {
      continue;
    }
    int64_t dimSize = inputType.getDimSize(i);
    if (dimSize > kReduceSumNonReduceAxisSizeLimit && isProductOfTwoPrimes(dimSize)) {
      return true;
    }
  }
  return false;
}

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

  if (op->getName().getStringRef() != "mfuse.cast") {
    return false;
  }

  // Check cast input type
  Value input0 = op->getOperand(0);
  if (auto outputType = getElementType(input0.getType())) {
    if (mlir::isa<ComplexType>(outputType)) {
      return true;
    }
  }
  return false;
}

bool isRankZeroTensor(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 0;
}

bool hasScalarMarker(Type type) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (!rankedType) {
    return false;
  }

  auto dictAttr = dyn_cast_or_null<DictionaryAttr>(rankedType.getEncoding());
  return dictAttr && dictAttr.contains(mfuse::kScalarMarkerAttr);
}

bool isFiniteScalarConstant(DenseElementsAttr denseAttr) {
  Type elementType = denseAttr.getElementType();
  if (!isa<FloatType>(elementType)) {
    return true;
  }
  auto value = *denseAttr.getValues<APFloat>().begin();
  return !value.isNaN() && !value.isInfinity();
}

// Keep DVM clusters aligned with the runtime scalar API surface in dvm.h.
// f64/i64 scalar constants are allowed only when convert-mfuse-to-dvm can
// safely normalize them to f32/i32; bool scalar constants are not supported.
bool isDvmSupportedScalarConstant(ConstantOp op) {
  auto rankedType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!rankedType || rankedType.getRank() != 0 || !hasScalarMarker(rankedType)) {
    return false;
  }

  auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValueAttr());
  if (!denseAttr || denseAttr.getNumElements() != 1 || !isFiniteScalarConstant(denseAttr)) {
    return false;
  }

  Type elementType = rankedType.getElementType();
  if (elementType.isF32() || elementType.isF16() || elementType.isBF16() || elementType.isInteger(32)) {
    return true;
  }
  if (elementType.isF64()) {
    double value = (*denseAttr.getValues<APFloat>().begin()).convertToDouble();
    constexpr double kMaxFloat = static_cast<double>(std::numeric_limits<float>::max());
    return std::isfinite(value) && value >= -kMaxFloat && value <= kMaxFloat;
  }
  if (elementType.isInteger(64)) {
    int64_t value = (*denseAttr.getValues<APInt>().begin()).getSExtValue();
    return value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max();
  }

  return false;
}

bool isSafeRankZeroProducer(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  if (auto constOp = dyn_cast<ConstantOp>(op)) {
    return isDvmSupportedScalarConstant(constOp);
  }
  if (isa<FullOp, CastOp>(op)) {
    return !llvm::any_of(op->getOperands(), [](Value operand) {
      if (!isRankZeroTensor(operand.getType())) {
        return false;
      }
      return !isSafeRankZeroProducer(operand.getDefiningOp());
    });
  }
  return false;
}

bool isDvmSupportedScalarOperand(Value operand) {
  if (!hasScalarMarker(operand.getType())) {
    return true;
  }
  return isSafeRankZeroProducer(operand.getDefiningOp());
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
    auto compareInputSupported = [](Operation *op) { return compareInputCheck(op); };
    auto inputCheckAll = [](Operation *op) { return inputCheck(op, {}); };
    auto inputCheckFirst = [](Operation *op) { return inputCheck(op, {0}); };
    auto castCheck = [](Operation *op) { return castCheckFunc(op); };
    auto floatOutputCheck = [](Operation *op) { return floatOutputCheckFunc(op); };
    auto floatIntOutputCheck = [](Operation *op) { return floatIntOutputCheckFunc(op); };
    auto floatIntBoolOutputCheck = [](Operation *op) { return floatIntBoolOutputCheckFunc(op); };
    auto boolOpCheck = [](Operation *op) { return boolOpCheckFunc(op); };
    auto boolTensorInputCheck = [](Operation *op) { return boolTensorInputCheckFunc(op); };
    // Should add collective_comm_op_check when support AllReduce.

    // cast op
    checkFunc_["mfuse.cast"] = {castCheck};
    // reduce sum op
    checkFunc_["mfuse.reduce_sum"] = {reduceSumCheck, inputCheckFirst};
    // cmp op
    checkFunc_["mfuse.eq"] = {boolOpCheck, compareInputSupported};
    checkFunc_["mfuse.ne"] = {boolOpCheck, compareInputSupported};
    checkFunc_["mfuse.gt"] = {boolOpCheck, compareInputSupported};
    checkFunc_["mfuse.ge"] = {boolOpCheck, compareInputSupported};
    checkFunc_["mfuse.lt"] = {boolOpCheck, compareInputSupported};
    checkFunc_["mfuse.le"] = {boolOpCheck, compareInputSupported};
    checkFunc_["mfuse.is_finite"] = {boolOpCheck, isFiniteOpCheckFunc};
    // select op
    checkFunc_["mfuse.select"] = {selectOpCheck, [](Operation *op) { return inputCheck(op, {kIndex1, kIndex2}); }};
    // float/int output ops
    checkFunc_["mfuse.add"] = {floatIntOutputCheckFunc, inputCheckAll};
    checkFunc_["mfuse.sub"] = {floatIntOutputCheckFunc, inputCheckAll};
    checkFunc_["mfuse.relu"] = {floatOutputCheck, inputCheckAll};
    checkFunc_["mfuse.mul"] = {mulOpCheck};
    checkFunc_["mfuse.maximum"] = {floatOutputCheck, inputCheckAll};
    checkFunc_["mfuse.minimum"] = {floatOutputCheck, inputCheckAll};
    checkFunc_["mfuse.neg"] = {floatIntOutputCheckFunc, inputCheckAll};
    checkFunc_["mfuse.abs"] = {floatIntOutputCheckFunc, inputCheckAll};
    checkFunc_["mfuse.logical_and"] = {boolOpCheck, boolTensorInputCheck};
    checkFunc_["mfuse.logical_or"] = {boolOpCheck, boolTensorInputCheck};
    checkFunc_["mfuse.logical_not"] = {boolOpCheck, boolTensorInputCheck};
    // Should add Assign check. There is no corresponding op in aten.
    checkFunc_["mfuse.broadcast_to"] = {floatIntBoolOutputCheck, inputCheckFirst};
    checkFunc_["mfuse.full"] = {floatIntBoolOutputCheck};
    // slice op
    checkFunc_["mfuse.slice"] = {sliceSupported, inputCheckFirst};
    // Should add StridedSlice check. There is no corresponding op in aten.
    //  matmul op
    checkFunc_["mfuse.matmul"] = {matmulOpCheck, inputCheckAll};
    checkFunc_["mfuse.batch_matmul"] = {matmulOpCheck, inputCheckAll};
    checkFunc_["mfuse.grouped_matmul"] = {groupedMatmulOpCheck, inputCheckAll};
    checkFunc_["mfuse.reshape"] = {floatIntOutputCheckFunc, inputCheckFirst};
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
    if (!outputType || !isFloatType(outputType)) {
      return false;
    }

    auto reduce = dyn_cast<ReduceSumOp>(op);
    if (!reduce) {
      return false;
    }
    // Temporary workaround for a DVM limitation. Remove this guard once DVM supports these shapes.
    if (hasLargeSemiprimeNonReduceAxis(reduce)) {
      MLOG(DEBUG) << "ReduceSum has unsupported large semiprime non-reduce axis";
      return false;
    }
    return true;
  }

  static bool isFiniteOpCheckFunc(Operation *op) {
    Type inputType = getElementType(op->getOperand(0).getType());
    return inputType && isFloatType(inputType);
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

  static bool floatIntOutputCheckFunc(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    return outputType && isFloatIntType(outputType);
  }

  static bool floatOutputCheckFunc(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    return outputType && isFloatType(outputType);
  }

  static bool floatIntBoolOutputCheckFunc(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    return outputType && (isFloatIntType(outputType) || isBoolType(outputType));
  }

  static bool boolOpCheckFunc(Operation *op) {
    Type outputType = getElementType(op->getResult(0).getType());
    return outputType && isBoolType(outputType);
  }

  static bool boolTensorInputCheckFunc(Operation *op) {
    for (Value operand : op->getOperands()) {
      Type operandType = operand.getType();
      if (hasScalarMarker(operandType) || isRankZeroTensor(operandType) || !isa<TensorType>(operandType)) {
        return false;
      }
      Type inputType = getElementType(operandType);
      if (!inputType || !isBoolType(inputType)) {
        return false;
      }
    }
    return true;
  }

  static bool mulOpCheck(Operation *op) {
    return mixTypeCheck(op, [](Type type) { return isFloatIntType(type); }, {});
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
      if (hasScalarMarker(operandType)) {
        if (!isDvmSupportedScalarOperand(operand)) {
          return false;
        }
        continue;
      }
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
      auto inputType = op->getOperand(idx).getType();
      if (hasScalarMarker(inputType)) {
        if (!isDvmSupportedScalarOperand(op->getOperand(idx))) {
          return false;
        }
        continue;
      }
      Type inputElemType = getElementType(inputType);
      if (!typeCheck(inputElemType)) {
        return false;
      }
    }
    return true;
  }

  static bool compareInputCheck(Operation *op) {
    if (op->getNumOperands() != 2) {
      return false;
    }
    Type lhsElemType = getElementType(op->getOperand(0).getType());
    if (!lhsElemType || !isFloatIntType(lhsElemType)) {
      return false;
    }

    Value rhs = op->getOperand(1);
    if (hasScalarMarker(rhs.getType())) {
      return isDvmSupportedScalarOperand(rhs);
    }
    Type rhsElemType = getElementType(rhs.getType());
    return rhsElemType && isFloatIntType(rhsElemType);
  }

  static bool isFloatType(Type type) { return type.isF32() || type.isF16() || type.isBF16(); }

  static bool isBoolType(Type type) { return type.isInteger(1); }

  /// Check if output type is float/int type
  static bool isFloatIntType(Type type) { return isFloatType(type) || type.isInteger(32); }

  std::unordered_map<std::string, std::vector<std::function<bool(Operation *)>>> checkFunc_;
};
}  // namespace

llvm::DenseSet<llvm::StringRef> DVMCluster::getClusterableOps() {
  // Clusterable operations for DVM backend.
  // Currently, we only support dvm supported operations.
  return llvm::DenseSet<llvm::StringRef>({
    "mfuse.abs",         "mfuse.add",          "mfuse.broadcast_to",
    "mfuse.cast",        "mfuse.exp",          "mfuse.log",
    "mfuse.maximum",     "mfuse.minimum",      "mfuse.mul",
    "mfuse.neg",         "mfuse.relu",         "mfuse.pow",
    "mfuse.full",        "mfuse.div",          "mfuse.real_div",
    "mfuse.reciprocal",  "mfuse.rsqrt",        "mfuse.sqrt",
    "mfuse.sub",         "mfuse.eq",           "mfuse.ne",
    "mfuse.gt",          "mfuse.ge",           "mfuse.lt",
    "mfuse.le",          "mfuse.logical_and",  "mfuse.logical_or",
    "mfuse.logical_not", "mfuse.select",       "mfuse.assign",
    "mfuse.reduce_sum",  "mfuse.is_finite",    "mfuse.reshape",
    "mfuse.floor",       "mfuse.ceil",         "mfuse.trunc",
    "mfuse.matmul",      "mfuse.batch_matmul", "mfuse.grouped_matmul",
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
