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

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mfusion/Dialect/Mfuse/Transforms/Cluster/BaseCluster.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/Utils.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"

#define DEBUG_TYPE "dvm-cluster"

namespace mlir {

namespace mfuse {

namespace {
constexpr int64_t kMaxDimSize = UINT16_MAX - UINT8_MAX;
constexpr int64_t kMinDimSize = 512;

/// Get element type from a value or type
Type getElementType(Type type) {
  if (auto tensor_type = dyn_cast<TensorType>(type)) {
    return tensor_type.getElementType();
  }
  if (auto ranked_tensor = dyn_cast<RankedTensorType>(type)) {
    return ranked_tensor.getElementType();
  }
  return Type();
}

bool isComplexDataType(Operation *op) {
  if (op->getNumResults() > 0) {
    Type output_type = op->getResult(0).getType();
    if (mlir::isa<ComplexType>(output_type)) {
      return true;
    }
  }

  if (op->getName().getStringRef() != "mfuse.cast" || op->getNumOperands() < 1) {
    return false;
  }

  // Get the output type of op's first input's defining op
  // This assumes the first operand is produced by another op
  Value input0 = op->getOperand(0);
  Operation *def_op = input0.getDefiningOp();
  if (!def_op) {
    return false;
  }

  // Get the output type of the defining op
  if (auto output_type = getElementType(def_op->getResult(0).getType())) {
    if (mlir::isa<ComplexType>(output_type)) {
      return true;
    }
  }
  return false;
}

/// DvmSupportChecker - Singleton class for checking DVM operator support.
class DvmSupportChecker {
 public:
  static DvmSupportChecker &Instance() {
    static DvmSupportChecker instance;
    return instance;
  }

  /// Check if operation is supported by DVM
  bool Check(Operation *op) const {
    // Check if operation format is supported by DVM, currently not implemented.
    if (!CheckFormat(op)) {
      return false;
    }
    std::string op_name = op->getName().getStringRef().str();

    // Find check function for this operation
    auto it = check_func_.find(op_name);
    if (it != check_func_.end()) {
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
    Type output_type = getElementType(op->getResult(0).getType());
    if (!output_type) {
      return false;  // Can't check without element type
    }
    return IsFloatType(output_type) && InputCheck(op, {});
  }

 private:
  DvmSupportChecker() { InitializeCheckFunc(); }

  // Check if operation format is supported by DVM, currently not implemented.
  bool CheckFormat(Operation *op) const { return true; }

  void InitializeCheckFunc() {
    auto input_same_type = [](Operation *op) { return IsElementTypesConsistent(op); };
    auto input_check_all = [](Operation *op) { return InputCheck(op, {}); };
    auto input_check_first = [](Operation *op) { return InputCheck(op, {0}); };
    auto cast_check = [](Operation *op) { return CastCheck(op); };
    auto int_op_check = [](Operation *op) { return IntOpCheck(op); };
    auto compare_check = [](Operation *op) { return CompareCheck(op); };
    auto transpose_op_check = [](Operation *op) { return TransposeOpCheck(op); };
    // TODO: Add collective_comm_op_check when support AllReduce.

    // cast op
    check_func_["mfuse.cast"] = {cast_check};
    // reduce sum op
    check_func_["mfuse.reduce_sum"] = {ReduceSumCheck, input_check_first};
    // cmp op
    check_func_["mfuse.eq"] = {compare_check, input_same_type};
    check_func_["mfuse.ne"] = {compare_check, input_same_type};
    check_func_["mfuse.gt"] = {compare_check, input_same_type};
    check_func_["mfuse.ge"] = {compare_check, input_same_type};
    check_func_["mfuse.lt"] = {compare_check, input_same_type};
    check_func_["mfuse.le"] = {compare_check, input_same_type};
    check_func_["mfuse.is_finite"] = {compare_check, IsFiniteOpCheck};
    // select op
    check_func_["mfuse.select"] = {SelectOpCheck, [](Operation *op) { return InputCheck(op, {1, 2}); }};
    // int op
    check_func_["mfuse.add"] = {int_op_check, input_check_all};
    check_func_["mfuse.sub"] = {int_op_check, input_check_all};
    check_func_["mfuse.mul"] = {MulOpCheck};
    check_func_["mfuse.maximum"] = {int_op_check, input_check_all};
    check_func_["mfuse.minimum"] = {int_op_check, input_check_all};
    check_func_["mfuse.neg"] = {int_op_check, input_check_all};
    check_func_["mfuse.abs"] = {int_op_check, input_check_all};
    // TODO: Add Assign check. There is no corresponding op in aten.
    check_func_["mfuse.broadcast_to"] = {int_op_check, input_check_first};
    // slice op
    check_func_["mfuse.slice"] = {SliceSupported, input_check_first};
    // TODO: Add StridedSlice check. There is no corresponding op in aten.
    //  matmul op
    check_func_["mfuse.matmul"] = {MatMulOpCheck, input_check_all};
    check_func_["mfuse.batch_matmul"] = {MatMulOpCheck, input_check_all};
    // TODO: Add GroupedMatmul check. There is no corresponding op in aten.
    // transpose op
    check_func_["mfuse.permute"] = {transpose_op_check, input_check_all};
    // collective comm op
    // TODO: Add AllReduce
    check_func_["mfuse.reshape"] = {ReshapeOpCheck};
  }

  static bool IsCastTypeSupported(Type type) {
    return type.isF16() || type.isF32() || type.isInteger(1) || type.isInteger(32) || type.isBF16();
  }

  static bool CastCheck(Operation *op) {
    Type output_type = getElementType(op->getResult(0).getType());
    Type input_type = getElementType(op->getOperand(0).getType());

    // Check input type
    if (!input_type || !output_type) {
      return false;
    }

    // Check if both input and output types are supported
    return IsCastTypeSupported(input_type) && IsCastTypeSupported(output_type);
  }

  static bool ReduceSumCheck(Operation *op) {
    // TODO: Check ReduceSum skip_mode attr for mindspore.
    Type output_type = getElementType(op->getResult(0).getType());
    return output_type && IsFloatType(output_type);
  }

  static bool CompareCheck(Operation *op) {
    Type input_type = getElementType(op->getOperand(0).getType());
    return input_type && IsFloatIntType(input_type);
  }

  static bool IsFiniteOpCheck(Operation *op) {
    Type input_type = getElementType(op->getOperand(0).getType());
    return input_type && !input_type.isInteger(32);
  }

  static bool SelectOpCheck(Operation *op) {
    // Check first operand is bool type
    Type cond_type = getElementType(op->getOperand(0).getType());
    if (!cond_type || (!cond_type.isInteger(1) && !cond_type.isSignlessInteger(1))) {
      LLVM_DEBUG(llvm::dbgs() << "Select op condition not bool\n");
      return false;
    }
    Type output_type = getElementType(op->getResult(0).getType());
    // Only support float type
    return output_type && IsFloatType(output_type);
  }

  static bool IntOpCheck(Operation *op) {
    Type output_type = getElementType(op->getResult(0).getType());
    return output_type && IsFloatIntType(output_type);
  }

  static bool MulOpCheck(Operation *op) {
    return MixTypeCheck(op, [](Type type) { return IsFloatIntType(type); }, {});
  }

  static bool TransposeOpCheck(Operation *op) {
    Type output_type = getElementType(op->getResult(0).getType());
    return output_type && (output_type.isF32() || output_type.isF16());
  }

  static bool MatMulOpCheck(Operation *op) {
    Type output_type = getElementType(op->getResult(0).getType());
    // Only support float16 and bfloat16
    if (!output_type.isF16() && !output_type.isBF16()) {
      LLVM_DEBUG(llvm::dbgs() << "MatMul op not float16/bfloat16\n");
      return false;
    }

    // Check if input or output has dynamic shape
    if (isDynamicShapeNode(op)) {
      return false;
    }

    // Check shape constraints
    return MatMulShapeCheck(op);
  }

  static bool ReshapeOpCheck(Operation *op) {
    auto node = op->getOperand(0);
    Operation *cube_op = nullptr;
    if (op->getName().getStringRef() == "mfuse.matmul" || op->getName().getStringRef() == "mfuse.batch_matmul" ||
        op->getName().getStringRef() == "mfuse.grouped_matmul") {
      cube_op = node.getDefiningOp();
    }
    return cube_op && DVMCluster::CanClusterableOp(DVMCluster::GetClusterableOps(), cube_op);
  }

  /// MatMul shape check helper
  static bool MatMulShapeCheck(Operation *op) {
    auto output_type_tensor = dyn_cast<TensorType>(op->getResult(0).getType());
    auto input1_type_tensor = dyn_cast<TensorType>(op->getOperand(0).getType());
    auto input2_type_tensor = dyn_cast<TensorType>(op->getOperand(1).getType());

    // Check tensor types
    if (!output_type_tensor || !input1_type_tensor || !input2_type_tensor) {
      return false;
    }
    auto a_shape = input1_type_tensor.getShape();
    auto b_shape = input2_type_tensor.getShape();
    auto c_shape = output_type_tensor.getShape();
    if (a_shape.back() > kMaxDimSize || b_shape.back() > kMaxDimSize) {
      return false;
    }
    if (op->getName().getStringRef() == "mfuse.matmul" && c_shape.back() <= kMinDimSize && c_shape.size() >= 2 &&
        c_shape[c_shape.size() - 2] <= kMinDimSize) {
      return false;
    }
    if (op->getName().getStringRef() == "mfuse.batch_matmul" && c_shape.size() > 4) {
      return false;
    }
    return true;
  }

  static bool SliceSupported(Operation *op) {
    constexpr size_t max_rank = 4;
    if (op->getNumOperands() < 1) {
      return false;
    }

    Type output_type = getElementType(op->getResult(0).getType());
    // Check output type is float/int
    if (!IsFloatIntType(output_type)) {
      return false;
    }

    auto output_type_tensor = dyn_cast<TensorType>(op->getResult(0).getType());
    auto input_type_tensor = dyn_cast<TensorType>(op->getOperand(0).getType());

    // Check tensor types
    if (!output_type_tensor || !input_type_tensor) {
      return false;
    }

    auto input_shape = input_type_tensor.getShape();
    if (input_shape.size() > max_rank) {
      return false;
    }
    auto output_shape = output_type_tensor.getShape();
    auto rank = output_shape.size();
    for (size_t i = 3; i < rank && rank >= i + 1; i++) {
      if (input_shape[rank - 1 - i] != output_shape[rank - 1 - i]) {
        return false;
      }
    }

    // TODO: Check StridedSlice specific: step_vector must be all 1.
    return true;
  }

  /// Check input types
  static bool InputCheck(Operation *op, const std::vector<size_t> &inputs_to_check) {
    Type output_type = getElementType(op->getResult(0).getType());
    size_t input_num = op->getNumOperands();

    std::vector<size_t> inputs;
    if (inputs_to_check.empty()) {
      for (size_t i = 0; i < input_num; ++i) {
        inputs.push_back(i);
      }
    } else {
      inputs = inputs_to_check;
    }

    for (size_t idx : inputs) {
      if (idx >= op->getNumOperands()) {
        continue;
      }
      Value operand = op->getOperand(idx);
      Type operand_type = operand.getType();
      // Skip tensor type check for non-tensor inputs
      if (!isa<TensorType>(operand_type)) {
        continue;
      }
      Type input_type = getElementType(operand_type);
      if (input_type != output_type) {
        return false;
      }
    }
    return true;
  }

  /// Check element type mix
  static bool MixTypeCheck(Operation *op, const std::function<bool(Type)> &type_check,
                           const std::vector<size_t> &inputs_to_check) {
    Type output_type = getElementType(op->getResult(0).getType());
    if (!type_check(output_type)) {
      return false;
    }

    size_t input_num = op->getNumOperands();
    std::vector<size_t> inputs;
    if (inputs_to_check.empty()) {
      for (size_t i = 0; i < input_num; ++i) {
        inputs.push_back(i);
      }
    } else {
      inputs = inputs_to_check;
    }

    for (size_t idx : inputs) {
      if (idx >= op->getNumOperands()) {
        continue;
      }
      Type input_type = getElementType(op->getOperand(idx).getType());
      if (!type_check(input_type)) {
        return false;
      }
    }
    return true;
  }

  /// Check if element types are consistent
  static bool IsElementTypesConsistent(Operation *op) {
    if (op->getNumOperands() <= 1) {
      return true;
    }

    Type first_elem_type = getElementType(op->getOperand(0).getType());
    if (!first_elem_type) {
      return false;  // Can't check without element type
    }

    for (size_t i = 1; i < op->getNumOperands(); ++i) {
      Type elem_type = getElementType(op->getOperand(i).getType());
      if (!elem_type || elem_type != first_elem_type) {
        return false;
      }
    }
    return true;
  }

  static bool IsFloatType(Type type) { return type.isF32() || type.isF16() || type.isBF16(); }

  /// Check if output type is float/int type
  static bool IsFloatIntType(Type type) { return IsFloatType(type) || type.isInteger(32); }

  std::unordered_map<std::string, std::vector<std::function<bool(Operation *)>>> check_func_;
};
}  // namespace

llvm::DenseSet<llvm::StringRef> DVMCluster::GetClusterableOps() {
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
bool DVMCluster::CanClusterableOp(const llvm::DenseSet<llvm::StringRef> &op_list, Operation *op) {
  if (op == nullptr) {
    return false;
  }
  // Check if operation is in clusterable list
  // TODO: Filter the ops with the opt level.
  std::string op_name = op->getName().getStringRef().str();
  if (op_list.find(op_name) == op_list.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Op not in cluster list: " << op_name << "\n");
    return false;
  }

  // Check if output type is complex type
  if (isComplexDataType(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Op has complex output type: " << op_name << "\n");
    return false;
  }

  // Check DVM-specific constraints
  if (!DvmSupportChecker::Instance().Check(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Op is not DVM supported: " << op_name << "\n");
    return false;
  }

  if (hasZeroShape(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Op has zero shape: " << op_name << "\n");
    return false;
  }

  // TODO: Check if inplace op has view inputs (may cause precision error)

  return true;
}

llvm::DenseSet<llvm::StringRef> DVMCluster::GetClusterableOpList() { return GetClusterableOps(); }

bool DVMCluster::IsClusterableOp(Operation *op) { return CanClusterableOp(op_list_, op); }

std::string DVMCluster::GetFusionType() { return "dvm"; }

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
#define GEN_PASS_DEF_DVMCLUSTER
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

struct DVMClusterPass : public impl::DVMClusterBase<DVMClusterPass> {
  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    DVMCluster cluster;
    if (cluster.Run(func_op)) {
      LLVM_DEBUG(llvm::dbgs() << "DVMCluster modified function: " << func_op.getName() << "\n");
    }
  }
};

}  // namespace mfuse

std::unique_ptr<Pass> createDVMClusterPass() { return std::make_unique<mfuse::DVMClusterPass>(); }

}  // namespace mlir
