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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_TOSA_TRANSFORMS_FOLDDIMENSION_H_
#define COMPILER_INCLUDE_AKG_DIALECT_TOSA_TRANSFORMS_FOLDDIMENSION_H_

#include <memory>
#include <string>
#include <utility>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func

#define GEN_PASS_DECL_FOLDDIMENSION
#include "akg/Dialect/Tosa/Passes.h.inc"

std::unique_ptr<OperationPass<func::FuncOp>> createFoldDimensionPass();

using TensorInfoMap =
  llvm::DenseMap<Value, std::pair<std::pair<llvm::SmallDenseSet<Value>, llvm::SmallDenseSet<Value>>,  // parent, child
                                  std::pair<llvm::SmallVector<int64_t>, std::string>  // foldable info, op type
                                  >>;

using FuncArgsMap = llvm::DenseMap<Value, llvm::SmallVector<int64_t>>;

class foldDimensionAnalyser {
 public:
  foldDimensionAnalyser() {}
  ~foldDimensionAnalyser() = default;

  void analyseFoldDimension(const func::FuncOp funcOp);
  void recordFuncArgs(func::FuncOp funcOp);
  void recordTensorCanFold();

  bool foldable{true};
  llvm::DenseMap<Value, Type> tensorToBeFolded;
  FuncArgsMap funcArgsMap;

 private:
  llvm::SmallVector<int64_t> sequentialize(const llvm::SmallVector<int64_t> inputInfo) const {
    // make foldableInfo consecutive: [0,2,2,3,5,5] -> [0,1,1,2,3,3]
    llvm::SmallVector<int64_t> outputInfo;
    outputInfo.reserve(inputInfo.size());
    int64_t index = 0;
    for (size_t i = 0; i < inputInfo.size(); i++) {
      if ((i > 0) && (inputInfo[i] > inputInfo[i - 1])) {
        index++;
      }
      (void)outputInfo.emplace_back(index);
    }
    return outputInfo;
  }

  bool backtrackUpdateTensors(const Value &value, const llvm::SmallVector<int64_t> info);
  void addOrUpdateTensorInfo(Operation *op);

  void analyseElementwiseOp(Operation *op);
  void analyseSymbolicBroadcastOp(const ShapedType ty0, const ShapedType ty1);
  void analyseTensorCastOp(Operation *op);
  void analyseElemwiseBroadcastOp(Operation *op);
  void analyseBroadcastToOp(Operation *op);
  void analyseReduceOp(Operation *op);
  void analyseReshapeOp(Operation *op);
  bool checkBroadcast(Operation *op) const;

  llvm::SmallVector<int64_t> getNormalizedFlattenShape(const Value &value);

  void updatefuncArgsMap(const Value &input, const llvm::SmallVector<int64_t> foldableInfo);
  void getFoldedTypeWithSymbol(SymbolicShapeAnalysis &analysis, const ShapedType inputTy,
                               const llvm::SmallVector<int64_t> foldableInfo,
                               llvm::SmallVector<int64_t> *flattenedShape,
                               llvm::SmallVector<std::string> *flattenedSymbolShape) const;
  void getFoldedType(const ShapedType inputTy, const llvm::SmallVector<int64_t> foldableInfo,
                     llvm::SmallVector<int64_t> *flattenedShape,
                     llvm::SmallVector<int64_t> *normalizedShapeAfter) const;

  TensorInfoMap tensorInfoMap;
  llvm::SmallVector<int64_t> opFoldableInfo;
};
}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_DIALECT_TOSA_TRANSFORMS_FOLDDIMENSION_H_

