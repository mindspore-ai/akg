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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_MINDSPORETOJSON_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_MINDSPORETOJSON_H_

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "llvm/Support/FileUtilities.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/ToolUtilities.h"

using json = nlohmann::json;

namespace mlir {
constexpr auto kJsonKeyOpDesc = "op_desc";
constexpr auto kJsonKeyAttr = "attr";
constexpr auto kJsonKeyInputDesc = "input_desc";
constexpr auto kJsonKeyFormat = "format";
constexpr auto kJsonKeyInferDataType = "infer_data_type";
constexpr auto kJsonKeyInferShape = "infer_shape";
constexpr auto kJsonKeyShape = "shape";
constexpr auto kJsonKeyDataType = "data_type";
constexpr auto kJsonKeyDataformat = "data_format";
constexpr auto kJsonKeyOutputDesc = "output_desc";
constexpr auto kJsonKeyName = "name";
constexpr auto kJsonKeyTensorName = "tensor_name";
constexpr auto kJsonKeyValue = "value";
constexpr auto kJsonKeyImplPath = "impl_path";
constexpr auto kJsonKeyProcess = "process";
constexpr auto kJsonKeyComposite = "composite";
constexpr auto kJsonValueComposite = "composite";
constexpr auto kJsonValueBasic = "basic";
constexpr auto kJsonKeyId = "id";
constexpr auto kJsonKeyOp = "op";
constexpr auto kJsonKeyPtrAddress = "ptr_address";
constexpr auto kJsonKeyCompositeGraph = "composite_graph";
constexpr auto kJsonKeyPlatform = "platform";
constexpr auto kJsonKeyOpFullName = "op_full_name";
constexpr auto kJsonKeyParallelFusion = "parallel_fusion";
constexpr auto kJsonKeyFusionType = "fusion_type";
constexpr auto kJsonKeySubGraph = "sub_graph";
constexpr auto kJsonKeyCoreNum = "core_num";
constexpr auto kJsonKeyTypeInfo = "type_info";
constexpr auto kJsonKeyRecomputeOps = "recompute_ops";
constexpr auto kJsonKeyBufferStitch = "buffer_stitch";
constexpr auto kJsonKeyStitchOp = "stitch_op";
constexpr auto kJsonKeyStitchAtomicOp = "stitch_atomic_op";
constexpr auto kJsonKeyVersion = "version";
constexpr auto kJsonKeyTargetInfo = "target_info";
constexpr auto kJsonKeyComputeCapability = "compute_capability";
constexpr auto kJsonKeySmCount = "sm_count";
constexpr auto kJsonKeySystem = "system";
constexpr auto kJsonKeyArch = "arch";
constexpr auto kJsonKeyCpuFeature = "feature";
constexpr auto kJsonKeyCpuType = "cpu";
constexpr auto kJsonKeyNodeName = "node_name";
constexpr auto kJsonKeyDynamicInputIndex = "dynamic_input_index";
constexpr auto kJsonKeyMultiGraph = "multi_graph";
constexpr auto kJsonKeyGraphDesc = "graph_desc";
constexpr auto kJsonKeyGraphMode = "graph_mode";
constexpr auto kJsonValueDefaultFormat = "DefaultFormat";
constexpr auto kJsonValueInt = "int";
constexpr auto kJsonValueFloat = "float";
constexpr auto kJsonValueBool = "bool";
constexpr auto kJsonValueString = "string";
constexpr auto kJsonValueUnknown = "unknown";
constexpr auto kJsonValueInput = "input";
constexpr auto kJsonValueOutput = "output";
constexpr auto kJsonKeySymbolicShape = "symbolic_shape";

std::string splitedMainFuncToJson(func::FuncOp &funcOp);
std::string mlirToJson(ModuleOp &moduleOp);
LogicalResult mlirToJsonOpt(int argc, char **argv, llvm::StringRef toolName, DialectRegistry &registry);
class JsonOpBuilder {
 public:
  JsonOpBuilder() = default;
  explicit JsonOpBuilder(Operation *newOp) : op(newOp) {}
  virtual ~JsonOpBuilder() = default;
  virtual json build();
  virtual json getAttrsJson();
  virtual json getAttrJson(NamedAttribute &attr);
  virtual json getAttrValue(Attribute &attrValue);
  virtual json getInputsJson();
  virtual json getInputJson(const Value &opnd);
  virtual json getOutputsJson();
  virtual json getOutputJson(const Value &opnd);
  virtual std::string getOpName();
  virtual json getTensorJson(const Value &opnd);
  virtual json getValueJson(const Value &opnd);
  static std::string getDataType(const Type &type);
  virtual std::string dump(const Value &value);
  virtual void addProto(const std::string name, JsonOpBuilder *builder) { protoMap[name] = builder; }
  virtual std::shared_ptr<JsonOpBuilder> clone(Operation *newOp) { return std::make_shared<JsonOpBuilder>(newOp); }
  static std::shared_ptr<JsonOpBuilder> getProto(const std::string name, Operation *newOp) {
    if (protoMap.find(name) == protoMap.end()) {
      return std::make_shared<JsonOpBuilder>(newOp);
    }
    return protoMap[name]->clone(newOp);
  }

  mindspore::MindSporeOp getMindSporeOp();
  std::string getOpAddress() const;
  json jsonListWrap(json js) const;
  json jsonListUnPack(json js) const;

 protected:
  Operation *op{nullptr};

 private:
  static std::unordered_map<std::string, JsonOpBuilder *> protoMap;
  mindspore::MindSporeOp mindsporeOp{nullptr};
};

class JsonFuncBuilder : public JsonOpBuilder {
 public:
  JsonFuncBuilder() = default;
  explicit JsonFuncBuilder(Operation *op) : JsonOpBuilder(op) { (void)getFuncOp(); }
  virtual ~JsonFuncBuilder() = default;
  json build() override;
  std::string getOpName() override;
  json getOutputsJson() override;
  json getInputsJson() override;
  json getInnerOpsJson();
  func::FuncOp getFuncOp();
  std::shared_ptr<JsonOpBuilder> opBuilderFactory(Operation *op);

 private:
  bool jumpOp(const Operation *op) const;
  SmallVector<Value, 4> inputs;
  SmallVector<Value, 4> outputs;
  func::FuncOp funcOp{nullptr};
  std::string platform{"AKG"};
  std::string process{"cuda"};
};

class MatMulOpBuilder : public JsonOpBuilder {
 public:
  explicit MatMulOpBuilder(Operation *op) : JsonOpBuilder(op) {}
  std::string getOpName() override;
  std::shared_ptr<JsonOpBuilder> clone(Operation *op) override { return std::make_shared<MatMulOpBuilder>(op); }

 private:
  static MatMulOpBuilder matmulProto;
  MatMulOpBuilder() { addProto("matmul", &matmulProto); }
};

class BatchMatMulOpBuilder : public JsonOpBuilder {
 public:
  explicit BatchMatMulOpBuilder(Operation *op) : JsonOpBuilder(op) {}
  std::string getOpName() override;
  std::shared_ptr<JsonOpBuilder> clone(Operation *op) override { return std::make_shared<BatchMatMulOpBuilder>(op); }

 private:
  static BatchMatMulOpBuilder batchMatmulProto;
  BatchMatMulOpBuilder() { addProto("batch_matmul", &batchMatmulProto); }
};

class TransposeOpBuilder : public JsonOpBuilder {
 public:
  explicit TransposeOpBuilder(Operation *op) : JsonOpBuilder(op) {}
  json getInputsJson() override;
  std::shared_ptr<JsonOpBuilder> clone(Operation *op) override { return std::make_shared<TransposeOpBuilder>(op); }

 private:
  static TransposeOpBuilder transposeProto;
  TransposeOpBuilder() { addProto("transpose", &transposeProto); }
};

class DivOpBuilder : public JsonOpBuilder {
 public:
  explicit DivOpBuilder(Operation *op) : JsonOpBuilder(op) {}
  std::string getOpName() override;
  json getInputsJson() override;
};

class UnknownOpBuilder : public JsonOpBuilder {
 public:
  explicit UnknownOpBuilder(Operation *op) : JsonOpBuilder(op) {}
  std::string getOpName() override;
  std::shared_ptr<JsonOpBuilder> clone(Operation *op) override { return std::make_shared<UnknownOpBuilder>(op); }

 private:
  static UnknownOpBuilder unKnownProto;
  UnknownOpBuilder() { addProto("unknown", &unKnownProto); }
};

}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_MINDSPORETOJSON_H_
