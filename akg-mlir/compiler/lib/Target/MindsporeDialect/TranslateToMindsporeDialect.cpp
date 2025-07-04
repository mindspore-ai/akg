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

#include <fstream>
#include <map>
#include <optional>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Target/MindsporeDialect/ToMindsporeDialect.h"
#include "akg/Utils/IOHelper.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "nlohmann/json.hpp"

using namespace llvm;
using namespace mlir;

namespace mlir {
namespace {
constexpr auto kProcess = "process";
constexpr auto kComputeCapability = "compute_capability";
constexpr auto kTargetInfo = "target_info";
constexpr auto kSymbolCalcExpr = "symbol_calc_expr";
constexpr auto kInputDesc = "input_desc";
constexpr auto kOutputDesc = "output_desc";
constexpr auto kTensorName = "tensor_name";
constexpr auto kShape = "shape";
constexpr auto kSymbolicShape = "symbolic_shape";
constexpr auto kDataType = "data_type";
constexpr auto kOpDesc = "op_desc";
constexpr auto kName = "name";
constexpr auto kAttr = "attr";
constexpr auto kValue = "value";
constexpr auto kPtrAddress = "ptr_address";
constexpr auto kOriOp = "ori_op";
constexpr auto kAxis = "axis";
constexpr auto kNewShape = "new_shape";
constexpr auto kTransposeA = "transpose_a";
constexpr auto kTransposeB = "transpose_b";
constexpr auto kInputNumTwo = 2;

class ValueNode {
 public:
  ValueNode(const std::string &, const std::string &, bool, const SmallVector<int64_t> &,
            const SmallVector<std::string> &);

  std::string tensorName;
  std::string dataType;
  bool isInput;
  SmallVector<int64_t> shape;
  SmallVector<std::string> symShape;
};

class OpNode {
 public:
  OpNode(std::string, nlohmann::json, nlohmann::json, nlohmann::json, std::string);
  OpNode(const OpNode &d);
  std::string opName;
  nlohmann::json inputDesc;
  nlohmann::json outputDesc;
  nlohmann::json attrs;
  std::string ptrAddress;
};

class MindBuilder {
 public:
  std::string moduleName;
  Operation *mlirModule;
  SmallVector<ValueNode> inputNodes;
  SmallVector<OpNode> opNodes;
  SmallVector<ValueNode> outputNodes;
  std::map<std::string, std::string> mindTypeMap;
  std::map<std::string, Value> operandList;
  std::map<std::string, nlohmann::json> funcAttributes;
  typedef void (MindBuilder::*pFunc)(OpBuilder, OpNode, SmallVector<Type>, SmallVector<Type>, SmallVector<Value>,
                                     SmallVector<NamedAttribute>);
  std::map<std::string, pFunc> mindOpFactory;
  // attrInputOpList defines the category where ops can input their attr as the second input (with type 1D index)
  SmallVector<std::string> attrInputOpList = {"Reshape",   "BroadcastTo", "ReduceMax", "ReduceMin",
                                              "ReduceSum", "ReducePrd",   "ReduceAll", "ReduceAny",
                                              "ArgMax",    "Transpose",   "Tile"};

  MindBuilder() = default;
  MindBuilder(const std::string &moduleName, const SmallVector<ValueNode> &inputNodes,
              const SmallVector<OpNode> &opNodes, const SmallVector<ValueNode> &outputNodes,
              const std::map<std::string, nlohmann::json> &funcAttributes);
  void initMindOpFactory();
  void initMindTypeMap();
  void convertToMLIR();
  void convertOpNode(OpBuilder builder, OpNode opNode);

  bool isStridedSliceWithAttr(OpNode &opNode) const;

  // helper functions
  mlir::FloatType getFloatType(std::string, OpBuilder) const;
  mlir::IntegerType getIntType(std::string, OpBuilder) const;
  SmallVector<int64_t> enableDynamicShape(
    SmallVector<int64_t>) const;  // replace dymShape -1 with minimum value of int64_t
  SmallVector<SmallVector<int64_t>> enableDynamicShape(SmallVector<SmallVector<int64_t>>);
  DenseElementsAttr buildDenseElementsAttr(OpBuilder builder, SmallVector<double>, RankedTensorType, std::string);
  RankedTensorType buildRankedTensorType(SmallVector<int64_t>, std::string, OpBuilder);
  NamedAttribute createMindsporeAttribute(OpBuilder, nlohmann::json) const;
  std::optional<DictionaryAttr> addOpSymShapeAttr(nlohmann::json inputDesc, nlohmann::json outputDesc,
                                                  MLIRContext *context) const;
  SmallVector<NamedAttribute> addFuncSymShapeAttr(SmallVector<NamedAttribute> attrs, SmallVector<ValueNode> inputNodes,
                                                  SmallVector<ValueNode> outputNodes, MLIRContext *context) const;
  bool isIndex1DAttrAsInput(std::string, int64_t);
  Value getIndexFromVector(OpBuilder builder, SmallVector<int64_t> vector);  // convert vector to 1D index tensor
  SmallVector<NamedAttribute> getDictAttrFromJson(nlohmann::json attrJson,
                                                  MLIRContext *context) const;  // get dict attrs from json
  template <typename AttrType>
  AttrType getAttrFromJson(const nlohmann::json &jsonAttr,
                           const std::string &attrName) const;  // get one attr from json
  template <typename AttrType>
  AttrType getAttrFromJson(const nlohmann::json &jsonAttr, const std::string &attrName, AttrType defaultValue) const;
  SmallVector<int64_t> getValueFromJson(nlohmann::json json) const;
  bool isConstInput(const nlohmann::json node) const;
  SmallVector<double> getConstInputValue(const nlohmann::json inputDesc, const std::string opName);

  void handleStridedSliceOpInput(OpBuilder &builder, OpNode &opNode, SmallVector<Type> &inputTys);
  void handleStridedSliceOperands(OpBuilder &builder, OpNode &opNode, SmallVector<Value> &operands,
                                  SmallVector<std::string> &operandNames);

  // functions for conversion
  void convertConstOperand(std::string, SmallVector<int64_t>, SmallVector<double>, std::string, OpBuilder);
  void convertStridedSliceOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                             SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertUnsortedSegmentSumOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                   SmallVector<Type> outputTys, SmallVector<Value> operands,
                                   SmallVector<NamedAttribute> attrs);
  void convertTransposeOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                          SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertInplaceAssignOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                              SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertConcatOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                       SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertTileOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                     SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertReshapeOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                        SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertBroadcastToOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                            SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertSliceOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                      SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertPadOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                    SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertGatherOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                       SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertSplitOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                      SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertMatMulOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                       SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertBatchMatMulOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                            SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertCastOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                     SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertAddNOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                     SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  void convertUnknownOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                        SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  template <typename TernaryOp>
  void convertTernaryOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                        SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  template <typename BinaryOp>
  void convertBinaryOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                       SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);
  template <typename UnaryOp>
  void convertUnaryOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                      SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);

  template <typename UnaryOp>
  void convertMultiOutputOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                            SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);

  template <typename ReduceOp>
  void convertReduceOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                       SmallVector<Value> operands, SmallVector<NamedAttribute> attrs);

  void unaryOpHelper(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                     SmallVector<Value> operands, SmallVector<NamedAttribute> attrs, std::string opName);
  void binaryOpHelper(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                      SmallVector<Value> operands, SmallVector<NamedAttribute> attrs, std::string opName);
  void reduceOpHelper(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys, SmallVector<Type> outputTys,
                      SmallVector<Value> operands, SmallVector<NamedAttribute> attrs, std::string opName);

  void handleOperands(OpBuilder &builder, OpNode &opNode, SmallVector<Value> &operands,
                      SmallVector<std::string> &operandNames);

  void handleOpInput(OpBuilder &builder, OpNode &opNode, SmallVector<Type> &inputTys);
};

class MindConverter {
 public:
  std::string inputFileName;
  std::string moduleName;
  nlohmann::json rawJson;
  MindBuilder builder;
  SmallVector<ValueNode> inputNodes = {};
  SmallVector<OpNode> opNodes = {};
  SmallVector<ValueNode> outputNodes = {};
  std::map<std::string, std::string> mindOpFactory;
  std::map<std::string, nlohmann::json> funcAttributes;
  std::map<std::string, SmallVector<int64_t>> fakeOutputShapes;

  std::map<std::string, std::string> outputNameToOpNameMap;
  std::map<std::string, SmallVector<int64_t>> opNameToOperandShapeMap;

  MindConverter() = default;
  explicit MindConverter(std::string);
  void parseJson();
  void parseInput();
  void parseOp();
  void parseOutput();
  MindBuilder initBuilder();
};

MindBuilder::MindBuilder(const std::string &moduleName, const SmallVector<ValueNode> &inputNodes,
                         const SmallVector<OpNode> &opNodes, const SmallVector<ValueNode> &outputNodes,
                         const std::map<std::string, nlohmann::json> &funcAttributes) {
  if (moduleName == "") {
    this->moduleName = "main";
  } else {
    this->moduleName = moduleName;
  }
  this->inputNodes = inputNodes;
  this->opNodes = opNodes;
  this->outputNodes = outputNodes;
  this->funcAttributes = funcAttributes;
}

SmallVector<int64_t> MindBuilder::enableDynamicShape(SmallVector<int64_t> oriShapes) const {
  SmallVector<int64_t> shapes = oriShapes;
  for (size_t i = 0; i < shapes.size(); i++) {
    if (shapes[i] == -1) {
      shapes[i] = ShapedType::kDynamic;
    }
  }
  return shapes;
}

ValueNode::ValueNode(const std::string &tensorName, const std::string &dtype, bool isInput,
                     const SmallVector<int64_t> &shape, const SmallVector<std::string> &symShape) {
  this->dataType = dtype;
  this->tensorName = tensorName;
  this->isInput = isInput;
  this->shape = shape;
  this->symShape = symShape;
}

OpNode::OpNode(std::string name, nlohmann::json inputDesc, nlohmann::json outputDesc, nlohmann::json attrs,
               std::string ptrAddress) {
  this->opName = name;
  this->inputDesc = inputDesc;
  this->outputDesc = outputDesc;
  this->attrs = attrs;
  this->ptrAddress = ptrAddress;
}

OpNode::OpNode(const OpNode &d1) {
  this->opName = d1.opName;
  this->inputDesc = d1.inputDesc;
  this->outputDesc = d1.outputDesc;
  this->attrs = d1.attrs;
  this->ptrAddress = d1.ptrAddress;
}

MindBuilder MindConverter::initBuilder() {
  MindBuilder opBuilder(this->moduleName, this->inputNodes, this->opNodes, this->outputNodes, this->funcAttributes);
  this->builder = opBuilder;
  return opBuilder;
}

void MindConverter::parseJson() {
  this->rawJson = DirUtils::checkAndReadJson(this->inputFileName);
  parseInput();
  parseOp();
  parseOutput();
  this->moduleName = this->rawJson.at("op");
  this->funcAttributes[kProcess] = this->rawJson.at(kProcess);
  this->funcAttributes[kComputeCapability] = "";
  if (this->rawJson.contains(kTargetInfo)) {
    auto targetInfo = this->rawJson.at(kTargetInfo);
    if (targetInfo.contains(kComputeCapability)) {
      this->funcAttributes[kComputeCapability] = targetInfo.at(kComputeCapability);
    }
  }
  if (this->rawJson.contains(kSymbolCalcExpr)) {
    this->funcAttributes[kSymbolCalcExpr] = this->rawJson.at(kSymbolCalcExpr);
  }
}

void MindConverter::parseInput() {
  for (size_t i = 0; i < this->rawJson.at(kInputDesc).size(); i++) {
    nlohmann::json inputDesc = rawJson.at(kInputDesc)[i][0];
    std::string tensorName = inputDesc.at(kTensorName);
    SmallVector<int64_t> shape = {1};
    if (inputDesc.contains(kShape)) {
      shape = inputDesc.at(kShape).get<SmallVector<int64_t>>();
    }
    SmallVector<std::string> symShape = {};
    if (inputDesc.contains(kSymbolicShape)) {
      symShape = inputDesc.at(kSymbolicShape).get<SmallVector<std::string>>();
    }
    std::string dtype = inputDesc.at(kDataType);
    if (dtype == "uint8")
      dtype = "int8";
    ValueNode node(tensorName, dtype, true, shape, symShape);
    inputNodes.push_back(node);
  }
}

void MindConverter::parseOp() {
  for (size_t i = 0; i < this->rawJson.at(kOpDesc).size(); i++) {
    nlohmann::json opDesc = rawJson.at(kOpDesc)[i];
    std::string opName = opDesc.at(kName);
    nlohmann::json attr = {};
    nlohmann::json inputDesc = opDesc.at(kInputDesc);
    nlohmann::json outputDesc;
    std::string ptrAddress = "";
    // deal with fake output
    bool fake_output = false;
    if (opDesc.contains(kAttr)) {
      attr = opDesc.at(kAttr);
    }
    for (auto kk : attr) {
      if ((kk.at(kName) == "fake_output") && kk.at(kValue) == true) {
        fake_output = true;
        break;
      }
    }
    if (fake_output) {
      outputDesc = opDesc.at(kInputDesc)[0];
      outputDesc[0].at(kTensorName) = opDesc.at(kOutputDesc)[0].at(kTensorName);
      SmallVector<int64_t> real_shape = outputDesc[0].at(kShape);
      this->fakeOutputShapes[outputDesc[0].at(kTensorName)] = real_shape;
    } else {
      outputDesc = opDesc.at(kOutputDesc);
    }
    if (opDesc.contains(kPtrAddress)) {
      ptrAddress = opDesc.at(kPtrAddress);
    }
    outputNameToOpNameMap[outputDesc[0].at(kTensorName)] = opName;
    // try to record the input shape;
    auto first_input_desc = inputDesc[0][0];
    SmallVector<int64_t> newShape = first_input_desc.at(kShape);
    opNameToOperandShapeMap[opName] = newShape;

    OpNode node(opName, inputDesc, outputDesc, attr, ptrAddress);
    opNodes.push_back(node);
  }
}

void MindConverter::parseOutput() {
  for (size_t i = 0; i < this->rawJson.at(kOutputDesc).size(); i++) {
    nlohmann::json outputDesc = rawJson.at(kOutputDesc)[i];
    std::string tensorName = outputDesc.at(kTensorName);

    std::string dtype = outputDesc.at(kDataType);
    if (dtype == "uint8")
      dtype = "int8";
    SmallVector<int64_t> shape = {};
    if (this->fakeOutputShapes.count(tensorName) != 0) {
      shape = this->fakeOutputShapes[tensorName];
    } else {
      if (!outputDesc.contains(kShape)) {
        shape = {1};
      } else {
        shape = outputDesc.at(kShape).get<SmallVector<int64_t>>();
      }
    }

    // try to get the op name
    if (outputNameToOpNameMap.find(tensorName) != outputNameToOpNameMap.end()) {
      std::string opName = outputNameToOpNameMap[tensorName];
      if (opName == "Assign" || opName == "InplaceAssign") {
        shape = opNameToOperandShapeMap[opName];
      }
    }

    SmallVector<std::string> symShape = {};
    if (outputDesc.contains(kSymbolicShape)) {
      symShape = outputDesc.at(kSymbolicShape).get<SmallVector<std::string>>();
    }
    ValueNode node(tensorName, dtype, false, shape, symShape);
    outputNodes.push_back(node);
  }
}  // namespace

void MindBuilder::initMindOpFactory() {
  this->mindOpFactory["Abs"] = &MindBuilder::convertUnaryOp<mindspore::AbsOp>;
  this->mindOpFactory["Neg"] = &MindBuilder::convertUnaryOp<mindspore::NegateOp>;
  this->mindOpFactory["Exp"] = &MindBuilder::convertUnaryOp<mindspore::ExpOp>;
  this->mindOpFactory["Log"] = &MindBuilder::convertUnaryOp<mindspore::LogOp>;
  this->mindOpFactory["Tanh"] = &MindBuilder::convertUnaryOp<mindspore::TanhOp>;
  this->mindOpFactory["ACos"] = &MindBuilder::convertUnaryOp<mindspore::AcosOp>;
  this->mindOpFactory["ASin"] = &MindBuilder::convertUnaryOp<mindspore::AsinOp>;
  this->mindOpFactory["Sin"] = &MindBuilder::convertUnaryOp<mindspore::SinOp>;
  this->mindOpFactory["Cos"] = &MindBuilder::convertUnaryOp<mindspore::CosOp>;
  this->mindOpFactory["Asinh"] = &MindBuilder::convertUnaryOp<mindspore::AsinhOp>;
  this->mindOpFactory["Acosh"] = &MindBuilder::convertUnaryOp<mindspore::AcoshOp>;
  this->mindOpFactory["Atan"] = &MindBuilder::convertUnaryOp<mindspore::AtanOp>;
  this->mindOpFactory["Rsqrt"] = &MindBuilder::convertUnaryOp<mindspore::RsqrtOp>;
  this->mindOpFactory["Floor"] = &MindBuilder::convertUnaryOp<mindspore::FloorOp>;
  this->mindOpFactory["Square"] = &MindBuilder::convertUnaryOp<mindspore::SquareOp>;
  this->mindOpFactory["Sqrt"] = &MindBuilder::convertUnaryOp<mindspore::SqrtOp>;
  this->mindOpFactory["IsInf"] = &MindBuilder::convertUnaryOp<mindspore::IsinfOp>;
  this->mindOpFactory["IsNan"] = &MindBuilder::convertUnaryOp<mindspore::IsnanOp>;
  this->mindOpFactory["Reciprocal"] = &MindBuilder::convertUnaryOp<mindspore::InvOp>;
  this->mindOpFactory["LogicalNot"] = &MindBuilder::convertUnaryOp<mindspore::LogicalNotOp>;

  this->mindOpFactory["LogicalAnd"] = &MindBuilder::convertBinaryOp<mindspore::LogicalAndOp>;
  this->mindOpFactory["LogicalOr"] = &MindBuilder::convertBinaryOp<mindspore::LogicalOrOp>;
  this->mindOpFactory["Pow"] = &MindBuilder::convertBinaryOp<mindspore::PowOp>;
  this->mindOpFactory["Add"] = &MindBuilder::convertBinaryOp<mindspore::AddOp>;
  this->mindOpFactory["Sub"] = &MindBuilder::convertBinaryOp<mindspore::SubOp>;
  this->mindOpFactory["Maximum"] = &MindBuilder::convertBinaryOp<mindspore::MaximumOp>;
  this->mindOpFactory["Minimum"] = &MindBuilder::convertBinaryOp<mindspore::MinimumOp>;
  this->mindOpFactory["Mul"] = &MindBuilder::convertBinaryOp<mindspore::MulOp>;
  this->mindOpFactory["Div"] = &MindBuilder::convertBinaryOp<mindspore::DivOp>;
  this->mindOpFactory["RealDiv"] = &MindBuilder::convertBinaryOp<mindspore::DivOp>;
  this->mindOpFactory["Equal"] = &MindBuilder::convertBinaryOp<mindspore::EqualOp>;
  this->mindOpFactory["GreaterEqual"] = &MindBuilder::convertBinaryOp<mindspore::GreaterEqualOp>;
  this->mindOpFactory["Greater"] = &MindBuilder::convertBinaryOp<mindspore::GreaterOp>;
  this->mindOpFactory["LessEqual"] = &MindBuilder::convertBinaryOp<mindspore::LessEqualOp>;
  this->mindOpFactory["Less"] = &MindBuilder::convertBinaryOp<mindspore::LessOp>;
  this->mindOpFactory["NotEqual"] = &MindBuilder::convertBinaryOp<mindspore::NotEqualOp>;
  this->mindOpFactory["Atan2"] = &MindBuilder::convertBinaryOp<mindspore::Atan2Op>;

  this->mindOpFactory["Select"] = &MindBuilder::convertTernaryOp<mindspore::SelectOp>;

  this->mindOpFactory["ReduceMax"] = &MindBuilder::convertReduceOp<mindspore::ReduceMaxOp>;
  this->mindOpFactory["ReduceMin"] = &MindBuilder::convertReduceOp<mindspore::ReduceMinOp>;
  this->mindOpFactory["ReduceSum"] = &MindBuilder::convertReduceOp<mindspore::ReduceSumOp>;
  this->mindOpFactory["ReducePrd"] = &MindBuilder::convertReduceOp<mindspore::ReduceProdOp>;
  this->mindOpFactory["ReduceAll"] = &MindBuilder::convertReduceOp<mindspore::ReduceAllOp>;
  this->mindOpFactory["ReduceAny"] = &MindBuilder::convertReduceOp<mindspore::ReduceAnyOp>;
  this->mindOpFactory["ElemAny"] = &MindBuilder::convertReduceOp<mindspore::ReduceAnyOp>;
  this->mindOpFactory["ArgMax"] = &MindBuilder::convertReduceOp<mindspore::ArgMaxOp>;

  this->mindOpFactory["BroadcastTo"] = &MindBuilder::convertBroadcastToOp;
  this->mindOpFactory["Slice"] = &MindBuilder::convertSliceOp;
  this->mindOpFactory["Tile"] = &MindBuilder::convertTileOp;
  this->mindOpFactory["Concat"] = &MindBuilder::convertConcatOp;
  this->mindOpFactory["Reshape"] = &MindBuilder::convertReshapeOp;
  this->mindOpFactory["Transpose"] = &MindBuilder::convertTransposeOp;
  this->mindOpFactory["Split"] = &MindBuilder::convertSplitOp;
  this->mindOpFactory["Pad"] = &MindBuilder::convertPadOp;
  this->mindOpFactory["Gather"] = &MindBuilder::convertGatherOp;
  this->mindOpFactory["InplaceAssign"] = &MindBuilder::convertInplaceAssignOp;
  this->mindOpFactory["Assign"] = &MindBuilder::convertInplaceAssignOp;
  this->mindOpFactory["UnsortedSegmentSum"] = &MindBuilder::convertUnsortedSegmentSumOp;
  this->mindOpFactory["StridedSlice"] = &MindBuilder::convertStridedSliceOp;
  this->mindOpFactory["AddN"] = &MindBuilder::convertAddNOp;
  this->mindOpFactory["MatMul"] = &MindBuilder::convertMatMulOp;
  this->mindOpFactory["BatchMatMul"] = &MindBuilder::convertBatchMatMulOp;
  this->mindOpFactory["Cast"] = &MindBuilder::convertCastOp;
}

void MindBuilder::initMindTypeMap() {
  this->mindTypeMap["bool"] = "int";
  this->mindTypeMap["int1"] = "int";
  this->mindTypeMap["int8"] = "int";
  this->mindTypeMap["uint8"] = "int";
  this->mindTypeMap["sint8"] = "int";
  this->mindTypeMap["int16"] = "int";
  this->mindTypeMap["int32"] = "int";
  this->mindTypeMap["int64"] = "int";

  this->mindTypeMap["float64"] = "float";
  this->mindTypeMap["float32"] = "float";
  this->mindTypeMap["float16"] = "float";
  this->mindTypeMap["bfloat16"] = "float";
}

NamedAttribute MindBuilder::createMindsporeAttribute(OpBuilder builder, nlohmann::json attr) const {
  MLIRContext *context = builder.getContext();
  std::string msAttrName = attr.at(kName);
  if (attr.at(kDataType) == "bool") {
    bool value = attr.at(kValue);
    return NamedAttribute(StringAttr::get(context, msAttrName), BoolAttr::get(context, value));
  } else if (attr.at(kDataType) == "str") {
    std::string value = attr.at(kValue);
    return NamedAttribute(StringAttr::get(context, msAttrName), StringAttr::get(context, value));
  } else if (attr.at(kDataType) == "listInt") {
    SmallVector<int64_t> values = attr.at(kValue).get<SmallVector<int64_t>>();
    if (values.empty()) {
      (void)values.emplace_back(1);
    }
    return NamedAttribute(StringAttr::get(context, msAttrName),
                          DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(values)));
  } else if (attr.at(kDataType) == "int") {
    int value = attr.at(kValue);
    return NamedAttribute(StringAttr::get(context, msAttrName), IntegerAttr::get(builder.getI64Type(), value));
  } else if (attr.at(kDataType) == "float") {
    float value = attr.at(kValue);
    return NamedAttribute(StringAttr::get(context, msAttrName), FloatAttr::get(builder.getF32Type(), value));
  } else {
    llvm::report_fatal_error(
      llvm::StringRef("Attribute type " + attr.at(kDataType).get<std::string>() + " is not supported!"));
  }
}

SmallVector<NamedAttribute> MindBuilder::getDictAttrFromJson(nlohmann::json attrJson, MLIRContext *context) const {
  SmallVector<NamedAttribute> ret;
  for (auto x : attrJson.items()) {
    (void)ret.emplace_back(StringAttr::get(context, x.key()), StringAttr::get(context, x.value().get<std::string>()));
  }
  return ret;
}

SmallVector<NamedAttribute> MindBuilder::addFuncSymShapeAttr(SmallVector<NamedAttribute> attrs,
                                                             SmallVector<ValueNode> inputNodes,
                                                             SmallVector<ValueNode> outputNodes,
                                                             MLIRContext *context) const {
  SmallVector<NamedAttribute> fSymbol;
  for (size_t i = 0; i < inputNodes.size(); i++) {
    if (inputNodes[i].symShape.size() != 0) {
      llvm::SmallVector<Attribute> symAttr;
      for (auto symbol : inputNodes[i].symShape) {
        (void)symAttr.emplace_back(StringAttr::get(context, symbol));
      }
      (void)fSymbol.emplace_back(StringAttr::get(context, "input_" + std::to_string(i)),
                                 ArrayAttr::get(context, symAttr));
    }
  }
  for (size_t i = 0; i < outputNodes.size(); i++) {
    if (outputNodes[i].symShape.size() != 0) {
      llvm::SmallVector<Attribute> symAttr;
      for (auto symbol : outputNodes[i].symShape) {
        (void)symAttr.emplace_back(StringAttr::get(context, symbol));
      }
      (void)fSymbol.emplace_back(StringAttr::get(context, "output_" + std::to_string(i)),
                                 ArrayAttr::get(context, symAttr));
    }
  }
  if (fSymbol.size() != 0) {
    (void)attrs.emplace_back(StringAttr::get(context, getFrontendSymbolAttrName()),
                             DictionaryAttr::get(context, ArrayRef<NamedAttribute>(fSymbol)));
  }
  return attrs;
}

void MindBuilder::convertToMLIR() {
  MLIRContext *context = this->mlirModule->getContext();
  OpBuilder builder(context);
  if (this->funcAttributes.count(kSymbolCalcExpr) != 0) {
    auto symbolCalcExpr = getDictAttrFromJson(this->funcAttributes[kSymbolCalcExpr], context);
    this->mlirModule->setAttr("mindspore.symbol_calc_expr",
                              DictionaryAttr::get(context, ArrayRef<NamedAttribute>(symbolCalcExpr)));
  }
  SmallVector<Type> inputs;
  SmallVector<Type> outputs;
  for (size_t i = 0; i < this->inputNodes.size(); i++) {
    Type temp =
      buildRankedTensorType(enableDynamicShape(this->inputNodes[i].shape), this->inputNodes[i].dataType, builder);
    (void)inputs.emplace_back(temp);
  }
  for (size_t i = 0; i < this->outputNodes.size(); i++) {
    Type temp =
      buildRankedTensorType(enableDynamicShape(this->outputNodes[i].shape), this->outputNodes[i].dataType, builder);
    (void)outputs.emplace_back(temp);
  }
  builder.setInsertionPointToStart(&this->mlirModule->getRegion(0).front());
  FunctionType funcTy = builder.getFunctionType(inputs, outputs);

  SmallVector<NamedAttribute> funcAttrs;
  funcAttrs = addFuncSymShapeAttr(funcAttrs, this->inputNodes, this->outputNodes, context);
  (void)funcAttrs.emplace_back(NamedAttribute(StringAttr::get(context, "mindspore_kernel"), UnitAttr::get(context)));
  (void)funcAttrs.emplace_back(NamedAttribute(
    StringAttr::get(context, kProcess), StringAttr::get(context, this->funcAttributes[kProcess].get<std::string>())));
  (void)funcAttrs.emplace_back(
    NamedAttribute(StringAttr::get(context, kComputeCapability),
                   StringAttr::get(context, this->funcAttributes[kComputeCapability].get<std::string>())));

  func::FuncOp function = builder.create<func::FuncOp>(UnknownLoc::get(context), this->moduleName, funcTy, funcAttrs);
  Block *entryBody = function.addEntryBlock();
  // set insert point in the function block
  builder.setInsertionPointToStart(entryBody);
  for (size_t i = 0; i < this->inputNodes.size(); i++) {
    this->operandList[this->inputNodes[i].tensorName] = entryBody->getArgument(i);
  }
  // convert ops
  for (size_t i = 0; i < this->opNodes.size(); i++) {
    convertOpNode(builder, opNodes[i]);
  }
  // create return op
  SmallVector<Value> outputOperands;
  for (size_t i = 0; i < this->outputNodes.size(); i++) {
    (void)outputOperands.emplace_back(this->operandList[this->outputNodes[i].tensorName]);
  }

  (void)builder.create<func::ReturnOp>(UnknownLoc::get(context), outputOperands);
}

bool MindBuilder::isIndex1DAttrAsInput(std::string opName, int64_t inputIdx) {
  if ((llvm::is_contained(this->attrInputOpList, opName)) && inputIdx == 1) {
    return true;
  }
  return false;
}

template <typename AttrType>
AttrType MindBuilder::getAttrFromJson(const nlohmann::json &jsonAttr, const std::string &attrName) const {
  for (auto attr : jsonAttr) {
    if (attr.at(kName) == attrName) {
      return attr.at(kValue).get<AttrType>();
    }
  }
  llvm_unreachable("cannot find attribute");
}

template <typename AttrType>
AttrType MindBuilder::getAttrFromJson(const nlohmann::json &jsonAttr, const std::string &attrName,
                                      AttrType defaultValue) const {
  for (auto attr : jsonAttr) {
    if (attr.at(kName) == attrName) {
      return attr.at(kValue).get<AttrType>();
    }
  }
  return defaultValue;
}

bool MindBuilder::isConstInput(const nlohmann::json node) const {
  if (node.contains(kShape)) {
    auto shape = node.at(kShape);
    if (shape.is_array() && shape.size() == 1 && shape[0] == 1) {
      return (node.contains(kValue) && node.at(kValue).is_number());
    }
  }
  return false;
}

SmallVector<double> MindBuilder::getConstInputValue(const nlohmann::json node, const std::string opName) {
  SmallVector<double> value;
  if (node.at(kValue).is_number()) {
    value.push_back(node.at(kValue));
  } else if (node.at(kValue).is_array()) {
    value = node.at(kValue).get<SmallVector<double>>();
  } else {
    llvm::report_fatal_error(llvm::StringRef("Unexpected value from const input of op: ") + opName);
  }
  return value;
}

std::optional<DictionaryAttr> MindBuilder::addOpSymShapeAttr(nlohmann::json inputDesc, nlohmann::json outputDesc,
                                                             MLIRContext *context) const {
  SmallVector<NamedAttribute> opSymbol;
  for (size_t i = 0; i < inputDesc.size(); i++) {
    if (inputDesc[i][0].contains(kSymbolicShape)) {
      llvm::SmallVector<Attribute> symAttr;
      for (auto symbol : inputDesc[i][0].at(kSymbolicShape).get<SmallVector<std::string>>()) {
        (void)symAttr.emplace_back(StringAttr::get(context, symbol));
      }
      if (symAttr.size() != 0) {
        (void)opSymbol.emplace_back(StringAttr::get(context, "input_" + std::to_string(i)),
                                    ArrayAttr::get(context, symAttr));
      }
    }
  }
  for (size_t i = 0; i < outputDesc.size(); i++) {
    if (outputDesc[i].contains(kSymbolicShape)) {
      llvm::SmallVector<Attribute> symAttr;
      for (auto symbol : outputDesc[i].at(kSymbolicShape).get<SmallVector<std::string>>()) {
        (void)symAttr.emplace_back(StringAttr::get(context, symbol));
      }
      if (symAttr.size() != 0) {
        (void)opSymbol.emplace_back(StringAttr::get(context, "output_" + std::to_string(i)),
                                    ArrayAttr::get(context, symAttr));
      }
    }
  }
  if (opSymbol.size()) {
    return DictionaryAttr::get(context, ArrayRef<NamedAttribute>(opSymbol));
  }
  return std::nullopt;
}

bool MindBuilder::isStridedSliceWithAttr(OpNode &opNode) const { return opNode.inputDesc.size() == 1; }

void MindBuilder::handleStridedSliceOpInput(OpBuilder &builder, OpNode &opNode, SmallVector<Type> &opInputTesnors) {
  constexpr auto kStridedSliceInputNum4 = 4;
  assert(opNode.inputDesc.size() == kStridedSliceInputNum4);
  // we only take the first input as the StrideSlice operands, the remain inpus we treat as attrs;
  std::string opInputType = opNode.inputDesc[0][0].at(kDataType);
  SmallVector<int64_t> opInputShape = opNode.inputDesc[0][0].at(kShape);
  Type temp = buildRankedTensorType(enableDynamicShape(opInputShape), opInputType, builder);
  (void)opInputTesnors.emplace_back(temp);
}

void MindBuilder::handleOpInput(OpBuilder &builder, OpNode &opNode, SmallVector<Type> &inputTys) {
  // special handle
  if (opNode.opName == "Concat") {
    assert(opNode.inputDesc.size() == 1);
    for (size_t i = 0; i < opNode.inputDesc.size(); i++) {
      for (size_t j = 0; j < opNode.inputDesc[i].size(); j++) {
        std::string inputType = opNode.inputDesc[i][j].at(kDataType);
        SmallVector<int64_t> inputShape = opNode.inputDesc[i][j].at(kShape);
        Type temp = buildRankedTensorType(enableDynamicShape(inputShape), inputType, builder);
        (void)inputTys.emplace_back(temp);
      }
    }
  } else if (opNode.opName == "StridedSlice" && !isStridedSliceWithAttr(opNode)) {
    handleStridedSliceOpInput(builder, opNode, inputTys);
  } else {
    for (size_t i = 0; i < opNode.inputDesc.size(); i++) {
      std::string inputType = opNode.inputDesc[i][0].at(kDataType);
      if (inputType == "uint8")
        inputType = "int8";
      SmallVector<int64_t> inputShape = opNode.inputDesc[i][0].at(kShape);
      Type temp = buildRankedTensorType(enableDynamicShape(inputShape), inputType, builder);
      (void)inputTys.emplace_back(temp);
    }
  }
}

void MindBuilder::handleStridedSliceOperands(OpBuilder &builder, OpNode &opNode, SmallVector<Value> &operands,
                                             SmallVector<std::string> &operandNames) {
  // only try to get the first input as operand;
  for (size_t i = 0; i < 1; i++) {
    if (isIndex1DAttrAsInput(opNode.opName, i)) {
      continue;  // Input Index1D represents an attribute, e.g. newshape attr in reshape op
    }
    nlohmann::json currInput = opNode.inputDesc[i][0];
    operandNames.push_back(currInput.at(kTensorName));
    if (this->operandList.count(operandNames[i]) == 0) {  // const input
      SmallVector<int64_t> operandShape = currInput.at(kShape);
      SmallVector<double> operandValue;
      assert(operandShape.size() != 0);
      if (currInput.at(kValue).is_number()) {
        operandValue.push_back(currInput.at(kValue));
      } else if (currInput.at(kValue).is_array()) {
        operandValue = currInput.at(kValue).get<SmallVector<double>>();
      } else {
        llvm::report_fatal_error(llvm::StringRef("Unexpected value from const input of op: ") + opNode.opName);
      }
      std::string operandDataType = currInput.at(kDataType);
      convertConstOperand(operandNames[i], operandShape, operandValue, operandDataType, builder);
    }
    (void)operands.emplace_back(this->operandList[operandNames[i]]);
  }
}

void MindBuilder::handleOperands(OpBuilder &builder, OpNode &opNode, SmallVector<Value> &operands,
                                 SmallVector<std::string> &operandNames) {
  if (opNode.opName == "Concat") {
    assert(opNode.inputDesc.size() == 1);
    for (size_t i = 0; i < opNode.inputDesc.size(); i++) {
      for (size_t j = 0; j < opNode.inputDesc[i].size(); j++) {
        if (isIndex1DAttrAsInput(opNode.opName, j)) {
          continue;  // Input Index1D represents an attribute, e.g. newshape attr in reshape op
        }
        nlohmann::json currInput = opNode.inputDesc[i][j];
        operandNames.push_back(currInput.at(kTensorName));
        if (this->operandList.count(operandNames[j]) == 0) {  // const input
          SmallVector<int64_t> shape = currInput.at(kShape);
          assert(shape.size() != 0);
          SmallVector<double> value = getConstInputValue(currInput, opNode.opName);
          convertConstOperand(operandNames[j], shape, value, currInput.at(kDataType), builder);
        }
        (void)operands.emplace_back(this->operandList[operandNames[j]]);
      }
    }
  } else if (opNode.opName == "StridedSlice" && !isStridedSliceWithAttr(opNode)) {
    handleStridedSliceOperands(builder, opNode, operands, operandNames);
  } else {
    for (size_t i = 0; i < opNode.inputDesc.size(); i++) {
      if (isIndex1DAttrAsInput(opNode.opName, i)) {
        continue;  // Input Index1D represents an attribute, e.g. newshape attr in reshape op
      }
      nlohmann::json currInput = opNode.inputDesc[i][0];
      operandNames.push_back(currInput.at(kTensorName));
      if (this->operandList.count(operandNames[i]) == 0) {  // const input
        SmallVector<int64_t> shape = currInput.at(kShape);
        assert(shape.size() != 0);
        SmallVector<double> value = getConstInputValue(currInput, opNode.opName);
        std::string dataType = currInput.at(kDataType);
        if (dataType == "uint8")
          dataType = "int8";
        convertConstOperand(operandNames[i], shape, value, dataType, builder);
      }
      (void)operands.emplace_back(this->operandList[operandNames[i]]);
    }
  }
}

void MindBuilder::convertOpNode(OpBuilder builder, OpNode opNode) {
  MLIRContext *context = builder.getContext();

  SmallVector<Type> inputTys;
  SmallVector<Type> outputTys;
  handleOpInput(builder, opNode, inputTys);

  for (size_t i = 0; i < opNode.outputDesc.size(); i++) {
    std::string dtype = opNode.outputDesc[i].at(kDataType);
    if (dtype == "uint8")
      dtype = "int8";
    llvm::SmallVector<int64_t> shape = opNode.outputDesc[i].at(kShape);
    Type temp = buildRankedTensorType(enableDynamicShape(shape), dtype, builder);
    (void)outputTys.emplace_back(temp);
  }
  // prepare operands
  SmallVector<Value> operands;
  SmallVector<std::string> operandNames;
  handleOperands(builder, opNode, operands, operandNames);

  // prepare all attrs
  SmallVector<NamedAttribute> allAttrs;
  // append symbol attrs
  std::optional<DictionaryAttr> symbolAttrs = addOpSymShapeAttr(opNode.inputDesc, opNode.outputDesc, context);
  if (symbolAttrs != std::nullopt) {
    (void)allAttrs.emplace_back(StringAttr::get(context, getFrontendSymbolAttrName()), *symbolAttrs);
  }
  // append ms attrs
  SmallVector<NamedAttribute> msAttrs;
  for (auto attr : opNode.attrs) {
    (void)msAttrs.emplace_back(createMindsporeAttribute(builder, attr));
  }
  if (msAttrs.size() != 0) {
    (void)allAttrs.emplace_back(StringAttr::get(context, "ms_attr"),
                                DictionaryAttr::get(context, ArrayRef<NamedAttribute>(msAttrs)));
  }
  // append ptr address
  if (opNode.ptrAddress != "") {
    (void)allAttrs.emplace_back(
      NamedAttribute(StringAttr::get(context, kPtrAddress), StringAttr::get(context, opNode.ptrAddress)));
  }
  std::string opName = opNode.opName;
  if (this->mindOpFactory.count(opName) == 0) {
    MindBuilder::convertUnknownOp(builder, opNode, inputTys, outputTys, operands, allAttrs);
  } else {
    (this->*mindOpFactory[opName])(builder, opNode, inputTys, outputTys, operands, allAttrs);
  }
}

template <typename UnaryOp>
void MindBuilder::convertUnaryOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                 SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  if (static_cast<int>(operands.size()) != 1) {
    llvm::report_fatal_error(llvm::StringRef("Error occurs when converting json to mlir: op name: " + opNode.opName +
                                             " in UnaryOp, input operand number must be 1\n"));
  }
  auto op = builder.create<UnaryOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op.getResult();
}

template <typename UnaryOp>
void MindBuilder::convertMultiOutputOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                       SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  if (static_cast<int>(operands.size()) != 1) {
    llvm::report_fatal_error(llvm::StringRef("Error occurs when converting json to mlir: op name: " + opNode.opName +
                                             " in UnaryOp, input operand number must be 1\n"));
  }
  auto op = builder.create<UnaryOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  auto res = op.getResult();

  assert(res.size() == opNode.outputDesc.size());
  for (size_t i = 0; i < res.size(); i++) {
    this->operandList[opNode.outputDesc[i].at(kTensorName)] = res[i];
  }
}

template <typename BinaryOp>
void MindBuilder::convertBinaryOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                  SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  constexpr auto kBinaryInputNum = 2;
  if (static_cast<int>(operands.size()) != kBinaryInputNum) {
    llvm::report_fatal_error(llvm::StringRef("Error occurs when converting json to mlir: op name: " + opNode.opName +
                                             " in op, input operand number must be 2\n"));
  }
  auto op = builder.create<BinaryOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op.getResult();
}

void MindBuilder::convertAddNOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  this->operandList[opNode.outputDesc[0].at(kTensorName)] =
    builder.create<mindspore::AddNOp>(UnknownLoc::get(context), outputTys, operands, attrs);
}

void MindBuilder::convertUnknownOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                   SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  (void)attrs.emplace_back(StringAttr::get(context, kOriOp), StringAttr::get(context, opNode.opName));
  this->operandList[opNode.outputDesc[0].at(kTensorName)] =
    builder.create<mindspore::UnknownOp>(UnknownLoc::get(context), outputTys, operands, attrs);
}

template <typename TernaryOp>
void MindBuilder::convertTernaryOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                   SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  constexpr auto kTernaryInputNum = 3;
  if (static_cast<int>(operands.size()) != kTernaryInputNum) {
    llvm::report_fatal_error(llvm::StringRef("Error occurs when converting json to mlir: op name: " + opNode.opName +
                                             " in ternary_op, input operand number must be 3\n"));
  }

  auto op = builder.create<TernaryOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op.getResult();
}

void MindBuilder::convertTransposeOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                     SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  std::string permName = opNode.opName + "_perm";
  auto permValue = (opNode.inputDesc.size() == 2) ? opNode.inputDesc[1][0].at(kValue).get<SmallVector<double>>()
                                                  : getAttrFromJson<SmallVector<double>>(opNode.attrs, "perm");
  SmallVector<int64_t> permShape;
  permShape.push_back(permValue.size());
  convertConstOperand(permName, permShape, permValue, "int64", builder);
  (void)operands.emplace_back(this->operandList[permName]);
  auto op = builder.create<mindspore::TransposeOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op.getResult();
}

void MindBuilder::convertUnsortedSegmentSumOp(OpBuilder builder, OpNode opNode, SmallVector<Type>,
                                              SmallVector<Type> outputTys, SmallVector<Value> operands,
                                              SmallVector<NamedAttribute> attrs) {
  constexpr auto kNumSegments = "numSegments";
  MLIRContext *context = builder.getContext();
  SmallVector<int64_t> numSegments;
  int64_t numSegment = 0;
  bool multiDimSegs = false;
  if ((opNode.inputDesc.size() > kInputNumTwo) && isConstInput(opNode.inputDesc[kInputNumTwo][0])) {
    numSegment = opNode.inputDesc[kInputNumTwo][0].at(kValue).get<int64_t>();
  } else {
    for (auto attr : opNode.attrs) {
      if (attr.at(kName) == kNumSegments) {
        if (attr.at(kDataType) == "int") {
          numSegment = attr.at(kValue).get<int64_t>();
        } else if (attr.at(kDataType) == "listInt") {
          numSegments = attr.at(kValue).get<SmallVector<int64_t>>();
          multiDimSegs = true;
        }
      }
    }
  }
  if (multiDimSegs) {
    (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kNumSegments),
                                            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(numSegments))));
  } else {
    if (numSegment == 0) {
      llvm::report_fatal_error(llvm::StringRef("UnsortedSegmentSumOp must have attr: numSegments"));
    }
    (void)attrs.emplace_back(
      NamedAttribute(StringAttr::get(context, kNumSegments), IntegerAttr::get(builder.getI64Type(), numSegment)));
  }
  if (opNode.inputDesc.size() > kInputNumTwo) {
    SmallVector<Value> newOperands(operands.begin(), std::next(operands.begin(), kInputNumTwo));
    operands = newOperands;
  }
  auto op = builder.create<mindspore::UnsortedSegmentSumOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op.getResult();
}

SmallVector<int64_t> MindBuilder::getValueFromJson(nlohmann::json operand) const {
  SmallVector<int64_t> newValue;
  auto shape = operand.at(kShape);
  assert(shape.size() != 0);
  if (shape.size() == 1 && shape.back() == 1) {
    (void)newValue.emplace_back(operand.at(kValue));
  } else {
    newValue = operand.at(kValue).get<SmallVector<int64_t>>();
  }
  return newValue;
}

void MindBuilder::convertStridedSliceOp(OpBuilder builder, OpNode opNode, SmallVector<Type>,
                                        SmallVector<Type> outputTys, SmallVector<Value> operands,
                                        SmallVector<NamedAttribute> attrs) {
  constexpr auto kBegin = "begin";
  constexpr auto kEnd = "end";
  constexpr auto kStrides = "strides";
  constexpr auto kBeginMask = "beginMask";
  constexpr auto kEndMask = "endMask";
  constexpr auto kEllipsisMask = "ellipsisMask";
  constexpr auto kNewAxisMask = "newAxisMask";
  constexpr auto kShrinkAxisMask = "shrinkAxisMask";
  constexpr auto kStartIdx = 1;
  constexpr auto kEndIdx = 2;
  constexpr auto kStridesIdx = 3;
  MLIRContext *context = builder.getContext();
  SmallVector<int64_t> start, end, strides;
  if (isStridedSliceWithAttr(opNode)) {
    start = getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kBegin);
    end = getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kEnd);
    strides = getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kStrides);
  } else {
    start = getValueFromJson(opNode.inputDesc[kStartIdx][0]);
    end = getValueFromJson(opNode.inputDesc[kEndIdx][0]);
    strides = getValueFromJson(opNode.inputDesc[kStridesIdx][0]);
  }

  for (size_t i = 0; i < end.size(); i++) {
    if (end[i] < 0) {
      end[i] = opNode.inputDesc[0][0].at(kShape).get<SmallVector<int64_t>>()[i] + end[i];
    }
  }

  auto beginMask = getAttrFromJson<int64_t>(opNode.attrs, kBeginMask, 0);
  auto endMask = getAttrFromJson<int64_t>(opNode.attrs, kEndMask, 0);
  auto ellipsisMask = getAttrFromJson<int64_t>(opNode.attrs, kEllipsisMask, 0);
  auto newAxisMask = getAttrFromJson<int64_t>(opNode.attrs, kNewAxisMask, 0);
  auto shrinkAxisMask = getAttrFromJson<int64_t>(opNode.attrs, kShrinkAxisMask, 0);
  (void)attrs.emplace_back(StringAttr::get(context, "start"),
                           DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(start)));
  (void)attrs.emplace_back(StringAttr::get(context, kEnd), DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(end)));
  (void)attrs.emplace_back(StringAttr::get(context, kStrides),
                           DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(strides)));
  (void)attrs.emplace_back(StringAttr::get(context, kBeginMask), IntegerAttr::get(builder.getI64Type(), beginMask));
  (void)attrs.emplace_back(StringAttr::get(context, kEndMask), IntegerAttr::get(builder.getI64Type(), endMask));
  (void)attrs.emplace_back(StringAttr::get(context, kEllipsisMask),
                           IntegerAttr::get(builder.getI64Type(), ellipsisMask));
  (void)attrs.emplace_back(StringAttr::get(context, kNewAxisMask), IntegerAttr::get(builder.getI64Type(), newAxisMask));
  (void)attrs.emplace_back(StringAttr::get(context, kShrinkAxisMask),
                           IntegerAttr::get(builder.getI64Type(), shrinkAxisMask));
  auto op = builder.create<mindspore::Strided_SliceOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op.getResult();
}

void MindBuilder::convertSliceOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                 SmallVector<Type> outputTys, SmallVector<Value> operands,
                                 SmallVector<NamedAttribute> attrs) {
  constexpr auto kBegin = "begin";
  constexpr auto kSize = "size";
  MLIRContext *context = builder.getContext();
  auto begin = getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kBegin);
  auto size = getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kSize);
  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, kBegin), DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(begin))));
  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, kSize), DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(size))));
  convertUnaryOp<mindspore::SliceOp>(builder, opNode, inputTys, outputTys, operands, attrs);
}

void MindBuilder::convertInplaceAssignOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                         SmallVector<Type>, SmallVector<Value> operands,
                                         SmallVector<NamedAttribute> attrs) {
  SmallVector<Value> takeOp;
  (void)takeOp.emplace_back(operands[0]);
  (void)takeOp.emplace_back(operands[1]);
  SmallVector<Type> giveOp;
  (void)giveOp.emplace_back(inputTys[0]);
  auto op = builder.create<mindspore::InplaceAssignOp>(UnknownLoc::get(builder.getContext()), giveOp, takeOp, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op.getResult();
}

void MindBuilder::convertConcatOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                  SmallVector<Type> outputTys, SmallVector<Value> operands,
                                  SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  auto axis = getAttrFromJson<int64_t>(opNode.attrs, kAxis, 0);
  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, kAxis), IntegerAttr::get(builder.getI64Type(), axis)));
  convertBinaryOp<mindspore::ConcatOp>(builder, opNode, inputTys, outputTys, operands, attrs);
}

void MindBuilder::convertTileOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                SmallVector<Type> outputTys, SmallVector<Value> operands,
                                SmallVector<NamedAttribute> attrs) {
  constexpr auto kMultiples = "multiples";
  MLIRContext *context = builder.getContext();
  auto multiples = (opNode.inputDesc.size() == 2) ? opNode.inputDesc[1][0].at(kValue).get<SmallVector<int64_t>>()
                                                  : getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kMultiples);
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kMultiples),
                                          DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(multiples))));
  convertUnaryOp<mindspore::TileOp>(builder, opNode, inputTys, outputTys, operands, attrs);
}

void MindBuilder::convertBroadcastToOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                       SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();

  bool existAttr = false;
  for (auto attr : opNode.attrs)
    if (attr.at(kName) == kShape)
      existAttr = true;

  ArrayRef<int64_t> newShape;
  if (existAttr)
    newShape = ArrayRef<int64_t>(getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kShape, {1}));
  else
    newShape = outputTys[0].cast<ShapedType>().getShape();

  if (newShape.size() == 0) {
    newShape = {1};
    emitWarning(UnknownLoc::get(context)) << "cannot find newshape in BroadcastToOp, take default value 1\n";
  }

  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, kNewShape), DenseI64ArrayAttr::get(context, newShape)));
  auto op_result = builder.create<mindspore::BroadcastToOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op_result;
}

void MindBuilder::convertPadOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                               SmallVector<Type> outputTys, SmallVector<Value> operands,
                               SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  auto padding = getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, "padding");
  auto value = getAttrFromJson<int64_t>(opNode.attrs, kValue, 0);
  auto mode = getAttrFromJson<std::string>(opNode.attrs, "mode", "constant");
  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, "padding"), DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(padding))));
  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, kValue), IntegerAttr::get(builder.getI64Type(), value)));
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, "mode"), StringAttr::get(context, mode)));
  convertUnaryOp<mindspore::PadOp>(builder, opNode, inputTys, outputTys, operands, attrs);
}

void MindBuilder::convertGatherOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                  SmallVector<Type> outputTys, SmallVector<Value> operands,
                                  SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  auto axis = 0;
  auto inputNum = opNode.inputDesc.size();
  if (inputNum > kInputNumTwo && isConstInput(opNode.inputDesc[kInputNumTwo][0])) {
    axis = opNode.inputDesc[kInputNumTwo][0].at(kValue).get<int64_t>();
  } else if (inputNum == kInputNumTwo) {
    axis = getAttrFromJson<int64_t>(opNode.attrs, kAxis, 0);
  }
  auto batch_dims = getAttrFromJson<int64_t>(opNode.attrs, "batch_dims");
  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, kAxis), IntegerAttr::get(builder.getI64Type(), axis)));
  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, "batch_dims"), IntegerAttr::get(builder.getI64Type(), batch_dims)));
  if (inputNum > kInputNumTwo) {
    SmallVector<Value> newOperands(operands.begin(), std::next(operands.begin(), kInputNumTwo));
    convertBinaryOp<mindspore::GatherOp>(builder, opNode, inputTys, outputTys, newOperands, attrs);
  } else {
    convertBinaryOp<mindspore::GatherOp>(builder, opNode, inputTys, outputTys, operands, attrs);
  }
}

void MindBuilder::convertSplitOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                 SmallVector<Type> outputTys, SmallVector<Value> operands,
                                 SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  SmallVector<int64_t> split_size_or_sections;
  auto axis = getAttrFromJson<int64_t>(opNode.attrs, kAxis, 0);
  for (auto t_attr : opNode.attrs) {
    if (t_attr.at(kName) == "split_size_or_sections") {
      split_size_or_sections = t_attr.at(kValue).get<SmallVector<int64_t>>();
    }
    if (t_attr.at(kName) == kAxis) {
      axis = t_attr.at(kValue);
    }
  }
  if (axis != -1) {
    (void)attrs.emplace_back(
      NamedAttribute(StringAttr::get(context, kAxis), IntegerAttr::get(builder.getI64Type(), axis)));
  }
  if (split_size_or_sections.size() == 1) {
    (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, "split_size_or_sections"),
                                            IntegerAttr::get(builder.getI64Type(), split_size_or_sections[0])));
  } else {
    (void)attrs.emplace_back(
      NamedAttribute(StringAttr::get(context, "split_size_or_sections"),
                     DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(split_size_or_sections))));
  }
  convertMultiOutputOp<mindspore::SplitOp>(builder, opNode, inputTys, outputTys, operands, attrs);
}

void MindBuilder::convertReshapeOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                   SmallVector<Type> outputTys, SmallVector<Value> operands,
                                   SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  constexpr auto kReshapeInputNum2 = 2;
  constexpr auto kMaxUnknownDim = 2;
  if (opNode.inputDesc.size() == kReshapeInputNum2) {
    nlohmann::json sec_input = opNode.inputDesc[1][0];
    if (sec_input.contains(kValue)) {
      auto newValue = sec_input.at(kValue);
      SmallVector<int64_t> newShape;
      assert(newValue.size() != 0);
      if (newValue.size() == 1) {
        auto value = newValue.get<int64_t>();
        (void)newShape.emplace_back(value);
      } else {
        auto values = newValue.get<SmallVector<int64_t>>();
        newShape = values;
      }
      int cnt = llvm::count(newShape, -1);
      if (cnt < kMaxUnknownDim) {
        (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kNewShape),
                                                DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(newShape))));
        convertUnaryOp<mindspore::ReshapeOp>(builder, opNode, inputTys, outputTys, operands, attrs);
      } else {
        (void)operands.emplace_back(getIndexFromVector(builder, newShape));
        convertBinaryOp<mindspore::ReshapeOp>(builder, opNode, inputTys, outputTys, operands, attrs);
      }
    } else {
      (void)operands.emplace_back(this->operandList[sec_input.at(kTensorName)]);
      convertBinaryOp<mindspore::ReshapeOp>(builder, opNode, inputTys, outputTys, operands, attrs);
    }
  } else {
    auto newShape = getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kShape, {1});
    (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kNewShape),
                                            DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(newShape))));
    convertUnaryOp<mindspore::ReshapeOp>(builder, opNode, inputTys, outputTys, operands, attrs);
  }
}

template <typename ReduceOp>
void MindBuilder::convertReduceOp(OpBuilder builder, OpNode opNode, SmallVector<Type>, SmallVector<Type> outputTys,
                                  SmallVector<Value> operands, SmallVector<NamedAttribute> attrs) {
  constexpr auto kEnableAtomicAdd = "enable_atomic_add";
  MLIRContext *context = builder.getContext();
  SmallVector<int64_t> axes;
  constexpr auto kReduceInputNum2 = 2;
  if (opNode.inputDesc.size() == kReduceInputNum2) {
    if (opNode.inputDesc[1][0].at(kValue).is_number()) {
      auto axis = opNode.inputDesc[1][0].at(kValue).get<int64_t>();
      if (axis < 0) {
        axis = opNode.inputDesc[0][0].at(kShape).get<SmallVector<int64_t>>().size() + axis;
      }
      (void)axes.emplace_back(axis);
    } else {
      axes = opNode.inputDesc[1][0].at(kValue).get<SmallVector<int64_t>>();
    }
  } else if (opNode.opName == "ElemAny") {
    // for ElemAny Op, all axis are "reduction" types.
    int64_t rank = opNode.inputDesc[0][0].at(kShape).get<SmallVector<int64_t>>().size();
    for (int64_t i = 0; i < rank; i++) {
      (void)axes.emplace_back(i);
    }
  } else {
    axes = getAttrFromJson<SmallVector<int64_t>>(opNode.attrs, kAxis);
  }
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kOriOp), StringAttr::get(context, opNode.opName)));
  auto keepDims = (opNode.opName == "ElemAny") ? true : getAttrFromJson<bool>(opNode.attrs, "keep_dims", false);
  (void)attrs.emplace_back(
    NamedAttribute(StringAttr::get(context, kAxis), DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(axes))));
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, "keepdims"), BoolAttr::get(context, keepDims)));

  auto op = builder.create<ReduceOp>(UnknownLoc::get(context), outputTys, operands, attrs);
  this->operandList[opNode.outputDesc[0].at(kTensorName)] = op.getResult();

  bool enableAtomicAdd = (getAttrFromJson<bool>(opNode.attrs, kEnableAtomicAdd, false) || (opNode.opName == "ElemAny"));
  if (enableAtomicAdd) {
    op->getParentOp()->setAttr(kEnableAtomicAdd, BoolAttr::get(context, true));
  }
}

void MindBuilder::convertConstOperand(std::string operand_name, SmallVector<int64_t> tensorShape,
                                      SmallVector<double> tensor_value, std::string dataType, OpBuilder builder) {
  tensorShape = enableDynamicShape(tensorShape);
  MLIRContext *context = builder.getContext();
  RankedTensorType AttrType = buildRankedTensorType(tensorShape, dataType, builder);
  DenseElementsAttr attr = buildDenseElementsAttr(builder, tensor_value, AttrType, dataType);
  auto op = builder.create<mindspore::ConstOp>(UnknownLoc::get(context), AttrType, attr);
  this->operandList[operand_name] = op.getResult();
}

void MindBuilder::convertMatMulOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                  SmallVector<Type> outputTys, SmallVector<Value> operands,
                                  SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  bool transposeA = getAttrFromJson<bool>(opNode.attrs, kTransposeA, false);
  bool transposeB = getAttrFromJson<bool>(opNode.attrs, kTransposeB, false);
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kTransposeA), BoolAttr::get(context, transposeA)));
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kTransposeB), BoolAttr::get(context, transposeB)));
  convertBinaryOp<mindspore::MatMulOp>(builder, opNode, inputTys, outputTys, operands, attrs);
}

void MindBuilder::convertBatchMatMulOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                       SmallVector<Type> outputTys, SmallVector<Value> operands,
                                       SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  bool transposeA = getAttrFromJson<bool>(opNode.attrs, kTransposeA, false);
  bool transposeB = getAttrFromJson<bool>(opNode.attrs, kTransposeB, false);
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kTransposeA), BoolAttr::get(context, transposeA)));
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, kTransposeB), BoolAttr::get(context, transposeB)));
  convertBinaryOp<mindspore::BatchMatMulOp>(builder, opNode, inputTys, outputTys, operands, attrs);
}

void MindBuilder::convertCastOp(OpBuilder builder, OpNode opNode, SmallVector<Type> inputTys,
                                SmallVector<Type> outputTys, SmallVector<Value> operands,
                                SmallVector<NamedAttribute> attrs) {
  MLIRContext *context = builder.getContext();
  SmallVector<Value> newOperands;
  newOperands.push_back(operands[0]);
  std::string outputTy = opNode.outputDesc[0].at(kDataType);
  (void)attrs.emplace_back(NamedAttribute(StringAttr::get(context, "dst_type"), StringAttr::get(context, outputTy)));
  convertUnaryOp<mindspore::CastOp>(builder, opNode, inputTys, outputTys, newOperands, attrs);
}

Value MindBuilder::getIndexFromVector(OpBuilder builder, SmallVector<int64_t> vector) {
  SmallVector<Value> index;
  vector = enableDynamicShape(vector);
  for (int64_t dim : vector) {
    (void)index.emplace_back(builder.create<arith::ConstantIndexOp>(UnknownLoc::get(builder.getContext()), dim));
  }
  return builder.create<tensor::FromElementsOp>(UnknownLoc::get(builder.getContext()), index);
}

RankedTensorType MindBuilder::buildRankedTensorType(SmallVector<int64_t> shape, std::string type, OpBuilder builder) {
  std::string mindType = this->mindTypeMap[type];
  RankedTensorType ret;
  if (mindType == "int") {
    ret = RankedTensorType::get(shape, getIntType(type, builder));
  } else if (mindType == "float") {
    ret = RankedTensorType::get(shape, getFloatType(type, builder));
  } else if (mindType == "index") {
    ret = RankedTensorType::get(shape, builder.getIndexType());
  }
  return ret;
}

DenseElementsAttr MindBuilder::buildDenseElementsAttr(OpBuilder builder, SmallVector<double> value,
                                                      RankedTensorType AttrType, std::string dataType) {
  if (this->mindTypeMap.count(dataType) != 0) {
    std::string valueType = this->mindTypeMap[dataType];
    SmallVector<Attribute> values;

    if (valueType == "int") {
      for (double v : value) {
        (void)values.emplace_back(builder.getIntegerAttr(getIntType(dataType, builder), v));
      }
    } else if (valueType == "float") {
      for (double v : value) {
        (void)values.emplace_back(builder.getFloatAttr(getFloatType(dataType, builder), v));
      }
    } else if (valueType == "index") {
      for (double v : value) {
        (void)values.emplace_back(builder.getIndexAttr(v));
      }
    }
    DenseElementsAttr attr = DenseElementsAttr::get(AttrType, ArrayRef<Attribute>(values));
    return attr;
  } else {
    llvm::report_fatal_error(llvm::StringRef("Error occurs when converting json to mlir: data type is not supported"));
  }
}  // namespace

mlir::FloatType MindBuilder::getFloatType(std::string dtype, OpBuilder builder) const {
  if (dtype == "float16") {
    return builder.getF16Type();
  } else if (dtype == "float32") {
    return builder.getF32Type();
  } else if (dtype == "float64") {
    return builder.getF64Type();
  } else if (dtype == "bfloat16") {
    return builder.getBF16Type();
  } else {
    llvm::report_fatal_error(
      llvm::StringRef("Error occurs when converting json to mlir: input float type is not supported"));
  }
}

mlir::IntegerType MindBuilder::getIntType(std::string dtype, OpBuilder builder) const {
  constexpr auto kIntegerSize8 = 8;
  if (dtype == "bool") {
    return builder.getI1Type();
  } else if (dtype == "int1") {
    return builder.getI1Type();
  } else if (dtype == "int8") {
    return builder.getI8Type();
  } else if (dtype == "uint8") {
    return builder.getIntegerType(kIntegerSize8, false);
  } else if (dtype == "sint8") {
    return builder.getIntegerType(kIntegerSize8, true);
  } else if (dtype == "int16") {
    return builder.getI16Type();
  } else if (dtype == "int32") {
    return builder.getI32Type();
  } else if (dtype == "int64") {
    return builder.getI64Type();
  } else {
    llvm::report_fatal_error(
      llvm::StringRef("Error occurs when converting json to mlir: input integar type is not supported"));
  }
}
}  // namespace
}  // namespace mlir

Operation *mlir::translateToMindsporeDialect(llvm::SourceMgr &sourceMgr, MLIRContext *context, std::string outputName) {
  llvm::MemoryBufferRef buffer = *sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  std::string jsonName = buffer.getBufferIdentifier().str();
  (void)context->getOrLoadDialect<func::FuncDialect>();
  (void)context->getOrLoadDialect<mindspore::MindSporeDialect>();
  (void)context->getOrLoadDialect<arith::ArithDialect>();
  (void)context->getOrLoadDialect<tensor::TensorDialect>();

  MindConverter msConverter;
  if (jsonName != "") {
    msConverter.inputFileName = jsonName;
  } else {
    llvm::report_fatal_error(llvm::StringRef("Error occurs when converting json to mlir: Please input a json file"));
  }
  msConverter.parseJson();
  MindBuilder mlirBuilder = msConverter.initBuilder();
  mlir::OpBuilder builder(context);
  mlirBuilder.mlirModule = builder.create<mlir::ModuleOp>(UnknownLoc::get(context));
  mlirBuilder.initMindOpFactory();
  mlirBuilder.initMindTypeMap();
  mlirBuilder.convertToMLIR();

  if (outputName != "") {
    std::error_code EC;
    llvm::raw_fd_ostream dataFile(outputName, EC);
    mlirBuilder.mlirModule->print(dataFile);
  }
  return mlirBuilder.mlirModule;
}
