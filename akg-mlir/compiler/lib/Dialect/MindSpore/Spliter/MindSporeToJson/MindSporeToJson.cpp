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
#include "akg/Dialect/MindSpore/Spliter/MindSporeToJson.h"

#include <nlohmann/json.hpp>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/Spliter/Utils.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace {
std::string extractLastSubstr(const std::string &input) {
  std::size_t pos = input.find_last_of('.');
  if (pos != std::string::npos) {
    return input.substr(pos + 1);
  } else {
    return input;
  }
}
}  // namespace

struct NameFactory {
  int count{0};
  std::string prefix;
  explicit NameFactory(const std::string &newPrefix) : prefix(newPrefix) {}
  std::string get() { return prefix + "_" + std::to_string(count++); }
  void reset() { count = 0; }
};

func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym) return nullptr;
  return dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

struct NameManager {
  DenseMap<Value, std::string> nameMap;
  DenseMap<Value, Value> callArgsMap;
  NameFactory inputNameFactory{NameFactory(kJsonValueInput)};
  NameFactory outputNameFactory{NameFactory(kJsonValueOutput)};
  void set(Value value, const std::string &name) { nameMap[value] = name; }
  void setInput(const Value &value) {
    if (nameMap.find(value) == nameMap.end()) {
      set(value, inputNameFactory.get());
    }
  }

  void setOutput(const Value &value) {
    if (nameMap.find(value) == nameMap.end()) {
      set(value, outputNameFactory.get());
    }
  }

  std::string getInput(const Value &value) {
    if (callArgsMap.find(value) != callArgsMap.end()) {
      setInput(callArgsMap[value]);
      return nameMap[callArgsMap[value]];
    }
    setInput(value);
    return nameMap[value];
  }

  std::string getOutput(const Value &value) {
    if (callArgsMap.find(value) != callArgsMap.end()) {
      setOutput(callArgsMap[value]);
      return nameMap[callArgsMap[value]];
    }
    setOutput(value);
    return nameMap[value];
  }

  void bind(func::CallOp callOp) {
    auto realInputs = SmallVector<Value>(callOp->getOperands());
    auto realOutputs = SmallVector<Value>(callOp->getResults());
    auto func = getCalledFunction(callOp);
    size_t i = 0;
    for (Value opnd : func.getBody().front().getArguments()) {
      callArgsMap[opnd] = realInputs[i++];
    }
    i = 0;
    func.walk([&](func::ReturnOp op) {
      for (mlir::Value opnd : op.getOperation()->getOperands()) {
        callArgsMap[opnd] = realOutputs[i++];
      }
    });
  }

  void clear() {
    nameMap.clear();
    callArgsMap.clear();
    inputNameFactory.reset();
    outputNameFactory.reset();
  }
} nameManager;

std::unordered_map<std::string, JsonOpBuilder *> JsonOpBuilder::protoMap;

std::string JsonOpBuilder::dump(const Value &value) {
  std::string tmp;
  llvm::raw_string_ostream str(tmp);
  str << value;
  return str.str();
}

std::string JsonOpBuilder::getDataType(const Type &type) {
  if (type.isa<IntegerType>()) {
    return kJsonValueInt + std::to_string(type.cast<IntegerType>().getWidth());
  } else if (type.isa<FloatType>()) {
    return kJsonValueFloat + std::to_string(type.cast<FloatType>().getWidth());
  } else {
    llvm::errs() << "unsupported data type for" << type << "\n";
    return "unknown";
  }
}

json JsonOpBuilder::getTensorJson(const Value &opnd) {
  json ret;
  Type type = opnd.getType();
  auto getDynamicShape = [](const RankedTensorType &tensorType) {
    auto shape = tensorType.getShape();
    std::vector<int64_t> dynShape;
    for (auto &dim : shape) {
      if (dim < 0) {
        dynShape.push_back(-1);
      } else {
        dynShape.push_back(dim);
      }
    }
    return dynShape;
  };
  if (type.isa<RankedTensorType>()) {
    auto tensorType = type.cast<RankedTensorType>();
    auto dataType = tensorType.getElementType();
    ret[kJsonKeyDataType] = getDataType(dataType);
    ret[kJsonKeyShape] = getDynamicShape(tensorType);
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    auto symShape = analysis.getSymbolicShape(tensorType);
    std::vector<std::string> symbolicShape;
    if (symShape) {
      for (auto str : (*symShape)) {
        symbolicShape.push_back(str);
      }
    }
    ret[kJsonKeySymbolicShape] = symbolicShape;
  } else {
    llvm::errs() << "unsupported tensor type for" << type << "\n";
    return "unknown";
  }
  // format may not be default
  ret[kJsonKeyFormat] = kJsonValueDefaultFormat;
  return ret;
}

json JsonOpBuilder::getValueJson(const Value &opnd) {
  json ret;
  return ret;
}

json JsonOpBuilder::jsonListWrap(json js) const {
  json ret;
  ret.push_back(js);
  return ret;
}

json JsonOpBuilder::jsonListUnPack(json js) const {
  if (js.is_array() && js.size() == 1) {
    return js[0];
  }
  return js;
}

json JsonOpBuilder::getInputJson(const Value &opnd) {
  json ret = getTensorJson(opnd);
  ret[kJsonKeyTensorName] = nameManager.getInput(opnd);
  auto definedOp = opnd.getDefiningOp();
  if (definedOp && isa<mindspore::ConstOp>(definedOp)) {
    auto constOp = cast<mindspore::ConstOp>(definedOp);
    auto elementsAttr = constOp.getValue();
    auto values = elementsAttr.getValues<Attribute>();
    json valueJs;
    for (auto value : values) {
      valueJs.push_back(getAttrValue(value)[kJsonKeyValue]);
    }
    ret[kJsonKeyValue] = jsonListUnPack(valueJs);
  }

  return ret;
}

json JsonOpBuilder::getOutputJson(const Value &opnd) {
  json ret = getTensorJson(opnd);
  ret[kJsonKeyTensorName] = nameManager.getOutput(opnd);
  return ret;
}

json JsonOpBuilder::getInputsJson() {
  NameFactory nameFactory(kJsonValueInput);
  auto inputs = SmallVector<Value>(op->getOperands());
  json ret;
  for (Value &input : inputs) {
    json inputJson = getInputJson(input);
    inputJson[kJsonKeyName] = nameFactory.get();
    ret.push_back(jsonListWrap(inputJson));
  }
  return ret;
}

json JsonOpBuilder::getOutputsJson() {
  NameFactory nameFactory(kJsonValueOutput);
  auto outputs = SmallVector<Value>(op->getResults());
  json ret;
  for (Value output : outputs) {
    json outputJson = getOutputJson(output);
    outputJson[kJsonKeyName] = nameFactory.get();
    ret.push_back(outputJson);
  }
  return ret;
}

json JsonOpBuilder::getAttrValue(Attribute &attrValue) {
  json ret;
  if (attrValue.isa<BoolAttr>()) {
    auto boolAttr = attrValue.cast<BoolAttr>();
    ret[kJsonKeyValue] = boolAttr.getValue();
    ret[kJsonKeyDataType] = kJsonValueBool;
  } else if (attrValue.isa<IntegerAttr>()) {
    auto intAttr = attrValue.cast<IntegerAttr>();
    ret[kJsonKeyValue] = intAttr.getInt();
    ret[kJsonKeyDataType] = getDataType(intAttr.getType());
  } else if (attrValue.isa<FloatAttr>()) {
    auto floatAttr = attrValue.cast<FloatAttr>();
    ret[kJsonKeyValue] = floatAttr.getValueAsDouble();
    ret[kJsonKeyDataType] = getDataType(floatAttr.getType());
  } else if (attrValue.isa<StringAttr>()) {
    auto stringAttr = attrValue.cast<StringAttr>();
    ret[kJsonKeyValue] = stringAttr.str();
    ret[kJsonKeyDataType] = "str";
  } else if (attrValue.isa<DenseI64ArrayAttr>()) {
    auto arrayAttr = attrValue.cast<DenseI64ArrayAttr>();
    ret[kJsonKeyDataType] = "listInt";
    ret[kJsonKeyValue] = arrayAttr.asArrayRef();
  } else {
    llvm::errs() << "attr value of type not implemented: " << attrValue << "\n";
    // error log here;
  }
  return ret;
}
json JsonOpBuilder::getAttrJson(NamedAttribute &attr) {
  Attribute attrValue = attr.getValue();
  auto ret = getAttrValue(attrValue);
  ret[kJsonKeyName] = attr.getName().getValue().str();
  return ret;
}

json JsonOpBuilder::getAttrsJson() {
  json ret;
  auto attrs = op->getAttrDictionary();
  if (attrs.contains("ms_attr")) {
    auto msAttrs = attrs.get("ms_attr").cast<DictionaryAttr>();
    for (auto attr : msAttrs) {
      ret.push_back(getAttrJson(attr));
    }
  }
  return ret;
}

mindspore::MindSporeOp JsonOpBuilder::getMindSporeOp() {
  assert(isa<mindspore::MindSporeOp>(op));
  return cast<mindspore::MindSporeOp>(op);
}

std::string JsonOpBuilder::getOpName() {
  auto toCamelCase = [](const std::string &input) {
    std::string result;
    bool capitalize = false;
    for (char c : input) {
      if (c == '_') {
        capitalize = true;
      } else {
        if (capitalize) {
          result.push_back(std::toupper(c));
          capitalize = false;
        } else {
          result.push_back(result.empty() ? std::toupper(c) : std::tolower(c));
        }
      }
    }
    return result;
  };
  std::string mindsporeOpName = op->getName().getStringRef().str();
  return toCamelCase(extractLastSubstr(mindsporeOpName));
}

std::string JsonOpBuilder::getOpAddress() const {
  if (!op->hasAttr(kJsonKeyPtrAddress)) {
    return "";
  }
  return op->getAttr(kJsonKeyPtrAddress).cast<StringAttr>().str();
}

json JsonOpBuilder::build() {
  json ret;
  ret[kJsonKeyAttr] = getAttrsJson();
  ret[kJsonKeyInputDesc] = getInputsJson();
  ret[kJsonKeyOutputDesc] = getOutputsJson();
  ret[kJsonKeyName] = getOpName();
  ret[kJsonKeyPtrAddress] = getOpAddress();
  return ret;
}

func::FuncOp JsonFuncBuilder::getFuncOp() {
  if (funcOp == nullptr) {
    assert(isa<func::FuncOp>(op));
    funcOp = cast<func::FuncOp>(op);
  }
  return funcOp;
}

std::shared_ptr<JsonOpBuilder> JsonFuncBuilder::opBuilderFactory(Operation *op) {
  std::string mindOpName = extractLastSubstr(op->getName().getStringRef().str());
  return JsonOpBuilder::getProto(mindOpName, op);
}

bool JsonFuncBuilder::jumpOp(const Operation *op) const {
  if (isa<mindspore::ConstOp>(op)) {
    return true;
  }
  return false;
}

json JsonFuncBuilder::getInnerOpsJson() {
  auto func = getFuncOp();
  Block &entryBlock = func.getBody().front();
  json ret;
  for (auto &opref : entryBlock.getOperations()) {
    auto op = &opref;
    if (!spliter::isGraphKernelOp(op) || jumpOp(op)) {
      continue;
    }
    auto opbuilder = opBuilderFactory(op);
    ret.push_back(opbuilder->build());
  }
  return ret;
}

json JsonFuncBuilder::getInputsJson() {
  auto func = getFuncOp();
  json ret;
  Block &entryBlock = func.getBody().front();
  for (Value opnd : entryBlock.getArguments()) {
    ret.push_back(jsonListWrap(getInputJson(opnd)));
  }
  return ret;
}

json JsonFuncBuilder::getOutputsJson() {
  auto func = getFuncOp();
  json ret;
  func.walk([&](func::ReturnOp op) {
    for (mlir::Value opnd : op.getOperation()->getOperands()) {
      ret.push_back(getOutputJson(opnd));
    }
  });
  return ret;
}

std::string JsonFuncBuilder::getOpName() {
  auto func = getFuncOp();
  return func.getSymNameAttr().str();
}

json JsonFuncBuilder::build() {
  auto opname = getOpName();
  json json;
  json[kJsonKeyOp] = opname;
  json[kJsonKeyComposite] = true;
  json[kJsonKeyPlatform] = platform;
  json[kJsonKeyProcess] = process;
  json[kJsonKeyInputDesc] = getInputsJson();
  json[kJsonKeyOpDesc] = getInnerOpsJson();
  json[kJsonKeyOutputDesc] = getOutputsJson();
  json[kJsonKeyGraphMode] = json[kJsonKeyOpDesc].size() > 1 ? kJsonValueComposite : kJsonValueBasic;
  return json;
}

MatMulOpBuilder MatMulOpBuilder::matmulProto;

std::string MatMulOpBuilder::getOpName() {
  if (op->hasAttr("ori_op")) {
    auto attr = op->getAttr("ori_op");
    auto oriOpname = attr.cast<StringAttr>().str();
    return oriOpname;
  }
  return "MatMul";
}

BatchMatMulOpBuilder BatchMatMulOpBuilder::batchMatmulProto;

std::string BatchMatMulOpBuilder::getOpName() { return "BatchMatMul"; }

TransposeOpBuilder TransposeOpBuilder::transposeProto;

json TransposeOpBuilder::getInputsJson() {
  NameFactory nameFactory(kJsonValueInput);
  auto input = op->getOperand(0);
  json ret;
  json inputJson = getInputJson(input);
  inputJson[kJsonKeyName] = nameFactory.get();
  ret.push_back(jsonListWrap(inputJson));
  return ret;
}

UnknownOpBuilder UnknownOpBuilder::unKnownProto;

std::string UnknownOpBuilder::getOpName() {
  auto attr = op->getAttr("ori_op");
  auto oriOpname = attr.cast<StringAttr>().str();
  return oriOpname;
}

json DivOpBuilder::getInputsJson() {
  NameFactory nameFactory(kJsonValueInput);
  Operation *reciprocal = op->getOperand(1).getDefiningOp();
  SmallVector<Value> inputs;
  inputs.push_back(op->getOperand(0));
  inputs.push_back(reciprocal->getOperand(0));
  json ret;
  for (Value &input : inputs) {
    json inputJson = getInputJson(input);
    inputJson[kJsonKeyName] = nameFactory.get();
    ret.push_back(jsonListWrap(inputJson));
  }
  return ret;
}

std::string DivOpBuilder::getOpName() {
  auto attr = op->getAttr("ori_op");
  auto oriOpname = attr.cast<StringAttr>().str();
  return oriOpname;
}

func::FuncOp getMainFunc(ModuleOp &moduleOp) {
  Region &moduleRegion = moduleOp.getBodyRegion();
  if (moduleRegion.empty()) {
    return nullptr;
  }
  Block &moduleBlock = moduleRegion.front();
  if (moduleBlock.empty()) {
    return nullptr;
  }
  auto funcRanges = moduleBlock.getOps<func::FuncOp>();
  auto funcOps = std::vector<func::FuncOp>(funcRanges.begin(), funcRanges.end());
  if (funcOps.size() > 1) {
    return funcOps.back();
  }
  return nullptr;
}

SmallVector<func::FuncOp> getFuncsFromModule(ModuleOp &moduleOp) {
  Region &moduleRegion = moduleOp.getBodyRegion();
  if (moduleRegion.empty()) {
    return SmallVector<func::FuncOp>();
  }
  Block &moduleBlock = moduleRegion.front();
  if (moduleBlock.empty()) {
    return SmallVector<func::FuncOp>();
  }
  return SmallVector<func::FuncOp>(moduleBlock.getOps<func::FuncOp>());
}

std::string singleFuncToJson(func::FuncOp &funcOp) {
  nameManager.clear();
  return JsonFuncBuilder(funcOp).build().dump();
}

std::string singleFuncToJson(ModuleOp &moduleOp) {
  nameManager.clear();
  auto funcList = getFuncsFromModule(moduleOp);
  if (funcList.empty()) {
    llvm::errs() << "No func op found.\n";
    return "";
  } else if (funcList.size() != 1) {
    llvm::errs() << "Multiple op funcs must be invoked in main func.\n";
    return "";
  } else {
    return singleFuncToJson(funcList[0]);
  }
}

std::string splitedMainFuncToJson(func::FuncOp &mainFunc) {
  nameManager.clear();
  Block &mainBlock = mainFunc.getBody().front();
  json graphModes;
  json funcs;
  for (func::CallOp callOp : mainBlock.getOps<func::CallOp>()) {
    nameManager.bind(callOp);
    auto funcJson = JsonFuncBuilder(getCalledFunction(callOp)).build();
    funcs.push_back(funcJson);
    graphModes.push_back(funcJson[kJsonKeyGraphMode]);
  }

  json ret = {{kJsonKeyMultiGraph, true}, {kJsonKeyGraphDesc, funcs}, {kJsonKeyGraphMode, graphModes}};
  return ret.dump();
}

std::string mlirToJson(ModuleOp &moduleOp) {
  func::FuncOp mainFunc = getMainFunc(moduleOp);
  if (!mainFunc) {
    // single func to json
    return singleFuncToJson(moduleOp);
  }
  return splitedMainFuncToJson(mainFunc);
}

struct Options {
  llvm::cl::opt<std::string> inputMlirFile{"mlir-to-json", llvm::cl::Positional,
                                           llvm::cl::desc("the input kernel json file"), llvm::cl::init("")};

  llvm::cl::opt<std::string> outputFile{"o", llvm::cl::Positional, llvm::cl::desc("the output .json file"),
                                        llvm::cl::init("")};
};

LogicalResult mlirToJsonOpt(int argc, char **argv, llvm::StringRef toolName, DialectRegistry &registry) {
  Options options;
  (void)llvm::cl::ParseCommandLineOptions(argc, argv, toolName);
  if (options.inputMlirFile.empty()) {
    llvm::errs() << "Empty input mlir file, use -mlir-to-json option to "
                    "indicate input mlir.\n";
    return LogicalResult::failure();
  }
  MLIRContext context(registry);
  auto moduleRef = parseSourceFile<ModuleOp>(options.inputMlirFile.data(), &context);
  if (!moduleRef) {
    llvm::errs() << "Parse input mlir file failed: " << options.inputMlirFile.data() << "\n";
    return LogicalResult::failure();
  }
  ModuleOp moduleOp = *moduleRef;
  auto res = mlirToJson(moduleOp);
  if (res.empty()) {
    llvm::errs() << "Convert mlir to json failed.\n";
    return LogicalResult::failure();
  }
  std::string inputFilename = std::string(options.inputMlirFile.data());
  std::string outputFilename = inputFilename.substr(0, inputFilename.find_last_of(".")) + ".json";
  if (!options.outputFile.empty()) {
    outputFilename = options.outputFile.data();
  }
  if (llvm::writeFileAtomically("tmp_%%%%%%%%.json", outputFilename, res)) {
    llvm::errs() << "Write json file to " << outputFilename << " failed.\n";
    return LogicalResult::failure();
  }
  llvm::outs() << "Convert " << inputFilename << " to " << outputFilename << " successfully.\n";
  return LogicalResult::success();
}

}  // namespace mlir
