// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2026 Huawei Technologies Co., Ltd
//
// SPDX-License-Identifier: Apache-2.0
//
// This file was copied from project tenstorrent/tt-mlir
// (https://github.com/tenstorrent/tt-mlir) and modified for AKG-MLIR.

#include <optional>
#include <regex>
#include <stack>
#include <string>
#include <unordered_set>

#include "akg/Target/Pypto/PyptoEmitter.h"

#include "akg/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "akg/Dialect/EmitPy/IR/EmitPyOps.h"
#include "akg/Dialect/EmitPy/IR/EmitPyTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;          // NOLINT(build/namespaces)
using namespace mlir::emitpy;  // NOLINT(build/namespaces)
using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras which typically
/// do not handle error propagation.
template <typename ForwardIterator, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(ForwardIterator begin, ForwardIterator end, UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  if (begin == end) {
    return success();
  }
  if (failed(eachFn(*begin))) {
    return failure();
  }
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin))) {
      return failure();
    }
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &container, UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  return interleaveWithError(container.begin(), container.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container container, raw_ostream &os, UnaryFunctor eachFn) {
  return interleaveWithError(container.begin(), container.end(), eachFn, [&]() { os << ", "; });
}

namespace {

// Forward declaration
struct PythonEmitter;

/// Helper class for building inline Python expressions
class ExpressionBuilder {
 public:
  explicit ExpressionBuilder(PythonEmitter &emitter) {}

  /// Register a block argument with its corresponding name
  void mapBlockArgument(Value blockArg, StringRef name) { blockArgNames[blockArg] = name.str(); }

  /// Build an expression string for a value recursively
  FailureOr<std::string> buildExpression(Value value);

 private:
  /// Build expression for specific operation types
  FailureOr<std::string> buildCallOpaqueExpr(CallOpaqueOp op);
  FailureOr<std::string> buildBinaryInfixExpr(CallOpaqueOp op, StringRef infix);
  FailureOr<std::string> buildConstantExpr(ConstantOp op);
  FailureOr<std::string> buildLiteralExpr(LiteralOp op);
  FailureOr<std::string> buildSubscriptExpr(SubscriptOp op);
  FailureOr<std::string> buildGetAttrExpr(GetAttrOp op);

  /// Map from Value to its expression string representation
  DenseMap<Value, std::string> expressionCache;

  /// Block arguments mapped to their names
  DenseMap<Value, std::string> blockArgNames;
};

/// Emitter that uses dialect specific emitters to emit Python code.
struct PythonEmitter {
  explicit PythonEmitter(raw_ostream &os, std::optional<std::string> fileId) : os(os), fileId(std::move(fileId)) {
    valueInScopeCount.push(0);
    usedNames.push(std::set<std::string>());
  }

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' or returns failure.
  LogicalResult emitOperation(Operation &op);

  /// TODO: implement this for type hints
  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits an assignment for a variable or
  /// returns failure.
  LogicalResult emitVariableAssignment(Operation &op);

  /// Allocates a variable name for the single result of `op`, reserving and
  /// mapping it, without emitting any text. Used by statements that need the
  /// result's name before emitting (e.g. select's if/else form); the
  /// naming mirrors emitVariableAssignment.
  std::string allocateResultName(Operation &op);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits multiple variables separated with comma for multi-valued
  /// operation;
  /// - emits single variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a global statement.
  LogicalResult emitGlobalStatement(GlobalStatementOp globalStatementOp);

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Emits value as an operand of an operation.
  LogicalResult emitOperand(Value value, std::string name);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value value, std::string name);

  /// Return the textual representation of a subscript operation.
  std::string getSubscriptName(SubscriptOp op);

  /// Bind `value` to a Python variable that another value already owns, as the
  /// init, body block argument, yielded value and result of a loop-carried
  /// value all are. The name must have come from getOrCreateName, which is what
  /// reserved it; no uniquing happens here, since aliasing is the point.
  void mapValueToName(Value value, StringRef name);

  /// Register a value with verbatim Python text to substitute for it wherever
  /// it is used. Nothing is reserved: the text is either not an identifier at
  /// all (a literal, an inlined expression) or a name the IR fixed and Python
  /// expects to be shared, as `global x` and a later assignment to `x` are.
  void registerDeferredValue(Value value, StringRef str);

  /// Decides whether the file should be emitted. If fileId is set, only
  /// the ops of the file with the matching id are emitted.
  bool shouldEmitFile(FileOp fileOp);

  /// Returns true if we're emitting multiple files (fileId is not set).
  bool isEmittingMultipleFiles() const { return !fileId.has_value(); }

  /// Emits a comment label for the file to separate outputs.
  void emitFileLabel(FileOp fileOp);

  /// RAII helper function to manage entering/exiting Python scopes.
  struct Scope {
    explicit Scope(PythonEmitter &emitter) : valueMapperScope(emitter.valueMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.usedNames.push(emitter.usedNames.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.usedNames.pop();
    }

   private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    PythonEmitter &emitter;
  };

  /// Returns whether the Value is assigned to a Python variable in the scope.
  bool hasValueInScope(Value value) { return valueMapper.count(value); }

  bool isInClassScope() const { return classDepth > 0; }

  /// RAII helper function to manage entering/exiting Python class scopes.
  struct ClassScope {
    explicit ClassScope(PythonEmitter &emitter) : emitter(&emitter) { emitter.classDepth++; }
    ~ClassScope() { emitter->classDepth--; }

   private:
    // Non-owning pointer: ClassScope is a stack guard created inside
    // printOperation(PythonEmitter&, ClassOp) and destroyed before that call
    // returns. The PythonEmitter object is owned by translateToPython and
    // outlives all nested ClassScope instances, so this pointer cannot dangle.
    PythonEmitter *emitter;
  };

  /// Reserve a name to prevent collisions.
  void reserveName(const std::string &name) { usedNames.top().insert(name); }

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; }

 private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Map from value to name of Python variable that contains the name.
  ValueMapper valueMapper;

  /// The number of values in the current scope per variable name. This is used
  /// to declare the names of values in a scope.
  std::stack<int64_t> valueInScopeCount;

  /// The set of names used in the current scope.
  std::stack<std::set<std::string>> usedNames;

  int classDepth = 0;

  /// The id of the current file. Only files with this id are emitted. If
  /// not set, all files are emitted as one.
  std::optional<std::string> fileId;
};
}  // namespace

// Implementation of ExpressionBuilder methods

/// Infix operator for a scalar comparison pseudo-callee, or an empty string
/// if `callee` is not one. The PyPTO parser accepts plain Python comparison
/// expressions (`a < b`), which it maps onto ir.eq/ne/lt/le/gt/ge; its
/// runtime's `pl.cmp` is a Tensor/Tile-only API, so scalar comparisons must
/// be spelled as infix comparisons in the emitted Python.
static StringRef infixCmpOpForCallee(StringRef callee) {
  return llvm::StringSwitch<StringRef>(callee)
    .Case("eq", "==")
    .Case("ne", "!=")
    .Case("lt", "<")
    .Case("le", "<=")
    .Case("gt", ">")
    .Case("ge", ">=")
    .Default(StringRef());
}

/// Infix operator for a scalar arithmetic pseudo-callee, or an empty string
/// if `callee` is not one. The PyPTO parser accepts plain Python arithmetic
/// expressions (`a + b`), which it maps onto pto.addi/subi/muli and, for
/// `/`, pto.divi on integers (truncated division, result stays integer) and
/// pto.divf on floats. Callees are the arith op names themselves; the
/// runtime's `pl.fmod` is a Tile-only API, so scalar remainders must be
/// spelled as the plain Python `%` operator in the emitted kernel.
static StringRef infixArithOpForCallee(StringRef callee) {
  return llvm::StringSwitch<StringRef>(callee)
    .Case("addi", "+")
    .Case("addf", "+")
    .Case("subi", "-")
    .Case("subf", "-")
    .Case("muli", "*")
    .Case("mulf", "*")
    .Case("divf", "/")
    .Case("divsi", "/")
    .Case("floordivsi", "//")
    .Case("remsi", "%")
    .Case("remui", "%")
    .Default(StringRef());
}

FailureOr<std::string> ExpressionBuilder::buildExpression(Value value) {
  // Check if already cached
  if (expressionCache.count(value)) {
    return expressionCache[value];
  }

  // Check if it's a block argument
  if (blockArgNames.count(value)) {
    return blockArgNames[value];
  }

  // Otherwise, it must be defined by an operation
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return failure();
  }

  // Build expression based on operation type
  FailureOr<std::string> result;
  if (auto callOp = dyn_cast<CallOpaqueOp>(defOp)) {
    result = buildCallOpaqueExpr(callOp);
  } else if (auto constantOp = dyn_cast<ConstantOp>(defOp)) {
    result = buildConstantExpr(constantOp);
  } else if (auto literalOp = dyn_cast<LiteralOp>(defOp)) {
    result = buildLiteralExpr(literalOp);
  } else if (auto subscriptOp = dyn_cast<SubscriptOp>(defOp)) {
    result = buildSubscriptExpr(subscriptOp);
  } else if (auto getAttrOp = dyn_cast<GetAttrOp>(defOp)) {
    result = buildGetAttrExpr(getAttrOp);
  } else {
    return defOp->emitOpError("operation type not supported in inline expression");
  }

  if (succeeded(result)) {
    expressionCache[value] = *result;
  }
  return result;
}

FailureOr<std::string> ExpressionBuilder::buildBinaryInfixExpr(CallOpaqueOp op, StringRef infix) {
  if (op.getNumOperands() != 2) {
    return op.emitOpError(op.getCallee()) << " expects two operands";
  }
  auto lhsExpr = buildExpression(op.getOperand(0));
  if (failed(lhsExpr)) {
    return failure();
  }
  auto rhsExpr = buildExpression(op.getOperand(1));
  if (failed(rhsExpr)) {
    return failure();
  }
  std::string expr;
  llvm::raw_string_ostream os(expr);
  os << "(" << *lhsExpr << " " << infix << " " << *rhsExpr << ")";
  return expr;
}

FailureOr<std::string> ExpressionBuilder::buildCallOpaqueExpr(CallOpaqueOp op) {
  std::string expr;
  llvm::raw_string_ostream os(expr);

  // Pseudo-callee printed as a Python infix arithmetic expression:
  // `(a + b)` (floordivsi prints `//`, remsi/remui print `%`).
  if (StringRef arithOp = infixArithOpForCallee(op.getCallee()); !arithOp.empty()) {
    return buildBinaryInfixExpr(op, arithOp);
  }

  // Pseudo-callee printed as a Python infix comparison: `(a < b)`.
  if (StringRef cmpOp = infixCmpOpForCallee(op.getCallee()); !cmpOp.empty()) {
    return buildBinaryInfixExpr(op, cmpOp);
  }

  // Pseudo-callee printed as a Python conditional expression: operands are
  // (condition, trueValue, falseValue) in the IR, Python spells it
  // `(trueValue if condition else falseValue)`.
  if (op.getCallee() == "select") {
    if (op.getNumOperands() != 3) {
      return op.emitOpError("select expects three operands");
    }
    auto condExpr = buildExpression(op.getOperand(0));
    if (failed(condExpr)) {
      return failure();
    }
    auto trueExpr = buildExpression(op.getOperand(1));
    if (failed(trueExpr)) {
      return failure();
    }
    auto falseExpr = buildExpression(op.getOperand(2));
    if (failed(falseExpr)) {
      return failure();
    }
    os << "(" << *trueExpr << " if " << *condExpr << " else " << *falseExpr << ")";
    return expr;
  }

  bool isList = (op.getCallee() == "util_create_list");
  os << (isList ? "[" : op.getCallee().str() + "(");

  bool first = true;
  for (Value operand : op.getOperands()) {
    if (!first) {
      os << ", ";
    }
    first = false;

    auto operandExpr = buildExpression(operand);
    if (failed(operandExpr)) {
      return failure();
    }
    os << *operandExpr;
  }

  os << (isList ? "]" : ")");
  return expr;
}

FailureOr<std::string> ExpressionBuilder::buildConstantExpr(ConstantOp op) {
  Attribute attr = op.getValue();
  if (auto oAttr = dyn_cast<mlir::emitpy::OpaqueAttr>(attr)) {
    return oAttr.getValue().str();
  }
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    SmallString<128> strValue;
    bool isSigned = true;
    if (auto intType = dyn_cast<IntegerType>(iAttr.getType())) {
      // Treat signless integers as signed.
      isSigned = !intType.isUnsigned();
    }
    iAttr.getValue().toString(strValue, 10, isSigned);
    return std::string{strValue.begin(), strValue.end()};
  }
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    APFloat value = fAttr.getValue();
    if (value.isInfinity()) {
      return std::string(value.isNegative() ? "float('-inf')" : "float('inf')");
    }
    if (value.isNaN()) {
      return std::string("float('nan')");
    }
    SmallString<128> strValue;
    value.toString(strValue);
    return std::string{strValue.begin(), strValue.end()};
  }
  if (auto sAttr = dyn_cast<StringAttr>(attr)) {
    return ("\"" + sAttr.getValue() + "\"").str();
  }
  return op.emitOpError("unsupported constant attribute type in expression");
}

FailureOr<std::string> ExpressionBuilder::buildLiteralExpr(LiteralOp op) { return std::string(op.getValue()); }

FailureOr<std::string> ExpressionBuilder::buildSubscriptExpr(SubscriptOp op) {
  std::string expr;
  llvm::raw_string_ostream os(expr);

  auto valueExpr = buildExpression(op.getContainer());
  if (failed(valueExpr)) {
    return failure();
  }

  auto indexExpr = buildExpression(op.getIndex());
  if (failed(indexExpr)) {
    return failure();
  }

  os << *valueExpr << "[" << *indexExpr << "]";
  return expr;
}

FailureOr<std::string> ExpressionBuilder::buildGetAttrExpr(GetAttrOp op) {
  std::string expr;
  llvm::raw_string_ostream os(expr);

  auto objectExpr = buildExpression(op.getObject());
  if (failed(objectExpr)) {
    return failure();
  }

  os << *objectExpr << "." << op.getAttrName();
  return expr;
}

/// Determine whether op result should be emitted in a deferred way.
static bool hasDeferredEmission(Operation *op) {
  // ExpressionOp with inline mode should also be deferred
  if (auto exprOp = dyn_cast_or_null<ExpressionOp>(op)) {
    return !exprOp.getDoNotInline();
  }
  return isa_and_nonnull<LiteralOp>(op);
}

StringRef PythonEmitter::getOrCreateName(Value value, std::string name) {
  if (!valueMapper.count(value)) {
    while (usedNames.top().count(name)) {
      name = name + "_" + std::to_string(valueInScopeCount.top()++);
    }
    valueMapper.insert(value, name);
    usedNames.top().insert(name);
  }
  return *valueMapper.begin(value);
}

std::string PythonEmitter::getSubscriptName(SubscriptOp op) {
  std::string name;
  llvm::raw_string_ostream ss(name);
  auto index = op.getIndex();

  // Only generate a new name if index doesn't already have one
  std::string indexName;
  if (!valueMapper.count(index)) {
    indexName = "index_" + std::to_string(valueInScopeCount.top()++);
  }

  ss << "[" << getOrCreateName(index, indexName) << "]";
  return name;
}

void PythonEmitter::mapValueToName(Value value, StringRef name) { registerDeferredValue(value, name); }

void PythonEmitter::registerDeferredValue(Value value, StringRef str) {
  if (!valueMapper.count(value)) {
    valueMapper.insert(value, str.str());
  }
}

bool PythonEmitter::shouldEmitFile(FileOp fileOp) { return isEmittingMultipleFiles() || *fileId == fileOp.getId(); }

void PythonEmitter::emitFileLabel(FileOp fileOp) { os << "# File: " << fileOp.getId() << "\n"; }

/// Prints one CallOpaqueOp argument: an operand reference (an IntegerAttr of
/// index type), a list mixing operand references with opaque source text
/// (an ArrayAttr), or a plain attribute.
template <typename PairT>
static LogicalResult emitCallOpaqueArg(PythonEmitter &emitter, Operation &op, raw_indented_ostream &os,
                                       const PairT &pair) {
  auto [attr, keywordArg] = pair;
  auto keywordArgStr = mlir::cast<StringAttr>(keywordArg).getValue();
  keywordArgStr != "" ? os << keywordArgStr << "=" : os << keywordArgStr;

  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (iAttr.getType().isIndex()) {
      int64_t idx = iAttr.getInt();
      Value operand = op.getOperand(idx);
      if (!emitter.hasValueInScope(operand)) {
        return op.emitOpError("operand ") << idx << "'s value not defined in scope";
      }
      if (failed(emitter.emitOperand(operand, "print_arg"))) {
        return failure();
      }
      return success();
    }
  }
  if (auto aAttr = dyn_cast<ArrayAttr>(attr)) {
    // A list-valued argument: each element is an operand reference
    // (IntegerAttr of index type) or opaque source text. Emits e.g.
    // `[row, 0]` for a dynamic offset pair.
    os << "[";
    bool firstElem = true;
    for (Attribute elem : aAttr.getValue()) {
      if (!firstElem) {
        os << ", ";
      }
      firstElem = false;
      if (auto iElem = dyn_cast<IntegerAttr>(elem)) {
        if (iElem.getType().isIndex()) {
          Value operand = op.getOperand(iElem.getInt());
          if (!emitter.hasValueInScope(operand)) {
            return op.emitOpError("operand ") << iElem.getInt() << "'s value not defined in scope";
          }
          if (failed(emitter.emitOperand(operand, "print_arg"))) {
            return failure();
          }
          continue;
        }
      }
      if (auto oElem = dyn_cast<mlir::emitpy::OpaqueAttr>(elem)) {
        os << oElem.getValue();
        continue;
      }
      return op.emitOpError("unsupported list argument element");
    }
    os << "]";
    return success();
  }
  if (failed(emitter.emitAttribute(op.getLoc(), attr))) {
    return failure();
  }
  return success();
}

static LogicalResult printUtilSelect(PythonEmitter &emitter, Operation &op) {
  raw_indented_ostream &os = emitter.ostream();
  if (op.getNumOperands() != 3) {
    return op.emitOpError("select expects three operands");
  }
  if (op.getNumResults() != 1) {
    return op.emitOpError("select expects a single result");
  }
  std::string name = emitter.allocateResultName(op);
  os << "if ";
  if (failed(emitter.emitOperand(op.getOperand(0), "print_arg"))) {
    return failure();
  }
  os << ":\n";
  os.indent();
  os << name << " = ";
  if (failed(emitter.emitOperand(op.getOperand(1), "print_arg"))) {
    return failure();
  }
  os << "\n";
  os.unindent();
  os << "else:\n";
  os.indent();
  os << name << " = ";
  if (failed(emitter.emitOperand(op.getOperand(2), "print_arg"))) {
    return failure();
  }
  os << "\n";
  os.unindent();
  return success();
}

/// Prints a two-operand pseudo-callee as `(lhs <infix> rhs)` after the
/// assign prefix has already been emitted.
static LogicalResult printBinaryInfixCall(PythonEmitter &emitter, CallOpaqueOp callOpaqueOp, StringRef infix) {
  raw_indented_ostream &os = emitter.ostream();
  Operation &op = *callOpaqueOp.getOperation();
  if (op.getNumOperands() != 2) {
    return op.emitOpError(callOpaqueOp.getCallee()) << " expects two operands";
  }
  os << "(";
  if (failed(emitter.emitOperand(op.getOperand(0), "print_arg"))) {
    return failure();
  }
  os << " " << infix << " ";
  if (failed(emitter.emitOperand(op.getOperand(1), "print_arg"))) {
    return failure();
  }
  os << ")";
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, CallOpaqueOp callOpaqueOp) {
  raw_indented_ostream &os = emitter.ostream();
  Operation &op = *callOpaqueOp.getOperation();
  StringRef callee = callOpaqueOp.getCallee();

  // Pseudo-callee printed as a Python if/else statement: operands are
  // (condition, trueValue, falseValue) in the IR. The PyPTO parser has no
  // support for Python conditional expressions (IfExp), so a scalar select
  // is spelled as an if/else that assigns the same variable in both
  // branches; the parser leaks the branch-local variable and its
  // ConvertToSSA pass inserts the phi.
  if (callee == "select") {
    return printUtilSelect(emitter, op);
  }

  if (failed(emitter.emitAssignPrefix(op))) {
    return failure();
  }

  // Pseudo-callee printed as a Python infix arithmetic expression:
  // `(a + b)` (floordivsi prints `//`, remsi/remui print `%`), which the
  // PyPTO parser maps onto scalar pto.addi/subi/muli/divi (divf) operations.
  if (StringRef arithOp = infixArithOpForCallee(callee); !arithOp.empty()) {
    return printBinaryInfixCall(emitter, callOpaqueOp, arithOp);
  }

  // Pseudo-callee printed as a Python infix comparison: `(a < b)`, which the
  // PyPTO parser maps onto scalar ir.eq/ne/lt/le/gt/ge comparisons.
  if (StringRef cmpOp = infixCmpOpForCallee(callee); !cmpOp.empty()) {
    return printBinaryInfixCall(emitter, callOpaqueOp, cmpOp);
  }

  if (callee == "util_create_list") {
    os << "[";
    if (failed(emitter.emitOperands(op))) {
      return failure();
    }
    os << "]";
  } else {
    os << callee << "(";
    if (callOpaqueOp.getArgs()) {
      auto args = llvm::zip(*callOpaqueOp.getArgs(), *callOpaqueOp.getKeywordArgs());
      auto emitArgs = [&](const auto &pair) { return emitCallOpaqueArg(emitter, op, os, pair); };
      if (failed(interleaveCommaWithError(args, os, emitArgs))) {
        return failure();
      }
    } else if (failed(emitter.emitOperands(op))) {
      return failure();
    }
    os << ")";
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, ImportOp importOp) {
  raw_indented_ostream &os = emitter.ostream();
  if (importOp.getImportAll()) {
    os << "from " << importOp.getModuleName() << " import *";
  } else if (importOp.getMembersToImport()) {
    os << "from " << importOp.getModuleName() << " import ";

    auto members = *importOp.getMembersToImport();
    os << mlir::cast<StringAttr>(members[0]).getValue();
    auto member_aliases = *importOp.getMemberAliases();
    auto member_alias = mlir::cast<StringAttr>(member_aliases[0]).getValue();
    if (!member_alias.empty()) {
      os << " as " << member_alias;
    }

    for (size_t i = 1; i < members.size(); i++) {
      os << ", " << mlir::cast<StringAttr>(members[i]).getValue();
      member_alias = mlir::cast<StringAttr>(member_aliases[i]).getValue();
      if (!member_alias.empty()) {
        os << " as " << member_alias;
      }
    }
  } else {
    os << "import " << importOp.getModuleName();
    if (importOp.getModuleAlias()) {
      os << " as " << importOp.getModuleAlias();
    }
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, func::CallOp callOp) {
  Operation &op = *callOp.getOperation();
  StringRef callee = callOp.getCallee();
  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(op))) {
    return failure();
  }
  os << callee;
  os << "(";
  if (failed(emitter.emitOperands(op))) {
    return failure();
  }
  os << ")";
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, ModuleOp moduleOp) {
  PythonEmitter::Scope scope(emitter);

  // When file ops exist, emit the content of a specified file if the fileId
  // is present, otherwise emit each file's contents with a label separator.
  bool hasFileOps = llvm::any_of(moduleOp.getOps<FileOp>(), [](FileOp) { return true; });

  if (hasFileOps) {
    for (auto fileOp : moduleOp.getOps<FileOp>()) {
      if (!emitter.shouldEmitFile(fileOp)) {
        continue;
      }
      if (emitter.isEmittingMultipleFiles()) {
        emitter.emitFileLabel(fileOp);
      }
      if (failed(emitter.emitOperation(*fileOp))) {
        return failure();
      }
    }
    return success();
  }

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op))) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PyPTO kernel emission
//===----------------------------------------------------------------------===//

/// Unit attribute set by convert-vector-to-pypto; marks a func as a PyPTO
/// @pl.jit kernel.
static constexpr llvm::StringLiteral kPyptoKernelAttr = "emitpy.pypto_kernel";

/// Argument directions recorded by convert-vector-to-pypto.
static constexpr llvm::StringLiteral kOutArgsAttr = "emitpy.out_args";
static constexpr llvm::StringLiteral kInOutArgsAttr = "emitpy.inout_args";

/// Returns whether `argNumber` appears in a dense i64 array attribute.
static bool pyptoArgAttrContains(DenseI64ArrayAttr attr, unsigned argNumber) {
  return attr && llvm::is_contained(attr.asArrayRef(), static_cast<int64_t>(argNumber));
}

/// Emits the PyPTO parameter annotation for a kernel argument: the verbatim
/// opaque type text (e.g. `pl.Tensor[[4, 8], pl.FP32]`), wrapped in
/// `pl.Out[...]` for outputs and `pl.InOut[...]` for read-modify-write
/// parameters as the PyPTO kernel model requires.
static LogicalResult emitPyptoParamAnnotation(PythonEmitter &emitter, func::FuncOp functionOp, BlockArgument arg) {
  raw_indented_ostream &os = emitter.ostream();
  auto opaqueType = dyn_cast<mlir::emitpy::OpaqueType>(arg.getType());
  if (!opaqueType) {
    return functionOp.emitOpError("PyPTO kernel parameter ")
           << arg.getArgNumber() << " must have an opaque annotation type";
  }
  os << ": ";
  if (pyptoArgAttrContains(functionOp->getAttrOfType<DenseI64ArrayAttr>(kOutArgsAttr), arg.getArgNumber())) {
    os << "pl.Out[" << opaqueType.getValue() << "]";
  } else if (pyptoArgAttrContains(functionOp->getAttrOfType<DenseI64ArrayAttr>(kInOutArgsAttr), arg.getArgNumber())) {
    os << "pl.InOut[" << opaqueType.getValue() << "]";
  } else {
    os << opaqueType.getValue();
  }
  return success();
}

static LogicalResult printFunctionArgs(PythonEmitter &emitter, Operation &op, Region::BlockArgListType arguments) {
  raw_indented_ostream &os = emitter.ostream();
  func::FuncOp functionOp = mlir::cast<func::FuncOp>(op);
  bool isPyptoKernel = functionOp->getAttr(kPyptoKernelAttr) != nullptr;

  return interleaveCommaWithError(arguments, os, [&](BlockArgument arg) {
    std::string argName = "inputs";
    if (auto suggestNameAttr = functionOp.getArgAttr(arg.getArgNumber(), "emitpy.name")) {
      argName = mlir::cast<StringAttr>(suggestNameAttr).getValue();
    } else if (emitter.isInClassScope() && arg.getArgNumber() == 0) {
      argName = "self";
    }
    if (failed(emitter.emitOperand(arg, argName))) {
      return failure();
    }
    if (isPyptoKernel) {
      return emitPyptoParamAnnotation(emitter, functionOp, arg);
    }
    return success();
  });
}

static LogicalResult printFunctionBody(PythonEmitter &emitter, Operation &op, Region::BlockListType &blocks) {
  raw_indented_ostream &os = emitter.ostream();
  os.indent();
  if (blocks.size() > 1) {
    return op.emitOpError(
      "PythonEmitter's printFunctionBody currently only supports "
      "single-block functions directly "
      "or relies on prior passes to structure control flow.");
  }

  for (Operation &bodyOp : blocks.front().getOperations()) {
    if (failed(emitter.emitOperation(bodyOp))) {
      return failure();
    }
  }
  os.unindent();
  return success();
}

/// Returns true if `type` is an opaque `pl.Tensor` annotation, as opposed to
/// device-level opaque values such as `pl.Tile` or `pl.Scalar`.
static bool isPyptoTensorType(Type type) {
  auto opaqueType = dyn_cast<mlir::emitpy::OpaqueType>(type);
  return opaqueType && opaqueType.getValue().starts_with("pl.Tensor");
}

/// Collects the tensor-level view ops of a PyPTO kernel body: ops that view
/// an opaque pl.Tensor (a kernel parameter or another tensor-level view) and
/// produce a memref, e.g. the pl.reshape calls generated from
/// memref.collapse_shape/expand_shape. Such ops must be emitted at host
/// level: inside the CORE_GROUP scope of pl.at they would be materialized as
/// tile ops operating on tensor values, which tile-level loads reject.
static llvm::DenseSet<Operation *> collectTensorViewOps(Block &entry) {
  llvm::DenseSet<Operation *> viewOps;
  for (Operation &op : entry.without_terminator()) {
    if (op.getNumResults() != 1 || !isa<MemRefType>(op.getResult(0).getType())) {
      continue;
    }
    bool viewsTensor = llvm::any_of(op.getOperands(), [&](Value operand) {
      Operation *defOp = operand.getDefiningOp();
      return isPyptoTensorType(operand.getType()) || (defOp != nullptr && viewOps.contains(defOp));
    });
    if (viewsTensor) {
      viewOps.insert(&op);
    }
  }
  return viewOps;
}

/// Prints the body of a @pl.jit PyPTO kernel: the operations are wrapped in
/// a pl.at scope with the terminating return outside it. Multi-core kernels
/// replace that scope with an SPMD launch: a pl.at wrapping pl.spmd has no
/// device counterpart, so kernels containing an spmd_scope emit their body
/// without the wrapper. Tensor-level views of the kernel parameters are
/// emitted before the device scope (see collectTensorViewOps).
static LogicalResult printPyptoKernelBody(PythonEmitter &emitter, func::FuncOp functionOp) {
  raw_indented_ostream &os = emitter.ostream();
  if (functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError("PyPTO kernels must have a single block");
  }
  Block &entry = functionOp.getBlocks().front();
  bool hasSpmd = false;
  entry.walk([&](SpmdScopeOp) { hasSpmd = true; });
  os.indent();
  llvm::DenseSet<Operation *> viewOps = collectTensorViewOps(entry);
  for (Operation &bodyOp : entry.without_terminator()) {
    if (!viewOps.contains(&bodyOp)) {
      continue;
    }
    if (failed(emitter.emitOperation(bodyOp))) {
      return failure();
    }
  }
  if (!hasSpmd) {
    os << "with pl.at(level=pl.Level.CORE_GROUP):\n";
    os.indent();
  }
  bool emittedInScope = false;
  for (Operation &bodyOp : entry.without_terminator()) {
    if (viewOps.contains(&bodyOp)) {
      continue;
    }
    if (failed(emitter.emitOperation(bodyOp))) {
      return failure();
    }
    emittedInScope = true;
  }
  if (!emittedInScope) {
    os << "pass\n";
  }
  if (!hasSpmd) {
    os.unindent();
  }
  LogicalResult terminatorStatus = emitter.emitOperation(entry.getOperations().back());
  os.unindent();
  if (failed(terminatorStatus)) {
    return failure();
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, func::FuncOp functionOp) {
  if (functionOp.isDeclaration()) {
    return success();
  }
  PythonEmitter::Scope scope(emitter);
  Operation &op = *functionOp.getOperation();
  raw_indented_ostream &os = emitter.ostream();
  StringRef callee = functionOp.getName();
  emitter.reserveName(callee.str());
  bool isPyptoKernel = functionOp->getAttr(kPyptoKernelAttr) != nullptr;
  if (auto methodKindAttr = functionOp->getAttrOfType<StringAttr>("emitpy.method_kind")) {
    StringRef methodKind = methodKindAttr.getValue();
    if (!emitter.isInClassScope()) {
      return functionOp.emitOpError("emitpy.method_kind is only valid inside a class");
    }
    if (methodKind == "classmethod") {
      os << "@classmethod\n";
    } else if (methodKind == "staticmethod") {
      os << "@staticmethod\n";
    }
  }
  if (isPyptoKernel) {
    os << "@pl.jit\n";
  }
  os << "def";
  os << " " << callee;
  os << "(";
  if (failed(printFunctionArgs(emitter, op, functionOp.getArguments()))) {
    return failure();
  }
  os << "): \n";

  if (isPyptoKernel) {
    if (failed(printPyptoKernelBody(emitter, functionOp))) {
      return failure();
    }
    return success();
  }

  if (failed(printFunctionBody(emitter, op, functionOp.getBlocks()))) {
    return failure();
  }

  if (!emitter.isInClassScope() && !isPyptoKernel && callee == "main") {
    os << "\n";
    os << "if __name__ == \'__main__\':\n";
    os << "  main()\n";
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, func::ReturnOp returnOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "return";
  // PyPTO kernels hand results back through their Out parameters: the IR
  // return carries no operands and the emitter reconstructs the value list
  // from the recorded output argument indices.
  func::FuncOp funcOp = returnOp->getParentOfType<func::FuncOp>();
  if (returnOp.getNumOperands() == 0 && funcOp && funcOp->getAttr(kPyptoKernelAttr)) {
    if (auto outArgs = funcOp->getAttrOfType<DenseI64ArrayAttr>(kOutArgsAttr)) {
      bool first = true;
      for (int64_t idx : outArgs.asArrayRef()) {
        if (!first) {
          os << ",";
        }
        first = false;
        os << " ";
        if (failed(emitter.emitOperand(funcOp.getArgument(idx), "out_arg"))) {
          return failure();
        }
      }
    }
    return success();
  }
  switch (returnOp.getNumOperands()) {
    case 0:
      break;
    case 1: {
      os << " ";
      if (failed(emitter.emitOperand(returnOp.getOperand(0), "return_arg"))) {
        return failure();
      }
      break;
    }
    default: {
      os << " ";
      if (failed(emitter.emitOperands(*returnOp.getOperation()))) {
        return failure();
      }
    }
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, SubscriptOp subscriptOp) {
  if (failed(emitter.emitAssignPrefix(*subscriptOp))) {
    return failure();
  }

  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitOperand(subscriptOp.getOperand(0), "subscript_arg"))) {
    return failure();
  }

  os << emitter.getSubscriptName(subscriptOp);

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, GetAttrOp getAttrOp) {
  if (failed(emitter.emitAssignPrefix(*getAttrOp))) {
    return failure();
  }

  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitOperand(getAttrOp.getObject(), "object"))) {
    return failure();
  }

  os << "." << getAttrOp.getAttrName();
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, SetAttrOp setAttrOp) {
  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitOperand(setAttrOp.getObject(), "object"))) {
    return failure();
  }

  os << "." << setAttrOp.getAttrName() << " = ";
  if (failed(emitter.emitOperand(setAttrOp.getValue(), "value"))) {
    return failure();
  }
  return success();
}

static LogicalResult emitClassBase(Attribute baseAttr, raw_indented_ostream &os) {
  if (auto symbolRef = dyn_cast<SymbolRefAttr>(baseAttr)) {
    os << symbolRef.getLeafReference();
    return success();
  }
  if (auto opaque = dyn_cast<mlir::emitpy::OpaqueAttr>(baseAttr)) {
    os << opaque.getValue();
    return success();
  }
  return failure();
}

static LogicalResult printOperation(PythonEmitter &emitter, ClassOp classOp) {
  PythonEmitter::Scope scope(emitter);
  PythonEmitter::ClassScope classScope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << "class " << classOp.getSymName();
  if (auto baseClasses = classOp.getBaseClasses()) {
    if (!baseClasses->empty()) {
      os << "(";
      if (failed(interleaveCommaWithError(*baseClasses, os,
                                          [&](Attribute baseAttr) { return emitClassBase(baseAttr, os); }))) {
        return classOp.emitOpError("invalid class base attribute");
      }
      os << ")";
    }
  }
  os << ":\n";

  os.indent();
  Block &body = classOp.getBody().front();
  if (body.empty()) {
    os << "pass\n";
    os.unindent();
    return success();
  }

  for (Operation &op : body.getOperations()) {
    if (failed(emitter.emitOperation(op))) {
      return failure();
    }
  }
  os.unindent();
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, NestedFuncOp funcOp) {
  emitter.reserveName(funcOp.getSymName().str());
  PythonEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << "def " << funcOp.getSymName() << "(";
  Block &body = funcOp.getBody().front();
  if (failed(interleaveCommaWithError(body.getArguments(), os,
                                      [&](BlockArgument arg) { return emitter.emitOperand(arg, "arg"); }))) {
    return failure();
  }
  os << "): \n";

  os.indent();
  for (Operation &op : body.getOperations()) {
    if (failed(emitter.emitOperation(op))) {
      return failure();
    }
  }
  os.unindent();
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, NestedFuncReturnOp returnOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "return";
  if (returnOp.getNumOperands() > 0) {
    os << " ";
    if (failed(emitter.emitOperands(*returnOp.getOperation()))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, ConstantOp constantOp) {
  Attribute value = constantOp.getValue();

  if (failed(emitter.emitAssignPrefix(*constantOp))) {
    return failure();
  }

  return emitter.emitAttribute(constantOp->getLoc(), value);
}

static LogicalResult printOperation(PythonEmitter &emitter, GlobalOp globalOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << globalOp.getSymName() << " = ";
  if (failed(emitter.emitAttribute(globalOp->getLoc(), globalOp.getInitialValue()))) {
    return failure();
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, AssignGlobalOp assignGlobalOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << assignGlobalOp.getName() << " = ";
  if (failed(emitter.emitOperand(assignGlobalOp.getValue(), assignGlobalOp.getName().str()))) {
    return failure();
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, GlobalStatementOp globalStatementOp) {
  emitter.registerDeferredValue(globalStatementOp.getResult(), globalStatementOp.getName());

  return emitter.emitGlobalStatement(globalStatementOp);
}

static LogicalResult printOperation(PythonEmitter &emitter, CreateDictOp createDictOp) {
  raw_indented_ostream &os = emitter.ostream();
  StringRef dictName = createDictOp.getDictName();

  emitter.registerDeferredValue(createDictOp.getResult(), dictName);

  os << dictName << " = ";

  if (createDictOp.getLiteralExpr()) {
    os << *createDictOp.getLiteralExpr();
    return success();
  }

  os << "{";
  auto items = createDictOp.getItems();
  for (size_t i = 0; i < items.size(); i += 2) {
    if (i > 0) {
      os << ", ";
    }
    if (failed(emitter.emitOperand(items[i], "dict_key"))) {
      return failure();
    }
    os << ": ";
    if (failed(emitter.emitOperand(items[i + 1], "dict_value"))) {
      return failure();
    }
  }
  os << "}";

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, AssignOp op) {
  raw_indented_ostream &os = emitter.ostream();

  if (failed(emitter.emitOperand(op.getTarget(), "target"))) {
    return failure();
  }

  os << " = ";

  if (failed(emitter.emitOperand(op.getValue(), "value"))) {
    return failure();
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, YieldOp yieldOp) {
  // YieldOp should only appear inside ExpressionOp
  // In the new design, yield should never be directly emitted
  // It's handled by the ExpressionBuilder
  return yieldOp.emitOpError("yield operation should not be directly emitted");
}

static LogicalResult printOperation(PythonEmitter &emitter, WhileYieldOp yieldOp) {
  // The loop-carried assignment is emitted by the parent, which is the only
  // place that knows the variable names.
  return yieldOp.emitOpError("while_yield operation should not be directly emitted");
}

// Helper function to build an expression string
static FailureOr<std::string> buildExpressionString(ExpressionOp expressionOp, PythonEmitter &emitter) {
  ExpressionBuilder builder(emitter);

  // Map block arguments to operands
  Block *bodyBlock = expressionOp.getBodyBlock();
  for (auto [blockArg, operand] : llvm::zip(bodyBlock->getArguments(), expressionOp.getOperands())) {
    // Get the name from the parent emitter and register it in the builder
    std::string argName = emitter.getOrCreateName(operand, "expr_arg").str();
    builder.mapBlockArgument(blockArg, argName);
  }

  // Find the yield op and build its expression
  auto yieldOp = cast<YieldOp>(bodyBlock->getTerminator());
  Value yieldValue = yieldOp.getResult();

  // Build the expression recursively
  return builder.buildExpression(yieldValue);
}

static LogicalResult printOperation(PythonEmitter &emitter, IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  auto emitConditionAndThenBody = [&](IfOp op, StringRef keyword) -> LogicalResult {
    os << keyword;

    auto items = op.parseFormatString();
    if (failed(items)) {
      return failure();
    }

    size_t idx = 0;
    for (auto &item : *items) {
      if (auto *str = std::get_if<StringRef>(&item)) {
        os << *str;
      } else {
        if (failed(emitter.emitOperand(op.getCondArgs()[idx++], ""))) {
          return failure();
        }
      }
    }

    os << ":\n";

    os.indent();
    for (Operation &bodyOp : op.getThenRegion().front().getOperations()) {
      if (failed(emitter.emitOperation(bodyOp))) {
        return failure();
      }
    }
    os.unindent();

    return success();
  };

  if (failed(emitConditionAndThenBody(ifOp, "if "))) {
    return failure();
  }

  while (!ifOp.getElseRegion().empty()) {
    Block &elseBlock = ifOp.getElseRegion().front();
    auto *firstOp = &elseBlock.front();
    if (auto nestedIf = dyn_cast<IfOp>(firstOp);
        nestedIf && elseBlock.getOperations().size() == 1 && nestedIf.getElseRegion().empty()) {
      if (failed(emitConditionAndThenBody(nestedIf, "elif "))) {
        return failure();
      }
      ifOp = nestedIf;
      continue;
    }
    os << "else:\n";
    os.indent();
    for (Operation &bodyOp : elseBlock.getOperations()) {
      if (failed(emitter.emitOperation(bodyOp))) {
        return failure();
      }
    }
    os.unindent();
    break;
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, VerbatimOp verbatimOp) {
  raw_indented_ostream &os = emitter.ostream();

  auto items = verbatimOp.parseFormatString();
  if (failed(items)) {
    return failure();
  }

  size_t idx = 0;
  for (auto &item : *items) {
    if (auto *str = std::get_if<StringRef>(&item)) {
      os << *str;
    } else {
      if (failed(emitter.emitOperand(verbatimOp.getFmtArgs()[idx++], ""))) {
        return failure();
      }
    }
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, WhileOp whileOp) {
  raw_indented_ostream &os = emitter.ostream();
  Block &body = whileOp.getBody().front();

  // The init, body block argument, yielded value and result of each
  // loop-carried value are one Python variable. Name it once, here, so it goes
  // through the emitter's collision check, and bind the rest to that name.
  //
  // No Scope is opened for the body: a Python loop body does not introduce one,
  // and the results have to stay visible after the loop.
  SmallVector<std::string> names;
  names.reserve(whileOp.getInits().size());
  for (auto [index, init] : llvm::enumerate(whileOp.getInits())) {
    names.emplace_back(emitter.getOrCreateName(body.getArgument(index), "carried_" + std::to_string(index)));
    const std::string &name = names.back();
    emitter.mapValueToName(whileOp.getResult(index), name);

    os << name << " = ";
    if (failed(emitter.emitOperand(init, ""))) {
      return failure();
    }
    os << "\n";
  }

  if (std::optional<int64_t> tripCount = whileOp.getTripCount()) {
    os << "for _ in range(" << *tripCount << "):\n";
  } else {
    os << "while ";

    auto items = whileOp.parseFormatString();
    if (failed(items)) {
      return failure();
    }

    size_t idx = 0;
    for (auto &item : *items) {
      if (auto *str = std::get_if<StringRef>(&item)) {
        os << *str;
      } else {
        if (failed(emitter.emitOperand(whileOp.getCondArgs()[idx++], ""))) {
          return failure();
        }
      }
    }

    os << ":\n";
  }

  os.indent();
  for (Operation &bodyOp : body.without_terminator()) {
    if (failed(emitter.emitOperation(bodyOp))) {
      return failure();
    }
  }

  if (names.empty()) {
    // Nothing carried, so nothing to assign; the body may well be empty too
    // and Python has no empty block.
    if (body.without_terminator().empty()) {
      os << "pass\n";
    }
  } else {
    // One tuple assignment, so the carry-back is simultaneous. A body that
    // yields its arguments in a different order would otherwise clobber a
    // value it still has to read.
    llvm::interleaveComma(names, os);
    os << " = ";
    if (failed(interleaveCommaWithError(whileOp.getYield().getResults(), os,
                                        [&](Value value) { return emitter.emitOperand(value, ""); }))) {
      return failure();
    }
    os << "\n";
  }
  os.unindent();

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, ForOp forOp) {
  raw_indented_ostream &os = emitter.ostream();
  Block &body = forOp.getBody().front();

  // The induction variable is the loop header's Python variable. No Scope is
  // opened for the body: a Python loop body does not introduce one (mirrors
  // emitpy.while).
  std::string name = emitter.getOrCreateName(body.getArgument(0), "loop_var").str();
  os << "for " << name << " in pl.range(";
  if (failed(emitter.emitOperand(forOp.getLowerBound(), ""))) {
    return failure();
  }
  os << ", ";
  if (failed(emitter.emitOperand(forOp.getUpperBound(), ""))) {
    return failure();
  }
  os << ", ";
  if (failed(emitter.emitOperand(forOp.getStep(), ""))) {
    return failure();
  }
  os << "):\n";
  os.indent();
  if (body.empty()) {
    os << "pass\n";
  } else {
    for (Operation &bodyOp : body.getOperations()) {
      if (failed(emitter.emitOperation(bodyOp))) {
        return failure();
      }
    }
  }
  os.unindent();

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, SpmdScopeOp spmdOp) {
  raw_indented_ostream &os = emitter.ostream();
  Block &body = spmdOp.getBody().front();

  // The block-index argument is the executing core's id; bind it to
  // pl.tile.get_block_idx() as the first body statement. No Scope is opened
  // for the body: a Python `with` block does not introduce one.
  std::string name = emitter.getOrCreateName(body.getArgument(0), "block_idx").str();
  os << "with pl.spmd(" << spmdOp.getCoreNum() << "):\n";
  os.indent();
  os << name << " = pl.tile.get_block_idx()\n";
  for (Operation &bodyOp : body.getOperations()) {
    if (failed(emitter.emitOperation(bodyOp))) {
      return failure();
    }
  }
  os.unindent();

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, ExpressionOp expressionOp) {
  Block *bodyBlock = expressionOp.getBodyBlock();

  bool shouldInline = !expressionOp.getDoNotInline();

  if (shouldInline) {
    // For inline expressions, build the expression string and cache it
    auto exprStrResult = buildExpressionString(expressionOp, emitter);
    if (failed(exprStrResult)) {
      return failure();
    }

    // Cache the expression for deferred emission
    emitter.registerDeferredValue(expressionOp.getResult(), *exprStrResult);

    // Return success - the expression will be emitted when it's used
    return success();
  } else {
    // Emit non-inline: emit the body operations normally
    PythonEmitter::Scope scope(emitter);

    // Map block arguments to operands
    for (auto [blockArg, operand] : llvm::zip(bodyBlock->getArguments(), expressionOp.getOperands())) {
      emitter.registerDeferredValue(blockArg, emitter.getOrCreateName(operand, "expr_arg"));
    }

    // Emit all operations in the body except the yield
    for (Operation &bodyOp : bodyBlock->getOperations()) {
      if (isa<YieldOp>(bodyOp)) {
        // For the yield, just emit an assignment to the expression result
        auto yieldOp = cast<YieldOp>(bodyOp);
        if (failed(emitter.emitAssignPrefix(*expressionOp))) {
          return failure();
        }
        return emitter.emitOperand(yieldOp.getResult(), "");
      }
      if (failed(emitter.emitOperation(bodyOp))) {
        return failure();
      }
    }
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, FileOp fileOp) {
  if (!emitter.shouldEmitFile(fileOp)) {
    return success();
  }

  for (Operation &op : fileOp.getRegion().getOps()) {
    if (failed(emitter.emitOperation(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult PythonEmitter::emitOperation(Operation &op) {
  LogicalResult status =
    llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      // Builtin ops.
      .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
      // EmitPy ops.
      .Case<CallOpaqueOp, ImportOp, AssignOp, GetAttrOp, SetAttrOp, ConstantOp, SubscriptOp, ClassOp, GlobalOp,
            AssignGlobalOp, GlobalStatementOp, CreateDictOp, ExpressionOp, YieldOp, IfOp, WhileOp, WhileYieldOp,
            VerbatimOp, NestedFuncOp, NestedFuncReturnOp, FileOp, ForOp, SpmdScopeOp>(
        [&](auto op) { return printOperation(*this, op); })
      .Case<LiteralOp>([&](auto op) {
        registerDeferredValue(op.getResult(), op.getValue());
        return success();
      })
      // Func ops.
      .Case<func::CallOp, func::FuncOp, func::ReturnOp>([&](auto op) { return printOperation(*this, op); })
      .Default([&](Operation *) { return op.emitOpError("unable to find printer for op"); });

  if (failed(status)) {
    return failure();
  }

  if (hasDeferredEmission(&op)) {
    return success();
  }

  if (!isa<IfOp, WhileOp, ForOp, SpmdScopeOp>(op)) {
    os << "\n";
  }

  return success();
}

LogicalResult PythonEmitter::emitOperand(Value value, std::string name) {
  // Check if this value is from an inline expression that has been cached
  if (auto *defOp = value.getDefiningOp()) {
    if (auto exprOp = dyn_cast<ExpressionOp>(defOp)) {
      if (!exprOp.getDoNotInline() && valueMapper.count(value)) {
        // Emit the cached expression string directly
        os << valueMapper.lookup(value);
        return success();
      }
    }
  }

  // Default behavior: emit the variable name
  os << getOrCreateName(value, name);

  return success();
}

LogicalResult PythonEmitter::emitOperands(Operation &op) {
  return interleaveCommaWithError(op.getOperands(), os, [&](Value value) {
    if (failed(emitOperand(value, "op_arg"))) {
      return failure();
    }

    return success();
  });
}

LogicalResult PythonEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](APInt attr) {
    SmallString<128> strValue;
    attr.toString(strValue, 10, true);
    return strValue;
  };
  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    os << printInt(iAttr.getValue());
    return success();
  }
  // Print float attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    APFloat value = fAttr.getValue();
    if (value.isInfinity()) {
      os << (value.isNegative() ? "float('-inf')" : "float('inf')");
    } else if (value.isNaN()) {
      os << "float('nan')";
    } else {
      SmallString<128> strValue;
      value.toString(strValue);
      os << strValue;
    }
    return success();
  }
  // Print string attributes.
  if (auto sAttr = dyn_cast<StringAttr>(attr)) {
    os << "\"" << sAttr.getValue() << "\"";
    return success();
  }
  // Print opaque attributes.
  if (auto oAttr = dyn_cast<mlir::emitpy::OpaqueAttr>(attr)) {
    os << oAttr.getValue();
    return success();
  }

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult PythonEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
    case 0:
      break;
    case 1: {
      if (failed(emitVariableAssignment(op))) {
        return failure();
      }
      break;
    }
    default: {
      interleaveComma(op.getResults(), os, [&](Value result) {
        std::string name = "v_" + std::to_string(valueInScopeCount.top()++);
        os << getOrCreateName(result, name);
      });
      os << " = ";
    }
  }
  return success();
}

LogicalResult PythonEmitter::emitGlobalStatement(GlobalStatementOp globalStatementOp) {
  os << "global " << globalStatementOp.getName();

  return success();
}

bool isValidPythonIdentifier(const std::string &name) {
  static const std::regex pattern(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
  static const std::unordered_set<std::string> keywords = {
    "False", "None",     "True",  "and",    "as",   "assert", "async",  "await",    "break",
    "class", "continue", "def",   "del",    "elif", "else",   "except", "finally",  "for",
    "from",  "global",   "if",    "import", "in",   "is",     "lambda", "nonlocal", "not",
    "or",    "pass",     "raise", "return", "try",  "while",  "with",   "yield"};

  return std::regex_match(name, pattern) && (keywords.find(name) == keywords.end());
}

std::string validateVariableName(const std::string &name) {
  std::string result;

  // Replace illegal characters with underscores
  for (char c : name) {
    if (std::isalnum(c) || c == '_') {
      result += c;
    } else {
      result += '_';
    }
  }

  if (!result.empty() && std::isdigit(result[0])) {
    result = "var_" + result;
  }

  if (!isValidPythonIdentifier(result)) {
    result = "";
  }

  return result;
}

// Allocate a variable name for the single result of the operation.
std::string PythonEmitter::allocateResultName(Operation &op) {
  std::string name = "";

  if (auto suggestNameAttr = op.getAttr("emitpy.name")) {
    name = mlir::cast<StringAttr>(suggestNameAttr).getValue();
    name = validateVariableName(name);
  }
  if (name.empty()) {
    name = "var_" + std::to_string(valueInScopeCount.top()++);
  }
  return getOrCreateName(op.getResult(0), name).str();
}

// Assign a variable name to the result of the operation.
LogicalResult PythonEmitter::emitVariableAssignment(Operation &op) {
  os << allocateResultName(op) << " = ";
  return success();
}

LogicalResult mlir::emitpy::translateToPython(Operation *op, raw_ostream &os, std::optional<std::string> fileId) {
  PythonEmitter emitter(os, std::move(fileId));
  return emitter.emitOperation(*op);
}
