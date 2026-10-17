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

#include "akg/Conversion/PyptoToEmitPy/PyptoToEmitPy.h"

#include <optional>
#include <string>

#include "akg/Dialect/EmitPy/IR/EmitPyOps.h"
#include "akg/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "akg/Dialect/Pypto/IR/PyptoOps.h"
#include "akg/Dialect/Pypto/IR/PyptoTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;  // NOLINT(build/namespaces)

namespace emitpy = mlir::emitpy;
namespace pypto = mlir::pypto;

namespace mlir {

#define GEN_PASS_DEF_CONVERTPYPTOTOEMITPY
#include "akg/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// PyPTO type mapping
//===----------------------------------------------------------------------===//

/// Maps an MLIR scalar element type onto the PyPTO dtype name, e.g.
/// f32 -> "pl.FP32", i32 -> "pl.INT32". Mirrors the mapping in
/// VectorToPypto.cpp; kept local so the pypto lowering does not depend on
/// the vector-to-emitpy conversion library.
static std::optional<std::string> pyptoDtypeName(Type type) {
  if (isa<FloatType>(type)) {
    if (type.isF16()) {
      return std::string("pl.FP16");
    }
    if (type.isBF16()) {
      return std::string("pl.BF16");
    }
    if (type.isF32()) {
      return std::string("pl.FP32");
    }
    return std::nullopt;  // f64 and friends have no PyPTO dtype
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    bool unsignedInt = intType.isUnsigned();
    switch (intType.getWidth()) {
    case 8:
      return std::string(unsignedInt ? "pl.UINT8" : "pl.INT8");
    case 16:
      return std::string(unsignedInt ? "pl.UINT16" : "pl.INT16");
    case 32:
      return std::string(unsignedInt ? "pl.UINT32" : "pl.INT32");
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

/// The dtype name carried by an op result: a tile's element type, or the
/// scalar type itself.
static std::optional<std::string> resultDtypeName(Type type) {
  if (auto tile = dyn_cast<pypto::TileType>(type)) {
    return pyptoDtypeName(tile.getElementType());
  }
  return pyptoDtypeName(type);
}

/// Renders a shape as a Python list literal, e.g. {4, 8} -> "[4, 8]".
static std::string shapeToPythonListText(ArrayRef<int64_t> shape) {
  std::string text;
  llvm::raw_string_ostream rso(text);
  rso << "[";
  llvm::interleaveComma(shape, rso);
  rso << "]";
  rso.flush();
  return text;
}

/// Extracts the constant integer of an index operand, or nullopt when the
/// operand is a runtime value.
static std::optional<int64_t> constIndexValue(Value idx) {
  auto constOp = idx.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return std::nullopt;
  }
  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr) {
    return std::nullopt;
  }
  return intAttr.getInt();
}

/// Builds the `[i0, i1, ...]` offsets argument for pl.load/pl.store.
/// Constant-only offsets render directly as an opaque list literal; dynamic
/// offsets become operand references (IntegerAttr of index type) and the
/// index values are appended to `operands`, matching the call_opaque
/// argument protocol used by the vector-to-emitpy lowering.
static Attribute buildOffsetsArg(ConversionPatternRewriter &rewriter,
                                 ValueRange offsets,
                                 SmallVectorImpl<Value> &operands) {
  bool allConstant = llvm::all_of(
      offsets, [](Value offset) { return constIndexValue(offset).has_value(); });
  if (allConstant) {
    std::string text = "[";
    bool first = true;
    for (Value offset : offsets) {
      if (!first) {
        text += ", ";
      }
      first = false;
      text += std::to_string(*constIndexValue(offset));
    }
    text += "]";
    return rewriter.getAttr<emitpy::OpaqueAttr>(text);
  }
  SmallVector<Attribute> items;
  for (Value offset : offsets) {
    operands.push_back(offset);
    items.push_back(rewriter.getIndexAttr(operands.size() - 1));
  }
  return rewriter.getArrayAttr(items);
}

/// The reduce-kind suffix shared by the `pl.row_*` / `pl.col_*` families.
static StringRef reduceKindSuffix(pypto::ReduceKind kind) {
  switch (kind) {
  case pypto::ReduceKind::sum:
    return "sum";
  case pypto::ReduceKind::prod:
    return "prod";
  case pypto::ReduceKind::min:
    return "min";
  case pypto::ReduceKind::max:
    return "max";
  }
  return "";
}

/// The axis prefix of the `pl.row_*` / `pl.col_*` families: `row` folds the
/// last axis, `col` the first one.
static StringRef reduceAxisPrefix(pypto::ReduceAxis axis) {
  switch (axis) {
  case pypto::ReduceAxis::row:
    return "row_";
  case pypto::ReduceAxis::col:
    return "col_";
  }
  return "";
}

/// Converts `!pypto.tile<elem, [shape]>` into the opaque PyPTO annotation
/// `!emitpy.opaque<"pl.Tile[[shape], dtype]">`: this is the single place
/// where the tile model is encoded as a Python-side annotation string.
class PyptoToEmitPyTypeConverter : public TypeConverter {
 public:
  PyptoToEmitPyTypeConverter() {
    // Fallback first: the framework tries conversions in reverse order of
    // registration, so the identity fallback must be added before the
    // tile-specific one (mirrors NPUVectorToVector.cpp).
    addConversion([](Type type) { return type; });
    // Returns `std::nullopt` (not an empty Type) when the element dtype has
    // no annotation spelling, so the framework falls back to the identity
    // conversion above: a null return is a hard failure and would leave
    // e.g. `!pypto.tile<i1, ...>` (cmp results) unconvertible.
    addConversion([](pypto::TileType type) -> std::optional<Type> {
      // A `?` extent cannot be spelled in the annotation; dynamic tiles
      // keep their source type, which the Python translator accepts for
      // call_opaque results (as it already does for dynamic vector tiles).
      if (llvm::any_of(type.getShape(), [](int64_t dim) {
            return ShapedType::isDynamic(dim);
          })) {
        return type;
      }
      auto dtype = pyptoDtypeName(type.getElementType());
      if (!dtype) {
        return std::nullopt;
      }
      std::string text;
      llvm::raw_string_ostream rso(text);
      rso << "pl.Tile[" << shapeToPythonListText(type.getShape()) << ", "
          << *dtype << "]";
      rso.flush();
      return emitpy::OpaqueType::get(type.getContext(), text);
    });
  }
};

//===----------------------------------------------------------------------===//
// Generic positional-call patterns
//===----------------------------------------------------------------------===//

/// Rewrites an op into `emitpy.call_opaque` with the given callee and all
/// operands passed positionally, e.g.
/// `pypto.add %a, %b` -> `%r = pl.add(a, b)`.
/// Covers the elementwise binaries/unaries, the scalar combines and
/// `pypto.select` (whose scratch tile is an explicit operand, unlike the
/// vector-side select that has to materialize one).
template <typename SourceOp>
class OpToCallOpaquePattern : public OpConversionPattern<SourceOp> {
 public:
  OpToCallOpaquePattern(TypeConverter &typeConverter, MLIRContext *context,
                        StringRef callee)
      : OpConversionPattern<SourceOp>(typeConverter, context), callee(callee) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, callee,
                                                      adaptor.getOperands());
    return success();
  }

 private:
  StringRef callee;
};

//===----------------------------------------------------------------------===//
// Constant and allocation producers
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.constant` into `pl.const(value, dtype)`: the value is
/// the Python literal text; the dtype keeps scalars from defaulting to
/// Python int/float64 semantics.
class ConstantOpPattern : public OpConversionPattern<pypto::ConstantOp> {
 public:
  using OpConversionPattern<pypto::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "constant type has no PyPTO dtype");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Attribute> args{
        rewriter.getAttr<emitpy::OpaqueAttr>(op.getValue()),
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.const", ValueRange{}, rewriter.getArrayAttr(args),
        rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.tile_full` into `pl.tile.full(shape, dtype, value)`:
/// a statically shaped result carries the shape in its type; a dynamic
/// result carries one extent operand per dimension (constants fold into
/// the list literal, runtime bounds become operand references, exactly
/// like offsets).
class TileFullOpPattern : public OpConversionPattern<pypto::TileFullOp> {
 public:
  using OpConversionPattern<pypto::TileFullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::TileFullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The result keeps the source type during the staged migration, so it
    // is either a `!pypto.tile<...>` (TileType is not a ShapedType) or
    // the original vector/npuvector type; both spell the element type
    // and, when static, the shape.
    Type sourceType = op.getResult().getType();
    Type elementType;
    SmallVector<int64_t> staticShape;
    bool hasStaticShape = false;
    if (auto tile = dyn_cast<pypto::TileType>(sourceType)) {
      elementType = tile.getElementType();
      if (!llvm::any_of(tile.getShape(), [](int64_t dim) {
            return ShapedType::isDynamic(dim);
          })) {
        staticShape = SmallVector<int64_t>(tile.getShape());
        hasStaticShape = true;
      }
    } else if (auto shaped = dyn_cast<ShapedType>(sourceType)) {
      elementType = shaped.getElementType();
      if (shaped.hasStaticShape()) {
        staticShape = SmallVector<int64_t>(shaped.getShape());
        hasStaticShape = true;
      }
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "tile_full result must be shaped");
    }
    auto dtype = pyptoDtypeName(elementType);
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "element type has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(sourceType);
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> fullOperands;
    Attribute shapeArg;
    if (adaptor.getOperands().empty()) {
      if (!hasStaticShape) {
        return rewriter.notifyMatchFailure(
            op, "tile_full without extents must be statically shaped");
      }
      shapeArg = rewriter.getAttr<emitpy::OpaqueAttr>(
          shapeToPythonListText(staticShape));
    } else {
      shapeArg =
          buildOffsetsArg(rewriter, adaptor.getOperands(), fullOperands);
    }
    SmallVector<Attribute> args{
        shapeArg,
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
        rewriter.getAttr<emitpy::OpaqueAttr>(op.getValue())};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.tile.full", ValueRange(fullOperands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.create_tile` into `pl.create_tile(shape, dtype=...,
/// target_memory=pl.MemorySpace.Vec)`: an uninitialized scratch tile in
/// vector-local memory.
class CreateTileOpPattern : public OpConversionPattern<pypto::CreateTileOp> {
 public:
  using OpConversionPattern<pypto::CreateTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::CreateTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tile = llvm::cast<pypto::TileType>(op.getResult().getType());
    if (llvm::any_of(tile.getShape(),
                     [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
      return rewriter.notifyMatchFailure(
          op, "pl.create_tile requires a statically shaped result");
    }
    auto dtype = pyptoDtypeName(tile.getElementType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "element type has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(tile);
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Attribute> args{
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(tile.getShape())),
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
        rewriter.getAttr<emitpy::OpaqueAttr>("pl.MemorySpace.Vec")};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("dtype"),
                                    rewriter.getStringAttr("target_memory")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.create_tile", ValueRange{},
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// View ops
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.slice` into `pl.slice(src, [window], [offsets],
/// drop_dims=[...])`: the window is full-rank over the source with unit
/// dims standing in for rank-reducing drops.
class SliceOpPattern : public OpConversionPattern<pypto::SliceOp> {
 public:
  using OpConversionPattern<pypto::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(op.getWindow())),
        offsetsArg};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    if (auto dropDims = op.getDropDims()) {
      if (!dropDims->empty()) {
        args.push_back(rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(*dropDims)));
        keywords.push_back(rewriter.getStringAttr("drop_dims"));
      }
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.slice", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.reshape` into `pl.reshape(src, [shape])`.
class ReshapeOpPattern : public OpConversionPattern<pypto::ReshapeOp> {
 public:
  using OpConversionPattern<pypto::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tile = dyn_cast<pypto::TileType>(op.getResult().getType());
    if (!tile || llvm::any_of(tile.getShape(), [](int64_t dim) {
          return ShapedType::isDynamic(dim);
        })) {
      return rewriter.notifyMatchFailure(
          op, "reshape result must be a statically shaped tile");
    }
    Type resultType = getTypeConverter()->convertType(tile);
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(tile.getShape()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.reshape", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.transpose` into `pl.transpose(src, i, j)`.
class TransposeOpPattern : public OpConversionPattern<pypto::TransposeOp> {
 public:
  using OpConversionPattern<pypto::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(op.getI())),
        rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(op.getJ()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.transpose", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.reinterpret_view` into
/// `pl.reinterpret_view(src, dtype)`: a bit-level reinterpretation of the
/// payload.
class ReinterpretViewOpPattern
    : public OpConversionPattern<pypto::ReinterpretViewOp> {
 public:
  using OpConversionPattern<
      pypto::ReinterpretViewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReinterpretViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "view type has no PyPTO dtype");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.reinterpret_view", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.cast` into `pl.cast(src, dtype)`.
class CastOpPattern : public OpConversionPattern<pypto::CastOp> {
 public:
  using OpConversionPattern<pypto::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "cast target has no PyPTO dtype");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.cast", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.cmp` into `pl.cmp(lhs, rhs, cmp_type=code)`: the enum
/// values already match the PyPTO comparison codes.
class CmpOpPattern : public OpConversionPattern<pypto::CmpOp> {
 public:
  using OpConversionPattern<pypto::CmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs()};
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(1),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            std::to_string(static_cast<uint32_t>(op.getPredicate())))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("cmp_type")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.cmp", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reduction and scalar IO
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.reduction` into `pl.row_<kind>(vector, scratch)` /
/// `pl.col_<kind>(vector, scratch)`: a keepdim in-place fold of the last
/// (row) or first (col) axis, working in the scratch tile.
class ReductionOpPattern : public OpConversionPattern<pypto::ReductionOp> {
 public:
  using OpConversionPattern<pypto::ReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef prefix = reduceAxisPrefix(op.getAxis());
    StringRef suffix = reduceKindSuffix(op.getKind());
    if (prefix.empty() || suffix.empty()) {
      return rewriter.notifyMatchFailure(
          op, "reduction kind/axis has no PyPTO counterpart");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    std::string callee = "pl." + prefix.str() + suffix.str();
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, callee,
        ValueRange{adaptor.getVector(), adaptor.getScratch()});
    return success();
  }
};

/// Rewrites `pypto.read` into `pl.read(src, [indices])`: the constant
/// coordinates render as a Python index list.
class ReadOpPattern : public OpConversionPattern<pypto::ReadOp> {
 public:
  using OpConversionPattern<pypto::ReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(op.getIndices()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.read", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.write` into `pl.write(dest, [offsets], value)`.
class WriteOpPattern : public OpConversionPattern<pypto::WriteOp> {
 public:
  using OpConversionPattern<pypto::WriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::WriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getDest(), adaptor.getValue()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0), offsetsArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "pl.write", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile movement between tensor memory and cores
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.load` into `pl.load`. Statically shaped tiles emit the
/// 3-argument form `pl.load(src, [offsets], [shape])`; dynamic tiles emit
/// the 4-argument form `pl.load(src, [offsets], [shapes], [valid])` with
/// the per-dimension allocation bounds and valid extents the op carries.
class LoadOpPattern : public OpConversionPattern<pypto::LoadOp> {
 public:
  using OpConversionPattern<pypto::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tile = llvm::cast<pypto::TileType>(op.getResult().getType());
    if (llvm::any_of(tile.getShape(), [](int64_t dim) {
          return ShapedType::isDynamic(dim);
        })) {
      return rewriteDynamicRead(op, adaptor, rewriter);
    }
    return rewriteStaticRead(op, tile, adaptor, rewriter);
  }

 private:
  LogicalResult rewriteStaticRead(pypto::LoadOp op, pypto::TileType tile,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    Type resultType = getTypeConverter()->convertType(tile);
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "tile element type has no PyPTO dtype");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        offsetsArg,
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(tile.getShape()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.load", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }

  LogicalResult rewriteDynamicRead(pypto::LoadOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    ValueRange paddedShapes = adaptor.getPaddedShapes();
    ValueRange validExtents = adaptor.getValidExtents();
    if (paddedShapes.empty() || validExtents.size() != paddedShapes.size()) {
      return rewriter.notifyMatchFailure(
          op, "dynamic load needs per-dimension padded/valid extents");
    }
    // A `?` extent cannot be spelled as an annotation; the call keeps the
    // source tile type, which the Python translator accepts for
    // call_opaque results (as it does for dynamic vector tiles).
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    Attribute shapesArg = buildOffsetsArg(rewriter, paddedShapes, operands);
    Attribute validArg = buildOffsetsArg(rewriter, validExtents, operands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), offsetsArg,
                                shapesArg, validArg};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.load", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.store` into `pl.store(t, [o...], dest)`; offsets may
/// be loop-dependent.
class StoreOpPattern : public OpConversionPattern<pypto::StoreOp> {
 public:
  using OpConversionPattern<pypto::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getValue(), adaptor.getDest()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0), offsetsArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "pl.store", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertPyptoToEmitPyPass
    : impl::ConvertPyptoToEmitPyBase<ConvertPyptoToEmitPyPass> {
  using impl::ConvertPyptoToEmitPyBase<
      ConvertPyptoToEmitPyPass>::ConvertPyptoToEmitPyBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitpy::EmitPyDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    PyptoToEmitPyTypeConverter typeConverter;
    ConversionTarget target(getContext());
    target.addIllegalDialect<pypto::PyptoDialect>();
    target.addLegalDialect<emitpy::EmitPyDialect>();
    // Tile-typed block arguments must be legalized together with their
    // signature, otherwise the conversion leaves dangling materializations.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op);
    });

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    patterns.add<ConstantOpPattern, TileFullOpPattern, CreateTileOpPattern,
                 SliceOpPattern, ReshapeOpPattern, TransposeOpPattern,
                 ReinterpretViewOpPattern, CastOpPattern, CmpOpPattern,
                 ReductionOpPattern, ReadOpPattern, WriteOpPattern,
                 LoadOpPattern, StoreOpPattern>(typeConverter,
                                                patterns.getContext());
    patterns.add<OpToCallOpaquePattern<pypto::AddOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.add");
    patterns.add<OpToCallOpaquePattern<pypto::SubOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.sub");
    patterns.add<OpToCallOpaquePattern<pypto::MulOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.mul");
    patterns.add<OpToCallOpaquePattern<pypto::DivOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.div");
    patterns.add<OpToCallOpaquePattern<pypto::FmodOp>>(typeConverter,
                                                       patterns.getContext(),
                                                       "pl.fmod");
    patterns.add<OpToCallOpaquePattern<pypto::MaximumOp>>(typeConverter,
                                                          patterns.getContext(),
                                                          "pl.maximum");
    patterns.add<OpToCallOpaquePattern<pypto::MinimumOp>>(typeConverter,
                                                          patterns.getContext(),
                                                          "pl.minimum");
    patterns.add<OpToCallOpaquePattern<pypto::AndOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.and_");
    patterns.add<OpToCallOpaquePattern<pypto::OrOp>>(typeConverter,
                                                     patterns.getContext(),
                                                     "pl.or_");
    patterns.add<OpToCallOpaquePattern<pypto::XorOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.xor");
    patterns.add<OpToCallOpaquePattern<pypto::ShlOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.shl");
    patterns.add<OpToCallOpaquePattern<pypto::ShrOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.shr");
    patterns.add<OpToCallOpaquePattern<pypto::NegOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.neg");
    patterns.add<OpToCallOpaquePattern<pypto::AbsOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.abs");
    patterns.add<OpToCallOpaquePattern<pypto::ExpOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.exp");
    patterns.add<OpToCallOpaquePattern<pypto::LogOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.log");
    patterns.add<OpToCallOpaquePattern<pypto::SqrtOp>>(typeConverter,
                                                       patterns.getContext(),
                                                       "pl.sqrt");
    patterns.add<OpToCallOpaquePattern<pypto::RsqrtOp>>(typeConverter,
                                                        patterns.getContext(),
                                                        "pl.rsqrt");
    patterns.add<OpToCallOpaquePattern<pypto::SinOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.sin");
    patterns.add<OpToCallOpaquePattern<pypto::CosOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.cos");
    patterns.add<OpToCallOpaquePattern<pypto::MaxOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.max");
    patterns.add<OpToCallOpaquePattern<pypto::MinOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.min");
    patterns.add<OpToCallOpaquePattern<pypto::SelectOp>>(typeConverter,
                                                         patterns.getContext(),
                                                         "pl.sel");

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace mlir

std::unique_ptr<Pass> mlir::createPyptoToEmitPyConversionPass() {
  return std::make_unique<ConvertPyptoToEmitPyPass>();
}
