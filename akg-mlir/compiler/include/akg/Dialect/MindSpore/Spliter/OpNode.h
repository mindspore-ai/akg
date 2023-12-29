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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_OPNODE_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_OPNODE_H_

#include "Node.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::spliter {
class PrimOp : public Node {
 public:
  enum class ComputeType : int {
    VIRTUAL = 0,
    RESHAPE = 1,
    ELEMWISE = 2,
    BROADCAST = 3,
    REDUCE = 4,
    OPAQUE = 5,
  };

  PrimOp(const std::string &newOp, ComputeType compute)
      : Node({{}, nullptr, kOpFormat_DEFAULT}), op(newOp), computeType(compute) {}
  ~PrimOp() = default;
  std::string toString() const override;
  NType nodeType() override { return NType::Primitive; }

  const std::string &getOp() const { return op; }
  ComputeType getComputeType() const { return computeType; }

 protected:
  std::string op;
  ComputeType computeType;
};

using PrimOpPtr = std::shared_ptr<PrimOp>;

class ReshapeOp : public PrimOp {
 public:
  explicit ReshapeOp(const std::string &op) : PrimOp(op, ComputeType::RESHAPE) {}
  ~ReshapeOp() = default;
};

class ElemwiseOp : public PrimOp {
 public:
  explicit ElemwiseOp(const std::string &op) : PrimOp(op, ComputeType::ELEMWISE) {}
  ~ElemwiseOp() = default;
};

class BroadcastOp : public PrimOp {
 public:
  explicit BroadcastOp(const std::string &op) : PrimOp(op, ComputeType::BROADCAST) {}
  ~BroadcastOp() = default;
};

class TileOp : public BroadcastOp {
 public:
  explicit TileOp(const std::string &op) : BroadcastOp(op) {}
  ~TileOp() = default;
};

class ReduceOp : public PrimOp {
 public:
  explicit ReduceOp(const std::string &op) : PrimOp(op, ComputeType::REDUCE) {}
  ~ReduceOp() = default;
};

class ArgReduceOp : public ReduceOp {
 public:
  explicit ArgReduceOp(const std::string &op) : ReduceOp(op) {}
  ~ArgReduceOp() = default;
};

class OpaqueOp : public PrimOp {
 public:
  explicit OpaqueOp(const std::string &op) : PrimOp(op, ComputeType::OPAQUE) {}
  ~OpaqueOp() = default;

 protected:
  // for pclint warning: 1790 public base symbol of symbol has no non-destructor
  // virtual functions
  virtual void doNothing() {}
};

class VirtualOp : public PrimOp {
 public:
  explicit VirtualOp(const std::string &op) : PrimOp(op, ComputeType::VIRTUAL) {}
  ~VirtualOp() = default;
};

class TransposeOp : public OpaqueOp {
 public:
  explicit TransposeOp(const std::string &op) : OpaqueOp(op) {}
  ~TransposeOp() = default;
};

class OneHotOp : public OpaqueOp {
 public:
  explicit OneHotOp(const std::string &op) : OpaqueOp(op) {}
  ~OneHotOp() = default;
};

class LayoutTransformOp : public OpaqueOp {
 public:
  explicit LayoutTransformOp(const std::string &op) : OpaqueOp(op) {}
  ~LayoutTransformOp() = default;
};

class ElemAnyOp : public OpaqueOp {
 public:
  explicit ElemAnyOp(const std::string &op) : OpaqueOp(op) {}
  ~ElemAnyOp() = default;
};

class ShapeOp : public OpaqueOp {
 public:
  explicit ShapeOp(const std::string &op) : OpaqueOp(op) {}
  ~ShapeOp() = default;
};

class ConstantOfShapeOp : public OpaqueOp {
 public:
  explicit ConstantOfShapeOp(const std::string &op) : OpaqueOp(op) {}
  ~ConstantOfShapeOp() = default;
};

class PadAkgOp : public OpaqueOp {
 public:
  explicit PadAkgOp(const std::string &op) : OpaqueOp(op) {}
  ~PadAkgOp() = default;
};

class UnPadAkgOp : public OpaqueOp {
 public:
  explicit UnPadAkgOp(const std::string &op) : OpaqueOp(op) {}
  ~UnPadAkgOp() = default;
};

class Conv2dOp : public OpaqueOp {
 public:
  explicit Conv2dOp(const std::string &op) : OpaqueOp(op) {}
  ~Conv2dOp() = default;
};

class GatherOp : public OpaqueOp {
 public:
  explicit GatherOp(const std::string &op) : OpaqueOp(op) {}
  ~GatherOp() = default;
};

class ConcatOp : public OpaqueOp {
 public:
  explicit ConcatOp(const std::string &op) : OpaqueOp(op) {}
  ~ConcatOp() = default;
};

class CImagRealOp : public ElemwiseOp {
 public:
  explicit CImagRealOp(const std::string &op) : ElemwiseOp(op) {}
  ~CImagRealOp() = default;
};

class Pool2DOp : public OpaqueOp {
 public:
  explicit Pool2DOp(const std::string &op) : OpaqueOp(op) {}
  ~Pool2DOp() = default;
};

class ComplexOp : public ElemwiseOp {
 public:
  explicit ComplexOp(const std::string &op) : ElemwiseOp(op) {}
  ~ComplexOp() = default;
};

class StandardNormalOp : public OpaqueOp {
 public:
  explicit StandardNormalOp(const std::string &op) : OpaqueOp(op) {}
  ~StandardNormalOp() = default;
};

class StridedSliceOp : public OpaqueOp {
 public:
  explicit StridedSliceOp(const std::string &op) : OpaqueOp(op) {}
  ~StridedSliceOp() = default;
};

class StridedSliceOnnxOp : public OpaqueOp {
 public:
  explicit StridedSliceOnnxOp(const std::string &op) : OpaqueOp(op) {}
  ~StridedSliceOnnxOp() = default;
};

class MatMulOp : public OpaqueOp {
 public:
  explicit MatMulOp(const std::string &op) : OpaqueOp(op) {}
  ~MatMulOp() = default;
};

class TupleGetItemOp : public VirtualOp {
 public:
  using VirtualOp::VirtualOp;
  ~TupleGetItemOp() = default;
};
}  // namespace mlir::spliter
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_OPNODE_H_
