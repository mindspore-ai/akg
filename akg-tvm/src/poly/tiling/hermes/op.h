/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef POLY_TILING_HERMES_OP_H_
#define POLY_TILING_HERMES_OP_H_

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace akg {
namespace ir {
namespace poly {
namespace HInputOp {
static char const input[] = "input";
static char const cst[] = "constant";
}  // namespace HInputOp

class Op {
 public:
  enum class OpType {
    Add,
    TensorAdd,
    AddN,
    Cast,
    Round,
    Trunc,
    Floor,
    Ceil,
    Exp,
    Fabs,
    Log,
    Pow,
    Mul,
    Maximum,
    Minimum,
    FArgMax,
    FArgMin,
    Relu,
    Neg,
    RealDiv,
    Reciprocal,
    Sub,
    Sqrt,
    Rsqrt,
    Tanh,
    Equal,
    Greater,
    Lower,
    Select,
    BitwiseAnd,
    BitwiseOr,
    BitwiseNot,
    BroadcastTo,
    TransData,
    InplaceAssign,
    Reshape,
    FourToFiveNCHW,
    Input,
    Constant,
    MatMul,
    BatchMatMul,
    ReduceSum,
    ReduceMax,
    ReduceMin,
    ReduceProd,
    ReduceDST,
    ReduceSRC,
    ReduceY,
    ReduceX,
    AllReduce,
    VMAdd,
    VMLA,
    IoU,
    IuF,
    NMS,
    LHS,
    RHS,
    Dropout,
    Orig,
    With,
    Assignment,
  };

  enum class OpCategory {
    Input,
    Injective,
    Broadcast,
    Reshape,
    Transpose,
    ReduceX,
    ReduceY,
    AllReduce,
    MatMul,
    Assignment
  };

  Op();
  explicit Op(OpType);
  explicit Op(const std::string &op_name);

  static Op::OpType OpTypeFromString(const std::string &op_type);
  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] std::string BufferName() const;
  static Op::OpType OpTypeFromBufferName(const std::string &op_type);

  [[nodiscard]] bool RemoveUselessInput() const;
  [[nodiscard]] bool IsReduce() const;
  [[nodiscard]] bool IsInput() const;
  [[nodiscard]] bool IsConstant() const;
  [[nodiscard]] bool IsNameless() const;
  [[nodiscard]] bool IsLonely() const;  // whose buffer name doesn't include its inputs

  [[nodiscard]] bool FitBufferName(const std::string &name, bool cstInput) const;

  [[nodiscard]] OpCategory Category() const;
  static Op::OpCategory DominantCategory(Op::OpCategory, Op::OpCategory);

  OpType op_type_;

  static const std::vector<std::tuple<OpType, std::string, std::string>> ops_;

 private:
  enum Priority {  // may be different from OpCategory
    Input = 0,
    Assignment = 1,
    Injective = 2,
    Broadcast = 3,
    Reshape = 4,
    Transpose = 5,
    AllReduce = 6,
    ReduceX = 7,
    ReduceY = 8,
    MatMul = 9
  };
  enum Source { Enum = 0, Info = 1, IR = 2 };

  std::unordered_map<OpCategory, Priority> op_category_priority_map_{
    {OpCategory::Input, Priority::Input},         {OpCategory::Injective, Priority::Injective},
    {OpCategory::Broadcast, Priority::Broadcast}, {OpCategory::Reshape, Priority::Reshape},
    {OpCategory::Transpose, Priority::Transpose}, {OpCategory::AllReduce, Priority::AllReduce},
    {OpCategory::ReduceX, Priority::ReduceX},     {OpCategory::ReduceY, Priority::ReduceY},
    {OpCategory::MatMul, Priority::MatMul},       {OpCategory::Assignment, Priority::Assignment}};

  static int Priority(Op::OpCategory cat);
};

std::string StringOfCategory(Op::OpCategory);

constexpr auto kBatchMatMul = "batchmatmul";
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_OP_H_
