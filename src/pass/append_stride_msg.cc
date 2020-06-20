/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pass/append_stride_msg.h"
#include "ir_pass.h"
namespace akg {
namespace ir {
using Arch::BurstLengthByte;
using Arch::GapUnitByte;
using ktvm::ir::intrinsic::tvm_access_ptr;
constexpr uint tvm_access_ptr_args_no_stride = 5U;
constexpr uint tvm_access_ptr_args_stride = 10U;
constexpr uint dma_call_num_args = 7U;
constexpr uint broadcast_arg_size = 6U;

// Simplify_cce the expr,
// When it expr is a const Whether it's IntImm or UntImm
// Convert it's type to int and return it.
// Other scene will only to Simplify_cce
inline Expr SimplifyAsInt(Expr expr) {
  expr = Simplify_cce(expr);
  const auto int_expr = as_const_int(expr);
  if (int_expr) {
    // Only int32 can be used Gen code without "cast",
    // because of parameter type restrictions
    return make_const(Int(32), *int_expr);
  }
  const auto uint_expr = as_const_uint(expr);
  if (uint_expr) {
    // Only int32 can be used Gen code without "cast",
    // because of parameter type restrictions
    return make_const(Int(32), *uint_expr);
  }
  return expr;
}

Expr AppendStrideMessage::Mutate_(const Call *op, const Expr &expr) {
  CHECK(op);
  if (op->name == tvm_access_ptr) {
    // Insert the stride message in tvm_access_ptr
    CHECK_EQ(op->args.size(), tvm_access_ptr_args_no_stride);
    CHECK_EQ(args_storage_.size(), tvm_access_ptr_args_no_stride)
      << "AppendStrideMessage's args_storage_ uninitialized";

    const auto int_first_value = args_storage_[0].as<IntImm>();
    const auto real_extent = op->args[3] * op->args[0].type().bytes();

    // When int_first_value is -1,
    // it means the intrinsic is not in process list,
    // and insert default value for stride info.
    // Otherwise, insert stride info which is got by intrinsic parameters.
    Array<Expr> args = int_first_value && (int_first_value->value == -1)
                         ? Array<Expr>{op->args[0], op->args[1],
                                       op->args[2], op->args[3],
                                       op->args[4], make_const(Int(32), 1),
                                       real_extent, make_const(Int(32), 1),
                                       real_extent, real_extent}
                         : Array<Expr>{op->args[0],
                                       op->args[1],
                                       op->args[2],
                                       op->args[3],
                                       op->args[4],
                                       SimplifyAsInt(args_storage_[0]),
                                       SimplifyAsInt(args_storage_[1]),
                                       SimplifyAsInt(args_storage_[2]),
                                       SimplifyAsInt(args_storage_[3]),
                                       SimplifyAsInt(args_storage_[4])};
    return Call::make(Handle(), tvm_access_ptr, args, Call::Intrinsic);
  } else {
    // Clear the args_storage_ message, Make sure the args are correct
    args_storage_.clear();
    auto it = call_to_func_.find(op->name);
    if (it != call_to_func_.end()) {
      // We find the op->name in call_to_func map,
      // Extend it's tvm_access_ptr's stride message by map's value function
      return it->second(expr);
    } else {
      // Also add default parameters for commands not covered
      const auto default_value = make_const(Int(32), -1);
      args_storage_ = std::vector<Expr>{default_value, default_value, default_value, default_value, default_value};
      return IRMutator::Mutate_(op, expr);
    }
  }
}

std::pair<int, int> AppendStrideMessage::GetDstSrcTypeLength(const Call *call, bool allow_exceed) {
  CHECK(call);

  CHECK_GE(call->args.size(), 2);
  auto buf_a = call->args[0].as<Call>();
  auto buf_b = call->args[1].as<Call>();

  CHECK(buf_a);
  CHECK(buf_b);
  CHECK(buf_a->name == tvm_access_ptr);
  CHECK(buf_b->name == tvm_access_ptr);

  if (allow_exceed) {
    CHECK_GE(buf_a->args.size(), tvm_access_ptr_args_no_stride);
    CHECK_GE(buf_b->args.size(), tvm_access_ptr_args_no_stride);
  } else {
    CHECK_EQ(buf_a->args.size(), tvm_access_ptr_args_no_stride);
    CHECK_EQ(buf_b->args.size(), tvm_access_ptr_args_no_stride);
  }

  int dst_type_length = buf_a->args[0].type().bytes();
  int src_type_length = buf_b->args[0].type().bytes();
  return std::make_pair(dst_type_length, src_type_length);
}

void AppendStrideMessage::CheckCommandLength(const std::string &cmd_name, int dst_type_length, int src_type_length) {
  auto sit = command_src_length_check_.find(cmd_name);
  if (sit != command_src_length_check_.end()) {
    CHECK_NE(sit->second.count(src_type_length), 0);
  }
  auto dit = command_dst_length_check_.find(cmd_name);
  if (dit != command_dst_length_check_.end()) {
    CHECK_NE(dit->second.count(dst_type_length), 0);
  }
}

Expr AppendStrideMessage::MutateDmaMov(const Expr &expr) {
  // Get the Dma base stride message
  auto dma_call = expr.as<Call>();
  int dst_type_length = 0;
  int src_type_length = 0;
  std::tie(dst_type_length, src_type_length) = GetDstSrcTypeLength(dma_call, true);
  CHECK_GE(dma_call->args.size(), dma_call_num_args);

  std::string command_name = dma_call->name;
  BurstLengthByte src_burst_length_byte, dst_burst_length_byte;
  Arch::GapUnitByte src_gap, dst_gap;

  auto SetBrustLengthByteWithType = [&dst_type_length, &src_type_length, &src_burst_length_byte,
                                     &dst_burst_length_byte](BurstLengthByte blb) {
    src_burst_length_byte = BurstLengthByte(blb * src_type_length / 2);
    dst_burst_length_byte = BurstLengthByte(blb * dst_type_length / 2);
  };
  auto SetBrustLengthByte = [&src_burst_length_byte, &dst_burst_length_byte](BurstLengthByte src, BurstLengthByte dst) {
    src_burst_length_byte = src;
    dst_burst_length_byte = dst;
  };
  auto SetGapWithByte = [&src_type_length, &dst_type_length, &src_gap, &dst_gap](GapUnitByte gub, bool is_src) {
    if (is_src) {
      src_gap = GapUnitByte(gub * src_type_length / 2);
      dst_gap = GapUnitByte::OTHER;
    } else {
      src_gap = GapUnitByte::OTHER;
      dst_gap = GapUnitByte(gub * dst_type_length / 2);
    }
  };
  CheckCommandLength(command_name, dst_type_length, src_type_length);
  if (command_name == "copy_matrix_ubuf_to_cc") {
    SetBrustLengthByteWithType(BurstLengthByte::BURST_LENGTH_512);
    SetGapWithByte(GapUnitByte::L0C16, false);
  } else if (command_name == "copy_vector_ubuf_to_cc") {
    SetBrustLengthByteWithType(BurstLengthByte::BURST_LENGTH_32);
    SetGapWithByte(GapUnitByte::L0C16V, false);
  } else if (command_name == "copy_small_matrix_ubuf_to_cc") {
    if (dst_type_length == 4) {
      SetBrustLengthByteWithType(BurstLengthByte::BURST_LENGTH_128);
      SetGapWithByte(GapUnitByte::L0CSC32, false);
    } else {
      SetBrustLengthByte(BurstLengthByte::BURST_LENGTH_512, BurstLengthByte::BURST_LENGTH_512);
      SetGapWithByte(GapUnitByte::DST_L0CSC16, false);
    }
  } else if (command_name == "copy_matrix_cc_to_ubuf") {
    SetBrustLengthByteWithType(BurstLengthByte::BURST_LENGTH_512);
    SetGapWithByte(GapUnitByte::L0C16, true);
  } else if (command_name == "copy_vector_cc_to_ubuf") {
    SetBrustLengthByteWithType(BurstLengthByte::BURST_LENGTH_32);
    SetGapWithByte(GapUnitByte::L0C16V, true);
  } else if (command_name == "copy_small_matrix_cc_to_ubuf") {
    SetBrustLengthByteWithType(BurstLengthByte::BURST_LENGTH_128);
    SetGapWithByte(GapUnitByte::SRC_L0CSC16, true);
  } else {
    SetBrustLengthByte(BurstLengthByte::BURST_LENGTH_32, BurstLengthByte::BURST_LENGTH_32);
    src_gap = GapUnitByte::OTHER;
    dst_gap = GapUnitByte::OTHER;
  }
  return AppendDmaStrideMsg(std::move(expr), src_burst_length_byte, dst_burst_length_byte, src_gap, dst_gap);
}

Expr AppendStrideMessage::MutateDmaLoad2D(const Expr &expr) {
  auto load2d_call = expr.as<Call>();
  int dst_type_length = 0;
  int src_type_length = 0;
  std::tie(dst_type_length, src_type_length) = GetDstSrcTypeLength(load2d_call, false);
  CheckCommandLength("dma_load_2d", dst_type_length, src_type_length);

  BurstLengthByte src_burst_length_byte = BurstLengthByte::BURST_LENGTH_512;

  const auto block_size = BurstLengthByte::BURST_LENGTH_512;
  CHECK_GT(load2d_call->args.size(), 4);
  const auto repeat = load2d_call->args[3];
  const auto src_stride = load2d_call->args[4] * static_cast<int>(src_burst_length_byte);
  Array<Expr> args;
  // The vector's elements are temp
  args_storage_ = std::vector<Expr>{repeat, src_stride, make_const(Int(32), 1), src_stride, block_size};
  Expr src_buffer = Mutate(load2d_call->args[1]);
  args_storage_.clear();

  args_storage_ = std::vector<Expr>{repeat, src_stride, make_const(Int(32), 1), block_size, block_size};
  Expr dst_buffer = Mutate(load2d_call->args[0]);
  args_storage_.clear();

  args.push_back(dst_buffer);
  args.push_back(src_buffer);
  for (unsigned i = 2; i < load2d_call->args.size(); ++i) {
    args.push_back(load2d_call->args[i]);
  }
  return Call::make(load2d_call->type, load2d_call->name, args, load2d_call->call_type);
}

Expr AppendStrideMessage::MutateDmaLoad3D(const Expr &expr) {
  auto load3d = expr.as<Call>();
  CHECK(load3d);
  CHECK_GE(load3d->args.size(), 2);
  Array<Expr> args;

  auto buf_dst = load3d->args[0].as<Call>();
  auto buf_src = load3d->args[1].as<Call>();

  CHECK(buf_src);
  CHECK(buf_src->name == tvm_access_ptr);
  CHECK_EQ(buf_src->args.size(), tvm_access_ptr_args_no_stride);

  CHECK(buf_dst);
  CHECK(buf_dst->name == tvm_access_ptr);
  CHECK_EQ(buf_dst->args.size(), tvm_access_ptr_args_no_stride);

  int src_type_length = buf_src->args[0].type().bytes();
  int dst_type_length = buf_dst->args[0].type().bytes();

  CheckCommandLength("dma_load_3d", dst_type_length, src_type_length);

  const auto n_dst_burst = buf_dst->args[3];
  const auto n_src_burst = buf_src->args[3];

  Expr dst_burst_length = dst_type_length * n_dst_burst;
  Expr src_burst_length = src_type_length * n_src_burst;

  args_storage_ = std::vector<Expr>{make_const(Int(32), 1), dst_burst_length, make_const(Int(32), 1), dst_burst_length,
                                    dst_burst_length};
  Expr dst_buffer = Mutate(load3d->args[0]);
  args_storage_.clear();

  args_storage_ = std::vector<Expr>{make_const(Int(32), 1), src_burst_length, make_const(Int(32), 1), src_burst_length,
                                    src_burst_length};
  Expr src_buffer = Mutate(load3d->args[1]);
  args_storage_.clear();

  args.push_back(dst_buffer);
  args.push_back(src_buffer);
  for (size_t i = 2; i < load3d->args.size(); ++i) {
    args.push_back(load3d->args[i]);
  }

  return Call::make(load3d->type, load3d->name, args, load3d->call_type);
}

Expr AppendStrideMessage::MutateDmaBroadCast(const Expr &expr) {
  auto broadcast = expr.as<Call>();
  CHECK(broadcast);
  CHECK_EQ(broadcast->args.size(), broadcast_arg_size);
  int dst_type_length = 0;
  int src_type_length = 0;
  std::tie(dst_type_length, src_type_length) = GetDstSrcTypeLength(broadcast, false);

  CheckCommandLength("dma_broad_cast", dst_type_length, src_type_length);

  BurstLengthByte src_burst_length_byte;
  BurstLengthByte dst_burst_length_byte;
  GapUnitByte src_gap_length_byte;
  GapUnitByte dst_gap_length_byte;

  if (src_type_length == 2) {
    src_burst_length_byte = BurstLengthByte::BURST_LENGTH_32;
    src_gap_length_byte = GapUnitByte::OTHER;
  } else {
    src_burst_length_byte = BurstLengthByte::BURST_LENGTH_16;
    src_gap_length_byte = GapUnitByte::OTHER;
  }

  if (dst_type_length == 2) {
    dst_burst_length_byte = BurstLengthByte::BURST_LENGTH_512;
    dst_gap_length_byte = GapUnitByte::L0C16;
  } else {
    dst_burst_length_byte = BurstLengthByte::BURST_LENGTH_256;
    dst_gap_length_byte = GapUnitByte::L0C8;
  }

  const auto n_burst = broadcast->args[2];
  const auto len_burst = broadcast->args[3];
  const auto src_gap = broadcast->args[4];
  const auto dst_gap = broadcast->args[5];
  Expr burst_stride =
    (static_cast<int>(src_burst_length_byte) * len_burst) + (src_gap * static_cast<int>(src_gap_length_byte));
  // no overflow
  Expr repeat_stride = n_burst * burst_stride;
  Array<Expr> args;

  // The vector's elements are temp
  args_storage_ = std::vector<Expr>{make_const(Int(32), 1), repeat_stride, n_burst, burst_stride,
                                    static_cast<int>(dst_burst_length_byte) * len_burst};
  Expr src_buffer = Mutate(broadcast->args[1]);
  args_storage_.clear();

  burst_stride =
    (static_cast<int>(dst_burst_length_byte) * len_burst) + (dst_gap * static_cast<int>(dst_gap_length_byte));
  // no overflow
  repeat_stride = n_burst * (burst_stride);
  args_storage_ = std::vector<Expr>{make_const(Int(32), 1), repeat_stride, n_burst, burst_stride,
                                    static_cast<int>(dst_burst_length_byte) * len_burst};
  Expr dst_buffer = Mutate(broadcast->args[0]);
  args_storage_.clear();

  args.push_back(dst_buffer);
  args.push_back(src_buffer);
  for (unsigned i = 2; i < broadcast->args.size(); ++i) {
    args.push_back(broadcast->args[i]);
  }

  return Call::make(broadcast->type, broadcast->name, args, broadcast->call_type);
}

Expr AppendStrideMessage::AppendDmaStrideMsg(const Expr &expr, BurstLengthByte src_burst_length_byte,
                                             BurstLengthByte dst_burst_length_byte, GapUnitByte src_gap_unit_byte,
                                             GapUnitByte dst_gap_unit_byte) {
  auto dma_call = expr.as<Call>();
  CHECK(dma_call);
  CHECK_GE(dma_call->args.size(), dma_call_num_args);

  // Get the dma parameter
  const auto n_burst = dma_call->args[3];
  const auto len_burst = dma_call->args[4];
  const auto src_stride = dma_call->args[5];
  const auto dst_stride = dma_call->args[6];

  const auto buf_a = dma_call->args[0].as<Call>();
  const auto buf_b = dma_call->args[1].as<Call>();

  CHECK(buf_a);
  CHECK(buf_a->name == tvm_access_ptr);
  CHECK_GE(buf_a->args.size(), tvm_access_ptr_args_no_stride);

  // Computed parameter value for dst
  Expr burst_stride =
    (static_cast<int>(dst_burst_length_byte) * len_burst) + (dst_stride * static_cast<int>(dst_gap_unit_byte));
  Expr repeat_stride = n_burst * burst_stride;
  args_storage_ = std::vector<Expr>{make_const(Int(32), 1), repeat_stride, n_burst, burst_stride,
                                    (static_cast<int>(dst_burst_length_byte) * len_burst)};

  // Reconstruct dst_buffer
  Expr dst_buffer = Mutate(dma_call->args[0]);
  args_storage_.clear();

  CHECK(buf_b);
  CHECK(buf_b->name == tvm_access_ptr);
  CHECK_GE(buf_b->args.size(), tvm_access_ptr_args_no_stride);

  // Computed parameter value for src

  burst_stride =
    (static_cast<int>(src_burst_length_byte) * len_burst) + (src_stride * static_cast<int>(src_gap_unit_byte));
  repeat_stride = n_burst * burst_stride;
  args_storage_ = std::vector<Expr>{make_const(Int(32), 1), repeat_stride, n_burst, burst_stride,
                                    (static_cast<int>(src_burst_length_byte) * len_burst)};

  // Reconstruct src_buffer
  Expr src_buffer = Mutate(dma_call->args[1]);
  args_storage_.clear();

  Array<Expr> args;
  args.push_back(dst_buffer);
  args.push_back(src_buffer);
  for (unsigned i = 2; i < dma_call->args.size(); ++i) {
    args.push_back(dma_call->args[i]);
  }
  // Reconstruct Dma stmt And return it
  return Call::make(dma_call->type, dma_call->name, args, dma_call->call_type);
}

Expr AppendStrideMessage::MutateSingleOpVector(const Expr &expr) {
  constexpr int SINGLE_OP_REPEAT_COUNT_INDEX = 2;
  constexpr int SINGLE_OP_BLOCK_STRIDE_INDEX = 3;
  constexpr int SINGLE_OP_REPEAT_STRIDE_INDEX = 5;

  return MutateVector(expr, SINGLE_OP_REPEAT_COUNT_INDEX, SINGLE_OP_BLOCK_STRIDE_INDEX, SINGLE_OP_REPEAT_STRIDE_INDEX);
}

Expr AppendStrideMessage::MutateSingleVSOpVector(const Expr &expr) {
  constexpr int SINGLE_VS_OP_REPEAT_COUNT_INDEX = 3;
  constexpr int SINGLE_VS_OP_BLOCK_STRIDE_INDEX = 4;
  constexpr int SINGLE_VS_OP_REPEAT_STRIDE_INDEX = 6;

  return MutateVector(expr, SINGLE_VS_OP_REPEAT_COUNT_INDEX, SINGLE_VS_OP_BLOCK_STRIDE_INDEX,
                      SINGLE_VS_OP_REPEAT_STRIDE_INDEX);
}

Expr AppendStrideMessage::MutateMultiOpVector(const Expr &expr) {
  constexpr int MULTI_OP_REPEAT_COUNT_INDEX = 3;
  constexpr int MULTI_OP_BLOCK_STRIDE_INDEX = 4;
  constexpr int MULTI_OP_REPEAT_STRIDE_INDEX = 7;

  return MutateVector(expr, MULTI_OP_REPEAT_COUNT_INDEX, MULTI_OP_BLOCK_STRIDE_INDEX, MULTI_OP_REPEAT_STRIDE_INDEX);
}

Expr AppendStrideMessage::MutateCMPOpVector(const Expr &expr) {
  constexpr int CMP_OP_REPEAT_COUNT_INDEX = 2;
  constexpr int CMP_OP_BLOCK_STRIDE_INDEX = 4;
  constexpr int CMP_OP_REPEAT_STRIDE_INDEX = 6;

  return MutateVector(expr, CMP_OP_REPEAT_COUNT_INDEX, CMP_OP_BLOCK_STRIDE_INDEX, CMP_OP_REPEAT_STRIDE_INDEX);
}

Expr AppendStrideMessage::MutateVector(const Expr &expr, int count_index, int b_stride_index, int r_stride_index) {
  auto vector_call = expr.as<Call>();
  CHECK(vector_call);

  Array<Expr> args_new;
  int args_size = static_cast<int>(vector_call->args.size());
  for (const auto &sub_expr : vector_call->args) {
    Expr arg = sub_expr;
    auto sub_call = sub_expr.as<Call>();
    // put args into sub_call
    if (sub_call && sub_call->name == tvm_access_ptr) {
      CHECK_GT(args_size, count_index);
      CHECK_GT(args_size, r_stride_index);
      CHECK_GT(args_size, b_stride_index);

      // as vector intrinsic pattern, repeat count is fixed,
      // the index of repeat stride and block stride should +1 every time
      args_storage_ = std::vector<Expr>{
        vector_call->args[count_index], SimplifyAsInt(vector_call->args[r_stride_index]) * Arch::Vector::BLOCK_SIZE,
        Arch::Vector::BLOCKS_PER_REPEAT, SimplifyAsInt(vector_call->args[b_stride_index]) * Arch::Vector::BLOCK_SIZE,
        Arch::Vector::BLOCK_SIZE};
      r_stride_index++;
      b_stride_index++;
      // Reconstruct src_buffer
      arg = Mutate(sub_expr);
      args_storage_.clear();
    }
    args_new.push_back(arg);
  }
  // Reconstruct Dma stmt And return it
  return Call::make(vector_call->type, vector_call->name, args_new, vector_call->call_type);
}

Expr AppendStrideMessage::MutateVCMPVOpVector(const Expr &expr) {
  constexpr int VCMPV_OP_REPEAT_COUNT_INDEX = 3;
  constexpr int VCMPV_OP_BLOCK_STRIDE_INDEX = 4;
  constexpr int VCMPV_OP_REPEAT_STRIDE_INDEX = 7;
  constexpr int VCMPV_OP_BLOCK_STRIDE_PRODUCT = 4;
  constexpr int VCMPV_OP_REPEAT_STRIDE_PRODUCT = 32;

  int count_index = VCMPV_OP_REPEAT_COUNT_INDEX;
  int r_stride_index = VCMPV_OP_REPEAT_STRIDE_INDEX;
  int b_stride_index = VCMPV_OP_BLOCK_STRIDE_INDEX;

  auto vector_call = expr.as<Call>();
  CHECK(vector_call);

  Array<Expr> args_new;
  int args_size = static_cast<int>(vector_call->args.size());
  bool is_first_dst = true;
  for (auto &sub_expr : vector_call->args) {
    Expr arg = sub_expr;
    auto sub_call = sub_expr.as<Call>();

    // put args into dst
    if (is_first_dst) {
      is_first_dst = false;
      CHECK(sub_call && sub_call->name == tvm_access_ptr);
      // their mul value is the same. which value id defined above
      CHECK_GT(sub_call->args.size(), 0U);
      int typeByte = sub_call->args[0].type().bytes();
      int block_byte_size = static_cast<int>(VCMPV_OP_BLOCK_STRIDE_PRODUCT) / typeByte;
      int repeat_byte_size = static_cast<int>(VCMPV_OP_REPEAT_STRIDE_PRODUCT) / typeByte;

      CHECK(args_size > count_index);
      // specially, handle here
      args_storage_ = std::vector<Expr>{vector_call->args[count_index], repeat_byte_size,
                                        Arch::Vector::BLOCKS_PER_REPEAT, block_byte_size, Arch::Vector::BLOCK_SIZE};
      // raise the index as normal
      // it can be defined well at first, here to make the same pattern
      r_stride_index++;
      b_stride_index++;
      // Reconstruct src_buffer
      arg = Mutate(sub_expr);
      args_storage_.clear();
      args_new.push_back(arg);
      continue;
    }
    // put args into src
    if (sub_call && sub_call->name == tvm_access_ptr) {
      CHECK_GT(args_size, count_index);
      CHECK_GT(args_size, r_stride_index);
      CHECK_GT(args_size, b_stride_index);
      // as vector intrinsic pattern, repeat count is fixed,
      // the index of repeat stride and block stride should +1 every time
      args_storage_ = std::vector<Expr>{
        vector_call->args[count_index], SimplifyAsInt(vector_call->args[r_stride_index]) * Arch::Vector::BLOCK_SIZE,
        Arch::Vector::BLOCKS_PER_REPEAT, SimplifyAsInt(vector_call->args[b_stride_index]) * Arch::Vector::BLOCK_SIZE,
        Arch::Vector::BLOCK_SIZE};
      r_stride_index++;
      b_stride_index++;
      // Reconstruct src_buffer
      arg = Mutate(sub_expr);
      args_storage_.clear();
    }
    args_new.push_back(arg);
  }
  // Reconstruct Dma stmt And return it
  return Call::make(vector_call->type, vector_call->name, args_new, vector_call->call_type);
}

Expr AppendStrideMessage::MutateReduceVCNmlOpVector(const Expr &expr) { return MutateReduceOpVector(expr, 1); }

Expr AppendStrideMessage::MutateReduceVCArgOpVector(const Expr &expr) { return MutateReduceOpVector(expr, 2); }

Expr AppendStrideMessage::MutateReduceVCGOpVector(const Expr &expr) {
  return MutateReduceOpVector(expr, Arch::Vector::BLOCKS_PER_REPEAT);
}

Expr AppendStrideMessage::MutateReduceOpVector(const Expr &expr, uint8_t coef) {
  uint8_t count_index = 2U;
  uint8_t b_stride_index = 3U;
  uint8_t r_stride_index = 5U;

  auto vector_call = expr.as<Call>();
  CHECK(vector_call);

  Array<Expr> args_new;
  int args_size = static_cast<int>(vector_call->args.size());
  bool is_first_dst = true;
  for (const auto &sub_expr : vector_call->args) {
    Expr arg = sub_expr;
    auto sub_call = sub_expr.as<Call>();

    // put args into dst
    if (is_first_dst) {
      is_first_dst = false;
      CHECK(sub_call && sub_call->name == tvm_access_ptr);
      // their mul value is the same. which value id defined above
      CHECK_GT(sub_call->args.size(), 0U);
      int repeat_byte_size = 0;
      int block_byte_size = 0;
      const auto uRepeatArgValue = vector_call->args[b_stride_index].as<UIntImm>();
      if (uRepeatArgValue) {
        repeat_byte_size =
          static_cast<int>(uRepeatArgValue->value) * Arch::Vector::BLOCKS_PER_REPEAT * Arch::Vector::BLOCK_SIZE / coef;
        block_byte_size = repeat_byte_size / Arch::Vector::BLOCKS_PER_REPEAT;
      } else {
        // compatible for tensorize
        const auto repeatArgValue = vector_call->args[b_stride_index].as<IntImm>();
        CHECK(repeatArgValue);
        repeat_byte_size =
          static_cast<int>(repeatArgValue->value) * Arch::Vector::BLOCKS_PER_REPEAT * Arch::Vector::BLOCK_SIZE / coef;
        block_byte_size = repeat_byte_size / Arch::Vector::BLOCKS_PER_REPEAT;
      }
      CHECK(args_size > count_index);
      // specially, handle here
      args_storage_ = std::vector<Expr>{vector_call->args[count_index], repeat_byte_size,
                                        Arch::Vector::BLOCKS_PER_REPEAT, block_byte_size, Arch::Vector::BLOCK_SIZE};
      // raise the index as normal
      // it can be defined well at first, here to make the same pattern
      b_stride_index++;
      // Reconstruct src_buffer
      arg = Mutate(sub_expr);
      args_storage_.clear();
      args_new.push_back(arg);
      continue;
    } else if (sub_call && sub_call->name == tvm_access_ptr) {
      // put args into src
      Expr repeat_byte_size;
      Expr block_byte_size;
      CHECK_GT(args_size, count_index);
      CHECK_GT(args_size, r_stride_index);
      CHECK_GT(args_size, b_stride_index);
      repeat_byte_size = SimplifyAsInt(vector_call->args[r_stride_index]) * Arch::Vector::BLOCK_SIZE;
      block_byte_size = SimplifyAsInt(vector_call->args[b_stride_index]) * Arch::Vector::BLOCK_SIZE;
      r_stride_index++;
      b_stride_index++;
      // as vector intrinsic pattern, repeat count is fixed,
      // the index of repeat stride and block stride should +1 every time
      args_storage_ = std::vector<Expr>{vector_call->args[count_index], repeat_byte_size,
                                        Arch::Vector::BLOCKS_PER_REPEAT, block_byte_size, Arch::Vector::BLOCK_SIZE};
      // Reconstruct src_buffer
      arg = Mutate(sub_expr);
      args_storage_.clear();
    }
    args_new.push_back(arg);
  }
  // Reconstruct Dma stmt And return it
  return Call::make(vector_call->type, vector_call->name, args_new, vector_call->call_type);
}

Expr RemoveStrideMessage::Mutate_(const Call *op, const Expr &expr) {
  CHECK(op);
  if (op->name == tvm_access_ptr) {
    CHECK_GE(op->args.size(), tvm_access_ptr_args_stride);
    Array<Expr> args{op->args[0], op->args[1], op->args[2], op->args[3], op->args[4]};
    return Call::make(Handle(), tvm_access_ptr, args, Call::Intrinsic);
  }
  return IRMutator::Mutate_(op, expr);
}

Stmt InjectAccessPtrMSG(Stmt stmt) { return AppendStrideMessage().Mutate(std::move(stmt)); }

Stmt RemoveAccessPtrMSG(Stmt stmt) { return RemoveStrideMessage().Mutate(std::move(stmt)); }
}  // namespace ir
}  // namespace akg
