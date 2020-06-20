/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/registry.h>

#include <unordered_set>
#include <map>
#include <numeric>
#include <set>
#include <algorithm>

#include "ir_pass.h"
#include "common/array_api.h"

#include "insn_with_variable.h"
#include "insn_builder.h"
#include "insn_info.h"
#include "insn_pattern.h"
#include "insn_emitter.h"

namespace akg {
namespace ir {
Expr GetVarCoefExpr(const Expr &index, const Var &loop_var) {
  Expr ret = Expr();
  Array<Expr> coefs = ktvm::arith::DetectLinearEquation(index, {loop_var});
  if (coefs.size() == 2) {
    ret = coefs[0];
  }
  return ret;
}

int GetVarCoefInt(const Expr &index, const Var &loop_var) {
  auto coef = GetVarCoefExpr(index, loop_var);
  if (coef.as<IntImm>()) {
    return static_cast<int>(coef.as<IntImm>()->value);
  }
  return -1;
}

void InsertArray(Array<Expr> &array, Array<Expr> append) {
  for (auto iter = append.begin(); iter != append.end(); ++iter) {
    array.push_back(*iter);
  }
}

std::string GetBufferType(Expr address) {
  CHECK(address.as<Variable>());
  std::string buffer_name = address.as<Variable>()->name_hint;
  return GetBufScope(buffer_name);
}

Stmt EmitCceInsn(const Type &type, const Array<Expr> &args, const std::string &intrin_name) {
  auto evaluate = Evaluate::make(Call::make(type, intrin_name, args, Call::Extern));
  return evaluate;
}

void RemoveVectorizedIndex(CCEInfo &t_info, int mode = 0) {
  Array<Var> del_vars;
  if (mode == 0) {
    if (t_info.loops_vars_.size() > 0) {
      del_vars = {t_info.loops_vars_.back()};
    } else {
      del_vars = {};
    }
  } else if (mode == 1) {
    size_t len = t_info.loops_vars_.size();
    CHECK_GE(len, 2);
    del_vars = {t_info.loops_vars_[len - 1], t_info.loops_vars_[len - 2]};
  }
  if (!del_vars.empty()) {
    t_info.dst_index = EliminateVarInExpr(t_info.dst_index, {del_vars});
    for (size_t i = 0; i < t_info.src_index.size(); ++i) {
      t_info.src_index.Set(i, EliminateVarInExpr(t_info.src_index[i], {del_vars}));
    }
  }
}

Array<Expr> GenInsnAddress(CCEInfo t_info, Map<std::string, Buffer> buffer_map, int mode = 0) {
  RemoveVectorizedIndex(t_info, mode);
  Array<Expr> args;
  CHECK(buffer_map.count(t_info.dst->name_hint));
  Buffer dstbuf = buffer_map[t_info.dst->name_hint];
  args.push_back(dstbuf.access_ptr(static_cast<int>(2), Handle(), 1, t_info.dst_index));
  CHECK(t_info.src.size() == t_info.src_index.size());
  for (size_t i = 0; i < t_info.src.size(); ++i) {
    CHECK(buffer_map.count(t_info.src[i]->name_hint));
    Buffer srcbuf = buffer_map[t_info.src[i]->name_hint];
    args.push_back(srcbuf.access_ptr(static_cast<int>(1), Handle(), 1, t_info.src_index[i]));
  }
  return args;
}

Array<Expr> GenInsnAddressWithOffset(CCEInfo t_info, Map<std::string, Buffer> buffer_map, Expr dst_offset,
                                     Array<Expr> src_offset) {
  RemoveVectorizedIndex(t_info, 1);
  Array<Expr> args;
  CHECK(buffer_map.count(t_info.dst->name_hint));
  Buffer dstbuf = buffer_map[t_info.dst->name_hint];
  args.push_back(dstbuf.access_ptr(static_cast<int>(2), Handle(), 1, t_info.dst_index + dst_offset));
  CHECK(t_info.src.size() == t_info.src_index.size());
  for (size_t i = 0; i < t_info.src.size(); ++i) {
    CHECK(buffer_map.count(t_info.src[i]->name_hint));
    Buffer srcbuf = buffer_map[t_info.src[i]->name_hint];
    args.push_back(srcbuf.access_ptr(static_cast<int>(1), Handle(), 1, t_info.src_index[i] + src_offset[i]));
  }
  return args;
}

Stmt SetMask(Type type, bool is_full = true, Expr len = 0, bool is_reduce_max_min = false) {
  const Expr ReduceAndMask = make_const(UInt(64), FullReduceMaskValue);
  Array<Expr> mask_args;
  if (is_full) {
    mask_args = {make_const(UInt(64), -1), make_const(UInt(64), -1)};
    if (is_reduce_max_min) {
      mask_args = {ReduceAndMask, ReduceAndMask};
    }
  } else {
    if (256 / type.bytes() > 64) {
      //  Float16, Int8, u_int8
      Expr cond = GE::make(len, 64);
      Expr lower_mask = if_then_else(cond, make_const(UInt(64), -1), (make_const(UInt(64), 1) << len) - 1);
      Expr high_mask = if_then_else(cond, (make_const(UInt(64), 1) << (len - 64)) - 1, make_const(UInt(64), 0));
      if (is_reduce_max_min) {
        lower_mask = lower_mask & ReduceAndMask;
        high_mask = high_mask & ReduceAndMask;
      }
      mask_args = {high_mask, lower_mask};
    } else {
      // Int32, u_int32, Float32
      Expr mask = (make_const(UInt(64), 1) << len) - 1;
      if (is_reduce_max_min) {
        mask = mask & ReduceAndMask;
      }
      mask_args = {make_const(UInt(64), 0), mask};
    }
  }
  return EmitCceInsn(type, mask_args, "set_vector_mask");
}

std::string GenCastIntrinsic(const Type &src_type, const Type &dst_type, const std::string &cur_intrinsic) {
  auto GetTypeMark = [](const Type &type) -> std::string {
    if (type == Int(8)) {
      return "s8";
    } else if (type == UInt(8)) {
      return "u8";
    } else if (type == Int(16)) {
      return "s16";
    } else if (type == Int(32)) {
      return "s32";
    } else if (type == Float(16)) {
      return "f16";
    } else if (type == Float(32)) {
      return "f32";
    }
    LOG(FATAL) << "Error: cannot cast the unsupported type";
    return "-1";
  };
  std::string cast_type = GetTypeMark(src_type) + "2" + GetTypeMark(dst_type);
  std::string full_intrinsic = "vconv_" + cast_type + cur_intrinsic;
  if (full_intrinsic == "vconv_s322f16") {
    full_intrinsic = "vconv_deq";
  } else if (full_intrinsic == "vconv_f162s32") {
    full_intrinsic = "vconv_f162s32f";
  } else if (full_intrinsic == "vconv_f322s32") {
    full_intrinsic = "vconv_f322s32r";
  }
  return full_intrinsic;
}

class HasScalarVarValue : public IRVisitor {
 public:
  bool Run(const Expr &e) {
    this->Visit(e);
    return flag_;
  }

  void Visit_(const Load *op) final { return; }

  void Visit_(const Variable *op) final {
    flag_ = true;
    return;
  }

  bool flag_{false};
};

class AdjustPragma : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (ktvm::ir::attr::IsPragmaKey(op->attr_key) && op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>()) {
      is_candidate_ = true;
      loop_vars_ = {};
      loop_extends_ = {};
      is_reduce_ = false;
      is_argmax_min_ = false;
      is_broadcast_ = false;
      is_transpose_ = false;
      is_scalar_dma_ = false;
      is_scalar_simd_ = false;
      is_vector_scalar_ = false;
      is_ub_dma_to_vadds_ = false;
      old_pragma_ = op->value.as<StringImm>()->value;
      attr_ptr = op;

      Stmt body = this->Mutate(op->body);
      is_candidate_ = false;
      if (op->value.as<StringImm>()->value == "dma_atomic_add") {
        // add atomic add config
        auto config_atomic_open = Evaluate::make(Call::make(UInt(64), "set_atomic_add_open", {}, Call::Extern));
        auto config_atomic_close = Evaluate::make(Call::make(UInt(64), "set_atomic_add_close", {}, Call::Extern));
        auto new_attr = AttrStmt::make(op->node, op->attr_key, Expr("dma_copy"), body);
        auto new_body = Block::make(Block::make(config_atomic_open, new_attr), config_atomic_close);
        return new_body;
      }

      const std::set<std::string> ReducePragma{"vec_binary_add", "vec_binary_max", "vec_binary_min"};
      if (ReducePragma.count(old_pragma_) && is_reduce_) {
        // add reduce pragma
        return AttrStmt::make(op->node, op->attr_key, reduce_type_, body);
      }
      if (old_pragma_ == "dma_copy" && (is_broadcast_ || is_transpose_ || is_scalar_dma_ || is_ub_dma_to_vadds_)) {
        // change "dma_copy" to ["broadcast", "transpose", scalar_dma, vec_single_adds]
        std::string new_pragma;
        if (is_broadcast_) {
          new_pragma = "broadcast";
        } else if (is_transpose_) {
          new_pragma = "dma_copy_transpose";
        } else if (is_scalar_dma_) {
          new_pragma = "scalar_dma";
        } else if (is_ub_dma_to_vadds_) {
          new_pragma = "vec_single_adds";
        }
        return AttrStmt::make(op->node, op->attr_key, Expr(new_pragma), body);
      }
      if (is_argmax_min_) {
        return AttrStmt::make(op->node, op->attr_key, reduce_type_, body);
      }
      if (is_vector_scalar_) {
        if (old_pragma_ == "vec_binary_add") {
          return AttrStmt::make(op->node, op->attr_key, Expr("vec_single_adds"), body);
        } else if (old_pragma_ == "vec_binary_mul") {
          return AttrStmt::make(op->node, op->attr_key, Expr("vec_single_muls"), body);
        }
      }
      if (is_scalar_simd_) {
        return body;
      }
      return AttrStmt::make(op->node, op->attr_key, op->value, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (is_candidate_) {
      loop_vars_.push_back(op->loop_var);
      loop_extends_.push_back(op->extent);
      Stmt body = this->Mutate(op->body);
      if (is_transpose_ && IsInArray(transpose_vars_, op->loop_var)) {
        return body;
      } else {
        return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
      }
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (op->value.as<Add>() || op->value.as<Max>() || op->value.as<Min>()) {
      is_reduce_ = IsReduce(op);
      if (is_reduce_) {
        return Store::make(op->buffer_var,
                           Call::make(op->value.type(), reduce_type_, {reduce_src_}, Call::CallType::Extern), op->index,
                           op->predicate);
      }
    } else if (op->value.as<Call>() &&
               (op->value.as<Call>()->name == "fargmin" || op->value.as<Call>()->name == "fargmax")) {
      auto call_ptr = op->value.as<Call>();
      Array<Expr> srcs = call_ptr->args;
      CHECK_EQ(srcs.size(), 2);
      is_argmax_min_ = true;
      reduce_type_ = (op->value.as<Call>()->name == "fargmin") ? "arg_min" : "arg_max";
      return Store::make(op->buffer_var, Call::make(call_ptr->type, reduce_type_, {srcs[1]}, Call::CallType::Extern),
                         op->index, op->predicate);
    } else if ((op->value.as<FloatImm>() || op->value.as<IntImm>() || op->value.as<UIntImm>()) &&
               GetVectorizedVarPosition(op->index, loop_vars_) != -1) {
      is_broadcast_ = true;
    } else if (op->value.as<Load>()) {
      transpose_vars_ = {};
      is_broadcast_ = IsBroadcast(op);
      if (!is_broadcast_) {
        is_transpose_ = IsTranspose(op, "Load_2D") || IsTranspose(op, "DMA_UB");
        auto load_ptr = op->value.as<Load>();
        if (is_transpose_) {
          Var trans_var("tt0");
          Expr dst_index = EliminateVarInExpr(op->index, transpose_vars_) + trans_var;
          Expr src_index = EliminateVarInExpr(load_ptr->index, transpose_vars_) + trans_var;
          Expr new_value = Load::make(load_ptr->type, load_ptr->buffer_var, src_index, load_ptr->predicate);
          Stmt new_store = Store::make(op->buffer_var, new_value, dst_index, op->predicate);
          Stmt new_for = For::make(trans_var, 0, 256, ForType::Serial, DeviceAPI::None, new_store);
          return new_for;
        } else {
          int dst_vec = GetVectorizedVarPosition(op->index, loop_vars_);
          int src_vec = GetVectorizedVarPosition(load_ptr->index, loop_vars_);
          if ((dst_vec != src_vec || src_vec == -1 || dst_vec == -1) && GetBufferType(op->buffer_var) == SCOPE_UBUF &&
              GetBufferType(load_ptr->buffer_var) == SCOPE_UBUF) {
            is_scalar_dma_ = true;
            return s;
          }
          if (GetBufferType(op->buffer_var) == SCOPE_UBUF && GetBufferType(load_ptr->buffer_var) == SCOPE_UBUF &&
              dst_vec == src_vec && dst_vec != -1 && loop_vars_.size() >= 2 && op->value.type().is_float()) {
            is_ub_dma_to_vadds_ = true;
            Expr value = Add::make(op->value, FloatImm::make(op->value.type(), 0.0));
            return Store::make(op->buffer_var, value, op->index, 1);
          }
        }
      }
    }

    if (!is_reduce_ && !is_argmax_min_ && !is_broadcast_ && !is_transpose_) {
      if (old_pragma_ == "vec_binary_add" || old_pragma_ == "vec_binary_mul") {
        is_vector_scalar_ = IsVectorScalar(op);
      }
      if (!is_vector_scalar_ && old_pragma_ != "vec_single_adds" && old_pragma_ != "vec_single_muls") {
        is_scalar_simd_ = IsScalar(op);
      }
      if (!is_scalar_simd_ && op->value.as<Select>()) {
        is_scalar_simd_ = IsSelectScalar(op->value);
      }
    }
    if (is_scalar_simd_) {
      return AttrStmt::make(attr_ptr->node, attr_ptr->attr_key, attr_ptr->value, s);
    }
    return s;
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    load_array_.push_back(e);
    return e;
  }

  bool IsReduce(const Store *op) {
    Array<Expr> srcs = GetSrcs(op);
    if (srcs.size() == 2 && srcs[0].as<Load>() && srcs[1].as<Load>()) {
      int src_pos0 = GetVectorizedVarPosition(srcs[0].as<Load>()->index, loop_vars_);
      int src_pos1 = GetVectorizedVarPosition(srcs[1].as<Load>()->index, loop_vars_);
      Expr dst = Load::make(op->value.type(), op->buffer_var, op->index, op->predicate);
      if (Equal(srcs[0], dst) && (src_pos0 != src_pos1) && src_pos1 >= 0 && !HasVars(op->index, loop_vars_[src_pos1])) {
        reduce_src_ = srcs[1];
        return true;
      } else if (Equal(srcs[1], dst) && (src_pos0 != src_pos1) && src_pos0 >= 0 &&
                 !HasVars(op->index, loop_vars_[src_pos0])) {
        reduce_src_ = srcs[0];
        return true;
      }
    }
    return false;
  }

  bool IsBroadcast(const Store *op) {
    if (op->value.as<Load>() && GetBufferType(op->buffer_var) == SCOPE_UBUF &&
        GetBufferType(op->value.as<Load>()->buffer_var) == SCOPE_UBUF) {
      int dst_pos = GetVectorizedVarPosition(op->index, loop_vars_);
      int src_pos = GetVectorizedVarPosition(op->value.as<Load>()->index, loop_vars_);
      if (dst_pos >= 0 && dst_pos != src_pos && !HasVars(op->value.as<Load>()->index, loop_vars_[dst_pos])) {
        return true;
      }
    }
    return false;
  }

  bool IsTranspose(const Store *op, const std::string &trans_type) {
    if (op->value.as<Load>()) {
      bool buffer_type_flag = false;
      if (trans_type == "Load_2D") {
        buffer_type_flag = (GetBufferType(op->buffer_var) == SCOPE_CA || GetBufferType(op->buffer_var) == SCOPE_CB);
      } else if (trans_type == "DMA_UB") {
        buffer_type_flag = op->value.type().bits() == 16 && GetBufferType(op->buffer_var) == SCOPE_UBUF &&
                           GetBufferType(op->value.as<Load>()->buffer_var) == SCOPE_UBUF;
      } else {
        CHECK(0) << "\ntrans_type must be 'L0' or 'UB'";
      }
      if (buffer_type_flag) {
        int dst_pos = GetVectorizedVarPosition(op->index, loop_vars_);
        int src_pos = GetVectorizedVarPosition(op->value.as<Load>()->index, loop_vars_);
        bool is_trans = dst_pos >= 0 && src_pos >= 0 && dst_pos != src_pos &&
                        HasVars(op->value.as<Load>()->index, loop_vars_[dst_pos]) &&
                        HasVars(op->index, loop_vars_[src_pos]) &&
                        GetVarCoefInt(op->index, loop_vars_[src_pos]) == 16 &&
                        GetVarCoefInt(op->value.as<Load>()->index, loop_vars_[dst_pos]) == 16 &&
                        loop_extends_[dst_pos].as<IntImm>() && loop_extends_[dst_pos].as<IntImm>()->value == 16 &&
                        loop_extends_[src_pos].as<IntImm>() && loop_extends_[src_pos].as<IntImm>()->value == 16;
        if (is_trans) {
          transpose_vars_.push_back(loop_vars_[src_pos]);
          transpose_vars_.push_back(loop_vars_[dst_pos]);
          return true;
        }
      }
    }
    return false;
  }

  bool IsVectorScalar(const Store *op) {
    load_array_ = {};
    static_cast<void>(this->Mutate(op->value));
    int dst_pos = GetVectorizedVarPosition(op->index, loop_vars_);
    CHECK_EQ(load_array_.size(), 2);
    CHECK(load_array_[0].as<Load>());
    CHECK(load_array_[1].as<Load>());
    int src_a_pos = GetVectorizedVarPosition(load_array_[0].as<Load>()->index, loop_vars_);
    int src_b_pos = GetVectorizedVarPosition(load_array_[1].as<Load>()->index, loop_vars_);
    return ((dst_pos == src_a_pos && dst_pos != -1 && dst_pos != src_b_pos) ||
            (dst_pos == src_b_pos && dst_pos != -1 && dst_pos != src_a_pos));
  }

  bool IsScalar(const Store *op) {
    load_array_ = {};
    static_cast<void>(this->Mutate(op->value));
    int dst_pos = GetVectorizedVarPosition(op->index, loop_vars_);
    bool flag = (dst_pos == -1);
    for (auto src : load_array_) {
      CHECK(src.as<Load>());
      int src_pos = GetVectorizedVarPosition(src.as<Load>()->index, loop_vars_);
      if (src_pos == -1) {
        flag = true;
        break;
      }
    }
    return flag;
  }

  bool IsSelectScalar(const Expr &e) { return HasScalarVarValue().Run(e); }

  Array<Expr> GetSrcs(const Store *op) {
    Array<Expr> srcs;
    if (op->value.as<Add>()) {
      srcs.push_back(op->value.as<Add>()->a);
      srcs.push_back(op->value.as<Add>()->b);
      reduce_type_ = "reduce_sum";
    } else if (op->value.as<Min>()) {
      srcs.push_back(op->value.as<Min>()->a);
      srcs.push_back(op->value.as<Min>()->b);
      reduce_type_ = "reduce_min";
    } else if (op->value.as<Max>()) {
      srcs.push_back(op->value.as<Max>()->a);
      srcs.push_back(op->value.as<Max>()->b);
      reduce_type_ = "reduce_max";
    }
    return srcs;
  }

  std::string old_pragma_;
  Array<Var> loop_vars_;
  Array<Expr> loop_extends_;
  Array<Expr> load_array_;
  Expr reduce_src_;
  std::string reduce_type_;
  bool is_reduce_{false};
  bool is_argmax_min_{false};
  bool is_broadcast_{false};
  bool is_candidate_{false};
  bool is_transpose_{false};
  bool is_scalar_dma_{false};
  bool is_scalar_simd_{false};
  bool is_vector_scalar_{false};
  bool is_ub_dma_to_vadds_{false};
  const AttrStmt *attr_ptr;
  Array<Var> transpose_vars_;
};

class TransposeTransform : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (ktvm::ir::attr::IsPragmaKey(op->attr_key) && op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value == "dma_copy") {
      pre_transpose_buffer = Var("srcTranspose_local_UB");
      post_transpose_buffer = Var("dstTranspose_local_UB");
      loop_vars_ = {};
      loop_extends_ = {};
      is_candidate_ = true;
      is_block_transpose_ = false;
      auto body = this->Mutate(op->body);
      is_candidate_ = false;
      if (is_block_transpose_) {
        is_block_transpose_ = false;
        auto allocate_pre_buffer = Allocate::make(pre_transpose_buffer, t_type, {TransTotalSize}, const_true(1), body);
        auto attr_pre_buffer =
          AttrStmt::make(pre_transpose_buffer, "storage_scope", Expr("local.UB"), allocate_pre_buffer);
        auto allocate_post_buffer =
          Allocate::make(post_transpose_buffer, t_type, {TransTotalSize}, const_true(1), attr_pre_buffer);
        auto attr_post_buffer =
          AttrStmt::make(post_transpose_buffer, "storage_scope", Expr("local.UB"), allocate_post_buffer);
        return attr_post_buffer;
      } else {
        return AttrStmt::make(op->node, op->attr_key, op->value, body);
      }
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (is_candidate_) {
      loop_vars_.push_back(op->loop_var);
      loop_extends_.push_back(op->extent);
      Stmt body = this->Mutate(op->body);
      if (is_block_transpose_ && IsInArray(trans_vars_, op->loop_var)) {
        return body;
      } else {
        return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (is_candidate_) {
      auto value = op->value;
      if (auto cast = op->value.as<Cast>()) {
        value = cast->value;
      }
      CHECK(value.as<Load>());
      auto src_ptr = value.as<Load>();
      if (GetBufferType(op->buffer_var) == SCOPE_UBUF && GetBufferType(src_ptr->buffer_var) == SCOPE_UBUF) {
        int dst_pos = GetVectorizedVarPosition(op->index, loop_vars_);
        int src_pos = GetVectorizedVarPosition(src_ptr->index, loop_vars_);
        if (dst_pos != -1 && src_pos != -1 && dst_pos != src_pos &&
            floormod(loop_extends_[dst_pos], TransAxisLen).as<IntImm>() &&
            floormod(loop_extends_[dst_pos], TransAxisLen).as<IntImm>()->value == 0 &&
            Equal(GetVarCoefExpr(op->index, loop_vars_[src_pos]), loop_extends_[dst_pos])) {
          if (loop_extends_[dst_pos].as<IntImm>() && loop_extends_[dst_pos].as<IntImm>()->value == TransAxisLen &&
              loop_extends_[src_pos].as<IntImm>() && loop_extends_[src_pos].as<IntImm>()->value == TransAxisLen) {
            return s;
          } else {
            is_block_transpose_ = true;
            t_type = src_ptr->type;
            trans_vars_ = {};
            trans_vars_.push_back(loop_vars_[src_pos]);
            trans_vars_.push_back(loop_vars_[dst_pos]);
            Expr ori_w = GetVarCoefExpr(src_ptr->index, loop_vars_[dst_pos]);
            Expr ori_h = loop_extends_[dst_pos];
            Expr ori_block_w = floordiv(ori_w, TransAxisLen);
            Expr ori_block_h = floordiv(ori_h, TransAxisLen);
            Var loop_w = Var("block_w");
            Var loop_h = Var("block_h");
            Expr src_base_index = EliminateVarInExpr(src_ptr->index, trans_vars_);
            Expr dst_base_index = EliminateVarInExpr(op->index, trans_vars_);

            Var tt0 = Var("tt0");
            Var tt1 = Var("tt1");
            auto pre_copy = Store::make(
              pre_transpose_buffer,
              Load::make(t_type, src_ptr->buffer_var,
                         src_base_index + loop_h * TransAxisLen * ori_w + loop_w * TransAxisLen + tt1 * ori_w + tt0, 1),
              tt1 * TransAxisLen + tt0, 1);
            auto pre_l0 = For::make(tt0, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, pre_copy);
            auto pre_l1 = For::make(tt1, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, pre_l0);
            auto pre_attr = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_copy"), pre_l1);

            auto transpose =
              Store::make(post_transpose_buffer, Load::make(t_type, pre_transpose_buffer, tt1 * TransAxisLen + tt0, 1),
                          tt0 * 16 + tt1, 1);
            auto trans_l0 = For::make(tt0, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, transpose);
            auto trans_l1 = For::make(tt1, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, trans_l0);
            auto trans_attr = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_copy"), trans_l1);

            auto post_copy = Store::make(
              op->buffer_var, Load::make(t_type, post_transpose_buffer, tt1 * TransAxisLen + tt0, 1),
              dst_base_index + loop_w * TransAxisLen * ori_h + loop_h * TransAxisLen + tt1 * ori_h + tt0, 1);
            auto post_l0 = For::make(tt0, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, post_copy);
            auto post_l1 = For::make(tt1, 0, TransAxisLen, ForType::Serial, DeviceAPI::None, post_l0);
            auto post_attr = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_copy"), post_l1);

            auto full_inner = Block::make(Block::make(pre_attr, trans_attr), post_attr);
            auto inner_w = For::make(loop_w, 0, ori_block_w, ForType::Serial, DeviceAPI::None, full_inner);
            auto inner_h = For::make(loop_h, 0, ori_block_h, ForType::Serial, DeviceAPI::None, inner_w);
            return inner_h;
          }
        }
      }
    }
    return s;
  }

  bool is_candidate_{false};
  bool is_block_transpose_{false};
  Array<Var> trans_vars_;
  Array<Var> loop_vars_;
  Array<Expr> loop_extends_;
  Type t_type;
  Var pre_transpose_buffer;
  Var post_transpose_buffer;
};

class IfReorder : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (ktvm::ir::attr::IsPragmaKey(op->attr_key) && op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value != "mad") {
      in_insn_ = true;
      for_vars_.clear();
      if_vars_.clear();
      for_vec_.clear();
      if_vec_.clear();
      auto body = this->Mutate(op->body);
      in_insn_ = false;
      if (!if_vec_.empty()) {
        Stmt new_s = AttrStmt::make(op->node, op->attr_key, op->value, body);
        for (auto if_op : if_vec_) {
          new_s = IfThenElse::make(if_op->condition, new_s);
        }

        for (auto for_op = for_vec_.rbegin(); for_op != for_vec_.rend(); ++for_op) {
          bool find_flag = false;
          for (auto for_iter = for_vars_.begin(); for_iter != for_vars_.end(); ++for_iter) {
            if (Equal((*for_iter), (*for_op)->loop_var)) {
              find_flag = true;
              break;
            }
          }
          if (find_flag) {
            new_s = For::make((*for_op)->loop_var, (*for_op)->min, (*for_op)->extent, ForType::Serial, DeviceAPI::None,
                              new_s);
          }
        }
        return new_s;
      } else {
        return s;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (in_insn_) {
      for_vec_.push_back(op);
      for_vars_.push_back(op->loop_var);
      Stmt body = this->Mutate(op->body);
      std::vector<Var>::iterator for_iter;
      for (for_iter = for_vars_.begin(); for_iter != for_vars_.end(); ++for_iter) {
        if (Equal((*for_iter), op->loop_var)) {
          break;
        }
      }

      if (!if_vec_.empty()) {
        std::vector<Var>::iterator if_iter;
        bool find_flag = false;
        for (if_iter = if_vars_.begin(); if_iter != if_vars_.end(); ++if_iter) {
          if (Equal((*if_iter), op->loop_var)) {
            find_flag = true;
            break;
          }
        }
        if (find_flag) {
          return body;
        } else {
          for_vars_.erase(for_iter);
          return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
        }
      } else {
        for_vars_.erase(for_iter);
        return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (in_insn_) {
      if_vec_.push_back(op);
      for (auto loop_var : for_vars_) {
        if (HasVars(op->condition, loop_var)) {
          if_vars_.push_back(loop_var);
        }
      }
      Stmt body = this->Mutate(op->then_case);
      return body;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (in_insn_) {
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

  bool in_insn_{false};
  std::vector<const IfThenElse *> if_vec_;
  std::vector<Var> if_vars_;
  std::vector<Var> for_vars_;
  std::vector<const For *> for_vec_;
  std::vector<const For *> before_if_;
};

class LoopReorder : public IRMutator {
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (ktvm::ir::attr::IsPragmaKey(op->attr_key) && op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>()) {
      in_insn_ = true;
      pragma = op->value.as<StringImm>()->value;
      for_map_.clear();
      ori_vars_ = {};
      var_order_.clear();
      auto ret = this->Mutate(op->body);
      in_insn_ = false;
      if (!has_changed_) {
        return s;
      } else {
        if (var_order_.empty()) {
          ret = AttrStmt::make(op->node, op->attr_key, op->value, ret);
          for (size_t i = 0; i < ori_vars_.size(); ++i) {
            CHECK_GT(for_map_.count(ori_vars_[i].get()), 0);
            auto ptr = for_map_[ori_vars_[i].get()];
            ret = For::make(ptr->loop_var, ptr->min, ptr->extent, ptr->for_type, ptr->device_api, ret);
          }
        } else {
          for (size_t i = 0; i < var_order_.size(); ++i) {
            CHECK_GT(for_map_.count(var_order_[i].get()), 0);
            auto ptr = for_map_[var_order_[i].get()];
            ret = For::make(ptr->loop_var, ptr->min, ptr->extent, ptr->for_type, ptr->device_api, ret);
          }
          ret = AttrStmt::make(op->node, op->attr_key, op->value, ret);
        }
        return ret;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (in_insn_) {
      for_map_[(op->loop_var).get()] = op;
      ori_vars_.push_back(op->loop_var);
      auto body = this->Mutate(op->body);
      return body;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    int dst_pos = GetVectorizedVarPosition(op->index, ori_vars_);
    int len = static_cast<int>(ori_vars_.size());

    std::vector<const Load *> srcs;
    auto get_loads = [&srcs](const NodeRef &node) {
      if (const auto v = node.as<Load>()) {
        srcs.push_back(v);
      }
    };
    PostOrderVisit(op->value, get_loads);

    bool same_pos = true;
    std::vector<int> srcs_pos;
    for (int i = 0; i < static_cast<int>(srcs.size()); ++i) {
      int temp_pos = GetVectorizedVarPosition(srcs[i]->index, ori_vars_);
      srcs_pos.push_back(temp_pos);
      if (temp_pos != dst_pos) {
        same_pos = false;
      }
    }

    has_changed_ = false;
    if (dst_pos >= 0 && len >= 2 && dst_pos != (len - 1) && (same_pos || pragma == "broadcast")) {
      // Src Load empty; all Load and Dst has the same key axis; broadcast
      has_changed_ = true;
      var_order_.push_back(ori_vars_[dst_pos]);
      for (int i = len - 1; i >= 0; i--) {
        if (i != dst_pos) {
          var_order_.push_back(ori_vars_[i]);
        }
      }
    } else if (pragma.find("reduce") != pragma.npos && len >= 2 && srcs_pos[0] != (len - 1)) {
      // based on dst key axis: reduce
      has_changed_ = true;
      var_order_.push_back(ori_vars_[srcs_pos[0]]);
      for (int i = len - 1; i >= 0; i--) {
        if (i != srcs_pos[0]) {
          var_order_.push_back(ori_vars_[i]);
        }
      }
    }

    return s;
  }

  std::unordered_map<const Variable *, const For *> for_map_;
  std::vector<Var> var_order_;
  Array<Var> ori_vars_;
  bool has_changed_{false};
  bool in_insn_{false};
  std::string pragma;
};

class ForVarUnique : public IRMutator {
 public:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto body = this->Mutate(op->body);
    if (var_maps_.count(op->loop_var.get())) {
      Var new_var = Var("ii" + std::to_string(++index_));
      std::unordered_map<const Variable *, Expr> value_map;
      value_map[op->loop_var.get()] = new_var;
      auto new_body = Substitute(body, value_map);
      var_maps_[new_var.get()] = 1;
      return For::make(new_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, new_body);
    } else {
      var_maps_[op->loop_var.get()] = 1;
      return For::make(op->loop_var, op->min, op->extent, ForType::Serial, DeviceAPI::None, body);
    }
  }

  std::unordered_map<const Variable *, int> var_maps_;
  int index_{0};
};

class GenSIMD {
 public:
  GenSIMD(CCEInfo &t_info, Map<std::string, Buffer> &buffer_map, const std::string &pragma)
      : t_info_(t_info), buffer_map_(buffer_map), pragma_(pragma) {}

  virtual ~GenSIMD() = default;

  virtual Stmt Run() {
    if (t_info_.loops_extent_.empty()) {
      vec_len_ = 1;
    } else {
      vec_len_ = t_info_.loops_extent_.back();
    }

    Adjust_info();
    Stmt ret;
    size_t len = t_info_.loops_vars_.size();
    if (len <= 1) {
      Stmt head = GenHead();
      Stmt tail = GenTail();
      ret = Block::make(head, tail);
      t_info_.loops_vars_.clear();
      t_info_.loops_extent_.clear();
    } else if (!HasVars(t_info_.dst_index, {t_info_.loops_vars_[len - 2]}) ||
               HasVars(t_info_.imm, {t_info_.loops_vars_[len - 2]})) {
      Stmt head = GenHead();
      Stmt tail = GenTail();
      ret = Block::make(head, tail);
      t_info_.loops_vars_.pop_back();
      t_info_.loops_extent_.pop_back();
    } else {
      Stmt head_one_loop = GenHead();
      Stmt tail_one_loop = GenTail();
      Stmt body_one_loop = Block::make(head_one_loop, tail_one_loop);
      Stmt branch1 = For::make(t_info_.loops_vars_[len - 2], 0, t_info_.loops_extent_[len - 2], ForType::Serial,
                               DeviceAPI::None, body_one_loop);
      Expr cond_one_loop = GT::make(vec_len_, 256 / t_info_.type.bytes());
      Stmt if_one_loop = IfThenElse::make(cond_one_loop, branch1, Stmt());

      Stmt branch2 = GenRepeatBody();
      Expr cond_two_loop = LE::make(vec_len_, 256 / t_info_.type.bytes());
      Stmt if_two_loop = IfThenElse::make(cond_two_loop, branch2, Stmt());

      ret = Block::make(if_one_loop, if_two_loop);
      t_info_.loops_vars_.pop_back();
      t_info_.loops_extent_.pop_back();
      t_info_.loops_vars_.pop_back();
      t_info_.loops_extent_.pop_back();
    }

    if (insn_intrinsic_ == "vconv_deq") {
      ret = InsertBody(Evaluate::make(Call::make(Float(16), "set_deqscale", {Expr(1)}, Call::Extern)), ret);
    }
    return ret;
  }

  void Adjust_info() {
    CHECK_GE(SIMDInsnMap.count(pragma_), 0);
    insn_intrinsic_ = SIMDInsnMap.find(pragma_)->second.first;
    insn_type_ = SIMDInsnMap.find(pragma_)->second.second;

    if (pragma_ == "broadcast" && !t_info_.src.empty()) {
      t_info_.imm = Load::make(t_info_.type, t_info_.src[0], t_info_.src_index[0], 1);
      t_info_.src = {};
      t_info_.src_index = {};
    }

    if (insn_type_ == "vector_scalar" && t_info_.src.size() == 2) {
      t_info_.imm = Load::make(t_info_.type, t_info_.src[1], t_info_.src_index[1], 1);
      t_info_.src.assign(t_info_.src.begin(), t_info_.src.begin() + 1);
      t_info_.src_index.assign(t_info_.src_index.begin(), t_info_.src_index.begin() + 1);
    }

    if (insn_type_ == "cast" && t_info_.src.size() == 1) {
      CHECK(buffer_map_.count(t_info_.dst->name_hint));
      CHECK(!t_info_.src.empty());
      CHECK(buffer_map_.count(t_info_.src[0]->name_hint));
      const Type dst_type = buffer_map_[t_info_.dst->name_hint]->dtype;
      const Type src_type = buffer_map_[t_info_.src[0]->name_hint]->dtype;
      insn_intrinsic_ = GenCastIntrinsic(src_type, dst_type, insn_intrinsic_);
      if (dst_type.bytes() >= src_type.bytes()) {
        t_info_.type = dst_type;
        dstoffset_factor_ = 1;
        srcoffset_factor_ = dst_type.bytes() / src_type.bytes();
      } else {
        t_info_.type = src_type;
        dstoffset_factor_ = src_type.bytes() / dst_type.bytes();
        srcoffset_factor_ = 1;
      }
    }
  }

  Array<Expr> GenM0M1(Expr repeat, int mode = 0) {
    Array<Expr> ReM0M1 = {repeat};
    Array<Expr> M0M1;
    Expr dst_m1;
    Array<Expr> src_m1;
    if (mode == 0) {
      dst_m1 = 8;
      src_m1.push_back(8);
      src_m1.push_back(8);
    } else if (mode == 1) {
      int block_unit = 32 / t_info_.type.bytes();
      size_t len = t_info_.loops_vars_.size();
      Array<Expr> strides_dst = ktvm::arith::DetectLinearEquation(t_info_.dst_index, {t_info_.loops_vars_[len - 2]});
      dst_m1 = Simplify(FloorDiv::make(strides_dst[0], block_unit));
      for (auto src_index : t_info_.src_index) {
        Array<Expr> strides_src = ktvm::arith::DetectLinearEquation(src_index, {t_info_.loops_vars_[len - 2]});
        src_m1.push_back(Simplify(FloorDiv::make(strides_src[0], block_unit)));
      }
    }

    if (insn_type_ == "binary" && src_m1.size() >= 2) {
      M0M1 = {1, 1, 1, dst_m1, src_m1[0], src_m1[1]};
    } else if ((insn_type_ == "single" || insn_type_ == "vector_scalar") && !src_m1.empty()) {
      M0M1 = {1, 1, dst_m1, src_m1[0]};
    } else if (insn_type_ == "vector_dup") {
      M0M1 = {1, 1, dst_m1, 0};
    } else if (insn_type_ == "cast") {
      M0M1 = {1, 1, Simplify(FloorDiv::make(dst_m1, dstoffset_factor_)),
              Simplify(FloorDiv::make(src_m1[0], srcoffset_factor_))};
    }

    InsertArray(ReM0M1, M0M1);
    if (insn_type_ == "binary" || insn_type_ == "single" || insn_type_ == "cast") {
      return ReM0M1;
    } else if (insn_type_ == "vector_scalar" || insn_type_ == "vector_dup") {
      Array<Expr> ImmArgs = {t_info_.imm};
      InsertArray(ImmArgs, ReM0M1);
      return ImmArgs;
    }
    return {};
  }

  virtual Stmt EmitInsn(Expr offset, Expr repeat) {
    CHECK(repeat.defined());
    CCEInfo tmp_info = t_info_;
    tmp_info.dst_index = tmp_info.dst_index + offset;
    for (size_t i = 0; i < tmp_info.src_index.size(); ++i) {
      tmp_info.src_index.Set(i, tmp_info.src_index[i] + offset);
    }
    Array<Expr> args = GenInsnAddress(tmp_info, buffer_map_);
    InsertArray(args, GenM0M1(repeat));
    auto insn = EmitCceInsn(tmp_info.type, args, insn_intrinsic_);
    return insn;
  }

  Stmt GenHead() {
    Expr head_len = FloorDiv::make(vec_len_, 256 / t_info_.type.bytes());
    Expr head_cond = NE::make(head_len, 0);
    Stmt head_body = GenHeadBody();
    Stmt head_tail = GenHeadTail();
    Stmt head_stmt = Block::make(head_body, head_tail);
    auto set_mask_stmt = SetMask(t_info_.type, true);
    head_stmt = Block::make(set_mask_stmt, head_stmt);
    Stmt head = IfThenElse::make(head_cond, head_stmt, Stmt());
    return head;
  }

  Stmt GenHeadBody() {
    Expr head_len = FloorDiv::make(vec_len_, 256 / t_info_.type.bytes());
    Expr head_body_len = FloorDiv::make(head_len, 255);

    Var bodyloop_var = Var("ll0");
    Expr head_body_offset = bodyloop_var * (256 / t_info_.type.bytes()) * 255;

    auto core_insn = EmitInsn(head_body_offset, 255);
    auto body_loop_stmt = For::make(bodyloop_var, 0, head_body_len, ForType::Serial, DeviceAPI::None, core_insn);
    return body_loop_stmt;
  }

  Stmt GenHeadTail() {
    Expr head_len = FloorDiv::make(vec_len_, 256 / t_info_.type.bytes());
    Expr head_tail_len = FloorMod::make(head_len, 255);
    Expr head_tail_cond = NE::make(head_tail_len, 0);
    Expr head_tail_offset = FloorDiv::make(head_len, 255) * 255 * (256 / t_info_.type.bytes());

    auto tail_insn = EmitInsn(head_tail_offset, head_tail_len);

    auto if_head_tail = IfThenElse::make(head_tail_cond, tail_insn, Stmt());
    return if_head_tail;
  }

  Stmt GenTail() {
    Expr head_len = FloorDiv::make(vec_len_, 256 / t_info_.type.bytes());
    Expr tail_len = FloorMod::make(vec_len_, 256 / t_info_.type.bytes());
    Expr tail_cond = NE::make(tail_len, 0);
    Expr tail_offset = head_len * (256 / t_info_.type.bytes());
    auto tail_insn = EmitInsn(tail_offset, 1);
    auto set_mask_stmt = SetMask(t_info_.type, false, tail_len);
    auto tail_stmt = Block::make(set_mask_stmt, tail_insn);
    auto if_tail = IfThenElse::make(tail_cond, tail_stmt, Stmt());
    return if_tail;
  }

  Stmt GenRepeatBody() {
    size_t len = t_info_.loops_vars_.size();
    auto set_mask_stmt = SetMask(t_info_.type, false, vec_len_);
    Expr repeat_times = t_info_.loops_extent_[len - 2];

    // repeat less equal 255
    Expr simple_repeat_cond = LE::make(repeat_times, 255);
    Array<Expr> simple_args = GenInsnAddress(t_info_, buffer_map_, 1);
    InsertArray(simple_args, GenM0M1(repeat_times, 1));
    auto simple_insn = EmitCceInsn(t_info_.type, simple_args, insn_intrinsic_);
    auto simple_stmt = IfThenElse::make(simple_repeat_cond, simple_insn, Stmt());

    // greater than 255
    const int max_repeat_cnt = 255;
    Expr complex_repeat_cond = GT::make(repeat_times, max_repeat_cnt);
    Expr full_repeat_times = FloorDiv::make(repeat_times, max_repeat_cnt);
    Expr tail_repeat_times = FloorMod::make(repeat_times, max_repeat_cnt);
    Var innerloop_var = Var("kk");

    Array<Expr> strides_dst = ktvm::arith::DetectLinearEquation(t_info_.dst_index, {t_info_.loops_vars_[len - 2]});
    Expr dst_m1_loop = Simplify(255 * strides_dst[0] * innerloop_var);
    Expr dst_m1_tail = Simplify(255 * strides_dst[0] * full_repeat_times);
    Array<Expr> src_m1_loop;
    Array<Expr> src_m1_tail;
    for (auto src_index : t_info_.src_index) {
      Array<Expr> strides_src = ktvm::arith::DetectLinearEquation(src_index, {t_info_.loops_vars_[len - 2]});
      src_m1_loop.push_back(Simplify(255 * strides_src[0] * innerloop_var));
      src_m1_tail.push_back(Simplify(255 * strides_src[0] * full_repeat_times));
    }

    // For Loop
    Array<Expr> complex_args_loop = GenInsnAddressWithOffset(t_info_, buffer_map_, dst_m1_loop, src_m1_loop);
    InsertArray(complex_args_loop, GenM0M1(255, 1));
    auto loop_insn = EmitCceInsn(t_info_.type, complex_args_loop, insn_intrinsic_);
    auto loop_stmt = For::make(innerloop_var, 0, full_repeat_times, ForType::Serial, DeviceAPI::None, loop_insn);

    // Tail
    Expr tail_repeat_cond = GT::make(tail_repeat_times, 0);
    Array<Expr> complex_args_tail = GenInsnAddressWithOffset(t_info_, buffer_map_, dst_m1_tail, src_m1_tail);
    InsertArray(complex_args_tail, GenM0M1(tail_repeat_times, 1));
    auto tail_insn = EmitCceInsn(t_info_.type, complex_args_tail, insn_intrinsic_);
    auto tail_stmt = IfThenElse::make(tail_repeat_cond, tail_insn, Stmt());

    auto loop_tail = Block::make(loop_stmt, tail_stmt);
    auto complex_stmt = IfThenElse::make(complex_repeat_cond, loop_tail, Stmt());

    auto full_stmt = Block::make(set_mask_stmt, Block::make(simple_stmt, complex_stmt));
    return full_stmt;
  }

  CCEInfo &t_info_;
  Expr vec_len_;
  Map<std::string, Buffer> buffer_map_;
  std::string pragma_;
  std::string insn_intrinsic_;
  std::string insn_type_;
  int dstoffset_factor_{1};
  int srcoffset_factor_{1};
};

class GenSelect : public GenSIMD {
 public:
  GenSelect(CCEInfo &t_info, Map<std::string, Buffer> &buffer_map, const std::string &pragma, const Store *op)
      : GenSIMD(t_info, buffer_map, pragma), store_ptr_(op) {
    if (!t_info.loops_vars_.empty()) {
      vec_axis = t_info.loops_vars_.back();
    } else {
      vec_axis = Var();
    }
  }
  ~GenSelect() override = default;

  Stmt Run() final {
    if (t_info_.loops_extent_.empty()) {
      vec_len_ = 1;
    } else {
      vec_len_ = t_info_.loops_extent_.back();
    }
    GenSrcInfo();
    auto ret = GenCmpSelect();
    for (size_t i = 0; i < dump_buffer_.size(); ++i) {
      ret = Block::make(dump_buffer_[i], ret);
    }
    ret = Block::make(SetMask(Float(16)), ret);
    for (size_t i = 0; i < make_buffer_.size(); ++i) {
      auto temp = make_buffer_[i];
      ret = Allocate::make(temp.first, temp.second, {256 / temp.second.bytes()}, const_true(), ret);
      ret = AttrStmt::make(producer_tensor[i], "storage_scope", Expr("local.UB"), ret);
    }
    return ret;
  }

  void GenTensors(const Expr &src, const Type &data_type) {
    CHECK((src.as<Load>() || src.as<FloatImm>()));
    if (src.as<Load>() && HasVars(src.as<Load>()->index, vec_axis)) {
      auto ptr = src.as<Load>();
      src_info_.tensor_var.push_back(ptr->buffer_var);
      src_info_.tensor_index.push_back(ptr->index);
      src_info_.offset_factor.push_back(1);
    } else {
      std::string name = "select_buffer_" + std::to_string(buffer_nunber_) + "_local_UB";
      Var buffer_var = Var(name);
      int number = 256 / data_type.bytes();
      Buffer var_buf =
        BufferNode::make(buffer_var, data_type, {number}, {}, 0, name, GetBufScope(name), 1, 1, BufferType::kDefault);
      ++buffer_nunber_;
      buffer_map_.Set(name, var_buf);
      src_info_.tensor_var.push_back(buffer_var);
      src_info_.tensor_index.push_back(0);
      src_info_.offset_factor.push_back(0);
      Array<Expr> args = {var_buf.access_ptr(static_cast<int>(2), Handle(), 1, 0), src, 1, 1, 1, 0, 0};
      dump_buffer_.push_back(Evaluate::make(Call::make(data_type, "vector_dup", args, Call::Extern)));
      producer_tensor.push_back(buffer_var);
      make_buffer_.push_back({buffer_var, data_type});
    }
  }

  void GetConditionInfo(const Select *ptr) {
    Expr cond_a, cond_b;
    Expr condition = ptr->condition;
    if (condition.as<EQ>()) {
      cond_a = condition.as<EQ>()->a;
      cond_b = condition.as<EQ>()->b;
      insn_intrinsic_ = "vcmp_eq";
    } else if (condition.as<NE>()) {
      cond_a = condition.as<NE>()->a;
      cond_b = condition.as<NE>()->b;
      insn_intrinsic_ = "vcmp_ne";
    } else if (condition.as<LT>()) {
      cond_a = condition.as<LT>()->a;
      cond_b = condition.as<LT>()->b;
      insn_intrinsic_ = "vcmp_lt";
    } else if (condition.as<LE>()) {
      cond_a = condition.as<LE>()->a;
      cond_b = condition.as<LE>()->b;
      insn_intrinsic_ = "vcmp_le";
    } else if (condition.as<GT>()) {
      cond_a = condition.as<GT>()->a;
      cond_b = condition.as<GT>()->b;
      insn_intrinsic_ = "vcmp_gt";
    } else if (condition.as<GE>()) {
      cond_a = condition.as<GE>()->a;
      cond_b = condition.as<GE>()->b;
      insn_intrinsic_ = "vcmp_ge";
    }

    Type data_type;
    if (cond_a.as<Load>()) {
      data_type = cond_a.as<Load>()->type;
    } else if (cond_b.as<Load>()) {
      data_type = cond_b.as<Load>()->type;
    }
    GenTensors(cond_a, data_type);
    GenTensors(cond_b, data_type);
    src_info_.data_type.push_back(data_type);
  }

  void GetValueInfo(const Select *ptr) {
    Expr true_value_ = ptr->true_value;
    Expr false_value_ = ptr->false_value;
    Type data_type = ptr->type;
    GenTensors(true_value_, data_type);
    GenTensors(false_value_, data_type);
    src_info_.data_type.push_back(data_type);
  }

  void GenSrcInfo() {
    CHECK(store_ptr_);
    auto sel_ptr = store_ptr_->value.as<Select>();
    CHECK(sel_ptr);
    GetConditionInfo(sel_ptr);
    GetValueInfo(sel_ptr);
    t_info_.dst_index = EliminateVarInExpr(t_info_.dst_index, {vec_axis});
  }

  Stmt GenCmpSelect() {
    Expr head_len = FloorDiv::make(vec_len_, 256 / t_info_.type.bytes());
    Expr tail_len = FloorMod::make(vec_len_, 256 / t_info_.type.bytes());
    Expr head_cond = NE::make(head_len, 0);
    Expr tail_cond = NE::make(tail_len, 0);

    // Head - For loops
    Var loop_var = Var("ll0");
    Expr head_cmp_offset = loop_var * (256 / src_info_.data_type[0].bytes());
    Expr head_sel_offset = loop_var * (256 / src_info_.data_type[1].bytes());
    auto head_mask = SetMask(Float(16));
    Array<Expr> args = GenCMPArgs(head_cmp_offset);
    Stmt cmp_insn = EmitCceInsn(src_info_.data_type[0], args, insn_intrinsic_);
    args = GenSELArgs(head_sel_offset);
    Stmt sel_insn = EmitCceInsn(src_info_.data_type[1], args, "vsel");
    auto core_stmt = Block::make(Block::make(head_mask, cmp_insn), sel_insn);
    auto head_stmt = For::make(loop_var, 0, head_len, ForType::Serial, DeviceAPI::None, core_stmt);
    auto if_head = IfThenElse::make(head_cond, head_stmt, Stmt());

    // Tail
    Expr tail_cmp_offset = head_len * (256 / src_info_.data_type[0].bytes());
    Expr tail_sel_offset = head_len * (256 / src_info_.data_type[1].bytes());
    auto set_cmp_mask_stmt = SetMask(src_info_.data_type[0]);
    args = GenCMPArgs(tail_cmp_offset);
    cmp_insn = EmitCceInsn(src_info_.data_type[0], args, insn_intrinsic_);

    auto set_sel_mask_stmt = SetMask(src_info_.data_type[1]);
    args = GenSELArgs(tail_sel_offset);
    sel_insn = EmitCceInsn(src_info_.data_type[1], args, "vsel");
    auto tail_stmt = Block::make(Block::make(set_cmp_mask_stmt, cmp_insn), Block::make(set_sel_mask_stmt, sel_insn));
    auto if_tail = IfThenElse::make(tail_cond, tail_stmt, Stmt());

    return Block::make(if_head, if_tail);
  }

  void RemoveVectorizedAxis(SelectInfo &info) {
    if (vec_axis.defined()) {
      Var vec_var = vec_axis;
      for (size_t i = 0; i < info.tensor_index.size(); ++i) {
        info.tensor_index.Set(i, EliminateVarInExpr(info.tensor_index[i], {vec_var}));
      }
    }
  }

  Array<Expr> GenCMPArgs(Expr offset) {
    SelectInfo tmp_info = src_info_;
    for (size_t i = 0; i < 2; ++i) {
      tmp_info.tensor_index.Set(i, tmp_info.tensor_index[i] + offset * tmp_info.offset_factor[i]);
    }
    RemoveVectorizedAxis(tmp_info);
    Array<Expr> args;
    for (size_t i = 0; i < 2; ++i) {
      CHECK(buffer_map_.count(tmp_info.tensor_var[i]->name_hint));
      Buffer srcbuf = buffer_map_[tmp_info.tensor_var[i]->name_hint];
      args.push_back(srcbuf.access_ptr(static_cast<int>(1), Handle(), 1, tmp_info.tensor_index[i]));
    }
    InsertArray(args, {1, 1, 1, 1, 0, 0, 0});
    return args;
  }

  Array<Expr> GenSELArgs(Expr offset) {
    SelectInfo tmp_info = src_info_;
    RemoveVectorizedAxis(tmp_info);
    for (size_t i = 2; i < tmp_info.tensor_index.size(); ++i) {
      tmp_info.tensor_index.Set(i, tmp_info.tensor_index[i] + offset * tmp_info.offset_factor[i]);
    }
    Array<Expr> args;
    CHECK(buffer_map_.count(t_info_.dst->name_hint));
    Buffer dstbuf = buffer_map_[t_info_.dst->name_hint];
    args.push_back(dstbuf.access_ptr(static_cast<int>(2), Handle(), 1, t_info_.dst_index + offset));
    for (size_t i = 2; i < tmp_info.tensor_index.size(); ++i) {
      CHECK(buffer_map_.count(tmp_info.tensor_var[i]->name_hint));
      Buffer srcbuf = buffer_map_[tmp_info.tensor_var[i]->name_hint];
      args.push_back(srcbuf.access_ptr(static_cast<int>(1), Handle(), 1, tmp_info.tensor_index[i]));
    }
    InsertArray(args, {1, 1, 1, 1, 0, 0, 0});
    return args;
  }

  Var vec_axis;
  const Store *store_ptr_;
  SelectInfo src_info_;
  int buffer_nunber_{0};
  std::vector<Var> producer_tensor;
  std::vector<Stmt> dump_buffer_;
  std::vector<std::pair<Var, Type>> make_buffer_;
};

class GenDMA : public GenSIMD {
 public:
  GenDMA(CCEInfo &t_info, Map<std::string, Buffer> &buffer_map, const std::string &pragma, bool has_if,
         bool has_transpose)
      : GenSIMD(t_info, buffer_map, pragma),
        loops_vars_(t_info.loops_vars_),
        loops_extent_(t_info.loops_extent_),
        has_if_(has_if),
        has_transpose_(has_transpose) {}
  ~GenDMA() override = default;

  Stmt Run() override {
    if (t_info_.loops_extent_.empty()) {
      vec_len_ = 1;
    } else {
      vec_len_ = t_info_.loops_extent_.back();
    }
    RemoveVectorizedIndex(t_info_, 0);

    std::map<std::string, std::string> intrin_buffer = {{DMA_COPY_GLOBAL, "gm"}, {SCOPE_UBUF, "ubuf"},
                                                        {SCOPE_CBUF, "cbuf"},    {SCOPE_CC, "matrix_cc"},
                                                        {SCOPE_CA, "ca"},        {SCOPE_CB, "cb"}};
    auto src_scope = GetBufferType(t_info_.src[0]);
    auto dst_scope = GetBufferType(t_info_.dst);
    std::string intrin_header = (dst_scope == SCOPE_CA || dst_scope == SCOPE_CB) ? "load_" : "copy_";
    if (intrin_buffer.count(src_scope) == 0 || intrin_buffer.count(dst_scope) == 0) {
      LOG(FATAL) << "\n_unsupported CCE_MOV scope strategy. \n" << t_info_.ori_stmt;
    }
    insn_intrinsic_ = intrin_header + intrin_buffer[src_scope] + "_to_" + intrin_buffer[dst_scope];

    if (insn_intrinsic_ == "copy_ubuf_to_ubuf" && has_transpose_) {
      Array<Expr> args = GenInsnAddress(t_info_, buffer_map_);
      ret = EmitCceInsn(t_info_.type, args, "vtranspose");
      if (!loops_vars_.empty()) {
        loops_extent_.pop_back();
        loops_vars_.pop_back();
      }
      return ret;
    }

    GenRepeat();

    if (insn_intrinsic_ == "copy_gm_to_ubuf" || insn_intrinsic_ == "copy_ubuf_to_ubuf" ||
        insn_intrinsic_ == "copy_gm_to_cbuf") {
      Array<Expr> args = GenInsnAddress(t_info_, buffer_map_);
      InsertArray(args,
                  {0, repeat_times_, floordiv(vec_len_ - 1, 32 / t_info_.type.bytes()) + 1, src_stride_, dst_stride_});
      if (insn_intrinsic_ == "copy_gm_to_cbuf") {
        auto pad = Call::make(Int(32), "tvm_cce_string_print", {StringImm::make("PAD_NONE")}, Call::PureIntrinsic);
        InsertArray(args, {pad});
      }
      ret = EmitCceInsn(t_info_.type, args, insn_intrinsic_);
    } else if (insn_intrinsic_ == "load_gm_to_cb" || insn_intrinsic_ == "load_gm_to_ca" ||
               insn_intrinsic_ == "load_cbuf_to_cb" || insn_intrinsic_ == "load_cbuf_to_ca") {
      Array<Expr> args = GenInsnAddress(t_info_, buffer_map_);
      InsertArray(args, {0, FloorDiv::make(vec_len_, 256), 1, 0});
      if (has_transpose_) {
        InsertArray(args, {1});
      } else {
        InsertArray(args, {0});
      }
      ret = EmitCceInsn(t_info_.type, args, insn_intrinsic_);
    } else if (insn_intrinsic_ == "copy_matrix_cc_to_ubuf") {
      Array<Expr> args = GenInsnAddress(t_info_, buffer_map_);
      InsertArray(args, {0, 1, floordiv(vec_len_ - 1, 256) + 1, 0, 0});
      std::string mod = "CRMODE_NONE";
      if (t_info_.src_type[0] == Float(32) && t_info_.type == Float(16)) {
        mod = "CRMODE_F32toF16_NONE";
      } else if (t_info_.src_type[0] == Float(16) && t_info_.type == Float(32)) {
        mod = "CRMODE_F16toF32_NONE";
      } else if (t_info_.src_type[0] == Int(32) && t_info_.type == Float(16)) {
        mod = "CRMODE_S32toF16_NONE";
      }
      auto cr_mod = Call::make(Int(32), "tvm_cce_string_print", {StringImm::make(mod)}, Call::PureIntrinsic);
      InsertArray(args, {cr_mod});
      ret = EmitCceInsn(t_info_.type, args, insn_intrinsic_);
    } else if (insn_intrinsic_ == "copy_ubuf_to_gm") {
      if (has_repeat_) {
        Array<Expr> args = GenInsnAddress(t_info_, buffer_map_);
        InsertArray(args, {0, repeat_times_, floordiv(vec_len_, 32 / t_info_.type.bytes()), src_stride_, dst_stride_});
        ret = EmitCceInsn(t_info_.type, args, insn_intrinsic_);
      } else {
        // Head Alignment (small condition)
        Expr small_align_cond = LT::make(vec_len_, 32 / t_info_.type.bytes());
        Stmt gm2ub = GenHeadAlign();
        Array<Expr> args1 = GenInsnAddress(t_info_, buffer_map_);
        InsertArray(args1, {0, 1, 1, 0, 0});
        Stmt ub2gm = EmitCceInsn(t_info_.type, args1, "copy_ubuf_to_gm");
        Stmt small_stmt = Block::make(gm2ub, ub2gm);

        // Body and Tail Alignment (large condition)
        Expr large_align_cond = GE::make(vec_len_, 32 / t_info_.type.bytes());
        Expr body_len = floordiv(vec_len_, 32 / t_info_.type.bytes());
        Array<Expr> args2 = GenInsnAddress(t_info_, buffer_map_);
        InsertArray(args2, {0, 1, body_len, 0, 0});
        Stmt body = EmitCceInsn(t_info_.type, args2, "copy_ubuf_to_gm");

        Expr tail_len = floormod(vec_len_, 32 / t_info_.type.bytes());
        Expr align_cond = GE::make(tail_len, 1);
        Stmt tail_stmt = GenTailAlign();
        Stmt if_tail = IfThenElse::make(align_cond, tail_stmt, Evaluate::make(0));

        Stmt large_stmt = Block::make(body, if_tail);

        // Whole Stmt
        Stmt if_small_copy = IfThenElse::make(small_align_cond, small_stmt, Evaluate::make(0));
        Stmt if_large_copy = IfThenElse::make(large_align_cond, large_stmt, Evaluate::make(0));
        ret = Block::make(if_small_copy, if_large_copy);
      }
    }
    if (!loops_vars_.empty()) {
      loops_extent_.pop_back();
      loops_vars_.pop_back();
    }
    return ret;
  }

  Stmt GenHeadAlign() {
    Expr fill_len = 32 / t_info_.type.bytes() - vec_len_;
    Expr gm_load_index = t_info_.dst_index + vec_len_;
    std::string align_buffer_name = "head_align_buffer_local_UB";
    Var align_var = Var(align_buffer_name);
    int align_buffer_len = 32 / t_info_.type.bytes();
    Buffer align_buffer = BufferNode::make(align_var, t_info_.type, {align_buffer_len}, {}, 0, align_buffer_name,
                                           GetBufScope(align_buffer_name), 1, 1, BufferType::kDefault);
    buffer_map_.Set(align_buffer_name, align_buffer);
    CHECK(buffer_map_.count(t_info_.dst->name_hint));
    Array<Expr> args = {align_buffer.access_ptr(static_cast<int>(2), Handle(), 1, 0),
                        buffer_map_[t_info_.dst->name_hint].access_ptr(static_cast<int>(1), Handle(), 1, gm_load_index),
                        0,
                        1,
                        1,
                        0,
                        0};
    Stmt gm2ub = EmitCceInsn(t_info_.type, args, "copy_gm_to_ubuf");
    Var loop_var = Var("ll0");
    Expr value = Load::make(t_info_.type, align_var, loop_var, 1);
    Stmt cp_gm_data = Store::make(t_info_.src[0], value, t_info_.src_index[0] + loop_var + vec_len_, 1);
    Stmt cp_ub = For::make(loop_var, 0, fill_len, ForType::Serial, DeviceAPI::None, cp_gm_data);
    Stmt fill_stmt = Block::make(gm2ub, cp_ub);
    Stmt allocate_stmt = Allocate::make(align_var, t_info_.type, {align_buffer_len}, const_true(1), fill_stmt);
    Stmt attr_stmt = AttrStmt::make(align_var, "storage_scope", Expr("local.UB"), allocate_stmt);
    return attr_stmt;
  }

  Stmt GenTailAlign() {
    Expr begin = vec_len_ - 32 / t_info_.type.bytes();
    Expr extend = 32 / t_info_.type.bytes();
    std::string align_buffer_name = "tail_align_buffer_local_UB";
    Var align_var = Var(align_buffer_name);
    int align_buffer_len = 32 / t_info_.type.bytes();
    Buffer align_buffer = BufferNode::make(align_var, t_info_.type, {align_buffer_len}, {}, 0, align_buffer_name,
                                           GetBufScope(align_buffer_name), 1, 1, BufferType::kDefault);
    buffer_map_.Set(align_buffer_name, align_buffer);
    Var loop_var = Var("ll0");
    Expr value = Load::make(t_info_.type, t_info_.src[0], loop_var + begin + t_info_.src_index[0], 1);
    Stmt move_ub_data = Store::make(align_var, value, loop_var, 1);
    Stmt move_ub = For::make(loop_var, 0, extend, ForType::Serial, DeviceAPI::None, move_ub_data);
    CHECK(buffer_map_.count(t_info_.dst->name_hint));
    Array<Expr> args = {
      buffer_map_[t_info_.dst->name_hint].access_ptr(static_cast<int>(2), Handle(), 1, t_info_.dst_index + begin),
      align_buffer.access_ptr(static_cast<int>(1), Handle(), 1, 0),
      0,
      1,
      1,
      0,
      0};
    Stmt ub2gm = EmitCceInsn(t_info_.type, args, "copy_ubuf_to_gm");
    Stmt tail_stmt = Block::make(move_ub, ub2gm);
    Stmt allocate_stmt = Allocate::make(align_var, t_info_.type, {align_buffer_len}, const_true(1), tail_stmt);
    Stmt attr_stmt = AttrStmt::make(align_var, "storage_scope", Expr("local.UB"), allocate_stmt);
    return attr_stmt;
  }

  void GenRepeat() {
    int burst_unit = 1024;
    bool is_dma_copy = (insn_intrinsic_ == "copy_gm_to_ubuf") || (insn_intrinsic_ == "copy_ubuf_to_ubuf") ||
                       (insn_intrinsic_ == "copy_gm_to_cbuf") || (insn_intrinsic_ == "copy_ubuf_to_gm");
    bool is_dma_load = (insn_intrinsic_ == "load_cbuf_to_cb") || (insn_intrinsic_ == "load_cbuf_to_ca");
    if (is_dma_copy) {
      burst_unit = 32 / t_info_.type.bytes();
    } else if (is_dma_load) {
      burst_unit = 256;
    } else {
      burst_unit = 1024;
    }
    has_repeat_ = false;
    repeat_times_ = 1;
    dst_stride_ = 0;
    src_stride_ = 0;
    Expr mod = Simplify(FloorMod::make(vec_len_, burst_unit));
    Expr div = Simplify(FloorDiv::make(vec_len_, burst_unit));
    if (!has_if_ && (is_dma_copy || (is_dma_load && div.as<IntImm>() && div.as<IntImm>()->value == 1)) &&
        loops_vars_.size() >= 2 && loops_extent_.size() >= 2 && mod.as<IntImm>() && mod.as<IntImm>()->value == 0) {
      Expr burst_len = (floordiv(vec_len_ - 1, burst_unit) + 1) * burst_unit;
      Var repeat_var = loops_vars_[loops_vars_.size() - 2];
      Expr dst_coef = GetVarCoefExpr(t_info_.dst_index, repeat_var);
      Expr src_coef = GetVarCoefExpr(t_info_.src_index[0], repeat_var);
      if (dst_coef.defined() && !Equal(dst_coef, 0) && src_coef.defined() && !Equal(src_coef, 0)) {
        has_repeat_ = true;
        repeat_times_ = loops_extent_[loops_extent_.size() - 2];
        dst_stride_ = FloorDiv::make(dst_coef - burst_len, burst_unit);
        t_info_.dst_index = EliminateVarInExpr(t_info_.dst_index, {repeat_var});
        src_stride_ = FloorDiv::make(src_coef - burst_len, burst_unit);
        for (size_t i = 0; i < t_info_.src_index.size(); ++i) {
          t_info_.src_index.Set(i, EliminateVarInExpr(t_info_.src_index[i], {repeat_var}));
        }
        loops_vars_.erase(loops_vars_.end() - 2);
        loops_extent_.erase(loops_extent_.end() - 2);
      }
    }
  }

  std::vector<Var> loops_vars_;
  std::vector<Expr> loops_extent_;
  bool has_if_{false};
  bool has_transpose_{false};
  bool has_repeat_{false};
  Expr repeat_times_;
  Expr dst_stride_;
  Expr src_stride_;
  Stmt ret;
};

class GenReduce {
 public:
  GenReduce(const CCEInfo &t_info, Map<std::string, Buffer> &buffer_map, const std::string &pragma)
      : t_info_(t_info), vec_len_(t_info.loops_extent_.back()), buffer_map_(buffer_map), pragma_(pragma) {}
  ~GenReduce() = default;

  Stmt Run(int pre_index) {
    is_arg_type_ = (pragma_ == "arg_max" || pragma_ == "arg_min");
    RemoveVectorizedIndex(t_info_, 0);
    if (pragma_.find("sum") != std::string::npos) {
      insn_intrinsic_ = "vcadd";
      expansion_factor_ = 1;
    } else if (pragma_.find("max") != std::string::npos) {
      insn_intrinsic_ = "vcmax";
      expansion_factor_ = 2;
    } else if (pragma_.find("min") != std::string::npos) {
      insn_intrinsic_ = "vcmin";
      expansion_factor_ = 2;
    }

    int repeat_size = 256 / t_info_.type.bytes();
    int one_pass_threshold = repeat_size;
    int two_pass_threshold = repeat_size * (repeat_size / expansion_factor_);
    int three_pass_threshold = repeat_size * 256;

    Expr if_gen_one_pass = LE::make(vec_len_, one_pass_threshold);
    Expr if_gen_two_pass = And::make(GT::make(vec_len_, one_pass_threshold), LE::make(vec_len_, two_pass_threshold));
    Expr if_gen_three_pass =
      And::make(GT::make(vec_len_, two_pass_threshold), LE::make(vec_len_, three_pass_threshold));

    Stmt if_one_pass_reduce = IfThenElse::make(if_gen_one_pass, GenReduceCond(pre_index, 1, 1));

    Stmt if_two_pass_reduce = IfThenElse::make(if_gen_two_pass, GenReduceCond(pre_index, 2, 2));

    Stmt if_three_pass_reduce = IfThenElse::make(if_gen_three_pass, GenReduceCond(pre_index, 3, 3));

    Stmt ret = Block::make(Block::make(if_one_pass_reduce, if_two_pass_reduce), if_three_pass_reduce);

    return ret;
  }

  Stmt GenReduceCond(int suffix1, int suffix2, int type) {
    int repeat_size = 256 / t_info_.type.bytes();
    Var first_pass_var =
      Var("first_pass_buffer_" + std::to_string(suffix1) + "_" + std::to_string(suffix2) + "_local_UB");
    Var second_pass_var =
      Var("second_pass_buffer" + std::to_string(suffix1) + "_" + std::to_string(suffix2) + "_local_UB");
    Var third_pass_var =
      Var("third_pass_buffer" + std::to_string(suffix1) + "_" + std::to_string(suffix2) + "_local_UB");

    Expr first_reduce_len = vec_len_;
    Expr second_reduce_len = (FloorDiv::make(vec_len_ - 1, repeat_size) + 1) * expansion_factor_;
    Expr third_reduce_len = (FloorDiv::make(second_reduce_len - 1, repeat_size) + 1) * expansion_factor_;
    Expr LastReduceLen = (FloorDiv::make(third_reduce_len - 1, repeat_size) + 1) * expansion_factor_;

    std::string data_format = "serial";
    if (insn_intrinsic_ == "vcmax" || insn_intrinsic_ == "vcmin") {
      data_format = "gap";
    }
    Stmt first_pass =
      GenReducePass(first_reduce_len, first_pass_var, 0, t_info_.src[0], t_info_.src_index[0], t_info_.type);
    Stmt second_pass =
      GenReducePass(second_reduce_len, second_pass_var, 0, first_pass_var, 0, t_info_.type, data_format);
    Stmt third_pass = GenReducePass(third_reduce_len, third_pass_var, 0, second_pass_var, 0, t_info_.type, data_format);

    if (type == 1) {
      // only one pass
      Stmt assign_store1;
      if (!is_arg_type_) {
        assign_store1 = GenMergeResults(first_pass_var);
      } else {
        assign_store1 = Store::make(t_info_.dst, Load::make(t_info_.type, first_pass_var, 1, 1), t_info_.dst_index, 1);
      }
      Stmt one_reduce = Block::make(first_pass, assign_store1);
      Stmt allocate_stmt = Allocate::make(first_pass_var, t_info_.type, {second_reduce_len}, const_true(1), one_reduce);
      Stmt one_attr1 = AttrStmt::make(first_pass_var, "storage_scope", Expr("local.UB"), allocate_stmt);
      return one_attr1;
    } else if (type == 2) {
      // only two pass
      Stmt assign_store2;
      if (is_arg_type_) {
        Type arg_type = UInt(16);
        Var arg_reg = Var("arg_reg_" + std::to_string(suffix1) + "_" + std::to_string(suffix2));
        Buffer arg_buf =
          BufferNode::make(arg_reg, arg_type, {16}, {}, 0, arg_reg->name_hint, "local.REG", 1, 1, BufferType::kDefault);
        buffer_map_.Set(arg_reg->name_hint, arg_buf);

        Stmt get_args2 = Store::make(arg_reg, Load::make(arg_type, second_pass_var, 1, 1), 0, 1);
        Expr get_index = Load::make(arg_type, arg_reg, 0, 1);
        Expr offset = get_index * 64;
        Stmt get_args00 = Store::make(arg_reg, Load::make(arg_type, first_pass_var, get_index + 1, 1), 2, 1);
        Stmt get_args1 =
          Store::make(t_info_.dst, Add::make(offset, Load::make(arg_type, arg_reg, 2, 1)), t_info_.dst_index, 1);
        Stmt gen_arg = Block::make(get_args2, Block::make(get_args00, get_args1));

        Stmt gen_allocate = Allocate::make(arg_reg, arg_type, {16}, const_true(1), gen_arg);
        assign_store2 = AttrStmt::make(arg_reg, "storage_scope", Expr("local.REG"), gen_allocate);
      } else {
        assign_store2 = GenMergeResults(second_pass_var);
      }
      Stmt two_reduce = Block::make(Block::make(first_pass, second_pass), assign_store2);
      Stmt allocate_stmt1 =
        Allocate::make(first_pass_var, t_info_.type, {second_reduce_len}, const_true(1), two_reduce);
      Stmt two_attr1 = AttrStmt::make(first_pass_var, "storage_scope", Expr("local.UB"), allocate_stmt1);
      Stmt allocate_stmt2 = Allocate::make(second_pass_var, t_info_.type, {third_reduce_len}, const_true(1), two_attr1);
      Stmt two_attr2 = AttrStmt::make(second_pass_var, "storage_scope", Expr("local.UB"), allocate_stmt2);
      return two_attr2;
    } else if (type == 3) {
      Stmt assign_store3;
      // only three pass
      if (is_arg_type_) {
        Type arg_type = UInt(16);
        Var arg_reg = Var("arg_reg_" + std::to_string(suffix1) + "_" + std::to_string(suffix2));
        Buffer arg_buf =
          BufferNode::make(arg_reg, arg_type, {16}, {}, 0, arg_reg->name_hint, "local.REG", 1, 1, BufferType::kDefault);
        buffer_map_.Set(arg_reg->name_hint, arg_buf);

        Stmt get_args3 = Store::make(arg_reg, Load::make(arg_type, third_pass_var, 1, 1), 0, 1);
        Expr get_index3 = Load::make(arg_type, arg_reg, 0, 1);
        Expr offset3 = get_index3 * 64 * 32;

        Stmt get_args2 = Store::make(arg_reg, Load::make(arg_type, second_pass_var, get_index3 + 1, 1), 2, 1);
        Expr get_index2 = Load::make(arg_type, arg_reg, 2, 1);
        Expr offset2 = get_index2 * 64;

        Stmt get_args1 = Store::make(arg_reg, Load::make(arg_type, first_pass_var, get_index2 + 1, 1), 4, 1);
        Expr get_index1 = Load::make(arg_type, arg_reg, 4, 1);
        Expr offset1 = get_index1;

        Expr index = Add::make(Add::make(offset3, offset2), offset1);
        Stmt get_args = Store::make(t_info_.dst, index, t_info_.dst_index, 1);
        Stmt gen_arg = Block::make(Block::make(get_args3, get_args2), Block::make(get_args1, get_args));
        Stmt gen_allocate = Allocate::make(arg_reg, arg_type, {16}, const_true(1), gen_arg);
        assign_store3 = AttrStmt::make(arg_reg, "storage_scope", Expr("local.REG"), gen_allocate);
      } else {
        assign_store3 = GenMergeResults(third_pass_var);
      }
      Stmt three_reduce = Block::make(Block::make(Block::make(first_pass, second_pass), third_pass), assign_store3);
      Stmt allocate_stmt1 =
        Allocate::make(first_pass_var, t_info_.type, {second_reduce_len}, const_true(1), three_reduce);
      Stmt three_attr1 = AttrStmt::make(first_pass_var, "storage_scope", Expr("local.UB"), allocate_stmt1);
      Stmt allocate_stmt2 =
        Allocate::make(second_pass_var, t_info_.type, {third_reduce_len}, const_true(1), three_attr1);
      Stmt three_attr2 = AttrStmt::make(second_pass_var, "storage_scope", Expr("local.UB"), allocate_stmt2);
      Stmt allocate_stmt3 = Allocate::make(third_pass_var, t_info_.type, {LastReduceLen}, const_true(1), three_attr2);
      Stmt three_attr3 = AttrStmt::make(third_pass_var, "storage_scope", Expr("local.UB"), allocate_stmt3);
      return three_attr3;
    } else {
      return Evaluate::make(1);
    }
  }

  Stmt GenReducePass(Expr reduce_len, Var dst_var, Expr dst_index, Var src_var, Expr src_index, Type t_type,
                     const std::string &source_type = "serial") {
    buf_len_ = FloorDiv::make(reduce_len - 1, 256 / t_type.bytes()) + 1;
    buf_len_ = buf_len_ * expansion_factor_;
    Buffer reduce_buf =
      BufferNode::make(dst_var, t_type, {buf_len_}, {}, 0, dst_var->name_hint, "local_UB", 1, 1, BufferType::kDefault);
    buffer_map_.Set(dst_var->name_hint, reduce_buf);

    Expr repeat = Div::make(reduce_len, 256 / t_type.bytes());
    Expr tail_len = Mod::make(reduce_len, 256 / t_type.bytes());
    Expr if_head_condition = GT::make(repeat, 0);
    Expr if_tail_condition = GT::make(tail_len, 0);
    Expr tail_dst_offset, tail_src_offset;
    tail_src_offset = reduce_len - tail_len;
    tail_dst_offset = repeat * expansion_factor_;

    // Head Stmt
    CCEInfo head_info(dst_var, dst_index, {src_var}, {src_index}, t_type);
    Array<Expr> head_args = GenInsnAddress(head_info, buffer_map_);
    InsertArray(head_args, {repeat, 1, 1, 8});
    Stmt head_mask;
    if (source_type == "serial") {
      head_mask = EmitCceInsn(t_type, {make_const(UInt(64), -1), make_const(UInt(64), -1)}, "set_vector_mask");
    } else {
      head_mask =
        EmitCceInsn(t_type, {make_const(UInt(64), FullReduceMaskValue), make_const(UInt(64), FullReduceMaskValue)},
                    "set_vector_mask");
    }
    Stmt head_reduce = EmitCceInsn(head_info.type, head_args, insn_intrinsic_);
    Stmt if_head = IfThenElse::make(if_head_condition, Block::make(head_mask, head_reduce));

    // Tail Stmt
    CCEInfo tail_info(dst_var, dst_index + tail_dst_offset, {src_var}, {src_index + tail_src_offset}, t_type);
    Array<Expr> tail_args = GenInsnAddress(tail_info, buffer_map_);
    InsertArray(tail_args, {1, 1, 1, 8});
    Stmt tail_mask;

    if (source_type == "serial") {
      tail_mask = SetMask(t_type, false, tail_len, false);
    } else {
      tail_mask = SetMask(t_type, false, tail_len, true);
    }
    Stmt tail_reduce = EmitCceInsn(tail_info.type, tail_args, insn_intrinsic_);
    Stmt if_tail = IfThenElse::make(if_tail_condition, Block::make(tail_mask, tail_reduce));
    Stmt reduce_stmt = Block::make(if_head, if_tail);
    return reduce_stmt;
  }

  Stmt GenMergeResults(Var &reducebuf) {
    Var buffer_var("temp_var_local_UB");
    Buffer arg_buf = BufferNode::make(buffer_var, t_info_.type, {1}, {}, 0, buffer_var->name_hint, "local.REG", 1, 1,
                                      BufferType::kDefault);
    buffer_map_.Set(buffer_var->name_hint, arg_buf);
    Stmt copy_from_dst = Store::make(buffer_var, Load::make(t_info_.type, t_info_.dst, t_info_.dst_index, 1), 0, 1);
    Stmt copy_to_dst = Store::make(t_info_.dst, Load::make(t_info_.type, buffer_var, 0, 1), t_info_.dst_index, 1);

    Stmt mask = EmitCceInsn(t_info_.type, {make_const(UInt(64), 0), make_const(UInt(64), 1)}, "set_vector_mask");
    CCEInfo temp_info;
    temp_info.dst = buffer_var;
    temp_info.dst_index = 0;
    temp_info.src.push_back(buffer_var);
    temp_info.src.push_back(reducebuf);
    temp_info.src_index.push_back(0);
    temp_info.src_index.push_back(0);
    auto args = GenInsnAddress(temp_info, buffer_map_);
    InsertArray(args, {1, 1, 1, 1, 0, 0, 0});
    std::string vectorinsn_name =
      (insn_intrinsic_ == "vcadd") ? "vadd" : ((insn_intrinsic_ == "vcmax") ? "vmax" : "vmin");
    Stmt vector_insn = EmitCceInsn(t_info_.type, args, vectorinsn_name);
    Stmt main_stmt = Block::make(Block::make(copy_from_dst, Block::make(mask, vector_insn)), copy_to_dst);
    Stmt allocate_stmt = Allocate::make(buffer_var, t_info_.type, {1}, const_true(1), main_stmt);
    Stmt attr_stmt = AttrStmt::make(buffer_var, "storage_scope", Expr("local.UB"), allocate_stmt);
    return attr_stmt;
  }

  CCEInfo t_info_;
  Expr vec_len_;
  Expr buf_len_;
  Map<std::string, Buffer> buffer_map_;
  std::string pragma_;
  std::string insn_intrinsic_;
  bool is_arg_type_{false};
  int expansion_factor_ = 1;
};

class EmitVariableInsns : public IRMutator {
 public:
  explicit EmitVariableInsns(const Map<Tensor, Buffer> &extern_buffer) {
    for (auto kv : extern_buffer) {
      buffer_map.Set(kv.second->name, kv.second);
    }
  }
  ~EmitVariableInsns() override = default;

  Stmt Emit(const Stmt &s) { return this->Mutate(s); }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    Buffer var_buf = BufferNode::make(op->buffer_var, op->type, op->extents, {}, 0, op->buffer_var->name_hint,
                                      GetBufScope(op->buffer_var->name_hint), 1, 1, BufferType::kDefault);
    buffer_map.Set(op->buffer_var->name_hint, var_buf);
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (ktvm::ir::attr::IsPragmaKey(op->attr_key) && op->attr_key == "pragma_emit_insn") {
      CHECK(op->value.as<StringImm>());
      pragma = op->value.as<StringImm>()->value;
      Stmt r;
      gen_cce = Stmt();
      if (pragma == "mad") {
        r = MadEmitter(s);
      } else if (pragma == "scalar_dma" || pragma == "scatter") {
        r = op->body;
      } else {
        in_emit_insn_ = true;
        loops_extent_.clear();
        loops_vars_.clear();
        if_vector_.clear();
        static_cast<void>(IRMutator::Mutate_(op, s));
        r = gen_cce;
        in_emit_insn_ = false;
      }
      CHECK(r.defined()) << "\nintrinsic rule must always return valid Expr for: " << pragma << "\n\n";
      if (!r.same_as(s)) {
        return r;
      }
    } else if (ktvm::ir::attr::IsPragmaKey(op->attr_key) &&
               (op->attr_key == "pragma_im2col" || op->attr_key == "pragma_load3d")) {
      if (paramters_.defined() && Downcast<Map<std::string, NodeRef>>(paramters_).count("feature")) {
        auto feature = Downcast<Map<std::string, NodeRef>>(paramters_)["feature"].as<StringImm>();
        CHECK(feature);
        std::string feature_tensor = feature->value + "_local_L1";
        CHECK(buffer_map.count(feature_tensor));
        Buffer src = buffer_map[feature_tensor];
        CHECK(op->node.as<StrMapNode>());
        Stmt r;
        if (op->attr_key == "pragma_im2col") {
          r = Im2ColEmitter(op->body, op->node.as<StrMapNode>()->data, src, true);
        } else {
          r = Im2ColEmitterL1UB(op->body, op->node.as<StrMapNode>()->data, src, true);
        }
        return r;
      }
    } else if (op->attr_key == "pragma_attrs") {
      paramters_ = op->node;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (in_emit_insn_) {
      loops_extent_.push_back(op->extent);
      loops_vars_.push_back(op->loop_var);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (in_emit_insn_) {
      if_vector_.push_back(op);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    CCEInfo t_info;
    t_info.ori_stmt = s;

    t_info.dst = op->buffer_var;
    t_info.dst_index = op->index;
    t_info.type = op->value.type();
    t_info.loops_extent_ = loops_extent_;
    t_info.loops_vars_ = loops_vars_;
    GetSrcInfo(t_info, op->value);

    if (pragma.find("vec_select") != std::string::npos) {
      EmitSelect(op, t_info);
    } else if (pragma.find("dma_copy") == 0) {
      EmitDMA(t_info);
    } else if (pragma.find("vec_binary") == 0 || pragma.find("vec_single") == 0) {
      EmitSIMD(t_info);
    } else if (pragma.find("reduce") == 0 || pragma.find("arg_") == 0) {
      EmitReduce(t_info);
    } else if (pragma.find("broadcast") == 0) {
      if (loops_vars_.empty()) {
        gen_cce = t_info.ori_stmt;
      } else {
        EmitSIMD(t_info);
      }
    } else {
      EmitIntrinsicCall(t_info);
    }

    GenForIfBlock();
    return IRMutator::Mutate_(op, s);
  }

  void GetSrcInfo(CCEInfo &t_info, Expr value) {
    auto FGetSource = [&t_info](const NodeRef &node) {
      if (node.as<Load>()) {
        t_info.src.push_back(node.as<Load>()->buffer_var);
        t_info.src_index.push_back(node.as<Load>()->index);
        t_info.src_type.push_back(node.as<Load>()->type);
      } else if (node.as<FloatImm>()) {
        t_info.imm = FloatImm::make(node.as<FloatImm>()->type, node.as<FloatImm>()->value);
      } else if (node.as<IntImm>()) {
        t_info.imm = IntImm::make(node.as<IntImm>()->type, node.as<IntImm>()->value);
      } else if (node.as<Variable>()) {
        t_info.imm = Var(node.as<Variable>()->name_hint, node.as<Variable>()->type);
      }
    };
    if (auto cast = value.as<Cast>()) {
      value = cast->value;
    }
    PostOrderVisit(value, FGetSource);
  }

  void GenForIfBlock() {
    for (auto iter = if_vector_.rbegin(); iter != if_vector_.rend(); ++iter) {
      gen_cce = IfThenElse::make((*iter)->condition, gen_cce);
    }
    if (!loops_extent_.empty()) {
      for (int i = static_cast<int>(loops_extent_.size() - 1); i >= 0; i--) {
        gen_cce = For::make(loops_vars_[i], 0, loops_extent_[i], ForType::Serial, DeviceAPI::None, gen_cce);
      }
    }
  }

  void EmitDMA(CCEInfo &t_info) {
    GenDMA dma_case = GenDMA(t_info, buffer_map, pragma, !if_vector_.empty(), (pragma == "dma_copy_transpose"));
    gen_cce = dma_case.Run();
    loops_vars_.assign(dma_case.loops_vars_.begin(), dma_case.loops_vars_.end());
    loops_extent_.assign(dma_case.loops_extent_.begin(), dma_case.loops_extent_.end());
    if (!gen_cce.defined()) {
      gen_cce = Evaluate::make(0);
    }
  }

  void EmitSIMD(CCEInfo &t_info) {
    gen_cce = GenSIMD(t_info, buffer_map, pragma).Run();
    loops_vars_ = t_info.loops_vars_;
    loops_extent_ = t_info.loops_extent_;
  }

  void EmitSelect(const Store *op, CCEInfo &t_info) {
    if (loops_extent_.empty()) {
      gen_cce = t_info.ori_stmt;
      return;
    }
    gen_cce = GenSelect(t_info, buffer_map, pragma, op).Run();
    PopBack(t_info);
  }

  void EmitReduce(CCEInfo &t_info) {
    gen_cce = GenReduce(t_info, buffer_map, pragma).Run(buffer_pre_index_++);
    PopBack(t_info);
  }

  void EmitIntrinsicCall(CCEInfo &t_info) {
    Array<Expr> args = GenInsnAddress(t_info, buffer_map);
    if ((pragma == "vec_argmax_cast" || pragma == "vec_argmin_cast")) {
      gen_cce = Evaluate::make(Call::make(t_info.type, "argmax_cast", args, Call::CallType::Extern));
      if (!loops_vars_.empty()) {
        size_t index = loops_vars_.size() - 1;
        gen_cce = For::make(loops_vars_[index], 0, loops_extent_[index], ForType::Serial, DeviceAPI::None, gen_cce);
      }
    }
  }

  void PopBack(CCEInfo &t_info) {
    CHECK(!t_info.loops_vars_.empty());
    CHECK(!t_info.loops_extent_.empty());
    t_info.loops_vars_.pop_back();
    t_info.loops_extent_.pop_back();
    loops_vars_ = t_info.loops_vars_;
    loops_extent_ = t_info.loops_extent_;
  }

 private:
  bool in_emit_insn_{false};
  std::vector<Expr> loops_extent_;
  std::vector<Var> loops_vars_;
  std::string pragma;
  Stmt gen_cce;
  Map<std::string, Buffer> buffer_map;
  NodeRef paramters_;
  std::vector<const IfThenElse *> if_vector_;
  int buffer_pre_index_{0};
};

Stmt EmitInsnWithDynamicShapes(const Stmt &s, const Map<Tensor, Buffer> &extern_buffer) {
  auto unique_for_var = ForVarUnique().Mutate(s);
  auto trans_stmt = TransposeTransform().Mutate(unique_for_var);
  auto adjust_stmt = AdjustPragma().Mutate(trans_stmt);
  auto for_stmt = LoopReorder().Mutate(adjust_stmt);
  auto if_stmt = IfReorder().Mutate(for_stmt);
  auto ret = EmitVariableInsns(extern_buffer).Emit(if_stmt);
  ret = RemoveNoOp(ret);
  ret = CanonicalSimplify(ret);
  return ret;
}
}  // namespace ir
}  // namespace akg
