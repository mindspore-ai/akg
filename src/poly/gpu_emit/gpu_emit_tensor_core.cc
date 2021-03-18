/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

/*!
 * \file gpu_emit_tensor_core.cc
 */

#include "emit_pass.h"
#include <stack>

namespace akg {
namespace ir {
namespace poly {

class CheckCast : public IRVisitor {
 public:
  explicit CheckCast() {}
  using IRVisitor::Visit_;

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == CAST_FLAG) {
      std::string mode = op->value.as<StringImm>()->value;
      if (mode == CAST_MODE_1) {
        origin_type_ = Float(32);
        cast_type_ = Float(16);
      }
      is_cast_ = true;
      IRVisitor::Visit_(op);
      return;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Call *op) final {
    if (op->is_intrinsic(air::ir::intrinsic::tvm_mma_sync)) {
      CHECK_EQ(op->args.size(), 8U);
      Expr arg2 = op->args[2];
      Expr arg4 = op->args[4];
      const Variable *a2 = arg2.as<Variable>();
      CHECK(a2);
      const Variable *a4 = arg4.as<Variable>();
      CHECK(a4);
      cast_tensors_.insert(SimplifyName(a2->name_hint));
      cast_tensors_.insert(SimplifyName(a4->name_hint));
    }
    IRVisitor::Visit_(op);
  }

  bool IsCastAdapt() { return is_cast_; }
  friend class CollectInfoToAdaptCast;

 private:
  Type origin_type_;
  Type cast_type_;
  bool is_cast_{false};
  std::set<std::string> cast_tensors_;
};

class CollectInfoToAdaptCast : public IRVisitor {
 public:
  explicit CollectInfoToAdaptCast(CheckCast &check_cast)
      : origin_type_(check_cast.origin_type_),
        cast_type_(check_cast.cast_type_),
        cast_tensors_(check_cast.cast_tensors_) {}
  using IRVisitor::Visit_;

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == GMREAD_FLAG) {
      is_global_to_shared_ = true;
      IRVisitor::Visit_(op);
      is_global_to_shared_ = false;
      return;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide *op) final {
    if (is_global_to_shared_) {
      global_to_shared_.insert(op);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Realize *op) final {
    std::string name = op->func->func_name();
    if (IsEndsWith(name, SHARE_SUFFIX) && cast_tensors_.count(SimplifyName(name))) {
      realize_need_cast_shared_.insert(name);
    } else if (IsEndsWith(name, LOCAL_SUFFIX) && cast_tensors_.count(SimplifyName(name))) {
      realize_need_cast_local_.insert(name);
    }
    IRVisitor::Visit_(op);
  }

  friend class AdaptCast;

 private:
  Type origin_type_;
  Type cast_type_;
  bool is_global_to_shared_{false};
  std::set<std::string> cast_tensors_;

  std::set<const Provide *> global_to_shared_;
  std::set<std::string> realize_need_cast_shared_;
  std::set<std::string> realize_need_cast_local_;
};

class AdaptCast : public IRMutator {
 public:
  explicit AdaptCast(CollectInfoToAdaptCast &info)
      : realize_need_cast_shared_(info.realize_need_cast_shared_),
        realize_need_cast_local_(info.realize_need_cast_local_),
        global_to_shared_(info.global_to_shared_),
        origin_type_(info.origin_type_),
        cast_type_(info.cast_type_) {}

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::buffer_bind_scope) {
      Array<NodeRef> arr = Downcast<Array<NodeRef>>(op->node);
      CHECK_EQ(arr.size(), 2U);
      const BufferNode *buffer = arr[0].as<BufferNode>();
      const TensorNode *tensor = arr[1].as<TensorNode>();
      const Call *tuple = op->value.as<Call>();
      CHECK(buffer && tensor);
      CHECK(tuple);
      if (realize_need_cast_local_.count(buffer->name)) {
        NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
        buffer_node->data = buffer->data;
        buffer_node->name = buffer->name;
        buffer_node->scope = buffer->scope;
        buffer_node->dtype = cast_type_;
        buffer_node->shape = buffer->shape;
        buffer_node->strides = buffer->strides;
        buffer_node->data_alignment = buffer->data_alignment;
        buffer_node->elem_offset = buffer->elem_offset;
        buffer_node->offset_factor = buffer->offset_factor;

        Buffer buffer_new(buffer_node);
        NodePtr<TensorNode> tensor_node = make_node<TensorNode>();
        tensor_node->value_index = tensor->value_index;
        tensor_node->op = tensor->op;
        tensor_node->shape = tensor->shape;
        tensor_node->dtype = cast_type_;
        Tensor tensor_new(tensor_node);

        Array<NodeRef> node = {buffer_new, tensor_new};
        Stmt body = this->Mutate(op->body);
        return AttrStmt::make(node, op->attr_key, op->value, body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    std::string tensor_name = op->func->func_name();
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Realize>();
    if (op != nullptr) {
      if (!realize_need_cast_shared_.count(tensor_name) && !realize_need_cast_local_.count(tensor_name)) {
        return stmt;
      }

      return Realize::make(op->func, op->value_index, cast_type_, op->bounds, op->condition, op->body);
    }
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (global_to_shared_.count(op)) {
      auto value = op->value;
      auto call = value.as<Call>();
      CHECK(call);
      CHECK(call->type == origin_type_);
      value = Cast::make(cast_type_, value);
      return Provide::make(op->func, op->value_index, value, op->args);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::set<std::string> realize_need_cast_shared_;
  std::set<std::string> realize_need_cast_local_;
  std::set<const Provide *> global_to_shared_;
  Type origin_type_;
  Type cast_type_;
};

class AdaptCastDesignOne : public IRMutator {
 public:
  explicit AdaptCastDesignOne(TensorCoreInfo &info) : cast_tensors_(info.cast_tensors_) {}

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    std::string tensor_name = op->func->func_name();
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Realize>();
    if (op != nullptr) {
      if (!cast_tensors_.count(tensor_name)) {
        return stmt;
      }

      return Realize::make(op->func, op->value_index, Float(16), op->bounds, op->condition, op->body);
    }
    return stmt;
  }

 private:
  std::unordered_set<std::string> cast_tensors_;
};

class ModifySizeOfLocal : public IRMutator {
 public:
  explicit ModifySizeOfLocal(TensorCoreInfo &info) : info_(info) {
    m_size_ = Expr(info_.warp_tile_.m);
    m_size_ = Mul::make(m_size_, info_.fragment_m_.defined() ? info_.fragment_m_ : make_const(Int(32), 1));
    n_size_ = Expr(info_.warp_tile_.n);
    n_size_ = Mul::make(n_size_, info_.fragment_n_.defined() ? info_.fragment_n_ : make_const(Int(32), 1));
  }
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::buffer_bind_scope) {
      Array<NodeRef> arr = Downcast<Array<NodeRef>>(op->node);
      CHECK_EQ(arr.size(), 2U);
      const BufferNode *buffer = arr[0].as<BufferNode>();
      const TensorNode *tensor = arr[1].as<TensorNode>();
      const Call *tuple = op->value.as<Call>();
      CHECK(buffer && tensor);
      CHECK(tuple);
      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      buffer_node->data = buffer->data;
      buffer_node->name = buffer->name;
      buffer_node->scope = buffer->scope;
      buffer_node->dtype = buffer->dtype;

      auto old_shape = buffer->shape;
      size_t len = old_shape.size();
      CHECK_GE(len, 2);

      std::string base_name = SimplifyName(buffer->name);
      auto matrix = info_.matrix_abc_[base_name];
      auto major = info_.matrix_major_[base_name];

      int mod_index = -1;
      Expr shape_mod;
      bool is_c_matrix = false;

      if (matrix == MATRIX_A) {
        if (major == ROW_MAJOR) {
          mod_index = len - 2;
        } else if (major == COL_MAJOR) {
          mod_index = len - 1;
        }
        shape_mod = m_size_;
      } else if (matrix == MATRIX_B) {
        if (major == ROW_MAJOR) {
          mod_index = len - 1;
        } else if (major == COL_MAJOR) {
          mod_index = len - 2;
        }
        shape_mod = n_size_;
      } else if (matrix == MATRIX_C) {
        is_c_matrix = true;
      }

      Array<Expr> new_shape;
      if (!is_c_matrix) {
        for (size_t i = 0; i < len; ++i) {
          if (i == static_cast<size_t>(mod_index)) {
            CHECK(shape_mod.defined());
            new_shape.push_back(shape_mod);
            continue;
          }
          new_shape.push_back(old_shape[i]);
        }
        buffer_node->shape = new_shape;

      } else {
        int len = buffer->shape.size();
        new_shape = ModifyCShape(buffer, len);
        buffer_node->shape = new_shape;
      }

      Array<Expr> strides;
      for (size_t i = 1; i < new_shape.size(); ++i) {
        Expr stride = IntImm::make(Int(32), 1);
        for (size_t j = new_shape.size() - 1; j >= i; --j) {
          stride = Mul::make(stride, new_shape[j]);
        }
        strides.push_back(stride);
      }
      strides.push_back(make_const(Int(32), 1));

      buffer_node->strides = strides;
      buffer_node->data_alignment = buffer->data_alignment;

      // elem_offset need modify now
      auto old_args = tuple->args;
      Array<Expr> new_args;
      if (!is_c_matrix) {
        size_t mod_index_arg = 2 * mod_index + 1;
        for (size_t i = 0; i < old_args.size(); ++i) {
          if (i == mod_index_arg) {
            new_args.push_back(shape_mod);
            continue;
          }
          new_args.push_back(old_args[i]);
        }
      } else {
        // C Matrix modify
        size_t mod_index_n = 2 * (len - 1) + 1;
        size_t mod_index_m = 2 * (len - 2) + 1;
        for (size_t i = 0; i < old_args.size(); ++i) {
          if (i == mod_index_n) {
            new_args.push_back(n_size_);
          } else if (i == mod_index_m) {
            new_args.push_back(m_size_);
          } else {
            new_args.push_back(old_args[i]);
          }
        }
      }

      Array<Expr> call_args;
      for (size_t i = 0; i < new_args.size();) {
        call_args.push_back(new_args[i]);
        i += 2;
      }

      Expr elem_offset_new = IntImm::make(Int(32), 0);
      auto min_bound = info_.min_bounds_[buffer->name];
      CHECK(min_bound.defined()) << "min_bound should be defined";
      CHECK_EQ(call_args.size(), min_bound.size());
      for (size_t i = 0; i < min_bound.size(); i++) {
        elem_offset_new = Add::make(elem_offset_new, Mul::make(strides[i], Sub::make(call_args[i], min_bound[i])));
      }

      buffer_node->elem_offset = elem_offset_new;
      buffer_node->offset_factor = buffer->offset_factor;

      Buffer buffer_new(buffer_node);
      NodePtr<TensorNode> tensor_node = make_node<TensorNode>();
      tensor_node->value_index = tensor->value_index;
      tensor_node->op = tensor->op;
      tensor_node->shape = new_shape;
      tensor_node->dtype = tensor->dtype;
      Tensor tensor_new(tensor_node);

      Array<NodeRef> node = {buffer_new, tensor_new};

      auto tuple_new =
        Call::make(tuple->type, tuple->name, new_args, tuple->call_type, tuple->func, tuple->value_index);

      Stmt body = this->Mutate(op->body);
      return AttrStmt::make(node, op->attr_key, tuple_new, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    std::string tensor_name = op->func->func_name();
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Realize>();
    if (op != nullptr) {
      if (!info_.frag_reg_.count(tensor_name)) {
        return stmt;
      }

      std::string base_name = SimplifyName(tensor_name);
      auto matrix = info_.matrix_abc_[base_name];
      auto major = info_.matrix_major_[base_name];

      Region new_bounds;
      size_t len = op->bounds.size();
      CHECK_GE(len, 2) << "bounds size should be greater than 2";

      int mod_index = -1;

      if (matrix == MATRIX_A) {
        if (major == ROW_MAJOR) {
          mod_index = len - 2;
        } else if (major == COL_MAJOR) {
          mod_index = len - 1;
        }
        new_bounds = ModifyAbRegion(op, mod_index, m_size_);
      } else if (matrix == MATRIX_B) {
        if (major == ROW_MAJOR) {
          mod_index = len - 1;
        } else if (major == COL_MAJOR) {
          mod_index = len - 2;
        }
        new_bounds = ModifyAbRegion(op, mod_index, n_size_);
      } else if (matrix == MATRIX_C) {
        new_bounds = ModifyCRegion(op, len);
      }
      return Realize::make(op->func, op->value_index, op->type, new_bounds, op->condition, op->body);
    }
    return stmt;
  }

  Region ModifyAbRegion(const Realize *op, int mod_index, Expr mod_size) {
    Region new_bounds;
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      if (i == static_cast<size_t>(mod_index)) {
        new_bounds.push_back(Range::make_by_min_extent(op->bounds[i]->min, mod_size));
        continue;
      } else {
        new_bounds.push_back(op->bounds[i]);
      }
    }
    return new_bounds;
  }

  Region ModifyCRegion(const Realize *op, int len) {
    Region new_bounds;
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      if (i == static_cast<size_t>(len - 1)) {
        new_bounds.push_back(Range::make_by_min_extent(op->bounds[i]->min, n_size_));
        continue;
      } else if (i == static_cast<size_t>(len - 2)) {
        new_bounds.push_back(Range::make_by_min_extent(op->bounds[i]->min, m_size_));
        continue;
      } else {
        new_bounds.push_back(op->bounds[i]);
      }
    }
    return new_bounds;
  }
  Array<Expr> ModifyCShape(const BufferNode *op, int len) {
    Array<Expr> new_shape;
    for (size_t i = 0; i < op->shape.size(); ++i) {
      if (i == static_cast<size_t>(len - 1)) {
        CHECK(n_size_.defined());
        new_shape.push_back(n_size_);
        continue;
      } else if (i == static_cast<size_t>(len - 2)) {
        CHECK(m_size_.defined());
        new_shape.push_back(m_size_);
        continue;
      } else {
        new_shape.push_back(op->shape[i]);
      }
    }
    return new_shape;
  }

 private:
  TensorCoreInfo info_;
  Expr m_size_;
  Expr n_size_;
};

class ModifyTheLocalOffset : public IRMutator {
 public:
  explicit ModifyTheLocalOffset(TensorCoreInfo &info) : info_(info) {}

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->is_intrinsic(air::ir::intrinsic::tvm_fill_fragment)) {
      CHECK_EQ(op->args.size(), 6U);
      Array<Expr> args = op->args;
      auto a0 = args[0].as<Variable>();
      CHECK(a0);
      std::string cur_tensor_name = a0->name_hint;
      std::string cur_base_name = SimplifyName(cur_tensor_name);

      auto a4 = args[4];
      a4 = ChangeTensorIndex(a4, cur_base_name);

      Array<Expr> new_args;
      for (unsigned int i = 0; i < args.size(); ++i) {
        if (i == 4) {
          new_args.push_back(a4);
          continue;
        }
        new_args.push_back(args[i]);
      }
      return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);

    } else if (op->is_intrinsic(air::ir::intrinsic::tvm_load_matrix_sync)) {
      CHECK_EQ(op->args.size(), 8U);
      Array<Expr> args = op->args;
      auto a0 = args[0].as<Variable>();
      CHECK(a0);
      std::string cur_tensor_name = a0->name_hint;
      std::string cur_base_name = SimplifyName(cur_tensor_name);
      auto a4 = args[4];
      a4 = ChangeTensorIndex(a4, cur_base_name);

      Array<Expr> new_args;
      for (unsigned int i = 0; i < args.size(); ++i) {
        if (i == 4) {
          new_args.push_back(a4);
          continue;
        }
        new_args.push_back(args[i]);
      }
      return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);

    } else if (op->is_intrinsic(air::ir::intrinsic::tvm_store_matrix_sync)) {
      CHECK_EQ(op->args.size(), 8U);
      Array<Expr> args = op->args;
      auto a0 = args[0].as<Variable>();
      CHECK(a0);
      std::string cur_tensor_name = a0->name_hint;
      std::string cur_base_name = SimplifyName(cur_tensor_name);
      auto a4 = args[4];
      a4 = ChangeTensorIndex(a4, cur_base_name);

      Array<Expr> new_args;
      for (unsigned int i = 0; i < args.size(); ++i) {
        if (i == 4) {
          new_args.push_back(a4);
          continue;
        }
        new_args.push_back(args[i]);
      }
      return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);

    } else if (op->is_intrinsic(air::ir::intrinsic::tvm_mma_sync)) {
      CHECK_EQ(op->args.size(), 8U);
      Array<Expr> args = op->args;
      auto a0 = args[0].as<Variable>();
      CHECK(a0);
      std::string a0_name = a0->name_hint;
      std::string a0_base_name = SimplifyName(a0_name);
      auto a1 = args[1];
      a1 = ChangeTensorIndex(a1, a0_base_name);

      auto a2 = args[2].as<Variable>();
      CHECK(a2);
      std::string a2_name = a2->name_hint;
      std::string a2_base_name = SimplifyName(a2_name);
      auto a3 = args[3];
      a3 = ChangeTensorIndex(a3, a2_base_name);

      auto a4 = args[4].as<Variable>();
      CHECK(a4);
      std::string a4_name = a4->name_hint;
      std::string a4_base_name = SimplifyName(a4_name);
      auto a5 = args[5];
      a5 = ChangeTensorIndex(a5, a4_base_name);

      auto a6 = args[6].as<Variable>();
      CHECK(a6);
      std::string a6_name = a6->name_hint;
      std::string a6_base_name = SimplifyName(a6_name);
      auto a7 = args[7];
      a7 = ChangeTensorIndex(a7, a6_base_name);

      Array<Expr> new_args;
      new_args.push_back(args[0]);
      new_args.push_back(a1);
      new_args.push_back(args[2]);
      new_args.push_back(a3);
      new_args.push_back(args[4]);
      new_args.push_back(a5);
      new_args.push_back(args[6]);
      new_args.push_back(a7);

      return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

 private:
  Expr ChangeTensorIndex(Expr e, std::string name) {
    auto matrix_map_info = info_.matrix_abc_;
    if ((matrix_map_info[name] == MATRIX_A) || (matrix_map_info[name] == MATRIX_B)) {
      if (e.as<Variable>()) {
        return e;
      }

      if (e.as<IntImm>()) {
        CHECK_EQ(e.as<IntImm>()->value, 0) << "A B matrix index should be 0";
        return e;
      }

      if (e.as<Mul>()) {
        auto mul = e.as<Mul>();
        auto a = mul->a;
        auto b = mul->b;
        CHECK(b.as<IntImm>()) << "A B matrix index format error";
        return a;
      }
      CHECK(false) << "A B matrix index error";

    } else if (matrix_map_info[name] == MATRIX_C) {
      if (e.as<Variable>()) {
        return e;
      }

      if (e.as<IntImm>()) {
        CHECK_EQ(e.as<IntImm>()->value, 0) << "C matrix index should be 0";
        return e;
      }

      if (e.as<Add>()) {
        auto add = e.as<Add>();
        auto a = add->a;
        auto b = add->b;
        CHECK(a.as<Mul>()) << "A B matrix index format error";
        auto mul = a.as<Mul>();
        auto mul_a = mul->a;
        auto mul_b = mul->b;
        CHECK(mul_b.as<IntImm>()) << "C matrix index mul format error";
        CHECK(info_.fragment_n_.defined());
        a = Mul::make(mul_a, info_.fragment_n_);
        if (b.as<Mul>()) {
          auto b_mul = b.as<Mul>();
          auto b_a = b_mul->a;
          auto b_b = b_mul->b;
          CHECK(b_b.as<IntImm>()) << "C matrix index b_b mul format error";
          b = b_a;
        }
        e = Add::make(a, b);
        return e;
      }

      if (e.as<Mul>()) {
        auto mul = e.as<Mul>();
        auto a = mul->a;
        auto b = mul->b;
        CHECK(b.as<IntImm>()) << "C matrix index format error";
        return a;
      }
      CHECK(false) << "C matrix index error";
    }

    return e;
  }
  TensorCoreInfo info_;
};

class DeleteUselessAttr : public IRMutator {
 public:
  explicit DeleteUselessAttr() {}
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if ((op->attr_key == GMREAD_FLAG) || (op->attr_key == MATRIX_A) || (op->attr_key == MATRIX_B) ||
        (op->attr_key == MMA_C) || (op->attr_key == MMA_SYNC) || (op->attr_key == FRAGMENT_A) ||
        (op->attr_key == FRAGMENT_B)) {
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }
};

Stmt EmitForTensorCoreDesignOne(Stmt stmt, TensorCoreInfo &info) {
  AdaptCastDesignOne adapt(info);
  stmt = adapt.Mutate(stmt);
  return stmt;
}

Stmt EmitForTensorCore(Stmt stmt, TensorCoreInfo &info) {
  stmt = ModifySizeOfLocal(info).Mutate(stmt);
  stmt = ModifyTheLocalOffset(info).Mutate(stmt);
  stmt = DeleteUselessAttr().Mutate(stmt);

  return stmt;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
