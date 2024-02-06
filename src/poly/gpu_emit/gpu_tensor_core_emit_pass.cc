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
#include <algorithm>
#include "build_module.h"

namespace akg {
namespace ir {
namespace poly {

class CheckTensorCoreValid : public IRVisitor {
 public:
  explicit CheckTensorCoreValid() {}
  using IRVisitor::Visit_;

  void Visit_(const AttrStmt *op) {
    auto key = op->attr_key;
    if (key == WARP_MARKER) {
      warp_marker_ = true;
    }
    return IRVisitor::Visit_(op);
  }

  bool IsValid() { return warp_marker_; }

 private:
  bool warp_marker_{false};
};

Array<Expr> GetTileSize(TensorCoreInfo &tensor_core_info, const std::string &name) {
  auto it = tensor_core_info.matrix_abc_.find(name);
  auto it2 = tensor_core_info.matrix_major_.find(name);
  CHECK(it != tensor_core_info.matrix_abc_.end() && it2 != tensor_core_info.matrix_major_.end())
    << "Cannot find matrix info for " << name;
  Expr size0 = make_const(Int(INT_32), 16);
  Expr size1 = make_const(Int(INT_32), 16);
  if (it->second == MMA_A && it2->second == COL_MAJOR) {
    size0 = make_const(Int(INT_32), tensor_core_info.warp_tile_.k);
    size1 = make_const(Int(INT_32), tensor_core_info.warp_tile_.m);
  }
  if (it->second == MMA_A && it2->second == ROW_MAJOR) {
    size0 = make_const(Int(INT_32), tensor_core_info.warp_tile_.m);
    size1 = make_const(Int(INT_32), tensor_core_info.warp_tile_.k);
  }
  if (it->second == MMA_B && it2->second == ROW_MAJOR) {
    size0 = make_const(Int(INT_32), tensor_core_info.warp_tile_.k);
    size1 = make_const(Int(INT_32), tensor_core_info.warp_tile_.n);
  }
  if (it->second == MMA_B && it2->second == COL_MAJOR) {
    size0 = make_const(Int(INT_32), tensor_core_info.warp_tile_.n);
    size1 = make_const(Int(INT_32), tensor_core_info.warp_tile_.k);
  }

  if (it->second == MATRIX_C || it->second == MATRIX_ELSE) {
    size0 = make_const(Int(INT_32), tensor_core_info.warp_tile_.m);
    size1 = make_const(Int(INT_32), tensor_core_info.warp_tile_.n);
  }
  Array<Expr> tile_size = {size0, size1};
  return tile_size;
}

class DeleteUselessFor : public air::ir::IRMutator {
 public:
  explicit DeleteUselessFor() {}
  ~DeleteUselessFor() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) {
    for_iters_.push_back(op->loop_var.get());
    Stmt stmt = IRMutator::Mutate_(op, s);
    for_iters_.pop_back();
    return stmt.as<For>()->body;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == air::ir::attr::buffer_bind_scope) {
      Array<NodeRef> arr = Downcast<Array<NodeRef>>(op->node);
      CHECK_EQ(arr.size(), 2U);
      const BufferNode *buffer = arr[0].as<BufferNode>();
      const TensorNode *tensor = arr[1].as<TensorNode>();
      CHECK(buffer && tensor);
      auto e = buffer->elem_offset;
      Expr ret = this->Mutate(e);
      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      buffer_node->data = buffer->data;
      buffer_node->name = buffer->name;
      buffer_node->scope = buffer->scope;
      buffer_node->dtype = buffer->dtype;
      buffer_node->strides = buffer->strides;
      buffer_node->shape = buffer->shape;
      buffer_node->data_alignment = buffer->data_alignment;
      buffer_node->elem_offset = ret;
      buffer_node->offset_factor = buffer->offset_factor;

      Buffer buffer_new(buffer_node);
      Array<NodeRef> node = {buffer_new, arr[1]};

      auto value = this->Mutate(op->value);
      auto body = this->Mutate(op->body);

      return AttrStmt::make(node, op->attr_key, value, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const EQ *op, const Expr &e) override {
    Expr a = op->a;
    Expr b = op->b;
    auto for_var = a.as<Variable>();
    if (for_var != nullptr) {
      for (auto &i : for_iters_) {
        if (i == for_var) {
          return EQ::make(b, b);
        }
      }
    }
    return e;
  }
  
  Expr Mutate_(const Variable *op, const Expr &e) {
    bool be_zero = false;
    for (auto &i : for_iters_) {
      if (i == op) {
        be_zero = true;
        break;
      }
    }

    if (be_zero) {
      return make_const(Int(INT_32), 0);
    }

    return e;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->is_intrinsic(air::ir::intrinsic::tvm_fill_fragment)) {
      CHECK_EQ(op->args.size(), 6U);
      return DeleteUselessForIndex(op, e);
    } else if (op->is_intrinsic(air::ir::intrinsic::tvm_load_matrix_sync)) {
      CHECK_EQ(op->args.size(), 8U);
      return DeleteUselessForIndex(op, e);

    } else if (op->is_intrinsic(air::ir::intrinsic::tvm_store_matrix_sync)) {
      CHECK_EQ(op->args.size(), 8U);
      return DeleteUselessForIndex(op, e);

    } else if (op->is_intrinsic(air::ir::intrinsic::tvm_mma_sync)) {
      CHECK_EQ(op->args.size(), 8U);
      return DeleteUselessForIndex(op, e);
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Expr DeleteUselessForIndex(const Call *op, const Expr &e) {
    Array<Expr> args = op->args;
    for (unsigned int i = 0; i < args.size(); ++i) {
      args.Set(i, Simplify(this->Mutate(args[i])));
    }
    if (args.same_as(op->args)) {
      return e;
    }
    return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
  }

 private:
  std::vector<const Variable *> for_iters_;
};

struct DataForLoad {
  Expr src;
  Expr stride;
  Expr major;
  const Call *call;
  const Provide *op;
  NodePtr<BufferNode> node;
};

struct DataForStore {
  Expr dst;
  Expr stride;
  const Call *call;
  NodePtr<BufferNode> node;
};

struct DataForFill {
  const Call *call;
  const Provide *op;
  NodePtr<BufferNode> node;
};

struct DataForSync {
  Expr a;
  Expr b;
  Expr c;
  NodePtr<BufferNode> node_a;
  NodePtr<BufferNode> node_b;
  NodePtr<BufferNode> node_c;
};

struct DataForElem {
  Expr a;
  Expr b;
  Expr c;
  NodePtr<BufferNode> node_a;
  NodePtr<BufferNode> node_b;
  NodePtr<BufferNode> node_c;
};


class EmitTensorCoreHelper {
 public:
  struct CompareExpr {
    bool operator()(const Expr &lhs, const Expr &rhs) const { return Compare(lhs, rhs) < 0; }
  };
  EmitTensorCoreHelper(TensorCoreInfo &info, ScopInfo &scop_info) : tensor_core_info_(info), scop_info_(scop_info) {}
  ~EmitTensorCoreHelper(){};

  void SetDataForLoad(Expr src, Expr stride, Expr major, const Call *call, const Provide *op,
                      NodePtr<BufferNode> &node);
  void SetDataForStore(Expr dst, Expr stride, const Call *call, NodePtr<BufferNode> &node);
  void SetDataForFill(const Provide *op, const Call *call, NodePtr<BufferNode> &node);
  void SetDataForSync(Expr a, Expr b, Expr c, NodePtr<BufferNode> &node_a, NodePtr<BufferNode> &node_b,
                      NodePtr<BufferNode> &node_c);
  void SetDataForElem(Expr a, Expr b, Expr c, NodePtr<BufferNode> &node_a, NodePtr<BufferNode> &node_b,
                      NodePtr<BufferNode> &node_c);

  void PrepareDataCore();

  Stmt MakeLoadTransform();
  Stmt MakeStoreTransform();
  Stmt MakeFillTransform();
  Stmt MakeSyncTransform();
  Stmt MakeFragmentElemTransform(Expr op_name);

 private:
  Array<NodeRef> node_;
  Expr tuple_;
  TensorCoreInfo &tensor_core_info_;

  DataForLoad data_for_load_={};
  DataForStore data_for_store_={};
  DataForFill data_for_fill_={};
  DataForSync data_for_sync_={};
  DataForElem data_for_elemwise_={};

  air::ir::TensorKey key_;
  const Call *call_={nullptr};
  NodePtr<BufferNode> buffer_node_;
  Type data_type_;
  ScopInfo &scop_info_;
  std::map<Expr, Expr, CompareExpr> fragment_offset_;
};

void EmitTensorCoreHelper::SetDataForLoad(Expr src, Expr stride, Expr major, const Call *call, const Provide *op,
                                          NodePtr<BufferNode> &node) {
  data_for_load_.src = src;
  data_for_load_.stride = stride;
  data_for_load_.major = major;
  data_for_load_.call = call;
  data_for_load_.op = op;
  data_for_load_.node = node;
}
void EmitTensorCoreHelper::SetDataForStore(Expr dst, Expr stride, const Call *call, NodePtr<BufferNode> &node) {
  data_for_store_.dst = dst;
  data_for_store_.stride = stride;
  data_for_store_.call = call;
  data_for_store_.node = node;
}
void EmitTensorCoreHelper::SetDataForFill(const Provide *op, const Call *call, NodePtr<BufferNode> &node) {
  data_for_fill_.call = call;
  data_for_fill_.op = op;
  data_for_fill_.node = node;
}
void EmitTensorCoreHelper::SetDataForSync(Expr a, Expr b, Expr c, NodePtr<BufferNode> &node_a,
                                          NodePtr<BufferNode> &node_b, NodePtr<BufferNode> &node_c) {
  data_for_sync_.a = a;
  data_for_sync_.b = b;
  data_for_sync_.c = c;
  data_for_sync_.node_a = node_a;
  data_for_sync_.node_b = node_b;
  data_for_sync_.node_c = node_c;
}
void EmitTensorCoreHelper::SetDataForElem(Expr a, Expr b, Expr c, NodePtr<BufferNode> &node_a,
                                          NodePtr<BufferNode> &node_b, NodePtr<BufferNode> &node_c) {
  data_for_elemwise_.a = a;
  data_for_elemwise_.b = b;
  data_for_elemwise_.c = c;
  data_for_elemwise_.node_a = node_a;
  data_for_elemwise_.node_b = node_b;
  data_for_elemwise_.node_c = node_c;
}

void EmitTensorCoreHelper::PrepareDataCore() {
  auto it = tensor_core_info_.bounds_.find(key_);
  CHECK(it != tensor_core_info_.bounds_.end());
  Array<Expr> min_bound;
  for (auto i : it->second) {
    min_bound.push_back(i->min);
  }

  CHECK_GE(it->second.size(), 2);
  Array<Expr> shape;
  for (size_t i = 0; i < it->second.size(); ++i) {
    shape.push_back(it->second[i]->extent);
  }

  auto tile_size = GetTileSize(tensor_core_info_, akg::common::GetGlobalName(call_->name));
  tensor_core_info_.min_bounds_[call_->name] = min_bound;

  Array<Expr> strides;
  for (size_t i = 1; i < shape.size(); ++i) {
    Expr stride = IntImm::make(Int(INT_32), 1);
    for (size_t j = shape.size() - 1; j >= i; --j) {
      stride = Mul::make(stride, shape[j]);
    }
    strides.push_back(stride);
  }
  strides.push_back(make_const(Int(INT_32), 1));

  // compute the local offset for fragment
  // example: (cc1, cc2)
  Expr fragment_elem_offset = IntImm::make(Int(INT_32), 0);
  CHECK_EQ(call_->args.size(), min_bound.size());
  for (size_t i = 0; i < min_bound.size(); i++) {
    auto arg = call_->args[i];
    arg = Simplify(arg);
    auto stride_val = strides[i];
    // tile_size[1] is the innermost axis of the tensor.
    // And this axis is used for wmma interface.
    // The fragment offset computing should make a division of the parameter.
    if (i != min_bound.size() - 1) {
      stride_val = Div::make(stride_val, tile_size[1]);
    }
    fragment_elem_offset = Add::make(fragment_elem_offset, Mul::make(stride_val, Sub::make(arg, min_bound[i])));
  }

  Expr elem_offset = IntImm::make(Int(INT_32), 0);
  CHECK_EQ(call_->args.size(), min_bound.size());
  for (size_t i = 0; i < min_bound.size(); i++) {
    auto arg = call_->args[i];
    arg = Simplify(arg);
    elem_offset = Add::make(elem_offset, Mul::make(strides[i], Sub::make(arg, min_bound[i])));
  }

  elem_offset = Simplify(elem_offset);

  // insert the fragment offset information
  fragment_offset_[elem_offset] = fragment_elem_offset;

  auto it2 = tensor_core_info_.matrix_abc_.find(akg::common::GetGlobalName(call_->name));
  CHECK(it2 != tensor_core_info_.matrix_abc_.end()) << "Cannot find matrix info for " << call_->name;
  buffer_node_->data = Variable::make(Handle(), call_->name);
  buffer_node_->name = call_->name;
  std::string name = it2->second;
  if (name == MATRIX_C || name == MATRIX_ELSE) {
    name = MMA_C;
  }
  buffer_node_->scope = "wmma." + name;
  buffer_node_->dtype = data_type_;
  buffer_node_->strides = strides;
  buffer_node_->shape = shape;
  buffer_node_->data_alignment = 1;
  buffer_node_->elem_offset = Simplify(elem_offset);
  buffer_node_->offset_factor = 1;
  Buffer buffer(buffer_node_);

  NodePtr<TensorNode> tensor_node = make_node<TensorNode>();
  tensor_node->value_index = key_.value_index;
  tensor_node->op = Downcast<Operation>(key_.f);
  tensor_node->shape = shape;
  tensor_node->dtype = data_type_;
  Tensor tensor(tensor_node);

  Array<Expr> args;
  for (size_t i = 0; i < call_->args.size(); ++i) {
    auto arg = call_->args[i];
    arg = Simplify(arg);
    args.push_back(arg);
    args.push_back(shape[i]);
  }
  tuple_ = Call::make(Handle(), air::ir::intrinsic::tvm_tuple, args, Call::Intrinsic);
  node_ = {buffer, tensor};
}

Stmt EmitTensorCoreHelper::MakeLoadTransform() {
  key_ = air::ir::TensorKey{data_for_load_.op->func, data_for_load_.op->value_index};
  call_ = data_for_load_.call;
  buffer_node_ = data_for_load_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(Call::make(
    Handle(), air::ir::intrinsic::tvm_load_matrix_sync,
    {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n, tensor_core_info_.warp_tile_.k,
     Simplify(fragment_offset_[buffer->elem_offset]), data_for_load_.src, data_for_load_.stride, data_for_load_.major},
    Call::Intrinsic));
  fragment_offset_.clear();
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeStoreTransform() {
  key_ = air::ir::TensorKey{data_for_store_.call->func, data_for_store_.call->value_index};
  call_ = data_for_store_.call;
  buffer_node_ = data_for_store_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(Call::make(
    Handle(), air::ir::intrinsic::tvm_store_matrix_sync,
    {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n, tensor_core_info_.warp_tile_.k,
     fragment_offset_[buffer->elem_offset], data_for_store_.dst, data_for_store_.stride, StringImm::make(ROW_MAJOR)},
    Call::Intrinsic));
  fragment_offset_.clear();
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeFillTransform() {
  key_ = air::ir::TensorKey{data_for_fill_.call->func, data_for_fill_.call->value_index};
  call_ = data_for_fill_.call;
  buffer_node_ = data_for_fill_.node;
  data_type_ = call_->type;

  PrepareDataCore();
  Buffer buffer = Downcast<Buffer>(node_[0]);
  Stmt stmt = Evaluate::make(
    Call::make(Handle(), air::ir::intrinsic::tvm_fill_fragment,
               {buffer->data, tensor_core_info_.warp_tile_.m, tensor_core_info_.warp_tile_.n,
                tensor_core_info_.warp_tile_.k, fragment_offset_[buffer->elem_offset], data_for_fill_.op->value},
               Call::Intrinsic));
  fragment_offset_.clear();
  return AttrStmt::make(node_, "buffer_bind_scope", tuple_, stmt);
}

Stmt EmitTensorCoreHelper::MakeSyncTransform() {
  bool is_cast = false;
  if (data_for_sync_.a.as<Call>()) {
    auto call_a = data_for_sync_.a.as<Call>();
    key_ = air::ir::TensorKey{call_a->func, call_a->value_index};
    call_ = call_a;
    buffer_node_ = data_for_sync_.node_a;
    data_type_ = call_->type;
    is_cast = true;
  } else if (data_for_sync_.a.as<Cast>()) {
    auto cast_a = data_for_sync_.a.as<Cast>();
    auto call_a = cast_a->value.as<Call>();
    CHECK(call_a);
    key_ = air::ir::TensorKey{call_a->func, call_a->value_index};
    call_ = call_a;
    buffer_node_ = data_for_sync_.node_a;
    data_type_ = call_->type;
    is_cast = true;
  }

  PrepareDataCore();

  auto tuple_a = tuple_;
  auto node_a = node_;

  if (data_for_sync_.b.as<Call>()) {
    auto call_b = data_for_sync_.b.as<Call>();
    key_ = air::ir::TensorKey{call_b->func, call_b->value_index};
    call_ = call_b;
    buffer_node_ = data_for_sync_.node_b;
    data_type_ = call_->type;
    is_cast = false;
  } else if (data_for_sync_.b.as<Cast>()) {
    auto cast_b = data_for_sync_.b.as<Cast>();
    auto call_b = cast_b->value.as<Call>();
    CHECK(call_b);
    key_ = air::ir::TensorKey{call_b->func, call_b->value_index};
    call_ = call_b;
    buffer_node_ = data_for_sync_.node_b;
    data_type_ = call_->type;
    is_cast = true;
  }

  PrepareDataCore();

  auto tuple_b = tuple_;
  auto node_b = node_;

  auto call_c = data_for_sync_.c.as<Call>();
  CHECK(call_c);
  key_ = air::ir::TensorKey{call_c->func, call_c->value_index};
  call_ = call_c;
  buffer_node_ = data_for_sync_.node_c;
  data_type_ = call_->type;

  PrepareDataCore();

  auto tuple_c = tuple_;
  auto node_c = node_;

  Buffer buffer_a(data_for_sync_.node_a);
  Buffer buffer_b(data_for_sync_.node_b);
  Buffer buffer = Downcast<Buffer>(node_c[0]);

  Stmt stmt = Evaluate::make(Call::make(
    Handle(), air::ir::intrinsic::tvm_mma_sync,
    {buffer->data, fragment_offset_[buffer->elem_offset], buffer_a->data, fragment_offset_[buffer_a->elem_offset],
     buffer_b->data, fragment_offset_[buffer_b->elem_offset], buffer->data, fragment_offset_[buffer->elem_offset]},
    Call::Intrinsic));
  fragment_offset_.clear();
  stmt = AttrStmt::make(node_c, "buffer_bind_scope", tuple_c, stmt);
  stmt = AttrStmt::make(node_b, "buffer_bind_scope", tuple_b, stmt);
  stmt = AttrStmt::make(node_a, "buffer_bind_scope", tuple_a, stmt);

  std::string cast_mode = CAST_MODE_1;
  if (is_cast) {
    stmt = AttrStmt::make(Expr("INFO"), CAST_FLAG, StringImm::make(cast_mode), stmt);
  }

  return stmt;
}

Stmt EmitTensorCoreHelper::MakeFragmentElemTransform(Expr op_name) {
  auto call_a = data_for_elemwise_.a.as<Call>();
  key_ = air::ir::TensorKey{call_a->func, call_a->value_index};
  call_ = call_a;
  buffer_node_ = data_for_elemwise_.node_a;
  data_type_ = call_->type;
  
  PrepareDataCore();

  auto tuple_a = tuple_;
  auto node_a = node_;

  auto call_c = data_for_elemwise_.c.as<Call>();
  CHECK(call_c);
  key_ = air::ir::TensorKey{call_c->func, call_c->value_index};
  call_ = call_c;
  buffer_node_ = data_for_elemwise_.node_c;
  data_type_ = call_->type;

  PrepareDataCore();

  auto tuple_c = tuple_;
  auto node_c = node_;

  Buffer buffer_a(data_for_elemwise_.node_a);
  Buffer buffer = Downcast<Buffer>(node_c[0]);

  auto call_b = data_for_elemwise_.b.as<Call>();
  if (call_b) {
    key_ = air::ir::TensorKey{call_b->func, call_b->value_index};
    call_ = call_b;
    buffer_node_ = data_for_elemwise_.node_b;
    data_type_ = call_->type;

    PrepareDataCore();

    auto tuple_b = tuple_;
    auto node_b = node_;
    Buffer buffer_b(data_for_elemwise_.node_b);
    Stmt stmt = Evaluate::make(
        Call::make(Handle(), air::ir::intrinsic::akg_fragment_elem,
                  {buffer->data, fragment_offset_[buffer->elem_offset], buffer_a->data,
                   fragment_offset_[buffer_a->elem_offset], buffer_b->data, fragment_offset_[buffer_b->elem_offset],
                   op_name}, Call::Intrinsic));
    fragment_offset_.clear();
    stmt = AttrStmt::make(node_c, "buffer_bind_scope", tuple_c, stmt);
    stmt = AttrStmt::make(node_b, "buffer_bind_scope", tuple_b, stmt);
    stmt = AttrStmt::make(node_a, "buffer_bind_scope", tuple_a, stmt);
    return stmt;
  } else {
    Stmt stmt = Evaluate::make(
        Call::make(Handle(), air::ir::intrinsic::akg_fragment_elem,
                  {buffer->data, fragment_offset_[buffer->elem_offset], buffer_a->data,
                   fragment_offset_[buffer_a->elem_offset], Expr(data_for_elemwise_.b),
                   op_name}, Call::Intrinsic));
    fragment_offset_.clear();
    stmt = AttrStmt::make(node_c, "buffer_bind_scope", tuple_c, stmt);
    stmt = AttrStmt::make(node_a, "buffer_bind_scope", tuple_a, stmt);
    return stmt;
  }
}

class AddMmaAttrFlag : public air::ir::IRMutator {
 public:
  explicit AddMmaAttrFlag(TensorCoreInfo &t) : tt(t) {}
  ~AddMmaAttrFlag() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (op->attr_key == air::ir::attr::realize_scope) {
      auto node = op->node.as<OperationNode>();
      if (node != nullptr) {
        if (!tt.frag_reg_.count(node->name)) {
          return stmt;
        }

        auto it = tt.matrix_abc_.find(akg::common::GetGlobalName(node->name));
        CHECK(it != tt.matrix_abc_.end()) << "Cannot find matrix info for " << node->name;
        std::string name = it->second;
        if (name == MATRIX_C || name == MATRIX_ELSE) {
          name = MMA_C;
        }

        auto matrix_abc = "wmma." + name;
        Stmt body = Mutate(op->body);
        return AttrStmt::make(op->node, op->attr_key, matrix_abc, body);
      }
    }
    return stmt;
  }

 private:
  TensorCoreInfo tt;
};

class LocalTensorAnalyser : public IRVisitor {
 public:
  explicit LocalTensorAnalyser(TensorCoreInfo &info, ScopInfo &scop_info)
      : matrix_abc_(info.matrix_abc_), matrix_major_(info.matrix_major_), frag_reg_(info.frag_reg_) {
    for (auto kv : scop_info.user_config_.GetOriginBind()) {
      BufferInfo bi;
      bi.name = kv.second->name;
      bi.dtype = kv.second->dtype;
      bi.external = true;
      buf_map_[air::ir::TensorKey{kv.first->op, kv.first->value_index}] = bi;
    }
  }
  using IRVisitor::Visit_;

  void Visit_(const Provide *op) final {
    IRVisitor::Visit_(op);
    air::ir::TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key.f;
    const BufferInfo &bi = it->second;
    CHECK(!bi.released) << "Read a buffer that is already out of scope";

    std::vector<int> tile_size;
    if (frag_reg_.count(bi.name)) {
      Expr dst = Call::make(bi.dtype, bi.name, op->args, Call::Halide, op->func, 0);
      frag_load_.insert(std::make_pair(op, dst));
    }

    const Call *value = op->value.as<Call>();

    if (value != nullptr && frag_reg_.count(value->name)) {
      Expr dst = Call::make(bi.dtype, bi.name, op->args, Call::Halide, op->func, 0);
      frag_store_.insert(std::make_pair(op, dst));
    }
  }

  void Visit_(const Realize *op) final {
    air::ir::TensorKey key{op->func, op->value_index};
    if (buf_map_.count(key)) {
      CHECK(buf_map_.at(key).external);
      Visit(op->body);
    } else {
      BufferInfo bi;
      bi.name = key.GetName();
      bi.dtype = op->type;

      buf_map_[key] = bi;
      Visit(op->body);
      buf_map_[key].released = true;
    }
  }

 private:
  struct BufferInfo {
    std::string name;
    Type dtype;
    bool external{false};
    bool released{false};
  };

  std::unordered_map<air::ir::TensorKey, BufferInfo> buf_map_;
  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_map<std::string, std::string> matrix_major_;
  std::set<std::string> frag_reg_;

 public:
  std::unordered_map<const Provide *, Expr> frag_load_;
  std::unordered_map<const Provide *, Expr> frag_store_;
};

class ExprUsedVarsVisitor : public IRVisitor {
 public:
  explicit ExprUsedVarsVisitor() {}

  void Visit_(const Variable *op) {
    if (op->name_hint != THREAD_IDX_X) {
      vars_.push_back(op);
    }
  }

  std::vector<const Variable *> Run(Expr e) {
    this->Visit(e);
    return vars_;
  }

 private:
  std::vector<const Variable *> vars_;
};

class ModifyTheLocalOffset : public IRMutator {
 public:
  explicit ModifyTheLocalOffset(TensorCoreInfo &info, ScopInfo &scop_info, LocalTensorAnalyser &local_analyser)
      : tensor_core_info_(info),
        scop_info_(scop_info),
        frag_load_(local_analyser.frag_load_),
        frag_store_(local_analyser.frag_store_) {}

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    auto it2 = frag_load_.find(op);
    if (it2 != frag_load_.end()) {
      if (op->value.as<FloatImm>() != nullptr || op->value.as<IntImm>() != nullptr) {
        Stmt stmt = ModifyTheOpIndexOfLoadFill(op, GetFragmentIndex(op));
        // The provide op is a new op.
        frag_load_new_.insert(stmt.as<Provide>());
        return stmt;
      }
      const Call *value = op->value.as<Call>();
      if (value != nullptr) {
        Stmt stmt = ModifyTheOpIndexOfLoadFill(op, GetFragmentIndex(op));
        frag_load_new_.insert(stmt.as<Provide>());
        return stmt;
      }

      Stmt stmt = ModifyTheOpIndexOfSync(op, GetFragmentIndex(op));
      frag_load_new_.insert(stmt.as<Provide>());
      return stmt;
    }

    auto it3 = frag_store_.find(op);
    if (it3 != frag_store_.end()) {
      auto value = op->value;
      auto call = value.as<Call>();
      CHECK(call);
      Stmt stmt;
      if (scop_info_.user_config_.GetEnableConvTensorCore()) {
        stmt = ModifyTheOpIndexOfStore(op, GetFragmentIndexConv(call));
      } else {
        stmt = ModifyTheOpIndexOfStore(op, GetFragmentIndex(call));
      }
      frag_store_new_.insert(stmt.as<Provide>());
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) {
    vec_for_vars_.push_back(op);
    Stmt stmt = IRMutator::Mutate_(op, s);
    vec_for_vars_.pop_back();
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) {
    if (sync_value_mod) {
      Array<Expr> real_index;
      if (scop_info_.user_config_.GetEnableConvTensorCore()) {
        real_index = GetFragmentIndexConv(op);
      } else {
        real_index = GetFragmentIndex(op);
      }
      return Call::make(op->type, op->name, real_index, op->call_type, op->func, op->value_index);
    }

    return IRMutator::Mutate_(op, e);
  }

  Array<Expr> GetFragmentIndex(const Provide *op) {
    auto call = frag_load_[op].as<Call>();
    CHECK(call);
    if (scop_info_.user_config_.GetEnableConvTensorCore()) {
      return GetFragmentIndexConv(call);
    }
    return GetFragmentIndex(call);
  }

  Expr GetCurrentIndex(const int outer_size, const std::vector<const Variable *> &used_vars) {
    Expr e = make_const(Int(INT_32), 0);
    int size = used_vars.size();
    for (int i = 0; i < outer_size; i++) {
      auto u = used_vars[i];
      Expr temp = Expr(GetObjPtr(u));
      for (int j = i + 1; j < size - 1; j++) {
        // The last var is used for wmma interface.
        // So in this place, the extent of last var is not used.
        temp = Mul::make(temp, FindExtentOfForVar(used_vars[j]));
      }
      e = Add::make(e, temp);
    }
    return e;
  }

  Array<Expr> GetFragmentIndex(const Call *call) {
    auto args = call->args;
    Array<Expr> new_index;
    for (auto &arg : args) {
      auto used_vars = ExprUsedVarsVisitor().Run(arg);
      int size = used_vars.size();
      // The last var is used for wmma interface.
      // So in this place, delete the related var infor.
      // example: (cc1*16+cc2) cc2 is the var using for wmma interface
      // cc1 is the var after warping
      // This index will be converted to cc1
      // If the index is without cc1, this will be converted to 0
      Expr e = GetCurrentIndex(size - 1, used_vars);
      new_index.push_back(e);
    }
    return new_index;
  }

  // for input_1 and output, the layout is n h w c
  // for input_2, the layout is o kh kw c
  // the h and w dimention is useful for the fragment compute.
  // the kh and kw dimension is splitted outer.
  Array<Expr> GetFragmentIndexConv(const Call *call) {
    auto args = call->args;
    Array<Expr> new_index;
    int len = args.size();
    constexpr auto H_DIMENSION_INDEX = 1;
    constexpr auto W_DIMENSION_INDEX = 2;
    for (int i = 0; i < len; i++) {
      auto used_vars = ExprUsedVarsVisitor().Run(args[i]);
      int size = used_vars.size();
      // The last var is used for wmma interface.
      // So in this place, delete the related var infor.
      // example: (cc1*16+cc2) cc2 is the var using for wmma interface
      // cc1 is the var after warping
      // This index will be converted to cc1
      // If the index is without cc1, this will be converted to 0
      // The H&W dimensions are not used for warp mapping and wmma interface.
      // So, for H and W dimension, this logic should be disabled.
      int outer_size = size;
      if (i != H_DIMENSION_INDEX && i != W_DIMENSION_INDEX) {
        outer_size -= 1;
      }
      Expr e = GetCurrentIndex(outer_size, used_vars);
      new_index.push_back(e);
    }
    return new_index;
  }

  Stmt ModifyTheOpIndexOfLoadFill(const Provide *op, Array<Expr> real_index) {
    return Provide::make(op->func, op->value_index, op->value, real_index);
  }

  Stmt ModifyTheOpIndexOfStore(const Provide *op, Array<Expr> real_index) {
    auto value = op->value;
    auto call = value.as<Call>();
    CHECK(call);
    value = Call::make(call->type, call->name, real_index, call->call_type, call->func, call->value_index);
    return Provide::make(op->func, op->value_index, value, op->args);
  }

  Stmt ModifyTheOpIndexOfSync(const Provide *op, Array<Expr> real_index) {
    auto value = op->value;
    sync_value_mod = true;
    value = this->Mutate(value);
    sync_value_mod = false;
    return Provide::make(op->func, op->value_index, value, real_index);
  }

  Expr FindExtentOfForVar(const Variable *var) {
    for (auto &v : vec_for_vars_) {
      if (v->loop_var.get() == var) {
        return v->extent;
      }
    }
    return Expr();
  }

  friend class TensorCoreInterfaceEmit;

 private:
  TensorCoreInfo &tensor_core_info_;
  ScopInfo &scop_info_;

  std::unordered_map<const Provide *, Expr> frag_load_;
  std::unordered_map<const Provide *, Expr> frag_store_;
  std::unordered_set<const Provide *> frag_load_new_;
  std::unordered_set<const Provide *> frag_store_new_;
  std::vector<const For *> vec_for_vars_;
  int for_count_{0};
  bool sync_value_mod{false};
};

class TensorCoreInterfaceEmit : public IRMutator {
 public:
  explicit TensorCoreInterfaceEmit(TensorCoreInfo &info, ScopInfo &scop_info, ModifyTheLocalOffset &warp)
      : tensor_core_info_(info),
        scop_info_(scop_info),
        frag_load_(warp.frag_load_new_),
        frag_store_(warp.frag_store_new_) {}

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    auto it2 = frag_load_.find(op);
    if (it2 != frag_load_.end()) {
      if (op->value.as<FloatImm>() != nullptr || op->value.as<IntImm>() != nullptr) {
        for_count_ = DATA_COMPUTE_FOR_DEPTH;
        return EmitFillStmt(stmt);
      }

      const Call *value = op->value.as<Call>();
      if (value != nullptr) {
        for_count_ = DATA_LOAD_STORE_FOR_DEPTH;
        return EmitLoadStmt(stmt);
      }

      if (Mma(stmt)) {
        return EmitSyncStmt(stmt);
      }

      Array<Expr> elemwise = GetBinaryOpExprChildren(op->value);
      if (!elemwise.empty()) {
        for_count_ = DATA_COMPUTE_FOR_DEPTH;
        return EmitFragmentElem(stmt);
      }

      return stmt;
    }

    auto it3 = frag_store_.find(op);
    if (it3 != frag_store_.end()) {
      for_count_ = DATA_LOAD_STORE_FOR_DEPTH;
      return EmitStoreStmt(stmt);
    }

    return IRMutator::Mutate_(op, s);
  }

  bool Mma(Stmt stmt) {
    auto op = stmt.as<Provide>();
    if (op == nullptr) {
      return false;
    }

    auto add_op = op->value.as<Add>();
    if (add_op == nullptr) {
      return false;
    }

    auto tensor_c = add_op->a.as<Call>();
    if (tensor_c == nullptr) {
      return false;
    }

    Type tensor_c_type = tensor_c->type;
    if (tensor_c_type != Float(16) && tensor_c_type != Float(32)) {
      return false;
    }

    auto mul_op = akg::common::SplitCast(add_op->b, tensor_c_type).as<Mul>();
    if (mul_op == nullptr) {
      return false;
    }

    return true;
  }

  Stmt Mutate_(const For *op, const Stmt &s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (for_count_ != 0) {
      for_count_--;
      if (for_count_ == 0) {
        stmt = DeleteUselessFor().Mutate(stmt);
      }
    }
    return stmt;
  }

  Stmt EmitLoadStmt(Stmt stmt) {
    auto op_new = stmt.as<Provide>();
    CHECK(op_new);
    const Call *call_value = op_new->value.as<Call>();
    CHECK(call_value != nullptr) << "Can only load fragment from a buffer";

    auto left_expr = MakeLeftCallFromProvide(op_new);
    auto left_call = left_expr.as<Call>();
    CHECK(left_call != nullptr) << "make right part call failed!";

    auto it = tensor_core_info_.strides_.find(call_value->name);
    CHECK(it != tensor_core_info_.strides_.end()) << "Cannot find stride for " << call_value->name;
    auto strides = it->second;
    CHECK_GE(strides.size(), 2);
    Expr stride = strides[strides.size() - 2];
    // set the stride information for conv operator
    // conv operator matrix a layout is "n h w ic"
    // The wmma interface uses the data of n. So the stride computing
    // should used the axises of h w ic.
    if (scop_info_.user_config_.GetEnableConvTensorCore() &&
        tensor_core_info_.matrix_abc_[akg::common::GetGlobalName(call_value->name)] == MATRIX_A) {
      CHECK_GE(strides.size(), CONV_MATRIXA_DIMENSION);
      stride = strides[strides.size() - CONV_MATRIXA_DIMENSION];
    }

    std::string call_name = op_new->func->func_name();
    Expr src = Call::make(call_value->type, "&", {op_new->value}, Call::Extern);

    Expr matrix_major;
    auto iter2 = tensor_core_info_.matrix_major_.find(akg::common::GetGlobalName(call_name));
    CHECK(iter2 != tensor_core_info_.matrix_major_.end()) << "Can not determine matrix major for " << call_name;
    if (iter2->second == COL_MAJOR) {
      matrix_major = StringImm::make(COL_MAJOR);
    } else if (iter2->second == ROW_MAJOR) {
      matrix_major = StringImm::make(ROW_MAJOR);
    } else {
      LOG(FATAL) << "invalid matrix major for " << call_name;
    }

    NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
    EmitTensorCoreHelper helper(tensor_core_info_, scop_info_);
    helper.SetDataForLoad(src, stride, matrix_major, left_call, op_new, buffer_node);
    return helper.MakeLoadTransform();
  }

  Stmt EmitSyncStmt(Stmt stmt) {
    auto op = stmt.as<Provide>();
    CHECK(op);

    auto left_expr = MakeLeftCallFromProvide(op);
    Type type = scop_info_.GetDtypeOf(op->func->func_name());
    auto *add = op->value.as<Add>();
    CHECK(add) << "format error of bmm";
    auto mul = akg::common::SplitCast(add->b, type).as<Mul>();
    CHECK(mul) << "format error of bmm";

    auto load_a_expr = akg::common::SplitCast(mul->a, type);
    auto load_b_expr = akg::common::SplitCast(mul->b, type);

    Expr a = load_a_expr;
    Expr b = load_b_expr;
    Expr c = left_expr;

    NodePtr<BufferNode> buffer_node_a = make_node<BufferNode>();
    NodePtr<BufferNode> buffer_node_b = make_node<BufferNode>();
    NodePtr<BufferNode> buffer_node_c = make_node<BufferNode>();

    EmitTensorCoreHelper helper(tensor_core_info_, scop_info_);
    helper.SetDataForSync(a, b, c, buffer_node_a, buffer_node_b, buffer_node_c);
    return helper.MakeSyncTransform();
  }

  Stmt EmitFillStmt(Stmt stmt) {
    auto op = stmt.as<Provide>();
    auto left_expr = MakeLeftCallFromProvide(op);
    auto left_call = left_expr.as<Call>();
    CHECK(left_call != nullptr) << "make right part call failed";

    if (op->value.as<FloatImm>() != nullptr || op->value.as<IntImm>() != nullptr) {
      NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
      EmitTensorCoreHelper helper(tensor_core_info_, scop_info_);
      helper.SetDataForFill(op, left_call, buffer_node);
      return helper.MakeFillTransform();
    } else {
      CHECK(false) << "mma init stmt format error";
    }
    return Stmt();
  }

  Stmt EmitStoreStmt(Stmt stmt) {
    auto op = stmt.as<Provide>();
    CHECK(op);

    auto lh_expr = MakeLeftCallFromProvide(op);
    auto lh_call = lh_expr.as<Call>();
    CHECK(lh_call != nullptr) << "make right part call failed!";

    auto it = tensor_core_info_.strides_.find(lh_call->name);
    CHECK(it != tensor_core_info_.strides_.end()) << "Cannot find stride for " << lh_call->name;
    auto strides = it->second;
    CHECK_GE(strides.size(), 2);
    Expr stride = strides[strides.size() - 2];

    // set the stride information for conv operator
    // conv operator output layout is "n h w o"
    // The wmma interface uses the data of n. So the stride computing
    // should used the axises of h w o.
    if (scop_info_.user_config_.GetEnableConvTensorCore()) {
      CHECK_GE(strides.size(), CONV_OUTPUT_DIMENSION);
      stride = strides[strides.size() - CONV_OUTPUT_DIMENSION];
    }

    Expr dst = lh_expr;
    dst = Call::make(Handle(), "&", {dst}, Call::Extern);

    auto call = op->value.as<Call>();
    NodePtr<BufferNode> buffer_node = make_node<BufferNode>();
    EmitTensorCoreHelper helper(tensor_core_info_, scop_info_);
    helper.SetDataForStore(dst, stride, call, buffer_node);
    return helper.MakeStoreTransform();
  }

  Stmt EmitFragmentElem(Stmt stmt) {
    auto op = stmt.as<Provide>();
    CHECK(op);

    auto elem = GetBinaryOpExprChildren(op->value);
    Expr op_name = GetBinaryOpName(op->value);

    Expr a = elem[0];
    Expr b = elem[1];
    auto left_expr = MakeLeftCallFromProvide(op);
    Expr c = left_expr;

    NodePtr<BufferNode> buffer_node_a = make_node<BufferNode>();
    NodePtr<BufferNode> buffer_node_b = make_node<BufferNode>();
    NodePtr<BufferNode> buffer_node_c = make_node<BufferNode>();

    EmitTensorCoreHelper helper(tensor_core_info_, scop_info_);
    helper.SetDataForElem(a, b, c, buffer_node_a, buffer_node_b, buffer_node_c);
    return helper.MakeFragmentElemTransform(op_name);
  }

  Expr MakeLeftCallFromProvide(const Provide *op) {
    std::string name = op->func->func_name();
    Type type = scop_info_.GetDtypeOf(name);
    Expr dst = Call::make(type, name, op->args, Call::Halide, op->func, 0);
    return dst;
  }

 private:
  TensorCoreInfo &tensor_core_info_;
  ScopInfo &scop_info_;
  bool load_stmt_{false};
  bool store_stmt_{false};
  bool sync_stmt_{false};
  std::unordered_set<const Provide *> frag_load_;
  std::unordered_set<const Provide *> frag_store_;
  std::stack<const For *> st;
  std::vector<const For *> vec_for_vars_;
  int for_count_{0};
};

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
      cast_tensors_.insert(akg::common::GetGlobalName(a2->name_hint));
      cast_tensors_.insert(akg::common::GetGlobalName(a4->name_hint));
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
    if (IsEndsWith(name, SHARE_SUFFIX) && cast_tensors_.count(akg::common::GetGlobalName(name))) {
      realize_need_cast_shared_.insert(name);
    } else if (IsEndsWith(name, LOCAL_SUFFIX) && cast_tensors_.count(akg::common::GetGlobalName(name))) {
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

class DeleteUselessAttr : public IRMutator {
 public:
  explicit DeleteUselessAttr() {}
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == GMREAD_FLAG) {
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }
};

class AddNoUnrollAttr : public IRMutator {
 public:
  explicit AddNoUnrollAttr() {}
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == CONV_KHKW_OUTER) {
      meet_khkw_outer_ = true;
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) {
    if (!meet_khkw_outer_) {
      return AttrStmt::make(Expr("INFO"), "no_unroll", StringImm::make("no_unroll"), IRMutator::Mutate_(op, s));
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool meet_khkw_outer_{false};
};

Stmt EmitForTensorCoreDesignOne(Stmt stmt, TensorCoreInfo &info) {
  AdaptCastDesignOne adapt(info);
  stmt = adapt.Mutate(stmt);
  return stmt;
}

bool CheckTileValid(Tile tile, TensorCoreInfo &info) {
  std::vector<int> tile_size{tile.m, tile.n, tile.k};
  auto it = find(AKG_TILE_SIZE.begin(), AKG_TILE_SIZE.end(), tile_size);
  if (it != AKG_TILE_SIZE.end()) {
    info.wmma_scope_ = "akg";
    return true;
  }
  it = find(NVCUDA_TILE_SIZE.begin(), NVCUDA_TILE_SIZE.end(), tile_size);
  if (it != NVCUDA_TILE_SIZE.end()) {
    info.wmma_scope_ = "nvcuda";
    return true;
  }
  return false;
}

void PrepareDataForTensorCore(TensorCoreInfo &info, ScopInfo &scop_info) {
  auto binds = scop_info.user_config_.GetBind();

  auto thread_cfg = scop_info.user_config_.GetThreadConfig();
  CHECK(thread_cfg) << "thread config is null";
  int tx = thread_cfg->GetX().second;
  int ty = thread_cfg->GetY().second;
  int tz = thread_cfg->GetZ().second;

  if (scop_info.user_config_.GetEnableOneDimThread()) {
    tx = tx * ty * tz;
    ty = 1;
    tz = 1;
  }

  for (auto i : binds) {
    if (!i.first.defined()) continue;
    if (!i.second.defined()) continue;
    auto t = i.first;
    auto b = i.second;

    std::string name = t->op->name;

    air::ir::TensorKey key{t->op, t->value_index};
    Region bounds;
    if (bounds.empty()) {
      for (auto j : t->shape) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), j));
      }
    }

    info.bounds_[key] = bounds;

    Array<Expr> strides;
    for (size_t i = 1; i < b->shape.size(); ++i) {
      Expr stride = IntImm::make(Int(INT_32), 1);
      for (size_t j = b->shape.size() - 1; j >= i; --j) {
        stride = Mul::make(stride, b->shape[j]);
      }
      strides.push_back(stride);
    }
    strides.push_back(make_const(Int(INT_32), 1));
    info.strides_[name] = strides;
  }

  auto mma = scop_info.analysis_result_.GetMmaMode();
  info.warp_tile_.m = mma.m;
  info.warp_tile_.n = mma.n;
  info.warp_tile_.k = mma.k;

  bool result = CheckTileValid(info.warp_tile_, info);
  CHECK(result) << "tile set is not valid!";

  info.matrix_abc_ = scop_info.analysis_result_.GetMatrixMatmulMap();
  info.matrix_major_ = scop_info.analysis_result_.GetMatrixMatmulMajor();

  for (auto &i : info.matrix_abc_) {
    info.frag_reg_.insert(i.first + LOCAL_SUFFIX);
  }
}

Stmt EmitForTensorCore(Stmt stmt, TensorCoreInfo &info, ScopInfo &scop_info) {
  CheckTensorCoreValid check;
  check.Visit(stmt);
  if (!check.IsValid()) {
    return stmt;
  }
  PrepareDataForTensorCore(info, scop_info);
  stmt = AddMmaAttrFlag(info).Mutate(stmt);
  LocalTensorAnalyser local_analyser(info, scop_info);
  local_analyser.Visit(stmt);

  ModifyTheLocalOffset warp(info, scop_info, local_analyser);
  stmt = warp.Mutate(stmt);

  stmt = TensorCoreInterfaceEmit(info, scop_info, warp).Mutate(stmt);
  stmt = DeleteUselessAttr().Mutate(stmt);

  if (scop_info.user_config_.GetEnableConvTensorCore()) {
    stmt = AddNoUnrollAttr().Mutate(stmt);
  }

  if (scop_info.analysis_result_.GetBatchAxisNumForMatmul()) {
    auto batch_axis_num = scop_info.analysis_result_.GetBatchAxisNumForMatmul();
    stmt = AttrStmt::make(Expr(""), "batch_axis_num", make_const(Int(INT_32), batch_axis_num), stmt);
  }

  // add tensor core plan two attr
  if (scop_info.user_config_.GetEnableTensorCore()) {
    auto tensor_core_mode = StringImm::make("");
    if (scop_info.user_config_.GetEnableTensorCoreUsePoly()) {
      tensor_core_mode = StringImm::make(TENSOR_CORE_MODE_TWO);
      stmt = AttrStmt::make(Expr(""), "pragma_tensor_core", tensor_core_mode, stmt);
      stmt = AttrStmt::make(Expr("INFO"), "wmma_scope", StringImm::make(info.wmma_scope_), stmt);
    } else {
      tensor_core_mode = StringImm::make(TENSOR_CORE_MODE_ONE);
      stmt = AttrStmt::make(Expr(""), "pragma_tensor_core", tensor_core_mode, stmt);
    }
    g_attrs.Set(kPragmaTensorCore, tensor_core_mode);
  }

  return stmt;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
