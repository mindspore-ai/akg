/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef POLY_GPU_ISL_EMITTER_H_
#define POLY_GPU_ISL_EMITTER_H_

#include "isl_emitter.h"

namespace akg {
namespace ir {
namespace poly {
#define TENSOR_CORE_DEV true
/*!
 * IslEmitter for GPU
 */
constexpr auto AKG_ALL_REDUCE = "akg_reduce::ALL_REDUCE";
constexpr auto AKG_X_REDUCE = "akg_reduce::REDUCE2D_X";
constexpr auto AKG_Y_REDUCE = "akg_reduce::REDUCE2D_Y";

constexpr auto MIND_TRICKS_SWIZZLE_MARKER = "mind_trick_swizzle_marker";
constexpr auto MIND_TRICKS_SWIZZLE_PRAGMA = "pragma_swizzle";

// example:
// red_init_SumOp_S_1_0
constexpr auto REDUCE_FLAG_SIZE = 6;
constexpr auto REDUCE_FLAG_TYPE_POS = 2;
constexpr auto REDUCE_FLAG_STMT_PREFIX_POS = 3;
constexpr auto REDUCE_FLAG_STMT_NUM_POS = 4;
constexpr auto REDUCE_FLAG_REDUCE_INDEX = 5;

// example:
// atomic_SumOp
constexpr auto REDUCE_ATOMIC_FLAG_SIZE = 2;
constexpr auto REDUCE_ATOMIC_FLAG = "atomic";
constexpr auto REDUCE_ATOMIC_FLAG_POS = 0;
constexpr auto REDUCE_ATOMIC_FLAG_TYPE_POS = 1;

constexpr auto DEFAULT_TENSOR_INDEX = "[0]";

constexpr auto USELESS_INDEX = "0";
constexpr auto USELESS_SHAPE_SIZE = "1";
constexpr auto SCALAR_TENSOR_PREFIX = "acc_";
constexpr auto SCALAR_KHT_PREFIX = "kahan_t";
constexpr auto SCALAR_KHY_PREFIX = "kahan_y";
constexpr auto SCALAR_KHC_PREFIX = "kahan_c";
constexpr auto SHARED_MEMORY_PREFIX = "__shared__";
constexpr auto SHARED_TENSOR_PREFIX = "red_buf";

constexpr auto REDUCE_LIB_TYPE_ORIGIN = "origin";
constexpr auto REDUCE_LIB_TYPE_PARIS = "paris";
constexpr auto AKG_REDUCE_LIB_SPACE = "akg_reduce";
constexpr auto AKG_REDUCE_LIB_NAME = "AkgReduce";
constexpr auto AKG_KAHAN_LIB_NAME = "AkgKahanAccumulation";
constexpr auto PARIS_REDUCE_LIB_SPACE = "paris_reduce";
constexpr auto PARIS_REDUCE_LIB_NAME = "ParisReduce";
constexpr auto AKG_REDUCE_RETURN_NAME = "AkgAtomicReturn";
constexpr auto PARIS_REDUCE_RETURN_NAME = "ParisReturn";
constexpr auto REDUCE_LIB_TYPE_FLAG = "reduceLibType";

constexpr auto MEM_TYPE_SHARED = "shared";
constexpr auto MEM_TYPE_LOCAL = "local";

// add for one dimension mapping
constexpr auto ORIGIN_THREAD_DIM_X = "bind_thread_x";

// add for tensor core
constexpr auto MMA_A = "matrix_a";
constexpr auto MMA_B = "matrix_b";
constexpr auto MMA_C = "accumulator";
constexpr auto MMA_SYNC = "matrix_sync";
constexpr auto MMA_PREFIX = "matrix_";
constexpr auto MMA_FILL_STMT_SERIAL = 2;
constexpr auto MMA_SYNC_STMT_SERIAL = 1;
constexpr auto ENABLE_SCHEME_TWO = "EnableSchemeTwo";
constexpr auto CAST_FLAG = "CAST";
constexpr auto CAST_MODE_1 = "mode1";
constexpr auto GMREAD_FLAG = "GMRead";
constexpr auto SHARED_MEM_PROMOTED_COMPLETE = "shared_mem_promoted_complete";
constexpr auto FRAGMENT_A = "fragment_a";
constexpr auto FRAGMENT_B = "fragment_b";
constexpr auto FRAGMENT_C = "fragment_c";

std::string SimplifyName(std::string input);
constexpr auto FOR_INFO_COLLECT_DEPTH = 3;
constexpr auto LOCAL_INDEX_POS = 4;
constexpr auto TENSOR_CORE_MODE_ONE = "1";
constexpr auto TENSOR_CORE_MODE_TWO = "2";
constexpr auto WARP_MARKER = "warp_marker";

class ReduceEmitInfo {
 public:
  std::string akg_reduce_api_;
  std::string akg_reduce_template_arg_;
  std::string output_promoted_tensor_name_for_atomic_;
  std::string akg_atomic_api_;
  std::string akg_atomic_template_arg_;
  std::set<std::string> atomic_tensors_;

  std::string promoted_tensor_name_for_reduce_;
  std::map<std::string, Stmt> reduce_stmt_;

  std::string shared_compute_name_;
  std::string scalar_tensor_name_;
  std::string scalar_kht_name_;
  std::string scalar_khy_name_;
  std::string scalar_khc_name_;
  Expr input_tensor_expr_;

  std::string reduce_op_;
  std::string reduce_stmt_index_;
  bool is_atomic{false};
  Type output_tensor_data_type_info_;
  Type reduce_data_type_info_;

  std::set<std::string> added_tensors_;
  Stmt reduce_area_stmt_;
  Stmt origin_reduce_stmt_;
  std::map<std::string, Tensor> scalar_tensor_;
  Tensor shared_tensor_;
  std::vector<Stmt> stmts_;
  Expr atomic_rhs_;
  Stmt gm_write_stmt_;

  bool init_stmt_emit_{false};
};

struct Tile {
  int m{-1};
  int n{-1};
  int k{-1};
};

class TensorCoreInfo {
 public:
  bool in_matrix_a_{false};
  bool in_matrix_b_{false};
  bool in_matrix_c_{false};
  bool in_matrix_sync_{false};

  std::map<std::string, bool> matrix_info_{{MMA_A, false}, {MMA_B, false}, {MMA_C, false}, {MMA_SYNC, false}};
  bool core_area_{false};
  bool fragment_axis_begin_{false};
  bool is_fragment_m_{false};
  bool is_fragment_n_{false};
  Expr fragment_m_;
  Expr fragment_n_;
  int warp_threads_y_{-1};
  int warp_threads_x_{-1};
  Tile warp_tile_;
  Tile thread_tile_;

  std::unordered_map<std::string, std::string> matrix_major_;
  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_map<air::ir::TensorKey, Region> bounds_;
  std::unordered_map<std::string, Array<Expr>> strides_;
  bool data_is_set_{false};
  std::set<std::string> frag_reg_;
  bool is_tensor_core_{false};
  bool for_mod_pos_found_{false};
  std::unordered_set<std::string> cast_tensors_;
  std::unordered_map<Var, Expr, ExprHash, ExprEqual> core_area_for_extent_;
  std::unordered_map<std::string, Array<Expr>> min_bounds_;

  std::string wmma_scope_;
};

class GpuIslEmitter : public IslEmitter {
 public:
  GpuIslEmitter(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) : IslEmitter(info, n, i) {}
  ~GpuIslEmitter() override = default;

  Stmt Emit(const isl::ast_node &node) final;
  Expr Interpret(const isl::ast_expr &e);

 private:
  // override emitters for GPU
  Stmt EmitBlock(const isl::ast_node_block &node) final;
  Stmt EmitStmt(const isl::ast_node_user &node) final;
  Stmt EmitFor(const isl::ast_node_for &node) final;
  Stmt EmitMark(const isl::ast_node_mark &node_id) override;
  Stmt EmitIf(const isl::ast_node_if &node) final;
  Stmt EmitUserStmt(const isl::ast_node_user &node) final;

  // DMA emitters for GPU
  Expr EmitLoad(const isl::ast_expr &lhs, Type type);
  Stmt EmitRead(const isl::ast_node_user &node);
  Stmt EmitWrite(const isl::ast_node_user &node);
  Stmt EmitWriteAtomic(const isl::ast_node_user &node);

  Stmt EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args);
  Stmt EmitAccessNodeFromPromoteAcsProvide(isl::id var, const Node *node, Array<Expr> &args);
  isl::multi_aff TensorAccessMultAff(isl::id &tensor_id, const Array<Expr> &subscripts, const isl::id &stmt_id);

  Stmt EmitSync();
  Stmt EmitReduceInit(const isl::ast_node_user &node);
  Stmt EmitReduceUpdate(const isl::ast_node_user &node);
  Stmt EmitReduceArea(const isl::ast_node_user &node);
  Stmt EmitAttr();  // thread_extent, virtual_thread

  // add for tensor core
  Stmt EmitUserStmtCore(const isl::ast_node_user &node);
  Stmt EmitUserStmtCoreSync(const isl::ast_node_user &node);
  Stmt EmitReadCore(const isl::ast_node_user &node);
  Stmt EmitWriteCore(const isl::ast_node_user &node);

  Type GetTypeOfTensor(std::string name);
  Expr MakeLeftCallFromProvide(const Provide *op);
  void PrepareDataForTensorCore();
  bool CheckTileValid(Tile tile);

  Expr FindRealizeScope(const isl::id &var);
  std::string FindRealizeScopeToString(const isl::id &var);
  Stmt InsertRealize(Stmt stmt, const isl::id &var);
  Stmt InsertRealizeWithMemType(Stmt stmt, const isl::id &var, std::string mem);

  Expr IterNameAdaptor(std::string name);
  Expr SingleConfigToMultiBand(std::string name);

  Expr AdaptPolyNewVar(std::string name);
  int GetThreadExtent(const std::string &name);

  Expr ModifyTheInitExpr(const Expr &e);
  Expr ModifyTheCondExpr(const Expr &e, int inc);
  Expr ModifyTheIterExpr(const VarExpr &iter, int inc, const Expr &init);

  Stmt EmitRealizeForGlobalTensor(Stmt stmt);

  bool NoNeedToEmitForTempTensor(const isl::id &id);

  void MakeAkgReduceFuncName();
  void ConstructAtomicReturnFuncName();
  void MakeReduceStmt();
  Stmt TransferToKaHanInterface();
  Stmt MakeAtomicStmt();

  void SetScalarTensorBind(std::string scalar_tensor_name);
  void SetSharedTensorBind();
  void ResetStatus();

  std::set<Tensor> realized_;

  std::unordered_map<const Variable *, Expr> stride_modify_iter_map_;
  std::map<std::string, VarExpr> iter_name_map_{{B0, VarExpr(BLOCK_IDX_X)},  {B1, VarExpr(BLOCK_IDX_Y)},
                                                {B2, VarExpr(BLOCK_IDX_Z)},  {T0, VarExpr(THREAD_IDX_X)},
                                                {T1, VarExpr(THREAD_IDX_Y)}, {T2, VarExpr(THREAD_IDX_Z)}};

  bool in_reduce_area_{false};
  bool update_stmt_out_{false};
  bool init_stmt_out_{false};
  bool is_out_most_stmt_{true};
  ReduceEmitInfo reduce_info_;
  TensorCoreInfo tensor_core_info_;
  bool is_sync_before_{false};
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

class DeleteThreadIdx : public air::ir::IRMutator {
 public:
  explicit DeleteThreadIdx() {}
  ~DeleteThreadIdx() override = default;
  Expr Mutate_(const Variable *op, const Expr &e) {
    if (op->name_hint == THREAD_IDX_X) {
      return make_const(Int(32), 0);
    }

    return e;
  }
};

class EmitTensorCoreHelper {
 public:
  EmitTensorCoreHelper(TensorCoreInfo &info) : tensor_core_info_(info) {}
  ~EmitTensorCoreHelper(){};

  void SetDataForLoad(Expr src, Expr stride, Expr major, const Call *call, const Provide *op,
                      NodePtr<BufferNode> &node);
  void SetDataForStore(Expr dst, Expr stride, const Call *call, NodePtr<BufferNode> &node);
  void SetDataForFill(const Provide *op, const Call *call, NodePtr<BufferNode> &node);
  void SetDataForSync(Expr a, Expr b, Expr c, NodePtr<BufferNode> &node_a, NodePtr<BufferNode> &node_b,
                      NodePtr<BufferNode> &node_c);

  void PrepareDataCore();

  Stmt MakeLoadTransform();
  Stmt MakeStoreTransform();
  Stmt MakeFillTransform();
  Stmt MakeSyncTransform();

  Array<Expr> GetTileSize(const std::string &name);

 private:
  Array<NodeRef> node_;
  Expr tuple_;
  TensorCoreInfo &tensor_core_info_;

  DataForLoad data_for_load_;
  DataForStore data_for_store_;
  DataForFill data_for_fill_;
  DataForSync data_for_sync_;

  air::ir::TensorKey key_;
  const Call *call_;
  NodePtr<BufferNode> buffer_node_;
  Type data_type_;
};

class AddMmaAttrFlag : public air::ir::IRMutator {
 public:
  explicit AddMmaAttrFlag(TensorCoreInfo t) : tt(t) {}
  ~AddMmaAttrFlag() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (op->attr_key == air::ir::attr::realize_scope) {
      auto node = op->node.as<OperationNode>();
      if (node != nullptr) {
        if (!tt.frag_reg_.count(node->name)) {
          return stmt;
        }

        auto it = tt.matrix_abc_.find(SimplifyName(node->name));
        CHECK(it != tt.matrix_abc_.end()) << "Cannot find matrix info for " << node->name;
        std::string name = it->second;
        if (name == MATRIX_C) {
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

class TensorSubstituteTensorCore : public air::ir::IRMutator {
 public:
  explicit TensorSubstituteTensorCore(const FunctionRef &a, const FunctionRef &b, int b_value_index)
      : a_(a), b_(b), b_value_index_(b_value_index) {}
  ~TensorSubstituteTensorCore() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == air::ir::attr::buffer_bind_scope) {
      Array<NodeRef> arr = Downcast<Array<NodeRef>>(op->node);
      CHECK_EQ(arr.size(), 2U);
      const BufferNode *buffer = arr[0].as<BufferNode>();
      const TensorNode *tensor = arr[1].as<TensorNode>();
      CHECK(buffer && tensor);
      if (tensor->op == a_) {
        Tensor new_tensor = TensorNode::make(tensor->shape, tensor->dtype, Downcast<Operation>(b_), b_value_index_);
        Array<NodeRef> node = {arr[0], new_tensor};
        return AttrStmt::make(node, op->attr_key, op->value, op->body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  FunctionRef a_, b_;
  int b_value_index_{0};
};

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

  Expr Mutate_(const Variable *op, const Expr &e) {
    bool be_zero = false;
    for (auto &i : for_iters_) {
      if (i == op) {
        be_zero = true;
        break;
      }
    }

    if (be_zero) {
      return make_const(Int(32), 0);
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

class AkgReduceStmtChange : public air::ir::IRMutator {
 public:
  explicit AkgReduceStmtChange(Tensor t, Array<Expr> args, std::string name) : t(t), args(args), name(name) {}
  ~AkgReduceStmtChange() override = default;

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == name) {
      return Call::make(op->type, t->op->func_name(), args, op->call_type, t->op, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    auto new_op = stmt.as<Provide>();
    CHECK(new_op);
    if (new_op->func->func_name() == name) {
      return Provide::make(t->op, new_op->value_index, new_op->value, args);
    }
    return stmt;
  }

 private:
  Tensor t;
  Array<Expr> args;
  std::string name;
};

class ConditionExprMod : public air::ir::IRMutator {
 public:
  explicit ConditionExprMod(std::string block_del, bool &is_found) : block_del_(block_del), is_found_(is_found) {}
  ~ConditionExprMod() override = default;

  Expr Mutate_(const And *op, const Expr &e) override {
    auto o_a = op->a;
    auto o_b = op->b;
    auto a = air::ir::IRMutator::Mutate(op->a);
    auto b = air::ir::IRMutator::Mutate(op->b);
    if (!a.defined() && !b.defined()) return Expr();
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    if (o_a.same_as(a) && o_b.same_as(b)) return e;
    return And::make(a, b);
  }

  Expr Mutate_(const Or *op, const Expr &e) override {
    auto o_a = op->a;
    auto o_b = op->b;
    auto a = air::ir::IRMutator::Mutate(op->a);
    auto b = air::ir::IRMutator::Mutate(op->b);
    if (!a.defined() && !b.defined()) return Expr();
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    if (o_a.same_as(a) && o_b.same_as(b)) return e;
    return Or::make(a, b);
  }

  Expr Mutate_(const EQ *op, const Expr &e) override {
    Expr a = op->a;
    Expr b = op->b;

    bool rh_zero = false;
    bool lh_block = false;
    if (b.as<IntImm>()) {
      auto v = b.as<IntImm>();
      if (v->value == 0) rh_zero = true;
    }

    if (a.as<Variable>()) {
      auto v = a.as<Variable>();
      if (v->name_hint == block_del_) {
        lh_block = true;
      }
    }

    if (rh_zero && lh_block) {
      is_found_ = true;
      return Expr();
    }
    return e;
  }

 private:
  std::string block_del_{""};
  bool &is_found_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_GPU_ISL_EMITTER_H_
