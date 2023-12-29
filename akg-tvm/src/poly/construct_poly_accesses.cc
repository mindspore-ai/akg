/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "poly/construct_poly_accesses.h"

#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

#include <tuple>
#include <string>

#include "poly/scop_builder.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
namespace poly {

std::pair<isl::map, isl::map> ConstructPolyAccess(const OperatorDomainSpace &domain, const Node *op,
                                                  const std::string &tensor, const Array<Expr> &dimensions,
                                                  AccessMap &accesses) {
  // create a tensor coordinate to store the accessed relation
  auto coordinate = CollectTensorCoordinate(domain.param_space,
    isl::id(domain.param_space.ctx(), tensor), dimensions.size());
  auto tensor_space = coordinate.get_space();

  // create a fully access set
  isl::set tensor_access = isl::set::universe(tensor_space);

  // add access relation constraint for each parameter of one dimension
  auto identity = isl::multi_aff::identity(tensor_space.map_from_set());
  for (size_t dim_idx = 0; dim_idx < dimensions.size(); ++dim_idx) {
    // make aff bounds of each dimension.
    auto domain_aff_bounds = Expr2Aff(domain.param_space, dimensions[dim_idx]);
    if (!domain_aff_bounds.is_null()) {
      domain_aff_bounds = domain_aff_bounds.unbind_params_insert_domain(coordinate);
      tensor_access = tensor_access.intersect(
        domain_aff_bounds.eq_set(identity.get_aff(static_cast<int>(dim_idx))));
    }
  }

  auto tensor_map = AddSuffix4Accesses(accesses,
    tensor_access.unbind_params_insert_domain(domain.tuple), op, domain.param_space.ctx());

  return {tensor_map, isl::map::from(identity)};
}

class AttrsExtractor final : public IRVisitor {
 public:
  AttrsExtractor() {}
  ~AttrsExtractor() override = default;

  void Apply(const Stmt &s) { IRVisitor::Visit(s); }

  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == ATTR_IM2COL_KEY) {
      Map<std::string, Expr> var_map = Downcast<Map<std::string, Expr>>(op->node);
      for (auto item : var_map) {
        if (item.first == ATTR_PRAGMA_OUT_H) {
          out_h_ = item.second.as<IntImm>() != nullptr ?
            static_cast<int>(item.second.as<IntImm>()->value) : 0;
        } else if (item.first == ATTR_PRAGMA_OUT_W) {
          out_w_ = item.second.as<IntImm>() != nullptr ?
            static_cast<int>(item.second.as<IntImm>()->value) : 0;
        }
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Evaluate *op) override {
    constexpr int kIm2colArgNum = 23;
    enum Im2colCallIndex {
      IDX_STRIDE_H = 7,
      IDX_STRIDE_W,
      IDX_KERNEL_H,
      IDX_KERNEL_W,
      IDX_PAD_TOP = 17,
      IDX_PAD_BOTTOM,
      IDX_PAD_LEFT,
      IDX_PAD_RIGHT
    };
    const Call *call = op->value.as<Call>();
    CHECK(call);
    auto GetCallValue = [&call](const Im2colCallIndex &idx) {
      if (auto item = call->args[static_cast<size_t>(idx)].as<IntImm>()) {
        return static_cast<int>(item->value);
      }
      return 0;
    };
    if (call->name == CALL_IM2COL_UB && call->args.size() >= kIm2colArgNum) {
      strid_h_ = GetCallValue(Im2colCallIndex::IDX_STRIDE_H);
      strid_w_ = GetCallValue(Im2colCallIndex::IDX_STRIDE_W);
      kernel_h_ = GetCallValue(Im2colCallIndex::IDX_KERNEL_H);
      kernel_w_ = GetCallValue(Im2colCallIndex::IDX_KERNEL_W);
      pad_top_ = GetCallValue(Im2colCallIndex::IDX_PAD_TOP);
      pad_bottom_ = GetCallValue(Im2colCallIndex::IDX_PAD_BOTTOM);
      pad_left_ = GetCallValue(Im2colCallIndex::IDX_PAD_LEFT);
      pad_right_ = GetCallValue(Im2colCallIndex::IDX_PAD_RIGHT);
    }
    IRVisitor::Visit_(op);
  }

  int KernelH() const { return kernel_h_; }
  int KernelW() const { return kernel_w_; }
  int OutH() const { return out_h_; }
  int OutW() const { return out_w_; }
  int StrideH() const { return strid_h_; }
  int StrideW() const { return strid_w_; }
  int PadLeft() const { return pad_left_; }
  int PadRight() const { return pad_right_; }
  int PadTop() const { return pad_top_; }
  int PadBottom() const { return pad_bottom_; }

 private:
  int kernel_h_{0};
  int kernel_w_{0};
  int out_h_{0};
  int out_w_{0};
  int strid_h_{0};
  int strid_w_{0};
  int pad_left_{0};
  int pad_right_{0};
  int pad_top_{0};
  int pad_bottom_{0};
};

class RelationAccessesParser final : public IRVisitor {
 public:
  isl::map ExtractIm2ColReadAccess(const std::string &tensor, const Array<Expr> &shape) {
    const int kArgSizeTwo = 2;
    const int kArgSizeThree = 3;
    const int arg_num = shape.size();
    isl::space param_space = domain_.param_space;
    isl::id tensor_id(param_space.ctx(), tensor);
    auto coordinate = CollectTensorCoordinate(param_space, tensor_id, arg_num);
    auto tensor_space = coordinate.get_space();

    isl::set access = isl::set::universe(tensor_space);
    auto identity = isl::multi_aff::identity(tensor_space.map_from_set());
    // need to optimize automatic add this exprs
    Array<Expr> args;
    auto arg_size = static_cast<size_t>(param_space.dim(isl_dim_param));
    int k_h = extractor_.KernelH();
    int k_w = extractor_.KernelW();
    int o_h = extractor_.OutH();
    int o_w = extractor_.OutW();
    if (arg_size == kArgSizeThree) {
      CHECK(shape[0].as<IntImm>());
      args.push_back(shape[0].as<IntImm>()->value > 0 ? static_cast<Expr>(Var("i")) : Expr(0));
    } else {
      args.push_back(VarExpr("j") * Expr(16) / Expr(o_h * o_w));
    }
    VarExpr k("k");
    CHECK_GT(k_h, 0);
    CHECK_GT(k_w, 0);
    Expr v = k / Expr(k_h * k_w);
    args.push_back(v);
    for (size_t i = 0; i < args.size(); ++i) {
      auto range_point = identity.get_aff(static_cast<int>(i));
      auto domain_point = Expr2Aff(param_space, args[i]);
      if (!domain_point.is_null()) {
        domain_point = domain_point.unbind_params_insert_domain(coordinate);
        access = access.intersect(domain_point.eq_set(range_point));
      }
    }
    auto map = access.unbind_params_insert_domain(domain_.tuple);

    const std::string kTag = "__poly_ref_0";
    isl::id tag_id(domain_.param_space.ctx(), kTag);
    auto domain_space = map.get_space().domain();
    auto tag_space = domain_space.params().add_named_tuple_id_ui(tag_id, 0);
    domain_space = domain_space.product(tag_space).unwrap();
    map = map.preimage_domain(isl::multi_aff::domain_map(domain_space));
    enum FeatureMapIndex { kBatchIndex = 0, kC1Index, kHIndex, kWIndex, kC0Index, KFeatureMapSiz };

    CHECK_EQ(shape.size(), FeatureMapIndex::KFeatureMapSiz);
    isl::set range = map.range();
    /***********************
     * no cut in H axis
     * 0<= arg2 <= fm_h-1
     * 0<= arg3 <= fm_w-1
     * 0<= arg4 <= 16-1
     ************************/
    if (arg_size == kArgSizeTwo) {
      range = range.lower_bound_si(isl_dim_set,
        static_cast<unsigned int>(FeatureMapIndex::kBatchIndex), 0);
      CHECK(shape[static_cast<size_t>(FeatureMapIndex::kBatchIndex)].as<IntImm>());
      range = range.upper_bound_si(isl_dim_set,
        static_cast<unsigned int>(FeatureMapIndex::kBatchIndex),
        shape[static_cast<size_t>(FeatureMapIndex::kBatchIndex)].as<IntImm>()->value - 1);
    }
    CHECK(shape[static_cast<size_t>(FeatureMapIndex::kHIndex)].as<IntImm>() &&
          shape[static_cast<size_t>(FeatureMapIndex::kWIndex)].as<IntImm>() &&
          shape[static_cast<size_t>(FeatureMapIndex::kC0Index)].as<IntImm>());

    range = range.lower_bound_si(isl_dim_set,
      static_cast<unsigned int>(FeatureMapIndex::kHIndex), 0);
    range = range.upper_bound_si(isl_dim_set,
      static_cast<unsigned int>(FeatureMapIndex::kHIndex),
      shape[static_cast<size_t>(FeatureMapIndex::kHIndex)].as<IntImm>()->value - 1);
    range = range.lower_bound_si(isl_dim_set,
      static_cast<unsigned int>(FeatureMapIndex::kWIndex), 0);
    range = range.upper_bound_si(isl_dim_set,
      static_cast<unsigned int>(FeatureMapIndex::kWIndex),
      shape[static_cast<size_t>(FeatureMapIndex::kWIndex)].as<IntImm>()->value - 1);
    range = range.lower_bound_si(isl_dim_set,
      static_cast<unsigned int>(FeatureMapIndex::kC0Index), 0);
    range = range.upper_bound_si(isl_dim_set,
      static_cast<unsigned int>(FeatureMapIndex::kC0Index),
      shape[static_cast<size_t>(FeatureMapIndex::kC0Index)].as<IntImm>()->value - 1);

    map = map.intersect_range(range);
    return map;
  }

  bool UpdateAccess(const Array<Expr> &shape) const {
    const size_t kHIndex = 2;
    const int kLargeHSize = 200;
    Expr fm_h = shape[kHIndex];
    if (extractor_.PadTop() > 0 && extractor_.PadBottom() > 0 &&
      extractor_.PadLeft() > 0 && extractor_.PadRight() > 0 &&
      Compare(fm_h, Expr(kLargeHSize)) > 0) {
      return true;
    }
    return false;
  }

  std::string GetConstraint(const std::string &min_j, const std::string &max_j, const std::string &min_h,
                            const std::string &max_h) {
    std::ostringstream ss;
    ss << "(" << min_j << " <= j <= " << max_j << " and " << min_h << " <= arg2 <= " << max_h << ")";
    std::string set_con = ss.str();
    return set_con;
  }

  std::string Body(bool left) {
    std::ostringstream ss;
    if (left) {
      ss << extractor_.StrideH() << "j/" << extractor_.KernelH() << " - " << extractor_.PadLeft();
    } else {
      ss << extractor_.StrideH() << "j/" << extractor_.KernelH() << " + " << extractor_.PadRight();
    }
    return ss.str();
  }

  void UpdatePaddingConstraint(const Expr &fmH) {
    int size_h = 0;
    const int kCutH = 2;
    const int kBlockMi = 16;

    if (fmH.as<IntImm>()) {
      size_h = static_cast<int>(fmH.as<IntImm>()->value);
    }
    int size_m = extractor_.OutH() * extractor_.OutW() / kBlockMi;
    int head_m = kCutH * extractor_.OutW() / kBlockMi;

    int head_h = extractor_.KernelH() + (kCutH - 1) * extractor_.StrideH() - extractor_.PadTop() - 1;
    int tail_h = (extractor_.OutH() - kCutH) * extractor_.StrideH() - extractor_.PadTop();

    std::string head_con = GetConstraint(std::to_string(0), std::to_string(head_m - 1),
      std::to_string(0), std::to_string(head_h));
    std::string tail_con = GetConstraint(std::to_string(size_m - head_m),
      std::to_string(size_m - 1), std::to_string(tail_h), std::to_string(size_h - 1));
    std::string body_con = GetConstraint(std::to_string(head_m),
      std::to_string(size_m - head_m - 1), Body(true), Body(false));

    auto map_str = reads_.to_str();
    std::string constraint = " (" + head_con + " or " + body_con + " or " + tail_con + ") ";
    size_t end_pos = map_str.find("}");
    std::string main = map_str.substr(0, end_pos);
    main = main + " and " + constraint + " }";
    isl_union_map *read_tmp = isl_union_map_read_from_str(reads_.ctx().get(), main.c_str());
    CHECK(read_tmp);
    reads_ = isl::manage(read_tmp);
  }

  isl::map ExtractIm2ColWriteAccess(const std::string &tensor, const Array<Expr> &shape) {
    int arg_num = shape.size();
    isl::space param_space = domain_.param_space;
    isl::id tensor_id(param_space.ctx(), tensor);
    auto coordinate = CollectTensorCoordinate(param_space, tensor_id, arg_num);
    auto tensor_space = coordinate.get_space();

    isl::set access = isl::set::universe(tensor_space);
    auto identity = isl::multi_aff::identity(tensor_space.map_from_set());
    // need to optimize automatic add this exprs
    auto arg_size = static_cast<size_t>(param_space.dim(isl_dim_param));
    Array<Expr> args;
    const std::vector<std::string> cons_str_5d = {"i", "j", "k", "mi", "ni"};
    const std::vector<std::string> cons_str_4d = {"j", "k", "mi", "ni"};
    enum ShapeDim { shape5D = 0, shape4D };
    ShapeDim mod = ShapeDim::shape5D;
    if (cons_str_5d.size() == shape.size()) {
      mod = ShapeDim::shape5D;
      for (size_t i = 0; i < arg_size; ++i) {
        if (i == 0) {
          CHECK(shape[0].as<IntImm>());
          Expr e = shape[0].as<IntImm>()->value > 0 ? static_cast<Expr>(Var(cons_str_5d[i])) : Expr(0);
          args.push_back(e);
        } else {
          args.push_back(static_cast<Expr>(Var(cons_str_5d[i])));
        }
      }
    } else if (cons_str_4d.size() == shape.size()) {
      mod = ShapeDim ::shape4D;
      for (size_t i = 0; i < arg_size; ++i) {
        args.push_back(static_cast<Expr>(Var(cons_str_4d[i])));
      }
    }

    for (size_t i = 0; i < args.size(); ++i) {
      auto range_point = identity.get_aff(static_cast<int>(i));
      auto domain_point = Expr2Aff(param_space, args[i]);
      if (!domain_point.is_null()) {
        domain_point = domain_point.unbind_params_insert_domain(coordinate);
        access = access.intersect(domain_point.eq_set(range_point));
      }
    }

    auto map = access.unbind_params_insert_domain(domain_.tuple);

    const std::string kTag = "__poly_ref_1";
    isl::id tag_id(domain_.param_space.ctx(), kTag);
    auto domain_space = map.get_space().domain();
    auto tag_space = domain_space.params().add_named_tuple_id_ui(tag_id, 0);
    domain_space = domain_space.product(tag_space).unwrap();
    map = map.preimage_domain(isl::multi_aff::domain_map(domain_space));

    enum FractalIndex { idxBatch = 0, idxMo, idxKo, idxMi, idxKi, fractalSize };
    /***********************
     * mi ni range definition
     * 0<= arg3 <= 16-1
     * 0<= arg4 <= 16-1
     ************************/
    CHECK_EQ(shape.size(), FractalIndex::fractalSize - mod);
    CHECK(shape[static_cast<uint32_t>(FractalIndex::idxMi - mod)].as<IntImm>() &&
          shape[static_cast<uint32_t>(FractalIndex::idxKi - mod)].as<IntImm>());
    isl::set range = map.range();

    range = range.lower_bound_si(isl_dim_set, static_cast<unsigned int>(FractalIndex::idxMi - mod), 0);
    range = range.upper_bound_si(isl_dim_set, static_cast<unsigned int>(FractalIndex::idxMi - mod),
                                 shape[static_cast<uint32_t>(FractalIndex::idxMi - mod)].as<IntImm>()->value - 1);

    range = range.lower_bound_si(isl_dim_set, static_cast<unsigned int>(FractalIndex::idxKi - mod), 0);
    range = range.upper_bound_si(isl_dim_set, static_cast<unsigned int>(FractalIndex::idxKi - mod),
                                 shape[static_cast<uint32_t>(FractalIndex::idxKi - mod)].as<IntImm>()->value - 1);
    map = map.intersect_range(range);

    return map;
  }

  void Visit_(const Evaluate *op) final {
    IRVisitor::Visit_(op);
    const Call *call_op = op->value.as<Call>();
    if (call_op && call_op->name == CALL_IM2COL_UB) {
      const int kMinArgSize = 2;
      CHECK_GE(call_op->args.size(), kMinArgSize);
      CHECK(call_op->args[0].as<Call>());
      CHECK_GE(call_op->args[0].as<Call>()->args.size(), kMinArgSize);
      CHECK(call_op->args[0].as<Call>()->args[1].as<Variable>());
      CHECK(call_op->args[1].as<Call>());
      CHECK_GE(call_op->args[1].as<Call>()->args.size(), kMinArgSize);
      CHECK(call_op->args[1].as<Call>()->args[1].as<Variable>());
      std::string write_buffer = call_op->args[0].as<Call>()->args[1].as<Variable>()->name_hint;
      std::string read_buffer = call_op->args[1].as<Call>()->args[1].as<Variable>()->name_hint;
      for (auto item : accesses_) {
        if (item.first->IsInstance<AttrStmt>()) {
          auto attr = static_cast<const AttrStmt *>(item.first);
          Array<NodeRef> array = Downcast<Array<NodeRef>>(attr->node);
          Buffer buffer = Downcast<Buffer>(array[0]);
          Tensor tensor = Downcast<Tensor>(array[1]);
          if (buffer->name == read_buffer) {
            isl::map readIm2Col = ExtractIm2ColReadAccess(tensor->op->name, tensor->shape);
            reads_ = reads_.unite(readIm2Col);
            if (UpdateAccess(tensor->shape)) {
              UpdatePaddingConstraint(tensor->shape[2]);
            }
          } else if (buffer->name == write_buffer) {
            isl::map writeIm2Col = ExtractIm2ColWriteAccess(tensor->op->name, tensor->shape);
            writes_ = writes_.unite(writeIm2Col);
          }
        }
      }
    }
  }

  void Visit_(const Call *op) final {
    IRVisitor::Visit_(op);
    if (op->call_type == Call::Halide) {
      isl::map reads, toinner;
      const std::string kSuffixV = "_v";
      std::string var_name = op->name;
      if (op->func.defined() && op->func->num_outputs() != 1) {
        var_name = var_name + kSuffixV + std::to_string(op->value_index);
      }
      std::tie(reads, toinner) = ConstructPolyAccess(domain_, op, var_name, op->args, accesses_);
      reads_ = reads_.unite(reads);
      to_inner_ = to_inner_.add_map(toinner);
    }
  }

  void Visit_(const Provide *op) final {
    IRVisitor::Visit_(op);
    isl::map writes, toinner;
    const std::string kSuffixV = "_v";
    std::string var_name = op->func->func_name();
    if (op->func->num_outputs() != 1) {
      var_name = var_name + kSuffixV + std::to_string(op->value_index);
    }
    std::tie(writes, toinner) = ConstructPolyAccess(domain_, op, var_name, op->args, accesses_);
    writes_ = writes_.unite(writes);
    to_inner_ = to_inner_.add_map(toinner);
  }

  void ConstructCmpOpAccesses(Expr a, Expr b) {
    isl::union_map reads, writes, toinner;

    Stmt stmt_a(GetObjPtr(a.get()));
    std::tie(reads, writes, toinner) = ConstructPolyAccesses(domain_, stmt_a, accesses_);
    reads_ = reads_.unite(reads);
    writes_ = writes_.unite(writes);
    to_inner_ = to_inner_.unite(toinner);

    Stmt stmt_b(GetObjPtr(b.get()));
    std::tie(reads, writes, toinner) = ConstructPolyAccesses(domain_, stmt_b, accesses_);
    reads_ = reads_.unite(reads);
    writes_ = writes_.unite(writes);
    to_inner_ = to_inner_.unite(toinner);
  }

  /* The conditionals of IfThenElse statements may fall in these cases.
   * The accesses should be updated to read sets of scop as such accesses
   * may only be read.
   *
   * More complicated cases like conditionals involving Store and/or
   * Provide should also update write sets.
   */
  void Visit_(const EQ *op) final {
    ConstructCmpOpAccesses(op->a, op->b);
  }

  void Visit_(const NE *op) final {
    ConstructCmpOpAccesses(op->a, op->b);
  }

  void Visit_(const LT *op) final {
    ConstructCmpOpAccesses(op->a, op->b);
  }

  void Visit_(const LE *op) final {
    ConstructCmpOpAccesses(op->a, op->b);
  }

  void Visit_(const GT *op) final {
    ConstructCmpOpAccesses(op->a, op->b);
  }

  void Visit_(const GE *op) final {
    ConstructCmpOpAccesses(op->a, op->b);
  }

  // End of conditionals of IfThenElse, more cases are pending.

  /* A For type statement may be visited in the presence of
   * IfThenElse in the scop, as the body of the enclosing
   * if statement.
   *
   * A Block type should be handled.
   */

  void Visit_(const For *op) final {
    IRVisitor::Visit_(op);
    isl::union_map reads, writes, toinner;

    std::tie(reads, writes, toinner) = ConstructPolyAccesses(domain_, op->body, accesses_);
    reads_ = reads_.unite(reads);
    writes_ = writes_.unite(writes);
    to_inner_ = to_inner_.unite(toinner);
  }

  const OperatorDomainSpace &domain_;
  AccessMap &accesses_;

  isl::union_map reads_, writes_;
  isl::union_map to_inner_;
  AttrsExtractor extractor_;

  RelationAccessesParser(const Stmt stmt, const OperatorDomainSpace &space, AccessMap &accesses)
      : domain_(space),
        accesses_(accesses),
        reads_(isl::union_map::empty(domain_.tuple.get_space())),
        writes_(isl::union_map::empty(domain_.tuple.get_space())),
        to_inner_(isl::union_map::empty(domain_.tuple.get_space())) {
    extractor_.Apply(stmt);
    IRVisitor::Visit(stmt);
  }
  ~RelationAccessesParser() override = default;
};

std::tuple<isl::union_map, isl::union_map, isl::union_map> ConstructPolyAccesses(const OperatorDomainSpace &domain,
                                                                                 const Stmt &s, AccessMap &accesses) {
  auto parser = RelationAccessesParser(s, domain, accesses);
  return std::make_tuple(parser.reads_, parser.writes_, parser.to_inner_);
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
