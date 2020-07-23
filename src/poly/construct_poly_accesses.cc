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
  auto coordinate =
    CollectTensorCoordinate(domain.param_space, isl::id(domain.param_space.ctx(), tensor), dimensions.size());
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
      tensor_access = tensor_access.intersect(domain_aff_bounds.eq_set(identity.get_aff(static_cast<int>(dim_idx))));
    }
  }

  auto tensor_map =
    AddSuffix4Accesses(accesses, tensor_access.unbind_params_insert_domain(domain.tuple), op, domain.param_space.ctx());

  return {tensor_map, isl::map::from(identity)};
}

std::tuple<isl::union_map, isl::union_map, isl::union_map> ConstructPolyAccesses(const OperatorDomainSpace &domain,
                                                                                 const Stmt &s, AccessMap &accesses) {
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
            m_out_h = item.second.as<IntImm>() != nullptr ? static_cast<int>(item.second.as<IntImm>()->value) : 0;
          } else if (item.first == ATTR_PRAGMA_OUT_W) {
            m_out_w = item.second.as<IntImm>() != nullptr ? static_cast<int>(item.second.as<IntImm>()->value) : 0;
          }
        }
      }
      IRVisitor::Visit_(op);
    }

    void Visit_(const Evaluate *op) override {
      CHECK(op);
      const int im2_col_arg_num = 23;
      enum Im2colCallIndex {
        idxStrideH = 7,
        idxStrideW,
        idxKernelH,
        idxKernelW,
        idxPadTop = 17,
        idxPadBottom,
        idxPadLeft,
        idxPadRight
      };
      const Call *call = op->value.as<Call>();
      CHECK(call);
      auto getCallValue = [&call](const Im2colCallIndex &idx) {
        if (auto item = call->args[static_cast<size_t>(idx)].as<IntImm>()) {
          return static_cast<int>(item->value);
        }
        return 0;
      };
      if (call->name == CALL_IM2COL_UB && call->args.size() == im2_col_arg_num) {
        m_strid_h = getCallValue(Im2colCallIndex::idxStrideH);
        m_strid_w = getCallValue(Im2colCallIndex::idxStrideW);
        m_kernel_h = getCallValue(Im2colCallIndex::idxKernelH);
        m_kernel_w = getCallValue(Im2colCallIndex::idxKernelW);
        m_pad_top = getCallValue(Im2colCallIndex::idxPadTop);
        m_pad_bottom = getCallValue(Im2colCallIndex::idxPadBottom);
        m_pad_left = getCallValue(Im2colCallIndex::idxPadLeft);
        m_pad_right = getCallValue(Im2colCallIndex::idxPadRight);
      }
      IRVisitor::Visit_(op);
    }

    int KernelH() const { return m_kernel_h; }

    int KernelW() const { return m_kernel_w; }
    int OutH() const { return m_out_h; }
    int OutW() const { return m_out_w; }
    int StrideH() const { return m_strid_h; }
    int StrideW() const { return m_strid_w; }
    int PadLeft() const { return m_pad_left; }
    int PadRight() const { return m_pad_right; }
    int PadTop() const { return m_pad_top; }
    int PadBottom() const { return m_pad_bottom; }

   private:
    int m_kernel_h{0};
    int m_kernel_w{0};
    int m_out_h{0};
    int m_out_w{0};
    int m_strid_h{0};
    int m_strid_w{0};
    int m_pad_left{0};
    int m_pad_right{0};
    int m_pad_top{0};
    int m_pad_bottom{0};
  };
  class RelationAccessesParser final : public IRVisitor {
   public:
    isl::map ExtractIm2ColReadAccess(const std::string &tensor, const Array<Expr> &shape) {
      const int arg_num = shape.size();
      isl::space param_space = domain.param_space;
      isl::id tensor_id(param_space.ctx(), tensor);
      auto coordinate = CollectTensorCoordinate(param_space, tensor_id, arg_num);
      auto tensor_space = coordinate.get_space();

      isl::set access = isl::set::universe(tensor_space);
      auto identity = isl::multi_aff::identity(tensor_space.map_from_set());
      // need to optimize automatic add this exprs
      Array<Expr> args;
      auto arg_size = static_cast<size_t>(param_space.dim(isl_dim_param));
      int k_h = extractor.KernelH();
      int k_w = extractor.KernelW();
      int o_h = extractor.OutH();
      int o_w = extractor.OutW();
      if (arg_size == 3) {
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
      auto map = access.unbind_params_insert_domain(domain.tuple);

      std::string tag = "__poly_ref_0";
      isl::id tag_id(domain.param_space.ctx(), tag);
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
      if (arg_size == 2) {
        range = range.lower_bound_si(isl_dim_set, static_cast<unsigned int>(FeatureMapIndex::kBatchIndex), 0);
        CHECK(shape[static_cast<size_t>(FeatureMapIndex::kBatchIndex)].as<IntImm>());
        range = range.upper_bound_si(isl_dim_set, static_cast<unsigned int>(FeatureMapIndex::kBatchIndex),
                                     shape[static_cast<size_t>(FeatureMapIndex::kBatchIndex)].as<IntImm>()->value - 1);
      }
      CHECK(shape[static_cast<size_t>(FeatureMapIndex::kHIndex)].as<IntImm>() &&
            shape[static_cast<size_t>(FeatureMapIndex::kWIndex)].as<IntImm>() &&
            shape[static_cast<size_t>(FeatureMapIndex::kC0Index)].as<IntImm>());

      range = range.lower_bound_si(isl_dim_set, static_cast<unsigned int>(FeatureMapIndex::kHIndex), 0);
      range = range.upper_bound_si(isl_dim_set, static_cast<unsigned int>(FeatureMapIndex::kHIndex),
                                   shape[static_cast<size_t>(FeatureMapIndex::kHIndex)].as<IntImm>()->value - 1);
      range = range.lower_bound_si(isl_dim_set, static_cast<unsigned int>(FeatureMapIndex::kWIndex), 0);
      range = range.upper_bound_si(isl_dim_set, static_cast<unsigned int>(FeatureMapIndex::kWIndex),
                                   shape[static_cast<size_t>(FeatureMapIndex::kWIndex)].as<IntImm>()->value - 1);
      range = range.lower_bound_si(isl_dim_set, static_cast<unsigned int>(FeatureMapIndex::kC0Index), 0);
      range = range.upper_bound_si(isl_dim_set, static_cast<unsigned int>(FeatureMapIndex::kC0Index),
                                   shape[static_cast<size_t>(FeatureMapIndex::kC0Index)].as<IntImm>()->value - 1);

      map = map.intersect_range(range);

      return map;
    }

    bool UpdateAccess(const Array<Expr> &shape) const {
      const size_t kHIndex = 2;
      const int largeHSize = 200;
      Expr fm_h = shape[kHIndex];
      if (extractor.PadTop() > 0 && extractor.PadBottom() > 0 && extractor.PadLeft() > 0 && extractor.PadRight() > 0 &&
          Compare(fm_h, Expr(largeHSize)) > 0) {
        return true;
      }
      return false;
    }

    std::string getConstraint(const std::string &min_j, const std::string &max_j, const std::string &min_h,
                              const std::string &max_h) {
      std::ostringstream ss;
      ss << "(" << min_j << " <= j <= " << max_j << " and " << min_h << " <= arg2 <= " << max_h << ")";
      std::string set_con = ss.str();
      return set_con;
    }

    std::string toString(int i) {
      std::ostringstream ss;
      ss << i;
      return ss.str();
    }

    std::string body(bool left) {
      std::ostringstream ss;
      if (left) {
        ss << extractor.StrideH() << "j/" << extractor.KernelH() << " - " << extractor.PadLeft();
      } else {
        ss << extractor.StrideH() << "j/" << extractor.KernelH() << " + " << extractor.PadRight();
      }
      return ss.str();
    }

    void UpdatePaddingConstraint(const Expr &fmH) {
      int size_h = 0;
      if (fmH.as<IntImm>()) {
        size_h = static_cast<int>(fmH.as<IntImm>()->value);
      }
      const int mi = 16;
      const int cut_h = 2;
      int size_m = extractor.OutH() * extractor.OutW() / mi;
      int head_m = cut_h * extractor.OutW() / mi;

      int head_h = extractor.KernelH() + (cut_h - 1) * extractor.StrideH() - extractor.PadTop() - 1;
      int tail_h = (extractor.OutH() - cut_h) * extractor.StrideH() - extractor.PadTop();

      std::string head_con = getConstraint(toString(0), toString(head_m - 1), toString(0), toString(head_h));
      std::string tail_con =
        getConstraint(toString(size_m - head_m), toString(size_m - 1), toString(tail_h), toString(size_h - 1));
      std::string body_con = getConstraint(toString(head_m), toString(size_m - head_m - 1), body(true), body(false));

      auto map_str = reads.to_str();
      std::string constraint = " (" + head_con + " or " + body_con + " or " + tail_con + ") ";
      size_t endPos = map_str.find("}");
      std::string main = map_str.substr(0, endPos);
      main = main + " and " + constraint + " }";
      isl_union_map *read_tmp = isl_union_map_read_from_str(reads.ctx().get(), main.c_str());
      CHECK(read_tmp);
      reads = isl::manage(read_tmp);
    }

    isl::map ExtractIm2ColWriteAccess(const std::string &tensor, const Array<Expr> &shape) {
      int arg_num = shape.size();
      isl::space param_space = domain.param_space;
      isl::id tensor_id(param_space.ctx(), tensor);
      auto coordinate = CollectTensorCoordinate(param_space, tensor_id, arg_num);
      auto tensor_space = coordinate.get_space();

      isl::set access = isl::set::universe(tensor_space);
      auto identity = isl::multi_aff::identity(tensor_space.map_from_set());
      // need to optimize automatic add this exprs
      auto arg_size = static_cast<size_t>(param_space.dim(isl_dim_param));
      Array<Expr> args;
      const std::vector<std::string> consStr5D = {"i", "j", "k", "mi", "ni"};
      const std::vector<std::string> consStr4D = {"j", "k", "mi", "ni"};
      enum ShapeDim { shape5D = 0, shape4D };
      ShapeDim mod = ShapeDim::shape5D;
      if (consStr5D.size() == shape.size()) {
        mod = ShapeDim::shape5D;
        for (size_t i = 0; i < arg_size; ++i) {
          if (i == 0) {
            CHECK(shape[0].as<IntImm>());
            Expr e = shape[0].as<IntImm>()->value > 0 ? static_cast<Expr>(Var(consStr5D[i])) : Expr(0);
            args.push_back(e);
          } else {
            args.push_back(static_cast<Expr>(Var(consStr5D[i])));
          }
        }
      } else if (consStr4D.size() == shape.size()) {
        mod = ShapeDim ::shape4D;
        for (size_t i = 0; i < arg_size; ++i) {
          args.push_back(static_cast<Expr>(Var(consStr4D[i])));
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

      auto map = access.unbind_params_insert_domain(domain.tuple);

      std::string tag = "__poly_ref_1";
      isl::id tag_id(domain.param_space.ctx(), tag);
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
        CHECK_GE(call_op->args.size(), 2);
        CHECK(call_op->args[0].as<Call>());
        CHECK_GE(call_op->args[0].as<Call>()->args.size(), 2);
        CHECK(call_op->args[0].as<Call>()->args[1].as<Variable>());
        CHECK(call_op->args[1].as<Call>());
        CHECK_GE(call_op->args[1].as<Call>()->args.size(), 2);
        CHECK(call_op->args[1].as<Call>()->args[1].as<Variable>());
        std::string write_buffer = call_op->args[0].as<Call>()->args[1].as<Variable>()->name_hint;
        std::string read_buffer = call_op->args[1].as<Call>()->args[1].as<Variable>()->name_hint;
        for (auto item : accesses) {
          if (item.first->IsInstance<AttrStmt>()) {
            auto attr = static_cast<const AttrStmt *>(item.first);
            Array<NodeRef> array = Downcast<Array<NodeRef>>(attr->node);
            Buffer buffer = Downcast<Buffer>(array[0]);
            Tensor tensor = Downcast<Tensor>(array[1]);
            if (buffer->name == read_buffer) {
              isl::map readIm2Col = ExtractIm2ColReadAccess(tensor->op->name, tensor->shape);
              reads = reads.unite(readIm2Col);
              if (UpdateAccess(tensor->shape)) {
                UpdatePaddingConstraint(tensor->shape[2]);
              }
            } else if (buffer->name == write_buffer) {
              isl::map writeIm2Col = ExtractIm2ColWriteAccess(tensor->op->name, tensor->shape);
              writes = writes.unite(writeIm2Col);
            }
          }
        }
      }
    }

    void Visit_(const Call *op) final {
      IRVisitor::Visit_(op);
      if (op->call_type == Call::Halide) {
        isl::map reads_tmp, toinner_tmp;
        std::string var_name = op->name;
        if (op->func.defined() && op->func->num_outputs() != 1) {
          var_name = var_name + "_v" + std::to_string(op->value_index);
        }
        std::tie(reads_tmp, toinner_tmp) = ConstructPolyAccess(domain, op, var_name, op->args, accesses);
        reads = reads.unite(reads_tmp);
        to_inner_ = to_inner_.add_map(toinner_tmp);
      }
    }

    void Visit_(const Provide *op) final {
      IRVisitor::Visit_(op);
      isl::map writes_tmp, toinner_tmp;
      std::string var_name = op->func->func_name();
      if (op->func->num_outputs() != 1) {
        var_name = var_name + "_v" + std::to_string(op->value_index);
      }
      std::tie(writes_tmp, toinner_tmp) = ConstructPolyAccess(domain, op, var_name, op->args, accesses);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.add_map(toinner_tmp);
    }

    /* The conditionals of IfThenElse statements may fall in these cases.
     * The accesses should be updated to read sets of scop as such accesses
     * may only be read.
     *
     * More complicated cases like conditionals involving Store and/or
     * Provide should also update write sets.
     */
    void Visit_(const EQ *op) final {
      isl::union_map reads_tmp, writes_tmp, toinner_tmp;

      Stmt stmt_a(GetObjPtr(op->a.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_a, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);

      Stmt stmt_b(GetObjPtr(op->b.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_b, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);
    }

    void Visit_(const NE *op) final {
      isl::union_map reads_tmp, writes_tmp, toinner_tmp;

      Stmt stmt_a(GetObjPtr(op->a.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_a, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);

      Stmt stmt_b(GetObjPtr(op->b.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_b, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);
    }

    void Visit_(const LT *op) final {
      isl::union_map reads_tmp, writes_tmp, toinner_tmp;

      Stmt stmt_a(GetObjPtr(op->a.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_a, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);

      Stmt stmt_b(GetObjPtr(op->b.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_b, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);
    }

    void Visit_(const LE *op) final {
      isl::union_map reads_tmp, writes_tmp, toinner_tmp;

      Stmt stmt_a(GetObjPtr(op->a.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_a, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);

      Stmt stmt_b(GetObjPtr(op->b.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_b, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);
    }

    void Visit_(const GT *op) final {
      isl::union_map reads_tmp, writes_tmp, toinner_tmp;

      Stmt stmt_a(GetObjPtr(op->a.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_a, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);

      Stmt stmt_b(GetObjPtr(op->b.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_b, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);
    }

    void Visit_(const GE *op) final {
      isl::union_map reads_tmp, writes_tmp, toinner_tmp;

      Stmt stmt_a(GetObjPtr(op->a.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_a, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);

      Stmt stmt_b(GetObjPtr(op->b.get()));
      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, stmt_b, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);
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
      isl::union_map reads_tmp, writes_tmp, toinner_tmp;

      std::tie(reads_tmp, writes_tmp, toinner_tmp) = ConstructPolyAccesses(domain, op->body, accesses);
      reads = reads.unite(reads_tmp);
      writes = writes.unite(writes_tmp);
      to_inner_ = to_inner_.unite(toinner_tmp);
    }

    const OperatorDomainSpace &domain;
    AccessMap &accesses;

    isl::union_map reads, writes;
    isl::union_map to_inner_;
    AttrsExtractor extractor;

    RelationAccessesParser(const Stmt stmt, const OperatorDomainSpace &space, AccessMap &accesses)
        : domain(space),
          accesses(accesses),
          reads(isl::union_map::empty(domain.tuple.get_space())),
          writes(isl::union_map::empty(domain.tuple.get_space())),
          to_inner_(isl::union_map::empty(domain.tuple.get_space())) {
      extractor.Apply(stmt);
      IRVisitor::Visit(stmt);
    }
    ~RelationAccessesParser() override = default;
  } parser(s, domain, accesses);
  return std::make_tuple(parser.reads, parser.writes, parser.to_inner_);
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
