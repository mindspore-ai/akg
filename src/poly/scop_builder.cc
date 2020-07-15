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
#include "poly/scop_builder.h"

#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

#include <set>
#include <utility>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <string>

#include "pass/utils.h"
namespace akg {
namespace ir {
namespace poly {

// Note: We can only handle empty param_space for now.
isl::space CreateParamsSpace(const isl::ctx &ctx) { return isl::space(ctx, 0); }

isl::space CreateParamsSpace(const isl::ctx &ctx, const std::unordered_map<std::string, air::Var> &params) {
  auto space = isl::space(ctx, 0);

  // set parameter names
  for (auto it = params.begin(); it != params.end(); ++it) {
    space = space.add_param(isl::id(ctx, it->second->name_hint));
  }
  return space;
}

isl::aff Int2Aff(const isl::space &s, int64_t v) { return isl::aff(isl::local_space(s), isl::val(s.ctx(), v)); }

template <typename T>
inline std::vector<isl::aff> ConcatAffs(const isl::space &space, T *op, bool allow_min, bool allow_max) {
  std::vector<isl::aff> result;

  for (const auto &aff : Expr2AffBounds(space, op->a, allow_min, allow_max)) {
    result.push_back(aff);
  }
  for (const auto &aff : Expr2AffBounds(space, op->b, allow_min, allow_max)) {
    result.push_back(aff);
  }

  return result;
}

template <typename T>
inline std::vector<isl::aff> UniteAffs(const isl::space &space, T *op, isl::aff (isl::aff::*unite)(isl::aff) const) {
  std::vector<isl::aff> bounds_l = Expr2AffBounds(space, op->a, false, false);
  std::vector<isl::aff> bounds_r = Expr2AffBounds(space, op->b, false, false);
  CHECK_LE(bounds_l.size(), 1u);
  CHECK_LE(bounds_r.size(), 1u);

  if (bounds_l.size() > 0 && bounds_r.size() > 0) {
    return {(bounds_l[0].*unite)(bounds_r[0])};
  }

  return {};
}

template <typename T>
bool ExprType(const Expr &e) {
  return (e.as<T>() != nullptr);
}

std::vector<isl::aff> Variable2AffBounds(const isl::space &space, const Variable *v, bool ignore_error) {
  isl::id id(space.ctx(), v->name_hint);
  if (space.has_param(id)) {
    return {isl::aff::param_on_domain(space, id)};
  }
  CHECK(ignore_error) << "Can not find var: " << v->name_hint << " in isl::space: " << space << '\n';
  return {};
}

std::vector<isl::aff> FloorDiv2AffBounds(const isl::space &space, const FloorDiv *f_div) {
  if (f_div->type.is_int() || f_div->type.is_uint()) {
    auto left = Expr2AffBounds(space, f_div->a, false, false);
    auto right = Expr2AffBounds(space, f_div->b, false, false);
    if (left.size() == 0 || right.size() == 0) {
      return {};
    }
    return {(left[0].div)(right[0]).floor()};
  }
  return UniteAffs(space, f_div, &isl::aff::div);
}

std::vector<isl::aff> Div2AffBounds(const isl::space &space, const Div *div) {
  if (div->type.is_int() || div->type.is_uint()) {
    auto left = Expr2AffBounds(space, div->a, false, false);
    auto right = Expr2AffBounds(space, div->b, false, false);
    if (left.size() == 0 || right.size() == 0) {
      return {};
    }
    return {(left[0].div)(right[0]).floor()};
  }
  return UniteAffs(space, div, &isl::aff::div);
}

std::vector<isl::aff> FloorMod2AffBounds(const isl::space &space, const FloorMod *f_mod, bool ignore_error) {
  auto left = Expr2AffBounds(space, f_mod->a, false, false);
  Expr right = f_mod->b;
  if (const int64_t *val = as_const_int(right)) {
    isl::val v = isl::val(space.ctx(), *val);
    return {left[0].mod(v)};
  }
  CHECK(ignore_error) << "Mod's denominator is not a const_int\n";
  return {};
}

std::vector<isl::aff> Mod2AffBounds(const isl::space &space, const Mod *mod, bool ignore_error) {
  auto left = Expr2AffBounds(space, mod->a, false, false);
  Expr right = mod->b;
  if (const int64_t *val = as_const_int(right)) {
    isl::val v = isl::val(space.ctx(), *val);
    return {left[0].mod(v)};
  }
  CHECK(ignore_error) << "Mod's denominator is not a const_int \n";
  return {};
}

std::vector<isl::aff> Select2AffBounds(const isl::space &space, const Select *sel) {
  /**************************************
   * Support Select expression aff bounds computation
   * select((15 < int32(ceil((float32(w)*5.000000f)))), 15, int32(ceil((float32(w)*5.000000f))))
   **************************************/
  auto true_aff_bounds = Expr2AffBounds(space, sel->true_value, false, false);
  auto false_aff_bounds = Expr2AffBounds(space, sel->false_value, false, false);
  if (true_aff_bounds.size() == 0 || false_aff_bounds.size() == 0) {
    return {};
  }
  /********************************************************
   * temp method just add true_value aff and false_value aff
   *******************************************************/
  return {(true_aff_bounds[0].add)(false_aff_bounds[0])};
}

std::vector<isl::aff> Expr2AffBounds(const isl::space &space, const Expr &e, bool allow_min, bool allow_max,
                                     bool ignore_error) {
  CHECK(!(allow_min && allow_max));

  if (ExprType<Variable>(e)) {
    return Variable2AffBounds(space, e.as<Variable>(), ignore_error);
  } else if (const int64_t *i = as_const_int(e)) {
    return {Int2Aff(space, *i)};
  } else if (ExprType<FloatImm>(e)) {
    return {Int2Aff(space, int64_t(e.as<FloatImm>()->value))};
  } else if (ExprType<Cast>(e)) {
    return Expr2AffBounds(space, e.as<Cast>()->value, false, false);
  } else if (const auto call = e.as<Call>()) {
    if ((call->name == "floor" || call->name == "ceil") && call->args.size() == 1) {
      return Expr2AffBounds(space, call->args[0], false, false);
    }
    LOG(INFO) << "not parse call type: " << call->name << " with expr :" << e;
  } else if (ExprType<Min>(e)) {
    if (!allow_min) return {};
    return ConcatAffs(space, e.as<Min>(), allow_min, allow_max);
  } else if (ExprType<Max>(e)) {
    if (!allow_max) return {};
    return ConcatAffs(space, e.as<Max>(), allow_min, allow_max);
  } else if (ExprType<Add>(e)) {
    return UniteAffs(space, e.as<Add>(), &isl::aff::add);
  } else if (ExprType<Sub>(e)) {
    return UniteAffs(space, e.as<Sub>(), &isl::aff::sub);
  } else if (ExprType<Mul>(e)) {
    return UniteAffs(space, e.as<Mul>(), &isl::aff::mul);
  } else if (ExprType<FloorDiv>(e)) {
    return FloorDiv2AffBounds(space, e.as<FloorDiv>());
  } else if (ExprType<Div>(e)) {
    return Div2AffBounds(space, e.as<Div>());
  } else if (ExprType<FloorMod>(e)) {
    return FloorMod2AffBounds(space, e.as<FloorMod>(), ignore_error);
  } else if (ExprType<Mod>(e)) {
    return Mod2AffBounds(space, e.as<Mod>(), ignore_error);
  } else if (ExprType<Select>(e)) {
    return Select2AffBounds(space, e.as<Select>());
  }

  CHECK(ignore_error) << "Expr2AffBounds " << e << "\n";
  return {};
}

std::vector<isl::aff> Expr2AffChecked(const isl::space &space, const Expr &e, bool allow_min, bool allow_max) {
  bool ignore_error = false;
  return Expr2AffBounds(space, e, allow_min, allow_max, ignore_error);
}

isl::aff Expr2Aff(const isl::space &space, const Expr &e) {
  auto list = Expr2AffChecked(space, e, false, false);
  return list.empty() ? isl::aff() : list[0];
}

isl::multi_id CollectTensorCoordinate(const isl::space &pspace, const isl::id &id, size_t dim) {
  isl::id_list args(pspace.ctx(), 0);
  for (size_t i = 0; i < dim; ++i) {
    auto name = std::string("arg") + std::to_string(i);
    args = args.add(isl::id(pspace.ctx(), name));
  }
  return isl::multi_id(pspace.add_named_tuple_id_ui(id, static_cast<unsigned int>(dim)), args);
}

isl::map AddSuffix4Accesses(AccessMap &accesses, const isl::map &in_map, const Node *op, const isl::ctx &ctx) {
  auto tensor_map = in_map;

  // Based on different condition, add suffix to the domain space
  std::string suffix;
  if (accesses.count(op) > 0) {  // reuse existing tag if the op is accessed previously
    suffix = accesses[op].to_str();
  } else {  // create a new tag with unique name
    suffix = "__poly_ref_" + std::to_string(accesses.size());
  }

  isl::id suffix_id(ctx, suffix);
  if (accesses.count(op) == 0) {  // only insert, not replace
    accesses.emplace(op, suffix_id);
  }

  auto domain_space = tensor_map.get_space().domain();
  auto tag_space = domain_space.params().add_named_tuple_id_ui(suffix_id, 0);
  domain_space = domain_space.product(tag_space).unwrap();
  tensor_map = tensor_map.preimage_domain(isl::multi_aff::domain_map(domain_space));

  return tensor_map;
}

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

isl::union_pw_aff GetUnionPwAffAtDomain(const isl::aff &f, const isl::union_set &domain, const OperatorDomainMap &map) {
  auto upa = isl::union_pw_aff::empty(domain.space());
  for (auto set : domain.get_set_list()) {
    upa = upa.union_add(isl::union_pw_aff(f.unbind_params_insert_domain(map.at(set.tuple_id()).tuple)));
  }
  return upa;
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

static const char kStatementLabel[] = "S_";

bool ParseWithStmt(const Expr &s, const Scop::Data &data) {
  class ParseWith final : public IRVisitor {
   public:
    void Visit_(const Call *op) final {
      if (!find_tensor && (0 != writes.size())) {
        if (op->call_type == Call::Halide) {
          if (writes.find(op->name) != writes.end()) {
            find_tensor = true;
          }
        }
      }
      IRVisitor::Visit_(op);
    }
    bool find_tensor{false};
    std::unordered_set<std::string> writes;

    bool GetResult() const { return find_tensor; }

    ParseWith(const Expr &stmt, const Scop::Data &data) {
      data.writes.foreach_map([&, this](const isl::map m) -> void {
        writes.insert(m.get_tuple_id(isl_dim_out).get_name());
        return;
      });
      IRVisitor::Visit(stmt);
    }
    ~ParseWith() override = default;
  } paserWith(s, data);

  return paserWith.GetResult();
}

std::map<std::string, PolyOpType> call_op_ = {
  {"log", PolyOpType::elewise_single_log},
  {"exp", PolyOpType::elewise_single_exp},
  {"sqrt", PolyOpType::elewise_single_sqrt},
  {"rsqrt", PolyOpType::elewise_single_rsqrt},
  {"fabs", PolyOpType::elewise_single_fabs},
  {"rec", PolyOpType::elewise_single_rec},
  {"floor", PolyOpType::vec_single_floor},
  {"round", PolyOpType::vec_single_round},
  {"ceil", PolyOpType::elewise_single_ceil},
  {"trunc", PolyOpType::vec_single_trunc},
  {"not", PolyOpType::elewise_single_not},
  {"relu", PolyOpType::elewise_single_relu},
  {"EQ", PolyOpType::elewise_binary_EQ},
  {"NE", PolyOpType::elewise_binary_NE},
  {"GT", PolyOpType::elewise_binary_GT},
  {"GE", PolyOpType::elewise_binary_GE},
  {"LT", PolyOpType::elewise_binary_LT},
  {"LE", PolyOpType::elewise_binary_LE},
  {"fargmax", PolyOpType::vec_argmax},
  {"fargmin", PolyOpType::vec_argmin},
  {"four2five_nchw", PolyOpType::four2five_nchw},
  {"vand", PolyOpType::elewise_binary_and},
  {"bitwise_and", PolyOpType::elewise_binary_bitwise_and},
  {"bitwise_or", PolyOpType::elewise_binary_bitwise_or},
  {"bitwise_not", PolyOpType::elewise_single_bitwise_not},
  {"proposal_sort", PolyOpType::elewise_binary_proposal_sort},
  {"topk_sort", PolyOpType::elewise_binary_topk_sort},
  {"nms", PolyOpType::elewise_binary_nms},
  {"dropout", PolyOpType::elewise_binary_dropout},
  {"iou", PolyOpType::elewise_binary_iou},
  {"vmadd", PolyOpType::vmadd},
  {"vmaddrelu", PolyOpType::vmaddrelu},
  {"vaxpy", PolyOpType::vaxpy},
  {"vmla", PolyOpType::vmla},
};

void ParseStmtOpCall(const isl::id &id, const Call *call, Scop::Data &data, const FunctionRef &func) {
  CHECK(call);
  if (call->call_type == Call::PureIntrinsic) {
    if (call_op_.count(call->name) > 0) {
      data.stmt_op_Info.at(id).ops.push_back(call_op_[call->name]);
    } else if (0 == strcmp(call->name.c_str(), "with")) {
      data.stmt_op_Info.at(id).ops.push_back(PolyOpType::with);
      if (!data.stmt_op_Info.at(id).isWith) {
        for (unsigned i = 0; i < call->args.size(); ++i) {
          if (ParseWithStmt(call->args[i], data)) {
            data.stmt_op_Info.at(id).isWith = true;
            break;
          }
        }
      }
    } else if (0 == strcmp(call->name.c_str(), "reshape")) {
      // do nothing
    } else if (0 == strcmp(call->name.c_str(), "transpose")) {
      // do nothing
    } else if (0 == strcmp(call->name.c_str(), "divide_var")) {
      // do nothing
    } else if (0 == strcmp(call->name.c_str(), "sub_relu")) {
      // do nothing
    } else if (0 == strcmp(call->name.c_str(), "load3d_l1_ub")) {
      data.stmt_op_Info.at(id).isLoad3d = true;
      ParseStmtOps(id, call->args[0], data, func);
    } else if (0 == strcmp(call->name.c_str(), "mad")) {
      data.stmt_op_Info.at(id).ops.push_back(PolyOpType::mad);
      data.stmt_op_Info.at(id).isCube = true;
      // assign + mad
      std::string name = id.get_name();
      size_t index = static_cast<size_t>(WrappedStrtol(name.substr(name.length() - 1)));
      std::string tmp = name.substr(0, name.length() - 1);
      std::stringstream ss;
      ss << tmp << index - 1;
      if (data.stmt_op_Info.count(isl::id(id.ctx(), ss.str())) > 0 &&
          data.stmt_op_Info.at(isl::id(id.ctx(), ss.str())).ops[0] == PolyOpType::broadcast)
        data.stmt_op_Info.at(isl::id(id.ctx(), ss.str())).isCubeAssign = true;
      // end
      data.stmt_op_Info.at(id).C_ = func->func_name();
      CHECK(call->args.size() == 2) << "invalid args of mad! ";
      auto mul_arg = call->args[0].as<Mul>() ? call->args[0].as<Mul>() : call->args[1].as<Mul>();

      if (call->args[1].as<Cast>()) {
        CHECK(call->args[1].as<Cast>()->value.as<Mul>());
        mul_arg = call->args[1].as<Cast>()->value.as<Mul>();
      }
      CHECK(mul_arg);
      auto a = mul_arg->a.as<Call>();
      auto b = mul_arg->b.as<Call>();
      // in gemm case, C = mad(C,  A * B)
      if (a && b) {
        data.stmt_op_Info.at(id).A_ = a->name;
        data.stmt_op_Info.at(id).B_ = b->name;
      }
      // in conv case, reassign A&B by attr
      if (func.as<ComputeOpNode>() != nullptr) {
        data.stmt_op_Info.at(id).MadType_ = call->args[1].as<Cast>() ? call->args[1].as<Cast>()->type : Float(16);
        for (auto i : func.as<ComputeOpNode>()->attrs) {
          if ("feature" == i.first) {
            data.stmt_op_Info.at(id).A_ = i.second.as<StringImm>()->value;
          }
          if ("filter" == i.first) {
            data.stmt_op_Info.at(id).B_ = i.second.as<StringImm>()->value;
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown pure intrinsic: " << call->name.c_str() << std::endl;
    }
  }
}

void ParseStmtOps(const isl::id &id, const Expr &val, Scop::Data &data, const FunctionRef &func) {
  data.stmt_op_Info.at(id).isCube = false;
  data.stmt_op_Info.at(id).isCubeAssign = false;
  if (auto add = val.as<Add>()) {
    if (isImm(add->a) || isImm(add->b)) {
      if (!isImm(add->a)) {  // if add->a is not a scalar, then put it into recursion
        ParseStmtOps(id, add->a, data, func);
      } else if (!isImm(add->b)) {  // if add->b is not a scalar, then put it into recursion
        ParseStmtOps(id, add->b, data, func);
      } else {  // if add->a and add->b are both scalar, then report error
        LOG(FATAL) << "Error: Scalar + Scalar, Please Check.";
      }
      data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_single_VS_add);
    } else {
      ParseStmtOps(id, add->a, data, func);
      ParseStmtOps(id, add->b, data, func);
      data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_add);
    }
  } else if (auto sub = val.as<Sub>()) {
    ParseStmtOps(id, sub->a, data, func);
    ParseStmtOps(id, sub->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_sub);
  } else if (auto mul = val.as<Mul>()) {
    if (isImm(mul->a) || isImm(mul->b)) {
      // if mul->a is not a scalar, then put it into recursion
      if (!isImm(mul->a)) {
        ParseStmtOps(id, mul->a, data, func);
      } else if (!isImm(mul->b)) {  // if mul->b is not a scalar, then put it into recursion
        ParseStmtOps(id, mul->b, data, func);
      } else {  // if mul->a and mul->b are both scalar, then report error
        LOG(FATAL) << "Error: Scalar + Scalar, Please Check.";
      }
      if (isZero(mul->b) || isZero(mul->a)) {
        data.stmt_op_Info.at(id).ops.push_back(PolyOpType::broadcast);
      } else {
        data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_single_VS_mul);
      }
    } else {
      ParseStmtOps(id, mul->a, data, func);
      ParseStmtOps(id, mul->b, data, func);
      data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_mul);
    }
  } else if (auto f_div = val.as<FloorDiv>()) {
    ParseStmtOps(id, f_div->a, data, func);
    ParseStmtOps(id, f_div->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_div);
  } else if (auto f_mod = val.as<FloorMod>()) {
    ParseStmtOps(id, f_mod->a, data, func);
    ParseStmtOps(id, f_mod->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_mod);
  } else if (auto div = val.as<Div>()) {
    ParseStmtOps(id, div->a, data, func);
    ParseStmtOps(id, div->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_div);
  } else if (auto mod = val.as<Mod>()) {
    ParseStmtOps(id, mod->a, data, func);
    ParseStmtOps(id, mod->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_mod);
  } else if (auto and_op = val.as<And>()) {
    ParseStmtOps(id, and_op->a, data, func);
    ParseStmtOps(id, and_op->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_and);
  } else if (auto or_op = val.as<Or>()) {
    ParseStmtOps(id, or_op->a, data, func);
    ParseStmtOps(id, or_op->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_or);
  } else if (auto min = val.as<Min>()) {
    ParseStmtOps(id, min->a, data, func);
    ParseStmtOps(id, min->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_min);
  } else if (auto max = val.as<Max>()) {
    ParseStmtOps(id, max->a, data, func);
    ParseStmtOps(id, max->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_max);
  } else if (auto ge = val.as<GE>()) {
    ParseStmtOps(id, ge->a, data, func);
    ParseStmtOps(id, ge->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto gt = val.as<GT>()) {
    ParseStmtOps(id, gt->a, data, func);
    ParseStmtOps(id, gt->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto le = val.as<LE>()) {
    ParseStmtOps(id, le->a, data, func);
    ParseStmtOps(id, le->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto lt = val.as<LT>()) {
    ParseStmtOps(id, lt->a, data, func);
    ParseStmtOps(id, lt->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto eq = val.as<EQ>()) {
    ParseStmtOps(id, eq->a, data, func);
    ParseStmtOps(id, eq->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto ne = val.as<NE>()) {
    ParseStmtOps(id, ne->a, data, func);
    ParseStmtOps(id, ne->b, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if ((isImm(val) || val.type().is_int()) && val.as<Call>() == nullptr) {
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::broadcast);
  } else if (auto sel = val.as<Select>()) {
    ParseStmtOps(id, sel->true_value, data, func);
    ParseStmtOps(id, sel->false_value, data, func);
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::pandora_select);
  } else if ((val.as<Cast>())) {
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::vec_single_cast);
  } else if (auto call = val.as<Call>()) {
    ParseStmtOpCall(id, call, data, func);
  } else {
    LOG(WARNING) << "====>> WARNING: operator unknown type! " << val << " type:" << val.type();
    data.stmt_op_Info.at(id).ops.push_back(PolyOpType::elewise_binary_unknown);
  }
}

void ParseStmtOps(const isl::id &id, const Evaluate *stmt, Scop::Data &data, const isl::union_map &new_reads,
                  const isl::union_map &new_writes) {
  if (!stmt) return;
  StmtOpInfo stmt_op_Info;
  for (auto a : new_reads.get_map_list()) {
    auto tensor_id = a.get_tuple_id(isl_dim_out);
    stmt_op_Info.readtensors.push_back(tensor_id);
  }

  if (stmt->value.as<Call>() && stmt->value.as<Call>()->name == CALL_IM2COL_UB) {
    stmt_op_Info.ops.push_back(PolyOpType::im2col);
    stmt_op_Info.isIm2col = true;
  }

  if (data.stmt_op_Info.count(id)) {
    // cce_img2col_ has no read_tensors and no write_tensors
    // maybe no need follows
    auto oldReadTensors = data.stmt_op_Info[id].readtensors;
    for (auto i : oldReadTensors) {
      stmt_op_Info.readtensors.push_back(i);
    }
    data.stmt_op_Info[id] = stmt_op_Info;
  } else {
    data.stmt_op_Info.emplace(id, stmt_op_Info);
  }
}

void ParseStmtOps(const isl::id &id, const Provide *stmt, Scop::Data &data, const isl::union_map &new_reads,
                  const isl::union_map &new_writes) {
  if (!stmt) return;
  StmtOpInfo stmt_op_Info;
  for (auto a : new_reads.get_map_list()) {
    auto tensor_id = a.get_tuple_id(isl_dim_out);
    stmt_op_Info.readtensors.push_back(tensor_id);
  }
  if ((stmt->value.as<Call>() && stmt->value.as<Call>()->call_type == Call::Halide)) {
    stmt_op_Info.ops.push_back(PolyOpType::assignment);
  }
  if (data.stmt_op_Info.count(id)) {
    // one id has multiple statements with readtensors
    auto oldReadTensors = data.stmt_op_Info[id].readtensors;
    for (auto i : oldReadTensors) {
      stmt_op_Info.readtensors.push_back(i);
    }
    data.stmt_op_Info[id] = stmt_op_Info;
  } else {
    data.stmt_op_Info.emplace(id, stmt_op_Info);
  }
  ParseStmtOps(id, stmt->value, data, stmt->func);
}

/* "macro_stmt" is introduced to handle IfThenElse IR Node type. In particular, an if statement with its body should
 * be handled as one "macro" statement. "macro_stmt" records the statement label of the entire "macro" statements. In
 * other words, each "kStatementLabel" of a Provide or Store type should be the same with its enclosing IfThenElse
 * statement, equal to "macro_stmt".
 *
 * "macro_stmt" should be set to -1 to represent scop building should follow the regular way, i.e., each statement
 * should be analyzed individually, and it may be updated to a nonnegative number once an IfThenElse Node type is
 * encountered.
 *
 * The first step is to update the reads/writes of scop by extracting access information in the conditional of an
 * IfThenElse type node, followed by the building of the then_case (and also else case) schedule tree, which only
 * updates reads/writes sets without updating schedule tree.
 *
 * Currently, the loop variables of those enclosed by an if statement appear as parameters of the schedule tree. This
 * may be updated in the future for better manipulation of schedule trees.
 */

isl::set CutSet(std::vector<Expr> cond_vec, const isl::set &set, bool is_else = false, bool is_or = false) {
  if (cond_vec.empty()) return set;

  isl::space space = set.get_space();
  std::vector<isl::set> set_vec;

  for (size_t index = 0; index < cond_vec.size(); index++) {
    auto i = cond_vec[index];
    if (const LT *lt = i.as<LT>()) {
      auto left = Expr2AffBounds(space, lt->a, false, true);
      auto right = Expr2AffBounds(space, lt->b, true, false);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].ge_set(right[0]));
      else
        set_vec.push_back(left[0].lt_set(right[0]));
    } else if (const LE *le = i.as<LE>()) {
      auto left = Expr2AffBounds(space, le->a, false, true);
      auto right = Expr2AffBounds(space, le->b, true, false);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].gt_set(right[0]));
      else
        set_vec.push_back(left[0].le_set(right[0]));
    } else if (const GT *gt = i.as<GT>()) {
      auto left = Expr2AffBounds(space, gt->a, true, false);
      auto right = Expr2AffBounds(space, gt->b, false, true);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].le_set(right[0]));
      else
        set_vec.push_back(left[0].gt_set(right[0]));
    } else if (const GE *ge = i.as<GE>()) {
      auto left = Expr2AffBounds(space, ge->a, true, false);
      auto right = Expr2AffBounds(space, ge->b, false, true);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].lt_set(right[0]));
      else
        set_vec.push_back(left[0].ge_set(right[0]));
    } else if (const EQ *eq = i.as<EQ>()) {
      auto left = Expr2AffBounds(space, eq->a, false, false);
      auto right = Expr2AffBounds(space, eq->b, false, false);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].ne_set(right[0]));
      else
        set_vec.push_back(left[0].eq_set(right[0]));
    } else if (const NE *ne = i.as<NE>()) {
      auto left = Expr2AffBounds(space, ne->a, false, false);
      auto right = Expr2AffBounds(space, ne->b, false, false);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].eq_set(right[0]));
      else
        set_vec.push_back(left[0].ne_set(right[0]));
    } else if (const And *and_op = i.as<And>()) {
      cond_vec.push_back(and_op->a);
      cond_vec.push_back(and_op->b);
    } else {
      CHECK(false) << " find unknown conditions: " << i;
    }
  }
  for (size_t i = 1; i < set_vec.size(); ++i) {
    if (is_or)
      set_vec[0] = set_vec[0].unite(set_vec[i]);
    else
      set_vec[0] = set_vec[0].intersect(set_vec[i]);
  }
  return set.intersect(set_vec[0]);
}

isl::schedule MakeScheduleTreeHelper(const NodeRef &s, Scop &scop, const isl::set &set, const isl::id_list &outer,
                                     ssize_t macro_stmt) {
  class ExtractCond : protected IRVisitor {
   public:
    ExtractCond() {}
    ~ExtractCond() override = default;
    std::vector<Expr> run(const Expr expr) {
      IRVisitor::Visit(Simplify_cce(expr));
      return result;
    }
    bool hasBothOrAndAnd() const { return (or_num && and_num); }
    bool IsOr() {
      if (!or_num && !and_num && result.size() > 1) {
        LOG(INFO) << "  result.size() > 1 and or(and)_num = 0";
      }
      return (or_num > 0);
    }
    std::vector<Expr> result;

   protected:
#define COMOP_VISIT_(OP)                    \
  void Visit_(const OP *op) final {         \
    has_tensor = false;                     \
    this->Visit(op->a);                     \
    this->Visit(op->b);                     \
    if (!has_tensor) {                      \
      Expr expr(GetRef<Expr>(op));          \
      result.push_back(Simplify_cce(expr)); \
    }                                       \
  }
    COMOP_VISIT_(EQ)
    COMOP_VISIT_(NE)
    COMOP_VISIT_(LT)
    COMOP_VISIT_(LE)
    COMOP_VISIT_(GT)
    COMOP_VISIT_(GE)

    void Visit_(const Call *op) final {
      IRVisitor::Visit_(op);
      if (op->call_type == Call::Halide) {
        has_tensor = true;
      }
    }
    void Visit_(const And *op) final {
      and_num++;
      this->Visit(op->a);
      this->Visit(op->b);
    }

    void Visit_(const Or *op) final {
      or_num++;
      this->Visit(op->a);
      this->Visit(op->b);
    }

    void Visit_(const Not *op) final {
      Expr expr(GetRef<Expr>(op));
      LOG(FATAL) << expr << " so far NOT is handled, please modify DSL";
    }

   private:
    int or_num{0};
    int and_num{0};
    bool has_tensor{false};
  };

  class MakeScheduleTree final : protected IRVisitor {
   public:
    MakeScheduleTree(const NodeRef s, Scop &scop, const isl::set set, const isl::id_list outer, ssize_t macro_stmt)
        : s(s), scop(scop), set(set), outer(outer), macro_stmt(macro_stmt) {
      IRVisitor::Visit(s);
    }
    ~MakeScheduleTree() override = default;

    const NodeRef s;
    Scop &scop;
    isl::set set;
    isl::id_list outer;
    isl::schedule sch;
    bool found{false};

    ssize_t macro_stmt{-1};

    /// Visitor implementation
    //@{
    void Visit_(const Provide *op) final {
      {
        size_t stmt_index = scop.data_.statements.size();
        isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                              : kStatementLabel + std::to_string(stmt_index));
        scop.data_.statements.emplace(id, op);
        auto tuple_space = isl::space(set.ctx(), 0);
        tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));
        OperatorDomainSpace op_domain;
        op_domain.param_space = set.get_space();
        op_domain.tuple = isl::multi_id(tuple_space, outer);
        scop.data_.domains.emplace(id, op_domain);
        auto domain = set.unbind_params(op_domain.tuple);
        sch = isl::schedule::from_domain(domain);

        isl::union_map new_reads, new_writes, new_to_inner;
        isl::union_map new_reads_with_conds, newWrites_with_conds;
        isl::set read_set = set;
        isl::set write_set = set;
        Stmt stmt = Downcast<Stmt>(s);
        std::tie(new_reads, new_writes, new_to_inner) = ConstructPolyAccesses(op_domain, stmt, scop.data_.accesses);

        new_reads_with_conds = new_reads.curry().intersect_domain(read_set.unbind_params(op_domain.tuple)).uncurry();
        /// has Select
#if (SELECT_DOMAIN_OPT)
        class CutSetTopDown final : protected IRVisitor {
         public:
          CutSetTopDown() {}
          ~CutSetTopDown() override = default;

          const isl::union_map Run(const Expr &expr, const isl::multi_id &tuple_, const isl::union_map &accesses_,
                                   const isl::set &read_set_) {
            accesses = accesses_;
            read_set = read_set_;
            tuple = tuple_;
            Visit(expr);
            return accesses;
          }

         private:
          static std::unordered_set<std::string> GatherCallTensors(const Expr &e) {
            std::unordered_set<std::string> tensor_names;
            PostOrderVisit(e, [&](const NodeRef &node) -> void {
              if (auto op = node.as<Call>()) {
                if (op->call_type == Call::CallType::Halide) {
                  tensor_names.insert(op->func->func_name());
                }
              }
            });
            return tensor_names;
          }

          void CutAccesses(const Expr &value, const std::vector<Expr> &conds, bool is_else, bool is_or) {
            auto may_access_tensors = GatherCallTensors(value);
            isl::union_map must_access = isl::union_map::empty(accesses.space());
            isl::union_map may_access = isl::union_map::empty(accesses.space());
            accesses.foreach_map([&](const isl::map &map) {
              auto tensor = map.get_tuple_id(isl_dim_out).get_name();
              if (may_access_tensors.count(tensor) == 0) {
                must_access = must_access.add_map(map);
              } else {
                may_access = may_access.add_map(map);
              }
            });
            read_set = CutSet(conds, read_set, is_else, is_or);
            auto cut_may_access = may_access.curry().intersect_domain(read_set.unbind_params(tuple)).uncurry();
            accesses = must_access.unite(cut_may_access);
          }

          void Visit_(const Select *sel) final {
            auto ec = ExtractCond();
            std::vector<Expr> conds = ec.run(sel->condition);
            if (!ec.hasBothOrAndAnd()) {
              if (isImm(sel->true_value)) {
                CutAccesses(sel->false_value, conds, true, !ec.IsOr());
              } else if (isImm(sel->false_value)) {
                CutAccesses(sel->true_value, conds, false, ec.IsOr());
              }
            }
          }

          isl::union_map accesses;
          isl::set read_set;
          isl::multi_id tuple;
        };

        new_reads_with_conds = CutSetTopDown().Run(op->value, op_domain.tuple, new_reads_with_conds, read_set);
#endif
        newWrites_with_conds = new_writes.curry().intersect_domain(write_set.unbind_params(op_domain.tuple)).uncurry();

        ParseStmtOps(id, op, scop.data_, new_reads, new_writes);

        // The parameters should be added as constraints of the reads/writes sets
        // otherwise isl may not be able to obtain a fixed box.
        if (macro_stmt >= 0) {
          auto params = domain.params();
          new_reads = new_reads.curry().intersect_domain(params).uncurry();
          new_writes = new_writes.curry().intersect_domain(params).uncurry();

          new_reads_with_conds = new_reads_with_conds.curry().intersect_domain(params).uncurry();
          newWrites_with_conds = newWrites_with_conds.curry().intersect_domain(params).uncurry();
        }

        scop.data_.reads = scop.data_.reads.unite(new_reads_with_conds);
        scop.data_.writes = scop.data_.writes.unite(newWrites_with_conds);
        found = true;
      }
    }

    void Visit_(const Block *op) final {
      auto sch_first = MakeScheduleTreeHelper(op->first, scop, set, outer, macro_stmt);
      auto sch_rest = MakeScheduleTreeHelper(op->rest, scop, set, outer, macro_stmt);
      if (macro_stmt >= 0)
        sch = sch_first;
      else
        sch = sch_first.sequence(sch_rest);
      found = true;
    }

    void Visit_(const IfThenElse *op) final {
      Expr cond = op->condition;

      size_t stmt_index = scop.data_.statements.size();
      isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                            : kStatementLabel + std::to_string(stmt_index));
      scop.data_.statements.emplace(id, op);
      auto tuple_space = isl::space(set.ctx(), 0);
      tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));
      OperatorDomainSpace op_domain;
      op_domain.param_space = set.get_space();
      op_domain.tuple = isl::multi_id(tuple_space, outer);
      scop.data_.domains.emplace(id, op_domain);
      auto domain = set.unbind_params(op_domain.tuple);
      sch = isl::schedule::from_domain(domain);

      isl::union_map new_reads, new_writes, new_to_inner;

      // Update the reads/writes sets of scop by analyzing the condition
      Stmt condition = Stmt(GetObjPtr<Object>(cond.get()));
      std::tie(new_reads, new_writes, new_to_inner) = ConstructPolyAccesses(op_domain, condition, scop.data_.accesses);
      StmtOpInfo stmt_op_Info;
      for (auto a : new_reads.get_map_list()) {
        auto tensor_id = a.get_tuple_id(isl_dim_out);
        stmt_op_Info.readtensors.push_back(tensor_id);
      }
      scop.data_.stmt_op_Info.emplace(id, stmt_op_Info);
      ParseStmtOps(id, cond, scop.data_, FunctionRef(GetObjPtr(cond.get())));

      scop.data_.reads = scop.data_.reads.unite(new_reads);
      scop.data_.writes = scop.data_.writes.unite(new_writes);

      // Update the flag for recording a macro statement
      if (macro_stmt < 0) macro_stmt = static_cast<int64_t>(stmt_index);
      // Build schedule for the then case without updating the schedule

      isl::set cut_set = set;

#if (SELECT_DOMAIN_OPT)
      auto ec = ExtractCond();
      std::vector<Expr> cond_vec = ec.run(cond);
      if (!ec.hasBothOrAndAnd()) {
        cut_set = CutSet(cond_vec, set, false, ec.IsOr());
      }
#endif
      static_cast<void>(MakeScheduleTreeHelper(op->then_case, scop, cut_set, outer, macro_stmt));

      // Build schedule for the else case without updating the schedule if defined
      if (op->else_case.defined()) {
#if (SELECT_DOMAIN_OPT)
        if (!ec.hasBothOrAndAnd()) {
          cut_set = CutSet(cond_vec, set, true, !ec.IsOr());
        }
#endif
        static_cast<void>(MakeScheduleTreeHelper(op->else_case, scop, cut_set, outer, macro_stmt));
      }

      found = true;
    }

    void Visit_(const Evaluate *op) final {
      const Call *call_op = op->value.as<Call>();
      if (call_op && call_op->name == CALL_IM2COL_UB) {
        size_t stmt_index = scop.data_.statements.size();
        isl::id id(set.ctx(), macro_stmt >= 0 ? kStatementLabel + std::to_string(macro_stmt)
                                              : kStatementLabel + std::to_string(stmt_index));
        scop.data_.statements.emplace(id, op);
        auto tuple_space = isl::space(set.ctx(), 0);
        tuple_space = tuple_space.add_named_tuple_id_ui(id, static_cast<unsigned int>(outer.size()));

        OperatorDomainSpace op_domain;
        op_domain.param_space = set.get_space();
        op_domain.tuple = isl::multi_id(tuple_space, outer);
        scop.data_.domains.emplace(id, op_domain);

        auto domain = set.unbind_params(op_domain.tuple);
        sch = isl::schedule::from_domain(domain);

        isl::union_map new_reads, new_writes, new_to_inner;
        Stmt stmt = Downcast<Stmt>(s);
        for (auto item : scop.data_.attrs) {
          if (item->attr_key == ATTR_IM2COL_KEY) {
            stmt = AttrStmt::make(item->node, item->attr_key, item->value, stmt);
          }
        }
        std::tie(new_reads, new_writes, new_to_inner) = ConstructPolyAccesses(op_domain, stmt, scop.data_.accesses);

        ParseStmtOps(id, op, scop.data_, new_reads, new_writes);

        if (macro_stmt >= 0) {
          auto params = domain.params();
          new_reads = new_reads.curry().intersect_domain(params).uncurry();
          new_writes = new_writes.curry().intersect_domain(params).uncurry();
          new_to_inner = new_to_inner.curry().intersect_domain(params).uncurry();
        }
        scop.data_.reads = scop.data_.reads.unite(new_reads);
        scop.data_.writes = scop.data_.writes.unite(new_writes);
        found = true;
      }
    }

    void AddLoopBoundConstraints(const isl::aff &loop_var, const isl::space &space, const Expr &expr, bool permit_min,
                                 bool permit_max) {
      auto constraint_bounds = Expr2AffChecked(space, expr, permit_min, permit_max);
      if (constraint_bounds.size() == 0u) LOG(INFO) << "could not obtain polyhedral lower / upper bounds from " << expr;
      for (const auto &item : constraint_bounds) {
        if (!permit_min && permit_max) {
          set = set.intersect(loop_var.ge_set(item));
        } else if (permit_min && !permit_max) {
          set = set.intersect(item.ge_set(loop_var));
        }
      }
    }

    void Visit_(const For *op) final {
      auto loop_var_id = isl::id(set.ctx(), op->loop_var->name_hint);
      auto space = set.get_space().add_param(loop_var_id);

      auto loop_var = isl::aff::param_on_domain(space, loop_var_id);

      // Add lower/upper loop bound constraints.
      AddLoopBoundConstraints(loop_var, space, op->min, false, true);
      Expr max = Simplify_cce(op->min + op->extent - 1);
      AddLoopBoundConstraints(loop_var, space, max, true, false);

      auto outer_add = outer.add(loop_var_id);
      auto outer_list = macro_stmt >= 0 ? outer : outer_add;
      auto body_schedule = MakeScheduleTreeHelper(op->body, scop, set, outer_list, macro_stmt);

      auto multi_union_pw_aff_func = isl::multi_union_pw_aff(GetUnionPwAffAtDomain(
        isl::aff::param_on_domain(space, loop_var_id), body_schedule.get_domain(), scop.data_.domains));

      sch = body_schedule.insert_partial_schedule(multi_union_pw_aff_func);
      found = true;
    }

    void Visit_(const Realize *op) final {
      IRVisitor::Visit_(op);
      auto name = op->func->func_name();
      CHECK_EQ(op->func->num_outputs(), 1);
      auto type = op->type;

      /// add old realize
      scop.realize_from_input_.insert(isl::id(scop.ctx_, name));

      for (auto i : scop.binds_) {
        if (i.first->op->name == name) return;
      }

      // add Realize's buf into binds
      Array<Expr> shapes;
      for (auto i : op->bounds) {
        shapes.push_back(i->extent);
      }
      Tensor tensor = placeholder(shapes, type, name);
      const Buffer buffer = decl_buffer(shapes, type, name);
      scop.binds_.Set(tensor, buffer);
    }

    void Visit_(const ProducerConsumer *op) final {
      sch = MakeScheduleTreeHelper(op->body, scop, set, outer, macro_stmt);
      found = true;
    }

    void Visit_(const AttrStmt *op) final {
      if (op->attr_key == air::ir::attr::reduce_update) {
        Array<IterVar> red = Downcast<Array<IterVar>>(op->node);
        const auto pro = op->body.as<Provide>();
        if (pro) {
          scop.data_.reduces.emplace(pro, red);
        } else {
          auto blo = op->body.as<Block>();
          if (blo) {
            while (blo->rest.defined() && blo->rest.as<Block>()) {
              blo = blo->rest.as<Block>();
            }
            const auto pro_first = blo->first.as<Provide>();
            const auto pro_rest = blo->rest.as<Provide>();
            if (pro_rest) {
              scop.data_.reduces.emplace(pro_rest, red);
            } else if (pro_first) {
              scop.data_.reduces.emplace(pro_first, red);
            }
          }
        }
      } else if (op->attr_key == air::ir::attr::buffer_bind_scope) {
        /* ******************************************
         * parse attr like below
         * // attr [[buffer(Abuf, 0x29ff3b0), Tensor(shape=[1, 32, 7, 7, 16], op.name=fmap)]] buffer_bind_scope =
         * tvm_tuple(0, 1, floordiv(k, 9), 1, ((j*16)/5), 3, 0, 7, 0, 16):handle:I
         * *******************************************/
        Array<NodeRef> array = Downcast<Array<NodeRef>>(op->node);
        Buffer buffer = Downcast<Buffer>(array[0]);
        Tensor tensor = Downcast<Tensor>(array[1]);
        Array<NodeRef> update_array;
        std::string update_name = tensor->op->name;
        std::string update_scope;
        if (tensor->op.as<PlaceholderOpNode>()) {
          update_name += "_local_L1";
          update_scope = "local.L1";
        } else {
          update_name += "_local_UB";
          update_scope = "local.UB";
        }
        Buffer update_buffer = BufferNode::make(buffer->data, buffer->dtype, buffer->shape, buffer->strides,
                                                buffer->elem_offset, buffer->name, update_scope, buffer->data_alignment,
                                                buffer->offset_factor, buffer->buffer_type);
        Tensor update_tensor = placeholder(tensor->shape, tensor->dtype, update_name);
        update_array.push_back(update_buffer);
        update_array.push_back(update_tensor);
        scop.data_.update_tensors.push_back(update_tensor);
        scop.data_.vecs.push_back(std::make_pair(update_array, op->value));
        scop.data_.accesses.emplace(op, isl::id(set.ctx(), tensor->op->name));
      } else if (op->attr_key == ATTR_IM2COL_KEY) {
        scop.data_.attrs.push_back(op);
      }

      sch = MakeScheduleTreeHelper(op->body, scop, set, outer, macro_stmt);
      found = true;
    }
  };

  MakeScheduleTree schedule_tree(s, scop, set, outer, macro_stmt);
  if (!schedule_tree.found) {
    LOG(FATAL) << "Unhandled " << s.get()->GetTypeKey() << " : " << s;
  }
  return schedule_tree.sch;
}

isl::schedule MakeScheduleTree(const isl::space &param_space, isl::set param_set, const Stmt &stmt, Scop &scop) {
  scop.data_.reads = isl::union_map::empty(param_space);
  scop.data_.copyin = isl::union_map::empty(param_space);
  scop.data_.writes = isl::union_map::empty(param_space);
  scop.data_.fake_copyin = isl::union_map::empty(param_space);
  scop.data_.transfer_stmt = isl::union_set::empty(param_space);
  scop.data_.inter_band_dependency = isl::union_map::empty(param_space);
  isl::set set = isl::set::universe(param_space);

  set = set.intersect_params(param_set);
  isl::id_list outer(param_space.ctx(), 0);
  ssize_t macro_stmt = -1;
  auto schedule = MakeScheduleTreeHelper(stmt, scop, set, outer, macro_stmt);

  return schedule;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
