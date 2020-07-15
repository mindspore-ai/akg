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

#include "scop.h"

namespace akg {
namespace ir {
namespace poly {
static NodeRef makeIntImm(int n) {
  const int bits = 32;
  return IntImm::make(Int(bits), n);
}

/************************************************************
 * steps:
 * 1. First construct gemm IR in conv operator
 * 2. get first setdim info(in operator python file)
 * 3. construct gemm setdim info of M,N,K tile spec
 * 4. setdim second times for gemm IR(resetdim)
 * 5. AutoPoly() get the new Stmt
 * 6. setdim third times with the backup info for conv operator(resetdim)
 *************************************************************/
Stmt Scop::ConstructPolyGemm(const Expr &mad_init_cond) {
  // construct gemm IR in conv operator
  Binds gemm_binds = BuildConvGemmBand();
  Stmt res = ConstructGemm(gemm_binds, mad_init_cond);
  std::string gmm_dim;
  // construct setdim info about gemm IR in conv
  if (!conv_mnk_dims_.empty()) {
    gmm_dim = AutoConstructGemmDimensionInfo();
  } else {
    gmm_dim = ConstructGemmDimensionInfo();
  }

  static_cast<void>(PartitionSingle::CreateInstance(1, -1, GetMAxisSetDim(), fractal_int_info_));

  Map<std::string, NodeRef> attrs;
  attrs.Set("conv_backprop_filter", makeIntImm(IsConvBackpropFilter()));
  attrs.Set("bypassL1", makeIntImm(bypassL1_));
  attrs.Set("dim", StringImm::make(gmm_dim));
  if (IsConvBackpropInput()) {
    attrs.Set("kernel_h", makeIntImm(matB_dim_h_));
    attrs.Set("kernel_w", makeIntImm(matB_dim_w_));
  }
  attrs.Set("isolated_idx", makeIntImm(++isolated_idx_));
  attrs.Set("dump_pass_ir", makeIntImm(dump_pass_ir_));
  attrs.Set("dump_poly_dir", StringImm::make(dump_poly_dir_));
  attrs.Set("pragma_tilesize_is_var", makeIntImm(tile_size_is_var_));
  Array<NodeRef> res_poly = AutoPoly(res, gemm_binds, attrs, true, is_dynamic_);
  CHECK_GE(res_poly.size(), 1);
  PartitionSingle::free();
  return air::Downcast<Stmt>(res_poly[0]);
}

void Scop::BuildConvGemmFeatureBand(Scop::Binds &new_bind) {
  std::string a_name =
    IsConvBackpropFilter() ? fractal_str_info_[ATTR_CONV_GMM_WEIGHT] : fractal_str_info_[ATTR_CONV_GMM_FEATURE];
  Tensor a = FindTensor(a_name);
  if (!CheckFeatureTensorShape(a->shape)) {
    Array<Expr> fm_shapes;
    std::vector<std::string> tensor_axis;
    if (IsConvBackpropFilter()) {
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_BATCH));
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_M_ALIGN));
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_K_ALIGN));
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_M_INNER));
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_K_INNER));
    } else {
      tensor_axis.emplace_back(std::string(ATTR_CONV_BATCH));
      tensor_axis.emplace_back(std::string(ATTR_CONV_TILE_M));
      tensor_axis.emplace_back(std::string(ATTR_CONV_TILE_K));
      tensor_axis.emplace_back(std::string(ATTR_CONV_M_INNER));
      tensor_axis.emplace_back(std::string(ATTR_CONV_K_INNER));
    }
    if (tile_size_is_var_ == 1) {
      fm_shapes.push_back(fractal_int_info_[tensor_axis[0]]);
      fm_shapes.push_back(Var("MO"));
      fm_shapes.push_back(Var("KO"));
      fm_shapes.push_back(fractal_int_info_[tensor_axis[3]]);
      fm_shapes.push_back(fractal_int_info_[tensor_axis[4]]);
    } else {
      for (const auto &axis : tensor_axis) {
        fm_shapes.push_back(ReplacePragmaPrimeByVar(fractal_int_info_[axis]));
      }
    }
    Tensor new_feature = placeholder(fm_shapes, a->dtype, a_name);
    const Buffer new_feature_buffer = decl_buffer(fm_shapes, a->dtype, a_name);
    new_bind.Set(new_feature, new_feature_buffer);
  } else {
    new_bind.Set(a, binds_[a]);
  }
}

void Scop::BuildConvGemmFilterBand(Scop::Binds &new_bind) {
  std::string b_name =
    IsConvBackpropFilter() ? fractal_str_info_[ATTR_CONV_GMM_FEATURE] : fractal_str_info_[ATTR_CONV_GMM_WEIGHT];
  Tensor b = FindTensor(b_name);
  if (!CheckFilterTensorShape(b->shape)) {
    Array<Expr> filter_shapes;
    std::vector<std::string> tensor_axis;
    if (IsConvBackpropFilter()) {
      // [Batch, Ko, No, Ni, Ki]
      tensor_axis.emplace_back(std::string(ATTR_CONV_BATCH));
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_K_ALIGN));
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_N_ALIGN));
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_N_INNER));
      tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_K_INNER));
    } else {
      tensor_axis.emplace_back(std::string(ATTR_CONV_TILE_K));
      tensor_axis.emplace_back(std::string(ATTR_CONV_TILE_N));
      tensor_axis.emplace_back(std::string(ATTR_CONV_N_INNER));
      tensor_axis.emplace_back(std::string(ATTR_CONV_K_INNER));
    }
    if (tile_size_is_var_ == 1) {
      filter_shapes.push_back(Var("KO"));
      filter_shapes.push_back(Var("NO"));
      filter_shapes.push_back(fractal_int_info_[tensor_axis[2]]);
      filter_shapes.push_back(fractal_int_info_[tensor_axis[3]]);
    } else {
      for (const auto &axis : tensor_axis) {
        filter_shapes.push_back(ReplacePragmaPrimeByVar(fractal_int_info_[axis]));
      }
    }
    Tensor new_filter = placeholder(filter_shapes, b->dtype, b_name);
    const Buffer new_filter_buffer = decl_buffer(filter_shapes, b->dtype, b_name);
    new_bind.Set(new_filter, new_filter_buffer);
  } else {
    new_bind.Set(b, binds_[b]);
  }
}

void Scop::BuildConvGemmResultBand(Scop::Binds &new_bind) {
  Array<Expr> shapes;
  std::vector<std::string> tensor_axis;
  if (IsConvBackpropFilter()) {
    tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_N_ALIGN));
    tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_M_ALIGN));
    tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_M_INNER));
    tensor_axis.emplace_back(std::string(ATTR_SPEC_GEMM_N_INNER));
  } else {
    tensor_axis.emplace_back(std::string(ATTR_CONV_BATCH));
    tensor_axis.emplace_back(std::string(ATTR_CONV_TILE_N));
    tensor_axis.emplace_back(std::string(ATTR_CONV_TILE_M));
    tensor_axis.emplace_back(std::string(ATTR_CONV_M_INNER));
    tensor_axis.emplace_back(std::string(ATTR_CONV_N_INNER));
  }
  if (tile_size_is_var_ == 1) {
    shapes.push_back(fractal_int_info_[tensor_axis[0]]);
    shapes.push_back(Var("NO"));
    shapes.push_back(Var("MO"));
    shapes.push_back(fractal_int_info_[tensor_axis[3]]);
    shapes.push_back(fractal_int_info_[tensor_axis[4]]);
  } else {
    for (const auto &axis : tensor_axis) {
      shapes.push_back(ReplacePragmaPrimeByVar(fractal_int_info_[axis]));
    }
  }
  Tensor t = placeholder(shapes, MadCastType(), fractal_str_info_[ATTR_CONV_GMM_RES]);
  const Buffer buffer = decl_buffer(shapes, MadCastType(), fractal_str_info_[ATTR_CONV_GMM_RES]);
  new_bind.Set(t, buffer);
}

Scop::Binds Scop::BuildConvGemmBand() {
  Binds new_bind;
  CheckConvGemmParam();
  BuildConvGemmFeatureBand(new_bind);
  BuildConvGemmFilterBand(new_bind);
  BuildConvGemmResultBand(new_bind);
  return new_bind;
}

Expr ZeroByDtype(const Tensor &t) {
  if (t->dtype.is_int()) {
    return IntImm::make(t->dtype, 0);
  } else if (t->dtype.is_uint()) {
    return UIntImm::make(t->dtype, 0);
  } else {
    CHECK(t->dtype.is_float());
    return FloatImm::make(t->dtype, 0.000000);
  }
}

Stmt Scop::ConstructGemmReduceBody(const Binds &gemm_bind, const Expr &mad_init_cond, const GemmVar &gv) {
  Tensor a;
  Tensor b;
  if (IsConvBackpropFilter()) {
    a = FindBindTensor(gemm_bind, fractal_str_info_[ATTR_CONV_GMM_WEIGHT]);
    b = FindBindTensor(gemm_bind, fractal_str_info_[ATTR_CONV_GMM_FEATURE]);
  } else {
    a = FindBindTensor(gemm_bind, fractal_str_info_[ATTR_CONV_GMM_FEATURE]);
    b = FindBindTensor(gemm_bind, fractal_str_info_[ATTR_CONV_GMM_WEIGHT]);
  }
  Tensor t = FindBindTensor(gemm_bind, fractal_str_info_[ATTR_CONV_GMM_RES]);

  Array<Expr> args_c_localUB, args_a_localL1, args_b_localL1;
  if (!IsConvBackpropFilter()) {
    args_c_localUB.push_back(gv.var_batch_name);
  }
  args_c_localUB.push_back(gv.var_no_name);
  args_c_localUB.push_back(gv.var_mo_name);
  args_c_localUB.push_back(gv.var_mi_name);
  args_c_localUB.push_back(gv.var_ni_name);

  args_a_localL1.push_back(gv.var_batch_name);
  args_a_localL1.push_back(gv.var_mo_name);
  args_a_localL1.push_back(gv.var_ko_name);
  args_a_localL1.push_back(gv.var_mi_name);
  args_a_localL1.push_back(gv.var_ki_name);

  if (IsConvBackpropFilter()) {
    args_b_localL1.push_back(gv.var_batch_name);
  }
  args_b_localL1.push_back(gv.var_ko_name);
  args_b_localL1.push_back(gv.var_no_name);
  args_b_localL1.push_back(gv.var_ni_name);
  args_b_localL1.push_back(gv.var_ki_name);

  Expr c_buffer = Call::make(t->dtype, t->op->name, args_c_localUB, Call::CallType::Halide, t->op, t->value_index);
  Expr a_buffer = Call::make(a->dtype, a->op->name, args_a_localL1, Call::CallType::Halide, a->op, a->value_index);
  Expr b_buffer = Call::make(b->dtype, b->op->name, args_b_localL1, Call::CallType::Halide, b->op, b->value_index);
  Expr added = Mul::make(a_buffer, b_buffer);
  if (MadCastType() == Float(32)) {
    added = Cast::make(Float(32), added);
  }
  Expr mad = Call::make(c_buffer.type(), "mad", {c_buffer, added}, Call::PureIntrinsic);
  Stmt provide = Provide::make(t->op, 0, mad, args_c_localUB);

  Stmt init = Provide::make(t->op, 0, ZeroByDtype(t), args_c_localUB);
  if (mad_init_cond.defined() && !is_zero(mad_init_cond)) {
    init = IfThenElse::make(mad_init_cond, init);
  }

  if (mad_init_cond.defined() && is_zero(mad_init_cond)) {
    init = AttrStmt::make(make_zero(Int(32)), "init", 1, init);
  } else {
    init = AttrStmt::make(make_zero(Int(32)), "init", 0, init);
  }

  Stmt ki = ConstructFor(0, fractal_int_info_[ATTR_CONV_K_INNER], gv.var_ki_name, provide);  // ki
  Stmt ko;
  if (tile_size_is_var_ == 1) {
    ko = ConstructFor(0, Var("KO"), gv.var_ko_name, ki);  // ko
  } else {
    ko = ConstructFor(0, ReplacePragmaPrimeByVar(fractal_int_info_[ATTR_CONV_TILE_K]), gv.var_ko_name, ki);  // ko
  }
  return Block::make(init, ko);
}

Stmt Scop::ConstructGemm(const Binds &gemm_bind, const Expr &mad_init_cond) {
  CheckConvGemmParam();
  GemmVar gv;
  Stmt reduce = ConstructGemmReduceBody(gemm_bind, mad_init_cond, gv);
  Stmt ni = ConstructFor(0, fractal_int_info_[ATTR_CONV_N_INNER], gv.var_ni_name, reduce);  // ni
  Stmt mi = ConstructFor(0, fractal_int_info_[ATTR_CONV_M_INNER], gv.var_mi_name, ni);      // mi
  Stmt mo;
  if (tile_size_is_var_ == 1) {
    mo = ConstructFor(0, Var("MO"), gv.var_mo_name, mi);  // mo
  } else {
    mo = ConstructFor(0, ReplacePragmaPrimeByVar(fractal_int_info_[ATTR_CONV_TILE_M]), gv.var_mo_name, mi);  // mo
  }

  Expr no_size = fractal_int_info_[ATTR_CONV_TILE_N];
  if (is_const_int(fractal_int_info_["isolate"], 1)) {
    no_size = fractal_int_info_["n_isolate"];
  }
  Stmt no;
  if (tile_size_is_var_ == 1) {
    no = ConstructFor(0, Var("NO"), gv.var_no_name, mo);  // no
  } else {
    no = ConstructFor(0, ReplacePragmaPrimeByVar(no_size), gv.var_no_name, mo);  // no
  }

  Stmt res = ConstructFor(0, fractal_int_info_[ATTR_CONV_BATCH], gv.var_batch_name, no);  // batch
  Tensor t = FindBindTensor(gemm_bind, fractal_str_info_[ATTR_CONV_GMM_RES]);
  res = ProducerConsumer::make(t->op, true, res);
  return res;
}

Stmt Scop::ConstructFor(int init, Expr cond_exp, const VarExpr &iter, const Stmt &s) {
  Expr initExp(init);
  Stmt res = For::make(iter, initExp, std::move(cond_exp), ForType::Serial, DeviceAPI::None, s);
  return res;
}

std::string Scop::AutoConstructGemmDimensionInfo() {
  std::ostringstream dim_out;
  std::vector<std::string> gmm_axis;
  if (!IsConvBackpropFilter()) {
    gmm_axis.emplace_back(std::string(ATTR_CONV_BATCH));
  }
  gmm_axis.emplace_back(std::string(ATTR_CONV_TILE_N));
  gmm_axis.emplace_back(std::string(ATTR_CONV_TILE_M));
  gmm_axis.emplace_back(std::string(ATTR_CONV_M_INNER));
  gmm_axis.emplace_back(std::string(ATTR_CONV_N_INNER));
  if (IsConvBackpropFilter()) {
    gmm_axis.emplace_back(std::string(ATTR_CONV_BATCH));
  }
  gmm_axis.emplace_back(std::string(ATTR_CONV_TILE_K));
  for (const auto &key : gmm_axis) {
    auto fractal_it = fractal_int_info_.find(key);
    if (fractal_it != fractal_int_info_.end()) {
      Expr axis_len;
      axis_len = fractal_it->second;
      if (!is_const_int(axis_len, 1)) {  // set dim for axis > 1
        for (const auto &dim : conv_mnk_dims_) {
          if (dim.axis == key) {
            int tile = static_cast<int>(dim.l0_tiling_size);
            dim_out << " " << dim.index << " " << dim.axis << " " << tile << " " << 0;
          }
        }
      }
    }
  }
  std::string gmm_dim = dim_out.str();

  return gmm_dim;
}

std::string Scop::ConstructGemmDimensionInfo() {
  std::ostringstream dim_out;
  int64_t index = 0;
  int64_t axis = 0;
  int64_t tile0 = 0;
  std::vector<std::string> gmm_axis;
  if (!IsConvBackpropFilter()) {
    gmm_axis.emplace_back(std::string(ATTR_CONV_BATCH));
  }
  gmm_axis.emplace_back(std::string(ATTR_CONV_TILE_N));
  gmm_axis.emplace_back(std::string(ATTR_CONV_TILE_M));
  gmm_axis.emplace_back(std::string(ATTR_CONV_M_INNER));
  gmm_axis.emplace_back(std::string(ATTR_CONV_N_INNER));
  if (IsConvBackpropFilter()) {
    gmm_axis.emplace_back(std::string(ATTR_CONV_BATCH));
  }
  gmm_axis.emplace_back(std::string(ATTR_CONV_TILE_K));
  for (const auto &key : gmm_axis) {
    auto fractal_it = fractal_int_info_.find(key);
    if (fractal_it != fractal_int_info_.end()) {
      Expr axis_len;
      axis_len = fractal_it->second;
      if (!is_const_int(axis_len, 1)) {  // set dim for axis > 1
        auto it = attr_info_.find(key);

        int64_t tile1 = axis_len.as<IntImm>()->value;  // init tile1
        if (it != attr_info_.end()) {                  // if find attr Info update tile1
          // auto tile 16 for N_out, K_outer, M_outer
          CHECK((*it).second.as<IntImm>());
          tile1 = AutoConvMNKTile(key, (*it).second.as<IntImm>()->value);  // attr Info
        }

        tile1 = (tile1 > axis_len.as<IntImm>()->value) ? axis_len.as<IntImm>()->value : tile1;  // min(attr, axis_len)

        dim_out << " " << index << " " << axis << " " << tile1 << " " << tile0;
        axis++;
      }
    }
  }

  std::string dim_info = dim_out.str();
  return dim_info;
}

void Scop::CheckConvGemmParam() {
  std::vector<std::string> str_param;
  str_param.emplace_back(std::string(ATTR_CONV_GMM_FEATURE));
  str_param.emplace_back(std::string(ATTR_CONV_GMM_WEIGHT));
  str_param.emplace_back(std::string(ATTR_CONV_GMM_RES));

  for (const auto &iter : str_param) {
    auto key = fractal_str_info_.find(iter);
    std::string err = "Error: You need to set" + iter + "in strInfo";
    CHECK(key != fractal_str_info_.end()) << err;
  }

  std::vector<std::string> int_param;

  int_param.emplace_back(std::string(ATTR_CONV_BATCH));

  int_param.emplace_back(std::string(ATTR_CONV_TILE_M));
  int_param.emplace_back(std::string(ATTR_CONV_TILE_K));
  int_param.emplace_back(std::string(ATTR_CONV_TILE_N));

  int_param.emplace_back(std::string(ATTR_CONV_M_INNER));
  int_param.emplace_back(std::string(ATTR_CONV_K_INNER));
  int_param.emplace_back(std::string(ATTR_CONV_N_INNER));

  int_param.emplace_back(std::string(ATTR_CONV_GMM_M));

  for (const auto &iter : int_param) {
    auto key = fractal_int_info_.find(iter);
    CHECK(key != fractal_int_info_.end()) << "Error: You need to set " << iter << " in intInfo";
  }
}

int64_t Scop::AutoConvMNKTile(const std::string &param_name, int64_t param_size) {
  int64_t result = param_size;
  if (param_name == ATTR_CONV_TILE_K || param_name == ATTR_CONV_TILE_M || param_name == ATTR_CONV_TILE_N) {
    const int64_t autoTileSize = 16;
    if (param_size > 0) {
      CHECK(param_size >= autoTileSize) << "Error: You need to set attr " << param_name << " >=" << autoTileSize
                                        << " in conv akg.tvm.compute";
    }
    result = result / autoTileSize;
  }
  return result;
}

bool Scop::CheckFilterTensorShape(const Array<Expr> &shape) {
  if (shape.size() != 4) return false;

  std::vector<std::string> keys;
  keys.emplace_back(std::string(ATTR_CONV_TILE_K));
  keys.emplace_back(std::string(ATTR_CONV_TILE_N));
  keys.emplace_back(std::string(ATTR_CONV_N_INNER));
  keys.emplace_back(std::string(ATTR_CONV_K_INNER));

  for (size_t i = 0; i < keys.size(); i++) {
    auto iter = fractal_int_info_.find(keys[i]);
    if (iter == fractal_int_info_.end()) return false;
    if (Compare(shape[i], iter->second) != 0) return false;
  }

  return true;
}

Tensor Scop::FindBindTensor(const Binds &bind, const std::string &name) {
  for (auto i : bind) {
    if (i.first->op->name == name) {
      return i.first;
    }
  }
  LOG(FATAL) << name << " is not declared in parameter binds";
  return Tensor();
}

bool Scop::CheckFeatureTensorShape(const Array<Expr> &shape) {
  if (shape.size() != 5) return false;

  std::vector<std::string> keys;
  keys.emplace_back(std::string(ATTR_CONV_BATCH));
  keys.emplace_back(std::string(ATTR_CONV_TILE_M));
  keys.emplace_back(std::string(ATTR_CONV_TILE_K));
  keys.emplace_back(std::string(ATTR_CONV_M_INNER));
  keys.emplace_back(std::string(ATTR_CONV_K_INNER));

  for (size_t i = 0; i < keys.size(); i++) {
    auto iter = fractal_int_info_.find(keys[i]);
    if (iter == fractal_int_info_.end()) return false;
    if (Compare(shape[i], iter->second) != 0) return false;
  }

  return true;
}

int Scop::GetMAxisSetDim() {
  int cut_m = GetAttrValue(ATTR_CONV_TILE_M);
  Expr e = fractal_int_info_[ATTR_CONV_TILE_M] * fractal_int_info_[ATTR_CONV_M_INNER];
  CHECK(is_const(e));
  CHECK(e.as<IntImm>());
  int gemm_m = e.as<IntImm>()->value;
  int result = cut_m < gemm_m ? cut_m : gemm_m;
  return result;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
