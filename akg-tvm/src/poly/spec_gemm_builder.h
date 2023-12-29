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

#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {
struct GemmVar {
  VarExpr var_batch_name{"b"};
  VarExpr var_no_name{"no"};
  VarExpr var_mo_name{"mo"};
  VarExpr var_mi_name{"mi"};
  VarExpr var_ni_name{"ni"};
  VarExpr var_ko_name{"ko"};
  VarExpr var_ki_name{"ki"};
};
class SpecGemmBuilder {
 public:
  explicit SpecGemmBuilder(ScopInfo& info) : info_(info) {}
  ~SpecGemmBuilder() = default;
  Stmt Build(const Expr &mad_init_cond);

 private:
  Expr ReplacePragmaPrimeByVar(Expr pragma);
  void BuildConvGemmFeatureBand(Binds &new_bind) ;
  void BuildConvGemmFilterBand(Binds &new_bind) ;
  void BuildConvGemmResultBand(Binds &new_bind) ;
  Binds BuildConvGemmBand() ;
  Expr ZeroByDtype(const Tensor &t) ;
  Stmt ConstructGemmReduceBody(const Binds &gemm_bind, const Expr &mad_init_cond, const GemmVar &gv);
  Stmt ConstructGemm(const Binds &gemm_bind, const Expr &mad_init_cond) ;
  Stmt ConstructFor(int init, Expr cond_exp, const VarExpr &iter, const Stmt &s) ;
  std::string AutoConstructGemmDimensionInfo() ;
  std::string ConstructGemmDimensionInfo() ;
  void CheckConvGemmParam() ;
  int64_t AutoConvMNKTile(const std::string &param_name, int64_t param_size) ;
  bool CheckFilterTensorShape(const Array<Expr> &shape) ;
  Tensor FindBindTensor(const Binds &bind, const std::string &name) ;
  bool CheckFeatureTensorShape(const Array<Expr> &shape) ;
  int GetMAxisSetDim() ;

  ScopInfo& info_;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
