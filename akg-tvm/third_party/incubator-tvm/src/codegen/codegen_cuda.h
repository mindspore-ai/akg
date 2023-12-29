/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_cuda.h
 * \brief Utility to generate cuda code
 */

/*
 * 2020.10.26
 *   Add function PrintReduce.
 */

/*
 * 2021.01.12
 * Add const for reduce
 */

/*
 * 2021.01.13
 * Add function Simplify_name.  
 * Add variables for TensorCore:
 *     warp_tile_m, warp_tile_n, warp_tile_k,
 *     matrix_a_major, matrix_b_major,
 *     matrix_abc,
 *     wmma_scope.
 */


/*
 * 2021.3.22
 *   Refactor the function Simplify_name.
 */
 
/*
 * 2021.05.17
 * Add const akg_reduce::AkgKahanAccumulation for reduce
 */

/*
 * 2021.7.07
 *   Add function for Mod op fusion: VisitExpr_(const Mod* op, std::ostream& os).
 *   If op is int type, generate code: op->a % op->b, else fmod(op->a, op->b).
 */

/*
 * 2021.8.19
 *   Add const standard_normal for random.
 *   Add need_random_lib_.
 */

/*
 * 2021.11.17
 *   Add csr_loop_stride for csr.
 */

#ifndef TVM_CODEGEN_CODEGEN_CUDA_H_
#define TVM_CODEGEN_CODEGEN_CUDA_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include <unordered_map>
#include "codegen_c.h"

namespace air {
namespace codegen {

constexpr auto REDUCE_LIB_TYPE = "reduceLibType";
constexpr auto AKG_REDUCE = "akg_reduce::AkgReduce";
constexpr auto AKG_ATOMIC_RETURN = "akg_reduce::AkgAtomicReturn";
constexpr auto AKG_KAHAN = "akg_reduce::AkgKahanAccumulation";
constexpr auto PARIS_REDUCE = "paris_reduce::ParisReduce";
constexpr auto PARIS_ATOMIC_RETURN = "paris_reduce::ParisReturn";
constexpr auto ORIGIN_REDUCE_LIB = "origin";
constexpr auto PARIS_REDUCE_LIB = "paris";
constexpr auto STANDARD_NORMAL = "StandardNormal";

class CodeGenCUDA final : public CodeGenC {
 public:
  CodeGenCUDA();
  void Init(bool output_ssa);
  void AddFunction(LoweredFunc f);
  std::string Finish();
  bool need_include_path() {
    return (enable_fp16_ || enable_int8_ || need_math_constants_h_ || need_mma_h_);
  }
  // override behavior
  void VisitStmt_(const ir::For* op) final;
  void PrintStorageSync(const Call* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(
      const std::string&op, Type t,
      Expr lhs, Expr rhs, std::ostream& os) final;  // NOLINT(*)
  void PrintType(Type t, std::ostream& os) final; // NOLINT(*)
  void PrintVecElemLoad(
      const std::string& vec, Type t, int i, std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(
      const std::string& vec, Type t, int i, const std::string& value) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  // overload visitor
  void VisitExpr_(const Ramp* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const Shuffle* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const Broadcast* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const FloatImm *op, std::ostream& os) final;
  void VisitExpr_(const Call *op, std::ostream& os) final;
  void VisitStmt_(const Evaluate *op) final;
  void VisitStmt_(const Allocate *op) final;
  void VisitStmt_(const AttrStmt *op) final;
  void VisitStmt_(const LetStmt *op) final;
  void VisitExpr_(const Variable *op, std::ostream &os) final;
  void VisitExpr_(const Load *op, std::ostream &os) final;
  void VisitExpr_(const Mod *op, std::ostream &os) final;
  void VisitStmt_(const Store *op) final;

 private:
  // Handle volatile loads.
  void HandleVolatileLoads(const std::string& value, const Load* op,
                           std::ostream& os) final;

  // Whether scope such as "__shared__" or "__constant__" is part of type.
  bool IsScopePartOfType() const final {
    return false;
  }

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need random lib
  bool need_random_lib_{false};

  // whether next store will be a reinterpret_cast
  bool is_reinterpret{false};
  // the extent of the swizzled loop
  Expr loop_extent{0};
  // whether next store is not an array
  bool simple_store{false};
  // whether store uses vector format
  bool vec_store{false};
  // true if we are currently unrolling loop
  bool unroll{false};
  // indicates we are unrolling a store
  bool unroll_store{false};
  // var from Loads to modify with vector load
  std::set<const Variable*> vec_loads;
  // whether replace cce variable with constant in vec_store
  bool replace_cce{false};
  // variable to replace
  const Variable* loop_var{nullptr};
  // index to replace cce with
  int current_index{0};
  // do not set value to next LetStmt if true
  bool no_init_value{false};
  // ignore next allocate stmt if true (trick to bypass some tests)
  int csr_loop_stride{1};
  // custom loop stride
//  bool ignore_next_allocate{false};

  // add for TensorCore
  // warp tile size for TensorCore interface
  Expr warp_tile_m = IntImm::make(Int(32), 1);
  Expr warp_tile_n = IntImm::make(Int(32), 1);
  Expr warp_tile_k = IntImm::make(Int(32), 1);
  // layout mode for TensorCore fragment
  Expr matrix_a_major = StringImm::make("row_major");
  Expr matrix_b_major = StringImm::make("col_major");
  std::unordered_map<std::string, std::string> matrix_abc;
  // indicate which TensorCore interface
  std::string wmma_scope = "nvcuda";

  std::unordered_map<const Variable*, int> sm_offsets;
  std::unordered_map<const Variable*, std::string> fragment_shapes;
  std::unordered_map<const Variable*, std::string> fragment_layouts;
  friend void PrintConst(const FloatImm* op, std::ostream& os, CodeGenCUDA* p);
  void PrintWmmaScope(const std::string& scope, Type t, const Variable* variable, std::ostream& os);
  int32_t GetWmmaFragmentSize(const std::string &scope, const Variable* variable, int32_t size);
};

}  // namespace codegen
}  // namespace air

#endif  // TVM_CODEGEN_CODEGEN_CUDA_H_
