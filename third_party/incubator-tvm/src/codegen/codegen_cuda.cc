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
 * \file codegen_cuda.cc
 */

/*
 * 2020.10.26
 *   Modify the functions:
 *     Finish()
 *     Delete the offset checkout of VisitStmt_(const ir::For* op)
 */

/*
 * 2021.01.12
 *   Modify the functions:
 *     VisitExpr_(const Call *op, std::ostream& os)
 * 
 */

/*
 * 2021.01.13
 *   Add function Simplify_name.
 *   Modify the functions:
 *     Finish()
 *     VisitExpr_(const Call *op, std::ostream& os)
 *     VisitStmt_(const AttrStmt* op)
 *     VisitStmt_(const Allocate* op)
 *     PrintWmmaScope
 *     GetWmmaFragmentSize
 */

/*
 * 2021.1.16
 *   Modify the functions:
 *     Add print total shared_memory of VisitStmt_(const AttrStmt* op)
 *     Print offset shared memory when use total shared_memory of VisitStmt_(const Allocate* op)
 */

/*
 * 2021.3.22
 *   Refactor the function Simplify_name.
 */
 
/*
 * 2021.5.17
 *   Modify the functions:
 *     add KaHan interface processing logic in VisitExpr_(const Call *op, std::ostream& os)
 *     for the reduce sum operator
 */

/*
 * 2021.5.27
 *   Add function for GEMM op fusion on TensorCore.
 */

#include "codegen_cuda.h"

#include <tvm/base.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <cmath>
#include <vector>
#include <string>
#include "common/common_util.h"
#include <tvm/ir_pass.h>
#include "literal/cuda_half_t.h"
#include "codegen_cuda.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace air {
namespace codegen {

CodeGenCUDA::CodeGenCUDA() {
  restrict_keyword_ = "__restrict__";
}

void CodeGenCUDA::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  vid_global_barrier_state_ = GetUniqueName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = GetUniqueName("__barrier_expect");
  CHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}

void CodeGenCUDA::AddFunction(LoweredFunc f) {
  this->stream << "extern \"C\" __global__ ";
  CodeGenC::AddFunction(f);
}

std::string CodeGenCUDA::Finish() {
#ifdef USE_CUDA
  int major, minor;
  std::string major_str;
  std::string minor_str;
  cudaError_t e1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    major_str = std::to_string(major);
    minor_str = std::to_string(minor);
    if (minor_str.length() == 1) {
      minor_str += "0";
    }
    decl_stream << "#ifndef __CUDA_ARCH__\n";
    decl_stream << "#define __CUDA_ARCH__ " << major_str + minor_str << "\n";
    decl_stream << "#endif\n";
  }
#endif

  if (need_reduce_lib_) {
    if (reduce_lib_type_ == ORIGIN_REDUCE_LIB) {
      decl_stream << "#include \"akg_reduce/reduce.cuh\"\n";
    } else if (reduce_lib_type_ == PARIS_REDUCE_LIB) {
      decl_stream << "#include \"paris_reduce/paris_reduce.cuh\"\n";
    }
  }
  if (enable_fp16_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)\n";
    decl_stream << "#include <cuda_fp16.h>\n";
    decl_stream << "__device__ half max"
                << "(half a, half b)\n"
                << "{\n  return __hgt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "__device__ half min(half a, half b)\n"
                << "{\n  return __hlt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "#else\n";
    decl_stream << _cuda_half_t_def;
    decl_stream << "#endif\n\n";
    decl_stream << _cuda_half_util;
  }

  if (enable_int8_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)\n";
    decl_stream << "#include <sm_61_intrinsics.h>\n";
    decl_stream << "#endif\n";
  }

  if (need_math_constants_h_) {
    decl_stream << "#include <math_constants.h>\n";
  }

  if (need_mma_h_) {
    if (wmma_scope == "akg") {
      decl_stream << "#include \"akg_mma_lib/wmma.hpp\"\n";
    } else{
      decl_stream << "#include <mma.h>\n";
    }
  }

  // TODO add condition
  decl_stream << "// built-in for half swizzle\n"
                 "#include <cuda_fp16.h>\n"
                 "struct __device_builtin__ __align__(8) half4 { half x, y, z, w; };\n"
                 "\n"
                 "#if defined(__CUDACC_RTC__)\n"
                 "#define __CUDA_FP16_DECL__ __host__ __device__\n"
                 "#else\n"
                 "#define __CUDA_FP16_DECL__ static __device__ __inline__\n"
                 "#endif\n"
                 "\n"
                 "// half4 ldg function support\n"
                 "#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)\n"
                 "#define __LDG_PTR   \"l\"\n"
                 "#else\n"
                 "// not sure about this one, it was copied from the half2 ldg() function\n"
                 "#define __LDG_PTR   \"r\"\n"
                 "#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/\n"
                 "\n"
                 "#define __HALF4_TO_UI(var) *(reinterpret_cast<unsigned long *>(&(var)))\n"
                 "__CUDA_FP16_DECL__ half4 __ldg(const  half4 *ptr)\n"
                 "{\n"
                 "    half4 ret;\n"
                 "    asm (\"ld.global.nc.b64 %0, [%1];\"  : \"=l\"(__HALF4_TO_UI(ret)) : __LDG_PTR(ptr));\n"
                 "    return ret;\n"
                 "}\n\n";

  return CodeGenC::Finish();
}

void CodeGenCUDA::VisitStmt_(const ir::For* op) {
  if (op->for_type == ir::ForType::Unrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  else if (op->for_type == ir::ForType::Swizzled) {
    // remove this loop
    PrintStmt(op->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, UInt(32), iv->var.type());
}

void CodeGenCUDA::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "do not yet support vector types";
    os << "void*"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        enable_fp16_ = true;
        if (lanes == 1) {
          os << "half";
        } else if (lanes <= 8) {
          CHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
          if (lanes <= 4) { // added for swizzle
            os << "half" << lanes;
          } else {
            os << "float" << lanes / 2;
          }
        } else {
          fail = true;
        }
        break;
      case 32: os << "float"; break;
      case 64: os << "double"; break;
      default: fail = true; break;
    }
    if (!fail && (lanes == 1 || t.bits() == 16)) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  } else if (t == Bool()) {
    os << "bool"; return;
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      if (t.lanes() != 1) {
        os << "u";
      } else {
        os << "unsigned ";
      }
    }
    switch (t.bits()) {
      case 8: {
        if (t.lanes() == 4) {
          // directly 4 8 bit int in integer.
          enable_int8_ = true;

          // We use int for int8x4 instead of char4 because using char4 is
          // likely to produce extra instructions to pack four int8 elements
          // into 32-bit data.
          os << "int"; return;
        } else if (t.lanes() == 8) {
          enable_int8_ = true;
          os << "int2"; return;
        } else if (t.lanes() == 16) {
          enable_int8_ = true;
          os << "int4"; return;
        } else if (!t.is_uint() && t.lanes() == 1) {
          os << "signed char"; break;
        } else {
          os << "char"; break;
        }
      }
      case 16: os << "short"; break;
      case 32: os << "int"; break;
      case 64: {
        if (sizeof(long) != 8) { // NOLINT(*)
          if (t.lanes() == 1) {
            os << "long long"; break;
          } else if (t.lanes() == 2) {
            os << "longlong"; break;
          } else {
            // No longlong3, longlong4
            LOG(FATAL) << "Cannot convert type " << t << " to CUDA type on a L32 platform";
          }
        } else {
          os << "long"; break;
        }
      }
      case 1: os << "int"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenCUDA::PrintVecBinaryOp(
    const std::string&op, Type t,
    Expr lhs, Expr rhs, std::ostream& os) {  // NOLINT(*)
  // unpacking operations.
  int lanes = t.lanes();

  {
    // The assignment below introduces side-effect, and the resulting value cannot
    // be reused across multiple expression, thus a new scope is needed
    int vec_scope = BeginScope();

    // default: unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.type());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.type());
    std::string sret = GetUniqueName("_");
    {
      // delcare type.
      this->PrintIndent();
      this->PrintType(t, stream);
      stream << ' ' << sret << ";\n";
    }
    for (int i = 0; i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.type(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.type(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.type(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.type(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
    os << sret;
    EndScope(vec_scope);
  }
}

void CodeGenCUDA::PrintVecElemLoad(
    const std::string& vec, Type t, int i, std::ostream& os) {  // NOLINT(*)
  static const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < 4);
  if (t.is_int() && t.bits() == 8) {
    os << "(0x000000ff & (" << vec << " >> " << i * 8 << "))";
  } else {
    os << vec << "." << access[i];
  }
}


void CodeGenCUDA::PrintVecElemStore(
    const std::string& vec, Type t, int i, const std::string& value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < 4);
  if (t.is_int() && t.bits() == 8) {
    stream << vec << "=" << vec << " & ~(0x000000ff << " << i * 8 << ") | ("
        << value << " << " << i * 8 << ");\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenCUDA::PrintStorageSync(const Call* op) {
  const std::string& sync = op->args[0].as<StringImm>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  } else if (sync == "global") {
    if (!need_global_barrier_) {
      need_global_barrier_ = true;
      this->decl_stream << "extern \"C\" __device__ unsigned "
                        << vid_global_barrier_state_ << ";\n";
    }
    // global synchronizer
    std::string is_load = PrintExpr(op->args[1]);
    std::string num_blocks = PrintExpr(op->args[2]);
    this->PrintIndent();
    // In theory only threadfence is needed
    // but we observed problems with only threadfence
    this->stream <<"__threadfence_system();\n";
    this->PrintIndent();
    this->stream <<"if (" << is_load << ") {\n";
    int wb = this->BeginScope();
    this->PrintIndent();
    this->stream << "atomicAdd(&" << vid_global_barrier_state_ << ", 1);\n";
    this->PrintIndent();
    std::string ptr = GetUniqueName("pf");
    this->stream << "volatile unsigned* "
                 << ptr << " = &" << vid_global_barrier_state_<< ";\n";
    this->PrintIndent();
    this->stream << vid_global_barrier_expect_ << " += " << num_blocks << ";\n";
    this->PrintIndent();
    this->stream <<"while (" << ptr << "[0] < " << vid_global_barrier_expect_ << ");\n";
    this->EndScope(wb);
    this->PrintIndent();
    this->stream <<"}\n";
    this->PrintIndent();
    this->stream <<"__syncthreads();\n";
  }
}

void CodeGenCUDA::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_NE(scope, "global");
  if (scope == "shared") {
    os << "__shared__";
  }
}

void CodeGenCUDA::VisitExpr_(const Call *op, std::ostream& os) {
  if (op->is_intrinsic(intrinsic::tvm_fill_fragment)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 6U);
    os << wmma_scope << "::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    Expr new_args = op->args[4];
    if (auto add = op->args[4].as<Add>()) {
      warp_tile_n = op->args[2];
      new_args = Div::make(add->a, warp_tile_n);
      new_args = Add::make(new_args, add->b);
    }
    if (is_scheme_two_) {
      this->PrintExpr(op->args[4], os);
    } else {
      this->PrintExpr(new_args, os);
    }
    os << "], ";
    Expr fill = FloatImm::make(Float(32), 0.0);
    this->PrintExpr(fill, os);
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_load_matrix_sync)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 8U);
    warp_tile_m = op->args[1];
    warp_tile_n = op->args[2];
    warp_tile_k = op->args[3];
    os << wmma_scope << "::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    Expr new_args = op->args[4];
    Expr warp_tile;
    auto var_node = op->args[0].as<Variable>();
    auto it_matrix = matrix_abc.find(akg::common::GetGlobalName(var_node->name_hint));
    if (it_matrix != matrix_abc.end()) {
      if (it_matrix->second == "matrix_a") {
        if (op->args[7].as<StringImm>()->value == "row_major") {
          warp_tile = warp_tile_k;
        } else {
          warp_tile = warp_tile_m;
        }
      } else if (it_matrix->second == "matrix_b") {
        if (op->args[7].as<StringImm>()->value == "col_major") {
          warp_tile = warp_tile_k;
        } else {
          warp_tile = warp_tile_n;
        }
      } else if (it_matrix->second == "accumulator") {
        if (op->args[7].as<StringImm>()->value == "row_major") {
          warp_tile = warp_tile_n;
        } else {
          LOG(FATAL) << "Not support matrix to load fragment accumulator!"; 
        }
      } else {
        LOG(FATAL) << "Not support matrix to load !"; 
      }
    }
    if (auto add = op->args[4].as<Add>()) {
      new_args = Div::make(add->a, warp_tile);
      new_args = Add::make(new_args, add->b);
    }
    if (is_scheme_two_) {
      this->PrintExpr(op->args[4], os);
    } else {
      this->PrintExpr(new_args, os);
    }
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    if (it_matrix != matrix_abc.end() && it_matrix->second == "accumulator") {
      os << ", nvcuda::wmma::mem_row_major";
    }
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_store_matrix_sync)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 8U);
    os << wmma_scope << "::wmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    Expr new_args = op->args[4];
    if (auto add = op->args[4].as<Add>()) {
      new_args = Div::make(add->a, warp_tile_n);
      new_args = Add::make(new_args, add->b);
    }
    if (is_scheme_two_) {
      this->PrintExpr(op->args[4], os);
    } else {
      this->PrintExpr(new_args, os);
    }
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImm *str = op->args[7].as<StringImm>()) {
      os << ", nvcuda::wmma::mem_" << str->value;
    } else {
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_mma_sync)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 8U);
    os << wmma_scope << "::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      Expr new_args = op->args[i * 2 + 1];
      if (auto add = op->args[i * 2 + 1].as<Add>()) {
        Expr warp_size;
        if (i == 0 || i == 3) {
          warp_size = warp_tile_n;
        } else {
          warp_size = warp_tile_k;
        }
        new_args = Div::make(add->a, warp_size);
        new_args = Add::make(new_args, add->b);
      }
      if (is_scheme_two_) {
        this->PrintExpr(op->args[i * 2 + 1], os);
      } else {
        this->PrintExpr(new_args, os);
      }
      os << "]" << ((i < 3) ? ", ": ")");
    }
  } else if (op->is_intrinsic(intrinsic::akg_fragment_elem)) {
    need_mma_h_ = true;
    os << wmma_scope << "::wmma::fragment_" << op->args[op->args.size() - 1].as<StringImm>()->value << "(";
    if (op->args.size() == 7) {
      for (int i = 0; i < 3; ++i) {
        this->PrintExpr(op->args[i * 2], os);
        os << "[";
        this->PrintExpr(op->args[i * 2 + 1], os);
        os << "]" << ((i < 2) ? ", " : ")");
      }
    } else if (op->args.size() == 6) {
      for (int i = 0; i < 2; ++i) {
        this->PrintExpr(op->args[i * 2], os);
        os << "[";
        this->PrintExpr(op->args[i * 2 + 1], os);
        os << "]" << ((i < 1) ? ", " : "");
      }
      os << ", " << op->args[4] << ")";
    }
  } else if ((op->call_type == Call::Extern) || (op->call_type == Call::PureExtern)) {
    if (op->name == "&") {
      CHECK_EQ(op->args.size(), 1);
      auto arg0 = op->args[0];
      auto CheckCast = [] (const Expr &input) -> bool {
        auto cast = input.as<Cast>();
        if (cast == nullptr) {
          return false;
        }
        return true;
      };

      while (CheckCast(arg0)) {
        Type t = arg0.as<Cast>()->type;
        os << "(";
        PrintType(t, os);
        os << "*)";
        arg0 = arg0.as<Cast>()->value;
      }
      os << op->name << "(";
      this->PrintExpr(arg0, os);
      os << ")";
      return;
    }

    if ((op->name == AKG_REDUCE) || (op->name == AKG_ATOMIC_RETURN) ||
        (op->name == PARIS_REDUCE) || (op->name == PARIS_ATOMIC_RETURN)) {
      CHECK_GE(op->args.size(), 2);
      os << op->name << "<";
      Expr template_arg0 = op->args[0];
      Expr template_arg1 = op->args[1];
      this->PrintType(template_arg0.type(), os);
      os << ",";
      CHECK(template_arg1.as<StringImm>());
      os << template_arg1.as<StringImm>()->value;
      os << ">(";
      for (size_t i = 2; i < op->args.size(); i++) {
        this->PrintExpr(op->args[i], os);
        if (i < op->args.size() - 1) {
          os << ", ";
        }
      }
      os << ")";
      return;
    }
    if (op->name == AKG_KAHAN) {
      CHECK_GE(op->args.size(), 1);
      os << op->name << "<";
      Expr template_arg0 = op->args[0];
      this->PrintType(template_arg0.type(), os);
      os << ">(";
      for (size_t i = 1; i < op->args.size(); i++) {
        this->PrintExpr(op->args[i], os);
        if (i < op->args.size() - 1) {
          os << ", ";
        }
      }
      os << ")";
      return;
    }
    CodeGenC::VisitExpr_(op, os);
  } else if (op->is_intrinsic(Call::reinterpret_cast_op)) {
    os << "*(reinterpret_cast<";
    PrintType(op->args[1].type(), os);
    if (op->args[0].as<IntImm>()->value > 0) os << op->args[0];
    os << "*>(&";
    auto ld = op->args[1].as<Load>();
    if (ld) {
      auto var_name = ld->buffer_var->name_hint;
      if (std::find_if(vec_loads.begin(), vec_loads.end(),
                       [var_name](const Variable *v) { return (v->name_hint == var_name); }) != vec_loads.end()) {
        os << "sw_" + var_name;
      } else this->PrintExpr(op->args[1], os);
    } else this->PrintExpr(op->args[1], os);
    os << "))";
  } else if (op->is_intrinsic(Call::ldg)) {
    os << "__ldg((";
    PrintType(op->args[1].type(), os);
    os << this->PrintExpr(op->args[0]) << " *)(&" << this->PrintExpr(op->args[1]) << "))";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCUDA::VisitStmt_(const LetStmt* op) {
  if (no_init_value){
    no_init_value = false;
    PrintIndent();
    if (op->var.type() == Handle() &&
        handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* "
             << AllocVarID(op->var.get())
             << ";\n";
    } else {
      PrintType(op->var.type(), this->stream);
      this->stream << ' '
                   << AllocVarID(op->var.get())
                   << ";\n";
    }
    PrintStmt(op->body);
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenCUDA::VisitExpr_(const Variable* op, std::ostream& os) {
  // replace cce var with const index
  if(replace_cce && loop_var == op){
    os << current_index;
  }
  else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCUDA::VisitExpr_(const Load* op, std::ostream& os) {
  int lanes = op->type.lanes();
  if (vec_store) {
    static const char access[] = {'x', 'y', 'z', 'w', 'a', 'b', 'c', 'd'};
    if (lanes == 2 || lanes == 4) {
      os << op->buffer_var->name_hint << "." << access[current_index];
    } else if(std::find_if(vec_loads.begin(), vec_loads.end(),
                           [op] (const Variable* v) { return (v->name_hint==op->buffer_var->name_hint); }) != vec_loads.end()){
      os << "sw_" << op->buffer_var->name_hint << "." << access[current_index];
    } else{
      // temp variable
      os << op->buffer_var->name_hint << "[";
      PrintExpr(op->index, os);
      os << "]";
    }
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCUDA::VisitStmt_(const Store* op) {
  Type t = op->value.type();
  if (is_reinterpret && t.lanes() == 1) {
    is_reinterpret = false;

    std::string value = this->PrintExpr(op->value);
    std::string ref  = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();

    Type elem_type = t.element_of();
    stream << "*(reinterpret_cast<";
    PrintType(elem_type, stream);
    stream << loop_extent << "*>(&" << ref << ")) = " << value << ";\n";

  } else if (vec_store) {
    replace_cce = true;
    static const char access[] = {'x', 'y', 'z', 'w', 'a', 'b', 'c', 'd'};
    int lanes = op->buffer_var.type().lanes();
    loop_extent = lanes;
    for (int i = 0; i < lanes; i++){
      this->PrintIndent();
      current_index = i;
      stream << op->buffer_var->name_hint << "." << access[i] << " = ";
      PrintExpr(op->value, stream);
      stream << ";\n";
    }
    replace_cce = false;
    vec_store = false;
  } else if (simple_store){
    simple_store = false;
    std::string value = this->PrintExpr(op->value);
    this->PrintIndent();
    stream << op->buffer_var->name_hint << " = " << value << ";\n";
  } else{
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenCUDA::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == attr::wmma_scope) {
    const StringImm* scope_str = op->value.as<StringImm>();
    wmma_scope = scope_str->value;
  } else if (op->attr_key == attr::fragment_shape) {
    const Variable* buffer = op->node.as<Variable>();
    const StringImm* shape_str = op->value.as<StringImm>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == attr::fragment_layout) {
    const Variable* buffer = op->node.as<Variable>();
    const StringImm* layout_str = op->value.as<StringImm>();
    fragment_layouts[buffer] = layout_str->value;
  } else if (op->attr_key == REDUCE_LIB_TYPE) {
    reduce_lib_type_ = op->value.as<StringImm>()->value;
    need_reduce_lib_ = true;
  } else if (op->attr_key == "pragma_tensor_core") {
    CHECK(op->value.as<StringImm>());
    if (op->value.as<StringImm>()->value == "2") {
      is_scheme_two_ = true;
    }
  } else if (op->attr_key == "total_shared_memory") {
    this->PrintIndent();
    stream << "__shared__ char total_shared_memory[" << op->value.as<IntImm>()->value << "];\n";
  } else if (op->attr_key == "shared_memory_offset") {
    const Variable* buffer = op->node.as<Variable>();
    int offset = op->value.as<IntImm>()->value;
    sm_offsets[buffer] = offset;
  } else if (op->attr_key == "reinterpret_store") {
    loop_extent = op->value;
    // mark next store statement to be a reinterpret cast
    is_reinterpret = true;
  } else if (op->attr_key == "vec_store") {
    loop_var = op->value.as<Variable>();
    // mark next store statement to be a vector store
    vec_store = true;
  } else if (op->attr_key == "simple_store") {
    // mark next store statement to be a basic store of type a = b
    simple_store = true;
  } else if (op->attr_key == "vec_load") {
    vec_loads.insert(op->value.as<Variable>());
  } else if (op->attr_key == "no_init_value") {
    // mark next let statement to be a simple, empty declaration
    no_init_value = true;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::VisitStmt_(const Allocate* op) {
  if (is_zero(op->condition)) {
    stream << "// ";
  }
  std::string vid = AllocVarID(op->buffer_var.get());
  if (op->new_expr.defined()) {
    // Prefer global static allocation for the program
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* "<< vid << '=' << new_data << ";\n";
  } else if (sm_offsets.find(op->buffer_var.as<Variable>()) != sm_offsets.end()) {
    // e.g:
    // __shared__ float input_shared[1000];
    // ==>
    // float* input_shared = (float*)(total_shared_memory + 0);
    const Variable* buffer = op->buffer_var.as<Variable>();
    int offset = sm_offsets.at(buffer);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* " << vid << " = (";
    PrintType(op->type, stream);
    stream << "*)(total_shared_memory + " << offset << ");\n";
  } else {
    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";
    const Variable* buffer = op->buffer_var.as<Variable>();
    std::string scope = alloc_storage_scope_.at(buffer);
    if (scope.find("wmma.") == 0) {
      std::string matrix_scope = scope;
      auto pos = matrix_scope.find(".");
      if (pos != std::string::npos) {
        matrix_scope = matrix_scope.substr(pos + 1);
      }
      matrix_abc.insert(std::make_pair(akg::common::GetGlobalName(vid), matrix_scope));
      if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
        CHECK(op->type == Float(16) || op->type == Int(8) || op->type == UInt(8))
          << "Matrix_a and matrix_b only support half or char or unsigned char type for now";
      } else {
        CHECK(op->type == Float(16) || op->type == Float(32) || op->type == Int(32))
          << "Accumulator only support half, float and int type for now";
      }
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
      PrintWmmaScope(scope, op->type, buffer, stream);
    } else {
      PrintStorageScope(scope, stream);
      stream << ' ';
      PrintType(op->type, stream);
    }
    stream << ' '<< vid << '['
           << constant_size << "];\n";
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenCUDA::VisitStmt_(const Evaluate *op) {
  if (is_const(op->value)) return;
  const Call* call = op->value.as<Call>();
  if (call && call->is_intrinsic(intrinsic::tvm_global_barrier_kinit)) {
    PrintIndent();
    stream << "__shared__ unsigned " << vid_global_barrier_expect_ << ";\n";
    PrintIndent();
    stream << "if (threadIdx.x == 0) {\n";
    PrintIndent();
    stream << "  " << vid_global_barrier_expect_ << " = 0;\n";
    PrintIndent();
    stream << "}\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenCUDA::VisitExpr_(const Ramp* op, std::ostream& os) {
  os << "((make_int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")" << "+(" << PrintExpr(op->stride) << "*" << i <<")";
    if (i != op->lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenCUDA::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  if (vec_store) {
    PrintExpr(op->value, os);
    return;
  }
  if (op->type.is_int() && op->type.bits() == 8 && op->lanes == 4) {
    // make_int8x4
    const int64_t *p = as_const_int(op->value);
    CHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    os << "(int)" << v;
    return;
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->type, os);
  os << '(';
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenCUDA::VisitExpr_(const Shuffle* op, std::ostream &os) {
  std::vector<std::string> to_shuffle(op->vectors.size());
  for (int i = 0, e = op->vectors.size(); i < e; ++i) {
    CHECK(op->vectors[i].type().lanes() == 1) << "Only scalars can be shuffled in CUDA!";
    to_shuffle[i] = PrintExpr(op->vectors[i]);
  }
  os << "make_";
  PrintType(op->type, os);
  os << '(';
  for (int i = 0, e = op->indices.size(); i < e; ++i) {
    const int64_t *val = as_const_int(op->indices[i]);
    CHECK(val && *val >= 0 && (int) *val < (int) to_shuffle.size());
    if (i != 0) os << ", ";
    os << to_shuffle[*val];
  }
  os << ')';
}

inline void PrintConst(const FloatImm* op, std::ostream& os, CodeGenCUDA* p) { // NOLINT(*)
  switch (op->type.bits()) {
    case 64: case 32: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << ((op->type.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
        p->need_math_constants_h_ = true;
      } else if (std::isnan(op->value)) {
        temp << ((op->type.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
        p->need_math_constants_h_ = true;
      } else {
        temp << std::scientific << op->value;
        if (op->type.bits() == 32) temp << 'f';
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << "__float2half_rn";
      os << '(' << std::scientific << op->value << 'f' << ')';
      break;
    }
    default: LOG(FATAL) << "Bad bit-width for float: " << op->type << "\n";
  }
}


void CodeGenCUDA::VisitExpr_(const FloatImm *op, std::ostream& os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenCUDA::PrintWmmaScope(const std::string &scope, Type t,
    const Variable* variable, std::ostream &os) {
  std::stringstream type;
  PrintType(t, type);
  std::string shape_str = fragment_shapes[variable];
  auto pos_wmma = scope.find("wmma");
  if (pos_wmma != std::string::npos) {
    os << wmma_scope << "::";
  }
  if (scope == "wmma.matrix_a") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    os << "wmma::fragment<nvcuda::wmma::matrix_a, "
      << shape_str << ", " << type.str() << ", nvcuda::wmma::" << layout_str <<">";
  } else if (scope == "wmma.matrix_b") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    os << "wmma::fragment<nvcuda::wmma::matrix_b, "
       << shape_str << ", " << type.str() << ", nvcuda::wmma::" << layout_str <<">";
  } else if (scope == "wmma.accumulator") {
    need_mma_h_ = true;
    os << "wmma::fragment<nvcuda::wmma::accumulator, "
       << shape_str << ", "<< "float" << ">";
  }
}

int32_t CodeGenCUDA::GetWmmaFragmentSize(const std::string &scope,
                                         const Variable* variable, int32_t size) {
  std::string shape_str = fragment_shapes[variable];
  size_t m, n, k;
  size_t last_pos = 0, pos = 0;
  pos = shape_str.find(", ", last_pos);
  m = std::stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  pos = shape_str.find(", ", last_pos);
  n = std::stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  k = std::stoi(shape_str.substr(last_pos, shape_str.length() - last_pos));
  if (scope == "wmma.matrix_a") {
    return size / m / k;
  } else if (scope == "wmma.matrix_b") {
    return size / n / k;
  } else if (scope == "wmma.accumulator") {
    return size / m / n;
  }
  return 0;
}

void CodeGenCUDA::HandleVolatileLoads(const std::string& value,
                                      const Load* op, std::ostream& os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  if (op->type.is_float16() && IsVolatile(op->buffer_var.get())) {
    os << "(";
    PrintType(op->type, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

}  // namespace codegen
}  // namespace air
