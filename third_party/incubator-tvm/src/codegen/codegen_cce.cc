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

/*
 * 2019.12.30 - Add file codegen_cce.cc.
 */

#include <tvm/base.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <arithmetic/compute_expr.h>
#include <contrib/cce_parm/cceconf.h>
#include <iomanip>
#include <vector>
#include <string>
#include "src/common/util.h"
#include "codegen/codegen_cce.h"
#include "algorithm"


using namespace akg::cceconf;

namespace air {
namespace codegen {
using namespace air;
using namespace air::ir;

void PrintMemoryQualifier(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  if (scope == "local.UB") {
    os << "__ubuf__ ";
  } else if (scope == "local.L1") {
    os << "__cbuf__ ";
  } else if (scope == "local.L0A") {
    os << "__ca__ ";
  } else if (scope == "local.L0B") {
    os << "__cb__ ";
  } else if (scope == "local.L0C") {
    os << "__cc__ ";
  } else if (scope == "local.REG") {
    os << "";
  } else {
    os << "__gm__ ";
  }
}

void PrintOverflowCheck(std::ostream& os) {
  os << "  \n";
  os << "  uint64_t *statusInScalarBuffer = (uint64_t *) 0x40000;\n";
  os << "  uint64_t status0 = get_status();\n";
  os << "  uint64_t overflowStatus = status0 & 0x00000000000005B8;\n";
  os << "  if (overflowStatus > 0) {\n";
  os << "    *statusInScalarBuffer = 1;\n";
  os << "  }\n";
}

bool PrintTypeFloat(const Type& t, std::ostream& os, bool fail, int lanes) {  // NOLINT(*)
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) {
      return true;
    } else if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void PrintTypeInt(const Type& t, std::ostream& os, int lanes) {  // NOLINT(*)
  if (t.bits() == 1) {
    os << "bool";
    return;
  }

  if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int";
      return;
    }
    switch (t.bits()) {
      case 8:
        os << "int8_t";
        break;
      case 16:
        os << "int16_t";
        break;
      case 32:
        os << "int32_t";
        break;
      case 64:
        os << "int64_t";
        break;
    }
  }
}

void CodeGenCCE::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }
  if (t.is_float()) {
    bool fail = false;
    bool flag = PrintTypeFloat(t, os, fail, lanes);
    if (flag == false) {
      LOG(FATAL) << "Cannot convert type " << t << " to CCE type";
    }
  }
  if (t.is_uint() || t.is_int()) {
    PrintTypeInt(t, os, lanes);
  }
}

CodeGenCCE::CodeGenCCE() {
  restrict_keyword_ = "__restrict__";
  iterName_ = "";
}

void CodeGenCCE::Initialize(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  tag = false;
}

void CodeGenCCE::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  if (!tag) {  // aicore need to check the scope,aicpu is not need
    CHECK_NE(scope, "global");
  }
  if (scope == "shared") {
    os << "__shared__";
  }
}

void CodeGenCCE::AddFunctionCore(const LoweredFunc f) {
  this->gvar_cce_.push_back(std::string("VA0"));
  this->gvar_cce_.push_back(std::string("VA1"));
  this->gvar_cce_.push_back(std::string("VA2"));
  this->gvar_cce_.push_back(std::string("VA3"));

  this->stream << "#ifdef __CCE_KT_TEST__\n"
               << "#define __aicore__ \n"
               << "#else\n"
               << "#define __aicore__ [aicore]\n"
               << "#endif\n\n";
  this->stream << "extern \"C\"  __global__ __aicore__ ";
  // clear previous generated state.
  this->InitFuncState(f);
  // skip the first underscore, so SSA variable starts from _1
  static_cast<void>(GetUniqueName("_"));
  // add to alloc buffer type.
  for (const auto& kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }
  std::string kernel_name = f->name.substr(0, f->name.find("_kernel", 0));
  this->stream << "void " << kernel_name << "_kernel0"
               << "(";
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    if (handle_data_type_.count(v.get())) {
      stream << "__gm__ ";
      PrintType(handle_data_type_.at(v.get()), stream);
      stream << "*";
      if (f->is_restricted && restrict_keyword_.length() != 0) {
        stream << ' ' << restrict_keyword_;
      }
    } else {
      PrintType(v.type(), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  CceConf* conf = CceConf::getInstance();
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  const std::string product_name = conf->getProductName();
  if (product_name == "cloud") {
    PrintOverflowCheck(this->stream);
  }
  this->stream << "}\n\n";
}

void CodeGenCCE::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  const auto buffer = op->buffer_var.as<Variable>();
  CHECK(buffer);
  std::string scope = alloc_storage_scope_.at(buffer);
  std::string vid = AllocVarID(op->buffer_var.get());
  if (tag) {
    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";
    PrintStorageScope(scope, stream);
    stream << ' ';
    PrintType(op->type, stream);
    stream << "* " << vid << " = "
           << "(";
    PrintType(op->type, stream);
    stream << '*' << ')' << "aicpu_malloc" << '(' << constant_size << '*' << (op->type.bits()) / 8 << ");\n";
    this->free_ids_.push_back(vid);
    RegisterHandleType(op->buffer_var.get(), op->type);
    this->PrintStmt(op->body);
  } else if (scope == "local.REG") {
    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";
    PrintType(op->type, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
    RegisterHandleType(op->buffer_var.get(), op->type);
    this->PrintStmt(op->body);
  } else {
    this->PrintIndent();
    PrintMemoryQualifier(scope, stream);
    if (op->new_expr.defined()) {
      CHECK_EQ(op->free_function, "nop");
      stream << ' ';
      PrintType(op->type, stream);
      /* modify start for O2 build problem */
      const auto buffer_new = op->buffer_var.as<Variable>();
      CHECK(buffer_new);
      std::string scope_new = alloc_storage_scope_.at(buffer_new);
      stream << "* " << vid << " = (";
      PrintMemoryQualifier(scope_new, stream);
      stream << ' ';
      PrintType(op->type, stream);
      stream << " *)(" << PrintExpr(op->new_expr) << ");\n";
      /* modify end for O2 build problem */
    } else {
      stream << ' ';
      int32_t constant_size = op->constant_allocation_size();
      CHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";
      const auto buffer_old = op->buffer_var.as<Variable>();
      CHECK(buffer_old);
      std::string scope_old = alloc_storage_scope_.at(buffer_old);
      PrintStorageScope(scope_old, stream);
      stream << ' ';
      PrintType(op->type, stream);
      stream << ' ' << vid << '[' << constant_size << "];\n";
    }
    RegisterHandleType(op->buffer_var.get(), op->type);
    this->PrintStmt(op->body);
  }
}

void CodeGenCCE::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = CastFromTo("block_idx" /*iv->thread_tag*/, UInt(32), iv->var.type());
}

void CodeGenCCE::VisitStmt_(const Evaluate* op) {
  if (air::is_const(op->value)) return;
  const auto call = op->value.as<Call>();
  if (call && call->is_intrinsic(intrinsic::tvm_global_barrier_kinit)) {
  } else {
    if (air::is_const(op->value)) return;
    const auto call_ = op->value.as<Call>();
    if (call_) {
      if (call_->is_intrinsic(intrinsic::tvm_storage_sync)) {
        return;
      } else if (call_->is_intrinsic(intrinsic::tvm_struct_set)) {
        CHECK_EQ(call_->args.size(), 4);
        std::string value = PrintExpr(call_->args[3]);
        if (const auto v = call_->args[2].as<IntImm>()) {
          std::string ref =
            GetStructRef(call_->args[3].type(), call_->args[0], call_->args[1], static_cast<int>(v->value));
          this->PrintIndent();
          this->stream << ref << " = " << value << ";\n";
          return;
        } else {
          return;
        }
      }
    }
    std::string vid = this->PrintExpr(op->value);
    if (call_ && call_->name != "null_op") {
      this->PrintIndent();
    }
    if (!(call_ && call_->name == "null_op")) {
      this->stream << vid << ";\n";
    }
  }
}

void CodeGenCCE::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == "pragma_insn_comment") {
    // Add comment
    CHECK(op->value.as<StringImm>());
    auto comments = air::common::Split(op->value.as<StringImm>()->value, '#');
    for (auto comment : comments) {
      if (!comment.empty()) {
        PrintIndent();
        stream << "/// \\param " << comment << "\n";
      }
    }
    PrintIndent();
    stream << "/// \\code\n";
    this->PrintStmt(op->body);
    PrintIndent();
    stream << "/// \\endcode\n";
  } else {
    // Call parent's function
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenCCE::VisitExpr_(const StringImm* op, std::ostream& os) {  // NOLINT(*)
  auto iter = find(gvar_cce_.begin(), gvar_cce_.end(), op->value);
  if ("nullptr" == op->value) {
    os << "nullptr";
  } else if (iter != gvar_cce_.end()) {
    os << op->value;
  } else {
    os << "\"" << op->value << "\"";
  }
}

void CodeGenCCE::VisitExpr_(const Call* op, std::ostream& os) {  // NOLINT(*)
  if (op->call_type == Call::Extern || op->call_type == Call::PureExtern) {
    PrintExternCall(op, os);
  } else if (op->is_intrinsic(intrinsic::tvm_address_of) || op->is_intrinsic(intrinsic::tvm_if_then_else) ||
             op->is_intrinsic(Call::bitwise_and)) {
    // if need other condition, pls add it form CodeGenC::VisitExpr_(const Call *op, std::ostream& os) func
    CodeGenC::VisitExpr_(op, os);
  } else if (op->call_type == Call::Intrinsic || op->call_type == Call::PureIntrinsic) {
    PrintPureCall(op, os);
  }
}

void CodeGenCCE::VisitExpr_(const Max* op, std::ostream& os) {  // NOLINT(*)
  os << "max" << '(';
  this->PrintExpr(op->a, os);
  os << ", ";
  this->PrintExpr(op->b, os);
  os << ')';
}

void CodeGenCCE::VisitExpr_(const FloatImm* op, std::ostream& os) {  // NOLINT(*)
  CHECK(op != nullptr);
  switch (op->type.bits()) {
    case 64: {
      std::ostringstream os64;
      os64 << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::scientific << op->value;
      this->MarkConst(os64.str());
      os << os64.str();
      break;
    }
    case 32: {
      std::ostringstream os32;
      // if the tvm.const is like 1.123456789 with float32 type. normally, it
      // will print 1.123457e+00f, but we want to get 1.1234568. Set precision
      // as what we want.
      os32 << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::scientific << op->value << 'f';
      this->MarkConst(os32.str());
      os << os32.str();
      break;
    }
    case 16: {
      os << '(';
      this->PrintType(op->type, os);
      os << ')' << std::setiosflags(std::ios::fixed) << std::scientific << op->value << 'f';
      break;
    }
    default: {
      LOG(FATAL) << "Bad bit-width for float: " << op->type << "\n";
    }
  }
}

void CodeGenCCE::PrintRegmov(const Call* op, std::ostream& os) {
  // decl with the param1
  int flag = 0;
  CHECK_GT(op->args.size(), 0);
  const auto opn = op->args[0].as<Call>();
  CHECK(opn);
  CHECK_GT(opn->args.size(), 0);
  const auto l = opn->args[0].as<Load>();
  CHECK(l);
  const auto buffer = l->buffer_var.as<Variable>();
  std::string scope = alloc_storage_scope_.at(buffer);
  if (scope == "local.REG") {
    flag = 1;
    this->PrintExpr(opn->args[0], os);
  } else {
    os << "(*(";
    PrintMemoryQualifier(scope, os);
    PrintType(op->type.element_of(), os);
    os << " * )";
    this->PrintCCEIntrinArgsType(op->args[0], os);
    os << " ) ";
  }

  os << " = ";
  // if the  left  type of "="  is reg,then the right type of "=" is  reg also.
  if (flag == 1) {
    os << "(";
    PrintType(opn->type.element_of(), os);
    os << ") ";
  }

  const auto opnn = op->args[1].as<Call>();
  if (opnn && op->args.size() == 2 && opnn->name == "reg") {  // if the second param is reg,then has not the third param
    this->PrintExpr(opnn->args[0], os);
  } else {
    // decl with the other param
    if (opnn &&
        opnn->is_intrinsic(intrinsic::tvm_address_of)) {  // if the second param is buf ptr, get value form the ptr
      const auto lo = opnn->args[0].as<Load>();
      CHECK(lo);
      const auto buffer_ = lo->buffer_var.as<Variable>();
      if (buffer_) {
        scope = alloc_storage_scope_.at(buffer_);
      }
      os << "(*( ";
      PrintMemoryQualifier(scope, os);
    } else {
      os << "(( ";
    }
    PrintType(op->type.element_of(), os);
    os << "* ) (";
    for (unsigned i = 1; i < static_cast<unsigned>(op->args.size()); i++) {
      this->PrintCCEIntrinArgsType(op->args[i], os);
      if (i < static_cast<unsigned>(op->args.size() - 1)) {
        os << " + ";
      }
    }
    os << "))";
  }
}

void CodeGenCCE::PrintArgmaxCast(const Call* op, std::ostream& os) {
  // decl with the param1
  CHECK_GT(op->args.size(), 0);
  const auto opn = op->args[0].as<Call>();
  CHECK(opn);
  CHECK_GT(opn->args.size(), 0);
  const auto l = opn->args[0].as<Load>();
  CHECK(l);
  const auto buffer = l->buffer_var.as<Variable>();
  std::string scope = alloc_storage_scope_.at(buffer);

  os << "(*(";
  PrintMemoryQualifier(scope, os);
  PrintType(op->type.element_of(), os);
  os << " * )";
  this->PrintCCEIntrinArgsType(op->args[0], os);
  os << " ) ";
  os << " = ";

  const auto opnn = op->args[1].as<Call>();
  // decl with the other param
  if (opnn &&
      opnn->is_intrinsic(intrinsic::tvm_address_of)) {  // if the second param is buf ptr, get value form the ptr
    const auto lo = opnn->args[0].as<Load>();
    CHECK(lo);
    const auto buffer_ = lo->buffer_var.as<Variable>();
    if (buffer_) {
      scope = alloc_storage_scope_.at(buffer_);
    }
    os << "(*( ";
    PrintMemoryQualifier(scope, os);
  } else {
    os << "(( ";
  }
  PrintType(UInt(16), os);
  os << "* ) (";
  for (unsigned i = 1; i < static_cast<unsigned>(op->args.size()); i++) {
    this->PrintCCEIntrinArgsType(op->args[i], os);
    if (i < static_cast<unsigned>(op->args.size() - 1)) {
      os << " + ";
    }
  }
  os << "))";
}

void CodeGenCCE::PrintPointerCast(const Call* op, std::ostream& os) {
  // decl with the param1
  CHECK_GT(op->args.size(), 0);
  const auto opn = op->args[0].as<Call>();
  CHECK(opn);
  CHECK_GT(opn->args.size(), 0);
  const auto l = opn->args[0].as<Load>();
  CHECK(l);
  const auto buffer = l->buffer_var.as<Variable>();
  std::string scope = alloc_storage_scope_.at(buffer);
  if (scope == "local.REG") {
    this->PrintExpr(opn->args[0], os);
  } else {
    os << "(*(";
    PrintMemoryQualifier(scope, os);
    PrintType(op->type.element_of(), os);
    os << " * )";
    this->PrintCCEIntrinArgsType(op->args[0], os);
    os << " ) ";
  }

  os << " = ";

  // decl with the other param
  os << "(( ";
  PrintType(UInt(64), os);
  os << ") (";
  for (unsigned i = 1; i < static_cast<unsigned>(op->args.size()); i++) {
    this->PrintCCEIntrinArgsType(op->args[i], os);
    if (i < static_cast<unsigned>(op->args.size() - 1)) {
      os << " + ";
    }
  }
  os << "))";
}

void CodeGenCCE::PrintBitMove(const Call* op, std::ostream& os, bool left) {
  CHECK_GT(op->args.size(), 0);
  const auto l = op->args[0].as<Load>();
  Expr r = op->args[1];
  CHECK(l);
  CHECK(r.defined());
  const auto buffer = l->buffer_var.as<Variable>();
  std::string scope = alloc_storage_scope_.at(buffer);
  if (scope == "local.REG") {
    this->PrintExpr(op->args[0], os);
  }

  if (left) {
    os << " << ";
  } else {
    os << " >> ";
  }

  // decl with the other param
  os << "(";
  PrintType(r.type(), os);
  os << ")";
  this->PrintExpr(r, os);
}

void CodeGenCCE::PrintSetAtomicAdd(const Call* op, std::ostream& os, bool open) {
  CHECK_LT(op->args.size(), 1);
  CceConf* conf = CceConf::getInstance();
  const std::string product_name = conf->getProductName();
  if (product_name != "cloud") {
    LOG(INFO) << "Atomic add only support cloud.";
  }

  if (open) {
    os << "set_ctrl((get_ctrl() & 0xcfffffffffffffff) | ((uint64_t)(0x1) << 60))";
  } else {
    os << "set_ctrl((get_ctrl() & 0xcfffffffffffffff) | ((uint64_t)(0x0) << 60))";
  }
}

void CodeGenCCE::PrintCCEIntrinArgsType(const Expr& e, std::ostream& os) {  // NOLINT(*)
  const auto call = e.as<Call>();
  if (!tag && call && call->is_intrinsic(intrinsic::tvm_address_of)) {
    const auto l = call->args[0].as<Load>();
    CHECK(call->args.size() == 1 && l);
    os << "((";
    const auto buffer = l->buffer_var.as<Variable>();
    std::string scope = "__gm__";
    if (alloc_storage_scope_.find(buffer) != alloc_storage_scope_.end()) scope = alloc_storage_scope_.at(buffer);
    PrintMemoryQualifier(scope, os);
    PrintType(l->type.element_of(), os);
    os << " *)" << this->GetVarID(buffer) << " + ";
    this->PrintExpr(l->index, os);
    os << ')';
    return;
  } else if (e->IsInstance<Load>()) {
    const auto l = e.as<Load>();
    os << "(";
    CHECK(l);
    PrintType(l->type.element_of(), os);
    os << ")";
  }

  this->PrintExpr(e, os);
}

void CodeGenCCE::PrintExternCall(const Call* op, std::ostream& os) {
  if (op->name != "null_op") {
    if (op->name == "reg_mov") {
      // call func to print, save the Cyclomatic complexity
      PrintRegmov(op, os);
    } else if (op->name == "argmax_cast") {
      // special codegen for argmax data cast
      PrintArgmaxCast(op, os);
    } else if (op->name == "printer_cast") {
      // special codegen for printer cast to uint64
      PrintPointerCast(op, os);
    } else if (op->name == "bit_move_left") {
      PrintBitMove(op, os, true);
    } else if (op->name == "bit_move_right") {
      PrintBitMove(op, os, false);
    } else if (op->name == "set_atomic_add_open") {
      PrintSetAtomicAdd(op, os, true);
    } else if (op->name == "set_atomic_add_close") {
      PrintSetAtomicAdd(op, os, false);
    } else {
      os << op->name << "(";
      for (unsigned i = 0; i < static_cast<unsigned>(op->args.size()); i++) {
        PrintCCEIntrinArgsType(op->args[i], os);
        if (i < static_cast<unsigned>(op->args.size() - 1)) {
          os << ", ";
        }
      }
      os << ")";
    }
  }
}

inline void PrintBinaryIntrinsic(const Call* op, const char* opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeGenCCE* p) {
  if (op->type.lanes() == 1) {
    CHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  }
}

void CodeGenCCE::PrintPureCall(const Call* op, std::ostream& os) {
  if (op->is_intrinsic(intrinsic::tvm_cce_string_print)) {
    for (unsigned i = 0; i < static_cast<unsigned>(op->args.size() - 1); i++) {
      if (const auto v = op->args[i].as<StringImm>()) {
        os << v->value << ", ";
      }
    }
    if (const auto ve = op->args[op->args.size() - 1].as<StringImm>()) {
      os << ve->value;
    }
  } else if (op->is_intrinsic(Call::bitwise_and)) {
    PrintBinaryIntrinsic(op, " & ", os, this);
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    PrintBinaryIntrinsic(op, " | ", os, this);
  } else if (op->is_intrinsic(Call::shift_right)) {
    PrintBinaryIntrinsic(op, " >> ", os, this);
  } else if (op->is_intrinsic(Call::shift_left)) {
    PrintBinaryIntrinsic(op, " << ", os, this);
  } else {
    if (tag) {  // when aicpu, change the intrinsic type in the intrin rule
      PrintExternCall(op, os);
    } else {
      os << op->name << " :printPureCall\n";
    }
  }
}
}  // namespace codegen
}  // namespace air
