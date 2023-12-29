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

#include "build_module.h"
#include <fstream>
#include <iostream>
#include <map>
#include <pass/utils.h>
#include <stdexcept>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tvm.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
using namespace std;

namespace akg {
namespace ir {
// var index
int cc = -1;
// flag for input/output that do not have realize op
bool is_io = false;
// flag for leave of expr
bool i = false;
// map var name to memref
map<string, string> memref = {};
// map var index to datatype
map<int, string> datatype = {};
// map loop var to its extern
map<string, int64_t> iter_ext = {};
// for allocing input/output tensor, size is needed
int64_t u, left, right;
// for allocing input/output tensor, datatype is needed
string io_datatype;
// string to dump
stringstream mlirstring;
// file to dump
ofstream outfile;

inline void VisitArray(const Array<Expr> &arr, IRVisitor *v) {
  for (size_t i = 0; i < arr.size(); i++) {
    v->Visit(arr[i]);
  }
}
inline void VisitRDom(const Array<IterVar> &rdom, IRVisitor *v) {
  for (size_t i = 0; i < rdom.size(); i++) {
    Range r = rdom[i]->dom;
    v->Visit(r->min);
    v->Visit(r->extent);
  }
}

/*Input output tensor do not have realize op. But each var in MLIR need to be
alloced.
This function is used to alloc input/output tensor.
*/
inline void AllocInpOut(const string name, Array<Expr> args, IRVisitor *v) {
  stringstream mem;
  mem << "memref<";
  is_io = true;
  for (size_t i = 0; i < args.size(); i++) {
    v->Visit(args[i]);
    LOG(DEBUG) << i << " " << args[i];
    mem << u + 1 << "x";
  }
  mem << io_datatype << ">\n";
  string s = mem.str();
  memref[name] = s;
  LOG(DEBUG) << "alloc input " << name << " " << s;
  outfile << "%" << name << " = alloc() : " << s;
  is_io = false;
}

// Throw not implement error
inline void throw_not_implement_error() { throw "Not implement yet"; }

/*Hilide IR visitor*/
class HalideIRVisitor : public IRVisitor {
public:
  HalideIRVisitor() {}
  ~HalideIRVisitor() override = default;

  /*Leaf of the tree*/
  void Visit_(const Variable *op) { u = iter_ext[op->name_hint]; }

  void Visit_(const For *op) final {
    // Record loop var name and its extent
    iter_ext[op->loop_var->name_hint] =
        op->extent.as<air::ir::IntImm>()->value - 1;
    // Hilide for op -> affine.for
    mlirstring << "affine.for %" << op->loop_var << " = " << op->min << " to "
               << op->extent + op->min << " {\n";
    IRVisitor *v = this;
    v->Visit(op->min);
    v->Visit(op->extent);
    v->Visit(op->body);
    mlirstring << "}\n";
  }

  void Visit_(const IfThenElse *op) final {
    this->Visit(op->condition);
    // Hilide ItThenElse -> scf.if
    mlirstring << "scf.if %" << cc << " {\n";
    this->Visit(op->then_case);
    if (op->else_case.defined()) {
      mlirstring << "} else {\n";
      this->Visit(op->else_case);
    }
    mlirstring << "}\n";
  }

  void Visit_(const Call *op) final {
    // No need to visit op->args
    // call by cases
    if (std::strcmp(op->name.c_str(), "exp") == 0) {
      VisitArray(op->args, this);
      int b = cc;
      cc += 1;
      datatype.insert(pair<int, string>(cc, datatype.at(cc - 1)));
      mlirstring << "    %" << cc << " = exp %" << b << " : "
                 << datatype.at(cc - 1) << "\n";
    } else if (std::strcmp(op->name.c_str(), "fabs") == 0) {
      VisitArray(op->args, this);
      int b = cc;
      cc += 1;
      datatype.insert(pair<int, string>(cc, datatype.at(cc - 1)));
      mlirstring << "    %" << cc << " = absf %" << b << " : "
                 << datatype.at(cc - 1) << "\n";
    } else if (std::strcmp(op->name.c_str(), "log") == 0) {
      VisitArray(op->args, this);
      int b = cc;
      cc += 1;
      datatype.insert(pair<int, string>(cc, datatype.at(cc - 1)));
      mlirstring << "    %" << cc << " = log %" << b << " : "
                 << datatype.at(cc - 1) << "\n";
    } else if (std::strcmp(op->name.c_str(), "sqrt") == 0) {
      VisitArray(op->args, this);
      int b = cc;
      cc += 1;
      datatype.insert(pair<int, string>(cc, datatype.at(cc - 1)));
      mlirstring << "    %" << cc << " = sqrt %" << b << " : "
                 << datatype.at(cc - 1) << "\n";
    } else if (op->call_type == Call::Halide) {
      cc += 1;
      stringstream ss;
      ss << op->args;
      string args_str = ss.str();
      string args_str_print = "";
      for (size_t i = 0; i < args_str.size(); i++) {
        if (std::isalpha(args_str[i]) && args_str[i - 1] != '_' &&
            !std::isalpha(args_str[i - 1]) && !std::isdigit(args_str[i - 1]))
          args_str_print = args_str_print + "%";
        args_str_print = args_str_print + args_str[i];
      }
      if (memref.find(op->name) == memref.end()) {
        AllocInpOut(op->name, op->args, this);
      }
      mlirstring << "    %" << cc << " = affine.load %" << op->name
                 << args_str_print << ": " << memref.at(op->name);
      size_t lasttime = memref.at(op->name).rfind("x");
      io_datatype = memref.at(op->name).substr(
          lasttime + 1, memref.at(op->name).size() - lasttime - 3);
      datatype.insert(pair<int, string>(cc, io_datatype));
    } else {
      throw_not_implement_error();
    }
  }
#define DEFINE_BINOP_VISIT_(OP)                                                \
  void Visit_(const OP *op) final {                                            \
    LOG(DEBUG) << "BINOP" << op->a << op->b << " " << cc;                      \
    this->Visit(op->a);                                                        \
    this->Visit(op->b);                                                        \
    throw_not_implement_error();                                               \
  }

  void Visit_(const Add *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    left = u;
    int b = cc;
    this->Visit(op->b);
    right = u;
    int c = cc;
    i = false;
    // only for computing tensor size of input/output
    if (is_io) {
      u = left + right;
      // otherwise
    } else {
      cc += 1;
      if ((a != b) && b != c) {
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        if (datatype.at(b).at(0) == 'f')
          mlirstring << "    %" << cc << " = addf %" << b << " , %" << c
                     << " : " << datatype.at(b) << "\n";
        else
          mlirstring << "    %" << cc << " = addi %" << b << " , %" << c
                     << " : " << datatype.at(b) << "\n";
      } else {
        throw_not_implement_error();
      }
    }
  }

  void Visit_(const Sub *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    datatype.insert(pair<int, string>(cc, datatype.at(b)));
    if ((a != b) && b != c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = subf %" << b << " , %" << c << " : "
                   << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc << " = subi %" << b << " , %" << c << " : "
                   << datatype.at(b) << "\n";
    } else {
      throw_not_implement_error();
    }
  }

  void Visit_(const Mul *op) final {
    LOG(DEBUG) << "mul" << op->a << op->b << " " << cc;
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    left = u;
    this->Visit(op->b);
    int c = cc;
    right = u;
    if (is_io) {
      u = left * right;
    } else {
      i = false;
      cc += 1;
      datatype.insert(pair<int, string>(cc, datatype.at(b)));
      if ((a != b) && b != c) {
        if (datatype.at(b).at(0) == 'f')
          mlirstring << "    %" << cc << " = mulf %" << b << " , %" << c
                     << " : " << datatype.at(b) << "\n";
        else
          mlirstring << "    %" << cc << " = muli %" << b << " , %" << c
                     << " : " << datatype.at(b) << "\n";
      } else {
        throw_not_implement_error();
      }
    }
  }
  void Visit_(const IntImm *op) final {
    u = op->value;
    if (i && !is_io) {
      cc += 1;
      // -> std.constant
      datatype.insert(
          pair<int, string>(cc, "i" + std::to_string(op->type.bits())));
      mlirstring << "    %" << cc << " = constant " << op->value << " : i"
                 << op->type.bits() << "\n";
      stringstream ss;
      ss << "i" << op->type.bits() << "\n";
      string data_str = ss.str();
      datatype.insert(pair<int, string>(cc, data_str));
    }
  }
  void Visit_(const FloatImm *op) final {
    if (i) {
      cc += 1;
      // -> std.constant
      datatype.insert(
          pair<int, string>(cc, "f" + std::to_string(op->type.bits())));
      mlirstring << "    %" << cc << " = constant " << std::scientific
                 << op->value << " : f" << op->type.bits() << "\n";
      stringstream ss;
      ss << "f" << op->type.bits() << "\n";
      string data_str = ss.str();
      datatype.insert(pair<int, string>(cc, data_str));
    }
  }

  void Visit_(const Div *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    datatype.insert(pair<int, string>(cc, datatype.at(b)));
    if ((a != b) && b != c)
      mlirstring << "    %" << cc << " = divf %" << b << " , %" << c << " : "
                 << datatype.at(b) << "\n";
    else {
      throw_not_implement_error();
    }
  }

  void Visit_(const Min *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 2;
    // both not loop var
    if ((a != b) && b != c) {
      datatype.insert(pair<int, string>(cc, datatype.at(b)));
      datatype.insert(pair<int, string>(cc - 1, "i1"));
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc - 1 << " = cmpf \"olt\" , %" << b << " , %"
                   << c << " : " << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc - 1 << " = cmpi \"slt\" , %" << b << " , %"
                   << c << " : " << datatype.at(b) << "\n";
      mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << b
                 << " , %" << c << " : " << datatype.at(b) << "\n";
      // op->b is loop var
    } else if (a != b && b == c) {
      if (datatype.at(b).at(0) == 'f') {
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        datatype.insert(pair<int, string>(cc - 1, "i1"));
        mlirstring << "    %" << cc - 1 << " = cmpf \"olt\" , %" << b << " , %"
                   << op->b << " : " << datatype.at(b) << "\n";
        mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << b
                   << " , %" << op->b << " : " << datatype.at(b) << "\n";
      } else {
        mlirstring << "    %" << cc - 1 << " = index_cast %" << op->b
                   << " : index to " << datatype.at(b) << "\n";
        datatype.insert(pair<int, string>(cc - 1, datatype.at(b)));
        datatype.insert(pair<int, string>(cc, "i1"));
        mlirstring << "    %" << cc << " = cmpi \"slt\" , %" << b << " , %"
                   << cc - 1 << " : " << datatype.at(b) << "\n";
        cc += 1;
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << b
                   << " , %" << op->b << " : " << datatype.at(b) << "\n";
      }
      // op->a is loop var
    } else if (a == b && b != c) {
      if (datatype.at(c).at(0) == 'f') {
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        datatype.insert(pair<int, string>(cc - 1, "i1"));
        mlirstring << "    %" << cc - 1 << " = cmpf \"olt\" , %" << c << " , %"
                   << op->a << " : " << datatype.at(c) << "\n";
        mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << c
                   << " , %" << op->a << " : " << datatype.at(c) << "\n";
      } else {
        mlirstring << "    %" << cc - 1 << " = index_cast %" << op->a
                   << " : index to " << datatype.at(c) << "\n";
        datatype.insert(pair<int, string>(cc - 1, datatype.at(c)));
        datatype.insert(pair<int, string>(cc, "i1"));
        mlirstring << "    %" << cc << " = cmpi \"slt\" , %" << c << " , %"
                   << cc - 1 << " : " << datatype.at(c) << "\n";
        cc += 1;
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << c
                   << " , %" << op->a << " : " << datatype.at(c) << "\n";
      }
      // both loop var
    } else {
      mlirstring << "    %" << cc - 1 << " = cmpi \"slt\" , %" << op->a
                 << " , %" << op->b << " : index\n";
      mlirstring << "    %" << cc << " = select %" << op->a << " , %" << op->b
                 << " , %" << c << " : index\n";
    }
  }

  void Visit_(const Max *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 2;
    // both not loop var
    if ((a != b) && b != c) {
      datatype.insert(pair<int, string>(cc, datatype.at(b)));
      datatype.insert(pair<int, string>(cc - 1, "i1"));
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc - 1 << " = cmpf \"ogt\" , %" << b << " , %"
                   << c << " : " << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc - 1 << " = cmpi \"sgt\" , %" << b << " , %"
                   << c << " : " << datatype.at(b) << "\n";
      mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << b
                 << " , %" << c << " : " << datatype.at(b) << "\n";
      // op->b is loop var
    } else if (a != b && b == c) {
      if (datatype.at(b).at(0) == 'f') {
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        datatype.insert(pair<int, string>(cc - 1, "i1"));
        mlirstring << "    %" << cc - 1 << " = cmpf \"ogt\" , %" << b << " , %"
                   << op->b << " : " << datatype.at(b) << "\n";
        mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << b
                   << " , %" << op->b << " : " << datatype.at(b) << "\n";
      } else {
        mlirstring << "    %" << cc - 1 << " = index_cast %" << op->b
                   << " : index to " << datatype.at(b) << "\n";
        datatype.insert(pair<int, string>(cc - 1, datatype.at(b)));
        datatype.insert(pair<int, string>(cc, "i1"));
        mlirstring << "    %" << cc << " = cmpi \"sgt\" , %" << b << " , %"
                   << cc - 1 << " : " << datatype.at(b) << "\n";
        cc += 1;
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << b
                   << " , %" << op->b << " : " << datatype.at(b) << "\n";
      }
      // op->a is loop var
    } else if (a == b && b != c) {
      if (datatype.at(c).at(0) == 'f') {
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        datatype.insert(pair<int, string>(cc - 1, "i1"));
        mlirstring << "    %" << cc - 1 << " = cmpf \"ogt\" , %" << c << " , %"
                   << op->a << " : " << datatype.at(c) << "\n";
        mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << c
                   << " , %" << op->a << " : " << datatype.at(c) << "\n";
      } else {
        mlirstring << "    %" << cc - 1 << " = index_cast %" << op->a
                   << " : index to " << datatype.at(c) << "\n";
        datatype.insert(pair<int, string>(cc - 1, datatype.at(c)));
        datatype.insert(pair<int, string>(cc, "i1"));
        mlirstring << "    %" << cc << " = cmpi \"sgt\" , %" << c << " , %"
                   << cc - 1 << " : " << datatype.at(c) << "\n";
        cc += 1;
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        mlirstring << "    %" << cc << " = select %" << cc - 1 << " , %" << c
                   << " , %" << op->a << " : " << datatype.at(c) << "\n";
      }
      // both loop var
    } else {
      mlirstring << "    %" << cc - 1 << " = cmpi \"sgt\" , %" << op->a
                 << " , %" << op->b << " : index\n";
      mlirstring << "    %" << cc << " = select %" << op->a << " , %" << op->b
                 << " , %" << c << " : index\n";
    }
  }

  void Visit_(const EQ *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    // both not loop var
    if ((a != b) && b != c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"oeq\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc << " = cmpi \"eq\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      // op->b is loop var
    } else if (a != b && b == c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"oeq\" , %" << b << " , %"
                   << op->b << " : " << datatype.at(b) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->b
                   << " : index to " << datatype.at(b) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"eq\" , %" << b << " , %"
                   << cc - 1 << " : " << datatype.at(b) << "\n";
      }
      // op->a is loop var
    } else if (a == b && b != c) {
      if (datatype.at(a).at(c) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"oeq\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->a
                   << " : index to " << datatype.at(c) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"eq\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      }
      // both loop var
    } else {
      mlirstring << "    %" << cc << " = cmpi \"eq\" , %" << op->a << " , %"
                 << op->b << " : index\n";
    }
    datatype.insert(pair<int, string>(cc, "i1"));
  }

  void Visit_(const NE *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    datatype.insert(pair<int, string>(cc, datatype.at(b)));
    // both not loop var
    if ((a != b) && b != c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"one\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc << " = cmpi \"ne\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      // op->b is loop var
    } else if (a != b && b == c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"one\" , %" << b << " , %"
                   << op->b << " : " << datatype.at(b) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->b
                   << " : index to " << datatype.at(b) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"ne\" , %" << b << " , %"
                   << cc - 1 << " : " << datatype.at(b) << "\n";
      }
      // op->a is loop var
    } else if (a == b && b != c) {
      if (datatype.at(a).at(c) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"one\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->a
                   << " : index to " << datatype.at(c) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"ne\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      }
      // both loop var
    } else {
      mlirstring << "    %" << cc << " = cmpi \"ne\" , %" << op->a << " , %"
                 << op->b << " : index\n";
    }
    datatype.insert(pair<int, string>(cc, "i1"));
  }

  void Visit_(const LT *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    // both not loop var
    if ((a != b) && b != c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"olt\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc << " = cmpi \"slt\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      // op->b is loop var
    } else if (a != b && b == c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"olt\" , %" << b << " , %"
                   << op->b << " : " << datatype.at(b) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->b
                   << " : index to " << datatype.at(b) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"slt\" , %" << b << " , %"
                   << cc - 1 << " : " << datatype.at(b) << "\n";
      }
      // op->a is loop var
    } else if (a == b && b != c) {
      if (datatype.at(a).at(c) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"olt\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->a
                   << " : index to " << datatype.at(c) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"slt\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      }
      // both loop var
    } else {
      mlirstring << "    %" << cc << " = cmpi \"slt\" , %" << op->a << " , %"
                 << op->b << " : index\n";
    }
    datatype.insert(pair<int, string>(cc, "i1"));
  }

  void Visit_(const LE *op) final {
    // LOG(DEBUG)<<"LT"<<op->a<<op->b<<" "<<cc;
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    // both not loop var
    if ((a != b) && b != c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"ole\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc << " = cmpi \"sle\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      // op->b is loop var
    } else if (a != b && b == c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"ole\" , %" << b << " , %"
                   << op->b << " : " << datatype.at(b) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->b
                   << " : index to " << datatype.at(b) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"sle\" , %" << b << " , %"
                   << cc - 1 << " : " << datatype.at(b) << "\n";
      }
      // op->a is loop var
    } else if (a == b && b != c) {
      if (datatype.at(a).at(c) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"ole\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->a
                   << " : index to " << datatype.at(c) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"sle\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      }
      // both loop var
    } else {
      mlirstring << "    %" << cc << " = cmpi \"sle\" , %" << op->a << " , %"
                 << op->b << " : index\n";
    }
    datatype.insert(pair<int, string>(cc, "i1"));
  }

  void Visit_(const GT *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    // both not loop var
    if ((a != b) && b != c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"ogt\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc << " = cmpi \"sgt\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      // op->b is loop var
    } else if (a != b && b == c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"ogt\" , %" << b << " , %"
                   << op->b << " : " << datatype.at(b) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->b
                   << " : index to " << datatype.at(b) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"sgt\" , %" << b << " , %"
                   << cc - 1 << " : " << datatype.at(b) << "\n";
      }
      // op->a is loop var
    } else if (a == b && b != c) {
      if (datatype.at(a).at(c) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"ogt\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->a
                   << " : index to " << datatype.at(c) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"sgt\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      }
      // both loop var
    } else {
      mlirstring << "    %" << cc << " = cmpi \"sgt\" , %" << op->a << " , %"
                 << op->b << " : index\n";
    }
    datatype.insert(pair<int, string>(cc, "i1"));
  }

  void Visit_(const GE *op) final {
    i = true;
    int a = cc;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    // both not loop var
    if ((a != b) && b != c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"oge\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      else
        mlirstring << "    %" << cc << " = cmpi \"sge\" , %" << b << " , %" << c
                   << " : " << datatype.at(b) << "\n";
      // op->b is loop var
    } else if (a != b && b == c) {
      if (datatype.at(b).at(0) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"oge\" , %" << b << " , %"
                   << op->b << " : " << datatype.at(b) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->b
                   << " : index to " << datatype.at(b) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(b)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"sge\" , %" << b << " , %"
                   << cc - 1 << " : " << datatype.at(b) << "\n";
      }
      // op->a is loop var
    } else if (a == b && b != c) {
      if (datatype.at(a).at(c) == 'f')
        mlirstring << "    %" << cc << " = cmpf \"oge\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      else {
        mlirstring << "    %" << cc << " = index_cast %" << op->a
                   << " : index to " << datatype.at(c) << "\n";
        datatype.insert(pair<int, string>(cc, datatype.at(c)));
        cc += 1;
        mlirstring << "    %" << cc << " = cmpi \"sge\" , %" << op->a << " , %"
                   << c << " : " << datatype.at(c) << "\n";
      }
      // both loop var
    } else {
      mlirstring << "    %" << cc << " = cmpi \"sge\" , %" << op->a << " , %"
                 << op->b << " : index\n";
    }
    datatype.insert(pair<int, string>(cc, "i1"));
  }

  void Visit_(const And *op) {
    i = true;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    datatype.insert(pair<int, string>(cc, datatype.at(b)));
    mlirstring << "    %" << cc << " =  and %" << b << " , %" << c << " : "
               << datatype.at(b) << "\n";
  }

  void Visit_(const Or *op) {
    i = true;
    this->Visit(op->a);
    int b = cc;
    this->Visit(op->b);
    int c = cc;
    i = false;
    cc += 1;
    datatype.insert(pair<int, string>(cc, datatype.at(b)));
    mlirstring << "    %" << cc << " =  or %" << b << " , %" << c << " : "
               << datatype.at(b) << "\n";
  }

  void Visit_(const Cast *op) {
    this->Visit(op->value);
    cc += 1;
    if (op->type.is_float() && op->type.bits() == 32) {
      datatype.insert(pair<int, string>(cc, "f32"));
      if (datatype.at(cc - 1) == "f16")
        mlirstring << "    %" << cc << " = fpext %" << cc - 1
                   << " : f16 to f32\n";
      else if (datatype.at(cc - 1) == "f64")
        mlirstring << "    %" << cc << " = fptrunc %" << cc - 1
                   << " : f64 to f32\n";
      else if (datatype.at(cc - 1).at(0) == 'i')
        mlirstring << "    %" << cc << " = fptosi %" << cc - 1 << " : "
                   << datatype.at(cc - 1) << " to f32\n";
    } else if (op->type.is_float() && op->type.bits() == 16) {
      datatype.insert(pair<int, string>(cc, "f16"));
      if (datatype.at(cc - 1) == "f32")
        mlirstring << "    %" << cc << " = fptrunc %" << cc - 1
                   << " : f32 to f16\n";
      else if (datatype.at(cc - 1) == "f64")
        mlirstring << "    %" << cc << " = fptrunc %" << cc - 1
                   << " : f16 to f16\n";
      else if (datatype.at(cc - 1).at(0) == 'i')
        mlirstring << "    %" << cc << " = fptosi %" << cc - 1 << " : "
                   << datatype.at(cc - 1) << " to f16\n";
    } else if (op->type.is_float() && op->type.bits() == 64) {
      datatype.insert(pair<int, string>(cc, "f64"));
      if (datatype.at(cc - 1) == "f32")
        mlirstring << "    %" << cc << " = fpext %" << cc - 1
                   << " : f32 to f64\n";
      else if (datatype.at(cc - 1) == "f16")
        mlirstring << "    %" << cc << " = fpext %" << cc - 1
                   << " : f16 to f64\n";
      else if (datatype.at(cc - 1).at(0) == 'i')
        mlirstring << "    %" << cc << " = fptosi %" << cc - 1 << " : "
                   << datatype.at(cc - 1) << " to f64\n";
    } else if (op->type.is_int() && op->type.bits() == 64) {
      datatype.insert(pair<int, string>(cc, "i64"));
      if (datatype.at(cc - 1) == "i32")
        mlirstring << "    %" << cc << " = zexti %" << cc - 1
                   << " : i32 to i64\n";
      else if (datatype.at(cc - 1) == "i16")
        mlirstring << "    %" << cc << " = zexti %" << cc - 1
                   << " : i16 to i64\n";
      else if (datatype.at(cc - 1) == "i8")
        mlirstring << "    %" << cc << " = zexti %" << cc - 1
                   << " : i8 to i64\n";
      else if (datatype.at(cc - 1).at(0) == 'f')
        mlirstring << "    %" << cc << " = sitofp %" << cc - 1 << " : "
                   << datatype.at(cc - 1) << " to i64\n";
    } else if (op->type.is_int() && op->type.bits() == 32) {
      datatype.insert(pair<int, string>(cc, "i32"));
      if (datatype.at(cc - 1) == "i64")
        mlirstring << "    %" << cc << " = trunci %" << cc - 1
                   << " : i64 to i32\n";
      else if (datatype.at(cc - 1) == "i16")
        mlirstring << "    %" << cc << " = zexti %" << cc - 1
                   << " : i16 to i32\n";
      else if (datatype.at(cc - 1) == "i8")
        mlirstring << "    %" << cc << " = zexti %" << cc - 1
                   << " : i8 to i32\n";
      else if (datatype.at(cc - 1).at(0) == 'f')
        mlirstring << "    %" << cc << " = sitofp %" << cc - 1 << " : "
                   << datatype.at(cc - 1) << " to i32\n";
    } else if (op->type.is_int() && op->type.bits() == 16) {
      datatype.insert(pair<int, string>(cc, "i16"));
      if (datatype.at(cc - 1) == "i64")
        mlirstring << "    %" << cc << " = trunci %" << cc - 1
                   << " : i64 to i16\n";
      else if (datatype.at(cc - 1) == "i32")
        mlirstring << "    %" << cc << " = trunci %" << cc - 1
                   << " : i32 to i16\n";
      else if (datatype.at(cc - 1) == "i8")
        mlirstring << "    %" << cc << " = zexti %" << cc - 1
                   << " : i8 to i16\n";
      else if (datatype.at(cc - 1).at(0) == 'f')
        mlirstring << "    %" << cc << " = sitofp %" << cc - 1 << " : "
                   << datatype.at(cc - 1) << " to i16\n";
    } else if (op->type.is_int() && op->type.bits() == 8) {
      datatype.insert(pair<int, string>(cc, "i8"));
      if (datatype.at(cc - 1) == "i64")
        mlirstring << "    %" << cc << " = trunci %" << cc - 1
                   << " : i64 to i8\n";
      else if (datatype.at(cc - 1) == "i32")
        mlirstring << "    %" << cc << " = trunci %" << cc - 1
                   << " : i32 to i8\n";
      else if (datatype.at(cc - 1) == "i16")
        mlirstring << "    %" << cc << " = trunci %" << cc - 1
                   << " : i16 to i8\n";
      else if (datatype.at(cc - 1).at(0) == 'f')
        mlirstring << "    %" << cc << " = sitofp %" << cc - 1 << " : "
                   << datatype.at(cc - 1) << " to i8\n";
    }
  }

  void Visit_(const Select *op) {
    this->Visit(op->condition);
    int b = cc;
    i = true;
    this->Visit(op->true_value);
    int c = cc;
    this->Visit(op->false_value);
    i = false;
    cc += 1;
    datatype.insert(pair<int, string>(cc, datatype.at(c)));
    mlirstring << "    %" << cc << " = select %" << b << " , %" << c << " , %"
               << cc - 1 << " : " << datatype.at(c) << "\n";
  }

  void Visit_(const Provide *op) {
    if (memref.find(op->func->func_name()) != memref.end()) {
      size_t lasttime = memref.at(op->func->func_name()).rfind("x");
      io_datatype =
          memref.at(op->func->func_name())
              .substr(lasttime + 1,
                      memref.at(op->func->func_name()).size() - lasttime - 3);
    }
    i = true;
    this->Visit(op->value);
    i = false;
    stringstream ss;
    ss << op->args;
    string args_str = ss.str();
    string args_str_print = "";
    for (size_t i = 0; i < args_str.size(); i++) {
      if (std::isalpha(args_str[i]) && args_str[i - 1] != '_' &&
          !std::isalpha(args_str[i - 1]) && !std::isdigit(args_str[i - 1]))
        args_str_print = args_str_print + "%";
      args_str_print = args_str_print + args_str[i];
    }
    if (memref.find(op->func->func_name()) == memref.end())
      AllocInpOut(op->func->func_name(), op->args, this);
    mlirstring << "    affine.store %" << cc << " , %" << op->func->func_name()
               << args_str_print << ": " << memref.at(op->func->func_name());
  }

  void Visit_(const Realize *op) {
    // todo: determine if var exit
    mlirstring << "%" << op->func->func_name() << " = alloc() : ";
    stringstream mem;
    mem << "memref<";
    for (size_t i = 0; i < op->bounds.size(); i++) {
      this->Visit(op->bounds[i]->min);
      this->Visit(op->bounds[i]->extent);
      mem << op->bounds[i]->extent;
      mem << "x";
    }
    if (op->type.is_float())
      mem << "f";
    else if (op->type.is_int())
      mem << "i";
    else if (op->type.is_bool())
      mem << "i";
    mem << op->type.bits() << ">\n";
    string s = mem.str(), f_name = op->func->func_name();
    memref.insert(pair<string, string>(f_name, s));
    mlirstring << s;
    this->Visit(op->body);
    this->Visit(op->condition);
    mlirstring << "dealloc %" << op->func->func_name() << " : " << s;
  }

  DEFINE_BINOP_VISIT_(Mod)
  DEFINE_BINOP_VISIT_(FloorDiv)
  DEFINE_BINOP_VISIT_(FloorMod)

  void Visit_(const Allocate *op) final {
    LOG(DEBUG) << "allocate\n";
    IRVisitor *v = this;
    for (size_t i = 0; i < op->extents.size(); i++) {
      v->Visit(op->extents[i]);
    }
    v->Visit(op->body);
    v->Visit(op->condition);
    if (op->new_expr.defined()) {
      v->Visit(op->new_expr);
    }
    throw_not_implement_error();
  }

  void Visit_(const Load *op) final {
    LOG(DEBUG) << "load\n";
    this->Visit(op->index);
    this->Visit(op->predicate);
    throw_not_implement_error();
  }

  void Visit_(const Store *op) final {
    LOG(DEBUG) << "store\n";
    this->Visit(op->value);
    this->Visit(op->index);
    this->Visit(op->predicate);
    throw_not_implement_error();
  }

  void Visit_(const Let *op) final {
    LOG(DEBUG) << "let\n";
    this->Visit(op->value);
    this->Visit(op->body);
    throw_not_implement_error();
  }

  void Visit_(const LetStmt *op) final {
    LOG(DEBUG) << "letstmt\n";
    this->Visit(op->value);
    this->Visit(op->body);
    throw_not_implement_error();
  }

  void Visit_(const Reduce *op) {
    LOG(DEBUG) << "Reduce";
    VisitRDom(op->axis, this);
    VisitArray(op->source, this);
    this->Visit(op->condition);
    throw_not_implement_error();
  }

  void Visit_(const Not *op) {
    LOG(DEBUG) << "not";
    this->Visit(op->a);
    throw_not_implement_error();
  }

  void Visit_(const Ramp *op) {
    LOG(DEBUG) << "ramp";
    this->Visit(op->base);
    this->Visit(op->stride);
    throw_not_implement_error();
  }

  void Visit_(const Shuffle *op) {
    LOG(DEBUG) << "shuffle";
    for (const auto &elem : op->indices)
      this->Visit(elem);
    for (const auto &elem : op->vectors)
      this->Visit(elem);
    throw_not_implement_error();
  }

  void Visit_(const Broadcast *op) {
    LOG(DEBUG) << "broadcast";
    this->Visit(op->value);
    throw_not_implement_error();
  }

  void Visit_(const AssertStmt *op) {
    LOG(DEBUG) << "assertstmt";
    this->Visit(op->condition);
    this->Visit(op->message);
    this->Visit(op->body);
    throw_not_implement_error();
  }

  void Visit_(const Prefetch *op) {
    LOG(DEBUG) << "prefetch";
    for (size_t i = 0; i < op->bounds.size(); i++) {
      this->Visit(op->bounds[i]->min);
      this->Visit(op->bounds[i]->extent);
    }
    throw_not_implement_error();
  }

  void Visit_(const Evaluate *op) {
    LOG(DEBUG) << "evaluate";
    this->Visit(op->value);
    throw_not_implement_error();
  }

  void Visit_(const Block *op) {
    this->Visit(op->first);
    this->Visit(op->rest);
  }

  void Visit_(const ProducerConsumer *op) { this->Visit(op->body); }

  void Visit_(const AttrStmt *op) final {
    this->Visit(op->value);
    this->Visit(op->body);
  }
};
/*Halide to MLIR
 */
Stmt ToMLIR(Stmt stmt) {
  LOG(DEBUG) << "To MLIR";
  mkdir("/tmp/", 777);
  outfile.open("/tmp/halide_2_affine.mlir");
  outfile << "  func @main() {\n";
  HalideIRVisitor hv;
  hv.Visit(stmt);
  outfile << mlirstring.str();
  outfile << "  return\n}\n";
  outfile.close();
  return stmt;
}
} // namespace ir
} // namespace akg
