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
 * 2019.12.30 - Add file codegen_cce.h.
 *              Utility to generate cce code.
 */

#ifndef CODEGEN_CODEGEN_CCE_H_
#define CODEGEN_CODEGEN_CCE_H_
#include <codegen/codegen_c.h>
#include <tvm/codegen.h>
#include <tvm/ir_visitor.h>
#include <tvm/packed_func_ext.h>
#include <map>
#include <set>
#include <string>
#include <vector>


namespace air {
namespace codegen {
using namespace air;
using namespace air::ir;

class CodeGenCCE final : public CodeGenC {
 public:
  CodeGenCCE();
  ~CodeGenCCE() override = default;
  void Initialize(bool output_ssa);
  void AddFunctionCore(LoweredFunc f);

  // override behavior
  void VisitStmt_(const Allocate *op) override;
  void VisitStmt_(const Evaluate *op) override;
  void VisitStmt_(const AttrStmt* op) override;

  void VisitExpr_(const StringImm *op, std::ostream &os) final;  // NOLINT(*)
  void VisitExpr_(const Call *op, std::ostream &os) final;       // NOLINT(*)
  void VisitExpr_(const Max *op, std::ostream &os) final;        // NOLINT(*)
  void VisitExpr_(const FloatImm *op, std::ostream &os) final;

  // tag for distinguish  aicpu and aicore
  bool tag{false};

 protected:
  inline void PrintBufferHeader(std::ostream& os) override {
    os << "__ubuf__";
  }

 private:
  // iteration name
  std::string iterName_;
  // Global cce vars
  std::vector<std::string> gvar_cce_;
  // needed to free malloc buf
  std::vector<std::string> free_ids_;
  void PrintType(Type t, std::ostream &os) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;   // NOLINT(*)
  void BindThreadIndex(const IterVar &iv);  // NOLINT(*)
  void PrintCCEIntrinArgsType(const Expr &e, std::ostream &os);
  void PrintPureCall(const Call *op, std::ostream &os);
  void PrintExternCall(const Call *op, std::ostream &os);

  // print the reg mov func
  void PrintRegmov(const Call *op, std::ostream &os);
  void PrintArgmaxCast(const Call *op, std::ostream &os);
  void PrintPointerCast(const Call *op, std::ostream &os);
  void PrintBitMove(const Call *op, std::ostream &os, bool left);
  void PrintSetAtomicAdd(const Call *op, std::ostream &os, bool open);
};
}  // namespace codegen
}  // namespace air

#endif  // CODEGEN_CODEGEN_CCE_H_
