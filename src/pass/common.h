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

#ifndef PASS_COMMON_H_
#define PASS_COMMON_H_
#include <tvm/ir_visitor.h>
#include <runtime/thread_storage_scope.h>
#include <tvm.h>
#include <string>
#include <list>
namespace akg {
namespace ir {
// pipe id
enum {
  PIPE_S = 1,     // Scalar Pipe
  PIPE_V = 2,     // Vector Pipe, including{VectorOP write UB,  L0C->UB write}
  PIPE_M = 3,     // Matrix Pipe, including{}
  PIPE_MTE1 = 4,  // L1->L0{A,B}
  PIPE_MTE2 = 5,  // OUT ->{L1, L0{A,B}, UB}
  PIPE_MTE3 = 6,  // UB ->{OUT,L1}
  PIPE_ALL = 7,
};

// for alias
struct MemInfo {
  const Variable *base;
  Expr offset;
  Expr extent;
  Type type;
  Expr repeatTime;
  Expr repeatStride;
  Expr blockNumber;
  Expr blockStride;
  Expr blockSize;
};

int GetIntrinPipe(std::string insn_name);

// base class for df analyzer
class DFAnalyzer {
 public:
  virtual ~DFAnalyzer() {}
  virtual void Plan(Stmt stmt) = 0;
  virtual bool DepForward(const AttrStmt *a, const AttrStmt *b) = 0;
  virtual bool DepBackward(const AttrStmt *a, const AttrStmt *b, const For *loop) = 0;
};

std::shared_ptr<DFAnalyzer> BuildDfAnalyzer(Stmt stmt, bool prebuild = false);

Stmt ConvertSingleCoprocForm(Stmt stmt);

using ktvm::runtime::StorageRank;
using ktvm::runtime::StorageScope;

class LivenessAnalyzer : public IRVisitor {
 public:
  struct StmtEntry {
    const Node *stmt;
    // variables generated
    std::vector<const Variable *> gen;
    // variables killed
    std::vector<const Variable *> kill;
  };

  struct AllocEntry {
    // scope of allocate
    StorageScope scope;
    // alloc of buffer
    const Allocate *alloc{nullptr};
    // scope level
    int level{0};
    //  touched stmt
    std::vector<int> touched;
  };

  void Analyze(Stmt stmt);
  void Visit_(const AttrStmt *op) final;
  void Visit_(const Allocate *op) final;
  void Visit_(const For *op) final;
  void Visit_(const IfThenElse *op) final;
  void Visit_(const Store *op) final;
  void Visit_(const Evaluate *op) final;
  void Visit_(const Load *op) final;
  void Visit_(const Variable *buf) final;

  std::vector<StmtEntry> liveness_;
  std::unordered_map<const Variable *, AllocEntry> alloc_;
  std::list<const Variable *> alloc_keys_;

 private:
  void TouchBuffer(const Variable *buf);
  void PushScope(const Node *stmt);
  void PopScope();

  struct ScopeTouch {
    int entry;
    std::unordered_set<const Variable *> touched;
  };
  std::vector<ScopeTouch> scope_touch_;
  bool in_insn_partition_{false};
};
}  // namespace ir
}  // namespace akg

#endif  // PASS_COMMON_H_
