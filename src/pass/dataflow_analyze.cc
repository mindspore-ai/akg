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

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <ir_pass.h>
#include <pass/storage_access.h>
#include "pass/common.h"
#include "pass/overflow_check.h"
#include "pass/dependency_graph.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
struct BranchTag {
  std::shared_ptr<std::vector<int>> data;
  // use COW for performance, because most of insn have same branch tag data
  // if no branch meet
  void CopyOnWrite() {
    if (!data.unique()) data = std::make_shared<std::vector<int>>(*data.get());
  }
};

// Touch Entry
struct TouchEntry {
  uint32_t index;
  std::vector<MemInfo> def;
  std::vector<MemInfo> use;
  std::set<const AttrStmt *> RAW;
  std::set<const AttrStmt *> WAR;
  std::set<const AttrStmt *> WAW;
  bool mask_insn{false};
  const AttrStmt *nest_attr{nullptr};
  BranchTag branch_tag;
};

// Analysis data flow info for each scope
class DFVisitor : public IRVisitor {
  Expr const_zero = make_const(ktvm::Int(32), 0);
  Expr const_one = make_const(ktvm::Int(32), 1);
  Var const_reg = Var("register");

  // const meminfo for vcmp
  VarExpr cmpmask = Variable::make(Int(32), "CMPMASK");
  MemInfo cmpmaskMemInfo = {cmpmask.get(),
                            const_zero,
                            const_one,
                            cmpmask.type(),
                            const_one,                           // repeatTime
                            const_one *cmpmask.type().bytes(),   // repeatStride
                            const_one,                           // blockNumber
                            const_one *cmpmask.type().bytes(),   // blockStride
                            const_one *cmpmask.type().bytes()};  // blockSize

  // const meminfo for setmask
  VarExpr vmask = Variable::make(Int(32), "VMASK");
  MemInfo vmaskkMemInfo = {vmask.get(), const_zero, const_one, vmask.type()};

  struct StorageRange {
    int64_t addr;
    int64_t size;
  };

 public:
  void Plan(const Stmt &stmt) {
    branch_tag_.data = std::make_shared<std::vector<int>>(1, 1);
    IRVisitor::Visit(stmt);
  }

  void Visit_(const Load *op) final {
    if (nullptr == op) {
      return;
    }

    if (in_scope_) {
      MemInfo m = {op->buffer_var.get(), op->index, 1, op->type};
      auto &touch = touched_[curr_attr_];
      if (touch.nest_attr != nullptr) {
        touched_[touch.nest_attr].use.push_back(m);
      } else {
        touch.use.push_back(m);
      }
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) final {
    if (nullptr == op) {
      return;
    }

    if (in_scope_) {
      MemInfo m = {op->buffer_var.get(), op->index, 1, op->value.type()};
      touched_[curr_attr_].def.push_back(m);
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const Evaluate *op) final {
    if (nullptr == op) {
      return;
    }

    const auto insn = op->value.as<Call>();
    if (in_scope_ && (insn != nullptr)) {
      const auto v = curr_attr_->value.as<IntImm>();
      if (v != nullptr) {
        int64_t pip = v->value;

        // make all vector insns alias with mask mem
        if (pip % 8 == 2) {
          if ((insn->name == "get_cmpmask") || (insn->name == "vsel")) {
            touched_[curr_attr_].use.emplace_back(cmpmaskMemInfo);
          } else if ((insn->name == "set_cmpmask") || (insn->name.find("vcmp") != std::string::npos)) {
            touched_[curr_attr_].def.emplace_back(cmpmaskMemInfo);
          } else if (insn->name == "set_vector_mask") {
            touched_[curr_attr_].def.emplace_back(vmaskkMemInfo);
          } else {
            touched_[curr_attr_].use.emplace_back(vmaskkMemInfo);
          }
        }
      }
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const Call *op) final {
    if (nullptr == op) {
      return;
    }

    if (!in_scope_) {
      IRVisitor::Visit_(op);
      return;
    }

    if (op->is_intrinsic(ktvm::ir::intrinsic::tvm_access_ptr)) {
      const auto buffer = op->args[1].as<Variable>();
      if (buffer == nullptr) {
        return;
      }

      const auto v = op->args[4].as<IntImm>();
      if (v == nullptr) {
        return;
      }

      int64_t rw = v->value;
      Expr offset = op->args[2];
      Expr extent = op->args[3];

      // MTE2/MTE3 copy data from/to global memory. the actual touched memory is align to 32B.
      // So we need to align the extent to follow this rule.
      if (curr_attr_ && curr_attr_->value.as<IntImm>()) {
        int cur_pip = static_cast<int>(curr_attr_->value.as<IntImm>()->value);
        if ((cur_pip == PIPE_MTE2) || (cur_pip == PIPE_MTE3)) {
          extent = AlignExtent(extent, op->args[0].type());
        }

        // align extent of VECTOR pipe to whole block memory size(32B). This is because we found
        // Vector and Scalar are conflicted if they access same block memory, even with vector mask.
        // for example:
        //   Vector(mask 0x3ffffff): tvm_access_ptr(int(8), res_local_UB, 0, 26)
        //   Scalar: for (scalar_idx, 0, 6) { (int8)res_local_UB[(scalar_idx + 26)] = xxx; }
        if (cur_pip == PIPE_V) {
          int64_t align = 32 / op->args[0].type().bytes();
          const auto ext_imm = extent.as<IntImm>();
          if ((ext_imm != nullptr) && (ext_imm->value % align != 0)) {
            CHECK(storage_range_.find(buffer) != storage_range_.end());
            StorageRange &range = storage_range_[buffer];
            int64_t align_ext = ext_imm->value + align - ext_imm->value % align;
            ktvm::arith::Analyzer analyzer_;
            Expr check = Simplify_cce(offset <= make_const(Int(32), range.size - align_ext), range_);
            if (analyzer_.CanProve(check)) {
              extent = make_const(Int(32), align_ext);
            }
          }
        }
      }

      Expr repeatTime, repeatStride, blockNum, blockStride, blockSize;
      if (op->args.size() == 10U) {
        repeatTime = op->args[5];
        repeatStride = op->args[6];
        blockNum = op->args[7];
        blockStride = op->args[8];
        blockSize = op->args[9];
      }

      MemInfo m = {buffer,       offset,   extent,      op->args[0].type(), repeatTime,
                   repeatStride, blockNum, blockStride, blockSize};
      if (rw == 1) {
        touched_[curr_attr_].use.push_back(m);
      } else {
        touched_[curr_attr_].def.push_back(m);
      }
    }

    for (Expr arg : op->args) {
      const auto imm = arg.as<StringImm>();
      if ((imm != nullptr) && (register_.count(imm->value) > 0)) {
        MemInfo m = {const_reg.get(), const_zero, const_zero, const_reg.type()};
        touched_[curr_attr_].def.push_back(m);

        break;
      }
    }

    if (op->name == "reg_mov") {
      RegMoveVisit(op);
      return;
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) final {
    if (nullptr == op) {
      return;
    }

    range_.Set(Var(op->loop_var), Range::make_by_min_extent(op->min, op->extent));
    IRVisitor::Visit_(op);
  }

  void Visit_(const IfThenElse *op) final {
    int delta = ((branch_tag_.data->back() > 0) ? 1 : -1);
    branch_tag_.CopyOnWrite();
    branch_tag_.data->back() += delta;
    branch_tag_.data->push_back(1);
    IRVisitor::Visit(op->then_case);

    if (op->else_case.defined()) {
      branch_tag_.CopyOnWrite();
      branch_tag_.data->back() = -1;
      IRVisitor::Visit(op->else_case);
    }

    branch_tag_.CopyOnWrite();
    branch_tag_.data->pop_back();
    branch_tag_.data->back() += delta;
  }

  void Visit_(const AttrStmt *op) final {
    if (nullptr == op) {
      return;
    }

    if (op->attr_key == ktvm::ir::attr::storage_scope) {
      const auto buf = op->node.as<Variable>();
      if (buf == nullptr) return;

      auto pragma = op->value.as<StringImm>();
      CHECK(pragma != nullptr);
      storage_scope_[buf] = StorageScope::make(pragma->value);
    }

    if (op->attr_key == ktvm::ir::attr::coproc_scope) {
      if (!in_scope_) {
        in_scope_ = true;
        curr_attr_ = op;
        IRVisitor::Visit_(op);
        curr_attr_ = nullptr;
        in_scope_ = false;
      } else {
        // nested coproc can only be scalar pipe
        CHECK(curr_attr_->value.as<IntImm>()->value == 1);
        touched_[op].nest_attr = curr_attr_;
        curr_attr_ = op;
        IRVisitor::Visit_(op);
      }

      touched_[op].index = index_++;
      touched_[op].branch_tag = branch_tag_;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Allocate *op) final {
    if (op == nullptr) {
      return;
    }

    const Variable *var = op->buffer_var.get();
    StorageRange &range = storage_range_[var];
    if (op->new_expr.defined() && op->new_expr.as<IntImm>()) {
      range.addr = op->new_expr.as<IntImm>()->value;
    } else {
      range.addr = 0;
    }

    int64_t size = op->type.bytes();
    for (auto &s : op->extents) {
      auto size_ = s.as<IntImm>();
      if (size_ == nullptr) {
        continue;
      }
      size *= size_->value;
    }

    range.size = size;
    IRVisitor::Visit_(op);
  }

  bool DepBetween(const std::vector<MemInfo> &a, const std::vector<MemInfo> &b) const {
    for (auto i : a) {
      if (std::any_of(b.begin(), b.end(), [&, this](const MemInfo &j) { return MemAlias(i, j); })) {
        return true;
      }
    }

    return false;
  }

  bool depLoopBack(const AttrStmt *from, const AttrStmt *to, const For *loop) {
    CHECK((from != nullptr) && (to != nullptr) && (loop != nullptr));
    TouchEntry &from_entry = touched_[from];
    TouchEntry &to_entry = touched_[to];
    if ((from_entry.nest_attr == to) || (to_entry.nest_attr == from)) {
      return false;
    }

    std::vector<MemInfo> to_def;
    std::vector<MemInfo> to_use;
    std::unordered_map<const Variable *, Expr> next_loop_map;
    next_loop_map[loop->loop_var.get()] = loop->loop_var + 1;
    auto to_entry_def = to_entry.def;
    std::transform(to_entry_def.begin(), to_entry_def.end(), std::back_inserter(to_def), [&](const MemInfo &def) {
      return (MemInfo{def.base, Substitute(def.offset, next_loop_map), Substitute(def.extent, next_loop_map), def.type,
                      def.repeatTime, def.repeatStride, def.blockNumber, def.blockStride, def.blockSize});
    });

    auto to_entry_use = to_entry.use;
    std::transform(to_entry_use.begin(), to_entry_use.end(), std::back_inserter(to_use), [&](const MemInfo &use) {
      return (MemInfo{use.base, Substitute(use.offset, next_loop_map), Substitute(use.extent, next_loop_map), use.type,
                      use.repeatTime, use.repeatStride, use.blockNumber, use.blockStride, use.blockSize});
    });

    return (DepBetween(from_entry.def, to_def) || DepBetween(from_entry.def, to_use) ||
            DepBetween(from_entry.use, to_def));
  }

 private:
  // Get current storage scope.
  StorageScope GetScope(const Variable *buf) const {
    auto it = storage_scope_.find(buf);
    return it != storage_scope_.end() ? it->second : StorageScope::make("global");
  }

  // check if two mem could overlap
  bool MemAlias(const MemInfo &a, const MemInfo &b) const {
    // if stmt access register directly, the dependency is unclear, just return true.
    if ((a.base == const_reg.get()) || (b.base == const_reg.get())) {
      return true;
    }

    StorageScope as = GetScope(a.base);
    StorageScope bs = GetScope(b.base);

    // different scope, just return false;
    if (as != bs) {
      return false;
    }

    // for global buffer, may need offset
    if (as.rank == StorageRank::kGlobal && a.base != b.base) {
      return false;
    }

    int64_t addr_a = 0;
    int64_t addr_b = 0;
    if (as.rank != StorageRank::kGlobal) {
      auto it_a = storage_range_.find(a.base);
      auto it_b = storage_range_.find(b.base);
      CHECK(it_a != storage_range_.end() && it_b != storage_range_.end());
      addr_a = it_a->second.addr;
      addr_b = it_b->second.addr;
      if ((addr_a >= addr_b + it_b->second.size) || (addr_b >= addr_a + it_a->second.size)) return false;
    }

    if (!MemAliasByExtent(a, b, addr_a, addr_b)) {
      return false;
    }

    return ((a.repeatTime.defined() && b.repeatTime.defined()) ? MemAliasByStride(a, b, addr_a, addr_b) : true);
  }

  bool MemAliasByExtent(const MemInfo &a, const MemInfo &b, int64_t addr_a, int64_t addr_b) const {
    Expr a_unit = make_const(a.offset.type(), a.type.bytes());
    Expr b_unit = make_const(b.offset.type(), b.type.bytes());
    const auto oa = a.offset.as<IntImm>();
    const auto ea = a.extent.as<IntImm>();
    const auto ob = b.offset.as<IntImm>();
    const auto eb = b.extent.as<IntImm>();
    const auto au = a_unit.as<IntImm>();
    const auto bu = b_unit.as<IntImm>();

    bool bCheckNull = (oa && ea && ob && eb && au && bu);

    if (bCheckNull) {
      auto a_offset = static_cast<uint32_t>(oa->value);
      auto a_extent = static_cast<uint32_t>(ea->value);
      auto b_offset = static_cast<uint32_t>(ob->value);
      auto b_extent = static_cast<uint32_t>(eb->value);
      auto a_begin = static_cast<uint32_t>(static_cast<int64_t>(a_offset) * au->value + addr_a);
      auto b_begin = static_cast<uint32_t>(static_cast<int64_t>(b_offset) * bu->value + addr_b);
      auto a_range = static_cast<uint32_t>(static_cast<int64_t>(a_offset + a_extent) * au->value + addr_a);
      auto b_range = static_cast<uint32_t>(static_cast<int64_t>(b_offset + b_extent) * bu->value + addr_b);

      if (b_begin > a_begin) {
        if (b_begin >= a_range) {
          // Disjoint
          return false;
        } else {
          // Overlap
          return true;
        }
      } else if (a_begin > b_begin) {
        if (a_begin >= b_range) {
          // Disjoint
          return false;
        } else {
          // Overlap
          return true;
        }
      } else {
        // Overlap
        return true;
      }
    }

    bCheckNull = (oa && ea && ob && ob->value == oa->value);
    if (bCheckNull) {
      return true;
    }

    bCheckNull = (ob && eb && oa && oa->value == ob->value);
    if (bCheckNull) {
      return true;
    }

    Expr delta = make_const(Int(32), addr_a - addr_b);
    Expr t = addr_a > addr_b ? Simplify_cce(a.offset * a_unit + delta >= (b.offset + b.extent) * b_unit, range_)
                             : Simplify_cce(b.offset * b_unit - delta >= (a.offset + a.extent) * a_unit, range_);

    ktvm::arith::Analyzer analyzer_;
    if (analyzer_.CanProve(t)) {
      return false;
    }

    return true;
  }

  /**
   * Check whether meminfo has necessary args for calculating stride.
   */
  bool checkAllMemInfoExists(const IntImm *offset, const IntImm *typeSize, const IntImm *repeatTime,
                             const IntImm *repeatStride, const IntImm *blockNumber, const IntImm *blockStride,
                             const IntImm *blockSize) const {
    return ((offset != nullptr) && (typeSize != nullptr) && (repeatTime != nullptr) && (repeatStride != nullptr) &&
            (blockNumber != nullptr) && (blockStride != nullptr) && (blockSize != nullptr));
  }

  /**
   * Check overflow for range of Tensor.
   */
  void CheckOverFlowForTensor(int64_t offset, int64_t typeSize, int64_t repeatTime, int64_t repeatStride,
                              int64_t blockNumber, int64_t blockStride, int64_t blockSize) const {
    int64_t useOffsetPos = offset * typeSize;
    CHECK_MUL_OVERFLOW(offset, typeSize, useOffsetPos);

    int64_t useRepeatTime = repeatTime - 1;
    int64_t maxRepeatRange = useRepeatTime * repeatStride;
    CHECK_MUL_OVERFLOW(useRepeatTime, repeatStride, maxRepeatRange);
    int64_t maxRepeatPos = useOffsetPos + maxRepeatRange;
    CHECK_ADD_OVERFLOW(useOffsetPos, maxRepeatRange, maxRepeatPos);

    int64_t useBlockNumber = blockNumber - 1;
    int64_t maxBlockRange = useBlockNumber * blockStride;
    CHECK_MUL_OVERFLOW(useBlockNumber, blockStride, maxBlockRange);
    int64_t maxBlockPos = maxRepeatPos + maxBlockRange;
    CHECK_ADD_OVERFLOW(maxRepeatPos, maxBlockRange, maxBlockPos);

    int64_t tailOfLastBlock = maxBlockPos + blockSize;
    CHECK_ADD_OVERFLOW(maxBlockPos, blockSize, tailOfLastBlock);
  }

  /**
   * Struct for a block, recording its head and tail.
   */
  struct blockRange {
    int64_t head;
    int64_t tail;
  };

  bool MemAliasByStride(const MemInfo &a, const MemInfo &b, int64_t addr_a, int64_t addr_b) const {
    Expr offsetA = make_const(Int(32), 0);
    auto offsetB = Simplify_cce(b.offset - a.offset, range_);
    const IntImm *offsetIntImmOfA = offsetA.as<IntImm>();
    const IntImm *offsetIntImmOfB = offsetB.as<IntImm>();
    if (offsetIntImmOfB == nullptr) {
      return true;
    }

    if (offsetIntImmOfB->value < 0) {
      offsetB = -offsetB;
      offsetIntImmOfA = offsetB.as<IntImm>();
      offsetIntImmOfB = offsetA.as<IntImm>();
    }

    CHECK(offsetIntImmOfA != nullptr);
    CHECK(offsetIntImmOfA->value >= 0) << "offset A must be Non-negative number";

    Expr typeSizeExprOfA = make_const(a.offset.type(), a.type.bytes());
    const auto typeSizeIntImmOfA = typeSizeExprOfA.as<IntImm>();
    const auto repeatTimeIntImmOfA = a.repeatTime.as<IntImm>();
    const auto repeatStrideIntImmOfA = a.repeatStride.as<IntImm>();
    const auto blockNumberIntImmOfA = a.blockNumber.as<IntImm>();
    const auto blockStrideIntImmOfA = a.blockStride.as<IntImm>();
    const auto blockSizeIntImmOfA = a.blockSize.as<IntImm>();

    Expr typeSizeExprOfB = make_const(b.offset.type(), b.type.bytes());
    const auto typeSizeIntImmOfB = typeSizeExprOfB.as<IntImm>();
    const auto repeatTimeIntImmOfB = b.repeatTime.as<IntImm>();
    const auto repeatStrideIntImmOfB = b.repeatStride.as<IntImm>();
    const auto blockNumberIntImmOfB = b.blockNumber.as<IntImm>();
    const auto blockStrideIntImmOfB = b.blockStride.as<IntImm>();
    const auto blockSizeIntImmOfB = b.blockSize.as<IntImm>();

    // if checkResult is true, it means there is enough info to calc stride. If not, func will return overlap(default).
    bool checkResult =
      checkAllMemInfoExists(offsetIntImmOfA, typeSizeIntImmOfA, repeatTimeIntImmOfA, repeatStrideIntImmOfA,
                            blockNumberIntImmOfA, blockStrideIntImmOfA, blockSizeIntImmOfA) &&
      checkAllMemInfoExists(offsetIntImmOfB, typeSizeIntImmOfB, repeatTimeIntImmOfB, repeatStrideIntImmOfB,
                            blockNumberIntImmOfB, blockStrideIntImmOfB, blockSizeIntImmOfB);
    if (!checkResult) {
      return true;
    }

    // Detect overlap with stride
    // Get value
    int64_t typeSizeOfA = typeSizeIntImmOfA->value;
    int64_t offsetOfA = offsetIntImmOfA->value * typeSizeOfA + addr_a;
    int64_t repeatTimeOfA = repeatTimeIntImmOfA->value;
    int64_t repeatStrideOfA = repeatStrideIntImmOfA->value;
    int64_t blockNumberOfA = blockNumberIntImmOfA->value;
    int64_t blockStrideOfA = blockStrideIntImmOfA->value;
    int64_t blockSizeOfA = blockSizeIntImmOfA->value;

    int64_t typeSizeOfB = typeSizeIntImmOfB->value;
    int64_t offsetOfB = offsetIntImmOfB->value * typeSizeOfB + addr_b;
    int64_t repeatTimeOfB = repeatTimeIntImmOfB->value;
    int64_t repeatStrideOfB = repeatStrideIntImmOfB->value;
    int64_t blockNumberOfB = blockNumberIntImmOfB->value;
    int64_t blockStrideOfB = blockStrideIntImmOfB->value;
    int64_t blockSizeOfB = blockSizeIntImmOfB->value;

    // Prune: if two tensors has same offset, they have overlap.
    if (offsetOfA == offsetOfB) {
      return true;
    }

    // Check add and mul overflow
    CheckOverFlowForTensor(offsetOfA, typeSizeOfA, repeatTimeOfA, repeatStrideOfA, blockNumberOfA, blockStrideOfA,
                           blockSizeOfA);
    CheckOverFlowForTensor(offsetOfB, typeSizeOfB, repeatTimeOfB, repeatStrideOfB, blockNumberOfB, blockStrideOfB,
                           blockSizeOfB);

    // Calc TotalBlockNum
    int64_t TotalBlockNumOfA = repeatTimeOfA * blockNumberOfA;
    int64_t TotalBlockNumOfB = repeatTimeOfB * blockNumberOfB;
    int64_t currentBlockCntOfA = 0;
    int64_t currentBlockCntOfB = 0;

    // Calc blockVec Of A and B
    std::vector<blockRange> blockVecOfA;
    for (int64_t i = 0; i < TotalBlockNumOfA; ++i) {
      // Calc start and end position of current block for A
      CHECK_NE(blockNumberOfA, 0);
      int64_t currentBlockHeadOfA =
        offsetOfA + (i / blockNumberOfA) * repeatStrideOfA + (i % blockNumberOfA) * blockStrideOfA;
      int64_t currentBlockTailOfA = currentBlockHeadOfA + blockSizeOfA;

      blockVecOfA.push_back({currentBlockHeadOfA, currentBlockTailOfA});
    }

    std::sort(blockVecOfA.begin(), blockVecOfA.end(), [](const blockRange &blockA, const blockRange &blockB) {
      return (blockA.head < blockB.head) || (blockA.head == blockB.head && blockA.tail < blockB.tail);
    });

    std::vector<blockRange> blockVecOfB;
    for (int64_t i = 0; i < TotalBlockNumOfB; i++) {
      // Calc start and end position of current block for B
      CHECK_NE(blockNumberOfB, 0);
      int64_t currentBlockHeadOfB =
        offsetOfB + (i / blockNumberOfB) * repeatStrideOfB + (i % blockNumberOfB) * blockStrideOfB;
      int64_t currentBlockTailOfB = currentBlockHeadOfB + blockSizeOfB;

      blockVecOfB.push_back({currentBlockHeadOfB, currentBlockTailOfB});
    }

    std::sort(blockVecOfB.begin(), blockVecOfB.end(), [](const blockRange &blockA, const blockRange &blockB) {
      return (blockA.head < blockB.head) || (blockA.head == blockB.head && blockA.tail < blockB.tail);
    });

    // This is the minimum condition for no overlap:
    //                              [blockA_i.head, blockA_i.tail)
    // [blockB_j.head, blockB_j.tail)                            [blockB_j+1.head, blockB_j+1.tail)
    while (currentBlockCntOfA < TotalBlockNumOfA && currentBlockCntOfB < TotalBlockNumOfB) {
      // If the head of A is greater than the tail of B, the pointer to the current B + 1;
      // If the tail of A is less than the head of B, the pointer to the current A + 1;
      // Otherwise, return overlap.
      if (blockVecOfA.at(currentBlockCntOfA).head >= blockVecOfB.at(currentBlockCntOfB).tail) {
        currentBlockCntOfB++;
        continue;
      }

      if (blockVecOfA.at(currentBlockCntOfA).tail <= blockVecOfB.at(currentBlockCntOfB).head) {
        currentBlockCntOfA++;
        continue;
      }

      // Overlap
      return true;
    }

    // If the A or B traversal ends, then there is no overlap.
    return false;
  }

  void RegMoveVisit(const Call *op) {
    if (nullptr == op) {
      return;
    }

    // if first dest arg is reg(load(buffer)), we need treat the load as def touch.
    bool first_arg = true;
    for (Expr arg : op->args) {
      const Call *arg_op = arg.as<Call>();
      if (first_arg) {
        first_arg = false;
        if ((arg_op != nullptr) && (arg_op->name == "reg")) {
          CHECK_EQ(arg_op->args.size(), 1);
          const Load *load = arg_op->args[0].as<Load>();
          CHECK(load != nullptr);
          MemInfo m = {load->buffer_var.get(), load->index, 1, load->type};
          touched_[curr_attr_].def.push_back(m);

          continue;
        }
      }

      IRVisitor::Visit(arg);
    }
  }

  Expr AlignExtent(Expr extent, const Type type) {
    int64_t align = 32 / type.bytes();
    const auto imm = extent.as<IntImm>();

    if (imm != nullptr && align != 0) {
      int64_t mod = imm->value % align;
      if (mod == 0) {
        return extent;
      }

      return make_const(Int(32), imm->value + (align - mod));
    }

    Expr ea = make_const(Int(32), align);
    if (DynamicRange()) {
      return ea;
    }

    return Simplify_cce((truncdiv((extent + make_const(Int(32), align - 1)), ea)) * ea, range_);
  }

  bool DynamicRange() {
    for (auto item : range_) {
      if (!(isImm(item.second->min) && isImm(item.second->extent))) {
        return true;
      }
    }

    return false;
  }

  const AttrStmt *curr_attr_{nullptr};
  bool in_scope_{false};
  uint32_t index_{0};
  // current branch tag
  BranchTag branch_tag_;
  // range info for Simplify_cce
  Map<Var, Range> range_;

  std::unordered_map<const AttrStmt *, TouchEntry> touched_;
  std::unordered_map<uint32_t, bool> dep_found_;
  // The storage scope of each buffer
  std::unordered_map<const Variable *, StorageScope> storage_scope_;
  // The base address and size of each buffer
  std::unordered_map<const Variable *, StorageRange> storage_range_;
  // all register names maybe touched directly in intrinsic
  std::unordered_set<std::string> register_{"VA0", "VA1", "VA2", "VA3"};

  friend class DFAnalyzeOnline;
  friend class DFAnalyzeOffline;
};

struct MemDependencyNode {
  // inverse pointer
  const AttrStmt *stmt;
  const DFVisitor *analyzer;
  // attribute: entry
  TouchEntry *entry;
};

class MemDependencyGraph : public DependencyGraph<MemDependencyNode> {
 public:
  explicit MemDependencyGraph(std::vector<MemDependencyNode> &nodes, bool check_redundant_arcs = false)
      : DependencyGraph(nodes, check_redundant_arcs) {}
  ~MemDependencyGraph() override = default;

  DepType GetDepType(const MemDependencyNode *a, const MemDependencyNode *b) override {
    CHECK(a && b);

    if (a->entry->index > b->entry->index) {
      if (b->entry->nest_attr == a->stmt) {
        return DepType::kNone;
      }

      const DFVisitor *self = a->analyzer;
      CHECK(self);

      if (self->DepBetween(a->entry->use, b->entry->def)) {
        return DepType::kRAW;
      } else if (self->DepBetween(a->entry->def, b->entry->def)) {
        return DepType::kWAW;
      } else if (self->DepBetween(a->entry->def, b->entry->use)) {
        return DepType::kWAR;
      }
    }

    return DepType::kNone;
  }

  void AddDepRelation(MemDependencyNode *a, MemDependencyNode *b, DepType type) override {
    CHECK(a && b);
    switch (type) {
      case DepType::kRAW:
        b->entry->RAW.insert(a->stmt);
        a->entry->RAW.insert(b->stmt);
        break;
      case DepType::kWAR:
        b->entry->WAR.insert(a->stmt);
        a->entry->WAR.insert(b->stmt);
        break;
      case DepType::kWAW:
        b->entry->WAW.insert(a->stmt);
        a->entry->WAW.insert(b->stmt);
        break;
      default:
        break;
    }
  }

  bool IsBranchAway(const MemDependencyNode *a, const MemDependencyNode *b) override {
    CHECK(a && b);
    std::vector<int> &va = *(a->entry->branch_tag.data.get());
    std::vector<int> &vb = *(b->entry->branch_tag.data.get());
    int n = std::min(va.size(), vb.size());

    for (int i = 0; i < n; i++) {
      if (va[i] * vb[i] < 0) return true;
      if (va[i] != vb[i]) return false;
    }

    return false;
  }
};

class DFAnalyzeOnline : public DFAnalyzer {
 public:
  DFAnalyzeOnline() {}
  ~DFAnalyzeOnline() override = default;

  void Plan(Stmt stmt) final { visitor_.Plan(stmt); }

  bool DepForward(const AttrStmt *a, const AttrStmt *b) final {
    TouchEntry &ea = visitor_.touched_[a];
    TouchEntry &eb = visitor_.touched_[b];
    if ((ea.nest_attr == b) || (eb.nest_attr == a)) {
      return false;
    }

    uint64_t dep_tag = ea.index < eb.index ? (ea.index << 16) | eb.index : (eb.index << 16) | ea.index;
    auto it = visitor_.dep_found_.find(dep_tag);
    if (it != visitor_.dep_found_.end()) {
      return it->second;
    }

    bool dep = (visitor_.DepBetween(ea.def, eb.def) || visitor_.DepBetween(ea.def, eb.use) ||
                visitor_.DepBetween(ea.use, eb.def));
    visitor_.dep_found_[dep_tag] = dep;

    return dep;
  }

  bool DepBackward(const AttrStmt *a, const AttrStmt *b, const For *loop) final {
    return visitor_.depLoopBack(a, b, loop);
  }

 private:
  DFVisitor visitor_;
};

class DFAnalyzeOffline : public DFAnalyzer {
 public:
  DFAnalyzeOffline() {}
  ~DFAnalyzeOffline() override = default;

  void Plan(Stmt stmt) final {
    visitor_.Plan(stmt);
    BackWardScanDF();
  }

  bool DepForward(const AttrStmt *a, const AttrStmt *b) final {
    auto entry = visitor_.touched_[a];
    if ((entry.RAW.count(b) != 0) || (entry.WAR.count(b) != 0) || (entry.WAW.count(b) != 0)) {
      return true;
    }

    return false;
  }

  bool DepBackward(const AttrStmt *a, const AttrStmt *b, const For *loop) final {
    return visitor_.depLoopBack(a, b, loop);
  }

  void BackWardScanDF() {
    std::vector<MemDependencyNode> nodes;
    // keep the order by index of touch entry
    nodes.resize(visitor_.touched_.size());
    for (auto &i : visitor_.touched_) {
      MemDependencyNode &n = nodes[i.second.index];
      n.stmt = i.first;
      n.analyzer = &visitor_;
      n.entry = &i.second;
    }

    MemDependencyGraph graph(nodes, false);
    graph.BuildGraph();
  }

 private:
  DFVisitor visitor_;
};

std::shared_ptr<DFAnalyzer> BuildDfAnalyzer(const Stmt stmt, bool prebuild) {
  DFAnalyzer *analyzer =
    prebuild ? static_cast<DFAnalyzer *>(new DFAnalyzeOffline()) : static_cast<DFAnalyzer *>(new DFAnalyzeOnline());
  analyzer->Plan(stmt);

  return std::shared_ptr<DFAnalyzer>(analyzer);
}
}  // namespace ir
}  // namespace akg
