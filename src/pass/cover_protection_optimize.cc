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

/**
 * The following pass simulates the protections needed in case of non alignment
 * of for-loops resulting of copy_ubuf_to_gm function.
 * The purpose is to optimize the number of these protections.
 * It simulates the following functions: copy_gm_to_ubuf then copy_ubuf_to_gm
 */

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <pass/ir_util.h>
#include <tvm.h>

#include "src/common/util.h"
#include "emit_insn/insn_info.h"

namespace akg {
namespace ir {
struct ForInfo {
  Expr var;
  Expr min = 0;
  Expr extent = 0;
};

struct DMAInfo {
  std::vector<ForInfo> external_for_info;
  std::vector<ForInfo> internal_for_info;
  std::vector<Expr> if_condition_info;
  Expr n_burst = 1;
  Expr len_burst = 1;
  Expr src_stride = 1;
  Expr dst_stride = 1;
  Expr block_size = 1;
  Expr real_block_size = 1;
  Expr dst_var_stride_expr = 1;
};

struct Block {
  int time;
  int s;
  int re;
  int e;
  int thread;
  int line;
  Map<Expr, Expr> iterators;

  Block(int time, int s, int re, int e, int thread, int line, Map<Expr, Expr> iter)
      : time(time), s(s), re(re), e(e), thread(thread), line(line), iterators(iter) {}

  bool Overlap(const Block &b) const {
    int s1 = s;
    int d1 = e - s;
    int s2 = b.s;
    int d2 = b.re - b.s;
    return s1 + d1 > s2 && s2 + d2 > s1;
  }
};

struct ConditionInfo {
  std::vector<std::vector<Block>> blocks;
  std::vector<int> position;
  int current_ubuf_line;
  // for stats
  std::vector<std::vector<Block>> blocks_logs;

  ConditionInfo() : current_ubuf_line(0) {}
  explicit ConditionInfo(int value) : current_ubuf_line(value) {}
};

struct GraphProtection {
  std::vector<std::vector<ForInfo *>> loops;
  std::vector<std::vector<Expr>> nodes_list;
  std::vector<Expr> dst_var_stride_expr;
  std::vector<Map<Expr, Expr>> nodes_index;
  std::vector<std::vector<Expr>> if_condition;
  std::vector<std::vector<int>> graph;
  std::vector<std::vector<int>> matrix;
  std::map<int, int> leaf_line_num;
  std::vector<int> buffer_size;
  std::vector<int> real_burst_size;
  std::vector<int> ranges;
  std::vector<int> iterators;
  std::vector<ForInfo> ubufs;
  std::vector<ForInfo *> nodes;
  ForInfo thread_node;
  int nodes_count = 0;
  int lines_count = 0;
  int temp_line_num = 0;
  bool risk_alert = false;
  size_t th_block = 0;

  void InitUbufNode() {
    ForInfo ubuf;
    ubuf.var = Var("ubuf");
    ubuf.extent = 1;
    ubufs.push_back(ubuf);
    loops[lines_count].push_back(&ubufs[lines_count]);
    nodes.push_back(&ubufs[ubufs.size() - 1]);
  }

  bool FindSameNode(ForInfo *node) {
    bool found = false;
    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      if (node->var.get() == nodes[i]->var.get()) {
        found = true;
      }
    }
    return found;
  }

  int GetNodeByIndex(ForInfo *node) {
    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      if (nodes[i]->var.get() == node->var.get()) {
        return i;
      }
    }
    return 0;
  }

  int SameRoot(int current_loop, int node) {
    if (current_loop == 0) return 0;
    if (loops[current_loop][node + 1]->var.get() == loops[current_loop - 1][node + 1]->var.get()) {
      return SameRoot(current_loop, node + 1);
    } else {
      return node;
    }
  }

  void BuildAdjacencyMatrix() {
    matrix.resize(nodes.size(), std::vector<int>(nodes.size(), 0));
    for (int i = 0; i < static_cast<int>(loops.size()); i++) {
      int root = SameRoot(i, 0);
      for (int j = root; j < static_cast<int>(loops[i].size() - 1); j++) {
        int ind_x = GetNodeByIndex(loops[i][j]);
        int ind_y = GetNodeByIndex(loops[i][j + 1]);
        matrix[ind_x][ind_y] = 1;
      }
    }
  }

  void BuildAdjacencyList() {
    for (int i = 0; i < static_cast<int>(matrix.size()); i++) {
      graph.emplace_back();
      for (int j = 0; j < static_cast<int>(matrix.size()); j++) {
        if (matrix[i][j] == 1) graph[i].push_back(j);
      }
    }
  }

  void BuildNodesOrder() {
    nodes.clear();
    for (auto &nodes_lines : loops) {
      for (auto &node : nodes_lines) {
        if (!FindSameNode(node)) {
          nodes.push_back(node);
        }
      }
    }
  }

  void PrintMatrix() {
    LOG(INFO) << "--- Matrix ---";
    for (int i = 0; i < static_cast<int>(matrix.size()); i++) {
      std::stringstream ss;
      ss << i + ": ";
      for (int j = 0; j < static_cast<int>(matrix[i].size()); j++) {
        ss << matrix[i][j] << " ";
      }
      LOG(INFO) << ss.str();
    }
  }

  void PrintLoops() {
    LOG(INFO) << "--- Loops ---";
    for (int i = 0; i < static_cast<int>(loops.size()); i++) {
      std::stringstream ss;
      for (int j = 0; j < static_cast<int>(loops[i].size()); j++) {
        ss << loops[i][j]->var.get() << " ";
      }
      LOG(INFO) << ss.str();
    }

    for (int i = 0; i < static_cast<int>(loops.size()); i++) {
      std::stringstream ss;
      for (int j = 0; j < static_cast<int>(loops[i].size()); j++) {
        ss << loops[i][j]->var << " ";
      }
      LOG(INFO) << ss.str();
    }
  }

  void PrintNodes() {
    LOG(INFO) << "--- Nodes ---";
    LOG(INFO) << "Number of nodes: " << nodes.size();
    std::stringstream ss;
    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      ss << nodes[i]->var << " | ";
    }
    LOG(INFO) << ss.str();
  }

  void PrintRanges() {
    LOG(INFO) << "--- Ranges ---";
    std::stringstream ss;
    for (int i = 0; i < static_cast<int>(ranges.size()); i++) {
      ss << nodes[i]->var << ": " << ranges[i] << " | ";
    }
    LOG(INFO) << ss.str();
  }

  void Increment(int current_node) {
    iterators[current_node] = (iterators[current_node] + 1) % GetInt32Const(ranges[current_node]);

    if (iterators[current_node] == 0) {
      int parent = 0;
      for (int i = 0; i < current_node; i++) {
        if (matrix[i][current_node] == 1) {
          parent = i;
          break;
        }
      }
      bool has_brother = false;
      for (int j = current_node + 1; j < nodes_count; j++) {
        if (matrix[parent][j] == 1) {
          has_brother = true;
          break;
        }
      }

      if (!has_brother) {
        Increment(parent);
      }
    }
  }

  void FindLeafParentIter(Map<Expr, Expr> &iter, int current_node) {
    for (int i = current_node; i >= 0; i--) {
      if (matrix[i][current_node] == 1) {
        if (i == 0 && Equal(thread_node.extent, 1)) return;
        iter.Set(nodes[i]->var, Expr(iterators[i]));
        FindLeafParentIter(iter, i);
        break;
      }
    }
  }

  void FindNodeFromExpr(const Expr &param) {
    if (auto add = param.as<Add>()) {
      FindNodeFromExpr(add->a);
      FindNodeFromExpr(add->b);
    } else if (auto sub = param.as<Sub>()) {
      FindNodeFromExpr(sub->a);
      FindNodeFromExpr(sub->b);
    } else if (auto mul = param.as<Mul>()) {
      FindNodeFromExpr(mul->a);
      FindNodeFromExpr(mul->b);
    } else if (auto div = param.as<Div>()) {
      FindNodeFromExpr(div->a);
      FindNodeFromExpr(div->b);
    } else if (auto f_div = param.as<FloorDiv>()) {
      FindNodeFromExpr(f_div->a);
      FindNodeFromExpr(f_div->b);
    } else if (auto mod = param.as<Mod>()) {
      FindNodeFromExpr(mod->a);
      FindNodeFromExpr(mod->b);
    } else if (auto f_mod = param.as<FloorMod>()) {
      FindNodeFromExpr(f_mod->a);
      FindNodeFromExpr(f_mod->b);
    } else if (param->IsInstance<Variable>()) {
      nodes_list[lines_count].push_back(param);
    }
  }

  Expr EvalParam(const Expr &param) {
    if (auto add = param.as<Add>()) {
      return Add::make(EvalParam(add->a), EvalParam(add->b));
    } else if (auto sub = param.as<Sub>()) {
      return Sub::make(EvalParam(sub->a), EvalParam(sub->b));
    } else if (auto mul = param.as<Mul>()) {
      return Mul::make(EvalParam(mul->a), EvalParam(mul->b));
    } else if (auto div = param.as<Div>()) {
      return Div::make(EvalParam(div->a), EvalParam(div->b));
    } else if (auto f_div = param.as<FloorDiv>()) {
      return FloorDiv::make(EvalParam(f_div->a), EvalParam(f_div->b));
    } else if (auto mod = param.as<Mod>()) {
      return Mod::make(EvalParam(mod->a), EvalParam(mod->b));
    } else if (auto f_mod = param.as<FloorMod>()) {
      return FloorMod::make(EvalParam(f_mod->a), EvalParam(f_mod->b));
    } else if (nodes_index[temp_line_num].find(param) != nodes_index[temp_line_num].end()) {
      return Expr(iterators[GetInt32Const(nodes_index[temp_line_num][param])]);
    } else {
      return param;
    }
  }

  Expr EvalParamsSymbol(const Expr &condition) {
    Expr left;
    Expr right;
    if (condition->IsInstance<EQ>()) {
      left = condition.as<EQ>()->a;
      right = condition.as<EQ>()->b;
      return Simplify(EQ::make(EvalParam(left), EvalParam(right)));
    } else if (condition->IsInstance<NE>()) {
      left = condition.as<NE>()->a;
      right = condition.as<NE>()->b;
      return Simplify(NE::make(EvalParam(left), EvalParam(right)));
    } else if (condition->IsInstance<LT>()) {
      left = condition.as<LT>()->a;
      right = condition.as<LT>()->b;
      return Simplify(LT::make(EvalParam(left), EvalParam(right)));
    } else if (condition->IsInstance<LE>()) {
      left = condition.as<LE>()->a;
      right = condition.as<LE>()->b;
      return Simplify(LE::make(EvalParam(left), EvalParam(right)));
    } else if (condition->IsInstance<GT>()) {
      left = condition.as<GT>()->a;
      right = condition.as<GT>()->b;
      return Simplify(GT::make(EvalParam(left), EvalParam(right)));
    } else if (condition->IsInstance<GE>()) {
      left = condition.as<GE>()->a;
      right = condition.as<GE>()->b;
      return Simplify(GE::make(EvalParam(left), EvalParam(right)));
    } else if (condition->IsInstance<Or>()) {
      left = condition.as<Or>()->a;
      right = condition.as<Or>()->b;
      return Simplify(Or::make(EvalParamsSymbol(left), EvalParamsSymbol(right)));
    } else if (condition->IsInstance<And>()) {
      left = condition.as<And>()->a;
      right = condition.as<And>()->b;
      return Simplify(And::make(EvalParamsSymbol(left), EvalParamsSymbol(right)));
    } else {
      LOG(INFO) << "Incorrect condition!";
      risk_alert = true;
      return condition;
    }
  }

  Expr EvalLineConditions() {
    if (if_condition[temp_line_num].empty()) return const_true();
    Expr is_valid = const_true();
    for (auto condition : if_condition[temp_line_num]) {
      if (condition->IsInstance<Not>()) {
        is_valid = is_valid && Simplify(Not::make(EvalParamsSymbol(condition.as<Not>()->a)));
      } else {
        is_valid = is_valid && EvalParamsSymbol(condition);
      }
    }
    return is_valid;
  }

  /**
   * Simulate the buffer creation (copy_gm_to_ubuf function) of the data following the order of
   * the IR for-loops through "blocks"
   */
  void Simulator(std::vector<Block> &blocks, int current_node) {
    if (graph[current_node].empty()) {
      temp_line_num = leaf_line_num[current_node];
      if (Equal(EvalLineConditions(), const_true())) {
        int s = GetInt32Const(Simplify(EvalParam(dst_var_stride_expr[temp_line_num])));
        int re = s + real_burst_size[temp_line_num];
        int e = s + buffer_size[temp_line_num];
        int time = blocks.size();
        Map<Expr, Expr> iter;
        FindLeafParentIter(iter, current_node);
        blocks.emplace_back(Block(time, s, re, e, iterators[0], temp_line_num, iter));
      }
      Increment(current_node);
    } else {
      for (int i = iterators[current_node]; i < ranges[current_node]; i++) {
        for (int j = 0; j < static_cast<int>(graph[current_node].size()); j++) {
          if (blocks.size() > th_block) return;
          Simulator(blocks, graph[current_node][j]);
        }
      }
    }
  }

  /**
   * Check which block (= buffer) could overlap another block which could
   * occurs during the writting of the data from a local buffer to the
   * general memory (copy_ubuf_to_gm function)
   */
  std::vector<bool> CheckOverlap(std::vector<Block> &blocks) {
    auto max_block =
      std::max_element(blocks.begin(), blocks.end(), [](const Block &a, const Block &b) { return a.e < b.e; });
    LOG(INFO) << "Simulated memory size: " << (*max_block).e;
    std::vector<int> memory((*max_block).e, 0);
    std::vector<std::vector<Block *>> info((*max_block).e);
    std::vector<bool> check(blocks.size(), false);
    for (int i = 0; i < static_cast<int>(blocks.size()); i++) {
      int j = blocks[i].s;
      info[j].push_back(&blocks[i]);
      if (j >= static_cast<int>(info.size()) && j >= static_cast<int>(memory.size())) break;
      for (auto block : info[j]) {
        if (blocks[i].thread != (*block).thread) {
          check[(*block).time] = true;
        }
      }
      memory[j] = 1;
      for (j = blocks[i].re; j < blocks[i].e; j++) {
        info[j].push_back(&blocks[i]);
        if (j >= static_cast<int>(info.size()) && j >= static_cast<int>(memory.size())) break;
        for (auto block : info[j]) {
          if (blocks[i].thread == (*block).thread && memory[j] == 1) {
            check[i] = true;
          } else if (blocks[i].thread != (*block).thread) {
            if (memory[j] == 1) {
              check[i] = true;
            } else {
              check[(*block).time] = true;
            }
          }
        }
        if (memory[j] == 0) {
          memory[j] = 2;
        }
      }
    }
    return check;
  }
};

class CoverProtectGather : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "gm_addr") {
      dma_copy_gm = true;
      current_info = DMAInfo();
      current_info.if_condition_info = if_condition_list;
      IRVisitor::Visit(op->body);
      CHECK(op->value.as<Variable>());
      if (infos.count(op->value.as<Variable>()) == 0) {
        infos[op->value.as<Variable>()] = {current_info};
      } else {
        infos[op->value.as<Variable>()].push_back(current_info);
      }
      dma_copy_gm = false;
      if (risk_rw.find(op->value.as<Variable>()->name_hint) != risk_rw.end()) risk_rwalert = true;
      return;
    } else if (op->attr_key == "external_for_loop") {
      ExtractForInfo(op->value, true);
    } else if (op->attr_key == "internal_for_loop") {
      flag_internal_for = true;
      IRVisitor::Visit(op->body);
      flag_internal_for = false;
      return;
    } else if (op->attr_key == "intrin_args") {
      flag_internal_for = false;
      ExtractIntrinInfo(op->value);
    } else if (op->attr_key == "src_var_stride") {
      ExtractVarStride(op->value, true);
    } else if (op->attr_key == "dst_var_stride") {
      ExtractVarStride(op->value, false);
    } else if (op->attr_key == "gm_addr_rw") {
      CHECK(op->value.as<Variable>());
      risk_rw.insert(op->value.as<Variable>()->name_hint);
    }

    IRVisitor::Visit(op->body);
  }

  void Visit_(const IfThenElse *op) {
    if (op->then_case.defined()) {
      if_condition_list.push_back(op->condition);
      this->Visit(op->then_case);
      if_condition_list.pop_back();
    }
    if (op->else_case.defined()) {
      if_condition_list.push_back(Not::make(op->condition));
      this->Visit(op->else_case);
      if_condition_list.pop_back();
    }
  }

  void Visit_(const For *op) {
    if (flag_internal_for) {
      ForInfo info;
      info.var = op->loop_var;
      info.extent = op->extent;
      current_info.internal_for_info.push_back(info);
    }
    IRVisitor::Visit(op->body);
  }

  std::unordered_map<const Variable *, std::vector<DMAInfo>> infos;
  std::vector<Expr> if_condition_list;
  std::set<std::string> risk_rw;
  bool risk_rwalert = false;
  Expr thread_var = 0;

 private:
  void ExtractForInfo(const Expr &comment, bool external) {
    if (IsZero(comment)) {
      return;
    }

    auto array = GetBinaryOpExprChildren(comment);
    if (comment.as<Sub>()) {
      ForInfo info;
      info.var = array[0];
      info.extent = array[1];

      if (external) {
        current_info.external_for_info.push_back(info);
      } else {
        current_info.internal_for_info.push_back(info);
      }
    } else {
      for (auto &expr : array) {
        if (expr.as<Mul>()) {
          ExtractForInfo(expr, external);
        } else if (expr.as<Sub>()) {
          auto key_value = GetBinaryOpExprChildren(expr);
          ForInfo info;
          info.var = key_value[0];
          info.extent = key_value[1];
          if (external) {
            current_info.external_for_info.push_back(info);
          } else {
            current_info.internal_for_info.push_back(info);
          }
        }
      }
    }
  }

  void ExtractIntrinInfo(const Expr &comment) {
    auto array = GetBinaryOpExprChildren(comment);
    for (auto &expr : array) {
      if (expr.as<Mul>()) {
        ExtractIntrinInfo(expr);
      } else if (expr.as<Sub>()) {
        auto key_value = GetBinaryOpExprChildren(expr);
        if (key_value[0].as<Variable>()->name_hint == "nBurst") {
          current_info.n_burst = key_value[1];
        } else if (key_value[0].as<Variable>()->name_hint == "lenBurst") {
          current_info.len_burst = key_value[1];
        } else if (key_value[0].as<Variable>()->name_hint == "srcStride") {
          current_info.src_stride = key_value[1];
        } else if (key_value[0].as<Variable>()->name_hint == "dstStride") {
          current_info.dst_stride = key_value[1];
        } else if (key_value[0].as<Variable>()->name_hint == "blockSize") {
          current_info.block_size = key_value[1];
        } else if (key_value[0].as<Variable>()->name_hint == "realBlockSize") {
          current_info.real_block_size = key_value[1];
        }
      }
    }
  }

  void FindThreadFromExpr(const Expr &param) {
    if (auto add = param.as<Add>()) {
      FindThreadFromExpr(add->a);
    } else if (auto sub = param.as<Sub>()) {
      FindThreadFromExpr(sub->a);
    } else if (auto mul = param.as<Mul>()) {
      FindThreadFromExpr(mul->a);
    } else if (auto div = param.as<Div>()) {
      FindThreadFromExpr(div->a);
    } else if (auto f_div = param.as<FloorDiv>()) {
      FindThreadFromExpr(f_div->a);
    } else if (auto mod = param.as<Mod>()) {
      FindThreadFromExpr(mod->a);
    } else if (auto f_mod = param.as<FloorMod>()) {
      FindThreadFromExpr(f_mod->a);
    } else if (param->IsInstance<Variable>()) {
      if (param.as<Variable>()->name_hint == "blockIdx.x") {
        thread_var = param;
      }
    }
  }

  void ExtractVarStride(const Expr &comment, bool src) {
    if (IsZero(comment)) {
      return;
    }

    if (!src) {
      if (!thread_var->IsInstance<Variable>()) {
        FindThreadFromExpr(comment);
        if (!thread_var->IsInstance<Variable>()) {
          thread_var = Var{"blockIdx.x"};
        }
      }
      current_info.dst_var_stride_expr = comment;
    }

    return;
  }

  DMAInfo current_info;
  bool dma_copy_gm{false};
  bool flag_internal_for{false};
};

class CoverProtector : public IRMutator {
 public:
  CoverProtector(const std::unordered_map<const Variable *, std::vector<DMAInfo>> &info, bool risk_rwalert,
                 Expr thread_var, size_t th_block, size_t th_protect)
      : infos(info), risk_alert(risk_rwalert), thread_var(thread_var), th_block(th_block), th_protect(th_protect) {}
  ~CoverProtector() override = default;

  void PrintLoopsLogs(std::vector<akg::ir::DMAInfo> &dma_infos) {
    for (auto &info : dma_infos) {
      LOG(INFO) << "external for info size " << info.external_for_info.size();
      LOG(INFO) << "internal for info size " << info.internal_for_info.size();
      LOG(INFO) << "intrin info is " << info.n_burst << " " << info.len_burst << " " << info.src_stride << " "
                << info.dst_stride << " " << info.block_size << " " << info.real_block_size;
      LOG(INFO) << "condition " << Array<Expr>(info.if_condition_info) << " \n";
      LOG(INFO) << "constants expr " << info.dst_var_stride_expr;
    }

    for (auto &info : dma_infos) {
      LOG(INFO) << "---------- EXTERNAL ----------";
      for (auto &node : info.external_for_info) {
        LOG(INFO) << "Print node var: " << node.var << " " << node.var.get();
        LOG(INFO) << "Print node extent: " << node.extent << " " << node.extent.get();
        LOG(INFO) << "Print node min:" << node.min << " " << node.min.get();
      }

      LOG(INFO) << "---------- INTERNAL ----------";
      for (auto &node : info.internal_for_info) {
        LOG(INFO) << "Print node var: " << node.var << " " << node.var.get();
        LOG(INFO) << "Print node extent: " << node.extent << " " << node.extent.get();
        LOG(INFO) << "Print node min: " << node.min << " " << node.min.get();
      }
    }
  }

  void GetInfoForCondition(const Variable *addr_gm) {
    auto dma_infos = this->infos[addr_gm];
    LOG(INFO) << "gm" << addr_gm->name_hint << " has " << dma_infos.size() << " dma infos";

    if (this->risk_alert) {
      LOG(INFO) << "Alerted: generate const true";
      data[addr_gm] = {ConditionInfo(-1)};
      return;
    }

    PrintLoopsLogs(dma_infos);

    // initialize structure and vectors
    ConditionInfo cond_info;
    GraphProtection graph;
    graph.ubufs.reserve(dma_infos.size());

    graph.th_block = this->th_block;
    graph.thread_node.var = this->thread_var;
    graph.thread_node.extent = this->thread_extent;
    graph.nodes.push_back(&graph.thread_node);

    for (auto &info : dma_infos) {
      graph.loops.emplace_back();
      graph.nodes_index.emplace_back();
      graph.nodes_list.emplace_back();
      graph.if_condition.emplace_back();
      cond_info.blocks.emplace_back();
      cond_info.blocks_logs.emplace_back();

      // Test eval expr in simulation
      graph.dst_var_stride_expr.push_back(info.dst_var_stride_expr);

      // set some intrin info
      graph.buffer_size.push_back(GetInt32Const(info.block_size * info.len_burst));
      graph.real_burst_size.push_back(GetInt32Const(info.real_block_size));

      if (!Equal(info.block_size * info.len_burst, info.real_block_size)) {
        cond_info.position.push_back(graph.lines_count);
      }

      // add the thread to the nodes list at the first position
      graph.loops[graph.lines_count].push_back(&graph.thread_node);
      graph.nodes_index[graph.lines_count].Set(graph.thread_node.var, Expr(0));

      // add nodes of current line
      info.external_for_info.insert(info.external_for_info.end(), info.internal_for_info.begin(),
                                    info.internal_for_info.end());
      for (int i = 0; i < static_cast<int>(info.external_for_info.size()); i++) {
        graph.loops[graph.lines_count].push_back(&info.external_for_info[i]);
      }

      // set conditions
      for (auto cond : info.if_condition_info) {
        graph.if_condition[graph.lines_count].push_back(cond);
      }

      // set ubuf variable
      graph.InitUbufNode();

      // find nodes per line from expression
      graph.FindNodeFromExpr(info.dst_var_stride_expr);

      graph.lines_count++;
    }

    graph.BuildNodesOrder();

    graph.ranges.push_back(GetInt32Const(graph.thread_node.extent));
    for (int i = 1; i < static_cast<int>(graph.nodes.size()); i++) {
      graph.ranges.push_back(GetInt32Const(graph.nodes[i]->extent));
      if (graph.nodes[i]->var.as<Variable>()->name_hint == "ubuf") {
        graph.leaf_line_num.emplace(i, graph.leaf_line_num.size());
        continue;
      }
      for (int j = 0; j < graph.lines_count; j++) {
        graph.nodes_index[j].Set(graph.nodes[i]->var, Expr(i));
      }
    }

    graph.nodes_count = graph.nodes.size();

    // Build adjacency matrix
    graph.BuildAdjacencyMatrix();

    // Build adjacency list
    graph.BuildAdjacencyList();

    // Initalize iterators
    graph.iterators.resize(graph.nodes_count);
    std::fill(graph.iterators.begin(), graph.iterators.end(), 0);

    graph.PrintLoops();
    graph.PrintNodes();
    graph.PrintRanges();

    // initialize blocks container
    std::vector<Block> blocks;

    // Launch the simulation
    graph.Simulator(blocks, 0);
    if (graph.risk_alert) {
      data[addr_gm] = {ConditionInfo(-1)};
      return;
    }

    LOG(INFO) << "Number of blocks created: " << blocks.size();

    if (blocks.size() > this->th_block) {
      LOG(INFO) << "Blocks threshold reached: " << th_protect;
      data[addr_gm] = {ConditionInfo(-1)};
      return;
    }

    // Run check function
    std::vector<bool> check = graph.CheckOverlap(blocks);
    int total_overlap_block = count(check.begin(), check.end(), true);
    LOG(INFO) << total_overlap_block << " protections are needed";

    // Get overlapped blocks per line
    for (int i = 0; i < static_cast<int>(check.size()); i++) {
      Block *block = &blocks[i];
      // for stats
      cond_info.blocks_logs[block->line].push_back(*block);
      if (check[i]) {
        cond_info.blocks[block->line].push_back(*block);
      }
    }

    data[addr_gm] = {cond_info};
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    std::set<std::string> attr_list = {"external_for_loop", "internal_for_loop", "intrin_args",
                                       "src_var_stride",    "dst_var_stride",    "gm_addr_rw"};

    if (op->attr_key == "gm_addr") {
      CHECK(op->value.as<Variable>());
      gm_addr = op->value.as<Variable>();
      auto stmt = IRMutator::Mutate(op->body);
      gm_addr = nullptr;
      return stmt;
    } else if (attr_list.count(op->attr_key)) {
      auto stmt = IRMutator::Mutate(op->body);
      return stmt;
    }

    if (op->attr_key == "overlap_optimize") {
      auto it = std::find_if(data.begin(), data.end(),
                             [&](const std::pair<const Variable *, ConditionInfo> &i) { return gm_addr == i.first; });
      if (it == data.end()) {
        GetInfoForCondition(gm_addr);
      }
      cover_case = true;
      auto stmt = IRMutator::Mutate(op->body);
      cover_case = false;
      return stmt;
    } else if (op->attr_key == "thread_extent") {
      thread_extent = op->value;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (cover_case) {
      static_cast<void>(IRMutator::Mutate_(op, s));
      CHECK(gm_addr);

      if (data[gm_addr].current_ubuf_line == -1) {
        LOG(INFO) << "they protected: all blocks";
        LOG(INFO) << "we protected: all blocks";
        return IfThenElse::make(const_true(), op->then_case, op->else_case);
      }

      LOG(INFO) << "they protected: "
                << data[gm_addr].blocks_logs[data[gm_addr].position[data[gm_addr].current_ubuf_line]].size()
                << " blocks";

      if (data[gm_addr].blocks[data[gm_addr].position[data[gm_addr].current_ubuf_line]].empty()) {
        Expr real_condition = const_false();
        LOG(INFO) << "we protected: 0 blocks";
        data[gm_addr].current_ubuf_line++;
        return IfThenElse::make(real_condition, op->then_case, op->else_case);
      }

      // 29760 --> 20209 --> 12765 --> 7620 --> 1120 --> 1024 --> 768 --> 512
      if (data[gm_addr].blocks[data[gm_addr].position[data[gm_addr].current_ubuf_line]].size() >= th_protect) {
        LOG(INFO) << "we protected: " << data[gm_addr].blocks_logs[data[gm_addr].current_ubuf_line].size() << " blocks";
        Expr real_condition = const_true();
        data[gm_addr].current_ubuf_line++;
        return IfThenElse::make(real_condition, op->then_case, op->else_case);
      }

      LOG(INFO) << "we protected: "
                << data[gm_addr].blocks[data[gm_addr].position[data[gm_addr].current_ubuf_line]].size() << " blocks";

      int lines_count = data[gm_addr].blocks.size();

      for (int i = 0; i < lines_count; i++) {
        LOG(INFO) << "Line " << i << ": " << data[gm_addr].blocks[i].size() << " blocks protected";
      }

      Expr real_condition = const_true();

      std::vector<int> blocks_per_line;
      blocks_per_line.resize(lines_count);
      std::fill(blocks_per_line.begin(), blocks_per_line.end(), 0);

      for (auto &block : data[gm_addr].blocks[data[gm_addr].position[data[gm_addr].current_ubuf_line]]) {
        if (blocks_per_line[data[gm_addr].position[data[gm_addr].current_ubuf_line]] > 0) {
          Expr orrealcondition = const_true();
          for (auto key_value : block.iterators) {
            auto subcondition = EQ::make(key_value.first, key_value.second);
            orrealcondition = And::make(orrealcondition, subcondition);
          }
          real_condition = Or::make(real_condition, orrealcondition);
        } else {
          for (auto key_value : block.iterators) {
            auto subcondition = EQ::make(Expr(key_value.first), key_value.second);
            real_condition = And::make(real_condition, subcondition);
          }
        }
        blocks_per_line[data[gm_addr].position[data[gm_addr].current_ubuf_line]]++;
      }
      data[gm_addr].current_ubuf_line++;
      return IfThenElse::make(real_condition, op->then_case, op->else_case);
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  Expr thread_extent = 1;
  bool cover_case{false};
  const Variable *gm_addr{nullptr};
  std::unordered_map<const Variable *, std::vector<DMAInfo>> infos;
  std::unordered_map<const Variable *, ConditionInfo> data;
  bool risk_alert{false};
  Expr thread_var = 0;
  size_t th_block = 0;
  size_t th_protect = 0;
};

Stmt CoverProtection(Stmt stmt, size_t th_block, size_t th_protect) {
  LOG(INFO) << "BEGIN_PASS";
  auto gather = CoverProtectGather();
  gather.Visit(stmt);
  stmt = CoverProtector(gather.infos, gather.risk_rwalert, gather.thread_var, th_block, th_protect).Mutate(stmt);
  LOG(INFO) << "END_PASS";
  return stmt;
}
};  // namespace ir
}  // namespace akg
