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

#include <utility>
#include <vector>
#include <map>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm.h>
#include <string>

#include "pass/utils.h"
#include "src/common/util.h"

namespace akg {
namespace ir {

class SwizzleFinder : public IRVisitor {
 public:
  SwizzleFinder() = default;

  ~SwizzleFinder() override = default;

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::thread_extent) {
      if (auto value = op->value.as<IntImm>()) {
        std::string name = op->node.as<IterVarNode>()->var->name_hint;
        LOG(DEBUG) << "Thread extent (" << name << ") : " << value->value;
        thread_extent[name] = value->value;
      }
      IRVisitor::Visit_(op);
    } else if (op->attr_key == air::ir::attr::realize_scope) {
      LOG(WARNING) << "Realize storage scope not implemented in swizzle pass (may not work as expected) : "
                   << op->value.as<StringImm>()->value;
      Visit(op->body);
    } else if (op->attr_key == "pragma_swizzle") {
      LOG(DEBUG) << "Pragma swizzle";
      pragma_swizzle = true;
      Visit(op->body);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const For *op) final {
    swizzle_length = GetExtent(op);
    if (swizzle_length == 2 || swizzle_length == 4) {
      LOG(INFO) << "Swizzle for loop ? var : " << op->loop_var->name_hint;

      loop_var = op->loop_var;
      if (pragma_swizzle) {
        swizzlable = true;
        swizzle_candidate = true;
      }
      loop_loads = {};
      loop_stores = {};
      Visit(op->body);
      if (!swizzle_candidate) {
        // loop contains another loop
        LOG(DEBUG) << op->loop_var->name_hint << " not swizzlable : loop contains another loop";
        swizzlable = false;
        return;
      }
      pragma_swizzle = false;
      swizzle_candidate = false;
      // get all load and store from this loop and test if their range is 2, 4 or 0 (const)
      for (auto l : loop_loads) {
        int ext = load_indexes[l].second - load_indexes[l].first;
        if (ext != (int)swizzle_length && ext != 0) {
          swizzlable = false;
          LOG(DEBUG) << op->loop_var->name_hint
                     << " not swizzlable : range of load is not 0, 2 or 4 : " << ExprToString(load_indexes[l].first);
          break;
        } else if (ext == 0) {
          // do not swizzle variables that are constant inside loop
          if (swizzle_temp_vars.count(l->buffer_var->name_hint) > 0 &&
              set_temp_vars.count(l->buffer_var->name_hint) == 0) {
            swizzle_temp_vars.erase(l->buffer_var->name_hint);
            temp_vars.insert(l->buffer_var->name_hint);
          }
        }
      }
      for (auto s : loop_stores) {
        int ext = store_indexes[s].second - store_indexes[s].first;
        if (ext != (int)swizzle_length && ext != 0) {
          swizzlable = false;
          LOG(DEBUG) << op->loop_var->name_hint
                     << " not swizzlable : range of store is not 0, 2 or 4 : " << ExprToString(store_indexes[s].first);
          break;
        } else if (ext == 0) {
          // cannot swizzle if variable is an argument with range 0 (disable loop swizzle for safety)
          if (swizzle_temp_vars.count(s->buffer_var->name_hint) == 0) {
            swizzlable = false;
            LOG(DEBUG) << op->loop_var->name_hint << " not swizzlable : range of store is 0 on variable : "
                       << ExprToString(store_indexes[s].first);
            break;
          }

          // check if store value contains var
          // Warning : this check does not verify loop variables that are not in a load
          if (!store_loads[s].empty() && swizzle_stores.count(s) == 0) {
            // Variable value changes at each loop iteration
            swizzle_stores.insert(s);
          } else {
            // remove from swizzle_temp_vars
            swizzle_temp_vars.erase(s->buffer_var->name_hint);
            temp_vars.insert(s->buffer_var->name_hint);
          }
        }
      }
      if (swizzlable) {
        swizzle_loops.push_back(op);
      }

    } else {
      Visit(op->body);
    }
  }

  // check potential temp variable to convert, making sure it only contains the loop var
  template <typename T>
  void checkSwizzleVar(const T *op, air::DataType t) {
    if (swizzle_temp_vars.count(op->buffer_var->name_hint) > 0) {
      auto *v = op->index.template as<Variable>();
      auto *i = op->index.template as<IntImm>();
      if (((v == nullptr || v->name_hint != loop_var->name_hint) && (i == nullptr || i->value != 0)) ||
          !((t.is_float() || t.is_int()) && t.bits() <= 32)) {
        // this temp var does not look like a swizzle var, treat it as any regular temp var
        LOG(DEBUG) << "Irregular potential swizzle var : " << op->buffer_var->name_hint;
        temp_vars.insert(op->buffer_var->name_hint);
        swizzle_temp_vars.erase(op->buffer_var->name_hint);
      } else {
        if (i && i->value == 0) {
          if (var_size.find(op->buffer_var->name_hint) == var_size.end() ||
              var_size[op->buffer_var->name_hint] < (int)swizzle_length) {
            var_size[op->buffer_var->name_hint] = swizzle_length;
          }
        }
      }
    }
  }

  void Visit_(const Load *op) final {
    if (swizzle_candidate && swizzlable) {
      LOG(DEBUG) << "Load : " << op->buffer_var->name_hint << " index : " << ExprToString(op->index);
      loop_loads.push_back(op);
      compute_var = true;
      current_min = 1;
      current_max = 1;

      checkSwizzleVar(op, op->type);

      IRVisitor::Visit(op->index);
      // add this load to array
      compute_var = false;
      if (current_min > current_max) std::swap(current_min, current_max);
      load_indexes.insert(std::make_pair(op, std::make_pair(current_min, current_max)));
      if (current_store != nullptr &&
          ((contains_iterator && (op->type.is_float() || op->type.is_int()) && op->type.bits() <= 32) ||
           swizzle_temp_vars.count(op->buffer_var->name_hint) > 0)) {
        LOG(DEBUG) << "Load contains iterator or swizzle temp var";
        store_loads[current_store].insert(op);
      }
      contains_iterator = false;
      LOG(DEBUG) << "End Load : " << op->buffer_var->name_hint << " range estimation : " << current_max - current_min;
      IRVisitor::Visit(op->predicate);
    }
    IRVisitor::Visit_(op);
  }

  // visit each operator for each store and load
  // compute value for each of its elements (a, b or array)
  // add map<Store, pair> to evaluate extent inside each load/store
  void Visit_(const Store *op) final {
    if (swizzle_candidate && swizzlable) {
      LOG(DEBUG) << "Store : " << op->buffer_var->name_hint << " index : " << ExprToString(op->index)
                 << "     value : " << ExprToString(op->value);
      loop_stores.push_back(op);
      compute_var = true;
      contains_iterator = false;
      current_store = op;
      store_loads[op] = {};
      current_min = 1;
      current_max = 1;
      checkSwizzleVar(op, op->value.type());
      if (swizzle_temp_vars.count(op->buffer_var->name_hint) > 0) {
        set_temp_vars.insert(op->buffer_var->name_hint);
      }

      IRVisitor::Visit(op->index);
      // check for single Load (allows inplace swizzle)
      const Load *ld = op->value.as<Load>();
      if (ld && ld->type == op->value.type() && (ld->type.is_float() || ld->type.is_int()) && ld->type.bits() <= 32) {
        LOG(DEBUG) << "Single store   Value : " << op->value << std::endl
                   << "ld : " << ld->buffer_var << "[" << ld->index << "]";
        single_stores.insert(current_store);
      }
      // add this store to array
      compute_var = false;
      if (current_min > current_max) std::swap(current_min, current_max);
      store_indexes.insert(std::make_pair(op, std::make_pair(current_min, current_max)));
      if (contains_iterator && (op->value.type().is_float() || op->value.type().is_int()) &&
          op->value.type().bits() <= 32) {
        swizzle_stores.insert(op);
      }
      contains_iterator = false;

      LOG(DEBUG) << "End Store index : " << op->buffer_var->name_hint
                 << " range estimation : " << current_max - current_min;
    }
    IRVisitor::Visit_(op);
    current_store = nullptr;
    LOG(DEBUG) << "End Store " << op->buffer_var->name_hint;
  }

  void Visit_(const Allocate *op) final {
    int size = op->constant_allocation_size();
    LOG(DEBUG) << "Allocate : " << op->buffer_var->name_hint << " size " << size;
    if (size == 1 || size == 2 || size == 4) {
      swizzle_temp_vars.insert(op->buffer_var->name_hint);
      var_size[op->buffer_var->name_hint] = size;
    } else {
      temp_vars.insert(op->buffer_var->name_hint);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const FloatImm *op) final {
    if (compute_var) {
      current_min = (int)op->value;
      current_max = current_min;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const IntImm *op) final {
    if (compute_var) {
      current_min = op->value;
      current_max = current_min;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Variable *op) final {
    if (swizzle_candidate && swizzlable && compute_var) {
      if (op->name_hint == loop_var->name_hint) {
        contains_iterator = true;
        current_min = 0;
        current_max = (int)swizzle_length;
      } else {
        auto it = thread_extent.find(op->name_hint);
        if (it != thread_extent.end()) {
          current_min = it->second;
          current_max = current_min;
        } else {
          // unknown variable value, consider it constant
          current_min = 1;
          current_max = 1;
        }
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Mul *op) final {
    if (swizzle_candidate && swizzlable) {
      if (compute_var) {
        Visit(op->a);
        int tmp_min = current_min, tmp_max = current_max;

        Visit(op->b);

        int tmp = std::min(std::min(current_min * tmp_min, current_max * tmp_min),
                           std::min(current_min * tmp_max, current_max * tmp_max));
        current_max = std::max(std::max(current_min * tmp_min, current_max * tmp_min),
                               std::max(current_min * tmp_max, current_max * tmp_max));
        current_min = tmp;

        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Add *op) final {
    if (swizzle_candidate && swizzlable && compute_var) {
      Visit(op->a);
      int tmp_min = current_min, tmp_max = current_max;

      Visit(op->b);
      int tmp = std::min(std::min(current_min + tmp_min, current_max + tmp_min),
                         std::min(current_min + tmp_max, current_max + tmp_max));
      current_max = std::max(std::max(current_min + tmp_min, current_max + tmp_min),
                             std::max(current_min + tmp_max, current_max + tmp_max));
      current_min = tmp;
      return;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Sub *op) final {
    if (swizzle_candidate && swizzlable && compute_var) {
      Visit(op->b);
      int tmp_min = current_min, tmp_max = current_max;

      Visit(op->a);
      int tmp = std::min(std::min(current_min - tmp_min, current_max - tmp_min),
                         std::min(current_min - tmp_max, current_max - tmp_max));
      current_max = std::max(std::max(current_min - tmp_min, current_max - tmp_min),
                             std::max(current_min - tmp_max, current_max - tmp_max));
      current_min = tmp;
      return;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Div *op) final {
    if (swizzle_candidate && swizzlable) {
      if (compute_var) {
        Visit(op->a);
        int tmp_min = current_min, tmp_max = current_max;

        Visit(op->b);

        if (current_min == 0 || current_max == 0) {
          swizzlable = false;
          LOG(WARNING) << "Possible division by zero detected in " << getDebugInfo(op);
        }
        int tmp = std::min(std::min(tmp_min / current_min, tmp_min / current_max),
                           std::min(tmp_max / current_min, tmp_max / current_max));
        current_max = std::max(std::max(tmp_min / current_min, tmp_min / current_max),
                               std::max(tmp_max / current_min, tmp_max / current_max));
        current_min = tmp;

        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const IfThenElse *op) final {
    if (swizzle_candidate) {
      swizzlable = false;
    }
    IRVisitor::Visit_(op);
  }

  // returns the extent of the loop if it's a constant integer, otherwise return -1
  static int GetExtent(const For *op) {
    // constant folding.
    Expr extent = Simplify(op->extent);
    const auto *v1 = extent.as<IntImm>();
    const auto *v2 = extent.as<UIntImm>();
    int value = -1;
    if (v1 != nullptr) {
      value = static_cast<int>(v1->value);
    }
    if (v2 != nullptr) {
      value = static_cast<int>(v2->value);
    }
    return value;
  }

  std::set<const Store *> single_stores{};
  std::vector<const For *> swizzle_loops{};
  std::set<std::basic_string<char>> temp_vars{};
  std::set<std::basic_string<char>> swizzle_temp_vars{};
  std::set<const Load *> input_arguments{};
  std::set<const Store *> output_arguments{};
  std::set<const Store *> swizzle_stores{};
  std::set<const Load *> swizzle_loads{};
  bool swizzlable{false};
  std::map<const Store *, std::set<const Load *>> store_loads;
  std::map<std::basic_string<char>, int> var_size{};

 private:
  // print a nice representation of what is happening inside the code
  template <typename T>
  std::string getDebugInfo(T *op) {
    Expr a = (Expr)(op->a);
    Expr b = (Expr)(op->b);
    std::string a_str, b_str;
    if (!a.defined()) {
      a_str = a->GetTypeKey();
    } else {
      a_str = "(" + a->GetTypeKey() + ") " + ExprToString(a);
    }
    if (!b.defined()) {
      b_str = b->GetTypeKey();
    } else {
      b_str = "(" + b->GetTypeKey() + ") " + ExprToString(b);
    }
    return a_str + " " + b_str;
  }

  const Store *current_store{};
  bool compute_var = false;
  int current_min = 0;
  int current_max = 0;
  bool swizzle_candidate{false};
  bool contains_iterator{false};
  bool pragma_swizzle{false};
  unsigned int swizzle_length{0};
  Var loop_var;
  // temp vars that are set inside a loop
  std::set<std::basic_string<char>> set_temp_vars{};
  std::vector<const Load *> loop_loads;
  std::vector<const Store *> loop_stores;
  std::unordered_map<const Load *, std::pair<int, int>> load_indexes{};
  std::unordered_map<const Store *, std::pair<int, int>> store_indexes{};
  std::map<std::string, int64_t> thread_extent = {{"blockIdx.x", 0}, {"threadIdx.x", 0}};
};

class Swizzle : public IRMutator {
 public:
  explicit Swizzle(std::basic_string<char> name) : finder() { kernel_name = std::move(name); };

  ~Swizzle() override = default;

  Stmt VisitAndMutate(Stmt stmt) {
    LOG(DEBUG) << "Visit statement";
    finder.Visit(stmt);
    if (!finder.swizzle_loops.empty()) {
      auto ret = Mutate(stmt);
      if (!ret.same_as(stmt)) {
        LOG(INFO) << "Total swizzled loops for " << kernel_name << " : " << finder.swizzle_loops.size();
        return ret;
      }
    }
    LOG(INFO) << "Total swizzled loops for " << kernel_name << " : 0";
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto f = std::find(std::begin(finder.swizzle_loops), std::end(finder.swizzle_loops), op);
    if (f != std::end(finder.swizzle_loops)) {
      swizzling = true;
      LOG(DEBUG) << "Swizzle " << op->loop_var->name_hint;
      Stmt s2 = swizzle_loop(op, s);
      swizzling = false;
      return s2;
    }

    auto body = Mutate(op->body);
    // auto unroll loops
    ForType t = op->for_type;
    int ext = SwizzleFinder::GetExtent(op);
    if (ext > 0 && t == ForType::Serial) t = ForType::Unrolled;
    return For::make(op->loop_var, op->min, op->extent, t, op->device_api, body);
  }

  // modify loop to apply swizzle
  Stmt swizzle_loop(const For *op, const Stmt &s) {
    auto min = op->min.as<IntImm>();
    loop_extent = SwizzleFinder::GetExtent(op);
    if (min && loop_extent > 0) {
      currentLoop = op;
      auto body = Mutate(op->body);
      // remove and unroll the loop
      return For::make(op->loop_var, op->min, op->extent, ForType::Swizzled, op->device_api, body);
    }

    // something wrong happened during extent evaluation, we do not mutate
    LOG(WARNING) << "Could not mutate loop (invalid loop extent)";
    ForType t = op->for_type;
    if (loop_extent > 0 && t == ForType::Serial) t = ForType::Unrolled;
    return For::make(op->loop_var, op->min, op->extent, t, op->device_api, op->body);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (swizzling) {
      loop_extent = SwizzleFinder::GetExtent(currentLoop);
      if (std::find(finder.single_stores.begin(), finder.single_stores.end(), op) != finder.single_stores.end()) {
        // this store has only one load attached, we can swizzle in place
        LOG(DEBUG) << "Swizzle in place " << op->buffer_var->name_hint;

        // modify store value
        Expr value;
        auto buffer_var = op->buffer_var;
        auto v = air::ir::Substitute(op->value, {{Var{currentLoop->loop_var}, make_const(Int(32), 0)}});
        v = Simplify_cce(v);
        Array<Expr> value_args{make_const(Int(32), loop_extent), v};
        // check if variable is a temp var or an output var
        if (finder.swizzle_temp_vars.count(op->buffer_var->name_hint) > 0) {
          // (swizzle temp var) sw_var ... ldg
          // check temp variable is declared
          CHECK(replace_vars.find(op->buffer_var->name_hint) != replace_vars.end());
          buffer_var = replace_vars[op->buffer_var->name_hint];
          value = Call::make(op->value.type(), Call::ldg, value_args, Call::Intrinsic);
          Stmt s2 = Store::make(buffer_var, value, op->index, op->predicate);
          return AttrStmt::make(s, "simple_store", Expr(0), s2);
        } else if (finder.temp_vars.count(op->buffer_var->name_hint) > 0) {
          // (temp var) reinterpret cast ... ldg
          value = Call::make(op->value.type(), Call::ldg, value_args, Call::Intrinsic);
        } else {
          // (output var) reinterpret cast ... reinterpret cast
          value = Call::make(op->value.type(), Call::reinterpret_cast_op, value_args, Call::Intrinsic);
        }

        auto index = air::ir::Substitute(op->index, {{Var{currentLoop->loop_var}, make_const(Int(32), 0)}});
        index = Simplify_cce(index);
        Stmt s2 = Store::make(buffer_var, value, index, op->predicate);
        return AttrStmt::make(s, "reinterpret_store", Expr(loop_extent), s2);
      } else {
        // store with != 1 load
        Expr value = Mutate(op->value);
        LOG(DEBUG) << "check type: " << op->value.type() << " " << op->value;
        CHECK(op->value.type().is_float() || op->value.type().is_int());
        CHECK_LE(op->value.type().bits(), 32);
        if (std::find(finder.swizzle_stores.begin(), finder.swizzle_stores.end(), op) != finder.swizzle_stores.end()) {
          LOG(DEBUG) << "Mutate Store : index contains loop var " << currentLoop->loop_var->name_hint << std::endl
                     << "Value :" << op->value;

          Stmt new_stmt, end_stmt;
          Expr index;
          Array<Expr> value_args;

          air::DataType t;
          Var new_var;
          if (replace_vars.find(op->buffer_var->name_hint) != replace_vars.end()) {
            new_var = replace_vars[op->buffer_var->name_hint];
            t = new_var.type();
          } else {
            if (op->value.type().is_int()) {
              t = Int(op->value.type().bits(), loop_extent);
            } else {
              // Float(16, 4) -> half4
              // 1. Generate the right DataType   -> half4
              t = Float(op->value.type().bits(), loop_extent);
            }
            new_var = Variable::make(t, "sw_" + op->buffer_var->name_hint);
          }

          LOG(DEBUG) << "(Vector store) replace previous buffer var : " << op->buffer_var
                     << " type : " << op->buffer_var->type << " with " << new_var << " type : " << new_var->type;
          Expr new_value = Broadcast::make(value, loop_extent);
          index = Ramp::make(0, 1, loop_extent);
          Expr predicate = Broadcast::make(Expr(1), loop_extent);

          new_stmt = Store::make(new_var, new_value, index, predicate);
          new_stmt = AttrStmt::make(s, "vec_store", Expr(currentLoop->loop_var), new_stmt);

          // do not reinterpret cast if var is in swizzle_temp_vars
          if (finder.swizzle_temp_vars.count(op->buffer_var->name_hint) == 0) {
            // last statement (store) : set value to initial value
            index = air::ir::Substitute(op->index, {{Var{currentLoop->loop_var}, make_const(Int(32), 0)}});
            index = Simplify_cce(index);
            value_args = {make_const(Int(32), 0), new_var};
            Expr reinterpret = Call::make(op->value.type(), Call::reinterpret_cast_op, value_args, Call::Intrinsic);
            // replace with new_var to remove second reinterpret (produces type error)
            end_stmt = Store::make(op->buffer_var, reinterpret, index, op->predicate);
            end_stmt = AttrStmt::make(new_stmt, "reinterpret_store", Expr(loop_extent), end_stmt);
          }

          // declare load variables
          LOG(DEBUG) << "Mutate Load : index contains loop var " << currentLoop->loop_var->name_hint
                     << ", Loop extent : " << loop_extent;
          for (auto ld : finder.store_loads[op]) {
            if (std::find(finder.temp_vars.begin(), finder.temp_vars.end(), ld->buffer_var->name_hint) ==
                finder.temp_vars.end()) {
              // declare swizzle variable

              Var new_var2;
              air::DataType t2;
              if (replace_vars.find(ld->buffer_var->name_hint) == replace_vars.end()) {
                if (ld->buffer_var->name_hint == op->buffer_var->name_hint) {
                  // this variable is both stored AND loaded for the first time
                  t2 = t;
                  new_var2 = new_var;
                } else {
                  if (ld->type.is_int()) {
                    t2 = Int(ld->type.bits(), loop_extent);
                  } else {
                    // Float(16, 4) -> half4
                    // 1. Generate the right DataType   -> half4
                    t2 = Float(ld->type.bits(), loop_extent);
                  }
                  new_var2 = Variable::make(t2, "sw_" + ld->buffer_var->name_hint);
                }
                replace_vars[ld->buffer_var->name_hint] = new_var2;
                declared.insert(ld->buffer_var->name_hint);

                // 2. declaration  half4 sw_T_where = __ldg( ... )
                index = air::ir::Substitute(ld->index, {{Var{currentLoop->loop_var}, make_const(Int(32), 0)}});
                index = Simplify_cce(index);
                Expr expr_ld = Load::make(ld->type, ld->buffer_var, index, ld->predicate);
                value_args = {make_const(Int(32), loop_extent), expr_ld};
                Expr ldg_input = Call::make(t2, Call::ldg, value_args, Call::Intrinsic);

                LOG(DEBUG) << "(Vector load) replace previous buffer var : " << ld->buffer_var
                           << " type : " << ld->buffer_var->type << " with " << new_var2
                           << " type : " << new_var2->type;

                new_stmt = LetStmt::make(new_var2, ldg_input, new_stmt);
                new_stmt = AttrStmt::make(s, "vec_load", ld->buffer_var, new_stmt);
              } else {
                new_stmt = AttrStmt::make(s, "vec_load", ld->buffer_var, new_stmt);
              }
            }
          }

          // declare store variable
          if (std::find(declared.begin(), declared.end(), op->buffer_var->name_hint) == declared.end()) {
            // declare swizzle variable
            declared.insert(op->buffer_var->name_hint);
            new_stmt = LetStmt::make(new_var, Broadcast::make(make_const(op->value.type(), 0), loop_extent), new_stmt);
            new_stmt = AttrStmt::make(s, "no_init_value", Expr(0), new_stmt);
          }

          if (replace_vars.find(op->buffer_var->name_hint) == replace_vars.end()) {
            replace_vars[op->buffer_var->name_hint] = new_var;
          }

          if (finder.swizzle_temp_vars.count(op->buffer_var->name_hint) == 0) {
            Stmt new_block = Block::make(new_stmt, end_stmt);
            return new_block;
          } else {
            return new_stmt;
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    if (finder.swizzle_temp_vars.count(op->buffer_var->name_hint) > 0) {
      // replace var with its swizzle counterpart
      int size = finder.var_size[op->buffer_var->name_hint];
      air::DataType t;
      if (op->type.is_int()) {
        t = Int(op->type.bits(), size);
      } else {
        t = Float(op->type.bits(), size);
      }
      Var new_var = Variable::make(t, "sw_" + op->buffer_var->name_hint);

      LOG(DEBUG) << "Allocate : replace previous buffer var : " << op->buffer_var << " type : " << op->type << " with "
                 << new_var << " type : " << new_var->type;

      replace_vars[op->buffer_var->name_hint] = new_var;
      declared.insert(op->buffer_var->name_hint);
      Stmt body = Mutate(op->body);
      Stmt new_stmt;
      Stmt let_stmt = LetStmt::make(new_var, Broadcast::make(make_const(op->type, 0), size), body);
      Stmt attr = AttrStmt::make(s, "no_init_value", Expr(0), let_stmt);
      Stmt new_allocate = Allocate::make(op->buffer_var, op->type, op->extents, const_false(), attr);

      return new_allocate;  // Block::make(attr, attr1);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  //  int nb_loops;
  SwizzleFinder finder;
  const For *currentLoop{};
  bool swizzling = false;
  int loop_extent{};
  std::set<std::basic_string<char>> declared{};
  std::unordered_map<std::basic_string<char>, Var> replace_vars{};
  std::basic_string<char> kernel_name;
};

static void ParseStringAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                            std::string *attr_to_set) {
  CHECK(attr_to_set != nullptr);
  if (attrs.count(attr_name) == 0) return;
  const NodeRef &e = attrs.at(attr_name);
  if (auto val = e.as<StringImm>()) {
    *attr_to_set = val->value;
  } else {
    LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as string";
  }
}

static void ParseIntAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, int *attr_to_set) {
  CHECK(attr_to_set != nullptr);
  if (attrs.count(attr_name) == 0) return;
  const NodeRef &e = attrs.at(attr_name);
  if (auto i = e.as<IntImm>()) {
    *attr_to_set = static_cast<int>(i->value);
  } else if (auto ui = e.as<UIntImm>()) {
    *attr_to_set = static_cast<int>(ui->value);
  } else {
    LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as integer";
  }
}

static void ParseBoolAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, bool *attr_to_set) {
  const int invalid_value = -1;
  int attr = invalid_value;
  ParseIntAttr(attrs, attr_name, &attr);
  if (attr != invalid_value) {
    CHECK(attr == 0 || attr == 1) << "Bool attribute " << attr_name << " must be 0 or 1, but found "
                                  << attrs.at(attr_name);
    *attr_to_set = static_cast<bool>(attr);
  }
}

Stmt SwizzleGPU(const Stmt &stmt, const Map<std::string, NodeRef> &attrs) {
  bool disable_swizzle = false;
  ParseBoolAttr(attrs, "disable_swizzle", &disable_swizzle);
  if (const char *env_p = std::getenv("MS_AKG_DISABLE_SWIZZLE"))
    if (!strcmp(env_p, "1")) disable_swizzle = true;
  if (disable_swizzle) {
    LOG(INFO) << "SwizzleGPU pass disabled";
    return stmt;
  }
  std::string kernel_name_;
  ParseStringAttr(attrs, "kernel_name", &kernel_name_);
  if (kernel_name_.empty())
    LOG(WARNING) << "Kernel name not found !";
  else
    LOG(INFO) << "BEGIN_PASS SwizzleGPU on " << kernel_name_;
  auto sw = Swizzle(kernel_name_);
  Stmt s = sw.VisitAndMutate(stmt);

  LOG(INFO) << "END_PASS";
  return s;
}

}  // namespace ir
}  // namespace akg
