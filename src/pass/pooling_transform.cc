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
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>

#include <string>
#include <utility>
#include <tuple>

#include "pass/ir_util.h"
#include "ir_pass.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
enum im2colArgIndex { kBatchIndex = 0, kC1Index, kKHIndex, kKWIndex, kOHIndex, kOWIndex, kC0Index, kIm2colArgSize };
enum LOAD3DCallArgs {
  idxTensorIdx = 0,
  idxPadTop,
  idxPadBottom,
  idxPadLeft,
  idxPadRight,
  idxFmH,
  idxFmW,
  idxStrideH,
  idxStrideW,
  idxFilterH,
  idxFilterW,
  idxDilationH,
  idxDilationW,
  idxRepeatMode,
  idxJmpOffset
};
enum firstMaxMaskCallArgs { fmmBatch = 0, fmmC1, fmmKh, fmmKw, fmmOh, fmmOw, fmmC0, fmmSize };
enum ReorderMode { modeOne = 0, modeTwo };

/* ***********************************************************
 * Fuse tiled im2col ir together
 * last pass result:
 *  for (cc8, 0, 2) {
      for (cc9, 0, 56) {
        for (cc10, 0, 3) {
          for (c15, 0, 3) {
            for (cc11, 0, 16) {
              im2col_row_major_local_UB(0, 0, cc10, c15, cc8, cc9, cc11) =
              load3d_l1_ub(select(((3 < ((cc8*2) + cc1    0)) || (111 < ((cc9*2) + c15))),
              -65504h, input_1_local_L1(0, 0, ((cc8*2) + cc10), ((cc9*2) + c15), cc11)), 0,
              1, 0, 1, 112, 112, 2, 2, 3, 3, 1, 1, 1, 1):float16:PI
            }
          }
        }
      }
    }
    for (cc8, 0, 2) {
      for (cc9, 0, 56) {
        for (cc11, 0, 16) {
          pooling_max_local_UB(0, 0, cc8, cc9, cc11) = -65504h
        }
      }
    }
    for (cc8, 0, 2) {
      for (cc9, 0, 56) {
        for (cc10, 0, 3) {
          for (cc11, 0, 16) {
            for (cc12, 0, 3) {
              pooling_max_local_UB(0, 0, cc8, cc9, cc11) =
              max(pooling_max_local_UB(0, 0, cc8, cc9, cc11),
              im2col_row_major_local_UB(0, 0, cc10, cc12, cc8, cc9, cc11))
            }
          }
        }
      }
    }
 *
 * current class result
 * for (cc8, 0, 2) {
     for (cc9, 0, 56) {
       for (cc11, 0, 16) {
         pooling_max_local_UB(0, 0, cc8, cc9, cc11) = -65504h
       }
     }
   }
   for (cc10, 0, 3) {
     for (c15, 0, 3) {
       // attr [{}] pragma_load3d = 1
       for (cc8, 0, 2) {
         for (cc9, 0, 56) {
           for (cc11, 0, 16) {
             im2col_row_major_local_UB(0, 0, cc10, c15, cc8, cc9, cc11) =
             load3d_l1_ub(select(((3 < ((cc8*2) + cc10)) || (111 < ((cc9*2) + c15))), -65504h,
             input_1_local_L1(0, 0, ((cc8*2) + cc10), ((cc9*2) + c15), cc11)),
             0, 1, 0, 1, 112, 112, 2, 2, 3, 3, 1, 1, 1, 1):float16:PI
           }
         }
       }
       for (cc8, 0, 2) {
         for (cc9, 0, 56) {
           for (cc11, 0, 16) {
             pooling_max_local_UB(0, 0, cc8, cc9, cc11) =
             max(pooling_max_local_UB(0, 0, cc8, cc9, cc11),
             im2col_row_major_local_UB(0, 0, cc10, c15, cc8, cc9, cc11))
           }
         }
       }
     }
   }

 *
 *
 * ***********************************************************/

class PoolingFusion : public IRMutator {
 public:
  explicit PoolingFusion(const std::string &name, bool is_dynamic) : call_name_(name), is_dynamic_(is_dynamic) {}
  ~PoolingFusion() override = default;

  Stmt Fusion(const Stmt &stmt) {
    Stmt s = this->Mutate(stmt);
    return s;
  }

  bool PatternCheck(const AttrStmt *op) {
    if (op->attr_key != ktvm::ir::attr::realize_scope) return false;
    const auto str = op->value.as<StringImm>();
    if (str == nullptr) return false;
    if (str->value != "local.L1") return false;
    if (!op->node.as<PlaceholderOpNode>()) return false;
    return true;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    Stmt res = IRMutator::Mutate_(op, s);
    if (PatternCheck(op)) {
      const auto op_node = op->node.as<PlaceholderOpNode>();
      CHECK(op_node);
      std::size_t pos = op_node->name.find("_local_L1");
      if (pos != std::string::npos) {
        std::string name = op_node->name.substr(0, pos);
        std::unordered_map<std::string, NodeRef> attrs;
        attrs["feature"] = Expr(name);
        res = AttrStmt::make(Map<std::string, NodeRef>(attrs.begin(), attrs.end()), "pragma_attrs", Expr(1), res);
      }
    }
    return res;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (!is_dynamic_) {
      const Call *call = op->value.as<Call>();
      if (call && call->name == call_name_) {
        is_load3d_ = true;
        CHECK_EQ(op->args.size(), static_cast<size_t>(im2colArgIndex::kIm2colArgSize));
        kh_var_ = op->args[static_cast<size_t>(im2colArgIndex::kKHIndex)].as<Variable>();
        kw_var_ = op->args[static_cast<size_t>(im2colArgIndex::kKWIndex)].as<Variable>();
        kh_expr_ = op->args[static_cast<size_t>(im2colArgIndex::kKHIndex)];
        kw_expr_ = op->args[static_cast<size_t>(im2colArgIndex::kKWIndex)];
        img2col_tensor_ = Downcast<Operation>(op->func).output(op->value_index);
      } else if (is_load3d_) {
        bool is_use = false;
        const Call *t_call = nullptr;
        std::tie(is_use, t_call) = GetCallIfUseTensor(op, img2col_tensor_);
        if (is_use) {
          detect_pool_compute_ = true;
          CHECK(t_call);
          CHECK_EQ(t_call->args.size(), static_cast<size_t>(im2colArgIndex::kIm2colArgSize));
          max_kh_ = t_call->args[static_cast<size_t>(im2colArgIndex::kKHIndex)].as<Variable>();
          max_kw_ = t_call->args[static_cast<size_t>(im2colArgIndex::kKWIndex)].as<Variable>();
          std::unordered_map<const Variable *, Expr> value_map;
          value_map[max_kh_] = kh_expr_;
          value_map[max_kw_] = kw_expr_;
          Stmt res = Substitute(s, value_map);
          return res;
        }
      }
    } else {
      const Call *call = op->value.as<Call>();
      const Max *max_compute = op->value.as<Max>();
      if (call && call->name == call_name_) {
        is_load3d_ = true;
        CHECK_EQ(op->args.size(), static_cast<size_t>(im2colArgIndex::kIm2colArgSize));
        kh_var_ = op->args[static_cast<size_t>(im2colArgIndex::kKHIndex)].as<Variable>();
        kw_var_ = op->args[static_cast<size_t>(im2colArgIndex::kKWIndex)].as<Variable>();
        kh_expr_ = op->args[static_cast<size_t>(im2colArgIndex::kKHIndex)];
        kw_expr_ = op->args[static_cast<size_t>(im2colArgIndex::kKWIndex)];
      } else if (is_load3d_ && max_compute != nullptr) {
        const Call *b_call = max_compute->b.as<Call>();
        CHECK(b_call);
        CHECK_EQ(b_call->args.size(), static_cast<size_t>(im2colArgIndex::kIm2colArgSize));
        max_kh_ = b_call->args[static_cast<size_t>(im2colArgIndex::kKHIndex)].as<Variable>();
        max_kw_ = b_call->args[static_cast<size_t>(im2colArgIndex::kKWIndex)].as<Variable>();
        std::unordered_map<const Variable *, Expr> value_map;
        value_map[max_kh_] = kh_expr_;
        value_map[max_kw_] = kw_expr_;
        Stmt res = Substitute(s, value_map);
        return res;
      } else if (is_load3d_ && isImm(op->value)) {
        is_max_init_ = true;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  bool ThreePartLoad3d(const Block *op) {
    std::set<std::string> call_set;
    PostOrderVisit(Block::make(op->first, op->rest), [&call_set](const NodeRef &node) {
      const Call *call = node.as<Call>();
      if (call != nullptr) {
        call_set.insert(call->name);
      }
      auto max = node.as<Max>();
      if (max != nullptr) {
        call_set.insert("max");
      }
    });

    if (call_set.count("max") > 0 && call_set.count("load3d_l1_ub") > 0) {
      return true;
    }
    return false;
  }

  Stmt FuseLoad3dAndPool(const Stmt &load3d, const Stmt &maxpooling) {
    Stmt res = load3d;
    if (kw_ != nullptr) {
      std::unordered_map<std::string, NodeRef> attrs;
      res = AttrStmt::make(Map<std::string, NodeRef>(attrs.begin(), attrs.end()), "pragma_load3d", Expr(1), res);
      res = Block::make(res, maxpooling);
      res = For::make(kw_->loop_var, kw_->min, kw_->extent, kw_->for_type, kw_->device_api, res);
    }
    if (kh_ != nullptr) {
      res = For::make(kh_->loop_var, kh_->min, kh_->extent, kh_->for_type, kh_->device_api, res);
    }
    return res;
  }

  Stmt FuseLoad3dAndMax(const Stmt &load3d, const Stmt &maxpooling) {
    Stmt res = load3d;
    if (kw_ != nullptr) {
      std::unordered_map<std::string, NodeRef> attrs;
      res = AttrStmt::make(Map<std::string, NodeRef>(attrs.begin(), attrs.end()), "pragma_load3d", Expr(1), res);
      res = Block::make(res, maxpooling);
      res = For::make(kw_->loop_var, kw_->min, kw_->extent, kw_->for_type, kw_->device_api, res);
    }
    if (kh_ != nullptr) {
      res = For::make(kh_->loop_var, kh_->min, kh_->extent, kh_->for_type, kh_->device_api, res);
    }
    return res;
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (!is_dynamic_) {
      Stmt first = this->Mutate(op->first);
      // load3d must be on the first part,
      // but pooling compute should be first or rest part.
      if (is_load3d_ && !info_cache_) {
        img2col_stmt_ = first;
        first = Evaluate::make(0);
        info_cache_ = true;
      } else if (is_load3d_ && detect_pool_compute_) {
        first = FuseLoad3dAndPool(img2col_stmt_, first);
        Init();
      }

      Stmt rest = this->Mutate(op->rest);
      if (is_load3d_ && detect_pool_compute_) {
        rest = FuseLoad3dAndPool(img2col_stmt_, rest);
        Init();
      }

      if (first.same_as(op->first) && rest.same_as(op->rest)) {
        return s;
      } else {
        if (first.same_as(Evaluate::make(0))) return rest;
        return Block::make(first, rest);
      }
    } else {
      if (op->first.get() != nullptr && op->rest.get() != nullptr && op->first.get()->GetTypeKey() == For::_type_key &&
          op->rest.get()->GetTypeKey() == Block::_type_key) {
        const auto rest = op->rest.as<Block>();
        CHECK(rest);
        if (rest->first.get()->GetTypeKey() == For::_type_key &&
            rest->rest.get()->GetTypeKey() == AttrStmt::_type_key) {
          Stmt first_for = Mutate(op->first);
          Stmt rest_for = Mutate(rest->first);
          if (is_load3d_) {
            first_for = FuseLoad3dAndMax(first_for, rest_for);
            Init();
            return Block::make(first_for, rest->rest);
          }
        } else if (ThreePartLoad3d(rest)) {
          Stmt first_for = Mutate(op->first);     // im2col
          Stmt rest_first = Mutate(rest->first);  // max init
          const auto rest_block = rest->rest.as<Block>();
          CHECK(rest_block);
          Stmt maxpooling = Mutate(rest_block->first);  // max compute
          if (is_load3d_ && is_max_init_) {
            first_for = FuseLoad3dAndMax(first_for, maxpooling);
            Init();
            first_for = Block::make(rest_first, first_for);
            return Block::make(first_for, rest_block->rest);
          }
        } else if (ThreePartLoad3d(op)) {
          Stmt im2col = Mutate(op->first);
          Stmt max_init = Mutate(rest->first);
          if (is_max_init_) {
            const auto rest_block = rest->rest.as<Block>();
            CHECK(rest_block);
            Stmt maxpooling = Mutate(rest_block->first);
            if (is_load3d_) {
              im2col = FuseLoad3dAndMax(im2col, maxpooling);
              Init();
              im2col = Block::make(max_init, im2col);
              return Block::make(im2col, rest_block->rest);
            }
          } else {
            Stmt maxpooling = max_init;
            if (is_load3d_) {
              im2col = FuseLoad3dAndMax(im2col, maxpooling);
              Init();
              return Block::make(im2col, rest->rest);
            }
          }
        }
      }
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Stmt res = Mutate(op->body);
    if (!is_load3d_) {
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, res);
    }

    if (op->loop_var.get() == kh_var_) {
      kh_ = op;
    } else if (op->loop_var.get() == kw_var_) {
      kw_ = op;
    }

    if (op != kh_ && op != kw_ && op->loop_var.get() != max_kh_ && op->loop_var.get() != max_kw_) {
      res = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, res);
    }

    return res;
  }

 private:
  std::pair<bool, const Call *> GetCallIfUseTensor(const Provide *op, const Tensor &tag) {
    bool is_use = false;
    const Call *consumer = nullptr;
    int consume_cnt = 0;
    PostOrderVisit(op->value, [&tag, &is_use, &consumer, &consume_cnt](const NodeRef &node) {
      auto call = node.as<Call>();
      if (call && call->call_type == Call::CallType::Halide &&
          Downcast<Operation>(call->func).output(call->value_index) == tag) {
        is_use = true;
        consumer = call;
        consume_cnt++;
      }
    });
    CHECK_LE(consume_cnt, 1);
    if (is_use) return std::make_pair(true, consumer);
    return std::make_pair(false, nullptr);
  }

  void Init() {
    is_load3d_ = false;
    is_max_init_ = false;
    info_cache_ = false;
    detect_pool_compute_ = false;
    kw_ = nullptr;
    kh_ = nullptr;
    max_kh_ = nullptr;
    max_kw_ = nullptr;
  }

  std::string call_name_;
  bool is_dynamic_{false};
  const For *kh_{nullptr};
  const For *kw_{nullptr};
  const Variable *max_kh_{nullptr};
  const Variable *max_kw_{nullptr};
  Expr kh_expr_;
  Expr kw_expr_;
  const Variable *kh_var_{nullptr};
  const Variable *kw_var_{nullptr};
  bool is_load3d_{false};
  bool is_max_init_{false};  // for dynamic
  bool info_cache_{false};
  Tensor img2col_tensor_;
  bool detect_pool_compute_{false};
  Stmt img2col_stmt_;
};

/* ****************************************************************
 * first mode
 * reorder ir from
 * for (cc8, 0, 3) {
     for (cc9, 0, 56) {
       for (cc10, 0, 3) {
         for (cc12, 0, 3) {
           for (cc11, 0, 16) {
             S_1
           }
           for (cc11, 0, 16) {
             S_2
           }
           for (cc11, 0, 16) {
             S_3
           }
         }
        }
       }
 --> to
 for (khw_var, 0, 8) {
   for (cc8, 0, 3) {
     for (cc9, 0, 56) {
       for (cc11, 0, 16) {
         S_1
   for (cc8, 0, 3) {
     for (cc9, 0, 56) {
       for (cc11, 0, 16) {
         S_2
   for (cc8, 0, 3) {
     for (cc9, 0, 56) {
       for (cc11, 0, 16) {
         S_3

 *
 * second mode
 * reorder ir from
  for (cc8, 0, 2) {
   for (c9, 0, 2) {
     for (c11, 0, 5) {
       for (c12, 0, 5) {
         for (c13, 0, 16) {
           S_1
         }}}}
   for (c9, 0, 2) {
     for (c11, 0, 5) {
       for (c12, 0, 5) {
         for (c13, 0, 16) {
           S_2
         }}}}
   for (c9, 0, 2) {
     for (c11, 0, 5) {
       for (c12, 0, 5) {
         for (c13, 0, 16) {
           S_3
         }}}}
 }
 to -->
 for (khw_var, 0, 3) {
   for (c11, 0, 5) {
     for (c12, 0, 5) {
       for (c13, 0, 16) {
         S_1
       }}}
   for (c11, 0, 5) {
     for (c12, 0, 5) {
       for (c13, 0, 16) {
         S_2
       }}}
   for (c11, 0, 5) {
     for (c12, 0, 5) {
       for (c13, 0, 16) {
         S_3
       }}}
 }
 * ***************************************************************/
class PoolingReorder : public IRMutator {
 public:
  PoolingReorder() = default;
  ~PoolingReorder() override = default;

  Stmt Reorder(const Stmt &stmt) {
    Stmt s = this->Mutate(stmt);
    return s;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (is_first_max_ && (op->name == call_name_ || op->args.size() == args_.size())) {
      Array<Expr> args;
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (i == im2colArgIndex::kKHIndex) {
          if (mode_ == ReorderMode::modeTwo) {
            kh_ = op->args[i].as<Variable>();
          }
          args.push_back(truncdiv(khw_var_ + Expr(1), kw_value_));
        } else if (i == im2colArgIndex::kKWIndex) {
          if (mode_ == ReorderMode::modeTwo) {
            kw_ = op->args[i].as<Variable>();
          }
          args.push_back(truncmod(khw_var_ + Expr(1), kw_value_));
        } else {
          args.push_back(op->args[i]);
        }
      }
      Expr res = Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
      args_ = args;
      return res;
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    const Max *max_compute = op->value.as<Max>();
    if (max_compute != nullptr && isImm(max_compute->b)) {
      is_first_max_ = true;
      if (mode_ == ReorderMode::modeOne) {
        oh_ = op->args[firstMaxMaskCallArgs::fmmOh].as<Variable>();
        ow_ = op->args[firstMaxMaskCallArgs::fmmOw].as<Variable>();
      }
      call_name_ = op->func->func_name();
      khw_var_ = VarExpr("khwVar");
      Expr kh = op->args[firstMaxMaskCallArgs::fmmKh];
      kh = Simplify(kh);
      kh_ = kh.as<Variable>();
      Expr kw = op->args[firstMaxMaskCallArgs::fmmKw];
      kw = Simplify(kw);
      kw_ = kw.as<Variable>();
      if (for_map_.count(kh_) > 0) {
        kh_value_ = for_map_[kh_]->extent;
      }
      if (for_map_.count(kw_) > 0) {
        kw_value_ = for_map_[kw_]->extent;
      }
      Expr e = this->Mutate(op->value);
      return Provide::make(op->func, op->value_index, e, args_);
    } else {
      if (is_first_max_ && op->func->func_name() == call_name_) {
        Expr e = this->Mutate(op->value);
        return Provide::make(op->func, op->value_index, e, args_);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    last_for_ = op;
    // collect for variables
    if (for_map_.count(op->loop_var.get()) == 0) {
      for_map_[op->loop_var.get()] = op;
    }

    Stmt body = Mutate(op->body);
    if (op->loop_var.get() == oh_) {
      oh_ = nullptr;
      return body;
    }

    if (op->loop_var.get() == ow_) {
      ow_ = nullptr;
      return body;
    }

    if (op->loop_var.get() == kw_) {
      kw_ = nullptr;
      return body;
    }

    if (op->loop_var.get() == kh_) {
      kh_ = nullptr;
      return body;
    }

    return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
  }

  Stmt AddFor(const For *op, Stmt body) {
    if (op == nullptr) {
      return body;
    }
    return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
  }

  Stmt AddVecFor(std::vector<const For *> vecs, Stmt body) {
    for (auto op : vecs) {
      body = AddFor(op, body);
    }
    return body;
  }

  bool IsModeTwo(const Stmt &s) { return s.as<For>() && s.as<For>()->body.as<For>(); }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (op->first.get()->GetTypeKey() == For::_type_key && op->rest.get()->GetTypeKey() == Block::_type_key) {
      const auto rest = op->rest.as<Block>();
      CHECK(rest);
      if (rest->first.get()->GetTypeKey() == For::_type_key && rest->rest.get()->GetTypeKey() == For::_type_key) {
        Stmt body = Mutate(rest->first);
        if (is_first_max_) {
          if (IsModeTwo(op->first)) {
            mode_ = ReorderMode::modeTwo;
          }
          Stmt first = Mutate(op->first);
          Stmt last = Mutate(rest->rest);
          const For *for_oh = nullptr;
          const For *for_ow = nullptr;
          if (for_map_.count(oh_) > 0) {
            for_oh = for_map_[oh_];
          }
          if (for_map_.count(ow_) > 0) {
            for_ow = for_map_[ow_];
          }

          std::vector<const For *> vecs;
          if (for_ow != nullptr) {
            vecs.push_back(for_ow);
          }
          if (for_oh != nullptr) {
            vecs.push_back(for_oh);
          }

          // pass down for axis for first part, body part and last part
          first = AddVecFor(vecs, first);
          body = AddVecFor(vecs, body);
          last = AddVecFor(vecs, last);

          // make three part block
          Stmt rest_stmt = Block::make(body, last);
          first = Block::make(first, rest_stmt);
          is_first_max_ = false;
          if (last_for_ != nullptr) {
            first = For::make(khw_var_, Expr(0), Simplify(kh_value_ * kw_value_ - Expr(1)), last_for_->for_type,
                              last_for_->device_api, first);
          }
          for_map_.clear();
          return first;
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  const Variable *oh_{nullptr};
  const Variable *ow_{nullptr};
  const Variable *kh_{nullptr};
  const Variable *kw_{nullptr};
  bool is_first_max_{false};
  VarExpr khw_var_;
  Expr kh_value_;
  Expr kw_value_;
  std::string call_name_;
  Array<Expr> args_;
  std::unordered_map<const Variable *, const For *> for_map_;
  ReorderMode mode_{ReorderMode::modeOne};
  const For *last_for_{nullptr};
};

enum PoolingHAxisIndex { INIT = 0, HEAD, BODY, TAIL };
/********************************************************
 *
 * extract pragma_load3d attributes and added into attributes map
 *
 *  for (c15, 0, 3) {
 *    // attr [{"dilation_h": 1, "stride_w": 2, "repeat_time": 7, "stride_h": 2, "h": 4, "filter_h": 3, "w": 112,
 *    "filter_w": 3, "repeat_mode": 1, "pos_h": cc10, "firstHi": 0, "firstWi": 0, "pad_bottom": 1, "pos_w": c15,
 *    "pad_top": 0, "di    lation_w": 1, "pad_left": 0, "jump_offset": 1, "pad_right": 1}] pragma_load3d = 1
 *     for (cc8, 0, 2) {
 *
 * ******************************************************/
class PoolingAttrProducer : public IRMutator {
 public:
  PoolingAttrProducer(const std::string &name, int times) : call_name_(name), load3d_times_(times) {
    if (load3d_times_ < static_cast<int>(PoolingHAxisIndex::TAIL)) {
      index_ = 1;
    }
  }
  ~PoolingAttrProducer() override = default;

  Stmt produceAttr(const Stmt &stmt) {
    Stmt s = this->Mutate(stmt);
    return s;
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    CHECK(op->func.as<PlaceholderOpNode>());
    std::size_t pos = op->func.as<PlaceholderOpNode>()->name.find("_local_L1");
    if (pos != std::string::npos) {
      CHECK(op->body.as<For>());
      fm_h = op->body.as<For>()->extent;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    const Call *call = op->value.as<Call>();

    if (call && call->name == call_name_) {
      index_ = index_ + 1;

      // default to set 0 with type int8, it will not valid becase there will be no pad.
      pad_value = IntImm::make(Int(8), 0);
      if (auto select_op = call->args[static_cast<size_t>(LOAD3DCallArgs::idxTensorIdx)].as<Select>()) {
        auto tv = select_op->true_value;
        auto fv = select_op->false_value;
        if (tv.as<FloatImm>() || tv.as<IntImm>() || tv.as<UIntImm>()) {
          pad_value = tv;
        } else if (fv.as<FloatImm>() || fv.as<IntImm>() || fv.as<UIntImm>()) {
          pad_value = fv;
        }
      }

      pad_t = call->args[static_cast<size_t>(LOAD3DCallArgs::idxPadTop)];
      pad_b = call->args[static_cast<size_t>(LOAD3DCallArgs::idxPadBottom)];
      pad_l = call->args[static_cast<size_t>(LOAD3DCallArgs::idxPadLeft)];
      pad_r = call->args[static_cast<size_t>(LOAD3DCallArgs::idxPadRight)];
      fm_w = call->args[static_cast<size_t>(LOAD3DCallArgs::idxFmW)];
      stride_h = call->args[static_cast<size_t>(LOAD3DCallArgs::idxStrideH)];
      stride_w = call->args[static_cast<size_t>(LOAD3DCallArgs::idxStrideW)];
      filter_h = call->args[static_cast<size_t>(LOAD3DCallArgs::idxFilterH)];
      filter_w = call->args[static_cast<size_t>(LOAD3DCallArgs::idxFilterW)];
      dilation_h = call->args[static_cast<size_t>(LOAD3DCallArgs::idxDilationH)];
      dilation_w = call->args[static_cast<size_t>(LOAD3DCallArgs::idxDilationW)];
      repeat_mode = call->args[static_cast<size_t>(LOAD3DCallArgs::idxRepeatMode)];
      jmp_offset = call->args[static_cast<size_t>(LOAD3DCallArgs::idxJmpOffset)];
      kh_expr_ = op->args[static_cast<size_t>(im2colArgIndex::kKHIndex)];
      kw_expr_ = op->args[static_cast<size_t>(im2colArgIndex::kKWIndex)];
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr valideWValue(const Expr &out_w) {
    Expr body = Mul::make(out_w - 1, stride_w);
    return Simplify_cce(Add::make(body, filter_w));
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    Stmt res = IRMutator::Mutate_(op, s);
    if (op->attr_key == "pragma_load3d") {
      std::unordered_map<std::string, NodeRef> attrs;
      attrs["pad_value"] = pad_value;

      attrs["pad_left"] = pad_l;
      attrs["pad_right"] = pad_r;

      CHECK_LE(index_, static_cast<int>(PoolingHAxisIndex::TAIL));
      updatePadValue();

      attrs["pad_top"] = pad_t;
      attrs["pad_bottom"] = pad_b;

      attrs["h"] = fm_h;
      attrs["w"] = fm_w;
      attrs["stride_h"] = stride_h;
      attrs["stride_w"] = stride_w;
      attrs["filter_h"] = filter_h;
      attrs["filter_w"] = filter_w;
      attrs["dilation_h"] = dilation_h;
      attrs["dilation_w"] = dilation_w;
      attrs["repeat_mode"] = repeat_mode;
      attrs["jump_offset"] = jmp_offset;
      attrs["pos_w"] = kw_expr_;
      attrs["pos_h"] = kh_expr_;
      attrs["firstWi"] = first_wi;
      attrs["firstHi"] = first_hi;
      Expr oh = truncdiv(fm_h + pad_t + pad_b - filter_h, stride_h) + Expr(1);
      Expr ow = truncdiv(fm_w + pad_l + pad_r - filter_w, stride_w) + Expr(1);
      if (is_const(oh) && is_const(ow)) {
        repeat_time = GetIntConst(truncmod(oh * ow, 16)) > 0 ? truncdiv(oh * ow, 16) + 1 : truncdiv(oh * ow, 16);
      } else {
        // dynamic mode
        repeat_time = truncdiv(oh * ow, 16) + 1;
      }
      attrs["repeat_time"] = repeat_time;
      // update fm_w value
      Expr valide_fm_w = valideWValue(ow);
      if (is_zero(pad_r) && Compare(Simplify_cce(Add::make(pad_r, fm_w)), valide_fm_w) > 0) {
        attrs["w"] = Simplify_cce(valide_fm_w - pad_r);
      }

      CHECK(res.as<AttrStmt>());
      res = AttrStmt::make(Map<std::string, NodeRef>(attrs.begin(), attrs.end()), op->attr_key, op->value,
                           res.as<AttrStmt>()->body);
    }
    return res;
  }

  void updatePadValue() {
    // no h tiling condition
    if (load3d_times_ == static_cast<int>(PoolingHAxisIndex::HEAD)) {
      first_hi = Simplify(Expr(0) - pad_t);
      first_wi = Simplify(Expr(0) - pad_l);
      return;
    }
    if (index_ == static_cast<int>(PoolingHAxisIndex::HEAD)) {
      pad_b = Expr(0);
      first_hi = Simplify(Expr(0) - pad_t);
      first_wi = Simplify(Expr(0) - pad_l);
    } else if (index_ == static_cast<int>(PoolingHAxisIndex::BODY)) {
      pad_t = Expr(0);
      pad_b = Expr(0);
      first_hi = Expr(0);
      first_wi = Simplify(Expr(0) - pad_l);
    } else if (index_ == static_cast<int>(PoolingHAxisIndex::TAIL)) {
      pad_t = Expr(0);
      first_hi = Expr(0);
      first_wi = Simplify(Expr(0) - pad_l);
    } else {
      LOG(FATAL) << "Invalid index " << index_;
    }
  }

 private:
  std::string call_name_;
  int index_{0};
  int load3d_times_{0};
  Expr pad_value;
  Expr pad_l;
  Expr pad_r;
  Expr pad_t;
  Expr pad_b;
  Expr fm_h;
  Expr fm_w;
  Expr first_hi;
  Expr first_wi;
  Expr stride_h;
  Expr stride_w;
  Expr filter_h;
  Expr filter_w;
  Expr dilation_h;
  Expr dilation_w;
  Expr repeat_mode;
  Expr jmp_offset;
  Expr kh_expr_;
  Expr kw_expr_;
  Expr repeat_time;
};

class PoolingCheck : public IRVisitor {
 public:
  explicit PoolingCheck(const std::string &name) : call_name_(name) {}
  ~PoolingCheck() override = default;
  void Visit_(const Provide *op) final {
    const Call *call = op->value.as<Call>();
    if (call != nullptr && call->name == call_name_) {
      find_ = true;
      load3d_times_++;
    }
    IRVisitor::Visit_(op);
  }

  bool find() const { return find_; }
  int times() const { return load3d_times_; }

 private:
  std::string call_name_;
  bool find_{false};
  int load3d_times_{0};
};

Stmt PoolingTransform(Stmt stmt, bool is_dynamic) {
  std::string call_name = "load3d_l1_ub";
  PoolingCheck checker(call_name);
  checker.Visit(stmt);
  if (!checker.find()) {
    return stmt;
  }

  PoolingFusion f(call_name, is_dynamic);
  stmt = f.Fusion(stmt);

  PoolingAttrProducer producer(call_name, checker.times());
  stmt = producer.produceAttr(stmt);

  PoolingReorder reorder;
  stmt = reorder.Reorder(stmt);

  return stmt;
}
}  // namespace ir
}  // namespace akg
