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
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <poly/poly_util.h>

#include "pass/utils.h"
#include "pass/convolution_model.h"
#include "build_module.h"

namespace akg {
namespace ir {
const int DY_L0B_LEN = 5;
const int DY_L0B_INDEX_MO = 1;
const int DY_L0B_INDEX_KO = 2;
const int DY_L0B_INDEX_MI = 3;
const int DY_L0B_INDEX_KI = 4;

const int FILTER_L0B_LEN = 4;
const int FILTER_L0B_INDEX_KO = 0;
const int FILTER_L0B_INDEX_NO = 1;
const int FILTER_L0B_INDEX_NI = 2;
const int FILTER_L0B_INDEX_KI = 3;

const int DW_L0C_LEN = 4;
const int DW_L0C_INDEX_NO = 0;
const int DW_L0C_INDEX_NI = 3;

const int OUTPUT_L0C_LEN = 5;
const int OUTPUT_L0C_INDEX_MO = 2;
const int OUTPUT_L0C_INDEX_MI = 3;

const int ISOLATE_NUM_ONE = 1;
const int ISOLATE_NUM_TWO = 2;
const int ISOLATE_NUM_THREE = 3;

const int BLOCK_INDEX = 16;

class MNKExtract : public IRVisitor {
 public:
  MNKExtract(const std::string &filter_name, bool conv_backprop_filter)
      : filter_(filter_name), conv_backprop_filter_(conv_backprop_filter) {}
  ~MNKExtract() override = default;

  void Visit_(const AttrStmt *op) final {
    if ((op->attr_key == "pragma_emit_insn") && (op->value.as<StringImm>()) &&
        (op->value.as<StringImm>()->value == "mad")) {
      is_mad_ = true;
      IRVisitor::Visit_(op);
      is_mad_ = false;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const For *op) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;

    CHECK(is_zero(op->min));
    extent_[name] = op->extent;
    IRVisitor::Visit_(op);
    extent_.erase(name);
  }

  void Visit_(const IfThenElse *op) final {
    if (!is_mad_) {
      IRVisitor::Visit_(op);
    }
  }

#define UPDATE_OUTER_AXIS(axis_, idx_)                 \
  do {                                                 \
    if (auto axis = op->args[(idx_)].as<Variable>()) { \
      axis_ = Range(0, extent_[axis->name_hint]);      \
    } else {                                           \
      CHECK(is_zero(op->args[(idx_)]));                \
    }                                                  \
  } while (0)

#define UPDATE_INNER_AXIS(axis_, idx_)           \
  do {                                           \
    auto axis = op->args[(idx_)].as<Variable>(); \
    CHECK(axis);                                 \
    axis_ = Range(0, extent_[axis->name_hint]);  \
  } while (0)

  void Visit_(const Provide *op) final {
    if (op->func->func_name() == filter_ + "_local_L1_local_L0B") {
      if (conv_backprop_filter_) {
        // dy_l0B[Batch, Mo, Ko, Mi, Ki]
        CHECK_EQ(op->args.size(), DY_L0B_LEN);

        UPDATE_OUTER_AXIS(mo_, DY_L0B_INDEX_MO);
        UPDATE_OUTER_AXIS(ko_, DY_L0B_INDEX_KO);

        UPDATE_INNER_AXIS(mi_, DY_L0B_INDEX_MI);
        UPDATE_INNER_AXIS(ki_, DY_L0B_INDEX_KI);
      } else {
        // filter_L0B[K0, No, Ni, Ki]
        CHECK_EQ(op->args.size(), FILTER_L0B_LEN);

        UPDATE_OUTER_AXIS(ko_, FILTER_L0B_INDEX_KO);
        UPDATE_OUTER_AXIS(no_, FILTER_L0B_INDEX_NO);

        UPDATE_INNER_AXIS(ni_, FILTER_L0B_INDEX_NI);
        UPDATE_INNER_AXIS(ki_, FILTER_L0B_INDEX_KI);
      }
    } else if (is_mad_) {
      if (conv_backprop_filter_) {
        // dw_L0C[No, Mo, Mi, Ni]
        CHECK_EQ(op->args.size(), DW_L0C_LEN);

        UPDATE_OUTER_AXIS(no_, DW_L0C_INDEX_NO);
        UPDATE_INNER_AXIS(ni_, DW_L0C_INDEX_NI);
      } else {
        // output_L0C[Batch, No, Mo, Mi, Ni]
        CHECK_EQ(op->args.size(), OUTPUT_L0C_LEN);

        UPDATE_OUTER_AXIS(mo_, OUTPUT_L0C_INDEX_MO);
        UPDATE_INNER_AXIS(mi_, OUTPUT_L0C_INDEX_MI);
      }
    } else {
      IRVisitor::Visit_(op);
    }
  }

  bool is_mad_{false};
  std::string filter_;
  bool conv_backprop_filter_;
  std::unordered_map<std::string, Expr> extent_;

  Range mo_{0, 1};
  Range no_{0, 1};
  Range ko_{0, 1};

  Range mi_{0, 1};
  Range ni_{0, 1};
  Range ki_{0, 1};
};

class Load3dCollector : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_attrs") {
      attrs = Downcast<Map<std::string, NodeRef>>(op->node);
    } else if (op->attr_key == "pragma_im2col") {
      find_ = true;
    }
    IRVisitor::Visit_(op);
  }

  Map<std::string, NodeRef> attrs;
  bool find_{false};
};

class IsolatedIdxCollector : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "isolated_idx") {
      CHECK(op->value.as<IntImm>());
      CHECK_EQ(idx_ + 1, op->value.as<IntImm>()->value);
      idx_++;
    }

    IRVisitor::Visit_(op);
  }

  int idx_{-1};
};

class RangeCalc : public IRVisitor {
 public:
  explicit RangeCalc(const std::unordered_map<std::string, Range> &r_map) : r_map_(r_map) {}
  ~RangeCalc() override = default;
  void Visit_(const Mul *op) final {
    Visit(op->a);
    auto a_min = min;
    auto a_max = max;
    Visit(op->b);
    auto b_min = min;
    auto b_max = max;
    IRVisitor::Visit_(op);
    min = Simplify_cce(a_min * b_min);
    max = Simplify_cce(a_max * b_max);
  }

  void Visit_(const Add *op) final {
    Visit(op->a);
    auto a_min = min;
    auto a_max = max;
    Visit(op->b);
    auto b_min = min;
    auto b_max = max;
    IRVisitor::Visit_(op);
    min = Simplify_cce(a_min + b_min);
    max = Simplify_cce(a_max + b_max);
  }

  void Visit_(const Sub *op) final {
    Visit(op->a);
    auto a_min = min;
    auto a_max = max;
    Visit(op->b);
    auto b_min = min;
    auto b_max = max;
    IRVisitor::Visit_(op);
    min = Simplify_cce(a_min - b_max);
    max = Simplify_cce(a_max - b_min);
  }

  void Visit_(const Variable *op) final {
    IRVisitor::Visit_(op);
    min = r_map_[op->name_hint]->min;
    max = Simplify_cce(r_map_[op->name_hint]->min + r_map_[op->name_hint]->extent - 1);
  }

  void Visit_(const IntImm *op) final {
    IRVisitor::Visit_(op);
    min = Expr(op->value);
    max = Expr(op->value);
  }

  Expr min{0};
  Expr max{0};

 private:
  std::unordered_map<std::string, Range> r_map_;
};

class FindOuterAxis : public IRVisitor {
 public:
  FindOuterAxis(const std::unordered_map<std::string, VarExpr> &lvMap, const std::string &name, int idx)
      : outerLoopvarMap_(lvMap), tensor_name_(name), idx_(idx) {}
  ~FindOuterAxis() override = default;

  void Visit_(const Provide *op) final {
    if ((op->func) && (op->func->func_name() == tensor_name_)) {
      isProvide_ = true;
      this->Visit(op->value);
      isProvide_ = false;
    }
  }

  void Visit_(const Call *op) final {
    if (isProvide_) {
      isIdx_ = true;
      this->Visit(op->args[idx_]);
      isIdx_ = false;
    }
  }

  void Visit_(const Variable *op) final {
    if (isIdx_) {
      for (auto kv : outerLoopvarMap_) {
        if (kv.first == op->name_hint) {
          var_ = kv.second;
        }
      }
    }
  }

 private:
  std::unordered_map<std::string, VarExpr> outerLoopvarMap_;
  std::string tensor_name_;
  int idx_;
  bool isProvide_{false};
  bool isIdx_{false};

 public:
  VarExpr var_{VarExpr("")};
};

class Load3dTransform : public IRMutator {
 public:
  Load3dTransform() {
    axis_map_["m"] = GemmAxis();
    axis_map_["k"] = GemmAxis();
    axis_map_["n"] = GemmAxis();
  }

  explicit Load3dTransform(const ConvolutionBackpropFilterModel conv) : conv_(conv) {
    isolate_idx_max_ = conv_.infer_L1_tile();
    axis_map_["m"] = GemmAxis();
    axis_map_["k"] = GemmAxis();
    axis_map_["n"] = GemmAxis();
    is_conv_backprop_filter_ = true;
  }

  ~Load3dTransform() override = default;

  Stmt transform(const Stmt &stmt) {
    Load3dCollector collector;

    collector.Visit(stmt);
    if (!collector.find_) {
      return stmt;
    }

    attrs_ = collector.attrs;

    CHECK(attrs_[ATTR_CONV_FEATURE_NAME].as<StringImm>());
    feature_ = attrs_[ATTR_CONV_FEATURE_NAME].as<StringImm>()->value;
    CHECK(attrs_[ATTR_CONV_FILTER_NAME].as<StringImm>());
    filter_ = attrs_[ATTR_CONV_FILTER_NAME].as<StringImm>()->value;

    if (attrs_.count(ATTR_CONV_BACKPROP_FILTER) > 0) {
      CHECK(attrs_[ATTR_CONV_BACKPROP_FILTER].as<IntImm>());
      conv_backprop_filter_ = attrs_[ATTR_CONV_BACKPROP_FILTER].as<IntImm>()->value;
    }

    if (!is_dynamic_) {
      CHECK(attrs_[ATTR_CONV_FEATURE_H].as<IntImm>());
      int h = attrs_[ATTR_CONV_FEATURE_H].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_PAD_TOP].as<IntImm>());
      int pad_top = attrs_[ATTR_CONV_PAD_TOP].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_PAD_BOTTOM].as<IntImm>());
      int pad_bottom = attrs_[ATTR_CONV_PAD_BOTTOM].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_STRIDE_H].as<IntImm>());
      int stride_h = attrs_[ATTR_CONV_STRIDE_H].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_STRIDE_W].as<IntImm>());
      int kernel_h = attrs_[ATTR_CONV_KERNEL_H].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_TILE_H].as<IntImm>());
      int tile_h = attrs_[ATTR_CONV_TILE_H].as<IntImm>()->value;
      if (tile_h == h) {
        tile_h += pad_top + pad_bottom;
      }

      int win_h = (h + pad_top + pad_bottom - kernel_h) / stride_h + 1;
      int win_tile_h = (tile_h - kernel_h) / stride_h + 1;

      bool head = (pad_top > 0);
      bool tail = ((win_h - 1) * stride_h + kernel_h > h + pad_top);
      isolated_base_h_ = (win_h + win_tile_h - 1) / win_tile_h;
      if (head) {
        if (tail) {
          if (isolated_base_h_ > ISOLATE_NUM_THREE) {
            isolated_base_h_ = ISOLATE_NUM_THREE;
          }
        } else {
          if (isolated_base_h_ > ISOLATE_NUM_TWO) {
            isolated_base_h_ = ISOLATE_NUM_TWO;
          }
        }
      } else {
        if (tail) {
          if (isolated_base_h_ > ISOLATE_NUM_TWO) {
            isolated_base_h_ = ISOLATE_NUM_TWO;
          }
        } else {
          if (isolated_base_h_ > ISOLATE_NUM_TWO) {
            isolated_base_h_ = ISOLATE_NUM_TWO;
          }
          if (win_h % win_tile_h == 0) {
            isolated_base_h_ = ISOLATE_NUM_ONE;
          }
        }
      }

      CHECK(attrs_[ATTR_CONV_FEATURE_W].as<IntImm>());
      int w = attrs_[ATTR_CONV_FEATURE_W].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_PAD_LEFT].as<IntImm>());
      int pad_left = attrs_[ATTR_CONV_PAD_LEFT].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_PAD_RIGHT].as<IntImm>());
      int pad_right = attrs_[ATTR_CONV_PAD_RIGHT].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_STRIDE_W].as<IntImm>());
      int stride_w = attrs_[ATTR_CONV_STRIDE_W].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_STRIDE_H].as<IntImm>());
      int kernel_w = attrs_[ATTR_CONV_KERNEL_W].as<IntImm>()->value;

      CHECK(attrs_[ATTR_CONV_TILE_W].as<IntImm>());
      int tile_w = attrs_[ATTR_CONV_TILE_W].as<IntImm>()->value;
      if (tile_w == w) {
        tile_w += pad_left + pad_right;
      }

      int win_w = (w + pad_left + pad_right - kernel_w) / stride_w + 1;
      int win_tile_w = (tile_w - kernel_w) / stride_w + 1;

      head = (pad_left > 0);
      tail = ((win_w - 1) * stride_w + kernel_w > w + pad_left);
      isolated_base_w_ = (win_w + win_tile_w - 1) / win_tile_w;
      if (head) {
        if (tail) {
          if (isolated_base_w_ > ISOLATE_NUM_THREE) {
            isolated_base_w_ = ISOLATE_NUM_THREE;
          }
        } else {
          if (isolated_base_w_ > ISOLATE_NUM_TWO) {
            isolated_base_w_ = ISOLATE_NUM_TWO;
          }
        }
      } else {
        if (tail) {
          if (isolated_base_w_ > ISOLATE_NUM_TWO) {
            isolated_base_w_ = ISOLATE_NUM_TWO;
          }
        } else {
          if (isolated_base_w_ > ISOLATE_NUM_TWO) {
            isolated_base_w_ = ISOLATE_NUM_TWO;
          }
          if (win_w % win_tile_w == 0) {
            isolated_base_w_ = ISOLATE_NUM_ONE;
          }
        }
      }
    } else {
      isolated_base_h_ = ISOLATE_NUM_TWO;
      isolated_base_w_ = ISOLATE_NUM_TWO;
    }

    return this->Mutate(stmt);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!pragma_gemm_l0_) {
      VarExpr var = op->loop_var;
      std::string name = var->name_hint;

      lv_map_[name] = var;
    }

    CHECK(op->loop_var.as<Variable>());
    std::string name = op->loop_var.as<Variable>()->name_hint;
    r_map_.emplace(std::pair<std::string, Range>{name, Range::make_by_min_extent(op->min, op->extent)});
    outerlv_map_.emplace(std::pair<std::string, VarExpr>{name, op->loop_var});
    Stmt stmt = IRMutator::Mutate_(op, s);
    outerlv_map_.erase(name);
    r_map_.erase(name);

    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    size_t idx1 = is_dynamic_ ? 1 : 0;
    size_t idx2 = is_dynamic_ ? 2 : 1;
    if (op->func->func_name() == feature_ + "_local_L1") {
      if (Equal(op->args[2], 0)) {
        pos_h_[idx1] = 0;
        pos_h_[idx2] = 1;
      } else {
        const auto h_args = op->args[2].as<Variable>();
        if (h_args) {
          std::string name = h_args->name_hint;
          pos_h_[idx1] = r_map_[name]->min;
          pos_h_[idx2] = Simplify_cce(r_map_[name]->min + r_map_[name]->extent);
        } else {
          auto f = RangeCalc(r_map_);
          f.Visit(op->args[2]);
          pos_h_[idx1] = f.min;
          pos_h_[idx2] = Simplify_cce(f.max + Expr(1));
        }
      }

      if (Equal(op->args[3], 0)) {
        pos_w_[idx1] = 0;
        pos_w_[idx2] = 1;
      } else {
        const auto w_args = op->args[3].as<Variable>();
        if (w_args) {
          std::string name = w_args->name_hint;
          pos_w_[idx1] = r_map_[name]->min;
          pos_w_[idx2] = Simplify_cce(r_map_[name]->min + r_map_[name]->extent);
        } else {
          auto f = RangeCalc(r_map_);
          f.Visit(op->args[3]);
          pos_w_[idx1] = f.min;
          pos_w_[idx2] = Simplify_cce(f.max + Expr(1));
        }
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (is_dynamic_) {
      if (op->func->func_name() == feature_ + "_local_L1") {
        auto region = op->bounds;

        CHECK_EQ(region.size(), 5);  // NC1HWC0

        // N[0, 1]
        auto range_n = region[0];
        CHECK(is_zero(range_n->min));
        CHECK(is_one(range_n->extent));

        auto range_h = region[2];
        pos_h_[0] = range_h->min;
        pos_h_[3] = Simplify_cce(range_h->min + range_h->extent);

        auto range_w = region[3];
        pos_w_[0] = range_w->min;
        pos_w_[3] = Simplify_cce(range_w->min + range_w->extent);
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "isolated_idx") {
      isolate_idx_++;
      if (is_conv_backprop_filter_) {
        gemm_idx_max_ = conv_.infer_L0_tile(isolate_idx_);
      }
      gemm_idx_ = -1;
    } else if (op->attr_key == "pragma_attrs") {
      Expr w = Downcast<Expr>(attrs_[ATTR_CONV_FEATURE_W]);
      Expr h = Downcast<Expr>(attrs_[ATTR_CONV_FEATURE_H]);
      Expr pad_left = Downcast<Expr>(attrs_[ATTR_CONV_PAD_LEFT]);
      Expr pad_right = Downcast<Expr>(attrs_[ATTR_CONV_PAD_RIGHT]);
      Expr pad_top = Downcast<Expr>(attrs_[ATTR_CONV_PAD_TOP]);
      Expr pad_bottom = Downcast<Expr>(attrs_[ATTR_CONV_PAD_BOTTOM]);

      Expr co = Downcast<Expr>(attrs_[ATTR_CONV_KERNEL_N]);
      Expr co_cut = Downcast<Expr>(attrs_[ATTR_CONV_TILE_CO]);
      Expr isolated_co = floormod(co, co_cut);
      int co_base = (!is_zero(isolated_co) ? ISOLATE_NUM_TWO : ISOLATE_NUM_ONE);

      Expr kh = Downcast<Expr>(attrs_[ATTR_CONV_KERNEL_H]);
      Expr sh = Downcast<Expr>(attrs_[ATTR_CONV_STRIDE_H]);
      CHECK(attrs_[ATTR_CONV_DILATION_H].as<IntImm>());
      int dh = attrs_[ATTR_CONV_DILATION_H].as<IntImm>()->value;
      Expr win_h = floordiv(h + pad_top + pad_bottom - ((kh - 1) * dh + 1), sh) + 1;

      Expr kw = Downcast<Expr>(attrs_[ATTR_CONV_KERNEL_W]);
      Expr sw = Downcast<Expr>(attrs_[ATTR_CONV_STRIDE_W]);
      CHECK(attrs_[ATTR_CONV_DILATION_W].as<IntImm>());
      int dw = attrs_[ATTR_CONV_DILATION_W].as<IntImm>()->value;
      Expr win_w = floordiv(w + pad_left + pad_right - ((kw - 1) * dw + 1), sw) + 1;

      Expr offset;
      Expr srcStrideFrom = Expr(floordiv(win_h * win_w + BLOCK_INDEX - 1, BLOCK_INDEX) * BLOCK_INDEX - 1);
      Expr srcStrideTo = Expr(win_h * win_w - 1);

      // filter_local_L1[Batch, Mo, Ko, Mi, Ki]
      FindOuterAxis axisMO(outerlv_map_, filter_ + "_local_L1", 1);
      axisMO.Visit(s);

      if (!is_zero(isolated_co)) {
        if (isolate_idx_ % co_base > 0) {
          offset = Expr((floordiv(win_h * win_w + BLOCK_INDEX - 1, BLOCK_INDEX) * BLOCK_INDEX - win_h * win_w) *
                        (floordiv(co, co_cut * co_cut)));
        } else {
          if (axisMO.var_->name_hint != "") {
            offset = (floordiv(win_h * win_w + BLOCK_INDEX - 1, BLOCK_INDEX) * BLOCK_INDEX - win_h * win_w) *
                     (axisMO.var_ * co_cut);
          } else {
            offset = (floordiv(win_h * win_w + BLOCK_INDEX - 1, BLOCK_INDEX) * BLOCK_INDEX - win_h * win_w) * co_cut;
          }
        }
      } else {
        if (axisMO.var_->name_hint != "") {
          offset = (floordiv(win_h * win_w + BLOCK_INDEX - 1, BLOCK_INDEX) * BLOCK_INDEX - win_h * win_w) *
                   (axisMO.var_ * co_cut);
        } else {
          offset = Expr(0);
        }
      }
      if (!is_dynamic_) {
        CHECK(attrs_[ATTR_CONV_FEATURE_N].as<IntImm>());
        int batch = attrs_[ATTR_CONV_FEATURE_N].as<IntImm>()->value;
        if (batch > 1 && attrs_.count(ATTR_CONV_TILE_B) > 0) {
          CHECK(attrs_[ATTR_CONV_TILE_B].as<IntImm>());
          int batch_cut = attrs_[ATTR_CONV_TILE_B].as<IntImm>()->value;
          CHECK_EQ(batch_cut, 1) << batch_cut;

          FindOuterAxis axisBO(outerlv_map_, filter_ + "_local_L1", 0);
          axisBO.Visit(s);
          CHECK_NE(axisBO.var_->name_hint, "");
          offset = offset + (floordiv((win_h * win_w + BLOCK_INDEX - 1), BLOCK_INDEX) * BLOCK_INDEX - win_h * win_w) *
                              (axisBO.var_ * co);
        }
      }
      Stmt stmt = IRMutator::Mutate_(op, s);

      std::unordered_map<std::string, Expr> attrs;
      attrs["offset"] = offset;
      attrs["srcStrideFrom"] = srcStrideFrom;
      attrs["srcStrideTo"] = srcStrideTo;

      return AttrStmt::make(Map<std::string, Expr>(attrs.begin(), attrs.end()), "UnalignedDMA", 0, stmt);
    } else if (op->attr_key == "pragma_gemm_l0") {
      gemm_idx_++;
      spec_gemm_ = true;
      static_cast<void>(IRMutator::Mutate_(op, s));
      spec_gemm_ = false;

      Map<std::string, Range> axisRange = Downcast<Map<std::string, Range>>(op->node);
      unsigned int bitmap = 0;

      for (auto kv : axisRange) {
        std::string name = kv.first;
        auto bitNum = static_cast<unsigned int>(name[0] - 'k');
        unsigned int bit = 1 << bitNum;
        if (name.compare(0, 3, "mo_") == 0) {
          CHECK_EQ((bitmap & bit), 0);
          updateAxis(axis_map_["m"], name, kv.second);
          bitmap |= bit;
        } else if (name.compare(0, 3, "no_") == 0) {
          CHECK_EQ((bitmap & bit), 0);
          updateAxis(axis_map_["n"], name, kv.second);
          bitmap |= bit;
        } else if (name.compare(0, 3, "ko_") == 0) {
          CHECK_EQ((bitmap & bit), 0);
          updateAxis(axis_map_["k"], name, kv.second);
          bitmap |= bit;
        }
      }
      CHECK(bitmap == ((1 << ('m' - 'k')) | (1 << ('n' - 'k')) | (1 << ('k' - 'k')))) << "bitmap: " << bitmap;

      // extract (m/n/k, ii/oi) axis
      MNKExtract inner(filter_, conv_backprop_filter_);
      inner.Visit(s);

      axis_map_["m"].oi = inner.mo_;
      axis_map_["m"].ii = inner.mi_;

      axis_map_["n"].oi = inner.no_;
      axis_map_["n"].ii = inner.ni_;

      axis_map_["k"].oi = inner.ko_;
      axis_map_["k"].ii = inner.ki_;

      pragma_gemm_l0_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      pragma_gemm_l0_ = false;

      return stmt;
    } else if (op->attr_key == "pragma_spec_gemm_attr") {
      if (spec_gemm_) {
        Map<std::string, VarExpr> varMap = Downcast<Map<std::string, VarExpr>>(op->node);

        for (auto kv : varMap) {
          lv_map_[kv.first] = kv.second;
        }

        return s;
      }

      return IRMutator::Mutate_(op, s);
    } else if (op->attr_key == "KH_axis") {
      kh_axis_ = op->value;
      Stmt stmt = IRMutator::Mutate_(op, s);
      kh_axis_ = Expr(0);

      return stmt;
    } else if (op->attr_key == "KW_axis") {
      kw_axis_ = op->value;
      Stmt stmt = IRMutator::Mutate_(op, s);
      kw_axis_ = Expr(0);

      return stmt;
    } else if (op->attr_key == "pragma_im2col") {
      std::unordered_map<std::string, NodeRef> attrs;

      if (conv_backprop_filter_ && pragma_gemm_l0_) {
        if (!is_dynamic_) {
          CHECK(attrs_[ATTR_CONV_FEATURE_W].as<IntImm>());
          int w = attrs_[ATTR_CONV_FEATURE_W].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_FEATURE_H].as<IntImm>());
          int h = attrs_[ATTR_CONV_FEATURE_H].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_PAD_LEFT].as<IntImm>());
          int pad_left = attrs_[ATTR_CONV_PAD_LEFT].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_PAD_RIGHT].as<IntImm>());
          int pad_right = attrs_[ATTR_CONV_PAD_RIGHT].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_PAD_TOP].as<IntImm>());
          int pad_top = attrs_[ATTR_CONV_PAD_TOP].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_PAD_BOTTOM].as<IntImm>());
          int pad_bottom = attrs_[ATTR_CONV_PAD_BOTTOM].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_KERNEL_H].as<IntImm>());
          int kh = attrs_[ATTR_CONV_KERNEL_H].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_STRIDE_H].as<IntImm>());
          int sh = attrs_[ATTR_CONV_STRIDE_H].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_DILATION_H].as<IntImm>());
          int dh = attrs_[ATTR_CONV_DILATION_H].as<IntImm>()->value;
          int win_h = (h + pad_top + pad_bottom - ((kh - 1) * dh + 1)) / sh + 1;

          CHECK(attrs_[ATTR_CONV_KERNEL_W].as<IntImm>());
          int kw = attrs_[ATTR_CONV_KERNEL_W].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_STRIDE_W].as<IntImm>());
          int sw = attrs_[ATTR_CONV_STRIDE_W].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_DILATION_W].as<IntImm>());
          int dw = attrs_[ATTR_CONV_DILATION_W].as<IntImm>()->value;
          int win_w = (w + pad_left + pad_right - ((kw - 1) * dw + 1)) / sw + 1;

          int b_base = conv_.b_base;
          int h_base = conv_.h_base;
          int w_base = conv_.w_base;

          // range_idx order as follow:
          // for (Ci Cut) {
          //   for (KH Cut) {
          //     for (KW Cut) {
          //       for (Co Cut) {
          //         for (Batch Cut) {
          //           for (H Cut) {
          //             for (W Cut) {
          //             }
          //           }
          //         }
          //       }
          //     }
          //   }
          // }

          CHECK_LT(isolate_idx_, isolate_idx_max_);
          CHECK_LT(gemm_idx_, gemm_idx_max_);

          if (!conv_.reduce_at_l1 || isolate_idx_ % (b_base * h_base * w_base) == 0) {
            mad_init_ = 1;
          } else {
            mad_init_ = 0;
          }

          int idx;
          IsolateInfo info;

          /* forward K axis <--> backward filter N axis */
          idx = conv_.get_n_idx(gemm_idx_);
          info = conv_.n_info[idx];
          Expr idx_k = Expr(conv_.calc_till_idx(&conv_.n_info, idx));

          CHECK(info.outer.as<IntImm>());
          if (info.outer.as<IntImm>()->value > 1) {
            // feature_fractal_L1_local_L0B[Batch, Mo, Ko, Ki, Mi]
            FindOuterAxis axisK(outerlv_map_, feature_ + "_fractal_L1_local_L0B", 2);
            axisK.Visit(s);
            idx_k += axisK.var_ * Expr(info.inner);
          }

          CHECK(info.inner.as<IntImm>());
          int k_l0 = info.inner.as<IntImm>()->value;

          /* forward M axis <--> backward filter K axis */
          idx = conv_.get_k_idx(gemm_idx_);
          info = conv_.k_info[idx];
          Expr idx_m = Expr(conv_.calc_till_idx(&conv_.k_info, idx));
          FindOuterAxis axisM(outerlv_map_, feature_ + "_fractal_L1_local_L0B", 1);
          axisM.Visit(s);

          CHECK(info.outer.as<IntImm>());
          if (info.outer.as<IntImm>()->value > 1) {
            // feature_fractal_L1_local_L0B[Batch, Mo, Ko, Ki, Mi]
            idx_m += axisM.var_ * info.inner;
          }
          outK_ = axisM.var_;

          CHECK(info.inner.as<IntImm>());
          int m_l0 = info.inner.as<IntImm>()->value;

          // regFMATRIX
          attrs["h"] = Expr(conv_.get_h_isolate_info(isolate_idx_).inner);
          attrs["w"] = Expr(conv_.get_w_isolate_info(isolate_idx_).inner);

          if (conv_.get_h_idx(isolate_idx_) == 0) {
            attrs["pad_top"] = Expr(pad_top);
          } else {
            attrs["pad_top"] = Expr(0);
          }

          if (conv_.get_h_idx(isolate_idx_) == h_base - 1) {
            attrs["pad_bottom"] = Expr(pad_bottom);
          } else {
            attrs["pad_bottom"] = Expr(0);
          }

          if (conv_.get_w_idx(isolate_idx_) == 0) {
            attrs["pad_left"] = Expr(pad_left);
          } else {
            attrs["pad_left"] = Expr(0);
          }

          if (conv_.get_w_idx(isolate_idx_) == w_base - 1) {
            attrs["pad_right"] = Expr(pad_right);
          } else {
            attrs["pad_right"] = Expr(0);
          }

          // regXm
          attrs["win_h"] = Expr(win_h);
          attrs["win_w"] = Expr(win_w);
          attrs["idx_m"] = idx_m;
          attrs["idx_k"] = idx_k;

          // regXt
          attrs["stride_w"] = attrs_[ATTR_CONV_STRIDE_W];
          attrs["stride_h"] = attrs_[ATTR_CONV_STRIDE_H];
          attrs["filter_w"] = attrs_[ATTR_CONV_KERNEL_W];
          attrs["filter_h"] = attrs_[ATTR_CONV_KERNEL_H];
          attrs["dilation_w"] = attrs_[ATTR_CONV_DILATION_W];
          attrs["dilation_h"] = attrs_[ATTR_CONV_DILATION_H];
          CHECK(conv_.get_kw_isolate_info(isolate_idx_).inner.as<IntImm>());
          CHECK(conv_.get_kh_isolate_info(isolate_idx_).inner.as<IntImm>());
          if (conv_.get_kw_isolate_info(isolate_idx_).inner.as<IntImm>()->value < kw ||
              conv_.get_kh_isolate_info(isolate_idx_).inner.as<IntImm>()->value < kh ||
              ((m_l0 + 15) / 16) >= (k_l0 / 16)) {
            attrs["jump_offset"] = Simplify_cce(k_l0 / 16);
            attrs["repeat_mode"] = Expr(1);
            attrs["repeat_time"] = Simplify_cce((m_l0 + 15) / 16);
          } else {
            attrs["jump_offset"] = Simplify_cce((m_l0 + 15) / 16);
            attrs["repeat_mode"] = Expr(0);
            attrs["repeat_time"] = Simplify_cce(k_l0 / 16);
          }

          idx = conv_.get_kw_idx(isolate_idx_);
          info = conv_.kw_info[idx];
          Expr idx_w = Expr(conv_.calc_till_idx(&conv_.kw_info, idx));
          CHECK(info.outer.as<IntImm>());
          if (info.outer.as<IntImm>()->value > 1) {
            idx_w += kw_axis_ * Expr(info.inner);
          }
          attrs["idx_w"] = idx_w;

          idx = conv_.get_kh_idx(isolate_idx_);
          info = conv_.kh_info[idx];
          Expr idx_h = Expr(conv_.calc_till_idx(&conv_.kh_info, idx));
          CHECK(info.outer.as<IntImm>());
          if (info.outer.as<IntImm>()->value > 1) {
            idx_h += kh_axis_ * Expr(info.inner);
          }
          attrs["idx_h"] = idx_h;

          attrs["kw_l0"] = Expr(conv_.get_kw_isolate_info(isolate_idx_).inner);
          attrs["kh_l0"] = Expr(conv_.get_kh_isolate_info(isolate_idx_).inner);

          attrs["fm_h"] = Expr(h);
          attrs["fm_w"] = Expr(w);
          attrs["pad"] = attrs_[ATTR_CONV_PAD_LEFT];

          return AttrStmt::make(Map<std::string, NodeRef>(attrs.begin(), attrs.end()), op->attr_key, op->value,
                                op->body);
        } else {
          return s;
        }
      } else {
        // regFMATRIX

        Expr w_l1_pad;
        Expr h_l1_pad;

        if (!is_dynamic_) {
          attrs["w"] = Simplify_cce(pos_w_[1] - pos_w_[0]);
          attrs["h"] = Simplify_cce(pos_h_[1] - pos_h_[0]);

          Expr w = Downcast<Expr>(attrs_[ATTR_CONV_FEATURE_W]);

          CHECK(attrs_[ATTR_CONV_PAD_LEFT].as<IntImm>());
          int pad_left = attrs_[ATTR_CONV_PAD_LEFT].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_PAD_RIGHT].as<IntImm>());
          int pad_right = attrs_[ATTR_CONV_PAD_RIGHT].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_STRIDE_W].as<IntImm>());
          int stride_w = attrs_[ATTR_CONV_STRIDE_W].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_KERNEL_W].as<IntImm>());
          int kernel_w = attrs_[ATTR_CONV_KERNEL_W].as<IntImm>()->value;

          Expr win_w = floordiv((w + pad_left + pad_right - kernel_w), stride_w) + 1;

          bool head = (pad_left > 0);
          bool tail = false;
          CHECK(CanonicalSimplify(win_w).as<IntImm>());
          CHECK(w.as<IntImm>());
          tail = ((CanonicalSimplify(win_w).as<IntImm>()->value - 1) * stride_w + kernel_w >
                  w.as<IntImm>()->value + pad_left);

          w_l1_pad = Simplify_cce(pos_w_[1] - pos_w_[0]);
          attrs["pad_left"] = Expr(0);
          attrs["pad_right"] = Expr(0);
          if (isolate_idx_ % isolated_base_w_ == 0 && head) {
            attrs["pad_left"] = Expr(pad_left);
            w_l1_pad = Simplify_cce(w_l1_pad + pad_left);
          }
          if (isolate_idx_ % isolated_base_w_ == isolated_base_w_ - 1 && tail) {
            attrs["pad_right"] = Expr(pad_right);
            w_l1_pad = Simplify_cce(w_l1_pad + pad_right);
          }

          Expr h = Downcast<Expr>(attrs_[ATTR_CONV_FEATURE_H]);

          CHECK(attrs_[ATTR_CONV_PAD_TOP].as<IntImm>());
          int pad_top = attrs_[ATTR_CONV_PAD_TOP].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_PAD_BOTTOM].as<IntImm>());
          int pad_bottom = attrs_[ATTR_CONV_PAD_BOTTOM].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_STRIDE_H].as<IntImm>());
          int stride_h = attrs_[ATTR_CONV_STRIDE_H].as<IntImm>()->value;

          CHECK(attrs_[ATTR_CONV_KERNEL_H].as<IntImm>());
          int kernel_h = attrs_[ATTR_CONV_KERNEL_H].as<IntImm>()->value;

          head = (pad_top > 0);
          CHECK(CanonicalSimplify(win_w).as<IntImm>());
          CHECK(h.as<IntImm>());
          tail = ((CanonicalSimplify(win_w).as<IntImm>()->value - 1) * stride_h + kernel_h >
                  h.as<IntImm>()->value + pad_top);

          h_l1_pad = Simplify_cce(pos_h_[1] - pos_h_[0]);
          attrs["pad_top"] = Expr(0);
          attrs["pad_bottom"] = Expr(0);
          if (isolate_idx_ / isolated_base_w_ % isolated_base_h_ == 0 && head) {
            attrs["pad_top"] = Expr(pad_top);
            h_l1_pad = Simplify_cce(h_l1_pad + pad_top);
          }
          if (isolate_idx_ / isolated_base_w_ % isolated_base_h_ == isolated_base_h_ - 1 && tail) {
            attrs["pad_bottom"] = Expr(pad_bottom);
            h_l1_pad = Simplify_cce(h_l1_pad + pad_bottom);
          }
        } else {
          attrs["w"] = Simplify_cce(pos_w_[2] - pos_w_[1]);
          attrs["h"] = Simplify_cce(pos_h_[2] - pos_h_[1]);

          w_l1_pad = Simplify_cce(pos_w_[3] - pos_w_[0]);
          attrs["pad_left"] = Simplify_cce(pos_w_[1] - pos_w_[0]);
          attrs["pad_right"] = Simplify_cce(pos_w_[3] - pos_w_[2]);

          h_l1_pad = Simplify_cce(pos_h_[3] - pos_h_[0]);
          attrs["pad_top"] = Simplify_cce(pos_h_[1] - pos_h_[0]);
          attrs["pad_bottom"] = Simplify_cce(pos_h_[3] - pos_h_[2]);
        }

        // regXm
        Expr lenM = Simplify_cce(axis_map_["m"].oi->extent * axis_map_["m"].ii->extent);
        attrs["m_l0"] = lenM;
        Expr m_idx;
        if (axis_map_["m"].var.defined()) {
          m_idx = Simplify_cce(axis_map_["m"].var * lenM + axis_map_["m"].base);
        } else {
          m_idx = Simplify_cce(axis_map_["m"].base);
        }

        Expr lenK = Simplify_cce(axis_map_["k"].oi->extent * axis_map_["k"].ii->extent);
        Expr k_idx;
        if (axis_map_["k"].var.defined()) {
          k_idx = Simplify_cce(axis_map_["k"].var * lenK + axis_map_["k"].base);
        } else {
          k_idx = Simplify_cce(axis_map_["k"].base);
        }

        Expr lenN = Simplify_cce(axis_map_["n"].oi->extent * axis_map_["n"].ii->extent);
        attrs["n_l0"] = lenN;

        Expr kh = Downcast<Expr>(attrs_[ATTR_CONV_KERNEL_H]);
        Expr sh = Downcast<Expr>(attrs_[ATTR_CONV_STRIDE_H]);
        Expr dh = Downcast<Expr>(attrs_[ATTR_CONV_DILATION_H]);
        Expr win_h = Simplify_cce((h_l1_pad - ((kh - 1) * dh + 1)) / sh + 1);
        attrs["win_h"] = win_h;
        Expr kw = Downcast<Expr>(attrs_[ATTR_CONV_KERNEL_W]);
        Expr sw = Downcast<Expr>(attrs_[ATTR_CONV_STRIDE_W]);
        Expr dw = Downcast<Expr>(attrs_[ATTR_CONV_DILATION_W]);
        Expr win_w = Simplify_cce((w_l1_pad - ((kw - 1) * dw + 1)) / sw + 1);
        attrs["win_w"] = win_w;

        attrs["idx_m"] = m_idx;
        attrs["idx_k"] = k_idx;

        // regXt
        attrs["stride_w"] = attrs_[ATTR_CONV_STRIDE_W];
        attrs["stride_h"] = attrs_[ATTR_CONV_STRIDE_H];
        attrs["filter_w"] = attrs_[ATTR_CONV_KERNEL_W];
        attrs["filter_h"] = attrs_[ATTR_CONV_KERNEL_H];
        attrs["dilation_w"] = attrs_[ATTR_CONV_DILATION_W];
        attrs["dilation_h"] = attrs_[ATTR_CONV_DILATION_H];
        attrs["jump_offset"] = Simplify_cce(floordiv(lenK, 16));
        attrs["repeat_mode"] = Expr(1);
        attrs["repeat_time"] = Simplify_cce(floordiv(lenM + 15, 16));

        attrs["idx_w"] = Expr(0);
        attrs["idx_h"] = Expr(0);

        attrs["kw_l0"] = kw;
        attrs["kh_l0"] = kh;
        if (!is_dynamic_) {
          attrs["fm_h"] = Simplify_cce(pos_h_[1] - pos_h_[0]);
          attrs["fm_w"] = Simplify_cce(pos_w_[1] - pos_w_[0]);
        } else {
          attrs["fm_h"] = Simplify_cce(pos_h_[2] - pos_h_[1]);
          attrs["fm_w"] = Simplify_cce(pos_w_[2] - pos_w_[1]);
        }
        attrs["pad"] = attrs_[ATTR_CONV_PAD_LEFT];

        static_cast<void>(IRMutator::Mutate_(op, s));
        return AttrStmt::make(Map<std::string, NodeRef>(attrs.begin(), attrs.end()), op->attr_key, op->value, op->body);
      }
    } else if (op->attr_key == "pragma_emit_insn") {
      if (op->value.as<StringImm>() && op->value.as<StringImm>()->value == "mad") {
        Stmt mad = AttrStmt::make(make_zero(Int(32)), "init", mad_init_, this->Mutate(op->body));
        mad = AttrStmt::make(op->node, op->attr_key, op->value, mad);
        if (outK_->name_hint != "" && conv_backprop_filter_) {
          mad = AttrStmt::make(make_zero(Int(32)), "pragma_mad_out_axis_k", outK_, mad);
        }
        return mad;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  struct GemmAxis {
    Expr base{0};
    VarExpr var;
    Range oo{0, 1};
    Range oi{0, 1};
    Range ii{0, 1};
  };

  void updateAxis(GemmAxis &axis, std::string n, const Range r) {
    if (is_zero(r->min)) {
      axis.base = Expr(0);
    } else if ((is_zero(Simplify_cce(axis.oo->min - r->min))) && (is_zero(Simplify_cce(axis.oo->extent - r->extent)))) {
      return;
    } else {
      CHECK(is_zero(Simplify_cce(axis.oo->min + axis.oo->extent - r->min)));
      axis.base = Simplify_cce(axis.base + axis.oo->extent * axis.oi->extent * axis.ii->extent);
    }

    if (is_one(r->extent)) {
      axis.var = VarExpr(ObjectPtr<Object>());
    } else {
      if (lv_map_.count(n) && lv_map_[n]->name_hint != n) {
        n = lv_map_[n]->name_hint;
      }

      if (lv_map_.count(n)) {
        axis.var = lv_map_[n];
      }
    }

    axis.oo = r;
    axis.oi = Range(0, 1);
    axis.ii = Range(0, 1);
  }

 private:
  ConvolutionBackpropFilterModel conv_;

  int isolate_idx_max_{0};
  int isolate_idx_{-1};
  int gemm_idx_max_{0};
  int gemm_idx_{-1};

  Map<std::string, NodeRef> attrs_;
  std::string feature_;
  std::string filter_;
  Expr pos_h_[4];
  Expr pos_w_[4];
  int mad_init_{1};
  int isolated_base_h_{0};
  int isolated_base_w_{0};
  bool conv_backprop_filter_{false};
  bool pragma_gemm_l0_{false};
  bool spec_gemm_{false};
  bool is_conv_backprop_filter_{false};
  std::unordered_map<std::string, GemmAxis> axis_map_;
  std::unordered_map<std::string, VarExpr> lv_map_;
  std::unordered_map<std::string, VarExpr> outerlv_map_;
  std::unordered_map<std::string, Range> r_map_;
  VarExpr outK_;
  Expr kh_axis_{0};
  Expr kw_axis_{0};
  bool is_dynamic_ = global_attrs.GetBoolAttr(kIsDynamic, false);
};

class L0C2UBTransform : public IRMutator {
 public:
  L0C2UBTransform() {}
  ~L0C2UBTransform() override = default;

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);

    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_cube_l0write") {
      convWrite = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      convWrite = false;

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (convWrite) {
      CHECK(Equal(op->min, 0));
      innerLoops_[op->loop_var->name_hint] = op->extent;
      Stmt stmt = IRMutator::Mutate_(op, s);
      if (innerLoops_.count(nInnerAxis_) == 1) {
        if (newAxis_.defined()) {
          stmt = For::make(newAxis_, 0, 16, ForType::Serial, op->device_api, stmt);
        }
      } else if ((op->loop_var->name_hint == matchAxis_)) {
        op = stmt.as<For>();

        CHECK(op);
        stmt = For::make(op->loop_var, 0, 16, op->for_type, op->device_api, op->body);
      }
      innerLoops_.erase(op->loop_var->name_hint);

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (convWrite) {
      CHECK(op->args.size() == 5) << "5dim: [Batch, Nout, Mout, Min, Nin]";
      CHECK(op->args[4].as<Variable>());
      nInnerAxis_ = op->args[4].as<Variable>()->name_hint;

      CHECK_EQ(innerLoops_.count(nInnerAxis_), 1);
      CHECK(Equal(innerLoops_[nInnerAxis_], 16)) << "N inner axis shoule be 16";

      if (Equal(op->args[3], 0)) {
        newAxis_ = VarExpr("i");
      } else {
        CHECK(op->args[3].as<Variable>());
        matchAxis_ = op->args[3].as<Variable>()->name_hint;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  bool convWrite{false};
  std::unordered_map<std::string, Expr> innerLoops_;
  VarExpr newAxis_;
  std::string matchAxis_;
  std::string nInnerAxis_;
};

class Load2dTranspose : public IRMutator {
 public:
  explicit Load2dTranspose(const std::string &name) : filter_name_(name) {}
  ~Load2dTranspose() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func->func_name() == filter_name_ + "_local_L1_local_L0A") {
      isProvide_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      isProvide_ = false;

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) override {
    if (isProvide_ && op->func->func_name() == filter_name_ + "_local_L1") {
      CHECK_EQ(op->args.size(), 5);
      std::vector<Expr> args(op->args.size());
      for (size_t i = 0; i < args.size() - 2; i++) {
        args[i] = op->args[i];
      }
      args[3] = op->args[4];
      args[4] = op->args[3];

      return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  std::string filter_name_;
  bool isProvide_{false};
};

class RealizeReshape : public IRMutator {
 public:
  explicit RealizeReshape(const std::string &name) : output_name_(name) {}
  ~RealizeReshape() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;

    loopvarMap_.emplace(std::pair<std::string, Expr>{name, op->extent});
    Stmt stmt = IRMutator::Mutate_(op, s);
    loopvarMap_.erase(name);

    return stmt;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    FunctionRef func = op->func;
    std::string name = func->func_name();
    if (name == output_name_ + "_local_UB_local_L0C") {
      Region bounds;
      Stmt body = this->Mutate(op->body);
      for (size_t i = 0; i < l0c_shape_.size(); i++) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), l0c_shape_[i]));
      }
      l0c_shape_.clear();

      return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, body);
    } else if (name == output_name_ + "_local_UB") {
      Region bounds;
      Stmt body = this->Mutate(op->body);
      for (size_t i = 0; i < ub_shape_.size(); i++) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), ub_shape_[i]));
      }
      ub_shape_.clear();

      return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, body);
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    FunctionRef func = op->func;
    std::string name = func->func_name();
    if (name == output_name_ + "_local_UB_local_L0C" && l0c_shape_.size() == 0) {
      for (size_t i = 0; i < op->args.size(); i++) {
        if (const auto var = op->args[i].as<Variable>()) {
          CHECK_GT(loopvarMap_.count(var->name_hint), 0);
          l0c_shape_.push_back(loopvarMap_[var->name_hint]);
        } else {
          CHECK(is_zero(op->args[i]));
          l0c_shape_.push_back(Expr(1));
        }
      }
    } else if (name == output_name_ + "_local_UB" && ub_shape_.size() == 0) {
      for (size_t i = 0; i < op->args.size(); i++) {
        if (const auto var = op->args[i].as<Variable>()) {
          CHECK_GT(loopvarMap_.count(var->name_hint), 0);
          ub_shape_.push_back(loopvarMap_[var->name_hint]);
        } else {
          CHECK(is_zero(op->args[i]));
          ub_shape_.push_back(Expr(1));
        }
      }
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<std::string, Expr> loopvarMap_;
  std::string output_name_;
  std::vector<Expr> l0c_shape_;
  std::vector<Expr> ub_shape_;
};

class RealizeElimination : public IRMutator {
 public:
  RealizeElimination() {}
  ~RealizeElimination() override = default;

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    FunctionRef func = op->func;
    std::string name = func->func_name();
    if (realizeMap_.count(name) > 0) {
      return this->Mutate(op->body);
    } else {
      realizeMap_.emplace(std::pair<std::string, FunctionRef>{name, func});
      Stmt stmt = IRMutator::Mutate_(op, s);
      realizeMap_.erase(name);

      return stmt;
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    FunctionRef func = op->func;
    std::string name = func->func_name();
    if (realizeMap_.count(name) > 0) {
      Expr value = this->Mutate(op->value);
      return Provide::make(realizeMap_[name], op->value_index, value, op->args);
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    FunctionRef func = op->func;
    std::string name = func->func_name();

    CHECK(op->name == name);
    if (realizeMap_.count(name) > 0) {
      return Call::make(op->type, op->name, op->args, Call::CallType::Halide, realizeMap_[name], op->value_index);
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  std::unordered_map<std::string, FunctionRef> realizeMap_;
};

class RealizeScopeElimination : public IRMutator {
 public:
  RealizeScopeElimination() {}
  ~RealizeScopeElimination() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == ktvm::ir::attr::realize_scope && !op->body.as<Realize>()) {
      return this->Mutate(op->body);
    }

    return IRMutator::Mutate_(op, s);
  }
};

class RealizeCount : public IRVisitor {
 public:
  ~RealizeCount() override = default;

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "isolated_idx") {
      isolate_num_++;
      gemm_num_ = 0;
      if (isolated_idx_level_ == -1) {
        isolated_idx_level_ = loop_level_;
      }
    } else if (op->attr_key == "pragma_gemm_l0") {
      gemm_num_++;
      if (gemm_idx_level_ == -1) {
        gemm_idx_level_ = loop_level_;
      }
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) final {
    loop_level_++;
    IRVisitor::Visit_(op);
    loop_level_--;
  }

  void Visit_(const IfThenElse *op) final {
    this->Visit(op->condition);

    if (!op->else_case.defined()) {
      IRVisitor::Visit_(op);
      return;
    }

    int isolate_num_bak = isolate_num_;
    int gemm_num_bak = gemm_num_;
    int isolated_idx_level_bak = isolated_idx_level_;
    int gemm_idx_level_bak = gemm_idx_level_;

    this->Visit(op->then_case);
    int isolate_num_if = isolate_num_;
    int gemm_num_if = gemm_num_;
    int isolated_idx_level_if = isolated_idx_level_;
    int gemm_idx_level_if = gemm_idx_level_;

    isolate_num_ = isolate_num_bak;
    gemm_num_ = gemm_num_bak;
    isolated_idx_level_ = isolated_idx_level_bak;
    gemm_idx_level_ = gemm_idx_level_bak;

    this->Visit(op->else_case);
    CHECK_EQ(isolate_num_, isolate_num_if);
    CHECK_EQ(gemm_num_, gemm_num_if);
    CHECK_EQ(isolated_idx_level_, isolated_idx_level_if);
    CHECK_EQ(gemm_idx_level_, gemm_idx_level_if);
  }

 private:
  int loop_level_{0};

 public:
  int isolate_num_{0};
  int gemm_num_{0};
  int isolated_idx_level_{-1};
  int gemm_idx_level_{-1};
};

class RealizeRescope : public IRMutator {
 public:
  RealizeRescope(ConvolutionBackpropFilterModel &conv, std::string &output_name)
      : conv_(conv), output_name_(output_name) {
    isolate_num_ = conv_.infer_L1_tile();
    isolated_idx_level_ = 0;

    CHECK(conv_.b_info[0].outer.as<IntImm>());
    if (conv_.b_info[0].outer.as<IntImm>()->value > 1) {
      isolated_idx_level_++;
    }

    CHECK(conv_.h_win_info[0].outer.as<IntImm>());
    if (conv_.h_win_info[0].outer.as<IntImm>()->value > 1) {
      isolated_idx_level_++;
    }

    CHECK(conv_.w_win_info[0].outer.as<IntImm>());
    if (conv_.w_win_info[0].outer.as<IntImm>()->value > 1) {
      isolated_idx_level_++;
    }
  }
  ~RealizeRescope() override = default;

 private:
  Stmt addResUBL0CRealize(Stmt body) {
    CHECK(realize_res_l0c_);
    CHECK(realize_res_ub_);

    Array<Expr> shape_l0c;
    for (auto range : realize_res_l0c_->bounds) {
      CHECK(is_zero(range->min));
      shape_l0c.push_back(range->extent);
    }
    auto t_l0c = placeholder(shape_l0c, realize_res_l0c_->type, realize_res_l0c_->func->func_name());
    body = TensorSubstitute(body, realize_res_l0c_->func, t_l0c->op, t_l0c->value_index);
    body = Realize::make(t_l0c->op, t_l0c->value_index, realize_res_l0c_->type, realize_res_l0c_->bounds,
                         realize_res_l0c_->condition, body);
    body = AttrStmt::make(t_l0c->op, ktvm::ir::attr::realize_scope, Expr("local.L0C"), body);

    Array<Expr> shape_ub;
    for (auto range : realize_res_ub_->bounds) {
      CHECK(is_zero(range->min));
      shape_ub.push_back(range->extent);
    }
    auto t_ub = placeholder(shape_ub, realize_res_ub_->type, realize_res_ub_->func->func_name());
    body = TensorSubstitute(body, realize_res_ub_->func, t_ub->op, t_ub->value_index);
    body = Realize::make(t_ub->op, t_ub->value_index, realize_res_ub_->type, realize_res_ub_->bounds,
                         realize_res_ub_->condition, body);
    body = AttrStmt::make(t_ub->op, ktvm::ir::attr::realize_scope, Expr("local.UB"), body);

    return body;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!mutate_) {
      RealizeCount count;
      count.Visit(op->body);

      if (conv_.reduce_at_l1 && count.isolate_num_ == conv_.l1_reduce_base) {
        if (count.isolated_idx_level_ == isolated_idx_level_) {
          mutate_ = true;
          Stmt stmt = this->Mutate(op->body);
          mutate_ = false;

          stmt = addResUBL0CRealize(stmt);
          stmt = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, stmt);

          return stmt;
        } else if (count.isolated_idx_level_ + 1 == isolated_idx_level_) {
          mutate_ = true;
          Stmt stmt = IRMutator::Mutate_(op, s);
          mutate_ = false;

          stmt = addResUBL0CRealize(stmt);

          return stmt;
        }
      }

      if (!conv_.reduce_at_l1 && count.gemm_num_ == conv_.l0_reduce_base) {
        if (count.gemm_idx_level_ == gemm_idx_level_) {
          mutate_ = true;
          Stmt stmt = this->Mutate(op->body);
          mutate_ = false;

          stmt = addResUBL0CRealize(stmt);
          stmt = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, stmt);

          return stmt;
        } else if (count.gemm_idx_level_ + 1 == gemm_idx_level_) {
          mutate_ = true;
          Stmt stmt = IRMutator::Mutate_(op, s);
          mutate_ = false;

          stmt = addResUBL0CRealize(stmt);

          return stmt;
        }
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    FunctionRef func = op->func;
    std::string name = func->func_name();
    if (name == output_name_ + "_local_UB_local_L0C") {
      realize_res_l0c_ = op;
      return this->Mutate(op->body);
    } else if (name == output_name_ + "_local_UB") {
      realize_res_ub_ = op;
      return this->Mutate(op->body);
    }

    return IRMutator::Mutate_(op, s);
  }

  template <class T>
  Stmt countRealizeAndAddResUBL0C(const T *op, const Stmt &s) {
    if (!mutate_) {
      RealizeCount count;
      count.Visit(s);

      if ((conv_.reduce_at_l1 && count.isolate_num_ == conv_.l1_reduce_base &&
           count.isolated_idx_level_ == isolated_idx_level_) ||
          (!conv_.reduce_at_l1 && count.gemm_num_ == conv_.l0_reduce_base &&
           count.gemm_idx_level_ == gemm_idx_level_)) {
        mutate_ = true;
        Stmt stmt = IRMutator::Mutate_(op, s);
        mutate_ = false;

        stmt = addResUBL0CRealize(stmt);
        return stmt;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final { return countRealizeAndAddResUBL0C(op, s); }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (!mutate_ && op->attr_key == "isolated_idx") {
      (void)conv_.infer_L0_tile(isolate_idx_++);
      gemm_idx_level_ = 0;
      CHECK(conv_.k_info[0].outer.as<IntImm>());
      if (conv_.k_info[0].outer.as<IntImm>()->value > 1) {
        gemm_idx_level_++;
      }
    }

    return countRealizeAndAddResUBL0C(op, s);
  }

  ConvolutionBackpropFilterModel conv_;
  std::string output_name_;
  int isolated_idx_level_{0};
  int gemm_idx_level_{0};
  const Realize *realize_res_ub_{nullptr};
  const Realize *realize_res_l0c_{nullptr};
  bool mutate_{false};
  int isolate_num_{0};
  int isolate_idx_{0};
};

const char const_outermost_mark[] = "load3d_transform_outermost_mark";

static Stmt AddOuterMostMark(const Stmt &stmt) {
  return AttrStmt::make(make_zero(Int(32)), const_outermost_mark, Expr(0), stmt);
}

class RemoveOutermostMarkMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == const_outermost_mark) {
      return IRMutator::Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
};

Stmt Load3dTrans(Stmt stmt, bool is_dynamic) {
  Load3dCollector collector;
  collector.Visit(stmt);
  if (!collector.find_) {
    return stmt;
  }

  stmt = AddOuterMostMark(stmt);
  bool conv_backprop_filter = false;
  if (IS_ATTR_EXIST(collector.attrs, ATTR_CONV_BACKPROP_FILTER)) {
    conv_backprop_filter = GET_INTIMM_ATTR(collector.attrs, ATTR_CONV_BACKPROP_FILTER);
  }

  if (conv_backprop_filter) {
    ConvolutionBackpropFilterModel conv(collector.attrs, is_dynamic);

    Load3dTransform trans3d(conv);
    stmt = trans3d.transform(stmt);

    std::string filter_name = GET_STRINGIMM_ATTR_DEFAULT(collector.attrs, ATTR_CONV_FILTER_NAME, "");
    stmt = Load2dTranspose(filter_name).Mutate(stmt);

    std::string output_name = GET_STRINGIMM_ATTR_DEFAULT(collector.attrs, ATTR_CONV_RES_NAME, "");
    if (!is_dynamic) stmt = RealizeRescope(conv, output_name).Mutate(stmt);
    stmt = RealizeElimination().Mutate(stmt);
    stmt = RealizeScopeElimination().Mutate(stmt);
    stmt = RealizeReshape(output_name).Mutate(stmt);
  } else {
    Load3dTransform trans3d;
    stmt = trans3d.transform(stmt);
    stmt = L0C2UBTransform().Mutate(stmt);
    stmt = RealizeElimination().Mutate(stmt);
    stmt = RealizeScopeElimination().Mutate(stmt);
  }

  stmt = RemoveOutermostMarkMutator().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
