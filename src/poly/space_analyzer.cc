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
#include "poly/space_analyzer.h"

#include <tvm/ir.h>

#include <algorithm>
#include <cmath>
#include <utility>

#include "poly/custom_tiling.h"
#include "poly/tiling_analyzer.h"

namespace akg {
namespace ir {
namespace poly {
class SpaceVisitor : public IRVisitor {
 public:
  explicit SpaceVisitor(TilingAnalyzer *analyzer) : analyzer_(analyzer) {}
  ~SpaceVisitor() override = default;
  using Band = TilingAnalyzer::Band;
  using VarNames = TilingAnalyzer::VarNames;
  using ProvideEntry = SpaceAnalyzer::ProvideEntry;
  using Tensor = SpaceAnalyzer::Tensor;

  void Collect() {
    CHECK(analyzer_);
    this->Visit(analyzer_->Halide());
  }

  void Visit_(const AttrStmt *op) final {
    cur_attr_ = op;
    IRVisitor::Visit_(op);
  }
  void Visit_(const Realize *op) final {
    local_buf_.insert(op->func->func_name());
    IRVisitor::Visit_(op);
  }
  void Visit_(const Provide *op) final {
    AnalyzeProvide(op);
    IRVisitor::Visit_(op);
  }
  void Visit_(const For *op) final {
    loop_count_ += 1;
    cur_loop_ = op;
    cur_band_.emplace_back(cur_loop_);
    AppendAttrForLoop();
    IRVisitor::Visit_(op);
    cur_loop_ = op;
    loop_count_ -= 1;
    // end of an outer band
    if (loop_count_ == 0) {
      band_count_ += 1;
      cur_band_.clear();
    }
  }
  void Visit_(const IfThenElse *op) final {
    cur_if_ = op;
    IRVisitor::Visit_(op);
    cur_if_ = op;
  }

  // Provides stmt after analysis.
  std::unordered_map<const For *, std::vector<ProvideEntry>> provides_ana_;

 private:
  TilingAnalyzer *analyzer_{nullptr};
  const For *cur_loop_{nullptr};
  const AttrStmt *cur_attr_{nullptr};
  const IfThenElse *cur_if_{nullptr};
  Band cur_band_;
  int loop_count_ = 0;
  size_t band_count_ = 0;
  std::unordered_set<std::string> local_buf_;
  std::unordered_map<std::string, int> op_pipe_map_ = {{"DMA2", PIPE_MTE2}, {"DMA3", PIPE_MTE3}, {"REDUCE", PIPE_V}};

  void AnalyzeProvide(const Provide *op) {
    if (cur_loop_ == nullptr) return;
    ProvideEntry prov;
    std::string basic_op_type = "";
    std::vector<Tensor> src_tensor;
    Tensor dst_tensor;
    std::vector<const Call *> src_call;
    auto GetSrc = [&, this](const NodeRef &op) {
      if (const auto call = op.as<Call>()) {
        src_call.emplace_back(call);
      } else if (op.as<Select>()) {
        basic_op_type = "CALL";
      }
    };
    air::ir::PostOrderVisit(op->value, GetSrc);

    for (auto call : src_call) {
      Tensor tensor;
      tensor.name = call->name;
      // get variable names
      for (auto arg : call->args) {
        VarNames vname;
        vname = analyzer_->VisitVarNames(arg, vname);
        tensor.var_names.emplace_back(vname);
      }
      tensor = MatchLoopByName(tensor);
      tensor.args = call->args;
      tensor.band_index = band_count_;
      tensor.type_byte = call->type.bytes();
      src_tensor.emplace_back(tensor);
    }

    auto src_length = static_cast<int>(src_tensor.size());
    for (auto st : src_tensor) {
      if (st.name == "mad" || st.name == "load_3d") {
        basic_op_type = "SP_CALL";
      }
    }
    if (src_length > 2 && basic_op_type != "SP_CALL") {
      LOG(WARNING) << "Detect provide has " << src_tensor.size() << " source tensors.";
      LOG(WARNING) << "After ToThreeAddress pass, number of ops in src should be less than 2.";
    }
    dst_tensor.name = op->func->func_name();
    for (auto arg : op->args) {
      VarNames vname;
      vname = analyzer_->VisitVarNames(arg, vname);
      dst_tensor.var_names.emplace_back(vname);
    }
    dst_tensor = MatchLoopByName(dst_tensor);
    dst_tensor.args = op->args;
    dst_tensor.band_index = band_count_;
    dst_tensor.type_byte = analyzer_->GetDataType(dst_tensor.name);
    prov.basic_op_type = basic_op_type.empty() ? GetBasicOpType(dst_tensor, src_tensor) : basic_op_type;
    prov.pipe = GetPipeFromBasicOpType(prov.basic_op_type);
    prov.band_index = band_count_;
    prov.src = src_tensor;
    prov.dst = dst_tensor;
    prov.op = op;
    prov.cond = cur_if_;
    provides_ana_[cur_loop_].emplace_back(prov);
  }

  std::unordered_set<int> GetPipeFromBasicOpType(const std::string &basic_op_type) {
    std::unordered_set<int> pipe;
    for (auto it : op_pipe_map_) {
      if (basic_op_type.find(it.first) != std::string::npos) pipe.insert(it.second);
    }
    if (basic_op_type.find("ELEMWISE") != std::string::npos && pipe.empty()) {
      pipe.insert(PIPE_V);
    }
    return pipe;
  }

  // Match variable to loop by name since the name in current band is unique.
  // If name is not unique, it means the axis is separated into different chunks
  // and they will need same alignment rule.
  Tensor MatchLoopByName(Tensor tensor) {
    std::unordered_map<size_t, std::vector<const For *>> loop_pos;
    for (size_t p = 0; p < tensor.var_names.size(); ++p) {
      for (auto name : tensor.var_names[p]) {
        for (auto i = static_cast<int>(cur_band_.size()) - 1; i >= 0; i--) {
          const For *loop = cur_band_[i];
          if (loop != nullptr && loop->loop_var.get()->name_hint == name) {
            loop_pos[p].emplace_back(loop);
            break;
          }
        }
      }
    }
    tensor.loops = loop_pos;
    return tensor;
  }

  std::string GetBasicOpType(const Tensor dst, const std::vector<Tensor> &srcs) {
    auto IsNum = [&](std::string name) -> bool {
      for (auto c : name)
        if (c > '9' || c < '0') return false;
      return true;
    };

    auto CountUniqueLoopName = [&](std::vector<VarNames> var_names) -> size_t {
      std::unordered_set<std::string> uni_name;
      for (auto names : var_names) {
        for (auto n : names) {
          if (IsNum(n)) continue;
          uni_name.insert(n);
        }
      }
      return uni_name.size();
    };

    auto GetSingleOpType = [&](const Tensor d, const Tensor s) -> std::string {
      auto dst_vars = d.var_names;
      auto src_vars = s.var_names;
      auto dst_vars_size = CountUniqueLoopName(dst_vars);
      auto src_vars_size = CountUniqueLoopName(src_vars);
      std::string type = "";
      if (this->local_buf_.find(s.name) == this->local_buf_.end()) type += "DMA2_";
      if (this->local_buf_.find(d.name) == this->local_buf_.end()) type += "DMA3_";

      if (dst_vars_size < src_vars_size) {
        if (d.loops.size() < s.loops.size() && d.name != s.name) {
          // A[i,0] = B[i,j]
          return type + "REDUCE";
        } else {
          return type + "UNKNOWN";
        }
      } else if (dst_vars_size > src_vars_size) {
        // A[i,j] = B[i,0]
        return type + "BROADCAST";
      } else {
        // Index size is the same.
        while (!dst_vars.empty() && !src_vars.empty()) {
          // detect transpose first
          VarNames dst_name = dst_vars.back();
          VarNames src_name = src_vars.back();
          dst_vars.pop_back();
          src_vars.pop_back();
          VarNames dst_pure_name;
          VarNames src_pure_name;
          for (auto n : dst_name)
            if (!IsNum(n)) dst_pure_name.emplace_back(n);
          for (auto n : src_name)
            if (!IsNum(n)) src_pure_name.emplace_back(n);
          if (dst_pure_name.size() == src_pure_name.size()) {
            for (size_t j = 0; j < dst_pure_name.size(); ++j) {
              if (dst_pure_name[j] != src_pure_name[j]) return type + "TRANSPOSE";
            }
          }
        }
        if (d.loops.size() == s.loops.size()) {
          // A[i,j] = B[i,j]
          return type + "ELEMWISE";
        } else {
          // A[0,i] = B[i,i]
          return type + "TRANSFORM";
        }
      }
      return type;
    };

    std::string basic_op_type = "";
    if (srcs.empty()) {
      // Dst = const
      if (this->local_buf_.find(dst.name) == this->local_buf_.end()) basic_op_type += "DMA3_";
      basic_op_type += "INIT";
    } else {
      for (auto s : srcs) {
        basic_op_type += GetSingleOpType(dst, s);
        basic_op_type += "_";
      }
    }
    return basic_op_type;
  }

  void AppendAttrForLoop() {
    if (cur_loop_ == nullptr || cur_attr_ == nullptr) return;
    TileAxis *axis = analyzer_->Axis(cur_loop_);
    if (axis != nullptr) axis->MarkWithAttr(AttrInfo{"ATTR", cur_attr_->attr_key});
    cur_attr_ = nullptr;
  }
};

// API for analysis, used in auto tiling.
void SpaceAnalyzer::AnalyzeSpecialAxes() {
  // Step 1: Collect info, mainly about provide stmt.
  SpaceVisitor visitor(analyzer_);
  visitor.Collect();
  provides_ana_ = std::move(visitor.provides_ana_);

  // Step 2: Use analyzed info to identify tiling space for special axis.
  IdentifyInsnType();
  IdentifyVectorizedAxes();
  IdentifyDmaUnderCondition();
  IdentifyAlignAxes();
  IdentifyReduceAxes();
  IdentifySharedAxes();
  IdentifyCastAxes();
  IdentifyModAxes();

  // Step 3: Support dynamic shape and custom tiling strategies.
  IdentifyDynamicShape();
  IdentifyCustomTiling();
}

void SpaceAnalyzer::IdentifyInsnType() {
  std::unordered_set<std::string> care_types = {"ELEMWISE", "BROADCAST", "DMA", "TRANSFORM"};
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      for (auto ct : care_types) {
        if (pe.basic_op_type.find(ct) == std::string::npos) continue;
        analyzer_->RootAxis()->MarkWithAttr(AttrInfo{pe.basic_op_type, pe.dst.name});
        for (auto src : pe.src) {
          analyzer_->RootAxis()->MarkWithAttr(AttrInfo{pe.basic_op_type, src.name});
        }
      }
    }
  }
}

void SpaceAnalyzer::IdentifyVectorizedAxes() {
  if (provides_ana_.empty()) return;
  std::string attr_key = "VECTORIZED";
  std::unordered_set<std::string> unsupported_insn = {"REDUCE", "TRANSFORM", "TRANSPOSE"};
  std::unordered_map<std::string, const For *> mark_dst_axes;
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      bool skip = false;
      for (auto insn : unsupported_insn) {
        if (pe.basic_op_type.find(insn) != std::string::npos) {
          skip = true;
          break;
        }
      }
      if (skip) {
        continue;
      }
      Tensor dst_tensor = pe.dst;
      const For *dst_last = GetBufferInnerAxis(dst_tensor);
      // skip if dst is scalar
      if (dst_last == nullptr) {
        continue;
      }

      const For *src_last = nullptr;
      for (auto src : pe.src) {
        const auto *last = GetBufferInnerAxis(src);
        if (last != nullptr && last == dst_last) {
          src_last = last;
          break;
        }
      }
      // skip if src tensor does not share same inner-most axis with dst tensor
      if (src_last == nullptr && !pe.src.empty()) {
        continue;
      }
      mark_dst_axes[dst_tensor.name] = dst_last;
    }
  }

  for (auto la : mark_dst_axes) {
    TileAxis *last_axis = analyzer_->Axis(la.second);
    if (last_axis != nullptr) last_axis->MarkWithAttr(AttrInfo{attr_key, la.first});
  }
}

void SpaceAnalyzer::IdentifyDmaUnderCondition() {
  std::string attr_key = "TOT";
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      if (pe.cond == nullptr) continue;
      if (pe.src.size() != 1U) continue;
      bool contain_tot = false;
      auto DetectToT = [&contain_tot](const NodeRef &op) {
        if (const auto eq = op.as<EQ>()) {
          if (((eq->a.as<Variable>() || eq->a.as<IntImm>()) &&
               (eq->b.as<Call>() && eq->b.as<Call>()->args.size() == 1U)) ||
              ((eq->b.as<Variable>() || eq->b.as<IntImm>()) &&
               (eq->a.as<Call>() && eq->a.as<Call>()->args.size() == 1U))) {
            contain_tot = true;
          }
        }
      };
      air::ir::PostOrderVisit(pe.cond->condition, DetectToT);
      if (!contain_tot) continue;
      TileAxis *tot_axis = analyzer_->Axis(GetBufferInnerAxis(pe.dst));
      if (tot_axis != nullptr) tot_axis->MarkWithAttr(AttrInfo{attr_key, ""});
    }
  }
}

void SpaceAnalyzer::IdentifyAlignAxes() {
  if (provides_ana_.empty()) return;
  std::string attr_key = "ALIGN";

  std::unordered_map<const For *, std::pair<std::string, std::string>> align_axes_attrs;
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      std::vector<Tensor> src_tensors = pe.src;
      Tensor dst_tensor = pe.dst;
      if (pe.basic_op_type.find("TRANSPOSE") != std::string::npos) {
        const For *dst_last = GetBufferInnerAxis(dst_tensor);
        if (dst_last != nullptr) {
          align_axes_attrs[dst_last] = std::make_pair(dst_tensor.name, pe.basic_op_type);
        } else {
          analyzer_->RootAxis()->MarkWithAttr(AttrInfo{"TRANSFORM", dst_tensor.name});
        }
      } else if (pe.basic_op_type.find("DMA") != std::string::npos) {
        const For *dst_last = GetBufferInnerAxis(dst_tensor);
        if (dst_last != nullptr) {
          align_axes_attrs[dst_last] = std::make_pair(dst_tensor.name, pe.basic_op_type);
        } else {
          // Pad op may create these DMA, which will aligned to 32(Bytes) / dtype
          // B((cc0 + 126794), (cc1 + 12), (cc2 + 1), 0) = input_1(cc0, cc1, cc2, 0)
          // Or B((cc0 + 126794), (cc1 + 12), (cc2 + 1), 7) = input_1(cc0, cc1, cc2, 0)
          VarNames last_names = dst_tensor.var_names.back();
          if (last_names.size() == 1U) {
            if (last_names[0] != "" &&
                static_cast<int64_t>(std::strtol(last_names[0].c_str(), nullptr, 10)) < ALIGN_BYTES)
              analyzer_->RootAxis()->MarkWithAttr(AttrInfo{"TRANSFORM", dst_tensor.name});
          }
        }
        for (auto t : src_tensors) {
          if (t.loops.size() <= dst_tensor.loops.size()) continue;
          const For *src_last = GetBufferInnerAxis(t);
          if (src_last != nullptr) {
            align_axes_attrs[src_last] = std::make_pair(t.name, pe.basic_op_type);
          }
        }
      } else {
        int64_t gm_block = 1;
        const For *src_last = nullptr;
        std::string src_name = "";
        auto IdentifySrcAlign = [this, &gm_block, &src_last, &src_name](const std::vector<Tensor> &src_tensors,
                                                                        const Tensor dst_tensor) {
          for (auto src : src_tensors) {
            if (src.name != dst_tensor.name) {
              src_last = GetBufferInnerAxis(src);
              src_name = src.name;
              break;
            }
          }
          if (src_last == nullptr) return;
          if (const auto i = src_last->extent.as<IntImm>()) gm_block = i->value;
        };
        if (pe.basic_op_type.find("REDUCE") != std::string::npos) {
          const For *dst_last = GetBufferInnerAxis(dst_tensor);
          int64_t ub_block = 1;
          if (dst_last != nullptr) {
            align_axes_attrs[dst_last] = std::make_pair(dst_tensor.name, pe.basic_op_type);
            if (const auto i = dst_last->extent.as<IntImm>()) ub_block = i->value;
          }
          IdentifySrcAlign(src_tensors, dst_tensor);
          if (src_last != nullptr) {
            TileAxis *align_axis = analyzer_->Axis(src_last);
            if ((align_axis != nullptr && !align_axis->children.empty()) || (ub_block != gm_block)) {
              align_axes_attrs[src_last] = std::make_pair(src_name, pe.basic_op_type);
            }
          }
        } else if (pe.basic_op_type.find("BROADCAST") != std::string::npos) {
          const For *dst_last = GetBufferInnerAxis(dst_tensor);
          int64_t ub_block = 1;
          if (dst_last == nullptr) continue;
          if (const auto i = dst_last->extent.as<IntImm>()) ub_block = i->value;
          IdentifySrcAlign(src_tensors, dst_tensor);
          if (ub_block != gm_block && src_last != nullptr) {
            align_axes_attrs[dst_last] = std::make_pair(dst_tensor.name, pe.basic_op_type);
            align_axes_attrs[src_last] = std::make_pair(src_name, pe.basic_op_type);
          }
        }
      }
    }
  }
  for (auto ai : align_axes_attrs) {
    TileAxis *align_axis = analyzer_->Axis(ai.first);
    std::string basic_op_type = ai.second.second;
    std::string key = attr_key + ":" + basic_op_type;
    if (align_axis != nullptr) align_axis->MarkWithAttr(AttrInfo{key, ai.second.first});
  }
}

const For *SpaceAnalyzer::GetBufferInnerAxis(Tensor t, int offset) {
  int last_dim = static_cast<int>(t.var_names.size()) - offset;
  auto it = t.loops.find(last_dim);
  if (it != t.loops.end() && it->second.size() == 1U) return it->second[0];
  return nullptr;
}

void SpaceAnalyzer::IdentifyReduceAxes() {
  if (provides_ana_.empty()) return;
  TileAxis *root = analyzer_->RootAxis();

  auto MarkAttr = [this](const For *loop, const std::string &key, const std::string &value) {
    TileAxis *axis = this->analyzer_->Axis(loop);
    if (axis != nullptr) axis->MarkWithAttr(AttrInfo{key, value});
  };
  for (auto &it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      if ((pe.basic_op_type.find("REDUCE") == std::string::npos)) continue;
      const For *dst_last = GetBufferInnerAxis(pe.dst);
      if (dst_last == nullptr) {
        // Reduce op like A[i, 0] = A[i, 0] op B[i, j], we need to mark axis `i` as dst last for dma align.
        for (auto offset = 0; offset < static_cast<int>(pe.dst.var_names.size()); ++offset) {
          dst_last = GetBufferInnerAxis(pe.dst, offset + 1);
          if (dst_last == nullptr) continue;
          MarkAttr(dst_last, "REDUCE_DST_LAST", pe.dst.name);
          break;
        }
      } else {
        MarkAttr(dst_last, "REDUCE_DST_LAST", pe.dst.name);
      }
      for (Tensor src : pe.src) {
        if (src.name == pe.dst.name) continue;
        const For *src_last = GetBufferInnerAxis(src);
        MarkAttr(src_last, "REDUCE_SRC_LAST", src.name);
        std::string flow = src.name + "->" + pe.dst.name;
        root->MarkWithAttr(AttrInfo{"REDUCE_FLOW", flow});
        std::unordered_set<const For *> src_axes;
        for (auto &lit : src.loops) {
          for (const For *l : lit.second) src_axes.insert(l);
        }
        for (auto &dit : pe.dst.loops) {
          for (const For *l : dit.second) {
            auto sit = src_axes.find(l);
            if (sit != src_axes.end()) src_axes.erase(sit);
          }
        }
        for (auto l : src_axes) {
          MarkAttr(l, "REDUCE_AXIS", pe.dst.name);
        }
      }
    }
  }
}

void SpaceAnalyzer::IdentifySharedAxes() const {
  auto SortByOffset = [](const For *l1, const For *l2) {
    const auto o1 = l1->min.as<IntImm>();
    const auto o2 = l2->min.as<IntImm>();
    return (o1 && o2 && o1->value < o2->value);
  };
  auto DetectShift = [this, SortByOffset](TileAxis *a) {
    if (a == nullptr) return;
    if (a->loops.size() <= 1U) return;
    std::string type = "";
    int64_t pre_off = -1;
    int64_t pre_ext = -1;
    int64_t shift_time = 0;
    int64_t bound = 1;
    std::vector<const For *> sorted_loop;
    sorted_loop.insert(sorted_loop.begin(), a->loops.begin(), a->loops.end());
    std::sort(sorted_loop.begin(), sorted_loop.end(), SortByOffset);
    for (size_t i = 0; i < sorted_loop.size(); ++i) {
      const For *loop = sorted_loop[i];
      const auto offset = loop->min.as<IntImm>();
      const auto extent = loop->extent.as<IntImm>();
      if (offset == nullptr) continue;
      if (extent == nullptr) {
        shift_time += 1;
        type = "DYNAMIC_SHIFT";
        if (pre_off != -1 && pre_off != 0 && offset->value != 0) {  // first time record offset
          bound = air::ir::gcd(offset->value, pre_off);
        }
        pre_off = offset->value;
      } else {
        if (pre_off == -1 && pre_ext == -1 && offset->value == 0) {
          pre_off = offset->value;
          pre_ext = extent->value;
        } else {
          if (extent->value == pre_ext) {
            if (pre_off == 0) {
              if (offset->value + 1 == pre_ext) {
                type = type.empty() ? "SHIFT" : type;
                shift_time += 1;
              } else if (offset->value == pre_ext) {
                type = type.empty() ? "MODSHIFT" : type;
                shift_time += 1;
              }
            } else if (type == "MODSHIFT" && offset->value == pre_ext) {
              shift_time += 1;
            } else if (type == "SHIFT" && ((offset->value + 1 + shift_time) == pre_ext * (shift_time + 1))) {
              shift_time += 1;
            }
          }
          pre_off = offset->value;
          pre_ext = extent->value;
        }
      }
    }
    if (type != "") {
      a->MarkWithAttr(AttrInfo{type, std::to_string(shift_time)});
    }
    if (bound != 1) {
      a->MarkWithAttr(AttrInfo{"DYNAMIC_BOUND", std::to_string(bound)});
    }
  };
  analyzer_->ForEachAxisTopDown(DetectShift);
}

void SpaceAnalyzer::IdentifyModAxes() {
  if (provides_ana_.empty()) return;
  std::string attr_key = "MOD";
  auto GetModValue = [](const Expr &arg) -> int {
    int64_t res = -1;
    if (const auto fm = arg.as<FloorMod>()) {
      if (const auto mod_value = fm->b.as<IntImm>()) res = mod_value->value;
    } else if (const auto m = arg.as<Mod>()) {
      if (const auto mod_value = m->b.as<IntImm>()) res = mod_value->value;
    }
    return res;
  };
  auto Process = [this, GetModValue, attr_key](Tensor t) {
    for (size_t a = 0; a < t.args.size(); ++a) {
      std::vector<Expr> constraints;
      constraints = FindModConstraint(t.args[a], constraints);
      for (auto c : constraints) {
        // Simply constraint lhs var to mod value, e.g. floormod((cc1+17), 5) -> cc1 mod 5 == 0.
        // Actually, this can be further improved to cc1 + 17 mod 5 == 0 and this needs lhs equation parsing.
        int64_t mod = GetModValue(c);
        auto lit = t.loops.find(a);
        if (lit == t.loops.end()) continue;
        for (auto loop : lit->second) {
          TileAxis *axis = analyzer_->Axis(loop);
          if (axis != nullptr) axis->MarkWithAttr(AttrInfo{attr_key, std::to_string(mod)});
        }
      }
    }
  };
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      Process(pe.dst);
      for (auto src : pe.src) Process(src);
    }
  }
}

std::vector<Expr> SpaceAnalyzer::FindModConstraint(const Expr &arg, std::vector<Expr> constraints) {
  if (arg.as<FloorMod>()) {
    constraints.emplace_back(arg);
  } else if (arg.as<Mod>()) {
    constraints.emplace_back(arg);
  } else if (const auto a = arg.as<Add>()) {
    constraints = FindModConstraint(a->a, constraints);
    constraints = FindModConstraint(a->b, constraints);
  } else if (const auto s = arg.as<Sub>()) {
    constraints = FindModConstraint(s->a, constraints);
    constraints = FindModConstraint(s->b, constraints);
  } else if (const auto m = arg.as<Mul>()) {
    constraints = FindModConstraint(m->a, constraints);
    constraints = FindModConstraint(m->b, constraints);
  }
  return constraints;
}

void SpaceAnalyzer::IdentifyCastAxes() {
  if (provides_ana_.empty()) return;
  std::string attr_key = "CAST";
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      Tensor dst = pe.dst;
      std::vector<Tensor> srcs = pe.src;
      std::string attr_value = "";
      for (auto s : srcs) {
        if (dst.type_byte == s.type_byte) continue;
        attr_value += s.name;
        attr_value += ":";
        attr_value += std::to_string(s.type_byte);
        attr_value += ",";
      }
      if (attr_value.empty()) continue;
      attr_value += "->";
      attr_value += dst.name + ":" + std::to_string(dst.type_byte);
      for (auto it1 : dst.loops) {
        std::vector<const For *> loops = it1.second;
        for (auto loop : loops) {
          TileAxis *axis = analyzer_->Axis(loop);
          if (axis != nullptr) axis->MarkWithAttr(AttrInfo{attr_key, attr_value});
        }
      }
    }
  }
}

void SpaceAnalyzer::IdentifyDynamicShape() {
  CHECK(analyzer_);
  for (auto node : analyzer_->dynamic_shape_) {
    if (auto dsn = node.as<air::DynamicShapeNode>()) {
      CHECK(dsn->tensor_name != "") << "Parse dynamic shape failed. Tensor name must be set.";
      SetAttrForTensor(dsn->tensor_name, dsn->pos, "DYN_SHAPE_LIMIT", std::to_string(dsn->dyn_shape_limit));
    }
  }
}

void SpaceAnalyzer::IdentifyCustomTiling() {
  CHECK(analyzer_);
  for (auto node : analyzer_->custom_tiling_) {
    if (auto ctn = node.as<air::CustomTilingNode>()) {
      const auto mode = ctn->tile_mode.as<StringImm>();
      CHECK(mode) << "Custom tiling mode must be set as string";
      if (mode->value == "COMMON") {
        if (ctn->mem_ratio != -1) {
          analyzer_->RootAxis()->MarkWithAttr(AttrInfo{"MEM_RATIO", std::to_string(ctn->mem_ratio)});
        }
      } else {
        std::string attr_value = "";
        std::string lv = ParseAllTypeExpr(ctn->tile_level);
        if (lv != "") {
          attr_value += ("LEVEL:" + lv);
          std::string min = ParseAllTypeExpr(ctn->tile_min);
          if (min != "" && min != "-1") {
            attr_value += ("_MIN:" + min);
          }
          std::string max = ParseAllTypeExpr(ctn->tile_max);
          if (max != "" && max != "-1") {
            attr_value += ("_MAX:" + max);
          }
          std::string factor = ParseAllTypeExpr(ctn->tile_factor);
          if (factor != "" && factor != "-1") {
            attr_value += ("_FACTOR:" + factor);
          }
          std::string candidate = ParseAllTypeExpr(ctn->tile_candidate);
          if (candidate != "" && candidate != "-1") {
            attr_value += ("_CANDIDATE:" + candidate);
          }
          std::string mod = ParseAllTypeExpr(ctn->tile_mod);
          if (mod != "" && mod != "-1") {
            attr_value += ("_MOD:" + mod);
          }
          if (ctn->forbid_isolate != -1) {
            attr_value += "_FORBIDISO:";
            attr_value += std::to_string(ctn->forbid_isolate);
          }
          if (ctn->priority != -1) {
            attr_value += "_PRIORITY:";
            attr_value += std::to_string(ctn->priority);
          }
          if (ctn->expansion != -1) {
            attr_value += "_EXPANSION:";
            attr_value += std::to_string(ctn->expansion);
          }
          if (const auto axis_info = ctn->axis_info.as<StringImm>()) {
            if (axis_info->value != "") {
              attr_value += "_AXISINFO:";
              attr_value += axis_info->value;
            }
          }
        }
        if (attr_value.empty()) continue;
        std::string key = "CUSTOM:" + mode->value;
        if (mode->value == "AXIS") {
          SetAttrForAxis(ctn->tile_band, ctn->tile_axis, key, attr_value);
        } else if (mode->value == "TENSOR") {
          const auto tn = ctn->tensor_name.as<StringImm>();
          CHECK(tn != nullptr && tn->value != "") << "Parse custom tiling failed. Tensor name must be set.";
          SetAttrForTensor(tn->value, ctn->tile_pos, key, attr_value);
        } else {
          CHECK(false) << "Custom tiling mode must be chosen from COMMON, AXIS or TENSOR";
        }
      }
    }
  }
}

void SpaceAnalyzer::SetAttrForAxis(int tile_band, int tile_axis, const std::string &attr_key,
                                   const std::string &attr_value) {
  CHECK(tile_band != -1 && tile_axis != -1) << "Axis mode requires field band and axis to be positive";
  std::vector<TileAxis *> custom_axes;
  auto ExtractAxis = [&, this](TileAxis *axis) {
    if ((axis->index == tile_band) && (static_cast<int>(axis->dim_axis) == tile_axis)) custom_axes.emplace_back(axis);
  };
  analyzer_->ForEachAxisTopDown(ExtractAxis);

  for (auto axis : custom_axes) {
    axis->MarkWithAttr(AttrInfo{attr_key, attr_value});
  }
}

std::string SpaceAnalyzer::ParseAllTypeExpr(const Expr constraint) {
  if (const auto intimm = constraint.as<IntImm>()) {
    return std::to_string(intimm->value);
  } else if (const auto strimm = constraint.as<StringImm>()) {
    return strimm->value;
  } else {
    return "";
  }
}

bool IsNameMatch(const std::string &match_from, const std::string &match_to) {
  std::vector<std::string> pattern = akg::common::Split(match_to, "*");
  bool fuzz = pattern.size() > 1U;
  bool match = match_from == match_to;
  if (fuzz) {
    for (auto p : pattern) {
      if (p.empty()) continue;
      if (match_from.find(p) != std::string::npos) {
        return true;
      }
    }
  }
  return match;
}

void SpaceAnalyzer::SetAttrForTensor(const std::string &tensor_name, int pos, const std::string &attr_key,
                                     const std::string &attr_value) {
  TileAxis *target = nullptr;
  if (pos == -1) target = analyzer_->RootAxis();
  bool found = false;
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      Tensor dst = pe.dst;
      if (IsNameMatch(dst.name, tensor_name)) {
        if (target == nullptr) {
          if (pos >= static_cast<int>(dst.var_names.size())) {
            if (dst.name == tensor_name)
              LOG(FATAL) << "Tile position " << pos << " exceeds tensor " << dst.name << "'s size "
                         << dst.var_names.size() << ", please check custom tiling setting in dsl";
            else
              continue;
          }
          std::vector<const For *> loops = dst.loops[pos];
          for (auto l : loops) {
            TileAxis *axis = analyzer_->Axis(l);
            if (axis != nullptr) axis->MarkWithAttr(AttrInfo{attr_key, attr_value});
          }
          found = true;
        } else {
          std::string target_info = dst.name + "->" + attr_value;
          target->MarkWithAttr(AttrInfo{attr_key, target_info});
          found = true;
        }
      }
      for (Tensor src : pe.src) {
        if (IsNameMatch(src.name, tensor_name)) {
          if (target == nullptr) {
            if (pos >= static_cast<int>(src.var_names.size())) {
              if (src.name == tensor_name)
                LOG(FATAL) << "Tile position " << pos << " exceeds tensor " << src.name << "'s size "
                           << src.var_names.size();
              else
                continue;
            }
            std::vector<const For *> loops = src.loops[pos];
            for (auto l : loops) {
              TileAxis *axis = analyzer_->Axis(l);
              if (axis != nullptr) axis->MarkWithAttr(AttrInfo{attr_key, attr_value});
            }
            found = true;
          } else {
            std::string target_info = src.name + "->" + attr_value;
            target->MarkWithAttr(AttrInfo{attr_key, target_info});
            found = true;
          }
        }
      }
    }
  }
  if (!found) {
    LOG(WARNING) << "Tensor name " << tensor_name << " does not match in generated ir, custom tiling is not working."
                 << " This may cause low efficiency or even error due to particularity of dsl."
                 << " Please use auto tiling instead.";
  }
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
