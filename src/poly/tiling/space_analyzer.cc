/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "space_analyzer.h"

#include <tvm/ir.h>

#include <algorithm>
#include <cmath>
#include <utility>

#include "poly/tiling/custom_tiling.h"
#include "poly/tiling/tiling_analyzer.h"
#include "poly/schedule_analysis/operator_info_collector.h"

namespace akg {
namespace ir {
namespace poly {

// API for analysis, used in auto tiling.
void SpaceAnalyzer::AnalyzeSpecialAxes() {
  // Step 1: Collect info, mainly about provide stmt.
  OpTypeCollector op_type_collector(analyzer_->scop_info_, analyzer_->body_);
  op_type_collector.Collect();
  provides_ana_ = std::move(op_type_collector.provides_ana_);
  count_op_tensor_ = op_type_collector.count_op_tensor_;

  // Step 2: Use analyzed info to identify tiling space for special axis.
  IdentifyInsnType();
  IdentifyVectorizedAxes();
  IdentifyDmaUnderCondition();
  IdentifyAlignAxes();
  IdentifyReduceAxes();
  IdentifyCountAxes();
  IdentifyPostFusionReduceTensors();
  IdentifySharedAxes();
  IdentifyCastAxes();
  IdentifyModAxes();
  if (analyzer_->scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    IdentifyInlinedTensors();
  }

  // Step 3: Support dynamic shape and custom tiling strategies.
  IdentifyDynamicShape();
  IdentifyCustomTiling();
}

void SpaceAnalyzer::IdentifyInlinedTensors() {
  std::unordered_map<std::string, int> tensor_count;
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      for (auto src : pe.src) {
        if (tensor_count.find(src.name) == tensor_count.end()) {
          tensor_count[src.name] = 1;
        } else {
          tensor_count[src.name] += 1;
        }
      }
    }
  }
  for (auto it : tensor_count) {
    analyzer_->RootAxis()->MarkWithAttr(AttrInfo{it.first, std::to_string(it.second)});
  }
}

void SpaceAnalyzer::MarkCaredType(ProvideEntry pe) {
  std::unordered_set<std::string> care_types = {AT_ELEMWISE, AT_BROADCAST, AT_TRANSPOSE, AT_DMA, AT_TRANSFORM, AT_PAD};
  for (auto ct : care_types) {
    if (pe.basic_op_type.find(ct) == std::string::npos) {
      continue;
    }
    if (ct == AT_PAD) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_OP_TYPE, AT_PAD});
    } else if (ct == AT_BROADCAST) {
      MarkBroadcastAxes(pe);
      TensorEntry target;
      for (auto src : pe.src) {
        if (src.loops.size() < pe.dst.loops.size()) {
          target = src;
          break;
        } else if (src.loops.size() > pe.dst.loops.size()) {
          target = pe.dst;
          break;
        }
      }
      MarkInnerMostAxis({target}, AT_BROADCAST_INNERMOST_AXIS);
    } else if (ct == AT_TRANSPOSE) {
      std::vector<TensorEntry> tensors = {pe.dst};
      for (auto src : pe.src) {
        tensors.emplace_back(src);
      }
      MarkInnerMostAxis(tensors, AT_TRANSPOSE_INNERMOST_AXIS);
    }
    analyzer_->RootAxis()->MarkWithAttr(AttrInfo{pe.basic_op_type, pe.dst.name});
    for (auto src : pe.src) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{pe.basic_op_type, src.name});
    }
  }
}

Template SpaceAnalyzer::GetOpTemplate(const Node *op) {
  CHECK(op->IsInstance<Provide>());
  isl::id s_id;
  auto &stmts = analyzer_->scop_info_.analysis_result_.GetProvideStmtsMap();
  for (auto &s : stmts) {
    if (s.second == op) {
      s_id = s.first;
      break;
    }
  }
  Template temp(Template::DEFAULT);
  if (s_id.is_null()) {
    return temp;
  }
  auto &bands = analyzer_->scop_info_.analysis_result_.GetAllOuterBandNode();
  for (auto &b : bands) {
    for (auto &i : b->stmts) {
      if (s_id == i) {
        return b->template_type;
      }
    }
  }
  return temp;
}

void SpaceAnalyzer::IdentifyInsnType() {
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      Template temp = GetOpTemplate(pe.op);
      bool is_gemm = ((temp == Template::CONV) || (temp == Template::MATMUL));
      auto target = analyzer_->scop_info_.user_config_.GetTarget();
      if ((target == TARGET_CUDA && is_gemm) || (target == TARGET_CPU && temp == Template::MATMUL)) {
        MarkGemmAxes(pe);
      }
      MarkCaredType(pe);
      for (auto src : pe.src) {
        if (HEAVY_ELTWISE_OP_MAP.find(src.name) != HEAVY_ELTWISE_OP_MAP.end()) {
          analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_HEAVY_ELTWISE, src.name});
        }
      }
    }
  }
}

void SpaceAnalyzer::EmplaceVarsInMatrices(const ProvideEntry &pe, int *index_a, int *index_b, VarNames &mx_c,
                                          VarNames &mx_a, VarNames &mx_b) {
  auto EmplaceVarsInTensor = [](TensorEntry tensor, VarNames &var_list) -> void {
    for (const auto &vars_i : tensor.var_names) {
      for (const auto &name : vars_i) {
        if (IsNum(name)) {
          continue;
        }
        var_list.emplace_back(name);
      }
    }
  };
  EmplaceVarsInTensor(pe.dst, mx_c);
  for (size_t i = 0; i < pe.src.size(); ++i) {
    auto src = pe.src[i];
    if (src.name == pe.dst.name) {
      VarNames src_c;
      EmplaceVarsInTensor(src, src_c);
      if (src_c.size() != mx_c.size()) {
        mx_c.clear();
        return;
      }
      for (size_t i = 0; i < src_c.size(); ++i) {
        if (src_c[i] != mx_c[i]) {
          mx_c.clear();
          return;
        }
      }
    } else if (*index_a == -1) {
      EmplaceVarsInTensor(src, mx_a);
      *index_a = i;
    } else if (*index_b == -1) {
      EmplaceVarsInTensor(src, mx_b);
      *index_b = i;
    } else {
      return;
    }
  }
}

void SpaceAnalyzer::FindAxisAndMark(std::unordered_map<std::string, std::string> loop_indices_map,
                                    const std::string &attr_key, Band loops) {
  for (const auto &loop : loops) {
    auto index = loop->loop_var.get()->name_hint;
    if (loop_indices_map.find(index) != loop_indices_map.end()) {
      SafeMarkAttr(loop, attr_key, loop_indices_map[index]);
    }
  }
}

void SpaceAnalyzer::MarkGemmAxes(const ProvideEntry &pe) {
  // Visit source tensors to fill mx_a and mx_b. Also, we need to check whether this provide stmt
  // is in `C = C + A * B` form and directly return if the form is broken.
  if (pe.src.size() != 3U) {
    return;
  }
  VarNames mx_c, mx_a, mx_b;
  int index_a = -1;
  int index_b = -1;
  EmplaceVarsInMatrices(pe, &index_a, &index_b, mx_c, mx_a, mx_b);

  if (mx_c.empty() || mx_a.empty()) {
    return;
  }

  // construct relationship between loop indices and loop type(b/m/n/k) and mark axis with corresponding attribute
  std::string attr_key = "";
  if (analyzer_->scop_info_.user_config_.GetEnableConvTensorCore()) {
    attr_key = AT_CONV;
  } else {
    attr_key = AT_GEMM;
  }

  std::unordered_map<std::string, std::string> loop_indices_map;
  if (analyzer_->scop_info_.user_config_.GetEnableConvTensorCore()) {
    loop_indices_map = ExtractLoopIndicesFromMatricesConv({mx_c, mx_a, mx_b});
  } else {
    loop_indices_map = ExtractLoopIndicesFromMatrices({mx_c, mx_a, mx_b});
  }

  // mark b/m/n through tensor C
  for (size_t i = 0; i < pe.dst.var_names.size(); ++i) {
    auto it = pe.dst.loops.find(i);
    if (it != pe.dst.loops.end()) {
      FindAxisAndMark(loop_indices_map, attr_key, it->second);
    }
  }
  // mark k through tensor A
  for (size_t i = 0; i < pe.src[index_a].var_names.size(); ++i) {
    auto it = pe.src[index_a].loops.find(i);
    if (it != pe.src[index_a].loops.end()) {
      FindAxisAndMark(loop_indices_map, attr_key, it->second);
    }
  }
}

void SpaceAnalyzer::MarkInnerMostAxis(std::vector<TensorEntry> tensors, const std::string &attr_key) {
  for (auto target : tensors) {
    for (int i = target.var_names.size() - 1; i >= 0; --i) {
      auto it = target.loops.find(i);
      if (it != target.loops.end()) {
        for (auto l : it->second) {
          SafeMarkAttr(l, attr_key, target.name);
        }
        break;
      }
    }
  }
}

void SpaceAnalyzer::MarkBroadcastAxes(const ProvideEntry &pe) {
  std::unordered_set<TileAxis *> broadcasted;
  for (auto dst_it : pe.dst.loops) {
    for (auto l : dst_it.second) {
      auto axis = analyzer_->Axis(l);
      if (axis != nullptr) {
        broadcasted.insert(axis);
      }
    }
  }

  for (auto src : pe.src) {
    if (src.loops.size() == 0 || src.loops.size() >= pe.dst.loops.size()) {
      continue;
    }
    for (auto src_it : src.loops) {
      for (auto l : src_it.second) {
        auto axis = analyzer_->Axis(l);
        if (axis != nullptr && broadcasted.count(axis)) {
          broadcasted.erase(axis);
        }
      }
    }
  }

  for (auto axis : broadcasted) {
    axis->MarkWithAttr(AttrInfo{AT_OP_TYPE, AT_BROADCAST});
  }
}

void SpaceAnalyzer::IdentifyVectorizedAxes() {
  if (provides_ana_.empty()) {
    return;
  }
  std::unordered_set<std::string> unsupported_insn = {AT_REDUCE, AT_TRANSFORM, AT_TRANSPOSE};
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
      TensorEntry dst_tensor = pe.dst;
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
    if (last_axis != nullptr) last_axis->MarkWithAttr(AttrInfo{AT_VECTORIZED, la.first});
  }
}

void SpaceAnalyzer::IdentifyDmaUnderCondition() {
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      if (pe.cond == nullptr) continue;
      if (pe.src.size() != 1U) continue;
      bool contain_tot = false;
      auto DetectToT = [&contain_tot](const NodeRef &op) {
        if (contain_tot) {
          return;
        }
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
      if (!contain_tot) {
        continue;
      }
      SafeMarkAttr(GetBufferInnerAxis(pe.dst), AT_TOT, "");
    }
  }
}

void SpaceAnalyzer::MarkTransposeAlign(
  const TensorEntry &dst_tensor, std::unordered_map<const For *, std::pair<std::string, std::string>> &align_axes_attrs,
  const std::string &basic_op_type) {
  const For *dst_last = GetBufferInnerAxis(dst_tensor);
  if (dst_last != nullptr) {
    align_axes_attrs[dst_last] = std::make_pair(dst_tensor.name, basic_op_type);
  } else {
    analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_TRANSFORM, dst_tensor.name});
  }
}

void SpaceAnalyzer::MarkDmaAlign(const TensorEntry &dst_tensor, std::vector<TensorEntry> src_tensors,
                                 std::unordered_map<const For *, std::pair<std::string, std::string>> &align_axes_attrs,
                                 const std::string &basic_op_type) {
  const For *dst_last = GetBufferInnerAxis(dst_tensor);
  if (dst_last != nullptr) {
    align_axes_attrs[dst_last] = std::make_pair(dst_tensor.name, basic_op_type);
  } else {
    // Pad op may create these DMA, which will aligned to 32(Bytes) / dtype
    // B((cc0 + 126794), (cc1 + 12), (cc2 + 1), 0) = input_1(cc0, cc1, cc2, 0)
    // Or B((cc0 + 126794), (cc1 + 12), (cc2 + 1), 7) = input_1(cc0, cc1, cc2, 0)
    VarNames last_names = dst_tensor.var_names.back();
    if (last_names.size() == 1U && !last_names[0].empty() && StrToDecimalInt64(last_names[0]) < ALIGN_BYTES) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_TRANSFORM, dst_tensor.name});
    }
  }
  for (auto t : src_tensors) {
    if (t.loops.size() <= dst_tensor.loops.size()) {
      continue;
    }
    const For *src_last = GetBufferInnerAxis(t);
    if (src_last != nullptr) {
      align_axes_attrs[src_last] = std::make_pair(t.name, basic_op_type);
    }
  }
}

void SpaceAnalyzer::MarkOneToManyAlign(
  const TensorEntry &dst_tensor, std::vector<TensorEntry> src_tensors,
  std::unordered_map<const For *, std::pair<std::string, std::string>> &align_axes_attrs,
  const std::string &basic_op_type) {
  if (basic_op_type.find(AT_REDUCE) == std::string::npos && basic_op_type.find(AT_BROADCAST) == std::string::npos) {
    return;
  }

  int64_t gm_block = 1;
  const For *src_last = nullptr;
  std::string src_name = "";
  auto IdentifySrcAlign = [this, &gm_block, &src_last, &src_name](const std::vector<TensorEntry> &src_tensors,
                                                                  const TensorEntry dst_tensor) {
    for (auto src : src_tensors) {
      if (src.name != dst_tensor.name) {
        src_last = GetBufferInnerAxis(src);
        src_name = src.name;
        break;
      }
    }
    if (src_last == nullptr) {
      return;
    }
    if (const auto i = src_last->extent.as<IntImm>()) {
      gm_block = i->value;
    }
  };
  const For *dst_last = GetBufferInnerAxis(dst_tensor);
  int64_t buf_block = 1;
  if (dst_last && dst_last->extent.as<IntImm>()) {
    buf_block = dst_last->extent.as<IntImm>()->value;
  }
  IdentifySrcAlign(src_tensors, dst_tensor);
  bool mark_dst = dst_last != nullptr;
  bool mark_src = src_last != nullptr;
  if (basic_op_type.find(AT_BROADCAST) != std::string::npos) {
    bool need_mark = (buf_block != gm_block && src_last != nullptr);
    mark_dst &= need_mark;
    mark_src &= need_mark;
  } else {
    TileAxis *align_axis = analyzer_->Axis(src_last);
    bool need_mark = (align_axis != nullptr && !align_axis->children.empty()) || (buf_block != gm_block);
    mark_src &= need_mark;
  }
  if (mark_dst) {
    align_axes_attrs[dst_last] = std::make_pair(dst_tensor.name, basic_op_type);
  }
  if (mark_src) {
    align_axes_attrs[src_last] = std::make_pair(src_name, basic_op_type);
  }
}

void SpaceAnalyzer::IdentifyAlignAxes() {
  if (provides_ana_.empty() || analyzer_->scop_info_.user_config_.GetTarget() != TARGET_CCE) {
    return;
  }

  std::unordered_map<const For *, std::pair<std::string, std::string>> align_axes_attrs;
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      std::vector<TensorEntry> src_tensors = pe.src;
      TensorEntry dst_tensor = pe.dst;
      if (pe.basic_op_type.find(AT_TRANSPOSE) != std::string::npos) {
        MarkTransposeAlign(dst_tensor, align_axes_attrs, pe.basic_op_type);
      } else if (pe.basic_op_type.find(AT_DMA) != std::string::npos) {
        MarkDmaAlign(dst_tensor, src_tensors, align_axes_attrs, pe.basic_op_type);
      } else {
        MarkOneToManyAlign(dst_tensor, src_tensors, align_axes_attrs, pe.basic_op_type);
      }
    }
    for (auto ai : align_axes_attrs) {
      TileAxis *align_axis = analyzer_->Axis(ai.first);
      std::string basic_op_type = ai.second.second;
      std::string key = AT_ALIGN;
      key = key + ":" + basic_op_type;
      if (align_axis != nullptr) {
        align_axis->MarkWithAttr(AttrInfo{key, ai.second.first});
      }
    }
  }
}

const For *SpaceAnalyzer::GetBufferInnerAxis(const TensorEntry &t, int offset) {
  int last_dim = static_cast<int>(t.var_names.size()) - offset;
  auto it = t.loops.find(last_dim);
  if (it != t.loops.end() && it->second.size() == 1U) {
    return it->second[0];
  }
  return nullptr;
}

void SpaceAnalyzer::SafeMarkAttr(const For *loop, const std::string &key, const std::string &value) {
  TileAxis *axis = analyzer_->Axis(loop);
  if (axis != nullptr) {
    axis->MarkWithAttr(AttrInfo{key, value});
  }
}

void SpaceAnalyzer::MarkReduceDstAxis(const TensorEntry &dst) {
  const For *dst_last = GetBufferInnerAxis(dst);
  if (dst_last == nullptr) {
    // Reduce op like A[i, 0] = A[i, 0] op B[i, j], we need to mark axis `i` as dst last for dma align.
    for (auto offset = 0; offset < static_cast<int>(dst.var_names.size()); ++offset) {
      dst_last = GetBufferInnerAxis(dst, offset + 1);
      if (dst_last != nullptr) {
        break;
      }
    }
  }
  SafeMarkAttr(dst_last, AT_REDUCE_DST_LAST, dst.name);
}

void SpaceAnalyzer::MarkReduceSrcAxis(const TensorEntry &dst, const TensorEntry &src) {
  std::unordered_set<const For *> src_axes;
  for (auto &lit : src.loops) {
    for (const For *l : lit.second) {
      src_axes.insert(l);
    }
  }
  for (auto &dit : dst.loops) {
    for (const For *l : dit.second) {
      auto sit = src_axes.find(l);
      if (sit != src_axes.end()) {
        src_axes.erase(sit);
      }
    }
  }
  for (auto l : src_axes) {
    SafeMarkAttr(l, AT_REDUCE_AXIS, src.name);
  }
}

void SpaceAnalyzer::IdentifyReduceAxes() {
  if (provides_ana_.empty()) {
    return;
  }
  TileAxis *root = analyzer_->RootAxis();

  for (auto &it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      if ((pe.basic_op_type.find(AT_REDUCE) == std::string::npos)) {
        continue;
      }
      MarkReduceDstAxis(pe.dst);

      for (TensorEntry src : pe.src) {
        if (src.name == pe.dst.name) {
          continue;
        }
        const For *src_last = GetBufferInnerAxis(src);
        SafeMarkAttr(src_last, AT_REDUCE_SRC_LAST, src.name);
        std::string flow = src.name + "->" + pe.dst.name;
        root->MarkWithAttr(AttrInfo{AT_REDUCE_FLOW, flow});
        MarkReduceSrcAxis(pe.dst, src);
      }
    }
  }
}

void SpaceAnalyzer::IdentifyCountAxes() {
  if (count_op_tensor_.loops.empty()) {
    return;
  }
  for (auto loop_entry : count_op_tensor_.loops) {
    for (auto loop : loop_entry.second) {
      SafeMarkAttr(loop, AT_COUNT_AXIS, count_op_tensor_.name);
    }
  }
}

void SpaceAnalyzer::IdentifyPostFusionReduceTensors() {
  if (provides_ana_.empty()) {
    return;
  }

  auto reduce_out_tensors_gpu = analyzer_->scop_info_.analysis_result_.GetReduceTensorInfoMap();
  if (reduce_out_tensors_gpu.empty()) {
    return;
  }
  auto IsReduceOut = [&reduce_out_tensors_gpu](const std::string &str) -> bool {
    for (auto &r : reduce_out_tensors_gpu) {
      if (r.second.write_tensor_name == str) {
        return true;
      }
    }
    return false;
  };

  TileAxis *root = analyzer_->RootAxis();
  for (auto &it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      for (auto src : pe.src) {
        bool is_reduce_src = (src.name != pe.dst.name && IsReduceOut(src.name));
        if (is_reduce_src) {
          root->MarkWithAttr(AttrInfo{AT_POST_FUSION_REDUCE_TENSOR, src.name});
        }
      }
    }
  }
}

void SpaceAnalyzer::ShiftHelper(const IntImm *offset, const IntImm *extent, int64_t *pre_off, int64_t *pre_ext,
                                int64_t *shift_time, int64_t *bound, std::string *type) const {
  if (offset == nullptr) {
    return;
  }
  if (extent == nullptr) {
    *shift_time += 1;
    *type = AT_DYNAMIC_SHIFT;
    if (*pre_off != -1 && *pre_off != 0 && offset->value != 0) {  // first time record offset
      *bound = air::ir::gcd(offset->value, *pre_off);
    }
    *pre_off = offset->value;
  } else if (*pre_off == -1 && *pre_ext == -1 && offset->value == 0) {
    *pre_off = offset->value;
    *pre_ext = extent->value;
  } else {
    if (extent->value == *pre_ext) {
      if (*pre_off == 0) {
        if (offset->value + 1 == *pre_ext) {
          *type = (*type).empty() ? AT_SHIFT : *type;
          *shift_time += 1;
        } else if (offset->value == *pre_ext) {
          *type = (*type).empty() ? AT_MODSHIFT : *type;
          *shift_time += 1;
        }
      } else if (*type == AT_MODSHIFT && offset->value == *pre_ext) {
        *shift_time += 1;
      } else if (*type == AT_SHIFT && ((offset->value + 1 + *shift_time) == *pre_ext * (*shift_time + 1))) {
        *shift_time += 1;
      }
    }
    *pre_off = offset->value;
    *pre_ext = extent->value;
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
      ShiftHelper(offset, extent, &pre_off, &pre_ext, &shift_time, &bound, &type);
    }
    if (type != "") {
      a->MarkWithAttr(AttrInfo{type, std::to_string(shift_time)});
    }
    if (bound != 1) {
      a->MarkWithAttr(AttrInfo{AT_DYNAMIC_BOUND, std::to_string(bound)});
    }
  };
  analyzer_->ForEachAxisTopDown(DetectShift);
}

void SpaceAnalyzer::IdentifyModAxes() {
  if (provides_ana_.empty()) {
    return;
  }
  auto GetModValue = [](const Expr &arg) -> int {
    int64_t res = -1;
    if (const auto fm = arg.as<FloorMod>()) {
      if (const auto mod_value = fm->b.as<IntImm>()) {
        res = mod_value->value;
      }
    } else if (const auto m = arg.as<Mod>()) {
      if (const auto mod_value = m->b.as<IntImm>()) {
        res = mod_value->value;
      }
    }
    return res;
  };
  auto Process = [this, GetModValue](TensorEntry t) {
    for (size_t a = 0; a < t.args.size(); ++a) {
      std::vector<Expr> constraints;
      constraints = FindModConstraint(t.args[a], constraints);
      for (auto c : constraints) {
        // Simply constraint lhs var to mod value, e.g. floormod((cc1+17), 5) -> cc1 mod 5 == 0.
        // Actually, this can be further improved to cc1 + 17 mod 5 == 0 and this needs lhs equation parsing.
        int64_t mod = GetModValue(c);
        auto lit = t.loops.find(a);
        if (lit == t.loops.end()) {
          continue;
        }
        for (auto loop : lit->second) {
          SafeMarkAttr(loop, AT_MOD, std::to_string(mod));
        }
      }
    }
  };
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      Process(pe.dst);
      for (auto src : pe.src) {
        Process(src);
      }
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
  } else if (const auto d = arg.as<FloorDiv>()) {
    constraints = FindModConstraint(d->a, constraints);
    constraints = FindModConstraint(d->b, constraints);
  }
  return constraints;
}

void SpaceAnalyzer::IdentifyCastAxes() {
  if (provides_ana_.empty()) {
    return;
  }
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      TensorEntry dst = pe.dst;
      std::vector<TensorEntry> srcs = pe.src;
      std::string attr_value = "";
      for (auto s : srcs) {
        if (dst.type_byte == s.type_byte) {
          continue;
        }
        attr_value += s.name;
        attr_value += ":";
        attr_value += std::to_string(s.type_byte);
        attr_value += ",";
      }
      if (attr_value.empty()) {
        continue;
      }
      attr_value += "->";
      attr_value += dst.name + ":" + std::to_string(dst.type_byte);
      if (dst.loops.empty()) {
        analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_CAST, attr_value});
      }
      for (auto it : dst.loops) {
        std::vector<const For *> loops = it.second;
        for (auto loop : loops) {
          SafeMarkAttr(loop, AT_CAST, attr_value);
        }
      }
    }
  }
}

void SpaceAnalyzer::IdentifyDynamicShape() {
  for (auto node : analyzer_->scop_info_.user_config_.GetDynamicShape()) {
    if (auto dsn = node.as<air::DynamicShapeNode>()) {
      CHECK(dsn->tensor_name != "") << "Parse dynamic shape failed. Tensor name must be set.";
      SetAttrForTensor(dsn->tensor_name, dsn->pos, "DYN_SHAPE_LIMIT", std::to_string(dsn->dyn_shape_limit));
    }
  }
}

std::string SpaceAnalyzer::ParseCustomValue(const air::CustomTilingNode *ctn) {
  std::string attr_value;
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
  return attr_value;
}

void SpaceAnalyzer::CustomTilingCustomMode(const air::CustomTilingNode *ctn, const std::string &mode) {
  std::string attr_value = "";
  std::string lv = ParseAllTypeExpr(ctn->tile_level);
  if (lv != "") {
    attr_value += ("LEVEL:" + lv);
    attr_value += ParseCustomValue(ctn);
  }
  if (attr_value.empty()) {
    return;
  }
  std::string key = "CUSTOM:" + mode;
  if (mode == "AXIS") {
    SetAttrForAxis(ctn->tile_band, ctn->tile_axis, key, attr_value);
  } else if (mode == "TENSOR") {
    const auto tn = ctn->tensor_name.as<StringImm>();
    CHECK(tn != nullptr && tn->value != "") << "Parse custom tiling failed. Tensor name must be set.";
    SetAttrForTensor(tn->value, ctn->tile_pos, key, attr_value);
  } else {
    CHECK(false) << "Custom tiling mode must be chosen from COMMON, AXIS or TENSOR";
  }
}

void SpaceAnalyzer::CustomTilingCommonMode(const air::CustomTilingNode *ctn) {
  if (analyzer_->scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    if (!ctn->thread_min.empty()) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_THREAD_MIN, ParseArrayExpr(ctn->thread_min)});
    }
    if (!ctn->thread_max.empty()) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_THREAD_MAX, ParseArrayExpr(ctn->thread_max)});
    }
    if (!ctn->thread_mod.empty()) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_THREAD_MOD, ParseArrayExpr(ctn->thread_mod)});
    }
    if (!ctn->block_min.empty()) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_BLOCK_MIN, ParseArrayExpr(ctn->block_min)});
    }
    if (!ctn->block_max.empty()) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_BLOCK_MAX, ParseArrayExpr(ctn->block_max)});
    }
    if (!ctn->block_mod.empty()) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_BLOCK_MOD, ParseArrayExpr(ctn->block_mod)});
    }
  } else {
    if (ctn->mem_ratio != -1) {
      analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_MEM_RATIO, std::to_string(ctn->mem_ratio)});
    }
  }
}

void SpaceAnalyzer::IdentifyCustomTiling() {
  for (auto node : analyzer_->scop_info_.user_config_.GetCustomTiling()) {
    if (auto ctn = node.as<air::CustomTilingNode>()) {
      const auto mode = ctn->tile_mode.as<StringImm>();
      CHECK(mode) << "Custom tiling mode must be set as string";
      if (mode->value == "COMMON") {
        CustomTilingCommonMode(ctn);
      } else {
        CustomTilingCustomMode(ctn, mode->value);
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

std::string SpaceAnalyzer::ParseArrayExpr(const Array<Expr> constraint) {
  std::stringstream ss;
  for (auto val : constraint) {
    ss << val;
    ss << ",";
  }
  return ss.str();
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

bool SpaceAnalyzer::TryMarkAttr(std::vector<TensorEntry> related_tensors, const std::string &tensor_name, int pos,
                                const std::string &attr_key, const std::string &attr_value, TileAxis *target) {
  bool marked = false;
  for (auto tensor : related_tensors) {
    if (target == nullptr) {
      if (pos >= static_cast<int>(tensor.var_names.size())) {
        CHECK_NE(tensor.name, tensor_name) << "Tile position " << pos << " exceeds tensor " << tensor.name << "'s size "
                                           << tensor.var_names.size() << ", please check custom tiling setting in dsl";
        continue;
      }
      std::vector<const For *> loops = tensor.loops[pos];
      for (auto l : loops) {
        SafeMarkAttr(l, attr_key, attr_value);
      }
      marked = true;
    } else {
      std::string target_info = tensor.name + "->" + attr_value;
      target->MarkWithAttr(AttrInfo{attr_key, target_info});
      marked = true;
    }
  }
  return marked;
}

void SpaceAnalyzer::SetAttrForTensor(const std::string &tensor_name, int pos, const std::string &attr_key,
                                     const std::string &attr_value) {
  TileAxis *target = nullptr;
  if (pos == -1) {
    target = analyzer_->RootAxis();
  }
  bool found = false;
  std::vector<TensorEntry> related_tensors;
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      TensorEntry dst = pe.dst;
      if (IsNameMatch(dst.name, tensor_name)) {
        related_tensors.emplace_back(dst);
      }
      for (TensorEntry src : pe.src) {
        if (IsNameMatch(src.name, tensor_name)) {
          related_tensors.emplace_back(src);
        }
      }
    }
  }
  found = TryMarkAttr(related_tensors, tensor_name, pos, attr_key, attr_value, target);
  if (!found) {
    LOG(WARNING) << "Tensor name " << tensor_name << " does not match in generated ir, custom tiling is not working."
                 << " This may cause low efficiency or even error due to particularity of dsl."
                 << " Please use auto tiling instead.";
  }
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
