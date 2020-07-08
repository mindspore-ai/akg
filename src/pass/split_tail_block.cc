/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <algorithm>
#include "emit_insn/insn_info.h"
#include "emit_insn/insn_pattern.h"

namespace akg {
namespace ir {

class TailSpliter : public IRMutator {
 public:
  TailSpliter() = default;

  ~TailSpliter() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn") {
      auto intrin_name = op->value.as<StringImm>()->value;
      if (include_intrin_list_.find(intrin_name) == include_intrin_list_.end()) {
        return s;
      }
      StmtInfoList dst_info_list;
      StmtInfoList src_info_list;
      StmtInfo if_info;
      StmtInfo for_info;

      GetCompactComputationInfo(op->body, dst_info_list, src_info_list, if_info, for_info, false);
      CHECK(!dst_info_list.empty());
      auto dst_info = dst_info_list[0];
      if (src_info_list.empty()) {
        src_info_list = {dst_info.Copy()};
      }
      auto get_info_list = [](const StmtStoreInfo &dst_info, const Array<StmtStoreInfo> &src_info_list) {
        Array<StmtStoreInfo> res;
        res.push_back(dst_info.Copy());
        for (auto it : src_info_list) {
          res.push_back(it.Copy());
        }
        return res;
      };
      auto info_list = get_info_list(dst_info, src_info_list);
      FillEmptyVar(info_list);
      auto axis_list = GetAixsList(for_info, info_list);
      auto get_last_axis_it = [](const std::list<InsnAxis> &axis_list) {
        for (auto it = axis_list.begin(); it != axis_list.end(); it++) {
          auto stride_list = it->stride_list;
          if (!(std::any_of(stride_list.begin(), stride_list.end(), [](int stride) { return stride > 1; }) ||
                std::all_of(stride_list.begin(), stride_list.end(), [](int stride) { return stride == 0; }))) {
            return it;
          }
        }
        return axis_list.end();
      };

      auto last_axis_it = get_last_axis_it(axis_list);
      if (last_axis_it == axis_list.end()) {
        return s;
      }
      auto last_axis = *last_axis_it;
      auto last_axis_shape = last_axis.extent;
      int dst_block_size = GetUbBlkSize(dst_info->dtype_);
      int src_block_size = GetUbBlkSize(src_info_list[0]->dtype_);
      int block_size = dst_block_size > src_block_size ? dst_block_size : src_block_size;
      int vec_max_len = block_size * FULL_BLOCK_NUM;

      if (last_axis_shape > vec_max_len && last_axis_shape % vec_max_len != 0) {
        return Block::make(TailMake(s, last_axis, vec_max_len, false), TailMake(s, last_axis, vec_max_len, true));
      }
      if (last_axis_shape < vec_max_len * tail_rate_ && last_axis_shape > block_size &&
          last_axis_shape % block_size != 0 && axis_list.size() > 1) {
        return Block::make(TailMake(s, last_axis, block_size, false), TailMake(s, last_axis, block_size, true));
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  std::list<InsnAxis> GetAixsList(const StmtInfo &for_info, const Array<StmtStoreInfo> &info_list) {
    std::list<InsnAxis> axis_list;
    auto GetStrideByAxis = [](const Array<Var> &vars, const Array<Expr> &strides, Var obj_var) {
      int index = 0;
      for (auto var_it : vars) {
        if (Equal(var_it, obj_var)) {
          return strides[index];
        }
        index++;
      }
      return Expr(0);
    };
    for (auto it : for_info.ops_) {
      InsnAxis axis;
      auto for_stmt = it.as<For>();
      CHECK(for_stmt);
      axis.var = for_stmt->loop_var;
      axis.extent = GetInt32Const(for_stmt->extent);
      axis.min = GetInt32Const(for_stmt->min);
      int index = 0;
      for (auto it : info_list) {
        auto stride = GetInt32Const(GetStrideByAxis(it->var_, it->strides_, axis.var));
        axis.stride_list.push_back(stride);
        if (index == 0) {
          axis.dst_stride = stride;
        } else {
          axis.src_stride_list.push_back(stride);
        }
        index++;
      }
      axis_list.push_back(axis);
    }
    return axis_list;
  }

  Stmt TailMake(const Stmt &s, const InsnAxis &tail_axis, int body_size, bool is_tail) {
    if (auto attr_stmt = s.as<AttrStmt>()) {
      return AttrStmt::make(attr_stmt->node, attr_stmt->attr_key, attr_stmt->value,
                            TailMake(attr_stmt->body, tail_axis, body_size, is_tail));
    }
    if (auto for_stmt = s.as<For>()) {
      if (Equal(for_stmt->loop_var, tail_axis.var) && GetIntConst(for_stmt->extent) == tail_axis.extent) {
        if (is_tail) {
          return For::make(for_stmt->loop_var, for_stmt->min, Expr(tail_axis.extent % body_size), for_stmt->for_type,
                           for_stmt->device_api, TailMake(for_stmt->body, tail_axis, body_size, is_tail));
        }
        CHECK_NE(body_size, 0);
        Expr remain_extent = Expr(tail_axis.extent / body_size * body_size);
        return For::make(for_stmt->loop_var, for_stmt->min, remain_extent, for_stmt->for_type, for_stmt->device_api,
                         TailMake(for_stmt->body, tail_axis, body_size, is_tail));
      }
      return For::make(for_stmt->loop_var, for_stmt->min, for_stmt->extent, for_stmt->for_type, for_stmt->device_api,
                       TailMake(for_stmt->body, tail_axis, body_size, is_tail));

    } 
    if (s.as<Store>() && is_tail) {
      return substitute(tail_axis.var, Add::make(Expr(tail_axis.extent / body_size * body_size), tail_axis.var), s);
    }
    return s;
  }

 private:
  const float tail_rate_{0.6};
  const std::set<std::string> include_intrin_list_ = {
    "vec_single_fabs",
    "vec_single_log",
    "vec_single_exp",
    "vec_single_rec",
    "vec_single_not",
    "vec_single_sqrt",
    "vec_single_rsqrt",
    "vec_single_relu",
    "vec_single_not",
    // vector_scalar
    "vec_single_muls",
    "vec_single_adds",
    // Mov
    "broadcast",
    // vector_cast
    "vec_single_cast",
    "vec_single_floor",
    "vec_single_round",
    "vec_single_ceil",
    "vec_single_trunc",
  };
};
Stmt SplitTail(Stmt stmt) { return TailSpliter().Mutate(stmt); }

}  // namespace ir
}  // namespace akg