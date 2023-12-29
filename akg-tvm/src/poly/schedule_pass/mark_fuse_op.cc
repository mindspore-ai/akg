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

#include "mark_fuse_op.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule MarkFuseOp::Run(isl::schedule schedule_mark) {
  auto fn = [&](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      std::string mark_id = node.as<isl::schedule_node_mark>().get_id().get_name();
      size_t pos = mark_id.find(UBL0);
      if (pos != std::string::npos) {
        if (IsMatmulPadding()) {
          node = MakeMatmulPaddingFuseOp(node);
        }
        std::string m = FUSE_VECTOR;
        node = node.insert_mark(isl::id(node.ctx(), m));
        node = node.parent();
      }
    }
    return node;
  };
  return schedule_mark.get_root().map_descendant_bottom_up(fn).get_schedule();
}

/* ******************************************************************************
 * Find new position in the schedule tree for matmul padding condition
 * between the filter node and the band node ahead of "realize_BUFC0" mark node
 * ******************************************************************************/
isl::schedule_node MarkFuseOp::MakeMatmulPaddingFuseOp(const isl::schedule_node &node) {
  isl::schedule_node new_node = node;
  isl::schedule_node cur_node = node;
  if (!cur_node.isa<isl::schedule_node_mark>()) {
    return new_node;
  }
  while (cur_node.has_parent()) {
    if (cur_node.parent().isa<isl::schedule_node_filter>() && cur_node.isa<isl::schedule_node_band>()) {
      new_node = cur_node;
      break;
    }
    cur_node = cur_node.parent();
  }
  return new_node;
}

bool MarkFuseOp::IsMatmulPadding() {
  if (scop_info_.mmu_info_.IsGemm()) {
    auto accul_mul = [](Array<Expr> shape_fractal) -> Expr {
      size_t sizes = shape_fractal.size();
      Array<Expr> shapes_ndim = shape_fractal;
      const int npu_fractal_all_ndim = 4;
      const int npu_fractal_basic_ndim = 2;
      if (shape_fractal.size() >= npu_fractal_all_ndim) {
        /* convert fractal shape to two dimesion shape
         * [2,3,16,16] -> [32,48]
         * ***********************************************/
        for (size_t i = 0; i < sizes - npu_fractal_basic_ndim; ++i) {
          if (i < sizes - npu_fractal_all_ndim) {
            shapes_ndim.push_back(shape_fractal[i]);
          } else {
            shapes_ndim.push_back(Simplify(shape_fractal[i] * shape_fractal[i + npu_fractal_basic_ndim]));
          }
        }
      }
      Expr res = Expr(0);
      for (auto shape : shapes_ndim) {
        res = res + shape;
      }
      return res;
    };

    auto write_map = scop_info_.StmtWriteMap();
    auto len = write_map.size();
    isl::id last_id = isl::id(scop_info_.ctx_, std::string("S_") + std::to_string(len - 1));
    isl::id out_tensor_id;
    if (write_map.count(last_id) > 0) {
      out_tensor_id = write_map[last_id][0];
    }
    Tensor mat_out = scop_info_.FindTensor(scop_info_.mmu_info_.GetCName());
    Tensor op_out = scop_info_.FindTensorInOrig(out_tensor_id);
    Expr mat_out_len = Simplify(accul_mul(mat_out->shape));
    Expr out_len = Simplify(accul_mul(op_out->shape));
    if (mat_out_len.as<IntImm>()->value > out_len.as<IntImm>()->value) {
      return true;
    }
  }
  return false;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
