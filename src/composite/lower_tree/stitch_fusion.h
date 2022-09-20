/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef STITCH_FUSION_H_
#define STITCH_FUSION_H_
#define GLOBAL "global"
#define LOCAL_L1 "local_L1"
#include "composite/utils/util.h"

namespace akg {
enum class StorageType { Shared, Global, Unknown };
enum class StitchOpType { Unknown = -1, Elem, Broadcast, Reduce2D_X, All_Reduce, Reduce2D_Y };

struct StitchBufferInfo {
  std::string name;
  StorageType type{StorageType::Unknown};
  std::string buf_name;
  uint64_t alloc_size = 0;
  Type dtype;
};
inline std::ostream &operator<<(std::ostream &os, const StitchBufferInfo &x) {
  os << "StitchBufferInfo: \n"
     << "name: " << x.name << "\n"
     << "type: " << static_cast<int>(x.type) << "\n"
     << "buf_name: " << x.buf_name << "\n"
     << "alloc_size: " << x.alloc_size << "\n";
  return os;
}

struct StitchAttrInfo {
  Expr broadcast_size;
  std::vector<StitchOpType> type_array;
};

struct IrAttrInfo {
  GridBlockDims dims;
  int grid_dims;
  int block_dims;
  Map<std::string, NodeRef> attrs;
  Expr broadcast_size{0};
  Expr elemwise_size{0};
};

class BufferStitchAttr : public GridBlockDimsAttr {
 public:
  explicit BufferStitchAttr(const std::function<Stmt(const StringImm *, const Map<std::string, NodeRef> &, bool, bool,
                                                     bool, std::vector<size_t> &)> &f)
      : func_(f){};

  void SetStitchType(const StitchOpType &stitch_type) {
    stitch_type_ = stitch_type > stitch_type_ ? stitch_type : stitch_type_;
  }
  void GetBufferStitchAttr(const Expr &json, std::vector<OpDesc> op_v, const Map<std::string, NodeRef> &attrs,
                           bool poly, bool fold_dim) {
    const StringImm *json_str = nullptr;
    for (auto &op : op_v) {
      auto out_shape = op.output_tensor_info[0].shape_;
      auto out_size = GetShapeSize(out_shape);
      if (IsReduce(op.op_name)) {
        json_str = json.as<StringImm>();
        CHECK_EQ(op.input_tensor_info.size(), 1) << "Number of Input for Reduce op should be only one.";
        auto reduce_axis = Downcast<Array<Integer>>(op.attrs["axis"]);
        bool reduce_inner = reduce_axis.empty();
        auto innermost_axis = static_cast<int>(op.input_tensor_info[0].shape_.size() - 1);
        for (auto &axis : reduce_axis) {
          if (static_cast<int>(axis) == innermost_axis || static_cast<int>(axis) == -1) {
            reduce_inner = true;
            break;
          }
        }
        if (reduce_inner) {
          if (Equal(out_size, 1)) {
            SetStitchType(StitchOpType::All_Reduce);
          } else {
            SetStitchType(StitchOpType::Reduce2D_X);
          }
        } else {
          // despite broadcast op or elemwise op following reduce, reduce outer axis will be regarded as Reduce_Elem
          SetStitchType(StitchOpType::Reduce2D_Y);
        }
        break;
      }

      if (IsElemwise(op.op_name)) {
        for (auto &input : op.input_tensor_info) {
          if (!input.has_value_) {
            if (!EqualShape(input.shape_, out_shape)) {
              auto input_size = GetShapeSize(input.shape_);
              CHECK(!Equal(input_size, 0));
              broadcast_size = out_size / input_size;
              SetStitchType(StitchOpType::Broadcast);
              break;
            } else {
              elemwise_size = out_size;
              SetStitchType(StitchOpType::Elem);
            }
          }
        }
      }
    }
    if (json_str) {
      auto stmt = func_(json_str, attrs, poly, true, fold_dim, split_index);
      Visit(stmt);
    }
  }
  static Expr GetShapeSize(Array<Expr> &shape) {
    Expr size{1};
    for (const auto &item : shape) {
      size *= item;
    }
    return size;
  }

 public:
  const std::function<Stmt(const StringImm *, const Map<std::string, NodeRef> &, bool, bool, bool,
                           std::vector<size_t> &)>
    func_;
  Expr broadcast_size;
  Expr elemwise_size;
  StitchOpType stitch_type_{StitchOpType::Unknown};
  std::vector<size_t> split_index;
};

class GetGpuMutateInfo : public IRVisitor {
 public:
  GetGpuMutateInfo(const std::vector<Stmt> &stitch_irs) : stitch_irs_(stitch_irs){};
  ~GetGpuMutateInfo() override = default;
  int get_total_block() {
    int total_block_ = 0;
    block_extent_ = 1;
    for (const auto &ir : stitch_irs_) {
      if (total_block_) {
        block_extent_ = 1;
      }
      IRVisitor::Visit(ir);
      if (!total_block_) {
        total_block_ = block_extent_;
      }
      if (block_extent_ != 1) {
        if (total_block_ != block_extent_) LOG(INFO) << "GridDim between splitted irs should be the same";
      }
    }
    return total_block_;
  }
  std::unordered_map<std::string, Region> get_buffer_region_map() {
    for (const auto &ir : stitch_irs_) {
      IRVisitor::Visit(ir);
    }
    return buf_region_map_;
  }

 private:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::thread_extent) {
      const auto *iv = op->node.as<air::IterVarNode>();
      CHECK(iv);
      std::string name = iv->var->name_hint;
      if (name.compare(0, BLOCKIDX_LEN, BLOCKIDX) == 0) {
        block_extent_ *= op->value.as<IntImm>()->value;
      }
    }
    IRVisitor::Visit(op->body);
  }

  void Visit_(const Realize *op) final {
    if (op->func.defined()) {
      std::string buffer_name = op->func->func_name();
      if (!buf_region_map_.count(buffer_name)) buf_region_map_[buffer_name] = op->bounds;
    }
    IRVisitor::Visit(op->body);
  }
  std::vector<Stmt> stitch_irs_;
  int block_extent_;
  std::unordered_map<std::string, Region> buf_region_map_;
};

IrAttrInfo GetIRAttr(StitchOpType type, BufferStitchAttr &stitch_attr_info, std::vector<StitchOpType> &type_array,
                     std::vector<GridBlockDims> &dim_array, const Map<std::string, NodeRef> &attrs);
Stmt StitchFusionAscend(std::vector<Stmt> &stitch_irs, const std::string &kernel_name,
                        std::unordered_map<std::string, NodeRef> &stitch_buffer,
                        const std::unordered_map<std::string, NodeRef> &real_outputs, Array<NodeRef> &workspace_args,
                        Map<Tensor, Buffer> &workspace_binds);
Stmt StitchFusionGPU(std::vector<Stmt> &stitch_irs, const std::string &kernel_name,
                     std::unordered_map<std::string, NodeRef> &stitch_buffer,
                     const std::unordered_map<std::string, NodeRef> &real_outputs, Array<NodeRef> &workspace_args,
                     Map<Tensor, Buffer> &workspace_binds);
}  // namespace akg

#endif  // STITCH_FUSION_H_
