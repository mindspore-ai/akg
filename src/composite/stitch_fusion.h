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
#include "composite/util.h"

namespace akg {
enum class StorageType { Shared, Global, Unknown };
enum class StitchOpType { Unknown = -1, Elem, Broadcast, Reduce2D_X, All_Reduce, Reduce2D_Y };

struct StitchBufferInfo {
  std::string name;
  StorageType type{StorageType::Unknown};
  std::string buf_name;
  uint64_t alloc_size = 0;
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

class StitchBufAlloc : public IRVisitor {
 public:
  explicit StitchBufAlloc(air::DataType data_type = Float(32)) : data_type_(data_type){};
  ~StitchBufAlloc() override = default;

  void BufferAllocReuse(const std::vector<Stmt> &stitch_irs, const Map<std::string, Array<NodeRef>> &alloc_map,
                        const Map<std::string, Array<NodeRef>> &reuse_map,
                        const Map<std::string, Array<NodeRef>> &global_stitch,
                        const std::unordered_map<std::string, NodeRef> &outputs2args) {
    for (const auto &ir : stitch_irs) {
      ir_idx_ += 1;
      if (total_block_) {
        block_extent_ = 1;
      }
      IRVisitor::Visit(ir);
      if (!total_block_) {
        total_block_ = block_extent_;
      }

      if (block_extent_ != 1) {
        if (total_block_ != block_extent_) LOG(INFO) << "GridDim between splitted irs should be the same.";
      }
    }

    for (const auto &it : global_stitch) {
      std::string name = it.first;
      if (name == "EMPTY") {
        break;
      }
      auto alloc_info = it.second[0].as<StringImm>()->value;
      auto alloc_size = it.second[1].as<IntImm>()->value;
      std::string ir_var = name;
      StitchBufferInfo info;
      info.name = name;
      info.type = StorageType::Global;
      info.buf_name = ir_var;
      info.alloc_size = alloc_size;
      stitch_buffer_map[ir_var] = info;
    }

    for (const auto &it : alloc_map) {
      std::string name = it.first;
      auto alloc_info = it.second[0].as<StringImm>()->value;
      CHECK_GT(total_block_, 0);
      uint64_t alloc_size_per_block = it.second[1].as<IntImm>()->value / total_block_;
      // first, check whether variable is already in buf_within_op_map. If NOT, do allocation or reuse.
      CHECK(outputs2args.find(name) != outputs2args.end());
      std::string ir_var = outputs2args.at(name).as<BufferNode>()->name;
      std::string shared_name = ir_var + "_shared";
      std::string ir_var_shared;
      if (buf_within_op_map.find(shared_name) != buf_within_op_map.end()) {
        ir_var_shared = shared_name;
      } else {
        allocated_share_size_ += alloc_size_per_block * data_type_.bytes();
        ir_var_shared = ir_var + "_shared_stitch";
      }

      StitchBufferInfo info;
      info.name = name;
      info.type = StorageType::Shared;
      info.buf_name = ir_var_shared;
      info.alloc_size = alloc_size_per_block;
      stitch_buffer_map[ir_var] = info;
      buf_alloc_op_[ir_var] = info;
    }

    for (const auto &it : reuse_map) {
      std::string name = it.first;
      if (name == "EMPTY") {
        break;
      }
      auto alloc_info = it.second[0].as<StringImm>()->value;
      uint64_t alloc_size_per_block = it.second[1].as<IntImm>()->value / total_block_;
      CHECK(outputs2args.find(alloc_info) != outputs2args.end());
      CHECK(outputs2args.find(name) != outputs2args.end());
      auto ir_var_reuse = outputs2args.at(alloc_info).as<BufferNode>()->name;
      auto ir_var = outputs2args.at(name).as<BufferNode>()->name;
      auto alloc_stitch_info = buf_alloc_op_[ir_var_reuse];
      std::string shared_name = ir_var + "_shared";

      if (buf_within_op_map.find(shared_name) != buf_within_op_map.end()) {
        // add replaced buffer into stitch_buffer_map.
        if (allocated_share_size_ >= alloc_size_per_block * data_type_.bytes()) {
          StitchBufferInfo info;
          info.name = name;
          info.type = StorageType::Shared;
          info.buf_name = alloc_stitch_info.buf_name;
          info.alloc_size = alloc_size_per_block;
          stitch_buffer_map[shared_name] = info;
          // remember to reduce allocated_share_size_ due to reuse.
          allocated_share_size_ -= alloc_size_per_block * data_type_.bytes();
          allocate_revoke.push_back(shared_name);
        }
      }
      StitchBufferInfo info;
      info.name = name;
      info.type = StorageType::Shared;
      info.buf_name = alloc_stitch_info.buf_name;
      info.alloc_size = alloc_size_per_block;
      stitch_buffer_map[ir_var] = info;
      buf_alloc_op_.erase(ir_var_reuse);
    }

    if (allocated_share_size_ >= MEM_LIMIT) {
      std::vector<std::pair<std::string, StitchBufferInfo>> reuse_free_map(buf_alloc_op_.begin(), buf_alloc_op_.end());
      std::sort(reuse_free_map.begin(), reuse_free_map.end(),
                [=](std::pair<std::string, StitchBufferInfo> &a, std::pair<std::string, StitchBufferInfo> &b) -> bool {
                  return (a.second.alloc_size > b.second.alloc_size);
                });
      auto overflow_size = allocated_share_size_ - MEM_LIMIT;
      bool cover_overflow_size = false;
      std::string moveout_var;
      for (const auto &iv : reuse_free_map) {
        if (iv.second.alloc_size * data_type_.bytes() >= overflow_size) {
          moveout_var = iv.first;
          cover_overflow_size = true;
        } else {
          break;
        }
      }

      if (cover_overflow_size) {
        auto moveout_info = stitch_buffer_map[moveout_var];
        moveout_info.type = StorageType::Global;
        moveout_info.buf_name = moveout_info.buf_name + "_global";
        stitch_buffer_map[moveout_var] = moveout_info;
      }
      if (!cover_overflow_size) {
        auto covered_size = 0;
        for (const auto &iv : reuse_free_map) {
          auto moveout_info = stitch_buffer_map[iv.first];
          covered_size += iv.second.alloc_size;
          moveout_info.type = StorageType::Global;
          moveout_info.buf_name = moveout_info.buf_name + "_global";
          stitch_buffer_map[iv.first].type = StorageType::Global;
          if (static_cast<uint64_t>(covered_size) >= overflow_size) {
            break;
          }
        }
      }
    }
  }

  void Dump() {
    LOG(INFO) << "=========buf_within_op_map=========: ";
    for (const auto &kv : buf_within_op_map) {
      LOG(INFO) << "EACH : ";
      LOG(INFO) << kv.first;
      LOG(INFO) << kv.second;
    }
    LOG(INFO) << "=========stitch_buffer_map=========:  ";
    for (const auto &kv : stitch_buffer_map) {
      LOG(INFO) << "EACH : ";
      LOG(INFO) << kv.first;
      LOG(INFO) << kv.second;
    }
  }

 public:
  std::unordered_map<std::string, StitchBufferInfo> buf_within_op_map;
  std::unordered_map<std::string, StitchBufferInfo> stitch_buffer_map;
  std::vector<std::string> allocate_revoke;

 private:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::storage_scope && op->value.as<StringImm>()->value == SHARED) {
      gather_shared_ = true;
    }
    if (op->attr_key == air::ir::attr::thread_extent) {
      const auto *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      std::string name = iv->var->name_hint;
      if (name.compare(0, BLOCKIDX_LEN, BLOCKIDX) == 0) {
        block_extent_ *= op->value.as<IntImm>()->value;
      }
    }
    IRVisitor::Visit(op->body);
  }

  void Visit_(const Allocate *op) final {
    if (gather_shared_) {
      const Variable *buf = op->buffer_var.get();
      std::string name = buf->name_hint;
      CHECK_GE(op->constant_allocation_size(), 0) << "allocation size < 0";
      auto size = static_cast<uint64_t>(op->constant_allocation_size());
      allocated_share_size_ += size * op->type.bytes();
      StitchBufferInfo info;
      info.name = name;
      info.type = StorageType::Shared;
      info.buf_name = name;
      info.alloc_size = size;
      buf_within_op_map[name] = info;

      gather_shared_ = false;
    }
    IRVisitor::Visit(op->body);
  }

  air::DataType data_type_;
  std::unordered_map<std::string, StitchBufferInfo> buf_alloc_op_;
  int ir_idx_ = 0;
  int total_block_ = 0;
  int block_extent_ = 1;
  uint64_t allocated_share_size_ = 0;
  bool gather_shared_ = false;
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

IrAttrInfo GetIRAttr(StitchOpType type, BufferStitchAttr &stitch_attr_info, std::vector<StitchOpType> &type_array,
                     std::vector<GridBlockDims> &dim_array, const Map<std::string, NodeRef> &attrs);
Stmt StitchFusionGpu(std::vector<Stmt> &stitch_irs, const std::string &kernel_name, StitchAttrInfo &store_attr,
                     std::unordered_map<std::string, StitchBufferInfo> &stitch_buffer_map,
                     std::unordered_map<std::string, StitchBufferInfo> &buf_within_op_map,
                     std::vector<std::string> &allocate_revoke,
                     const std::unordered_map<std::string, NodeRef> &real_outputs);
}  // namespace akg

#endif  // STITCH_FUSION_H_
