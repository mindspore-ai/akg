/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "poly/dma_dataflow.h"
#include "poly/poly_util.h"

namespace akg {
namespace ir {
namespace poly {

isl::id GetNpuIndexDstId(const isl::ctx &ctx, const isl::id &id, const int index) {
  CHECK_GE(index, 0);
  if (index == 0) return id;
  std::string id_name = id.get_name();
  size_t pos = id_name.find("_local_");
  std::string new_id_name = id_name;
  if (pos != std::string::npos) {
    std::stringstream ss;
    ss << id_name.substr(0, pos) << PROMOTION_INFIX << index << id_name.substr(pos, id_name.size() - pos);
    new_id_name = ss.str();
  }
  return isl::id(ctx, new_id_name);
}

isl::id GetGpuIndexDstId(const GpuMemType &type, const isl::id &id, const int index) {
  std::string pos_fix = (type == GpuMemType::SHARED ? SHARE_SUFFIX : LOCAL_SUFFIX);
  if (index == 0) {
    return isl::id(id.ctx(), id.get_name() + pos_fix);
  }
  return isl::id(id.ctx(), id.get_name() + PROMOTION_INFIX + std::to_string(index) + pos_fix);
}

bool BufferDefInfo::CompareScheduleMarkNode(const isl::schedule_node_mark &mark1,
                                            const isl::schedule_node_mark &mark2) {
  return (mark1.get_id().get_name() == mark2.get_id().get_name());
}

std::shared_ptr<TensorFootprintCluster> BufferDefInfo::GetFootPrintCluster(const isl::schedule_node &mark_node) {
  for (const auto &key : footprint_cluster_map) {
    if (key.first.isa<isl::schedule_node_mark>() && mark_node.isa<isl::schedule_node_mark>() &&
        CompareScheduleMarkNode(key.first.as<isl::schedule_node_mark>(), mark_node.as<isl::schedule_node_mark>())) {
      isl::union_map umap1 = LocalSchedule(key.first);
      isl::union_map umap2 = LocalSchedule(mark_node);
      if (umap1.is_equal(umap2)) return key.second;
    }
  }
  /************************
   * This is for conv op im2col foorprint cluster
   * The computation of this foorprint cluster is at realize_L1 mark node
   * and add extension is at realize_L0 mark node
   *************************/
  if (footprint_cluster_map.size() == 1 && mark_node.isa<isl::schedule_node_mark>() &&
      mark_node.as<isl::schedule_node_mark>().get_id().get_name() != REALIZE_BUF) {
    return footprints_cluster;
  }
  return nullptr;
}

std::shared_ptr<TensorFootprintCluster> BufferDefInfo::GetFootPrintClusterGPU(const isl::schedule_node &node) {
  for (const auto &key : footprint_cluster_map) {
    if (key.first.isa<isl::schedule_node_band>() && node.isa<isl::schedule_node_band>()) {
      isl::union_map umap1 = LocalSchedule(key.first);
      isl::union_map umap2 = LocalSchedule(node);
      if (umap1.is_equal(umap2)) return key.second;
    }
  }
  return nullptr;
}

std::vector<size_t> BufferDefInfo::TensorSize(const isl::schedule_node &mark_node) {
  std::vector<size_t> res;
  /********************************************
   * some time the markNode is not the supported mark node in schedule tree
   * batch_matmul case 2
   ********************************************/
  if (sizes_map_.size() == 1) {
    return sizes;
  }

  for (const auto &key : sizes_map_) {
    if (key.first.isa<isl::schedule_node_mark>() && mark_node.isa<isl::schedule_node_mark>() &&
        CompareScheduleMarkNode(key.first.as<isl::schedule_node_mark>(), mark_node.as<isl::schedule_node_mark>())) {
      isl::union_map map1 = LocalSchedule(key.first);
      isl::union_map map2 = LocalSchedule(mark_node);
      if (map1.is_equal(map2)) return key.second;
    }
  }
  return res;
}

std::vector<std::pair<isl::id, MemType>> BufferDefInfo::MakeDataStream(const isl::id new_dst_id) {
  std::vector<std::pair<isl::id, MemType>> dataStream;
  for (const auto &item : data_stream) {
    if (item.first.get_name() == dst_tensor_id.get_name()) {
      dataStream.emplace_back(std::make_pair(new_dst_id, item.second));
    } else {
      dataStream.push_back(item);
    }
  }
  return dataStream;
}

std::vector<std::pair<isl::id, MemType>> BufferDefInfo::PartialDataStream(size_t start_idx) {
  std::vector<std::pair<isl::id, MemType>> stream;
  if (start_idx >= data_stream.size()) return stream;
  for (size_t idx = start_idx; idx < data_stream.size(); ++idx) {
    stream.push_back(data_stream[idx]);
  }
  return stream;
}

void BufferDefInfo::AddSize(const isl::schedule_node &node, const std::vector<size_t> &sizes) {
  sizes_map_.emplace_back(std::make_pair(node, sizes));
}

isl::id BufferDefInfo::NextTensorDstId() {
  isl::id result_id = dst_tensor_id;
  if (data_stream.size() > static_cast<size_t>(DataStreamIndex::DS_SECOND)) {
    result_id = data_stream[static_cast<size_t>(DataStreamIndex::DS_SECOND)].first;
  }
  return result_id;
}

bool BufferDefInfo::IsMmuCC1Write() {
  const int l1WriteDFLen = static_cast<int>(DataStreamIndex::DS_THIRD);
  if (data_stream.size() == l1WriteDFLen) {
    if (data_stream[static_cast<size_t>(DataStreamIndex::DS_ZERO)].second == MemType::DDR &&
        data_stream[static_cast<size_t>(DataStreamIndex::DS_FIRST)].second == MemType::BUF_ &&
        data_stream[static_cast<size_t>(DataStreamIndex::DS_SECOND)].second == MemType::C0C_)
      return true;
  }
  return false;
}

bool BufferDefInfo::IsPreMmuC1Write() {
  const int preCubeL1WriteDFLen = static_cast<int>(DataStreamIndex::DS_THIRD);
  if (data_stream.size() == preCubeL1WriteDFLen) {
    if (data_stream[static_cast<size_t>(DataStreamIndex::DS_ZERO)].second == MemType::DDR &&
        data_stream[static_cast<size_t>(DataStreamIndex::DS_FIRST)].second == MemType::C1_ &&
        data_stream[static_cast<size_t>(DataStreamIndex::DS_SECOND)].second == MemType::BUF_C1_)
      return true;
  }
  return false;
}

bool BufferDefInfo::IsPreMmuTile2Write() {
  const int preCubeTile2WriteDFLen = static_cast<int>(DataStreamIndex::DS_SECOND);
  if (data_stream.size() == preCubeTile2WriteDFLen) {
    if (data_stream[static_cast<size_t>(DataStreamIndex::DS_ZERO)].second == MemType::C1_ &&
        data_stream[static_cast<size_t>(DataStreamIndex::DS_FIRST)].second == MemType::BUF_C1_)
      return true;
  }
  return false;
}

bool BufferDefInfo::IsGemmDataC12C0() { return (SrcMemType() == MemType::C1_ && DstMemType() == MemType::C0A_); }

bool BufferDefInfo::IsGemmWeightC12C0() { return (SrcMemType() == MemType::C1_ && DstMemType() == MemType::C0B_); }

bool BufferDefInfo::IsIm2col() { return (SrcMemType() == MemType::C1_ && DstMemType() == MemType::C1_); }

bool BufferDefInfo::IsBindCopyinDataFlow() {
  return (data_stream.size() == 2) && (data_stream[1].second == MemType::BUF_C1_);
}

MemType BufferDefInfo::SrcMemType() {
  // tensor dataflow at least one data
  CHECK_GE(data_stream.size(), 1);
  return data_stream[0].second;
}

MemType BufferDefInfo::DstMemType() {
  // tensor dataflow at least one data
  CHECK_GE(data_stream.size(), 1);
  if (data_stream.size() >= 2) return data_stream[1].second;
  // the last tensor in dataflow, memType is DDR
  return MemType::DDR;
}

void DispatchDataFlow(STMT_OP_TYPE op_type, const isl::id &stmt_id, const StmtOpInfo &stmt_op, StmtIdHashMap &read_map,
                      StmtIdHashMap &write_map) {
  auto GenerateDf = [](const isl::id &stmt_id, StmtIdHashMap &rw_map, bool rw, const DataFlowAttrs &attr) {
    if (rw_map.find(stmt_id) != rw_map.end()) {
      for (const auto &id : rw_map[stmt_id]) {
        if (id.get_name() != "") {
          DataFlow::Get().AddFlow(stmt_id.get_name(), id.get_name(), rw, attr);
        }
      }
    }
  };
  auto GenerateVetorDf = [&]() {
    GenerateDf(stmt_id, read_map, true, Inst_BUF);
    GenerateDf(stmt_id, write_map, false, Inst_BUF);
  };
  auto GenerateGemmDf = [&]() {
    DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.A_, true, Mmu_Gemm_A);
    DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.B_, true, Mmu_Gemm_B);
    DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.C_, false, Mmu_Gemm_C);
    DataFlow::Get().SetMmuFlow(stmt_id.get_name());
  };
  auto GenerateConvDf = [&]() {
    DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.A_, true, Mmu_Conv_A);
    DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.B_, true, Mmu_Conv_B);
    DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.C_, false, Mmu_Conv_C);
    DataFlow::Get().SetMmuFlow(stmt_id.get_name());
  };

  auto GenerateSpecGemmDf = [&]() {
    auto GetPrefixByPostfix = [&](const std::string &buffer_name, const char *const postfix) {
      std::size_t pos = buffer_name.find(postfix);
      if (pos != std::string::npos) {
        return buffer_name.substr(0, pos);
      }
      return std::string("");
    };
    std::string prefix = GetPrefixByPostfix(stmt_op.A_, _FRACTAL_C1);
    if (prefix != "") {
      DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.A_, true, Mmu_Spec_Gemm_A, prefix);
    } else {
      prefix = GetPrefixByPostfix(stmt_op.A_, LOCAL_C1);
      CHECK_NE(prefix, "");
      DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.A_, true, Mmu_Spec_Gemm_A_, prefix);
    }
    DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.B_, true, Mmu_Spec_Gemm_B);
    DataFlow::Get().AddFlow(stmt_id.get_name(), stmt_op.C_, false, Mmu_Spec_Gemm_C);
    DataFlow::Get().SetMmuFlow(stmt_id.get_name());
  };
  auto GenerateIm2colDf = [&]() {
    GenerateDf(stmt_id, read_map, true, Im2Col_C1);
    GenerateDf(stmt_id, write_map, false, Inst_BUF);
  };

  const std::map<STMT_OP_TYPE, std::function<void(void)>> dispatch_map = {
    {STMT_OP_TYPE::INST, GenerateVetorDf},
    {STMT_OP_TYPE::MMU_GEMM, GenerateGemmDf},
    {STMT_OP_TYPE::MMU_CONV, GenerateConvDf},
    {STMT_OP_TYPE::IM2COL_BUF, GenerateIm2colDf},
    {STMT_OP_TYPE::MMU_SPEC_GEMM, GenerateSpecGemmDf}};

  if (dispatch_map.find(op_type) != dispatch_map.end()) {
    dispatch_map.find(op_type)->second();
  }
}

void DataFlow::Print() {
  auto PrintTensorDF = [](const TensorDfMap &df) {
    for (auto it : df) {
      auto tensor = it.first;
      std::cout << tensor << ": " << std::endl;
      for (auto f_it : it.second) {
        std::cout << f_it.first << "," << f_it.second << std::endl;
      }
    }
  };
  for (auto s_it : op_data_flow_) {
    auto stmt = s_it.first;
    std::cout << stmt << ":" << std::endl;
    auto reads = s_it.second.read;
    auto writes = s_it.second.write;
    std::cout << "read: " << std::endl;
    PrintTensorDF(reads);
    std::cout << "write: " << std::endl;
    PrintTensorDF(writes);
  }
}

void DataFlow::AddFlow(const std::string &stmt_id, const std::string &tensor, bool read, const DataFlowAttrs &flow,
                       std::string buffer_prefix) {
  if (buffer_prefix == "") {
    buffer_prefix = tensor;
  }
  auto NamedFlow = [&](const DataFlowAttrs &flow) {
    DataFlowAttrs res;
    for (auto it : flow) {
      res.push_back({it.first, buffer_prefix + it.second});
    }
    return res;
  };
  if (read) {
    op_data_flow_[stmt_id].read[tensor] = NamedFlow(flow);
  } else {
    op_data_flow_[stmt_id].write[tensor] = NamedFlow(flow);
  }
}
std::pair<std::map<std::string, MemFlow>, std::map<std::string, std::vector<std::string>>>
DataFlow::ExtractCombinedFlow() {
  auto combined_flow = GetCombinedFlow();
  std::map<std::string, std::vector<std::string>> tensor_name_flows;
  std::map<std::string, MemFlow> tensor_mem_flows;
  for (auto &flow : combined_flow) {
    for (auto &flow_type : flow.second) {
      tensor_mem_flows[flow.first].push_back(flow_type.first);
      tensor_name_flows[flow.first].push_back(flow_type.second);
    }
  }
  this->Clear();
  return {tensor_mem_flows, tensor_name_flows};
}

void FusionAnalysis() {
  std::string mmu_stmt_id = DataFlow::Get().GetMmuId();
  int s_count = DataFlow::Get().GetOpFlow().size();
  if (s_count <= 1 || mmu_stmt_id.empty()) return;
  auto pre_count = PreFusionAnalysis(mmu_stmt_id);
  if (pre_count + 1 == s_count) return;
  for (auto &s : DataFlow::Get().GetOpFlow()) {
    auto stmt_id = s.first;
    UpdateMemType(stmt_id, BUF_C0_);
  }
}

void UpdateMemType(std::string stmt_id, MemType type) {
  auto UpdateTensorMemType = [&](TensorDfMap &rw_map) {
    for (auto &tensor_df : rw_map) {
      for (auto &flow : tensor_df.second) {
        if (flow.first == MemType::BUF_) {
          flow.first = type;
        }
      }
    }
  };
  auto &stmt_flow = DataFlow::Get().GetStmtFlow(stmt_id);
  UpdateTensorMemType(stmt_flow.read);
  UpdateTensorMemType(stmt_flow.write);
}

int PreFusionAnalysis(const std::string &target) {
  auto GetReadTensor = [](const std::string &target) {
    std::unordered_set<std::string> tensors;
    for (auto &it : DataFlow::Get().GetStmtFlow(target).read) {
      tensors.insert(it.first);
    }
    return tensors;
  };
  auto FindTensor = [](TensorDfMap &write_map, std::unordered_set<std::string> &read_set) {
    for (auto w : write_map) {
      if (read_set.find(w.first) != read_set.end()) {
        return true;
      };
    }
    return false;
  };

  int res = 0;
  auto rtensors = GetReadTensor(target);
  for (auto &s : DataFlow::Get().GetOpFlow()) {
    if (s.first == DataFlow::Get().GetMmuId()) continue;
    auto writes = s.second.write;
    auto stmt_id = s.first;
    if (FindTensor(writes, rtensors)) {
      UpdateMemType(s.first, MemType::BUF_C1_);
      res += PreFusionAnalysis(s.first) + 1;
    }
  }
  return res;
}

std::unordered_map<std::string, TensorDF> DataFlow::GetCombinedFlow() {
  TensorDfMap res;
  auto CombineFlow = [&](TensorDfMap src) {
    auto MergedFlow = [](const TensorDF &left, const TensorDF &right) {
      TensorDF result = left;
      uint64_t i = 0;
      while (i < right.size()) {
        if (i >= left.size() || left[i].first != right[i].first) {
          result.push_back(right[i]);
        }
        i = i + 1;
      }
      return result;
    };
    for (auto it : src) {
      auto tensor_name = it.first;
      auto tensor_flow = it.second;
      if (res.find(tensor_name) == res.end()) {
        res[tensor_name] = tensor_flow;
      } else {
        res[tensor_name] = MergedFlow(res[tensor_name], tensor_flow);
      }
    }
  };
  for (auto it : op_data_flow_) {
    auto &tensor_df = it.second;
    CombineFlow(tensor_df.read);
    CombineFlow(tensor_df.write);
  }
  return res;
}

void DataFlow::SetMmuFlow(const std::string &stmt_id) { mmu_stmt_id_ = stmt_id; }

std::string DataFlow::GetMmuId() { return mmu_stmt_id_; }

StmtDataFlow &DataFlow::GetStmtFlow(const std::string &stmt_id) { return op_data_flow_[stmt_id]; }

OpDataFlow &DataFlow::GetOpFlow() { return op_data_flow_; }

void DataFlow::Clear() {
  op_data_flow_.clear();
  mmu_stmt_id_.clear();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
