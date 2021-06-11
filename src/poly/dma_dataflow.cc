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

isl::id GpuDstId(GpuMemType type, isl::id tensor_id) {
  std::string pos_fix = (type == GpuMemType::SHARED ? "_shared" : "_local");
  return isl::id(tensor_id.ctx(), tensor_id.get_name() + pos_fix);
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
isl::id BufferDefInfo::GetIndexDstId(const isl::ctx &ctx, const isl::id &id, const int index) {
  CHECK_GE(index, 0);
  if (index == 0) return id;
  std::string id_name = id.get_name();
  size_t pos = id_name.find("_local_");
  std::string new_id_name = id_name;
  if (pos != std::string::npos) {
    std::stringstream ss;
    ss << id_name.substr(0, pos) << "_promotion_" << index << id_name.substr(pos, id_name.size() - pos);
    new_id_name = ss.str();
  }
  return isl::id(ctx, new_id_name);
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

bool BufferDefInfo::IsBindCopyinDataFlow() { return (data_stream.size() == 2) && (data_stream[1].second == MemType::BUF_C1_); }

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

void TensorDataFlow::Initial(const std::string &name, const DataFlowAttrs &attrs) {
  mem_type_flow_.clear();
  name_flow_.clear();
  for (const auto &one_pair : attrs) {
    mem_type_flow_.push_back(one_pair.first);
    name_flow_.push_back(name + one_pair.second);
  }
}

void StmtDataFlowInfo::AddReadTensor(const std::string &name, TENSOR_DATAFLOW_TYPE type) {
  if (name == "") return;
  if (reads_.find(name) == reads_.end()) {
    TensorDataFlow dataflow;
    CreateTensorDataFlow(type, name, dataflow);
    reads_[name] = dataflow;
  }
}

void StmtDataFlowInfo::AddWriteTensor(const std::string &name, TENSOR_DATAFLOW_TYPE type) {
  if (name == "") return;
  if (writes_.find(name) == writes_.end()) {
    TensorDataFlow dataflow;
    CreateTensorDataFlow(type, name, dataflow);
    writes_[name] = dataflow;
  }
}

void StmtDataFlowInfo::CreateTensorDataFlow(TENSOR_DATAFLOW_TYPE type, const std::string &name,
                                            TensorDataFlow &dataflow) {
  CHECK_NE(name, "");
  switch (type) {
    case TENSOR_DATAFLOW_TYPE::MMU_CONV_A:
      MmuConvA(name, dataflow);
      break;
    case TENSOR_DATAFLOW_TYPE::MMU_CONV_B:
      MmuConvB(name, dataflow);
      break;
    case TENSOR_DATAFLOW_TYPE::MMU_CONV_C:
      MmuConvC(name, dataflow);
      break;
    case TENSOR_DATAFLOW_TYPE::MMU_GEMM_A:
      MmuGEMMA(name, dataflow);
      break;
    case TENSOR_DATAFLOW_TYPE::MMU_GEMM_B:
      MmuGEMMB(name, dataflow);
      break;
    case TENSOR_DATAFLOW_TYPE::MMU_GEMM_C:
      MmuGEMMC(name, dataflow);
      break;
    case TENSOR_DATAFLOW_TYPE::INST_BUF:
      InstBUF(name, dataflow);
      break;
    case TENSOR_DATAFLOW_TYPE::IM2COL_C1:
      Im2colC1(name, dataflow);
      break;
    default:
      CHECK(false) << "CreateTensorDataFlow type error!!! ";
  }
}

void StmtDataFlowInfo::MmuConvA(const std::string &name, TensorDataFlow &dataflow) {
  dataflow.Initial(name, Mmu_Conv_A);
}

void StmtDataFlowInfo::MmuConvB(const std::string &name, TensorDataFlow &dataflow) {
  dataflow.Initial(name, Mmu_Conv_B);
}

void StmtDataFlowInfo::MmuConvC(const std::string &name, TensorDataFlow &dataflow) {
  dataflow.Initial(name, Mmu_Conv_C);
}

void StmtDataFlowInfo::MmuGEMMA(const std::string &name, TensorDataFlow &dataflow) {
  std::size_t start1 = name.find(_FRACTAL_C1);
  if (start1 != std::string::npos) {
    // spec gemm A
    std::string head = name.substr(0, start1);
    dataflow.Initial(head, Mmu_Spec_Gemm_A);
    return;
  }
  std::size_t start2 = name.find(LOCAL_C1);
  if (start2 != std::string::npos) {
    // spec gemm A
    std::string head = name.substr(0, start2);
    dataflow.Initial(head, Mmu_Spec_Gemm_A_);
    return;
  }
  dataflow.Initial(name, Mmu_Gemm_A);
  return;
}

void StmtDataFlowInfo::MmuGEMMB(const std::string &name, TensorDataFlow &dataflow) {
  std::size_t start1 = name.find(LOCAL_C1);
  if (start1 != std::string::npos) {
    // spec gemm B
    dataflow.Initial(name, Mmu_Spec_Gemm_B);
    return;
  }
  std::size_t start2 = name.find(_FRACTAL_C1);
  if (start2 != std::string::npos) {
    // spec gemm B
    dataflow.Initial(name, Mmu_Spec_Gemm_B_);
    return;
  }
  dataflow.Initial(name, Mmu_Gemm_B);
  return;
}

void StmtDataFlowInfo::MmuGEMMC(const std::string &name, TensorDataFlow &dataflow) {
  std::size_t start = name.find(LOCAL_BUF);
  if (start != std::string::npos) {
    // spec gemm C
    // this is only for conv fusion condition,
    dataflow.Initial(name, Mmu_Spec_Gemm_C);
  } else {
    dataflow.Initial(name, Mmu_Gemm_C);
  }
}

void StmtDataFlowInfo::InstBUF(const std::string &name, TensorDataFlow &dataflow) { dataflow.Initial(name, Inst_BUF); }
void StmtDataFlowInfo::Im2colC1(const std::string &name, TensorDataFlow &dataflow) {
  dataflow.Initial(name, Im2Col_C1);
}

void StmtDataFlowInfo::UpdateTensorMemType(MemType update_type) {
  for (auto read = reads_.begin(); read != reads_.end(); ++read) {
    for (uint64_t index = 0; index < read->second.mem_type_flow_.size(); ++index) {
      if (read->second.mem_type_flow_[index] == MemType::BUF_) {
        read->second.mem_type_flow_[index] = update_type;
      }
    }
  }
  for (auto write = writes_.begin(); write != writes_.end(); ++write) {
    for (uint64_t index = 0; index < write->second.mem_type_flow_.size(); ++index) {
      if (write->second.mem_type_flow_[index] == MemType::BUF_) {
        write->second.mem_type_flow_[index] = update_type;
      }
    }
  }
}

bool StmtDataFlowInfo::SameNameFlow(const std::vector<std::string> &left, const std::vector<std::string> &right) {
  if (left.size() != right.size()) {
    return false;
  }

  for (uint64_t index = 0; index < left.size(); ++index) {
    if (left[index] != right[index]) return false;
  }

  return true;
}

bool StmtDataFlowInfo::SameMemFlow(const MemFlow &left, const MemFlow &right) const {
  if (left.size() != right.size()) {
    return false;
  }

  for (uint64_t index = 0; index < left.size(); ++index) {
    if (left[index] != right[index]) return false;
  }

  return true;
}

template <typename T>
std::vector<T> StmtDataFlowInfo::MergedFlow(const std::vector<T> &left, const std::vector<T> &right) {
  std::vector<T> result;

  uint64_t left_idx = 0;  // left array index
  uint64_t rightIdx = 0;  // right array index

  while (left_idx < left.size() || rightIdx < right.size()) {
    if (left_idx < left.size() && rightIdx < right.size() && left[left_idx] == right[rightIdx]) {
      result.push_back(left[left_idx]);
      left_idx++;
      rightIdx++;
    } else {
      if (left_idx < left.size()) {
        result.push_back(left[left_idx]);
        left_idx++;
      } else if (rightIdx < right.size()) {
        result.push_back(right[rightIdx]);
        rightIdx++;
      }
    }
  }
  return result;
}

void StmtDataFlowInfo::UpdateFlowInfo(std::map<std::string, std::vector<std::string>> &nameflow,
                                      std::map<std::string, MemFlow> &memflow) {
  for (const auto &read : reads_) {
    if (nameflow.find(read.first) == nameflow.end()) {
      nameflow[read.first] = read.second.name_flow_;
    } else {
      if (!SameNameFlow(nameflow[read.first], read.second.name_flow_)) {
        // merge two name flow
        nameflow[read.first] = MergedFlow(nameflow[read.first], read.second.name_flow_);
      }
    }

    if (memflow.find(read.first) == memflow.end()) {
      memflow[read.first] = read.second.mem_type_flow_;
    } else {
      if (!SameMemFlow(memflow[read.first], read.second.mem_type_flow_)) {
        // merge two mem flow
        memflow[read.first] = MergedFlow(memflow[read.first], read.second.mem_type_flow_);
      }
    }
    CHECK_EQ(nameflow[read.first].size(), memflow[read.first].size());
  }

  CHECK_EQ(nameflow.size(), memflow.size());

  for (const auto &write : writes_) {
    if (nameflow.find(write.first) == nameflow.end()) {
      nameflow[write.first] = write.second.name_flow_;
    } else {
      if (!SameNameFlow(nameflow[write.first], write.second.name_flow_)) {
        // merge two name flow
        nameflow[write.first] = MergedFlow(nameflow[write.first], write.second.name_flow_);
      }
    }

    if (memflow.find(write.first) == memflow.end()) {
      memflow[write.first] = write.second.mem_type_flow_;
    } else {
      if (!SameMemFlow(memflow[write.first], write.second.mem_type_flow_)) {
        // merge two mem flow
        memflow[write.first] = MergedFlow(memflow[write.first], write.second.mem_type_flow_);
      }
    }
    CHECK_EQ(nameflow[write.first].size(), memflow[write.first].size());
  }

  CHECK_EQ(nameflow.size(), memflow.size());
}

void DMADataFlow::CreateStmtDataFlow(STMT_OP_TYPE op_type, const isl::id &stmt_id, const StmtOpInfo &stmt_op,
                                     StmtIdHashMap &read_map, StmtIdHashMap &write_map) {
  /**********************************
   * stmt is classify three type
   * 1.1 cube: conv
   * 1.2 cube: gemm
   * 2.  vector
   *********************************/
  std::string state = stmt_id.get_name();
  if (op_data_flow_.find(state) == op_data_flow_.end()) {
    StmtDataFlowInfo stmtDataflow(stmt_id, stmt_op.isMMU);
    op_data_flow_[state] = stmtDataflow;
  }
  if (op_type == STMT_OP_TYPE::MMU_CONV) {
    // create memflow for A,B,C
    op_data_flow_[state].AddReadTensor(stmt_op.A_, TENSOR_DATAFLOW_TYPE::MMU_CONV_A);
    op_data_flow_[state].AddReadTensor(stmt_op.B_, TENSOR_DATAFLOW_TYPE::MMU_CONV_B);
    op_data_flow_[state].AddWriteTensor(stmt_op.C_, TENSOR_DATAFLOW_TYPE::MMU_CONV_C);
  }

  if (op_type == STMT_OP_TYPE::MMU_GEMM) {
    // create memflow for A,B,C
    op_data_flow_[state].AddReadTensor(stmt_op.A_, TENSOR_DATAFLOW_TYPE::MMU_GEMM_A);
    op_data_flow_[state].AddReadTensor(stmt_op.B_, TENSOR_DATAFLOW_TYPE::MMU_GEMM_B);
    op_data_flow_[state].AddWriteTensor(stmt_op.C_, TENSOR_DATAFLOW_TYPE::MMU_GEMM_C);
  }

  if (op_type == STMT_OP_TYPE::IM2COL_BUF) {
    // create memflow for A, B
    if (read_map.find(stmt_id) != read_map.end()) {
      for (const auto &id : read_map[stmt_id]) {
        if (id.get_name() != "") {
          op_data_flow_[state].AddReadTensor(id.get_name(), TENSOR_DATAFLOW_TYPE::IM2COL_C1);
        }
      }
    }

    //  UB vector write tensors
    if (write_map.find(stmt_id) != write_map.end()) {
      for (const auto &id : write_map[stmt_id]) {
        if (id.get_name() != "") {
          op_data_flow_[state].AddWriteTensor(id.get_name(), TENSOR_DATAFLOW_TYPE::INST_BUF);
        }
      }
    }
  }

  if (op_type == STMT_OP_TYPE::INST) {
    // UB vector read tensors
    if (read_map.find(stmt_id) != read_map.end()) {
      for (const auto &id : read_map[stmt_id]) {
        if (id.get_name() != "") {
          op_data_flow_[state].AddReadTensor(id.get_name(), TENSOR_DATAFLOW_TYPE::INST_BUF);
        }
      }
    }

    // UB vector write tensors
    if (write_map.find(stmt_id) != write_map.end()) {
      for (const auto &id : write_map[stmt_id]) {
        if (id.get_name() != "") {
          op_data_flow_[state].AddWriteTensor(id.get_name(), TENSOR_DATAFLOW_TYPE::INST_BUF);
        }
      }
    }
  }
}

void DMADataFlow::FusionAnalysis() {
  bool has_mmu = false;
  bool mmu_pre_fusion = has_mmu;
  bool mmu_post_fusion = has_mmu;
  int state_num = 0;

  // analysis has cube pre fusion and post fusion
  for (const auto &state : op_data_flow_) {
    if (has_mmu && state_num > 0) mmu_post_fusion = true;

    if (state.second.is_cube_) {
      has_mmu = true;
      if (state_num > 0) mmu_pre_fusion = true;
    }
    state_num++;
  }

  if (!mmu_pre_fusion && !mmu_post_fusion) return;

  bool start_pre_fusion = has_mmu;
  bool start_post_fusion = false;

  for (auto state = op_data_flow_.begin(); state != op_data_flow_.end(); ++state) {
    if (state->second.is_cube_) {
      start_pre_fusion = false;
      start_post_fusion = true;
    }

    if (mmu_pre_fusion && start_pre_fusion) {
      state->second.UpdateTensorMemType(MemType::BUF_C1_);
    }

    if (mmu_post_fusion && start_post_fusion) {
      state->second.UpdateTensorMemType(MemType::BUF_C0_);
    }
  }
}

void DMADataFlow::OpDataflowInfo(std::map<std::string, std::vector<std::string>> &nameflow,
                                 std::map<std::string, MemFlow> &memflow) {
  for (auto state : op_data_flow_) {
    state.second.UpdateFlowInfo(nameflow, memflow);
  }
  CHECK_EQ(nameflow.size(), memflow.size());
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
