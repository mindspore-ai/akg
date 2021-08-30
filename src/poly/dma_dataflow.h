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
#ifndef POLY_DMA_DATAFLOW_H_
#define POLY_DMA_DATAFLOW_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <string>
#include "ir_pass.h"
#include "poly/isl.h"
#include "poly/stmt_parse.h"
#include "poly/dsa_utils.h"
namespace akg {
namespace ir {
namespace poly {
class TensorFootprintCluster;
struct TensorDataFlow;
class StmtDataFlowInfo;

enum DataStreamIndex { DS_ZERO = 0, DS_FIRST, DS_SECOND, DS_THIRD };
enum GpuMemType { SHARED = 0, LOCAL };

isl::id GpuDstId(GpuMemType type, isl::id tensor_id);

struct BufferDefInfo {
  isl::id tensor_id;
  isl::id dst_tensor_id;
  isl::id ancester_tensor_id;
  MemType mem_type;
  std::string mark_tag;
  bool find_buffer;
  bool is_bind_tensor;

  std::vector<std::pair<isl::id, MemType>> data_stream;
  Tensor tensor;
  Type data_type;
  std::vector<size_t> sizes;

  std::shared_ptr<TensorFootprintCluster> footprints_cluster;
  isl::union_map outer_schedule;
  // solve same mark tag problem
  // temp solution
  // better solution need init schedule tree has different mark tag
  // realize_BUF1, realize_BUF2, and so on
  std::vector<std::pair<isl::schedule_node, std::shared_ptr<TensorFootprintCluster>>> footprint_cluster_map;
  std::vector<std::pair<isl::schedule_node, std::vector<size_t>>> sizes_map_;

  std::shared_ptr<TensorFootprintCluster> GetFootPrintCluster(const isl::schedule_node &mark_node);
  std::shared_ptr<TensorFootprintCluster> GetFootPrintClusterGPU(const isl::schedule_node &node);
  bool CompareScheduleMarkNode(const isl::schedule_node_mark &mark1, const isl::schedule_node_mark &mark2);
  void AddSize(const isl::schedule_node &node, const std::vector<size_t> &sizes);
  isl::id GetIndexDstId(const isl::ctx &ctx, const isl::id &id, const int index);
  std::vector<std::pair<isl::id, MemType>> MakeDataStream(const isl::id new_dst_id);
  std::vector<std::pair<isl::id, MemType>> PartialDataStream(size_t start_idx);
  std::vector<size_t> TensorSize(const isl::schedule_node &node);

  /* ************************************************
   * A -> A_local_C1 -> A_fractal_C1
   * input is BufferDefInfo about tensor A
   * output is isl::id A_fractal_C1
   * return the next tensor id in dataflow
   **************************************************/
  isl::id NextTensorDstId();
  bool IsMmuCC1Write();
  bool IsPreMmuC1Write();
  bool IsPreMmuTile2Write();
  bool IsGemmDataC12C0();
  bool IsGemmWeightC12C0();
  bool IsIm2col();
  bool IsBindCopyinDataFlow();
  MemType SrcMemType();
  MemType DstMemType();
};

inline std::ostream &operator<<(std::ostream &os, const BufferDefInfo &def_info) {
  os << "\nBufferDefInfo:"
     << "\n tensor_id: " << def_info.tensor_id << "\n dst_tensor_id: " << def_info.dst_tensor_id
     << "\n ancester_tensor_id: " << def_info.ancester_tensor_id
     << "\n mem_mype: " << static_cast<int>(def_info.mem_type) << "\n mark_tag: " << def_info.mark_tag
     << "\n find_buffer: " << (def_info.find_buffer ? "true" : "false")
     << "\n is_bind_tensor: " << (def_info.is_bind_tensor ? "true" : "false") << "\n datastream: ";
  for (auto i : def_info.data_stream) {
    os << "\n     (id,Memtype): "
       << "( " << i.first << ", " << static_cast<int>(i.second);
  }
  os << ")"
     << "\n tensor: " << def_info.tensor << "\n datatype: " << def_info.data_type << "\n sizes: [ ";
  for (auto i : def_info.sizes) {
    os << i << " ";
  }
  os << "]"
     << "\n footprint_cluster: " << def_info.footprints_cluster << "\n outer_schedule: " << def_info.outer_schedule
     << "\n footprint_cluster item: " << def_info.footprint_cluster_map.size();

  return os;
}


enum STMT_OP_TYPE { MMU_CONV = 1, MMU_GEMM, INST, IM2COL_BUF, MMU_SPEC_GEMM };

using StmtIdHashMap = std::unordered_map<isl::id, std::vector<isl::id>, isl::IslIdIslHash>;
using MemFlow = std::vector<MemType>;
using TensorDF = std::vector<std::pair<MemType, std::string>>;
using TensorDfMap = std::unordered_map<std::string, TensorDF>;
struct StmtDataFlow {
  TensorDfMap read;
  TensorDfMap write;
};
struct CmpByStmtOrder {
  bool operator()(const std::string &a, const std::string &b) {
    return a.length() < b.length() || (a.length() == b.length() && a < b);
  }
};
using OpDataFlow = std::map<std::string, StmtDataFlow, CmpByStmtOrder>;

/******************************************
 * conv pre fusion case
 *      DDR -> BUFC1
 *      DDR -> C1 -> C1 -> C0A
 * merged to:
 *      DDR -> BUFC1 -> C1 -> C1 -> C0A
 * conv post fusion case
 *      DDR -> BUFC0 -> C0C
 *      DDR -> BUFC0
 * merged to:
 *      DDR -> BUFC0 -> C0C
 * ****************************************/
class DataFlow {
 public:
  DataFlow(const DataFlow &) = delete;
  DataFlow &operator=(const DataFlow &) = delete;
  virtual ~DataFlow() = default;

  void AddFlow(const std::string &stmt_id, const std::string &tesnor, bool read, const DataFlowAttrs &flow,
               std::string buffer_prefix = "");
  OpDataFlow &GetOpFlow();
  std::unordered_map<std::string, TensorDF> GetCombinedFlow();
  std::pair<std::map<std::string, MemFlow>, std::map<std::string, std::vector<std::string>>> ExtractCombinedFlow();
  void SetMmuFlow(const std::string &stmt_id);
  StmtDataFlow &GetStmtFlow(const std::string &stmt_id);
  std::string GetMmuId();
  void Print();
  void Clear();

  static DataFlow &Get() {
    static DataFlow instance;
    return instance;
  };

 private:
  DataFlow() = default;
  OpDataFlow op_data_flow_;
  std::string mmu_stmt_id_;
};

void DispatchDataFlow(STMT_OP_TYPE op_type, const isl::id &stmt_id, const StmtOpInfo &stmt_op, StmtIdHashMap &read_map,
                      StmtIdHashMap &write_map);
void FusionAnalysis();

void UpdateMemType(std::string stmt_id, MemType type);
int PreFusionAnalysis(const std::string &target);

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_DMA_DATAFLOW_H_
