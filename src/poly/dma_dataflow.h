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

using StmtIdHashMap = std::unordered_map<isl::id, std::vector<isl::id>, isl::IslIdIslHash>;
using MemFlow = std::vector<MemType>;
using FlowMap = std::unordered_map<std::string, TensorDataFlow>;
using StateFlowMap = std::map<std::string, StmtDataFlowInfo>;

inline std::ostream &operator<<(std::ostream &os, const StmtIdHashMap &sthash) {
  os << "\nStmtIdHashMap:";
  for (auto i : sthash) {
    os << "\n     stmt_id: "
       << "( " << i.first << " )";
    for (auto j : i.second) {
      os << "\n        tensor: "
         << "( " << j << " )";
    }
  }
  return os;
}

enum STMT_OP_TYPE { MMU_CONV = 1, MMU_GEMM, INST, IM2COL_BUF };
enum TENSOR_DATAFLOW_TYPE {
  MMU_CONV_A = 1,
  MMU_CONV_B,
  MMU_CONV_C,
  MMU_GEMM_A,
  MMU_GEMM_B,
  MMU_GEMM_C,
  IM2COL_C1,
  INST_BUF
};

struct TensorDataFlow {
  std::vector<std::string> name_flow_;
  MemFlow mem_type_flow_;

  void Initial(const std::string &name, const DataFlowAttrs &attrs);
};

class StmtDataFlowInfo {
 public:
  StmtDataFlowInfo(const isl::id &id, bool is_cube) : stmt_id_(id), is_cube_(is_cube) {}
  StmtDataFlowInfo() { is_cube_ = false; }
  ~StmtDataFlowInfo() {}

  void AddReadTensor(const std::string &name, TENSOR_DATAFLOW_TYPE type);
  void AddWriteTensor(const std::string &name, TENSOR_DATAFLOW_TYPE type);

  void CreateTensorDataFlow(TENSOR_DATAFLOW_TYPE type, const std::string &name, TensorDataFlow &dataflow);

  void MmuConvA(const std::string &name, TensorDataFlow &dataflow);
  void MmuConvB(const std::string &name, TensorDataFlow &dataflow);
  void MmuConvC(const std::string &name, TensorDataFlow &dataflow);
  void MmuGEMMA(const std::string &name, TensorDataFlow &dataflow);
  void MmuGEMMB(const std::string &name, TensorDataFlow &dataflow);
  void MmuGEMMC(const std::string &name, TensorDataFlow &dataflow);
  void InstBUF(const std::string &name, TensorDataFlow &dataflow);
  void Im2colC1(const std::string &name, TensorDataFlow &dataflow);

  /******************************************
   *  update all memType BUF to updateType
   ******************************************/
  void UpdateTensorMemType(MemType upateType);

  /******************************************
   * push reads_/writes_ tensor info to nameflow and memflow
   * if tensor exists in map merge two flow:
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
   * if two flow is same, not changed.
   * ****************************************/
  void UpdateFlowInfo(std::map<std::string, std::vector<std::string>> &nameflow,
                      std::map<std::string, MemFlow> &memflow);

  bool SameNameFlow(const std::vector<std::string> &left, const std::vector<std::string> &right);
  bool SameMemFlow(const MemFlow &left, const MemFlow &right) const;

  template <typename T>
  std::vector<T> MergedFlow(const std::vector<T> &left, const std::vector<T> &right);

  isl::id stmt_id_;
  bool is_cube_;
  FlowMap reads_;
  FlowMap writes_;
};

class DMADataFlow {
 public:
  DMADataFlow() {}
  ~DMADataFlow() {}

  void CreateStmtDataFlow(STMT_OP_TYPE opType, const isl::id &stmtId, const StmtOpInfo &stmtOp, StmtIdHashMap &readMap,
                          StmtIdHashMap &writeMap);
  /********************************************
   *  analysis stmt fusion condition
   *  change mem type
   ********************************************/
  void FusionAnalysis();

  /********************************************
   *  get nameflow and memflow of each tensor
   *******************************************/
  void OpDataflowInfo(std::map<std::string, std::vector<std::string>> &nameflow,
                      std::map<std::string, MemFlow> &memflow);

 private:
  /*********************************************
   * The key of OpDataFlow is operator stmt id in schedule tree.
   * S_0, S_1, S_2, S_3, S_4
   *
   *********************************************/
  StateFlowMap op_data_flow_;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_DMA_DATAFLOW_H_
