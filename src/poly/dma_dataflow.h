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

namespace akg {
namespace ir {
namespace poly {
class TensorFootprintCluster;
struct TensorDataFlow;
class StmtDataFlowInfo;

enum MemType { DDR = 1, L1_, UB_, L0A_, L0B_, L0C_, UBL0_, UBL1_ };
enum DataStreamIndex { DS_ZERO = 0, DS_FIRST, DS_SECOND, DS_THIRD };

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
  // realize_UB1, realize_UB2, and so on
  std::vector<std::pair<isl::schedule_node, std::shared_ptr<TensorFootprintCluster>>> footprint_cluster_map;
  std::vector<std::pair<isl::schedule_node, std::vector<size_t>>> sizes_map_;

  std::shared_ptr<TensorFootprintCluster> GetFootPrintCluster(const isl::schedule_node &mark_node);
  bool CompareScheduleMarkNode(const isl::schedule_node_mark &mark1, const isl::schedule_node_mark &mark2);
  void AddSize(const isl::schedule_node &node, const std::vector<size_t> &sizes);
  isl::id GetIndexDstId(const isl::ctx &ctx, const isl::id &id, const int index);
  std::vector<std::pair<isl::id, MemType>> MakeDataStream(const isl::id new_dst_id);
  std::vector<std::pair<isl::id, MemType>> PartialDataStream(size_t start_idx);
  std::vector<size_t> TensorSize(const isl::schedule_node &node);

  /* ************************************************
   * A -> A_local_L1 -> A_fractal_L1
   * input is BufferDefInfo about tensor A
   * output is isl::id A_fractal_L1
   * return the next tensor id in dataflow
   **************************************************/
  isl::id NextTensorDstId();
  bool IsCubeCL1Write();
  bool IsPreCubeL1Write();
  bool IsPreCubeTile2Write();
  bool IsGemmDataL12L0();
  bool IsGemmWeightL12L0();
  bool IsIm2col();
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
using DataFlowAttrs = std::vector<std::pair<MemType, std::string>>;

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

enum STMT_OP_TYPE { CUBE_CONV = 1, CUBE_GEMM, VECTOR, IM2COL_UB };
enum TENSOR_DATAFLOW_TYPE {
  CUBE_CONV_A = 1,
  CUBE_CONV_B,
  CUBE_CONV_C,
  CUBE_GEMM_A,
  CUBE_GEMM_B,
  CUBE_GEMM_C,
  IM2COL_L1,
  VECTOR_UB
};

struct TensorDataFlow {
  std::vector<std::string> name_flow_;
  MemFlow mem_type_flow_;

  void Initial(const std::string &name, const DataFlowAttrs &attrs);
};

const DataFlowAttrs Cube_Conv_A = {{MemType::DDR, ""},
                                   {MemType::L1_, "_local_L1"},
                                   {MemType::L1_, "_fractal_L1"},
                                   {MemType::L0A_, "_local_L1_local_L0A"}};
const DataFlowAttrs Cube_Conv_B = {
  {MemType::DDR, ""}, {MemType::L1_, "_local_L1"}, {MemType::L0B_, "_local_L1_local_L0B"}};
const DataFlowAttrs Cube_Conv_C = {
  {MemType::DDR, ""}, {MemType::UB_, "_local_UB"}, {MemType::L0C_, "_local_UB_local_L0C"}};
const DataFlowAttrs Cube_Spec_Gemm_A = {{MemType::L1_, "_fractal_L1"}, {MemType::L0A_, "_fractal_L1_local_L0A"}};
const DataFlowAttrs Cube_Spec_Gemm_A_ = {{MemType::L1_, "_local_L1"}, {MemType::L0A_, "_local_L1_local_L0A"}};
const DataFlowAttrs Cube_Gemm_A = {
  {MemType::DDR, ""}, {MemType::L1_, "_local_L1"}, {MemType::L0A_, "_local_L1_local_L0A"}};
const DataFlowAttrs Cube_Spec_Gemm_B = {{MemType::L1_, ""}, {MemType::L0B_, "_local_L0B"}};
const DataFlowAttrs Cube_Spec_Gemm_B_ = {{MemType::L1_, ""}, {MemType::L0B_, "_local_L0B"}};
const DataFlowAttrs Cube_Gemm_B = {
  {MemType::DDR, ""}, {MemType::L1_, "_local_L1"}, {MemType::L0B_, "_local_L1_local_L0B"}};
const DataFlowAttrs Cube_Spec_Gemm_C = {{MemType::UBL0_, ""}, {MemType::L0C_, "_local_L0C"}};
const DataFlowAttrs Cube_Gemm_C = {
  {MemType::DDR, ""}, {MemType::UB_, "_local_UB"}, {MemType::L0C_, "_local_UB_local_L0C"}};
const DataFlowAttrs Vector_UB = {{MemType::DDR, ""}, {MemType::UB_, "_local_UB"}};
const DataFlowAttrs Im2Col_L1 = {{MemType::DDR, ""}, {MemType::L1_, "_local_L1"}};

class StmtDataFlowInfo {
 public:
  StmtDataFlowInfo(const isl::id &id, bool is_cube) : stmt_id_(id), is_cube_(is_cube) {}
  StmtDataFlowInfo() { is_cube_ = false; }
  ~StmtDataFlowInfo() {}

  void AddReadTensor(const std::string &name, TENSOR_DATAFLOW_TYPE type);
  void AddWriteTensor(const std::string &name, TENSOR_DATAFLOW_TYPE type);

  void CreateTensorDataFlow(TENSOR_DATAFLOW_TYPE type, const std::string &name, TensorDataFlow &dataflow);

  void CubeConvA(const std::string &name, TensorDataFlow &dataflow);
  void CubeConvB(const std::string &name, TensorDataFlow &dataflow);
  void CubeConvC(const std::string &name, TensorDataFlow &dataflow);
  void CubeGEMMA(const std::string &name, TensorDataFlow &dataflow);
  void CubeGEMMB(const std::string &name, TensorDataFlow &dataflow);
  void CubeGEMMC(const std::string &name, TensorDataFlow &dataflow);
  void VectorUB(const std::string &name, TensorDataFlow &dataflow);
  void Im2colL1(const std::string &name, TensorDataFlow &dataflow);

  /******************************************
   *  update all memType UB to updateType
   ******************************************/
  void UpdateTensorMemType(MemType upateType);

  /******************************************
   * push reads_/writes_ tensor info to nameflow and memflow
   * if tensor exists in map merge two flow:
   * conv pre fusion case
   *      DDR -> UBL1
   *      DDR -> L1 -> L1 -> L0A
   * merged to:
   *      DDR -> UBL1 -> L1 -> L1 -> L0A
   * conv post fusion case
   *      DDR -> UBL0 -> L0C
   *      DDR -> UBL0
   * merged to:
   *      DDR -> UBL0 -> L0C
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
