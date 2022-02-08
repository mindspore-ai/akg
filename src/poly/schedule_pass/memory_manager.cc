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

#include "memory_manager.h"
#include "poly/dma_inject.h"
#include "poly/scop_builder.h"
#include "poly/schedule_tree_util.h"
#include "poly/dsa_utils.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule MemoryManager::Run(isl::schedule sch) {
  schedule_ = sch;

  AddStateTensorsDataFlow();
  ReorderBufferedDefInfos();

  auto schedule = sch;
  GetVisitedStmts(schedule.get_root());
  for (size_t index = 0; index < scop_info_.analysis_result_.buffer_def_infos_.size(); index++) {
    if (scop_info_.analysis_result_.buffer_def_infos_[index].find_buffer) continue;
    std::string mark_tag = scop_info_.analysis_result_.buffer_def_infos_[index].mark_tag;
    if (scop_info_.analysis_result_.buffer_def_infos_[index].IsIm2col()) {
      isl::id nextTensorId = scop_info_.analysis_result_.buffer_def_infos_[index].NextTensorDstId();
      mark_tag = scop_info_.analysis_result_.GetBufferDefInfo(nextTensorId).mark_tag;
    }
    schedule = HoistBufferFootprintAtMarkNode(schedule.get_root(), mark_tag, index);
  }
  CHECK_EQ(buffer_footprint_queue_.size(), 0);
  if (scop_info_.user_config_.GetEnableHoistCondWrite()) {
    scop_info_.CollectConditionalWritePromotions();
  }
  return schedule;
}

isl::schedule MemoryManager::HoistBufferFootprintAtMarkNode(const isl::schedule_node &root, const std::string &mark_tag,
                                                            size_t index) {
  auto fn = [mark_tag, index, this](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_mark>()) {
      std::string mark_id = node.as<isl::schedule_node_mark>().get_id().get_name();
      if (mark_id == mark_tag) {
        node = HoistBufferFootprintAtMarkNode(node.get_child(0), index);
      }
    }
    return node;
  };

  return MapDescendantTopDown(root, fn).get_schedule();
}

isl::schedule_node MemoryManager::HoistBufferFootprintAtMarkNode(const isl::schedule_node &tree, size_t index) {
  auto schedule = LocalSchedule(tree);

  // hoist cluster and add extension to schedule tree
  return HoistTensorClusterFootprint(tree, index, schedule);
}

isl::schedule_node MemoryManager::HoistTensorClusterFootprint(isl::schedule_node tree, size_t buffered_fp_idx,
                                                              const isl::union_map &schedule) {
  BufferDefInfo &tensor_info = scop_info_.analysis_result_.buffer_def_infos_[buffered_fp_idx];
  isl::union_map sch_map = scop_info_.analysis_result_.GetScheduleMapBeforeTile();

  isl::schedule_node mark_node = tree;
  if (tree.has_parent()) {
    mark_node = tree.parent();
  }

  isl::id src_tensor_id = tensor_info.tensor_id;
  isl::id dst_tensor_id = tensor_info.dst_tensor_id;
  bool is_bind_tensor = tensor_info.is_bind_tensor;

  auto fp_cluster = tensor_info.GetFootPrintCluster(mark_node);
  if ((fp_cluster == nullptr) || (!fp_cluster->foot_print_.box.is_valid())) {
    LOG(INFO) << "FootprintsClusters: fp_cluster is null or box is invalid! src: " << src_tensor_id
              << ", dst: " << dst_tensor_id;
    return tree;
  }

  auto active_domains = CollectDomain(tree);
  auto active_buf_fp = CollectBufferedFootprints(active_domains, src_tensor_id);
  auto foot_prints = isl::set::empty(fp_cluster->GetSingleAccessRange().get_space());
  auto all_read_only = fp_cluster->UnWriteable();
  for (const auto &buf_fp : active_buf_fp) {
    foot_prints = foot_prints.unite(buf_fp.second.cluster->GetSingleAccessRange());
    all_read_only = all_read_only && buf_fp.second.cluster->UnWriteable();
  }

  if (is_bind_tensor && tensor_info.mem_type != MemType::BUF_C0_) {
    if (!(scop_info_.mmu_info_.IsGemm() && tensor_info.IsMmuCC1Write())) {
      bool insert_buf_to_c1 = false;
      if (!scop_info_.analysis_result_.GetFakeCopyin().is_empty()) {
        scop_info_.analysis_result_.GetFakeCopyin().foreach_map(
          [&insert_buf_to_c1, &src_tensor_id, &dst_tensor_id](const isl::map &m) -> void {
            if ((m.get_tuple_id(isl_dim_out).get_name() == src_tensor_id.get_name()) &&
                (src_tensor_id.get_name() + LOCAL_C1 == dst_tensor_id.get_name())) {
              insert_buf_to_c1 = true;
            }
          });
      }
      if (insert_buf_to_c1) {
        isl::id outer_tensorId = isl::id(src_tensor_id.ctx(), src_tensor_id.get_name() + LOCAL_BUF);
        tree = PlaceInnerDataCopyBelow(scop_info_, tree, *fp_cluster, *fp_cluster, src_tensor_id, dst_tensor_id,
                                       outer_tensorId, sch_map);
      } else {
        tree = PlaceOuterDataCopyBelow(scop_info_, tree, *fp_cluster, src_tensor_id, dst_tensor_id, sch_map,
                                       schedule_.get_domain().get_space());
      }
    } else {
      buffer_footprint_queue_.push(src_tensor_id);
    }
    // If the new buffer_footprint is not a strict subset of any other parent
    auto cluster = std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster));
    scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(
      std::make_pair(active_domains, BufferedFootPrintInfo{cluster, schedule, dst_tensor_id}));
    tensor_info.find_buffer = true;
    return tree;
  }

  if (tensor_info.IsIm2col()) {
    isl::id cluster_id = tensor_info.NextTensorDstId();
    auto l0_fp_cluster = GetFootPrintsCluster(dst_tensor_id);
    CHECK(l0_fp_cluster != nullptr);
    tree = PlaceIm2colBelow(scop_info_, tree, *l0_fp_cluster, *fp_cluster, cluster_id, dst_tensor_id);
    // If the new buffer_footprint is not a strict subset of any other parent
    auto cluster = std::shared_ptr<TensorFootprintCluster>(std::move(l0_fp_cluster));
    scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(
      std::make_pair(active_domains, BufferedFootPrintInfo{cluster, schedule, dst_tensor_id}));
    tensor_info.find_buffer = true;
    SetFindBuffer(dst_tensor_id, true);
    return tree;
  }

  if (tensor_info.IsGemmDataC12C0()) {
    if (scop_info_.mmu_info_.IsGemmDataTranspose()) {
      const isl::id &trans_id = dst_tensor_id;
      const isl::id &cluster_id = dst_tensor_id;
      tree = PlaceGemmTranspose(scop_info_, tree, *gemm_a_transpose_fp_cluster_, *fp_cluster, trans_id, cluster_id);
      scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(
        std::make_pair(active_domains, BufferedFootPrintInfo{gemm_a_transpose_fp_cluster_, schedule, cluster_id}));
    }
  }

  if (tensor_info.IsGemmWeightC12C0()) {
    if (scop_info_.mmu_info_.IsGemmWeightTranspose()) {
      const isl::id &trans_id = dst_tensor_id;
      const isl::id &cluster_id = dst_tensor_id;
      tree = PlaceGemmTranspose(scop_info_, tree, *gemm_b_transpose_fp_cluster_, *fp_cluster, trans_id, cluster_id);
      scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(
        std::make_pair(active_domains, BufferedFootPrintInfo{gemm_b_transpose_fp_cluster_, schedule, cluster_id}));
    }
  }
  auto scop_cluster = fp_cluster;
  if (scop_info_.mmu_info_.IsGemm() && (tensor_info.IsGemmDataC12C0() || tensor_info.IsGemmWeightC12C0())) {
    scop_cluster = scop_info_.analysis_result_.GetBufferDefInfo(tensor_info.tensor_id).footprints_cluster;
  }
  if (tensor_info.IsPreMmuTile2Write()) {
    auto info = scop_info_.analysis_result_.GetBufferDefInfo(tensor_info.tensor_id);
    auto new_scop_group = info.GetFootPrintCluster(mark_node);
    if (new_scop_group != nullptr) {
      scop_cluster = new_scop_group;
    }
  }
  tree = PlaceInnerDataCopyBelow(scop_info_, tree, *fp_cluster, *scop_cluster, src_tensor_id, dst_tensor_id,
                                 src_tensor_id, sch_map);
  if (scop_info_.mmu_info_.IsGemm() && !buffer_footprint_queue_.empty() &&
      buffer_footprint_queue_.front().get_name() == tensor_info.ancester_tensor_id.get_name()) {
    tree = PlaceOuterDataCopyBelow(scop_info_, tree, *fp_cluster, tensor_info.ancester_tensor_id, src_tensor_id,
                                   sch_map, schedule_.get_domain().get_space());
    buffer_footprint_queue_.pop();
  }

  // If the new buffer_footprint is not a strict subset of any other parent
  auto group = std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster));

  scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(
    std::make_pair(active_domains, BufferedFootPrintInfo{group, schedule, dst_tensor_id}));
  tensor_info.find_buffer = true;
  return tree;
}

std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> MemoryManager::CollectBufferedFootprints(
  const isl::union_set &active_domains, const isl::id &tensor_id) const {
  std::vector<std::pair<isl::union_set, BufferedFootPrintInfo>> result;

  for (auto idx : CollectBufferedFootprintsIndexes(active_domains, tensor_id)) {
    result.emplace_back(scop_info_.analysis_result_.active_buffer_footprints_[idx]);
  }
  return result;
}

std::vector<size_t> MemoryManager::CollectBufferedFootprintsIndexes(const isl::union_set &active_domains,
                                                                    const isl::id &tensor_id) const {
  std::vector<size_t> result;

  for (size_t i = 0, e = scop_info_.analysis_result_.active_buffer_footprints_.size(); i < e; ++i) {
    const auto &act_fp = scop_info_.analysis_result_.active_buffer_footprints_[i];
    if (act_fp.first.intersect(active_domains).is_empty()) {
      continue;
    }

    auto cluster_id = act_fp.second.cluster_id;
    for (const auto &def_iter : scop_info_.analysis_result_.BufferDefInfos()) {
      if (def_iter.dst_tensor_id.get_name() == cluster_id.get_name() &&
          def_iter.tensor_id.get_name() == tensor_id.get_name()) {
        result.push_back(i);
        break;
      }
    }
  }
  return result;
}

std::shared_ptr<TensorFootprintCluster> MemoryManager::GetFootPrintsCluster(const isl::id &tensor_id) {
  for (const auto &info : scop_info_.analysis_result_.buffer_def_infos_) {
    if (info.tensor_id.get_name() == tensor_id.get_name()) {
      return info.footprints_cluster;
    }
  }
  return nullptr;
}

// set the findPromote to the given tensor_id in buffered_decl_infos_
// based on tensor_id_
void MemoryManager::SetFindBuffer(const isl::id &tensor_id, bool find_buffer) {
  for (auto &info : scop_info_.analysis_result_.buffer_def_infos_) {
    if (info.tensor_id.get_name() == tensor_id.get_name()) {
      info.find_buffer = find_buffer;
      return;
    }
  }
  LOG(FATAL) << "hosited tensor" << tensor_id << "has no declaration";
}

PartitionSingle::PartitionSingle(int times, int tile_start, int cut_m,
                                 const std::map<std::string, Expr> &fractal_int_info) {
  m_times_ = times;
  m_cut_m_ = cut_m;
  m_fractal_int_info_ = fractal_int_info;
}

PartitionSingle *PartitionSingle::single_ = nullptr;
int PartitionSingle::m_times_ = 0;
int PartitionSingle::m_cut_m_ = 0;
std::map<std::string, Expr> PartitionSingle::m_fractal_int_info_;

void MemoryManager::GatherBufferFootprintDefInfo(const isl::schedule_node &tree, BufferDefInfo &tensor_info) {
  auto fp_cluster = tensor_info.GetFootPrintCluster(tree);
  std::vector<size_t> sizes;
  if (fp_cluster == nullptr) {
    tensor_info.AddSize(tree, sizes);
    return;
  }
  sizes = fp_cluster->GetFixedBoxSizes();

  isl::id tensor_id = tensor_info.tensor_id;
  isl::id cluster_id = tensor_info.dst_tensor_id;

  // build a Halide Node for cluster_id
  Array<Expr> shapes;
  for (auto i : sizes) {
    shapes.push_back(Expr(static_cast<int>(i)));
  }

  Type type = scop_info_.GetDtypeOf(tensor_id);
  Tensor tensor = placeholder(shapes, type, cluster_id.get_name());
  const Buffer buffer = decl_buffer(shapes, scop_info_.GetDtypeOf(tensor_id), cluster_id.get_name());
  scop_info_.user_config_.SetBind(tensor, buffer);

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(tree, sizes);
}

void MemoryManager::CollectBufferFootprintDefInfo(BufferDefInfo &tensor_info, const isl::union_map &schedule_prom,
                                                  const isl::schedule_node &node) {
  auto writes = scop_info_.analysis_result_.GetWrites();
  if (scop_info_.IsInBinds(tensor_info.tensor_id) &&
      scop_info_.IsFunctionalCopyin(tensor_info.tensor_id.name(), scop_info_.StmtBindCopyinMap()) &&
      tensor_info.IsBindCopyinDataFlow()) {
    writes = writes.unite(scop_info_.analysis_result_.GetBindCopyin());
  }
  tensor_info.footprints_cluster = TensorFootprintCluster::HoistBufferFootprintCluster(
    schedule_prom, tensor_info.ancester_tensor_id, scop_info_.analysis_result_.GetReads(),
    scop_info_.analysis_result_.GetCopyin(), writes, scop_info_.analysis_result_.GetFakeCopyin());
  if (tensor_info.footprints_cluster != nullptr) {
    tensor_info.footprint_cluster_map.emplace_back(std::make_pair(node, tensor_info.footprints_cluster));
    GatherBufferFootprintDefInfo(node, tensor_info);
  }
}

void MemoryManager::HoistIm2colBufferFootprintCluster(const isl::union_map &schedule, const isl::schedule_node &node,
                                                      const int index, BufferDefInfo &tensor_info) {
  im2col_fp_cluster = ConstructAffineFpCluster(scop_info_, scop_info_.analysis_result_.GetReads(), schedule.domain(),
                                               schedule, ReferenceType::Read, AffineType::AFFINE_IM2COL);
  tensor_info.footprints_cluster =
    ConstructAffineFpCluster(scop_info_, scop_info_.analysis_result_.GetReads(), schedule.domain(), schedule,
                             ReferenceType::Read, AffineType::AFFINE_FRACTAL);
  CHECK_EQ(index, 0);
  CHECK(im2col_fp_cluster != nullptr) << "im2col_fp_cluster must be not null";
  CHECK(tensor_info.footprints_cluster != nullptr) << "footprint cluster in Im2col must be defined";
  tensor_info.footprint_cluster_map.emplace_back(std::make_pair(node, tensor_info.footprints_cluster));

  if ((tensor_info.footprints_cluster->foot_print_.box.is_valid()) && (im2col_fp_cluster->foot_print_.box.is_valid())) {
    GatherBufferFootprintDefInfo(node, tensor_info);
    // this update info is used for spec gemm
    scop_info_.mmu_info_.UpdateFractalIntFirstInfo(scop_info_.mmu_info_.IsConvBackpropFilter(),
                                                   im2col_fp_cluster->GetFixedBoxSizes(),
                                                   tensor_info.footprints_cluster->GetFixedBoxSizes());
  } else {
    int64_t t_ci = 1;
    int64_t k_h = 0;
    int64_t k_w = 0;
    int64_t t_h = 1;
    int64_t t_w = 1;
    int64_t s_h = 1;
    int64_t s_w = 1;
    int64_t t_ho = 1;
    int64_t t_wo = 1;
    int64_t c_in = 0;
    int64_t block_size = 16;
    LOG(INFO) << "im2col or fractal foot_print_ box is invalid.";

    Map<std::string, NodeRef> attr_info = scop_info_.mmu_info_.GetConvAttrInfo();
    auto it = attr_info.find(ATTR_CONV_KERNEL_H);
    if ((it != attr_info.end()) && (*it).second.as<IntImm>()) k_h = (*it).second.as<IntImm>()->value;
    it = attr_info.find(ATTR_CONV_KERNEL_W);
    if ((it != attr_info.end()) && (*it).second.as<IntImm>()) k_w = (*it).second.as<IntImm>()->value;
    it = attr_info.find(ATTR_CONV_STRIDE_H);
    if ((it != attr_info.end()) && (*it).second.as<IntImm>()) s_h = (*it).second.as<IntImm>()->value;
    it = attr_info.find(ATTR_CONV_STRIDE_W);
    if ((it != attr_info.end()) && (*it).second.as<IntImm>()) s_w = (*it).second.as<IntImm>()->value;
    it = attr_info.find(ATTR_CONV_TILE_H);
    if ((it != attr_info.end()) && (*it).second.as<IntImm>()) t_h = (*it).second.as<IntImm>()->value;
    it = attr_info.find(ATTR_CONV_TILE_W);
    if ((it != attr_info.end()) && (*it).second.as<IntImm>()) t_w = (*it).second.as<IntImm>()->value;
    it = attr_info.find(ATTR_CONV_FEATURE_C);
    if ((it != attr_info.end()) && (*it).second.as<IntImm>()) c_in = (*it).second.as<IntImm>()->value;

    t_ho = (t_h - k_h) / s_h + 1;
    t_wo = (t_w - k_w) / s_w + 1;

    bool replace_ci = false;
    auto dynamic_shape = scop_info_.user_config_.GetDynamicShape();
    if (!dynamic_shape.empty()) {
      for (const auto &ds : dynamic_shape) {
        if (auto dsn = ds.as<air::DynamicShapeNode>()) {
          if (dsn->tensor_name == "CI1") {
            t_ci = (int64_t)(dsn->poly_upper_bound - 1);
            replace_ci = true;
          }
        }
      }
    }
    if (!replace_ci) {
      t_ci = (int64_t)(c_in + block_size - 1) / block_size;
    }

    std::vector<size_t> sizes;
    sizes.push_back(1);                                                      // 1
    sizes.push_back((size_t)((t_ho * t_wo + block_size - 1) / block_size));  // 109
    sizes.push_back((size_t)(t_ci * k_h * k_w));                             // 43648
    sizes.push_back(block_size);                                             // BLOCK_SIZE
    sizes.push_back(block_size);                                             // BLOCK_SIZE
    scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_GMM_M] = t_ho * t_wo;   // 1739
    int tmp_size = 0;
    scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_BATCH] = (int64_t)sizes[tmp_size++];
    scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_TILE_M] = (int64_t)sizes[tmp_size++];
    scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_TILE_K] = (int64_t)sizes[tmp_size++];
    scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_M_INNER] = (int64_t)sizes[tmp_size++];
    scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_K_INNER] = (int64_t)sizes[tmp_size];
    GatherFractalDefInfo(node, tensor_info, sizes);
  }
  scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_FEATURE_W] =
    scop_info_.mmu_info_.ExtractExprFromAttrs(ATTR_CONV_FEATURE_W);
  scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_PAD_LEFT] =
    scop_info_.mmu_info_.ExtractExprFromAttrs(ATTR_CONV_PAD_LEFT);
  scop_info_.mmu_info_.fractal_int_info_[ATTR_CONV_PAD_RIGHT] =
    scop_info_.mmu_info_.ExtractExprFromAttrs(ATTR_CONV_PAD_RIGHT);
}

void MemoryManager::MakeMultiBufferFootprint(const isl::union_map &schedule, const isl::schedule_node &node, int &index,
                                             BufferDefInfo &tensor_info) {
  if (!scop_info_.IsCopyinTensor(tensor_info.ancester_tensor_id.get_name())) {
    CollectBufferFootprintDefInfo(tensor_info, schedule, node);
  } else {
    if (index == 0) {
      CollectBufferFootprintDefInfo(tensor_info, schedule, node);
    } else {
      isl::id new_dst_id = GetNpuIndexDstId(scop_info_.ctx_, tensor_info.dst_tensor_id, index);
      BufferDefInfo new_footprint_info = BufferDefInfo{tensor_info.tensor_id,
                                                       new_dst_id,
                                                       tensor_info.ancester_tensor_id,
                                                       tensor_info.mem_type,
                                                       tensor_info.mark_tag,
                                                       false,
                                                       tensor_info.is_bind_tensor,
                                                       tensor_info.MakeDataStream(new_dst_id),
                                                       Tensor(),
                                                       Handle(),
                                                       tensor_info.sizes,
                                                       nullptr,
                                                       isl::union_map::empty(CreateParamsSpace(scop_info_.ctx_))};
      CollectBufferFootprintDefInfo(new_footprint_info, schedule, node);
      scop_info_.analysis_result_.buffer_def_infos_.push_back(new_footprint_info);
    }
  }
}

void MemoryManager::AddStateTensorsDataFlow() {
  // build init list
  // init list   TensorID   input0      DDR --> C1 --> C1 --> C0A
  //             TensorID   input1      DDR --> C0B
  //             TensorID   input2      DDR --> BUF
  //             TensorID   output0     DDR <-- BUF <-- C0C
  //             TensorID   max_1       BUF  --> DDR
  // build whole list
  // add below node
  //   TensorID  input0_local_C1               C1 --> C1 --> C0A
  //   TensorID  input0_fractal_C1             C1 --> C0A
  //   TensorID  input0_fractal_C1_local_C0A   C0A
  //   TensorID  input1_local_C1_local_C0B     C0B
  //   TensorID  output0_local_BUF               BUF <-- C0C
  //   TensorID  output0_local_BUF_local_C0C     C0C
  //   TensorID  input2_local_BUF               BUF
  //   TensorID   max_1_local_BUF               BUF
  auto tensor_name_flows = scop_info_.analysis_result_.GetTensorNameFlows();
  auto tensor_mem_flows = scop_info_.analysis_result_.GetTensorMemFlows();
  CHECK_EQ(tensor_mem_flows.size(), tensor_name_flows.size());
  CHECK_GT(tensor_mem_flows.size(), 0);
  for (const auto &tensor : tensor_mem_flows) {
    std::string name = tensor.first;
    if (tensor_name_flows.find(name) == tensor_name_flows.end()) continue;
    auto it = std::find(tensor_mem_flows[name].begin(), tensor_mem_flows[name].end(), BUF_C1_);
    auto it2 = std::find(tensor_mem_flows[name].begin(), tensor_mem_flows[name].end(), C1_);
    if (it != tensor_mem_flows[name].end() && it2 != tensor_mem_flows[name].end()) {
      std::vector<std::string> name_flow1, name_flow2, name_flow3;
      MemFlow mem_flow1, mem_flow2, mem_flow3;
      int64_t zeroth_pos = 0;
      int64_t first_pos = 1;
      int64_t second_pos = 2;
      int64_t third_pos = 3;
      int64_t fourth_pos = 4;
      if (scop_info_.mmu_info_.IsConv() || scop_info_.mmu_info_.IsGemm()) {
        name_flow1.push_back(tensor_name_flows[name][zeroth_pos]);
        mem_flow1.push_back(tensor_mem_flows[name][zeroth_pos]);
        name_flow1.push_back(tensor_name_flows[name][second_pos]);
        mem_flow1.push_back(tensor_mem_flows[name][second_pos]);
        name_flow1.push_back(tensor_name_flows[name][first_pos]);
        mem_flow1.push_back(tensor_mem_flows[name][first_pos]);

        name_flow2.push_back(tensor_name_flows[name][zeroth_pos]);
        mem_flow2.push_back(tensor_mem_flows[name][zeroth_pos]);
        name_flow2.push_back(tensor_name_flows[name][second_pos]);
        mem_flow2.push_back(tensor_mem_flows[name][second_pos]);
        name_flow2.push_back(tensor_name_flows[name][third_pos]);
        mem_flow2.push_back(tensor_mem_flows[name][third_pos]);
      }

      if (scop_info_.mmu_info_.IsConv() && scop_info_.mmu_info_.IsA(name)) {
        name_flow2.push_back(tensor_name_flows[name][fourth_pos]);
        mem_flow2.push_back(tensor_mem_flows[name][fourth_pos]);
      }

      if (scop_info_.IsInBinds(name)) {
        // add copyin tensor, gm write dataflow
        name_flow3.push_back(tensor_name_flows[name][zeroth_pos]);
        mem_flow3.push_back(tensor_mem_flows[name][zeroth_pos]);
        name_flow3.push_back(tensor_name_flows[name][first_pos]);
        mem_flow3.push_back(tensor_mem_flows[name][first_pos]);
        AddTensorDataFlow(mem_flow3, name_flow3, REALIZE_C1);
      }

      AddTensorDataFlow(mem_flow1, name_flow1);
      AddTensorDataFlow(mem_flow2, name_flow2);

      continue;
    }
    AddTensorDataFlow(tensor.second, tensor_name_flows[name]);
  }

  size_t length = scop_info_.analysis_result_.buffer_def_infos_.size();
  for (size_t tensor_idx = 0; tensor_idx < length; tensor_idx++) {
    if (scop_info_.analysis_result_.buffer_def_infos_[tensor_idx].data_stream.size() == 1) continue;

    isl::id ancestor_id = scop_info_.analysis_result_.buffer_def_infos_[tensor_idx].tensor_id;
    for (size_t idx = 1; idx < scop_info_.analysis_result_.buffer_def_infos_[tensor_idx].data_stream.size(); ++idx) {
      if (idx + 1 == scop_info_.analysis_result_.buffer_def_infos_[tensor_idx].data_stream.size()) continue;
      std::vector<std::pair<isl::id, MemType>> sub_data_stream =
        scop_info_.analysis_result_.buffer_def_infos_[tensor_idx].PartialDataStream(idx);
      AddOneBufferDefInfo(ancestor_id, sub_data_stream);
    }
  }
}

void MemoryManager::AddOneBufferDefInfo(const isl::id &ancestor_id,
                                        const std::vector<std::pair<isl::id, MemType>> &data_stream) {
  if (data_stream.empty()) return;

  auto target = data_stream[0];
  isl::id tensor_id = target.first;
  MemType mem_type = target.second;
  constexpr auto TENSORLISTTAILNAME = "TensorListTail";
  isl::id dst_tensorId = isl::id(scop_info_.ctx_, TENSORLISTTAILNAME);
  MemType dst_mem_type = MemType::DDR;
  if (0 < data_stream.size() - 1) {
    dst_tensorId = data_stream[1].first;
    dst_mem_type = data_stream[1].second;
  }

  MemFlow mem_flow;
  for (const auto &item : data_stream) {
    mem_flow.push_back(item.second);
  }
  std::string mark_tag = TensorMarkTag(dst_mem_type, mem_flow);
  if (mark_tag.empty()) return;

  std::vector<size_t> sizes;
  BufferDefInfo promoted_info = BufferDefInfo{tensor_id,
                                              dst_tensorId,
                                              ancestor_id,
                                              mem_type,
                                              mark_tag,
                                              false,
                                              false,
                                              data_stream,
                                              Tensor(),
                                              Handle(),
                                              sizes,
                                              nullptr,
                                              isl::union_map::empty(isl::space(scop_info_.ctx_, 0))};
  MakeBufferFootprintCluster(promoted_info);
  scop_info_.analysis_result_.buffer_def_infos_.push_back(promoted_info);
}

void MemoryManager::AddTensorDataFlow(const std::vector<MemType> &memflow, const std::vector<std::string> &nameflow,
                                      std::string mark_tag_specific) {
  CHECK(memflow.size() == nameflow.size());
  uint64_t i = 0;
  /*********************************************
   *
   * init mem_type:        DDR
   * init tensor_id:       input0
   * init dst_tensorId:    input0_local_C1
   * init ancestor_id:     input0
   *
   * init mark_tag:        base on dst_tensorId mem_type, realize_C1
   * init data_stream:     input0 --> input0_local_C1 --> input0_fractal_C1 --> input0_fractal_C1_local_C0A
   **********************************************/
  std::string tensor_name = nameflow[i];
  MemType mem_type = memflow[i];

  isl::id tensor_id = isl::id(scop_info_.ctx_, tensor_name);
  isl::id ancestor_id = tensor_id;
  isl::id dst_tensorId = isl::id(scop_info_.ctx_, tensor_name);
  if (i < nameflow.size() - 1) {
    std::string dst_tensor_name = nameflow[i + 1];
    dst_tensorId = isl::id(scop_info_.ctx_, dst_tensor_name);
  }
  std::vector<std::pair<isl::id, MemType>> data_stream;

  for (size_t j = i; j < nameflow.size(); j++) {
    std::string tmp_name = nameflow[j];
    isl::id tmp_id = isl::id(scop_info_.ctx_, tmp_name);
    MemType tmp_mem_type = memflow[j];
    data_stream.emplace_back(std::make_pair(tmp_id, tmp_mem_type));
  }
  MemType dst_mem_type = MemType::DDR;
  if (data_stream.size() > 1) {
    dst_mem_type = data_stream[1].second;
  }
  std::string mark_tag = TensorMarkTag(dst_mem_type, memflow);
  if (scop_info_.mmu_info_.IsIm2col() && mark_tag == REALIZE_C1) {
    mark_tag = REALIZE_BUF;
  }

  bool isCopyin = scop_info_.IsCopyinTensor(tensor_id.get_name());
  if (!isCopyin && dst_mem_type == MemType::BUF_C1_) {
    mark_tag = REALIZE_C1BUFC1;
  }
  if (!mark_tag_specific.empty()) {
    mark_tag = mark_tag_specific;
  }

  std::vector<size_t> sizes;
  bool is_bind_tensor = true;
  BufferDefInfo promoted_info = BufferDefInfo{tensor_id,
                                              dst_tensorId,
                                              ancestor_id,
                                              mem_type,
                                              mark_tag,
                                              false,
                                              is_bind_tensor,
                                              data_stream,
                                              Tensor(),
                                              Handle(),
                                              sizes,
                                              nullptr,
                                              isl::union_map::empty(isl::space(scop_info_.ctx_, 0))};
  MakeBufferFootprintCluster(promoted_info);
  scop_info_.analysis_result_.buffer_def_infos_.push_back(promoted_info);
}

void MemoryManager::MakeBufferFootprintCluster(BufferDefInfo &tensor_info) {
  std::vector<isl::schedule_node> nodes = CollectMarkNode(schedule_.get_root(), tensor_info.mark_tag);
  int index = 0;
  for (const auto &node : nodes) {
    isl::schedule_node tree = node.get_child(0);
    auto schedule = LocalSchedule(tree);

    // get TensorFootPrintsCluster for each tensor
    if (tensor_info.IsIm2col()) {
      HoistIm2colBufferFootprintCluster(schedule, node, index, tensor_info);
    } else {
      if (tensor_info.IsGemmDataC12C0() || tensor_info.IsGemmWeightC12C0()) {
        AddGemmTransposeFpCluster(schedule);
      }
      MakeMultiBufferFootprint(schedule, node, index, tensor_info);
      scop_info_.mmu_info_.UpdateSpecGemmFractalInfo(tensor_info);
    }
    index++;
  }
}

void MemoryManager::ReorderBufferedDefInfos() {
  if (scop_info_.analysis_result_.GetFakeCopyin().is_empty()) {
    return;
  }

  std::unordered_set<std::string> tensors;
  scop_info_.analysis_result_.GetFakeCopyin().foreach_map(
    [&tensors](const isl::map &m) -> void { tensors.insert(m.get_tuple_id(isl_dim_out).get_name()); });

  for (size_t index = 1; index < scop_info_.analysis_result_.buffer_def_infos_.size(); index++) {
    if ((scop_info_.analysis_result_.buffer_def_infos_[index].mark_tag == REALIZE_C1) &&
        (tensors.find(scop_info_.analysis_result_.buffer_def_infos_[index].tensor_id.get_name()) != tensors.end())) {
      BufferDefInfo promoted_info = scop_info_.analysis_result_.buffer_def_infos_[index];
      scop_info_.analysis_result_.buffer_def_infos_.erase(scop_info_.analysis_result_.buffer_def_infos_.begin() +
                                                          static_cast<int>(index));
      scop_info_.analysis_result_.buffer_def_infos_.insert(scop_info_.analysis_result_.buffer_def_infos_.begin(),
                                                           promoted_info);
    }
  }
}

void MemoryManager::AddGemmTransposeFpCluster(const isl::union_map &schedule) {
  auto domain = schedule.domain();
  if (scop_info_.mmu_info_.IsGemmDataTranspose()) {
    if (scop_info_.mmu_info_.IsGemmDataTransposeBlock()) {
      gemm_a_transpose_fp_cluster_ =
        ConstructAffineFpCluster(scop_info_, scop_info_.analysis_result_.GetReads(), domain, schedule,
                                 ReferenceType::Read, AffineType::AFFINE_GEMMBLOCK, AffineTensor::LEFT_TENSOR);
    } else if (scop_info_.mmu_info_.IsGemmDataTransposeInnerBlock()) {
      gemm_a_transpose_fp_cluster_ =
        ConstructAffineFpCluster(scop_info_, scop_info_.analysis_result_.GetReads(), domain, schedule,
                                 ReferenceType::Read, AffineType::AFFINE_GEMMBLOCKIN, AffineTensor::LEFT_TENSOR);
    } else {
      gemm_a_transpose_fp_cluster_ =
        ConstructAffineFpCluster(scop_info_, scop_info_.analysis_result_.GetReads(), domain, schedule,
                                 ReferenceType::Read, AffineType::AFFINE_GEMM, AffineTensor::LEFT_TENSOR);
    }
  }
  if (scop_info_.mmu_info_.IsGemmWeightTranspose()) {
    if (scop_info_.mmu_info_.IsGemmWeightTransposeBlock()) {
      gemm_b_transpose_fp_cluster_ =
        ConstructAffineFpCluster(scop_info_, scop_info_.analysis_result_.GetReads(), domain, schedule,
                                 ReferenceType::Read, AffineType::AFFINE_GEMMBLOCK, AffineTensor::RIGHT_TENSOR);
    } else if (scop_info_.mmu_info_.IsGemmWeightTransposeInnerBlock()) {
      gemm_b_transpose_fp_cluster_ =
        ConstructAffineFpCluster(scop_info_, scop_info_.analysis_result_.GetReads(), domain, schedule,
                                 ReferenceType::Read, AffineType::AFFINE_GEMMBLOCKIN, AffineTensor::RIGHT_TENSOR);
    } else {
      gemm_b_transpose_fp_cluster_ =
        ConstructAffineFpCluster(scop_info_, scop_info_.analysis_result_.GetReads(), domain, schedule,
                                 ReferenceType::Read, AffineType::AFFINE_GEMM, AffineTensor::RIGHT_TENSOR);
    }
  }
}

void MemoryManager::GatherFractalDefInfo(const isl::schedule_node &tree, BufferDefInfo &tensor_info,
                                         std::vector<size_t> &sizes) {
  isl::id tensor_id = tensor_info.tensor_id;
  isl::id cluster_id = tensor_info.dst_tensor_id;

  Array<Expr> shapes;
  for (auto i : sizes) {
    shapes.push_back(Expr(static_cast<int>(i)));
  }

  Type type = scop_info_.GetDtypeOf(tensor_id);
  Tensor tensor = placeholder(shapes, type, cluster_id.get_name());
  const Buffer buffer = decl_buffer(shapes, scop_info_.GetDtypeOf(tensor_id), cluster_id.get_name());
  scop_info_.user_config_.SetBind(tensor, buffer);

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(tree, sizes);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
