/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "poly/create_cluster.h"
#include "cpu_memory_manager.h"
#include "poly/schedule_tree_util.h"
#include "poly/scop.h"
#include "poly/dma_inject.h"
#include "poly/poly_util.h"
#include <vector>
#include <numeric>

namespace akg {
namespace ir {
namespace poly {

isl::schedule CpuMemoryManager::Run(isl::schedule sch) {
  if (!scop_info_.user_config_.GetUseSharedMemory() ||
      (!scop_info_.user_config_.GetEnableMatmul() && !scop_info_.user_config_.GetEnableConv2dDirect())) {
    return sch;
  }

  schedule_ = sch;
  return HoistCpuMemory();
}

isl::schedule_node CpuMemoryManager::HoistCpuMemoryOnMark(const isl::schedule_node &orig_node) {
  current_outer_bn_ = scop_info_.analysis_result_.GetOuterBandNode(band_index_);
  if (!current_outer_bn_->use_shared_memory) {
    return orig_node;
  }

  CreateClusterForOperator(orig_node);

  std::string mark_name = "";
  auto GetMarkNode = [this, &mark_name](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_mark>()) {
      return node;
    }

    std::string tmp_mark_name = node.as<isl::schedule_node_mark>().get_id().get_name();
    if (tmp_mark_name != mark_name) {
      return node;
    }

    node = node.del().parent();
    return HoistClusters(node).child(0);
  };
  auto node = orig_node;
  for (auto name : mark_names_) {
    mark_name = name;
    node = MapDescendantTopDown(node, GetMarkNode);
  }
  node = InsertMarkerForVectorization(node);
  return node;
}

isl::schedule CpuMemoryManager::HoistCpuMemory() {
  isl::schedule_node node = GetOuterBand(schedule_.root());
  if (node.isa<isl::schedule_node_band>()) {
    node = HoistCpuMemoryOnMark(node);
  } else {
    int number = static_cast<int>(node.n_children());
    for (int i = 0, current_band_index = 0; i < number; ++i) {
      auto promotion_node = node.child(i).child(0);
      if (!IsContainBandNode(promotion_node)) {
        continue;
      }
      band_index_ = current_band_index;
      mark_names_.clear();
      node = HoistCpuMemoryOnMark(promotion_node);
      node = node.parent().parent();
      ++current_band_index;
    }
  }

  return node.get_schedule();
}

void CpuMemoryManager::CreateClusterForOperator(const isl::schedule_node &orig_node) {
  CpuCreateCluster create_cluster(scop_info_, band_index_);
  if (current_outer_bn_->template_type == Template::MATMUL) {
    // matmul operator
    mark_names_ = {PROMOTE_GLOBAL_TO_REGISTER};
    create_cluster.CreateClusterListForGemm(orig_node, mark_names_);
  } else if (current_outer_bn_->template_type == Template::CONV) {
    // conv operator
    mark_names_.emplace(PROMOTE_GLOBAL_TO_REGISTER_AB);
    create_cluster.CreateClusterListForConv(orig_node, mark_names_);
  }
}

isl::schedule_node CpuMemoryManager::InsertMarkerForVectorization(const isl::schedule_node &orig_node) {
  isl::schedule_node node = orig_node;
  node = InsertMarkerForPromotedNode(node, WRITE_ID_NAME, FOR_VECTORIZED, -1);
  node = InsertMarkerForPromotedNode(node, READ_ID_NAME, FOR_VECTORIZED, -1);
  return node;
}

isl::schedule_node CpuMemoryManager::HoistClusters(const isl::schedule_node &node) {
  auto partial_sched_mupa = ShortScheduleMupa(schedule_.root(), node);

  auto res_node = node;
  for (size_t index = 0; index < scop_info_.analysis_result_.buffer_def_infos_.size(); index++) {
    BufferDefInfo &buffer_info = scop_info_.analysis_result_.buffer_def_infos_[index];
    auto fp_cluster = buffer_info.GetFootPrintClusterGPU(node);
    if ((fp_cluster == nullptr || !fp_cluster->foot_print_.box.is_valid())) {
      continue;
    }

    auto id = buffer_info.tensor_id;

    auto box_sizes = fp_cluster->GetFixedBoxSizes();
    if (box_sizes.size() == 0) {
      LOG(FATAL) << "Can not manage a scalar tensor";
    }

    GatherBufferFootprintDefInfo(res_node, buffer_info);
    auto dst_id = buffer_info.dst_tensor_id;
    res_node = HoistMemory(res_node, GpuMemType::LOCAL, id, dst_id, *(fp_cluster), true);

    // collect active_buffer_footprints_ info for codegen
    auto out_schedule = LocalSchedule(res_node);
    auto active_domains = CollectDomain(res_node);

    scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(std::make_pair(
      active_domains,
      BufferedFootPrintInfo{std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster)), out_schedule, dst_id}));
    buffer_info.find_buffer = true;
  }

  return res_node;
}

void CpuMemoryManager::GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info) {
  auto fp_cluster = tensor_info.footprints_cluster;
  std::vector<size_t> sizes;
  if (fp_cluster == nullptr) {
    tensor_info.AddSize(node, sizes);
    return;
  }
  sizes = fp_cluster->GetFixedBoxSizes();

  isl::id tensor_id = tensor_info.tensor_id;
  Type type = scop_info_.GetDtypeOf(tensor_id);

  isl::id cluster_id = tensor_info.dst_tensor_id;

  // build a Halide Node for cluster_id
  Array<Expr> shapes;
  for (auto i : sizes) {
    shapes.push_back(Expr(static_cast<int>(i)));
  }

  Tensor tensor = placeholder(shapes, type, cluster_id.get_name());
  const Buffer buffer = decl_buffer(shapes, scop_info_.GetDtypeOf(tensor_id), cluster_id.get_name());
  scop_info_.user_config_.SetBind(tensor, buffer);

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(node, sizes);
}

isl::schedule_node CpuMemoryManager::HoistMemory(isl::schedule_node &tree, GpuMemType type, const isl::id &tensor_id,
                                                 const isl::id &dst_tensor_id, TensorFootprintCluster &cluster,
                                                 bool force_last_extension_odd) {
  auto out_schedule = LocalSchedule(tree);
  auto active_domains = CollectDomain(tree);
  auto sizes = cluster.GetFixedBoxSizes();
  auto res_node = PlaceOuterDataCopyBelow(scop_info_, tree, cluster, tensor_id, dst_tensor_id, out_schedule,
                                          tree.get_schedule().get_domain().get_space());
  return res_node;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
