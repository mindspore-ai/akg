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
#include "poly/schedule_pass_gpu/operator_shared_strategy.h"
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
  if (!scop_info_.user_config_.GetUseSharedMemory() || !scop_info_.user_config_.GetEnableMatmul()) {
    return sch;
  }

  schedule_ = sch;
  return HoistCpuMemory();
}

isl::schedule CpuMemoryManager::HoistCpuMemory() {
  std::string mark_name = "";
  auto HoistCpuMemoryOnMark = [this, &mark_name](isl::schedule_node node) -> isl::schedule_node {
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

  auto HoistCoreFunc = [this, HoistCpuMemoryOnMark,
                        &mark_name](const isl::schedule_node &orig_node) -> isl::schedule_node {
    current_outer_bn_ = scop_info_.analysis_result_.GetOuterBandNode(band_index_);
    if (!current_outer_bn_->use_shared_memory) {
      return orig_node;
    }

    mark_names_ = {PROMOTE_GLOBAL_TO_REGISTER};
    CpuMemoryStrategy other_op(scop_info_, mark_names_, band_index_);
    other_op.CreateClusterList(orig_node);
    auto node = orig_node;
    for (auto name : mark_names_) {
      mark_name = name;
      node = MapDescendantTopDown(node, HoistCpuMemoryOnMark);
    }
    return node;
  };

  isl::schedule_node node = GetOuterBand(schedule_.root());
  if (node.isa<isl::schedule_node_band>()) {
    node = HoistCoreFunc(node);
  } else {
    int number = static_cast<int>(node.n_children());
    for (int i = 0, current_band_index = 0; i < number; ++i) {
      auto promotion_node = node.child(i).child(0);
      if (promotion_node.isa<isl::schedule_node_leaf>()) {
        continue;
      }
      band_index_ = current_band_index;
      node = HoistCoreFunc(promotion_node).ancestor(2);
      ++current_band_index;
    }
  }

  return node.get_schedule();
}

bool CpuMemoryManager::IsInitFilter(const isl::schedule_node &orig_node) {
  if (!orig_node.isa<isl::schedule_node_filter>()) {
    return false;
  }

  bool is_init_filter = false;
  auto filter_node = orig_node.as<isl::schedule_node_filter>();
  auto filter = filter_node.get_filter();
  filter.foreach_set([&is_init_filter, this](const isl::set &set) -> void {
    auto id_name = set.get_tuple_id();
    auto all_init_id = scop_info_.analysis_result_.GetReduceInitIds();
    auto it = std::find(all_init_id.begin(), all_init_id.end(), id_name);
    if (it != all_init_id.end()) {
      is_init_filter = true;
      return;
    }
  });
  return is_init_filter;
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
  if (scop_info_.user_config_.GetVectorLength()) {
    scop_info_.analysis_result_.RecordSharedTensorBitsMap(tensor_id.get_name(),
                                                          scop_info_.GetDtypeOf(tensor_id).bits());
  }

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

bool CpuMemoryManager::CoalescingAccessWay(const isl::schedule_node &node, const TensorFootprintCluster &cluster) {
  isl::union_map original = cluster.OrigianlAccessRelations();
  size_t tensor_dim = cluster.foot_print_.GetBoxDim();
  std::vector<isl::schedule_node> thread_marker = CollectFnNode(IsThreadMappedMark, schedule_.root());
  for (auto item : thread_marker) {
    if (!(item.isa<isl::schedule_node_mark>()) && !(item.has_children()) &&
        !(item.child(0).isa<isl::schedule_node_filter>())) {
      continue;
    }
    isl::schedule_node thread_filter = item.child(0);
    if (!thread_filter.has_children()) {
      continue;
    }
    isl::schedule_node thread_band = thread_filter.child(0);
    if (!thread_band.has_children()) {
      continue;
    }
    isl::schedule_node inner_band = thread_band.child(0);
    size_t num_mapped_thread = inner_band.schedule_depth() - thread_band.schedule_depth();
    if (num_mapped_thread == 0) {
      continue;
    }
    size_t inner_depth = inner_band.schedule_depth();
    auto active_domains = CollectDomain(thread_band);
    auto local_access = original.intersect_domain(active_domains);
    auto schedule = ShortSchedule(inner_band);
    auto schedule_access = local_access.apply_domain(schedule);
    for (auto access : schedule_access.get_map_list()) {
      if (!IsSubsetForIncreaseDim(access, tensor_dim - 1, inner_depth - 1)) {
        return true;
      }
    }
  }
  return false;
}

isl::schedule CpuMemoryManager::InsertVectorizedMarker(const isl::schedule &sch) {
  auto GetPromotedWriteFilter = [this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_band>() || !node.has_parent() || !node.parent().isa<isl::schedule_node_filter>()) {
      return node;
    }
    auto filter_node = node.parent().as<isl::schedule_node_filter>();
    isl::union_set uset = filter_node.get_filter();
    bool is_gm_read = false;
    uset.foreach_set([&is_gm_read](isl::set s) {
      if (s.get_tuple_name() == READ_ID_NAME) {
        is_gm_read = true;
      }
    });

    if (!is_gm_read) {
      return node;
    }

    auto name_uset = uset.unwrap().domain().unwrap().range();
    name_uset.foreach_set([this, &node](isl::set s) {
      auto matrix_name = s.get_tuple_name();
      std::unordered_map<std::string, std::string> matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();

      if (matmul_map.find(matrix_name) == matmul_map.end()) {
        return;
      }

      auto matrix_matmul_map = matmul_map[matrix_name];
      if (matrix_matmul_map != MATRIX_A && matrix_matmul_map != MATRIX_B) {
        return;
      }

      auto band_node = node.as<isl::schedule_node_band>();
      node = band_node.split(band_node.n_member() - 1).child(0);
      node = node.insert_mark(FOR_VECTORIZED);
      node = node.parent();
      return;
    });

    return node;
  };
  auto final_sch = sch.get_root().map_descendant_bottom_up(GetPromotedWriteFilter).schedule();
  return final_sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
