/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "register_memory_manager.h"
#include "poly/create_cluster.h"
#include "poly/scop.h"
#include "poly/dma_inject.h"
#include "poly/poly_util.h"

#include <numeric>
namespace akg {
namespace ir {
namespace poly {

isl::schedule RegisterMemoryManager::Run(isl::schedule sch) {
  if (!scop_info_.user_config_.GetUseRegisterMemory()) {
    return sch;
  }

  schedule_ = sch;
  sch = HoistRegisterMemory();
  return sch;
}

isl::schedule_node RegisterMemoryManager::HoistRegisterMemoryOnMark(const isl::schedule_node &orig_node) {
  current_outer_bn_ = scop_info_.analysis_result_.GetOuterBandNode(band_index_);
  if (!current_outer_bn_->use_register_memory) {
    return orig_node;
  }

  CreateClusterForOperator(orig_node);

  std::string mark_name;
  auto GetMarkNode = [this, &mark_name](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_mark>()) {
      return node;
    }

    std::string tmp_mark_name = node.as<isl::schedule_node_mark>().get_id().get_name();
    if (tmp_mark_name != mark_name) {
      return node;
    }

    return HoistClusters(node.parent()).child(0);
  };

  auto node = orig_node;
  for (auto name : mark_names_) {
    mark_name = name;
    node = MapDescendantTopDown(node, GetMarkNode);
  }
  node = InsertMarkerForEmit(node);
  node = DeleUselessMarker(node, mark_names_);
  return node;
}

isl::schedule RegisterMemoryManager::HoistRegisterMemory() {
  isl::schedule_node node = GetOuterBand(schedule_.root());
  if (node.isa<isl::schedule_node_band>()) {
    node = HoistRegisterMemoryOnMark(node);
  } else {
    int number = static_cast<int>(node.n_children());
    for (int i = 0, current_band_index = 0; i < number; ++i) {
      auto promotion_node = node.child(i).child(0);
      if (promotion_node.isa<isl::schedule_node_leaf>()) continue;

      mark_names_.clear();
      band_index_ = current_band_index;
      node = HoistRegisterMemoryOnMark(promotion_node);
      node = node.parent().parent();
      ++current_band_index;
    }
  }

  return node.get_schedule();
}

void RegisterMemoryManager::SetPromotedWriteNameForGemm(std::string &local_tensor_c) {
  write_name_ = GML_WRITE_ID_NAME;
  std::string shared_tensors;
  for (const auto &buffer : scop_info_.analysis_result_.active_buffer_footprints_) {
    auto cluster_id = buffer.second.cluster_id;
    shared_tensors += cluster_id.name() + " ";
  }
  if (shared_tensors.find(local_tensor_c) != std::string::npos) {
    write_name_ = SHARED_WRITE_ID_NAME;
  }
}

void RegisterMemoryManager::CreateClusterForOperator(const isl::schedule_node &node) {
  RegisterCreateCluster create_cluster(scop_info_, band_index_);
  if (scop_info_.user_config_.GetEnableMatmul()) {
    // matmul operator
    std::string local_tensor_c = GetMatmulTensorsName(scop_info_)[MATRIX_C];
    SetPromotedWriteNameForGemm(local_tensor_c);

    auto config_shared_tensors = scop_info_.user_config_.GetSharedTensors();
    auto c_mark = PROMOTE_GLOBAL_TO_REGISTER_C;
    if (config_shared_tensors.find(local_tensor_c) != config_shared_tensors.end()) {
      c_mark = PROMOTE_SHARED_TO_REGISTER_C;
    }

    mark_names_.emplace(PROMOTE_SHARED_TO_REGISTER_AB);
    mark_names_.emplace(c_mark);

    create_cluster.CreateClusterListForGemm(node, mark_names_);
  } else {
    mark_names_.emplace(PROMOTE_GLOBAL_TO_REGISTER);
    create_cluster.CreateClusterListForElementWise(node, mark_names_);
  }
}

isl::schedule_node RegisterMemoryManager::InsertMarkerForEmit(const isl::schedule_node &orig_node) {
  auto node = orig_node;
  if (scop_info_.user_config_.GetEnableMatmul()) {
    if (scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
      node = TileTensorAccordingInterfaceValue(orig_node);
    }
    std::string marker_name = PROMOTE_REGISTER_TO_GLOBAL;
    if (write_name_ == SHARED_WRITE_ID_NAME) {
      marker_name = PROMOTE_REGISTER_TO_SHARED;
    }
    node = InsertMarkerForThreadGroup(node, write_name_, marker_name);
  } else if (current_outer_bn_->enable_vectorization) {
    node = InsertMarkerForThreadGroup(node, GML_READ_ID_NAME, PROMOTE_VECTORIZATION);
    node = InsertMarkerForThreadGroup(node, GML_WRITE_ID_NAME, PROMOTE_VECTORIZATION);
  }
  return node;
}

isl::schedule_node RegisterMemoryManager::HoistClusters(const isl::schedule_node &node) {
  auto res_node = node;
  isl::schedule_node root_node = node.get_schedule().get_root();

  isl::schedule sch = schedule_;

  for (size_t index = 0; index < scop_info_.analysis_result_.buffer_def_infos_.size(); index++) {
    BufferDefInfo &buffer_info = scop_info_.analysis_result_.buffer_def_infos_[index];

    if (buffer_info.dst_tensor_id.to_str().find(SHARE_SUFFIX) != std::string::npos) {
      continue;
    }

    auto fp_cluster = buffer_info.GetFootPrintClusterGPU(node);

    if (fp_cluster == nullptr || !fp_cluster->foot_print_.box.is_valid()) {
      continue;
    }

    auto tensor_id = buffer_info.tensor_id;
    RegisterCreateCluster create_cluster(scop_info_, band_index_);
    isl::union_map partial_sched = create_cluster.GetPartialSchedule(node);

    auto active_domains = CollectDomain(node);
    isl::id dst_tensor_id = buffer_info.dst_tensor_id;
    GatherBufferFootprintDefInfo(node, buffer_info);
    if (scop_info_.user_config_.GetEnableMatmul()) {
      if (tensor_id.get_name().find(SHARE_SUFFIX) != std::string::npos) {
        std::shared_ptr<TensorFootprintCluster> src_fp_cluster;
        isl::union_map sch_map = scop_info_.analysis_result_.GetScheduleMapBeforeTile();
        for (auto &buffer : scop_info_.analysis_result_.active_buffer_footprints_) {
          if (tensor_id == buffer.second.cluster_id) {
            src_fp_cluster = buffer.second.cluster;
            break;
          }
        }
        if (src_fp_cluster != nullptr) {
          if (!GetMarkerName(res_node.child(0), PROMOTE_SHARED_TO_REGISTER_C).empty()) {
            res_node = res_node.child(0).del();
            res_node = res_node.parent();
          }
          res_node = PlaceInnerDataCopyBelow(scop_info_, res_node, *fp_cluster, *src_fp_cluster, tensor_id,
                                             dst_tensor_id, tensor_id, sch_map);
        }
      } else {
        res_node = PlaceOuterDataCopyBelow(scop_info_, res_node, *fp_cluster, tensor_id, dst_tensor_id, partial_sched,
                                           schedule_.get_domain().get_space());
      }
    } else {
      res_node = PlaceOuterDataCopyBelow(scop_info_, res_node, *fp_cluster, tensor_id, dst_tensor_id, partial_sched,
                                         schedule_.get_domain().get_space());
    }

    // active_buffer_footprints for codegen
    scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(std::make_pair(
      active_domains, BufferedFootPrintInfo{std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster)),
                                            partial_sched, dst_tensor_id}));
    buffer_info.find_buffer = true;
  }
  return res_node;
}

void RegisterMemoryManager::GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info) {
  auto fp_cluster = tensor_info.footprints_cluster;
  std::vector<size_t> sizes;
  if (fp_cluster == nullptr) {
    tensor_info.AddSize(node, sizes);
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
  tensor_info.AddSize(node, sizes);
}

// According to the value of the conv interface, the size of the tensor is split to confirm the size of the fragment.
isl::schedule_node RegisterMemoryManager::TileTensorAccordingInterfaceValue(const isl::schedule_node &orig_node) {
  CHECK(scop_info_.user_config_.GetReplaceConfig().count(WARP_COMPUTE)) << "Cannot map to warp.";
  auto CollectReadWriteFilter = [this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>()) {
      return node;
    }
    bool is_all_sets_read_or_write = IsReadOrWriteTensor(node, SHARED_READ_ID_NAME, write_name_);
    if (!is_all_sets_read_or_write) {
      return node;
    }

    auto start_depth = node.get_tree_depth();

    auto band_node = GetCanMappingNode(node);
    std::string id_name = GetPromotionTensorName(band_node, scop_info_.analysis_result_.buffer_def_infos_);
    if (id_name.empty() || !scop_info_.analysis_result_.GetMatrixMatmulMap().count(id_name) ||
        !scop_info_.analysis_result_.GetMatrixMatmulMajor().count(id_name)) {
      return node;
    }

    bool is_conv = scop_info_.user_config_.GetEnableConvTensorCore();
    if (is_conv) {
      band_node = AdjustConvScheduleTreeStructure(band_node);
    }

    auto mapping_cfg = scop_info_.user_config_.GetReplaceConfig()[WARP_COMPUTE];
    CHECK(mapping_cfg != nullptr) << "mapping config is null";
    // split member that does not involved in thread mapping
    auto mem_size = band_node.as<isl::schedule_node_band>().n_member();
    if (mem_size > mapping_cfg->bound) {
      band_node = band_node.as<isl::schedule_node_band>().split(mem_size - mapping_cfg->bound);
      band_node = band_node.child(0);
    }

    std::string matrix_name = scop_info_.analysis_result_.GetMatrixMatmulMap()[id_name];
    std::string matrix_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[id_name];
    isl::multi_val tile_size_val = GetRealTileSizeVal(band_node, matrix_name, matrix_major);
    band_node = TileBand(band_node, tile_size_val);

    node = band_node.ancestor(band_node.get_tree_depth() - start_depth);
    return node;
  };

  return orig_node.map_descendant_bottom_up(CollectReadWriteFilter);
}

isl::schedule_node RegisterMemoryManager::AdjustConvScheduleTreeStructure(const isl::schedule_node &orig_node) {
  auto node = orig_node;
  if (!node.isa<isl::schedule_node_band>()) {
    return node;
  }

  auto band_node = node.as<isl::schedule_node_band>();
  auto orig_number = band_node.n_member();
  if (orig_number <= 2) {
    return node;
  }

  // original node
  auto orig_partial_schedule = band_node.get_partial_schedule();
  bool orig_permutable = band_node.get_permutable();
  std::vector<bool> orig_coincident;
  for (int i = 0; i < static_cast<int>(orig_number); ++i) {
    orig_coincident.push_back(band_node.member_get_coincident(i));
  }

  isl::union_pw_aff_list new_partial_schedule(node.ctx(), orig_number);
  auto InsertPartialSchedule = [&new_partial_schedule](isl::schedule_node node) -> void {
    auto partial_schedule = node.as<isl::schedule_node_band>().get_partial_schedule();
    for (int i = 0; i < static_cast<int>(partial_schedule.size()); ++i) {
      new_partial_schedule = new_partial_schedule.add(partial_schedule.get_at(i));
    }
  };

  // split n axis
  node = node.as<isl::schedule_node_band>().split(1);
  auto n_node = node;
  node = node.del();

  // split h and w axis
  const int h_w_axis_size = 2;
  int real_h_w_axis_size = static_cast<int>(orig_number) - h_w_axis_size;
  node = node.as<isl::schedule_node_band>().split(real_h_w_axis_size);
  InsertPartialSchedule(node);
  node = node.del();
  InsertPartialSchedule(n_node);

  // split o and other axis
  InsertPartialSchedule(node);

  node = node.insert_partial_schedule(isl::multi_union_pw_aff(orig_partial_schedule.get_space(), new_partial_schedule));
  band_node = node.as<isl::schedule_node_band>();
  band_node = band_node.set_permutable(orig_permutable);
  for (int i = 0; i < static_cast<int>(orig_number); ++i) {
    band_node = band_node.member_set_coincident(i, orig_coincident[i]);
  }
  return band_node;
}

isl::multi_val RegisterMemoryManager::GetRealTileSizeVal(const isl::schedule_node &node, const std::string &matrix_name,
                                                         const std::string &matrix_major) {
  auto ctx = node.ctx();
  auto space = node.as<isl::schedule_node_band>().get_space();
  isl::multi_val tile_size_val = isl::multi_val::zero(space);

  int m = scop_info_.analysis_result_.GetMmaMode().m;
  int n = scop_info_.analysis_result_.GetMmaMode().n;
  int k = scop_info_.analysis_result_.GetMmaMode().k;
  std::vector<int> tile_size_number;
  bool need_reverse = false;
  if (matrix_name == MATRIX_B) {
    need_reverse = (matrix_major == ROW_MAJOR) ? true : false;
    tile_size_number.emplace_back(m);
    tile_size_number.emplace_back(k);
  } else if (matrix_name == MATRIX_A) {
    need_reverse = (matrix_major == COL_MAJOR) ? true : false;
    tile_size_number.emplace_back(n);
    tile_size_number.emplace_back(k);
  } else {
    tile_size_number.emplace_back(m);
    tile_size_number.emplace_back(n);
  }

  auto len = static_cast<int>(tile_size_number.size());
  for (auto i = 0; i < len; ++i) {
    int pos = need_reverse ? len - 1 - i : i;
    tile_size_val = tile_size_val.set_val(pos, isl::val(ctx, tile_size_number[i]));
  }

  return tile_size_val;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
