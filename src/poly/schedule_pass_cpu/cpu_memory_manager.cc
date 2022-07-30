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
#include "poly/isolate_tile_manager.h"
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
  if (!scop_info_.user_config_.GetUseRegisterMemory()) {
    return sch;
  }

  schedule_ = sch;
  return HoistCpuMemory();
}

isl::schedule_node CpuMemoryManager::HoistCpuMemoryOnMark(const isl::schedule_node &orig_node) {
  current_outer_bn_ = scop_info_.analysis_result_.GetOuterBandNode(band_index_);
  if (!current_outer_bn_->use_register_memory) {
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

    if (current_outer_bn_->template_type != Template::TRANSPOSE_OP) {
      node = node.del();
    }
    node = node.parent();
    return HoistClusters(node).child(0);
  };
  auto node = orig_node;
  for (auto name : mark_names_) {
    mark_name = name;
    node = MapDescendantTopDown(node, GetMarkNode);
  }
  node = InsertMarkerForEmit(node);
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
    mark_names_.emplace(PROMOTE_GLOBAL_TO_REGISTER_B);
    mark_names_.emplace(PROMOTE_GLOBAL_TO_REGISTER_A);
    mark_names_.emplace(PROMOTE_GLOBAL_TO_REGISTER_C);
    create_cluster.CreateClusterListForGemm(orig_node, mark_names_);
  } else if (current_outer_bn_->template_type == Template::CONV) {
    // conv operator
    mark_names_.emplace(PROMOTE_GLOBAL_TO_REGISTER_AB);
    mark_names_.emplace(PROMOTE_GLOBAL_TO_REGISTER_C);
    create_cluster.CreateClusterListForConv(orig_node, mark_names_);
  } else if (current_outer_bn_->template_type == Template::TRANSPOSE_OP) {
    // transpose operator
    mark_names_.emplace(PROMOTE_TRANSPOSE);
    create_cluster.CreateClusterListForTranspose(orig_node, mark_names_);
  }
}

isl::schedule_node CpuMemoryManager::InsertMarkerForEmit(const isl::schedule_node &orig_node) {
  isl::schedule_node node = orig_node;
  std::unordered_map<std::string, PromoteMarkerInfo> filter_marker_map;
  if (current_outer_bn_->template_type == Template::MATMUL) {
    // matmul operator
    node = InsertMarkerForGemm(node);
  } else if (current_outer_bn_->template_type == Template::CONV) {
    // conv operator
    PromoteMarkerInfo write_info;
    write_info.markers = {FOR_VECTORIZED, FOR_UNROLLED};
    write_info.axis_pos = -1;
    filter_marker_map[WRITE_ID_NAME] = write_info;

    PromoteMarkerInfo read_info;
    read_info.markers = {FOR_VECTORIZED};
    read_info.axis_pos = -1;
    filter_marker_map[READ_ID_NAME] = read_info;
    node = InsertMarkerForPromotedNode(node, filter_marker_map);
  } else if (current_outer_bn_->template_type == Template::TRANSPOSE_OP) {
    // transpose operator
    PromoteMarkerInfo read_write_info;
    read_write_info.markers = {FOR_VECTORIZED, FOR_UNROLLED};
    read_write_info.axis_pos = -1;
    filter_marker_map[WRITE_ID_NAME] = read_write_info;
    filter_marker_map[READ_ID_NAME] = read_write_info;
    node = InsertMarkerForPromotedNode(node, filter_marker_map);
  }
  return node;
}

// Insert relevant marker nodes for gemm operators.
isl::schedule_node CpuMemoryManager::InsertMarkerForGemm(const isl::schedule_node &orig_node) {
  auto GetPromotedFilter = [this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>() || !node.has_children() ||
        !node.child(0).isa<isl::schedule_node_band>()) {
      return node;
    }
    size_t start_depth = node.get_tree_depth();
    // Get the promoted filter node.
    isl::union_set uset = node.as<isl::schedule_node_filter>().get_filter();
    bool is_gm_filter = false;
    uset.foreach_set([&is_gm_filter](isl::set s) {
      if (s.get_tuple_name() == WRITE_ID_NAME || s.get_tuple_name() == READ_ID_NAME) {
        is_gm_filter = true;
        return;
      }
    });
    if (!is_gm_filter) {
      return node;
    }

    // Get the name of the current promoted tensor to get the row-column mode of A, B, C.
    std::string tensor_name;
    uset.unwrap().range().foreach_set([&tensor_name](const isl::set &s) -> void {
      tensor_name = s.get_tuple_name();
      size_t pos = 0;
      if ((pos = tensor_name.find(LOCAL_SUFFIX)) != std::string::npos) {
        tensor_name = tensor_name.erase(pos, tensor_name.size() - pos);
      }
    });

    std::string tensor_mark = GetTensorMark(tensor_name, scop_info_);
    std::string matmul_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[tensor_name];
    if (tensor_mark.empty() || matmul_major.empty()) {
      return node;
    }

    // Insert relevant marker nodes for gemm operators.
    auto band_node = node.child(0).as<isl::schedule_node_band>();
    node = TileVectorizationForGemm(band_node, tensor_name, tensor_mark);
    node = node.ancestor(node.get_tree_depth() - start_depth);
    return node;
  };
  return orig_node.map_descendant_bottom_up(GetPromotedFilter);
}

isl::schedule_node CpuMemoryManager::TileVectorizationForGemm(const isl::schedule_node &orig_node,
                                                              const std::string &tensor_name,
                                                              const std::string &tensor_mark) {
  if (gemm_tensor_info_map_.count(tensor_name) == 0) {
    return orig_node;
  }
  size_t start_depth = orig_node.get_tree_depth();
  GemmTensorInfo gemm_tensor_info = gemm_tensor_info_map_[tensor_name];
  auto need_transpose = gemm_tensor_info.need_transpose;
  auto vectorized_pos = gemm_tensor_info.vectorized_pos;
  auto unrolled_pos = gemm_tensor_info.unrolled_pos;
  auto tensor_shape = gemm_tensor_info.tensor_shape;

  auto band_node = orig_node.as<isl::schedule_node_band>();
  auto ctx = band_node.ctx();
  auto space = band_node.get_space();
  isl::multi_val tile_size = isl::multi_val::zero(space);

  int mn_size = scop_info_.analysis_result_.GetMmaMode().n;
  if (tensor_mark == TENSOR_A) {
    mn_size = scop_info_.analysis_result_.GetMmaMode().m;
  }

  // For the determination of the unrolled axis in the following two scenarios, if the size of the axis before
  // the vectorized axis can be determined, the unrolled marker can be inserted, otherwise it will not be inserted.
  auto node = orig_node;
  std::unique_ptr<IsolateTileManager> isolate_tile = std::make_unique<IsolateTileManager>(scop_info_, true);
  if (need_transpose) {
    // For the scene that needs to be transposed, it can be tilled directly on the promoted statement(make sure the data
    // is storable on registers), and insert vectorized marker on consecutive axes.
    const int transpose_vectorize = 4;
    const int transpose_unroll = 8;
    int current_size = 1;
    for (int i = 0; i < static_cast<int>(band_node.n_member()); ++i) {
      if (i == vectorized_pos) {
        current_size = std::min(mn_size, transpose_vectorize);
      } else if (i == unrolled_pos) {
        if (tensor_mark == TENSOR_B) {
          current_size = scop_info_.analysis_result_.GetMmaMode().k;
        } else {
          current_size = std::min(mn_size, transpose_unroll);
        }
      }
      tile_size = tile_size.set_val(i, isl::val(ctx, current_size));
    }

    node = isolate_tile->IsolateTilesForCudaAndCpu(node, tile_size, 0, tile_size.size());

    // When tensor B needs to be transposed, in order to improve performance, it is necessary to vectorize the tensor
    // before and after the transposition, but this operation cannot be completed in poly. Therefore, it is necessary to
    // tile the tensor so that the subsequent pass can be easily processed.
    if (tensor_mark == TENSOR_B) {
      tile_size = tile_size.set_val(unrolled_pos, isl::val(ctx, std::min(mn_size, transpose_unroll)));
      tile_size = tile_size.set_val(vectorized_pos, isl::val(ctx, std::min(mn_size, transpose_vectorize)));
      node = isolate_tile->IsolateTilesForCudaAndCpu(node, tile_size, 0, tile_size.size());
    }

    node = node.insert_mark(PROMOTE_TRANSPOSE).child(0);
    node = InsertMarkerForLoop(node, FOR_UNROLLED, true, unrolled_pos).child(0);
    auto split_pos = (unrolled_pos != -1) ? (vectorized_pos - unrolled_pos) : vectorized_pos;
    node = InsertMarkerForLoop(node, FOR_VECTORIZED, true, split_pos);
    node = node.ancestor(node.get_tree_depth() - start_depth);
    return node;
  } else {
    // For scenes that do not need transposition, you need to find the corresponding m/n axis and insert the vectorized
    // marker.
    for (int i = 0; i < static_cast<int>(band_node.n_member()); ++i) {
      if (i == vectorized_pos) {
        tile_size = tile_size.set_val(i, isl::val(ctx, mn_size));
      } else {
        tile_size = tile_size.set_val(i, isl::val(ctx, 1));
      }
    }

    auto node = isolate_tile->IsolateTilesForCudaAndCpu(band_node, tile_size, 0, tile_size.size());
    int vectorized_loop_size = scop_info_.analysis_result_.GetVectorizedLoopSize();
    if (vectorized_loop_size < mn_size) {
      tile_size = tile_size.set_val(vectorized_pos, isl::val(ctx, vectorized_loop_size));
      node = isolate_tile->IsolateTilesForCudaAndCpu(node, tile_size, 0, tile_size.size()).parent();
      node = InsertMarkerForLoop(node, FOR_UNROLLED, true, vectorized_pos).child(0);
      node = node.child(0);
    }
    node = InsertMarkerForLoop(node, FOR_VECTORIZED, true, vectorized_pos);
    node = node.ancestor(node.get_tree_depth() - start_depth);
    if (tensor_mark == TENSOR_C) {
      node = InsertMarkerForLoop(node, FOR_UNROLLED, true, unrolled_pos);
    }
    return node;
  }
  return orig_node;
}

void CpuMemoryManager::GetRealPosition(const std::string &tensor_name, const std::vector<size_t> &tensor_shape) {
  int tensor_size = static_cast<int>(tensor_shape.size());
  // Get the axis that needs to be vectorized for the current tensor.
  bool need_transpose = false;
  int vectorized_pos = tensor_size - 1;
  int unrolled_pos = vectorized_pos - 1;
  std::string tensor_mark = GetTensorMark(tensor_name, scop_info_);
  std::string matmul_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[tensor_name];
  // For matrix A/B, transpose is required when the k-axis is a continuous axis.
  if ((tensor_mark == TENSOR_A && matmul_major == ROW_MAJOR) ||
      (tensor_mark == TENSOR_B && matmul_major == COL_MAJOR)) {
    need_transpose = true;
  }

  // If there is a case where the axis length is 0 during the promotion stage, it cannot be displayed on the
  // schedule_tree, and additional judgment is required.
  auto AdjustRealPos = [this, tensor_size, tensor_shape, need_transpose](int &before_pos) -> void {
    if (tensor_shape[before_pos] <= 1 && !need_transpose) {
      before_pos = -1;
      return;
    }
    int real_pos = before_pos;
    for (int i = 0; i < tensor_size; ++i) {
      if (tensor_shape[i] > 1) {
        continue;
      }
      if (i < real_pos || (need_transpose && i == real_pos)) {
        --before_pos;
      }
    }
  };

  AdjustRealPos(vectorized_pos);
  AdjustRealPos(unrolled_pos);

  GemmTensorInfo gemm_tensor_info;
  gemm_tensor_info.need_transpose = need_transpose;
  gemm_tensor_info.vectorized_pos = vectorized_pos;
  gemm_tensor_info.unrolled_pos = unrolled_pos;
  gemm_tensor_info.tensor_shape = tensor_shape;
  gemm_tensor_info_map_[tensor_name] = gemm_tensor_info;
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

    GetRealPosition(id.get_name(), box_sizes);

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
