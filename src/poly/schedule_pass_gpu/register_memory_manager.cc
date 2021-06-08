/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <numeric>

#include "poly/scop.h"
#include "poly/dma_inject.h"
#include "poly/schedule_tree_util.h"
#include "poly/poly_util.h"

namespace akg {
namespace ir {
namespace poly {

isl::union_set RegisterMemoryManager::GatherMappingsTo(MappingCfg *cfg) {
  isl::schedule_node root = schedule_.get_root();
  auto domain_node = root.as<isl::schedule_node_domain>();
  auto domain = domain_node.domain();
  auto mapping_filters = CollectNode<isl::schedule_node_filter>(schedule_);

  std::vector<isl::id> filters;
  for (size_t idx = 0; idx < cfg->bound; ++idx) {
    auto value = cfg->GetAt(idx);
    auto id = isl::id(root.ctx(), value.first);
    filters.push_back(id);
  }
  mapping_filters = FilterNode(mapping_filters, filters);

  auto mapping = isl::union_set::empty(domain.ctx());
  for (auto item : mapping_filters) {
    if (item.isa<isl::schedule_node_filter>()) {
      auto filter = item.as<isl::schedule_node_filter>();
      if (filter.has_parent() && !filter.parent().isa<isl::schedule_node_mark>()) {
        continue;
      }

      isl::union_set uset = filter.get_filter();
      std::vector<isl::set> vset;
      uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });
      if (!vset.empty()) {
        auto filter_name = vset[0].get_tuple_name();
        if (filter_name == READ_ID_NAME || filter_name == WRITE_ID_NAME) {
          continue;
        }
      }

      mapping = mapping.unite(filter.filter());
    }
  }
  return mapping;
}

void RegisterMemoryManager::SharedTensors() {
  for (const auto &buffer : scop_info_.analysis_result_.active_buffer_footprints_) {
    auto cluster_id = buffer.second.cluster_id;
    auto buf_def = scop_info_.analysis_result_.GetBufferDefInfo(cluster_id);
    shared_tensors_ += buf_def.tensor_id.name() + " ";
  }
}

isl::schedule RegisterMemoryManager::HoistRegisterMemoryOnDepth(isl::schedule_node &node, size_t depth) {
  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  auto res_node = node;
  auto block_mapping = GatherMappingsTo(block_cfg);
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  auto mapping = GatherMappingsTo(thread_cfg).intersect(block_mapping);

  auto partial_sched = LocalSchedule(node);
  auto tmp_sched = partial_sched.intersect_domain(mapping);
  if (scop_info_.user_config_.GetEnableMatmul() && scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
    tmp_sched = partial_sched;
  }
  CreateTensorCluster(node, tmp_sched);

  isl::schedule_node root_node = node.get_schedule().get_root();

  isl::schedule sch = schedule_;
  if (memory_exceeding_) {
    depth = depth + 1;
    sch = HoistRegisterMemory(root_node, depth);
    return sch;
  } else {
    auto thread_schedule = MapDomainAllWithType(root_node, thread_cfg, scop_info_.upa_node_mapping_, THREAD_MARKER);
    auto block_schedule = MapDomainAllWithType(root_node, block_cfg, scop_info_.upa_node_mapping_, BLOCK_MARKER);

    auto tmp_node = res_node;
    if (node.isa<isl::schedule_node_band>()) {
      tmp_node = res_node.child(0);
    }

    auto partial_sched_mupa = ShortScheduleMupa(root_node, tmp_node);
    auto partial_sched_with_block = isl::union_map::from(partial_sched_mupa).intersect_domain(block_mapping);
    partial_sched_mupa = partial_sched_mupa.flat_range_product(block_schedule).flat_range_product(thread_schedule);
    for (size_t index = 0; index < scop_info_.analysis_result_.buffer_def_infos_.size(); index++) {
      BufferDefInfo &buffer_info = scop_info_.analysis_result_.buffer_def_infos_[index];

      if (buffer_info.dst_tensor_id.to_str().find(SHARE_SUFFIX) != std::string::npos) {
        continue;
      }

      if (scop_info_.user_config_.GetEnableMatmul() && !hoist_tensor_all_) {
        if (!hoist_compute_local_tensor_) {
          if (buffer_info.dst_tensor_id.get_name() == local_tensor_c_ + LOCAL_SUFFIX) {
            continue;
          }
        } else {
          if (buffer_info.dst_tensor_id.get_name() != local_tensor_c_ + LOCAL_SUFFIX) {
            continue;
          }
        }
      }

      auto fp_cluster = buffer_info.GetFootPrintClusterGPU(res_node);

      if (fp_cluster == nullptr || !fp_cluster->foot_print_.box.is_valid()) {
        continue;
      }

      auto tensor_id = buffer_info.tensor_id;
      auto box_sizes = fp_cluster->GetFixedBoxSizes();

      if (box_sizes.size() == 0) {
        LOG(FATAL) << "Can not manage a scalar tensor in register memory promotion";
      }

      if (!IsPromote(*fp_cluster, partial_sched_mupa, thread_schedule)) {
        continue;
      }

      if (!scop_info_.user_config_.GetEnableTensorCore() && !scop_info_.user_config_.GetEnableMatmul()) {
        if (!ReuseTensorCluster(*fp_cluster, partial_sched_mupa)) {
          continue;
        }
      }

      auto active_domains = CollectDomain(res_node);
      isl::id dst_tensor_id = buffer_info.dst_tensor_id;
      GatherBufferFootprintDefInfo(res_node, buffer_info);
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
            node = PlaceInnerDataCopyBelow(scop_info_, node, *fp_cluster, *src_fp_cluster, tensor_id, dst_tensor_id,
                                           tensor_id, sch_map);
          }
        } else {
          node = PlaceOuterDataCopyBelow(scop_info_, node, *fp_cluster, tensor_id, dst_tensor_id, partial_sched,
                                         schedule_.get_domain().get_space());
        }
      } else {
        node = PlaceOuterDataCopyBelow(scop_info_, node, *fp_cluster, tensor_id, dst_tensor_id, partial_sched,
                                       schedule_.get_domain().get_space());
      }

      // active_buffer_footprints for codegen
      scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(std::make_pair(
        active_domains, BufferedFootPrintInfo{std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster)),
                                              partial_sched, dst_tensor_id}));
      buffer_info.find_buffer = true;
    }
    sch = node.get_schedule();
    return sch;
  }
}

bool RegisterMemoryManager::UnrolledLoop(const TensorFootprintCluster &fp_cluster) {
  auto box_sizes = fp_cluster.GetFixedBoxSizes();
  size_t tmp_size = 1;
  for (auto size : box_sizes) {
    tmp_size = tmp_size * size;
  }
  if (tmp_size != 1) {
    return true;
  }
  return false;
}

/*Check if the given "group" can be promoted to registers for the given
 * mapping to thread identifiers and within the given outer schedule */
bool RegisterMemoryManager::IsPromote(const TensorFootprintCluster &fp_cluster,
                                      const isl::multi_union_pw_aff &partial_sched_mupa,
                                      const isl::multi_union_pw_aff &thread_schedule) {
  /* compute the mapping relation between single thread and outer schedule space and tensor elements pair */
  isl::union_map state_schedule_mapping =
    ScheduleTensorMapping(partial_sched_mupa, fp_cluster.OrigianlAccessRelations());
  isl::union_map thread_schedule_mapping = state_schedule_mapping.apply_domain(isl::union_map::from(thread_schedule));
  /* check that whether the mapping relation between single thread
   * and outer schedule points and group elements pair is injective. */
  return thread_schedule_mapping.is_injective();
}

/* Check that whether the mapping relation between instance statement
 * and outer schedule points and tensor elements pair is reusable. */
bool RegisterMemoryManager::ReuseTensorCluster(const TensorFootprintCluster &cluster,
                                               const isl::multi_union_pw_aff &outer_pw_aff) {
  /* compute the mapping relation between statement instance and outer schedule space and tensor elements pair */
  isl::union_map state_schedule_mapping = ScheduleTensorMapping(outer_pw_aff, cluster.OrigianlAccessRelations());
  return !state_schedule_mapping.is_injective();
}

void RegisterMemoryManager::CreateTensorCluster(const isl::schedule_node &node, const isl::union_map &outer_sch) {
  isl::union_map reads = scop_info_.analysis_result_.GetReads();
  isl::union_map writes = scop_info_.analysis_result_.GetWrites();
  isl::union_map copyin = scop_info_.analysis_result_.GetCopyin();
  isl::union_map fake_copyin = scop_info_.analysis_result_.GetFakeCopyin();

  auto read_map = scop_info_.StmtReadMap();
  auto write_map = scop_info_.StmtWriteMap();
  auto stmt_map = scop_info_.analysis_result_.GetStmtOpInfoMap();
  std::vector<isl::id> tensor_list;
  std::unordered_set<isl::id, isl::IslIdIslHash> id_sets;
  for (auto item : read_map) {
    for (auto item_id : item.second) {
      id_sets.insert(item_id);
    }
  }
  for (auto item : write_map) {
    for (auto item_id : item.second) {
      id_sets.insert(item_id);
    }
  }

  std::set<std::string> shared_dst_tensor_ids;
  for (auto buffer : scop_info_.analysis_result_.active_buffer_footprints_) {
    shared_dst_tensor_ids.insert(buffer.second.cluster_id.get_name());
  }
  if (!configed_tensors_.empty()) {
    std::set<std::string> tensor_sets;
    for (const auto &item : configed_tensors_) {
      if (tensor_sets.count(item) == 0) {
        tensor_sets.emplace(item);
      }
    }
    id_sets.clear();
    for (auto item : tensor_sets) {
      id_sets.insert(isl::id(scop_info_.ctx_, item));
    }
  }

  for (auto item : id_sets) {
    if (scop_info_.user_config_.GetEnableMatmul()) {
      tensor_list.push_back(item);
    } else {
      if (!shared_dst_tensor_ids.count(item.get_name() + SHARE_SUFFIX)) {
        tensor_list.push_back(item);
      }
    }
  }

  std::vector<BufferDefInfo> promoted_infos;

  if (scop_info_.user_config_.GetEnableMatmul()) {
    std::unordered_map<std::string, std::string> matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();
    for (auto i : matmul_map) {
      if (i.second == MATRIX_C) {
        local_tensor_c_ = i.first;
      }
    }
  }

  for (const auto &item : tensor_list) {
    if (scop_info_.user_config_.GetEnableMatmul() && !hoist_tensor_all_) {
      if (!hoist_compute_local_tensor_) {
        if (item.get_name() == local_tensor_c_) {
          continue;
        }
      } else {
        if (item.get_name() != local_tensor_c_) {
          continue;
        }
      }
    }

    isl::id dst_tensor_id = GpuDstId(GpuMemType::LOCAL, item);
    std::vector<size_t> buffer_sizes;
    std::vector<std::pair<isl::id, MemType>> data_stream;
    MemType memtype;
    BufferDefInfo promoted_info;
    isl::id tmp_item;
    if (!shared_dst_tensor_ids.count(item.get_name() + SHARE_SUFFIX)) {
      tmp_item = item;
      data_stream.push_back(std::make_pair(item, MemType::DDR));
      data_stream.push_back(std::make_pair(item, MemType::LOCAL_));
      memtype = MemType::DDR;
    } else {
      tmp_item = isl::id(scop_info_.ctx_, item.get_name() + SHARE_SUFFIX);
      data_stream.push_back(std::make_pair(item, MemType::SHARED_));
      data_stream.push_back(std::make_pair(item, MemType::LOCAL_));
      memtype = MemType::SHARED_;
    }
    promoted_info = BufferDefInfo{tmp_item,
                                  dst_tensor_id,
                                  tmp_item,
                                  memtype,
                                  "",
                                  false,
                                  false,
                                  data_stream,
                                  Tensor(),
                                  Handle(),
                                  buffer_sizes,
                                  nullptr,
                                  isl::union_map::empty(isl::space(scop_info_.ctx_, 0))};
    promoted_info.footprints_cluster =
      TensorFootprintCluster::HoistBufferFootprintCluster(outer_sch, item, reads, copyin, writes, fake_copyin);
    if (promoted_info.footprints_cluster != nullptr) {
      promoted_info.footprint_cluster_map.emplace_back(std::make_pair(node, promoted_info.footprints_cluster));
      promoted_infos.push_back(promoted_info);
    }
  }

  IsOutofMemory(promoted_infos);

  if (!memory_exceeding_) {
    for (auto promoted_info : promoted_infos) {
      scop_info_.analysis_result_.buffer_def_infos_.push_back(promoted_info);
    }
  }
}

void RegisterMemoryManager::IsOutofMemory(std::vector<BufferDefInfo> promoted_infos) {
  int64_t alloc_threads = 1;
  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  if (thread_cfg != nullptr) {
    for (size_t i = 0; i < thread_cfg->bound; ++i) {
      alloc_threads *= thread_cfg->GetAt(i).second;
    }
  }
  size_t total_alloc_size = 0;
  memory_exceeding_ = false;
  for (auto promoted_info : promoted_infos) {
    auto box_sizes = promoted_info.footprints_cluster->GetFixedBoxSizes();
    if (!box_sizes.empty()) {
      auto tensor_size = std::accumulate(box_sizes.begin(), box_sizes.end(), 1, std::multiplies<size_t>());
      auto data_bytes = scop_info_.user_config_.GetDataType(promoted_info.tensor_id.get_name());
      total_alloc_size += tensor_size * std::max<int>(1, data_bytes / BYTES_PER_REGISTER);
      if (total_alloc_size * alloc_threads >= MAX_REGISTER_PER_THREAD_BLOCK * REGISTER_ALLOC_RATIO) {
        memory_exceeding_ = true;
        break;
      }
    }
  }
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

size_t RegisterMemoryManager::UpdateDepth(const isl::schedule_node &node) {
  auto band = node.as<isl::schedule_node_band>();
  for (size_t i = 0; i < band.n_member(); i++) {
    if (!band.member_get_coincident(i)) {
      if (i == 0) {
        return band.n_member();
      } else {
        return i;
      }
    }
  }
  return band.n_member() + node.schedule_depth();
}

isl::schedule RegisterMemoryManager::HoistRegisterMemory(isl::schedule_node root, size_t depth) {
  auto bands = BandsContainingScheduleDepth(root, depth);
  bands = FilterWithFunc(
    [root, depth](isl::schedule_node node) {
      auto band = node.as<isl::schedule_node_band>();
      return !IsThreadMappedMark(node) || node.schedule_depth() + band.n_member() == depth;
    },
    bands);
  bands = BandsSplitAfterDepth(bands, root, depth);

  isl::schedule tmp_sch = root.get_schedule();
  int distance_to_extension = 3;
  for (auto band : bands) {
    if (IsThreadMappedMark(band)) {
      band = band.child(0);
    }

    if (IsReadOrWriteBand(band)) {
      continue;
    }

    if (band.has_parent() && band.parent().has_parent() && band.parent().parent().has_parent() &&
        band.ancestor(distance_to_extension) &&
        band.ancestor(distance_to_extension).isa<isl::schedule_node_extension>()) {
      break;
    }
    tmp_sch = HoistRegisterMemoryOnDepth(band, depth);
    break;
  }
  return tmp_sch;
}

bool RegisterMemoryManager::IsReadOrWriteBand(isl::schedule_node node) {
  if (node.parent().isa<isl::schedule_node_filter>()) {
    auto filter = node.parent().as<isl::schedule_node_filter>();

    isl::union_set uset = filter.get_filter();
    std::vector<isl::set> vset;
    uset.foreach_set([&vset](isl::set s) { vset.push_back(s); });
    if (!vset.empty()) {
      auto filter_name = vset[0].get_tuple_name();
      if (filter_name == READ_ID_NAME || filter_name == WRITE_ID_NAME) {
        return true;
      }
    }
  }
  return false;
}

isl::schedule_node RegisterMemoryManager::GetRegisterPromotedNode(isl::schedule_node &root) {
  isl::schedule_node hoist_register_node = root;
  root.foreach_descendant_top_down([&hoist_register_node](const isl::schedule_node &node) -> bool {
    if (auto sequence_node = node.as<isl::schedule_node_sequence>()) {
      if (sequence_node.parent().isa<isl::schedule_node_extension>() &&
          sequence_node.parent().parent().isa<isl::schedule_node_band>()) {
        hoist_register_node = sequence_node.parent().parent();
        return false;
      } else if (sequence_node.parent().isa<isl::schedule_node_band>()) {
        hoist_register_node = sequence_node.parent();
        return false;
      }
    }
    if (auto mark_node = node.as<isl::schedule_node_mark>()) {
      if (mark_node.get_id().get_name() == THREAD_MARKER && mark_node.parent().isa<isl::schedule_node_band>()) {
        hoist_register_node = mark_node.parent();
        return false;
      }
    }
    return true;
  });
  return hoist_register_node;
}

isl::schedule_node RegisterMemoryManager::CollectMarkNode(isl::schedule_node root,
                                                          const std::string local_position_mark) {
  isl::schedule_node hoist_node;
  root.foreach_descendant_top_down([&hoist_node, &local_position_mark](const isl::schedule_node &node) -> bool {
    if (auto mark_node = node.as<isl::schedule_node_mark>()) {
      // ignore nested mark nodes
      if (mark_node.get_id().get_name() == local_position_mark) {
        hoist_node = mark_node;
        return false;
      }
    }
    return true;
  });
  return hoist_node;
}

isl::schedule RegisterMemoryManager::HoistRegisterMemoryOnMark(isl::schedule_node root) {
  std::string config_shared_tensors = scop_info_.user_config_.GetSharedTensors();
  auto c_mark = PROMOTE_GLOBAL_TO_REGISTER_C;
  if (config_shared_tensors.find(local_tensor_c_) != std::string::npos) {
    c_mark = PROMOTE_GLOBAL_TO_SHARED_C;
  }

  auto mark_node = CollectMarkNode(root, c_mark);
  auto tmp_hoist_node = mark_node.parent();

  while (!tmp_hoist_node.isa<isl::schedule_node_band>()) {
    tmp_hoist_node = tmp_hoist_node.parent();
  }

  auto depth = tmp_hoist_node.child(0).schedule_depth();
  auto hoist_compute_node = tmp_hoist_node.as<isl::schedule_node_band>();
  for (size_t i = 0; i < hoist_compute_node.n_member(); ++i) {
    if (!hoist_compute_node.member_get_coincident(i)) {
      if (scop_info_.user_config_.GetEnableTensorCoreUsePoly() && i == 0) {
        hoist_tensor_all_ = true;
        auto hoist_node = mark_node.del().parent();
        auto sch = HoistRegisterMemoryOnDepth(hoist_node, depth);
        return sch;
      }
      hoist_compute_node = hoist_compute_node.split(i);
      depth = depth - hoist_compute_node.n_member() + i;
    }
  }
  auto sch = HoistRegisterMemoryOnDepth(hoist_compute_node, depth);

  auto hoist_ab_root = sch.get_root();
  auto ab_mark = PROMOTE_SHARED_TO_REGISTER;
  auto mark_ab_node = CollectMarkNode(hoist_ab_root, ab_mark);
  auto hoist_ab_node = mark_ab_node.del().parent();
  auto hoist_ab_depth = hoist_ab_node.schedule_depth();
  hoist_compute_local_tensor_ = false;
  sch = HoistRegisterMemoryOnDepth(hoist_ab_node, hoist_ab_depth);

  return sch;
}

isl::schedule_node RegisterMemoryManager::MapPromotionTensorToWarps(isl::schedule_node &root) {
  std::string write_name = WRITE_ID_NAME;
  std::string shared_tensors = shared_tensors_;
  if (shared_tensors.find(local_tensor_c_) != std::string::npos) {
    write_name = SHARED_WRITE_ID_NAME;
  }
  auto CollectReadWriteFilter = [this, write_name](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>()) {
      return node;
    }
    bool is_all_sets_read_or_write = IsReadOrWriteTensor(node, SHARED_READ_ID_NAME, write_name);
    if (!is_all_sets_read_or_write) {
      return node;
    }
    auto band_node = GetCanMappingNode(node);
    CHECK(scop_info_.user_config_.GetReplaceConfig().count(WARP_COMPUTE)) << "Cannot map to warp.";
    auto mapping_cfg = scop_info_.user_config_.GetReplaceConfig()[WARP_COMPUTE];
    auto original_x = mapping_cfg->GetX().second;
    auto original_y = mapping_cfg->GetY().second;

    std::string id_name = GetPromotionTensorName(band_node, scop_info_.analysis_result_.buffer_def_infos_);
    if (id_name.empty() || !scop_info_.analysis_result_.GetMatrixMatmulMap().count(id_name) ||
        !scop_info_.analysis_result_.GetMatrixMatmulMajor().count(id_name)) {
      return band_node;
    }

    // split member that does not involved in thread mapping
    bool has_split = false;
    auto mem_size = band_node.as<isl::schedule_node_band>().n_member();
    if (mem_size > mapping_cfg->bound) {
      band_node = band_node.as<isl::schedule_node_band>().split(mem_size - mapping_cfg->bound);
      band_node = band_node.child(0);
      has_split = true;
    }

    auto matrix_name = scop_info_.analysis_result_.GetMatrixMatmulMap()[id_name];
    auto matrix_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[id_name];
    isl::multi_val tile_size_val = GetRealTileSizeVal(band_node, matrix_name, matrix_major);
    band_node = TileBand(band_node, tile_size_val);

    // In order to ensure that the data when promotion and calculation are consistent, map the m axis of MATRIX_A to
    // w0, and map the n axis of MATRIX_B to w1.
    bool need_coalesce = false;
    if (matrix_name == MATRIX_A) {
      need_coalesce = (matrix_major == ROW_MAJOR) ? true : false;
      mapping_cfg->ModifySize(N_POSITION, MAPPING_INVALID_WARP);
    } else if (matrix_name == MATRIX_B) {
      need_coalesce = (matrix_major == ROW_MAJOR) ? true : false;
      mapping_cfg->ModifySize(M_POSITION, MAPPING_INVALID_WARP);
    } else {
      need_coalesce = true;
    }

    Mapping mapping;
    auto after_map_pair = MapInnerDimToThreads(band_node, true, mapping_cfg, mapping, need_coalesce);
    band_node = after_map_pair.first;

    if (matrix_name == MATRIX_A) {
      need_coalesce = true;
      mapping_cfg->ModifySize(N_POSITION, original_y);
    } else if (matrix_name == MATRIX_B) {
      mapping_cfg->ModifySize(M_POSITION, original_x);
    }

    bool locate_is_child = false;
    if (band_node.child(0).as<isl::schedule_node_mark>()) {
      band_node = band_node.child(0);
      locate_is_child = true;
    }
    if (band_node.as<isl::schedule_node_mark>()) {
      auto marker_name = band_node.as<isl::schedule_node_mark>().get_id().get_name();
      if (marker_name.find(THREAD_MARKER) != std::string::npos) {
        band_node = band_node.del().insert_mark(isl::id(band_node.ctx(), matrix_name));
      }
    }
    band_node = locate_is_child ? band_node.parent() : band_node;
    std::string fragment_mark = FRAGMENT;
    fragment_mark += matrix_name.at(matrix_name.size() - 1);
    band_node = band_node.insert_mark(fragment_mark);

    band_node = has_split ? band_node.parent() : band_node;

    node = band_node.parent();
    return node;
  };

  return root.map_descendant_bottom_up(CollectReadWriteFilter);
}

isl::multi_val RegisterMemoryManager::GetRealTileSizeVal(const isl::schedule_node &node, const std::string &matrix_name,
                                                         const std::string &matrix_major) {
  auto title_size_count = static_cast<int>(pass_info_.tile_sizes_.size());
  auto ctx = node.ctx();
  auto space = node.as<isl::schedule_node_band>().get_space();
  isl::multi_val tile_size_val = isl::multi_val::zero(space);

  auto init_number = title_size_count > M_N_K_COUNT ? title_size_count - M_N_K_COUNT : 0;
  auto del_position = init_number;
  bool need_coalesce = false;
  if (matrix_name == MATRIX_B) {
    need_coalesce = (matrix_major == ROW_MAJOR) ? true : false;
    del_position += M_POSITION;
  } else if (matrix_name == MATRIX_A) {
    need_coalesce = (matrix_major == COL_MAJOR) ? true : false;
    del_position += N_POSITION;
  } else {
    del_position += K_POSITION;
  }

  std::vector<int> tile_size_number;
  for (auto i = init_number; i < title_size_count; ++i) {
    if (i == del_position) {
      continue;
    }
    tile_size_number.emplace_back(static_cast<int>(pass_info_.tile_sizes_[i].c0_tiling_size));
  }

  auto len = static_cast<int>(tile_size_number.size());
  for (auto i = 0; i < len; ++i) {
    int pos = need_coalesce ? len - 1 - i : i;
    tile_size_val = tile_size_val.set_val(pos, isl::val(ctx, tile_size_number[i]));
  }

  return tile_size_val;
}

isl::schedule RegisterMemoryManager::Run(isl::schedule sch) {
  auto GetGMWriteFilter = [this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>()) {
      return node;
    }
    isl::union_set uset = node.as<isl::schedule_node_filter>().get_filter();
    bool is_gm_write = false;
    uset.foreach_set([&is_gm_write](isl::set s) {
      if (s.get_tuple_name() == WRITE_ID_NAME) {
        is_gm_write = true;
      }
    });
    if (is_gm_write && node.has_parent() && node.parent().isa<isl::schedule_node_sequence>()) {
      node = node.child(0).insert_mark(PROMOTE_LOCAL_TO_GLOBAL);
      node = node.parent();
    }
    return node;
  };

  auto node = sch.root().child(0);
  if (node.isa<isl::schedule_node_context>()) {
    node = node.del();
  }
  node = InsertContextNode(node, scop_info_);
  sch = node.schedule();

  if (!scop_info_.user_config_.UseRegisterMemory()) {
    return sch;
  }

  if (scop_info_.user_config_.GetEnableAkgReduceLib()) {
    return sch;
  }

  schedule_ = sch;
  auto root = sch.get_root();

  auto res_node = GetRegisterPromotedNode(root);
  if (res_node.isa<isl::schedule_node_band>()) {
    auto depth = UpdateDepth(res_node);
    if (scop_info_.user_config_.GetRegisterDepth() >= 0) {
      depth = scop_info_.user_config_.GetRegisterDepth();
    }
    if (scop_info_.user_config_.GetEnableMatmul()) {
      sch = HoistRegisterMemoryOnMark(root);
      if (scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
        root = sch.get_root();
        sch = MapPromotionTensorToWarps(root).get_schedule();
      }
      sch = sch.get_root().map_descendant_bottom_up(GetGMWriteFilter).get_schedule();
    } else {
      sch = HoistRegisterMemory(root, depth);
    }
  }

  return sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
