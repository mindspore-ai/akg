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
#include "poly/poly_util.h"

namespace akg {
namespace ir {
namespace poly {

void RegisterMemoryManager::GetActualPromotedSharedTensors() {
  for (const auto &buffer : scop_info_.analysis_result_.active_buffer_footprints_) {
    auto cluster_id = buffer.second.cluster_id;
    shared_tensors_ += cluster_id.name() + " ";
  }
}

isl::schedule RegisterMemoryManager::HoistRegisterMemoryOnDepth(isl::schedule_node &node, size_t depth) {
  auto res_node = node;
  isl::schedule_node root_node = node.get_schedule().get_root();

  auto block_cfg = scop_info_.user_config_.GetBlockConfig();
  CHECK(block_cfg != nullptr) << "block config is null";
  auto replace_cfg = scop_info_.user_config_.GetReplaceConfig();
  MappingStrategyAxisMap mapping_strategy = scop_info_.user_config_.GetOuterMappingStrategy(0);
  std::unordered_set<std::string> non_repeated_idx = GetNonRepeatedIdx(mapping_strategy);
  auto block_mapping = GetMappingFilterInfo(root_node, block_cfg, replace_cfg, non_repeated_idx);

  auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
  CHECK(thread_cfg != nullptr) << "thread config is null";
  auto thread_mapping = isl::union_set::empty(block_mapping.ctx());
  mapping_strategy = scop_info_.user_config_.GetInnerMappingStrategy(0);
  non_repeated_idx = GetNonRepeatedIdx(mapping_strategy);
  thread_mapping = GetMappingFilterInfo(root_node, thread_cfg, replace_cfg, non_repeated_idx);

  auto partial_sched = LocalSchedule(node);
  if (!thread_mapping.is_empty() && !block_mapping.is_empty()) {
    auto mapping = block_mapping.intersect(thread_mapping);
    partial_sched = partial_sched.intersect_domain(mapping);
  } else if (!thread_mapping.is_empty()) {
    partial_sched = partial_sched.intersect_domain(thread_mapping);
  } else if (!block_mapping.is_empty()) {
    partial_sched = partial_sched.intersect_domain(block_mapping);
  }

  CreateTensorCluster(node, partial_sched);

  isl::schedule sch = schedule_;

  auto thread_schedule = MapDomainAllWithType(root_node, thread_cfg, scop_info_.upa_node_mapping_, THREAD_MARKER);
  auto block_schedule = MapDomainAllWithType(root_node, block_cfg, scop_info_.upa_node_mapping_, BLOCK_MARKER);

  auto tmp_node = res_node;
  if (node.isa<isl::schedule_node_band>()) {
    tmp_node = res_node.child(0);
  }

  int64_t alloc_threads = 1;
  if (thread_cfg != nullptr) {
    for (size_t i = 0; i < thread_cfg->bound; ++i) {
      alloc_threads *= thread_cfg->GetAt(i).second;
    }
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
        if (GetTensorMark(buffer_info.dst_tensor_id.get_name(), scop_info_) == TENSOR_C) {
          continue;
        }
      } else {
        if (GetTensorMark(buffer_info.dst_tensor_id.get_name(), scop_info_) != TENSOR_C) {
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

    if (!scop_info_.user_config_.GetEnableTensorCore() && !scop_info_.user_config_.GetEnableMatmul() &&
        !scop_info_.user_config_.GetEnableVectorization()) {
      if (!ReuseTensorCluster(*fp_cluster, partial_sched_mupa)) {
        continue;
      }
    }

    auto tensor_size = std::accumulate(box_sizes.begin(), box_sizes.end(), 1, std::multiplies<size_t>());
    if (scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
      tensor_size = (buffer_info.tensor_id.get_name() == local_tensor_c_) ? (tensor_size / alloc_threads)
                                                                          : (tensor_size * 2 / alloc_threads);
    }
    auto data_bytes = scop_info_.user_config_.GetDataBytes(buffer_info.tensor_id.get_name());
    tensor_size = tensor_size * std::max<int>(1, data_bytes / BYTES_PER_REGISTER);
    size_t memory_requirement = tensor_size * alloc_threads;

    if (memory_requirement <= remain_memory_) {
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

      remain_memory_ -= memory_requirement;

      // active_buffer_footprints for codegen
      scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(std::make_pair(
        active_domains, BufferedFootPrintInfo{std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster)),
                                              partial_sched, dst_tensor_id}));
      buffer_info.find_buffer = true;
    }
  }
  sch = node.get_schedule();
  return sch;
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
      if (shared_dst_tensor_ids.count(item.get_name() + SHARE_SUFFIX)) {
        continue;
      }

      std::unordered_set<std::string> tensors = scop_info_.analysis_result_.GetTensorsNotPromote();
      if (tensors.count(item.get_name())) {
        continue;
      }

      tensor_list.push_back(item);
    }
  }

  std::vector<BufferDefInfo> promoted_infos;

  for (const auto &item : tensor_list) {
    if (scop_info_.user_config_.GetEnableMatmul() && !hoist_tensor_all_) {
      if (!hoist_compute_local_tensor_) {
        if (GetTensorMark(item.get_name(), scop_info_) == TENSOR_C) {
          continue;
        }
      } else {
        if (GetTensorMark(item.get_name(), scop_info_) != TENSOR_C) {
          continue;
        }
      }
    }

    isl::id dst_tensor_id = GetGpuIndexDstId(GpuMemType::LOCAL, item);
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

  for (auto promoted_info : promoted_infos) {
    auto name = promoted_info.tensor_id.get_name();
    if (CheckRAW(name)) {
      scop_info_.analysis_result_.buffer_def_infos_.insert(scop_info_.analysis_result_.buffer_def_infos_.begin(),
                                                           promoted_info);
    } else {
      scop_info_.analysis_result_.buffer_def_infos_.push_back(promoted_info);
    }
  }
}

bool RegisterMemoryManager::CheckRAW(std::string &name) {
  isl::union_map reads = scop_info_.analysis_result_.GetReads();
  isl::union_map reads_filter = isl::union_map::empty(scop_info_.GetCtx());
  auto reads_range = reads.range();
  isl::union_set reads_uset = isl::union_set::empty(scop_info_.GetCtx());
  reads_range.foreach_set([this, name, &reads_uset](const isl::set &s) -> void {
    std::string tensor_name = s.get_tuple_name();
    if (tensor_name == name) {
      reads_uset = reads_uset.add_set(s);
    }
  });

  reads_filter = reads.intersect_range(reads_uset);
  isl::union_map writes = scop_info_.analysis_result_.GetWrites();
  isl::union_map writes_filter = isl::union_map::empty(scop_info_.GetCtx());
  auto writes_range = writes.range();
  isl::union_set writes_uset = isl::union_set::empty(scop_info_.GetCtx());
  writes_range.foreach_set([this, name, &writes_uset](const isl::set &s) -> void {
    std::string tensor_name = s.get_tuple_name();
    if (tensor_name == name) {
      writes_uset = writes_uset.add_set(s);
    }
  });

  writes_filter = writes.intersect_range(writes_uset);
  isl::union_map rawmap = ComputeRAW(scop_info_.origin_schedule_, reads_filter, writes_filter);
  if (rawmap.is_empty()) {
    return false;
  } else {
    return true;
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
  if (!bands.size()) {
    return tmp_sch;
  }

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
  root.foreach_descendant_top_down([&hoist_register_node, this](const isl::schedule_node &node) -> bool {
    if (node.isa<isl::schedule_node_sequence>()) {
      auto sequence_node = node.as<isl::schedule_node_sequence>();
      if (sequence_node.parent().isa<isl::schedule_node_extension>() &&
          sequence_node.parent().parent().isa<isl::schedule_node_band>()) {
        hoist_register_node = sequence_node.parent().parent();
        return false;
      } else if (sequence_node.parent().isa<isl::schedule_node_band>()) {
        hoist_register_node = sequence_node.parent();
        return false;
      }
    }

    if (node.isa<isl::schedule_node_mark>()) {
      auto mark_node = node.as<isl::schedule_node_mark>();
      if (scop_info_.user_config_.GetEnableVectorization()) {
        if (mark_node.get_id().get_name() == THREAD_MARKER &&
            mark_node.child(0).child(0).isa<isl::schedule_node_band>()) {
          hoist_register_node = mark_node.child(0).child(0);
          return false;
        }
      } else if (mark_node.get_id().get_name() == THREAD_MARKER && mark_node.parent().isa<isl::schedule_node_band>()) {
        hoist_register_node = mark_node.parent();
        return false;
      }
    }
    return true;
  });
  return hoist_register_node;
}

isl::schedule_node RegisterMemoryManager::PromotedNodeUnderSequence(isl::schedule_node_sequence &node) {
  int band_node_num = 0;
  auto root = node.get_schedule().get_root();
  auto tmp_node = root;

  for (size_t i = 0; i < node.n_children(); ++i) {
    if (IsReadOrWriteBand(node.child(i).child(0))) {
      continue;
    }
    band_node_num += 1;
    tmp_node = node.child(i);
  }

  auto hoist_register_node = root;
  if (band_node_num == 1) {
    tmp_node.foreach_descendant_top_down([&hoist_register_node](const isl::schedule_node &node) -> bool {
      if (node.isa<isl::schedule_node_mark>()) {
        auto mark_node = node.as<isl::schedule_node_mark>();
        if (mark_node.get_id().get_name() == THREAD_MARKER &&
            mark_node.child(0).child(0).isa<isl::schedule_node_band>()) {
          hoist_register_node = mark_node.child(0).child(0);
          return false;
        }
      }
      return true;
    });
  }
  return hoist_register_node;
}

isl::schedule_node RegisterMemoryManager::GetVectorizationPromotedNode(isl::schedule_node &root) {
  isl::schedule_node hoist_register_node = root;
  root.foreach_descendant_top_down([&hoist_register_node, this](const isl::schedule_node &node) -> bool {
    if (node.isa<isl::schedule_node_sequence>()) {
      auto sequence_node = node.as<isl::schedule_node_sequence>();
      if (sequence_node.parent().isa<isl::schedule_node_extension>() &&
          sequence_node.parent().parent().isa<isl::schedule_node_band>()) {
        hoist_register_node = PromotedNodeUnderSequence(sequence_node);
        return false;
      } else if (sequence_node.parent().isa<isl::schedule_node_band>()) {
        return false;
      }
    }

    if (node.isa<isl::schedule_node_mark>()) {
      auto mark_node = node.as<isl::schedule_node_mark>();
      if (mark_node.get_id().get_name() == THREAD_MARKER &&
          mark_node.child(0).child(0).isa<isl::schedule_node_band>()) {
        hoist_register_node = mark_node.child(0).child(0);
        return false;
      }
    }
    return true;
  });
  return hoist_register_node;
}

isl::schedule RegisterMemoryManager::HoistRegisterMemoryOnMark(isl::schedule_node root) {
  std::string config_shared_tensors = scop_info_.user_config_.GetSharedTensors();
  auto c_mark = PROMOTE_GLOBAL_TO_REGISTER_C;
  if (config_shared_tensors.find(local_tensor_c_) != std::string::npos) {
    c_mark = PROMOTE_SHARED_TO_REGISTER_C;
  }

  auto mark_node = CollectMarkNode(root, c_mark).at(0);
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
  auto ab_mark = PROMOTE_SHARED_TO_REGISTER_AB;
  auto mark_ab_node = CollectMarkNode(hoist_ab_root, ab_mark).at(0);
  auto hoist_ab_node = mark_ab_node.del().parent();
  auto hoist_ab_depth = hoist_ab_node.schedule_depth();
  hoist_compute_local_tensor_ = false;
  sch = HoistRegisterMemoryOnDepth(hoist_ab_node, hoist_ab_depth);

  return sch;
}

std::string RegisterMemoryManager::GetPromotedWriteName() {
  std::string write_name = GML_WRITE_ID_NAME;
  std::string shared_tensors = shared_tensors_;
  if (shared_tensors.find(local_tensor_c_) != std::string::npos) {
    write_name = SHARED_WRITE_ID_NAME;
  }
  return write_name;
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

// According to the value of the conv interface, the size of the tensor is split to confirm the size of the fragment.
isl::schedule_node RegisterMemoryManager::TileTensorAccordingInterfaceValue(isl::schedule_node &root) {
  CHECK(scop_info_.user_config_.GetReplaceConfig().count(WARP_COMPUTE)) << "Cannot map to warp.";
  std::string write_name = GetPromotedWriteName();
  auto CollectReadWriteFilter = [this, write_name](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>()) {
      return node;
    }
    bool is_all_sets_read_or_write = IsReadOrWriteTensor(node, SHARED_READ_ID_NAME, write_name);
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

  return root.map_descendant_bottom_up(CollectReadWriteFilter);
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

isl::schedule RegisterMemoryManager::RunMatmul(isl::schedule_node root) {
  GetActualPromotedSharedTensors();
  auto sch = HoistRegisterMemoryOnMark(root);
  if (scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
    root = sch.get_root();
    sch = TileTensorAccordingInterfaceValue(root).get_schedule();
  }
  std::string write_name = GetPromotedWriteName();
  std::string marker_name = PROMOTE_REGISTER_TO_GLOBAL;
  if (write_name == SHARED_WRITE_ID_NAME) {
    marker_name = PROMOTE_REGISTER_TO_SHARED;
  }
  sch = InsertMarkerForThreadGroup(sch, write_name, marker_name);
  return sch;
}

isl::schedule RegisterMemoryManager::RunReduce(isl::schedule_node root) {
  auto sch = root.get_schedule();
  auto res_node = GetRegisterPromotedNode(root);
  if (res_node.isa<isl::schedule_node_band>()) {
    auto depth = UpdateDepth(res_node);
    sch = HoistRegisterMemory(root, depth);
  }
  return sch;
}

isl::schedule RegisterMemoryManager::RunElementWise(isl::schedule_node root) {
  auto sch = root.get_schedule();
  auto CollectGMLReadWriterFilter = [this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>()) {
      return node;
    }

    bool is_all_sets_read_or_write = IsReadOrWriteTensor(node, GML_READ_ID_NAME, GML_WRITE_ID_NAME);
    if (!is_all_sets_read_or_write) {
      return node;
    }

    auto filter = node.as<isl::schedule_node_filter>().filter();
    auto filter_set = filter.unwrap();
    bool is_vectorization_tensor = false;
    filter_set.range().foreach_set([this, &is_vectorization_tensor](const isl::set &s) -> void {
      std::string promoted_tensor = s.get_tuple_name();
      for (auto buffer : scop_info_.analysis_result_.active_buffer_footprints_) {
        auto cluster_id = buffer.second.cluster_id;
        if (cluster_id.get_name() == promoted_tensor) {
          auto cluster = buffer.second.cluster;
          auto box_sizes = cluster->GetFixedBoxSizes();
          auto local_size = 1;
          for (auto i : box_sizes) {
            local_size = local_size * i;
          }
          if (local_size == 4 || local_size == 8) {
            // vectorization mode fp32 or fp16
            is_vectorization_tensor = true;
          }
        }
      }
    });

    if (!is_vectorization_tensor) {
      return node;
    }

    if (node.n_children() > 0 && node.child(0).isa<isl::schedule_node_band>()) {
      node = node.child(0).insert_mark(PROMOTE_VECTORIZATION);
      node = node.parent();
    }
    return node;
  };

  isl::schedule_node res_node = root;
  if (scop_info_.user_config_.GetEnableVectorization()) {
    res_node = GetVectorizationPromotedNode(root);
    if (res_node.isa<isl::schedule_node_domain>()) {
      return sch;
    }
  } else {
    res_node = GetRegisterPromotedNode(root);
  }

  if (res_node.isa<isl::schedule_node_band>()) {
    auto depth = UpdateDepth(res_node);

    sch = HoistRegisterMemory(root, depth);

    if (scop_info_.user_config_.GetEnableVectorization()) {
      auto tmp_root = sch.get_root();
      tmp_root = tmp_root.map_descendant_bottom_up(CollectGMLReadWriterFilter);
      sch = tmp_root.get_schedule();
    }
  }
  return sch;
}

isl::schedule RegisterMemoryManager::Run(isl::schedule sch) {
  if (!scop_info_.user_config_.GetLocalTensors().empty()) {
    configed_tensors_ = Split(scop_info_.user_config_.GetLocalTensors(), " ");
  }
  if (scop_info_.user_config_.GetEnableMatmul()) {
    local_tensor_c_ = GetMatmulTensorsName(scop_info_)[MATRIX_C];
  }

  sch = InsertContextNode(sch, scop_info_);

  if (!scop_info_.user_config_.GetUseRegisterMemory()) {
    return sch;
  }

  schedule_ = sch;
  auto root = sch.get_root();

  if (scop_info_.user_config_.GetEnableMatmul()) {
    sch = RunMatmul(root);
  } else if (scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    sch = RunReduce(root);
  } else {
    sch = RunElementWise(root);
  }

  return sch;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
