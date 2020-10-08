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

#include "shared_memory_manager.h"
#include "poly/schedule_tree_util.h"
#include "poly/scop.h"
#include "poly/dma_inject.h"
#include "poly/poly_util.h"
#include <vector>
#include <numeric>

namespace akg {
namespace ir {
namespace poly {

isl::schedule SharedMemoryManager::Run(isl::schedule sch) {
  if (!scop_info_.user_config_.UseSharedMemory()) {
    return sch;
  }
  schedule_ = sch;
  auto root = sch.get_root();

  // Update the variable/tensor to share
  if (!scop_info_.user_config_.GetSharedTensors().empty()) {
    configed_tensors_ = Split(scop_info_.user_config_.GetSharedTensors(), " ");
  }

  // Compute the depth where the shared memory have to be generated
  UpdateDepth(root);
  if (scop_info_.user_config_.GetSharedDepth() >= 0) {
    depth_ = scop_info_.user_config_.GetSharedDepth();
    use_config_ = true;
  }
  CHECK_GE(depth_, 0) << "shared depth should be greater than or equal with zero!";
  bank_conflict_ = scop_info_.user_config_.GetEnableBankConflict();
  shared_inversed_thread_map_ = scop_info_.user_config_.GetSharedInversedThreadMap();
  shared_vector_align_ = scop_info_.user_config_.GetSharedVectorAlign();

  // collect all bands at the given depth in the schedule tree
  size_t remain_memory = share_memory_size_;

  if (scop_info_.user_config_.GetEnableMatmul()) {
    remain_memory = tensor_core_share_memory_size_;
    root = HoistSharedMemoryOnMark(root, remain_memory, depth_);
  } else {
    root = HoistSharedMemoryOnDepth(root, remain_memory, depth_);
  }
  bool unroll_shared = scop_info_.user_config_.GetUnrollShared();
  root = MapCopiesToThreads(root, unroll_shared);
  schedule_ = root.get_schedule();

  auto node = schedule_.root().child(0);
  if (node.as<isl::schedule_node_context>()) {
    node = node.del();
  }
  node = InsertContextNode(node, scop_info_);
  schedule_ = node.schedule();

  return schedule_;
}

isl::schedule_node SharedMemoryManager::CollectMarkNode(isl::schedule_node root, const std::string mark) {
  isl::schedule_node hoist_node;
  root.foreach_descendant_top_down([&hoist_node, &mark](const isl::schedule_node &node) -> bool {
    if (auto mark_node = node.as<isl::schedule_node_mark>()) {
      // ignore nested mark nodes
      if (mark_node.get_id().get_name() == mark) {
        hoist_node = mark_node;
        return false;
      }
    }
    return true;
  });
  return hoist_node;
}

isl::schedule_node SharedMemoryManager::HoistSharedMemoryOnMark(const isl::schedule_node &root, size_t &remain_memory,
                                                                size_t depth) {
  auto ab_mark_node = CollectMarkNode(root, PROMOTE_GLOBAL_TO_SHARED_AB);
  auto ab_promote_node = ab_mark_node.parent();
  hoist_tensor_c_ = false;
  auto ab_res_node = ManageToShareBelow(this->schedule_, ab_promote_node, remain_memory);
  if (find(configed_tensors_.begin(), configed_tensors_.end(), tensor_c_) != configed_tensors_.end()) {
    auto c_mark_node = CollectMarkNode(ab_res_node.get_schedule().get_root(), PROMOTE_GLOBAL_TO_SHARED_C);
    auto c_promote_node = c_mark_node.parent();
    hoist_tensor_c_ = true;
    auto c_res_node = ManageToShareBelow(c_promote_node.get_schedule(), c_promote_node, remain_memory);
    return c_res_node;
  }
  return ab_res_node;
}

isl::schedule_node SharedMemoryManager::HoistSharedMemoryOnDepth(const isl::schedule_node &root, size_t &remain_memory,
                                                                 size_t depth) {
  auto fn = [depth, &remain_memory, this](isl::schedule_node node) -> isl::schedule_node {
    auto res_node = node;
    if (node.isa<isl::schedule_node_band>()) {
      if (ContainsDepth(node, depth)) {
        auto node_splitted = BandSplitAtDepth(node, depth);
        if (!use_config_ && IsAncestorMapToThread(node_splitted)) {
          LOG(INFO) << "a subtree under the thread_marker cannot "
                    << "be promoted.";
          return node;
        }
        res_node = ManageToShareBelow(this->schedule_, node_splitted, remain_memory);
      }
    }
    return res_node;
  };

  auto root_node = root;
  if (depth == 0) {
    root_node = GenerateEmptyBandInRoot(root_node);
    auto node_splitted = BandSplitAtDepth(root_node, depth);
    node_splitted = ManageToShareBelow(schedule_, node_splitted, remain_memory);
    return node_splitted;
  }

  return MapDescendantTopDown(root, fn);
}

isl::union_set SharedMemoryManager::GatherMappingsTo(MappingCfg *cfg) {
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
      auto filter_domain = filter.filter().intersect(CollectDomain(item));
      mapping = mapping.unite(filter.filter());
    }
  }
  return mapping;
}

isl::schedule_node SharedMemoryManager::MapCopiesToThreads(isl::schedule_node &root, bool unroll) {
  auto CollectReadWriteFilter = [&unroll, this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_filter>()) {
      return node;
    }

    bool is_all_sets_read_or_write = IsReadOrWriteTensor(node, READ_ID_NAME, WRITE_ID_NAME);
    if (!is_all_sets_read_or_write) {
      return node;
    }

    auto band_node = GetCanMappingNode(node);
    std::string atomic_type = InAtomicTensors(node);
    // split member that does not involved in thread mapping
    bool has_split = false;
    auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
    auto mem_size = band_node.as<isl::schedule_node_band>().n_member();
    if (mem_size > thread_cfg->bound) {
      band_node = band_node.as<isl::schedule_node_band>().split(mem_size - thread_cfg->bound);
      band_node = band_node.child(0);
      has_split = true;
    }

    if (shared_inversed_thread_map_) {
      // Pretille - To make a vectorize loop more apparent with only the information of the mapping
      const auto &domain =  band_node.as<isl::schedule_node_band>().get_partial_schedule().domain();
      const isl::id &current_computing_id_shared = domain.unwrap().range().set_list().get_at(0).get_tuple_id();

      std::vector<size_t> tensor_size;
      for (BufferDefInfo &buffer_info : scop_info_.analysis_result_.buffer_def_infos_) {
        if (current_computing_id_shared == buffer_info.dst_tensor_id) {
          tensor_size = buffer_info.sizes;
        }
      }
      // Reverse because thread is innermost map
      std::reverse(tensor_size.begin(),tensor_size.end());

      auto ctx = band_node.ctx();
      const auto &space = band_node.as<isl::schedule_node_band>().get_space();
      const auto n_member = band_node.as<isl::schedule_node_band>().n_member();
      isl::multi_val tile_size = isl::multi_val::zero(space);
      for (size_t i = 0; i < n_member; ++i) {
        const size_t size = tensor_size[i]/thread_cfg->GetAt(i).second;
        tile_size = tile_size.set_val(n_member-1 - i, isl::val(ctx, size!=0?size:1));
      }
      band_node = TileBand(band_node, tile_size);
    }

    auto mapping_cfg = thread_cfg;
    if (scop_info_.user_config_.GetVectorLoadType() || scop_info_.user_config_.GetEnableTensorCoreUsePoly()) {
      scop_info_.user_config_.SetEnableOneDimThread(true);
    }
    if (scop_info_.user_config_.GetEnableOneDimThread()) {
      mapping_cfg = GetCurrentConfig(band_node);

      bool use_thread_cfg = true;
      if (mapping_cfg != nullptr && mapping_cfg->bound != thread_cfg->bound) {
        use_thread_cfg = false;
      } else if (mapping_cfg != nullptr && mapping_cfg->bound == thread_cfg->bound) {
        for (size_t i = 0; i < mapping_cfg->bound; ++i) {
          if (mapping_cfg->GetAt(i).second != thread_cfg->GetAt(i).second) {
            use_thread_cfg = false;
            break;
          }
        }
      }
      if (use_thread_cfg) {
        mapping_cfg = thread_cfg;
      }
    }
    Mapping mapping;
    auto after_map_pair = MapInnerDimToThreads(band_node, true, mapping_cfg, mapping, false);
    band_node = after_map_pair.first;
    auto InsertAtomicMarker = [atomic_type, this](isl::schedule_node atomic_node) -> isl::schedule_node {
      if (atomic_type != "" && atomic_node.has_children() && atomic_node.child(0).isa<isl::schedule_node_filter>()) {
        atomic_node =
          atomic_node.child(0).child(0).insert_mark(isl::id(atomic_node.ctx(), AtomicMarker("_" + atomic_type)));
        atomic_node = atomic_node.parent().parent();
      }
      return atomic_node;
    };
    if (band_node.isa<isl::schedule_node_mark>()) {
      band_node = InsertAtomicMarker(band_node);
    } else if (band_node.has_children() && band_node.child(0).isa<isl::schedule_node_mark>()) {
      band_node = InsertAtomicMarker(band_node.child(0));
      band_node = band_node.parent();
    }

    if (has_split) {
      band_node = band_node.parent();
    }

    if (unroll) {
      band_node = UnrollByMarkOptions(band_node, scop_info_.user_config_.GetMaxUnrollLoop());
    }

    node = band_node.parent();
    return node;
  };

  return root.map_descendant_bottom_up(CollectReadWriteFilter);
}

MappingCfg *SharedMemoryManager::GetCurrentConfig(isl::schedule_node &node) {
  std::string id_name = GetPromotionTensorName(node, scop_info_.analysis_result_.buffer_def_infos_);
  if (id_name.empty()) {
    return nullptr;
  }

  bool enable_vectorization = true;
  auto vector_load_type = scop_info_.user_config_.GetVectorLoadType();
  if (vector_load_type == 0) {
    enable_vectorization = false;
  }

  auto shares_tensor_bits_map = scop_info_.analysis_result_.GetSharedTensorBitsMap();
  if (enable_vectorization && !shares_tensor_bits_map.count(id_name)) {
    enable_vectorization = false;
  }

  int vectorization_loop = 0;
  if (enable_vectorization) {
    vectorization_loop = vector_load_type / shares_tensor_bits_map[id_name];
  }

  auto replace_cfg_map = scop_info_.user_config_.GetReplaceConfig();
  id_name = PROMOTE + id_name;
  if (replace_cfg_map.count(id_name) == 0) {
    auto partial_schedule = node.as<isl::schedule_node_band>().get_partial_schedule();
    auto upa_list = GetUPAList(node, partial_schedule, true, false);

    auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg != nullptr) << "thread config is null";

    int total_thread = 1;
    for (size_t i = 0; i < thread_cfg->bound; ++i) {
      total_thread *= thread_cfg->GetAt(i).second;
    }

    std::string new_cfg = "";
    int mapping_dim = std::min(static_cast<int>(upa_list.size()), static_cast<int>(thread_cfg->MaxDim()));
    for (int i = 0; i < mapping_dim; ++i) {
      auto extend = upa_list.get_at(i).max_val().get_num_si() + 1;
      if (extend > total_thread || (i == mapping_dim - 1 && extend < total_thread)) {
        new_cfg += (std::to_string(total_thread) + " ");
        break;
      }

      if (i == 0 && enable_vectorization) {
        extend /= vectorization_loop;
      }
      total_thread /= extend;
      new_cfg += (std::to_string(extend) + " ");
    }

    if (new_cfg.empty()) {
      return nullptr;
    }
    scop_info_.user_config_.RecordReplaceConfig(id_name, new_cfg, MappingType::REPLACE_THREADS);
  }
  auto mapping_cfg = scop_info_.user_config_.GetReplaceConfig()[id_name];

  if (enable_vectorization) {
    isl::multi_val tile_size;
    auto ctx = node.ctx();
    auto space = node.as<isl::schedule_node_band>().get_space();
    tile_size = isl::multi_val::zero(space);

    auto n_member = node.as<isl::schedule_node_band>().n_member();
    for (size_t i = 0; i < n_member - 1; ++i) {
      tile_size = tile_size.set_val(i, isl::val(ctx, 1));
    }
    tile_size = tile_size.set_val(n_member - 1, isl::val(ctx, vectorization_loop));

    node = TileBand(node, tile_size).child(0);
    node = node.insert_mark(PROMOTE_VECTORIZATION).parent();
  }
  return mapping_cfg;
}

isl::schedule_node SharedMemoryManager::ManageToShareBelow(const isl::schedule &root_sch, isl::schedule_node &node,
                                                           size_t &remaining_memory) {
  isl::schedule_node root_node = root_sch.get_root();

  CHECK(use_config_ || !IsAncestorMapToThread(node)) << "shared memory promotion cannot below thread_marker.";

  auto partial_sched = LocalSchedule(node);

  // Collect block config.
  auto cfg = scop_info_.user_config_.GetBlockConfig();
  isl::union_set mapping;
  for (auto it : scop_info_.user_config_.GetReplaceConfig()) {
    auto cfg = it.second;
    if (cfg->type == MappingType::REPLACE_BLOCKS) {
      if (mapping.is_null()) {
        mapping = GatherMappingsTo(cfg);
      } else {
        mapping = mapping.intersect(GatherMappingsTo(cfg));
      }
    }
  }
  if (mapping.is_null()) {
    mapping = GatherMappingsTo(cfg);
  }

  auto out_sched = partial_sched.intersect_domain(mapping);
  CreateClusterList(node, out_sched);
  auto new_node = HoistClusters(root_node, node, remaining_memory);
  auto sync_manager = scop_info_.sync_manager_;
  return sync_manager.InsertPromotionSync(new_node);
}

std::set<std::string> SharedMemoryManager::AnalysisReduceTensors() {
  std::set<std::string> id_sets;
  if (!scop_info_.user_config_.GetEnableAkgReduceLib()) {
    return id_sets;
  }

  /*************************************************
   * In order to enable cuda atomic operator, add
   * these tensors for shared memory promotion list
   *************************************************/
  auto atomic_tensors = scop_info_.analysis_result_.GetAtomicTensors();
  if (!atomic_tensors.empty()) {
    id_sets.clear();
    for (const auto &item : atomic_tensors) {
      if (id_sets.count(item.tensor_name) == 0) {
        id_sets.emplace(item.tensor_name);
      }
    }
  }

  /***********************************************
   * For the condition that it is without cuda
   * atomic usage, but with reduce operation.
   * Also need to add these tensors for shared memory
   * promotion list.
   *********************************************/
  auto reduce_out_tensors = scop_info_.analysis_result_.GetReduceOutTensors();
  for (const auto &item : reduce_out_tensors) {
    if (id_sets.count(item) == 0) {
      id_sets.emplace(item);
    }
  }

  return id_sets;
}

void SharedMemoryManager::CreateClusterList(const isl::schedule_node &node, const isl::union_map &outer_sch) {
  isl::union_map reads = scop_info_.analysis_result_.GetReads();
  isl::union_map writes = scop_info_.analysis_result_.GetWrites();
  isl::union_map copyin = scop_info_.analysis_result_.GetCopyin();
  isl::union_map fake_copyin = scop_info_.analysis_result_.GetFakeCopyin();

  auto read_map = scop_info_.StmtReadMap();
  auto write_map = scop_info_.StmtWriteMap();
  auto stmt_map = scop_info_.analysis_result_.GetStmtOpInfoMap();
  std::vector<isl::id> tensor_list;
  std::set<std::string> id_sets;
  std::set<std::string> read_sets;
  std::set<std::string> write_sets;
  for (auto item : read_map) {
    for (auto item_id : item.second) {
      if (read_sets.count(item_id.get_name()) == 0) {
        read_sets.insert(item_id.get_name());
      }
    }
  }
  for (auto item : write_map) {
    for (auto item_id : item.second) {
      if (write_sets.count(item_id.get_name()) == 0) {
        write_sets.insert(item_id.get_name());
      }
    }
  }
  /*********************************************************
   * manage only read tensors to share memory
   * for read and write tensor, should be managed to local memory
   ********************************************************/
  std::set_difference(read_sets.begin(), read_sets.end(), write_sets.begin(), write_sets.end(),
                      std::inserter(id_sets, id_sets.begin()));

  if (scop_info_.user_config_.GetEnableAkgReduceLib()) {
    id_sets = AnalysisReduceTensors();
  }

  if (!configed_tensors_.empty()) {
    id_sets.clear();
    for (const auto &item : configed_tensors_) {
      if (id_sets.count(item) == 0) {
        id_sets.emplace(item);
      }
    }
  }

  if (scop_info_.user_config_.GetEnableMatmul()) {
    std::unordered_map<std::string, std::string> matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();
    for (auto i : matmul_map) {
      if (i.second == MATRIX_C) {
        tensor_c_ = i.first;
      } else if (i.second == MATRIX_A) {
        tensor_a_ = i.first;
      } else if (i.second == MATRIX_B) {
        tensor_b_ = i.first;
      }
    }
    if (id_sets.count(tensor_a_) == 0) {
      id_sets.emplace(tensor_a_);
    }
    if (id_sets.count(tensor_b_) == 0) {
      id_sets.emplace(tensor_b_);
    }
  }

  for (auto item : id_sets) {
    tensor_list.push_back(isl::id(scop_info_.ctx_, item));
  }

  for (const auto &item : tensor_list) {
    if (scop_info_.user_config_.GetEnableMatmul()) {
      if (!hoist_tensor_c_) {
        if (item.get_name().find(tensor_a_) == std::string::npos &&
            item.get_name().find(tensor_b_) == std::string::npos) {
          continue;
        }
      } else {
        if (item.get_name().find(tensor_a_) != std::string::npos ||
            item.get_name().find(tensor_b_) != std::string::npos) {
          continue;
        }
      }
    }

    isl::id dst_tensor_id = GpuDstId(GpuMemType::SHARED, item);
    std::vector<size_t> buffer_sizes;
    std::vector<std::pair<isl::id, MemType>> data_stream;
    data_stream.push_back(std::make_pair(item, MemType::DDR));
    data_stream.push_back(std::make_pair(item, MemType::SHARED_));
    BufferDefInfo promoted_info = BufferDefInfo{item,
                                                dst_tensor_id,
                                                item,
                                                MemType::DDR,
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
      scop_info_.analysis_result_.buffer_def_infos_.push_back(promoted_info);
    }
  }
}

void SharedMemoryManager::GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info) {
  auto fp_cluster = tensor_info.footprints_cluster;
  std::vector<size_t> sizes;
  if (fp_cluster == nullptr) {
    tensor_info.AddSize(node, sizes);
    return;
  }
  sizes = fp_cluster->GetFixedBoxSizes();
  if (scop_info_.user_config_.GetEnableMatmul() && sizes.back() % 2 == 0) {
    sizes.back() += 16;
  }

  sizes = OptimizeSharedDimension(sizes);

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
  if (scop_info_.user_config_.GetVectorLoadType()) {
    scop_info_.analysis_result_.RecoreSharedTensorBitsMap(tensor_id.get_name(),
                                                          scop_info_.GetDtypeOf(tensor_id).bits());
  }

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(node, sizes);
}

isl::schedule_node SharedMemoryManager::HoistClusters(const isl::schedule_node &root_node,
                                                      const isl::schedule_node &node, size_t &remaining_memory) {
  auto partial_sched_mupa = ShortScheduleMupa(root_node, node);
  auto res_node = node;
  for (size_t index = 0; index < scop_info_.analysis_result_.buffer_def_infos_.size(); index++) {
    BufferDefInfo &buffer_info = scop_info_.analysis_result_.buffer_def_infos_[index];
    auto fp_cluster = buffer_info.GetFootPrintClusterGPU(node);
    if ((fp_cluster == nullptr || !fp_cluster->foot_print_.box.is_valid())) {
      continue;
    }
    auto id = buffer_info.tensor_id;

    if (scop_info_.user_config_.GetEnableMatmul()) {
      if (!hoist_tensor_c_) {
        if (id.get_name().find(tensor_a_) == std::string::npos && id.get_name().find(tensor_b_) == std::string::npos) {
          continue;
        }
      } else {
        if (id.get_name().find(tensor_a_) != std::string::npos || id.get_name().find(tensor_b_) != std::string::npos) {
          continue;
        }
      }
    }

    auto box_sizes = fp_cluster->GetFixedBoxSizes();
    if (box_sizes.size() == 0) {
      LOG(FATAL) << "Can not manage a scalar tensor";
    }

    box_sizes = OptimizeSharedDimension(box_sizes);

    auto approximation_size = std::accumulate(box_sizes.begin(), box_sizes.end(), 1, std::multiplies<size_t>());
    size_t byte = Bytes(id);
    size_t memory_requirement = approximation_size * byte;
    bool use_reuse_filter = true;
    if (InAtomicTensors(buffer_info.tensor_id.name()) || InReduceTensors(buffer_info.tensor_id.name())) {
      use_reuse_filter = false;
    }
    bool is_injective = !ReuseTensorCluster(*fp_cluster, partial_sched_mupa);
    if (scop_info_.user_config_.GetEnableMatmul()) {
      is_injective = false;
    }
    if (memory_requirement < remaining_memory) {
      if (use_reuse_filter && is_injective && !CoalescingAccessWay(root_node, res_node, *fp_cluster)) {
        continue;
      }
      GatherBufferFootprintDefInfo(res_node, buffer_info);
      res_node = HoistToBlockThreadMemory(res_node, GpuMemType::SHARED, id, *(fp_cluster), true);
      remaining_memory -= memory_requirement;

      // collect active_buffer_footprints_ info for codegen
      auto out_schedule = LocalSchedule(res_node);
      auto active_domains = CollectDomain(res_node);
      auto dst_id = GpuDstId(GpuMemType::SHARED, id);
      scop_info_.analysis_result_.active_buffer_footprints_.emplace_back(std::make_pair(
        active_domains,
        BufferedFootPrintInfo{std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster)), out_schedule, dst_id}));
      buffer_info.find_buffer = true;
    }
  }
  return res_node;
}

isl::schedule_node SharedMemoryManager::HoistToBlockThreadMemory(isl::schedule_node &tree, GpuMemType type,
                                                                 const isl::id &tensor_id,
                                                                 TensorFootprintCluster &cluster,
                                                                 bool force_last_extension_odd) {
  auto out_schedule = LocalSchedule(tree);
  auto active_domains = CollectDomain(tree);

  isl::id dst_tensor_id = GpuDstId(type, tensor_id);
  auto sizes = cluster.GetFixedBoxSizes();
  if (force_last_extension_odd) {
    sizes = OptimizeSharedDimension(sizes);
  }

  auto res_node = PlaceOuterDataCopyBelow(scop_info_, tree, cluster, tensor_id, dst_tensor_id, out_schedule,
                                          schedule_.get_domain().get_space());
  return res_node;
}

bool SharedMemoryManager::ReuseTensorCluster(const TensorFootprintCluster &cluster,
                                             const isl::multi_union_pw_aff &outer_pw_aff) {
  isl::union_map state_schedule_mapping = ScheduleTensorMapping(outer_pw_aff, cluster.OrigianlAccessRelations());
  /* Here we use the property of bijective to decide whether promote this tensor to shared.
   * For element wise operator, S -> tensor_schedule is bijective.
   * It should not be promoted to shared memory.
   * For reduced operator, S -> tensor_schedule is not bijective.
   * It should be promoted to shared memory.
   * For stencil operator in sciencetific computing, S -> tensor_schedule is not bijective.
   * It should be promoted to shared memory.
   * *******************************************************************************************/
  return !(state_schedule_mapping.is_bijective());
}

bool SharedMemoryManager::CoalescingAccessWay(const isl::schedule_node &root, const isl::schedule_node &node,
                                              const TensorFootprintCluster &cluster) {
  isl::union_map original = cluster.OrigianlAccessRelations();
  size_t tensor_dim = cluster.foot_print_.GetBoxDim();
  std::vector<isl::schedule_node> thread_marker = CollectFnNode(IsThreadMappedMark, root);
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
      auto schedule_space = access.get_space().domain();
      auto tensor_space = access.get_space().range();
      auto element_next = CreateMapIncreaseDim(tensor_space, tensor_dim - 1);
      auto schedule_next = CreateMapIncreaseDim(schedule_space, inner_depth - 1);
      auto access_by_adjacent_inner = schedule_next.apply_domain(access).apply_range(access);
      if (!access_by_adjacent_inner.is_subset(element_next)) {
        return true;
      }
    }
  }
  return false;
}

void SharedMemoryManager::UpdateDepth(const isl::schedule_node &root) {
  auto outer_band = GetOuterBand(root);
  auto cfg = scop_info_.user_config_.GetBlockConfig();
  if (outer_band.isa<isl::schedule_node_band>()) {
    auto block_depth = cfg->bound + 1;
    auto outer_band_depth = outer_band.as<isl::schedule_node_band>().n_member();
    block_depth = std::min<int>(block_depth, outer_band_depth);
    if (block_depth > outer_band_depth && !UnderThreadMarker(block_depth)) {
      depth_ = block_depth;
    } else {
      depth_ = outer_band_depth;
    }
  }
}

bool SharedMemoryManager::UnderThreadMarker(size_t depth) {
  isl::schedule_node root = this->schedule_.get_root();
  auto bands = BandsContainingScheduleDepth(root, depth);
  for (auto item : bands) {
    if (IsAncestorMapToThread(item)) {
      return true;
    }
  }
  return false;
}

std::string SharedMemoryManager::InAtomicTensors(isl::schedule_node &node) {
  if (!node.isa<isl::schedule_node_filter>()) {
    return "";
  }
  auto filter = node.as<isl::schedule_node_filter>().filter();
  auto filter_set = filter.unwrap();
  std::string atomic_type = "";
  filter_set.range().foreach_set([this, &atomic_type](const isl::set &s) -> void {
    std::string promoted_tensor = s.get_tuple_name();
    std::string posfix = SHARE_SUFFIX;
    std::string::size_type pos = promoted_tensor.find(posfix);
    if (pos != std::string::npos) {
      std::string tensor = promoted_tensor.substr(0, pos);
      for (const auto &item : scop_info_.analysis_result_.GetAtomicTensors()) {
        if (item.tensor_name == tensor) {
          atomic_type = item.tensor_type;
        }
      }
    }
  });
  return atomic_type;
}

bool SharedMemoryManager::InAtomicTensors(std::string name) {
  for (const auto &item : scop_info_.analysis_result_.GetAtomicTensors()) {
    if (item.tensor_name == name) {
      return true;
    }
  }
  return false;
}

bool SharedMemoryManager::InReduceTensors(std::string name) {
  for (const auto &item : scop_info_.analysis_result_.GetReduceOutTensors()) {
    if (item == name) {
      return true;
    }
  }
  return false;
}

std::string SharedMemoryManager::AtomicMarker(std::string type) { return ATOMIC_MARKER + type; }

size_t SharedMemoryManager::Bytes(const isl::id tensor_id) {
  Type type = scop_info_.GetDtypeOf(tensor_id);
  return static_cast<size_t>(type.bytes());
}

std::vector<size_t> SharedMemoryManager::OptimizeSharedDimension(std::vector<size_t> sizes) {
  std::vector<size_t> res = sizes;
  res = OptimizeBankConflict(res);
  res = OptimizeVectorAlign(res);
  return res;
}

std::vector<size_t> SharedMemoryManager::OptimizeBankConflict(std::vector<size_t> sizes) {
  std::vector<size_t> res = sizes;
  if (res.back() % 2 == 0) {
    if (scop_info_.user_config_.GetEnableMatmul()) {
      res.back() += 16;
    } else if (bank_conflict_ && res.back() < 32) {
      res.back() = 33;
    } else {
      res.back() += 1;
    }
  }
  return res;
}

std::vector<size_t> SharedMemoryManager::OptimizeVectorAlign(std::vector<size_t> sizes) {
  std::vector<size_t> res = sizes;
  if (shared_vector_align_ != 0) {
    int padsize = res.back() % shared_vector_align_;
    res.back() += padsize?(shared_vector_align_-padsize):0;
  }
  return res;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
