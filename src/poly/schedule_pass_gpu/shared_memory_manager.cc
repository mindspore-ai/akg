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
#include "poly/schedule_pass_gpu/operator_shared_strategy.h"
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

  PrepareInfoForPromotion(root);

  if (scop_info_.user_config_.GetEnableAkgReduceLib()) {
    return RunReduce(root);
  } else if (scop_info_.user_config_.GetEnableMatmul()) {
    return RunMatmul(root);
  } else {
    return RunElemwise(root);
  }
}

isl::schedule SharedMemoryManager::RunReduce(isl::schedule_node &root) {
  is_reduce_ = true;
  // collect all bands at the given depth in the schedule tree
  root = HoistSharedMemoryOnDepth(root).root();
  root = MapCopiesToThreads(root, unroll_shared_);
  schedule_ = root.get_schedule();
  schedule_ = InsertContextNode(schedule_, scop_info_);

  return schedule_;
}
isl::schedule SharedMemoryManager::RunElemwise(isl::schedule_node &root) {
  // collect all bands at the given depth in the schedule tree
  root = HoistSharedMemoryOnDepth(root).root();
  root = MapCopiesToThreads(root, unroll_shared_);
  schedule_ = root.get_schedule();
  schedule_ = InsertContextNode(schedule_, scop_info_);

  return schedule_;
}

isl::schedule SharedMemoryManager::RunMatmul(isl::schedule_node &root) {
  is_matmul_ = true;
  // collect all bands at the given depth in the schedule tree
  remain_memory_ = akg::common::ADVANCED_SHARED_MEMORY_SIZE;
  root = HoistSharedMemoryOnMark(root).root();
  root = MapCopiesToThreads(root, unroll_shared_);
  schedule_ = root.get_schedule();
  schedule_ = InsertMarkerForThreadGroup(schedule_, WRITE_ID_NAME, PROMOTE_SHARED_TO_GLOBAL);
  schedule_ = InsertContextNode(schedule_, scop_info_);

  return schedule_;
}

void SharedMemoryManager::PrepareInfoForPromotion(const isl::schedule_node &root) {
  // Update the variable/tensor to share
  if (!scop_info_.user_config_.GetSharedTensors().empty()) {
    configed_tensors_ = Split(scop_info_.user_config_.GetSharedTensors(), " ");
  }

  // Compute the depth where the shared memory have to be generated
  auto outer_band = GetOuterBand(root);
  if (outer_band.isa<isl::schedule_node_band>()) {
    depth_ = outer_band.as<isl::schedule_node_band>().n_member();
  }

  if (scop_info_.user_config_.GetSharedDepth() >= 0) {
    depth_ = scop_info_.user_config_.GetSharedDepth();
    use_config_ = true;
  }
  CHECK_GE(depth_, 0) << "shared depth should be greater than or equal with zero!";
  if (scop_info_.user_config_.HasTranspose()) {
    scop_info_.user_config_.SetEnableBankConflict(true);
  }
  bank_conflict_ = scop_info_.user_config_.GetEnableBankConflict();
  shared_inversed_thread_map_ = scop_info_.user_config_.GetSharedInversedThreadMap();
  shared_vector_align_ = scop_info_.user_config_.GetSharedVectorAlign();
  if (scop_info_.user_config_.GetVectorLoadType() && !scop_info_.user_config_.GetEnableVectorization() &&
      !scop_info_.user_config_.EnableStitchFusion()) {
    scop_info_.user_config_.SetEnableOneDimThread(true);
  }
  unroll_shared_ = scop_info_.user_config_.GetUnrollShared();
}

isl::schedule_node SharedMemoryManager::HoistSharedMemoryOnMark(const isl::schedule_node &root) {
  auto ab_mark_node = CollectMarkNodeOnPromotion(root, PROMOTE_GLOBAL_TO_SHARED_AB);
  auto ab_promote_node = ab_mark_node.parent();
  hoist_tensor_c_ = false;
  auto ab_res_node = ManageToShareBelow(this->schedule_, ab_promote_node);
  auto tensor_c_name = GetMatmulTensorsName(scop_info_)[MATRIX_C];
  if (find(configed_tensors_.begin(), configed_tensors_.end(), tensor_c_name) != configed_tensors_.end()) {
    auto c_mark_node = CollectMarkNodeOnPromotion(ab_res_node.get_schedule().get_root(), PROMOTE_GLOBAL_TO_SHARED_C);
    auto c_promote_node = c_mark_node.parent();
    hoist_tensor_c_ = true;
    remain_memory_ = akg::common::ADVANCED_SHARED_MEMORY_SIZE;
    auto c_res_node = ManageToShareBelow(c_promote_node.get_schedule(), c_promote_node);
    return c_res_node;
  }
  return ab_res_node;
}

isl::schedule_node SharedMemoryManager::HoistSharedMemoryOnDepth(const isl::schedule_node &root) {
  auto fn = [this](isl::schedule_node node) -> isl::schedule_node {
    if (!node.isa<isl::schedule_node_band>()) {
      return node;
    }

    size_t depth_before = node.schedule_depth();
    size_t depth_after = depth_before + node.as<isl::schedule_node_band>().n_member();
    bool is_contain_depth = (depth_before < depth_ && depth_after >= depth_);
    if (!is_contain_depth) {
      return node;
    }

    auto node_splitted = BandSplitAtDepth(node, depth_);
    if (!use_config_ && IsAncestorMapToThread(node_splitted)) {
      LOG(INFO) << "a subtree under the thread_marker cannot "
                << "be promoted.";
      return node;
    }
    auto res_node = ManageToShareBelow(this->schedule_, node_splitted);
    return res_node;
  };

  auto root_node = root;
  if (depth_ == 0) {
    root_node = GenerateEmptyBandInRoot(root_node);
    auto node_splitted = BandSplitAtDepth(root_node, depth_);
    node_splitted = ManageToShareBelow(schedule_, node_splitted);
    return node_splitted;
  }

  return MapDescendantTopDown(root, fn);
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

    auto thread_cfg = scop_info_.user_config_.GetThreadConfig();

    auto mapping_cfg = thread_cfg;

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

    // split member that does not involved in mapping_cfg
    bool has_split = false;
    auto mem_size = band_node.as<isl::schedule_node_band>().n_member();
    if (mem_size > mapping_cfg->bound) {
      band_node = band_node.as<isl::schedule_node_band>().split(mem_size - mapping_cfg->bound);
      band_node = band_node.child(0);
      has_split = true;
    }

    if (shared_inversed_thread_map_) {
      // Pretille - To make a vectorize loop more apparent with only the information of the mapping
      const auto &domain = band_node.as<isl::schedule_node_band>().get_partial_schedule().domain();
      const isl::id &current_computing_id_shared = domain.unwrap().range().set_list().get_at(0).get_tuple_id();

      std::vector<size_t> tensor_size;
      for (BufferDefInfo &buffer_info : scop_info_.analysis_result_.buffer_def_infos_) {
        if (current_computing_id_shared == buffer_info.dst_tensor_id) {
          tensor_size = buffer_info.sizes;
        }
      }
      // Reverse because thread is innermost map
      std::reverse(tensor_size.begin(), tensor_size.end());

      auto ctx = band_node.ctx();
      const auto &space = band_node.as<isl::schedule_node_band>().get_space();
      const auto n_member = band_node.as<isl::schedule_node_band>().n_member();
      isl::multi_val tile_size = isl::multi_val::zero(space);
      for (size_t i = 0; i < n_member; ++i) {
        const size_t size = tensor_size[i] / thread_cfg->GetAt(i).second;
        tile_size = tile_size.set_val(n_member - 1 - i, isl::val(ctx, size != 0 ? size : 1));
      }
      band_node = TileBand(band_node, tile_size);
    }

    Mapping mapping;
    band_node = MapInnerDimToThreads(band_node, mapping_cfg, mapping, true, false);
    if (is_reduce_) {
      std::string atomic_type = InAtomicTensors(node);
      auto InsertAtomicMarker = [atomic_type, this](isl::schedule_node atomic_node) -> isl::schedule_node {
        if (atomic_type != "" && atomic_node.has_children() && atomic_node.child(0).isa<isl::schedule_node_filter>()) {
          atomic_node =
            atomic_node.child(0).child(0).insert_mark(isl::id(atomic_node.ctx(), AtomicMarker("_" + atomic_type)));
          scop_info_.analysis_result_.RecordAtomicMarkers(AtomicMarker("_" + atomic_type));
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

  // vectorization for elementwise OP
  if (!scop_info_.user_config_.GetEnableMatmul()) {
    enable_vectorization = false;
  }

  int vectorization_loop = 0;
  if (enable_vectorization) {
    vectorization_loop = vector_load_type / shares_tensor_bits_map[id_name];

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
    int mapping_dim = static_cast<int>(upa_list.size());
    for (int i = 0; i < mapping_dim; ++i) {
      auto extend = upa_list.get_at(i).floor().max_val().get_num_si() + 1;
      if (extend >= total_thread || (i == mapping_dim - 1 && extend < total_thread)) {
        new_cfg += (std::to_string(total_thread) + " ");
        break;
      }

      total_thread /= extend;
      new_cfg += (std::to_string(extend) + " ");
    }

    if (new_cfg.empty()) {
      return nullptr;
    }
    scop_info_.user_config_.RecordReplaceConfig(id_name, new_cfg, MappingType::REPLACE_THREADS, false);
  }
  auto mapping_cfg = scop_info_.user_config_.GetReplaceConfig()[id_name];

  return mapping_cfg;
}

isl::schedule_node SharedMemoryManager::ManageToShareBelow(const isl::schedule &root_sch, isl::schedule_node &node) {
  isl::schedule_node root_node = root_sch.get_root();

  CHECK(use_config_ || !IsAncestorMapToThread(node)) << "shared memory promotion cannot below thread_marker.";

  // Collect block config.
  auto cfg_block = scop_info_.user_config_.GetBlockConfig();
  isl::union_set mapping;
  if (is_matmul_) {
    for (auto it : scop_info_.user_config_.GetReplaceConfig()) {
      auto cfg = it.second;
      if (cfg->type == MappingType::REPLACE_BLOCKS) {
        if (mapping.is_null()) {
          mapping = GatherMappingsTo(root_node, cfg);
        } else {
          mapping = mapping.intersect(GatherMappingsTo(root_node, cfg));
        }
      }
    }
  }
  if (mapping.is_null()) {
    mapping = GatherMappingsTo(root_node, cfg_block);
  }

  auto partial_sched = LocalSchedule(node);
  auto out_sched = partial_sched.intersect_domain(mapping);
  if (is_reduce_) {
    ReduceSharedStrategy reduce_op(scop_info_);
    reduce_op.CreateClusterList(node, out_sched);
  } else if (is_matmul_) {
    BatchMatmulSharedStrategy matmul_op(scop_info_);
    matmul_op.CreateClusterList(node, out_sched, hoist_tensor_c_);
  } else {
    OperatorSharedStrategy other_op(scop_info_);
    other_op.CreateClusterList(node, out_sched);
  }
  return HoistClusters(root_node, node);
}

void SharedMemoryManager::GatherBufferFootprintDefInfo(const isl::schedule_node &node, BufferDefInfo &tensor_info) {
  auto fp_cluster = tensor_info.footprints_cluster;
  std::vector<size_t> sizes;
  if (fp_cluster == nullptr) {
    tensor_info.AddSize(node, sizes);
    return;
  }
  sizes = fp_cluster->GetFixedBoxSizes();

  isl::id tensor_id = tensor_info.tensor_id;
  Type type = scop_info_.GetDtypeOf(tensor_id);

  if (is_matmul_ && tensor_id.get_name() == GetMatmulTensorsName(scop_info_)[MATRIX_C]) {
    sizes.back() += 8;
  }

  if (bank_conflict_) {
    OptimizeSharedDimension(sizes, type);
  }

  isl::id cluster_id = tensor_info.dst_tensor_id;

  // build a Halide Node for cluster_id
  Array<Expr> shapes;
  for (auto i : sizes) {
    shapes.push_back(Expr(static_cast<int>(i)));
  }

  Tensor tensor = placeholder(shapes, type, cluster_id.get_name());
  const Buffer buffer = decl_buffer(shapes, scop_info_.GetDtypeOf(tensor_id), cluster_id.get_name());
  scop_info_.user_config_.SetBind(tensor, buffer);
  if (scop_info_.user_config_.GetVectorLoadType()) {
    scop_info_.analysis_result_.RecordSharedTensorBitsMap(tensor_id.get_name(),
                                                          scop_info_.GetDtypeOf(tensor_id).bits());
  }

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(node, sizes);
}

isl::schedule_node SharedMemoryManager::HoistClusters(const isl::schedule_node &root_node,
                                                      const isl::schedule_node &node) {
  auto partial_sched_mupa = ShortScheduleMupa(root_node, node);
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

    OptimizeSharedDimension(box_sizes, scop_info_.GetDtypeOf(id));

    auto approximation_size = std::accumulate(box_sizes.begin(), box_sizes.end(), 1, std::multiplies<size_t>());
    size_t byte = Bytes(id);
    size_t memory_requirement = approximation_size * byte;
    bool use_reuse_filter = true;
    if (InAtomicTensors(buffer_info.tensor_id.name()) || InReduceTensors(buffer_info.tensor_id.name()) || is_matmul_ ||
        scop_info_.user_config_.HasTranspose()) {
      use_reuse_filter = false;
    }
    bool is_injective = !ReuseTensorCluster(*fp_cluster, partial_sched_mupa);

    if (memory_requirement < remain_memory_) {
      bool need_shared_memory =
        !use_reuse_filter || !is_injective || CoalescingAccessWay(root_node, res_node, *fp_cluster);
      if (!need_shared_memory) {
        continue;
      }
      GatherBufferFootprintDefInfo(res_node, buffer_info);
      res_node = HoistToBlockThreadMemory(res_node, GpuMemType::SHARED, id, *(fp_cluster), true);
      remain_memory_ -= memory_requirement;

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
    OptimizeSharedDimension(sizes, scop_info_.GetDtypeOf(dst_tensor_id));
  }
  auto res_node = PlaceOuterDataCopyBelow(scop_info_, tree, cluster, tensor_id, dst_tensor_id, out_schedule,
                                          schedule_.get_domain().get_space());
  return res_node;
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

bool SharedMemoryManager::InAtomicTensors(const std::string &name) {
  for (const auto &item : scop_info_.analysis_result_.GetAtomicTensors()) {
    if (item.tensor_name == name) {
      return true;
    }
  }
  return false;
}

bool SharedMemoryManager::InReduceTensors(const std::string &name) {
  for (const auto &item : scop_info_.analysis_result_.GetReduceTensorInfoMap()) {
    if (item.second.write_tensor_name == name) {
      return true;
    }
  }
  return false;
}

std::string SharedMemoryManager::AtomicMarker(const std::string &type) { return ATOMIC_MARKER + type; }

size_t SharedMemoryManager::Bytes(const isl::id tensor_id) {
  Type type = scop_info_.GetDtypeOf(tensor_id);
  return static_cast<size_t>(type.bytes());
}

void SharedMemoryManager::OptimizeSharedDimension(std::vector<size_t> &sizes, Type type) {
  OptimizeBankConflict(sizes, type);
  OptimizeVectorAlign(sizes);
}

void SharedMemoryManager::OptimizeBankConflict(std::vector<size_t> &sizes, Type type) {
  if (sizes.back() % 2 != 0) {
    return;
  }
  if (bank_conflict_ && sizes.back() < 32) {
    sizes.back() = 33;
  } else {
    size_t pad = 1;
    if (scop_info_.user_config_.HasTranspose()) {
      pad = std::max<size_t>(1, 4 / type.bytes());
    }
    sizes.back() += pad;
  }
}

void SharedMemoryManager::OptimizeVectorAlign(std::vector<size_t> &sizes) {
  if (shared_vector_align_ == 0) {
    return;
  }
  int padsize = sizes.back() % shared_vector_align_;
  sizes.back() += padsize ? (shared_vector_align_ - padsize) : 0;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
