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
#include "poly/create_cluster.h"
#include "poly/schedule_pass_gpu/operator_mapping_strategy.h"
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
  if (!scop_info_.user_config_.GetUseSharedMemory()) {
    return sch;
  }
  schedule_ = sch;
  PrepareInfoForPromotion();
  schedule_ = HoistSharedMemory();
  schedule_ = InsertContextNode(schedule_, scop_info_);
  return schedule_;
}

void SharedMemoryManager::PrepareInfoForPromotion() {
  // Update the variable/tensor to share
  configed_tensors_ = scop_info_.user_config_.GetSharedTensors();
  bank_conflict_ = scop_info_.user_config_.GetEnableBankConflict();
  shared_inversed_thread_map_ = scop_info_.user_config_.GetSharedInversedThreadMap();
  shared_vector_align_ = scop_info_.user_config_.GetSharedVectorAlign();
  unroll_shared_ = scop_info_.user_config_.GetUnrollShared();
}

isl::schedule_node SharedMemoryManager::HoistSharedMemoryOnMark(const isl::schedule_node &orig_node) {
  current_outer_bn_ = scop_info_.analysis_result_.GetOuterBandNode(band_index_);
  SetPromotedMarkNames();
  auto node = orig_node;
  if (!current_outer_bn_->use_shared_memory) {
    node = InsertMarkerForRegisterPromotion(node);
    node = DeleUselessMarker(node, mark_names_);
    return node;
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

    if (tmp_mark_name == PROMOTE_GLOBAL_TO_SHARED_C) {
      remain_memory_ = akg::common::ADVANCED_SHARED_MEMORY_SIZE;
    }

    return HoistClusters(node.parent()).child(0);
  };

  for (auto name : mark_names_) {
    mark_name = name;
    node = MapDescendantTopDown(node, GetMarkNode);
  }
  node = MapCopiesToThreads(node, unroll_shared_);
  node = InsertMarkerForRegisterPromotion(node);
  node = DeleUselessMarker(node, mark_names_);

  return node;
}

isl::schedule SharedMemoryManager::HoistSharedMemory() {
  isl::schedule_node node = GetOuterBand(schedule_.root());
  if (node.isa<isl::schedule_node_band>()) {
    node = HoistSharedMemoryOnMark(node);
  } else {
    int number = static_cast<int>(node.n_children());
    for (int i = 0, current_band_index = 0; i < number; ++i) {
      auto promotion_node = node.child(i).child(0);
      if (!IsContainBandNode(promotion_node)) {
        continue;
      }

      remain_memory_ = akg::common::SHARED_MEMORY_SIZE;
      mark_names_.clear();
      band_index_ = current_band_index;
      node = HoistSharedMemoryOnMark(promotion_node);
      node = node.parent().parent();
      ++current_band_index;
    }
  }

  return node.get_schedule();
}

void SharedMemoryManager::SetPromotedMarkNames() {
  if (scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    // reduce operator
    mark_names_.emplace(PROMOTE_GLOBAL_TO_SHARED);
  } else if (scop_info_.user_config_.GetEnableMatmul()) {
    // matmul operator
    auto tensor_c_name = GetMatmulTensorsName(scop_info_)[MATRIX_C];
    if (std::find(configed_tensors_.begin(), configed_tensors_.end(), tensor_c_name) != configed_tensors_.end()) {
      mark_names_.emplace(PROMOTE_GLOBAL_TO_SHARED_C);
    }
    mark_names_.emplace(PROMOTE_GLOBAL_TO_SHARED_AB);
  } else {
    mark_names_.emplace(PROMOTE_GLOBAL_TO_SHARED);
  }
}

void SharedMemoryManager::CreateClusterForOperator(const isl::schedule_node &node) {
  SharedCreateCluster create_cluster(scop_info_, band_index_);
  if (scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    // reduce operator
    create_cluster.CreateClusterListForReduce(node, mark_names_);
  } else if (scop_info_.user_config_.GetEnableMatmul()) {
    // matmul operator
    remain_memory_ = akg::common::ADVANCED_SHARED_MEMORY_SIZE;
    create_cluster.CreateClusterListForGemm(node, mark_names_);
  } else if (current_outer_bn_->template_type == Template::PARTIAL_ELEM) {
    create_cluster.CreateClusterListForPartialElementWise(node, mark_names_);
  } else {
    create_cluster.CreateClusterListForElementWise(node, mark_names_);
  }
}

isl::schedule_node SharedMemoryManager::InsertMarkerForRegisterPromotion(const isl::schedule_node &orig_node) {
  isl::schedule_node hoist_register_node = orig_node;

  if (scop_info_.user_config_.GetEnableMatmul()) {
    if (mark_names_.find(PROMOTE_GLOBAL_TO_SHARED_C) != mark_names_.end()) {
      hoist_register_node = orig_node.child(0).insert_mark(PROMOTE_SHARED_TO_REGISTER_C);
    }
    hoist_register_node = InsertMarkerForPromotedNode(hoist_register_node, WRITE_ID_NAME, PROMOTE_SHARED_TO_GLOBAL);
    return ReplaceMarker(hoist_register_node, PROMOTE_GLOBAL_TO_SHARED_AB, SHARED_MEM_PROMOTED_COMPLETE);
  }

  size_t start_depth = orig_node.get_tree_depth();

  orig_node.foreach_descendant_top_down([&hoist_register_node, this](const isl::schedule_node &node) -> bool {
    if (node.isa<isl::schedule_node_sequence>()) {
      auto sequence_node = node.as<isl::schedule_node_sequence>();
      if (sequence_node.parent().isa<isl::schedule_node_extension>()) {
        hoist_register_node = sequence_node.parent().insert_mark(PROMOTE_GLOBAL_TO_REGISTER);
        return false;
      }
    }

    if (node.isa<isl::schedule_node_mark>()) {
      auto mark_node = node.as<isl::schedule_node_mark>();
      if (mark_node.get_id().get_name() == THREAD_MARKER) {
        hoist_register_node = mark_node.insert_mark(PROMOTE_GLOBAL_TO_REGISTER);
        return false;
      }
    }
    return true;
  });

  hoist_register_node = hoist_register_node.ancestor(hoist_register_node.get_tree_depth() - start_depth);
  return hoist_register_node;
}

isl::schedule_node SharedMemoryManager::MapCopiesToThreads(const isl::schedule_node &orig_node, bool unroll) {
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

    OperatorMappingStrategy others_op(scop_info_, mapping_cfg, band_index_, true, true);
    others_op.SetRequiredMappingCfg(band_node);
    // Map band under thread_root from inner dim to outer dim.
    band_node = others_op.MapDimToThreadsBlocks(band_node);
    if (scop_info_.analysis_result_.GetUseGpuReduceLib()) {
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

  return orig_node.map_descendant_bottom_up(CollectReadWriteFilter);
}

MappingCfg *SharedMemoryManager::GetCurrentConfig(isl::schedule_node &node) {
  std::string id_name = GetPromotionTensorName(node, scop_info_.analysis_result_.buffer_def_infos_);
  if (id_name.empty()) {
    return nullptr;
  }

  bool enable_vectorization = true;
  auto vector_length = scop_info_.analysis_result_.GetVectorizedLength();
  if (vector_length == 0) {
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
    vectorization_loop = vector_length / shares_tensor_bits_map[id_name];

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
    node = node.insert_mark(FOR_VECTORIZED).parent();
  }

  auto replace_cfg_map = scop_info_.user_config_.GetReplaceConfig();
  id_name = PROMOTE + id_name;
  if (replace_cfg_map.count(id_name) == 0) {
    auto thread_cfg = scop_info_.user_config_.GetThreadConfig();
    CHECK(thread_cfg != nullptr) << "thread config is null";
    int total_cfg = 1;
    for (size_t i = 0; i < thread_cfg->bound; ++i) {
      total_cfg *= thread_cfg->GetAt(i).second;
    }
    OperatorMappingStrategy others_op(scop_info_, thread_cfg, band_index_, true, true);
    std::string new_cfg = others_op.SetOneConfigForMulAxis(node, total_cfg);

    if (new_cfg.empty()) {
      return nullptr;
    }
    scop_info_.user_config_.RecordReplaceConfig(id_name, new_cfg, MappingType::REPLACE_THREADS, false);
  }
  auto mapping_cfg = scop_info_.user_config_.GetReplaceConfig()[id_name];

  return mapping_cfg;
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

  if (scop_info_.user_config_.GetEnableMatmul() && tensor_id.get_name() == GetMatmulTensorsName(scop_info_)[MATRIX_C]) {
    sizes.back() += 8;
  }

  if (bank_conflict_ || current_outer_bn_->template_type == Template::TRANSPOSE_OP) {
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
  if (scop_info_.analysis_result_.GetVectorizedLength()) {
    scop_info_.analysis_result_.RecordSharedTensorBitsMap(tensor_id.get_name(),
                                                          scop_info_.GetDtypeOf(tensor_id).bits());
  }

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(node, sizes);
}

isl::schedule_node SharedMemoryManager::HoistClusters(const isl::schedule_node &node) {
  auto res_node = node;
  for (size_t index = 0; index < scop_info_.analysis_result_.buffer_def_infos_.size(); index++) {
    BufferDefInfo &buffer_info = scop_info_.analysis_result_.buffer_def_infos_[index];
    auto fp_cluster = buffer_info.GetFootPrintClusterGPU(node);
    if ((fp_cluster == nullptr || !fp_cluster->foot_print_.box.is_valid())) {
      continue;
    }

    if (!node.has_children() || !node.child(0).isa<isl::schedule_node_mark>() ||
        buffer_info.mark_tag != node.child(0).as<isl::schedule_node_mark>().get_id().get_name()) {
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

    if (memory_requirement >= remain_memory_) {
      continue;
    }
    GatherBufferFootprintDefInfo(res_node, buffer_info);
    auto dst_id = buffer_info.dst_tensor_id;
    res_node = HoistToBlockThreadMemory(res_node, GpuMemType::SHARED, id, dst_id, *(fp_cluster), true);
    remain_memory_ -= memory_requirement;

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

isl::schedule_node SharedMemoryManager::HoistToBlockThreadMemory(isl::schedule_node &tree, GpuMemType type,
                                                                 const isl::id &tensor_id, const isl::id &dst_tensor_id,
                                                                 TensorFootprintCluster &cluster,
                                                                 bool force_last_extension_odd) {
  auto out_schedule = LocalSchedule(tree);
  auto active_domains = CollectDomain(tree);
  auto sizes = cluster.GetFixedBoxSizes();
  if (force_last_extension_odd) {
    OptimizeSharedDimension(sizes, scop_info_.GetDtypeOf(dst_tensor_id));
  }
  auto res_node = PlaceOuterDataCopyBelow(scop_info_, tree, cluster, tensor_id, dst_tensor_id, out_schedule,
                                          schedule_.get_domain().get_space());
  return res_node;
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
  const int64_t even_check = 2;
  if (sizes.back() % even_check != 0) {
    return;
  }
  const int64_t bank_conflict_size = 32;
  if (bank_conflict_ && sizes.back() < bank_conflict_size) {
    sizes.back() = bank_conflict_size + 1;
  } else {
    size_t pad = 1;
    if (current_outer_bn_->template_type == Template::TRANSPOSE_OP) {
      const int64_t pad_size = 4;
      pad = std::max<size_t>(1, pad_size / type.bytes());
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
