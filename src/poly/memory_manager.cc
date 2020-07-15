/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "scop.h"
#include "poly/dma_inject.h"
#include "scop_builder.h"

namespace akg {
namespace ir {
namespace poly {
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

void GetVisitedStmts(const isl::schedule_node &root) {
  int n = root.n_children();
  if (n <= 0) return;

  isl::schedule_node node;
  if (root.isa<isl::schedule_node_sequence>()) {
    isl::union_set visited_stmts;
    for (int i = 0; i < n; ++i) {
      node = root.child(i);
      auto filter_node = node.as<isl::schedule_node_filter>();
      CHECK(filter_node) << "expected children of sequence to be filters";
      auto filter = filter_node.get_filter().universe();
      if (visited_stmts.get()) {
        CHECK(visited_stmts.intersect(filter).is_empty()) << "filters are expected to be disjoint as stmt level";
        visited_stmts = visited_stmts.unite(filter);
      } else {
        visited_stmts = filter;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    node = root.child(i);
    GetVisitedStmts(node);
  }
}

std::vector<isl::schedule_node> CollectMarkNode(const isl::schedule_node &tree, const std::string &mark_tag) {
  std::vector<isl::schedule_node> mark_nodes;
  tree.foreach_descendant_top_down([&mark_nodes, &mark_tag](const isl::schedule_node &node) -> bool {
    if (auto mark_node = node.as<isl::schedule_node_mark>()) {
      // ignore nested mark nodes
      if (mark_node.get_id().get_name() == mark_tag) {
        mark_nodes.push_back(node);
        return false;
      }
    }
    return true;
  });
  return mark_nodes;
}

const BufferDefInfo &Scop::GetBufferDefInfo(const isl::id &tensor_id) const {
  for (const auto &idx : BufferDefInfos()) {
    if (idx.dst_tensor_id.get_name() == tensor_id.get_name()) {
      return idx;
    }
  }
  LOG(FATAL) << "Hoist footprint of tensor " << tensor_id << " has no buffer definition";
  return place_holder_;
}

void Scop::RecordAllTensorBufferFootprintToExtension() {
  GetVisitedStmts(schedule_.get_root());
  for (size_t index = 0; index < buffer_def_infos_.size(); index++) {
    if (buffer_def_infos_[index].find_buffer) continue;
    std::string mark_tag = buffer_def_infos_[index].mark_tag;
    if (buffer_def_infos_[index].IsIm2col()) {
      isl::id nextTensorId = buffer_def_infos_[index].NextTensorDstId();
      mark_tag = GetBufferDefInfo(nextTensorId).mark_tag;
    }
    this->schedule_ = HoistBufferFootprintAtMarkNode(schedule_.get_root(), mark_tag, index);
  }
  CHECK_EQ(buffer_footprint_queue_.size(), 0);
}

isl::schedule_node MapDescendantTopDown(isl::schedule_node node,
                                        const std::function<isl::schedule_node(isl::schedule_node)> &fn) {
  unsigned int depth_ = node.get_tree_depth();
  do {
    do {
      node = fn(node);
    } while (node.has_children() && (node = node.first_child()));

    while (node.get_tree_depth() > depth_ && !node.has_next_sibling()) {
      node = node.parent();
    }

    if (node.get_tree_depth() > depth_) {
      node = node.next_sibling();
    }
  } while (node.get_tree_depth() > depth_);

  return node;
}

isl::schedule Scop::HoistBufferFootprintAtMarkNode(const isl::schedule_node &root, const std::string &mark_tag,
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

isl::schedule_node Scop::HoistBufferFootprintAtMarkNode(const isl::schedule_node &tree, size_t index) {
  auto schedule = LocalSchedule(tree);

  // hoist cluster and add extension to schedule tree
  return HoistTensorClusterFootprint(tree, index, schedule);
}

void Scop::GatherBufferFootprintDefInfo(const isl::schedule_node &tree, BufferDefInfo &tensor_info) {
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

  Type type = GetDtypeOf(tensor_id);
  Tensor tensor = placeholder(shapes, type, cluster_id.get_name());
  const Buffer buffer = decl_buffer(shapes, GetDtypeOf(tensor_id), cluster_id.get_name());
  binds_.Set(tensor, buffer);

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(tree, sizes);
}

void Scop::CollectBufferFootprintDefInfo(BufferDefInfo &tensor_info, const isl::union_map &schedule_prom,
                                         const isl::schedule_node &node) {
  tensor_info.footprints_cluster = TensorFootprintCluster::HoistBufferFootprintCluster(
    schedule_prom, tensor_info.ancester_tensor_id, data_.reads, data_.copyin, data_.writes, data_.fake_copyin);
  if (tensor_info.footprints_cluster != nullptr) {
    tensor_info.footprint_cluster_map.emplace_back(std::make_pair(node, tensor_info.footprints_cluster));
    GatherBufferFootprintDefInfo(node, tensor_info);
  }
}

void Scop::HoistIm2colBufferFootprintCluster(const isl::union_map &schedule, const isl::schedule_node &node,
                                             const int index, BufferDefInfo &tensor_info) {
  im2col_fp_cluster = ConstructAffineFpCluster(*this, data_.reads, schedule.domain(), schedule, ReferenceType::Read,
                                               AffineType::AFFINE_IM2COL);
  tensor_info.footprints_cluster = ConstructAffineFpCluster(*this, data_.reads, schedule.domain(), schedule,
                                                            ReferenceType::Read, AffineType::AFFINE_FRACTAL);
  CHECK_EQ(index, 0);
  CHECK(im2col_fp_cluster != nullptr) << "im2col_fp_cluster must be not null";
  CHECK(tensor_info.footprints_cluster != nullptr) << "footprint cluster in Im2col must be defined";
  tensor_info.footprint_cluster_map.emplace_back(std::make_pair(node, tensor_info.footprints_cluster));

  if ((tensor_info.footprints_cluster->foot_print_.box.is_valid()) && (im2col_fp_cluster->foot_print_.box.is_valid())) {
    GatherBufferFootprintDefInfo(node, tensor_info);
    // this update info is used for spec gemm
    UpdateFractalIntFirstInfo(IsConvBackpropFilter(), im2col_fp_cluster->GetFixedBoxSizes(),
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
    LOG(INFO) << "im2col or fractal foot_print_ box is invalid.";

    auto it = attr_info_.find(ATTR_CONV_KERNEL_H);
    if ((it != attr_info_.end()) && (*it).second.as<IntImm>()) k_h = (*it).second.as<IntImm>()->value;
    it = attr_info_.find(ATTR_CONV_KERNEL_W);
    if ((it != attr_info_.end()) && (*it).second.as<IntImm>()) k_w = (*it).second.as<IntImm>()->value;
    it = attr_info_.find(ATTR_CONV_STRIDE_H);
    if ((it != attr_info_.end()) && (*it).second.as<IntImm>()) s_h = (*it).second.as<IntImm>()->value;
    it = attr_info_.find(ATTR_CONV_STRIDE_W);
    if ((it != attr_info_.end()) && (*it).second.as<IntImm>()) s_w = (*it).second.as<IntImm>()->value;
    it = attr_info_.find(ATTR_CONV_TILE_H);
    if ((it != attr_info_.end()) && (*it).second.as<IntImm>()) t_h = (*it).second.as<IntImm>()->value;
    it = attr_info_.find(ATTR_CONV_TILE_W);
    if ((it != attr_info_.end()) && (*it).second.as<IntImm>()) t_w = (*it).second.as<IntImm>()->value;
    it = attr_info_.find(ATTR_CONV_FEATURE_C);
    if ((it != attr_info_.end()) && (*it).second.as<IntImm>()) c_in = (*it).second.as<IntImm>()->value;

    t_ho = (t_h - k_h) / s_h + 1;
    t_wo = (t_w - k_w) / s_w + 1;

    bool replace_ci = false;
    if (!dynamic_shape_.empty()) {
      for (const auto &ds : dynamic_shape_) {
        if (auto dsn = ds.as<air::DynamicShapeNode>()) {
          if (dsn->tensor_name == "CI1") {
            t_ci = (int64_t)(dsn->poly_upper_bound - 1);
            replace_ci = true;
          }
        }
      }
    }
    if (!replace_ci) {
      t_ci = (int64_t)(c_in + 15) / 16;
    }

    std::vector<size_t> sizes;
    sizes.push_back(1);                                  // 1
    sizes.push_back((size_t)((t_ho * t_wo + 15) / 16));  // 109
    sizes.push_back((size_t)(t_ci * k_h * k_w));         // 43648
    sizes.push_back(16);                                 // 16
    sizes.push_back(16);                                 // 16
    fractal_int_info_[ATTR_CONV_GMM_M] = t_ho * t_wo;    // 1739
    fractal_int_info_[ATTR_CONV_BATCH] = (int64_t)sizes[0];
    fractal_int_info_[ATTR_CONV_TILE_M] = (int64_t)sizes[1];
    fractal_int_info_[ATTR_CONV_TILE_K] = (int64_t)sizes[2];
    fractal_int_info_[ATTR_CONV_M_INNER] = (int64_t)sizes[3];
    fractal_int_info_[ATTR_CONV_K_INNER] = (int64_t)sizes[4];
    GatherFractalDefInfo(node, tensor_info, sizes);
  }
  fractal_int_info_[ATTR_CONV_FEATURE_W] = ExtractExprFromAttrs(ATTR_CONV_FEATURE_W);
  fractal_int_info_[ATTR_CONV_PAD_LEFT] = ExtractExprFromAttrs(ATTR_CONV_PAD_LEFT);
  fractal_int_info_[ATTR_CONV_PAD_RIGHT] = ExtractExprFromAttrs(ATTR_CONV_PAD_RIGHT);
}

void Scop::MakeMultiBufferFootprint(const isl::union_map &schedule, const isl::schedule_node &node, int &index,
                                    BufferDefInfo &tensor_info) {
  if (!IsCopyinTensor(tensor_info.ancester_tensor_id.get_name())) {
    CollectBufferFootprintDefInfo(tensor_info, schedule, node);
  } else {
    if (index == 0) {
      CollectBufferFootprintDefInfo(tensor_info, schedule, node);
    } else {
      isl::id new_dst_id = tensor_info.GetIndexDstId(ctx_, tensor_info.dst_tensor_id, index);
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
                                                       isl::union_map::empty(CreateParamsSpace(ctx_))};
      CollectBufferFootprintDefInfo(new_footprint_info, schedule, node);
      buffer_def_infos_.push_back(new_footprint_info);
    }
  }
}

void Scop::UpdateSpecGemmFractalInfo(const BufferDefInfo &tensor_info) {
  if (IsConv() && IsB(tensor_info.tensor_id.get_name())) {
    CHECK(tensor_info.footprints_cluster != nullptr);
    UpdateFractalIntLastInfo(tensor_info.footprints_cluster->GetFixedBoxSizes());
    fractal_str_info_[ATTR_CONV_GMM_WEIGHT] = tensor_info.dst_tensor_id.get_name();
    CHECK_NE(tensor_info.dst_tensor_id.get_name(), "");
  } else if (IsConv() && IsA(tensor_info.tensor_id.get_name())) {
    fractal_str_info_[ATTR_CONV_GMM_FEATURE] = tensor_info.data_stream[2].first.get_name();
    CHECK_NE(tensor_info.dst_tensor_id.get_name(), "");
  } else if (IsConv() && IsC(tensor_info.tensor_id.get_name())) {
    fractal_str_info_[ATTR_CONV_GMM_RES] = tensor_info.dst_tensor_id.get_name();
    CHECK_NE(tensor_info.dst_tensor_id.get_name(), "");
  }
}

void Scop::MakeBufferFootprintCluster(BufferDefInfo &tensor_info) {
  std::vector<isl::schedule_node> nodes = CollectMarkNode(schedule_.get_root(), tensor_info.mark_tag);
  int index = 0;
  for (const auto &node : nodes) {
    isl::schedule_node tree = node.get_child(0);
    auto schedule = LocalSchedule(tree);

    // get TensorFootPrintsCluster for each tensor
    if (tensor_info.IsIm2col()) {
      HoistIm2colBufferFootprintCluster(schedule, node, index, tensor_info);
    } else {
      if (tensor_info.IsGemmDataL12L0() || tensor_info.IsGemmWeightL12L0()) {
        AddGemmTransposeFpCluster(schedule);
      }
      MakeMultiBufferFootprint(schedule, node, index, tensor_info);
      UpdateSpecGemmFractalInfo(tensor_info);
    }
    index++;
  }
}

isl::union_set CollectDomain(const isl::schedule_node &node) {
  int depth = node.get_tree_depth();
  isl::schedule_node tmp_node;
  isl::union_set domain = node.get_domain();
  for (int i = 0; i < depth; ++i) {
    tmp_node = node.ancestor(depth - i);
    if (auto filter_node = tmp_node.as<isl::schedule_node_filter>()) {
      domain = domain.intersect(filter_node.get_filter());
    }
    if (auto extension_node = tmp_node.as<isl::schedule_node_extension>()) {
      auto parent_schedule = ShortSchedule(tmp_node);
      auto extension = extension_node.get_extension();
      parent_schedule = parent_schedule.intersect_domain(domain);
      domain = domain.unite(parent_schedule.range().apply(extension));
    }
  }
  return domain;
}

std::vector<size_t> Scop::CollectBufferedFootprintsIndexes(const isl::union_set &active_domains,
                                                           const isl::id &tensor_id) const {
  std::vector<size_t> result;

  for (size_t i = 0, e = active_buffer_footprints_.size(); i < e; ++i) {
    const auto &act_fp = active_buffer_footprints_[i];
    if (act_fp.first.intersect(active_domains).is_empty()) {
      continue;
    }

    auto cluster_id = act_fp.second.cluster_id;
    for (const auto &def_iter : BufferDefInfos()) {
      if (def_iter.dst_tensor_id.get_name() == cluster_id.get_name() &&
          def_iter.tensor_id.get_name() == tensor_id.get_name()) {
        result.push_back(i);
        break;
      }
    }
  }
  return result;
}

std::vector<std::pair<isl::union_set, Scop::BufferedFootPrintInfo>> Scop::CollectBufferedFootprints(
  const isl::union_set &active_domains, const isl::id &tensor_id) const {
  std::vector<std::pair<isl::union_set, Scop::BufferedFootPrintInfo>> result;

  for (auto idx : CollectBufferedFootprintsIndexes(active_domains, tensor_id)) {
    result.emplace_back(active_buffer_footprints_[idx]);
  }
  return result;
}

std::shared_ptr<TensorFootprintCluster> Scop::GetFootPrintsCluster(const isl::id &tensor_id) {
  for (const auto &info : buffer_def_infos_) {
    if (info.tensor_id.get_name() == tensor_id.get_name()) {
      return info.footprints_cluster;
    }
  }
  return nullptr;
}

isl::schedule_node Scop::HoistTensorClusterFootprint(isl::schedule_node tree, size_t buffered_fp_idx,
                                                     const isl::union_map &schedule) {
  BufferDefInfo &tensor_info = buffer_def_infos_[buffered_fp_idx];

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

  if (is_bind_tensor && tensor_info.mem_type != MemType::UBL0_) {
    if (!(IsGemm() && tensor_info.IsCubeCL1Write())) {
      bool insert_ub_to_l1 = false;
      if (!data_.fake_copyin.is_empty()) {
        data_.fake_copyin.foreach_map([&insert_ub_to_l1, &src_tensor_id, &dst_tensor_id](const isl::map &m) -> void {
          if ((m.get_tuple_id(isl_dim_out).get_name() == src_tensor_id.get_name()) &&
              (src_tensor_id.get_name() + "_local_L1" == dst_tensor_id.get_name())) {
            insert_ub_to_l1 = true;
          }
        });
      }
      if (insert_ub_to_l1) {
        isl::id outer_tensorId = isl::id(src_tensor_id.ctx(), src_tensor_id.get_name() + "_local_UB");
        tree =
          PlaceInnerDataCopyBelow(*this, tree, *fp_cluster, *fp_cluster, src_tensor_id, dst_tensor_id, outer_tensorId);
      } else {
        tree = PlaceOuterDataCopyBelow(*this, tree, *fp_cluster, src_tensor_id, dst_tensor_id);
      }
    } else {
      buffer_footprint_queue_.push(src_tensor_id);
    }
    // If the new buffer_footprint is not a strict subset of any other parent
    auto cluster = std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster));
    active_buffer_footprints_.emplace_back(
      std::make_pair(active_domains, BufferedFootPrintInfo{cluster, schedule, dst_tensor_id}));
    tensor_info.find_buffer = true;
    return tree;
  }

  if (tensor_info.IsIm2col()) {
    isl::id cluster_id = tensor_info.NextTensorDstId();
    auto l0_fp_cluster = GetFootPrintsCluster(dst_tensor_id);
    CHECK(l0_fp_cluster != nullptr);
    tree = PlaceIm2colBelow(*this, tree, *l0_fp_cluster, *fp_cluster, cluster_id, dst_tensor_id);
    // If the new buffer_footprint is not a strict subset of any other parent
    auto cluster = std::shared_ptr<TensorFootprintCluster>(std::move(l0_fp_cluster));
    active_buffer_footprints_.emplace_back(
      std::make_pair(active_domains, BufferedFootPrintInfo{cluster, schedule, dst_tensor_id}));
    tensor_info.find_buffer = true;
    SetFindBuffer(dst_tensor_id, true);
    return tree;
  }

  if (tensor_info.IsGemmDataL12L0()) {
    if (IsGemmDataTranspose()) {
      const isl::id &trans_id = dst_tensor_id;
      const isl::id &cluster_id = dst_tensor_id;
      tree = PlaceIm2colBelow(*this, tree, *gemm_a_transpose_fp_cluster_, *fp_cluster, trans_id, cluster_id);
      active_buffer_footprints_.emplace_back(
        std::make_pair(active_domains, BufferedFootPrintInfo{gemm_a_transpose_fp_cluster_, schedule, cluster_id}));
    }
  }

  if (tensor_info.IsGemmWeightL12L0()) {
    if (IsGemmWeightTranspose()) {
      const isl::id &trans_id = dst_tensor_id;
      const isl::id &cluster_id = dst_tensor_id;
      tree = PlaceIm2colBelow(*this, tree, *gemm_b_transpose_fp_cluster_, *fp_cluster, trans_id, cluster_id);
      active_buffer_footprints_.emplace_back(
        std::make_pair(active_domains, BufferedFootPrintInfo{gemm_b_transpose_fp_cluster_, schedule, cluster_id}));
    }
  }
  auto scop_cluster = fp_cluster;
  if (IsGemm() && (tensor_info.IsGemmDataL12L0() || tensor_info.IsGemmWeightL12L0())) {
    scop_cluster = GetBufferDefInfo(tensor_info.tensor_id).footprints_cluster;
  }
  if (tensor_info.IsPreCubeTile2Write()) {
    auto info = GetBufferDefInfo(tensor_info.tensor_id);
    auto new_scop_group = info.GetFootPrintCluster(mark_node);
    if (new_scop_group != nullptr) {
      scop_cluster = new_scop_group;
    }
  }
  tree = PlaceInnerDataCopyBelow(*this, tree, *fp_cluster, *scop_cluster, src_tensor_id, dst_tensor_id, src_tensor_id);
  if (IsGemm() && !buffer_footprint_queue_.empty() &&
      buffer_footprint_queue_.front().get_name() == tensor_info.ancester_tensor_id.get_name()) {
    tree = PlaceOuterDataCopyBelow(*this, tree, *fp_cluster, tensor_info.ancester_tensor_id, src_tensor_id);
    buffer_footprint_queue_.pop();
  }

  // If the new buffer_footprint is not a strict subset of any other parent
  auto group = std::shared_ptr<TensorFootprintCluster>(std::move(fp_cluster));

  active_buffer_footprints_.emplace_back(
    std::make_pair(active_domains, BufferedFootPrintInfo{group, schedule, dst_tensor_id}));
  tensor_info.find_buffer = true;
  return tree;
}

void Scop::ReorderBufferedDefInfos() {
  if (data_.fake_copyin.is_empty()) {
    return;
  }

  std::unordered_set<std::string> tensors;
  data_.fake_copyin.foreach_map(
    [&tensors](const isl::map &m) -> void { tensors.insert(m.get_tuple_id(isl_dim_out).get_name()); });

  for (size_t index = 1; index < buffer_def_infos_.size(); index++) {
    if ((buffer_def_infos_[index].mark_tag == REALIZE_L1) &&
        (tensors.find(buffer_def_infos_[index].tensor_id.get_name()) != tensors.end())) {
      BufferDefInfo promoted_info = buffer_def_infos_[index];
      buffer_def_infos_.erase(buffer_def_infos_.begin() + static_cast<int>(index));
      buffer_def_infos_.insert(buffer_def_infos_.begin(), promoted_info);
    }
  }
}

int Scop::CountBufferDefInfo(const isl::id &tensor_id) const {
  int num = 0;
  for (const auto &tensorIter : BufferDefInfos()) {
    if (tensorIter.dst_tensor_id.get_name() == tensor_id.get_name()) {
      num++;
    }
  }
  return num;
}

void Scop::AddGemmTransposeFpCluster(const isl::union_map &schedule) {
  auto domain = schedule.domain();
  if (IsGemmDataTranspose()) {
    if (IsGemmDataTransposeBlock()) {
      gemm_a_transpose_fp_cluster_ = ConstructAffineFpCluster(*this, data_.reads, domain, schedule, ReferenceType::Read,
                                                              AffineType::AFFINE_GEMMBLOCK, AffineTensor::LEFT_TENSOR);
    } else if (IsGemmDataTransposeInnerBlock()) {
      gemm_a_transpose_fp_cluster_ =
        ConstructAffineFpCluster(*this, data_.reads, domain, schedule, ReferenceType::Read,
                                 AffineType::AFFINE_GEMMBLOCKIN, AffineTensor::LEFT_TENSOR);
    } else {
      gemm_a_transpose_fp_cluster_ = ConstructAffineFpCluster(*this, data_.reads, domain, schedule, ReferenceType::Read,
                                                              AffineType::AFFINE_GEMM, AffineTensor::LEFT_TENSOR);
    }
  }
  if (IsGemmWeightTranspose()) {
    if (IsGemmWeightTransposeBlock()) {
      gemm_b_transpose_fp_cluster_ = ConstructAffineFpCluster(*this, data_.reads, domain, schedule, ReferenceType::Read,
                                                              AffineType::AFFINE_GEMMBLOCK, AffineTensor::RIGHT_TENSOR);
    } else if (IsGemmWeightTransposeInnerBlock()) {
      gemm_b_transpose_fp_cluster_ =
        ConstructAffineFpCluster(*this, data_.reads, domain, schedule, ReferenceType::Read,
                                 AffineType::AFFINE_GEMMBLOCKIN, AffineTensor::RIGHT_TENSOR);
    } else {
      gemm_b_transpose_fp_cluster_ = ConstructAffineFpCluster(*this, data_.reads, domain, schedule, ReferenceType::Read,
                                                              AffineType::AFFINE_GEMM, AffineTensor::RIGHT_TENSOR);
    }
  }
}

void GetAffOffsetAndNumVars(const isl::aff &aff, int &offset, int &num_vars) {
  offset = aff.get_constant_val().get_num_si();

  num_vars = 0;
  int dim = isl_aff_dim(aff.get(), isl_dim_in);
  CHECK_GE(dim, 0);
  for (int j = 0; j < dim; ++j) {
    isl_val *coef = isl_aff_get_coefficient_val(aff.get(), isl_dim_in, j);
    int coef_val = isl_val_get_num_si(coef);
    static_cast<void>(isl_val_free(coef));
    if (coef_val != 0) ++num_vars;
  }
}

/*
 * Check the isl::aff is in the form of { [i0, i1, i2, i3, i4] -> [(-64 + i2)] }
 * i.e. the mapping is one variable plus a non-zero constant offset.
 */
bool IsAffVarPlusOffset(const isl::aff &aff) {
  int offset = 0, num_vars = 0;
  GetAffOffsetAndNumVars(aff, offset, num_vars);
  return offset != 0 && num_vars == 1;
}

/*
 * Check the isl::aff is in the form of { [i0, i1, i2, i3, i4] -> [(64)] }
 * i.e. the mapping is a non-zero constant.
 */
bool IsAffNonZeroConst(const isl::aff &aff) {
  int offset = 0, num_vars = 0;
  GetAffOffsetAndNumVars(aff, offset, num_vars);
  return offset != 0 && num_vars == 0;
}

static isl::pw_multi_aff ComputeNewBufferFootprint(const std::shared_ptr<TensorFootprintCluster> &fp_cluster,
                                                   const isl::pw_multi_aff &buffer_footprint) {
  if (!fp_cluster->UnWriteable()) return buffer_footprint;
  if (!fp_cluster->foot_print_.is_valid) return buffer_footprint;
  unsigned num_dims = fp_cluster->foot_print_.GetBoxDim();

  isl::pw_multi_aff new_buffer_footprint = buffer_footprint;
  for (unsigned dim = 0; dim < num_dims; ++dim) {
    isl::aff lower_bound = fp_cluster->foot_print_.GetBoxLowerBound(dim);
    isl::pw_aff dim_buf_fp = buffer_footprint.get_pw_aff(dim);
    if (dim_buf_fp.n_piece() != 1) return buffer_footprint;
    // there is only one piece, but we have to use the foreach API
    dim_buf_fp.foreach_piece([&lower_bound, &new_buffer_footprint, &dim](const isl::set &set,
                                                                         const isl::aff &aff) -> void {
      if (IsAffVarPlusOffset(lower_bound) && IsAffNonZeroConst(aff)) {
        isl::pw_aff zero = isl::pw_aff(isl::manage(isl_aff_set_constant_si(aff.copy(), 0)));
        new_buffer_footprint = isl::manage(isl_pw_multi_aff_set_pw_aff(new_buffer_footprint.copy(), dim, zero.copy()));
      }
    });
  }
  return new_buffer_footprint;
}

/*
 * Remove the constant offset from provide args, e.g. input_1_local_UB(32, 7, cc2, cc3) = input_1(...)
 * Check the footprint cluster of the hoisted var to confirm this input tensor has multiple accesses
 * from shifted tiles. This should be improved by computing the new footprint with footprint_per_access(),
 * but from isl AST we do not know the footprint ID that corresponds to the GM -> UB copy.
 */
isl::pw_multi_aff Scop::RemoveConstOffsetFromBufferFootprint(const isl::pw_multi_aff &buffer_footprint) {
  const isl::id buffer_id = buffer_footprint.get_tuple_id(isl_dim_out);
  for (const auto &act_buf : ActiveBufferFootprints()) {
    if (act_buf.second.cluster_id == buffer_id) {
      const auto &footprint_cluster = act_buf.second.cluster;
      return ComputeNewBufferFootprint(footprint_cluster, buffer_footprint);
    }
  }
  return buffer_footprint;
}

bool Scop::HasBufferDefInfo(const isl::id &tensor_id) const {
  for (const auto &idx : BufferDefInfos()) {
    if (idx.dst_tensor_id.get_name() == tensor_id.get_name()) {
      return true;
    }
  }
  return false;
}

void Scop::UpdateFractalIntFirstInfo(bool is_conv_backprop_filter, const std::vector<size_t> &im2col_fp_cluster_size,
                                     const std::vector<size_t> &fractal_fp_cluster_size) {
  if (is_conv_backprop_filter) {
    UpdateFractalIntFirstInfoConvBackpropFilter(im2col_fp_cluster_size, fractal_fp_cluster_size);
  } else {
    UpdateFractalIntFirstInfoConvForward(im2col_fp_cluster_size, fractal_fp_cluster_size);
  }
}

void Scop::UpdateFractalIntLastInfo(std::vector<size_t> filter_fp_cluster_size) {
  if (IsConvBackpropInput()) {
    CHECK_EQ(filter_fp_cluster_size.size(), 4);
    // conv_backprop_input filter: [ko, no, ni, ki]
    int64_t kh = ExtractIntFromAttrs(ATTR_CONV_KERNEL_H);
    int64_t kw = ExtractIntFromAttrs(ATTR_CONV_KERNEL_W);
    fractal_int_info_[ATTR_CONV_TILE_CO] = (int64_t)filter_fp_cluster_size[0] / (kh * kw);
    fractal_int_info_[ATTR_CONV_TILE_N] = (int64_t)filter_fp_cluster_size[0] / (kh * kw);

    fractal_int_info_[ATTR_CONV_N_INNER] = (int64_t)filter_fp_cluster_size[2];
  } else if (IsConvBackpropFilter()) {
    CHECK_EQ(filter_fp_cluster_size.size(), 5);
    // conv_backprop_filter filter: [batch, no, mo, ni, mi]
    fractal_int_info_[ATTR_CONV_TILE_M] = (int64_t)filter_fp_cluster_size[1];
    fractal_int_info_[ATTR_CONV_M_INNER] = (int64_t)filter_fp_cluster_size[3];
    fractal_int_info_[ATTR_CONV_GMM_M] = (int64_t)filter_fp_cluster_size[1] * filter_fp_cluster_size[3];
  } else {
    CHECK_EQ(filter_fp_cluster_size.size(), 4);
    // conv_forward filter: [ko, no, ni, ki]
    fractal_int_info_[ATTR_CONV_TILE_CO] = (int64_t)filter_fp_cluster_size[1];
    fractal_int_info_[ATTR_CONV_TILE_N] = (int64_t)filter_fp_cluster_size[1];
    fractal_int_info_[ATTR_CONV_N_INNER] = (int64_t)filter_fp_cluster_size[2];
  }
}

// set the findPromote to the given tensor_id in buffered_decl_infos_
// based on tensor_id_
void Scop::SetFindBuffer(const isl::id &tensor_id, bool find_buffer) {
  for (auto &info : buffer_def_infos_) {
    if (info.tensor_id.get_name() == tensor_id.get_name()) {
      info.find_buffer = find_buffer;
      return;
    }
  }
  LOG(FATAL) << "hosited tensor" << tensor_id << "has no declaration";
}

void Scop::UpdateFractalIntFirstInfoConvBackpropFilter(std::vector<size_t> im2col_fp_cluster_size,
                                                       std::vector<size_t> fractal_fp_cluster_size) {
  CHECK_EQ(fractal_fp_cluster_size.size(), 5);
  fractal_int_info_[ATTR_CONV_BATCH] = (int64_t)fractal_fp_cluster_size[0];
  fractal_int_info_[ATTR_CONV_TILE_K] = (int64_t)fractal_fp_cluster_size[1];
  fractal_int_info_[ATTR_CONV_TILE_N] = (int64_t)fractal_fp_cluster_size[2];
  fractal_int_info_[ATTR_CONV_N_INNER] = (int64_t)fractal_fp_cluster_size[3];
  fractal_int_info_[ATTR_CONV_K_INNER] = (int64_t)fractal_fp_cluster_size[4];

  fractal_int_info_[ATTR_CONV_TILE_CO] = (int64_t)fractal_fp_cluster_size[2];

  CHECK_EQ(im2col_fp_cluster_size.size(), 6);
  fractal_int_info_[ATTR_CONV_GMM_K] = (int64_t)im2col_fp_cluster_size[1];
}

void Scop::UpdateFractalIntFirstInfoConvForward(std::vector<size_t> im2col_fp_cluster_size,
                                                std::vector<size_t> fractal_fp_cluster_size) {
  CHECK_EQ(fractal_fp_cluster_size.size(), 5);
  fractal_int_info_[ATTR_CONV_BATCH] = (int64_t)fractal_fp_cluster_size[0];
  fractal_int_info_[ATTR_CONV_TILE_M] = (int64_t)fractal_fp_cluster_size[1];
  fractal_int_info_[ATTR_CONV_TILE_K] = (int64_t)fractal_fp_cluster_size[2];
  fractal_int_info_[ATTR_CONV_M_INNER] = (int64_t)fractal_fp_cluster_size[3];
  fractal_int_info_[ATTR_CONV_K_INNER] = (int64_t)fractal_fp_cluster_size[4];

  CHECK_EQ(im2col_fp_cluster_size.size(), 6);
  fractal_int_info_[ATTR_CONV_GMM_M] = (int64_t)im2col_fp_cluster_size[1];
}

isl::union_map LocalScheduleImpl(const isl::schedule_node &node, bool use_node) {
  int tree_depth = node.get_tree_depth();
  int new_tree_depth = tree_depth;
  if (use_node) ++new_tree_depth;
  isl::schedule_node tmp_node;
  isl::union_map schedule = isl::union_map::from_domain(node.get_domain());
  for (int i = 0; i < new_tree_depth; ++i) {
    tmp_node = node.ancestor(tree_depth - i);
    if (auto band_node = tmp_node.as<isl::schedule_node_band>()) {
      if (band_node.n_member() > 0) {
        schedule = schedule.flat_range_product(band_node.get_partial_schedule_union_map());
      }
    } else if (auto filter_node = tmp_node.as<isl::schedule_node_filter>()) {
      schedule = schedule.intersect_domain(filter_node.get_filter());
    } else if (auto extension_node = tmp_node.as<isl::schedule_node_extension>()) {
      schedule = schedule.unite(extension_node.get_extension().reverse().intersect_range(schedule.range()));
    }
  }
  return schedule;
}

isl::union_map ShortSchedule(const isl::schedule_node &node) { return LocalScheduleImpl(node, false); }

isl::union_map LocalSchedule(const isl::schedule_node &node) { return LocalScheduleImpl(node, true); }

void Scop::GatherFractalDefInfo(const isl::schedule_node &tree, BufferDefInfo &tensor_info,
                                std::vector<size_t> &sizes) {
  isl::id tensor_id = tensor_info.tensor_id;
  isl::id cluster_id = tensor_info.dst_tensor_id;

  Array<Expr> shapes;
  for (auto i : sizes) {
    shapes.push_back(Expr(static_cast<int>(i)));
  }

  Type type = GetDtypeOf(tensor_id);
  Tensor tensor = placeholder(shapes, type, cluster_id.get_name());
  const Buffer buffer = decl_buffer(shapes, GetDtypeOf(tensor_id), cluster_id.get_name());
  binds_.Set(tensor, buffer);

  tensor_info.sizes = sizes;
  tensor_info.tensor = tensor;
  tensor_info.data_type = type;
  tensor_info.AddSize(tree, sizes);
}

/*
 * Update sizes of a specific tensor in order to support realize shape expansion in UB -> L1 strided copy
 * param new_sizes: new shape of the tensor
 * return: found or not found
 */
bool Scop::UpdateBufferDefInfoSizes(const isl::id &tensor_id, const std::vector<size_t> &new_sizes) {
  for (auto &info : buffer_def_infos_) {
    // update the first occurrence
    if (info.dst_tensor_id == tensor_id) {
      auto old_sizes = info.sizes;
      CHECK(old_sizes.size() == new_sizes.size());
      Array<Expr> shapes;
      for (size_t dim = 0; dim < new_sizes.size(); ++dim) {
        size_t new_size = std::max(new_sizes[dim], old_sizes[dim]);
        shapes.push_back(Expr(static_cast<int>(new_size)));
      }
      Tensor tensor = placeholder(shapes, info.data_type, tensor_id.get_name());
      const Buffer buffer = decl_buffer(shapes, info.data_type, tensor_id.get_name());
      binds_.Set(tensor, buffer);

      info.sizes = new_sizes;
      info.tensor = tensor;
      return true;
    }
  }
  return false;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
