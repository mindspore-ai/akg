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

#ifndef POLY_DMA_INJECT_H_
#define POLY_DMA_INJECT_H_

#include <isl/constraint.h>
#include <memory>
#include "poly/isl.h"
#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {
enum class ReferenceType : int16_t { Read, Write };

struct ScopedFootprint {
  size_t GetBoxDim() const { return box.get_size().size(); }
  isl::val GetBoxSizeValue(int pos) const { return box.get_size().get_val(pos); }
  isl::aff GetBoxLowerBound(int pos) const { return box.get_offset().get_aff(pos); }
  isl::val GetStrideValue(int pos) const { return stride_values.get_val(pos); }
  isl::aff GetStrideOffset(int pos) const { return stride_offsets.get_aff(pos); }

  isl::fixed_box box;
  isl::multi_val stride_values;
  isl::multi_aff stride_offsets;

  // read-and-write unmergeable memory hoists are invalid, and should use identity memory hoist for invalid dims
  bool is_valid{false};
  std::vector<int> invalid_dims;

  // read-only or write-only unmergeable memory hoists should be split to separate memory hoists
  bool should_split{false};
};

struct TensorFootprint {
  TensorFootprint(const isl::map &original_access, const isl::map &scoped_access, ReferenceType type, bool need_dma,
                  bool need_extension)
      : original_access(original_access.domain_factor_domain()),
        scoped_access(scoped_access.domain_factor_domain()),
        type(type),
        id(scoped_access.get_space().domain().unwrap().get_tuple_id(isl_dim_out)),
        need_dma(need_dma),
        need_extension(need_extension) {}

  isl::map original_access;
  isl::map scoped_access;
  ReferenceType type;
  isl::id id;

  // Whether this footprint needs dma copy
  bool need_dma;

  // whether this footprint needs an extension mark in tree
  bool need_extension;
};
class TensorFootprintCluster;
using TensorClusterInfo = std::vector<std::unique_ptr<TensorFootprintCluster>>;
using TensorCluster = std::unordered_map<isl::id, TensorClusterInfo, isl::IslIdIslHash>;

class TensorFootprintCluster {
 private:
  TensorFootprintCluster() = default;

 public:
  ~TensorFootprintCluster() = default;

  static std::unique_ptr<TensorFootprintCluster> HoistBufferFootprintCluster(
    const isl::union_map &outer_schedule, const isl::id &target_id, const isl::union_map &reads,
    const isl::union_map &copyin, const isl::union_map &writes, const isl::union_map &fake_copyin);

  bool UnWriteable() const;
  bool UnReadable() const;

  isl::map RichWriteRelations() const;
  isl::map RichReadRelations() const;
  isl::map RichAccessRelations() const { return RichWriteRelations().unite(RichReadRelations()); }

  bool WriteNeedDma() const;
  bool ReadNeedDma() const;

  bool WriteNeedExtension() const;
  bool ReadNeedExtension() const;

  isl::map ExtractSingleAccessRelation() const;

  isl::set GetSingleAccessRange() const { return ExtractSingleAccessRelation().range(); }

  isl::multi_aff ComputeBufferedFootprints() const;
  isl::multi_aff IdentityBufferFootprint() const;
  isl::multi_aff UnshiftedBufferFootprint(const isl::multi_aff &default_promotion, const isl::id &ref_id) const;
  isl::aff LowerBound(const isl::aff &offset, const isl::val &stride, const isl::aff &stride_offset) const;
  isl::aff UpperBound(const isl::val &size, const isl::aff &offset, const isl::val &stride,
                      const isl::aff &stride_offset) const;
  isl::set BufferedFootprint() const;

  std::vector<size_t> GetFixedBoxSizes() const;

  static std::unique_ptr<TensorFootprintCluster> ClusteringFootprints(
    std::unique_ptr<TensorFootprintCluster> &&cluster1, std::unique_ptr<TensorFootprintCluster> &&cluster2);

  static std::unique_ptr<TensorFootprintCluster> ComputeFootprintCluster(const isl::map &original_access,
                                                                         const isl::map &scoped_access,
                                                                         ReferenceType type, bool need_dma,
                                                                         bool need_extension = false);

  std::vector<std::unique_ptr<TensorFootprint>> tensor_foot_prints;
  ScopedFootprint foot_print_;
  isl::map footprint_map_;

 private:
  isl::multi_aff ComputeBufferedFootprints(bool with_strides, bool with_lower_bounds) const;
};

inline std::ostream &operator<<(std::ostream &os, const ScopedFootprint &scoped_fp) {
  if (!scoped_fp.box) {
    return os;
  }
  os << "{ offset: " << scoped_fp.box.get_offset() << ", size: " << scoped_fp.box.get_size() << " }\n";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const TensorFootprint &foot_print) {
  os << (foot_print.type == ReferenceType::Read ? "read" : "write")
     << "\n original_access: " << foot_print.original_access << "\n scoped_access: " << foot_print.scoped_access
     << "\n footprint_id: " << foot_print.id << "\n need_dma: " << (foot_print.need_dma ? "true" : "false")
     << "\n need_extension: " << (foot_print.need_extension ? "true" : "false");
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const TensorFootprintCluster &fp_cluster) {
  os << "\n Reference with footprint: " << fp_cluster.foot_print_ << "\n";
  for (const auto &fp : fp_cluster.tensor_foot_prints) {
    os << *fp << "\n";
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const TensorClusterInfo &cluster_info) {
  for (const auto &info : cluster_info) {
    os << *info << " ";
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const TensorCluster &cluster) {
  size_t i = 0;
  for (const auto &t : cluster) {
    os << "id: " << t.first << "; \n acc: " << t.second;
    if (++i < cluster.size()) {
      os << std::endl;
    }
  }
  return os;
}

std::vector<int> ExpandInvalidDims(const std::vector<int> &invalid_dims, const isl::multi_aff &default_footprint,
                                   int &first_invalid_domain_dim);
isl::multi_aff ComputeBufferFootprint(const isl::map &access, const ScopedFootprint &foot_print);

isl::schedule_node PlaceDataCopyBelowImpl(ScopInfo &scop_info, isl::schedule_node tree,
                                          const TensorFootprintCluster &cluster, const isl::map &buffer_footprint,
                                          const isl::id &tensor_id, const isl::set &original_elements,
                                          const isl::map &exact_reads, const isl::map &exact_writes,
                                          const isl::union_map &sch);

void PlaceDataCopyBelowImplReadWrite(ScopInfo &scop_info, isl::schedule_node &tree,
                                     const TensorFootprintCluster &cluster, const isl::map &footprint,
                                     const isl::id &tensor_id, const isl::set &original_elements,
                                     const isl::map &exact_writes, isl::map &read_extension,
                                     isl::set &buffered_footprint, const isl::id &cluster_id, isl::map &extension_map,
                                     isl::id &read_id);

void PlaceDataCopyBelowImplFakeReads(ScopInfo &scop_info, isl::schedule_node &tree,
                                     const TensorFootprintCluster &cluster, isl::map &read_extension,
                                     const isl::id &cluster_id, const isl::union_map &sch);

isl::schedule_node PlaceInnerDataCopyBelow(ScopInfo &scop_info, const isl::schedule_node &tree,
                                           const TensorFootprintCluster &cluster,
                                           const TensorFootprintCluster &outer_scope_cluster, const isl::id &tensor_id,
                                           const isl::id &cluster_id, const isl::id &outer_scope_cluster_id,
                                           const isl::union_map &sch);

isl::schedule_node PlaceOuterDataCopyBelow(ScopInfo &scop_info, const isl::schedule_node &tree,
                                           const TensorFootprintCluster &cluster, const isl::id &tensor_id,
                                           const isl::id &cluster_id, const isl::union_map &sch,
                                           const isl::space &sch_space);

isl::schedule_node PlaceIm2colBelow(ScopInfo &scop_info, const isl::schedule_node &tree,
                                    const TensorFootprintCluster &cluster,
                                    const TensorFootprintCluster &outer_scope_cluster, const isl::id &cluster_id,
                                    const isl::id &outer_scope_cluster_id);

enum AffineTensor { LEFT_TENSOR = 0, RIGHT_TENSOR, OUT_TENSOR };

class AffineBase {
 public:
  virtual ~AffineBase() = default;
  virtual isl::map ConstructAffine(isl::map original) = 0;
  virtual bool NotNeedConstruct(std::string name, ScopInfo &scop_info) = 0;
};

class GemmInnerTransposeAffine : public AffineBase {
 public:
  GemmInnerTransposeAffine() = default;
  ~GemmInnerTransposeAffine() override = default;

  isl::map ConstructAffine(isl::map original_map) final;
  void SetRightMatrix(AffineTensor v) { is_right_matrix_ = v; }

  bool NotNeedConstruct(std::string name, ScopInfo &scop_info) override {
    // right matrix filter !B tensor
    if (is_right_matrix_ == AffineTensor::RIGHT_TENSOR && !scop_info.cube_info_.IsB(name)) {
      return true;
    }
    // left matrix filter !A tensor
    if (is_right_matrix_ == AffineTensor::LEFT_TENSOR && !scop_info.cube_info_.IsA(name)) {
      return true;
    }
    return false;
  }

 private:
  AffineTensor is_right_matrix_ = AffineTensor::LEFT_TENSOR;
};

class GemmTransposeAffine : public AffineBase {
 public:
  GemmTransposeAffine() = default;
  ~GemmTransposeAffine() override = default;

  isl::map ConstructAffine(isl::map original_map) final;

  void SetRightMatrix(AffineTensor v) { is_right_matrix_ = v; }

  bool NotNeedConstruct(std::string name, ScopInfo &scop_info) override {
    // right matrix filter !B tensor
    if (is_right_matrix_ == AffineTensor::RIGHT_TENSOR && !scop_info.cube_info_.IsB(name)) {
      return true;
    }
    // left matrix filter !A tensor
    if (is_right_matrix_ == AffineTensor::LEFT_TENSOR && !scop_info.cube_info_.IsA(name)) {
      return true;
    }
    return false;
  }

 private:
  AffineTensor is_right_matrix_ = AffineTensor::LEFT_TENSOR;
};

class GemmTransposeBlockAffine : public AffineBase {
 public:
  GemmTransposeBlockAffine() = default;
  ~GemmTransposeBlockAffine() override = default;

  isl::map ConstructAffine(isl::map original_map) final;

  void SetRightMatrix(AffineTensor v) { is_right_matrix_ = v; }

  bool NotNeedConstruct(std::string name, ScopInfo &scop_info) override {
    // right matrix filter !B tensor
    if (AffineTensor::RIGHT_TENSOR == is_right_matrix_ && !scop_info.cube_info_.IsB(name)) {
      return true;
    }
    // left matrix filter !A tensor
    if (is_right_matrix_ == AffineTensor::LEFT_TENSOR && !scop_info.cube_info_.IsA(name)) {
      return true;
    }

    if (AffineTensor::OUT_TENSOR == is_right_matrix_ && !scop_info.cube_info_.IsC(name)) {
      return true;
    }

    return false;
  }

 private:
  AffineTensor is_right_matrix_ = AffineTensor::LEFT_TENSOR;
};

class Im2colAffine : public AffineBase {
 public:
  Im2colAffine() = default;
  ~Im2colAffine() override = default;

  isl::map ConstructAffine(isl::map original_map) final;

  void ConstructAffineMap(isl::map &footprint, std::vector<isl::aff> &v_aff_x, std::vector<isl::aff> &v_aff_y,
                          const isl::map &original_map, isl::local_space &ls);

  bool NotNeedConstruct(std::string name, ScopInfo &scop_info) override {
    if (!scop_info.cube_info_.IsA(name)) {
      return true;
    }
    return false;
  }

  Map<std::string, NodeRef> attrInfo_;
};

class WeightAffine : public AffineBase {
 public:
  WeightAffine() = default;
  ~WeightAffine() override = default;

  isl::map ConstructAffine(isl::map original_map) final;

  bool NotNeedConstruct(std::string name, ScopInfo &scop_info) override {
    if (!scop_info.cube_info_.IsB(name)) {
      return true;
    }
    return false;
  }

  Map<std::string, NodeRef> attrInfo_;
};

class FractalAffine : public AffineBase {
 public:
  FractalAffine() = default;
  ~FractalAffine() override = default;

  isl::map ConstructAffine(isl::map original_map) final;

  void ConstructAffineMap(isl::map &footprint, std::vector<isl::aff> &v_aff_x, std::vector<isl::aff> &v_aff_y,
                          const isl::map &original_map, isl::local_space &ls);

  bool NotNeedConstruct(std::string name, ScopInfo &scop_info) override {
    if (!scop_info.cube_info_.IsA(name)) {
      return true;
    }
    return false;
  }

  Map<std::string, NodeRef> attrInfo_;
};

enum AffineType {
  AFFINE_GEMM = 0,
  AFFINE_GEMMBLOCK,
  AFFINE_GEMMBLOCKIN,
  AFFINE_IM2COL,
  AFFINE_WEIGHTTRANS,
  AFFINE_FRACTAL
};

class AffineRefGroupConstructor {
 public:
  explicit AffineRefGroupConstructor(AffineType type) : type_(type) {}

  ~AffineRefGroupConstructor() {
    if (affine_ != nullptr) {
      delete affine_;
      affine_ = nullptr;
    }
  }

  void create();

  std::unique_ptr<TensorFootprintCluster> ConstructRefGroup(ScopInfo &scop_info, const isl::union_map &accesses,
                                                            const isl::union_set &domain,
                                                            const isl::union_map &schedule, ReferenceType type);

  std::unique_ptr<TensorFootprintCluster> ConstructAffineMapFootprintCluster(const isl::union_map &schedule,
                                                                             const isl::map &access, ReferenceType type,
                                                                             bool need_dma);

  std::unique_ptr<TensorFootprintCluster> FractalAffineMapFootprintCluster(const isl::union_map &schedule,
                                                                           const isl::map &access, ReferenceType type,
                                                                           bool need_dma);

  std::unique_ptr<TensorFootprintCluster> AffineMapFootprintCluster(const isl::union_map &schedule,
                                                                    const isl::map &access, ReferenceType type,
                                                                    bool need_dma);

  AffineBase *affine_ = nullptr;
  AffineType type_ = AffineType::AFFINE_GEMM;
};

std::unique_ptr<TensorFootprintCluster> ConstructAffineFpCluster(ScopInfo &info, const isl::union_map &accesses,
                                                                 const isl::union_set &domain,
                                                                 const isl::union_map &schedule, ReferenceType type,
                                                                 AffineType affine_type,
                                                                 AffineTensor right_matrix = AffineTensor::LEFT_TENSOR);
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_DMA_INJECT_H_
