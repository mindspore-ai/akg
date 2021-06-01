/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef POLY_OPRATOR_MAPPING_STRATEGY_H_
#define POLY_OPRATOR_MAPPING_STRATEGY_H_

#include "poly/schedule_pass.h"
#include "poly/reduce_manager.h"

namespace akg {
namespace ir {
namespace poly {

class OperatorMappingStrategy {
 public:
  explicit OperatorMappingStrategy(PassInfo &pass_info, ScopInfo &scop_info)
      : pass_info_(pass_info), scop_info_(scop_info) {}
  ~OperatorMappingStrategy() {}

  size_t GetFinalMappingThreadNumber(isl::schedule_node &node, const size_t thread_cfg_bound,
                                     const size_t n_thread_map);
  virtual size_t MapThreadHelper(isl::schedule_node &thread_root);
  virtual isl::schedule_node MapBlockHelper(const isl::schedule_node &orig_node, MappingCfg *block_cfg,
                                            size_t n_block_map, bool check_extent,
                                            std::unordered_map<size_t, size_t> map_idx_shift = {});

 protected:
  PassInfo &pass_info_;
  ScopInfo &scop_info_;
};

class ReduceMappingStrategy : public OperatorMappingStrategy {
 public:
  explicit ReduceMappingStrategy(PassInfo &pass_info, ScopInfo &scop_info)
      : OperatorMappingStrategy(pass_info, scop_info) {}
  ~ReduceMappingStrategy() {}

  size_t MapThreadHelper(isl::schedule_node &thread_root);

  bool NeedAtomicAdd(const isl::schedule_node_band &band, size_t n_block_map);
  void MarkAtomicAddTensor(const isl::schedule_node_band &band);
  size_t GetReduceId() const;
  isl::schedule DetectAndMarkReduce(const isl::schedule &sch);
  isl::schedule InsertReduceMarker(const isl::schedule &sch);
  isl::schedule_node InsertReduceExtension(const isl::schedule_node &node);
};

class BatchMatmulMappingStrategy : public OperatorMappingStrategy {
 public:
  explicit BatchMatmulMappingStrategy(PassInfo &pass_info, ScopInfo &scop_info)
      : OperatorMappingStrategy(pass_info, scop_info) {}
  ~BatchMatmulMappingStrategy() {}

  size_t MapThreadHelper(isl::schedule_node &thread_root);
};

class ConvMappingStrategy : public OperatorMappingStrategy {
 public:
  explicit ConvMappingStrategy(PassInfo &pass_info, ScopInfo &scop_info)
      : OperatorMappingStrategy(pass_info, scop_info) {}
  ~ConvMappingStrategy() {}

  isl::schedule_node ResetConvBlockMappingConfig(const isl::schedule_node &orig_node, MappingCfg *block_cfg,
                                                 const bool check_extent);
  isl::schedule MoveKernelHWBand(isl::schedule sch);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_OPRATOR_MAPPING_STRATEGY_H_