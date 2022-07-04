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
#ifndef POLY_PASS_H_
#define POLY_PASS_H_

#include <set>
#include <ostream>
#include "poly/isl.h"
#include "poly/scop_info.h"
#include "poly/pass_info.h"

#ifdef AKG_USE_MLS
#include "poly/mls.h"
#endif

namespace akg {
namespace ir {
namespace poly {

constexpr auto MAX_STRIDE = 65535;

class SchedulePass {
 public:
  virtual ~SchedulePass() {}
  virtual isl::schedule Run(isl::schedule sch) = 0;

  std::string GetPassName() { return pass_name_; }
  std::string pass_name_;

  std::set<std::string> disabled_passes_;
};

bool LoadScheduleTreeFromFile(const std::string &filename, isl::schedule &schedule);

isl::union_map DependenceAnalysis(const isl::union_map &sources, const isl::union_map &targets,
                                  const isl::union_map &kills, const isl::union_map &sch);
isl::union_map ComputeAllDependences(const isl::schedule &schedule, const isl::union_map &reads_um,
                                     const isl::union_map &writes_um);
isl::union_map ComputeRAW(const isl::schedule &schedule, const isl::union_map &reads_um,
                          const isl::union_map &writes_um);
isl::schedule_node GetOuterBand(const isl::schedule_node &root);
bool IsSequenceOrSet(const isl::schedule_node &node);

/*
 * Compute copyin for each filter node, by intersecting the domains of
 * reads and writes of the entire scop.
 */
isl::union_map ComputeFilterCopyin(const isl::schedule_node &node, const isl::union_map &ori_reads,
                                   const isl::union_map &ori_writes, const isl::schedule ori_schedule);

bool CompareSchTreeWithString(const std::string &compare_sch, const isl::schedule &sch);

isl::schedule_constraints MakeScheduleConstraints(const isl::schedule &schedule, PassInfo &pass_info);

isl::union_map RemoveReduceOpSelfDependence(ScopInfo &scop_info, PassInfo &pass_info);

isl::union_map RemoveSelfDependence(PassInfo &pass_info, std::map<std::string, std::string> tensor_name_map = {});

/*
 * Compute copyin for each filter and return the union of such copyins.
 * In particular, return an empty result when the outermost band node
 * is not a sequence/set node.
 *
 * "result" is the union of "copyin" from each filter node, which in
 * turn is computed by ComputeFilterCopyin.
 */
isl::union_map ComputeFakeCopyin(const isl::schedule &schedule, const isl::union_map &fake_copyin,
                                 const isl::union_map &ori_reads, const isl::union_map &ori_writes);

/*
 * Insert a context node beyond to determine bound block and thread sizes for Gpu.
 */
isl::schedule InsertContextNode(const isl::schedule &sch, ScopInfo &scop_info);

/*
 * Get the number of axis whose coincidence is 1 in the current band node.
 */
size_t CountConsecutiveCoincident(const isl::schedule_node &node);

/*
 * Tile a node band based on given tile sizes.
 */
isl::schedule_node TileBand(isl::schedule_node node, const isl::multi_val &sizes);

/*
 * Obtain the information needed during the data promotion phase.
 */
std::string GetPromotionTensorName(const isl::schedule_node &node, const std::vector<BufferDefInfo> &buffer_def_infos);

bool IsReadOrWriteTensor(const isl::schedule_node &node, const std::string &read_name, const std::string &write_name);

isl::schedule_node GetCanMappingNode(const isl::schedule_node &node);

#ifdef AKG_USE_MLS
/// \brief Unwrap and remove extra refs from an isl::union_map
/// \param[in] umap isl::union_map to sanitize
/// \return Sanitized isl::union_map
isl::union_map UnwrappedAccesses(const isl::union_map &umap);

/// \brief Determine whether the MLSched scheduler should be used
/// \param[in] scop_info ScopInfo to maybe inspect
/// \return A boolean value that indicates whether MLSched should be used
/// \retval true if MLSched should be used
/// \retval false otherwise
bool MLSchedShouldBeUsed(akg::ir::poly::ScopInfo &scop_info);

/// \brief Initialize runtime options for MLSched
/// \param[in] scop_info ScopInfo to maybe inspect
/// \param[in] pass_info PassInfo to maybe inspect
/// \result Options for MLSched
///
/// The method initializes and returns runtime options for the MLSched scheduler.
/// The options may be decided arbitrarily, from the environment or from \a pass_info and \a scop_info.
mls::bin::Options MLSchedOptionsInit(const akg::ir::poly::PassInfo &pass_info,
                                     const akg::ir::poly::ScopInfo &scop_info);

/// \brief Extract the directives informations from the information coming from AKG scop
/// \result return an hint object that can be used for MLSched scheduler.
mls::bin::Hints ExtractDirectivesFromAKG(ScopInfo &scop_info);
#endif

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_PASS_H_
