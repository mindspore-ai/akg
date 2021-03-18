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
#include "compute_transfer_copyin.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule ComputeTransferCopyin::Run(isl::schedule sch) {
  // compute fake copyin
  auto ori_reads = scop_info_.analysis_result_.GetReads();
  auto ori_writes = scop_info_.analysis_result_.GetWrites();
  auto ori_fake_copyin = scop_info_.analysis_result_.GetFakeCopyin();
  isl::union_map fake_copyin = ComputeFakeCopyin(sch, ori_fake_copyin, ori_reads, ori_writes);
  fake_copyin = fake_copyin.subtract(scop_info_.analysis_result_.GetCopyin());
  scop_info_.analysis_result_.RecordFakeCopyin(fake_copyin);
  isl::union_map raw_writes = ori_writes.domain_factor_domain();
  isl::union_map raw_reads = ori_reads.domain_factor_domain();
  isl::union_map raw_copyin = scop_info_.analysis_result_.GetCopyin().domain_factor_domain();
  isl::union_map reads = fake_copyin.domain_factor_domain();
  isl::union_map transfer_copyin = fake_copyin;
  while (!reads.is_empty()) {
    isl::union_map writes = raw_writes.intersect_range(reads.range());
    isl::union_map dependence = DependenceAnalysis(writes, reads, writes, sch.get_map());
    isl::union_set stmt = dependence.domain().universe();
    scop_info_.analysis_result_.RecordTransferStmt(scop_info_.analysis_result_.GetTransferStmt().unite(stmt));
    reads = raw_reads.intersect_domain(stmt);

    // compute transfer copyin
    isl::union_map target_acc = raw_writes.intersect_domain(stmt);
    isl::union_map relation = target_acc.reverse().apply_range(reads);
    transfer_copyin = transfer_copyin.apply_range(relation);
    isl::union_map copyin = transfer_copyin.intersect_range(raw_copyin.range().universe());
    scop_info_.analysis_result_.RecordReads(scop_info_.analysis_result_.GetReads().unite(copyin));
    scop_info_.analysis_result_.RecordCopyin(scop_info_.analysis_result_.GetCopyin().unite(copyin));
    transfer_copyin = transfer_copyin.subtract(copyin);
    reads = reads.subtract(raw_copyin);
    reads = reads.subtract(fake_copyin.domain_factor_domain());
  }
  return sch;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
