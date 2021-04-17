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
#ifndef POLY_MARK_FUSE_OP_H_
#define POLY_MARK_FUSE_OP_H_

#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {

/* *********************************************************************
 * This class is used to add "fused_vector" mark in the schedule tree
 * for the usage of convolution and matmul operator.
 * General condition, it add the "fused_vector" mark ahead of "realize_BUFC0" as the
 * direct parent of of "realize_BUFC0".
 *- filter: "{ S_2[n_i1, m_i1, m_i0, n_i0] }"
             child:
               schedule: "[{ S_2[n_i1, m_i1, m_i0, n_i0] -> [(0)] },"
                          "{ S_2[n_i1, m_i1, m_i0, n_i0] -> [(0)] },"
                          "{ S_2[n_i1, m_i1, m_i0, n_i0] -> [(0)] },"
                          "{ S_2[n_i1, m_i1, m_i0, n_i0] -> [(0)] },"
                          "{ S_2[n_i1, m_i1, m_i0, n_i0] -> [(0)] }]"
               permutable: 1
               coincident: [ 1, 1, 1, 1, 0 ]
               options: "{ isolate[[i0, i1, 0, 0, i4] -> [0, 0, 0, 0, 0]] : 0 <= i0 <= 767 and 0 <= i1 <= 47 and 0 <= i4
 <= 255 }" child: mark: "fuse_vector" child: mark: "realize_UBL0" child: sequence:
 * Padding condition for matmul, it add the "fused_vector" mark between the "filter" node and band node
 * ahead of "realize_BUFC0" mark node in the schedule tree.
 *- filter: "{ S_2[i, j, k, i_0];"
             "S_3[j, k, i_0] }"
   child:
     mark: "fuse_vector"
     child:
       schedule: "[{ S_2[i, j, k, i_0] -> [(0)];"
                    "S_3[j, k, i_0] -> [(0)] },"
                  "{ S_2[i, j, k, i_0] -> [(0)];"
                    "S_3[j, k, i_0] -> [(0)] },"
                  "{ S_2[i, j, k, i_0] -> [(0)];"
                    "S_3[j, k, i_0] -> [(0)] },"
                  "{ S_2[i, j, k, i_0] -> [(0)];"
                    "S_3[j, k, i_0] -> [(0)] },"
                  "{ S_2[i, j, k, i_0] -> [(0)];"
                    "S_3[j, k, i_0] -> [(0)] }]"
       permutable: 1
       coincident: [ 1, 1, 1, 1, 0 ]
       options: "{ isolate[[i0, i1, 0, 0, i4] -> [0, 0, 0, 0, 0]] : 0 <= i0 <= 1907 and 0 <= i1 <= 113 and 0 <= i4 <= 63
 }" child: mark: "realize_UBL0" child: sequence:
           - filter: "{ S_2[i, j, k, i_0] }"
 * *********************************************************************/
class MarkFuseOp : public SchedulePass {
 public:
  MarkFuseOp(ScopInfo &scop_info) : scop_info_(scop_info) { pass_name_ = __FUNCTION__; };
  ~MarkFuseOp(){};

  virtual isl::schedule Run(isl::schedule schedule_mark) final;

 private:
  bool IsMatmulPadding();
  isl::schedule_node MakeMatmulPaddingFuseOp(const isl::schedule_node &node);

  ScopInfo &scop_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_MARK_FUSE_OP_H_
