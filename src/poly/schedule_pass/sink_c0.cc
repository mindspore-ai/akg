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

#include "sink_c0.h"

namespace akg {
namespace ir {
namespace poly {

bool SinkC0::FindC0Schedule(const isl::pw_aff_list &paList) {
  for (unsigned int upaIdx = 0; upaIdx < paList.size(); ++upaIdx) {
    isl::pw_aff pa = paList.get_at(upaIdx);
    int64_t inDimSize = isl_pw_aff_dim(pa.get(), isl_dim_in);
    CHECK_NE(inDimSize, -1);
    const char *lastInDim = isl_pw_aff_get_dim_name(pa.get(), isl_dim_in, inDimSize - 1);
    if (lastInDim == nullptr) {
      continue;
    }
    std::string lastAxis = lastInDim;
    // pw_aff { S_4[n, c1, kh, oh, c0] -> [(n)] }
    // to do use isl api to mark schedule axis
    std::string pwAffStr = pa.to_str();
    std::size_t arrowPos = pwAffStr.find("->");
    if (arrowPos == std::string::npos) {
      continue;
    }
    std::string rang = pwAffStr.substr(arrowPos + 2, pwAffStr.size() - (arrowPos + 2));
    std::size_t leftBracket = rang.find("(");
    std::size_t rightBracket = rang.find(")");
    if ((leftBracket == std::string::npos) || (rightBracket == std::string::npos) ||
        (rightBracket <= leftBracket + 1)) {
      continue;
    }
    std::string scheduleAxis = rang.substr(leftBracket + 1, rightBracket - leftBracket - 1);
    if (lastAxis == scheduleAxis) {
      // lastIdxSchedule[i] = true;
      // findC0Schedule = true;
      // break;
      return true;
    }
  }
  return false;
}

void SinkC0::ExchangeCoincident(std::vector<int> &coincident, const isl::schedule_node &node,
                                const std::unordered_map<int, bool> lastIdxSchedule, const int &n) {
  // save coincident value for this band
  std::vector<int> coincidentOld;
  for (int i = 0; i < n; ++i) {
    coincidentOld.push_back(node.as<isl::schedule_node_band>().member_get_coincident(i));
  }

  // exchange last axis coincident to last position
  for (int i = 0; i < n; ++i) {
    if (lastIdxSchedule.count(i) > 0) {
      continue;
    }
    coincident.push_back(coincidentOld[i]);
  }

  for (auto item : lastIdxSchedule) {
    CHECK_GE(item.first, 0) << "index of coincident can not be negative: " << item.first;
    coincident.push_back(coincidentOld[item.first]);
  }
}

/* *****************************************************
 * Initialization part:
 * get partial_schedule info and union_pw_aff_list from band node
 * partial_schedule is a multi_union_pw_aff as follows:
 * [
    { S_4[n, c1, kh, oh, c0] -> [(n)]; S_3[n, c1, oh, ow, c0] -> [(n)]; S_5[n, c1, kh, oh, ow, c0] -> [(n)]; S_6[n,
c1, kh, kw, oh, ow, c0] -> [(n)] }, { S_4[n, c1, kh, oh, c0] -> [(c1)]; S_3[n, c1, oh, ow, c0] -> [(c1)]; S_5[n, c1, kh,
oh, ow, c0] -> [(c1)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(c1)] }, { S_4[n, c1, kh, oh, c0] -> [(oh)]; S_3[n, c1, oh,
ow, c0] -> [(oh)]; S_5[n, c1, kh, oh, ow, c0] -> [(oh)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(oh)] }, { S_4[n, c1, kh,
oh, c0] -> [(0)]; S_3[n, c1, oh, ow, c0] -> [(ow)]; S_5[n, c1, kh, oh, ow, c0] -> [(1 + ow)]; S_6[n, c1, kh, kw, oh, ow,
c0] -> [(ow)] }, { S_4[n, c1, kh, oh, c0] -> [(c0)]; S_3[n, c1, oh, ow, c0] -> [(c0)]; S_5[n, c1, kh, oh, ow, c0] ->
[(c0)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(c0)] }, { S_4[n, c1, kh, oh, c0] -> [(kh)]; S_3[n, c1, oh, ow, c0] -> [(0)];
S_5[n, c1, kh, oh, ow, c0] -> [(kh)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(kh)] }, { S_4[n, c1, kh, oh, c0] -> [(0)];
S_3[n, c1, oh, ow, c0] -> [(0)]; S_5[n, c1, kh, oh, ow, c0] -> [(0)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(-kw)] }
   ]
 * Is union_pw_aff_list(upal) the other form of multi_union_pw_aff ? and it can not print in LOG(INFO)
 * but we need it during update, at least we make a new multi_union_pw_aff from union_pw_aff_list
 * and add it to the band node, shown in the following pseudo-code
 * isl::union_pw_aff_list upal = isl::union_pw_aff_list();
 * ... ...
 * update strategy of upal ...
 * ... ...
 * isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff(partial_schedule.get_space(), upal);
 * node = node.del();
 * node = node.insert_partial_schedule(mupa);
 *
 * The update strategy of SinkC0 is moving the schedule of axis of C0 with every statement
 * to the end of the multi_union_pw_aff, the purpose result is shown in the following:
 *
[
{ S_4[n, c1, kh, oh, c0] -> [(n)]; S_3[n, c1, oh, ow, c0] -> [(n)]; S_5[n, c1, kh, oh, ow, c0] -> [(n)]; S_6[n, c1, kh,
kw, oh, ow, c0] -> [(n)] }, { S_4[n, c1, kh, oh, c0] -> [(c1)]; S_3[n, c1, oh, ow, c0] -> [(c1)]; S_5[n, c1, kh, oh, ow,
c0] -> [(c1)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(c1)] }, { S_4[n, c1, kh, oh, c0] -> [(oh)]; S_3[n, c1, oh, ow, c0] ->
[(oh)]; S_5[n, c1, kh, oh, ow, c0] -> [(oh)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(oh)] }, { S_4[n, c1, kh, oh, c0] ->
[(0)]; S_3[n, c1, oh, ow, c0] -> [(ow)]; S_5[n, c1, kh, oh, ow, c0] -> [(1 + ow)]; S_6[n, c1, kh, kw, oh, ow, c0] ->
[(ow)] }, del { S_4[n, c1, kh, oh, c0] -> [(c0)]; S_3[n, c1, oh, ow, c0] -> [(c0)]; S_5[n, c1, kh, oh, ow, c0] ->
[(c0)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(c0)] }, |  { S_4[n, c1, kh, oh, c0] -> [(kh)]; S_3[n, c1, oh, ow, c0] ->
[(0)]; S_5[n, c1, kh, oh, ow, c0] -> [(kh)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(kh)] }, v  { S_4[n, c1, kh, oh, c0] ->
[(0)]; S_3[n, c1, oh, ow, c0] -> [(0)]; S_5[n, c1, kh, oh, ow, c0] -> [(0)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(-kw)] }
add { S_4[n, c1, kh, oh, c0] -> [(c0)]; S_3[n, c1, oh, ow, c0] -> [(c0)]; S_5[n, c1, kh, oh, ow, c0] -> [(c0)]; S_6[n,
c1, kh, kw, oh, ow, c0] -> [(c0)] },
]
 * This strategy is designed for Davinci architecture, for its five dimension data format.
 * We suppose two steps to achieve this strategy:
 * 1. find the last axis C0 schedule in the multi_union_pw_aff
 * 2. if find this schedule, move it to the end of the multi_union_pw_aff
 * 3. add the updated multi_union_pw_aff to the band node
 * *****************************************************/
isl::schedule_node SinkC0::SinkC0Schedule(isl::schedule_node &node) {
  if (!node.isa<isl::schedule_node_band>()) {
    return node;
  }
  auto schedule = node.as<isl::schedule_node_band>().get_partial_schedule();
  isl::union_pw_aff_list upal = isl::union_pw_aff_list();
  std::unordered_map<int, bool> lastIdxSchedule;

  // make new union pw aff list
  for (unsigned int i = 0; i < schedule.size(); ++i) {
    isl::union_pw_aff upa = schedule.get_union_pw_aff(i);
    isl::pw_aff_list paList = upa.get_pw_aff_list();
    bool findC0Schedule = FindC0Schedule(paList);
    if (findC0Schedule) {
      lastIdxSchedule[i] = true;
      continue;
    }
    if (upal.is_null()) {
      upal = isl::union_pw_aff_list(upa);
    } else {
      upal = upal.add(upa);
    }
  }

  // save permutable value for this band
  int permutable = node.as<isl::schedule_node_band>().get_permutable();
  if (!lastIdxSchedule.empty() && permutable == 1) {
    for (auto idx : lastIdxSchedule) {
      isl::union_pw_aff upa = schedule.get_union_pw_aff(idx.first);
      if (upal.is_null()) {
        upal = isl::union_pw_aff_list(upa);
      } else {
        upal = upal.add(upa);
      }
    }
  } else {
    return node;
  }

  std::vector<int> coincident;
  int n = node.as<isl::schedule_node_band>().n_member();
  ExchangeCoincident(coincident, node, lastIdxSchedule, n);

  // make multi_union_pw_aff
  isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff(schedule.get_space(), upal);

  // delete old node
  node = node.del();

  // insert new node
  node = node.insert_partial_schedule(mupa);
  node = node.as<isl::schedule_node_band>().set_permutable(permutable);
  for (int i = 0; i < n; ++i) {
    node = node.as<isl::schedule_node_band>().member_set_coincident(i, coincident[i]);
  }
  return node;
}

isl::schedule SinkC0::Run(isl::schedule sch) {
  auto fn = [&, this](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_band>()) {
      node = SinkC0Schedule(node);
    }
    return node;
  };

  return sch.get_root().map_descendant_bottom_up(fn).get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
