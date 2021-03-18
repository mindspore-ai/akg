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

#ifndef POLY_SYNC_MANAGER_H_
#define POLY_SYNC_MANAGER_H_

#include "isl.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace akg {
namespace ir {
namespace poly {
enum SyncLevel { EMPTY = 0, WARP, BLOCK };

constexpr auto SYNC_PREFIX = "_sync_";
constexpr auto WARP_SYNC_PREFIX = "_warpSync_";

constexpr auto SYNC_WARP = "warp";
constexpr auto SYNC_BLOCK = "block";
constexpr auto SYNC_GRID = "grid";
constexpr auto WARP_SIZE = 32;

struct SyncCandidate;

struct Synchronization {
  Synchronization(SyncLevel l) : level(l){};
  Synchronization(SyncLevel l, int p) : level(l), pos(p){};

  SyncLevel level;
  int pos{0};

  bool IsEqual(const Synchronization &sync) { return (this->pos == sync.pos && this->level == sync.level); }
};

using SyncBetween = std::pair<SyncCandidate*, Synchronization>;

struct SyncCandidate {
  SyncCandidate(int i, int len) : idx(i), length(len){};
  int idx{0};     // index of position in an isl sequence node
  int length{0};  // total length of sequence node
  std::unique_ptr<SyncCandidate> next;
  std::vector<SyncBetween> sync;
  isl::union_set domain;
  std::unordered_map<SyncCandidate *, int> num_block_sync_to;
  std::unordered_map<SyncCandidate *, int> num_warp_sync_to;

  void ForEachCandidateTopDown(const std::function<void(SyncCandidate *)> &fn) {
    fn(this);
    auto cur = this->next.get();
    while (cur && cur != this) {
      fn(cur);
      cur = cur->next.get();
    }
  }

  void InsertSyncBetween(SyncCandidate *node, Synchronization sync) {
    this->sync.emplace_back(std::make_pair(node, sync));
  }

  int GetNumOfSyncBetween(SyncCandidate *node, SyncLevel level = SyncLevel::EMPTY) {
    int count = 0;
    bool finished = false;
    this->ForEachCandidateTopDown([this, &count, &finished, node, level](SyncCandidate *cand) {
      if (finished) {
        return;
      }
      if (cand == node) {
        finished = true;
      }
      for (const auto &s : cand->sync) {
        if (s.second.level == level) {
          ++count;
        }
      }
    });
    return count;
  }

  SyncCandidate *NextNCandidate(int n) {
    auto res = this;
    n = n % this->length;
    while (n) {
      res = res->next.get();
      --n;
    }
    return res;
  }

  std::pair<SyncCandidate *, int> GetOptimalSyncPos(SyncLevel level) {
    std::pair<SyncCandidate *, int> opt = std::make_pair(nullptr, -1);
    if (level == SyncLevel::EMPTY) {
      return std::make_pair(this, 0);
    }
    auto target = level == SyncLevel::BLOCK ? this->num_block_sync_to : this->num_warp_sync_to;
    for (const auto &s : target) {
      if (s.first == this) {
        continue;
      }
      if (opt.second == -1 || s.second < opt.second || (s.second == opt.second && s.first->idx < opt.first->idx)) {
        opt = s;
      }
    }
    return opt;
  }

  void Dump() {
    this->ForEachCandidateTopDown([](SyncCandidate *node) {
      std::cout << "[No." << node->idx << "]: ";
      for (const auto &s : node->sync) {
        std::cout << "sync " << s.second.level << " -> No." << s.first->idx << "; ";
      }
      std::cout << std::endl;
      std::cout << "Block level sync count: " << std::endl;
      for (const auto &p : node->num_block_sync_to) {
        std::cout << "[No." << node->idx << "]" << " -> [No." << p.first->idx << "] : #" << p.second << " sync." << std::endl;
      }
      std::cout << "Warp level sync count: " << std::endl;
      for (const auto &p : node->num_warp_sync_to) {
        std::cout << "[No." << node->idx << "]" << " -> [No." << p.first->idx << "] : #" << p.second << " sync." << std::endl;
      }
      std::cout << "====================================================" << std::endl;
    });
  }
};

class SyncManager {
 public:
  explicit SyncManager(isl::ctx ctx) : ctx_(ctx) {}
  ~SyncManager() {}

  isl::schedule_node InsertExtensionNode(const isl::schedule_node &node, SyncLevel level, bool after);
  isl::schedule_node InsertPromotionSync(const isl::schedule_node &tree);

 private:
  isl::ctx ctx_;
  int extension_distance_from_original_pos_ = 3;

  isl::id MakeUniqueId(SyncLevel level);
  isl::id GetSyncId() const;
  isl::id GetWarpSyncId() const;

  isl::map GetExtensionSpace(const isl::schedule_node &node, SyncLevel level);
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_SYNC_MANAGER_H_
