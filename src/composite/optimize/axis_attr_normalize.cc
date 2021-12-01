/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "composite/optimize/axis_attr_normalize.h"

#include <algorithm>
#include <vector>

namespace akg {
class AxisAttrNormalizer : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    Stmt stmt = s;
    if (op->attr_key == "attrs") {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      if (attrs.find("is_csr") != attrs.end()) {
        is_csr_ = true;
      }
      if (attrs.find("axis") != attrs.end()) {
        rank_ = 0;
        is_allreduce_ = false;
        Stmt body = IRMutator::Mutate(op->body);
        if (is_allreduce_) {
          attrs.Set("axis", Array<Expr>());
          return AttrStmt::make(attrs, op->attr_key, op->value, body);
        }
        if (attrs["axis"].as<IntImm>()) {
          Array<Expr> new_axis = {make_const(Int(32), CalIndex(attrs["axis"].as<IntImm>()->value))};
          attrs.Set("axis", new_axis);
        } else {
          Array<Expr> axis = Downcast<Array<Expr>>(attrs["axis"]);
          Array<Expr> new_axis;
          if (axis.empty()) {
            for (int64_t i = 0; i < rank_; i++) {
              new_axis.push_back(make_const(Int(32), i));
            }
          } else {
            std::vector<int64_t> new_axis_val;
            for (auto val : axis) {
              auto imm = val.as<IntImm>();
              CHECK(imm);
              new_axis_val.push_back(CalIndex(imm->value));
            }
            std::sort(new_axis_val.begin(), new_axis_val.end());
            for (auto val : new_axis_val) {
              new_axis.push_back(make_const(Int(32), val));
            }
          }
          attrs.Set("axis", new_axis);
        }
        stmt = AttrStmt::make(attrs, op->attr_key, op->value, body);
      }
    }
    is_csr_ = false;
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) {
    auto prim = op->value.as<Call>();
    CHECK(prim && prim->args.size() > 0);
    auto input = prim->args[0].as<Call>();
    if (input) {
      rank_ = input->args.size();
      if (rank_ == 0) {
        rank_ = 1;
      }
      if (GetOpName(op) == "ExpandDims") {
        // the valid axis of ExpandDims is in [-1-rank, rank]
        rank_++;
      }
      if (is_csr_) {
        rank_ = kCSRRank;
      }
    }
    if (IsReduce(GetOpName(op)) && ShapeIsOne(op->args)) {
      is_allreduce_ = true;
    }
    return s;
  }

  int64_t CalIndex(int64_t x) {
    // rank == 0 means the real body is eliminated
    if (rank_ == 0) return x;

    // check axis in range [-rank, rank)
    CHECK_GE(x, -rank_);
    CHECK_LE(x, rank_ - 1);
    return x >= 0 ? x : x + rank_;
  }

 private:
  int64_t rank_{0};
  bool is_allreduce_{false};
  bool is_csr_{false};
  static constexpr int kCSRRank = 2;
};

Stmt AxisAttrNormalize::Run(const Stmt &stmt) { return AxisAttrNormalizer().Mutate(stmt); }
}  // namespace akg
