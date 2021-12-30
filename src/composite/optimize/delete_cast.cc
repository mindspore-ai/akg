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
#include "composite/optimize/pass.h"

namespace akg {
class DeleteCastMatcher : public IRVisitor {
 public:
  void Visit_(const Provide *op) {
    auto call = op->value.as<Call>();
    if (call == nullptr) {
      return IRVisitor::Visit_(op);
    }
    if (call->name == "Cast") {
      auto in = call->args[0].as<Call>();
      if (in && in->name == matmul_output_) {
        cast_matmul_output_ = true;
        cast_map_[op->func.as<PlaceholderOpNode>()->name] = in->name;
        cast_func_map_[op->func.as<PlaceholderOpNode>()->name] = in->func;
        cast_dtype_[in->name] = op->func.as<PlaceholderOpNode>()->dtype;
      }
    } else if (call->name == "BatchMatMul") {
      if (auto gemm = op->func.as<PlaceholderOpNode>()) {
        matmul_output_ = gemm->name;
      }
    }
    return IRVisitor::Visit_(op);
  }

  inline bool Matched() { return cast_matmul_output_; }

  friend class DeleteCastMutator;

 private:
  bool cast_matmul_output_{false};
  std::unordered_map<std::string, std::string> cast_map_;
  std::unordered_map<std::string, FunctionRef> cast_func_map_;
  std::unordered_map<std::string, air::DataType> cast_dtype_;
  std::string matmul_output_;
};

// delete cast for MatMul / BatchMatMul fusion op
class DeleteCastMutator : public IRMutator {
 public:
  explicit DeleteCastMutator(const DeleteCastMatcher &deletecast_matcher)
      : cast_map_(deletecast_matcher.cast_map_),
        cast_func_map_(deletecast_matcher.cast_func_map_),
        cast_dtype_(deletecast_matcher.cast_dtype_),
        matmul_output_(deletecast_matcher.matmul_output_) {}
  ~DeleteCastMutator() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
    if (attrs.find("is_backed_cast") != attrs.end()) {
      for (auto &val : attrs) {
        std::string key = val.first;
        auto pos = key.find("_format");
        if (pos != std::string::npos) {
          std::string src_tensor = key.substr(0, pos);
          if (src_tensor == matmul_output_) {
            return IRMutator::Mutate(op->body);
          }
        }
      }
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<AttrStmt>();
    if (op == nullptr) {
      return stmt;
    }

    for (auto &val : attrs) {
      std::string key = val.first;
      auto pos = key.find("_format");
      if (pos != std::string::npos) {
        std::string src_tensor = key.substr(0, pos);
        if (cast_map_.find(src_tensor) != cast_map_.end()) {
          std::string dst_tensor = cast_map_[src_tensor] + "_format";
          attrs.Set(dst_tensor, val.second);
        }
      }
      if (val.first == "Akg") {
        attrs.Set("dst_type", Expr("float32"));
      }
    }
    return AttrStmt::make(attrs, op->attr_key, op->value, op->body);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Block>();
    if (op == nullptr) {
      return stmt;
    }
    auto pro = op->first.as<Provide>();
    if (pro == nullptr) {
      return stmt;
    }
    if (auto call = pro->value.as<Call>()) {
      CHECK(call->args[0].as<Call>() != nullptr) << "Cast tensor is not Call op";
      if (call->name == "Cast" && cast_dtype_.find(call->args[0].as<Call>()->name) != cast_dtype_.end()) {
        return op->rest;
      }
    }
    return stmt;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    Array<Expr> args;
    for (const auto &arg : op->args) {
      if (auto tensor = arg.as<Call>()) {
        if (cast_map_.find(tensor->name) != cast_map_.end()) {
          args.push_back(Call::make(cast_dtype_.at(cast_map_.at(tensor->name)), cast_map_.at(tensor->name),
                                    tensor->args, tensor->call_type, cast_func_map_.at(tensor->name)));
        } else {
          args.push_back(arg);
        }
      } else {
        args.push_back(arg);
      }
    }
    if (cast_map_.find(op->name) != cast_map_.end()) {
      return Call::make(cast_dtype_.at(cast_map_.at(op->name)), cast_map_.at(op->name), args, op->call_type,
                        cast_func_map_.at(op->name), op->value_index);
    }
    return Call::make(op->type, op->name, args, op->call_type, op->func);
  }

 private:
  std::unordered_map<std::string, std::string> cast_map_;
  std::unordered_map<std::string, FunctionRef> cast_func_map_;
  std::unordered_map<std::string, air::DataType> cast_dtype_;
  std::string matmul_output_;
};

Stmt DeleteCast(const Stmt &s, BuildInfo*) {
  DeleteCastMatcher deletecast_matcher;
  deletecast_matcher.Visit(s);
  if (!deletecast_matcher.Matched()) {
    return s;
  }
  return DeleteCastMutator(deletecast_matcher).Mutate(s);
}
}  // namespace akg
