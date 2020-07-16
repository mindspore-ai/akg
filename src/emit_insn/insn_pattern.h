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

#ifndef EMIT_INSN_INSN_PATTERN_H_
#define EMIT_INSN_INSN_PATTERN_H_

#include <string>

#include "common/array_api.h"
#include "tvm.h"
#include "ir_pass.h"
#include "cce_params.h"
#include "insn_info.h"

namespace akg {
bool IsScalarMode(const StmtInfoList &info_list);

void SplitAxis(StmtInfoList &com_info_list, StmtInfo &for_info, const Var &axis_var, int new_size);

struct PatternResult {
  ArgInfo arg_info;
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtInfo for_info;
};

class PatternGenerator {
 public:
  PatternGenerator(const StmtInfoList &dst_info_list, const StmtInfo &for_info)
      : for_info(for_info),
        not_this_pattern(-1.0f),
        split_latency_coef(10.0f),
        repeat_latency_coef(3.0f),
        offset_latency_coef(0.1f) {
    CHECK(!dst_info_list.empty());
    dst_info = dst_info_list[0];
  }
  virtual ~PatternGenerator() = default;
  virtual PatternResult GetInsnArgs() = 0;

 protected:
  int GetNonZeroShape(const Expr &dst_shape, const Expr &src0_shape, const Expr &src1_shape = Expr());
  void GetShapeInfoAndSwap(Array<Var> &var, Array<Expr> &shape, Array<Expr> &strides, int idx1, int idx2);

  virtual float Compute3DPatternMaskRate() { return not_this_pattern; }
  virtual float Compute2DBlockPatternMaskRate() { return not_this_pattern; }
  virtual float Compute2DPatternMaskRate() { return not_this_pattern; }
  virtual float Compute1DPatternMaskRate() { return not_this_pattern; }
  virtual Array<Var> Get3DPattern() { return {}; }
  virtual Array<Var> Get2DBlockPattern() { return {}; }
  virtual Array<Var> Get2DPattern() { return {}; }
  virtual Array<Var> Get1DPattern() { return {}; }
  virtual PatternResult GenResult(const Array<Var> &elim_var) = 0;

  StmtStoreInfo dst_info;
  StmtInfo for_info;

  const float not_this_pattern;
  const float split_latency_coef;
  const float repeat_latency_coef;
  const float offset_latency_coef;
};

class SingleVecPatternGenerator : public PatternGenerator {
 public:
  SingleVecPatternGenerator(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list,
                            const StmtInfo &for_info, const std::string &mode = "elewise")
      : PatternGenerator(dst_info_list, for_info),
        arg_info(ArgInfo(make_node<ArgInfoNode>())),
        body_args(VectorArgInfo()),
        tail_args(VectorArgInfo()),
        mode(mode) {
    if (src_info_list.empty()) {
      src_info = dst_info.Copy();
    } else {
      CHECK(!src_info_list.empty());
      src_info = src_info_list[0];
    }
  }
  ~SingleVecPatternGenerator() override = default;
  PatternResult GetInsnArgs() final;

 protected:
  float Compute3DPatternMaskRate() final;
  float Compute2DBlockPatternMaskRate() final;
  float Compute2DPatternMaskRate() final;
  float Compute1DPatternMaskRate() final;
  float Compute3DsPatternMaskRate();
  float Compute2DRepeatPatternMaskRate();
  Array<Var> Get3DPattern() final;
  Array<Var> Get2DBlockPattern() final;
  Array<Var> Get2DPattern() final;
  Array<Var> Get1DPattern() final;
  Array<Var> Get3DsPattern();
  Array<Var> Get2DRepeatPattern();
  PatternResult GenResult(const Array<Var> &elim_var) final;

 private:
  void CalcParams();
  int GetLastDimShape(const Expr &dst_shape, const Expr &src_shape);

  struct Params {
    Array<Var> dst_var;
    Array<Var> src_var;
    Array<Expr> dst_shape;
    Array<Expr> src_shape;
    Array<Expr> dst_strides;
    Array<Expr> src_strides;
    int non_zero_shape1 = 0;
    int non_zero_shape2 = 0;
    int non_zero_shape3 = 0;
    int all_points = 0;
    int dst_block_size = 0;
    int src_block_size = 0;
    int mask_block_size = 0;
    int dst_bits = 0;
    int src_bits = 0;
    int max_bits = 0;
    int dst_vec_max_len = 0;
    int vec_max_len = 0;
    int block_offset = 0;
  };

  StmtStoreInfo src_info;
  Params params;
  ArgInfo arg_info;
  VectorArgInfo body_args;
  VectorArgInfo tail_args;
  std::string mode;
  Type data_type;
};

class BinaryVecPatternGenerator : public PatternGenerator {
 public:
  BinaryVecPatternGenerator(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list,
                            const StmtInfo &for_info, const std::string &mode, bool expand_mask = true)
      : PatternGenerator(dst_info_list, for_info),
        src_info_list(src_info_list),
        arg_info(ArgInfo(make_node<ArgInfoNode>())),
        body_args(VectorArgInfo()),
        tail_args(VectorArgInfo()),
        empty_var(Var("")),
        mode(mode),
        expand_mask(expand_mask) {}
  ~BinaryVecPatternGenerator() override = default;

  PatternResult GetInsnArgs() final;

 protected:
  float Compute3DPatternMaskRate() final;
  float Compute2DBlockPatternMaskRate() final;
  float Compute2DPatternMaskRate() final;
  float Compute1DPatternMaskRate() final;
  Array<Var> Get3DPattern() final;
  Array<Var> Get2DBlockPattern() final;
  Array<Var> Get2DPattern() final;
  Array<Var> Get1DPattern() final;
  PatternResult GenResult(const Array<Var> &elim_var) final;

 private:
  void CalcParams();
  bool IsSamePatternComInfo(const StmtStoreInfo &info_a, const StmtStoreInfo &info_b);
  bool IsNonZeroShapeEqual(const Array<Expr> &shape_list);
  void AppendEmptyVar(StmtInfoList &info_list);

  struct Params {
    Array<Var> dst_var;
    Array<Expr> dst_shape;
    Array<Expr> dst_strides;
    Array<Var> src_var0;
    Array<Expr> src_shape0;
    Array<Expr> src_strides0;
    Array<Var> src_var1;
    Array<Expr> src_shape1;
    Array<Expr> src_strides1;
    int non_zero_shape1 = 0;
    int non_zero_shape2 = 0;
    int non_zero_shape3 = 0;
    int all_points = 0;
    int block_size = 0;
    int last_dim_shape = 0;
    int vec_max_len = 0;
  };

  StmtInfoList src_info_list;
  ArgInfo arg_info;
  VectorArgInfo body_args;
  VectorArgInfo tail_args;
  Params params;
  Var empty_var;
  std::string mode;
  bool expand_mask;
};

class ReduceLastAxisPatternGenerator : public PatternGenerator {
 public:
  ReduceLastAxisPatternGenerator(const StmtStoreInfo &dst_info, const StmtStoreInfo &src_info, const StmtInfo &for_info,
                                 const std::string &intrin_name)
      : PatternGenerator({dst_info}, for_info),
        src_info(src_info),
        arg_info(ArgInfo(make_node<ArgInfoNode>())),
        body_args(VectorArgInfo()),
        tail_args(VectorArgInfo()),
        intrin_name(intrin_name) {}
  PatternResult GetInsnArgs() final;
  ~ReduceLastAxisPatternGenerator() override = default;

 protected:
  float Compute2DBlockPatternMaskRate() final;
  Array<Var> Get2DBlockPattern() final;
  Array<Var> Get1DPattern() final;
  PatternResult GenResult(const Array<Var> &elim_var) final;

 private:
  void CalcParams();

  struct Params {
    Array<Var> src_var;
    int block_size = 0;
    int vec_max_len = 0;
    int last_dim_shape = 0;
    Expr insn_offset_scale_factor;
  };

  StmtStoreInfo src_info;
  ArgInfo arg_info;
  VectorArgInfo body_args;
  VectorArgInfo tail_args;
  Array<VectorArgInfo> mix_vec_arg_list;
  std::string intrin_name;
  Params params;
};

std::string GetSingleVecComputationInfo(const Stmt &stmt, const std::string &intrin_name,
                                        Array<StmtStoreInfo> &dst_info_list, Array<StmtStoreInfo> &src_info_list,
                                        StmtInfo &if_info, StmtInfo &for_info, bool need_compact = true);

ArgInfo GetBinaryVecInsnArgs(const Stmt &stmt, std::string intrin_name, StmtInfoList &dst_info_list,
                             StmtInfoList &src_info_list, StmtInfo &if_info, StmtInfo &for_info,
                             bool enable_bisect = true);

ArgInfo GetMultiVecInsnArgs(StmtInfoList &dst_info_list, StmtInfoList &src_info_list, StmtInfo &for_info);

void FillLastDim(StmtInfoList &dst_info_list, StmtInfoList &src_info_list, StmtInfo &for_info);

void FillEmptyVar(Array<StmtStoreInfo> &info_list);

void CleanZeroStrides(StmtStoreInfo &info);

void CleanZeroStrides(Array<StmtStoreInfo> &info_list);

Array<Expr> GetVecMask(int data_len, int data_num, Type data_type, int begin = 0);

Map<std::string, Expr> GetDmaLoad2DInsnArgs(const std::string &intrin_name, const StmtInfoList &dst_info_list,
                                            const StmtInfoList &src_info_list, StmtInfo &for_info);

void GetDmaComputationInfo(const Stmt &stmt, StmtInfoList &dst_info_list, StmtInfoList &src_info_list,
                           StmtInfo &if_info, StmtInfo &for_info, std::string &dma_mode, std::string &intrin_name);

Map<std::string, Expr> GetDmaCopyInsnArgs(std::string &intrin_name, const StmtInfoList &dst_info_list,
                                          const StmtInfoList &src_info_list, StmtInfo &for_info);

Map<std::string, Expr> GetDmaCopyInsnArgs(std::string &intrin_name, const StmtInfoList &dst_info_list,
                                          const StmtInfoList &src_info_list, StmtInfo &for_info,
                                          Map<std::string, Expr> &ub_copy_pre, Map<std::string, Expr> &ub_copy_post);

BisectionInfoWrapper SeparateComInfoToBisectionInfoList(const StmtInfoList &dst_info_list,
                                                        const StmtInfoList &src_info_list, const StmtInfo &for_info,
                                                        StmtInfo &if_info, bool last_axis, int postfix);

extern const char *const DummyLastVar;
}  // namespace akg
#endif  // EMIT_INSN_INSN_PATTERN_H_
