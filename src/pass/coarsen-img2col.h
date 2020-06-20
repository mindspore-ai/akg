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

#ifndef PASS_COARSEN_IMG2COL_H_
#define PASS_COARSEN_IMG2COL_H_

namespace akg {
namespace ir {
// This high-level intrinsic will be split into the two D-intrinsics,
// set_fmatrix and img2col_cbuf_to_{ca, cb, ub}
const char kCCEImg2ColIntrinName[] = "cce_img2col_";
const char kCCEImg2ColUBIntrinName[] = "cce_img2col_ub";
const int kCCEImg2ColArgNum = 23;

// Indices of args in the high-level cce_img2col_ intrinsic call
enum CCEImg2ColArgIdx {
  kOutputPtr = 0,
  kInputPtr,
  kFPosW,
  kFPosH,
  kInitFMPosW,
  kInitFMPosH,
  kInitFMPosC1,
  kStrideW,
  kStrideH,
  kKernelW,
  kKernelH,
  kDilationW,
  kDilationH,
  kDestJumpOffset,
  kRepeatMode,
  kRepeats,
  kC0Mode,
  kLastImg2ColArg,
  kPadTop = kLastImg2ColArg,
  kPadBottom,
  kPadLeft,
  kPadRight,
  kFMHeight,
  kFMWidth,
  kArgNums
};

enum CCEImg2colTiling { kImg2colHead = 0, kImg2colBody, kImg2colTail };

class LowerImg2ColMutator : public IRMutator {
 public:
  void setTimes(int times) { kImg2ColAllTimes = times; }

 private:
  // The D-intrinsics
  std::string kSetFmatrixName = "set_fmatrix";
  std::string kImg2ColName = "img2col_cbuf_to_ca";
  int kImg2ColTimes{0};
  int kImg2ColAllTimes{0};
  bool kImg2ColL1UB_{false};

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final;

  Stmt MutateImg2Col(const Call *op);

  Array<Expr> MakeSetFmatrixArgs(const Array<Expr> &all_args, bool extent_fmh = false);

  Array<Expr> UpdateImg2ColArgs(const Array<Expr> &args);

  bool NeedUpdateInputBuffer(const Call *op);

  bool NeedTiling() const;
  Expr GetOutValue(const Expr &fmW, const Expr &padL, const Expr &padR, const Expr &kernelW, const Expr &strideW);
};

class CoarsenImg2ColMutator : public IRMutator {
 private:
  struct LoopNestInfo {
    Var loopVar;
    Expr loopExtent;
    bool isEvaluate;
  };

  const int kFractalSize = 16;

  // The D-intrinsic that copies data from GM to L1
  std::string kDMACopyName = "copy_gm_to_cbuf";
  // Index of the burst length argument to copy_gm_to_cbuf
  const int kBurstLengthArgIdx = 4;

  bool img2col_coarsened_{false};
  int burst_length_ = 0;
  bool opt_turn_off_coarsening_ = false;

  Stmt Mutate_(const For *op, const Stmt &s) final;

  Expr Mutate_(const Call *op, const Expr &e) final;

  Expr MutateImg2Col(const Call *op, const Expr &e);

  int GetInteger(Expr extent);

  // Return fm_height
  bool GetRowsFromBaseAddress(const Expr &base_address, const Var &innermost_var, const Expr &e_pattern,
                              int64_t kernel_h, int64_t kernel_w, int64_t fm_width, int32_t *fm_height);

  Expr MakeFPW(const Expr &e, int32_t kernel_w);
  Expr MakeFPH(const Expr &e, int32_t kernel_w, int32_t kernel_h);

  bool CheckEqualMod(const Expr &e1, const Expr &e2, int m);

  std::vector<LoopNestInfo> loopNest_;
};
}  // namespace ir
}  // namespace akg
#endif  // PASS_COARSEN_IMG2COL_H_
