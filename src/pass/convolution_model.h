/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef PASS_CONVOLUTION_MODEL_H_
#define PASS_CONVOLUTION_MODEL_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include "tvm.h"

namespace akg {
namespace ir {
namespace {
constexpr auto PRAGMA_CONV_BACKPROP_INPUT = "pragma_conv_backprop_input";
constexpr auto PRAGMA_CONV_BACKPROP_FILTER = "pragma_conv_backprop_filter";
constexpr auto PRAGMA_CONV_FEATURE_N = "pragma_conv_fm_n";
constexpr auto PRAGMA_CONV_FEATURE_C = "pragma_conv_fm_c";
constexpr auto PRAGMA_CONV_FEATURE_H = "pragma_conv_fm_h";
constexpr auto PRAGMA_CONV_FEATURE_W = "pragma_conv_fm_w";
constexpr auto PRAGMA_CONV_KERNEL_N = "pragma_conv_kernel_n";
constexpr auto PRAGMA_CONV_KERNEL_H = "pragma_conv_kernel_h";
constexpr auto PRAGMA_CONV_KERNEL_W = "pragma_conv_kernel_w";
constexpr auto PRAGMA_CONV_PAD_TOP = "pragma_conv_padding_top";
constexpr auto PRAGMA_CONV_PAD_BOTTOM = "pragma_conv_padding_bottom";
constexpr auto PRAGMA_CONV_PAD_LEFT = "pragma_conv_padding_left";
constexpr auto PRAGMA_CONV_PAD_RIGHT = "pragma_conv_padding_right";
constexpr auto PRAGMA_CONV_STRIDE_H = "pragma_conv_stride_h";
constexpr auto PRAGMA_CONV_STRIDE_W = "pragma_conv_stride_w";
constexpr auto PRAGMA_CONV_DILATION_H = "pragma_conv_dilation_h";
constexpr auto PRAGMA_CONV_DILATION_W = "pragma_conv_dilation_w";
constexpr auto PRAGMA_CONV_TILE_B = "pragma_conv_batch_cut";
constexpr auto PRAGMA_CONV_TILE_CIN = "pragma_conv_cin_cut";
constexpr auto PRAGMA_CONV_TILE_CO = "pragma_conv_co_cut";
constexpr auto PRAGMA_CONV_TILE_H = "pragma_conv_h_cut";
constexpr auto PRAGMA_CONV_TILE_W = "pragma_conv_w_cut";
constexpr auto PRAGMA_CONV_TILE_KH = "pragma_conv_kh_cut";
constexpr auto PRAGMA_CONV_TILE_KW = "pragma_conv_kw_cut";
constexpr auto PRAGMA_CONV_TILE_M = "pragma_conv_m_cut";
constexpr auto PRAGMA_CONV_TILE_K = "pragma_conv_k_cut";
constexpr auto PRAGMA_CONV_TILE_N = "pragma_conv_n_cut";
}  // namespace

#define IS_ATTR_EXIST(attrs_, key_) ((attrs_).count(key_) > 0)

#define GET_INTIMM_ATTR(attrs_, key_) (attrs_)[key_].as<IntImm>()->value
#define GET_STRINGIMM_ATTR(attrs_, key_) (attrs_)[key_].as<StringImm>()->value
#define IS_STATIC true

#define GET_EXPR_ATTR(attrs_, key_, default_) \
  (IS_ATTR_EXIST(attrs_, key_) ? (Downcast<Expr>((attrs_)[key_])) : (default_))

#define GET_INTIMM_ATTR_DEFAULT(attrs_, key_, default_)                                                               \
  ((IS_ATTR_EXIST(attrs_, key_) && ((attrs_)[key_].as<IntImm>())) ? (static_cast<int>(GET_INTIMM_ATTR(attrs_, key_))) \
                                                                  : (default_))

#define GET_STRINGIMM_ATTR_DEFAULT(attrs_, key_, default_) \
  ((IS_ATTR_EXIST(attrs_, key_) && ((attrs_)[key_].as<StringImm>())) ? (GET_STRINGIMM_ATTR(attrs_, key_)) : (default_))

class Convolution : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_attrs") {
      attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
      if (attrs_.count(PRAGMA_CONV_BACKPROP_INPUT) > 0) {
        CHECK(attrs_[PRAGMA_CONV_BACKPROP_INPUT].as<IntImm>());
        isConvBackpropInput_ = GET_INTIMM_ATTR(attrs_, PRAGMA_CONV_BACKPROP_INPUT);
      }
      if (attrs_.count(PRAGMA_CONV_BACKPROP_FILTER) > 0) {
        CHECK(attrs_[PRAGMA_CONV_BACKPROP_FILTER].as<IntImm>());
        isConvBackpropFilter_ = GET_INTIMM_ATTR(attrs_, PRAGMA_CONV_BACKPROP_FILTER);
      }
    } else if (op->attr_key == "pragma_im2col") {
      isConv_ = true;
    }

    IRVisitor::Visit_(op);
  }

  Map<std::string, NodeRef> attrs_;
  bool isConv_{false};
  bool isConvBackpropInput_{false};
  bool isConvBackpropFilter_{false};
};

struct NCHW {
  Expr n{0};
  Expr c{0};
  Expr h{0};
  Expr w{0};

  NCHW() : n(0), c(0), h(0), w(0) {}

  NCHW(Expr n, Expr c, Expr h, Expr w) : n(n), c(floordiv(c + 15, 16) * 16), h(h), w(w) {}
};

struct Weight {
  Expr cout{0};
  Expr cin{0};
  Expr kh{0};
  Expr kw{0};
  Expr d_kh{0};
  Expr d_kw{0};

  Weight() : cout(0), cin(0), kh(0), kw(0), d_kh(0), d_kw(0) {}

  Weight(Expr cout, Expr cin, Expr kh, Expr kw)
      : cout(floordiv(cout + 15, 16) * 16), cin(floordiv(cin + 15, 16) * 16), kh(kh), kw(kw), d_kh(0), d_kw(0) {}

  Weight(Expr cout, Expr cin, Expr kh, Expr kw, Expr d_kh, Expr d_kw)
      : cout(floordiv(cout + 15, 16) * 16), cin(floordiv(cin + 15, 16) * 16), kh(kh), kw(kw), d_kh(d_kh), d_kw(d_kw) {}
};

struct Pad {
  Expr pad_top{0};
  Expr pad_bottom{0};
  Expr pad_left{0};
  Expr pad_right{0};

  Pad() : pad_top(0), pad_bottom(0), pad_left(0), pad_right(0) {}

  explicit Pad(Expr pad) : pad_top(pad), pad_bottom(pad), pad_left(pad), pad_right(pad) {}

  Pad(Expr pad_top, Expr pad_bottom, Expr pad_left, Expr pad_right)
      : pad_top(pad_top), pad_bottom(pad_bottom), pad_left(pad_left), pad_right(pad_right) {}
};

struct Stride {
  Expr stride_h{0};
  Expr stride_w{0};

  Stride() : stride_h(0), stride_w(0) {}

  explicit Stride(Expr stride) : stride_h(stride), stride_w(stride) {}

  Stride(Expr stride_h, Expr stride_w) : stride_h(stride_h), stride_w(stride_w) {}
};

struct Dilation {
  Expr dilation_h{0};
  Expr dilation_w{0};

  Dilation() : dilation_h(0), dilation_w(0) {}

  explicit Dilation(Expr dilation) : dilation_h(dilation), dilation_w(dilation) {}

  Dilation(Expr dilation_h, Expr dilation_w) : dilation_h(dilation_h), dilation_w(dilation_w) {}
};

struct Conv {
  NCHW input;
  Weight filter;
  NCHW output;
  Pad pad;
  Stride stride;
  Dilation dilation;
  Expr block_size{16};

  void infer_output_shape() {
    Expr out_n = this->input.n;
    Expr out_c = this->filter.cout;

    Expr in_h = this->input.h;
    Expr pad_top = this->pad.pad_top;
    Expr pad_bottom = this->pad.pad_bottom;
    Expr d_kh = this->filter.d_kh;
    Expr stride_h = this->stride.stride_h;
    Expr out_h = (in_h + pad_top + pad_bottom - d_kh) / stride_h + 1;

    Expr in_w = this->input.w;
    Expr pad_left = this->pad.pad_left;
    Expr pad_right = this->pad.pad_right;
    Expr d_kw = this->filter.d_kw;
    Expr stride_w = this->stride.stride_w;
    Expr out_w = (in_w + pad_left + pad_right - d_kw) / stride_w + 1;

    this->output = NCHW(out_n, out_c, out_h, out_w);
  }

  Conv() {}

  Conv(Expr n, Expr c, Expr h, Expr w, Expr cout, Expr kh, Expr kw, Expr pad, Expr stride, Expr dilation)
      : input(n, c, h, w),
        filter(cout, c, kh, kw, (kh - 1) * dilation + 1, (kw - 1) * dilation + 1),
        pad(pad),
        stride(stride),
        dilation(dilation) {
    infer_output_shape();
  }

  Conv(Expr n, Expr c, Expr h, Expr w, Expr cout, Expr kh, Expr kw, Expr pad_top, Expr pad_bottom, Expr pad_left,
       Expr pad_right, Expr stride_h, Expr stride_w, Expr dilation_h, Expr dilation_w)
      : input(n, c, h, w),
        filter(cout, c, kh, kw, (kh - 1) * dilation_h + 1, (kw - 1) * dilation_w + 1),
        pad(pad_top, pad_bottom, pad_left, pad_right),
        stride(stride_h, stride_w),
        dilation(dilation_h, dilation_w) {
    infer_output_shape();
  }
};

struct Tiling {
  Expr cut_b{0};
  Expr cut_ci{0};
  Expr cut_co{0};
  Expr cut_h{0};
  Expr cut_w{0};

  Expr cut_kh{0};
  Expr cut_kw{0};

  Expr cut_m{0};
  Expr cut_k{0};
  Expr cut_n{0};

  Tiling() : cut_b(0), cut_ci(0), cut_co(0), cut_h(0), cut_w(0), cut_kh(0), cut_kw(0), cut_m(0), cut_k(0), cut_n(0) {}

  Tiling(Expr cut_b, Expr cut_ci, Expr cut_co, Expr cut_h, Expr cut_w, Expr cut_kh, Expr cut_kw, Expr cut_m, Expr cut_k,
         Expr cut_n)
      : cut_b(cut_b),
        cut_ci(cut_ci),
        cut_co(cut_co),
        cut_h(cut_h),
        cut_w(cut_w),
        cut_kh(cut_kh),
        cut_kw(cut_kw),
        cut_m(cut_m),
        cut_k(cut_k),
        cut_n(cut_n) {}
};

struct IsolateInfo {
  Expr outer{0};
  Expr inner{0};

  IsolateInfo() : outer(0), inner(0) {}

  IsolateInfo(Expr outer, Expr inner) : outer(outer), inner(inner) {}
};

class ConvolutionModel {
 public:
  ConvolutionModel() {}

  ConvolutionModel(const Map<std::string, NodeRef> &attrs, bool is_dynamic);

  virtual ~ConvolutionModel() {}

  int infer_isolate(std::vector<IsolateInfo> *info, Expr len, Expr cut);

  int infer_isolate_overlap(std::vector<IsolateInfo> *info, std::vector<IsolateInfo> *win_info, Expr len, Expr cut,
                            Expr stride, Expr kernel_dilation, Expr head, Expr tail);

  /* CA1 isolate Info */
  virtual int infer_CA1_tile() = 0;

  virtual int get_ci_idx(int isolate_idx) const;
  virtual int get_co_idx(int isolate_idx) const;
  virtual int get_b_idx(int isolate_idx) const;
  virtual int get_h_idx(int isolate_idx) const;
  virtual int get_w_idx(int isolate_idx) const;
  virtual int get_kh_idx(int isolate_idx) const;
  virtual int get_kw_idx(int isolate_idx) const;

  IsolateInfo get_ci_isolate_info(int isolate_idx);
  IsolateInfo get_co_isolate_info(int isolate_idx);
  IsolateInfo get_b_isolate_info(int isolate_idx);
  IsolateInfo get_h_isolate_info(int isolate_idx);
  IsolateInfo get_h_win_isolate_info(int isolate_idx);
  IsolateInfo get_w_isolate_info(int isolate_idx);
  IsolateInfo get_w_win_isolate_info(int isolate_idx);
  IsolateInfo get_kh_isolate_info(int isolate_idx);
  IsolateInfo get_kw_isolate_info(int isolate_idx);

  /* CA0 isolate Info */
  virtual int infer_CA0_tile(int isolate_idx) = 0;

  int get_n_idx(int gemm_idx) const;
  int get_m_idx(int gemm_idx) const;
  int get_k_idx(int gemm_idx) const;

  IsolateInfo get_n_isolate_info(int gemm_idx);
  IsolateInfo get_m_isolate_info(int gemm_idx);
  IsolateInfo get_k_isolate_info(int gemm_idx);

  Expr calc_till_idx(std::vector<IsolateInfo> *info, int idx) const;

 protected:
  Map<std::string, NodeRef> attrs_;

 public:
  bool is_dynamic_{false};
  bool reduce_at_ca1{false};
  int ca1_reduce_base{1};
  int ca0_reduce_base{1};

  Conv conv_;
  Tiling tile_;

  /* ca1 info */
  std::vector<IsolateInfo> b_info;
  int b_base{0};
  std::vector<IsolateInfo> ci_info;
  int ci_base{0};
  std::vector<IsolateInfo> co_info;
  int co_base{0};
  std::vector<IsolateInfo> h_info;
  std::vector<IsolateInfo> h_win_info;
  int h_base{0};
  std::vector<IsolateInfo> w_info;
  std::vector<IsolateInfo> w_win_info;
  int w_base{0};
  std::vector<IsolateInfo> kh_info;
  int kh_base{0};
  std::vector<IsolateInfo> kw_info;
  int kw_base{0};

  /* ca0 info */
  std::vector<IsolateInfo> m_info;
  int m_base{0};
  std::vector<IsolateInfo> k_info;
  int k_base{0};
  std::vector<IsolateInfo> n_info;
  int n_base{0};
};

class ConvolutionForwardModel : public ConvolutionModel {
 public:
  ConvolutionForwardModel() {}

  ConvolutionForwardModel(const Map<std::string, NodeRef> &attrs, bool is_dynamic);

  ~ConvolutionForwardModel() override = default;

  /* CA1 isolate Info */
  int infer_CA1_tile() override;

  int get_co_idx(int isolate_idx) const override;
  int get_h_idx(int isolate_idx) const override;
  int get_w_idx(int isolate_idx) const override;

  /* CA0 isolate Info */
  int infer_CA0_tile(int isolate_idx) override;
};

class ConvolutionBackpropInputModel : public ConvolutionModel {
 public:
  ConvolutionBackpropInputModel() {}

  ConvolutionBackpropInputModel(const Map<std::string, NodeRef> &attrs, bool is_dynamic);

  ~ConvolutionBackpropInputModel() override = default;

  /* CA1 isolate Info */
  int infer_CA1_tile() override;

  int get_co_idx(int isolate_idx) const override;
  int get_h_idx(int isolate_idx) const override;
  int get_w_idx(int isolate_idx) const override;

  /* CA0 isolate Info */
  int infer_CA0_tile(int isolate_idx) override;
};

class ConvolutionBackpropFilterModel : public ConvolutionModel {
 public:
  ConvolutionBackpropFilterModel() {}

  explicit ConvolutionBackpropFilterModel(const Map<std::string, NodeRef> &attrs, bool is_dynamic);

  ~ConvolutionBackpropFilterModel() override = default;

  /* CA1 isolate Info */
  int infer_CA1_tile() override;

  int get_ci_idx(int isolate_idx) const override;
  int get_co_idx(int isolate_idx) const override;
  int get_b_idx(int isolate_idx) const override;
  int get_h_idx(int isolate_idx) const override;
  int get_w_idx(int isolate_idx) const override;
  int get_kh_idx(int isolate_idx) const override;
  int get_kw_idx(int isolate_idx) const override;

  /* CA0 isolate Info */
  int infer_CA0_tile(int isolate_idx) override;
};
}  // namespace ir
}  // namespace akg

#endif  // PASS_CONVOLUTION_MODEL_H_
