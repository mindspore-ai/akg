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

#include "pass/convolution_model.h"

namespace akg {
namespace ir {
/* ConvolutionModel */
ConvolutionModel::ConvolutionModel(const Map<std::string, NodeRef> &attrs, bool is_dynamic)
    : attrs_(attrs), is_dynamic_(is_dynamic) {
  Expr n = GET_EXPR_ATTR(attrs_, ATTR_CONV_FEATURE_N, 0);
  Expr c = GET_EXPR_ATTR(attrs_, ATTR_CONV_FEATURE_C, 0);
  CHECK_GT(attrs_.count(ATTR_CONV_FEATURE_H), 0);
  Expr h = GET_EXPR_ATTR(attrs_, ATTR_CONV_FEATURE_H, 0);
  Expr w = GET_EXPR_ATTR(attrs_, ATTR_CONV_FEATURE_W, 0);

  Expr cout = GET_EXPR_ATTR(attrs_, ATTR_CONV_KERNEL_N, 0);
  Expr kh = GET_EXPR_ATTR(attrs_, ATTR_CONV_KERNEL_H, 0);
  Expr kw = GET_EXPR_ATTR(attrs_, ATTR_CONV_KERNEL_W, 0);

  Expr pad_top = GET_EXPR_ATTR(attrs_, ATTR_CONV_PAD_TOP, 0);
  Expr pad_bottom = GET_EXPR_ATTR(attrs_, ATTR_CONV_PAD_BOTTOM, 0);
  Expr pad_left = GET_EXPR_ATTR(attrs_, ATTR_CONV_PAD_LEFT, 0);
  Expr pad_right = GET_EXPR_ATTR(attrs_, ATTR_CONV_PAD_RIGHT, 0);

  Expr stride_h = GET_EXPR_ATTR(attrs_, ATTR_CONV_STRIDE_H, 0);
  Expr stride_w = GET_EXPR_ATTR(attrs_, ATTR_CONV_STRIDE_W, 0);

  Expr dilation_h = GET_EXPR_ATTR(attrs_, ATTR_CONV_DILATION_H, 0);
  Expr dilation_w = GET_EXPR_ATTR(attrs_, ATTR_CONV_DILATION_W, 0);

  conv_ = {n,          c,        h,         w,        cout,     kh,         kw,        pad_top,
           pad_bottom, pad_left, pad_right, stride_h, stride_w, dilation_h, dilation_w};

  Expr cut_b = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_B, 0);
  Expr cut_ci = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_CIN, 0);
  Expr cut_co = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_CO, 0);
  Expr cut_h = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_H, 0);
  Expr cut_w = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_W, 0);
  Expr cut_kh = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_KH, kh);
  Expr cut_kw = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_KW, kw);
  Expr cut_m = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_M, 0);
  Expr cut_k = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_K, 0);
  Expr cut_n = GET_EXPR_ATTR(attrs_, ATTR_CONV_TILE_N, 0);

  tile_ = {cut_b, cut_ci, cut_co, cut_h, cut_w, cut_kh, cut_kw, cut_m, cut_k, cut_n};
}

int ConvolutionModel::infer_isolate(std::vector<IsolateInfo> *info, Expr len, Expr cut) {
  info->clear();
  if (!is_dynamic_) {
    CHECK(len.as<IntImm>() && cut.as<IntImm>());
    CHECK(len.as<IntImm>()->value >= cut.as<IntImm>()->value) << len << " : " << cut;
    if (len.as<IntImm>()->value % cut.as<IntImm>()->value > 0) {
      info->emplace_back(IsolateInfo(len / cut, cut));
      info->emplace_back(IsolateInfo(1, len % cut));
    } else {
      info->emplace_back(IsolateInfo(len / cut, cut));
    }
  } else {
    info->emplace_back(IsolateInfo(len / cut, cut));
    info->emplace_back(IsolateInfo(1, len % cut));
  }
  return static_cast<int>(info->size());
}

int ConvolutionModel::infer_isolate_overlap(std::vector<IsolateInfo> *info, std::vector<IsolateInfo> *win_info,
                                            Expr len, Expr cut, Expr stride, Expr kernel_dilation, Expr head,
                                            Expr tail) {
  info->clear();
  win_info->clear();

#define WIN_TO_LEN(win_) (((win_)-1) * stride + kernel_dilation)
  if (!is_dynamic_) {
    CHECK(len.as<IntImm>() && cut.as<IntImm>());
    CHECK(tail.as<IntImm>());
    CHECK(head.as<IntImm>());
    CHECK_NE(cut.as<IntImm>()->value, 0);
    int base = (len.as<IntImm>()->value + cut.as<IntImm>()->value - 1) / cut.as<IntImm>()->value;
    // [head body tail]
    if (base >= 3) {
      if (head.as<IntImm>()->value) {
        info->emplace_back(IsolateInfo(1, WIN_TO_LEN(cut) - head));
        win_info->emplace_back(IsolateInfo(1, cut));
      }

      if ((tail.as<IntImm>()->value > 0) || (len.as<IntImm>()->value % cut.as<IntImm>()->value > 0)) {
        info->emplace_back(IsolateInfo(base - (head.as<IntImm>()->value ? Expr(2) : Expr(1)), WIN_TO_LEN(cut)));
        info->emplace_back(IsolateInfo(
          1, WIN_TO_LEN((len.as<IntImm>()->value % cut.as<IntImm>()->value == 0) ? cut : (len % cut)) - tail));

        win_info->emplace_back(IsolateInfo(base - (head.as<IntImm>()->value ? Expr(2) : Expr(1)), cut));
        win_info->emplace_back(
          IsolateInfo(1, (len.as<IntImm>()->value % cut.as<IntImm>()->value == 0) ? cut : (len % cut)));
      } else {
        info->emplace_back(IsolateInfo(base - (head.as<IntImm>()->value ? Expr(1) : Expr(0)), WIN_TO_LEN(cut) - tail));
        win_info->emplace_back(IsolateInfo(base - (head.as<IntImm>()->value ? Expr(1) : Expr(0)), cut));
      }
    } else if (base == 2) {  // [head body] or [head tail]
      if (head.as<IntImm>()->value) {
        info->emplace_back(IsolateInfo(1, WIN_TO_LEN(cut) - head));
        win_info->emplace_back(IsolateInfo(1, cut));
      }

      if ((tail.as<IntImm>()->value > 0) || (len.as<IntImm>()->value % cut.as<IntImm>()->value > 0)) {
        if (!head.as<IntImm>()->value) {
          info->emplace_back(IsolateInfo(base - 1, WIN_TO_LEN(cut)));
          win_info->emplace_back(IsolateInfo(base - 1, cut));
        }
        info->emplace_back(IsolateInfo(
          1, WIN_TO_LEN((len.as<IntImm>()->value % cut.as<IntImm>()->value == 0) ? cut : (len % cut)) - tail));
        win_info->emplace_back(
          IsolateInfo(1, (len.as<IntImm>()->value % cut.as<IntImm>()->value == 0) ? cut : (len % cut)));
      } else {
        info->emplace_back(IsolateInfo(base - (head.as<IntImm>()->value ? Expr(1) : Expr(0)), WIN_TO_LEN(cut)));
        win_info->emplace_back(IsolateInfo(base - (head.as<IntImm>()->value ? Expr(1) : Expr(0)), cut));
      }
    } else {
      CHECK_EQ(base, 1);
      info->emplace_back(IsolateInfo(base, WIN_TO_LEN(cut) - head - tail));
      win_info->emplace_back(IsolateInfo(base, cut));
    }
  } else {
    info->emplace_back(IsolateInfo(len / cut, WIN_TO_LEN(cut)));
    info->emplace_back(IsolateInfo(1, WIN_TO_LEN(len % cut)));

    win_info->emplace_back(IsolateInfo(len / cut, cut));
    win_info->emplace_back(IsolateInfo(1, len % cut));
  }
#undef WIN_TO_LEN

  return static_cast<int>(info->size());
}

Expr ConvolutionModel::calc_till_idx(std::vector<IsolateInfo> *info, int idx) const {
  Expr len{0};
  for (int i = 0; i < idx; i++) {
    len += info->at(i).inner * info->at(i).outer;
  }

  return len;
}

int ConvolutionModel::get_ci_idx(int isolate_idx) const {
  // without Tile Ci, isolate index is 0
  return 0;
}

int ConvolutionModel::get_co_idx(int isolate_idx) const {
  // without Tile Co, isolate index is 0
  return 0;
}

int ConvolutionModel::get_b_idx(int isolate_idx) const {
  // without Tile Batch, isolate index is 0
  return 0;
}

int ConvolutionModel::get_h_idx(int isolate_idx) const {
  // without Tile H, isolate index is 0
  return 0;
}

int ConvolutionModel::get_w_idx(int isolate_idx) const {
  // without Tile W, isolate index is 0
  return 0;
}

int ConvolutionModel::get_kh_idx(int isolate_idx) const {
  // without Tile KH, isolate index is 0
  return 0;
}

int ConvolutionModel::get_kw_idx(int isolate_idx) const {
  // without Tile KW, isolate index is 0
  return 0;
}

IsolateInfo ConvolutionModel::get_ci_isolate_info(int isolate_idx) { return ci_info[get_ci_idx(isolate_idx)]; }

IsolateInfo ConvolutionModel::get_co_isolate_info(int isolate_idx) { return co_info[get_co_idx(isolate_idx)]; }

IsolateInfo ConvolutionModel::get_b_isolate_info(int isolate_idx) { return b_info[get_b_idx(isolate_idx)]; }

IsolateInfo ConvolutionModel::get_h_isolate_info(int isolate_idx) { return h_info[get_h_idx(isolate_idx)]; }

IsolateInfo ConvolutionModel::get_h_win_isolate_info(int isolate_idx) { return h_win_info[get_h_idx(isolate_idx)]; }

IsolateInfo ConvolutionModel::get_w_isolate_info(int isolate_idx) { return w_info[get_w_idx(isolate_idx)]; }

IsolateInfo ConvolutionModel::get_w_win_isolate_info(int isolate_idx) { return w_win_info[get_w_idx(isolate_idx)]; }

IsolateInfo ConvolutionModel::get_kh_isolate_info(int isolate_idx) { return kh_info[get_kh_idx(isolate_idx)]; }

IsolateInfo ConvolutionModel::get_kw_isolate_info(int isolate_idx) { return kw_info[get_kw_idx(isolate_idx)]; }

int ConvolutionModel::get_n_idx(int gemm_idx) const {
  CHECK_NE((m_base * k_base), 0);
  CHECK_NE(n_base, 0);
  return (gemm_idx / (m_base * k_base) % n_base);
}

int ConvolutionModel::get_m_idx(int gemm_idx) const {
  CHECK_NE(k_base, 0);
  CHECK_NE(m_base, 0);
  return (gemm_idx / k_base % m_base);
}

int ConvolutionModel::get_k_idx(int gemm_idx) const {
  CHECK_NE(k_base, 0);
  return (gemm_idx % k_base);
}

IsolateInfo ConvolutionModel::get_n_isolate_info(int gemm_idx) { return n_info[get_n_idx(gemm_idx)]; }

IsolateInfo ConvolutionModel::get_m_isolate_info(int gemm_idx) { return m_info[get_m_idx(gemm_idx)]; }

IsolateInfo ConvolutionModel::get_k_isolate_info(int gemm_idx) { return k_info[get_k_idx(gemm_idx)]; }

/* ConvolutionForwardModel */
ConvolutionForwardModel::ConvolutionForwardModel(const Map<std::string, NodeRef> &attrs, bool is_dynamic)
    : ConvolutionModel(attrs, is_dynamic) {}

int ConvolutionForwardModel::infer_L1_tile() {
  Expr co = conv_.output.c;
  Expr cut_co = tile_.cut_co;
  if (!is_dynamic_) {
    const auto int_co = co.as<IntImm>();
    const auto int_cut_co = cut_co.as<IntImm>();
    CHECK(int_co && int_cut_co);
    if (int_cut_co->value > int_co->value) {
      cut_co = co;
    }
  }
  co_base = infer_isolate(&co_info, co, cut_co);

  Expr h = conv_.input.h;
  Expr d_kh = conv_.filter.d_kh;
  Expr stride_h = conv_.stride.stride_h;
  Expr pad_top = conv_.pad.pad_top;
  Expr pad_bottom = conv_.pad.pad_bottom;
  Expr cut_h = tile_.cut_h;
  Expr win_h = (h + pad_top + pad_bottom - d_kh) / stride_h + 1;
  Expr win_cut_h = (cut_h - d_kh) / stride_h + 1;
  if (!is_dynamic_) {
    CHECK(win_cut_h.as<IntImm>());
    CHECK(win_h.as<IntImm>());
    if (win_cut_h.as<IntImm>()->value > win_h.as<IntImm>()->value) {
      win_cut_h = win_h;
      tile_.cut_h = (win_h - 1) * stride_h + d_kh;
    }
  }
  Expr h_head = pad_top;
  Expr h_tail = Expr(0);
  if (!is_dynamic_) {
    CHECK(win_cut_h.as<IntImm>());
    CHECK(pad_top.as<IntImm>());
    CHECK(win_h.as<IntImm>());
    CHECK(d_kh.as<IntImm>());
    CHECK(h.as<IntImm>());

    CHECK(win_cut_h.as<IntImm>()->value * stride_h.as<IntImm>()->value >= pad_top.as<IntImm>()->value)
      << "Only one head for cut H axis";
    CHECK((((win_h.as<IntImm>()->value + win_cut_h.as<IntImm>()->value - 1) / win_cut_h.as<IntImm>()->value - 1) *
             win_cut_h.as<IntImm>()->value -
           1) *
              stride_h.as<IntImm>()->value +
            d_kh.as<IntImm>()->value <=
          h.as<IntImm>()->value + pad_top.as<IntImm>()->value)
      << "Only one tail for cut H axis";
  }
  h_base = infer_isolate_overlap(&h_info, &h_win_info, win_h, win_cut_h, stride_h, d_kh, h_head, h_tail);

  Expr w = conv_.input.w;
  Expr d_kw = conv_.filter.d_kw;
  Expr stride_w = conv_.stride.stride_w;
  Expr pad_left = conv_.pad.pad_left;
  Expr pad_right = conv_.pad.pad_right;
  Expr cut_w = tile_.cut_w;
  Expr win_w = (w + pad_left + pad_right - d_kw) / stride_w + 1;
  Expr win_cut_w = (cut_w - d_kw) / stride_w + 1;
  if (!is_dynamic_) {
    CHECK(win_w.as<IntImm>());
    CHECK(win_cut_w.as<IntImm>());
    if (win_cut_w.as<IntImm>()->value > win_w.as<IntImm>()->value) {
      win_cut_w = win_w;
      tile_.cut_w = (win_w - 1) * stride_w + d_kw;
    }
  }
  Expr w_head = pad_left;
  Expr w_tail = Expr(0);
  if (!is_dynamic_) {
    CHECK(win_cut_w.as<IntImm>());
    CHECK(stride_w.as<IntImm>());
    CHECK(pad_left.as<IntImm>());
    CHECK(win_w.as<IntImm>());
    CHECK(d_kw.as<IntImm>());
    CHECK(w.as<IntImm>());

    CHECK(win_cut_w.as<IntImm>()->value * stride_w.as<IntImm>()->value >= pad_left.as<IntImm>()->value)
      << "Only one head for cut W axis";
    CHECK((((win_w.as<IntImm>()->value + win_cut_w.as<IntImm>()->value - 1) / win_cut_w.as<IntImm>()->value - 1) *
             win_cut_w.as<IntImm>()->value -
           1) *
              stride_w.as<IntImm>()->value +
            d_kw.as<IntImm>()->value <=
          w.as<IntImm>()->value + pad_left.as<IntImm>()->value)
      << "Only one tail for cut W axis";
  }
  w_base = infer_isolate_overlap(&w_info, &w_win_info, win_w, win_cut_w, stride_w, d_kw, w_head, w_tail);

  l1_reduce_base = 1;
  reduce_at_l1 = false;

  /* co / h / w */
  return co_base * h_base * w_base;
}

int ConvolutionForwardModel::infer_L0_tile(int isolate_idx) {
  /* m_l1 = oh * ow */
  Expr m_l1 = get_h_win_isolate_info(isolate_idx).inner * get_w_win_isolate_info(isolate_idx).inner;
  m_l1 = (m_l1 + conv_.block_size - 1) / conv_.block_size * conv_.block_size;

  /* k_l1 = kh * kw * cin */
  Expr k_l1 = conv_.filter.d_kh * conv_.filter.d_kw * get_ci_isolate_info(isolate_idx).inner;

  /* n_l1 = cout */
  Expr n_l1 = get_co_isolate_info(isolate_idx).inner;

  Expr cut_m = tile_.cut_m;
  if (!is_dynamic_) {
    CHECK(cut_m.as<IntImm>());
    CHECK(m_l1.as<IntImm>());
    if (cut_m.as<IntImm>()->value > m_l1.as<IntImm>()->value) {
      cut_m = m_l1;
    }
  }
  m_base = infer_isolate(&m_info, m_l1, cut_m);

  Expr cut_n = tile_.cut_n;
  if (!is_dynamic_) {
    CHECK(cut_n.as<IntImm>());
    CHECK(n_l1.as<IntImm>());
    if (cut_n.as<IntImm>()->value > n_l1.as<IntImm>()->value) {
      cut_n = n_l1;
    }
  }

  n_base = infer_isolate(&n_info, n_l1, cut_n);

  Expr cut_k = tile_.cut_k;
  if (!is_dynamic_) {
    CHECK(cut_k.as<IntImm>());
    CHECK(k_l1.as<IntImm>());
    if (cut_k.as<IntImm>()->value > k_l1.as<IntImm>()->value) {
      cut_k = k_l1;
    }
  }

  k_base = infer_isolate(&k_info, k_l1, cut_k);

  l0_reduce_base = k_base;

  return m_base * n_base * k_base;
}

int ConvolutionForwardModel::get_co_idx(int isolate_idx) const {
  CHECK_NE((h_base * w_base), 0);
  CHECK_NE(co_base, 0);
  return (isolate_idx / (h_base * w_base) % co_base);
}

int ConvolutionForwardModel::get_h_idx(int isolate_idx) const {
  CHECK_NE(w_base, 0);
  CHECK_NE(h_base, 0);
  return (isolate_idx / w_base % h_base);
}

int ConvolutionForwardModel::get_w_idx(int isolate_idx) const { return (isolate_idx % w_base); }

/* ConvolutionBackpropInputModel */
ConvolutionBackpropInputModel::ConvolutionBackpropInputModel(const Map<std::string, NodeRef> &attrs, bool is_dynamic)
    : ConvolutionModel(attrs, is_dynamic) {}

int ConvolutionBackpropInputModel::infer_L1_tile() {
  if (!is_dynamic_) {
    CHECK(conv_.output.c.as<IntImm>());
    CHECK(tile_.cut_co.as<IntImm>());
    int co = conv_.output.c.as<IntImm>()->value;
    int cut_co = tile_.cut_co.as<IntImm>()->value;
    if (cut_co > co) {
      cut_co = co;
    }
    co_base = infer_isolate(&co_info, co, cut_co);

    CHECK(conv_.input.h.as<IntImm>());
    CHECK(conv_.filter.d_kh.as<IntImm>());
    CHECK(conv_.stride.stride_h.as<IntImm>());
    CHECK(conv_.pad.pad_top.as<IntImm>());
    CHECK(conv_.pad.pad_bottom.as<IntImm>());
    CHECK(tile_.cut_h.as<IntImm>());
    int h = conv_.input.h.as<IntImm>()->value;
    int d_kh = conv_.filter.d_kh.as<IntImm>()->value;
    int stride_h = conv_.stride.stride_h.as<IntImm>()->value;
    int pad_top = conv_.pad.pad_top.as<IntImm>()->value;
    int pad_bottom = conv_.pad.pad_bottom.as<IntImm>()->value;
    int cut_h = tile_.cut_h.as<IntImm>()->value;
    int win_h = (h + pad_top + pad_bottom - d_kh) / stride_h + 1;
    int win_cut_h = (cut_h - d_kh) / stride_h + 1;
    if (win_cut_h > win_h) {
      win_cut_h = win_h;
      tile_.cut_h = (win_h - 1) * stride_h + d_kh;
    }
    int h_head = pad_top;
    int h_tail = ((win_h - 1) * stride_h + d_kh) - (h + pad_top);
    CHECK(win_cut_h * stride_h >= pad_top) << "Only one head for cut H axis";
    CHECK_NE(win_cut_h, 0);
    CHECK((((win_h + win_cut_h - 1) / win_cut_h - 1) * win_cut_h - 1) * stride_h + d_kh <= h + pad_top)
      << "Only one tail for cut H axis";
    h_base = infer_isolate_overlap(&h_info, &h_win_info, win_h, win_cut_h, stride_h, d_kh, h_head, h_tail);

    CHECK(conv_.input.w.as<IntImm>());
    CHECK(conv_.filter.d_kw.as<IntImm>());
    CHECK(conv_.stride.stride_w.as<IntImm>());
    CHECK(conv_.pad.pad_right.as<IntImm>());
    CHECK(tile_.cut_w.as<IntImm>());
    int w = conv_.input.w.as<IntImm>()->value;
    int d_kw = conv_.filter.d_kw.as<IntImm>()->value;
    int stride_w = conv_.stride.stride_w.as<IntImm>()->value;
    int pad_left = conv_.pad.pad_left.as<IntImm>()->value;
    int pad_right = conv_.pad.pad_right.as<IntImm>()->value;
    int cut_w = tile_.cut_w.as<IntImm>()->value;
    int win_w = (w + pad_left + pad_right - d_kw) / stride_w + 1;
    int win_cut_w = (cut_w - d_kw) / stride_w + 1;
    if (win_cut_w > win_w) {
      win_cut_w = win_w;
      tile_.cut_w = (win_w - 1) * stride_w + d_kw;
    }
    int w_head = pad_left;
    int w_tail = ((win_w - 1) * stride_w + d_kw) - (w + pad_left);
    CHECK(win_cut_w * stride_w >= pad_left) << "Only one head for cut W axis";
    CHECK_NE(win_cut_w, 0);
    CHECK((((win_w + win_cut_w - 1) / win_cut_w - 1) * win_cut_w - 1) * stride_w + d_kw <= w + pad_left)
      << "Only one tail for cut W axis";
    w_base = infer_isolate_overlap(&w_info, &w_win_info, win_w, win_cut_w, stride_w, d_kw, w_head, w_tail);

    l1_reduce_base = 1;
    reduce_at_l1 = false;
  }

  /* co / h / w */
  return co_base * h_base * w_base;
}

int ConvolutionBackpropInputModel::infer_L0_tile(int isolate_idx) {
  if (!is_dynamic_) {
    CHECK(get_h_win_isolate_info(isolate_idx).inner.as<IntImm>());
    CHECK(get_w_win_isolate_info(isolate_idx).inner.as<IntImm>());
    CHECK(conv_.block_size.as<IntImm>());
    /* m_l1 = oh * ow */
    int m_l1 = get_h_win_isolate_info(isolate_idx).inner.as<IntImm>()->value *
               get_w_win_isolate_info(isolate_idx).inner.as<IntImm>()->value;
    m_l1 = (m_l1 + conv_.block_size.as<IntImm>()->value - 1) / conv_.block_size.as<IntImm>()->value *
           conv_.block_size.as<IntImm>()->value;

    /* k_l1 = kh * kw * cin */
    CHECK(conv_.filter.d_kh.as<IntImm>());
    CHECK(conv_.filter.d_kw.as<IntImm>());
    CHECK(get_ci_isolate_info(isolate_idx).inner.as<IntImm>());
    int k_l1 = conv_.filter.d_kh.as<IntImm>()->value * conv_.filter.d_kw.as<IntImm>()->value *
               get_ci_isolate_info(isolate_idx).inner.as<IntImm>()->value;

    /* n_l1 = cout */
    CHECK(get_co_isolate_info(isolate_idx).inner.as<IntImm>());
    int n_l1 = get_co_isolate_info(isolate_idx).inner.as<IntImm>()->value;

    CHECK(tile_.cut_m.as<IntImm>());
    int cut_m = tile_.cut_m.as<IntImm>()->value;
    if (cut_m > m_l1) {
      cut_m = m_l1;
    }
    m_base = infer_isolate(&m_info, m_l1, cut_m);

    CHECK(tile_.cut_n.as<IntImm>());
    int cut_n = tile_.cut_n.as<IntImm>()->value;
    if (cut_n > n_l1) {
      cut_n = n_l1;
    }
    n_base = infer_isolate(&n_info, n_l1, cut_n);

    CHECK(tile_.cut_k.as<IntImm>());
    int cut_k = tile_.cut_k.as<IntImm>()->value;
    if (cut_k > k_l1) {
      cut_k = k_l1;
    }
    k_base = infer_isolate(&k_info, k_l1, cut_k);

    l0_reduce_base = k_base;
  }

  return m_base * n_base * k_base;
}

int ConvolutionBackpropInputModel::get_co_idx(int isolate_idx) const {
  CHECK_NE((h_base * w_base), 0);
  CHECK_NE(co_base, 0);
  return (isolate_idx / (h_base * w_base) % co_base);
}

int ConvolutionBackpropInputModel::get_h_idx(int isolate_idx) const {
  CHECK_NE(w_base, 0);
  CHECK_NE(h_base, 0);
  return (isolate_idx / w_base % h_base);
}

int ConvolutionBackpropInputModel::get_w_idx(int isolate_idx) const { return (isolate_idx % w_base); }

/* ConvolutionBackpropFilterModel */
ConvolutionBackpropFilterModel::ConvolutionBackpropFilterModel(const Map<std::string, NodeRef> &attrs, bool is_dynamic)
    : ConvolutionModel(attrs, is_dynamic) {
  if (!is_dynamic_) {
    CHECK(tile_.cut_b.as<IntImm>());
    CHECK_EQ(tile_.cut_b.as<IntImm>()->value, 1) << "Only support Batch Cut 1 now";
  }
}

int ConvolutionBackpropFilterModel::infer_L1_tile() {
  if (!is_dynamic_) {
    CHECK(conv_.input.n.as<IntImm>());
    CHECK(tile_.cut_b.as<IntImm>());
    int batch = conv_.input.n.as<IntImm>()->value;
    int cut_b = tile_.cut_b.as<IntImm>()->value;
    if (cut_b > batch) {
      cut_b = batch;
    }
    b_base = infer_isolate(&b_info, batch, cut_b);

    CHECK(conv_.input.c.as<IntImm>());
    CHECK(tile_.cut_ci.as<IntImm>());
    int ci = conv_.input.c.as<IntImm>()->value;
    int cut_ci = tile_.cut_ci.as<IntImm>()->value;
    if (cut_ci > ci) {
      cut_ci = ci;
    }
    ci_base = infer_isolate(&ci_info, ci, cut_ci);

    CHECK(conv_.output.c.as<IntImm>());
    CHECK(tile_.cut_co.as<IntImm>());
    int co = conv_.output.c.as<IntImm>()->value;
    int cut_co = tile_.cut_co.as<IntImm>()->value;
    if (cut_co > co) {
      cut_co = co;
    }
    co_base = infer_isolate(&co_info, co, cut_co);

    CHECK(conv_.filter.kh.as<IntImm>());
    CHECK(tile_.cut_kh.as<IntImm>());
    int kh = conv_.filter.kh.as<IntImm>()->value;
    int cut_kh = tile_.cut_kh.as<IntImm>()->value;
    if (cut_kh > kh) {
      cut_kh = kh;
    }
    kh_base = infer_isolate(&kh_info, kh, cut_kh);

    CHECK(conv_.filter.kw.as<IntImm>());
    CHECK(tile_.cut_kw.as<IntImm>());
    int kw = conv_.filter.kw.as<IntImm>()->value;
    int cut_kw = tile_.cut_kw.as<IntImm>()->value;
    if (cut_kw > kw) {
      cut_kw = kw;
    }
    kw_base = infer_isolate(&kw_info, kw, cut_kw);

    CHECK(conv_.input.h.as<IntImm>());
    CHECK(conv_.filter.d_kh.as<IntImm>());
    CHECK(conv_.stride.stride_h.as<IntImm>());
    CHECK(conv_.pad.pad_top.as<IntImm>());
    CHECK(conv_.pad.pad_bottom.as<IntImm>());
    CHECK(tile_.cut_h.as<IntImm>());
    int h = conv_.input.h.as<IntImm>()->value;
    int d_kh = conv_.filter.d_kh.as<IntImm>()->value;
    int stride_h = conv_.stride.stride_h.as<IntImm>()->value;
    int pad_top = conv_.pad.pad_top.as<IntImm>()->value;
    int pad_bottom = conv_.pad.pad_bottom.as<IntImm>()->value;
    int cut_h = tile_.cut_h.as<IntImm>()->value;
    int win_h = (h + pad_top + pad_bottom - d_kh) / stride_h + 1;
    int win_cut_h = (cut_h - d_kh) / stride_h + 1;
    if (win_cut_h > win_h) {
      win_cut_h = win_h;
      tile_.cut_h = (win_h - 1) * stride_h + d_kh;
    }
    int h_head = pad_top;
    int h_tail = ((win_h - 1) * stride_h + d_kh) - (h + pad_top);
    CHECK(win_cut_h * stride_h >= pad_top) << "Only one head for cut H axis";
    CHECK_NE(win_cut_h, 0);
    CHECK((((win_h + win_cut_h - 1) / win_cut_h - 1) * win_cut_h - 1) * stride_h + d_kh <= h + pad_top)
      << "Only one tail for cut H axis";
    h_base = infer_isolate_overlap(&h_info, &h_win_info, win_h, win_cut_h, stride_h, d_kh, h_head, h_tail);

    CHECK(conv_.input.w.as<IntImm>());
    CHECK(conv_.filter.d_kw.as<IntImm>());
    CHECK(conv_.stride.stride_w.as<IntImm>());
    CHECK(conv_.pad.pad_left.as<IntImm>());
    CHECK(conv_.pad.pad_right.as<IntImm>());
    CHECK(conv_.pad.pad_right.as<IntImm>());
    CHECK(tile_.cut_w.as<IntImm>());
    int w = conv_.input.w.as<IntImm>()->value;
    int d_kw = conv_.filter.d_kw.as<IntImm>()->value;
    int stride_w = conv_.stride.stride_w.as<IntImm>()->value;
    int pad_left = conv_.pad.pad_left.as<IntImm>()->value;
    int pad_right = conv_.pad.pad_right.as<IntImm>()->value;
    int cut_w = tile_.cut_w.as<IntImm>()->value;
    int win_w = (w + pad_left + pad_right - d_kw) / stride_w + 1;
    int win_cut_w = (cut_w - d_kw) / stride_w + 1;
    if (win_cut_w > win_w) {
      win_cut_w = win_w;
      tile_.cut_w = (win_w - 1) * stride_w + d_kw;
    }
    int w_head = pad_left;
    int w_tail = ((win_w - 1) * stride_w + d_kw) - (w + pad_left);
    CHECK(win_cut_w * stride_w >= pad_left) << "Only one head for cut W axis";
    CHECK_NE(win_cut_w, 0);
    CHECK((((win_w + win_cut_w - 1) / win_cut_w - 1) * win_cut_w - 1) * stride_w + d_kw <= w + pad_left)
      << "Only one tail for cut W axis";
    w_base = infer_isolate_overlap(&w_info, &w_win_info, win_w, win_cut_w, stride_w, d_kw, w_head, w_tail);

    l1_reduce_base = b_base * h_base * w_base;
    if (win_cut_w < win_w || win_cut_h < win_h || cut_b < batch) {
      reduce_at_l1 = true;
    }
  }

  /* ci / kh / kw / co / batch / h / w */
  return ci_base * kh_base * kw_base * co_base * b_base * h_base * w_base;
}

int ConvolutionBackpropFilterModel::infer_L0_tile(int isolate_idx) {
  if (!is_dynamic_) {
    /* m_l1 = cout */
    CHECK(get_co_isolate_info(isolate_idx).inner.as<IntImm>());
    int m_l1 = get_co_isolate_info(isolate_idx).inner.as<IntImm>()->value;

    /* n_l1 = kh * kw * cin */
    CHECK(get_kh_isolate_info(isolate_idx).inner.as<IntImm>());
    CHECK(get_kw_isolate_info(isolate_idx).inner.as<IntImm>());
    CHECK(get_ci_isolate_info(isolate_idx).inner.as<IntImm>());
    int n_l1 = get_kh_isolate_info(isolate_idx).inner.as<IntImm>()->value *
               get_kw_isolate_info(isolate_idx).inner.as<IntImm>()->value *
               get_ci_isolate_info(isolate_idx).inner.as<IntImm>()->value;

    /* k_l1 = batch * oh * ow */
    CHECK(get_b_isolate_info(isolate_idx).inner.as<IntImm>());
    CHECK(get_h_win_isolate_info(isolate_idx).inner.as<IntImm>());
    CHECK(get_w_win_isolate_info(isolate_idx).inner.as<IntImm>());
    int k_l1 = get_b_isolate_info(isolate_idx).inner.as<IntImm>()->value *
               get_h_win_isolate_info(isolate_idx).inner.as<IntImm>()->value *
               get_w_win_isolate_info(isolate_idx).inner.as<IntImm>()->value;
    CHECK(conv_.block_size.as<IntImm>());
    k_l1 = (k_l1 + conv_.block_size.as<IntImm>()->value - 1) / conv_.block_size.as<IntImm>()->value *
           conv_.block_size.as<IntImm>()->value;
    CHECK(tile_.cut_m.as<IntImm>());
    int cut_m = tile_.cut_m.as<IntImm>()->value;

    CHECK(tile_.cut_n.as<IntImm>());
    int cut_n = tile_.cut_n.as<IntImm>()->value;

    CHECK(tile_.cut_k.as<IntImm>());
    int cut_k = tile_.cut_k.as<IntImm>()->value;
    if (reduce_at_l1) {
      if (cut_m > m_l1) {
        cut_m = m_l1;
      }
      m_base = infer_isolate(&m_info, m_l1, cut_m);

      if (cut_n > n_l1) {
        cut_n = n_l1;
      }
      n_base = infer_isolate(&n_info, n_l1, cut_n);

      if (cut_k > k_l1) {
        cut_k = k_l1;
      }
      k_base = infer_isolate(&k_info, k_l1, cut_k);
    } else {
      if (cut_m > m_l1) {
        cut_m = m_l1;
      }
      m_base = infer_isolate(&m_info, m_l1, cut_m);

      if (cut_n > n_l1) {
        cut_n = n_l1;
      }
      n_base = infer_isolate(&n_info, n_l1, cut_n);

      if (cut_k > k_l1) {
        cut_k = k_l1;
      }
      k_base = infer_isolate(&k_info, k_l1, cut_k);
    }

    l0_reduce_base = k_base;
  }
  return m_base * n_base * k_base;
}

int ConvolutionBackpropFilterModel::get_ci_idx(int isolate_idx) const {
  int value = kh_base * kw_base * co_base * b_base * h_base * w_base;
  CHECK_NE(value, 0);
  CHECK_NE(ci_base, 0);
  return (isolate_idx / value % ci_base);
}

int ConvolutionBackpropFilterModel::get_kh_idx(int isolate_idx) const {
  int value = kw_base * co_base * b_base * h_base * w_base;
  CHECK_NE(value, 0);
  CHECK_NE(kh_base, 0);
  return (isolate_idx / value % kh_base);
}

int ConvolutionBackpropFilterModel::get_kw_idx(int isolate_idx) const {
  int value = co_base * b_base * h_base * w_base;
  CHECK_NE(value, 0);
  CHECK_NE(kw_base, 0);
  return (isolate_idx / value % kw_base);
}

int ConvolutionBackpropFilterModel::get_co_idx(int isolate_idx) const {
  int value = b_base * h_base * w_base;
  CHECK_NE(value, 0);
  CHECK_NE(co_base, 0);
  return (isolate_idx / value % co_base);
}

int ConvolutionBackpropFilterModel::get_b_idx(int isolate_idx) const {
  int value = h_base * w_base;
  CHECK_NE(value, 0);
  CHECK_NE(b_base, 0);
  return (isolate_idx / value % b_base);
}

int ConvolutionBackpropFilterModel::get_h_idx(int isolate_idx) const {
  CHECK_NE(w_base, 0);
  CHECK_NE(h_base, 0);
  return (isolate_idx / w_base % h_base);
}

int ConvolutionBackpropFilterModel::get_w_idx(int isolate_idx) const {
  CHECK_NE(w_base, 0);
  return (isolate_idx % w_base);
}
}  // namespace ir
}  // namespace akg
