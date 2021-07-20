# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""space generating functions for operators"""
from functools import partial
from typing import NamedTuple
from collections import namedtuple
from tests.common.test_run import matmul_run
from akg.utils import validation_check as vc_util
from tests.prev_version_auto_tune.type_definitions import ConvDesc, ConvBackpropDesc, MatmulCubeDesc, ConvConfig, ConvBackpropInputConfig, ConvBackpropFilterConfig, MatmulCubeConfig
from tests.prev_version_auto_tune.space import ListConfigSpace
from tests.prev_version_auto_tune.kernel_compiler import compile_kernel


def _get_space_vector(op_type: str, op_desc):
    """get config space of vector operator"""
    space_res, key, expect, input_for_mod = compile_kernel(op_type, op_desc, None, None, None, 0,
                                                           gen_tiling_spaces=True)

    if space_res is None:
        raise RuntimeError('no space returned')
    if 'index' not in space_res or 'tuning_space' not in space_res:
        raise RuntimeError('invalid space returned')
    index_table = space_res['index']
    tiling_spaces = space_res['tuning_space']

    if not tiling_spaces:
        raise RuntimeError('empty tiling spaces')

    dim_names = ['tiling_' + str(i) for i in range(len(tiling_spaces[0]))]
    input_type = namedtuple(op_type, dim_names)
    space = ListConfigSpace(input_type)
    for tiling_space in tiling_spaces:
        config = input_type(*tiling_space)
        space.add(config)
    return index_table, space, key, expect, input_for_mod


def _get_space_conv(op_desc: ConvDesc, tunning_attrs):
    """get config space of convolution"""
    if not isinstance(op_desc, ConvDesc):
        raise TypeError('op_desc must be ConvDesc')

    stride_ = op_desc.stride
    pad_ = op_desc.pad
    dilation_ = op_desc.dilation
    vc_util.convolution_format_check(op_desc.fmap_shape, op_desc.filter_shape, pad_, stride_, dilation_)
    config_space = ListConfigSpace(ConvConfig)

    # if double buff is not enabled, set it's value to 1
    size_scale = 1

    l1_max_size = (1024 * 1024) // size_scale
    l0a_max_size = (64 * 1024) // size_scale
    l0b_max_size = (64 * 1024) // size_scale
    l0c_max_size = ((256 - 8) * 1024) // size_scale // 2

    _, in_c, in_h, in_w = op_desc.fmap_shape
    k_n, _, k_h, k_w = op_desc.filter_shape
    padding = (pad_[0], pad_[1], pad_[2], pad_[3])
    p_top, p_bottom, p_left, p_right = padding
    s_h, s_w = stride_

    in_c = ((in_c - 1) // 16 + 1) * 16
    tile_c = in_c
    tile_co_start = 16

    data_len = 2

    h_max = in_h + p_top + p_bottom
    win_h = (h_max - k_h) // s_h + 1
    h_max = (h_max - k_h) // s_h * s_h + k_h
    w_max = in_w + p_left + p_right
    win_w = (w_max - k_w) // s_w + 1
    w_max = (w_max - k_w) // s_w * s_w + k_w

    bypass_options = [0, 1]

    for bypass in bypass_options:
        for tile_h in range(h_max, k_h - 1, -s_h):
            size_h = tile_h
            if tile_h == h_max:
                w_range = range(w_max, k_w - 1, -s_w)
                size_h = in_h
            else:
                w_range = [w_max]
                win_tile_h = (tile_h - k_h) // s_h + 1
                h_tiles = (win_h + win_tile_h - 1) // win_tile_h
                if h_tiles == 2:
                    size_h = max(tile_h - p_top, in_h + p_top - tile_h + k_h - s_h)

            for tile_w in w_range:
                size_w = tile_w
                if size_w == w_max:
                    size_w = in_w
                else:
                    win_tile_w = (tile_w - k_w) // s_w + 1
                    w_tiles = (win_w + win_tile_w - 1) // win_tile_w
                    if w_tiles == 2:
                        size_w = max(tile_w - p_left, in_w + p_left - tile_w + k_w - s_w)

                k_n_ = ((k_n - 1) // 16 + 1) * 16
                co_range = range(k_n_, tile_co_start - 1, -16)
                for tile_co in co_range:
                    if bypass == 1:
                        if tile_co != k_n:
                            continue
                        l1_size = data_len * (size_h * size_w * in_c)
                    else:
                        l1_size = data_len * (size_h * size_w * in_c +
                                              tile_co * tile_c * k_h * k_w)

                    if l1_size > l1_max_size:
                        continue

                    tile_co_ = ((tile_co - 1) // 16 + 1) * 16
                    for tile_n in range(tile_co_, 15, -16):
                        k_max = in_c * k_h * k_w
                        k_max_ = ((k_max - 1) // 16 + 1) * 16
                        k_size = l0b_max_size // data_len // tile_n
                        k_size_ = k_size // 16 * 16
                        for tile_k in range(min(k_max_, k_size_), 15, -16):
                            m_max = (int(((tile_h - k_h) // (s_h)) + 1)) * (int(((tile_w - k_w) // (s_w)) + 1))
                            m_max_ = ((m_max - 1) // 16 + 1) * 16
                            m_size1 = l0a_max_size // data_len // tile_k
                            m_size1_ = m_size1 // 16 * 16
                            m_size2 = l0c_max_size // data_len // tile_n
                            m_size2_ = m_size2 // 16 * 16
                            for tile_m in range(min(m_max_, m_size1_, m_size2_), 15, -16):
                                config_space.add(ConvConfig(tile_h, tile_co, tile_m, tile_k,
                                                            tile_n, tile_w, bypass))

    return None, config_space, op_desc.__str__(), None, None


def _get_space_conv_bn1(op_desc: ConvDesc, tunning_attrs):
    """get config space of convolution"""
    if not isinstance(op_desc, ConvDesc):
        raise TypeError('op_desc must be ConvDesc')

    stride_ = op_desc.stride
    pad_ = op_desc.pad
    dilation_ = op_desc.dilation
    vc_util.convolution_format_check(op_desc.fmap_shape, op_desc.filter_shape, pad_, stride_, dilation_)
    config_space = ListConfigSpace(ConvConfig)

    # if double buff is not enabled, set it's value to 1
    size_scale = 1

    l1_max_size = (1024 * 1024) // size_scale
    l0a_max_size = (64 * 1024) // size_scale
    l0b_max_size = (64 * 1024) // size_scale
    l0c_max_size = ((256 - 8) * 1024) // size_scale // 2 // 4

    _, in_c, in_h, in_w = op_desc.fmap_shape
    k_n, _, k_h, k_w = op_desc.filter_shape
    padding = (pad_[0], pad_[1], pad_[2], pad_[3])
    p_top, p_bottom, p_left, p_right = padding
    s_h, s_w = stride_

    in_c = ((in_c - 1) // 16 + 1) * 16
    tile_c = in_c
    tile_co_start = 16

    data_len = 2

    h_max = in_h + p_top + p_bottom
    win_h = (h_max - k_h) // s_h + 1
    h_max = (h_max - k_h) // s_h * s_h + k_h
    w_max = in_w + p_left + p_right
    win_w = (w_max - k_w) // s_w + 1
    w_max = (w_max - k_w) // s_w * s_w + k_w

    bypass_options = [0, 1]

    for bypass in bypass_options:
        h_range = range(h_max, k_h - 1, -s_h)
        for tile_h in h_range:
            size_h = tile_h
            if tile_h == h_max:
                w_range = range(w_max, k_w - 1, -s_w)
                size_h = in_h
            else:
                w_range = [w_max]
                win_tile_h = (tile_h - k_h) // s_h + 1
                h_tiles = (win_h + win_tile_h - 1) // win_tile_h
                if h_tiles == 2:
                    size_h = max(tile_h - p_top, in_h + p_top - tile_h + k_h - s_h)

            for tile_w in w_range:
                size_w = tile_w
                if size_w == w_max:
                    size_w = in_w
                else:
                    win_tile_w = (tile_w - k_w) // s_w + 1
                    w_tiles = (win_w + win_tile_w - 1) // win_tile_w
                    if w_tiles == 2:
                        size_w = max(tile_w - p_left, in_w + p_left - tile_w + k_w - s_w)

                k_n_ = ((k_n - 1) // 16 + 1) * 16
                co_range = range(k_n_, tile_co_start - 1, -16)
                for tile_co in co_range:
                    if bypass == 1:
                        if tile_co != k_n:
                            continue
                        l1_size = data_len * (size_h * size_w * in_c)
                    else:
                        l1_size = data_len * (size_h * size_w * in_c +
                                              tile_co * tile_c * k_h * k_w)

                    if l1_size > l1_max_size:
                        continue

                    tile_co_ = ((tile_co - 1) // 16 + 1) * 16
                    for tile_n in range(tile_co_, 15, -16):
                        k_max = in_c * k_h * k_w
                        k_max_ = ((k_max - 1) // 16 + 1) * 16
                        k_size = l0b_max_size // data_len // tile_n
                        k_size_ = k_size // 16 * 16
                        for tile_k in range(min(k_max_, k_size_), 15, -16):
                            m_max = (int(((tile_h - k_h) // (s_h)) + 1)) * (int(((tile_w - k_w) // (s_w)) + 1))
                            m_max_ = ((m_max - 1) // 16 + 1) * 16
                            m_size1 = l0a_max_size // data_len // tile_k
                            m_size1_ = m_size1 // 16 * 16
                            m_size2 = l0c_max_size // data_len // tile_n
                            m_size2_ = m_size2 // 16 * 16
                            for tile_m in range(min(m_max_, m_size1_, m_size2_), 15, -16):
                                config_space.add(ConvConfig(tile_h, tile_co, tile_m, tile_k,
                                                            tile_n, tile_w, bypass))

    return None, config_space, op_desc.__str__(), None, None


def _get_space_conv_backprop_input(op_desc: ConvBackpropDesc, tunning_attrs):
    """get config space of convolution backprop input"""
    if not isinstance(op_desc, ConvBackpropDesc):
        raise TypeError('op_desc must be ConvDesc')

    stride_ = op_desc.stride
    pad_ = op_desc.pad
    dilation_ = op_desc.dilation
    vc_util.convolution_format_check(op_desc.fmap_shape, op_desc.filter_shape, pad_, stride_, dilation_)
    config_space = ListConfigSpace(ConvBackpropInputConfig)

    # if double buff is not enabled, set it's value to 1
    size_scale = 1
    block_size = 16

    l1_max_size = (1024 * 1024) // size_scale
    l0a_max_size = (64 * 1024) // size_scale
    l0b_max_size = (64 * 1024) // size_scale
    l0c_max_size = ((256 - 8) * 1024) // size_scale // 2
    ub_max_size = l0c_max_size

    _, in_c, in_h, in_w = op_desc.fmap_shape
    k_n, _, k_h, k_w = op_desc.filter_shape

    in_c = (in_c + block_size - 1) // block_size * block_size
    k_n = (k_n + block_size - 1) // block_size * block_size

    pad_top, pad_bottom, pad_left, pad_right = pad_
    stride_h, stride_w = stride_

    out_c = k_n
    out_h = (in_h + pad_top + pad_bottom - k_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - k_w) // stride_w + 1

    out_h = out_h * stride_h
    out_w = out_w * stride_w

    p_top = k_h - pad_[0] - 1
    p_bottom = in_h + pad_[0] - stride_[0] * ((in_h + pad_[0] + pad_[1] - k_h) // stride_[0] + 1)
    p_left = k_w - pad_[2] - 1
    p_right = in_w + pad_[2] - stride_[1] * ((in_w + pad_[2] + pad_[3] - k_w) // stride_[1] + 1)

    s_h = 1
    s_w = 1

    tile_c = out_c
    tile_co_start = 16

    data_len = 2

    h_max = out_h + p_top + p_bottom
    win_h = (h_max - k_h) // s_h + 1
    h_max = (h_max - k_h) // s_h * s_h + k_h
    w_max = out_w + p_left + p_right
    win_w = (w_max - k_w) // s_w + 1
    w_max = (w_max - k_w) // s_w * s_w + k_w

    for tile_h in range(h_max, k_h - 1, -s_h):
        size_h = tile_h
        if tile_h == h_max:
            w_range = range(w_max, k_w - 1, -s_w)
            size_h = in_h
        else:
            w_range = [w_max]
            win_tile_h = (tile_h - k_h) // s_h + 1
            h_tiles = (win_h + win_tile_h - 1) // win_tile_h
            if h_tiles == 2:
                size_h = max(tile_h - p_top, in_h + p_top - tile_h + k_h - s_h)

        for tile_w in w_range:
            size_w = tile_w
            if size_w == w_max:
                size_w = in_w
            else:
                win_tile_w = (tile_w - k_w) // s_w + 1
                w_tiles = (win_w + win_tile_w - 1) // win_tile_w
                if w_tiles == 2:
                    size_w = max(tile_w - p_left, in_w + p_left - tile_w + k_w - s_w)

            k_n_ = ((k_n - 1) // 16 + 1) * 16
            co_range = range(k_n_, tile_co_start - 1, -16)
            for tile_co in co_range:
                l1_size = data_len * (size_h * size_w * out_c +
                                      tile_co * tile_c * k_h * k_w)
                if l1_size > l1_max_size:
                    continue
                ub_size = data_len * (size_h * size_w * out_c)
                if ub_size > ub_max_size:
                    continue

                tile_co_ = ((tile_co - 1) // 16 + 1) * 16
                for tile_n in range(tile_co_, 15, -16):
                    k_max = out_c * k_h * k_w
                    k_base = 16 * k_h * k_w
                    k_max_ = ((k_max - 1) // k_base + 1) * k_base
                    k_size = l0b_max_size // data_len // tile_n
                    k_size_ = k_size // k_base * k_base
                    for tile_k in range(min(k_max_, k_size_), k_base - 1, -k_base):
                        m_max = (int(((tile_h - k_h) // (s_h)) + 1)) * (int(((tile_w - k_w) // (s_w)) + 1))
                        m_max_ = ((m_max - 1) // 16 + 1) * 16
                        m_size1 = l0a_max_size // data_len // tile_k
                        m_size1_ = m_size1 // 16 * 16
                        m_size2 = l0c_max_size // data_len // tile_n
                        m_size2_ = m_size2 // 16 * 16
                        for tile_m in range(min(m_max_, m_size1_, m_size2_), 15, -16):
                            config_space.add(ConvBackpropInputConfig(tile_h, tile_co, tile_m,
                                                                     tile_k, tile_n, tile_w))
    return None, config_space, op_desc.__str__(), None, None


def _get_space_conv_backprop_filter(op_desc: ConvBackpropDesc, tunning_attrs):
    """get config space of convolution backwprop filter"""
    if not isinstance(op_desc, ConvBackpropDesc):
        raise TypeError('op_desc must be ConvBackpropDesc')

    stride_ = op_desc.stride
    pad_ = op_desc.pad
    dilation_ = op_desc.dilation
    vc_util.convolution_format_check(op_desc.fmap_shape, op_desc.filter_shape, pad_, stride_, dilation_)
    config_space = ListConfigSpace(ConvBackpropFilterConfig)

    # if double buff is not enabled, set it's value to 1
    size_scale = 1
    block_size = 16

    l1_max_size = (1024 * 1024) // size_scale
    l0a_max_size = (64 * 1024) // size_scale
    l0b_max_size = (64 * 1024) // size_scale
    l0c_max_size = ((256 - 8) * 1024) // size_scale // 2

    in_n, in_c, in_h, in_w = op_desc.fmap_shape
    cout, _, k_h, k_w = op_desc.filter_shape
    k_n = cout

    in_c = (in_c + block_size - 1) // block_size * block_size
    cout = (cout + block_size - 1) // block_size * block_size

    pad_top, pad_bottom, pad_left, pad_right = pad_
    s_h, s_w = stride_
    tile_co_start = 16
    tile_ci_start = 16
    data_len = 2
    h_max = in_h + pad_top + pad_bottom
    win_h = (h_max - k_h) // s_h + 1
    h_max = (h_max - k_h) // s_h * s_h + k_h
    w_max = in_w + pad_left + pad_right
    win_w = (w_max - k_w) // s_w + 1
    w_max = (w_max - k_w) // s_w * s_w + k_w

    for tile_h in range(h_max, k_h - 1, -s_h):
        size_h = tile_h
        win_tile_h = (tile_h - k_h) // s_h + 1
        # Only one head for cut H axis
        if win_tile_h * s_h < pad_top:
            continue
        # Only one tail for cut H axis
        if (((win_h + win_tile_h - 1) // win_tile_h - 1) * win_tile_h - 1) * s_h + k_h > in_h + pad_top:
            continue
        if tile_h == h_max:
            w_range = range(w_max, k_w - 1, -s_w)
            size_h = in_h
        else:
            w_range = [w_max]
            h_tiles = (win_h + win_tile_h - 1) // win_tile_h
            if h_tiles == 2:
                size_h = max(tile_h - pad_top, in_h + pad_top - tile_h + k_h - s_h)

        for tile_w in w_range:
            size_w = tile_w
            win_tile_w = (tile_w - k_w) // s_w + 1
            # Only one head for cut W axis
            if win_tile_w * s_w < pad_left:
                continue
            # Only one tail for cut W axis
            if (((win_w + win_tile_w - 1) // win_tile_w - 1) * win_tile_w - 1) * s_w + k_w > in_w + pad_left:
                continue
            if size_w == w_max:
                size_w = in_w
            else:
                w_tiles = (win_w + win_tile_w - 1) // win_tile_w
                if w_tiles == 2:
                    size_w = max(tile_w - pad_left, in_w + pad_left - tile_w + k_w - s_w)
            for tile_kh in range(k_h, 0, -1):
                for tile_kw in range(k_w, 0, -1):
                    k_n_ = ((k_n - 1) // 16 + 1) * 16
                    co_range = range(k_n_, tile_co_start - 1, -16)
                    for tile_co in co_range:
                        in_c_ = ((in_c - 1) // 16 + 1) * 16
                        ci_range = range(in_c_, tile_ci_start - 1, -16)
                        for tile_ci in ci_range:
                            tile_batch = 1
                            l1_size = data_len * tile_batch * (tile_co * win_tile_h * win_tile_w +
                                                               tile_ci * size_h * size_w)
                            if l1_size > l1_max_size:
                                continue

                            if (tile_batch != in_n or tile_co != k_n_ or tile_ci != in_c_):
                                tile_m = tile_co
                                tile_n = tile_ci * tile_kh * tile_kw
                                l0c_size = data_len * tile_n * tile_m
                                if l0c_size > l0c_max_size:
                                    continue
                                k_max = tile_batch * tile_h * tile_w
                                k_max_ = ((k_max - 1) // 16 + 1) * 16
                                k_size1 = l0a_max_size // data_len // tile_m
                                k_size1_ = k_size1 // 16 * 16
                                k_size2 = l0b_max_size // data_len // tile_n
                                k_size2_ = k_size2 // 16 * 16
                                for tile_k in range(min(k_max_, k_size1_, k_size2_), 15, -16):
                                    config_space.add(ConvBackpropFilterConfig(tile_ci, tile_kh, tile_kw, tile_co,
                                                                              tile_batch, tile_h, tile_w, tile_m,
                                                                              tile_k, tile_n))
                            else:
                                for tile_n in range(tile_ci * tile_kh * tile_kw, 15, -16):
                                    k_max = tile_batch * tile_h * tile_w
                                    k_max_ = ((k_max - 1) // 16 + 1) * 16
                                    k_size = l0b_max_size // data_len // tile_n
                                    k_size_ = k_size // 16 * 16
                                    for tile_k in range(min(k_max_, k_size_), 15, -16):
                                        m_max = tile_co
                                        m_max_ = ((m_max - 1) // 16 + 1) * 16
                                        m_size1 = l0a_max_size // data_len // tile_k
                                        m_size1_ = m_size1 // 16 * 16
                                        m_size2 = l0c_max_size // data_len // tile_n
                                        m_size2_ = m_size2 // 16 * 16
                                        for tile_m in range(min(m_max_, m_size1_, m_size2_), 15, -16):
                                            config_space.add(ConvBackpropFilterConfig(tile_ci, tile_kh, tile_kw,
                                                                                      tile_co, tile_batch, tile_h,
                                                                                      tile_w, tile_m, tile_k, tile_n))
    return None, config_space, op_desc.__str__(), None, None

def gen_bool_list(attr_list):
    bool_list = []
    for _ in attr_list:
        if len(bool_list) == 0:
            bool_list = [[True], [False]]
        else:
            tmp_list = []
            for attr_option in bool_list:
                tmp = attr_option[:]
                tmp.append(True)
                tmp1 = tmp[:]
                tmp.pop()
                tmp.append(False)
                tmp2 = tmp[:]
                tmp_list.append(tmp1)
                tmp_list.append(tmp2)
            bool_list = tmp_list
    return bool_list

def _get_space_matmul_cube(op_desc: MatmulCubeDesc, tuning_attrs):
    """get config space of matmul_cube"""
    if not isinstance(op_desc, MatmulCubeDesc):
        raise TypeError('op_desc must be MatmulCubeDesc')
    config_attrs =  ['n_l1', 'n_l0', 'm_l1', 'm_l0', 'k_l1', 'k_l0', 'bypass']
    config_attrs.extend(tuning_attrs)
    MatmulCubeConfig = namedtuple('MatmulCubeConfig', config_attrs) 
    config_space = ListConfigSpace(MatmulCubeConfig)
    batch_tuple, m, k, n = matmul_run.extract_dim(op_desc.x_shape, op_desc.y_shape, op_desc.adj_x, op_desc.adj_y)

    mmax = (m + 15) // 16
    nmax = (n + 15) // 16
    kmax = (k + 15) // 16

    double_buffer = True
    mad_fp32 = True

    l1_max_size  = (1024 * 1024)      # L1  MEM 1024KB
    l0a_max_size = (64 * 1024)        # L0A MEM 64KB
    l0b_max_size = (64 * 1024)        # L0B MEM 64KB
    l0c_max_size = (256 * 1024)       # L0C MEM 256KB
    ub_max_size  = ((256 - 8) * 1024) # UB  MEM 248KB, 8KB reserved for compiler

    if double_buffer:
        l1_max_size = l1_max_size // 2
        l0a_max_size = l0a_max_size // 2
        l0b_max_size = l0b_max_size // 2
        l0c_max_size = l0c_max_size // 2
        ub_max_size = ub_max_size // 2

    if mad_fp32:
        l0c_max_size = l0c_max_size // 2
    if op_desc.out_dtype == 'float32':
        ub_max_size = ub_max_size // 2

    bypass_options = [0, 1, 2]

    for bypass in bypass_options:
        if (bypass == 2) and ((op_desc.adj_x == False and op_desc.left_format[0].lower() == 'n') or
                              (op_desc.adj_x == True and op_desc.left_format[0].lower() == 'z')):
            continue

        if (bypass == 1) and ((op_desc.adj_y == False and op_desc.right_format[0].lower() == 'z') or
                              (op_desc.adj_y == True and op_desc.right_format[0].lower() == 'n')):
            continue

        for k_l1 in range(1, kmax + 1):
            if kmax % k_l1 != 0:
                continue
            for k_l0 in range(1, k_l1 + 1):
                if k_l1 % k_l0 != 0:
                    continue

                # no need to cut from l1 to l0 for m and n when k is cut
                for m_l1 in range(1, mmax + 1):
                    if mmax % m_l1 != 0:
                        continue
                    m_l0_range = [m_l1] if k_l1 != kmax else range(1, m_l1 + 1)
                    for m_l0 in m_l0_range:
                        if m_l1 % m_l0 != 0:
                            continue
                        for n_l1 in range(1, nmax + 1):
                            if nmax % n_l1 != 0:
                                continue
                            n_l0_range = [n_l1] if k_l1 != kmax else range(1, n_l1 + 1)
                            for n_l0 in n_l0_range:
                                if n_l1 % n_l0 != 0:
                                    continue

                                if m_l0 * 16 * k_l0 * 16 > l0a_max_size:
                                    continue

                                if n_l0 * 16 * k_l0 * 16 > l0b_max_size:
                                    continue

                                if m_l0 * 16 * n_l0 * 16 > l0c_max_size:
                                    continue

                                if m_l0 * 16 * n_l0 * 16 > ub_max_size:
                                    continue

                                if bypass == 2:
                                    l1_size = n_l1 * 16 * k_l1 * 16
                                elif bypass == 1:
                                    l1_size = m_l1 * 16 * k_l1 * 16
                                else:
                                    l1_size = (m_l1 * 16 + n_l1 * 16) * k_l1 * 16
                                if l1_size > l1_max_size:
                                    continue

                                if nmax == 1:
                                    n_l1 = 0
                                    n_l0 = 0
                                if mmax == 1:
                                    m_l1 = 0
                                    m_l0 = 0
                                if kmax == 1:
                                    k_l1 = 16
                                    k_l0 = 16
                                tiling_space = [n_l1, n_l0, m_l1, m_l0, k_l1, k_l0, bypass]
                                if len(tuning_attrs) == 0:
                                    config_space.add(MatmulCubeConfig(*tiling_space))
                                else:
                                    attr_options = gen_bool_list(tuning_attrs)
                                    for attr_option in attr_options:
                                        tmp = tiling_space[:]
                                        tmp.extend(attr_option)
                                        config = MatmulCubeConfig(*tmp)
                                        config_space.add(config)
    shape_xx, shape_yy, _, _, k = matmul_run.get_converted_shapes(m, n, k, batch_tuple, op_desc.adj_x, op_desc.adj_y,
                                                                  op_desc.bias, op_desc.left_format,
                                                                  op_desc.right_format, op_desc.out_format)
    return None, config_space, str((shape_xx, shape_yy, op_desc.bias, op_desc.left_format, op_desc.right_format,
                                    op_desc.out_format, op_desc.adj_x, op_desc.adj_y, op_desc.dtype,
                                    op_desc.out_dtype)), None, None


_get_space_func = {
    'conv': _get_space_conv,
    'conv_bn1': _get_space_conv_bn1,
    'conv_backprop_input': _get_space_conv_backprop_input,
    'conv_backprop_filter': _get_space_conv_backprop_filter,
    'matmul': _get_space_matmul_cube,
}


def get_space(op_type: str, op_desc: NamedTuple, tuning_attrs=[]):
    """get space of an operator"""
    func = _get_space_func.get(op_type, None)
    if func is None:
        func = partial(_get_space_vector, op_type=op_type)
    return func(op_desc=op_desc, tuning_attrs=tuning_attrs)
