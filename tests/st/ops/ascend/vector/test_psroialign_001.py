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

import akg
import akg.lang.cce
from akg.utils import kernel_exec as utils


def do_align(in_num, aligned_size):
    return ((in_num + aligned_size - 1) // aligned_size) * aligned_size


def psroialign_compute(fm_shape, roi_shape, class_num, group_size, sample_h, sample_w, scale):
    '''

    :param fm_shape:   (n, c_dim, h, w) where: c_dim = group_size * group_size * (class_num + 1)
    :param roi_shape:  (roi_num, 16, 1, 1). there are 5 value on dim C: score, x1, y1, x2, y2. The other 11 num is pads
    :param class_num:
    :param group_size:
    :param sample_h:
    :param sample_w:
    :param scale:
    :return:

    '''

    dtype = "float16"

    fm_data = akg.tvm.placeholder(fm_shape, name="fm_data", dtype=dtype)
    roi_data = akg.tvm.placeholder(roi_shape, name="roi_data", dtype=dtype)
    scale_const = akg.tvm.const(scale, dtype=dtype)

    sample_h_const = akg.tvm.const(sample_h, "int32")
    sample_w_const = akg.tvm.const(sample_w, "int32")
    two_const = akg.tvm.const(2, "float16")
    one_const = akg.tvm.const(1, "float16")
    group_size_const = akg.tvm.const(group_size, "int32")

    bin_num = group_size * group_size

    # ==============================================================
    # step 1: scale coordinates size in original image to size in feature map
    # ==============================================================

    COSIZE = 16
    roi_num = roi_shape[0]
    aligned_roi_num = do_align(roi_num, COSIZE)

    # 4 means x1, y1, x2, y2
    # roi_shape[0] must be equal to COSIZE
    scaled_coors = akg.tvm.compute((4, aligned_roi_num, 1, 1), lambda n, c, h, w: roi_data[c, 1 + n, h, w] * scale_const,
                               name='scaled_coors')

    # ==============================================================
    # step 2: compute the width and height of roi
    # ==============================================================

    # 2 stands for width and height
    width_height_shape = (2, aligned_roi_num, 1, 1)
    width_height_of_rois = akg.tvm.compute(width_height_shape, lambda n, c, h, w: scaled_coors[n + 2, c, h, w]
                                       - scaled_coors[n, c, h, w], name='width_height_of_rois')

    width_shape = (aligned_roi_num,)
    width_of_rois = akg.tvm.compute(width_shape, lambda n: scaled_coors[2, n, 0, 0]
                                - scaled_coors[0, n, 0, 0], name='width_of_rois')
    width_shape = (aligned_roi_num,)
    height_of_rois = akg.tvm.compute(width_shape, lambda n: scaled_coors[1, n, 0, 0]
                                 - scaled_coors[3, n, 0, 0], name='height_of_rois')

    # ==============================================================
    # step 3: compute the bias of the coordinates of all samples
    # ==============================================================

    # samples_shape = (aligned_roi_num, bin_num, sample_h, sample_w)

    # unit_nums = akg.tvm.compute((2,), lambda i: two_const * group_size_const \
    #                                         * akg.tvm.expr.Select(i == 0, sample_w_const, sample_h_const), name = 'uint_nums')

    # width_height_shape(0, x, x, x) indicates the width of a single unit which is separated by samples
    # and width_height_shape(1, x, x, x) the height
    # unit_lengths = akg.tvm.compute(width_height_shape, lambda n, c, h, w: width_height_of_rois(n, c, h, w) / unit_nums(n), \
    #                            name = 'uint_lengths')

    unit_w_lengths = akg.tvm.compute(width_shape, lambda n: width_of_rois(n) / sample_w_const * group_size_const,
                                 name='uint_w_lengths')
    unit_h_lengths = akg.tvm.compute(width_shape, lambda n: height_of_rois(n) / sample_h_const * group_size_const,
                                 name='uint_h_lengths')

    # samples_coors_x_shape = (aligned_roi_num, 1, group_size * sample_h, group_size * sample_w)
    # samples_x_coors_bias = akg.tvm.compute(samples_coors_x_shape, lambda n, c, h, w: unit_w_lengths[n] * \
    #                                         (one_const + w * two_const), name = 'samples_x_coors_bias')
    #
    # samples_y_coors_bias = akg.tvm.compute(samples_coors_x_shape, lambda n, c, h, w: unit_h_lengths[n] * \
    #                                         (one_const + w * two_const), name = 'samples_y_coors_bias')
    #
    # samples_x_coors = akg.tvm.compute(samples_coors_x_shape, lambda n, c, h, w: \
    #     samples_x_coors_bias(n, c, h, w) + scaled_coors(1, c, 1, 1), name = 'samples_x_coors')
    # samples_y_coors = akg.tvm.compute(samples_coors_x_shape, lambda n, c, h, w: \
    #     samples_y_coors_bias(n, c, h, w) + scaled_coors(2, c, 1, 1), name = 'samples_y_coors')

    sample_w_bias_shape = (1, group_size, sample_w, aligned_roi_num)
    # sample_w_bias = akg.tvm.compute(sample_w_bias_shape, lambda n, c, h, w: unit_w_lengths[w] * \
    #                                 (one_const + two_const * (c * sample_w_const + h)), name = 'samples_w_bias')
    # sample_w_bias = akg.tvm.compute(sample_w_bias_shape, lambda n, c, h, w: unit_w_lengths[w] * \
    #                                   (one_const + two_const * (sample_w_const)), name = 'samples_w_bias')

    sample_h_bias_shape = (1, group_size, sample_h, aligned_roi_num)
    # sample_h_bias = akg.tvm.compute(sample_h_bias_shape, lambda n, c, h, w: unit_h_lengths[w] * \
    #                                 (one_const + two_const * (c * sample_h_const + h)), name = 'samples_h_bias')
    # sample_h_bias = akg.tvm.compute(sample_h_bias_shape, lambda n, c, h, w: unit_h_lengths[w] * \
    #                                   (one_const + two_const * (sample_h_const)), name = 'samples_h_bias')

    @akg.tvm.hybrid.script(capture=locals())
    def gen_bias(h_value, unit_lengths, ratio):
        output = output_tensor((1, group_size, h_value, aligned_roi_num), 'float16')

        strides = allocate((aligned_roi_num, ), 'float16', 'local')
        for w in range(0, aligned_roi_num):
            strides[w] = half(0.0)

        for c in range(0, group_size):
            for h in range(0, 1):
                for w in range(0, aligned_roi_num):
                    output[0, c, h, w] = unit_lengths[w]
                    # strides[w] += unit_lengths[w] * ratio * half(h_value)

            for h in range(1, h_value):
                for w in range(0, aligned_roi_num):
                    output[0, c, h, w] = output[0, c, h - 1, w] + ratio * unit_lengths[w]

        return output

    sample_w_bias = gen_bias(sample_w_const, unit_w_lengths, two_const)
    sample_h_bias = gen_bias(sample_h_const, unit_h_lengths, two_const)

    samples_x_coors = akg.tvm.compute(sample_w_bias_shape, lambda n, c, h, w: sample_w_bias(n, c, h, w) +
                                  scaled_coors(0, w, 0, 0), name='samples_x_coors')

    samples_y_coors = akg.tvm.compute(sample_h_bias_shape, lambda n, c, h, w: sample_h_bias(n, c, h, w) +
                                  scaled_coors(1, w, 0, 0), name='samples_y_coors')

    # ==============================================================
    # step 4: compute the low and high coordinates of samples for bilinear
    # ==============================================================
    # samples_x_coors_low = akg.tvm.compute(sample_w_bias_shape, lambda *indices: \
    #     akg.lang.cce.floor(samples_x_coors(*indices)), name = 'samples_x_coors_low')
    # samples_x_coors_high = akg.tvm.compute(sample_w_bias_shape, lambda *indices: \
    #     akg.lang.cce.ceil(samples_x_coors(*indices)), name = 'samples_x_coors_high')
    # samples_y_coors_low = akg.tvm.compute(sample_h_bias_shape, lambda *indices: \
    #     akg.lang.cce.floor(samples_y_coors(*indices)), name = 'samples_y_coors_low')
    # samples_y_coors_high = akg.tvm.compute(sample_h_bias_shape, lambda *indices: \
    #     akg.lang.cce.ceil(samples_y_coors(*indices)), name = 'samples_y_coors_high')
    samples_x_coors_low = akg.lang.cce.floor(samples_x_coors)
    samples_x_coors_high = akg.lang.cce.ceil(samples_x_coors)
    samples_y_coors_low = akg.lang.cce.floor(samples_y_coors)
    samples_y_coors_high = akg.lang.cce.ceil(samples_y_coors)

    # samples_x_coors_low = akg.tvm.compute(sample_w_bias_shape, lambda *indices: \
    #     akg.topi.cast(samples_x_coors(*indices), 'int32'), name = 'samples_x_coors_low')
    # samples_x_coors_high = akg.tvm.compute(sample_w_bias_shape, lambda *indices: \
    #     samples_x_coors_low(*indices) + akg.topi.cast(one_const, 'int32'), name = 'samples_x_coors_high')
    # samples_y_coors_low = akg.tvm.compute(sample_h_bias_shape, lambda *indices: \
    #     akg.topi.cast(samples_y_coors(*indices), 'int32'), name = 'samples_y_coors_low')
    # samples_y_coors_high = akg.tvm.compute(sample_h_bias_shape, lambda *indices: \
    #     samples_y_coors_low(*indices) + akg.topi.cast(one_const, 'int32'), name = 'samples_y_coors_high')

    # ==============================================================
    # step 5: compute the weight of low and high coordinates for bilinear
    # ==============================================================
    # wlx = akg.tvm.compute(samples_coors_x_shape, lambda *indices: samples_x_coors_high(*indices) - samples_x_coors(*indices))
    # whx = akg.tvm.compute(samples_coors_x_shape, lambda *indices: one_const - wlx(*indices))
    #
    # wly = akg.tvm.compute(samples_coors_x_shape, lambda *indices: samples_y_coors_high(*indices) - samples_y_coors(*indices))
    # why = akg.tvm.compute(samples_coors_x_shape, lambda *indices: one_const - wly(*indices))
    #
    # wlxXwly = akg.tvm.compute(samples_coors_x_shape, lambda *indices: wlx(*indices) * wly(*indices))
    # whxXwly = akg.tvm.compute(samples_coors_x_shape, lambda *indices: whx(*indices) * wly(*indices))
    # wlxXwhy = akg.tvm.compute(samples_coors_x_shape, lambda *indices: wlx(*indices) * why(*indices))
    # whxXwhy = akg.tvm.compute(samples_coors_x_shape, lambda *indices: whx(*indices) * why(*indices))

    wlx = akg.tvm.compute(sample_w_bias_shape, lambda *indices: samples_x_coors_high(*indices) - samples_x_coors(*indices),
                      name='wlx')
    whx = akg.tvm.compute(sample_w_bias_shape, lambda *indices: one_const - wlx(*indices), name='whx')

    wly = akg.tvm.compute(sample_h_bias_shape, lambda *indices: samples_y_coors_high(*indices) - samples_y_coors(*indices),
                      name='wly')
    why = akg.tvm.compute(sample_h_bias_shape, lambda *indices: one_const - wly(*indices), name='why')

    samples_shape = (group_size, group_size, sample_h, sample_w, aligned_roi_num)
    wlxXwly = akg.tvm.compute(samples_shape, lambda i, j, m, n, k: wlx(0, j, n, k) * wly(0, i, m, k), name='wlxXwly')
    whxXwly = akg.tvm.compute(samples_shape, lambda i, j, m, n, k: whx(0, j, n, k) * wly(0, i, m, k), name='whxXwly')
    wlxXwhy = akg.tvm.compute(samples_shape, lambda i, j, m, n, k: wlx(0, j, n, k) * why(0, i, m, k), name='wlxXwhy')
    whxXwhy = akg.tvm.compute(samples_shape, lambda i, j, m, n, k: whx(0, j, n, k) * why(0, i, m, k), name='whxXwhy')

    boundaries_values_shape = (4, sample_h, sample_w, aligned_roi_num)
    bin_values_shape = (1, class_num + 1, bin_num, aligned_roi_num)
    gap_values_shape = (class_num + 1, aligned_roi_num)

    @akg.tvm.hybrid.script
    def fetch_data(shape, fm_in, c_idx, bin_idx, bin_num, group_size, sample_h, sample_w, roi_num,
                   x_low, x_high, y_low, y_high, one_value):
        boundaries_values = output_tensor(shape, 'float16')

        for i in range(0, sample_h):
            for j in range(0, sample_w):
                for k in range(0, roi_num):
                    # assume batch is 1

                    # w_low_idx =  x_low[0, bin_idx % group_size, j, k]
                    # w_high_idx =  x_high[0, bin_idx % group_size, j, k]
                    #
                    # h_low_idx =  y_low[0, bin_idx // group_size, i, k]
                    # h_high_idx =  y_high[0, bin_idx // group_size, i, k]

                    #x_low, y_low
                    boundaries_values[0, i, j, k] = one_value
                    boundaries_values[1, i, j, k] = one_value
                    boundaries_values[2, i, j, k] = one_value
                    boundaries_values[3, i, j, k] = one_value
                    # boundaries_values[0, i, j, k] = fm_in[0, c_idx * bin_num + bin_idx, h_low_idx, w_low_idx]
                    #
                    # #x_high, y_low
                    # boundaries_values[1, i, j, k] = fm_in[0, c_idx * bin_num + bin_idx, h_low_idx, w_high_idx]
                    #
                    # #x_low, y_high
                    # boundaries_values[2, i, j, k] = fm_in[0, c_idx * bin_num + bin_idx, h_high_idx, w_low_idx]
                    #
                    # #x_high, y_high
                    # boundaries_values[3, i, j, k] = fm_in[0, c_idx * bin_num + bin_idx, h_high_idx, w_high_idx]

        return boundaries_values

    @akg.tvm.hybrid.script(capture=locals())
    def compute_bilinear_maxpool_gap(fm_in, x_low, x_high, y_low, y_high, wlxXwly_, whxXwly_, wlxXwhy_, whxXwhy_, one_value):

        bin_values = allocate(bin_values_shape, 'float16', 'local')

        # global average result
        gap_values = output_tensor(gap_values_shape, 'float16')

        for c in range(0, class_num + 1):
            for b in range(0, bin_num):
                boundaries_values = fetch_data(boundaries_values_shape, fm_in, c, b, bin_num, group_size, sample_h, sample_w, roi_num,
                                               x_low, x_high, y_low, y_high, one_value)

                k_w = b % group_size
                k_h = b // group_size

                for n in range(0, roi_num):
                    bin_values[0, c, b, n] = half(0.0)

                for h in range(0, sample_h):
                    for w in range(0, sample_w):
                        for n in range(0, roi_num):
                            # bilinear
                            tmp = boundaries_values[0, h, w, n] * wlxXwly_[k_h, k_w, h, w, n] + \
                                boundaries_values[1, h, w, n] * whxXwly_[k_h, k_w, h, w, n] + \
                                boundaries_values[2, h, w, n] * wlxXwhy_[k_h, k_w, h, w, n] + \
                                boundaries_values[3, h, w, n] * whxXwhy_[k_h, k_w, h, w, n]

                            # maxpooling
                            if tmp > bin_values[0, c, b, n]:
                                bin_values[0, c, b, n] = tmp

            # global average pooling
            for j in range(0, roi_num):
                tmp1 = bin_values[0, c, 0, j]
                for k in range(1, bin_num):
                    tmp1 += bin_values[0, c, k, j]

                gap_values[c, j] = tmp1 / bin_num

        return gap_values

    # ==============================================================
    # step 6: compute results of bilinear, maxpooling and global average pooling
    # ==============================================================
    out = compute_bilinear_maxpool_gap(fm_data, samples_x_coors_low, samples_x_coors_high,
                                       samples_y_coors_low, samples_y_coors_high,
                                       wlxXwly, whxXwly, wlxXwhy, whxXwhy, one_const)

    # out = wlxXwhy

    # info = dim.Dim()
    # info.setdim(index=0, head = 0, body = 0, tail = 0, tilel1 = 1, tilel0 = 1)
    # info.setdim(index=0, head = 0, body = 0, tail = 0, tilel1 = 1, tilel0 = 1)

    s = akg.tvm.create_schedule(out.op)
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        # mod = akg.tvm.build(s, [fm_data, roi_data, out], "cce", name="psroialign", attrs = {"dim" : str(info)}, polyhedral=True)
        mod = akg.build(s, [fm_data, roi_data, out], "cce", name="psroialign", polyhedral=True)

    return mod


if __name__ == "__main__":
    group_size = 3
    class_num = 9
    h = w = 196
    scale = 0.0625  # 1 / 16
    roi_num = 40
    sample_h = sample_w = 2

    # fm_shape = (1, group_size * group_size * (class_num + 1), h, w)
    # roi_shape = (roi_num, 16, 1, 1)
    # psroialign_compute(fm_shape, roi_shape, class_num, group_size, sample_h, sample_w, scale)
